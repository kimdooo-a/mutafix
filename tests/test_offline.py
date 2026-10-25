"""Offline tests: patch application, retrieval and pytest parsing. No API calls."""
import textwrap

import pytest
from mutafix.agent import PatchError, Workspace, parse_json
from mutafix.index import RepoIndex
from mutafix.testrun import run_pytest

LIB = textwrap.dedent('''\
    def clamp(x, lo, hi):
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x


    def mean(xs):
        return sum(xs) / len(xs)
    ''')
TEST = textwrap.dedent('''\
    from pkg.lib import clamp, mean

    def test_clamp():
        assert clamp(5, 0, 3) == 3

    def test_mean():
        assert mean([1, 2, 3]) == 2
    ''')


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "lib.py").write_text(LIB.replace("x > hi", "x < hi"))  # injected bug
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_lib.py").write_text(TEST)
    return tmp_path


def test_reproduce_and_parse(repo):
    res = run_pytest(str(repo), ["tests"], ["-o", "pythonpath=."])
    assert not res["ok"]
    assert res["failed"] == ["tests/test_lib.py::test_clamp"]
    assert any("assert" in l for l in res["errors"]["tests/test_lib.py::test_clamp"]["e_lines"])


def test_retrieval_reaches_called_function(repo):
    idx = RepoIndex(str(repo), "pkg")
    src, names = idx.test_source("tests/test_lib.py::test_clamp")
    assert "clamp" in names
    order, depth = idx.graph_rank(names, [0.0] * len(idx.funcs))
    assert idx.funcs[order[0]]["qualname"] == "clamp"


def test_patch_apply_then_tests_pass(repo):
    idx = RepoIndex(str(repo), "pkg")
    f = next(f for f in idx.funcs if f["qualname"] == "clamp")
    ws = Workspace(str(repo))
    diff = ws.apply(f, "    if x < hi:", "    if x > hi:")
    assert "+    if x > hi:" in diff
    assert run_pytest(str(repo), ["tests"], ["-o", "pythonpath=."])["ok"]
    ws.restore()
    assert "x < hi" in (repo / "pkg" / "lib.py").read_text()


def test_patch_rejects_ambiguous_and_syntax_errors(repo):
    idx = RepoIndex(str(repo), "pkg")
    f = next(f for f in idx.funcs if f["qualname"] == "clamp")
    ws = Workspace(str(repo))
    with pytest.raises(PatchError):
        ws.apply(f, "        return", "        return 0")  # occurs 2 times
    with pytest.raises(PatchError):
        ws.apply(f, "    if x < hi:", "    if x >>> hi:")
    with pytest.raises(PatchError):
        ws.apply(f, "def mean", "def average")  # outside the candidate function


def test_patch_preserves_crlf(tmp_path):
    (tmp_path / "pkg").mkdir()
    p = tmp_path / "pkg" / "m.py"
    p.write_bytes(b"def f(a):\r\n    return a + 1\r\n")
    idx = RepoIndex(str(tmp_path), "pkg")
    Workspace(str(tmp_path)).apply(idx.funcs[0], "    return a + 1", "    return a - 1")
    assert p.read_bytes() == b"def f(a):\r\n    return a - 1\r\n"


def test_parse_json_tolerates_prose():
    assert parse_json('sure: {"action": "read", "candidate": 2, "why": "x"} done') == \
        {"action": "read", "candidate": 2, "why": "x"}
    assert parse_json("no json") is None


def test_patch_tolerates_gutter_and_indentation(repo):
    idx = RepoIndex(str(repo), "pkg")
    f = next(f for f in idx.funcs if f["qualname"] == "clamp")
    ws = Workspace(str(repo))
    # line-number gutter copied from the prompt
    ws.apply(f, "    4 |     if x < hi:", "    4 |     if x > hi:")
    assert "if x > hi:" in (repo / "pkg" / "lib.py").read_text()
    ws.restore()
    # indentation lost: matched on stripped lines, new text re-indented
    ws.apply(f, "if x < hi:\n    return hi", "if x > hi:\n    return hi")
    text = (repo / "pkg" / "lib.py").read_text()
    assert "    if x > hi:\n        return hi" in text
    assert run_pytest(str(repo), ["tests"], ["-o", "pythonpath=."])["ok"]
    ws.restore()
