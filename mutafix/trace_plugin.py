"""pytest plugin: record, per test, which functions under a directory ran, and the test outcome.

Loaded with `-p mutafix.trace_plugin`. Configured by environment variables:
  MUTAFIX_TRACE_ROOT  only code objects whose file is under this directory are recorded
  MUTAFIX_TRACE_OUT   JSON written at the end: {"tests": {nodeid: {"outcome": ..., "funcs": [[path, line], ...]}}}
This is the input for spectrum-based fault localization (Ochiai). On Python 3.12+ it uses
sys.monitoring: each function is reported once per test and then disabled until the next test,
so the overhead is small. Older interpreters fall back to sys.setprofile.
Only the standard library is used, so it runs inside the target project's interpreter.
"""
import json
import os
import sys
import threading

_ROOT = os.path.normcase(os.path.abspath(os.environ.get("MUTAFIX_TRACE_ROOT", ".")))
_OUT = os.environ.get("MUTAFIX_TRACE_OUT")
_TOOL = 4
_tests = {}
_cur = None
_under = {}


def _under_root(fn):
    r = _under.get(fn)
    if r is None:
        try:
            r = os.path.normcase(os.path.abspath(fn)).startswith(_ROOT)
        except Exception:
            r = False
        _under[fn] = r
    return r


def _start(code, offset):
    if _cur is not None and _under_root(code.co_filename):
        _tests[_cur]["funcs"].add((code.co_filename, code.co_firstlineno))
    return sys.monitoring.DISABLE


def _profile(frame, event, arg):
    if event == "call" and _cur is not None:
        co = frame.f_code
        if _under_root(co.co_filename):
            _tests[_cur]["funcs"].add((co.co_filename, co.co_firstlineno))


def pytest_configure(config):
    if not _OUT:
        return
    if hasattr(sys, "monitoring"):
        sys.monitoring.use_tool_id(_TOOL, "mutafix")
        sys.monitoring.register_callback(_TOOL, sys.monitoring.events.PY_START, _start)
        sys.monitoring.set_events(_TOOL, sys.monitoring.events.PY_START)
    else:
        sys.setprofile(_profile)
        threading.setprofile(_profile)


def pytest_runtest_setup(item):
    global _cur
    if not _OUT:
        return
    _cur = item.nodeid
    _tests.setdefault(_cur, {"outcome": "unknown", "funcs": set()})
    if hasattr(sys, "monitoring"):
        sys.monitoring.restart_events()


def pytest_runtest_logreport(report):
    if not _OUT or report.nodeid not in _tests:
        return
    t = _tests[report.nodeid]
    if report.failed:
        t["outcome"] = "failed"
    elif report.when == "call" and t["outcome"] != "failed":
        t["outcome"] = report.outcome


def pytest_unconfigure(config):
    if not _OUT:
        return
    if hasattr(sys, "monitoring"):
        sys.monitoring.set_events(_TOOL, 0)
        sys.monitoring.free_tool_id(_TOOL)
    else:
        sys.setprofile(None)
    out = {nid: {"outcome": t["outcome"],
                 "funcs": sorted({(os.path.relpath(f, _ROOT).replace("\\", "/"), n) for f, n in t["funcs"]})}
           for nid, t in _tests.items()}
    with open(_OUT, "w", encoding="utf-8") as fh:
        json.dump({"tests": out}, fh)
