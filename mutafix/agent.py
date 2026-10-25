"""The MutaFix repair loop.

  reproduce  run the test suite, collect failing tests and their error lines
  retrieve   static call graph from the failing test + BM25 -> 20 candidate functions
  rank       Nemotron Nano (no thinking, JSON schema) orders the candidates, top 5 kept
  repair     Nemotron Super (thinking on) acts through JSON tool calls:
               read(candidate)            -> full source with line numbers
               patch(candidate, old, new) -> applied, then the failing tests and the full
                                             suite are run and the result is sent back
               give_up(why)
             A failing patch is reverted before the next turn, so every attempt starts
             from the original code. The agent can only edit indexed package functions,
             never the tests.
"""
import ast
import difflib
import json
import math
import os
import random
import re
import tempfile
import time

from .index import RepoIndex
from .llm import PATCHER_MODEL, RANKER_MODEL, BudgetExceeded
from .testrun import run_pytest

K_CANDIDATES = 20
TOP_SHOWN = 5

RANK_SCHEMA = {"type": "object", "properties": {"ranking": {"type": "array", "items": {"type": "integer"}}},
               "required": ["ranking"]}
ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["read", "patch", "give_up"]},
        "candidate": {"type": "integer"},
        "old": {"type": "string"},
        "new": {"type": "string"},
        "why": {"type": "string"},
    },
    "required": ["action", "why"],
}

PATCHER_SYSTEM = """You are MutaFix, a test-driven repair agent working inside a Python repository.
Some tests fail. Find the bug in the library code and fix it with the smallest correct change.

You act only by replying with one JSON object per turn:
  {"action": "read", "candidate": K, "why": "..."}          show the full source of candidate K
  {"action": "patch", "candidate": K, "old": "...", "new": "...", "why": "..."}
        replace the text `old` (copied exactly from candidate K, one or more whole lines,
        same indentation, no line numbers) with `new`. The tests run after every patch.
  {"action": "give_up", "why": "..."}

Rules:
- Edit only the candidate functions. Never edit tests. Do not add debugging prints.
- `old` must occur exactly once in that candidate.
- If a patch fails, it is reverted; use the test output to try a different fix."""


def numbered(code, start, max_lines=None):
    lines = code.splitlines()
    cut = max_lines is not None and len(lines) > max_lines
    if cut:
        lines = lines[:max_lines]
    out = "\n".join(f"{start + i:5d} | {ln}" for i, ln in enumerate(lines))
    if cut:
        out += f"\n      | ... ({len(code.splitlines()) - max_lines} more lines, use read)"
    return out


def truncate(text, n):
    ls = text.splitlines()
    return text if len(ls) <= n else "\n".join(ls[:n] + [f"# ... ({len(ls) - n} more lines)"])


def parse_json(text):
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e < s:
        return None
    try:
        return json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None


def short_name(model):
    m = model.lower()
    return "Nano" if "nano" in m else "Super" if "super" in m else "Ultra" if "ultra" in m else model.split("/")[-1]


class PatchError(Exception):
    pass


NUM_PREFIX = re.compile(r"^\s*\d+ \| ?")


def strip_line_numbers(text):
    """Models sometimes copy the '  757 | ' gutter we show them; remove it if every line has it."""
    body = [l for l in text.split("\n") if l.strip()]
    if body and all(NUM_PREFIX.match(l) for l in body):
        return "\n".join(NUM_PREFIX.sub("", l, count=1) for l in text.split("\n"))
    return text


def _indent(line):
    return len(line) - len(line.lstrip(" "))


def match_lines(seg_lines, old, new, exact_count):
    """Whole-line match when the exact text does not occur exactly once:
    1) ignoring trailing whitespace, 2) ignoring indentation (new is re-indented by the same offset).
    The window must be unique, otherwise the patch is rejected."""
    old_l = old.strip("\n").split("\n")
    new_l = new.rstrip("\n").split("\n") if new.strip() else []
    n = len(old_l)
    for key in (str.rstrip, str.strip):
        want = [key(l) for l in old_l]
        hits = [i for i in range(len(seg_lines) - n + 1) if [key(l) for l in seg_lines[i:i + n]] == want]
        if len(hits) == 1:
            i = hits[0]
            delta = _indent(seg_lines[i]) - _indent(old_l[0]) if key is str.strip else 0
            if delta > 0:
                new_l = [(" " * delta + l) if l.strip() else l for l in new_l]
            elif delta < 0:
                new_l = [l[min(-delta, _indent(l)):] for l in new_l]
            return seg_lines[:i] + new_l + seg_lines[i + n:]
        if len(hits) > 1:
            raise PatchError(f"`old` matches {len(hits)} places in the candidate (must be exactly once)")
    raise PatchError(f"`old` occurs {exact_count} times in the candidate (must be exactly once); "
                     "copy whole lines exactly as shown, without the line-number gutter")


class Workspace:
    """Applies text patches inside one function and can restore the original files."""

    def __init__(self, root):
        self.root = root
        self.backup = {}

    def apply(self, func, old, new):
        path = os.path.join(self.root, func["file"])
        raw = open(path, encoding="utf-8", newline="").read()
        nl = "\r\n" if "\r\n" in raw else "\n"
        text = raw.replace("\r\n", "\n")
        lines = text.split("\n")
        s, e = func["start"] - 1, func["end_lineno"]
        seg = "\n".join(lines[s:e])
        old_n, new_n = strip_line_numbers(old.replace("\r\n", "\n")), strip_line_numbers(new.replace("\r\n", "\n"))
        if not old_n.strip():
            raise PatchError("`old` is empty")
        cnt = seg.count(old_n)
        if cnt == 1:
            new_seg = seg.replace(old_n, new_n.rstrip("\n") if not old_n.endswith("\n") else new_n, 1)
            seg_lines = new_seg.split("\n")
        else:
            seg_lines = match_lines(seg.split("\n"), old_n, new_n, cnt)
        new_lines = lines[:s] + seg_lines + lines[e:]
        new_text = "\n".join(new_lines)
        try:
            ast.parse(new_text)
        except SyntaxError as ex:
            raise PatchError(f"patched file does not parse: {ex.msg} (line {ex.lineno})")
        if path not in self.backup:
            self.backup[path] = raw
        open(path, "w", encoding="utf-8", newline="").write(new_text.replace("\n", nl))
        diff = "".join(difflib.unified_diff(
            text.splitlines(True), new_text.splitlines(True),
            fromfile=f"a/{func['file']}", tofile=f"b/{func['file']}", n=2))
        return diff

    def restore(self):
        for path, raw in self.backup.items():
            open(path, "w", encoding="utf-8", newline="").write(raw)
        self.backup = {}

    def keep(self):
        self.backup = {}


class Agent:
    def __init__(self, tf, root, package, test_args, pytest_opts=(), python=None,
                 ranker=RANKER_MODEL, patcher=PATCHER_MODEL, max_turns=8, max_patches=3, max_reads=3,
                 test_timeout=120, log=print, seed=0, keep_fix=True, thinking=True, think_tokens=4096, dynamic=True,
                 rank_order="ordered", rank_only=False, rank_mode="auto", cascade=True,
                 first_model=RANKER_MODEL, first_turns=2):
        self.tf, self.root, self.package = tf, root, package
        self.test_args, self.pytest_opts, self.python = list(test_args), list(pytest_opts), python
        self.ranker, self.patcher = ranker, patcher
        self.max_turns, self.max_patches, self.max_reads = max_turns, max_patches, max_reads
        self.test_timeout = test_timeout
        self.log, self.seed = log, seed
        self.keep_fix = keep_fix
        self.thinking, self.think_tokens = thinking, think_tokens
        self.dynamic = dynamic
        self.rank_order, self.rank_only, self.rank_mode = rank_order, rank_only, rank_mode
        self.cascade, self.first_model, self.first_turns = cascade, first_model, first_turns
        self.spectrum = None
        self.steps = []
        self.t0 = None

    def _step(self, kind, **kw):
        rec = {"kind": kind, "t": round(time.time() - self.t0, 2), **kw}
        self.steps.append(rec)
        return rec

    def _tests(self, args=None):
        return run_pytest(self.root, args if args is not None else self.test_args, self.pytest_opts,
                          timeout=self.test_timeout, python=self.python)

    # ------------------------------------------------------------------ phases
    def reproduce(self):
        res = self._tests()
        self._step("reproduce", ok=res["ok"], failed=res["failed"][:20], n_failed=len(res["failed"]),
                   counts=res["counts"], wall_s=res["wall_s"],
                   errors={k: v["e_lines"] for k, v in list(res["errors"].items())[:5]})
        self.log(f"[reproduce] {len(res['failed'])} failing test(s) in {res['wall_s']}s")
        return res

    def retrieve(self, idx, res):
        primary = next((f for f in res["failed"] if "::" in f), res["failed"][0])
        test_src, names = idx.test_source(primary)
        err = "\n".join(res["errors"].get(primary, {}).get("e_lines", []))
        query = test_src + "\n" + err + "\n" + primary
        bm_order, bm_scores = idx.bm25_rank(query)
        g_order, depth = idx.graph_rank(names, bm_scores)
        spec = self.trace(idx, res) if self.dynamic else None
        self.spectrum = spec
        if spec:
            # spectrum-based fault localization: Ochiai over the tests in the failing test files,
            # call-graph/BM25 order breaks ties and fills the list with functions that never ran
            pos = {i: k for k, i in enumerate(g_order)}
            ran = [i for i in spec["score"] if spec["score"][i] > 0]
            ran.sort(key=lambda i: (-spec["score"][i], pos[i]))
            cands = (ran + [i for i in g_order if i not in spec["score"] or spec["score"][i] == 0])[:K_CANDIDATES]
        else:
            cands = g_order[:K_CANDIDATES]
        self._step("retrieve", primary=primary, n_functions=len(idx.funcs), n_reachable=len(depth),
                   n_executed=len([i for i in (spec or {}).get("score", {}) if spec["score"][i] > 0]) if spec else None,
                   candidates=[{"id": idx.funcs[i]["id"], "depth": depth.get(i),
                                "ochiai": round(spec["score"].get(i, 0.0), 3) if spec else None,
                                "fail_hits": spec["nf"].get(i, 0) if spec else None,
                                "pass_hits": spec["np"].get(i, 0) if spec else None} for i in cands])
        extra = (f", spectrum over {spec['n_failed']} failing / {spec['n_passed']} passing tests"
                 if spec else "")
        self.log(f"[retrieve] {len(idx.funcs)} functions indexed, {len(depth)} reachable from "
                 f"{primary.split('::')[-1]}{extra}, kept {len(cands)}")
        return primary, test_src, err, cands, depth

    def trace(self, idx, res):
        """Re-run the test files that contain failing tests with the trace plugin and score each
        function with Ochiai: failed(f) / sqrt(total_failed * (failed(f) + passed(f)))."""
        files = []
        for f in res["failed"]:
            if "::" in f:
                path = f.split("::")[0]
                if path not in files:
                    files.append(path)
        files = files[:5]
        if not files:
            return None
        out = os.path.join(tempfile.gettempdir(), f"mutafix-trace-{os.getpid()}.json")
        if os.path.exists(out):
            os.remove(out)
        pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = {"MUTAFIX_TRACE_ROOT": os.path.abspath(self.root), "MUTAFIX_TRACE_OUT": out,
               "PYTHONPATH": pkg_parent + os.pathsep + os.environ.get("PYTHONPATH", "")}
        r = run_pytest(self.root, files, [*self.pytest_opts, "-p", "mutafix.trace_plugin"],
                       timeout=self.test_timeout, python=self.python, env_extra=env)
        if not os.path.exists(out):
            self._step("trace", files=files, error="no trace written (timeout?)", wall_s=r["wall_s"])
            self.log("[trace] no trace written, falling back to static retrieval")
            return None
        tests = json.load(open(out, encoding="utf-8"))["tests"]
        os.remove(out)
        key = {}
        for i, fn in enumerate(idx.funcs):
            key[(fn["file"], fn["start"])] = i
            key[(fn["file"], fn["lineno"])] = i
        nf, np_ = {}, {}
        n_failed = n_passed = 0
        for t in tests.values():
            if t["outcome"] not in ("failed", "passed"):
                continue
            hit = {key[(f, n)] for f, n in t["funcs"] if (f, n) in key}
            if t["outcome"] == "failed":
                n_failed += 1
                for i in hit:
                    nf[i] = nf.get(i, 0) + 1
            else:
                n_passed += 1
                for i in hit:
                    np_[i] = np_.get(i, 0) + 1
        if not n_failed:
            self._step("trace", files=files, error="no failing test in trace", wall_s=r["wall_s"])
            return None
        score = {i: nf.get(i, 0) / math.sqrt(n_failed * (nf.get(i, 0) + np_.get(i, 0)))
                 for i in set(nf) | set(np_)}
        self._step("trace", files=files, n_failed=n_failed, n_passed=n_passed,
                   n_executed=len(nf), wall_s=r["wall_s"])
        self.log(f"[trace] {len(files)} test file(s): {n_failed} failing, {n_passed} passing tests, "
                 f"{len(nf)} functions ran in a failing test ({r['wall_s']}s)")
        return {"score": score, "nf": nf, "np": np_, "n_failed": n_failed, "n_passed": n_passed}

    def _spec_note(self, i):
        sp = getattr(self, "spectrum", None)
        if not sp:
            return ""
        return (f"  [ran in {sp['nf'].get(i, 0)}/{sp['n_failed']} failing and "
                f"{sp['np'].get(i, 0)}/{sp['n_passed']} passing tests]")

    def rank(self, idx, primary, test_src, err, cands):
        shown = list(cands)
        if self.rank_order == "shuffled":
            random.Random(f"{self.seed}|{primary}").shuffle(shown)  # no position hint from the retriever
        parts = [f"Repository package: {self.package}",
                 "One function in this repository has a bug. The following test fails.",
                 f"Failing test: {primary}",
                 "Test source:\n```python\n" + truncate(test_src, 60) + "\n```",
                 "Failure output:\n```\n" + (err or "(no assertion message)") + "\n```",
                 f"Candidate functions ({len(shown)}, " +
                 ("in random order)" if self.rank_order == "shuffled" else
                  "ordered by a fault-localization score, most suspicious first)") +
                 (". The bracket after a name says in how many failing and passing tests of the same "
                  "test files the function ran:" if getattr(self, "spectrum", None) else ":")]
        for k, i in enumerate(shown, 1):
            f = idx.funcs[i]
            parts.append(f"[{k}] {f['qualname']}  (file: {f['file']}){self._spec_note(i)}\n```python\n{truncate(f['code'], 40)}\n```")
        parts.append('Which candidate most likely contains the bug? Answer with JSON only: '
                     '{"ranking": [k1, k2, k3, k4, k5]}, most likely first.')
        r = self.tf.chat(self.ranker, [{"role": "user", "content": "\n\n".join(parts)}],
                         max_tokens=200, thinking=False, schema=RANK_SCHEMA, tag="rank")
        obj = parse_json(r["content"]) or {}
        picks = []
        for x in obj.get("ranking", []):
            if isinstance(x, int) and 1 <= x <= len(shown) and shown[x - 1] not in picks:
                picks.append(shown[x - 1])
        rest = [i for i in cands if i not in picks]
        order = picks[:TOP_SHOWN] + [i for i in picks[TOP_SHOWN:] + rest if i not in picks[:TOP_SHOWN]]
        self._step("rank", model=self.ranker, ranking=[idx.funcs[i]["id"] for i in picks[:TOP_SHOWN]],
                   prompt_tokens=r["prompt_tokens"], completion_tokens=r["completion_tokens"],
                   wall_s=r["wall_s"], cost_usd=r["cost_usd"], valid=bool(picks))
        self.log(f"[rank] {self.ranker.split('/')[-1]}: top-1 {idx.funcs[order[0]]['qualname']} "
                 f"({r['wall_s']}s, {r['prompt_tokens']} tok)")
        return order

    def _test_report(self, before, res):
        if res["ok"]:
            return "All tests pass."
        now = set(res["failed"])
        fixed = [f for f in before if f not in now]
        new = [f for f in res["failed"] if f not in set(before)]
        lines = [f"Tests still failing: {len(res['failed'])} (was {len(before)}). "
                 f"Fixed by this patch: {len(fixed)}. Newly failing: {len(new)}. "
                 "Your patch has been reverted."]
        for nid in (new[:2] + [f for f in res["failed"] if f not in new][:2])[:3]:
            e = res["errors"].get(nid, {}).get("e_lines", [])
            lines.append(f"- {nid}\n" + "\n".join(e[:8]))
        if res["timed_out"]:
            lines.append("The test run timed out (possible infinite loop).")
        return "\n".join(lines)

    def repair(self, idx, primary, test_src, res, order):
        ws = Workspace(self.root)
        cand = order
        view = ["Failing tests (" + str(len(res["failed"])) + "):\n" + "\n".join(res["failed"][:8])]
        for nid in res["failed"][:3]:
            view.append(f"{nid}\n" + "\n".join(res["errors"].get(nid, {}).get("e_lines", [])[:10]))
        view.append(f"Source of {primary}:\n```python\n{truncate(test_src, 60)}\n```")
        view.append("Candidate functions, ranked by a fault-localization model (most likely first):")
        for k, i in enumerate(cand[:TOP_SHOWN], 1):
            f = idx.funcs[i]
            view.append(f"[{k}] {f['qualname']}  ({f['file']}){self._spec_note(i)}\n```\n{numbered(f['code'], f['start'], 120)}\n```")
        others = [f"[{k}] {idx.funcs[i]['qualname']}  ({idx.funcs[i]['file']}){self._spec_note(i)}"
                  for k, i in enumerate(cand[TOP_SHOWN:], TOP_SHOWN + 1)]
        if others:
            view.append("More candidates (use read to see them):\n" + "\n".join(others))
        messages = [{"role": "system", "content": PATCHER_SYSTEM},
                    {"role": "user", "content": "\n\n".join(view)}]
        before = list(res["failed"])
        patches = reads = first_patches = 0
        seen = {idx.funcs[i]["id"] for i in cand[:TOP_SHOWN] if len(idx.funcs[i]["code"].splitlines()) <= 120}
        for turn in range(1, self.max_turns + 1):
            # cascade: the small model gets the first patch attempt (fast, no reasoning);
            # after one patch, or after first_turns turns, the large reasoning model takes over
            first = self.cascade and first_patches == 0 and turn <= self.first_turns
            model = self.first_model if first else self.patcher
            thinking = False if first else self.thinking
            lab = short_name(model)
            r = self.tf.chat(model, messages, max_tokens=self.think_tokens if thinking else 2048,
                             thinking=thinking, schema=ACTION_SCHEMA, tag=f"turn{turn}")
            if not (r["content"] or "").strip() and r["finish_reason"] == "length":
                # reasoning ran out of room: ask again without thinking, keep both costs
                r2 = self.tf.chat(model, messages, max_tokens=2048, thinking=False,
                                  schema=ACTION_SCHEMA, tag=f"turn{turn}-nothink")
                for key in ("prompt_tokens", "completion_tokens", "wall_s", "cost_usd"):
                    r2[key] = r2[key] + r[key]
                r2["reasoning"] = r["reasoning"]
                r = r2
            act = parse_json(r["content"])
            messages.append({"role": "assistant", "content": r["content"] or "{}"})
            meta = {"turn": turn, "model": model, "prompt_tokens": r["prompt_tokens"],
                    "completion_tokens": r["completion_tokens"], "wall_s": r["wall_s"],
                    "cost_usd": r["cost_usd"], "thought": (r["reasoning"] or "")[:1500]}
            if not act or act.get("action") not in ("read", "patch", "give_up"):
                self._step("turn", action="invalid", raw=(r["content"] or "")[:300], **meta)
                self.log(f"[turn {turn} {lab}] invalid reply ({r['finish_reason']})")
                messages.append({"role": "user", "content": "Reply with one JSON action object."})
                continue
            a = act["action"]
            if a == "give_up":
                self._step("turn", action="give_up", why=act.get("why", ""), **meta)
                self.log(f"[turn {turn} {lab}] give up: {act.get('why', '')[:80]}")
                return False, None
            k = act.get("candidate")
            if not isinstance(k, int) or not 1 <= k <= len(cand):
                self._step("turn", action=a, target=None, error=f"bad candidate number {k!r}", **meta)
                messages.append({"role": "user", "content": f"candidate must be 1..{len(cand)}"})
                continue
            f = idx.funcs[cand[k - 1]]
            left = self.max_turns - turn
            if a == "read":
                again = f["id"] in seen
                if not again and reads >= self.max_reads:
                    why_not = f"read limit ({self.max_reads}) reached"
                    self._step("turn", action="read", target=f["id"], why=act.get("why", ""),
                               error=why_not, **meta)
                    self.log(f"[turn {turn} {lab}] read refused ({why_not})")
                    messages.append({"role": "user", "content":
                                     f"Read refused: {why_not}. Propose a patch now. Turns left: {left}."})
                    continue
                if not again:  # re-showing code already in the conversation does not use up a read
                    reads += 1
                    seen.add(f["id"])
                self._step("turn", action="read", target=f["id"], why=act.get("why", ""), **meta)
                self.log(f"[turn {turn} {lab}] read {f['qualname']}")
                messages.append({"role": "user", "content":
                                 f"[{k}] {f['qualname']}  ({f['file']})\n```\n{numbered(f['code'], f['start'])}\n```"
                                 f"\nTurns left: {left}. Reads left: {self.max_reads - reads}."})
                continue
            # patch
            if first:
                first_patches += 1
            else:
                patches += 1
            meta["edit"] = {"old": act.get("old", "")[:2000], "new": act.get("new", "")[:2000]}
            try:
                diff = ws.apply(f, act.get("old", ""), act.get("new", ""))
            except PatchError as ex:
                self._step("turn", action="patch", target=f["id"], why=act.get("why", ""),
                           error=str(ex), **meta)
                self.log(f"[turn {turn} {lab}] patch rejected: {ex}")
                messages.append({"role": "user", "content": f"Patch not applied: {ex}. Turns left: {left}."})
                if patches >= self.max_patches:
                    break
                continue
            quick = self._tests([b for b in before if "::" in b][:20] or None)
            full = self._tests() if quick["ok"] else quick
            passed = full["ok"]
            st = self._step("turn", action="patch", target=f["id"], why=act.get("why", ""), diff=diff,
                            quick={"ok": quick["ok"], "failed": quick["failed"][:10], "wall_s": quick["wall_s"]},
                            full={"ok": full["ok"], "failed": full["failed"][:10], "counts": full["counts"],
                                  "wall_s": full["wall_s"], "ran": quick["ok"]},
                            **meta)
            self.log(f"[turn {turn} {lab}] patch {f['qualname']}: "
                     f"{'ALL GREEN' if passed else str(len(full['failed'])) + ' failing'}")
            if passed:
                ws.keep() if self.keep_fix else ws.restore()
                return True, {"diff": diff, "target": f["id"], "step": st}
            ws.restore()
            messages.append({"role": "user", "content": self._test_report(before, full) + f"\nTurns left: {left}."})
            if patches >= self.max_patches:
                break
        ws.restore()
        return False, None

    def run(self):
        self.t0 = time.time()
        out = {"root_name": os.path.basename(os.path.abspath(self.root)), "package": self.package,
               "ranker": self.ranker, "patcher": self.patcher, "steps": self.steps}
        res = self.reproduce()
        if res["ok"]:
            out.update(status="nothing_to_fix", wall_s=round(time.time() - self.t0, 2))
            return out
        idx = RepoIndex(self.root, self.package)
        try:
            primary, test_src, err, cands, depth = self.retrieve(idx, res)
            if self.rank_mode == "llm" or (self.rank_mode == "auto" and not self.spectrum):
                order = self.rank(idx, primary, test_src, err, cands)
            else:  # the spectrum order is kept: in our rank-only experiment Nano did not improve on it
                order = list(cands)
                self._step("spectrum_rank", ranking=[idx.funcs[i]["id"] for i in order[:TOP_SHOWN]])
                self.log(f"[rank] spectrum order kept: top-1 {idx.funcs[order[0]]['qualname']}")
            if self.rank_only:
                raise StopIteration
            ok, fix = self.repair(idx, primary, test_src, res, order)
            status = "fixed" if ok else "not_fixed"
        except StopIteration:
            ok, fix, status = False, None, "ranked_only"
        except BudgetExceeded as ex:
            ok, fix, status = False, None, "budget"
            self._step("budget", message=str(ex))
        llm = [s for s in self.steps if "cost_usd" in s]
        out.update(status=status, fix=fix, wall_s=round(time.time() - self.t0, 2),
                   llm_calls=len(llm), cost_usd=sum(s["cost_usd"] for s in llm),
                   prompt_tokens=sum(s["prompt_tokens"] for s in llm),
                   completion_tokens=sum(s["completion_tokens"] for s in llm))
        self.log(f"[result] {status} in {out['wall_s']}s, {len(llm)} model calls, ${out['cost_usd']:.4f}")
        return out
