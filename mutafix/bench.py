"""Benchmark mode: inject a known one-line bug (a MutaLoc task), let the agent repair it
without seeing the answer, then score the result and restore the repository."""
import json
import os
import subprocess
import time

from .agent import Agent
from .index import RepoIndex

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.normpath(os.path.join(HERE, "..", "bench"))


def load_config(path=None):
    path = path or os.path.join(BENCH_DIR, "repos.json")
    cfg = json.load(open(path, encoding="utf-8"))
    base = os.path.dirname(os.path.abspath(path))
    for name, c in cfg.items():
        c["abs_root"] = os.path.normpath(os.path.join(base, c["root"]))
        c["name"] = name
    return cfg


def load_tasks(repo, tasks_dir=None):
    p = os.path.join(tasks_dir or os.path.join(BENCH_DIR, "tasks"), f"{repo}.jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def fetch(cfg):
    """Clone each benchmark repository at its pinned commit (idempotent)."""
    for name, c in cfg.items():
        root = c["abs_root"]
        if not os.path.isdir(os.path.join(root, ".git")):
            os.makedirs(os.path.dirname(root), exist_ok=True)
            subprocess.run(["git", "clone", "-q", c["url"], root], check=True)
        subprocess.run(["git", "-C", root, "-c", "advice.detachedHead=false", "checkout", "-q", c["commit"]], check=True)
        print(f"{name}: {c['commit'][:10]}")


def git_clean(root):
    r = subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
                       capture_output=True, text=True)
    return r.stdout.strip() == ""


def restore(root):
    subprocess.run(["git", "-C", root, "checkout", "-q", "--", "."], check=True)


def inject(root, task):
    path = os.path.join(root, task["gt_file"])
    raw = open(path, encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    lines = raw.replace("\r\n", "\n").split("\n")
    i = task["gt_line"] - 1
    if lines[i] != task["line_before"]:
        raise RuntimeError(f"{task['task_id']}: line {task['gt_line']} does not match the task")
    lines[i] = task["line_after"]
    open(path, "w", encoding="utf-8", newline="").write("\n".join(lines).replace("\n", nl))


def current_line(root, task):
    path = os.path.join(root, task["gt_file"])
    return open(path, encoding="utf-8").read().split("\n")[task["gt_line"] - 1].rstrip("\r")


def run_task(tf, c, task, log=print, python=None, **agent_kw):
    root = c["abs_root"]
    if not git_clean(root):
        restore(root)
    inject(root, task)
    try:
        gt_id = None
        idx = RepoIndex(root, c["package"])
        gi = idx.find(task["gt_file"], task["gt_line"])
        gt_id = idx.funcs[gi]["id"] if gi is not None else None
        agent = Agent(tf, root, c["package"], c["test_args"], c.get("pytest_opts", []), python=python,
                      log=log, **agent_kw)
        out = agent.run()
        steps = out["steps"]
        retr = next((s for s in steps if s["kind"] == "retrieve"), {})
        rank = next((s for s in steps if s["kind"] in ("rank", "spectrum_rank")), {})
        cand_ids = [x["id"] for x in retr.get("candidates", [])]
        ranking = rank.get("ranking", [])
        restored = current_line(root, task) == task["line_before"] if out["status"] == "fixed" else False
        out["bench"] = {
            "task_id": task["task_id"], "repo": task["repo"], "operator": task["operator"],
            "gt_function": gt_id,
            "mutation": {"before": task["line_before"].strip(), "after": task["line_after"].strip(),
                         "file": task["gt_file"], "line": task["gt_line"]},
            "gt_in_candidates": gt_id in cand_ids,
            "gt_rank_retriever": cand_ids.index(gt_id) + 1 if gt_id in cand_ids else None,
            "gt_rank_model": ranking.index(gt_id) + 1 if gt_id in ranking else None,
            "patched_gt_function": bool(out.get("fix")) and out["fix"]["target"] == gt_id,
            "exact_restore": restored,
            "mutant_failed": out["status"] != "nothing_to_fix",
        }
        rescore([out], {c["name"]: c})  # count the enclosing function as the right place too
        return out
    finally:
        restore(root)
        assert git_clean(root), f"{root} not clean after task"


def summarize(rows):
    """Aggregate metrics over task results (rows with a 'bench' block)."""
    rows = [r for r in rows if r.get("bench", {}).get("mutant_failed")]
    n = len(rows)
    if not n:
        return {"n": 0}
    pct = lambda k: round(100.0 * k / n, 1)
    fixed = [r for r in rows if r["status"] == "fixed"]
    by_repo = {}
    for r in rows:
        b = by_repo.setdefault(r["bench"]["repo"], {"n": 0, "fixed": 0, "top1": 0})
        b["n"] += 1
        b["fixed"] += r["status"] == "fixed"
        b["top1"] += r["bench"]["gt_rank_model"] == 1
    turns = [s for r in rows for s in r["steps"] if s["kind"] == "turn"]
    ranks = [s for r in rows for s in r["steps"] if s["kind"] == "rank"]
    return {
        "n": n,
        "fixed": len(fixed), "fixed_pct": pct(len(fixed)),
        "exact_restore": sum(r["bench"]["exact_restore"] for r in rows),
        "plausible_not_exact": sum(1 for r in fixed if not r["bench"]["exact_restore"]),
        "gt_in_candidates_pct": pct(sum(r["bench"]["gt_in_candidates"] for r in rows)),
        "retriever_top1_pct": pct(sum(r["bench"]["gt_rank_retriever"] == 1 for r in rows)),
        "ranker_top1_pct": pct(sum(r["bench"]["gt_rank_model"] == 1 for r in rows)),
        "ranker_top5_pct": pct(sum(r["bench"]["gt_rank_model"] is not None for r in rows)),
        "patched_gt_function_pct": pct(sum(r["bench"]["patched_gt_function"] for r in rows)),
        "status": {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})},
        "mean_wall_s": round(sum(r["wall_s"] for r in rows) / n, 1),
        "median_wall_s_fixed": sorted(r["wall_s"] for r in fixed)[len(fixed) // 2] if fixed else None,
        "mean_rank_call_s": round(sum(s["wall_s"] for s in ranks) / max(1, len(ranks)), 2),
        "mean_patch_call_s": round(sum(s["wall_s"] for s in turns) / max(1, len(turns)), 2),
        "mean_turns": round(len(turns) / n, 2),
        "cost_usd": round(sum(r.get("cost_usd", 0) for r in rows), 4),
        "cost_per_task_usd": round(sum(r.get("cost_usd", 0) for r in rows) / n, 5),
        "cost_per_fix_usd": round(sum(r.get("cost_usd", 0) for r in rows) / max(1, len(fixed)), 5),
        "tokens": {"prompt": sum(r.get("prompt_tokens", 0) for r in rows),
                   "completion": sum(r.get("completion_tokens", 0) for r in rows)},
        "by_repo": by_repo,
        "generated": time.strftime("%Y-%m-%d %H:%M"),
    }


def rescore(rows, cfg):
    """Count a function as 'the buggy one' if it is the innermost function holding the mutated
    line or an enclosing function (a fix in thread_first that edits its nested evalform_front
    is a fix in the right place). Line spans come from the clean checkout: a one-line mutation
    does not move any line."""
    idx_cache = {}
    for r in rows:
        b = r.get("bench")
        if not b:
            continue
        repo = b["repo"]
        if repo not in idx_cache:
            idx_cache[repo] = RepoIndex(cfg[repo]["abs_root"], cfg[repo]["package"])
        idx = idx_cache[repo]
        spans = {f["id"]: f for f in idx.funcs}
        m = b["mutation"]

        def holds(fid):
            f = spans.get(fid)
            return bool(f) and f["file"] == m["file"] and f["start"] <= m["line"] <= f["end_lineno"]

        rank = next((s for s in r["steps"] if s["kind"] in ("rank", "spectrum_rank")), {}).get("ranking", [])
        retr = [c["id"] for c in next((s for s in r["steps"] if s["kind"] == "retrieve"), {}).get("candidates", [])]
        b["gt_rank_model"] = next((k for k, fid in enumerate(rank, 1) if holds(fid)), None)
        b["gt_rank_retriever"] = next((k for k, fid in enumerate(retr, 1) if holds(fid)), None)
        b["gt_in_candidates"] = b["gt_rank_retriever"] is not None
        b["patched_gt_function"] = bool(r.get("fix")) and holds(r["fix"]["target"])
        if r.get("fix") and git_clean(cfg[repo]["abs_root"]):
            b["exact_restore"] = b["exact_restore"] or ast_restore(r, cfg)
    return rows


def apply_unified_diff(text, diff):
    """Apply a unified diff produced by Workspace.apply to text (both '\n' separated)."""
    import re as _re
    lines = text.split("\n")
    out, pos = [], 0
    hunks = _re.split(r"^(@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@.*)$", diff, flags=_re.M)
    for i in range(1, len(hunks), 2):
        start = int(_re.match(r"@@ -(\d+)", hunks[i]).group(1)) - 1
        body = hunks[i + 1].split("\n")
        if body and body[0] == "":
            body = body[1:]
        out += lines[pos:start]
        pos = start
        for b in body:
            if b.startswith(" "):
                out.append(lines[pos]); pos += 1
            elif b.startswith("-"):
                pos += 1
            elif b.startswith("+"):
                out.append(b[1:])
    out += lines[pos:]
    return "\n".join(out)


def ast_restore(r, cfg):
    """True if the patched file has the same AST as the original (clean) file: the fix undoes the
    bug exactly, ignoring formatting such as quote style. Needs the clean checkout."""
    import ast as _ast
    b, fix = r["bench"], r.get("fix")
    if not fix:
        return False
    c = cfg[b["repo"]]
    clean = open(os.path.join(c["abs_root"], b["mutation"]["file"]), encoding="utf-8").read().replace("\r\n", "\n")
    lines = clean.split("\n")
    i = b["mutation"]["line"] - 1
    indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
    lines[i] = indent + b["mutation"]["after"]
    try:
        patched = apply_unified_diff("\n".join(lines), fix["diff"])
        return _ast.dump(_ast.parse(patched)) == _ast.dump(_ast.parse(clean))
    except Exception:
        return False
