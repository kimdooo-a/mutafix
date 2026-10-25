"""mutafix command line.

  mutafix run --repo PATH --package PKG [--tests ...]      repair a real failing test suite
  mutafix run --bench toolz --task toolz-007               repair one injected benchmark bug
  mutafix bench --per-repo 12 --out runs/bench             run the benchmark and build the report
  mutafix report runs/bench                                rebuild the benchmark report
  mutafix fetch                                            clone the benchmark repositories
"""
import argparse
import json
import os
import sys
import time

from . import bench as B
from .agent import Agent
from .llm import PATCHER_MODEL, RANKER_MODEL, TokenFactory
from .report import bench_page, write_run


def _agent_kw(a):
    return {"ranker": a.ranker, "patcher": a.patcher, "max_turns": a.max_turns, "max_patches": a.max_patches,
            "thinking": a.thinking == "on", "think_tokens": a.think_tokens, "dynamic": a.retrieval == "dynamic",
            "rank_order": a.rank_order, "rank_only": getattr(a, "rank_only", False),
            "rank_mode": a.rank, "cascade": a.cascade == "on", "first_model": a.ranker}


def cmd_run(a):
    os.makedirs(a.out, exist_ok=True)
    print(f"models via Nebius Token Factory: fast={a.ranker.split('/')[-1]}  reasoning={a.patcher.split('/')[-1]}"
          f" (thinking {a.thinking}, cascade {a.cascade})")
    tf = TokenFactory(ledger_path=os.path.join(a.out, "ledger.jsonl"), budget_usd=a.budget)
    if a.bench:
        cfg = B.load_config(a.config)[a.bench]
        task = next(t for t in B.load_tasks(a.bench) if t["task_id"] == a.task)
        print(f"MutaFix - benchmark task {a.task} (bug injected, answer hidden from the agent)")
        out = B.run_task(tf, cfg, task, python=a.python, **_agent_kw(a))
        name = a.task
    else:
        if not a.repo:
            sys.exit("--repo or --bench is required")
        print(f"MutaFix - repairing {a.repo} (package {a.package})")
        agent = Agent(tf, a.repo, a.package, a.tests, a.pytest_opt, python=a.python,
                      keep_fix=not a.dry_run, **_agent_kw(a))
        out = agent.run()
        name = os.path.basename(os.path.abspath(a.repo))
        if out.get("fix"):
            open(os.path.join(a.out, f"{name}.patch"), "w", encoding="utf-8").write(out["fix"]["diff"])
            if a.dry_run:
                print("(dry run) patch saved, working tree left as it was before the fix")
    json.dump(out, open(os.path.join(a.out, f"{name}.json"), "w", encoding="utf-8"), indent=1)
    write_run(out, os.path.join(a.out, f"{name}.html"))
    print(f"report: {os.path.join(a.out, name + '.html')}")
    if out.get("fix"):
        print(out["fix"]["diff"])


def pick_tasks(repo, per_repo, skip=()):
    tasks = [t for t in B.load_tasks(repo) if t["task_id"] not in skip]
    return tasks[:per_repo]  # task files are already seed-shuffled by the generator


def cmd_bench(a):
    os.makedirs(os.path.join(a.out, "runs"), exist_ok=True)
    cfg = B.load_config(a.config)
    tf = TokenFactory(ledger_path=os.path.join(a.out, "ledger.jsonl"), budget_usd=a.budget)
    res_path = os.path.join(a.out, "results.jsonl")
    done = set()
    if os.path.exists(res_path):
        done = {json.loads(l)["bench"]["task_id"] for l in open(res_path, encoding="utf-8")}
    for repo in a.repos:
        for t in pick_tasks(repo, a.per_repo):
            if t["task_id"] in done:
                continue
            print(f"\n=== {t['task_id']} ({t['operator']})  spent so far ${tf.spent:.4f}", flush=True)
            try:
                out = B.run_task(tf, cfg[repo], t, python=a.python, **_agent_kw(a))
            except Exception as ex:  # API outage etc.: not recorded, so a re-run retries the task
                print(f"!! {t['task_id']} aborted: {ex!r} (will be retried on the next run)", flush=True)
                continue
            write_run(out, os.path.join(a.out, "runs", f"{t['task_id']}.html"))
            with open(res_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(out) + "\n")
            if out["status"] == "budget":
                print("budget reached, stopping")
                return build_report(a.out)
    build_report(a.out)


def build_report(out_dir, config=None):
    rows = [json.loads(l) for l in open(os.path.join(out_dir, "results.jsonl"), encoding="utf-8")]
    rows = [r for r in rows if r["status"] != "budget"]
    B.rescore(rows, B.load_config(config))
    s = B.summarize(rows)
    scored = [r for r in rows if r["bench"]["mutant_failed"]]
    links = [f"runs/{r['bench']['task_id']}.html" for r in scored]
    os.makedirs(os.path.join(out_dir, "runs"), exist_ok=True)
    for r in rows:  # re-render every trace page with the current report code
        write_run(r, os.path.join(out_dir, "runs", f"{r['bench']['task_id']}.html"))
    open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8").write(bench_page(s, scored, links))
    json.dump(s, open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8"), indent=1)
    skipped = [r["bench"]["task_id"] for r in rows if not r["bench"]["mutant_failed"]]
    print(json.dumps({k: v for k, v in s.items() if k != "by_repo"}, indent=1))
    if skipped:
        print("not scored (mutant did not fail in this environment):", skipped)
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mutafix", description="Test-driven repair agent on Nemotron + Nebius Token Factory")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--ranker", default=RANKER_MODEL)
        p.add_argument("--patcher", default=PATCHER_MODEL)
        p.add_argument("--max-turns", type=int, default=8)
        p.add_argument("--thinking", choices=["on", "off"], default="on", help="patcher reasoning mode")
        p.add_argument("--think-tokens", type=int, default=4096)
        p.add_argument("--rank", choices=["auto", "llm", "spectrum"], default="auto",
                       help="auto: keep the spectrum order when a trace exists, otherwise ask the ranker model")
        p.add_argument("--cascade", choices=["on", "off"], default="on",
                       help="on: the small model (--ranker) makes the first patch attempt, then --patcher takes over")
        p.add_argument("--rank-order", choices=["shuffled", "ordered"], default="ordered",
                       help="how the 20 candidates are shown to the ranker")
        p.add_argument("--retrieval", choices=["dynamic", "static"], default="dynamic",
                       help="dynamic: functions executed by the failing tests first; static: call graph + BM25 only")
        p.add_argument("--max-patches", type=int, default=3)
        p.add_argument("--budget", type=float, default=1.0, help="stop when this many USD are spent")
        p.add_argument("--python", default=None, help="interpreter that runs the target's tests")
        p.add_argument("--config", default=None, help="benchmark repos.json")

    r = sub.add_parser("run")
    common(r)
    r.add_argument("--repo")
    r.add_argument("--package", help="package directory inside the repo, e.g. src/mylib")
    r.add_argument("--tests", nargs="*", default=[], help="pytest arguments (default: whole suite)")
    r.add_argument("--pytest-opt", action="append", default=[])
    r.add_argument("--bench")
    r.add_argument("--task")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--out", default=os.path.join("runs", time.strftime("%Y%m%d-%H%M%S")))
    r.set_defaults(fn=cmd_run)

    b = sub.add_parser("bench")
    common(b)
    b.add_argument("--repos", nargs="*", default=["toolz", "more-itertools", "boltons", "sortedcontainers"])
    b.add_argument("--per-repo", type=int, default=12)
    b.add_argument("--rank-only", action="store_true", help="stop after the ranking step (localization experiment)")
    b.add_argument("--out", default=os.path.join("runs", "bench"))
    b.set_defaults(fn=cmd_bench)

    rp = sub.add_parser("report", help="rebuild HTML from a bench dir or a single run .json")
    rp.add_argument("out")
    rp.set_defaults(fn=lambda a: build_report(a.out) if os.path.isdir(a.out) else
                    write_run(json.load(open(a.out, encoding="utf-8")), a.out[:-5] + ".html"))

    f = sub.add_parser("fetch")
    f.add_argument("--config", default=None)
    f.set_defaults(fn=lambda a: B.fetch(B.load_config(a.config)))

    a = ap.parse_args(argv)
    if a.cmd == "run" and a.repo and not a.package:
        ap.error("--package is required with --repo")
    a.fn(a)


if __name__ == "__main__":
    main()
