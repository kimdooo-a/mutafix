# MutaFix

**A test-driven repair agent for Python repositories, built on NVIDIA Nemotron models served by Nebius Token Factory.**

Point MutaFix at a repository whose test suite fails. It runs pytest, re-runs the failing test
files with a small tracing plugin to see which library functions ran in failing and in passing
tests, and ranks every function with a spectrum score (Ochiai). **Nemotron 3 Nano** gets the
first repair attempt: it reads code and proposes a patch through a JSON tool loop, with
reasoning off, in about two seconds per turn. Every patch is applied and the failing tests,
then the full suite, are run. If the suite is still red, the patch is reverted and
**Nemotron 3 Super** takes over the same conversation with reasoning on, sees what failed, and
keeps going until the whole suite is green or the turn budget is spent.

Each run writes an HTML trace a reviewer can read in a minute: the failing tests, the trace,
the candidates with their scores, every patch as a diff, every test result, the model's
reasoning, and the tokens, latency and cost of each call.

Hackathon track: **Coding and Agentic Engineering** (agents that write, run, and test code in Token Factory).

- Demo video (under 3 minutes): linked from the Devpost submission (YouTube)
- Reports on GitHub Pages: [benchmark index](https://kimdooo-a.github.io/mutafix/) ·
  [quick-start trace](https://kimdooo-a.github.io/mutafix/demo/invoice.html) ·
  [benchmark trace with escalation](https://kimdooo-a.github.io/mutafix/demo/toolz-008.html)

## Results

48 injected one-line bugs, 12 in each of four open-source libraries (toolz, more-itertools,
boltons, sortedcontainers at pinned commits). The agent never sees the answer: the bug is
written into a clean checkout and the agent starts from pytest, exactly as in normal use.

| | final pipeline |
|---|---|
| repaired (full test suite green) | **40 / 48 (83%)** |
| patch restores the original code exactly (same AST) | 30 / 48 |
| passes every test with a different edit | 10 (see [Honest limits](#honest-limits)) |
| buggy function among the 20 candidates / in the top 5 / first | 98% / 96% / 83% |
| repairs made by Nano alone / after escalation to Super | 20 / 20 |
| API cost, all 48 tasks | $0.39 (**$0.0081 per task**, $0.0097 per repair) |
| median wall time of a repair | 33 s (mean per task, including failures: 51 s) |
| mean time per model turn | Nano 2.3 s · Super 17.9 s (reasoning on) |

Per repository (repaired / 12): toolz 9, more-itertools 10, boltons 12, sortedcontainers 9.

### What moved the numbers

Every row is the same 48 tasks. The full rows, per-task traces and the cost ledger of each
experiment are in [`results/`](results/).

| configuration | repaired | bug in 20 candidates | bug in top 5 | cost / task | time / task |
|---|---|---|---|---|---|
| static call graph + BM25, Nano re-ranks, Super patches (thinking on) | 31 | 77% | 65% | $0.0107 | 58 s |
| + test trace with Ochiai scores, Nano re-ranks (shuffled), Super patches | 39 | 98% | 71% | $0.0101 | 62 s |
| same, Super with thinking off | 29 | 98% | 69% | $0.0062 | 27 s |
| **final: trace order kept, Nano patches first, Super takes over** | **40** | 98% | 96% | **$0.0081** | 51 s |

Three findings decided the final design:

1. **The trace was the biggest single gain.** With static retrieval, the buggy function was
   missing from the 20 candidates in 11 of 48 tasks, and no model can fix a function it is
   never shown. Re-running the failing test files with the trace plugin brought that to 1 of 48.
2. **Nano re-ranking did not beat the spectrum, so we stopped using Nano for it.** A
   ranking-only experiment (no patching, `results/rank-only-*`) put the bug in Nano's top 5 in
   67% of tasks when the candidates were shuffled and 96% when they were shown in spectrum
   order, while the spectrum order alone had it first in 83% and in the top 5 in 96%. Nano
   never rescued a bug the spectrum had ranked below 5th.
3. **Nano is worth more as the first patcher.** Half of the repairs (20 of 40) needed only
   Nano turns at about 2 seconds each. The other 20 came after Nano's patch had failed and
   Super, with reasoning, saw the test output. Compared with the previous configuration
   (Nano re-ranking, Super from the first turn), cost per task fell by about 20% and the repair
   count stayed level (40 vs 39). Turning Super's reasoning off
   instead saves more money but loses 10 repairs.

## Where Nemotron and Token Factory are used

| step | model (Token Factory id) | settings | measured |
|---|---|---|---|
| first repair attempt (up to 2 turns, 1 patch) | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | `enable_thinking: false`, JSON-schema actions | 2.3 s per turn, 95 turns cost $0.023 in total |
| escalation: read, patch, react to test output | `nvidia/nemotron-3-super-120b-a12b` | `enable_thinking: true` (4096-token cap), JSON-schema actions | 17.9 s per turn, 78 turns cost $0.37 in total |
| ranking when no trace is available (no passing tests, timeout) | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | thinking off, JSON schema | about 1 s, $0.0003 per call |

All calls go to the OpenAI-compatible endpoint `https://api.tokenfactory.nebius.com/v1/chat/completions`
([`mutafix/llm.py`](mutafix/llm.py)). The Token Factory features the design relies on:

1. **`chat_template_kwargs.enable_thinking`** turns reasoning on and off per call, so one model
   family covers the fast first attempt and the slow careful one. Super's reasoning is capped
   at 4096 tokens; if a reply hits the cap with no answer, MutaFix asks again with thinking off.
2. **Structured output (`response_format: json_schema`)** makes every reply a valid action,
   so the tool loop never has to repair malformed JSON.
3. **Per-token prices from `/v1/models`** are read at start-up. Every call is written to
   `ledger.jsonl` with its tokens, latency and real cost, and the run stops at a budget (`--budget`, USD).
4. **Retries on 429/5xx/529** with backoff. One benchmark run hit a 529 (overloaded) mid-task; the
   harness now skips the task, keeps going, and retries it on the next pass.

## Install and run

Requires Python 3.10+ and a Token Factory API key ([tokenfactory.nebius.com](https://tokenfactory.nebius.com)).
Python 3.12+ is recommended for the target project: the trace plugin uses `sys.monitoring`
there and falls back to `sys.setprofile` on older versions.

```bash
git clone https://github.com/kimdooo-a/mutafix && cd mutafix
uv venv && uv pip install -e .          # or: python -m venv .venv && pip install -e .
export NEBIUS_API_KEY=...               # PowerShell: $env:NEBIUS_API_KEY="..."
```

**Quick start (a few seconds, about $0.0002):** `examples/invoice` has a hand-written
off-by-one bug in `paginate`; two of its six tests fail.

```bash
mutafix run --repo examples/invoice --package invoice --dry-run --out runs/invoice
# open runs/invoice/invoice.html
```

`--dry-run` saves the patch to `runs/invoice/invoice.patch` and leaves the files as they were;
without it the fix stays applied.

**Your own repository:**

```bash
mutafix run --repo path/to/repo --package src/yourpkg --tests tests --python path/to/repo/.venv/bin/python
```

`--python` is the interpreter that has the project's dependencies; MutaFix runs
`python -m pytest` with it and puts its own trace plugin (standard library only) on `PYTHONPATH`.
`--tests` passes pytest arguments (default: the whole suite).

**One benchmark task, or the whole benchmark:**

```bash
mutafix fetch                                   # clone the 4 libraries at pinned commits into bench/repos
mutafix run --bench toolz --task toolz-008      # inject one bug, repair it, restore the repo
mutafix bench --per-repo 12 --out runs/bench    # 48 tasks; writes runs/bench/index.html
```

Flags for the experiments above: `--retrieval static` (no trace), `--rank llm` and
`--rank-order shuffled|ordered` (Nano re-ranking), `--cascade off` (Super from the first turn),
`--thinking off`, `--rank-only` (stop after localization), plus `--max-turns 8`,
`--max-patches 3`, `--budget 1.0`, and `--ranker` / `--patcher` to swap Token Factory models
(for example `nvidia/Nemotron-3-Ultra-550b-a55b` as the patcher).

## How it works

```
reproduce   pytest -rfE                          failing tests and their "E" error lines
trace       pytest -p mutafix.trace_plugin       per test: which library functions ran, pass/fail
            on the failing tests' files
localize    Ochiai = failed(f) / sqrt(total_failed * (failed(f) + passed(f)))
            ties and never-run functions ordered by static call distance from the test, then BM25
            -> 20 candidates, top 5 shown in full with "ran in 3/3 failing, 2/40 passing tests"
repair      tool loop with JSON actions:  read(k) | patch(k, old, new) | give_up
            turns 1-2: Nemotron 3 Nano (thinking off), one patch
            then:      Nemotron 3 Super (thinking on), same conversation, up to 3 patches
verify      after each patch: the failing tests, then the full suite
            red -> revert, send fixed/still failing/newly failing tests back to the model
report      static HTML trace + JSON + cost ledger
```

Safety limits: a patch can only replace text inside one indexed library function, so the
agent cannot edit tests or files outside the package. A patch that does not parse is rejected
before it reaches the disk. A failing patch is reverted at once, so each attempt starts from
the original code. The patch matcher tolerates the line-number gutter the model sometimes
copies and lost indentation, but it rejects any `old` text that matches more than one place.
Reads are capped at 3 new functions; re-reading code already in the conversation is allowed
and does not count.

## Honest limits

- **The benchmark bugs are injected one-line mutations** (negated conditions, swapped
  comparison or arithmetic operators, changed constants). Real regressions often span several
  lines or files; MutaFix patches one function per attempt.
- **"Repaired" means the full test suite passes.** 30 of the 40 repairs restore the original
  code exactly. Of the other 10, by our own reading of the diffs, 8 are equivalent rewrites
  (for example `if not seqs:` for `if len(seqs) == 0:`) and 2 are workarounds the tests do not
  catch: in toolz-012 the patch returns `None` where the original returned `False`, and in
  boltons-001 the patch edits a different function to hide the symptom. Every diff is in the
  per-task traces.
- **Where the 8 failures come from:** in 7 the buggy function was in the top 5 but neither
  model found a patch that turned the suite green within the budget (in more-itertools-008 a
  patch caused an infinite loop and the 120-second test timeout stopped it); in 1 (toolz-004)
  the buggy function was not among the 20 candidates.
- Each configuration was run once. Model output is not fully deterministic even at
  temperature 0 (on 19 tasks run under two nearly identical early configurations the count
  differed by one), so differences of one or two tasks between rows should be read as noise.
- The trace needs passing and failing tests in the same files. Without it MutaFix falls back to
  static retrieval and Nano ranking, which the table above shows is weaker.

## What was built during the hackathon

Pre-existing component: the **MutaLoc** task generator and the static retrieval code in
[`mutafix/index.py`](mutafix/index.py) (AST index, call graph, BM25), written in October 2026 by the
same author for a fault-localization study. MutaFix uses its task files as the benchmark.

New in the submission period: the repair agent and its tool loop with the Nano-to-Super
cascade ([`mutafix/agent.py`](mutafix/agent.py)), the trace plugin and the spectrum scoring
([`mutafix/trace_plugin.py`](mutafix/trace_plugin.py)), the Token Factory client with the cost
ledger, retries and budget stop ([`mutafix/llm.py`](mutafix/llm.py)), pytest execution and failure
parsing ([`mutafix/testrun.py`](mutafix/testrun.py)), the benchmark harness with injection,
AST-level scoring and restore ([`mutafix/bench.py`](mutafix/bench.py)), the HTML reports
([`mutafix/report.py`](mutafix/report.py)), the CLI, the quick-start example, the tests and all
experiments in `results/`.

## Tests

```bash
python -m pytest tests -q      # offline: patch application, retrieval, pytest parsing (no API calls)
```

## Credits and licenses

MutaFix is MIT licensed. Benchmark tasks quote single lines from four libraries that are cloned
(not vendored) at pinned commits: toolz (BSD-3-Clause), more-itertools (MIT), boltons
(BSD-3-Clause), sortedcontainers (Apache-2.0). See [`bench/tasks/NOTICE.md`](bench/tasks/NOTICE.md).
The code and documentation were written with the help of Claude Code, an AI coding assistant.
