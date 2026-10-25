# Experiment results

Each folder holds `results.jsonl` (one full agent trace per task: steps, diffs, tests,
tokens, cost), `ledger.jsonl` (every Token Factory call with its usage and cost) and
`summary.json` (the aggregate, rebuilt with `mutafix report results/<folder>`).
Local paths and the user name were removed from error messages before publishing.

| folder | configuration | tasks |
|---|---|---|
| `final` | trace + Ochiai order kept, Nano first patch, Super (thinking on) takes over | 48 |
| `ablation-v3-llm-rank` | trace + Ochiai, Nano re-ranks shuffled candidates, Super from turn 1 | 48 |
| `ablation-v3-no-thinking` | as above, Super with thinking off | 48 |
| `ablation-v2-static` | static call graph + BM25 only, Nano re-ranks, Super from turn 1 | 48 |
| `ablation-v1-first19` | earlier patch matcher and read policy (stopped after 19 tasks) | 19 |
| `rank-only-shuffled` / `rank-only-ordered` | localization only: Nano ranks the 20 trace candidates, shuffled or in spectrum order | 48 each |
| `pilot-thinking-on` / `pilot-thinking-off` | first pilot, 6 tasks | 6 each |

The per-task HTML traces of `final` are published under [`../docs/`](../docs/) (GitHub Pages).
