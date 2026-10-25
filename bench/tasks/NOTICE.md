# Benchmark task files

Each line describes one injected bug in a pinned commit of an open-source library:
the file, line number, the original line and the mutated line, the failing test and
its error lines. Only these single quoted lines are taken from the libraries; the
libraries themselves are not included and are cloned by `mutafix fetch`.

| repository | license | commit |
|---|---|---|
| [pytoolz/toolz](https://github.com/pytoolz/toolz) | BSD-3-Clause | 451af60 |
| [more-itertools/more-itertools](https://github.com/more-itertools/more-itertools) | MIT | b21ec4e |
| [mahmoud/boltons](https://github.com/mahmoud/boltons) | BSD-3-Clause | 4e5faa3 |
| [grantjenks/python-sortedcontainers](https://github.com/grantjenks/python-sortedcontainers) | Apache-2.0 | 3ac3586 |

The tasks were produced by the MutaLoc generator (October 2026, same author), which
injects one single-line semantic mutation (negated condition, swapped comparison or
arithmetic operator, changed boolean operator, changed small integer constant) into one
function and keeps the mutant only if at least one test fails.
