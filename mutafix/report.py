"""Static HTML reports: one page per repair run, one index page per benchmark."""
import html
import json
import os

CSS = """
:root{--bg:#fbfbf9;--fg:#1d1f21;--mute:#6a6f75;--line:#e3e3de;--card:#ffffff;--ok:#1f7a3f;--okbg:#e7f4ea;
--bad:#a8321f;--badbg:#fbe9e5;--acc:#5b3fd1;--code:#f4f4f0;--add:#e6f4e8;--del:#fbe8e6}
@media (prefers-color-scheme:dark){:root{--bg:#141517;--fg:#e6e6e3;--mute:#9a9ea3;--line:#2c2e31;--card:#1b1c1f;
--ok:#6fcf8e;--okbg:#17301f;--bad:#f08a75;--badbg:#3a1d17;--acc:#a993ff;--code:#202226;--add:#173022;--del:#3a1d1a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1060px;margin:0 auto;padding:28px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:34px 0 10px}.sub{color:var(--mute);margin:0 0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:10px 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi b{display:block;font-size:24px}.kpi span{color:var(--mute);font-size:13px}
.step{display:grid;grid-template-columns:120px minmax(0,1fr);gap:14px;border-top:1px solid var(--line);padding:14px 0}
.step:first-child{border-top:0}.tag{font:600 12px/1 Consolas,"Cascadia Mono",Menlo,monospace;text-transform:uppercase;
letter-spacing:.04em;color:var(--acc);padding-top:4px}.meta{color:var(--mute);font-size:13px}
pre{background:var(--code);border-radius:8px;padding:10px 12px;overflow:auto;font:13px/1.45 Consolas,"Cascadia Mono",Menlo,monospace;margin:8px 0}
.dl{display:block}.thought{white-space:pre-wrap}.d-add{background:var(--add);display:block}.d-del{background:var(--del);display:block}.d-hdr{color:var(--mute);display:block}
.pill{display:inline-block;border-radius:99px;padding:2px 10px;font-size:12px;font-weight:600}
.ok{background:var(--okbg);color:var(--ok)}.bad{background:var(--badbg);color:var(--bad)}.neu{background:var(--code);color:var(--mute)}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}
th{color:var(--mute);font-weight:600;font-size:13px}td code,li code{font:12.5px Consolas,"Cascadia Mono",Menlo,monospace}
ol.cands{margin:6px 0;padding-left:22px}ol.cands li{margin:2px 0}.gt{font-weight:700}
details summary{cursor:pointer;color:var(--mute);font-size:13px}a{color:var(--acc)}
.wrap{overflow-x:auto}
"""


def esc(x):
    return html.escape(str(x))


def diff_html(diff):
    out = []
    for ln in (diff or "").splitlines():
        cls = "d-hdr" if ln.startswith(("---", "+++", "@@")) else "d-add" if ln.startswith("+") else \
            "d-del" if ln.startswith("-") else ""
        out.append(f'<span class="dl {cls}">{esc(ln) or " "}</span>')
    return '<pre class="diff">' + "".join(out) + "</pre>"


def counts(c):
    return ", ".join(f"{v} {k}" for k, v in (c or {}).items()) if isinstance(c, dict) else esc(c)


def pill(ok, yes="pass", no="fail"):
    return f'<span class="pill {"ok" if ok else "bad"}">{yes if ok else no}</span>'


def page(title, body):
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>{CSS}</style>'
            f'</head><body><main>{body}</main></body></html>')


def run_page(out):
    b = out.get("bench") or {}
    gt = b.get("gt_function")
    title = f"MutaFix run: {b.get('task_id') or out.get('root_name')}"
    status = out.get("status")
    head = [f"<h1>{esc(title)}</h1>",
            f'<p class="sub">ranker <code>{esc(out["ranker"])}</code> · patcher <code>{esc(out["patcher"])}</code> '
            f'· via Nebius Token Factory</p>',
            '<div class="kpis">',
            f'<div class="kpi"><b>{pill(status == "fixed", "fixed", status.replace("_", " "))}</b><span>outcome</span></div>',
            f'<div class="kpi"><b>{out.get("wall_s", 0):.0f}s</b><span>wall time</span></div>',
            f'<div class="kpi"><b>{out.get("llm_calls", 0)}</b><span>model calls</span></div>',
            f'<div class="kpi"><b>${out.get("cost_usd", 0):.4f}</b><span>API cost</span></div></div>']
    if b:
        m = b["mutation"]
        head.append('<div class="card"><b>Benchmark ground truth</b> (hidden from the agent)<br>'
                    f'<span class="meta">{esc(m["file"])}:{m["line"]} · operator {esc(b["operator"])} · '
                    f'function <code>{esc(gt)}</code></span>'
                    f'<pre><span class="d-del">- {esc(m["after"])}   (injected bug)</span>'
                    f'<span class="d-add">+ {esc(m["before"])}   (original)</span></pre>'
                    f'retriever rank {esc(b["gt_rank_retriever"])} of 20 · ranker position '
                    f'{esc(b["gt_rank_model"] or "not in top 5")} · patched the right function '
                    f'{pill(b["patched_gt_function"], "yes", "no")} · exact restore {pill(b["exact_restore"], "yes", "no")}</div>')
    steps = []
    for s in out["steps"]:
        k = s["kind"]
        if k == "reproduce":
            errs = "".join(f"<b>{esc(n)}</b><pre>{esc(chr(10).join(e) or '(no message)')}</pre>"
                           for n, e in list(s.get("errors", {}).items())[:2])
            body = (f'{s["n_failed"]} failing test(s) · {counts(s.get("counts"))} · {s["wall_s"]}s'
                    f'<details><summary>failing tests and errors</summary><pre>{esc(chr(10).join(s["failed"]))}</pre>{errs}</details>')
        elif k == "trace":
            if s.get("error"):
                body = f'trace of {esc(", ".join(s["files"]))}: {esc(s["error"])} · static retrieval used'
            else:
                body = (f're-ran {len(s["files"])} test file(s) with the trace plugin: {s["n_failed"]} failing and '
                        f'{s["n_passed"]} passing tests, {s["n_executed"]} library functions ran in a failing test · '
                        f'{s["wall_s"]}s<br><span class="meta">{esc(", ".join(s["files"]))}</span>')
        elif k == "retrieve":
            def cand(c):
                sp = (f' · Ochiai {c["ochiai"]:.2f} (failing {c["fail_hits"]}, passing {c["pass_hits"]})'
                      if c.get("ochiai") is not None else "")
                return (f'<li class="{"gt" if c["id"] == gt else ""}"><code>{esc(c["id"])}</code> '
                        f'<span class="meta">call depth {esc(c["depth"] or "-")}{sp}</span></li>')
            li = "".join(cand(c) for c in s["candidates"])
            how = ("ranked by spectrum score (Ochiai), call graph and BM25 break ties"
                   if s.get("n_executed") is not None else "ranked by call distance, BM25 breaks ties")
            body = (f'{s["n_functions"]} functions indexed, {s["n_reachable"]} reachable from '
                    f'<code>{esc(s["primary"])}</code> in the static call graph. Kept {len(s["candidates"])} candidates, {how}.'
                    f'<details><summary>candidates in retriever order</summary><ol class="cands">{li}</ol></details>')
        elif k == "spectrum_rank":
            li = "".join(f'<li class="{"gt" if r == gt else ""}"><code>{esc(r)}</code></li>' for r in s["ranking"])
            body = f'<span class="meta">spectrum order kept (no model call)</span><ol class="cands">{li}</ol>'
            k = "rank"
        elif k == "rank":
            li = "".join(f'<li class="{"gt" if r == gt else ""}"><code>{esc(r)}</code></li>' for r in s["ranking"])
            body = (f'<span class="meta">{esc(s["model"])} · {s["prompt_tokens"]} in / {s["completion_tokens"]} out · '
                    f'{s["wall_s"]}s · ${s["cost_usd"]:.5f}</span><ol class="cands">{li}</ol>')
        elif k == "turn":
            meta = (f'<span class="meta">turn {s["turn"]} · {esc(s.get("model", "").split("/")[-1])} · {s["prompt_tokens"]} in / {s["completion_tokens"]} out · '
                    f'{s["wall_s"]}s · ${s["cost_usd"]:.5f}</span>')
            a = s["action"]
            why = f'<p>{esc(s.get("why", ""))}</p>' if s.get("why") else ""
            thought = (f'<details><summary>model reasoning (excerpt)</summary><pre class="thought">{esc(s["thought"])}</pre></details>'
                       if s.get("thought") else "")
            if a == "read":
                body = f'{meta}<p>read <code>{esc(s.get("target"))}</code> {esc(s.get("error", ""))}</p>{why}{thought}'
            elif a == "patch" and s.get("error"):
                body = (f'{meta}<p>patch on <code>{esc(s.get("target"))}</code> rejected: {esc(s["error"])}</p>{why}'
                        f'<details><summary>the edit it proposed</summary><pre>{esc((s.get("edit") or {}).get("old", ""))}</pre>'
                        f'<pre>{esc((s.get("edit") or {}).get("new", ""))}</pre></details>{thought}')
            elif a == "patch":
                full = s["full"]
                res = (pill(True, "all tests pass") if full["ok"] else
                       pill(False, no=f'{len(full["failed"])} failing' + ("" if full["ran"] else " (target tests)")))
                fl = f'<pre>{esc(chr(10).join(full["failed"]))}</pre>' if full["failed"] else ""
                body = f'{meta}{why}{diff_html(s["diff"])}<p>tests: {res} <span class="meta">{counts(full.get("counts"))}</span></p>{fl}{thought}'
            else:
                body = f'{meta}<p>{esc(a)}: {esc(s.get("why") or s.get("raw") or s.get("error", ""))}</p>{thought}'
            k = a
        else:
            body = esc(json.dumps({x: y for x, y in s.items() if x not in ("kind", "t")}))
        steps.append(f'<div class="step"><div class="tag">{esc(k)}<div class="meta">+{s["t"]}s</div></div><div>{body}</div></div>')
    fix = out.get("fix")
    tail = f"<h2>Final patch</h2>{diff_html(fix['diff'])}" if fix else ""
    return page(title, "".join(head) + '<h2>Trace</h2><div class="card">' + "".join(steps) + "</div>" + tail)


def bench_page(summary, rows, links):
    s = summary
    k = [("repaired (all tests green)", f'{s["fixed"]}/{s["n"]} ({s["fixed_pct"]}%)'),
         ("exact restore of the original line", f'{s["exact_restore"]}/{s["n"]}'),
         ("ranker top-1 = buggy function", f'{s["ranker_top1_pct"]}%'),
         ("retriever top-1 (no model)", f'{s["retriever_top1_pct"]}%'),
         ("bug in the 20 candidates", f'{s["gt_in_candidates_pct"]}%'),
         ("API cost per task", f'${s["cost_per_task_usd"]:.4f}'),
         ("mean ranker call", f'{s["mean_rank_call_s"]}s'),
         ("mean patcher call", f'{s["mean_patch_call_s"]}s')]
    kp = "".join(f'<div class="kpi"><b>{esc(v)}</b><span>{esc(n)}</span></div>' for n, v in k)
    trs = []
    for r, link in zip(rows, links):
        b = r["bench"]
        trs.append(f'<tr><td><a href="{esc(link)}">{esc(b["task_id"])}</a></td><td>{esc(b["operator"])}</td>'
                   f'<td><code>{esc((b["gt_function"] or "").split("::")[-1])}</code></td>'
                   f'<td>{esc(b["gt_rank_model"] or "-")}</td>'
                   f'<td>{pill(r["status"] == "fixed", "fixed", r["status"].replace("_", " "))}</td>'
                   f'<td>{"exact" if b["exact_restore"] else ("plausible" if r["status"] == "fixed" else "-")}</td>'
                   f'<td>{sum(1 for x in r["steps"] if x["kind"] == "turn")}</td>'
                   f'<td>{r["wall_s"]:.0f}s</td><td>${r.get("cost_usd", 0):.4f}</td></tr>')
    repo_rows = "".join(f'<tr><td>{esc(n)}</td><td>{v["n"]}</td><td>{v["fixed"]}</td><td>{v["top1"]}</td></tr>'
                        for n, v in s["by_repo"].items())
    body = (f'<h1>MutaFix benchmark</h1><p class="sub">{s["n"]} injected one-line bugs in 4 open-source Python '
            f'libraries · Nemotron on Nebius Token Factory · generated {esc(s["generated"])}</p>'
            f'<div class="kpis">{kp}</div>'
            f'<h2>By repository</h2><div class="card wrap"><table><tr><th>repository</th><th>tasks</th><th>repaired</th>'
            f'<th>ranker top-1</th></tr>{repo_rows}</table></div>'
            f'<h2>Every task</h2><div class="card wrap"><table><tr><th>task</th><th>bug type</th><th>buggy function</th>'
            f'<th>ranker pos.</th><th>outcome</th><th>fix</th><th>turns</th><th>time</th><th>cost</th></tr>'
            f'{"".join(trs)}</table></div>'
            f'<p class="meta">"exact" = the patch restored the original line. "plausible" = all tests pass with a '
            f'different edit; the test suite cannot tell it apart from the original. Total cost ${s["cost_usd"]:.4f} for '
            f'{s["tokens"]["prompt"]:,} prompt and {s["tokens"]["completion"]:,} completion tokens.</p>')
    return page("MutaFix benchmark", body)


def write_run(out, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    open(path, "w", encoding="utf-8").write(run_page(out))
