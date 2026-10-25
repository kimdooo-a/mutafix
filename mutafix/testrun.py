"""Run pytest in a subprocess and parse which tests failed and why."""
import os
import re
import subprocess
import sys
import time

SUMMARY_RE = re.compile(r"^(FAILED|ERROR) (\S+?)(?: - (.*))?$")
HEADER_RE = re.compile(r"^_{3,} (.+?) _{3,}$")
COUNTS_RE = re.compile(r"(\d+) (passed|failed|errors?|skipped|deselected|xfailed|xpassed)")


def run_pytest(root, args, opts=(), timeout=300, python=None, env_extra=None):
    """Return a dict: ok, failed (node ids), errors_by_test, counts, wall_s, tail."""
    cmd = [python or sys.executable, "-m", "pytest", "-q", "-rfE", "--tb=short", "--color=no",
           "-p", "no:cacheprovider", *opts, *args]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0", **(env_extra or {}))
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=root, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout, env=env)
        out, rc, timed_out = p.stdout + p.stderr, p.returncode, False
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else (e.stdout or b"").decode("utf-8", "replace")
        rc, timed_out = -1, True
    wall = time.time() - t0
    lines = out.splitlines()
    failed, short = [], {}
    for ln in lines:
        m = SUMMARY_RE.match(ln.strip())
        if m:
            nid = m.group(2)
            if nid not in failed:
                failed.append(nid)
                short[nid] = m.group(3) or ""
    # per-test failure blocks ("____ test_name ____" ... until the next header)
    blocks, cur, buf = {}, None, []
    for ln in lines:
        h = HEADER_RE.match(ln)
        if h or ln.startswith("=") and "short test summary" in ln:
            if cur:
                blocks[cur] = buf
            cur, buf = (h.group(1) if h else None), []
        elif cur:
            buf.append(ln)
    if cur:
        blocks[cur] = buf
    errors = {}
    for nid in failed:
        tail = nid.split("::")[-1]
        key = next((k for k in blocks if k.endswith(tail) or k.replace(".", "::").endswith(tail)), None)
        blk = blocks.get(key, [])
        e_lines = [l for l in blk if l.startswith("E ")][:15]
        errors[nid] = {"e_lines": e_lines or ([f"E   {short.get(nid, '')}"] if short.get(nid) else []),
                       "block": blk[-40:]}
    counts = {}
    for n, k in COUNTS_RE.findall("\n".join(lines[-3:])):
        counts[k.rstrip("s") if k.startswith("error") else k] = int(n)
    if timed_out:
        failed = failed or ["<timeout>"]
    ok = rc == 0 and not failed
    return {"ok": ok, "rc": rc, "timed_out": timed_out, "failed": failed, "errors": errors,
            "counts": counts, "wall_s": round(wall, 2), "tail": "\n".join(lines[-12:])}
