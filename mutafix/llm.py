"""Nebius Token Factory client (OpenAI-compatible chat completions) with a cost ledger.

Every call is appended to a JSONL ledger with its model, token usage, latency and
dollar cost computed from the per-token prices that /v1/models publishes. A budget
cap stops the run before it is exceeded.
"""
import json
import os
import threading
import time

import httpx

BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")

# Defaults: Nano ranks candidates (fast, cheap), Super writes patches.
RANKER_MODEL = os.environ.get("MUTAFIX_RANKER_MODEL", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B")
PATCHER_MODEL = os.environ.get("MUTAFIX_PATCHER_MODEL", "nvidia/nemotron-3-super-120b-a12b")

# Used only if /v1/models cannot be read: deliberately pessimistic ($ per token).
FALLBACK_PRICE = {"prompt": 0.5e-6, "completion": 2.0e-6}


class BudgetExceeded(RuntimeError):
    pass


class TokenFactory:
    def __init__(self, ledger_path=None, budget_usd=None, api_key=None, timeout=180):
        self.key = api_key or os.environ.get("NEBIUS_API_KEY")
        if not self.key:
            raise SystemExit("NEBIUS_API_KEY is not set (get a key at https://tokenfactory.nebius.com).")
        self.http = httpx.Client(base_url=BASE_URL, timeout=timeout,
                                 headers={"Authorization": f"Bearer {self.key}"})
        self.ledger_path = ledger_path
        self.budget = budget_usd
        self.spent = 0.0
        self.calls = []
        self._lock = threading.Lock()
        self.prices = self._load_prices()
        if ledger_path and os.path.exists(ledger_path):
            for line in open(ledger_path, encoding="utf-8"):
                self.spent += json.loads(line).get("cost_usd", 0.0)

    def _load_prices(self):
        try:
            r = self.http.get("/models", params={"verbose": "true"})
            r.raise_for_status()
            out = {}
            for m in r.json().get("data", []):
                p = m.get("pricing") or {}
                if p.get("prompt") is not None:
                    out[m["id"]] = {"prompt": float(p["prompt"]), "completion": float(p["completion"])}
            return out
        except Exception:
            return {}

    def price(self, model):
        return self.prices.get(model, FALLBACK_PRICE)

    def chat(self, model, messages, *, max_tokens=1024, temperature=0.0, thinking=False,
             schema=None, tag=""):
        if self.budget is not None and self.spent >= self.budget:
            raise BudgetExceeded(f"budget ${self.budget:.2f} reached (spent ${self.spent:.4f})")
        body = {"model": model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature,
                "chat_template_kwargs": {"enable_thinking": bool(thinking)}}
        if schema is not None:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "answer", "schema": schema}}
        t0 = time.time()
        last = None
        for attempt in range(6):
            try:
                r = self.http.post("/chat/completions", json=body)
                if r.status_code in (429, 500, 502, 503, 504, 529):  # 529 = overloaded
                    last = f"HTTP {r.status_code}: {r.text[:200]}"
                    time.sleep(min(60, 4 * 2 ** attempt))
                    continue
                r.raise_for_status()
                data = r.json()
                break
            except httpx.TransportError as e:
                last = repr(e)
                time.sleep(min(60, 4 * 2 ** attempt))
        else:
            raise RuntimeError(f"Token Factory call failed: {last}")
        wall = time.time() - t0
        msg = data["choices"][0]["message"]
        usage = data.get("usage") or {}
        pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        pr = self.price(model)
        cost = pt * pr["prompt"] + ct * pr["completion"]
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": tag, "model": model,
               "prompt_tokens": pt, "completion_tokens": ct,
               "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
               "wall_s": round(wall, 2), "cost_usd": cost,
               "finish_reason": data["choices"][0].get("finish_reason")}
        with self._lock:
            self.spent += cost
            self.calls.append(rec)
            if self.ledger_path:
                with open(self.ledger_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        return {"content": content, "reasoning": reasoning, **rec}
