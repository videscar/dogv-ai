#!/usr/bin/env python3
"""Warm the chat server's long-prefill attention kernel.

vLLM + FlashInfer JIT-compiles its prefill kernel for a given shape on the FIRST
request that hits it. For this RAG app the real prompts are ~16k tokens, so the
first user question after a (re)start otherwise pays a one-time ~8-10s compile.
Firing one large prompt at startup moves that cost off the user's first query.

Stdlib only, so it can run as a systemd ExecStartPost with any python3.
Idempotent and best-effort: never fails the unit (always exits 0).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request


def build_prompt(approx_tokens: int) -> str:
    # ~1.6 tokens/word for this Spanish legal filler; size to the target.
    unit = ("Articulo {n}. La Generalitat Valenciana, en el ejercicio de sus competencias, "
            "regula mediante las bases reguladoras las condiciones, requisitos y plazos de "
            "justificacion de las subvenciones publicadas en el Diari Oficial de la Generalitat "
            "Valenciana para el fomento del desarrollo rural sostenible. ")
    words_target = int(approx_tokens / 1.6)
    out, n, words = [], 1, 0
    while words < words_target:
        s = unit.format(n=n)
        out.append(s)
        words += len(s.split())
        n += 1
    out.append("\nPregunta: resume en una frase las obligaciones de los beneficiarios.")
    return "".join(out)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("WARM_CHAT_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--model", default=os.environ.get("WARM_CHAT_MODEL", "qwen3.6-27b"))
    ap.add_argument("--tokens", type=int, default=int(os.environ.get("WARM_CHAT_TOKENS", "16000")))
    ap.add_argument("--timeout", type=float, default=120.0)
    a = ap.parse_args()

    prompt = build_prompt(a.tokens)
    body = {
        "model": a.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        a.base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=a.timeout) as r:
            o = json.load(r)
        pt = (o.get("usage") or {}).get("prompt_tokens")
        print(f"warm_chat_longctx.ok prompt_tokens={pt} elapsed={time.time()-t0:.2f}s")
    except Exception as e:  # best-effort: never block startup
        print(f"warm_chat_longctx.skip error={type(e).__name__}: {e} elapsed={time.time()-t0:.2f}s",
              file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
