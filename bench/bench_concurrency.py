#!/usr/bin/env python3
"""Throughput sweep against nano-vllm's OpenAI-compatible server.

Mirror of `~/Desktop/setup/spark/bench_vllm.py`. Each (N, prefix_tokens)
cell sends N concurrent requests sharing one random prefix to
`/v1/completions`; nano-vllm's prefix cache is expected to absorb the
prefill cost after the first request. Per-request output length is
pinned via `ignore_eos=True` and `max_tokens`, with the output budget per
cell GEN_BUDGET_PER_CELL split N ways and clamped to
[OUTPUT_TOKENS_MIN, OUTPUT_TOKENS_MAX].

A separate per-row `cold_prefill` probe (output_tokens=1, fresh seed so
the prefix cache is cold) reports prefill time as the last column —
matches the prefill measurement in `~/Desktop/setup/spark/bench_prefill.py`,
folded into the same table here for convenience.

Start the server first::

    uv run python -m nanovllm.server --model ~/huggingface/Qwen3-0.6B/ \
        --gpu-memory-utilization 0.65 --max-num-batched-tokens 16384

Then run from project root::

    uv run python bench/bench_concurrency.py
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import sys
import threading
import time
import urllib.request


# Default thread stack is 8 MB; with 1024 threads that's 8 GB of VM, enough
# to draw OOM-killer attention on a box where the engine owns most of the
# unified memory.
threading.stack_size(512 * 1024)


PREFIX_LENGTHS_DEFAULT = [1, 4096, 32768]
CONCURRENCIES_DEFAULT = [1, 4, 16, 64, 256, 1024]
GEN_BUDGET_PER_CELL = 128 * 1024  # 131 072 total output tokens per cell
OUTPUT_TOKENS_MIN = 64
OUTPUT_TOKENS_MAX = 1024
TOK_LO, TOK_HI = 1000, 100_000  # safe vocab range for Qwen3 (vocab 151 936)


def random_token_ids(n_tokens: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(TOK_LO, TOK_HI) for _ in range(n_tokens)]


def output_tokens_for(N: int) -> int:
    return max(OUTPUT_TOKENS_MIN, min(OUTPUT_TOKENS_MAX, GEN_BUDGET_PER_CELL // N))


def one(url: str, model: str, prompt_ids: list[int], output_tokens: int, api_key: str):
    body = json.dumps({
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": output_tokens,
        "min_tokens": output_tokens,   # vLLM extension; nano-vllm ignores
        "ignore_eos": True,
        "temperature": 0.0,
        "stop": [],
    }).encode()
    headers = {"Content-Type": "application/json", "User-Agent": "curl/8.5.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=body, headers=headers)
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read())
    dt = time.perf_counter() - t0
    u = data["usage"]
    return dt, u["prompt_tokens"], u["completion_tokens"]


def sweep_cell(url: str, model: str, N: int, prefix_tokens: int, seed_base: int, api_key: str):
    shared_prefix = random_token_ids(prefix_tokens, seed=seed_base)
    out_tok = output_tokens_for(N)
    plan = [(shared_prefix, out_tok) for _ in range(N)]
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=N) as ex:
        futs = [ex.submit(one, url, model, p, k, api_key) for (p, k) in plan]
        results = [f.result() for f in cf.as_completed(futs)]
    wall = time.perf_counter() - t0
    p_tok = sum(r[1] for r in results)
    c_tok = sum(r[2] for r in results)
    return c_tok / wall, len(plan), p_tok, c_tok, wall


def cold_prefill(url: str, model: str, prefix_tokens: int, seed: int, api_key: str) -> float:
    """Wall time of a single cold-cache request asking for one decode step.

    Wall ≈ prefill + ~1 decode step. One decode step is tens of ms — a rounding
    error against any non-trivial prefill, so this column reads as prefill.
    """
    prompt = random_token_ids(prefix_tokens, seed=seed)
    dt, _, _ = one(url, model, prompt, output_tokens=1, api_key=api_key)
    return dt


def warmup(url: str, model: str, api_key: str) -> None:
    print("# warmup…", file=sys.stderr)
    one(url, model, random_token_ids(64, seed=99_000), output_tokens=8, api_key=api_key)


def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1/completions")
    ap.add_argument(
        "--model",
        default=os.environ.get("BENCH_MODEL", os.path.expanduser("~/huggingface/Qwen3-0.6B/")),
        help="Model id passed in the request body (must match what the server loaded).",
    )
    ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""))
    ap.add_argument("--prefix-lengths", type=parse_int_list, default=PREFIX_LENGTHS_DEFAULT)
    ap.add_argument("--concurrencies", type=parse_int_list, default=CONCURRENCIES_DEFAULT)
    args = ap.parse_args()

    print(
        f"# url={args.url} model={args.model} prefixes={args.prefix_lengths} "
        f"concurrencies={args.concurrencies}",
        file=sys.stderr,
    )
    warmup(args.url, args.model, args.api_key)

    header = (
        "prefix \\ N            | "
        + " | ".join(f"{N:>5d}" for N in args.concurrencies)
        + " | prefill (s)"
    )
    print(header)
    print("-" * len(header))

    seed_base = 1
    for L in args.prefix_lengths:
        prefill_s = cold_prefill(args.url, args.model, L, seed=seed_base + 50_000, api_key=args.api_key)
        row = [f"{L:>22d}"]
        for N in args.concurrencies:
            try:
                tps, n_req, p_tok, c_tok, wall = sweep_cell(
                    args.url, args.model, N, L, seed_base, args.api_key
                )
                row.append(f"{tps:>5.0f}")
                print(
                    f"#   L={L:>5} N={N:>3}: {n_req:>4} reqs, p={p_tok:>8d}, "
                    f"c={c_tok:>7d}, wall={wall:>6.1f}s, out_tps={tps:>6.1f}",
                    file=sys.stderr,
                )
            except Exception as e:
                row.append("  ERR")
                print(f"#   L={L} N={N}: {e}", file=sys.stderr)
            seed_base += N + 7
        print("  | ".join(row) + f" | {prefill_s:>6.2f}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
