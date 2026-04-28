#!/usr/bin/env python3
"""Throughput sweep against vLLM's /v1/completions endpoint.

Mirror of `bench_concurrency.py` but pointed at vLLM's HTTP server. Each
(N, prefix_tokens) cell sends N concurrent requests sharing one random
prefix; vLLM's prefix cache absorbs the prefill cost after the first
request and the cell measures decode throughput at concurrency N.

Output length is pinned via `min_tokens=max_tokens` and `ignore_eos=True`
so each request emits exactly `clamp(131072/N, 64, 1024)` tokens. The
last column reports cold-cache prefill wall time at each prefix length
— matches `~/Desktop/setup/spark/bench_prefill.py` folded into the same
table.
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import sys
import threading
import time
import urllib.request

# Default thread stack is 8 MB; 1024 threads × 8 MB = 8 GB of VM, enough to
# draw OOM-killer attention on a box where vLLM owns most of the unified
# memory. 512 KB is plenty for a thread that just does a urlopen.
threading.stack_size(512 * 1024)

URL_DEFAULT = "http://localhost:8000/v1/completions"
MODEL_DEFAULT = "Qwen/Qwen3-0.6B"

OUTPUT_TOKENS_MIN = 64
OUTPUT_TOKENS_MAX = 1024
GEN_BUDGET_PER_CELL = 128 * 1024  # 131 072 total output tokens per cell

PREFIX_LENGTHS_DEFAULT = [1, 4096, 32768]
CONCURRENCIES_DEFAULT = [1, 4, 16, 64, 256, 1024]

TOK_LO, TOK_HI = 1000, 100_000


def random_token_ids(n_tokens, seed):
    rng = random.Random(seed)
    return [rng.randint(TOK_LO, TOK_HI) for _ in range(n_tokens)]


def one(url, model, prompt_ids, output_tokens, api_key):
    body = json.dumps({
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": output_tokens,
        "min_tokens": output_tokens,
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


def output_tokens_for(N):
    return max(OUTPUT_TOKENS_MIN, min(OUTPUT_TOKENS_MAX, GEN_BUDGET_PER_CELL // N))


def sweep_cell(url, model, N, prefix_tokens, seed_base, api_key):
    shared_prefix = random_token_ids(prefix_tokens, seed=seed_base)
    out_tok = output_tokens_for(N)
    plan = [(shared_prefix, out_tok) for _ in range(N)]
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=N) as ex:
        futs = [ex.submit(one, url, model, p, k, api_key) for (p, k) in plan]
        results = [f.result() for f in cf.as_completed(futs)]
    wall = time.perf_counter() - t0
    c_tok = sum(r[2] for r in results)
    p_tok = sum(r[1] for r in results)
    return c_tok / wall, len(plan), p_tok, c_tok, wall


def cold_prefill(url, model, prefix_tokens, seed, api_key):
    """Single cold-cache request asking for one decode step. Wall time
    ≈ prefill (one decode is tens of ms — rounding error against any
    non-trivial prefill)."""
    prompt = random_token_ids(prefix_tokens, seed=seed)
    dt, _, _ = one(url, model, prompt, output_tokens=1, api_key=api_key)
    return dt


def warmup(url, model, api_key, max_concurrency):
    Nw = min(8, max_concurrency)
    shared_prefix = random_token_ids(64, seed=99_000)
    plan = [(shared_prefix, 16) for _ in range(Nw)]
    with cf.ThreadPoolExecutor(max_workers=Nw) as ex:
        list(ex.map(lambda pk: one(url, model, pk[0], pk[1], api_key), plan))
    print("#   warmup done", file=sys.stderr)


def parse_int_list(s):
    return [int(x) for x in s.split(",") if x.strip()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL_DEFAULT)
    ap.add_argument("--model", default=os.environ.get("BENCH_MODEL", MODEL_DEFAULT))
    ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""))
    ap.add_argument("--prefix-lengths", type=parse_int_list, default=PREFIX_LENGTHS_DEFAULT)
    ap.add_argument("--concurrencies", type=parse_int_list, default=CONCURRENCIES_DEFAULT)
    args = ap.parse_args()

    print(
        f"# url={args.url} model={args.model} prefixes={args.prefix_lengths} "
        f"concurrencies={args.concurrencies}",
        file=sys.stderr,
    )
    print("# Warming up engine…", file=sys.stderr)
    warmup(args.url, args.model, args.api_key, max(args.concurrencies))

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
