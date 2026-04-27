#!/usr/bin/env python3
"""nano-vllm prefix-length × concurrency throughput sweep.

Mirror of ~/Desktop/setup/spark/bench_vllm.py and ~/Desktop/inference/scripts/bench.py
but talks to nano-vllm in-process via the LLM.generate batched API instead of
HTTP. Each (N, prefix_tokens) cell submits N requests sharing one random
prefix; nano-vllm's prefix cache is expected to absorb the prefill cost
after the first request.

Per-request output length is pinned via ignore_eos=True and max_tokens; the
output budget per cell is GEN_BUDGET_PER_CELL split N ways and clamped to
[OUTPUT_TOKENS_MIN, OUTPUT_TOKENS_MAX].

Run from project root:
    uv run python bench/bench_concurrency.py
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

from nanovllm import LLM, SamplingParams


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


def sweep_cell(llm: LLM, N: int, prefix_tokens: int, seed_base: int):
    shared_prefix = random_token_ids(prefix_tokens, seed=seed_base)
    out_tok = output_tokens_for(N)
    prompts = [list(shared_prefix) for _ in range(N)]
    sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=out_tok) for _ in range(N)]
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sps, use_tqdm=False)
    wall = time.perf_counter() - t0
    c_tok = sum(len(o["token_ids"]) for o in outputs)
    p_tok = prefix_tokens * N
    return c_tok / wall, N, p_tok, c_tok, wall


def cold_prefill(llm: LLM, prefix_tokens: int, seed: int) -> float:
    prompt = random_token_ids(prefix_tokens, seed=seed)
    sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=1)
    t0 = time.perf_counter()
    llm.generate([prompt], [sp], use_tqdm=False)
    return time.perf_counter() - t0


def warmup(llm: LLM) -> None:
    print("# warmup…", file=sys.stderr)
    sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=8)
    llm.generate([random_token_ids(32, seed=99_000)], [sp], use_tqdm=False)


def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    ap.add_argument("--max-model-len", type=int, default=34816)  # 32768 + 2048 headroom
    ap.add_argument("--max-num-seqs", type=int, default=1024)
    ap.add_argument("--max-num-batched-tokens", type=int, default=16384)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--prefix-lengths", type=parse_int_list, default=PREFIX_LENGTHS_DEFAULT)
    ap.add_argument("--concurrencies", type=parse_int_list, default=CONCURRENCIES_DEFAULT)
    args = ap.parse_args()

    print(
        f"# model={args.model} max_model_len={args.max_model_len} "
        f"max_num_seqs={args.max_num_seqs} mbt={args.max_num_batched_tokens} "
        f"util={args.gpu_memory_utilization} eager={args.enforce_eager}",
        file=sys.stderr,
    )
    llm = LLM(
        args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    warmup(llm)

    header = "prefix \\ N            | " + " | ".join(f"{N:>5d}" for N in args.concurrencies) + " | prefill (s)"
    print(header)
    print("-" * len(header))

    seed_base = 1
    for L in args.prefix_lengths:
        prefill_s = cold_prefill(llm, L, seed=seed_base + 50_000)
        row = [f"{L:>22d}"]
        for N in args.concurrencies:
            try:
                tps, n_req, p_tok, c_tok, wall = sweep_cell(llm, N, L, seed_base)
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
        print("  | ".join(row) + f" | {prefill_s:>6.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
