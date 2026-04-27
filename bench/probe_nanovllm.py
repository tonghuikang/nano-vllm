#!/usr/bin/env python3
"""Focused probe: re-run a single (L, N) cell under varying scheduler knobs
to localise where nano-vllm trails vLLM.

Tests:
  A. baseline (max_num_batched_tokens=16384, prefix cold)
  B. max_num_batched_tokens=2048 (force serial cold prefill so prefix-cache
     race only burns 1 prompt instead of 4)
  C. warm prefix (run a pre-pass that establishes the cache, then time only
     decode-only workload)
  D. Same as A but max_num_seqs=2048 (hypothesis: decode is throttled by
     max_num_seqs default of 512 once admitted seqs > 512)
"""
import os, sys, time, random, gc, argparse
import torch
from nanovllm import LLM, SamplingParams

PREFIX = 4096
N = 64
OUTPUT = max(64, min(1024, 128 * 1024 // N))   # 1024 for N=64
TOK_LO, TOK_HI = 1000, 100_000
MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B/")


def random_token_ids(n_tokens, seed):
    rng = random.Random(seed)
    return [rng.randint(TOK_LO, TOK_HI) for _ in range(n_tokens)]


def time_cell(llm, n, prefix_len, output_tokens, seed, label, warm_prefix=False):
    prefix = random_token_ids(prefix_len, seed=seed)
    if warm_prefix:
        # one priming request to warm the prefix cache
        sp_w = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=1)
        llm.generate([list(prefix)], [sp_w], use_tqdm=False)
    prompts = [list(prefix) for _ in range(n)]
    sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_tokens) for _ in range(n)]
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sps, use_tqdm=False)
    dt = time.perf_counter() - t0
    c_tok = sum(len(o["token_ids"]) for o in outs)
    print(f"[{label}] L={prefix_len} N={n} out={output_tokens}  wall={dt:.1f}s  c={c_tok}  tps={c_tok/dt:.1f}", flush=True)
    return c_tok / dt


def fresh(**kwargs):
    """Spin up a fresh LLM with given knobs, run probe, return tps. Writes to LLM destructor on return."""
    cfg = dict(model=MODEL, max_model_len=8192, max_num_seqs=1024, max_num_batched_tokens=16384,
               gpu_memory_utilization=0.65, enforce_eager=False)
    cfg.update(kwargs)
    print(f"\n=== {cfg} ===", flush=True)
    return LLM(**cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["A", "B", "C", "D", "E"], required=True)
    args = ap.parse_args()

    if args.mode == "A":
        llm = fresh()
        time_cell(llm, N, PREFIX, OUTPUT, seed=10, label="A: baseline mbt=16384")

    elif args.mode == "B":
        llm = fresh(max_num_batched_tokens=2048)
        time_cell(llm, N, PREFIX, OUTPUT, seed=20, label="B: mbt=2048 (force serial cold prefill)")

    elif args.mode == "C":
        llm = fresh()
        time_cell(llm, N, PREFIX, OUTPUT, seed=30, label="C: warm prefix (cache primed)", warm_prefix=True)

    elif args.mode == "D":
        # max_num_seqs already 1024; rerun with 256 to show throttling
        llm = fresh(max_num_seqs=256)
        time_cell(llm, N, PREFIX, OUTPUT, seed=40, label="D: max_num_seqs=256")

    elif args.mode == "E":
        # eager-only (no torch.compile / CUDA graph)
        llm = fresh(enforce_eager=True)
        time_cell(llm, N, PREFIX, OUTPUT, seed=50, label="E: enforce_eager=True")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
