# Cascade decode — what it is and why KV caching doesn't replace it

## What it is

Optimization for batches where N decode sequences share a common prefix in the
KV cache (which is exactly what `bench/bench_concurrency.py` creates: one
random prefix shared by all N requests).

Code: `nanovllm/layers/attention.py:_cascade_decode`. Gate:
`N · shared_prefix_len ≥ 32 K`, in `ModelRunner.plan_cascade_decode`.

Per-layer per-step, replaces the standard paged-attention call with three
passes:

1. **Prefix pass** — all N queries treated as one varlen "sequence" attending
   *non-causally* over the shared prefix blocks. Kernel loads the prefix K/V
   *once* and reuses it across the N queries from L1/L2/SMEM.
2. **Suffix pass** — each seq's query attends *causally* over its own unique
   tail (`block_table[num_shared_blocks:]`).
3. **Merge** — combine the two outputs via softmax-LSE composition.
   `out = (p_se · out_p + s_se · out_s) / (p_se + s_se)` where
   `p_se = exp(lse_p - max(lse_p, lse_s))`, similarly for `s_se`.

All three passes run inside one `Attention.forward()` call for one decode step.
Repeated 28 times (one per layer) per step.

## Synchronization requirements

- The N sequences must be in the same scheduler batch (same `prepare_decode`
  call) — that's the whole point.
- The merge formula is only valid when `out_p` and `out_s` come from the
  **same query** attending over disjoint key spans. Different queries → no
  valid merge. So all three passes are tightly coupled in the same step.
- The K/V cache itself is persistent across steps; cascade just reads it
  differently.

## Why KV caching doesn't already cover this

KV caching avoids *recompute*. Cascade avoids *redundant HBM reads*.

Even with KV caching, standard decode reads the prefix K/V **once per
sequence per step**. For Qwen3-0.6B with N=1024 sharing a 4k-token prefix:

- Prefix K/V size: `4096 tokens · 28 layers · 8 KV heads · 128 dim · 2 bytes · 2 (K+V) ≈ 47 MB / layer`, ~1.3 GB across all layers.
- Without cascade: each of 1024 seqs reads its slice → effectively
  ~`47 MB · 1024 = 48 GB / layer / step` from HBM.
- With cascade: kernel reads it once, broadcasts across 1024 queries →
  ~47 MB / layer / step.

At Spark's ~270 GB/s HBM bandwidth that's the difference between ~5 ms and
~0.2 ms per layer for prefix reads alone. Across 28 layers and many decode
steps, it dominates.

So: caching avoids recompute; cascade avoids re-reading the same cached
bytes N times.

## References

- **Hydragen** — Juravsky et al., 2024. arxiv:2402.05099. Original
  prefix/suffix decomposition + LSE merge for shared-prefix batch decoding.
- **FlashInfer** — Ye et al., 2025. arxiv:2501.01005, §2.2. Derives the LSE
  composition formula in clean form (the source of the comment in
  `attention.py:78`). Also the source of the
  `MultiLevelCascadeAttentionWrapper` and
  `BatchDecodeWithSharedPrefixPagedKVCacheWrapper` kernels that the
  `flashinfer` / `flashinfer_shared` opt-in backends call into.
- The underlying online-softmax / chunked-attention math that LSE
  composition rests on goes back to FlashAttention-2 (Dao, 2023).
