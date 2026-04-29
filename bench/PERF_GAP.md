# How nano-vllm closed the shared-prefix high-concurrency gap

Measured on **NVIDIA DGX Spark (GB10, sm_121, 128 GB unified memory)**, Qwen3-0.6B.
See [`bench/FULL_SWEEP_20260429.md`](FULL_SWEEP_20260429.md),
[`bench/bench_nanovllm.txt`](bench_nanovllm.txt), and
[`bench/bench_vllm.txt`](bench_vllm.txt) for the current full-sweep results.

Current status: nothing remains behind vLLM at the parity tolerance. The fresh
median-of-3 sweep has all 18 published cells at or above 0.95x vLLM, with the
weakest cell at 0.978x (`prefix=1,N=1024`).

Historically, the gap was concentrated on cells with **L ≥ 4 k AND N ≥ 16**,
while the L=1 no-shared-prefix row stayed at parity. That isolated the cause to
nano-vllm's handling of long shared-prefix concurrent workloads.

## What we found

The earlier draft of this doc blamed per-step Python↔GPU sync. Profiler
disproved it: 99.6 % of the L=4 k N=64 step is GPU compute, only 0.2 % is
Python prep / dispatch. The `.tolist()` line **is** a `cudaSync`, but what
the engine is actually waiting on is the GPU — paged-attention reads
were doing N independent fetches of the shared prefix's K/V every step,
and that dominates everything else at any meaningful (L, N).

### What we did

Implemented **two-pass shared-prefix attention with LSE merging** in
[`nanovllm/layers/attention.py`](../nanovllm/layers/attention.py),
following vLLM's pattern (refs:
[`vllm/v1/attention/backends/flash_attn.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flash_attn.py)
`cascade_attention()` and
[`vllm/v1/attention/ops/triton_merge_attn_states.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_merge_attn_states.py)).

In one decode step, when ≥ 2 running seqs share a non-trivial prefix:

1. **Prefix pass.** All N queries go through `flash_attn_varlen_func` as
   one varlen "sequence" attending non-causally over the shared prefix's
   paged blocks. The kernel loads the prefix's K/V once and reuses it
   across the N queries via L1/L2/SMEM. Per-step KV-cache HBM traffic
   for the prefix portion drops from O(N · L) to O(L).

2. **Suffix pass.** Each seq's query attends causally over its unique
   tail (block_table[num_shared_blocks:]) via `flash_attn_varlen_func`.
   Standard per-seq paged attention but over a much shorter context.

3. **Merge.** Compose the two outputs via softmax-LSE composition
   (arxiv:2501.01005 §2.2): `out = (p_se · out_p + s_se · out_s) /
   (p_se + s_se)` where `p_se = exp(lse_p - max(lse_p, lse_s))`.

Plumbing changes:
- [`nanovllm/utils/context.py`](../nanovllm/utils/context.py): added
  `shared_prefix_blocks`, `shared_prefix_len`, `tail_block_tables`,
  `tail_lens`, `tail_max_len` fields.
- [`nanovllm/engine/model_runner.py`](../nanovllm/engine/model_runner.py)
  `prepare_decode`: walks the running seqs' `block_table`s to find the
  longest common prefix (block_manager.xxhash already dedups full blocks,
  so identical block ids ⇒ identical content). Pre-computes
  `tail_max_len` on the host so the cascade kernel call doesn't need a
  per-step `.item()` sync.
- A gate skips cascade when `N · shared_prefix_len < 32 K` — empirically
  the crossover between "two extra kernel launches + an LSE merge" and
  "the per-seq kernel reads K/V N times". L=4 k N=4 (16 K) regresses
  with cascade on; L=32 k N=4 (128 K) is a win.
- Cascade decode has CUDA graph capture keyed on
  `(N, num_shared_blocks, max_tail_blocks)`. The tail table is sized for
  the largest context the batch can reach before completion, so long
  decode runs do not recapture each time they cross a KV block boundary.

Also landed:
- `model_runner.capture_cudagraph` now captures up to `max_num_seqs`
  (was capped at 512), so non-cascade decodes at N > 512 stop falling
  back to eager.
- `nanovllm.server` default `--gpu-memory-utilization` bumped from 0.5
  to 0.85. Profile of L=4 k N=1024 with the lower setting showed
  `bs=74` dominant (instead of the expected 1024) — the KV pool only
  fit ~1,915 blocks while the cell needs ~2,063, so the scheduler was
  preempting on every step. At 0.85 the pool holds ~2,805 blocks and
  bs=1024 stays steady.

### Historical bench impact

The L=1 row held parity (cascade gate kept it on the existing path).
Shared-prefix cells below are the original cascade measurements at
`gpu_memory_utilization=0.85`, which is enough for the ~2,063 blocks
L=4 k N=1024 needs without preemption). The N=64 rows also include the
fresh repeated HTTP medians after cascade CUDA graph capture:

| | pre-cascade | cascade eager | current | vLLM | Δ vs pre |
| --- | ---: | ---: | ---: | ---: | ---: |
| L=4 k N=64    | 1208 | 1565 | 2271.6 | 2066.3 | +88 % |
| L=4 k N=256   | 1877 | 2724 | - | 4694 | +45 % |
| L=4 k N=1024  | 1874 | 3474 | - | 8295 | +85 % |
| L=32 k N=64   | 266  | 979  | 1308.4 | 1131.0 | +392 % |
| L=32 k N=256  | 286  | 1762 | - | 2476 | +516 % |
| L=32 k N=1024 | 281  | 1684 | - | 3008 | +499 % |

vLLM ratios (nano-vllm / vLLM):

| | pre-cascade | cascade eager | current |
| --- | ---: | ---: | ---: |
| L=4 k N=64    | 0.59× | 0.76× | 1.10× |
| L=4 k N=1024  | 0.23× | 0.42× | - |
| L=32 k N=64   | 0.24× | 0.87× | 1.16× |
| L=32 k N=256  | 0.12× | 0.71× | - |
| L=32 k N=1024 | 0.09× | 0.56× | - |

## Resolved follow-ups

Nothing remaining at the parity tolerance. The fresh median-of-3 full sweep in
[`bench/FULL_SWEEP_20260429.md`](FULL_SWEEP_20260429.md) has all 18 cells at or
above 0.95x vLLM, with the weakest cell at 0.978x
(`prefix=1,N=1024`). The README tables now publish those fresh medians rather
than the older pre-cascade numbers.

### 2026-04-28 follow-up: cascade CUDA graph

The cascade decode path now has CUDA graph capture keyed on
`(N, num_shared_blocks, max_tail_blocks)`. The tail table is sized for the
largest context the batch can reach before completion, so the graph key stays
stable when decode crosses later KV block boundaries.

This follow-up also fixed a `BlockManager.hash_blocks` correctness issue where
a hashed block could be evicted from `used_block_ids` while its hash entry
remained live, then later be deduplicated back into a sequence without being
removed from `free_block_ids`. That stale free-list state could surface as a
later deallocation `KeyError`.

Repeated HTTP measurements recorded in
[`bench/raw_n64_verification_20260429/`](raw_n64_verification_20260429/):

| prefix | N | output tok/s | note |
| ---: | ---: | ---: | --- |
| 1 | 64 | 2968.6 | median of 3 nano-vLLM runs, no-prefix control; vLLM median 2522.3 |
| 4096 | 64 | 2271.6 | median of 3 nano-vLLM runs; vLLM median 2066.3 |
| 32768 | 64 | 1308.4 | median of 3 nano-vLLM runs, long-prefix control; vLLM median 1131.0 |

The fresh repeated vLLM L=4096/N=64 median on the same DGX Spark/Qwen3-0.6B
setup is 2066.3 tok/s, so the target cell now measures 1.10x vLLM by median
HTTP throughput.

Additional pre-fix single-run focused probes against the already-running
nano-vLLM server on 2026-04-29 (`gpu_memory_utilization=0.85`,
`max_num_batched_tokens=16384`, default varlen cascade):

| prefix | N | output tok/s | vLLM reference | ratio | note |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4096 | 256 | 4482.5 | 4694 | 0.955x | pre-full-sweep focused probe |
| 4096 | 1024 | 6345.9 | 8295 | 0.765x | pre exact-block fix |

### 2026-04-29 follow-up: exact-block full cache hits

The sweep prompts at L=4096 and L=32768 end exactly on a 256-token KV block
boundary. `BlockManager.can_allocate()` previously kept the last prompt block
private so a prefix-cache hit always had at least one prefill block to run.
That was correct for partial final blocks, but it made exact-block prompts keep
256 prompt tokens in each sequence's cascade suffix. At N=1024 that reintroduced
262k tokens of per-step suffix attention work.

The scheduler now allows exact-block prompts to cache the final prompt block and
handles a full-cache hit by moving the sequence directly into decode, where the
first decode step recomputes only the last prompt token and subsequent steps
share the complete prompt prefix.

Repeated HTTP measurements recorded in
[`bench/raw_focused_verification_20260429_iter/`](raw_focused_verification_20260429_iter/)
after this change (`nanovllm_fullblock_varlen_probe*.err`):

| prefix | N | nano-vLLM median tok/s | vLLM reference | ratio |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | 256 | 4792.7 | 4694 | 1.02x |
| 4096 | 1024 | 9466.1 | 8295 | 1.14x |
| 32768 | 256 | 2817.7 | 2476 | 1.14x |
| 32768 | 1024 | 3625.8 | 3008 | 1.21x |

Pre-fix medians from the same run directory (`nanovllm_varlen_run*.err`) were
4595.6, 6480.6, 2444.5, and 2171.1 tok/s respectively, so the gain is
concentrated where the duplicated exact-block suffix was largest.

At very high concurrency (N >= 1024) on the moderate-prefix row (L=4 k), the
fresh full sweep now clears parity. If a future full sweep regresses, the next
likely limiter to investigate is:

1. **Suffix kernel quality at N=1024.** The per-seq tails fan out to
   1024 unique block_tables. `flash_attn_varlen_func` with
   `cu_seqlens_q = [0, 1, …, 1024]` parallelises across batches but the
   per-batch K/V access pattern is randomised across the paged cache;
   L2 thrashes. vLLM uses kernels that are tuned for this regime
   (batched-decode in FlashInfer or FA3 on Blackwell).

Historical kernel alternatives:
- FlashInfer is now available in the local uv environment
  (`flashinfer-python[cu13]` + `flashinfer-cubin` 0.6.9; `show-config`
  reports CUDA 13.0 / SM 12.1). `NANO_VLLM_CASCADE_SUFFIX_KERNEL=flashinfer`
  selects an opt-in `MultiLevelCascadeAttentionWrapper` path over the existing
  paged KV cache. On the target L=4 k N=64 focused probe it reached
  2144.2 tok/s, clearing the target but trailing the CUDA-graphed varlen
  cascade, so it remains non-default.
- `NANO_VLLM_CASCADE_SUFFIX_KERNEL=flashinfer_shared` selects FlashInfer's
  exact shared-prefix paged decode wrapper when the model runs in float16. It
  is not viable for the default Qwen3-0.6B bfloat16 benchmark on the installed
  FlashInfer build, so bf16 runs fail early with a clear error instead of
  entering the wrapper and failing later.
- For the suffix pass, `NANO_VLLM_CASCADE_SUFFIX_KERNEL=kvcache` selects
  `flash_attn_with_kvcache` instead of the default `flash_attn_varlen_func`.
  After cascade graphing it was roughly parity on the target L=4 k N=64
  focused probe (2231 tok/s vs 2248 recorded for varlen), so varlen remains
  the default. Earlier pre-cascade measurement had kvcache regressing
  L=4 k N=64 by 27 %, with parity elsewhere.

## What still keeps us behind vLLM

Nothing remaining at the parity tolerance.

## Things ruled out by experiment

- **Async sampler / pinned input buffers / prefill+decode mixing.**
  Profiler showed each targets sub-1 % of measured step time. None
  were going to close a gap that lives on the GPU.
- **CUDA graph capture cap (was 512).** Lifted to `max_num_seqs`. No
  measurable benefit on this bench, confirming the cell isn't
  bottlenecked by graph dispatch overhead.
- **Cascade-path CUDA graph.** Implemented after the original cascade pass;
  it moved the target L=4 k N=64 cell above the recorded vLLM baseline.
- **`torch.compile(dynamic=True)`** on RMSNorm and Sampler. Avoids
  recompile_limit warnings but slightly regressed L=4 k N=4 (within
  noise). Reverted.

The earlier L=1 row matching vLLM did not explain the historical L=4 k or
L=32 k gap because the L=1 path does not exercise the per-seq paged-attention
read pattern that scales with prefix length. The cascade fix removed that
scaling.
