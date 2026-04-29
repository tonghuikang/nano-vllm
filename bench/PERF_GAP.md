# Why nano-vllm trails vLLM on shared-prefix high-concurrency cells

Measured on **NVIDIA DGX Spark (GB10, sm_121, 128 GB unified memory)**, Qwen3-0.6B.
See `bench_nanovllm.log` and `bench_vllm.log` for raw runs.

The gap is concentrated on cells with **L ≥ 4 k AND N ≥ 16** — the L=1
no-shared-prefix row stays at parity (within 5–10 % either way). That isolated
the cause to nano-vllm's handling of long shared-prefix concurrent workloads.

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
- CUDA graph is bypassed when cascade is active (the cascade path is
  not graph-friendly; eager dispatch is fine because the kernels do all
  the heavy lifting).

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

### Bench impact

L=1 row holds parity (cascade gate keeps it on the existing path).
Shared-prefix cells (final numbers at `gpu_memory_utilization=0.85`,
which is enough for the ~2,063 blocks L=4 k N=1024 needs without
preemption):

| | pre-cascade | post-cascade | vLLM | Δ vs pre |
| --- | ---: | ---: | ---: | ---: |
| L=4 k N=64    | 1208 | 1565 | 2051 | +30 % |
| L=4 k N=256   | 1877 | 2724 | 4694 | +45 % |
| L=4 k N=1024  | 1874 | 3474 | 8295 | +85 % |
| L=32 k N=64   | 266  | 979  | 1130 | +268 % |
| L=32 k N=256  | 286  | 1762 | 2476 | +516 % |
| L=32 k N=1024 | 281  | 1684 | 3008 | +499 % |

vLLM ratios (nano-vllm / vLLM):

| | pre-cascade | post-cascade |
| --- | ---: | ---: |
| L=4 k N=1024  | 0.23× | 0.42× |
| L=32 k N=64   | 0.24× | 0.87× |
| L=32 k N=256  | 0.12× | 0.71× |
| L=32 k N=1024 | 0.09× | 0.56× |

## What still keeps us behind vLLM

At very high concurrency (N ≥ 256) on the moderate-prefix row (L=4 k),
nano-vllm is at ~0.4× of vLLM. Two likely causes:

1. **Suffix kernel quality at N=1024.** The per-seq tails fan out to
   1024 unique block_tables. `flash_attn_varlen_func` with
   `cu_seqlens_q = [0, 1, …, 1024]` parallelises across batches but the
   per-batch K/V access pattern is randomised across the paged cache;
   L2 thrashes. vLLM uses kernels that are tuned for this regime
   (batched-decode in FlashInfer or FA3 on Blackwell).
2. **No cascade-path CUDA graph.** Cascade dispatch is eager, so the
   28 layers each pay launch + dispatch overhead per step. Not the
   dominant cost at L=32 k (compute-bound) but matters at L=4 k where
   the kernels are quick.

Concrete next steps:
- Try `flashinfer.BatchDecodeWithSharedPrefix` — same concept but a
  kernel tuned for it (vLLM's choice when the dep is available).
- Capture a cascade-mode CUDA graph keyed on `(N, num_shared_blocks,
  max_tail_blocks)` since within a benchmark cell those are stable.
- For the suffix pass, if `flash_attn_with_kvcache` ever beats
  `flash_attn_varlen_func` for high-N moderate-tail decodes, switch
  per-cell. (We measured: kvcache regresses L=4 k N=64 by 27 %, parity
  elsewhere — varlen is the better default.)

## Things ruled out by experiment

- **Async sampler / pinned input buffers / prefill+decode mixing.**
  Profiler showed each targets sub-1 % of measured step time. None
  were going to close a gap that lives on the GPU.
- **CUDA graph capture cap (was 512).** Lifted to `max_num_seqs`. No
  measurable benefit on this bench, confirming the cell isn't
  bottlenecked by graph dispatch overhead.
- **`torch.compile(dynamic=True)`** on RMSNorm and Sampler. Avoids
  recompile_limit warnings but slightly regressed L=4 k N=4 (within
  noise). Reverted.

The L=1 row already matching vLLM tells you nothing about the L=4 k or
L=32 k gap — the L=1 path doesn't exercise the per-seq paged-attention
read pattern that scales with prefix length. The cascade fix is what
removed that scaling.
