# Parity plan: nano-vllm matches vLLM on every published bench cell

Drive nano-vllm to throughput parity (or better) with vLLM across the full
prefix-length × concurrency sweep that the README publishes, on the current
DGX Spark host (GB10, sm_121, 128 GB unified memory, Qwen3-0.6B).

The repo is in the middle of this work. `bench/FULL_SWEEP_20260429.md` claims
all 18 cells are within 0.95× of fresh vLLM medians, but the README still
publishes pre-cascade numbers. Confirm the current code actually delivers
those numbers, then make the README and `bench/PERF_GAP.md` reflect reality.

## Current status, 2026-04-29

- Fresh median-of-3 nano-vLLM and vLLM sweeps are documented in
  `bench/FULL_SWEEP_20260429.md`.
- `README.md`, `bench/PERF_GAP.md`, `bench/bench_nanovllm.txt`, and
  `bench/bench_vllm.txt` have been updated to publish the fresh medians.
- The weakest fresh cell is `prefix=1,N=1024`, where nano-vLLM is 0.978x
  vLLM; all 18 cells clear the 0.95x acceptance threshold.
- `uv run pytest tests/` passes.

## Sweep grid

- prefix lengths: `1`, `4096`, `32768`
- concurrencies: `1`, `4`, `16`, `64`, `256`, `1024`
- 18 total cells

For each cell, send N concurrent requests sharing one random prefix to
`/v1/completions`; each request pinned at `output_tokens = clamp(131072/N,
64, 1024)`, `min_tokens=max_tokens`, `ignore_eos=true`. Cell value is the
sum of decode tokens divided by wall time (tok/s). The harness lives at
`bench/bench_concurrency.py` (nano-vllm) and `bench/bench_vllm.py` (vLLM).

## Acceptance criteria — every one of these must hold

1. **All 18 cells ≥ 0.95× vLLM.** A fresh median-of-3 nano-vllm sweep and
   a fresh median-of-3 vLLM sweep must be on disk under `bench/`, and for
   every (prefix, N) cell, `nano_median / vllm_median ≥ 0.95`. The
   weakest cell ratio must be reported in the published markdown. Both
   sweeps must come from the *current* commit / current vLLM image on
   the *same* hardware in the same session — no mixing old logs with
   new ones.

2. **Reproducible commands.** The exact server-launch command and the
   exact bench harness command for both engines must be in the
   published markdown, copy-pasteable, runnable from the project root
   on this host. The vLLM image tag, the nano-vllm flags
   (`--gpu-memory-utilization`, `--max-num-batched-tokens`,
   `--max-num-seqs`, `--max-model-len`, `--kvcache-block-size`), and
   the model path used must all be stated.

3. **README reflects the fresh sweep.** The two tables under
   "Prefix-length × concurrency sweep" in `README.md` (nano-vllm and
   vLLM) must show the fresh medians from this run, not the older
   pre-cascade numbers. The "Speed ratio (vLLM / Nano-vLLM)" table
   must be computed from those medians and must not contain any cell
   above 1.10× (i.e., nano-vllm is allowed to be at most 10 % slower,
   matching the parity tolerance the README already calls out as
   "≤ 1.10×"). If any cell exceeds 1.10×, the loop is not done.

4. **Per-cell raw logs preserved.** Every per-iteration `.err`/`.out`
   from both engines lives under a directory in `bench/` named with
   today's date. The published markdown links to that directory and to
   the nano-vllm and vLLM `.txt` summary files (`bench/bench_nanovllm.txt`,
   `bench/bench_vllm.txt`).

5. **`bench/PERF_GAP.md` reflects what is actually true today.** The
   "What still keeps us behind vLLM" section must either be empty
   ("nothing remaining at parity tolerance") or list only items that
   the current sweep actually shows. No stale claims about cells
   that the fresh data clears.

6. **Tests still pass.** `uv run pytest tests/` runs to completion
   with zero failures from the project root. The cascade plumbing in
   `nanovllm/layers/attention.py`, `nanovllm/utils/context.py`,
   `nanovllm/engine/model_runner.py`, `nanovllm/engine/scheduler.py`,
   and `nanovllm/engine/block_manager.py` is exercised end-to-end
   by `tests/test_prefix_cache_scheduler.py` and any other test
   present.

7. **Working tree is committed.** `git status` shows a clean tree
   (no staged or unstaged changes) and the `main` branch tip
   contains the README/PERF_GAP/PARITY_PLAN updates. **Do not push
   to any remote.** Local commits only.

## Execution constraints

- **Hardware in use.** Single DGX Spark, GB10, sm_121, 128 GB
  unified memory. There is one GPU; both engines cannot occupy max
  utilization at the same time. Run nano-vllm and vLLM sequentially:
  start one server, sweep, stop it, start the next.
- **Model path.** `~/huggingface/Qwen3-0.6B/` for nano-vllm. vLLM
  uses the same weights via the path the existing
  `bench/FULL_SWEEP_20260429.md` records. If `gpu-memory-utilization
  0.85` rejects on startup because of leftover GPU memory, drop to
  `0.80` (matching what the existing full-sweep doc had to do) and
  document it.
- **vLLM image.** `nvcr.io/nvidia/vllm:26.03.post1-py3` — the only
  image with working sm_121 kernels at the time the bench was set
  up. Do not switch images.
- **Server lifecycle.** Always check that the prior server has fully
  released GPU memory before starting the next one. If a sweep aborts
  mid-cell, don't keep the partial data — re-run from scratch.
- **No destructive git ops.** No `git push`, no `git push --force`,
  no `git reset --hard`, no `git branch -D`, no `--no-verify`. New
  commits only.
- **No external uploads.** Logs / numbers stay on disk under `bench/`.
- **Don't drop the cascade gate.** The L=1 row depends on cascade
  staying off when `N · shared_prefix_len < 32 K`. If a "fix" you
  consider would regress L=1 N≥64, abandon it.
- **Don't disable prefix caching.** Both engines must run with
  prefix caching enabled — the bench is designed around it.
- **Output-length budget is fixed.** `GEN_BUDGET_PER_CELL = 131 072`
  in `bench/bench_concurrency.py` and `bench/bench_vllm.py`. If you
  change it, change it in both and update the README accordingly,
  but the default policy is don't change it.

## How to fix gaps if a cell fails

If after a clean median-of-3 sweep some cell fails the 0.95× bar,
the gap will live in one of these places. Investigate in this order:

1. **Cascade decode kernel quality at large N.** The suffix pass is
   `flash_attn_varlen_func` over `cu_seqlens_q = [0, 1, …, N]`. At
   N = 1024 each per-batch K/V access pattern is randomised across
   the paged cache; L2 thrashes. Possible levers:
   - `NANO_VLLM_CASCADE_SUFFIX_KERNEL=kvcache` switches the suffix
     to `flash_attn_with_kvcache`. Already wired in
     `nanovllm/layers/attention.py`. Try it cell by cell.
   - `NANO_VLLM_CASCADE_SUFFIX_KERNEL=flashinfer` selects FlashInfer's
     `MultiLevelCascadeAttentionWrapper`. The local uv env already has
     `flashinfer-python[cu13]` and `flashinfer-cubin` 0.6.9. Test on
     N ≥ 256 L=4 k cells where suffix-kernel quality is suspected.

2. **Cascade graph key churn.** Cascade graphs are keyed on
   `(N, num_shared_blocks, max_tail_blocks)`. If the tail length is
   undersized for the longest decode the batch will reach, capture
   thrashes when seqs cross block boundaries. Check
   `model_runner._make_cascade_graph_key` and the tail upper bound.

3. **CUDA graph fallback at large bs.** `model_runner.capture_cudagraph`
   should cover up to `max_num_seqs`. Verify N=1024 is captured rather
   than running eager.

4. **Preemption under KV pressure.** Profile with `bs=1024` cells and
   confirm the running batch holds at 1024, not e.g. 74. If the KV
   pool runs short, re-tune `--gpu-memory-utilization` (current default
   `0.85`).

5. **Per-step Python overhead at low N.** At N ≤ 4 the step is short
   and overhead matters. Profiler already showed 99.6 % of step time
   is GPU, so this is unlikely — confirm before chasing.

vLLM source for ideas:
- `vllm/v1/attention/backends/flash_attn.py` — `cascade_attention()`.
- `vllm/v1/attention/ops/triton_merge_attn_states.py` — LSE merge.
- `vllm/v1/worker/gpu_model_runner.py` — graph capture & dispatch.

## Verification each iteration

After each iteration of work:

1. Restart the nano-vllm server with the documented flags.
2. Run a 3-iteration sweep:
   ```bash
   for i in 1 2 3; do
     uv run python bench/bench_concurrency.py \
       --url http://127.0.0.1:8001/v1/completions \
       > bench/raw_full_sweep_$(date +%Y%m%d)_iter/nanovllm_full_run${i}.out \
       2> bench/raw_full_sweep_$(date +%Y%m%d)_iter/nanovllm_full_run${i}.err
   done
   ```
3. Compute medians per cell.
4. If the vLLM reference is older than today's session, restart vLLM
   and re-run the same way. Otherwise reuse today's vLLM medians.
5. Compute ratios. Report the worst cell.
6. If every cell is ≥ 0.95×, update README + PERF_GAP + bench markdown
   with the new numbers, run `uv run pytest tests/`, commit.

## Critical files

- `nanovllm/layers/attention.py` — cascade two-pass + LSE merge.
- `nanovllm/engine/model_runner.py` — cascade detection, graph capture.
- `nanovllm/engine/scheduler.py` — full-block prefix cache hit path.
- `nanovllm/engine/block_manager.py` — block hash + dedup.
- `nanovllm/utils/context.py` — cascade plumbing fields.
- `bench/bench_concurrency.py`, `bench/bench_vllm.py` — harnesses.
- `bench/FULL_SWEEP_20260429.md`, `bench/PERF_GAP.md` — current
  documented state.
- `README.md` — published headline tables.
- `tests/test_prefix_cache_scheduler.py` — exercises the full-block
  cache hit fix.
