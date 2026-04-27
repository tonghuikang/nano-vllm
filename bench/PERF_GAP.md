# Why nano-vllm trails vLLM on shared-prefix high-concurrency cells

Measured on **NVIDIA DGX Spark (GB10, sm_121, 128 GB unified memory)**, Qwen3-0.6B,
2026-04-27. See `bench_nanovllm.log` and `bench_vllm.log` for raw runs.

The gap is concentrated on cells with **L ≥ 4096 AND N ≥ 4** — the L=1
no-shared-prefix row stays within 1.14× of vLLM. That isolates the cause to
nano-vllm's handling of shared-prefix concurrent workloads.

## Diagnosis (on the L=4096 N=64 cell, vLLM 2026 → nano-vllm 1152 tok/s, gap 1.75×)

Five focused experiments using the same model/hardware (`probe_nanovllm.py`):

| variant | knob change | tok/s | Δ vs baseline |
| --- | --- | ---: | ---: |
| A baseline | default                            | 1152 | — |
| B serial cold prefill | `max_num_batched_tokens=2048`         | 1192 | +3% |
| C prefix pre-warmed | warmup pass before timed cell        | 1198 | +4% |
| D smaller decode batch | `max_num_seqs=256`                | 1154 | +0% |
| E eager mode | `enforce_eager=True` (no CUDA graph)        | 1123 | −3% |

**vLLM is 1.75× faster, but no nano-vllm knob recovers more than 4%.** The
bottleneck is not in the configurable scheduler/cache machinery — it's in
the per-step engine path.

## Root cause: per-step CPU↔GPU synchronization

`engine/model_runner.py` runs decode like this for every single token:

```python
def run(self, seqs, is_prefill):
    input_ids, positions = self.prepare_decode(seqs)        # CPU lists → pin_memory → cuda(async)
    temperatures = self.prepare_sample(seqs)                # same
    logits = self.run_model(input_ids, positions, False)    # CUDA-graph replay
    token_ids = self.sampler(logits, temperatures).tolist() # ⚠ forces cudaSynchronize + D2H
    reset_context()
    return token_ids
```

The `.tolist()` blocks until sampling finishes, copies tokens to host, and
**only then** can the engine call `scheduler.postprocess` and start the next
step. Per-step Python work + sync sits in series with GPU compute, not in
parallel.

Order-of-magnitude estimate at L=4096 N=64:

- **GPU compute / step** for batch-64 decode at ~5 k tokens of KV ≈ 30 ms
  (memory-bandwidth-bound on Spark; Qwen3-0.6B has GQA-8/28-layers).
- **Total decode**: 1024 steps × 30 ms ≈ 30 s.
- **Observed wall**: 55 s. → **per-step overhead ≈ 24 ms** sitting between
  GPU steps.
- **vLLM wall**: 32 s → per-step overhead ≈ 3 ms.

So nano-vllm spends **~22 ms per decode step waiting on Python**, doubling
wall time. vLLM eliminates this by:

1. **Async sampler** — next-token IDs are produced on the GPU and the
   scheduler kicks off the next step's `prepare_decode` *before* the
   sampler's host copy lands. (See vLLM's `output_processor` thread.)
2. **Pre-allocated input buffers** — vLLM's CUDA graph captures
   include the input tensors, so per-step host work is just
   `memcpy_async` into pinned device buffers, not Python list construction
   + `torch.tensor(...).cuda()`.
3. **Mixed prefill+decode batches** — vLLM admits a small number of
   prefill tokens *into the same step* as running decodes, so prefill
   doesn't stall in-flight decodes. nano-vllm's `Scheduler.schedule()`
   returns `(seqs, is_prefill)` — strictly one mode per step (see
   `engine/scheduler.py:25-73`), so during the few prefill steps at the
   start of a high-N cell, all already-running decodes are paused.

(3) compounds (1)+(2) at higher concurrency: every prefill step is a wider
stall, which is why the gap grows from 1.04× at N=1 to 4.30× at N=1024.

## Other things ruled out

- **Prefix-cache race in `Scheduler`** (multiple cold prefills of the same
  prefix admitted in a single step at `max_num_batched_tokens=16384`): real,
  but worth only ~3% (probe B). At L=4096 only 4 prompts can fit, so worst
  case 3 redundant cold prefills ≈ 1.5 s out of a 55 s cell.
- **Prefill kernel quality**: nano-vllm uses `flash_attn_varlen_func` (same
  family as vLLM's). Probe C (warm cache → almost no prefill work) doesn't
  close the gap.
- **CUDA graph capture**: probe E shows graphs save only ~3% on this cell.
  The cell isn't compute-bound enough for graph replay vs eager to matter.
- **Decode batch ceiling (`max_num_seqs=512`)**: probe D drops it to 256
  with no effect, ruling out batch-size-related throttling at this N.
- **GPU memory contention**: nano-vllm here gets `gpu_memory_utilization=0.65`
  → ~32 GB KV pool, well within Spark's 128 GB. KV cache is not the
  binding constraint.

## Where the gap could be closed (priority order)

1. **Async sampler / overlap host↔device sync with the next prepare**.
   Easy win — biggest single contributor to the per-step overhead. Pattern:
   issue sampler kernel, return a CUDA event, start the next
   `prepare_decode` immediately, only wait on the event right before the
   block-manager bookkeeping that needs the actual token IDs.
2. **Capture `prepare_decode` into the CUDA graph**. Today the graph
   captures `model(input_ids, positions)` but `input_ids`, `positions`,
   `slot_mapping`, `context_lens`, `block_tables` are rebuilt host-side
   every step (`prepare_decode` at `model_runner.py:172-188`). Move those
   into pre-allocated pinned-host buffers and slot-fill in place.
3. **Mix prefill+decode in one step.** Allow `Scheduler.schedule()` to
   return a small prefill chunk *plus* the running decode batch, so the
   first few prefill steps of a high-N cell don't stall in-flight
   decodes. Hardest of the three; meaningful only at N ≥ 16.

The L=1 row already matching vLLM tells you the kernels and the basic
block manager are fine — the gap is engine-overhead-shaped, not
compute-shaped.
