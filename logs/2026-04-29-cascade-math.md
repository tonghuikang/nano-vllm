# Cascade attention — the math behind the speedup

Companion to [`2026-04-29-cascade-decode.md`](2026-04-29-cascade-decode.md), which
covers what cascade does at a system level. This doc derives why it is
correct and how much it can save.

## 1. Softmax attention as a reduction over keys

For one query `q ∈ ℝ^d` against keys `K = [k_1; …; k_L] ∈ ℝ^{L×d}` and
values `V = [v_1; …; v_L] ∈ ℝ^{L×d}`, attention output is

```
o = Σ_i softmax(s)_i · v_i           with   s_i = (q · k_i) / √d
   = Σ_i exp(s_i − m) v_i  /  Σ_i exp(s_i − m)
```

where `m = max_i s_i` is a stability shift. Define the log-sum-exp

```
ℓ = m + log Σ_i exp(s_i − m)         (= log Σ_i exp(s_i) directly)
```

The pair `(o, ℓ)` is *all* the information a downstream consumer needs to
combine this attention with another partial attention over a disjoint key
range. Once you have `o` and `ℓ`, the un-normalized numerator is recoverable
as `o · exp(ℓ)`, and the denominator is `exp(ℓ)`.

This is the FlashAttention online-softmax identity (Dao, 2023) and the
foundation everything below rests on.

## 2. Partition identity: split keys, merge with LSE

Partition the key/value range into two disjoint chunks `A` and `B` of
length `L_A` and `L_B`. Run attention over each independently:

```
o_A, ℓ_A = attn(q, K_A, V_A)
o_B, ℓ_B = attn(q, K_B, V_B)
```

Each is a "complete" attention over its own chunk, with its own normalization.
Because softmax is *not* linear, you cannot just average `o_A` and `o_B`. But
you *can* recover the joint result exactly using the LSE values as soft
weights:

```
o = (o_A · exp(ℓ_A) + o_B · exp(ℓ_B))
    / (exp(ℓ_A) + exp(ℓ_B))
```

Proof (sketch). Let `Z_A = exp(ℓ_A) = Σ_{i ∈ A} exp(s_i)`, similarly `Z_B`.
The joint denominator is `Z_A + Z_B`. The joint numerator is
`Σ_{i ∈ A∪B} exp(s_i) v_i = Z_A · o_A + Z_B · o_B` (by definition of `o_A`).
Dividing gives the formula above. ∎

Numerically stable form (used in `nanovllm/layers/attention.py:_cascade_decode`,
following arxiv:2501.01005 §2.2):

```
m       = max(ℓ_A, ℓ_B)
w_A     = exp(ℓ_A − m),   w_B = exp(ℓ_B − m)
o       = (w_A · o_A + w_B · o_B) / (w_A + w_B)
```

The shift by `m` keeps both weights `≤ 1` and avoids overflow when `ℓ_A`
and `ℓ_B` are large in magnitude.

This identity holds for **any** partition of the keys, not just
prefix/suffix — and that is exactly what makes it useful here.

## 3. Application to shared-prefix decode

In a batch of `N` decode steps where every sequence shares a prefix of
length `L_p` and has its own tail `t_n`, sequence `n`'s key range is

```
K^(n) = [K_prefix ; K_tail^(n)]    with |K_prefix| = L_p, |K_tail^(n)| = t_n
```

Two-pass cascade decode:

| pass | input | output |
| ---- | ----- | ------ |
| prefix | all N queries `{q_n}`, shared `K_prefix`/`V_prefix` | `o_p^(n), ℓ_p^(n)` for each `n` |
| suffix | per-seq queries, per-seq `K_tail^(n)`/`V_tail^(n)` | `o_s^(n), ℓ_s^(n)` |
| merge  | the four tensors above | `o^(n)` for each `n` |

Pass 1 is implemented as a single `flash_attn_varlen_func` call with
`cu_seqlens_q = [0, 1, …, N]` (so all N queries form one varlen "batch")
and a *single* shared `block_table` over the prefix blocks. The kernel
loads each `K_prefix[b], V_prefix[b]` block **once** into SMEM and
broadcasts it across all N queries before evicting.

Pass 2 is the standard per-seq paged decode but only over the per-seq
tail blocks — much shorter than the full context.

The merge runs in plain PyTorch on the `(N, num_heads, head_dim)`
tensors plus their `(num_heads, N)` LSEs. Cost is `O(N · num_heads · head_dim)`
elementwise ops, negligible next to the attention passes themselves.

## 4. I/O complexity — where the speedup comes from

The attention output value `o` itself is the same with or without
cascade. Speed difference lives entirely in **bytes moved between HBM
and on-chip cache**.

For Qwen3-0.6B (`num_kv_heads=8`, `head_dim=128`, bf16, `num_layers=28`),
each cached token occupies

```
2 (K+V) · num_kv_heads · head_dim · 2 bytes = 2 · 8 · 128 · 2 = 4 KiB / token / layer
```

i.e. **112 KiB / token** across all 28 layers.

### Naive paged decode (no cascade)

For one decode step over N sequences with full context length `L`, each
sequence reads its full `L`-token K/V across all 28 layers. Naive HBM
traffic per step:

```
B_naive = N · L · 112 KiB
```

For `N=1024, L=32 768`:

```
B_naive = 1024 · 32768 · 112 KiB ≈ 3.7 TiB per step
```

That's a *floor* — what the access pattern would cost if every K/V
fetch went to HBM with no L2/SMEM reuse. In practice the standard
paged-attention kernel does get partial reuse within each per-block
tile, so the effective bandwidth demand is lower (we never hit the
naive floor).

### Cascade decode

Cascade replaces the per-seq read of the prefix with one shared read:

```
B_cascade = (L_p + N · t_avg) · 112 KiB
         ≈ L_p · 112 KiB           when t_avg ≪ L_p · N (early decode)
```

For `N=1024, L_p=32 768, t_avg=0`:

```
B_cascade ≈ 32768 · 112 KiB ≈ 3.7 GiB per step
```

— **1024× less** than the naive floor at this shape.

### Speedup formula

If attention reads dominate step time (true at the long-prefix
high-N corner), the speedup is bounded by the byte-count ratio:

```
speedup ≤ B_naive / B_cascade
       = N · L / (L_p + N · t_avg)
       → N            when t_avg → 0
```

So at `t_avg = 0` the theoretical ceiling is `N`× — but the actual gain
plateaus much sooner because:

1. **Non-attention work doesn't shrink.** Embedding, FFN, layernorms,
   sampler, scheduler bookkeeping are all `O(N · 1)` per step and
   independent of `L_p`. Once cascade pushes attention below those,
   they become the bottleneck.
2. **Standard paged decode already gets some L2 reuse.** The "naive"
   floor overestimates the pre-cascade cost — within one kernel
   launch, queries sharing K/V blocks coalesce.
3. **Cascade adds overhead.** Two kernel launches instead of one, plus
   the LSE merge.

### Observed vs. theoretical (post-cascade benchmarks, this repo)

| (N, L_p) | pre-cascade tok/s | cascade tok/s | observed × | theoretical max × |
| -------- | ----------------: | ------------: | ---------: | ----------------: |
| 1024, 32 768 |  281 | 3 723 | 13.2× | up to 1024× |
| 256, 32 768  |  286 | 2 722 |  9.5× | up to 256×  |
|  64, 32 768  |  266 | 1 308 |  4.9× | up to 64×   |
| 1024,  4 096 | 1 874 | 9 834 | 5.2× | up to 1024× |

The 13× observed at `N=1024, L=32 768` is what the architecture lets
through after non-attention work, dispatch overhead, and partial pre-cascade
L2 reuse take their slices. The fact that the gain *grows with both N and
L_p* — and that the slope flattens once non-attention work catches up — is
the signature of a memory-bound kernel becoming compute/dispatch-bound.

### Why the gate sits at `N · L_p ≥ 32 K`

Cascade's overhead (extra launch + merge) is roughly constant per layer
per step. Its benefit scales with `N · L_p · 112 KiB / HBM_bw`. The
crossover is at the prefix-read-cost ≈ overhead point — empirically
around `N · L_p ≈ 32 768` tokens for Qwen3-0.6B on Spark. Below that the
two-pass scheme is a small loss; the gate keeps L=1 and small-prefix
runs on the original path.

## 5. What this does *not* save

Cascade only moves bytes. Specifically it does **not** help:

- **L=1** (no shared prefix). No prefix to amortize. The gate skips
  cascade and the standard kernel runs.
- **Compute-bound regimes.** If the GPU is FLOP-limited rather than
  bandwidth-limited (very small `L`, very large head dim, or matmul
  shapes that already saturate tensor cores), reducing HBM reads
  doesn't translate to speedup.
- **The decode tail.** As decode runs and per-seq tails grow, the
  `N · t_avg` term in `B_cascade` grows linearly — the suffix pass
  becomes the new bandwidth-dominant term once `t_avg ≳ L_p / N`.
  At that point a "second-level" cascade (recursive partition of the
  tail across N seqs that share *some* tail prefix) would help,
  but nano-vllm doesn't currently exploit that.
- **Prefill.** The math still holds, but prefill workloads in this
  repo's bench don't exhibit the long-shared-prefix pattern that
  cascade exists to handle (each prefill is the prefix being created,
  not consumed N ways).

## References

- Dao, *FlashAttention-2*, 2023. Online-softmax identity used in §1.
- Juravsky et al., *Hydragen*, 2024 (arxiv:2402.05099). Original
  prefix/suffix decomposition + LSE merge for shared-prefix batched
  decoding.
- Ye et al., *FlashInfer*, 2025 (arxiv:2501.01005, §2.2). Clean
  derivation of the LSE merge formula in the form used in
  `_cascade_decode`.
