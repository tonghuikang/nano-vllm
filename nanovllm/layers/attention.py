import os

import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


def _read_cascade_suffix_kernel() -> str:
    kernel = os.environ.get("NANO_VLLM_CASCADE_SUFFIX_KERNEL", "varlen").strip().lower()
    if kernel not in {"varlen", "kvcache", "flashinfer", "flashinfer_shared"}:
        raise ValueError(
            "NANO_VLLM_CASCADE_SUFFIX_KERNEL must be one of: varlen, kvcache, flashinfer, flashinfer_shared"
        )
    return kernel


CASCADE_SUFFIX_KERNEL = _read_cascade_suffix_kernel()


def _check_flashinfer_scale(q: torch.Tensor, scale: float):
    default_scale = q.size(-1) ** -0.5
    if abs(scale - default_scale) > 1e-12:
        raise RuntimeError(
            "FlashInfer cascade backends currently require the default "
            "attention scale head_dim ** -0.5"
        )


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def _cascade_decode(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, scale: float, ctx) -> torch.Tensor:
    """Two-pass shared-prefix attention via flash_attn_varlen_func.

    Pass 1 (prefix): treat all N queries as a single varlen "sequence" attending
    *non-causally* over the shared prefix blocks. The kernel loads the prefix
    K/V once and reuses it across the N queries from L1/L2/SMEM, dropping
    per-step KV-cache HBM traffic from O(N·L_prefix) to O(L_prefix).

    Pass 2 (suffix): each seq's query attends *causally* over its unique tail
    (block_table[num_shared_blocks:]). Standard per-seq paged attention, but
    over a much shorter context.

    Pass 3 (merge): compose via softmax-LSE — see arxiv:2501.01005 §2.2.

    cu_seqlens tensors live on `ctx`; they're built once per step by
    `prepare_decode` and reused across all 28 attention layers.
    """
    N = q.size(0)

    if CASCADE_SUFFIX_KERNEL == "flashinfer":
        _check_flashinfer_scale(q, scale)
        return ctx.cascade_wrapper.run(q, (k_cache, v_cache))
    if CASCADE_SUFFIX_KERNEL == "flashinfer_shared":
        _check_flashinfer_scale(q, scale)
        shared_k = k_cache.index_select(0, ctx.shared_prefix_blocks).flatten(0, 1)
        shared_v = v_cache.index_select(0, ctx.shared_prefix_blocks).flatten(0, 1)
        return ctx.cascade_wrapper.forward(q, shared_k, shared_v, (k_cache, v_cache))

    # Pass 1: prefix attention.
    out_p, lse_p, _ = flash_attn_varlen_func(
        q, k_cache, v_cache,
        cu_seqlens_q=ctx.cu_q_pref, cu_seqlens_k=ctx.cu_k_pref,
        max_seqlen_q=N, max_seqlen_k=ctx.shared_prefix_len,
        softmax_scale=scale, causal=False,
        block_table=ctx.shared_prefix_blocks.unsqueeze(0),
        return_attn_probs=True,
    )

    # Pass 2: suffix attention.
    if CASCADE_SUFFIX_KERNEL == "kvcache":
        out_s, lse_s = flash_attn_with_kvcache(
            q.unsqueeze(1), k_cache, v_cache,
            cache_seqlens=ctx.tail_lens, block_table=ctx.tail_block_tables,
            softmax_scale=scale, causal=True, return_softmax_lse=True,
        )
        out_s = out_s.squeeze(1)
        lse_s = lse_s.squeeze(-1).transpose(0, 1)
    else:
        out_s, lse_s, _ = flash_attn_varlen_func(
            q, k_cache, v_cache,
            cu_seqlens_q=ctx.cu_q_suff, cu_seqlens_k=ctx.cu_k_suff,
            max_seqlen_q=1, max_seqlen_k=ctx.tail_max_len,
            softmax_scale=scale, causal=True,
            block_table=ctx.tail_block_tables,
            return_attn_probs=True,
        )

    # Pass 3: merge. lse_p, lse_s shape [num_heads, N]; out_p, out_s [N, num_heads, head_dim].
    max_lse = torch.maximum(lse_p, lse_s)
    p_se = torch.exp(lse_p - max_lse)
    s_se = torch.exp(lse_s - max_lse)
    inv = 1.0 / (p_se + s_se)
    p_scale = (p_se * inv).transpose(0, 1).unsqueeze(-1).to(out_p.dtype)
    s_scale = (s_se * inv).transpose(0, 1).unsqueeze(-1).to(out_s.dtype)
    return out_p * p_scale + out_s * s_scale


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        elif context.shared_prefix_len > 0:    # decode w/ cascade
            o = _cascade_decode(q, k_cache, v_cache, self.scale, context)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o
