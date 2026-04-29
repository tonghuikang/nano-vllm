from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    # Cascade-attention plumbing for the decode path. When `shared_prefix_len`
    # is non-zero, every running seq's first N block_table entries are
    # identical. PagedAttention can read the prefix's K/V *once* via
    # flash_attn_func against a dense tensor, then attend over the per-seq
    # unique tail via flash_attn_with_kvcache, and merge the two via LSE
    # composition. Cuts per-step KV-cache HBM traffic from O(N·L) to
    # O(L + N·tail) when there's a long shared prefix.
    shared_prefix_blocks: torch.Tensor | None = None  # int32 tensor of shared block ids
    shared_prefix_len: int = 0                        # tokens covered by shared prefix
    shared_prefix_signature: int = 0                  # hash of shared_prefix_blocks for cache-invalidation
    tail_block_tables: torch.Tensor | None = None     # (N, max_tail_blocks) int32
    tail_lens: torch.Tensor | None = None             # (N,) int32
    tail_max_len: int = 0                             # max(tail_lens) on host — avoids per-step .item() sync
    cu_q_pref: torch.Tensor | None = None             # [0, N], int32 — built once per step
    cu_k_pref: torch.Tensor | None = None             # [0, L_p], int32
    cu_q_suff: torch.Tensor | None = None             # [0, 1, …, N], int32
    cu_k_suff: torch.Tensor | None = None             # [0, tail_lens[0], …, sum(tail_lens)], int32

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None,
                shared_prefix_blocks=None, shared_prefix_len=0, shared_prefix_signature=0, tail_block_tables=None, tail_lens=None, tail_max_len=0,
                cu_q_pref=None, cu_k_pref=None, cu_q_suff=None, cu_k_suff=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables,
                      shared_prefix_blocks, shared_prefix_len, shared_prefix_signature, tail_block_tables, tail_lens, tail_max_len,
                      cu_q_pref, cu_k_pref, cu_q_suff, cu_k_suff)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
