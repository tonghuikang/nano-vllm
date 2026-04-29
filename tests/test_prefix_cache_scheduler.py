from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.attention import Attention
from nanovllm.utils.context import get_context, reset_context


def make_config(num_blocks=16, block_size=256, max_num_seqs=4, max_num_batched_tokens=1024):
    return SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        eos=-1,
        num_kvcache_blocks=num_blocks,
        kvcache_block_size=block_size,
    )


class PrefixCacheSchedulerTest(unittest.TestCase):
    def setUp(self):
        Sequence.block_size = 256

    def test_exact_block_prompt_can_cache_final_block(self):
        manager = BlockManager(num_blocks=8, block_size=256)
        first = Sequence(list(range(512)))
        manager.allocate(first, num_cached_blocks=0)
        first.num_scheduled_tokens = 512
        manager.hash_blocks(first)

        second = Sequence(list(range(512)))

        self.assertEqual(manager.can_allocate(second), 2)

    def test_full_cache_hit_enters_decode_without_empty_prefill(self):
        scheduler = Scheduler(make_config())
        first = Sequence(list(range(512)))
        scheduler.block_manager.allocate(first, num_cached_blocks=0)
        first.num_scheduled_tokens = 512
        scheduler.block_manager.hash_blocks(first)
        first.num_scheduled_tokens = 0

        second = Sequence(list(range(512)))
        scheduler.add(second)

        seqs, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(seqs, [second])
        self.assertEqual(second.num_cached_tokens, 512)
        self.assertEqual(second.num_scheduled_tokens, 1)
        self.assertFalse(second.is_prefill)

    def test_cached_decode_batch_uses_cascade_attention_context(self):
        batch_size = 128
        block_size = 256
        prompt_len = block_size + 44
        config = make_config(
            num_blocks=batch_size + 4,
            block_size=block_size,
            max_num_seqs=batch_size * 2,
            max_num_batched_tokens=batch_size * (prompt_len - block_size),
        )
        scheduler = Scheduler(config)
        shared_prefix = list(range(block_size))

        seed = Sequence(shared_prefix + list(range(10_000, 10_044)))
        scheduler.block_manager.allocate(seed, num_cached_blocks=0)
        seed.num_scheduled_tokens = block_size
        scheduler.block_manager.hash_blocks(seed)

        for i in range(batch_size):
            unique_tail = list(range(20_000 + i * 100, 20_044 + i * 100))
            scheduler.add(Sequence(shared_prefix + unique_tail))

        prefill_seqs, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(len(prefill_seqs), batch_size)
        scheduler.postprocess(prefill_seqs, list(range(30_000, 30_000 + batch_size)), is_prefill=True)

        decode_seqs, is_prefill = scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(len(decode_seqs), batch_size)

        runner = ModelRunner.__new__(ModelRunner)
        runner.block_size = block_size
        with patch.object(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor):
            input_ids, positions = runner.prepare_decode(decode_seqs)
        ctx = get_context()
        try:
            self.assertEqual(input_ids.tolist(), [seq.last_token for seq in decode_seqs])
            self.assertEqual(positions.tolist(), [len(seq) - 1 for seq in decode_seqs])
            self.assertFalse(ctx.is_prefill)
            self.assertIsNone(ctx.block_tables)
            self.assertEqual(ctx.shared_prefix_len, block_size)
            self.assertEqual(ctx.shared_prefix_blocks.tolist(), [decode_seqs[0].block_table[0]])
            self.assertEqual(ctx.tail_block_tables.shape, (batch_size, 1))
            self.assertEqual(ctx.tail_block_tables[:, 0].tolist(), [seq.block_table[1] for seq in decode_seqs])
            self.assertEqual(ctx.tail_lens.tolist(), [45] * batch_size)
            self.assertEqual(ctx.tail_max_len, block_size)
            self.assertEqual(ctx.cu_q_pref.tolist(), [0, batch_size])
            self.assertEqual(ctx.cu_k_pref.tolist(), [0, block_size])
            self.assertEqual(ctx.cu_q_suff.tolist(), list(range(batch_size + 1)))
            self.assertEqual(ctx.cu_k_suff.tolist(), [45 * i for i in range(batch_size + 1)])
            self.assertEqual(ctx.slot_mapping.tolist(), [seq.block_table[1] * block_size + 44 for seq in decode_seqs])

            attn = Attention(num_heads=2, head_dim=4, scale=0.5, num_kv_heads=2)
            attn.k_cache = torch.empty(config.num_kvcache_blocks, block_size, 2, 4)
            attn.v_cache = torch.empty(config.num_kvcache_blocks, block_size, 2, 4)
            q = torch.empty(batch_size, 2, 4)
            k = torch.empty(batch_size, 2, 4)
            v = torch.empty(batch_size, 2, 4)
            expected = torch.empty(batch_size, 2, 4)

            with patch("nanovllm.layers.attention.store_kvcache") as store_kvcache, \
                 patch("nanovllm.layers.attention._cascade_decode", return_value=expected) as cascade_decode:
                out = attn(q, k, v)

            self.assertIs(out, expected)
            store_kvcache.assert_called_once_with(k, v, attn.k_cache, attn.v_cache, ctx.slot_mapping)
            cascade_decode.assert_called_once_with(q, attn.k_cache, attn.v_cache, attn.scale, ctx)
        finally:
            reset_context()


if __name__ == "__main__":
    unittest.main()
