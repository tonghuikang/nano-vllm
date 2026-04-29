from types import SimpleNamespace
import unittest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence


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


if __name__ == "__main__":
    unittest.main()
