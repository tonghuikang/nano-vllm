import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.graph_max_bs = 0
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens_t = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)

        # Find the longest common block_table prefix across all running seqs.
        # block_manager dedups full blocks via xxhash, so identical block ids
        # mean identical content. The last block of each seq is unique
        # (newly-allocated for decode), so the common prefix never extends to
        # any seq's tail. Required: every seq must have at least one block
        # *beyond* the shared prefix (so the cascade suffix pass is non-empty).
        #
        # Cascade gate: N * shared_prefix_len ≥ 32 K. The cost we save is the
        # N-fold redundant K/V re-read of the prefix; the cost we pay is two
        # kernel launches plus an LSE merge. Empirically the crossover is
        # around 32 K: L=4 k N=4 (16 K) regresses, L=32 k N=4 (128 K) wins.
        shared_blocks: list[int] = []
        shared_prefix_len = 0
        cascade_min_blocks = 1
        cascade_min_work = 32768
        if len(seqs) >= 2:
            common = list(seqs[0].block_table)
            for seq in seqs[1:]:
                bt = seq.block_table
                i = 0
                m = min(len(common), len(bt))
                while i < m and common[i] == bt[i]:
                    i += 1
                common = common[:i]
                if len(common) < cascade_min_blocks:
                    common = []
                    break
            min_blocks_per_seq = min(len(seq.block_table) for seq in seqs)
            if (len(common) >= cascade_min_blocks
                    and len(common) < min_blocks_per_seq
                    and len(seqs) * len(common) * self.block_size >= cascade_min_work):
                shared_blocks = common
                shared_prefix_len = len(common) * self.block_size

        if shared_prefix_len > 0:
            N = len(seqs)
            num_shared = len(shared_blocks)
            tail_rows = []
            tail_lens_list = []
            max_tail_blocks = max(len(seq.block_table) for seq in seqs) - num_shared
            for seq, ctxlen in zip(seqs, context_lens):
                tail = seq.block_table[num_shared:]
                tail_rows.append(tail + [-1] * (max_tail_blocks - len(tail)))
                tail_lens_list.append(ctxlen - shared_prefix_len)
            shared_prefix_blocks_t = torch.tensor(shared_blocks, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            tail_block_tables_t = torch.tensor(tail_rows, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            tail_lens_t = torch.tensor(tail_lens_list, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            # Hoist the cu_seqlens tensors here so the 28 attention layers
            # don't each pay the alloc + cumsum cost (was ~ms per step at
            # high N).
            cu_q_pref = torch.tensor([0, N], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            cu_k_pref = torch.tensor([0, shared_prefix_len], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            cu_q_suff = torch.arange(N + 1, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            cu_k_suff_host = [0]
            acc = 0
            for tl in tail_lens_list:
                acc += tl
                cu_k_suff_host.append(acc)
            cu_k_suff = torch.tensor(cu_k_suff_host, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            sig = hash(tuple(shared_blocks))
            tail_max = max(tail_lens_list)
        else:
            shared_prefix_blocks_t = None
            tail_block_tables_t = None
            tail_lens_t = None
            cu_q_pref = cu_k_pref = cu_q_suff = cu_k_suff = None
            sig = 0
            tail_max = 0

        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens_t, block_tables=block_tables,
                    shared_prefix_blocks=shared_prefix_blocks_t, shared_prefix_len=shared_prefix_len,
                    shared_prefix_signature=sig, tail_block_tables=tail_block_tables_t, tail_lens=tail_lens_t,
                    tail_max_len=tail_max,
                    cu_q_pref=cu_q_pref, cu_k_pref=cu_k_pref, cu_q_suff=cu_q_suff, cu_k_suff=cu_k_suff)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        ctx = get_context()
        # Cascade decode goes through flash_attn_varlen_func twice + an LSE
        # merge, none of which are in the captured graph. Run eager.
        if is_prefill or self.enforce_eager or ctx.shared_prefix_len > 0 or input_ids.size(0) > self.graph_max_bs:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = self.config.max_num_seqs
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # Up to 512 we keep the original step-16 density (fine-grained padding
        # waste matters for moderate batches). Above 512 we step by 64 — at
        # those sizes one decode step is already long, padding 1023→1024 is
        # noise but capturing 32 extra graphs would inflate startup time.
        coarse = list(range(576, max_bs + 1, 64)) if max_bs > 512 else []
        self.graph_bs = [1, 2, 4, 8] + list(range(16, min(max_bs, 512) + 1, 16)) + coarse
        if self.graph_bs[-1] != max_bs:
            self.graph_bs.append(max_bs)
        self.graph_max_bs = self.graph_bs[-1]
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
