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


def _device_int_tensor(values, device):
    device = torch.device(device)
    pin_memory = device.type == "cuda" and torch.cuda.is_available()
    return torch.tensor(values, dtype=torch.int32, pin_memory=pin_memory).to(device, non_blocking=pin_memory)


def _config_torch_dtype(hf_config, feature: str = "model execution"):
    dtype_names = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    supported_dtypes = set(dtype_names.values())
    for attr in ("dtype", "torch_dtype"):
        dtype = getattr(hf_config, attr, None)
        if isinstance(dtype, torch.dtype):
            if dtype in supported_dtypes:
                return dtype
            continue
        if isinstance(dtype, str):
            dtype_name = dtype.strip().lower().removeprefix("torch.")
            if dtype_name in dtype_names:
                return dtype_names[dtype_name]
    raise TypeError(
        f"{feature} requires hf_config.dtype or hf_config.torch_dtype "
        "to be a torch.dtype or recognized dtype string"
    )


def _flashinfer_dtype(hf_config):
    return _config_torch_dtype(hf_config, "FlashInfer cascade planning")


def _flashinfer_dtype_name(dtype: torch.dtype) -> str:
    dtype_names = {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }
    try:
        return dtype_names[dtype]
    except KeyError as exc:
        raise TypeError(f"FlashInfer cascade planning does not support dtype {dtype}") from exc


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
        model_dtype = _config_torch_dtype(hf_config)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(model_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.graph_max_bs = 0
        self.cascade_graphs = {}
        self.cascade_graph_failures = set()
        self.cascade_flashinfer_wrapper = None
        self.cascade_flashinfer_workspace = None
        self.cascade_flashinfer_backend = None
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
        self.release_cudagraphs()
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def release_cudagraphs(self):
        for name in (
            "graphs",
            "graph_vars",
            "graph_bs",
            "graph_pool",
            "cascade_graphs",
            "cascade_flashinfer_wrapper",
            "cascade_flashinfer_workspace",
            "cascade_flashinfer_backend",
        ):
            if hasattr(self, name):
                delattr(self, name)
        if hasattr(self, "cascade_graph_failures"):
            self.cascade_graph_failures.clear()
        self.graph_max_bs = 0

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
        model_dtype = _config_torch_dtype(hf_config)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * model_dtype.itemsize
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

    @staticmethod
    def build_flashinfer_cascade_plan_inputs(
        N: int,
        shared_blocks: list[int],
        tail_rows: list[list[int]],
        tail_lens_list: list[int],
        block_size: int,
    ):
        num_shared = len(shared_blocks)
        unique_indptr_host = [0]
        unique_indices_host = []
        for tail in tail_rows:
            live_tail = [block_id for block_id in tail if block_id != -1]
            unique_indices_host.extend(live_tail)
            unique_indptr_host.append(len(unique_indices_host))

        unique_last_page_len_host = []
        for tail_len in tail_lens_list:
            last_page_len = tail_len % block_size
            unique_last_page_len_host.append(last_page_len or block_size)

        return {
            "query_indptr": [[0, N], list(range(N + 1))],
            "kv_indptr": [[0, num_shared], unique_indptr_host],
            "kv_indices": [shared_blocks, unique_indices_host],
            "last_page_len": [[block_size], unique_last_page_len_host],
        }

    def plan_flashinfer_cascade(self, N: int, shared_blocks: list[int], tail_rows: list[list[int]], tail_lens_list: list[int], device):
        try:
            import flashinfer
        except ImportError as exc:
            raise RuntimeError(
                "NANO_VLLM_CASCADE_SUFFIX_KERNEL=flashinfer or flashinfer_shared "
                "requires flashinfer-python and flashinfer-cubin to be installed"
            ) from exc

        from nanovllm.layers.attention import CASCADE_SUFFIX_KERNEL

        hf_config = self.config.hf_config
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        dtype = _flashinfer_dtype(hf_config)
        dtype_name = _flashinfer_dtype_name(dtype)
        if CASCADE_SUFFIX_KERNEL == "flashinfer_shared" and dtype is not torch.float16:
            raise RuntimeError(
                "NANO_VLLM_CASCADE_SUFFIX_KERNEL=flashinfer_shared currently requires "
                "a float16 model dtype; use flashinfer for bfloat16 models"
            )

        if getattr(self, "cascade_flashinfer_workspace", None) is None:
            self.cascade_flashinfer_workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        if getattr(self, "cascade_flashinfer_backend", None) != CASCADE_SUFFIX_KERNEL:
            if CASCADE_SUFFIX_KERNEL == "flashinfer_shared":
                wrapper_cls = getattr(flashinfer, "BatchDecodeWithSharedPrefixPagedKVCacheWrapper", None)
                if wrapper_cls is None:
                    raise RuntimeError(
                        "NANO_VLLM_CASCADE_SUFFIX_KERNEL=flashinfer_shared requires "
                        "FlashInfer BatchDecodeWithSharedPrefixPagedKVCacheWrapper"
                    )
                self.cascade_flashinfer_wrapper = wrapper_cls(
                    self.cascade_flashinfer_workspace, "NHD"
                )
            else:
                wrapper_cls = getattr(flashinfer, "MultiLevelCascadeAttentionWrapper", None)
                if wrapper_cls is None:
                    raise RuntimeError(
                        "NANO_VLLM_CASCADE_SUFFIX_KERNEL=flashinfer requires "
                        "FlashInfer MultiLevelCascadeAttentionWrapper"
                    )
                self.cascade_flashinfer_wrapper = wrapper_cls(
                    2, self.cascade_flashinfer_workspace, "NHD"
                )
            self.cascade_flashinfer_backend = CASCADE_SUFFIX_KERNEL

        plan_inputs = self.build_flashinfer_cascade_plan_inputs(
            N, shared_blocks, tail_rows, tail_lens_list, self.block_size
        )

        shared_query_indptr = _device_int_tensor(plan_inputs["query_indptr"][0], device)
        unique_query_indptr = _device_int_tensor(plan_inputs["query_indptr"][1], device)
        shared_indptr = _device_int_tensor(plan_inputs["kv_indptr"][0], device)
        unique_indptr = _device_int_tensor(plan_inputs["kv_indptr"][1], device)
        shared_indices = _device_int_tensor(plan_inputs["kv_indices"][0], device)
        unique_indices = _device_int_tensor(plan_inputs["kv_indices"][1], device)
        shared_last_page_len = _device_int_tensor(plan_inputs["last_page_len"][0], device)
        unique_last_page_len = _device_int_tensor(plan_inputs["last_page_len"][1], device)

        if CASCADE_SUFFIX_KERNEL == "flashinfer_shared":
            self.cascade_flashinfer_wrapper.begin_forward(
                unique_indptr,
                unique_indices,
                unique_last_page_len,
                hf_config.num_attention_heads // self.world_size,
                hf_config.num_key_value_heads // self.world_size,
                head_dim,
                self.block_size,
                data_type=dtype_name,
            )
        else:
            self.cascade_flashinfer_wrapper.plan(
                [shared_query_indptr, unique_query_indptr],
                [shared_indptr, unique_indptr],
                [shared_indices, unique_indices],
                [shared_last_page_len, unique_last_page_len],
                hf_config.num_attention_heads // self.world_size,
                hf_config.num_key_value_heads // self.world_size,
                head_dim,
                self.block_size,
                causal=True,
                sm_scale=head_dim ** -0.5,
                q_data_type=dtype_name,
                kv_data_type=dtype_name,
            )
        return self.cascade_flashinfer_wrapper

    @staticmethod
    def plan_cascade_decode(seqs: list[Sequence], block_size: int):
        cascade_min_blocks = 1
        cascade_min_work = 32768
        empty_plan = ([], 0, 0, [], [])
        if len(seqs) < 2:
            return empty_plan

        first_table = seqs[0].block_table
        common_len = len(first_table)
        min_blocks_per_seq = common_len
        for seq in seqs[1:]:
            block_table = seq.block_table
            min_blocks_per_seq = min(min_blocks_per_seq, len(block_table))
            limit = min(common_len, len(block_table))
            i = 0
            while i < limit and first_table[i] == block_table[i]:
                i += 1
            common_len = i
            if common_len < cascade_min_blocks:
                return empty_plan

        if common_len >= min_blocks_per_seq:
            return empty_plan
        shared_prefix_len = common_len * block_size
        if len(seqs) * shared_prefix_len < cascade_min_work:
            return empty_plan

        shared_blocks = first_table[:common_len]
        max_tail_blocks = 0
        tail_rows = []
        tail_lens_list = []
        for seq in seqs:
            ctxlen = len(seq)
            remaining_decode_steps = max(seq.max_tokens - seq.num_completion_tokens - 1, 0)
            max_ctxlen = ctxlen + remaining_decode_steps
            max_blocks = (max_ctxlen + block_size - 1) // block_size
            current_tail_blocks = len(seq.block_table) - common_len
            max_tail_blocks = max(max_tail_blocks, max_blocks - common_len, current_tail_blocks)
            tail_lens_list.append(ctxlen - shared_prefix_len)
        for seq in seqs:
            tail = seq.block_table[common_len:]
            tail_rows.append(tail + [-1] * (max_tail_blocks - len(tail)))
        return shared_blocks, shared_prefix_len, max_tail_blocks, tail_rows, tail_lens_list

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
        shared_blocks, shared_prefix_len, max_tail_blocks, tail_rows, tail_lens_list = self.plan_cascade_decode(seqs, self.block_size)

        if shared_prefix_len > 0:
            N = len(seqs)
            num_shared = len(shared_blocks)
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
            # Use the allocated tail span, not the current max tail length, as
            # the FA varlen launch bound. This keeps the cascade graph key
            # stable for every decode step that stays within the same tail
            # block count.
            tail_max = max_tail_blocks * self.block_size
            block_tables = None
            from nanovllm.layers.attention import CASCADE_SUFFIX_KERNEL
            if CASCADE_SUFFIX_KERNEL in {"flashinfer", "flashinfer_shared"}:
                cascade_wrapper = self.plan_flashinfer_cascade(
                    N, shared_blocks, tail_rows, tail_lens_list, shared_prefix_blocks_t.device
                )
            else:
                cascade_wrapper = None
        else:
            shared_prefix_blocks_t = None
            tail_block_tables_t = None
            tail_lens_t = None
            cu_q_pref = cu_k_pref = cu_q_suff = cu_k_suff = None
            sig = 0
            tail_max = 0
            block_tables = self.prepare_block_tables(seqs)
            cascade_wrapper = None

        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens_t, block_tables=block_tables,
                    shared_prefix_blocks=shared_prefix_blocks_t, shared_prefix_len=shared_prefix_len,
                    shared_prefix_signature=sig, tail_block_tables=tail_block_tables_t, tail_lens=tail_lens_t,
                    tail_max_len=tail_max,
                    cu_q_pref=cu_q_pref, cu_k_pref=cu_k_pref, cu_q_suff=cu_q_suff, cu_k_suff=cu_k_suff,
                    cascade_wrapper=cascade_wrapper)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        ctx = get_context()
        if ctx.shared_prefix_len > 0 and ctx.cascade_wrapper is not None:
            return self.model.compute_logits(self.model(input_ids, positions))
        if ctx.shared_prefix_len > 0 and not is_prefill and not self.enforce_eager:
            return self.run_cascade_graph(input_ids, positions)
        if is_prefill or self.enforce_eager or input_ids.size(0) > self.graph_max_bs:
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

    def cascade_graph_key(self, ctx):
        missing = [
            name
            for name in (
                "slot_mapping",
                "context_lens",
                "shared_prefix_blocks",
                "tail_block_tables",
                "tail_lens",
                "cu_q_pref",
                "cu_k_pref",
                "cu_q_suff",
                "cu_k_suff",
            )
            if getattr(ctx, name, None) is None
        ]
        if missing:
            raise RuntimeError(
                "cascade CUDA graph requires a complete decode context; "
                f"missing {', '.join(missing)}"
            )
        if ctx.slot_mapping.dim() != 1:
            raise RuntimeError(
                "cascade CUDA graph requires slot_mapping to be a 1D tensor; "
                f"got shape {tuple(ctx.slot_mapping.shape)}"
            )
        batch_size = ctx.slot_mapping.size(0)
        expected_batch_tensors = (
            "context_lens",
            "tail_lens",
        )
        for name in expected_batch_tensors:
            tensor = getattr(ctx, name)
            if tensor.dim() != 1 or tensor.size(0) != batch_size:
                raise RuntimeError(
                    "cascade CUDA graph requires decode context tensors to "
                    f"match batch size {batch_size}; {name} has shape {tuple(tensor.shape)}"
                )
        if ctx.tail_block_tables.dim() != 2 or ctx.tail_block_tables.size(0) != batch_size:
            raise RuntimeError(
                "cascade CUDA graph requires tail_block_tables to have one row "
                f"per sequence; got shape {tuple(ctx.tail_block_tables.shape)} "
                f"for batch size {batch_size}"
            )
        expected_cu_len = batch_size + 1
        for name in ("cu_q_suff", "cu_k_suff"):
            tensor = getattr(ctx, name)
            if tensor.dim() != 1 or tensor.size(0) != expected_cu_len:
                raise RuntimeError(
                    "cascade CUDA graph requires suffix cu_seqlens to have "
                    f"length {expected_cu_len}; {name} has shape {tuple(tensor.shape)}"
                )
        for name in ("cu_q_pref", "cu_k_pref"):
            tensor = getattr(ctx, name)
            if tensor.dim() != 1 or tensor.size(0) != 2:
                raise RuntimeError(
                    "cascade CUDA graph requires prefix cu_seqlens to have "
                    f"length 2; {name} has shape {tuple(tensor.shape)}"
                )
        if ctx.shared_prefix_blocks.dim() != 1:
            raise RuntimeError(
                "cascade CUDA graph requires shared_prefix_blocks to be a "
                f"1D tensor; got shape {tuple(ctx.shared_prefix_blocks.shape)}"
            )
        return (
            batch_size,
            ctx.shared_prefix_blocks.size(0),
            ctx.tail_block_tables.size(1),
        )

    @staticmethod
    def restore_decode_context(ctx):
        set_context(
            False,
            slot_mapping=ctx.slot_mapping,
            context_lens=ctx.context_lens,
            block_tables=ctx.block_tables,
            shared_prefix_blocks=ctx.shared_prefix_blocks,
            shared_prefix_len=ctx.shared_prefix_len,
            shared_prefix_signature=ctx.shared_prefix_signature,
            tail_block_tables=ctx.tail_block_tables,
            tail_lens=ctx.tail_lens,
            tail_max_len=ctx.tail_max_len,
            cu_q_pref=ctx.cu_q_pref,
            cu_k_pref=ctx.cu_k_pref,
            cu_q_suff=ctx.cu_q_suff,
            cu_k_suff=ctx.cu_k_suff,
            cascade_wrapper=ctx.cascade_wrapper,
        )

    @torch.inference_mode()
    def run_cascade_graph(self, input_ids: torch.Tensor, positions: torch.Tensor):
        ctx = get_context()
        key = self.cascade_graph_key(ctx)
        batch_size = key[0]
        for name, tensor in (("input_ids", input_ids), ("positions", positions)):
            if tensor.dim() != 1 or tensor.size(0) != batch_size:
                raise RuntimeError(
                    "cascade CUDA graph requires input_ids and positions to "
                    f"match batch size {batch_size}; {name} has shape {tuple(tensor.shape)}"
                )
        if key in self.cascade_graph_failures:
            return self.model.compute_logits(self.model(input_ids, positions))
        if key not in self.cascade_graphs:
            try:
                self.capture_cascade_cudagraph(key, ctx, input_ids, positions)
            except RuntimeError:
                self.cascade_graph_failures.add(key)
                return self.model.compute_logits(self.model(input_ids, positions))
            finally:
                # capture_cascade_cudagraph installs a capture context; restore
                # the real step context before copying this step's live values
                # or falling back to eager execution.
                self.restore_decode_context(ctx)

        entry = self.cascade_graphs[key]
        graph_vars = entry["vars"]
        graph_vars["input_ids"].copy_(input_ids)
        graph_vars["positions"].copy_(positions)
        graph_vars["slot_mapping"].copy_(ctx.slot_mapping)
        graph_vars["context_lens"].copy_(ctx.context_lens)
        graph_vars["shared_prefix_blocks"].copy_(ctx.shared_prefix_blocks)
        graph_vars["tail_block_tables"].copy_(ctx.tail_block_tables)
        graph_vars["tail_lens"].copy_(ctx.tail_lens)
        graph_vars["cu_q_pref"].copy_(ctx.cu_q_pref)
        graph_vars["cu_k_pref"].copy_(ctx.cu_k_pref)
        graph_vars["cu_q_suff"].copy_(ctx.cu_q_suff)
        graph_vars["cu_k_suff"].copy_(ctx.cu_k_suff)
        entry["graph"].replay()
        return self.model.compute_logits(graph_vars["outputs"][:batch_size])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        try:
            temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
            logits = self.run_model(input_ids, positions, is_prefill)
            return self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        finally:
            reset_context()

    @staticmethod
    def cudagraph_batch_sizes(max_bs: int):
        dense = [1, 2, 4, 8] + list(range(16, min(max_bs, 512) + 1, 16))
        coarse = list(range(576, max_bs + 1, 64)) if max_bs > 512 else []
        graph_bs = [bs for bs in dense + coarse if bs <= max_bs]
        if not graph_bs or graph_bs[-1] != max_bs:
            graph_bs.append(max_bs)
        return graph_bs

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
        self.graph_bs = self.cudagraph_batch_sizes(max_bs)
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

    @torch.inference_mode()
    def capture_cascade_cudagraph(self, key, ctx, input_ids: torch.Tensor, positions: torch.Tensor):
        bs, num_shared_blocks, max_tail_blocks = key
        hf_config = self.config.hf_config
        device = input_ids.device
        model_dtype = _config_torch_dtype(hf_config)

        graph_vars = dict(
            input_ids=torch.empty(bs, dtype=torch.int64, device=device),
            positions=torch.empty(bs, dtype=torch.int64, device=device),
            slot_mapping=torch.empty(bs, dtype=torch.int32, device=device),
            context_lens=torch.empty(bs, dtype=torch.int32, device=device),
            shared_prefix_blocks=torch.empty(num_shared_blocks, dtype=torch.int32, device=device),
            tail_block_tables=torch.empty(bs, max_tail_blocks, dtype=torch.int32, device=device),
            tail_lens=torch.empty(bs, dtype=torch.int32, device=device),
            cu_q_pref=torch.empty(2, dtype=torch.int32, device=device),
            cu_k_pref=torch.empty(2, dtype=torch.int32, device=device),
            cu_q_suff=torch.empty(bs + 1, dtype=torch.int32, device=device),
            cu_k_suff=torch.empty(bs + 1, dtype=torch.int32, device=device),
            outputs=torch.empty(bs, hf_config.hidden_size, dtype=model_dtype, device=device),
        )
        graph_vars["input_ids"].copy_(input_ids)
        graph_vars["positions"].copy_(positions)
        graph_vars["slot_mapping"].copy_(ctx.slot_mapping)
        graph_vars["context_lens"].copy_(ctx.context_lens)
        graph_vars["shared_prefix_blocks"].copy_(ctx.shared_prefix_blocks)
        graph_vars["tail_block_tables"].copy_(ctx.tail_block_tables)
        graph_vars["tail_lens"].copy_(ctx.tail_lens)
        graph_vars["cu_q_pref"].copy_(ctx.cu_q_pref)
        graph_vars["cu_k_pref"].copy_(ctx.cu_k_pref)
        graph_vars["cu_q_suff"].copy_(ctx.cu_q_suff)
        graph_vars["cu_k_suff"].copy_(ctx.cu_k_suff)

        set_context(
            False,
            slot_mapping=graph_vars["slot_mapping"],
            context_lens=graph_vars["context_lens"],
            block_tables=None,
            shared_prefix_blocks=graph_vars["shared_prefix_blocks"],
            shared_prefix_len=ctx.shared_prefix_len,
            shared_prefix_signature=ctx.shared_prefix_signature,
            tail_block_tables=graph_vars["tail_block_tables"],
            tail_lens=graph_vars["tail_lens"],
            tail_max_len=ctx.tail_max_len,
            cu_q_pref=graph_vars["cu_q_pref"],
            cu_k_pref=graph_vars["cu_k_pref"],
            cu_q_suff=graph_vars["cu_q_suff"],
            cu_k_suff=graph_vars["cu_k_suff"],
        )

        graph = torch.cuda.CUDAGraph()
        graph_vars["outputs"][:] = self.model(graph_vars["input_ids"], graph_vars["positions"])
        with torch.cuda.graph(graph, self.graph_pool):
            graph_vars["outputs"][:] = self.model(graph_vars["input_ids"], graph_vars["positions"])
        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        torch.cuda.synchronize()
        self.cascade_graphs[key] = {"graph": graph, "vars": graph_vars}
