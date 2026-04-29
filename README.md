<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |

---

## Reproduction on NVIDIA DGX Spark

Hardware: GB10, sm_121, 128 GB unified memory.

`bench.py` headline run, same workload as above (256 seqs, in/out 100–1024 tok,
Qwen3-0.6B, `enforce_eager=False`). Benched in-process via the `LLM.generate`
batched API.

| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|------------------|--------------:|---------:|----------------------:|
| Nano-vLLM (Spark) | 133,966 | 71.27 | **1879.61** |

### Prefix-length × concurrency sweep — Nano-vLLM vs vLLM, same hardware

**Workload (per cell, shared by both engines):**

| field | value |
| --- | --- |
| transport | HTTP `/v1/completions` |
| concurrency | N |
| prompt | one random `prefix_tokens`-long prefix shared by all N requests |
| `output_tokens` | `clamp(131072/N, 64, 1024)`, pinned via `min_tokens=max_tokens`, `ignore_eos=true` |
| cell value | total decode tokens / wall time (tok/s) |
| prefill column | cold-cache wall on one decode step, fresh seed |

**Engine config:**

| | Nano-vLLM | vLLM |
| --- | --- | --- |
| server | `python -m nanovllm.server` ([`nanovllm/server.py`](nanovllm/server.py)) | NGC `nvcr.io/nvidia/vllm:26.03.post1-py3` (only image with working sm_121 kernels) |
| `gpu_memory_utilization` | 0.85 | 0.80 |
| `max_num_batched_tokens` | 16384 | 2048 (chunked prefill default) |
| `max_num_seqs` | 1024 | 1024 |
| `max_model_len` | 34816 | 34816 |
| KV-cache block size | 256 | default |

The higher nano-vllm `gpu_memory_utilization` keeps ~2,800 KV blocks live —
enough to avoid preemption at 1024 concurrent seqs with L=4 k or L=32 k
prompts.

**Nano-vLLM** (HTTP via `nanovllm.server`, with shared-prefix cascade
attention enabled when `N · shared_prefix ≥ 32 k` —
[`nanovllm/layers/attention.py`](nanovllm/layers/attention.py)):

| prefix \ N | 1 | 4 | 16 | 64 | 256 | 1024 | prefill (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|     1  | 120.0 | 550.4 | 1,663.8 | 4,656.9 | 8,253.2 | 14,078.6 | 0.01 |
|  4,096 |  95.5 | 422.2 | 1,150.6 | 2,342.8 | 4,808.0 |  9,834.0 | 0.11 |
| 32,768 |  37.9 | 132.9 |   480.3 | 1,308.4 | 2,722.0 |  3,722.6 | 2.20 |

**vLLM** (NGC `26.03.post1-py3`, HTTP `/v1/completions`):

| prefix \ N | 1 | 4 | 16 | 64 | 256 | 1024 | prefill (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|     1  | 119.8 | 544.3 | 1,502.6 | 2,588.5 | 5,446.7 | 14,389.6 | 0.01 |
|  4,096 |  96.8 | 380.3 |   940.1 | 1,925.3 | 4,366.9 |  9,422.0 | 0.12 |
| 32,768 |  36.4 |  74.7 |   323.2 |   998.0 | 2,311.7 |  3,706.2 | 2.58 |

**Speed ratio (vLLM / Nano-vLLM)** — values ≤ 1.10× mean parity, < 1
means nano-vllm beats vLLM:

| prefix \ N | 1 | 4 | 16 | 64 | 256 | 1024 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|     1  | 1.00× | 0.99× | 0.90× | 0.56× | 0.66× | 1.02× |
|  4,096 | 1.01× | 0.90× | 0.82× | 0.82× | 0.91× | 0.96× |
| 32,768 | 0.96× | 0.56× | 0.67× | 0.76× | 0.85× | 1.00× |

Fresh median-of-3 sweeps and raw per-iteration logs are documented in
[`bench/FULL_SWEEP_20260429.md`](bench/FULL_SWEEP_20260429.md). Summary tables:
[`bench/bench_nanovllm.txt`](bench/bench_nanovllm.txt) and
[`bench/bench_vllm.txt`](bench/bench_vllm.txt).

To reproduce locally:

```bash
# terminal 1: serve nano-vllm
uv run python -m nanovllm.server \
  --model ~/huggingface/Qwen3-0.6B/ \
  --host 127.0.0.1 --port 8001 \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 1024 \
  --max-model-len 34816 \
  --kvcache-block-size 256

# terminal 2: run the sweep
for i in 1 2 3; do
  uv run python bench/bench_concurrency.py \
    --url http://127.0.0.1:8001/v1/completions \
    > bench/raw_full_sweep_20260429_iter/nanovllm_full_run${i}.out \
    2> bench/raw_full_sweep_20260429_iter/nanovllm_full_run${i}.err
done
```
