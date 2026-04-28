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

Both engines benched over HTTP `/v1/completions` so the request paths
match. Nano-vLLM is served by the bundled OpenAI-compatible server
(`python -m nanovllm.server`, sources in
[`nanovllm/server.py`](nanovllm/server.py)). Each `(N, prefix_tokens)`
cell sends N concurrent requests sharing one random prefix,
`output_tokens = clamp(131072/N, 64, 1024)`, pinned with
`min_tokens=max_tokens` and `ignore_eos=true`. Cell value is total
decode tokens / wall time (tok/s); the last column is cold-cache
prefill wall at that prefix length (one decode step, fresh seed). vLLM
uses NGC image `nvcr.io/nvidia/vllm:26.03.post1-py3` (only image with
working sm_121 kernels), `--gpu-memory-utilization 0.75`,
`max_num_batched_tokens=2048` (chunked prefill default). Nano-vLLM uses
`gpu_memory_utilization=0.65`, `max_num_batched_tokens=16384`,
`max_num_seqs=1024`.

**Nano-vLLM** (HTTP via `nanovllm.server`):

| prefix \ N | 1 | 4 | 16 | 64 | 256 | 1024 | prefill (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|     1  | 120 | 544 | 1,518 | 2,671 | 5,475 | 13,431 | 0.20 |
|  4,096 |  96 | 395 |   835 | 1,208 | 1,877 |  1,874 | 0.11 |
| 32,768 |  38 |  85 |   228 |   266 |   286 |    281 | 2.21 |

**vLLM** (NGC `26.03.post1-py3`, HTTP `/v1/completions`):

| prefix \ N | 1 | 4 | 16 | 64 | 256 | 1024 | prefill (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|     1  | 124 | 533 | 1,380 | 2,515 | 5,709 | 12,716 | 0.01 |
|  4,096 |  98 | 398 |   972 | 2,051 | 4,694 |  8,295 | 0.12 |
| 32,768 |  38 |  77 |   373 | 1,130 | 2,476 |  3,008 | 2.41 |

**Speed ratio (vLLM / Nano-vLLM)** — values < 1.10× mean parity:

| prefix \ N | 1 | 4 | 16 | 64 | 256 | 1024 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|     1  | 1.03× | 0.98× | 0.91× | 0.94× | 1.04× | 0.95× |
|  4,096 | 1.02× | 1.01× | **1.16×** | **1.70×** | **2.50×** | **4.43×** |
| 32,768 | 1.00× | 0.91× | **1.64×** | **4.25×** | **8.66×** | **10.71×** |

Parity row at L=1 (no shared prefix to amortize): nano-vllm matches or
beats vLLM. The gap opens up specifically on shared-prefix cells with
concurrency ≥ 16, and widens with both prefix length and concurrency —
nano-vllm's throughput plateaus around 280 tok/s at L=32768 while vLLM
keeps scaling to 3 k tok/s. Root causes documented in
[`bench/PERF_GAP.md`](bench/PERF_GAP.md). Harnesses + raw logs live
under [`bench/`](bench/): `bench_concurrency.py` drives the nano-vllm
HTTP server, `bench_vllm.py` the vLLM container,
`probe_nanovllm.py` runs the focused diagnostic experiments.

To reproduce locally:

```bash
# terminal 1: serve nano-vllm
uv run python -m nanovllm.server \
  --model ~/huggingface/Qwen3-0.6B/ \
  --gpu-memory-utilization 0.65 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 1024 \
  --max-model-len 34816

# terminal 2: run the sweep
uv run python bench/bench_concurrency.py
```
