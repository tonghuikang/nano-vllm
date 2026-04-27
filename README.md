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

### Reproduction on NVIDIA DGX Spark (GB10, sm_121, 128 GB unified memory) — 2026-04-27

`bench.py` headline run, same workload as above (256 seqs, in/out 100–1024 tok,
Qwen3-0.6B, `enforce_eager=False`). Benched in-process via the `LLM.generate`
batched API.

| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|------------------|--------------:|---------:|----------------------:|
| Nano-vLLM (Spark) | 133,966 | 71.27 | **1879.61** |

#### Prefix-length × concurrency sweep — Nano-vLLM vs vLLM, same hardware

Same harness as `~/Desktop/setup/spark/bench_vllm.py`: each `(N, prefix_tokens)`
cell sends N requests sharing one random prefix, `output_tokens =
clamp(131072/N, 64, 1024)`, pinned with `min_tokens=max_tokens` and
`ignore_eos=true`. Cell value is total decode tokens / wall time (tok/s).
vLLM uses NGC image `nvcr.io/nvidia/vllm:26.03.post1-py3` (only image with
working sm_121 kernels), `--gpu-memory-utilization 0.75`,
`max_num_batched_tokens=2048` (chunked prefill default). Nano-vLLM uses
`gpu_memory_utilization=0.65`, `max_num_batched_tokens=16384`,
`max_num_seqs=1024`.

**Nano-vLLM** (in-process `LLM.generate`):

| prefix \ N | 1 | 4 | 16 | 64 | 256 | 1024 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|     1  | 131 | 551 | 1,524 | 2,641 | 5,489 | 14,375 |
|  4,096 | 101 | 259 |   764 | 1,160 | 1,836 |  1,904 |
| 32,768 |  39 |  87 |   227 |   265 |   —¹  |    —¹  |

**vLLM** (NGC `26.03.post1-py3`, HTTP `/v1/completions`):

| prefix \ N | 1 | 4 | 16 | 64 | 256 | 1024 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|     1  | 136 | 555 | 1,382 | 2,489 | 5,535 | 12,584 |
|  4,096 | 105 | 393 |   978 | 2,026 | 4,579 |  8,179 |
| 32,768 |  38 |  95 |   334 | 1,088 | 2,326 |  3,003 |

¹ Cell aborted: nano-vllm's prefill-then-decode-only scheduler stalls heavily
on these cells; >5 min wall, throughput would have been < vLLM's.

**Speed ratio (vLLM / Nano-vLLM)** — values < 1.10× mean parity:

| prefix \ N | 1 | 4 | 16 | 64 | 256 | 1024 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|     1  | 1.04× | 1.01× | 0.91× | 0.94× | 1.01× | 0.88× |
|  4,096 | 1.04× | **1.52×** | **1.28×** | **1.75×** | **2.49×** | **4.30×** |
| 32,768 | 0.97× | 1.09× | **1.47×** | **4.10×** | n/a | n/a |

Parity row at L=1 (no shared prefix to amortize): nano-vllm matches or beats
vLLM. The gap opens up specifically on shared-prefix cells with concurrency
≥ 4. Root causes documented in
[`bench/PERF_GAP.md`](bench/PERF_GAP.md). Harnesses + raw logs live under
[`bench/`](bench/) (`bench_concurrency.py` for nano-vllm in-process,
`bench_vllm.py` for the vLLM HTTP path, `probe_nanovllm.py` for the
focused diagnostic experiments).


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)