# Full-sweep verification, 2026-04-29

Hardware: DGX Spark / GB10, Qwen3-0.6B.

## nano-vLLM

Command:

```bash
uv run python -m nanovllm.server --model ~/huggingface/Qwen3-0.6B/ \
  --host 127.0.0.1 --port 8001 \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 1024 \
  --max-model-len 34816 \
  --kvcache-block-size 256

for i in 1 2 3; do
  uv run python bench/bench_concurrency.py \
    --url http://127.0.0.1:8001/v1/completions \
    > bench/raw_full_sweep_20260429_iter/nanovllm_full_run${i}.out \
    2> bench/raw_full_sweep_20260429_iter/nanovllm_full_run${i}.err
done
```

Median output tok/s across 3 complete runs:

Summary file: [`bench/bench_nanovllm.txt`](bench_nanovllm.txt). Raw logs:
[`bench/raw_full_sweep_20260429_iter/`](raw_full_sweep_20260429_iter/).

| prefix \ N | 1 | 4 | 16 | 64 | 256 | 1024 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 120.0 | 550.4 | 1663.8 | 4656.9 | 8253.2 | 14078.6 |
| 4096 | 95.5 | 422.2 | 1150.6 | 2342.8 | 4808.0 | 9834.0 |
| 32768 | 37.9 | 132.9 | 480.3 | 1308.4 | 2722.0 | 3722.6 |

Ratio vs the checked-in `bench/bench_vllm.txt` reference:

| prefix \ N | 1 | 4 | 16 | 64 | 256 | 1024 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.00x | 1.01x | 1.11x | 1.80x | 1.52x | 0.98x |
| 4096 | 0.99x | 1.11x | 1.22x | 1.22x | 1.10x | 1.04x |
| 32768 | 1.04x | 1.78x | 1.49x | 1.31x | 1.18x | 1.00x |

All 18 cells are >= 0.95x against that reference.

## vLLM

The local NGC image `nvcr.io/nvidia/vllm:26.03.post1-py3` is present. The
exact 0.85 memory-utilization launch was blocked by an unrelated existing GPU
process:

```text
ValueError: Free memory on device cuda:0 (101.61/121.69 GiB) on startup is less
than desired GPU memory utilization (0.85, 103.44 GiB).
```

A lower-utilization vLLM server was launched at 0.80 with the same model and
max sequence/concurrency settings:

```bash
docker run --rm --gpus all --ipc=host -p 8002:8000 \
  -v /srv/vllm:/srv/vllm:ro \
  nvcr.io/nvidia/vllm:26.03.post1-py3 \
  vllm serve /srv/vllm/hf/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 34816 \
    --gpu-memory-utilization 0.80 \
    --max-num-seqs 1024 \
    --enable-prefix-caching
```

The vLLM benchmark client was updated to accept `--timeout-s` and to serialize
the identical per-cell request body once, avoiding the client-side timeout and
memory pressure that invalidated the earlier attempt. Command:

```bash
for i in 1 2 3; do
  uv run python bench/bench_vllm.py \
    --url http://127.0.0.1:8002/v1/completions \
    --model /srv/vllm/hf/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
    --timeout-s 2400 \
    > bench/raw_vllm_full_sweep_20260429_valid/vllm_full_run${i}.out \
    2> bench/raw_vllm_full_sweep_20260429_valid/vllm_full_run${i}.err
done
```

Median output tok/s across 3 complete runs:

Summary file: [`bench/bench_vllm.txt`](bench_vllm.txt). Raw logs:
[`bench/raw_vllm_full_sweep_20260429_valid/`](raw_vllm_full_sweep_20260429_valid/).

| prefix \ N | 1 | 4 | 16 | 64 | 256 | 1024 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 119.8 | 544.3 | 1502.6 | 2588.5 | 5446.7 | 14389.6 |
| 4096 | 96.8 | 380.3 | 940.1 | 1925.3 | 4366.9 | 9422.0 |
| 32768 | 36.4 | 74.7 | 323.2 | 998.0 | 2311.7 | 3706.2 |

## Cell-by-cell comparison vs fresh vLLM medians

Ratio = nano-vLLM median / fresh vLLM median:

| prefix \ N | 1 | 4 | 16 | 64 | 256 | 1024 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.00x | 1.01x | 1.11x | 1.80x | 1.52x | 0.98x |
| 4096 | 0.99x | 1.11x | 1.22x | 1.22x | 1.10x | 1.04x |
| 32768 | 1.04x | 1.78x | 1.49x | 1.31x | 1.18x | 1.00x |

All 18 cells are >= 0.95x against the fresh vLLM medians from the same 0.80
capacity configuration. The lowest ratio is 0.978x at `prefix=1,N=1024`.
