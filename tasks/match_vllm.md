# Match vLLM throughput across the full sweep

## Goal

On every (N, prefix_length) cell of the bench sweep, reach ≥ 0.95× of vLLM's
output throughput, on DGX Spark (GB10) with Qwen3-0.6B. No regressions on
cells that already match.

## Sweep grid

`{N: 1, 4, 16, 64, 256, 1024} × {prefix: 1, 4096, 32768}`.

## Starting state

From `bench/bench_nanovllm.txt` and `bench/bench_vllm.txt` (pre-cascade-graph
snapshot — ratios = nano-vllm / vLLM):

| L \ N | 1 | 4 | 16 | 64 | 256 | 1024 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1     | 0.97× | 1.04× | 1.12× | 1.07× | 1.00× | 1.11× |
| 4096  | 0.97× | 0.99× | 0.91× | 0.79× | 0.60× | 0.44× |
| 32768 | 1.00× | 1.23× | 0.86× | 0.79× | 0.69× | 0.56× |

## How to verify

```bash
# Server (per README); for vLLM use NGC sm_121 container.
uv run python bench/bench_concurrency.py    # full sweep, nano-vllm
uv run python bench/bench_vllm.py           # full sweep, vLLM
```

Run each ≥ 3 times; report median per cell. Compare cell-by-cell.

