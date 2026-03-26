# Full-Model FFN Latency Parallel Comparison

Config: `30B` — 5120d/13824h/96L/seq512

| mode | compile(s) | warmup(ms) | median forward(ms) | tok/s | peak RSS(MB) | avg layer(ms) | total FFN(ms) | total non-FFN(ms) | FFN % | max abs diff | mean abs diff | cosine | loss delta | win vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 3.14 | 19201.8 | 17579.5 | 29.12 | 119561 | 183.120 | 16980.2 | 599.3 | 96.6% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +0.00% |
| shard2 | 0.67 | 14705.3 | 13168.0 | 38.88 | 198966 | 137.167 | 7347.6 | 5820.4 | 55.8% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +25.09% |
| shard4 | 0.74 | 12890.8 | 11236.9 | 45.56 | 204544 | 117.051 | 5340.3 | 5896.6 | 47.5% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +36.08% |
| shard8 | 0.95 | 11980.8 | 10221.0 | 50.09 | 200246 | 106.469 | 4355.4 | 5865.7 | 42.6% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +41.86% |
| shard12 | 1.27 | 11658.5 | 10044.7 | 50.97 | 198436 | 104.633 | 4216.9 | 5827.9 | 42.0% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +42.86% |
| shard16 | 1.46 | 12036.1 | 10396.3 | 49.25 | 198246 | 108.295 | 4542.8 | 5853.5 | 43.7% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +40.86% |

## Thresholds

- max abs diff <= `2.000e-2`
- mean abs diff <= `5.000e-3`
- cosine similarity >= `0.999`
- loss abs diff <= `1.000e-4`
- full forward win >= `5.0%`

Primary success: **PASS**

Winning modes: `shard2`, `shard4`, `shard8`, `shard12`, `shard16`

## Notes

- Baseline wall time comes from `full_model::forward_only_ws(...)`.
- Baseline FFN totals come from a separate reference pass using `layer::forward_timed(...)` per layer.
- Sharded FFN totals come from the custom sharded full-model loop.
