# FFN Latency Parallel Scale Comparison

## Thresholds

- max abs diff <= `2.000e-2`
- mean abs diff <= `5.000e-3`
- cosine similarity >= `0.999`
- full-layer win >= `5.0%`

## Best Mode Per Scale

| scale | baseline(ms) | best mode | best full-layer(ms) | win vs baseline | primary success |
| --- | ---: | --- | ---: | ---: | --- |
| 30B | 218.6 | shard8 | 90.1 | +58.78% | PASS |
| 50B | 215.9 | shard8 | 92.9 | +56.96% | PASS |
| 80B | 217.9 | shard8 | 91.8 | +57.89% | PASS |
| 100B | 216.2 | shard8 | 92.6 | +57.18% | PASS |

## 30B — 5120d/13824h/96L/seq512

| mode | compile(s) | median full-layer(ms) | median ffn(ms) | peak RSS(MB) | max abs diff | mean abs diff | cosine | win vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 3.13 | 218.6 | 165.9 | 3324 | 0.0000 | 0.0000 | 1.000000 | +0.00% |
| shard2 | 0.67 | 122.6 | 76.3 | 5412 | 0.0000 | 0.0000 | 1.000000 | +43.93% |
| shard4 | 0.73 | 102.8 | 56.0 | 6688 | 0.0000 | 0.0000 | 1.000000 | +52.96% |
| shard6 | 0.95 | 95.7 | 48.9 | 6316 | 0.0000 | 0.0000 | 1.000000 | +56.21% |
| shard8 | 0.97 | 90.1 | 43.9 | 5608 | 0.0000 | 0.0000 | 1.000000 | +58.78% |

Primary success: **PASS**

Winning modes: `shard2`, `shard4`, `shard6`, `shard8`


## 50B — 5120d/13824h/160L/seq512

| mode | compile(s) | median full-layer(ms) | median ffn(ms) | peak RSS(MB) | max abs diff | mean abs diff | cosine | win vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 3.09 | 215.9 | 163.1 | 4188 | 0.0000 | 0.0000 | 1.000000 | +0.00% |
| shard2 | 0.65 | 122.9 | 76.6 | 5955 | 0.0000 | 0.0000 | 1.000000 | +43.06% |
| shard4 | 0.73 | 102.6 | 56.0 | 6796 | 0.0000 | 0.0000 | 1.000000 | +52.48% |
| shard6 | 0.94 | 94.3 | 48.0 | 6380 | 0.0000 | 0.0000 | 1.000000 | +56.34% |
| shard8 | 0.98 | 92.9 | 45.6 | 5586 | 0.0000 | 0.0000 | 1.000000 | +56.96% |

Primary success: **PASS**

Winning modes: `shard2`, `shard4`, `shard6`, `shard8`


## 80B — 5120d/13824h/256L/seq512

| mode | compile(s) | median full-layer(ms) | median ffn(ms) | peak RSS(MB) | max abs diff | mean abs diff | cosine | win vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 3.09 | 217.9 | 164.9 | 4168 | 0.0000 | 0.0000 | 1.000000 | +0.00% |
| shard2 | 0.65 | 122.4 | 75.9 | 6070 | 0.0000 | 0.0000 | 1.000000 | +43.81% |
| shard4 | 0.73 | 102.2 | 55.7 | 6914 | 0.0000 | 0.0000 | 1.000000 | +53.10% |
| shard6 | 0.93 | 94.8 | 47.9 | 6367 | 0.0000 | 0.0000 | 1.000000 | +56.48% |
| shard8 | 0.97 | 91.8 | 45.1 | 5566 | 0.0000 | 0.0000 | 1.000000 | +57.89% |

Primary success: **PASS**

Winning modes: `shard2`, `shard4`, `shard6`, `shard8`


## 100B — 5120d/13824h/320L/seq512

| mode | compile(s) | median full-layer(ms) | median ffn(ms) | peak RSS(MB) | max abs diff | mean abs diff | cosine | win vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 3.08 | 216.2 | 163.4 | 4146 | 0.0000 | 0.0000 | 1.000000 | +0.00% |
| shard2 | 0.65 | 121.4 | 74.9 | 6183 | 0.0000 | 0.0000 | 1.000000 | +43.84% |
| shard4 | 0.73 | 102.7 | 56.7 | 6876 | 0.0000 | 0.0000 | 1.000000 | +52.52% |
| shard6 | 0.92 | 95.2 | 48.5 | 6350 | 0.0000 | 0.0000 | 1.000000 | +55.96% |
| shard8 | 0.97 | 92.6 | 45.8 | 5610 | 0.0000 | 0.0000 | 1.000000 | +57.18% |

Primary success: **PASS**

Winning modes: `shard2`, `shard4`, `shard6`, `shard8`

## Notes

- Baseline full-layer timing comes from `layer::forward_timed(...)`.
- Baseline FFN buckets are coarse runtime buckets; sharded buckets are max-per-shard phase times plus merge wall.
