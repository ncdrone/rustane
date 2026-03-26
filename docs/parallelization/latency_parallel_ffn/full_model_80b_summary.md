# Full-Model FFN Latency Parallel Comparison

Config: `80B` — 5120d/13824h/256L/seq512

| mode | compile(s) | warmup(ms) | mean forward(ms) | tok/s | peak RSS(MB) | ANE hw(ms) | ANE busy % | avg layer(ms) | total FFN(ms) | total non-FFN(ms) | FFN % | max abs diff | mean abs diff | cosine | loss delta | win vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.90 | 64650.2 | 56343.2 | 9.09 | 311832 | 0.0 | 0.0% | 220.091 | 43949.9 | 12393.3 | 78.0% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +0.00% |
| shard2 | 0.71 | 51069.3 | 51193.7 | 10.00 | 313313 | 0.0 | 0.0% | 199.976 | 39193.4 | 12000.3 | 76.6% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +9.14% |
| shard4 | 0.81 | 36202.1 | 36506.4 | 14.02 | 313924 | 0.0 | 0.0% | 142.603 | 24394.5 | 12111.9 | 66.8% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +35.21% |
| shard6 | 0.98 | 32693.3 | 32210.9 | 15.90 | 314357 | 0.0 | 0.0% | 125.824 | 20252.9 | 11958.0 | 62.9% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +42.83% |
| shard8 | 1.00 | 29796.6 | 29681.3 | 17.25 | 313103 | 0.0 | 0.0% | 115.943 | 17685.9 | 11995.4 | 59.6% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +47.32% |
| shard12 | 1.33 | 27550.1 | 27502.9 | 18.62 | 313092 | 0.0 | 0.0% | 107.433 | 15497.4 | 12005.5 | 56.3% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +51.19% |
| shard16 | 1.51 | 26885.1 | 26588.8 | 19.26 | 312879 | 0.0 | 0.0% | 103.862 | 14645.0 | 11943.8 | 55.1% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +52.81% |
| shard24 | 2.03 | 27698.8 | 27685.0 | 18.49 | 313123 | 0.0 | 0.0% | 108.144 | 15776.5 | 11908.4 | 57.0% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +50.86% |
| shard32 | 2.54 | 30353.8 | 30180.8 | 16.96 | 313438 | 0.0 | 0.0% | 117.894 | 18285.6 | 11895.2 | 60.6% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +46.43% |

## Thresholds

- max abs diff <= `2.000e-2`
- mean abs diff <= `5.000e-3`
- cosine similarity >= `0.999`
- loss abs diff <= `1.000e-4`
- full forward win >= `5.0%`

Primary success: **PASS**

Winning modes: `shard2`, `shard4`, `shard6`, `shard8`, `shard12`, `shard16`, `shard24`, `shard32`

## Notes

- Each mode runs 1 warmup plus 2 measured stats-enabled full-model passes.
- Reported wall time, FFN totals, RSS, and ANE utilization all come from the same measured samples.
- This harness uses the mirrored stats-enabled path, not `full_model::forward_only_ws(...)`.
