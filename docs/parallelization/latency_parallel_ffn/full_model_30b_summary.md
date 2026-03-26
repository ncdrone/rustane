# Full-Model FFN Latency Parallel Comparison

Config: `30B` — 5120d/13824h/96L/seq512

| mode | compile(s) | warmup(ms) | mean forward(ms) | tok/s | peak RSS(MB) | ANE hw(ms) | ANE busy % | avg layer(ms) | total FFN(ms) | total non-FFN(ms) | FFN % | max abs diff | mean abs diff | cosine | loss delta | win vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.87 | 24267.2 | 20944.1 | 24.45 | 118207 | 0.0 | 0.0% | 218.167 | 16274.6 | 4669.4 | 77.7% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +0.00% |
| shard2 | 0.69 | 18657.4 | 18740.5 | 27.32 | 119687 | 0.0 | 0.0% | 195.214 | 14204.8 | 4535.7 | 75.8% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +10.52% |
| shard4 | 0.80 | 13123.3 | 13181.1 | 38.84 | 120298 | 0.0 | 0.0% | 137.303 | 8698.1 | 4483.0 | 66.0% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +37.07% |
| shard6 | 0.99 | 11892.2 | 11877.7 | 43.11 | 120731 | 0.0 | 0.0% | 123.726 | 7401.2 | 4476.5 | 62.3% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +43.29% |
| shard8 | 1.04 | 10883.5 | 10856.0 | 47.16 | 119477 | 0.0 | 0.0% | 113.084 | 6372.9 | 4483.2 | 58.7% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +48.17% |
| shard12 | 1.36 | 10156.5 | 10098.3 | 50.70 | 119466 | 0.0 | 0.0% | 105.190 | 5615.4 | 4482.9 | 55.6% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +51.78% |
| shard16 | 1.52 | 9998.4 | 9912.6 | 51.65 | 119252 | 0.0 | 0.0% | 103.256 | 5423.6 | 4489.0 | 54.7% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +52.67% |

## Thresholds

- max abs diff <= `2.000e-2`
- mean abs diff <= `5.000e-3`
- cosine similarity >= `0.999`
- loss abs diff <= `1.000e-4`
- full forward win >= `5.0%`

Primary success: **PASS**

Winning modes: `shard2`, `shard4`, `shard6`, `shard8`, `shard12`, `shard16`

## Notes

- Each mode runs 1 warmup plus 2 measured stats-enabled full-model passes.
- Reported wall time, FFN totals, RSS, and ANE utilization all come from the same measured samples.
- This harness uses the mirrored stats-enabled path, not `full_model::forward_only_ws(...)`.
