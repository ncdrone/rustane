# Full-Model FFN Latency Parallel Comparison

Config: `50B` — 5120d/13824h/160L/seq512

| mode | compile(s) | warmup(ms) | mean forward(ms) | tok/s | peak RSS(MB) | ANE hw(ms) | ANE busy % | avg layer(ms) | total FFN(ms) | total non-FFN(ms) | FFN % | max abs diff | mean abs diff | cosine | loss delta | win vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.92 | 40673.8 | 35340.9 | 14.49 | 195658 | 0.0 | 0.0% | 220.881 | 27568.5 | 7772.4 | 78.0% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +0.00% |
| shard2 | 0.71 | 32043.8 | 32069.3 | 15.97 | 197136 | 0.0 | 0.0% | 200.433 | 24511.0 | 7558.4 | 76.4% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +9.26% |
| shard4 | 0.84 | 22648.2 | 22621.7 | 22.63 | 197748 | 0.0 | 0.0% | 141.386 | 15066.5 | 7555.2 | 66.6% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +35.99% |
| shard6 | 0.99 | 20242.8 | 20127.4 | 25.44 | 198181 | 0.0 | 0.0% | 125.797 | 12613.7 | 7513.8 | 62.7% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +43.05% |
| shard8 | 1.02 | 18369.6 | 18347.1 | 27.91 | 196927 | 0.0 | 0.0% | 114.669 | 10869.4 | 7477.6 | 59.2% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +48.09% |
| shard12 | 1.37 | 17056.0 | 17006.9 | 30.11 | 196918 | 0.0 | 0.0% | 106.293 | 9538.3 | 7468.6 | 56.1% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +51.88% |
| shard16 | 1.53 | 16767.5 | 16718.6 | 30.62 | 196704 | 0.0 | 0.0% | 104.491 | 9242.1 | 7476.5 | 55.3% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +52.69% |

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
