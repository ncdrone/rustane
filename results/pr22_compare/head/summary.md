# Forward Utilization Summary

## Full-model scale ladder

| name | params(B) | compile(s) | alloc(s) | median fwd(ms) | tok/s | est RAM(GB) | rss compile(MB) | rss alloc(MB) | rss warm(MB) | rss peak(MB) | dispatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5B | 5.01 | 1.2 | 3.3 | 3294.5 | 155.4 | 25.2 | 16 | 13327 | 28576 | 28576 | 220 |
| 13B | 12.73 | 3.2 | 8.4 | 9036.4 | 56.7 | 58.6 | 3526 | 37230 | 70555 | 70555 | 200 |
| 20B | 20.34 | 3.2 | 13.1 | 14425.9 | 35.5 | 93.3 | 10073 | 58800 | 109428 | 109428 | 320 |

## RSS checkpoints

| name | after compile | after alloc | after warmup | peak timed |
| --- | ---: | ---: | ---: | ---: |
| 5B | 16 MB | 13327 MB | 28576 MB | 28576 MB |
| 13B | 3526 MB | 37230 MB | 70555 MB | 70555 MB |
| 20B | 10073 MB | 58800 MB | 109428 MB | 109428 MB |

## Single-layer bucket shares

| scale | total(ms) | ane(ms) | stage(ms) | read(ms) | cpu(ms) | ane% | stage% | read% | cpu% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5B | 79.45 | 64.38 | 11.66 | 1.68 | 0.74 | 81.0% | 14.7% | 2.1% | 0.9% |
| 13B | 225.66 | 200.65 | 19.43 | 2.79 | 1.14 | 88.9% | 8.6% | 1.2% | 0.5% |
| 20B | 225.04 | 199.87 | 19.68 | 2.94 | 1.19 | 88.8% | 8.7% | 1.3% | 0.5% |

## Per-kernel wall vs hardware

| scale | kernel | wall median(us) | hw median(ns) | overhead(us) | overhead% |
| --- | --- | ---: | ---: | ---: | ---: |
| 5B | sdpa_fwd | 6981.0 | 0 | 6981.0 | 100.0% |
| 5B | wo_fwd | 1990.5 | 0 | 1990.5 | 100.0% |
| 5B | ffn_w1_fwd | 4861.7 | 0 | 4861.7 | 100.0% |
| 5B | ffn_w3_fwd | 4878.9 | 0 | 4878.9 | 100.0% |
| 5B | ffn_w2_fwd | 7977.3 | 0 | 7977.3 | 100.0% |
| 13B | sdpa_fwd | 33525.5 | 0 | 33525.5 | 100.0% |
| 13B | wo_fwd | 8328.5 | 0 | 8328.5 | 100.0% |
| 13B | ffn_w1_fwd | 24843.9 | 0 | 24843.9 | 100.0% |
| 13B | ffn_w3_fwd | 24887.9 | 0 | 24887.9 | 100.0% |
| 13B | ffn_w2_fwd | 41872.7 | 0 | 41872.7 | 100.0% |
| 20B | sdpa_fwd | 33470.1 | 0 | 33470.1 | 100.0% |
| 20B | wo_fwd | 8356.9 | 0 | 8356.9 | 100.0% |
| 20B | ffn_w1_fwd | 24838.5 | 0 | 24838.5 | 100.0% |
| 20B | ffn_w3_fwd | 24938.2 | 0 | 24938.2 | 100.0% |
| 20B | ffn_w2_fwd | 41915.2 | 0 | 41915.2 | 100.0% |

## Conclusion

Current classification: **staging/readback dominated**.

## Recommendation

Next action: **request/dispatch reduction**.

