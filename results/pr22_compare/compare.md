# PR #22 Forward Comparison

## Single-stream common-path comparison

| scale | base tok/s | head tok/s | tok/s delta | base fwd(ms) | head fwd(ms) | fwd delta | base peak RSS(MB) | head peak RSS(MB) | RSS delta | head FFN mode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 5B | 0.0 | 155.4 | n/a | 0.0 | 3294.5 | +3294.5 ms | 15 | 28576 | +28562 MB | w1_w3_w2 |
| 13B | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| 20B | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |

## Layer bucket deltas

| scale | base ane% | head ane% | ane delta | base stage% | head stage% | stage delta | base read% | head read% | read delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5B | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| 13B | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |

## Head-only lean forward addendum

| scale | common tok/s | lean tok/s | lean delta | common peak RSS(MB) | lean peak RSS(MB) | lean RSS delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5B | 155.4 | 156.3 | +0.6% | 28576 | 20253 | -8323 MB |
| 13B | 56.7 | 56.9 | +0.4% | 70555 | 54262 | -16293 MB |
| 20B | 35.5 | 35.6 | +0.3% | 109428 | 82075 | -27353 MB |

## Notes

- `head FFN mode` is reported from the head-side scale results.
- Kernel wall-vs-hw results are intentionally excluded from the delta table until non-zero hardware timing is available.

