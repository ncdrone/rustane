# 600M Sharded Backward Sweep Summary

Date: 2026-03-25

Commands run:

```bash
make sweep-600m
make sweep-600m FFN_SHARDS=4
make sweep-600m ATTN_SHARDS=2
make sweep-600m ATTN_SHARDS=2 FFN_SHARDS=4
```

## Results

| Config | ms/step | fwd | bwd | upd | tok/s | Loss delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1042 | 258 | 698 | 66 | 491 | -0.0251 |
| `FFN_SHARDS=4` | 2017 | 911 | 1023 | 62 | 254 | -0.0251 |
| `ATTN_SHARDS=2` | 1385 | 261 | 1050 | 57 | 370 | -0.0251 |
| `ATTN_SHARDS=2 FFN_SHARDS=4` | 2144 | 933 | 1134 | 60 | 239 | -0.0251 |

## Regression Collapse vs Original Plan Numbers

| Config | Original `bwd` | Current `bwd` | Improvement |
| --- | ---: | ---: | ---: |
| `ATTN_SHARDS=2` | 1174 | 1050 | -124 ms |
| `FFN_SHARDS=4` | 1978 | 1023 | -955 ms |
| Combined | 2258 | 1134 | -1124 ms |

## Notes

- All four runs converged normally over the 10-step training check.
- FFN backward parallelization delivered the largest win.
- Attention backward parallelization helped, but less than FFN.
- Combined backward time is much better than the original combined regression, but forward time remains heavily inflated in FFN-sharded modes.
- The FFN-sharded training forward path previously cloned the full `x_buf` once per layer. That host copy was removed and `make sweep-600m FFN_SHARDS=4` was re-run immediately afterward. The rerun measured `2070 ms/step (fwd=937 bwd=1043 upd=63)`, which is within noise of the original `2017 / 911 / 1023 / 62` result. Conclusion: the clone was real overhead, but not the primary forward bottleneck.
