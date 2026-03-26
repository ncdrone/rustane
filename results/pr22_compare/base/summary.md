# Forward Utilization Summary

## Full-model scale ladder

| name | params(B) | compile(s) | alloc(s) | median fwd(ms) | tok/s | est RAM(GB) | rss compile(MB) | rss alloc(MB) | rss warm(MB) | rss peak(MB) | dispatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5B | 5.01 | FAIL | FAIL | FAIL | FAIL | 25.2 | 15 | 15 | 15 | 15 | 132 |

## RSS checkpoints

| name | after compile | after alloc | after warmup | peak timed |
| --- | ---: | ---: | ---: | ---: |
| 5B | 15 MB | 15 MB | 15 MB | 15 MB |

## Single-layer bucket shares

| scale | total(ms) | ane(ms) | stage(ms) | read(ms) | cpu(ms) | ane% | stage% | read% | cpu% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5B | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| 13B | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| 20B | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |

## Per-kernel wall vs hardware

| scale | kernel | wall median(us) | hw median(ns) | overhead(us) | overhead% |
| --- | --- | ---: | ---: | ---: | ---: |
| 5B | sdpa_fwd | FAIL | FAIL | FAIL | FAIL |
| 5B | wo_fwd | 1999.8 | 0 | 1999.8 | 100.0% |
| 5B | ffn_fused | FAIL | FAIL | FAIL | FAIL |
| 13B | sdpa_fwd | FAIL | FAIL | FAIL | FAIL |
| 13B | wo_fwd | 8314.2 | 0 | 8314.2 | 100.0% |
| 13B | ffn_fused | FAIL | FAIL | FAIL | FAIL |
| 20B | sdpa_fwd | FAIL | FAIL | FAIL | FAIL |
| 20B | wo_fwd | 8316.9 | 0 | 8316.9 | 100.0% |
| 20B | ffn_fused | FAIL | FAIL | FAIL | FAIL |

## Recorded failures

- scale 5B: sdpaFwd compile: Compile("_ANECompiler : ANECCompile() FAILED")
- layer 5B: sdpaFwd compile: Compile("_ANECompiler : ANECCompile() FAILED")
- layer 13B: sdpaFwd compile: Compile("_ANECompiler : ANECCompile() FAILED")
- layer 20B: sdpaFwd compile: Compile("_ANECompiler : ANECCompile() FAILED")
- kernel 5B / sdpa_fwd: sdpa_fwd compile: Compile("_ANECompiler : ANECCompile() FAILED")
- kernel 5B / ffn_fused: ffn_fused compile: Compile("_ANECompiler : ANECCompile() FAILED")
- kernel 13B / sdpa_fwd: sdpa_fwd compile: Compile("_ANECompiler : ANECCompile() FAILED")
- kernel 13B / ffn_fused: ffn_fused compile: Compile("_ANECompiler : ANECCompile() FAILED")
- kernel 20B / sdpa_fwd: sdpa_fwd compile: Compile("_ANECompiler : ANECCompile() FAILED")
- kernel 20B / ffn_fused: ffn_fused compile: Compile("_ANECompiler : ANECCompile() FAILED")

## Conclusion

Current classification: **mixed**.

## Recommendation

Next action: **graph shaping**.

