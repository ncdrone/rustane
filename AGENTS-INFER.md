# AGENTS-INFER.md — Rustane Inference Engine

Instructions for AI agents optimizing MoE inference on Apple M4 Max 128GB.
Read this ENTIRE file before writing any code.

## Branch Policy

- **rustane-infer** is the stable inference branch. Only verified improvements go here.
- Work on your agent branch: `infer-opt/auto-{agent_id}`
- NEVER push directly to rustane-infer.
- NEVER modify locked files (see Locked Files section).

## Build & Test

```bash
cargo build -p moe-infer --release                                    # build
cargo test -p moe-infer --release                                     # full test suite
cargo test -p moe-infer --test test_fused_metal --release             # Metal kernel correctness
cargo test -p moe-infer --test test_generation --release -- --ignored  # HF match (requires weights)
cargo test -p moe-infer --test bench_tok_per_sec --release -- --ignored --nocapture  # tok/s benchmark
cargo test -p moe-infer --test bench_decode_layer_breakdown --release -- --ignored --nocapture  # per-layer breakdown
```

## Metric Matrix — The Anti-Cheat Gate

You MUST pass ALL of Tier 1 and not regress Tier 2 to commit an improvement.

### Tier 1 — Hard Gates (any failure = instant revert)

| # | Test | Threshold | Command |
|---|------|-----------|---------|
| 1 | HF greedy match | ≥ 18/20 | `cargo test -p moe-infer --test test_generation --release -- test_generation_matches_hf --ignored` |
| 2 | Full test suite | zero failures | `cargo test -p moe-infer --release` |
| 3 | Metal kernel correctness | max_diff < 1e-3 | `cargo test -p moe-infer --test test_fused_metal --release` |
| 4 | Build clean | zero errors | `cargo build -p moe-infer --release` |
| 5 | Your custom test | pass | `cargo test -p moe-infer --test auto_<name> --release` |
| 6 | Locked files untouched | git diff clean | Verified by optimize-infer.sh post-check |

### Tier 2 — Performance (must improve or hold, BOTH regimes)

| # | Metric | Baseline | Rule | Command |
|---|--------|----------|------|---------|
| 7 | Decode tok/s (median of 3) | 19.6 | no regress >3% | `bench_tok_per_sec` |
| 8 | Prefill latency (13 tokens) | 310ms | no regress >10% | `bench_tok_per_sec` |
| 9 | Decode p95 latency | ~55ms | no regress >10% | `bench_decode_layer_breakdown` |

### Tier 3 — Lie Detector (logged, not gated)

| # | Metric | Why |
|---|--------|-----|
| 10 | Per-layer breakdown (metal_ms, attn_ms) | Your improvement MUST show up in components |
| 11 | Time-to-first-token | User-facing latency |
| 12 | Peak RSS during generation | Catch memory leaks |

**Improvement must be visible in the breakdown.** "5% faster decode" but identical per-layer numbers = measurement noise. Revert.

## Locked Files — DO NOT MODIFY

These files are verified unmodified after every iteration. Touching them = full revert.

```
crates/moe-infer/tests/test_generation.rs
crates/moe-infer/tests/bench_tok_per_sec.rs
crates/moe-infer/tests/bench_decode_layer_breakdown.rs
crates/moe-infer/tests/test_fused_metal.rs
crates/moe-infer/tests/bench_metal_dispatch_latency.rs
crates/moe-infer/tests/bench_blas_sizes.rs
configs/qwen3-moe-30b.toml
weights/references/greedy_generation.json
AGENTS-INFER.md
```

## Architecture Overview

### Current Decode Pipeline (Qwen3-MoE-30B, 19.6 tok/s)

```
Per token (48 layers):
  CPU: RMSNorm → BLAS sgemv (Q/K/V/O projections) → softmax → attention scores
  Metal: fused gate+up+SiLU + down GEMVs (single cmd_buf per layer, scratch buffers)
  CPU: weighted expert combine → residual add
```

### Current Bottleneck (51ms/token)

```
Metal MoE dispatch:  ~31ms (60%)  — fused kernel + down, 48 commits/token
CPU attention:       ~25ms (38%)  — 4×sgemv + softmax + scores per layer
Everything else:      ~1ms (2%)   — RMSNorm, router, residuals
```

### Target Architecture (DeepSeek-V3, 671B, MLA + SSD streaming)

```
Per token (61 layers):
  Backbone (RAM-resident, 17GB f16):
    MLA: Q LoRA (W_qa → norm → W_qb), KV compress (W_kva → norm)
    Absorbed attention: q@W_UK scores, softmax, v@W_UV reconstruct
    O projection: [16384→7168]
  Expert FFN (SSD-streamed, 343GB 4-bit):
    pread 8 experts from SSD → Metal 4-bit dequant GEMV
    + 1 shared expert (RAM-resident)
```

## M4 Max Hardware Facts (Verified)

- **GPU peak**: ~15 TFLOPS fp16 (80% of real 18.4T)
- **ANE peak**: 7.3 TFLOPS single kernel, 17.8 TFLOPS fused
- **Memory bandwidth**: ~400 GB/s unified
- **Single P-core bandwidth**: ~80 GB/s
- **SSD sequential read**: ~17.5 GB/s (measured via pread)
- **Metal cmd_buf commit overhead**: ~85µs per commit
- **Metal dispatch overhead**: ~4µs per dispatch within a commit
- **ANE dispatch overhead**: ~0.095ms per XPC round-trip
- **L2 cache**: buffers <16MB are effectively free to copy (~0.01ms for 1.5MB)
- **IOSurface**: width must be multiple of 16 (silent data corruption otherwise)
- **ANE compiler**: fails on rsqrt/sqrt after reduce ops — use pow(-0.5)
- **vDSP/vecLib FFI**: ~0.5µs per call overhead

## What's Been Tried (Stage 3 → Stage 4)

### Stage 3: 0.4 → 14.0 tok/s (35x)
| Change | Impact |
|--------|--------|
| BLAS sgemv for attention (AMX) | 0.4 → 1.0 tok/s |
| Metal expert FFN (zero-copy mmap) | 1.0 → 3.5 tok/s |
| FMA kernel (pre-factored scale*x) | 3.5 → 4.5 tok/s |
| Zero-copy Metal + f32 pre-conversion | 4.5 → 11.7 tok/s |
| ANE batched prefill | prefill 0.80 → 0.71s |

### Stage 4: 14.0 → 19.6 tok/s (40%)
| Change | Impact |
|--------|--------|
| ROWS_PER_TG=8 shader | 14.0 → 17.2 tok/s |
| Fused gate+up+SiLU kernel | 17.2 → 17.1 tok/s (prerequisite for single cmd_buf) |
| Scratch buffers + single cmd_buf/layer | 17.1 → 19.1 tok/s |
| sgemm batched O_proj (prefill) | prefill 400 → 310ms |
| ANE run_cached_direct | 19.1 → 19.6 tok/s |

## Proven Dead Ends — DO NOT RETRY

| Experiment | Why it failed | Source |
|-----------|---------------|--------|
| CPU SiLU optimization | Only 0.5% of decode time (367µs total) — not a bottleneck | Stage 4 Task 1 |
| IOSurface staging parallelism | Serializes at memory level regardless of thread count | Training auto-max |
| Metal GPU for Adam | 99.8% driver overhead for small ops | Training auto-max |
| Double-buffer IOSurface | Apple UMA is fully cached, zero benefit | Training auto-max |
| Multi-pass vDSP for fused ops | LLVM auto-vectorization beats explicit vDSP | Training auto-max |

## What DOES Work

- **ROWS_PER_TG=8**: 8 SIMD groups share x_cache. Reduces threadgroups 8x.
- **Fused kernels**: gate+up+SiLU in one dispatch eliminates CPU roundtrips.
- **Single cmd_buf per layer**: halves Metal commit overhead (96→48 commits/token).
- **Pre-allocated scratch buffers**: zero allocations in decode hot path.
- **sgemm over sgemv for batching**: 3.5x speedup for batched O_proj in prefill.
- **BLAS for f32 dense matmuls**: AMX hardware at 3 TFLOPS beats Metal for non-quantized weights.
- **run_cached_direct for ANE**: bypasses XPC daemon, saves ~0.095ms/dispatch.

## Remaining Optimization Targets

Ranked by expected impact (from Stage 5 research):

| # | Target | Expected Saving | Complexity |
|---|--------|----------------|------------|
| 1 | Batched QKV sgemv (3→1 call) | -2.4ms/token | Low |
| 2 | Eliminate 240 allocs/token in attention | -1.5ms/token | Low |
| 3 | Metal attention scores kernel | -2 to -4ms/token | Medium |
| 4 | Metal f32 sgemv for attention projections | -5 to -10ms/token | Medium |
| 5 | Single cmd_buf ALL 48 layers | -4ms/token | High (needs R4) |

## Research Corpus

Detailed research in `rustane-research/mla-1t/`:
- `start-here/context.md` — what exists, what we're building
- `start-here/architecture-overview.md` — MLA math, FLOPs, weight format
- `stage0-foundations-2026-03-21/` — 3 key findings
- `stage1-external-2026-03-21/FINAL.md` — external research (absorbed attention, Metal shaders, FP8 conversion)

Reference implementations:
- `flash-moe-ane` (`/Users/dan/Dev/flash-moe-ane/`) — Metal shaders, SSD streaming, 4.36 tok/s on Qwen3.5-397B
- ane-infer — single cmd_buf for all layers pattern (32 tok/s Q8 on M5)
- uzu — online softmax attention kernel, quantized GEMV fast paths

## Code Conventions

- One variable per experiment. Never combine changes.
- Read existing code before modifying it.
- Write a custom test (auto_*.rs) that proves YOUR change is semantically equivalent.
- Log EVERY experiment to system/experiments-infer.tsv (even failures).
- Commit message: `"perf: <what> — <before> → <after> tok/s (<X>%)"`
- Write status to /tmp/rustane-infer-status-{agent_id} at each phase.
- Status phases: READING, THINKING, CODING, TESTING, BENCHMARKING, LOGGING, DONE.
