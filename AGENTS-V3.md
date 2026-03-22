# AGENTS-V3.md — DeepSeek-V3 (671B) Inference Optimization

Instructions for AI agents optimizing DeepSeek-V3 inference on Apple M4 Max 128GB.
Read this ENTIRE file before writing any code.

## Branch Policy

- **v3-optimize** is the stable V3 branch. Only verified improvements go here.
- Work on your agent branch: `v3-opt/auto-{agent_id}`
- NEVER push directly to v3-optimize.
- NEVER modify locked files (see Locked Files section).

## Build & Test

```bash
cargo build -p moe-infer --release                                                        # build
cargo test -p moe-infer --release                                                          # full test suite
cargo test -p moe-infer --test test_v3_validation --release -- --ignored --nocapture        # V3 correctness
cargo test -p moe-infer --test test_model_validation --release -- --ignored --nocapture     # V2-Lite regression
cargo test -p moe-infer --test bench_v3_tok_per_sec --release -- --ignored --nocapture      # V3 tok/s benchmark
```

## Metric Matrix — The Anti-Cheat Gate

You MUST pass ALL of Tier 1 and not regress Tier 2 to commit an improvement.

### Tier 1 — Hard Gates (any failure = instant revert)

| # | Test | Threshold | Command |
|---|------|-----------|---------|
| 1 | Build clean | zero errors | `cargo build -p moe-infer --release` |
| 2 | Full test suite | zero failures | `cargo test -p moe-infer --release` |
| 3 | V3 validation | cosine > 0.99 | `cargo test -p moe-infer --test test_v3_validation --release -- --ignored` |
| 4 | V2-Lite regression | 4/4 levels pass | `cargo test -p moe-infer --test test_model_validation --release -- --ignored` |
| 5 | Your custom test | pass | `cargo test -p moe-infer --test auto_<name> --release` |
| 6 | Locked files untouched | git diff clean | Verified by optimize-v3.sh post-check |

### Tier 2 — Performance (median of 3 runs, must improve or hold)

| # | Metric | Baseline | Rule | Command |
|---|--------|----------|------|---------|
| 7 | Warm decode tok/s | 0.7 | no regress >5% | `bench_v3_tok_per_sec` |

### Tier 3 — Lie Detector (logged, not gated)

| # | Metric | Why |
|---|--------|-----|
| 8 | Per-phase timing in benchmark stderr | Your improvement MUST show up in a specific component |
| 9 | Cold vs warm tok/s gap | Catch expert cache regressions |

**Improvement must be visible in timing.** "10% faster decode" but identical per-phase numbers = measurement noise. Revert.

## Locked Files — DO NOT MODIFY

These files are verified unmodified after every iteration. Touching them = full revert.

```
crates/moe-infer/tests/bench_v3_tok_per_sec.rs
crates/moe-infer/tests/test_v3_validation.rs
crates/moe-infer/tests/test_model_validation.rs
configs/deepseek-v3.toml
AGENTS-V3.md
```

## Architecture Overview

### V3 Decode Pipeline (0.7 tok/s, 61 layers)

```
Per token:
  Backbone (f16 mmap, 34.2 GB):
    f16→f32 conversion (rayon parallel, ~7ms/layer)
    Q LoRA: x → W_qa [1536,7168] → RMSNorm → W_qb [24576,1536] → q
    KV compress: x → W_kva → RMSNorm → split(kv_nope, kv_pe)
    Absorbed attention: q@W_UK scores + softmax + v@W_UV reconstruct
    O projection: [16384→7168] sgemv
  Expert FFN (INT4, 343 GB on SSD, pread):
    Sigmoid routing with e_score_correction_bias → top 8 of 256
    pread 8 experts from SSD (~3ms, parallel QD=4-8)
    Metal INT4 dequant GEMV (fused gate+up+SiLU + down)
    + 1 shared expert (RAM-resident, 3× sgemv)
  Residual add → next layer
```

### Current Bottleneck (22ms/layer × 61 = 1,340 ms/token)

```
f16→f32 conversion:     ~7ms  (32%)  — Memory BW limited
MLA attention sgemv:    ~5ms  (23%)  — Memory BW (AMX), scalar f64 loops
Shared expert FFN:      ~4ms  (18%)  — Memory BW (3× sgemv)
Expert pread from SSD:  ~3ms  (14%)  — NVMe + page cache
Metal expert dispatch:  ~3ms  (14%)  — GPU compute + dispatch overhead
```

## M4 Max Hardware Facts (Verified)

- **GPU peak**: ~15 TFLOPS fp16 (80% of real 18.4T)
- **ANE peak**: 7.3 TFLOPS single kernel, 17.8 TFLOPS fused
- **Memory bandwidth**: ~400 GB/s unified (546 GB/s peak measured)
- **Single P-core bandwidth**: ~80 GB/s
- **SSD sequential read**: ~17.5 GB/s (measured via pread)
- **Metal cmd_buf commit overhead**: ~85us per commit
- **Metal dispatch overhead**: ~4us per dispatch within a commit
- **L2 cache**: buffers <16MB are effectively free to copy
- **vDSP/vecLib FFI**: ~0.5us per call overhead

## What's Been Tried (V3 Phase 1: 0.03 → 0.7 tok/s)

| # | Change | tok/s | Speedup | Key Insight |
|---|--------|-------|---------|-------------|
| 1 | Baseline (serial convert, clones) | 0.03 | 1x | ~100 GB memcpy + 55 GB conversion per token |
| 2 | Zero-copy borrows | 0.03 | - | Eliminated Vec::clone() epidemic |
| 3 | Buffer reuse (single buf) | 0.03 | - | Zero allocs after warmup |
| 4 | Expert pager (pread) | 0.2 | 7x | Replaced 348 GB mmap thrashing |
| 5 | Rayon parallel conversion | 0.5 | 17x | Saturate memory bandwidth across cores |
| 6 | Cached Metal staging buffer | 0.7 | 23x | Eliminate per-layer Metal buffer creation |
| 7 | Parallel pread (QD>1) | 0.7 | 23x | NVMe needs queue depth for throughput |

## Proven Dead Ends — DO NOT RETRY

| Experiment | Why it failed |
|-----------|---------------|
| Chunked f16 sgemv | 4x slower (86ms vs 22ms/layer). Per-chunk cblas_sgemv dispatch overhead dominates. |
| Channel-based double-buffer pipeline | Alloc thrashing. Pipeline overhead > overlap benefit when conversion (7ms) < compute (13ms). |
| mmap for 348 GB expert files | Catastrophic page cache thrashing. pread is 100x better. |
| mlock backbone | No benefit after switching experts to pread — no eviction pressure. |
| Serial expert pread | 25% slower than parallel. NVMe needs QD>1 for throughput. |

## What DOES Work

- **Expert pager pread**: Targeted reads instead of mmap. 21x speedup.
- **Rayon parallel f16→f32**: Saturates memory bandwidth. 8.5x speedup on conversion.
- **Cached Metal staging buffer**: Create once, reuse for all MoE dispatches.
- **Buffer reuse (convert_layer_into)**: Zero allocations after warmup.
- **Zero-copy borrows**: `&'a [f32]` instead of `Vec<f32>` clones.
- **Parallel pread**: QD=4-8 for NVMe throughput.
- **Backbone warmup + madvise**: Pre-fault all backbone pages at load time.

## Known Bugs (DO NOT FIX — avoid in optimizations)

These are documented for awareness. Bug fixes require manual sessions.

| ID | Bug | Impact |
|----|-----|--------|
| B1 | `x_cache[4096]` in Metal but V3 hidden=7168 | Expert gate/up OOB — results partially wrong |
| B2 | `staging_ptr` mutable ref from immutable self | Active UB in hot path (aliasing violation) |
| B3 | `wrap_mmap` pads beyond Vec allocation | Metal reads unowned memory |
| B4 | `pread` short reads not checked | Silent expert weight corruption possible |
| B5 | ExpertPool imported but never wired | Every expert pread'd every token (20 GB I/O vs 0.5 GB with pool) |
| B6 | Scalar f64 attention loops | ~10ms/layer wasted (sgemm would be 0.01ms) |
| B7 | routed_scaling_factor dead code | scaling_factor param always 1.0, actual scaling in accumulation |

## Memory Budget (128 GB total)

```
macOS + Metal:          ~10 GB  (fixed)
Backbone mmap (f16):    ~34 GB  (page cache, madvise'd)
Expert staging:          ~2 GB  (reused per layer)
KV cache (61 layers):    ~1 GB
Metal buffers:           ~1 GB
────────────────────────────────
Used:                   ~48 GB
Available:              ~80 GB
```

**WARNINGS:**
- f32 backbone pre-load = 68 GB. Does NOT fit alongside current usage.
- f32 compute path max 8.4 tok/s (physical limit: 57 GB/token at 546 GB/s).
- f16 compute path max ~14 tok/s (28.5 GB/token at 546 GB/s).
- Expert pool of 3300 experts = ~77 GB. Would consume all remaining memory.

## Scope Constraint

**You are making SMALL, SAFE, TESTABLE optimizations within the existing architecture.**

If a change touches more than ~100 lines or requires a new subsystem, log it as PLANNED and exit.

Good targets for auto-optimization:
- Buffer sizing (scratch too small/large?)
- Loop ordering (better cache behavior?)
- BLAS call batching (3 sgemv → 1 sgemm?)
- Threading parameters (rayon chunk size, pread parallelism)
- Allocation elimination (unnecessary Vec::collect, .to_vec())
- Memory layout (struct field ordering, alignment)
- Constant tuning (e.g., ROWS_PER_TG=8 was a 23% win on another model)

NOT for auto-optimization:
- Architecture changes (S1-S8 milestones)
- Bug fixes (B1-B7)
- New subsystems (ExpertPool wiring, Metal attention kernel)

## Research Context

Read files in `research-context/` for deep background. Key files:

- `research-context/stage3/04-stage3-findings.md` — double-buffer design, expert pool sizing
- `research-context/stage2/01-audit-full.md` — bugs, showstoppers, 14x gap analysis
- `research-context/stage2/03-architecture-10toks.md` — 10 tok/s blueprint, physical limits
- `research-context/stage1/FINAL.md` — 10 architecture corrections from 9 research agents
- `research-context/01-internal-architecture.md` — MLA math, compute budget, weight tensors

## Code Conventions

- One variable per experiment. Never combine changes.
- Read existing code before modifying it.
- Write a custom test (`auto_*.rs`) that proves YOUR change is semantically equivalent.
- Log EVERY experiment to `system/experiments-v3.tsv` (even failures).
- Commit message: `"perf: <what> — <before> → <after> tok/s (<X>%)"`
- Write status to `/tmp/rustane-v3-status-{agent_id}` at each phase.
- Status phases: READING, THINKING, CODING, TESTING, BENCHMARKING, LOGGING, DONE.

## Key Source Files

```
crates/moe-infer/src/generate_v2.rs       ← THE hot path (V3 decode loop)
crates/moe-infer/src/mla_attention.rs     ← MLA forward pass
crates/moe-infer/src/weights.rs           ← weight loading + buffer reuse
crates/moe-infer/src/blas.rs              ← Accelerate BLAS FFI
crates/moe-infer/src/config.rs            ← config parsing
crates/moe-infer/src/fp8.rs               ← FP8 dequant
crates/moe-infer/src/rmsnorm.rs           ← RMSNorm
crates/moe-infer/src/yarn_rope.rs         ← YaRN RoPE
crates/expert-pager/src/pool.rs           ← expert pool (built, NOT wired)
crates/expert-pager/src/loader.rs         ← pread expert loader
crates/moe-router/src/lib.rs              ← sigmoid routing + route_sigmoid_v3
```
