# AGENTS-K2.md — Kimi-K2 (1T) Inference Optimization

Instructions for AI agents optimizing Kimi-K2 inference on Apple M4 Max 128GB.
Read this ENTIRE file before writing any code.

## Branch Policy

- **v3-optimize** is the stable branch (shared with V3). Only verified improvements go here.
- Work on your agent branch: `k2-opt/auto-{agent_id}`
- NEVER push directly to v3-optimize.

## Build & Test

```bash
cargo build -p moe-infer --release
cargo test -p moe-infer --release
cargo test -p moe-infer --test bench_k2_tok_per_sec --release -- --ignored --nocapture
```

## Metric Matrix

### Tier 1 — Hard Gates (any failure = instant revert)
| # | Test | Command |
|---|------|---------|
| 1 | Build clean | `cargo build -p moe-infer --release` |
| 2 | Full test suite | `cargo test -p moe-infer --release` |
| 3 | Your custom test | `cargo test -p moe-infer --test auto_<name> --release` |
| 4 | Locked files untouched | Verified by optimizer post-check |

### Tier 2 — Performance (median of 3)
| # | Metric | Baseline | Command |
|---|--------|----------|---------|
| 5 | K2 warm decode tok/s | TBD (first warm run pending) | `bench_k2_tok_per_sec` |

## Locked Files
```
crates/moe-infer/tests/bench_k2_tok_per_sec.rs
configs/kimi-k2.toml
AGENTS-K2.md
```

## Architecture Overview

### K2 vs V3 — Same Code, Different Constants
K2 uses the SAME inference code (generate_v2.rs, mla_attention.rs) as V3. All V3 optimizations
(pipeline decode, pread overlap, deferred convert, cached dense, sgemm attention) apply automatically.

Key differences:
- 64 attention heads (V3=128) — half the W_UK/W_UV/Q/O projection work
- 384 experts per MoE layer (V3=256) — 50% more experts, 9 GB/layer files (V3=5.5 GB)
- n_group=1 (V3=8) — no grouped routing, plain top-8 from 384
- first_k_dense=1 (V3=3) — only layer 0 is dense
- vocab_size=163840 (V3=129280) — larger embedding + lm_head (4.5 GB vs 3.7 GB)
- routed_scaling_factor=2.827 (V3=2.5)
- bf16 source weights (V3=FP8) — already converted to same rustane format

### Current Decode Pipeline
```
Per token (61 layers):
  Dense layer 0: cached f32, no conversion needed
  MoE layers 1-60:
    Background: convert layer N+1 f16→f32 (deferred to FFN phase)
    MLA: Q LoRA → W_UK absorption (64 heads) → sgemm attention → W_UV → O proj
    FFN: shared expert sgemv (CPU) || pread 8 of 384 experts (SSD)
         then Metal INT4 dispatch
```

### Expected Bottleneck (K2 vs V3)
K2 has LESS MLA compute (64 heads vs 128) but MORE expert data (384 × 9 GB files).
- MLA should be ~50% faster than V3 (half the per-head sgemv calls)
- Expert pread same speed (still 8 experts per token, just from larger files)
- Metal dispatch same (still 8 expert GEMV per layer)
- Backbone smaller (23.4 GB vs 34 GB) — faster warmup

## M4 Max Hardware Facts (same as V3)
- Memory bandwidth: ~400 GB/s unified (546 GB/s peak)
- SSD sequential read: ~17.5 GB/s (pread)
- Metal cmd_buf commit: ~85µs per commit
- AMX sgemv: ~150 GB/s per P-core (measured, not estimated 80 GB/s)
- Apple BLAS: internally multi-threads for large matrices (>100 MB)

## Memory Budget
```
macOS + Metal:          ~10 GB
K2 backbone mmap (f16): ~23 GB (smaller than V3's 34 GB)
Expert staging:          ~2 GB
KV cache (61 layers):    ~1 GB
Metal buffers:           ~1 GB
────────────────────────
Used:                   ~37 GB
Available:              ~91 GB (more headroom than V3)
```

## V3 Learnings That Apply to K2
From 40+ V3 experiments:
- Overlap wins: thread::scope overlapping work on DIFFERENT hardware resources
- f16 CPU compute: DEAD END (needs Metal, not CPU)
- CPU parallelization: Apple BLAS already optimal
- Allocation elimination: invisible at this scale
- Metal per-dispatch: too expensive alone, needs batching
- Shared FFN overlap is load-bearing — cannot move to Metal without replacement CPU work

## Scope
You are making SMALL, SAFE, TESTABLE optimizations. Under 100 lines.
NEVER use EXHAUSTED as a verdict. There is ALWAYS something to try.
You MUST implement and benchmark something every iteration.

## Code Conventions
- One variable per experiment
- Write a custom test (auto_*.rs)
- Log to `system/experiments-k2.tsv`
- Commit: `"perf: <what> — <before> → <after> tok/s (<X>%)"`
- Status: `/tmp/rustane-k2-status-{agent_id}`

## Key Source Files
```
crates/moe-infer/src/generate_v2.rs       ← decode loop (shared with V3)
crates/moe-infer/src/mla_attention.rs     ← MLA forward (parameterized by heads)
crates/moe-infer/src/weights.rs           ← weight loading
crates/moe-infer/src/blas.rs              ← BLAS FFI
crates/expert-pager/src/pool.rs           ← expert pool (built, NOT wired)
crates/expert-pager/src/loader.rs         ← pread loader
crates/moe-router/src/lib.rs             ← sigmoid routing (n_group=1 for K2)
configs/kimi-k2.toml                      ← K2 config
```
