# DeepSeek-V3 Phase 2: 0.7 → 3+ tok/s

**You are continuing optimization work from a previous session.** That session took V3 from 0.03 to 0.7 tok/s through 6 structural fixes. Your job is to get the remaining 2-4× to hit ≥1 tok/s (target 3-5).

**Hardware:** M4 Max, 128GB unified memory, NVMe SSD (~60 GB/s pread)
**Model:** DeepSeek-V3, 671B params, 61 layers, 128 heads, 256 experts/layer
**Current perf:** 0.7 tok/s = ~1,430 ms/token

## What Was Already Done (DO NOT REDO)

| Optimization | tok/s | Speedup | Key Insight |
|---|---|---|---|
| Baseline (serial convert, clones) | 0.03 | 1× | ~100 GB memcpy + 55 GB conversion per token |
| Zero-copy borrows | 0.03 | - | Eliminated Vec::clone() epidemic |
| Buffer reuse (single buf) | 0.03 | - | Zero allocs after warmup |
| Expert pager (pread) | 0.2 | 7× | Replaced 348 GB mmap thrashing with targeted pread |
| Rayon parallel conversion | 0.5 | 17× | Saturate memory bandwidth across cores |
| Cached Metal staging buffer | 0.7 | 23× | Eliminate per-layer Metal buffer creation |
| Rayon parallel pread | 0.7 | 23× | NVMe needs QD>1 for throughput |

## Phase 1: Load Everything Into Context

Read ALL of these before writing code. You have 1M tokens.

### Codebase (read in full — the previous session modified many of these)
```
dev/CURRENT.md                                 ← START HERE
crates/moe-infer/src/generate_v2.rs            ← V3 generation loop (MOST IMPORTANT)
crates/moe-infer/src/mla_attention.rs          ← MLA forward pass (may have been refactored to use borrows)
crates/moe-infer/src/weights.rs                ← weight loading
crates/moe-infer/src/blas.rs                   ← Accelerate BLAS FFI
crates/moe-infer/src/config.rs                 ← config
crates/moe-infer/src/fp8.rs                    ← FP8 dequant
crates/moe-infer/src/rmsnorm.rs                ← RMSNorm
crates/moe-infer/src/yarn_rope.rs              ← YaRN RoPE
crates/moe-infer/src/bin/infer.rs              ← CLI binary
crates/expert-pager/src/pool.rs                ← Least-Stale expert cache
crates/expert-pager/src/loader.rs              ← pread expert loader
crates/moe-router/src/lib.rs                   ← routing
crates/moe-kernels/src/lib.rs                  ← Metal shaders (READ BUT DON'T MODIFY)
```

### Research (read in full — the answers to your optimization questions are here)
```
/Users/dan/Dev/rustane-research/mla-1t/stage3-v3-runtime-2026-03-22/
  04-stage3-findings.md                ← CRITICAL: performance model, memory architecture
  wave1-rq1-lazy-conversion.md         ← double-buffer design, Neon throughput
  wave1-rq2-expert-pool.md             ← hit rate analysis
  wave1-rq3-wuk-optimization.md        ← W_UK benchmarks

/Users/dan/Dev/rustane-research/mla-1t/
  01-internal-architecture.md          ← MLA math, tensor shapes, compute budget
  model-comparison.md                  ← V2-Lite vs V3 dimensions
  precision-notes.md                   ← error budget
```

### Recent git history
```
git log --oneline -15                  ← see what the previous session committed
system/experiments-infer.tsv           ← previous session's measurements
```

## Phase 2: Profile First

Before writing any optimization code, **instrument and measure** where the 1,430 ms/token is actually going. Add timing to `run_layer_v2`:

```rust
let t_convert = Instant::now();
// ... convert_layer_f32 or buffer reuse ...
let convert_ms = t_convert.elapsed().as_secs_f64() * 1000.0;

let t_attn = Instant::now();
// ... MLA attention ...
let attn_ms = t_attn.elapsed().as_secs_f64() * 1000.0;

let t_ffn = Instant::now();
// ... MoE FFN or dense FFN ...
let ffn_ms = t_ffn.elapsed().as_secs_f64() * 1000.0;

eprintln!("  L{layer:02} conv={convert_ms:.1}ms attn={attn_ms:.1}ms ffn={ffn_ms:.1}ms");
```

Run ONE decode token and read the per-layer breakdown. This tells you exactly what to optimize.

### Expected breakdown (from research performance model):
| Component | Predicted ms | % |
|---|---|---|
| f16→f32 conversion | ~15ms/layer × 61 = 915ms | 64% |
| Q LoRA + KV compression | ~35ms/layer × 61 = 2135ms | -- |
| Attention scores | ~15ms/layer × 61 = 915ms | -- |
| O projection | ~20ms/layer × 61 = 1220ms | -- |
| MoE routing + shared FFN | ~33ms/layer × 58 = 1914ms | -- |
| Routed expert FFN (Metal) | ~100ms/layer × 58 = 5800ms | -- |

But these are PREDICTIONS. The actual numbers may be very different. Measure first.

## Phase 3: Attack the Bottleneck

Based on profiling, implement the highest-impact fix. Here are the likely candidates:

### If conversion dominates (>40% of time):
**Double-buffer with background thread.** The research design:
- Allocate 2 MlaLayerF32 buffers
- `std::thread::spawn` converts layer N+1 into buffer B while layer N runs on buffer A
- Swap buffers after each layer
- Conversion (~15ms) hides behind compute (~20ms) = FREE

### If attention dominates (>30% of time):
**Rayon parallel W_UK absorption.** 128 heads × sgemv_f32_trans is embarrassingly parallel:
```rust
use rayon::prelude::*;
(0..n_heads).into_par_iter().for_each(|h| {
    sgemv_f32_trans(&w_uk[h*nope*kv_rank..], &q_nope[h*nope..], &mut q_absorbed[h*kv_rank..], kv_rank, nope);
});
```

### If expert FFN dominates (>30% of time):
**Batch Metal dispatches.** Currently 8 separate Metal dispatches per MoE layer (one per routed expert). Batch into 1 dispatch with 8 threadgroups. The fused_and_down_single_cmdbuf already partially does this — verify it's actually batching.

### If expert loading dominates (>10% of time):
**Increase pool capacity.** Currently may be undersized. Research says 2000 experts = 90% hit rate at ~45 GB. Check `pool.stats.hit_rate()` and adjust.

### If memory bandwidth dominates:
**Consider f16 compute paths.** Accelerate's `cblas_hgemm` (if available on M4 Max) or Metal compute shaders for the large projections (o_proj 7168×16384, q_b_proj 24576×1536). This eliminates conversion entirely.

## Phase 4: Iterate

After each optimization:
1. Profile again — did the bottleneck shift?
2. Run V2-Lite regression
3. Log to experiments-infer.tsv
4. Commit with `perf:` prefix and tok/s number
5. Target the new bottleneck

## Commands

```bash
cargo build -p moe-infer --release

# V3 inference (THE benchmark)
cargo run -p moe-infer --release --bin infer -- \
  --config configs/deepseek-v3.toml \
  --weights weights/rustane-v3 \
  --tokenizer weights/deepseek-v3/tokenizer.json \
  --prompt "The capital of France is" \
  --max-tokens 10

# V2-Lite regression (MUST pass)
cargo test -p moe-infer --test test_model_validation --release -- --ignored --nocapture

# All lib tests
cargo test -p moe-infer --lib --release
```

## Known Bugs from Audit (fix if you hit them)

**C1.** Q LoRA partial load falls through to empty q_proj — UB in release. Add validation at load time.

**C2.** `route_sigmoid_v3` scaling_factor parameter is dead code (always 1.0). Actual scaling happens in accumulation loop. Coincidentally correct but fragile.

**I3.** `final_norm.to_vec()` called every token — unnecessary copy. Remove `.to_vec()`.

## Guardrails

1. **Profile before optimizing.** No guessing.
2. **V2-Lite regression must ALWAYS pass.**
3. **One change at a time.** Measure impact before stacking.
4. **Commit after each win.** `perf:` prefix, tok/s in message.
5. **If stuck >30 min, try different approach.**
6. **Read research before reimplementing.**

## Success Criteria

| Level | tok/s | ms/token |
|---|---|---|
| Previous session | 0.7 | 1,430 |
| Minimum viable | 1.0 | 1,000 |
| Good | 2.0 | 500 |
| Target | 3-5 | 200-333 |
| Research prediction | 4.3 | 230 |
