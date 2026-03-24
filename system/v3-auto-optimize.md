# DeepSeek-V3 (671B) Inference — Deep Architecture Session

**You are a senior systems engineer with the FULL codebase and ALL research in context.** Your goal is to get DeepSeek-V3 generating coherent text at ≥1 tok/s on M4 Max 128GB. You have 12+ hours and a 1M token context window. Use them.

**Hardware:** M4 Max, 128GB unified memory, NVMe SSD (~60 GB/s pread)
**Model:** DeepSeek-V3, 671B params, 61 layers, 128 heads, 256 experts/layer, MLA attention
**Codebase:** Rust inference engine (`crates/moe-infer/`), branch `rustane-infer`
**Current perf:** ~0.03 tok/s (target: ≥1 tok/s, research predicts 4.3 tok/s achievable)

## Phase 1: Load Everything Into Context

Read ALL of these before writing any code. You have the context window for it.

### Codebase (read in full)
```
crates/moe-infer/src/generate_v2.rs    ← V3 generation loop (THE critical file)
crates/moe-infer/src/mla_attention.rs  ← MLA forward pass
crates/moe-infer/src/weights.rs        ← weight loading
crates/moe-infer/src/config.rs         ← config parsing
crates/moe-infer/src/fp8.rs            ← FP8 dequant
crates/moe-infer/src/blas.rs           ← Accelerate BLAS FFI
crates/moe-infer/src/rmsnorm.rs        ← RMSNorm
crates/moe-infer/src/yarn_rope.rs      ← YaRN RoPE
crates/moe-infer/src/sampler.rs        ← token sampling
crates/moe-infer/src/lib.rs            ← module exports
crates/moe-infer/src/bin/infer.rs      ← CLI binary
crates/expert-pager/src/pool.rs        ← Least-Stale expert cache
crates/expert-pager/src/loader.rs      ← pread expert loader
crates/moe-router/src/lib.rs           ← sigmoid/softmax routing + route_sigmoid_v3
```

### Research (read in full — this is your advantage)
```
/Users/dan/Dev/rustane-research/mla-1t/stage3-v3-runtime-2026-03-22/
  04-stage3-findings.md                ← CRITICAL: double-buffer design, perf model, memory arch
  wave1-rq1-lazy-conversion.md         ← f16→f32 throughput data, Neon FCVTL
  wave1-rq2-expert-pool.md             ← hit rate vs pool size, expert locality
  wave1-rq3-wuk-optimization.md        ← W_UK benchmarks, NOT a bottleneck

/Users/dan/Dev/rustane-research/mla-1t/stage2-deepseekv3-execution-2026-03-21-1246/
  POST-MORTEM.md                       ← what was built, 8 tasks, FP8 proof
  REFLECTIONS.md                       ← learnings, process, what we got wrong
  stage2-deepseekv3-execution-external-research-2026-03-21-1246/
    FINAL.md                           ← executive summary from 9 research agents
    wave2-fp8-converter-dense-ffn.md   ← converter details
    wave2-expert-pager-design.md       ← Least-Stale eviction design
    wave3-backbone-loading.md          ← backbone.bin layout, mmap strategy

/Users/dan/Dev/rustane-research/mla-1t/
  01-internal-architecture.md          ← MLA math, tensor shapes, compute budget
  model-comparison.md                  ← V2-Lite vs V3 vs K2 differences
  precision-notes.md                   ← error budget, debugging playbook
  testing-framework.md                 ← 4-level validation hierarchy
```

### Project state
```
dev/CURRENT.md                         ← live state, read FIRST
research/RESULTS-moe-1T.md            ← full implementation history with benchmarks
configs/deepseek-v3.toml               ← V3 config
configs/deepseek-v2-lite.toml          ← V2-Lite config (regression reference)
```

## Phase 2: Think Before You Code

After loading everything, spend time reasoning about:

1. **Where is the 1000× gap?** Research predicts 230ms/token (4.3 tok/s). We're at ~30,000ms/token (0.03 tok/s). That's 130× slower. Where is the time going? The research says it's the serial f16→f32 conversion — but is that the ONLY problem? Or are there architectural issues in how `run_layer_v2` is structured?

2. **What does the compute graph actually look like?** Trace one token through all 61 layers. Where are the memory allocations? Where are the copies? Where are the BLAS calls? Is there unnecessary work?

3. **What would a clean V3 inference loop look like?** The current `generate_v2.rs` was built for V2-Lite (2048 hidden, 16 heads, 27 layers) and extended for V3. Would a purpose-built V3 path be cleaner and faster?

4. **What does the memory access pattern look like?** The backbone.bin is 34.2 GB mmap'd. Each layer's weights are scattered across it. Is there locality? Could we restructure the backbone for sequential access per layer?

5. **Can we avoid f32 entirely for some operations?** The research mentions Accelerate's cblas_sgemv requires f32. But what about f16 BLAS via Metal? Or mixed precision?

## Phase 3: Implement Architecturally

Don't make 50 small tweaks. Make 3-5 deep structural changes. Each should move the needle by 5-50×.

**Known high-impact changes (from research):**

### A. Double-Buffer Weight Streaming (~100× improvement)
The single biggest win. Design from `04-stage3-findings.md`:
- Two f32 scratch buffers (~1.8 GB each)
- Background thread converts layer N+1 while layer N computes
- f16→f32 at ~100 GB/s on Neon = ~15ms/layer, fully hidden behind compute

### B. Eliminate Unnecessary Allocations
`run_layer_v2` creates new Vecs on every call: `q` (24K elements), `q_nope` (16K), `q_pe` (8K), `kv_out` (576), `scores` (128×seq), etc. For 61 layers × 6 tokens = 366 calls, that's thousands of allocations. Pre-allocate scratch buffers.

### C. Profile-Driven Optimization
Add timing instrumentation FIRST. Know where time goes before optimizing:
```rust
let t = std::time::Instant::now();
// ... operation ...
let elapsed = t.elapsed();
eprintln!("  layer {layer} attn: {:.1}ms, ffn: {:.1}ms, convert: {:.1}ms", ...);
```

### D. Consider: Skip Pre-Conversion Entirely for Small Tensors
Norms ([7168] f32), router ([256×7168] f16), bias ([256] f32) are tiny. Keep them in the backbone mmap and convert inline. Only the large projections need the double-buffer.

### E. Consider: Restructure Backbone for Sequential Access
Currently backbone.bin has ALL layers' weights interleaved by tensor type. If we restructured to group by layer, the mmap page cache would be much more effective.

## Commands

```bash
# Build
cargo build -p moe-infer --release

# Run V3 inference (THE test)
cargo run -p moe-infer --release --bin infer -- \
  --config configs/deepseek-v3.toml \
  --weights weights/rustane-v3 \
  --tokenizer weights/deepseek-v3/tokenizer.json \
  --prompt "The capital of France is" \
  --max-tokens 10

# V2-Lite regression (MUST pass after every change)
cargo test -p moe-infer --test test_model_validation --release -- --ignored --nocapture

# All lib tests
cargo test -p moe-infer --lib --release

# Qwen3 regression
cargo test -p moe-infer --test bench_tok_per_sec --release -- --ignored --nocapture
```

## Guardrails

1. **V2-Lite regression must ALWAYS pass.** Run after every structural change.
2. **One architectural change at a time.** Measure the impact before stacking another.
3. **Commit after each successful change.** `perf:` prefix, include tok/s in message.
4. **Log to `system/experiments-infer.tsv`** — date, name, variable, tok_s, result, verdict.
5. **If V3 takes >5 min per token, something is wrong.** Don't wait — profile and fix.
6. **Read the research before reimplementing.** The answers are likely already there.

## Key Architecture Details

- **MLA attention**: two dot products (nope + rope) summed. Scale = 1/sqrt(192) × mscale².
- **Q LoRA (V3)**: x → W_qa [1536, 7168] → RMSNorm → W_qb [24576, 1536] → q.
- **Routing (V3)**: sigmoid + grouped top-k with frozen e_score_correction_bias.
- **routed_scaling_factor = 2.5**: applied per-expert weight, NOT combined output.
- **3 dense FFN layers** (0, 1, 2): intermediate=18432. Layers 3-60 are MoE.
- **Lazy conversion threshold**: >64 Q heads → lazy per-layer f16→f32.
- **Expert files**: `layer_XX_experts.bin`, 256 experts, INT4 quantized, mmap'd.

## Success Criteria

| Level | tok/s | Status |
|-------|-------|--------|
| Current | 0.03 | Broken (serial conversion) |
| Minimum viable | 0.5 | Can iterate |
| Target | 1.0 | Session goal |
| Stretch | 3-5 | Research prediction |

Get to 1.0 tok/s. If you get there early, push for 3-5.
