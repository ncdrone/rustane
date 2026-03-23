# V3 Optimization Ideas — Synthesized from 43 Research Files
> Generated 2026-03-22 from 4 parallel research agents mining the full MLA-1T corpus.
> Deduplicated and categorized. Cross-referenced against experiments-v3.tsv (23 experiments tried).

## Category Status

| Category | Status | Experiments Run | Verdict |
|----------|--------|----------------|---------|
| CPU overlap/scheduling | EXHAUSTED | 6 (pipeline, pread-ffn, deferred-convert, metal-convert, convert0-lastlayer, parallel-gate-up) | 3 wins, then ceiling |
| CPU parallelization | EXHAUSTED | 4 (manual-gemv, rayon-perhead, parallel-oproj, parallel-gate-up) | Apple BLAS already optimal |
| Allocation elimination | EXHAUSTED | 2 (elim-norm-allocs, cached-layer0) | <50µs invisible |
| f16 CPU compute | DEAD END | 3 (f16-sgemv, f16-direct-path, f16-shared-expert) | Needs Metal, not CPU |
| **BLAS batching** | OPEN | 1 (sgemm-attention: correct but invisible at seq_len=5) | Q LoRA batching untried |
| **Metal GPU compute** | OPEN | 0 | Biggest untapped category |
| **Expert caching/pool** | OPEN | 0 | ExpertPool built, unwired |
| **Speculative decoding** | OPEN | 0 | V2-Lite available |
| **Memory layout** | OPEN | 0 | INT8 KV, f16 buffers untried |
| **LM head** | OPEN | 0 | 4.35 GB, ~11ms/token |
| **Profiling** | OPEN | 0 | No per-component MLA breakdown |

---

## SMALL OPTIMIZATIONS (auto-agent scope, <100 lines each)

These can go directly into the gossip "Suggested Next" list.

### S1. Per-Component MLA Profiling
- **Category:** Profiling
- **Impact:** Enables all other optimizations (no direct perf gain)
- **How:** Add Instant::now() timing inside mla_forward_decode for: W_qa, W_qb, W_kva, W_UK absorption, attention scores, value recon, O_proj individually. Print per-layer breakdown. One decode run reveals real bottleneck.
- **Effort:** 30 min, ~20 lines
- **Source:** v3-phase2-optimize.md, stage4 research prompt

### S2. Batch Q LoRA W_qa + W_qb
- **Category:** BLAS batching
- **Impact:** Save ~30µs/layer dispatch overhead (two sequential sgemv on same input)
- **How:** Pre-concatenate W_qa [1536,7168] and W_qb [24576,1536] at load time. One sgemm call produces both outputs. Split result after.
- **Effort:** 1 hour, ~30 lines
- **Source:** stage5-plan, gossip suggestion

### S3. Overlap LM Head with Next Token Embedding
- **Category:** Overlap
- **Impact:** Hide ~5-11ms of lm_head sgemv ([151936,7168])
- **How:** thread::scope — lm_head on main thread while background thread does embedding lookup + first layer's norm for next token. Different data, no BW conflict.
- **Effort:** 1 hour, ~40 lines
- **Source:** gossip suggestion, POST-MORTEM

### S4. Pre-Convert Small Tensors at Load Time
- **Category:** Memory layout
- **Impact:** Eliminate repeated micro-conversions for norms, router, bias (~0.5 GB total)
- **How:** At model load, convert all RMSNorm weights, router gates, biases to f32 and store permanently. Skip conversion for these in the per-layer loop.
- **Effort:** 30 min, ~20 lines
- **Source:** wave1-rq1-lazy-conversion.md

### S5. 2MB-Aligned Expert Buffers
- **Category:** I/O optimization
- **Impact:** 3.6x effective pread throughput (Apple SSD DMA coalesces 2MB reads)
- **How:** Replace expert buffer allocation with posix_memalign(buf, 2*1024*1024, size). Same buffers reused for pread + Metal.
- **Effort:** 30 min, ~10 lines
- **Source:** wave2-flashmoe-llamacpp-ssd.md

### S6. Expert Pool Stats Logging (Diagnostic)
- **Category:** Expert caching / profiling
- **Impact:** Reveals actual hit rate, miss pattern, hot expert distribution
- **How:** Wire ExpertPool::stats() into decode loop, log hit_rate/miss_count every 10 tokens. Don't wire the pool for caching yet — just measure what WOULD happen.
- **Effort:** 30 min, ~15 lines
- **Source:** wave1-rq2-expert-pool.md

### S7. Remove final_norm.to_vec() (Already Done but Verify)
- **Category:** Allocation elimination
- **Impact:** 4 unnecessary copies per token
- **How:** Verify the elim-norm-allocs experiment actually removed this. If not, borrow directly.
- **Effort:** 5 min
- **Source:** AUDIT quick wins

### S8. RMSNorm via NEON Intrinsics
- **Category:** Compute
- **Impact:** Avoid vDSP call overhead for small vectors (7168 elements)
- **How:** Replace vDSP_sve + vDSP_vsdiv with inline NEON: float32x4_t accumulate, vrsqrteq_f32 for reciprocal sqrt, vmulq_f32 for scale. Avoids ~0.5µs/call FFI overhead × 183 calls/token.
- **Effort:** 1 hour, ~40 lines
- **Source:** gossip suggestion

---

## MEDIUM OPTIMIZATIONS (borderline auto-agent scope, 50-150 lines)

### M1. Wire ExpertPool for Stats + Caching
- **Category:** Expert caching
- **Impact:** 90-95% cache hit rate → eliminate most pread I/O (~15ms/token saved)
- **How:** Wire pool.rs request() into generate_v2.rs decode loop. On miss: pread to pool slot. On hit: use cached buffer directly. Start with 2000 experts (~45 GB).
- **Effort:** 2 hours, ~80 lines
- **Source:** wave1-rq2-expert-pool.md, pool.rs already built

### M2. Expert Speculation (Previous-Token Prediction)
- **Category:** Expert caching
- **Impact:** 50-65% miss prediction → reduce miss penalty from 17ms to 6-9ms
- **How:** Cache last token's routing per layer (464 ints). During attention phase, async pread predicted experts. Promote on hit.
- **Effort:** 2 hours, ~50 lines
- **Source:** mr6b-expert-speculation.md

### M3. Dynamic Expert Pool Sizing
- **Category:** Expert caching
- **Impact:** Avoid OOM, graceful degradation under memory pressure
- **How:** HOST_VM_INFO64 every 64 tokens. Shrink pool if free < 4 GB, grow if > 8 GB.
- **Effort:** 1 hour, ~30 lines
- **Source:** wave1-rq2-expert-pool.md

### M4. INT8 KV Cache (Per-Layer Absmax)
- **Category:** Memory layout
- **Impact:** 72% KV memory savings + 4x attention bandwidth reduction
- **How:** Quantize kv_latent [512] to INT8 per-layer. Keep k_pe [64] in f16. Use ARM SDOT intrinsics for INT8 dot products.
- **Effort:** 3 hours, ~100 lines
- **Source:** mr6c-int8-attention-cache.md

---

## ARCHITECTURE CHANGES (manual session only, >150 lines)

### A1. Metal f16 GEMV for Backbone Projections
- **Category:** Metal GPU compute
- **Impact:** 1.13 → 3+ tok/s (eliminate f32 conversion entirely)
- **How:** Write Metal f16 GEMV kernel. Replace all CPU sgemv for Q/KV/O projections. f16 weights stay on GPU, no conversion.
- **Effort:** 4 hours
- **Source:** mr3-gpu-projections.md, 03-architecture-10toks.md

### A2. GPU INT4 Backbone Projections
- **Category:** Metal GPU compute
- **Impact:** 488ms → 12-25ms total backbone (20-40x)
- **How:** Reuse expert INT4 dequant kernel with backbone dimensions. 5.1 GB persistent MTLBuffers.
- **Effort:** 6 hours
- **Source:** mr3-gpu-projections.md

### A3. V2-Lite Speculative Decoding
- **Category:** Speculative decoding
- **Impact:** 1.9-2.4x multiplier on whatever base speed we achieve
- **How:** V2-Lite drafts K=4-5 tokens, V3 verifies in batch. Rejection sampling. KV rollback.
- **Effort:** 2-3 days
- **Source:** mr4-speculative-decoding.md

### A4. Fused Metal SDPA Kernel (MLA Absorbed Attention)
- **Category:** Metal GPU compute
- **Impact:** Additional 0.5-1ms/layer
- **How:** Single Metal kernel: q_absorbed·kv_latent + q_pe·k_pe → online softmax → value recon.
- **Effort:** 4 hours
- **Source:** stage4 research prompt RQ4, wave2-mlx-mla-impl.md

### A5. Async Metal Pipeline (Remove waitUntilCompleted)
- **Category:** Metal GPU compute
- **Impact:** +0.5 tok/s (overlap GPU+CPU between layers)
- **How:** MTLSharedEvent signaling instead of synchronous waits. GPU does projections while CPU does norms+routing.
- **Effort:** 4 hours
- **Source:** mr1-heterogeneous-pipeline.md

### A6. Single Unified Command Buffer (All 61 Layers)
- **Category:** Metal GPU compute
- **Impact:** Eliminate 60 × 85µs = 5ms dispatch overhead
- **How:** One MTLCommandBuffer for entire token. Requires GPU-primary architecture.
- **Effort:** 6 hours
- **Source:** stage5-plan R2, ane-infer pattern

---

## CORRECTNESS FIXES (manual session, urgent)

### B1. x_cache[4096] → [7168] (Metal OOB)
### B2. staging_ptr aliasing UB
### B3. wrap_mmap OOB padding
### B4. pread short reads
### B5. ExpertPool not wired (I/O waste)
### B6. Scalar attention (invisible at seq_len=5 but wrong)
### B7. routed_scaling_factor dead code

---

## RECOMMENDED AUTO-AGENT SEQUENCE

For the next auto-optimize run, seed these into gossip "Suggested Next":

1. **S1** — Profile per-component MLA (reveals real targets)
2. **S5** — 2MB-aligned expert buffers (10 lines, could be significant)
3. **S3** — Overlap lm_head with next token (proven pattern)
4. **S4** — Pre-convert small tensors (eliminate micro-conversions)
5. **S2** — Batch Q LoRA (BLAS dispatch savings)
6. **S6** — ExpertPool stats logging (diagnostic, informs M1)
7. **S8** — RMSNorm NEON intrinsics (eliminate FFI overhead)
