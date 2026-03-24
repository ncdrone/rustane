# K2 Optimization Gossip

## Current State
tok/s: 1.57 | ms/layer: 12.0 | wins: 1 | experiments: 7

## Model Facts
- Kimi-K2: 1 trillion parameters, 61 layers
- 64 attention heads (MLA with Q LoRA)
- 384 experts per MoE layer, top-8 routed, INT4 quantized
- 1 dense layer (layer 0), 60 MoE layers
- Backbone: 23.4 GB f16, Expert files: 9 GB each (524 GB total)
- Tokenizer: tiktoken-based, vocab 163840

## Hardware
- M4 Max 128 GB unified memory
- CPU (AMX): 3 TFLOPS — currently handles MLA attention + shared FFN
- Metal GPU: 15 TFLOPS — currently handles expert INT4 dispatch only
- ANE: 17.8 TFLOPS — currently UNUSED (ane-bridge crate exists)
- NVMe SSD: 17.5 GB/s pread

## Per-Layer Breakdown (MEASURED, warm)
- **MLA: 2.2ms/layer** (18%) — Q LoRA (1.3ms) + o_proj (1.2ms) dominate. NOT 0.4ms as previously estimated.
- **FFN: 9.0ms/layer** (75%) — max(pread ~5ms, shared_ffn ~4ms) + Metal ~3.5ms
- **convert_wait: ~0ms** — f16→f32 conversion fully hidden behind FFN in double-buffer pipeline
- **lm_head: 24ms** (3.3%) — single-threaded cblas_sgemv, already bandwidth-saturated
- **other: 1.6ms total** (<0.3%) — thread::scope overhead + residual adds

## The Goal
Get tok/s as high as possible. Theoretical max ~5 tok/s.

## Iteration Log

### Iteration 1: f16 direct decode path — REVERTED
- **Experiment**: Replace f32 double-buffer pipeline with f16 inline decode (run_layer_f16)
- **Result**: 0.32 tok/s (scalar conversion), 0.75 tok/s (SIMD convert_to_f32_slice) vs 1.54 baseline
- **Verdict**: REVERTED — f16 inline conversion adds to critical path (no pipelining)
- **Insight**: The f32 double-buffer hides ~1.6ms/layer conversion behind 8ms FFN compute. f16 path puts conversion on critical path. Also: scalar `f16::to_f32()` in a loop is 40x slower than `convert_to_f32_slice()` (SIMD FCVTL) — always use bulk SIMD conversion.
- **Infrastructure kept**: SIMD fix in blas.rs (sgemv_f16 + sgemv_f16_trans now use convert_to_f32_slice). No effect on current f32 path.

### Iteration 2: lm_head parallel sgemv — NO EFFECT
- **Experiment**: Parallel rayon sgemv_f32 for lm_head (163840×7168 = 4.7 GB), also tried sgemv_f16_par and sgemv_f16
- **Result**: 1.54 tok/s (identical to baseline).
- **Verdict**: NO EFFECT — lm_head is only 24ms (3.3% of total). Not a bottleneck.
- **Insight**: cblas_sgemv already saturates M4 Max memory bandwidth single-threaded. f16 conversion throughput (FCVTL) is SLOWER than DRAM read.

### Iteration 3: Disable pool write-back — WIN (+0.03 tok/s)
- **Experiment**: Disable ExpertPool write-back that copies 23 MB per expert miss to pool buffers
- **Result**: 1.57 tok/s (median warm: 1.58, 1.57, 1.57) vs 1.54 baseline
- **Verdict**: WIN — pool write-back was causing 3x regression (0.49 tok/s with default 3000 cap)
- **Root cause**: 480 expert misses/token × 23 MB alloc+copy = 11 GB memory churn per token. This evicts OS page cache entries for expert files, making pread slower. Net effect: write-back costs more than the cache hits save.
- **Fix**: Disabled write-back, OS page cache handles expert caching natively. Pool tracking still runs for statistics.
- **Profiling data**: MLA=2.2ms/layer (18%), FFN=9.0ms/layer (75%), convert_wait≈0, lm_head=24ms, other=1.6ms total.

### Iteration 4: Pool disable default (pool_cap=0) — NO EFFECT
- **Experiment**: Change default pool_cap from 3000→0 (pool=None) since write-back is disabled
- **Result**: 1.55 tok/s (median warm: 1.37, 1.55, 1.57) vs 1.57 baseline
- **Verdict**: NO EFFECT — -1.3%, within noise. Committed as code cleanup.
- **Insight**: Pool HashMap tracking overhead (~0.3ms/layer = ~18ms/token) is real but too small to reliably measure against 650ms/token total. The 1.37 outlier shows page cache state dominates run-to-run variance. RUSTANE_POOL_CAP env var preserved for future pool experiments.

### Iteration 5: Split pread + pipelined Metal fused/down — NO EFFECT (REVERTED)
- **Experiment**: Split expert pread into gate+up (phase 1, overlaps shared_ffn) then down (phase 2, overlaps GPU fused dispatch). Two Metal command buffers instead of one.
- **Result**: 1.56 tok/s (warm) vs 1.57 baseline
- **Verdict**: NO EFFECT — pread_dn (1.5-2.5ms) > GPU fused (~1.75ms), so GPU finishes before pread. No overlap benefit.
- **Detailed profiling** (warm, layers 1-3): pread_gu=2-3ms, shared_ffn=2.3-4ms, pread_dn=1.5-2.5ms, Metal_total=3.7-4ms
- **Insight**: The GPU expert dispatch (fused gate+up+SiLU) completes faster than a single pread of 62 MB down data. Splitting the Metal dispatch into two command buffers adds overhead (~0.3ms/layer) without pipeline benefit. The fundamental bottleneck is pread_total + Metal_total = serial, and neither can be hidden behind the other because pread feeds Metal.
- **Infrastructure kept**: const u32 buffer caching in MetalDequantGemv (eliminates 360 Metal buffer allocs/token).

### Iteration 6: Cached constant u32 Metal buffers — NO EFFECT
- **Experiment**: Pre-cache u32 constant Metal buffers (hidden=7168, moe_inter=2048, group_size=128) in MetalDequantGemv struct, eliminating all per-dispatch HashMap creation and newBufferWithBytes_length_options calls.
- **Result**: 1.56 tok/s (median warm: 1.50, 1.56, 1.56) vs 1.57 baseline
- **Verdict**: NO EFFECT — eliminated ~360 Metal buffer allocs/token but each was only ~1μs (4-byte buffer). Total savings ~0.18ms/token, unmeasurable against 650ms total.
- **Insight**: Metal constant buffer CPU-side overhead is negligible. The 3.5ms/layer Metal dispatch time is GPU kernel execution, not buffer setup. Committed as code cleanup — removes 3 HashMap allocations per dispatch call.

### Iteration 7: Parallel per-head W_UK/W_UV sgemv — NO EFFECT
- **Experiment**: Parallelized 64 per-head W_UK absorption and W_UV projection sgemv calls using rayon par_chunks_mut. Each head's sgemv is independent ([128,512] matrix, 256 KB), GPU idle during MLA.
- **Result**: 1.51 tok/s (median warm: 1.48, 1.51, 1.53) vs 1.57 baseline
- **Verdict**: NO EFFECT — synthetic timing shows only 315µs→286µs (1.10x) for both loops combined. Over 61 layers: ~1.8ms savings, unmeasurable against 650ms total.
- **Insight**: Per-head sgemv on [128,512] is ~5µs/call (compute-bound, data fits L2). 64 sequential calls = 320µs/layer. Rayon parallel dispatch overhead (~15µs) nearly cancels the parallelism benefit. AMX sgemv on tiny matrices has per-call overhead that dominates; parallelism doesn't help because the BLAS function-call overhead is already the bottleneck, not the sequential execution time.
- **Infrastructure**: Fixed pre-existing build error in make_attn_weights (removed dead f16 field assignments that referenced non-existent struct fields).

## Dead Ends (do not retry)
- **lm_head optimization**: Only 3.3% of total time. cblas_sgemv saturates bandwidth single-threaded.
- **f16 inline decode**: Puts conversion on critical path. f32 double-buffer hides it behind FFN compute.
- **Pool write-back on critical path**: 23 MB alloc+copy per miss causes page cache eviction. Must be async or deferred.
- **Split pread + pipeline Metal fused/down**: pread_dn (1.5-2.5ms) > GPU fused (~1.75ms). No overlap benefit, extra cmd buffer overhead cancels.
- **Metal constant buffer caching**: 4-byte u32 buffers are ~1μs each via newBufferWithBytes. ~360/token = 0.18ms, unmeasurable.
- **Parallel per-head sgemv (W_UK/W_UV)**: 64×[128,512] sgemv = 320µs/layer total. Each call ~5µs, dominated by BLAS function-call overhead, not compute. Rayon parallelism saves <30µs/layer = 1.8ms/token.

## Suggested Next Experiments
1. **Async Metal dispatch overlap with MLA** — Metal dispatch (3.5ms/layer) and MLA (2.2ms/layer) use different hardware (GPU vs CPU/AMX). Currently serial. If Metal were launched async and MLA(N+1) ran concurrently, saves 2.2ms/layer × 61 = 134ms (20%). Requires: split fused_and_down into launch+wait, double-buffer staging for Metal, restructure decode loop. ~150 lines.
2. **Reduce MLA 2.2ms → ~1.0ms** — o_proj is 1.2ms (235 MB sgemv). Dispatch o_proj on Metal GPU using F32_GEMV_SHADER already compiled in dequant.rs, overlapped with next layer's convert.
3. **Reduce FFN pread time** — when page cache warm, pread is fast (~3ms). When cold, 10ms+. Pre-populate pool buffers during prefill (not decode). Or use madvise(MADV_WILLNEED) to prefault expert pages.
4. **Async pool write-back** — overlap pool buffer copy with next layer's MLA (CPU idle while Metal runs). Recovers pool benefit without decode-path overhead.
5. **Reduce Metal GPU kernel time** — 3.5ms for 8 experts. Try: larger threadgroups (TG_SIZE=512, ROWS_PER_TG=16), batching all 8 experts in single kernel dispatch, or compute-aware scheduling.
6. **ANE for norms/activations** — 17.8 TFLOPS sitting idle. Could offload RMSNorm + SiLU.
