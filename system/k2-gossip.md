# K2 Optimization Gossip

## Current State
tok/s: 1.57 | ms/layer: 12.0 | wins: 1 | experiments: 4

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

## Dead Ends (do not retry)
- **lm_head optimization**: Only 3.3% of total time. cblas_sgemv saturates bandwidth single-threaded.
- **f16 inline decode**: Puts conversion on critical path. f32 double-buffer hides it behind FFN compute.
- **Pool write-back on critical path**: 23 MB alloc+copy per miss causes page cache eviction. Must be async or deferred.

## Suggested Next Experiments
1. **Reduce MLA 2.2ms → ~1.0ms** — o_proj is 1.2ms (235 MB sgemv). Cache o_proj f32 for all layers (14.3 GB) to eliminate from double-buffer, freeing bandwidth. Or overlap o_proj with FFN start.
2. **Reduce FFN pread time** — when page cache warm, pread is fast (~3ms). When cold, 10ms+. Pre-populate pool buffers during prefill (not decode). Or use madvise(MADV_WILLNEED) to prefault expert pages.
3. **Async pool write-back** — overlap pool buffer copy with next layer's MLA (CPU idle while Metal runs). Recovers pool benefit without decode-path overhead.
4. **Reduce Metal dispatch overhead** — 3.5ms for 8 experts. Cache constant u32 buffers, reuse command encoders.
5. **ANE for norms/activations** — 17.8 TFLOPS sitting idle. Could offload RMSNorm + SiLU.
