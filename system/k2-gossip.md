# K2 Optimization Gossip

## Current State
tok/s: 1.54 | ms/layer: 10.6 | wins: 0 | experiments: 1

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

## The Goal
Get tok/s as high as possible. Theoretical max ~5 tok/s.

## Iteration Log

### Iteration 1: f16 direct decode path — REVERTED
- **Experiment**: Replace f32 double-buffer pipeline with f16 inline decode (run_layer_f16)
- **Result**: 0.32 tok/s (scalar conversion), 0.75 tok/s (SIMD convert_to_f32_slice) vs 1.54 baseline
- **Verdict**: REVERTED — f16 inline conversion adds to critical path (no pipelining)
- **Insight**: The f32 double-buffer hides ~1.6ms/layer conversion behind 8ms FFN compute. f16 path puts conversion on critical path. Also: scalar `f16::to_f32()` in a loop is 40x slower than `convert_to_f32_slice()` (SIMD FCVTL) — always use bulk SIMD conversion.
- **Infrastructure kept**: SIMD fix in blas.rs (sgemv_f16 + sgemv_f16_trans now use convert_to_f32_slice). No effect on current f32 path.
- **Previous commit 06afdd0 was wrong**: claimed 1.43 tok/s but benchmarked at 0.32 (scalar conversion not fixed). Reverted.

## Suggested Next Experiments
1. **Reduce "Other" 4ms bucket** — profile what's in the conversion/norms/Metal overhead. Could be L2 cache misses from f32 buffer thrashing.
2. **Overlap expert pread with MLA** — currently sequential. MLA is only 0.4ms but pread could start earlier.
3. **Batch expert Metal dispatches** — 8 expert dispatches per layer, each with command buffer overhead. Single fused dispatch?
4. **Wire expert-pager pool** — pool.rs is built but not wired into decode. Could eliminate pread for cached experts.
5. **ANE for norms/activations** — 17.8 TFLOPS sitting idle. Could offload RMSNorm + SiLU.
