# K2 Optimization Gossip

## Current State
tok/s: 1.43 | ms/layer: 13.3 | wins: 1 | experiments: 1

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

## Bottleneck (updated iter 1)
Per-token decode: ~700ms. Per-layer: ~13.3ms.
- MLA attention: ~3ms (Q LoRA + KV compress + W_UK absorb + attn scores + W_UV + O proj)
- FFN overlap: ~10ms (shared expert sgemv_f32 || expert pread from SSD || Metal INT4 dispatch)
- Convert(N+1) f16→f32: ~6ms (overlapped with FFN, fully hidden)
- lm_head: ~6ms (f16_par, was ~31ms with f32 single-thread)
Expert pread (8 × 22 MB = 176 MB/layer, 60 layers = 10.5 GB/token) dominates.
When experts are in page cache, ~17.5 GB/s. When cold, much slower.

## Dead Ends
(none yet for K2 with internal SSD configuration)

## Suggested Next
1. Wire ExpertPool into decode loop for explicit expert caching (~44 GB for 2000 experts, ~90% hit rate)
2. Profile per-component K2 timing with RUSTANE_MLA_PROFILE=1 to validate bottleneck model
3. Use sgemv_f16 for shared expert FFN (overlap phase — halves DRAM contention with convert thread)
4. mlock backbone mmap to prevent page eviction under memory pressure
5. Overlap lm_head with next token's embedding lookup

## Iteration Log
[iter 1] RESULT: s3-f16-decode-path — IMPROVED 1.43 tok/s. Combined optimization: (1) f16 direct decode path via run_layer_f16, eliminating double-buffer f32 conversion + thread::scope pipeline. Halves backbone DRAM traffic per layer (sgemv_f16 chunked L2 convert reads only f16 from DRAM). (2) sgemm_nt for f16 MLA attention scores (was scalar f64 loops). (3) f16 lm_head via sgemv_f16_par, saves 4.7 GB RAM, halves logit traffic. Establishes K2 internal SSD baseline.
[iter 1] INSIGHT: K2 on internal SSD with cached backbone delivers 1.4+ tok/s, 286x faster than external SSD cold start. The bottleneck is expert pread from SSD (~10.5 GB/token for 60 layers × 8 experts × 22 MB). RAM savings help by giving OS more page cache for expert data. The f16 direct path is simpler code AND faster — no thread::scope overhead, no double-buffer management.
