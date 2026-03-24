# K2 Optimization Gossip

## Current State
tok/s: 1.3 | ms/layer: 12.4 | wins: 0 | experiments: 0

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
