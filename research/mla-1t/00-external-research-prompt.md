# External Research Prompt for Orchestrator

> Give this to your research orchestrator. It should dispatch agents to answer these specific questions.
> Results should be saved to: research/mla-1t/02-external-findings.md

---

## Context

We are building DeepSeek-V3 (671B MoE, MLA attention) inference in Rust on Apple M4 Max 128GB. Expert FFN weights stream from SSD (352GB at 4-bit). Attention weights are RAM-resident. We already have working Qwen3-MoE-30B inference at 19.6 tok/s with Metal 4-bit GEMV shaders. We need to add Multi-Latent Attention (MLA) support.

## Research Questions

### RQ1: MLA Absorbed Attention — Implementation Details

Search for actual MLA implementations (not just the paper). We need the EXACT computation graph for single-token decode.

Specific questions:
- In DeepSeek-V3's official inference code (github.com/deepseek-ai/DeepSeek-V3), how is `absorbed attention` implemented? Find the exact einsum/matmul sequence.
- How does `kv_b_proj` get split into W_UK (key reconstruction) and W_UV (value reconstruction)? Is it a simple reshape, or is there a learned split?
- The `q_a_proj` → `q_a_layernorm` → `q_b_proj` two-stage projection: is q_a_layernorm applied BEFORE or AFTER the reshape into heads?
- How is `kv_a_proj_with_mqa` structured? The "with_mqa" suffix suggests the rope key portion is multi-query (shared across heads). Confirm: is k_pe [64] shared across all 128 heads, or per KV-head group?
- What is the `scoring_func = "sigmoid"` routing? How exactly does sigmoid + topk + e_score_correction_bias work vs softmax + topk?

### RQ2: How Do Other Frameworks Implement MLA Decode?

Search for MLA implementations in:
- **llama.cpp** — search github for "MLA" or "deepseek" in ggml/llama code. How do they handle absorbed attention? What GGML ops?
- **vLLM** — how does vLLM handle MLA KV cache? Do they decompress or use absorbed form?
- **SGLang** — they claim fastest DeepSeek-V3 inference. What's their MLA strategy?
- **mlx-lm** — Apple's MLX framework. Do they support DeepSeek-V3? How do they handle MLA on Metal?
- **ExLlamaV2** — known for fast quantized inference. MLA support?

For each: what's the decode tok/s they achieve, and what's their absorbed attention implementation approach?

### RQ3: Metal Shader Design for MLA Attention Scores

The absorbed attention scores kernel computes:
```
scores[h, t] = q_absorbed[h, 512] · cache_latent[t, 512] + q_pe[h, 64] · cache_rope[t, 64]
```
for 128 heads × seq_len positions.

Questions:
- Is there an existing Metal shader for this dual-component dot product + addition pattern?
- What's the optimal threadgroup geometry? (heads as threadgroups? positions as threads?)
- Should the nope and rope components be computed separately or fused?
- How does flash-attention / online softmax interact with the two-component score?
- At seq_len=1000, this is 128 * 1000 * 576 = 73.7M FLOPs. Is this GPU-bound or memory-bound on M4 Max?

### RQ4: FP8 to 4-bit Weight Conversion

DeepSeek-V3 safetensors use FP8 (e4m3) format with per-block scale_inv tensors.

Questions:
- What is the exact FP8 e4m3 dequantization formula? `value = fp8_bits * scale_inv[block_idx]`?
- What block size do the scale_inv tensors use? (128x128? per-row? per-channel?)
- Has anyone published a FP8 → INT4 conversion pipeline? (dequant FP8 to f32, then quantize to 4-bit with group scales/zeros)
- What quality loss does FP8 → INT4 conversion introduce? Any published perplexity numbers?
- Alternative: are there pre-quantized INT4/GPTQ/AWQ versions of DeepSeek-V3 on HuggingFace?

### RQ5: DeepSeek-V3 Inference on Apple Silicon — What Exists?

Search for anyone who has run DeepSeek-V3 on Apple Silicon (M-series Macs):
- llama.cpp + DeepSeek-V3 GGUF — what tok/s? What quant levels?
- MLX + DeepSeek-V3 — does it work? Performance?
- Any custom implementations (like our rustane or flash-moe-ane)?
- What are the practical bottlenecks people report? (memory, compute, I/O)
- Does anyone stream expert weights from SSD, or do they require everything in RAM?

### RQ6: Shared Expert Implementation

DeepSeek-V3 has 1 shared expert per MoE layer (always activated, no routing).

Questions:
- Is the shared expert output simply ADDED to the routed expert output, or is there a learned gate?
- In the official code, does the shared expert run in PARALLEL with routed experts or SEQUENTIALLY?
- Some implementations mention `shared_expert_gate` — does DeepSeek-V3 have this? (our target-1t.toml doesn't show one)
- What's the compute cost? Same dims as routed experts (2048 intermediate) = 88.1M FLOPs per layer.

### RQ7: Multi-Token Prediction (MTP)

DeepSeek-V3 has `num_nextn_predict_layers = 1` (one MTP module).

Questions:
- Is MTP required for correct inference, or only for speculative decoding?
- Can we skip MTP entirely for basic generation and add it later as an optimization?
- What does the MTP module look like architecturally? (extra attention layer? MLP head?)
- Does Kimi-K2 have MTP? (our research says num_nextn_predict_layers = 0)

---

## Output Format

For each question, provide:
1. **Answer** — the specific finding
2. **Source** — URL or repo path where you found it
3. **Confidence** — high/medium/low based on source quality
4. **Implication for rustane** — what this means for our implementation
