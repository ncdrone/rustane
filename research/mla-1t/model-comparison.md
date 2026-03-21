# MLA Model Comparison: V2-Lite → V3 → Kimi-K2

> What changes between models, what stays the same.
> Use this to predict what breaks when scaling up.

## Architecture Comparison

| Parameter | V2-Lite (15.7B) | V3 (671B) | Kimi-K2 (1T) |
|-----------|----------------|-----------|--------------|
| **hidden_size** | 2048 | 7168 | 7168 |
| **num_layers** | 27 | 61 | 61+ |
| **num_heads** | 16 | 128 | 128 |
| **kv_lora_rank** | 512 | 512 | 512 |
| **qk_nope_head_dim** | 128 | 128 | 128 |
| **qk_rope_head_dim** | 64 | 64 | 64 |
| **v_head_dim** | 128 | 128 | 128 |
| **q_lora_rank** | None (direct) | 1536 | 1536 |
| **num_experts** | 64 | 256 | 256 |
| **num_experts_per_tok** | 6 | 8 | 8 |
| **n_group** | 1 | 8 | 1 |
| **topk_group** | 1 | 4 | N/A |
| **shared_experts** | 2 | 1 | 1 |
| **scoring_func** | softmax | sigmoid | sigmoid |
| **routed_scaling_factor** | 1.0 | 2.5 | 2.5 |
| **rope_theta** | 10000 | 10000 | 10000 |
| **rope_scaling factor** | 40 | 40 | 40 |
| **mscale** | 0.707 | 1.0 | 1.0 |
| **first_k_dense_replace** | 1 | 1 | 1 |
| **vocab_size** | 102400 | 129280 | 129280 |
| **weight dtype** | bf16 | FP8 (e4m3) | FP8 (e4m3) |
| **model type** | base | base | instruction-tuned |

## What Changes Between Models

### V2-Lite → V3 (scale + Q LoRA + sigmoid)

| Change | Impact | Code change needed |
|--------|--------|-------------------|
| hidden 2048→7168 | Larger matmuls, more memory | Config only |
| 16→128 heads | W_UK absorption: 16→128 per-head sgemv (needs Metal kernel) | Metal kernel |
| Q LoRA (rank 1536) | Two-stage Q: x→W_qa→norm→W_qb→split | Already implemented (Task 9) |
| softmax→sigmoid routing | Different scoring + grouped top-k | Already implemented (Task 9) |
| n_group=8, topk_group=4 | Grouped expert selection | Already implemented |
| routed_scaling_factor=2.5 | Expert output scaled | Config only |
| mscale 0.707→1.0 | Different attention scale | Config only |
| FP8 weights | New converter pipeline (fp8→int4) | New converter code |
| 64→256 experts | 4x expert files, larger mmap | Config only |
| vocab 102400→129280 | Larger embedding + LM head | Config only |

### V3 → Kimi-K2 (instruction tuning + minor arch)

| Change | Impact | Code change needed |
|--------|--------|-------------------|
| n_group: 8→1 | Simpler routing (no grouped selection) | Config only |
| Instruction tuning | Chat template needed for prompts | Tokenizer config |
| Same MLA architecture | No attention code changes | None |
| Same FP8 weights | Same converter | None |

### What DOESN'T change (portable code)

- MLA absorbed attention path (two dot products summed)
- KV cache structure (latent [512] + k_pe [64])
- YaRN RoPE with partial rotary
- RMSNorm
- SwiGLU dense FFN
- 4-bit quantized expert dispatch (Metal)
- Weight loading (backbone.bin + index)
- Generation loop structure

## Risk Assessment for Scaling

### V3 (671B) — Known Risks

1. **128 per-head sgemv for W_UK absorption** — 1.28ms overhead on CPU. Must have Metal kernel. This is the #1 performance blocker.

2. **FP8→INT4 conversion** — New weight format. Block-wise dequant (128×128 tiles). Research doc has exact algorithm.

3. **Memory budget** — 671B at 4-bit ≈ 84GB + KV cache. Fits M4 Max 128GB but tight. Expert paging needed for 8K context.

4. **256 experts × 61 layers** — 15,616 expert files at ~280MB each = 4.3TB total. Need lazy loading / expert paging.

### Kimi-K2 (1T) — Additional Risks

5. **Instruction tuning format** — Must use chat template, not raw text. Tokenizer handles this.

6. **1T parameter count** — Similar to V3 in active params (37B), but more experts. Same memory footprint if routed sparsely.

7. **Tool-calling evaluation** — K2's primary benchmark is agentic, not perplexity. Testing framework needs structured eval.

## File Locations

| What | Where |
|------|-------|
| V2-Lite HF weights | `weights/deepseek-v2-lite/` |
| V2-Lite converted | `weights/rustane-deepseek-v2-lite/` |
| V2-Lite config | `configs/deepseek-v2-lite.toml` |
| V2-Lite references | `weights/references/deepseek_v2_lite_*` |
| V3 HF weights | `weights/deepseek-v3/` (downloading) |
| V3 config | `configs/deepseek-v3.toml` (TBD) |
| K2 HF weights | TBD |
| Research (external) | `rustane-research/mla-1t/stage1-external-2026-03-21/` |
| Research (internal) | `research/mla-1t/` |
| Test code | `crates/moe-infer/tests/` |
