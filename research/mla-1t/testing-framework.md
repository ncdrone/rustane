# MLA-1T Testing Framework

> Portable validation suite: same test code for V2-Lite → V3 → Kimi-K2.
> Date: 2026-03-21. Updated as models are added.

## Philosophy

**Correctness before performance.** There's no point measuring tok/s on gibberish. Tests are ordered by what they diagnose — earliest tests catch the cheapest bugs.

**Model-agnostic test code.** Only the reference data and config paths change between models. The test logic is identical. To port: regenerate references, update one config block.

**Greedy token match is the wrong gate.** It requires bit-exact precision across every layer. A single bf16→f16 rounding difference at layer 0 cascades into a completely different token sequence. Use logit distribution comparison instead.

---

## The 4-Level Validation Hierarchy

### Level 1: Embedding Check (ms, catches converter bugs)

**What:** Compare `rmsnorm(embed(last_token))` against HF's `layer0_hidden_input`.

**Why first:** If the embedding is wrong, nothing else matters. Catches:
- Weight converter bf16→f16 precision issues
- Tensor offset bugs in `backbone_index.json`
- Embedding table shape misalignment

**Threshold:** max_diff < 0.02 (bf16→f16 loses ~0.5 ULP)

**Current state (V2-Lite):** max_diff=0.017. PASS.

### Level 2: Single Layer Check (ms, catches attention/FFN bugs)

**What:** Run all prompt tokens through layer 0 only. Compare last-token output against HF's `layer0_output`.

**Why:** Isolates MLA attention + FFN without accumulation across 27+ layers. Catches:
- W_UK/W_UV transposition (was a real bug — caught here)
- RoPE frequency errors
- Softmax numerics
- Dense FFN SwiGLU bugs

**Threshold:** cosine_sim > 0.90, max_diff < 1.0

**Current state (V2-Lite):** cosine_sim=0.938, max_diff=0.152. PASS.

### Level 3: Logit Distribution (seconds, catches accumulation)

**What:** Full forward pass through all layers. Compare final logit vector against HF.

**Why:** Precision differences accumulate across layers. This catches:
- Systematic bias (e.g., wrong attention scale compounds)
- MoE routing bugs (wrong experts selected)
- Shared expert integration errors
- KV cache corruption

**Metrics:**
- **Top-1 match:** Does our argmax = HF's argmax?
- **Top-5 overlap:** How many of HF's top-5 are in our top-5?
- **Our top-1 in HF top-k:** Is our pick reasonable?
- **Cosine similarity:** Full distribution alignment
- **KL divergence:** Information-theoretic distance

**Gate:** Our top-1 must be in HF's top-10. Cosine > 0.5.

**Current state (V2-Lite):** Our top-1 ("a") is HF's #2. In HF top-5. Model works, precision shifts the argmax.

### Level 4: Greedy Generation (seconds, the strictest test)

**What:** Generate 20 tokens, compare against HF greedy reference.

**Why:** The aspirational gate. If this passes, the implementation is bit-exact.

**Gate:** Report only. Do NOT use as a blocking gate — it's too strict for f16 inference. Use Level 3 as the real gate.

**Current state (V2-Lite):** 1/20 match (first token wrong, cascade failure). Expected until precision debugging complete.

---

## Reference Data Per Model

Each model needs these files generated from HF (one-time, takes ~2 min):

| File | Contents | Used by |
|------|----------|---------|
| `*_intermediates.npz` | Layer 0/1 inputs, Q/KV projections, W_UK/W_UV, outputs | Levels 1, 2 |
| `*_last_logits.npy` | Full logit vector [vocab_size] for a fixed prompt | Level 3 |
| `*_greedy.json` | 20-token greedy generation + prompt | Level 4 |

**Generator script:** `scripts/generate_deepseek_v2_ref.py`

To port to a new model:
1. Copy the script, change `MODEL_DIR`
2. If instruction-tuned (K2), use chat template for the prompt
3. Run once to generate reference files
4. Update `model_config()` in `test_model_validation.rs`

---

## What to Measure at Scale

### Correctness (blocks everything)

| Metric | V2-Lite (15.7B) | V3 (671B) | K2 (1T) |
|--------|-----------------|-----------|---------|
| Level 1 embedding max_diff | 0.017 | TBD | TBD |
| Level 2 layer 0 cosine_sim | 0.938 | TBD | TBD |
| Level 3 our top-1 in HF top-5 | YES | TBD | TBD |
| Level 3 cosine similarity | TBD | TBD | TBD |

### Performance (only after correct)

| Metric | V2-Lite | V3 | K2 |
|--------|---------|-----|------|
| Prefill tok/s | TBD | TBD | TBD |
| Decode tok/s | TBD | TBD | TBD |
| KV cache MB/1K ctx | 254 MB | TBD | TBD |
| Model load time (s) | ~1s | TBD | TBD |
| Peak memory (GB) | TBD | TBD | TBD |

### Quality (after correct + fast)

| Metric | Applicable to | Notes |
|--------|--------------|-------|
| WikiText-2 perplexity | V2-Lite (base), V3 (base) | NOT K2 (instruction-tuned) |
| MMLU 5-shot | All | Standard LLM benchmark |
| HumanEval pass@1 | All | Code quality |
| Tool-calling accuracy | K2 only | K2's primary design goal |

---

## Test Files

```
crates/moe-infer/tests/
├── test_model_validation.rs      # 4-level portable suite (THE gate)
├── test_mla_attention.rs         # MLA unit tests vs HF intermediates
├── test_mla_layer_divergence.rs  # Layer-by-layer divergence diagnosis
├── test_v2_lite_logits.rs        # Logit cosine/KL/top-k deep analysis
├── test_v2_lite_generation.rs    # Greedy token match (strict)
├── test_yarn_rope.rs             # YaRN RoPE unit tests
├── test_q_lora.rs                # V3 Q LoRA path (synthetic)
├── bench_v2_lite_tok_per_sec.rs  # Throughput benchmark
└── bench_tok_per_sec.rs          # Qwen3 throughput (regression)
```

---

## Known Precision Issues

1. **bf16→f16 conversion:** The weight converter converts bf16 (HF) → f32 → f16 (rustane). This loses precision in the low bits. Embedding max_diff=0.017 from this.

2. **BLAS accumulation order:** Accelerate's sgemv accumulates in a different order than PyTorch's BLAS. For large matmuls (e.g., 3072×2048), this can differ by up to ~0.05.

3. **Softmax numerics:** Our softmax uses f32 with f64 dot products. HF uses bf16 internally (when loaded as bf16). The f32 path is actually MORE precise, which paradoxically causes different results.

4. **W_UK transposition:** Fixed. Was using wrong BLAS call — W_UK stored as [nope, kv_rank] needs transposed multiply (sgemv_f32_trans). Caught by Level 2 test.

5. **MoE expert quantization:** 4-bit quantization of routed experts introduces ~1% relative error per expert. With top-6 routing, this accumulates. Shared experts (f16, no quant) are more precise.
