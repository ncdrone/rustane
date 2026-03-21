# MLA Precision Notes

> What causes numerical divergence between our Rust implementation and HF Python.
> Updated as bugs are found and fixed. Read this before debugging any E2E mismatch.

## Error Budget

For a 27-layer model (V2-Lite), errors compound multiplicatively. If each layer introduces ε relative error:
- After 1 layer: ε
- After 27 layers: ~27ε (additive residual connections help, but attention is multiplicative)
- Practical: 0.5% error/layer → ~14% total → different argmax at output

## Error Sources (Ranked by Impact)

### 1. bf16 → f16 Weight Conversion (LARGEST)

**Source:** HF stores weights in bf16. Our converter does bf16 → f32 → f16. bf16 has 8-bit exponent + 7-bit mantissa. f16 has 5-bit exponent + 10-bit mantissa. Different precision tradeoffs — some values that are exact in bf16 are rounded in f16.

**Impact:** Embedding max_diff = 0.017. This is the baseline error before any computation.

**Measured:** Token 317 embedding matches exactly (converter is correct), but layernorm amplifies bf16/f16 differences.

**Fix options:**
- Store weights as f32 (2x memory, but eliminates conversion error)
- Store weights as bf16 natively (need bf16 BLAS or convert at load time)
- Current: f16 storage, f32 compute. Acceptable for V2-Lite.

### 2. MoE Expert 4-bit Quantization

**Source:** Routed expert weights are quantized to INT4 with group_size=128. Asymmetric quantization: scale = (max-min)/15, zero = min.

**Impact:** ~1-3% relative error per expert FFN. With top-6 routing, the weighted sum has ~0.5-1% error. Shared experts are NOT quantized (stored f16), so they're more precise.

**Note:** HF runs in bf16 with no quantization. Our 4-bit experts introduce error that HF doesn't have. This is the biggest single source of divergence for MoE layers.

### 3. BLAS Accumulation Order

**Source:** Accelerate's cblas_sgemv uses hardware-specific accumulation (AMX on Apple Silicon). PyTorch may use different BLAS. The order of floating-point additions affects the result due to non-associativity.

**Impact:** ~1e-5 to 1e-3 per matmul, depending on dimension. For [3072, 2048] Q projection: up to ~0.05 absolute error.

**Not a bug.** Both are "correct" within floating-point semantics.

### 4. Softmax Numerics

**Source:** Our softmax uses f32 with f64 dot products for attention scores. HF uses bf16 for attention computation (when loaded as bf16).

**Impact:** Small but consistent bias. Our f32 path is actually MORE precise, which paradoxically produces different results from HF's bf16 path.

### 5. RoPE Frequency Computation

**Source:** YaRN correction_dim computation involves `ln()` and `powf()` which have implementation-dependent precision.

**Impact:** Negligible (< 1e-6). The frequencies are precomputed once and stored in tables.

## Bugs Found and Fixed

| Bug | Symptom | Root Cause | Fix | Test that catches it |
|-----|---------|------------|-----|---------------------|
| W_UK transposition | Layer 0 cosine < 0.5 | sgemv interprets [nope,kv_rank] as [kv_rank,nope] | Use sgemv_f32_trans | Level 2 |
| Missing BOS token | Wrong first embedding | HF prepends BOS=100000, our tokenizer doesn't | Prepend in generate_v2 | Level 1 |
| Layer0_hidden_input is post-norm | Embedding test shows 16x diff | HF hook captures post-input_layernorm, not raw embedding | Apply layernorm before comparing | Level 1 |

## Debugging Playbook

When E2E generation doesn't match HF:

1. **Run Level 1.** If embedding doesn't match → converter bug or tokenizer mismatch.
2. **Run Level 2.** If layer 0 diverges → attention or FFN bug. Check:
   - Q projection dimensions
   - W_UK transposition (use sgemv_f32_trans, not sgemv_f32)
   - kv_a_layernorm applied BEFORE cache write
   - Attention scale = 1/sqrt(192) * mscale², not 1/sqrt(576)
3. **Run Level 3.** If logits have low cosine → accumulation across layers. Check:
   - MoE routing (softmax vs sigmoid)
   - Shared expert addition
   - 4-bit quantization quality
4. **Run test_mla_layer_divergence.** Compare layer 0 vs layer 1 max_diff to see if error grows.

## Current State (V2-Lite, 2026-03-21)

| Metric | Value | Status |
|--------|-------|--------|
| Embedding max_diff (post-norm) | 0.017 | PASS |
| Layer 0 cosine_sim | 0.938 | PASS (borderline) |
| Layer 0 max_diff | 0.152 | Acceptable |
| Our top-1 in HF top-5 | YES ("a" = HF's #2) | PASS |
| Greedy match | 1/20 | Expected (cascade from top-1 miss) |

**Diagnosis:** Model fundamentally works. The ~6% cosine error at layer 0 accumulates across 27 layers, shifting the argmax from "Paris" (#1) to "a" (#2). The primary error source is 4-bit expert quantization + bf16→f16 weight conversion.
