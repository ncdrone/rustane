# MLA Inference: DeepSeek-V2-Lite Dry Run

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get DeepSeek-V2-Lite (15.7B, MLA attention, 64 MoE experts) generating correct tokens on M4 Max 128GB. Validate the MLA absorbed attention path that scales to DeepSeek-V3 (671B) and Kimi-K2 (1T) via config changes.

**Architecture:** MLA replaces GQA's 4 simple projections (Q/K/V/O) with a 2-stage LoRA pipeline + absorbed attention. KV cache stores compressed [512] latent + [64] rope key per token (71x smaller than full MHA). Attention scores computed as TWO SEPARATE DOT PRODUCTS (nope + rope) summed — NOT concatenated. Scale = 1/sqrt(192) × mscale².

**Tech Stack:** Rust, Accelerate BLAS (f32 sgemv/sgemm), Metal (existing 4-bit GEMV + new W_UK absorption kernel), safetensors, memmap2, half (f16).

**Research source:** `rustane-research/mla-1t/stage1-external-2026-03-21/` — 3 waves of agent research. Key files:
- `FINAL.md` — corrected architecture decisions, P0 risks
- `wave3-implementation-blueprint.md` — exact Rust structs, Metal kernel code, shape tables
- `wave3-risks-and-traps.md` — 6 risks, 3 must-fix-before-code showstoppers

**Critical corrections from research (must get right):**
1. Score computation: TWO separate dot products summed, NOT concatenated [576]
2. Scale factor: 1/sqrt(192) × mscale², NOT 1/sqrt(576) — 3.25x difference
3. `kv_a_layernorm` MUST be applied before caching kv_latent — silent bug if missed
4. k_pe stored POST-RoPE in cache; q_pe gets RoPE at decode time
5. W_UK absorption needs batched kernel, NOT 128 separate sgemv (1.28ms overhead)
6. DO NOT use llama.cpp as ground truth — it converts MLA to flattened GQA

**Validation model:** DeepSeek-V2-Lite (weights at `weights/deepseek-v2-lite/`, 31GB bf16).
**V2-Lite specifics:** `q_lora_rank=None` (direct q_proj, no Q LoRA), `scoring_func=softmax` (not sigmoid), 16 heads (not 128), 27 layers, hidden=2048, 64 experts top-6, 2 shared experts.

---

## Current State

- Qwen3-MoE-30B: 19.6 tok/s decode, 20/20 HF match, GQA attention
- MLA scaffolding: `mla.rs` has MlaConfig, MlaKvCache, CPU reference stubs
- Config: `config.rs` has `attention.kind` supporting "mla" + `kv_lora_rank`
- Metal: 4-bit dequant GEMV with fused kernel, parameterized for any dims
- V2-Lite: downloaded at `weights/deepseek-v2-lite/` (bf16, 4 shards)

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `crates/moe-infer/src/mla_attention.rs` | MLA forward: Q proj, KV compress, absorbed attention (two dot products), O proj |
| `crates/moe-infer/src/yarn_rope.rs` | YaRN RoPE: frequency scaling, mscale, partial rotary (64 of 192 dims) |
| `crates/moe-infer/src/generate_v2.rs` | DeepSeek-V2 generation loop: MLA + shared experts + dense FFN |
| `configs/deepseek-v2-lite.toml` | V2-Lite inference config |
| `crates/moe-infer/src/bin/convert_deepseek.rs` | Rust binary: bf16 safetensors → rustane format |
| `scripts/generate_deepseek_v2_ref.py` | One-time: generate HF golden reference outputs |
| `crates/moe-infer/tests/test_mla_attention.rs` | MLA unit tests: absorbed attention, scale, W_UK split |
| `crates/moe-infer/tests/test_yarn_rope.rs` | YaRN RoPE vs Python reference values |
| `crates/moe-infer/tests/test_v2_lite_generation.rs` | E2E: generation matches HF reference |
| `crates/moe-infer/tests/test_q_lora.rs` | V3 prep: Q LoRA unit test (synthetic data) |
| `crates/moe-router/tests/test_sigmoid_v3.rs` | V3 prep: sigmoid + grouped top-k routing |

### Modified Files

| File | Change |
|------|--------|
| `crates/moe-infer/src/lib.rs` | Add `pub mod mla_attention; pub mod yarn_rope; pub mod generate_v2;` |
| `crates/moe-infer/src/config.rs` | MLA fields, dense layer config, shared experts, scoring_func |
| `crates/moe-infer/src/weights.rs` | MLA weight loading, shared experts, parameterize layer count |
| `crates/moe-kernels/src/mla.rs` | W_UK/W_UV split from kv_b_proj at load time |
| `crates/moe-kernels/src/dequant.rs` | Add `mla_q_absorb` Metal kernel (batched per-head GEMV) |
| `crates/moe-router/src/lib.rs` | Add `route_sigmoid_v3()` with bias + grouped top-k |

---

## Task 1: Generate Python Reference Tensors

**Files:**
- Create: `scripts/generate_deepseek_v2_ref.py`

**What it does:** Runs HF transformers on V2-Lite to produce golden values. This MUST happen first — everything else validates against these. Per the research: "Your Rust implementation isn't correct until it matches these vectors."

**Two outputs:**
1. **Per-layer intermediate tensors** (for Task 4 MLA validation): q_nope, q_pe, kv_latent (post-norm), k_pe (post-rope), attention scores, v_out, layer output — for layer 0 and layer 1 with a fixed input.
2. **Greedy generation reference** (for Task 8 E2E test): prompt → 20 tokens, greedy, save to `weights/references/deepseek_v2_lite_greedy.json`.

- [ ] **Step 1: Write the reference generator**

```python
# scripts/generate_deepseek_v2_ref.py
# Extracts per-layer intermediates + greedy generation reference
# Uses official HF transformers DeepSeek-V2 implementation
# Outputs:
#   weights/references/deepseek_v2_lite_intermediates.npz (per-layer tensors)
#   weights/references/deepseek_v2_lite_greedy.json (20-token greedy generation)
```

Key: hook into `model.model.layers[0].self_attn` to capture intermediates. Save q after split (q_nope, q_pe), kv_out after kv_a_proj, kv_latent after norm, k_pe after rope, attention weights, and final output.

- [ ] **Step 2: Run the generator**

```bash
python3 scripts/generate_deepseek_v2_ref.py
```

- [ ] **Step 3: Verify outputs exist and are non-trivial, commit**

**Gate:** Reference files exist at `weights/references/deepseek_v2_lite_*.{npz,json}`. Intermediates have expected shapes.

---

## Task 2: Config + TOML

**Files:**
- Modify: `crates/moe-infer/src/config.rs`
- Create: `configs/deepseek-v2-lite.toml`

**What it does:** Extend InferConfig for MLA parameters, dense FFN layers, shared experts, scoring function, YaRN rope config.

- [ ] **Step 1: Add MLA fields to AttentionSection**

```rust
#[serde(default)]
pub qk_nope_head_dim: Option<usize>,
#[serde(default)]
pub qk_rope_head_dim: Option<usize>,
#[serde(default)]
pub v_head_dim: Option<usize>,
#[serde(default)]
pub q_lora_rank: Option<usize>,  // None = direct q_proj (V2-Lite), Some(1536) = Q LoRA (V3)
```

- [ ] **Step 2: Add dense/shared/scoring fields to FfnSection**

```rust
#[serde(default)]
pub first_k_dense_replace: usize,  // layers 0..N are dense FFN, not MoE
#[serde(default)]
pub dense_inter_size: Option<usize>,
#[serde(default = "default_scoring")]
pub scoring_func: String,  // "softmax" or "sigmoid"
#[serde(default = "default_scaling")]
pub routed_scaling_factor: f32,
```

- [ ] **Step 3: Add YaRN rope config section**

```rust
#[derive(Clone, Debug, Deserialize)]
pub struct RopeScalingSection {
    pub factor: f32,           // 40.0
    pub original_max_position_embeddings: usize,  // 4096
    pub beta_fast: f32,        // 32.0
    pub beta_slow: f32,        // 1.0
    pub mscale: f32,           // 0.707 (V2-Lite), 1.0 (V3)
    pub mscale_all_dim: f32,   // 0.707 (V2-Lite), 1.0 (V3)
}
```

- [ ] **Step 4: Fix `is_moe_layer()` to use first_k_dense_replace**

```rust
pub fn is_moe_layer(&self, layer: usize) -> bool {
    layer >= self.ffn.first_k_dense_replace
}
```

- [ ] **Step 5: Write `configs/deepseek-v2-lite.toml`**

- [ ] **Step 6: Test both configs parse (V2-Lite + Qwen3 no regression), commit**

**Gate:** Both TOMLs parse. `is_moe_layer(0)` returns false for V2-Lite, true for Qwen3.

---

## Task 3: YaRN RoPE

**Files:**
- Create: `crates/moe-infer/src/yarn_rope.rs`
- Create: `crates/moe-infer/tests/test_yarn_rope.rs`

**What it does:** YaRN RoPE with 3 key differences from standard RoPE:
1. Frequency scaling: dims split into low/mid/high bands by beta_fast/beta_slow
2. mscale: attention scale multiplier for long sequences (seqlen > 4096)
3. Partial rotary: apply to only qk_rope_head_dim=64 dims, not full head

**From the research (Risk 2):**
- YaRN correction ONLY applies if `seqlen > original_max_position_embeddings` (4096)
- k_pe is [64] NOT per-head — single shared vector
- mscale formula: `0.1 * config.mscale * ln(factor) + 1.0`

- [ ] **Step 1: Write YaRN frequency computation**

Match the official `precompute_freqs_cis` exactly, including the `seqlen > original_seq_len` branch and `find_correction_range` / `linear_ramp_factor`.

- [ ] **Step 2: Write `apply_rope_partial`**

Apply RoPE to ONLY the rope_dim slice of a tensor. Standard neox-style pairing.

- [ ] **Step 3: Write tests against Python reference**

Test at pos=0 (trivial), pos=100 (standard), pos=4096 (boundary — YaRN should NOT activate for V2-Lite at this length if seq < original_max), pos=10000 (YaRN active if applicable).

- [ ] **Step 4: Test mscale computation**

Verify: for V2-Lite (mscale=0.707, factor=40): `mscale_used = 0.1 * 0.707 * ln(40) + 1.0 ≈ 1.261`
For V3 (mscale=1.0, factor=40): `mscale_used = 0.1 * 1.0 * ln(40) + 1.0 ≈ 1.369`

- [ ] **Step 5: Run tests, commit**

**Gate:** RoPE values match HF reference within 1e-5. mscale matches formula.

---

## Task 4: MLA Absorbed Attention Forward Pass

**Files:**
- Create: `crates/moe-infer/src/mla_attention.rs`
- Create: `crates/moe-infer/tests/test_mla_attention.rs`
- Modify: `crates/moe-kernels/src/mla.rs` (W_UK/W_UV split)

**What it does:** The core MLA forward. This is the most critical piece. Must follow the corrected architecture exactly.

**The corrected decode path (from FINAL.md):**
```
STEP 1: Q projection
  V2-Lite: q = x @ q_proj^T → split into q_nope [H,128] + q_pe [H,64]
  V3: q_latent = x @ W_qa^T → RMSNorm → q = q_latent @ W_qb^T → split

STEP 2: KV compression
  kv_out = x @ W_kva^T → split [kv_latent(512), k_pe(64)]
  kv_latent = RMSNorm(kv_latent, kv_a_layernorm)  ← CRITICAL: norm BEFORE cache
  k_pe = apply_yarn_rope(k_pe, pos)                ← stored POST-rope
  cache.write(layer, pos, kv_latent, k_pe)

STEP 3: Absorbed attention — TWO SEPARATE DOT PRODUCTS
  q_absorbed = einsum("hd,hdc->hc", q_nope, W_UK)  [H, 512]
  scores_nope = q_absorbed @ kv_latent_cache^T       [H, seq_len]
  scores_rope = q_pe @ k_pe_cache^T                  [H, seq_len]
  scale = 1/sqrt(192) × mscale²
  scores = (scores_nope + scores_rope) * scale
  weights = softmax(scores)

STEP 4: Value combination
  v_latent = weights @ kv_latent_cache               [H, 512]
  v = einsum("hc,hdc->hd", v_latent, W_UV)           [H, 128]

STEP 5: Output projection
  output = concat(v) @ W_o^T                          [hidden]
```

- [ ] **Step 1: Add W_UK/W_UV split to mla.rs**

Split kv_b_proj [num_heads*(nope+v), kv_lora_rank] into W_UK and W_UV at load time. Verify shapes against blueprint: W_UK [H, nope, kv_lora_rank], W_UV [H, v_head_dim, kv_lora_rank].

- [ ] **Step 2: Update MlaKvCache**

Store kv_latent and k_pe as SEPARATE tensors (not concatenated [576]) per the corrected architecture. The blueprint shows concatenated storage but the FINAL.md corrects this.

- [ ] **Step 3: Write `mla_forward_decode` — CPU path first**

All BLAS sgemv for projections. W_UK absorption as loop over heads (128 sgemv calls for V3, 16 for V2-Lite). CPU dot products for attention scores. This is the correctness reference — Metal optimization comes later.

- [ ] **Step 4: Write unit tests against Python reference tensors (from Task 1)**

```rust
#[test]
fn mla_q_projection_matches_hf() { /* compare q_nope, q_pe */ }
#[test]
fn mla_kv_compress_matches_hf() { /* compare kv_latent (post-norm), k_pe (post-rope) */ }
#[test]
fn mla_absorbed_attention_matches_hf() { /* compare attention weights, v_out */ }
#[test]
fn mla_full_layer_matches_hf() { /* compare full layer output */ }
#[test]
fn mla_scale_factor_correct() { /* verify 1/sqrt(192) × mscale² */ }
```

Load reference tensors from `weights/references/deepseek_v2_lite_intermediates.npz`.

- [ ] **Step 5: Run tests, iterate until matching, commit**

**Gate:** All MLA operations match HF reference within 1e-3 (f16 precision).

---

## Task 5: Rust Weight Converter

**Files:**
- Create: `crates/moe-infer/src/bin/convert_deepseek.rs`

**What it does:** Reads V2-Lite safetensors (bf16), writes rustane format. All Rust, no Python (except the one-time reference generator from Task 1).

**Key differences from Qwen3 converter:**
- MLA tensors: `kv_a_proj_with_mqa`, `kv_b_proj`, `kv_a_layernorm`, direct `q_proj` (not q/k/v separate)
- Shared experts: `mlp.shared_experts.{gate,up,down}_proj`
- Dense FFN (layer 0): `mlp.{gate,up,down}_proj` (no experts)
- Split kv_b_proj → W_UK + W_UV at conversion time (pre-split, stored separately)
- bf16 input (V2-Lite). FP8 input support deferred to V3 task.

- [ ] **Step 1: Write the converter**
- [ ] **Step 2: Run on V2-Lite**
- [ ] **Step 3: Spot-check converted tensors against safetensors**
- [ ] **Step 4: Commit**

**Gate:** Converted weights load. Spot-checked tensors match within f16 precision.

---

## Task 6: MLA Weight Loading

**Files:**
- Modify: `crates/moe-infer/src/weights.rs`

**What it does:** Load MLA-specific tensors. Parameterize layer count (remove hardcoded 48). Add shared expert mmap loading.

- [ ] **Step 1: Add `MlaLayerWeights` struct**
- [ ] **Step 2: Add `mla_layer_weights()` method**
- [ ] **Step 3: Parameterize layer count from config**
- [ ] **Step 4: Add shared expert weight loading**
- [ ] **Step 5: Test with converted V2-Lite, commit**

**Gate:** All MLA tensors load. Shapes match config.

---

## Task 7: Generation Loop

**Files:**
- Create: `crates/moe-infer/src/generate_v2.rs`

**What it does:** Full decode loop for DeepSeek-V2. Handles: MLA attention (Task 4), dense FFN (layer 0), MoE FFN with shared experts (layers 1-26), YaRN RoPE (Task 3).

- [ ] **Step 1: Write `ModelV2` struct + `load()`**

Pre-split W_UK/W_UV at load time. Pre-convert attention weights to f32 for BLAS. Build YaRN rope tables.

- [ ] **Step 2: Write `run_layer_v2()`**

Dispatches to MLA attention + either dense FFN or MoE FFN.

- [ ] **Step 3: Dense FFN path (layer 0)**

Standard SwiGLU. V2-Lite dense_inter=10944. Use BLAS sgemv on pre-converted f32 weights.

- [ ] **Step 4: Shared expert path**

For MoE layers: run shared experts (same SwiGLU, BLAS or Metal 4-bit) + routed experts (existing Metal path). Sum outputs. V2-Lite has 2 shared experts.

- [ ] **Step 5: Wire `generate_v2()` — full loop**

Embed → 27 layers → final norm → LM head → sample → repeat.

- [ ] **Step 6: Commit**

**Gate:** Compiles. No runtime test yet (next task).

---

## Task 8: E2E Generation Test

**Files:**
- Create: `crates/moe-infer/tests/test_v2_lite_generation.rs`

**What it does:** Compare our generation against the HF reference from Task 1.

- [ ] **Step 1: Write test**

```rust
#[test]
#[ignore = "requires converted weights + tokenizer"]
fn test_v2_lite_generation_matches_hf() {
    // Load model, generate 20 tokens greedy
    // Compare against weights/references/deepseek_v2_lite_greedy.json
    // Target: ≥15/20 token match
}
```

- [ ] **Step 2: Run, debug, iterate**
- [ ] **Step 3: Commit when passing**

**Gate:** ≥15/20 HF greedy match.

---

## Task 9: V3 Prep — Q LoRA + Sigmoid Routing (Unit Tests Only)

**Files:**
- Modify: `crates/moe-infer/src/mla_attention.rs` (Q LoRA path)
- Modify: `crates/moe-router/src/lib.rs` (sigmoid routing)
- Create: `crates/moe-infer/tests/test_q_lora.rs`
- Create: `crates/moe-router/tests/test_sigmoid_v3.rs`

**What it does:** V3 features that V2-Lite doesn't exercise. Validated with unit tests on synthetic data.

- [ ] **Step 1: Q LoRA path** — when `q_a_proj` is Some: x → W_qa → RMSNorm → W_qb → split
- [ ] **Step 2: Unit test Q LoRA** (small dims, verify two-stage matches composed)
- [ ] **Step 3: Sigmoid routing with bias + grouped top-k** (from FINAL.md exact algorithm)

```rust
pub fn route_sigmoid_v3(
    gate_logits: &[f32],
    bias: &[f32],          // e_score_correction_bias
    n_group: usize,        // 8 (V3) or 1 (K2)
    topk_group: usize,     // 4
    top_k: usize,          // 8
    scaling_factor: f32,   // 2.5
) -> RouteResult
```

- [ ] **Step 4: Unit test routing** (correct experts, unbiased weights, scaling factor)
- [ ] **Step 5: Commit**

**Gate:** Unit tests pass. Ready for V3 weights.

---

## Task 10: Benchmark + Final State

**Files:**
- Create: `crates/moe-infer/tests/bench_v2_lite_tok_per_sec.rs`

- [ ] **Step 1: Write V2-Lite benchmark**
- [ ] **Step 2: Run both benchmarks** (V2-Lite + Qwen3 no regression)
- [ ] **Step 3: Log to experiments-infer.tsv**
- [ ] **Step 4: Commit**

**Gate:** V2-Lite generates correct tokens. Qwen3 not regressed. Baseline recorded.

---

## Verification

**After every task:**
```bash
cargo build -p moe-infer --release
cargo test -p moe-infer --release  # no regressions
```

**After Task 4 (MLA correctness):**
```bash
cargo test -p moe-infer --test test_mla_attention --release -- --nocapture
# All intermediate tensors match HF < 1e-3
```

**After Task 8 (E2E):**
```bash
cargo test -p moe-infer --test test_v2_lite_generation --release -- --ignored
cargo test -p moe-infer --test test_generation --release -- --ignored  # Qwen3 still works
```

**Exit gate:**
```bash
cargo test -p moe-infer --release                                               # all pass
cargo test -p moe-infer --test test_mla_attention --release                     # MLA correctness
cargo test -p moe-infer --test test_v2_lite_generation --release -- --ignored    # V2-Lite HF match
cargo test -p moe-infer --test test_generation --release -- --ignored            # Qwen3 HF match
```
