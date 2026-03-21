# MLA Inference: DeepSeek-V2-Lite Dry Run

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get DeepSeek-V2-Lite (15.7B, MLA attention, 64 MoE experts) generating correct tokens on M4 Max 128GB. Validate MLA absorbed attention, shared experts, dense FFN layers, and YaRN RoPE. All in Rust. This is the dry run — same code will scale to DeepSeek-V3 (671B) and Kimi-K2 (1T) via config changes.

**Architecture:** Extend the existing Qwen3-MoE-30B inference pipeline with MLA attention (absorbed form), shared expert FFN, dense layer support, and YaRN RoPE. V2-Lite uses direct Q projection (no Q LoRA), so Q LoRA (needed for V3) is implemented as a separate task with unit tests only. The existing Metal 4-bit dequant GEMV shaders handle V2-Lite expert dims without modification.

**Tech Stack:** Rust, Accelerate BLAS (cblas_sgemv/sgemm), Metal (existing 4-bit GEMV shaders), safetensors (dev-dependency for weight conversion tests), memmap2, half (f16).

**Validation model:** DeepSeek-V2-Lite (weights at `weights/deepseek-v2-lite/`, 31GB bf16, already downloaded).

**Research source:** `rustane-research/mla-1t/stage1-external-2026-03-21/FINAL.md` — corrected MLA math (two separate dot products, scale=1/sqrt(128+64)×mscale², kv_a_layernorm before caching).

---

## Current State

- Qwen3-MoE-30B: 19.6 tok/s decode, 20/20 HF match, GQA attention
- MLA scaffolding: `mla.rs` has MlaConfig, MlaKvCache, CPU reference functions
- Config: `config.rs` already has `attention.kind` supporting "mla" and `kv_lora_rank`
- Metal: 4-bit dequant GEMV with fused gate+up+SiLU, parameterized for any dims
- V2-Lite: downloaded at `weights/deepseek-v2-lite/` (4 safetensor shards, bf16)

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `crates/moe-infer/src/mla_attention.rs` | MLA forward pass: Q projection, KV compression, absorbed attention, O projection |
| `crates/moe-infer/src/yarn_rope.rs` | YaRN RoPE with mscale, partial rotary dim |
| `crates/moe-infer/src/generate_v2.rs` | DeepSeek-V2 generation loop (MLA + shared experts + dense FFN) |
| `configs/deepseek-v2-lite.toml` | V2-Lite inference config |
| `scripts/convert_deepseek_v2.rs` | Rust binary: safetensors → rustane format (backbone.bin + expert files) |
| `crates/moe-infer/tests/test_mla_attention.rs` | MLA unit tests (absorbed attention math, KV cache, Q absorption) |
| `crates/moe-infer/tests/test_yarn_rope.rs` | YaRN RoPE tests against known values |
| `crates/moe-infer/tests/test_v2_lite_generation.rs` | E2E generation test with HF reference |

### Modified Files

| File | Change |
|------|--------|
| `crates/moe-infer/src/lib.rs` | Add `pub mod mla_attention; pub mod yarn_rope; pub mod generate_v2;` |
| `crates/moe-infer/src/config.rs` | Add MLA fields (qk_nope_head_dim, qk_rope_head_dim, v_head_dim, q_lora_rank), dense layer config, shared expert config, scoring_func |
| `crates/moe-infer/src/weights.rs` | Add MLA weight loading (kv_a_proj, kv_b_proj, kv_a_layernorm, o_proj_mla), shared expert weights, dense FFN weights. Parameterize layer count. |
| `crates/moe-kernels/src/mla.rs` | Update MlaConfig to match V2-Lite dims, add W_UK/W_UV split |
| `crates/moe-router/src/lib.rs` | Add `route_sigmoid_v3()` with bias correction and grouped top-k (unit tested, not used by V2-Lite which uses softmax) |

### Unchanged (but referenced)

| File | Why |
|------|-----|
| `crates/moe-kernels/src/dequant.rs` | Metal shaders work for V2-Lite dims unchanged |
| `crates/moe-infer/src/generate.rs` | Qwen3-MoE-30B path preserved, not modified |
| `crates/moe-infer/src/blas.rs` | sgemv/sgemm used by MLA projections |
| `crates/moe-infer/src/rmsnorm.rs` | Used by MLA (kv_a_layernorm, q_a_layernorm) |

---

## Task 1: Config + TOML for DeepSeek-V2-Lite

**Files:**
- Modify: `crates/moe-infer/src/config.rs`
- Create: `configs/deepseek-v2-lite.toml`

**What it does:** Extend InferConfig to support MLA attention parameters, dense FFN layers, shared experts, and scoring function. Write the V2-Lite TOML config.

- [ ] **Step 1: Add MLA fields to AttentionSection**

Add to `config.rs` AttentionSection:
```rust
/// MLA-only: non-positional Q/K head dim
#[serde(default)]
pub qk_nope_head_dim: Option<usize>,
/// MLA-only: positional (RoPE) Q/K head dim
#[serde(default)]
pub qk_rope_head_dim: Option<usize>,
/// MLA-only: V head dim
#[serde(default)]
pub v_head_dim: Option<usize>,
/// MLA-only: Q LoRA rank (None = direct projection, as in V2-Lite)
#[serde(default)]
pub q_lora_rank: Option<usize>,
```

- [ ] **Step 2: Add dense layer + shared expert + scoring fields to FfnSection**

```rust
/// Number of initial dense (non-MoE) layers. 0 = all MoE.
#[serde(default)]
pub first_k_dense_replace: usize,
/// Dense FFN intermediate size (for dense layers only).
#[serde(default)]
pub dense_inter_size: Option<usize>,
/// "softmax" or "sigmoid"
#[serde(default = "default_scoring")]
pub scoring_func: String,
/// Routed scaling factor (DeepSeek-V3 = 2.5, V2-Lite = 1.0)
#[serde(default = "default_scaling")]
pub routed_scaling_factor: f32,
```

- [ ] **Step 3: Update `is_moe_layer()` to use `first_k_dense_replace`**

```rust
pub fn is_moe_layer(&self, layer: usize) -> bool {
    layer >= self.ffn.first_k_dense_replace
}
```

- [ ] **Step 4: Add convenience accessors for MLA params**

- [ ] **Step 5: Write `configs/deepseek-v2-lite.toml`**

```toml
[model]
name = "deepseek-v2-lite"
vocab_size = 102400
hidden_size = 2048
num_layers = 27
max_position_embeddings = 163840
bos_token_id = 100000
eos_token_id = 100001
rms_norm_eps = 1e-6

[attention]
kind = "mla"
num_q_heads = 16
num_kv_heads = 16
head_dim = 128
rope_theta = 10000.0
kv_lora_rank = 512
qk_nope_head_dim = 128
qk_rope_head_dim = 64
v_head_dim = 128

[ffn]
all_moe = false
first_k_dense_replace = 1
dense_inter_size = 10944
moe_inter_size = 1408
num_experts = 64
num_experts_per_tok = 6
shared_expert_count = 2
norm_topk_prob = true
scoring_func = "softmax"
routed_scaling_factor = 1.0

[quantization]
bits = 4
group_size = 128
```

- [ ] **Step 6: Add config parse test**

- [ ] **Step 7: Run tests, commit**

```bash
cargo test -p moe-infer --lib config --release
git commit -m "feat: extend InferConfig for MLA + DeepSeek-V2-Lite TOML"
```

**Gate:** Config parses V2-Lite TOML. Qwen3 config still parses (no regression).

---

## Task 2: YaRN RoPE

**Files:**
- Create: `crates/moe-infer/src/yarn_rope.rs`
- Create: `crates/moe-infer/tests/test_yarn_rope.rs`
- Modify: `crates/moe-infer/src/lib.rs`

**What it does:** Implement YaRN RoPE with frequency scaling, mscale factor, and partial rotary dimension. DeepSeek-V2/V3 applies RoPE to only 64 of 192 Q/K dims (qk_rope_head_dim=64). The mscale factor adjusts attention scale for long sequences.

- [ ] **Step 1: Write YaRN RoPE module**

```rust
// yarn_rope.rs
pub struct YarnRopeConfig {
    pub rope_theta: f32,
    pub qk_rope_head_dim: usize,  // 64 for DeepSeek
    pub original_max_position_embeddings: usize,  // 4096
    pub factor: f32,       // 40.0
    pub beta_fast: f32,    // 32.0
    pub beta_slow: f32,    // 1.0
    pub mscale: f32,       // 0.707 (V2-Lite), computed from factor for V3
    pub mscale_all_dim: f32,  // 0.707
}

pub struct YarnRopeTables {
    cos_cache: Vec<f32>,  // [max_seq, rope_dim/2]
    sin_cache: Vec<f32>,  // [max_seq, rope_dim/2]
    pub mscale_factor: f32,  // attention scale multiplier
}

impl YarnRopeTables {
    pub fn build(config: &YarnRopeConfig, max_seq: usize) -> Self { ... }
    pub fn apply(&self, x: &mut [f32], num_heads: usize, pos: usize) { ... }
}
```

Key math: YaRN computes per-dimension frequency scaling based on `beta_fast`, `beta_slow`, and `factor`. Dimensions near the "fast" end get divided by `factor`, dimensions near the "slow" end are unchanged, and intermediate dimensions get a smooth interpolation.

- [ ] **Step 2: Write tests against known values**

Generate reference values from Python (HF transformers `DeepseekV2YarnRotaryEmbedding`). Test at positions 0, 100, 4096 (boundary), 10000 (beyond original_max).

- [ ] **Step 3: Run tests, commit**

```bash
cargo test -p moe-infer --test test_yarn_rope --release
git commit -m "feat: YaRN RoPE with mscale for DeepSeek-V2/V3"
```

**Gate:** RoPE values match HF reference within 1e-5.

---

## Task 3: MLA Attention Forward Pass

**Files:**
- Create: `crates/moe-infer/src/mla_attention.rs`
- Create: `crates/moe-infer/tests/test_mla_attention.rs`
- Modify: `crates/moe-infer/src/lib.rs`
- Modify: `crates/moe-kernels/src/mla.rs`

**What it does:** The core MLA forward pass for single-token decode. Uses absorbed attention (two separate dot products, NOT concatenated). This is the most complex and correctness-critical piece.

**V2-Lite specifics:** `q_lora_rank=None` → direct `q_proj` [q_total_dim, hidden] instead of two-stage Q LoRA. Still has absorbed attention via kv_b_proj.

- [ ] **Step 1: Update MlaConfig and add W_UK/W_UV split**

In `mla.rs`, update the config to support both V2-Lite (no Q LoRA) and V3 (with Q LoRA). Add a function to split kv_b_proj into W_UK and W_UV at load time:

```rust
/// Split kv_b_proj [num_heads*(nope+v_head_dim), kv_lora_rank] into
/// W_UK [num_heads, nope_dim, kv_lora_rank] and W_UV [num_heads, v_head_dim, kv_lora_rank].
/// Done once at model load.
pub fn split_kv_b_proj(
    kv_b_proj: &[f32],
    num_heads: usize,
    qk_nope_head_dim: usize,
    v_head_dim: usize,
    kv_lora_rank: usize,
) -> (Vec<f32>, Vec<f32>) { ... }
```

- [ ] **Step 2: Write `mla_forward_decode` — the absorbed attention function**

```rust
// mla_attention.rs

/// MLA forward pass for single-token decode.
/// Implements absorbed attention: two separate dot products (nope + rope), combined.
pub fn mla_forward_decode(
    x: &[f32],                    // [hidden_size]
    // Projections (f32, pre-converted at load time)
    q_proj: &[f32],              // [q_total_dim, hidden] (V2-Lite) or None if Q LoRA
    q_a_proj: Option<&[f32]>,    // [q_lora_rank, hidden] (V3 only)
    q_a_norm: Option<&[f32]>,    // [q_lora_rank] (V3 only)
    q_b_proj: Option<&[f32]>,    // [q_total_dim, q_lora_rank] (V3 only)
    kv_a_proj: &[f32],           // [kv_lora_rank + rope_dim, hidden]
    kv_a_norm: &[f32],           // [kv_lora_rank]
    w_uk: &[f32],                // [num_heads, nope_dim, kv_lora_rank] (pre-split)
    w_uv: &[f32],                // [num_heads, v_head_dim, kv_lora_rank] (pre-split)
    o_proj: &[f32],              // [hidden, num_heads * v_head_dim]
    // State
    cache: &mut MlaKvCache,
    layer: usize,
    pos: usize,
    rope: &YarnRopeTables,
    config: &MlaConfig,
) -> Vec<f32> {
    // STEP 1: Q projection
    // V2-Lite: direct q = x @ q_proj^T → split into q_nope + q_pe
    // V3: q_latent = x @ q_a^T → norm → q = q_latent @ q_b^T → split

    // STEP 2: KV compression
    // kv_out = x @ kv_a^T → split [kv_latent, k_pe]
    // kv_latent = RMSNorm(kv_latent)
    // k_pe = apply_yarn_rope(k_pe, pos)  — stored POST-rope
    // cache.append(layer, kv_latent, k_pe)

    // STEP 3: Absorbed attention (TWO separate dot products)
    // q_absorbed = einsum("hd,hdc->hc", q_nope, W_UK)  [num_heads, kv_lora_rank]
    // scores_nope = q_absorbed @ cache_latent^T          [num_heads, seq_len]
    // scores_rope = q_pe @ cache_rope^T                  [num_heads, seq_len]
    // scale = 1/sqrt(qk_nope_head_dim + qk_rope_head_dim) * mscale^2
    // scores = (scores_nope + scores_rope) * scale
    // weights = softmax(scores)

    // STEP 4: Value combination
    // v_latent = weights @ cache_latent                  [num_heads, kv_lora_rank]
    // v = einsum("hc,hdc->hd", v_latent, W_UV)          [num_heads, v_head_dim]

    // STEP 5: Output projection
    // output = concat(v) @ o_proj^T                      [hidden_size]
}
```

- [ ] **Step 3: Write unit tests with synthetic weights**

Test absorbed attention math independently:
1. Q absorption: q_nope @ W_UK matches reference
2. Two-dot-product scores: nope + rope components sum correctly
3. Scale factor: verify 1/sqrt(192) × mscale² at various seq_len
4. Full forward: small dims (2 heads, hidden=64, kv_lora_rank=16) against CPU reference

- [ ] **Step 4: Run tests, commit**

```bash
cargo test -p moe-infer --test test_mla_attention --release -- --nocapture
git commit -m "feat: MLA absorbed attention forward pass (V2-Lite + V3 compatible)"
```

**Gate:** All unit tests pass. Absorbed attention matches CPU reference < 1e-5.

---

## Task 4: Weight Converter (Rust Binary)

**Files:**
- Create: `crates/moe-infer/src/bin/convert_deepseek.rs`
- Modify: `crates/moe-infer/Cargo.toml` (add safetensors as dependency)

**What it does:** Rust binary that reads DeepSeek-V2-Lite safetensors (bf16) and writes rustane format (backbone.bin + backbone_index.json + layer_XX_experts.bin). All in Rust, no Python.

V2-Lite weights are bf16 (not FP8). V3 weights will be FP8 — add FP8 support in a later task.

- [ ] **Step 1: Add safetensors as a dependency (non-dev)**

```toml
safetensors = "0.7"  # move from dev-dependencies to dependencies
```

- [ ] **Step 2: Write the converter binary**

```rust
// src/bin/convert_deepseek.rs
// Reads safetensors, writes:
//   backbone.bin + backbone_index.json (MLA attention weights in f16, norms in f32)
//   layer_XX_experts.bin (4-bit quantized routed experts)
//   layer_XX_shared_experts.bin (4-bit quantized shared experts)
//
// Key differences from convert_qwen3:
// - MLA tensors: kv_a_proj_with_mqa, kv_b_proj, kv_a_layernorm, q_proj (not q/k/v separate)
// - Shared experts: mlp.shared_experts.{gate,up,down}_proj
// - Dense FFN (layer 0): mlp.{gate,up,down}_proj (no experts)
// - bf16 input (not FP8)
```

- [ ] **Step 3: Run converter on V2-Lite**

```bash
cargo run -p moe-infer --release --bin convert_deepseek -- \
  --model-dir weights/deepseek-v2-lite \
  --output-dir weights/rustane-deepseek-v2-lite \
  --config configs/deepseek-v2-lite.toml
```

- [ ] **Step 4: Verify converted weights**

Spot-check: load backbone.bin, read one tensor, compare against safetensors directly.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: Rust weight converter for DeepSeek-V2-Lite (bf16 → rustane format)"
```

**Gate:** Converted weights load without error. Spot-checked tensors match within f16 precision.

---

## Task 5: MLA Weight Loading

**Files:**
- Modify: `crates/moe-infer/src/weights.rs`

**What it does:** Extend BackboneWeights to load MLA-specific tensors: kv_a_proj, kv_b_proj, kv_a_layernorm, o_proj (different shape from GQA), shared expert mmaps. Parameterize layer count (was hardcoded to 48).

- [ ] **Step 1: Add MLA layer weights struct**

```rust
/// MLA attention weights for one layer (zero-copy slices).
pub struct MlaLayerWeights<'a> {
    pub input_norm: &'a [f32],
    pub post_attn_norm: &'a [f32],
    // Q projection (V2-Lite: direct, V3: q_a + q_b LoRA)
    pub q_proj: Option<&'a [f16]>,       // V2-Lite: [q_total, hidden]
    pub q_a_proj: Option<&'a [f16]>,     // V3: [q_lora_rank, hidden]
    pub q_a_layernorm: Option<&'a [f32]>,// V3: [q_lora_rank]
    pub q_b_proj: Option<&'a [f16]>,     // V3: [q_total, q_lora_rank]
    // KV compression
    pub kv_a_proj: &'a [f16],           // [kv_lora_rank + rope_dim, hidden]
    pub kv_a_layernorm: &'a [f32],      // [kv_lora_rank]
    pub kv_b_proj: &'a [f16],           // [num_heads*(nope+v), kv_lora_rank]
    // O projection
    pub o_proj: &'a [f16],             // [hidden, num_heads*v_head_dim]
    // Router (MoE layers only)
    pub router: Option<&'a [f16]>,
}
```

- [ ] **Step 2: Add `mla_layer_weights()` method**

- [ ] **Step 3: Parameterize layer count (remove hardcoded 48)**

- [ ] **Step 4: Add shared expert mmap loading**

- [ ] **Step 5: Test with converted V2-Lite weights, commit**

**Gate:** All MLA tensors load correctly from converted weights.

---

## Task 6: Generation Loop (generate_v2.rs)

**Files:**
- Create: `crates/moe-infer/src/generate_v2.rs`
- Modify: `crates/moe-infer/src/lib.rs`

**What it does:** Full DeepSeek-V2 generation loop. Handles: MLA attention, dense FFN (layer 0), MoE FFN with shared experts (layers 1-26), YaRN RoPE, MLA KV cache. Reuses existing Metal 4-bit GEMV for expert dispatch.

- [ ] **Step 1: Write `ModelV2` struct and `load()`**

Similar to existing `Model` but with MLA weights, pre-split W_UK/W_UV, YaRN rope tables.

- [ ] **Step 2: Write `run_layer_v2()` — single layer forward**

Dispatches to MLA attention + either dense FFN or MoE FFN based on layer index.

- [ ] **Step 3: Write dense FFN path**

For layer 0 (first_k_dense_replace=1): standard SwiGLU with gate_proj, up_proj, down_proj. These are f16 backbone weights, not 4-bit quantized. Use BLAS sgemv.

- [ ] **Step 4: Wire shared expert into MoE FFN path**

For layers 1-26: run routed experts (existing Metal path) + shared expert (BLAS sgemv on f16 weights or Metal 4-bit). Sum outputs.

- [ ] **Step 5: Write `generate_v2()` — full generation function**

Embed → 27 layers → final norm → LM head → sample → repeat.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: DeepSeek-V2 generation loop with MLA + shared experts + dense FFN"
```

**Gate:** Compiles. No runtime test yet (next task).

---

## Task 7: Generate HF Reference + E2E Test

**Files:**
- Create: `scripts/generate_deepseek_v2_ref.py` (one-time Python script for reference generation)
- Create: `crates/moe-infer/tests/test_v2_lite_generation.rs`

**What it does:** Generate golden reference outputs from HF transformers for V2-Lite, then test our Rust implementation against them. This is the acceptance test.

- [ ] **Step 1: Write Python reference generator**

```python
# One-time script to generate reference tensors
# Run: python scripts/generate_deepseek_v2_ref.py
# Outputs: weights/references/deepseek_v2_lite_greedy.json
# Contains: prompt, generated_ids (20 tokens, greedy)
```

- [ ] **Step 2: Run the reference generator**

```bash
python3 scripts/generate_deepseek_v2_ref.py
```

- [ ] **Step 3: Write E2E generation test**

```rust
#[test]
#[ignore = "requires converted weights + tokenizer"]
fn test_v2_lite_generation_matches_hf() {
    // Load model, generate 20 tokens greedy, compare against reference
    // Target: ≥15/20 token match (looser than Qwen3 because of f16 precision)
}
```

- [ ] **Step 4: Run, debug, iterate until passing**

```bash
cargo test -p moe-infer --test test_v2_lite_generation --release -- --ignored --nocapture
```

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: DeepSeek-V2-Lite E2E generation — X/20 HF match"
```

**Gate:** ≥15/20 HF greedy match. MLA attention producing correct output through all 27 layers.

---

## Task 8: Q LoRA + Sigmoid Routing (V3 Prep, Unit Tests Only)

**Files:**
- Modify: `crates/moe-infer/src/mla_attention.rs`
- Modify: `crates/moe-router/src/lib.rs`
- Create: `crates/moe-infer/tests/test_q_lora.rs`
- Create: `crates/moe-router/tests/test_sigmoid_v3.rs`

**What it does:** Add the V3-only features that V2-Lite doesn't exercise. Validated with unit tests on synthetic data — no model weights needed.

- [ ] **Step 1: Implement Q LoRA path in mla_forward_decode**

When `q_a_proj` is Some: q_latent = x @ W_qa^T → RMSNorm → q = q_latent @ W_qb^T

- [ ] **Step 2: Unit test Q LoRA with synthetic weights**

Small dims (hidden=64, q_lora_rank=16). Verify: two-stage LoRA matches direct projection composed.

- [ ] **Step 3: Add `route_sigmoid_v3()` to moe-router**

DeepSeek-V3 routing: sigmoid scores + e_score_correction_bias + grouped top-k (8 groups of 32, select top-4 groups, then top-8 from 128 candidates).

```rust
pub fn route_sigmoid_v3(
    &mut self,
    gate_logits: &[f32],
    bias: &[f32],           // e_score_correction_bias [num_experts]
    n_group: usize,         // 8
    topk_group: usize,      // 4
    scaling_factor: f32,    // 2.5
) -> RouteResult { ... }
```

- [ ] **Step 4: Unit test sigmoid routing**

Test: correct experts selected, weights use unbiased scores, scaling factor applied.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: Q LoRA + sigmoid grouped routing (V3 prep, unit tested)"
```

**Gate:** All unit tests pass. Ready for V3 weights when download completes.

---

## Task 9: Benchmark + Commit Final State

**Files:**
- Create: `crates/moe-infer/tests/bench_v2_lite_tok_per_sec.rs`

**What it does:** Benchmark DeepSeek-V2-Lite decode performance. Record baseline tok/s. Verify Qwen3-MoE-30B not regressed.

- [ ] **Step 1: Write V2-Lite benchmark**

- [ ] **Step 2: Run both benchmarks**

```bash
# V2-Lite
cargo test -p moe-infer --test bench_v2_lite_tok_per_sec --release -- --ignored --nocapture
# Qwen3 (no regression)
cargo test -p moe-infer --test bench_tok_per_sec --release -- --ignored --nocapture
```

- [ ] **Step 3: Log results to experiments-infer.tsv**

- [ ] **Step 4: Final commit**

```bash
git commit -m "feat: DeepSeek-V2-Lite MLA inference — X tok/s, Y/20 HF match"
```

**Gate:** V2-Lite generates correct tokens. Qwen3 not regressed. Baseline tok/s recorded.

---

## Verification

**After every task:**
```bash
cargo build -p moe-infer --release              # must compile
cargo test -p moe-infer --release                # no regressions
```

**After Task 7 (E2E):**
```bash
cargo test -p moe-infer --test test_v2_lite_generation --release -- --ignored  # HF match
cargo test -p moe-infer --test test_generation --release -- --ignored          # Qwen3 still works
```

**Exit gate:**
```bash
cargo test -p moe-infer --release                                              # all pass
cargo test -p moe-infer --test test_v2_lite_generation --release -- --ignored  # V2-Lite HF match
cargo test -p moe-infer --test test_generation --release -- --ignored          # Qwen3 HF match
cargo test -p moe-infer --test bench_v2_lite_tok_per_sec --release -- --ignored # V2-Lite benchmark
```
