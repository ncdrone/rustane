# DeepSeek-V3 (671B) Execution on M4 Max

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get DeepSeek-V3 (671B, MLA attention, 256 MoE experts) generating correct tokens on M4 Max 128GB, streaming experts from SSD at 3-5 tok/s steady state.

**Architecture:** FP8 weights (641 GB) → converted to INT4 experts (358 GB on SSD) + f16 backbone (22 GB in RAM). Expert pool with Least-Stale eviction pages experts from NVMe via parallel pread. Same MLA attention path as V2-Lite — Q LoRA and sigmoid routing now exercised with real weights.

**Tech Stack:** Rust, Accelerate BLAS (f32 sgemv/sgemm), Metal (4-bit GEMV), safetensors, memmap2, half (f16), Rayon (parallel FP8 conversion + W_UK absorption).

**Research source:** `rustane-research/mla-1t/stage2-deepseekv3-execution-2026-03-21-1246/`

**Prerequisite:** Stage 1 complete — MLA pipeline validated on V2-Lite (15.7B). Branch: `rustane-infer`.

---

## Current State

- V2-Lite: full MLA pipeline, 39 tests, 4-level validation (L1-L3 pass)
- V3 weights: downloaded at `weights/deepseek-v3/` (641 GB, 163 FP8 shards)
- Expert pager crate: built (`crates/expert-pager/`), not wired into generation loop
- Q LoRA + sigmoid routing: implemented, unit tested on synthetic data
- Config system: supports MLA fields, YaRN, dense/MoE layers

---

## Critical Corrections from Stage 2 Research

These MUST be applied — getting any wrong causes silent bugs or wrong eviction:

1. **first_k_dense_replace = 3** — V3 has THREE dense layers (0, 1, 2), not just 1
2. **Least-Stale eviction, NOT LRU** — evict by minimum `last_used_layer` (SpecMD, Apple 2026)
3. **L-2 prefetcher REMOVED** — tested -18% performance, 25% hit rate. Trust OS page cache.
4. **FP8 dequant: LUT[256] → f32 × scale_inv[row/128][col/128]** — block-wise, not per-tensor
5. **e_score_correction_bias** is per-layer [256] f32 — frozen bias for sigmoid routing
6. **`topk_method` and `num_nextn_predict_layers`** — NO effect on inference, ignore
7. **shared_expert_count = 1** (V3) not 2 (V2-Lite) — shared expert intermediate = 2048, not 4096
8. **norm_topk_prob = true** (V3) — routing weights normalized before `routed_scaling_factor` applied
9. **routed_scaling_factor scales expert weights, NOT combined output** — current generate_v2.rs is wrong (was invisible at scale=1.0 for V2-Lite, breaks at 2.5 for V3)
10. **Expert file naming: `layer_{layer:02}_experts.bin`** — 2-digit format, consistent with V2-Lite converter and weight loader

## Memory Budget (128 GB)

```
macOS + Metal overhead:   ~7 GB
Backbone (f16, mmap'd):  ~22 GB
KV cache (4K ctx, f32):  ~0.3 GB
Expert pool (INT4):      ~96 GB  (4,314 experts @ 22.3 MB each)
Headroom:                ~2.7 GB
```

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `crates/moe-infer/src/bin/convert_v3.rs` | FP8→INT4 converter with Rayon parallelism |
| `crates/moe-infer/src/fp8.rs` | FP8 e4m3fn dequant (LUT + block-scale) |
| `configs/deepseek-v3.toml` | V3 inference config |
| `scripts/generate_v3_ref.py` | Partial-load HF reference generator (~200 MB) |
| `scripts/v3_api_reference.py` | DeepSeek API reference for L3 behavioral test |
| `crates/moe-infer/tests/test_v3_validation.rs` | V3 4-level validation suite |
| `crates/moe-infer/tests/test_fp8_dequant.rs` | FP8 conversion unit tests |

### Modified Files

| File | Change |
|------|--------|
| `crates/expert-pager/src/pool.rs` | Replace LRU with Least-Stale eviction |
| `crates/expert-pager/src/prefetch.rs` | Remove L-2 prefetcher (causes -18%) |
| `crates/moe-infer/src/generate_v2.rs` | Wire expert pager, support V3 loading |
| `crates/moe-infer/src/config.rs` | Add `e_score_correction_bias` field, `topk_group` |
| `crates/moe-infer/src/weights.rs` | Add `load_with_layers` V3 path, bias loading |
| `crates/moe-infer/src/mla_attention.rs` | Rayon parallel W_UK sgemv |
| `crates/moe-infer/Cargo.toml` | Add `rayon`, `ml-dtypes` (for FP8) |

## Task Dependencies

```
T1 (Config) ──────────────────────────────┐
T2 (FP8 dequant) ──→ T3 (Converter) ─────┤
T4 (Expert pager) ────────────────────────┼──→ T5 (Wiring) ──→ T7 (Validation) ──→ T8 (Benchmark)
T6 (Python refs) ─────────────────────────┘
```

Tasks 1, 2, 4, 6 can run in parallel. Task 5 is the integration point. Task 6 reads original FP8 safetensors directly (doesn't need converter output).

---

## Task 1: V3 Config + TOML

**Files:**
- Create: `configs/deepseek-v3.toml`
- Modify: `crates/moe-infer/src/config.rs`

**What it does:** Extend config for V3-specific fields. Verify against config.json.

- [ ] **Step 1: Add V3 fields to config.rs**

```rust
// In FfnSection:
#[serde(default)]
pub topk_group: usize,          // 4 (V3), 1 (V2-Lite)
#[serde(default)]
pub n_group: usize,             // 8 (V3), 1 (V2-Lite)

// In ModelSection:
#[serde(default)]
pub first_k_dense_replace: Option<usize>,  // 3 for V3 (override ffn.first_k_dense_replace)
```

- [ ] **Step 2: Write `configs/deepseek-v3.toml`**

All parameters from V3 config.json. Key differences from V2-Lite: hidden=7168, layers=61, q_lora_rank=1536, 256 experts, sigmoid scoring, routed_scaling_factor=2.5, first_k_dense_replace=3, mscale=1.0, bos=0, eos=1, shared_expert_count=1 (not 2), norm_topk_prob=true, n_group=8, topk_group=4.

- [ ] **Step 3: Write test**

```rust
#[test]
fn parse_deepseek_v3_toml() {
    let config = InferConfig::from_toml(&ws_root.join("configs/deepseek-v3.toml")).unwrap();
    assert_eq!(config.model.hidden_size, 7168);
    assert_eq!(config.model.num_layers, 61);
    assert!(config.is_mla());
    assert_eq!(config.attention.q_lora_rank, Some(1536));
    assert!(!config.is_moe_layer(0));  // dense
    assert!(!config.is_moe_layer(2));  // dense
    assert!(config.is_moe_layer(3));   // MoE starts here
    assert_eq!(config.ffn.scoring_func, "sigmoid");
    assert_eq!(config.ffn.routed_scaling_factor, 2.5);
}
```

- [ ] **Step 4: Run tests (V3 + V2-Lite + Qwen3 all parse), commit**

**Gate:** All three TOMLs parse. `is_moe_layer(2)` returns false for V3.

---

## Task 2: FP8 Dequantization Module

**Files:**
- Create: `crates/moe-infer/src/fp8.rs`
- Create: `crates/moe-infer/tests/test_fp8_dequant.rs`

**What it does:** FP8 e4m3fn byte → f32 conversion with block-wise scale lookup. This is the foundation for the weight converter.

- [ ] **Step 1: Build FP8 LUT**

```rust
/// Precomputed LUT: fp8_e4m3fn byte → f32 value.
/// 256 entries covering all possible byte values.
/// e4m3fn: 1 sign + 4 exponent + 3 mantissa, no infinities, NaN = 0x7F
fn build_fp8_e4m3fn_lut() -> [f32; 256] { ... }
```

- [ ] **Step 2: Block-wise dequant function**

```rust
/// Dequantize FP8 tensor with block-wise scales.
/// weight: [M, K] as raw bytes (1 byte per element)
/// scale_inv: [ceil(M/128), ceil(K/128)] as f32
/// Returns: [M, K] as f32
pub fn dequant_fp8_block(
    weight: &[u8], scale_inv: &[f32],
    m: usize, k: usize, block_size: usize,
) -> Vec<f32> { ... }
```

- [ ] **Step 3: Unit tests against known values**

Test with hand-computed FP8 bytes. Verify: `fp8_byte=0x38 (1.0 in e4m3fn) × scale_inv=2.0 → 2.0`.

- [ ] **Step 4: Cross-validate against Python ml_dtypes**

```python
import ml_dtypes, numpy as np
val = np.array([0x38], dtype=np.uint8).view(ml_dtypes.float8_e4m3fn)
print(float(val[0]))  # should be 1.0
```

- [ ] **Step 5: Run tests, commit**

**Gate:** LUT matches ml_dtypes for all 256 byte values. Block dequant handles edge cases (partial blocks at matrix edges).

---

## Task 3: FP8→INT4 Weight Converter

**Files:**
- Create: `crates/moe-infer/src/bin/convert_v3.rs`
- Modify: `crates/moe-infer/Cargo.toml` (add `rayon`)

**What it does:** Convert 641 GB of FP8 safetensors → backbone.bin (22 GB f16) + per-layer expert files (358 GB INT4). Rayon parallelism at expert level.

- [ ] **Step 1: Backbone conversion**

Read all non-expert tensors (embedding, MLA weights, dense FFN layers 0-2, shared experts, routers, norms, LM head). FP8 → f32 → f16 for projections, keep f32 for norms. Pre-split kv_b_proj → W_UK + W_UV. Write backbone.bin + backbone_index.json.

- [ ] **Step 2: Expert conversion with Rayon**

Per MoE layer (3-60): read 256 experts from safetensors, FP8 dequant → INT4 quantize, write to `layer_{NN}_experts.bin`. Use Rayon `par_iter` over experts within each layer.

- [ ] **Step 3: Dense FFN layers 0-2**

These are FP8 too (gate/up/down with intermediate=18432). Convert to f16 and store in backbone.bin. Same FP8 dequant → f16 path as MLA projections.

- [ ] **Step 4: e_score_correction_bias extraction**

Per MoE layer: extract `mlp.gate.e_score_correction_bias` [256] f32 tensor. Store in backbone.bin alongside router gate weights.

- [ ] **Step 5: Run converter on 2 layers first**

```bash
cargo run -p moe-infer --release --bin convert_v3 -- \
  --model-dir weights/deepseek-v3 --output-dir weights/rustane-v3 --max-layers 5
```

- [ ] **Step 6: Spot-check converted tensors, commit**

- [ ] **Step 7: Run full conversion (all 61 layers)**

```bash
cargo run -p moe-infer --release --bin convert_v3 -- \
  --model-dir weights/deepseek-v3 --output-dir weights/rustane-v3
```

Expected: ~30-45 minutes with Rayon (research estimate), ~380 GB output.

**Gate:** backbone.bin exists (~22 GB), 58 expert files exist (~6 GB each), spot-checked tensors match FP8 originals.

---

## Task 4: Expert Pager — Least-Stale Eviction

**Files:**
- Modify: `crates/expert-pager/src/pool.rs`
- Modify: `crates/expert-pager/src/prefetch.rs`

**What it does:** Replace LRU with Least-Stale eviction policy. Remove L-2 prefetcher.

- [ ] **Step 1: Replace LRU with Least-Stale in pool.rs**

Change eviction from `min_by_key(|e| e.last_used)` (LRU clock) to `min_by_key(|e| e.last_used_layer)` (layer index). Key changes to `(layer_idx, expert_idx)` tuple.

- [ ] **Step 2: Remove L-2 prefetcher**

Delete or disable the `ExpertPrefetcher` — research showed -18% performance, 25% hit rate. Trust the OS page cache instead.

- [ ] **Step 3: Unit tests for Least-Stale**

```rust
#[test]
fn least_stale_evicts_lowest_layer() {
    let mut pool = ExpertPool::new(2);
    pool.request(3, 10);   // layer 3, expert 10
    pool.request(59, 20);  // layer 59, expert 20
    pool.request(30, 5);   // layer 30, expert 5 — triggers eviction
    // Layer 3 should be evicted (lowest layer = furthest from reuse)
    assert!(!pool.is_resident(3, 10));
    assert!(pool.is_resident(59, 20));  // kept (highest layer)
    assert!(pool.is_resident(30, 5));   // just added
}
```

- [ ] **Step 4: Run tests, commit**

**Gate:** Least-Stale evicts lowest-layer expert. Old LRU tests updated or removed.

---

## Task 5: V3 Weight Loading + Generation Wiring

**Files:**
- Modify: `crates/moe-infer/src/weights.rs`
- Modify: `crates/moe-infer/src/generate_v2.rs`
- Modify: `crates/moe-infer/src/config.rs`

**What it does:** Load V3 backbone with Q LoRA weights, wire expert pager into the MoE FFN path, handle e_score_correction_bias.

- [ ] **Step 1: Add Q LoRA fields to `MlaLayerWeights` in weights.rs**

The zero-copy loader struct in `weights.rs` (lines 61-81) needs Q LoRA fields added:
```rust
pub q_a_proj: Option<&'a [f16]>,
pub q_a_layernorm: Option<&'a [f32]>,
pub q_b_proj: Option<&'a [f16]>,
```
Load these in `mla_layer_weights()` when `q_lora_rank` is Some. Note: the runtime struct in `mla_attention.rs` already has these fields — this closes the gap in the loader.

- [ ] **Step 2: Add e_score_correction_bias loading**

Load `mlp.gate.e_score_correction_bias` [256] f32 per MoE layer. Pass to `route_sigmoid_v3()` which already accepts bias + n_group + topk_group parameters.

- [ ] **Step 3: Fix routed_scaling_factor bug in generate_v2.rs**

**BUG:** Current code (lines 362-370) applies `routed_scaling_factor` to the entire `combined` output (shared + routed). This is wrong — it should scale only the routed expert weights BEFORE combining. Was invisible for V2-Lite (scale=1.0) but breaks V3 (scale=2.5).

Fix: move scaling to the per-expert weight multiplication:
```rust
// BEFORE (wrong): combined[d] += weight * down_results[i][d]; ... combined *= scale;
// AFTER (correct): combined[d] += (weight * scale) * down_results[i][d];
// Shared expert output added separately, unscaled.
```

- [ ] **Step 4: Wire expert pager into generate_v2**

Replace the mmap-all-experts approach with expert pool lookups:
```rust
// For each routed expert in this layer:
let (slot, is_hit) = expert_pool.request(layer, expert_id);
if !is_hit {
    expert_loader.load_expert(expert_id, &mut pool_buffers[slot]);
}
// Use pool_buffers[slot] for Metal dispatch
```

- [ ] **Step 5: Handle 3 dense FFN layers**

Update the `is_moe_layer()` check — layers 0, 1, 2 all use dense FFN with intermediate=18432.

- [ ] **Step 6: Run V2-Lite regression tests**

```bash
cargo test -p moe-infer --test test_model_validation --release -- --ignored --nocapture
```
Verify V2-Lite L1-L3 still pass after generate_v2 changes.

- [ ] **Step 7: Build + compile check, commit**

**Gate:** Compiles. V2-Lite validation still passes. Expert pager wired for V3, bypassed for V2-Lite (mmap path).

---

## Task 6: Python Reference Generation

**Files:**
- Create: `scripts/generate_v3_ref.py`
- Create: `scripts/v3_api_reference.py`

**What it does:** Generate golden reference outputs for V3 validation. Two approaches: partial safetensors load for L1/L2 (embedding + layer 0), DeepSeek API for L3 (logits).

- [ ] **Step 1: Write partial-load reference generator**

Load only ~200 MB of weights (embedding + layer 0 MLA + layer 0 dense FFN). Run forward pass for a single layer. Save intermediates to `weights/references/deepseek_v3_intermediates.npz`.

Includes FP8 dequant in Python (via ml_dtypes).

- [ ] **Step 2: Run it and verify shapes**

```bash
python3 scripts/generate_v3_ref.py --weights-dir weights/deepseek-v3 --output-dir weights/references/
```

- [ ] **Step 3: Write API reference script**

Hit DeepSeek API with `logprobs: true, top_logprobs: 20` for "The capital of France is". Save logits + greedy generation for L3 validation.

- [ ] **Step 4: Run API script, commit**

**Gate:** `deepseek_v3_intermediates.npz` exists with layer 0 tensors. API reference has top-20 logprobs.

---

## Task 7: V3 Validation Suite

**Files:**
- Create: `crates/moe-infer/tests/test_v3_validation.rs`

**What it does:** Same 4-level framework as V2-Lite, pointing at V3 config and references.

- [ ] **Step 1: Copy test_model_validation.rs pattern**

Update `model_config()` to point at V3 paths. Adjust thresholds (V3 has more layers = more accumulation).

- [ ] **Step 2: L1 — Embedding check**

Compare `embed(token) → layernorm` against partial-load reference. Threshold: cos > 0.999.

- [ ] **Step 3: L2 — Layer 0 output**

Run all prompt tokens through layer 0 (Q LoRA + dense FFN). Compare against reference. Threshold: cos > 0.99.

- [ ] **Step 4: L3 — Logit distribution (API reference)**

Full 61-layer forward pass. Compare argmax and top-5 overlap against DeepSeek API reference.

- [ ] **Step 5: Run suite, iterate, commit**

**Gate:** L1 and L2 pass. L3 reports metrics (may not match API exactly due to quantization).

---

## Task 8: End-to-End Generation + Benchmark

**Files:**
- Create: `crates/moe-infer/tests/bench_v3_tok_per_sec.rs`

**What it does:** Generate tokens, measure throughput, verify coherent output.

- [ ] **Step 1: Write benchmark test**

```rust
#[test]
#[ignore = "requires converted V3 weights"]
fn bench_v3_decode_throughput() {
    // Load model, generate 20 tokens greedy
    // Report: cold tok/s, warm tok/s, expert cache hit rate
    // Log to experiments-infer.tsv
}
```

- [ ] **Step 2: Run benchmark, record baseline**

Target: 3-5 tok/s warm, 0.3-0.9 cold.

- [ ] **Step 3: Add Rayon parallel W_UK absorption**

Replace 128× serial `sgemv_f32_trans` with Rayon `par_chunks`. Measure improvement.

- [ ] **Step 4: Tune expert pool capacity**

Try different pool sizes (500, 1000, 2000, 4000 experts). Measure hit rate vs memory.

- [ ] **Step 5: Log results, commit**

**Gate:** V3 generates coherent text. Baseline tok/s recorded. Qwen3 + V2-Lite no regressions.

---

## Verification

**After every task:**
```bash
cargo build -p moe-infer --release
cargo test -p moe-infer --lib --release     # 28+ lib tests
cargo test -p moe-router --lib --release    # 5 router tests
```

**After Task 3 (converter):**
```bash
cargo run -p moe-infer --release --bin convert_v3 -- \
  --model-dir weights/deepseek-v3 --output-dir weights/rustane-v3 --max-layers 5
# Verify: backbone.bin + 2 expert files exist
```

**After Task 7 (validation):**
```bash
cargo test -p moe-infer --test test_v3_validation --release -- --ignored --nocapture
# L1: cos > 0.999, L2: cos > 0.99
```

**After Task 8 (benchmark):**
```bash
cargo test -p moe-infer --test bench_v3_tok_per_sec --release -- --ignored --nocapture
# Target: 3-5 tok/s warm
```

**Exit gate:**
```bash
cargo test -p moe-infer --release                                           # all pass
cargo test -p moe-infer --test test_v3_validation --release -- --ignored     # L1+L2 pass
cargo test -p moe-infer --test test_generation --release -- --ignored        # Qwen3 still works
cargo test -p moe-infer --test test_model_validation --release -- --ignored  # V2-Lite still works
```
