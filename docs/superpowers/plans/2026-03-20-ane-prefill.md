# ANE Prefill for Qwen3-MoE-30B

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace sequential token-by-token CPU attention during prefill with a fused ANE graph that processes all prompt tokens in one dispatch per layer. Expert FFN stays on Metal GPU (different tokens route to different experts — ANE can't batch that). Expected: prefill 5-10x faster (0.8s → 0.1-0.2s for 13 tokens).

**Architecture:** For each layer during prefill, the ANE graph processes all seq tokens' Q/K/V projections + QK-norm + RoPE + causal SDPA + O_proj in a single dispatch (~53 ops, 94% ANE utilization). K/V outputs are written to the CPU KV cache. Expert FFN then runs per-token on Metal as before.

**Tech Stack:** Rust (moe-kernels, moe-infer), ane-bridge Graph API, IOSurface staging, existing Metal GEMV for expert FFN.

**Existing reference:** `crates/engine/src/kernels/sdpa_fwd.rs` — working ANE attention graph for training. Must be adapted for: (1) neox-style RoPE, (2) QK-norm, (3) GQA 8:1 ratio, (4) Qwen3 dims.

**Safety net:** `test_generation_matches_hf` (20/20 HF greedy match) must pass after every task.

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `crates/moe-kernels/src/gqa_prefill.rs` | ANE fused GQA attention graph builder |
| `crates/moe-infer/tests/test_ane_prefill.rs` | ANE vs CPU attention correctness test |

### Modified Files

| File | Change |
|------|--------|
| `crates/moe-kernels/src/lib.rs` | Add `pub mod gqa_prefill;` |
| `crates/moe-infer/src/generate.rs` | Add prefill branch: ANE attention + Metal expert FFN |
| `crates/moe-infer/src/attention.rs` | Export `gqa_forward_f32` signature details for CPU reference |

### Unchanged (but referenced)

| File | Why |
|------|-----|
| `crates/engine/src/kernels/sdpa_fwd.rs` | Reference ANE attention pattern (RoPE, matmul, softmax, causal mask) |
| `crates/moe-kernels/src/expert_ffn.rs` | Reference dynamic conv1x1 pattern |
| `crates/moe-infer/src/kv_cache.rs` | KV cache write-back target |

---

## Critical Dimension Facts

| Parameter | Value | Notes |
|-----------|-------|-------|
| hidden_size | 2048 | Input/output dim |
| q_dim | 4096 | 32 heads × 128 head_dim |
| kv_dim | 512 | 4 heads × 128 head_dim |
| head_dim | 128 | Per-head dimension |
| num_q_heads | 32 | Q heads |
| num_kv_heads | 4 | KV heads (GQA 8:1) |
| gqa_group | 8 | Q heads per KV head |
| rope_theta | 1e6 | Neox-style RoPE |
| rms_norm_eps | 1e-6 | For QK-norm |
| max_prefill_seq | 256 | SRAM cap (see analysis) |

### SRAM Analysis (seq=256)

| Working set | Size |
|-------------|------|
| Q [32, 256, 128] fp16 | 2.0 MB |
| K [4, 256, 128] fp16 | 0.25 MB |
| V [4, 256, 128] fp16 | 0.25 MB |
| QK^T [32, 256, 256] fp16 | 4.0 MB |
| Attn_out [32, 256, 128] fp16 | 2.0 MB |
| Causal mask [256, 256] fp16 | 0.13 MB |
| RoPE tables [256, 64] × 2 fp16 | 0.06 MB |
| **Total** | **~8.7 MB** |
| **SRAM limit** | **32 MB** |

Fits comfortably. Cap at seq ≤ 256.

### GQA Broadcasting Strategy

No native `tile` op on ANE. Use slice + concat to expand KV heads:
```
For each kv_head h (0..3):
  k_h = slice(K, channels=h)           # [1, 1, seq, hd]
  k_h_x8 = concat([k_h] × 8, axis=1)  # [1, 8, seq, hd]
K_expanded = concat([k0_x8, ..., k3_x8], axis=1)  # [1, 32, seq, hd]
```
Same for V. Total: 4 slices + 4 concat(8) + 1 concat(4) = 9 ops per K/V.

### Neox RoPE (vs engine's interleaved)

Engine sdpa_fwd uses interleaved pairs: `(x[2i], x[2i+1])`.
Qwen3 uses neox: `(x[i], x[i + half])` where half = head_dim/2 = 64.

```
# Neox rotation:
x_first  = x[..., :64]     # first half
x_second = x[..., 64:]     # second half
rotated  = concat(-x_second, x_first, axis=-1)
result   = x * cos + rotated * sin
```

On ANE (4D layout [1, heads, seq, hd]):
```
x_first  = slice(x, width_start=0, width_size=64)
x_second = slice(x, width_start=64, width_size=64)
neg_second = multiply(x_second, scalar(-1))
rotated = concat([neg_second, x_first], axis=3)  # width axis
```

### QK-Norm (per-head RMSNorm)

Q shape after projection: [1, 32, seq, 128]. Need RMSNorm per head:
```
q_sq = multiply(q, q)                    # [1, 32, seq, 128]
q_mean = reduce_mean(q_sq, axis=3)       # [1, 32, seq, 1]
q_mean_eps = addition(q_mean, eps_const) # add 1e-6 for numerical stability
q_rms = power(q_mean_eps, scalar(-0.5))  # [1, 32, seq, 1]  ← NOT rsqrt (ANE compiler bug)
q_normed = multiply(q, q_rms)            # [1, 32, seq, 128] (broadcasts)
q_scaled = multiply(q_normed, q_norm_w)  # apply learned scale (q_norm_w is a graph constant)
```
Same 6-op pattern for K (with 4 KV heads instead of 32). k_norm_w is also a graph constant.

**CRITICAL:** Use `pow(-0.5)`, NOT `rsqrt` — ANE compiler fails on rsqrt after reduce ops.
**CRITICAL:** Must add epsilon BEFORE pow(-0.5) — without it, zero-magnitude vectors produce inf in fp16.
**NOTE:** q_norm_w and k_norm_w are small (128 floats each) — bake as graph constants, not IOSurface inputs.

---

## Task 6: Build ANE GQA Prefill Graph

**Files:**
- Create: `crates/moe-kernels/src/gqa_prefill.rs`
- Modify: `crates/moe-kernels/src/lib.rs`
- Create: `crates/moe-infer/tests/test_ane_prefill.rs`

**Context:** Adapts the existing `engine/src/kernels/sdpa_fwd.rs` pattern for Qwen3-MoE-30B GQA inference. Key differences: neox RoPE, QK-norm, GQA 8:1 ratio, output K/V for cache.

**O_proj is NOT in the ANE graph** (same as sdpa_fwd.rs). O_proj has shape [q_dim=4096, hidden=2048] which doesn't fit the channels=hidden=2048 input IOSurface. O_proj is applied on CPU/BLAS after reading ANE output — fast since it's a pre-converted f32 matvec.

**IOSurface spatial width alignment:**
- Input: `seq + q_dim + kv_dim + kv_dim` = `seq + 5120`. Pad to next multiple of 16.
- Output: `seq`. **Must be ≥ 64** (ANE MIN_SPATIAL_WIDTH) and multiple of 16.
- **Minimum prefill seq: 64 tokens.** For prompts < 64 tokens, pad with zeros and ignore extra outputs. For prompts > 256, process in chunks of 256.
- Graph is pre-compiled at model load for fixed seq sizes: 64, 128, 256. Select the smallest that fits.

**GQA concat dedup safety:** The plan uses `concat([k_h, k_h, ...×8], axis=1)` with the same tensor handle. If ANE's MIL compiler rejects duplicate bottoms, use 8 separate identity slices (each `slice(k_h, [0,0,0,0], full_shape)`) as distinct handles before concat.

- [ ] **Step 1: Write the failing test**

Create `crates/moe-infer/tests/test_ane_prefill.rs`:

```rust
//! ANE prefill attention vs CPU reference.

use moe_kernels::gqa_prefill;

#[test]
fn ane_gqa_matches_cpu_seq64() {
    // seq=64 (minimum for ANE spatial width ≥ 64), hidden=2048
    // 32 Q heads, 4 KV heads, hd=128
    // 1. Generate random input [seq, hidden]
    // 2. Generate random Q/K/V projection weights (O_proj tested separately)
    // 3. Run CPU gqa_forward_f32 sequentially for seq tokens
    // 4. Run ANE build_gqa_prefill for all seq tokens at once
    // 5. Compare attention outputs: max_diff < 0.05 (fp16 tolerance)
    // NOTE: seq must be ≥ 64 for ANE MIN_SPATIAL_WIDTH constraint
    todo!("implement after graph builder exists")
}
```

Run: `cargo test -p moe-infer --test test_ane_prefill --release -- ane_gqa_matches_cpu_seq64`
Expected: FAIL — module doesn't exist yet.

- [ ] **Step 2: Add module to lib.rs**

Add to `crates/moe-kernels/src/lib.rs`:
```rust
pub mod gqa_prefill;
```

- [ ] **Step 3: Build the GQA prefill graph**

Create `crates/moe-kernels/src/gqa_prefill.rs`. The graph takes ONE IOSurface input containing:
- Activations: `[1, hidden, 1, seq]` (spatial dim = seq)
- Wq: `[1, hidden, 1, q_dim]` (spatial dim = q_dim=4096)
- Wk: `[1, hidden, 1, kv_dim]` (spatial dim = kv_dim=512)
- Wv: `[1, hidden, 1, kv_dim]` (spatial dim = kv_dim=512)
- **No Wo** (O_proj done on CPU after ANE dispatch — different channel dim)

Total input spatial width: `pad16(seq + q_dim + kv_dim + kv_dim)` = `pad16(seq + 5120)`
For seq=64: 5184 (÷16=324, ✓). For seq=128: 5248 (÷16=328, ✓). For seq=256: 5376 (÷16=336, ✓).

Output: `[1, q_dim + kv_dim + kv_dim, 1, seq]`
- Channels 0..q_dim = raw attention output `[32 heads × 128 hd]` (O_proj applied on CPU)
- Channels q_dim..q_dim+kv_dim = K after RoPE (for KV cache)
- Channels q_dim+kv_dim.. = V (for KV cache)

Output seq must be ≥ 64 and multiple of 16 (enforced by graph seq parameter).

The graph internally:
1. QKV projections via matmul (like sdpa_fwd)
2. Reshape to heads
3. QK-norm via reduce_mean + pow(-0.5) + multiply
4. Neox RoPE via slice + negate + concat + cos/sin multiply
5. GQA expand K/V via slice + concat (4→32 heads)
6. Attention: Q @ K^T × scale + causal_mask → softmax → @ V
7. O_proj via matmul
8. Concat output + K_rope + V for cache write-back

```rust
pub fn build_gqa_prefill(
    hidden: usize,      // 2048
    num_q_heads: usize,  // 32
    num_kv_heads: usize, // 4
    head_dim: usize,     // 128
    seq: usize,          // variable, max 256
    rope_theta: f32,     // 1e6
    eps: f32,            // 1e-6 for QK-norm
) -> Graph { ... }
```

See dimension facts and strategy sections above for implementation details.

- [ ] **Step 4: Add CPU reference for batched GQA**

Add `gqa_forward_batch_f32()` to `attention.rs` — processes all seq tokens at once with proper causal masking (not just sequential single-token calls). This is the ground truth for the ANE comparison.

- [ ] **Step 5: Implement the test**

Fill in `test_ane_prefill.rs` with:
1. Random weights and input
2. CPU reference via `gqa_forward_batch_f32`
3. ANE dispatch via `build_gqa_prefill` + IOSurface staging
4. Compare outputs within fp16 tolerance (max_diff < 0.05)

Run: `cargo test -p moe-infer --test test_ane_prefill --release -- ane_gqa_matches_cpu_seq64`
Expected: PASS

- [ ] **Step 6: Test at seq=128**

Add `ane_gqa_matches_cpu_seq128` test. Same structure, larger sequence.
Expected: PASS, max_diff < 0.05

- [ ] **Step 7: Commit**

```bash
git add crates/moe-kernels/src/gqa_prefill.rs crates/moe-kernels/src/lib.rs \
  crates/moe-infer/tests/test_ane_prefill.rs crates/moe-infer/src/attention.rs
git commit -m "feat: ANE GQA prefill graph — fused attention for batch tokens"
```

---

## Task 7: Wire ANE Prefill into Generation Pipeline

**Files:**
- Modify: `crates/moe-infer/src/generate.rs`

**Context:** Replace the sequential token-by-token prefill loop with: ANE attention (batch) → CPU routing → Metal expert FFN (per-token) → CPU residual. The expert FFN stays on Metal because different tokens route to different experts (can't batch on ANE).

- [ ] **Step 1: Add `prefill_layer_ane()` function**

This processes ONE layer for ALL prefill tokens:

```rust
fn prefill_layer_ane(
    model: &Model,
    cache: &mut KvCache,
    router: &mut MoeRouter,
    layer: usize,
    xs: &[Vec<f32>],      // [seq] of [hidden]
    positions: &[usize],  // [seq] position indices
) -> Result<Vec<Vec<f32>>> {
    // 1. RMSNorm all tokens (CPU, tiny)
    // 2. Stage all tokens + weights into IOSurface
    // 3. ANE dispatch: fused GQA attention for all tokens
    // 4. Extract attention outputs + K/V from ANE output
    // 5. Write K/V to cache for each position
    // 6. Residual add (CPU)
    // 7. RMSNorm (CPU)
    // 8. Per-token: MoE routing + Metal expert FFN (same as decode)
    // 9. Residual add (CPU)
    // Return: [seq] of [hidden]
}
```

- [ ] **Step 2: Add ANE graph cache to Model**

Pre-compile the ANE graph at model load time for a fixed max prefill seq length (e.g. 256). Store the compiled graph in Model struct.

```rust
pub struct Model {
    // ... existing fields ...
    pub ane_prefill_graph: Option<CompiledGraph>,  // Pre-compiled for max_prefill_seq
    pub max_prefill_seq: usize,
}
```

- [ ] **Step 3: Update generate() to use ANE prefill**

Replace the sequential prefill loop:

```rust
// OLD: sequential (one token at a time through all 48 layers)
for (i, &token_id) in input_ids.iter().enumerate() {
    for layer in 0..num_layers {
        x = run_layer(model, &mut cache, &mut router, layer, &x, i)?;
    }
}

// NEW: batched (all tokens through each layer together)
let mut xs: Vec<Vec<f32>> = input_ids.iter()
    .map(|&id| embed_f16_to_f32(embed_table, id as usize, hidden))
    .collect();
let positions: Vec<usize> = (0..input_ids.len()).collect();

if model.ane_prefill_graph.is_some() && input_ids.len() <= model.max_prefill_seq {
    // ANE prefill: all tokens through each layer
    for layer in 0..num_layers {
        xs = prefill_layer_ane(model, &mut cache, &mut router, layer, &xs, &positions)?;
    }
} else {
    // Fallback: sequential CPU (for seq > max_prefill_seq)
    // ... existing sequential loop ...
}
// Sample from last token's hidden state
```

- [ ] **Step 4: Run 20/20 HF match**

Run: `cargo test -p moe-infer --test test_generation --release -- test_generation_matches_hf --ignored --nocapture`
Expected: PASS — 20/20 tokens match. **Critical gate.**

- [ ] **Step 5: Run benchmark**

Run: `cargo test -p moe-infer --test bench_tok_per_sec --release -- --ignored --nocapture`
Expected: Prefill time significantly reduced (0.8s → 0.1-0.2s). Decode unchanged at ~14 tok/s.

- [ ] **Step 6: Commit**

```bash
git add crates/moe-infer/src/generate.rs
git commit -m "feat: ANE prefill pipeline — batched attention, per-token Metal FFN"
```

---

## Why Expert FFN Stays on Metal (not ANE) for Prefill

The original plan proposed batched ANE expert FFN for prefill. After investigation, this is impractical because:

1. **Variable routing:** Each token routes to different top-8 experts. For seq=64, that's potentially 64 × 8 = 512 different expert dispatches across dozens of unique experts.

2. **4-bit quantized weights:** Expert weights are 4-bit packed. ANE conv1x1 requires f32/f16 weights. Dequantizing all potentially-needed experts per layer adds significant CPU overhead.

3. **Scatter-gather pattern:** Tokens must be grouped by which expert they route to, processed, then results scattered back. This doesn't map well to a single ANE graph.

4. **Metal is already fast:** Expert FFN on Metal takes ~0.8ms/layer for decode (seq=1). For prefill, each token still goes through Metal sequentially — the overhead per layer is 8 × 0.8ms = ~6.4ms for seq=8 tokens. With 48 layers, that's ~300ms total for prefill expert FFN.

5. **The win is in attention:** Prefill attention is the bottleneck that scales with seq (O(seq²) per layer). Going from 13 sequential layer-passes to 1 batched pass is the 13x improvement. Expert FFN is O(seq) and already fast on Metal.

**Future optimization:** If expert FFN becomes the prefill bottleneck (longer prompts), we can revisit ANE batching by grouping tokens by expert assignment and dequantizing on-demand. But that's a separate, more complex project.

---

## Verification

**After every task:** `cargo test -p moe-infer --test test_generation --release -- test_generation_matches_hf --ignored --nocapture` → 20/20 match

**Exit gate:**
```bash
cargo test -p moe-infer --test test_generation --release -- --ignored       # 20/20 HF match
cargo test -p moe-infer --test test_ane_prefill --release                   # ANE vs CPU
cargo test -p moe-infer --test bench_tok_per_sec --release -- --ignored --nocapture  # timing
cargo test -p moe-infer --release                                           # no regressions
```

---

## Key Files (read before executing)

| File | What to look for |
|------|-----------------|
| `crates/engine/src/kernels/sdpa_fwd.rs` | **THE reference** — working ANE attention graph. Copy the matmul+RoPE+softmax pattern, adapt for GQA/neox/QK-norm |
| `crates/moe-kernels/src/expert_ffn.rs` | Dynamic conv1x1 pattern, IOSurface staging approach |
| `crates/moe-infer/src/attention.rs` | CPU GQA forward — the ground truth to match |
| `crates/moe-infer/src/generate.rs` | Current prefill loop to replace |
| `crates/moe-infer/src/kv_cache.rs` | KV cache store/get API |
| `CLAUDE.md` | ANE gotchas: rsqrt fails after reduce, spatial width must be multiple of 16 |
