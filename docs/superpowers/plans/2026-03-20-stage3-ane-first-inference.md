# Stage 3: ANE-First Inference — 0.4 → 25+ tok/s

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace CPU scalar GEMV with Metal GPU + Accelerate BLAS dispatch, add ANE prefill path, achieve 25-30 tok/s decode on Qwen3-MoE-30B. All changes preserve the 20/20 HF greedy token match.

**Architecture:** ANE prefill (seq ≥ 64) + Metal/CPU decode (seq=1). PMetal confirmed: seq=1 is too small for ANE dispatch overhead (0.095ms). Decode uses Metal GEMV for expert FFN (the bottleneck — 95% of time), Accelerate cblas_sgemv for attention projections (40x over scalar loops), and ANE for the LM head. Prefill later adds fused ANE graphs for attention + batched expert FFN.

**Tech Stack:** Rust (moe-infer, moe-kernels), Metal shaders (dequant GEMV), Accelerate BLAS (cblas_sgemv), ANE (ane-bridge, conv1x1 graphs for prefill)

**Key Research:** `rustane-research/moe-1T/stage2-v2-2026-03-20/stage2-FINAL.md` — complete dispatch tables, SRAM analysis, code sketches.

**Pre-execution gap analysis:** `rustane-research/moe-1T/pre-execution-gap-analysis-2026-03-20.md` — all 4 investigations resolved, zero blockers.

**Safety net:** `test_generation_matches_hf` (20/20 HF greedy match) must pass after EVERY task. The `--features dual-path` flag runs both CPU and Metal/ANE paths in parallel for validation.

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `crates/moe-infer/src/blas.rs` | Accelerate BLAS bindings (cblas_sgemv wrapper for decode attention) |
| `crates/moe-infer/tests/test_metal_decode.rs` | Metal GEMV decode path validation (vs CPU reference) |
| `crates/moe-kernels/src/attention_ane.rs` | ANE fused GQA prefill graph (Q/K/V + QK-norm + RoPE + SDPA) |
| `crates/moe-infer/tests/test_ane_prefill.rs` | ANE prefill validation (vs CPU reference) |

### Modified Files

| File | Change |
|------|--------|
| `crates/moe-infer/src/generate.rs` | Wire Metal GEMV for expert FFN, BLAS for attention, ANE prefill branch |
| `crates/moe-infer/src/attention.rs` | Replace `matvec_f16()` with `blas::sgemv_f16()` for decode path |
| `crates/moe-infer/src/lib.rs` | Add `pub mod blas;` |
| `crates/moe-infer/Cargo.toml` | Add `dual-path` feature flag |
| `crates/moe-kernels/src/dequant.rs` | Add `encode_into()` for batched dispatch, FMA kernel upgrade |
| `crates/moe-kernels/src/expert_ffn.rs` | Add `build_batched_gate_up_conv()`, `build_batched_down_conv_n2()` |
| `crates/moe-infer/src/bin/infer.rs` | Add `--metal` flag, timing breakdown output |

### Unchanged (but referenced)

| File | Why |
|------|-----|
| `crates/moe-infer/src/rmsnorm.rs` | Keep as-is — CPU is correct for tiny 2048-float norms |
| `crates/moe-infer/src/kv_cache.rs` | Keep as-is — CPU/RAM regardless of compute target |
| `crates/moe-infer/src/sampler.rs` | Keep as-is — sampling always CPU |
| `crates/moe-router/src/lib.rs` | Keep as-is — routing always CPU |
| `crates/moe-infer/tests/test_generation.rs` | THE acceptance gate — must pass 20/20 after every task |

---

## Task 1: Dual-Path Safety Net

**Files:**
- Modify: `crates/moe-infer/Cargo.toml`
- Modify: `crates/moe-infer/src/generate.rs`

- [ ] **Step 1: Add dual-path feature flag**

Add to `crates/moe-infer/Cargo.toml` under `[features]`:
```toml
[features]
default = []
mla = []
dual-path = []
```

- [ ] **Step 2: Add validation comparator to generate.rs**

Add at the top of `generate.rs`:
```rust
/// Compare accelerated output against CPU reference.
/// Only active with `--features dual-path`.
#[cfg(feature = "dual-path")]
fn validate_output(accel: &[f32], cpu: &[f32], layer: usize, op: &str) -> Vec<f32> {
    let max_diff = accel.iter().zip(cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    if max_diff > 1e-2 {
        eprintln!("WARN: L{layer} [{op}] max_diff={max_diff:.6} — using CPU fallback");
        cpu.to_vec()
    } else {
        accel.to_vec()
    }
}
```

- [ ] **Step 3: Verify dual-path compiles**

Run: `cargo build -p moe-infer --release --features dual-path`
Expected: Clean build, no errors.

- [ ] **Step 4: Commit**

```bash
git add crates/moe-infer/Cargo.toml crates/moe-infer/src/generate.rs
git commit -m "feat: add dual-path safety net for Metal/ANE validation"
```

---

## Task 2: Accelerate BLAS for Decode Attention

**Files:**
- Create: `crates/moe-infer/src/blas.rs`
- Modify: `crates/moe-infer/src/lib.rs`
- Modify: `crates/moe-infer/src/attention.rs`

**Context:** The current `matvec_f16()` is a scalar CPU loop doing f16→f64 conversion per element. Accelerate's `cblas_sgemv` uses AMX hardware — 40x faster. Pre-execution investigation confirmed: no Cargo.toml changes needed, `#[link]` attribute is sufficient.

- [ ] **Step 1: Write the failing test**

```rust
// In crates/moe-infer/src/blas.rs (inline test)
#[test]
fn sgemv_matches_naive() {
    let rows = 4096;
    let cols = 2048;
    let w: Vec<f32> = (0..rows*cols).map(|i| (i as f32 * 0.001).sin()).collect();
    let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01).cos()).collect();
    let mut blas_out = vec![0.0f32; rows];
    sgemv_f32(&w, &x, &mut blas_out, rows, cols);

    // Naive reference
    let mut naive_out = vec![0.0f32; rows];
    for i in 0..rows {
        let mut sum = 0.0f64;
        for j in 0..cols { sum += w[i*cols+j] as f64 * x[j] as f64; }
        naive_out[i] = sum as f32;
    }

    let max_diff = blas_out.iter().zip(naive_out.iter())
        .map(|(a,b)| (a-b).abs()).fold(0f32, f32::max);
    assert!(max_diff < 1e-3, "BLAS vs naive max_diff={max_diff}");
}
```

- [ ] **Step 2: Write blas.rs**

Create `crates/moe-infer/src/blas.rs` with the copy-paste-ready code from the pre-execution gap analysis (Investigation 1). Contains `sgemv_f32()` and `sgemv_f16()`.

- [ ] **Step 3: Add module to lib.rs**

Add `pub mod blas;` to `crates/moe-infer/src/lib.rs`.

- [ ] **Step 4: Run test**

Run: `cargo test -p moe-infer --lib --release -- blas::tests::sgemv_matches_naive`
Expected: PASS

- [ ] **Step 5: Replace matvec_f16 in attention.rs with blas::sgemv_f16**

In `crates/moe-infer/src/attention.rs`, replace the `matvec_f16` function body (keep the signature for the CPU reference under `#[cfg(feature = "dual-path")]`):

```rust
fn matvec_f16(w: &[f16], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_dim];
    crate::blas::sgemv_f16(w, x, &mut out, out_dim, in_dim);
    out
}
```

Also replace `matvec_f16` in `generate.rs` (the duplicate function for LM head / router gate).

- [ ] **Step 6: Run acceptance test**

Run: `cargo test -p moe-infer --test test_generation --release -- test_generate_one_token --ignored --nocapture`
Expected: PASS, noticeably faster than before (attention ~40x faster, but expert FFN still CPU).

- [ ] **Step 7: Commit**

```bash
git add crates/moe-infer/src/blas.rs crates/moe-infer/src/lib.rs \
  crates/moe-infer/src/attention.rs crates/moe-infer/src/generate.rs
git commit -m "perf: Accelerate BLAS for decode attention — 40x faster projections"
```

---

## Task 3: Wire Metal GEMV for Expert FFN

**Files:**
- Modify: `crates/moe-infer/src/generate.rs`
- Create: `crates/moe-infer/tests/test_metal_decode.rs`

**Context:** This is THE highest-ROI change. The existing `MetalDequantGemv` in `moe-kernels/src/dequant.rs` is already compiled and tested (matches CPU <1e-6). We just need to call it from `generate.rs` instead of the CPU `dequant_expert_ffn()`.

- [ ] **Step 1: Write the failing test**

```rust
// crates/moe-infer/tests/test_metal_decode.rs
#[test]
fn metal_expert_matches_cpu() {
    // Load one expert's packed weights from the expert file
    // Run CPU dequant_expert_ffn and Metal metal_gemv_expert
    // Assert max_diff < 1e-3
}
```

- [ ] **Step 2: Add `parse_packed_weights()` helper to generate.rs**

This converts a flat byte slice (from the expert mmap) into a `PackedWeights4Bit` struct that `MetalDequantGemv::gemv()` can consume:

```rust
fn parse_packed_weights(
    data: &[u8], out_features: usize, in_features: usize, group_size: usize,
) -> PackedWeights4Bit {
    let packed_u32s = out_features * in_features / 8;
    let packed_bytes = packed_u32s * 4;
    let num_groups = out_features * (in_features / group_size);

    let data_u32 = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u32, packed_u32s)
    }.to_vec();
    let scales = unsafe {
        std::slice::from_raw_parts(data[packed_bytes..].as_ptr() as *const f16, num_groups)
    }.to_vec();
    let zeros = unsafe {
        std::slice::from_raw_parts(
            data[packed_bytes + num_groups * 2..].as_ptr() as *const f16, num_groups
        )
    }.to_vec();

    PackedWeights4Bit { data: data_u32, scales, zeros, out_features, in_features, group_size }
}
```

- [ ] **Step 3: Add `metal_gemv_expert()` wrapper**

```rust
fn metal_gemv_expert(
    metal: &moe_kernels::dequant::MetalDequantGemv,
    data: &[u8], x: &[f32], hidden: usize, inter: usize, group_size: usize,
) -> Vec<f32> {
    let matrix_packed = inter * hidden / 2;
    let num_groups = inter * (hidden / group_size);
    let scales_size = num_groups * 2;
    let matrix_total = matrix_packed + scales_size * 2;

    let gate = parse_packed_weights(&data[0..matrix_total], inter, hidden, group_size);
    let up = parse_packed_weights(&data[matrix_total..2*matrix_total], inter, hidden, group_size);
    let down = parse_packed_weights(&data[2*matrix_total..3*matrix_total], hidden, inter, group_size);

    let gate_out = metal.gemv(&gate, x);
    let up_out = metal.gemv(&up, x);

    // SiLU(gate) * up
    let h: Vec<f32> = gate_out.iter().zip(up_out.iter())
        .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u).collect();

    metal.gemv(&down, &h)
}
```

- [ ] **Step 4: Add `MetalDequantGemv` to Model struct**

```rust
pub struct Model {
    pub weights: BackboneWeights,
    pub config: InferConfig,
    pub rope: RopeTables,
    pub gqa_config: GqaConfig,
    pub metal: Option<moe_kernels::dequant::MetalDequantGemv>,
}
```

Initialize in `Model::load()`:
```rust
let metal = moe_kernels::dequant::MetalDequantGemv::new();
if metal.is_some() { println!("Metal GPU: enabled"); }
```

- [ ] **Step 5: Replace CPU expert dispatch in `moe_ffn()` with Metal**

Change `moe_ffn()` signature to accept `metal: Option<&MetalDequantGemv>`, then replace the inner loop:

```rust
let expert_out = if let Some(m) = metal {
    metal_gemv_expert(m, &expert_data[base..base + expert_stride], x, hidden, moe_inter, group_size)
} else {
    dequant_expert_ffn(&expert_data[base..base + expert_stride], x, hidden, moe_inter, group_size)
};
```

Update `run_layer()` to pass `model.metal.as_ref()` to `moe_ffn()`.

- [ ] **Step 6: Run acceptance test**

Run: `cargo test -p moe-infer --test test_generation --release -- test_generate_one_token --ignored --nocapture`
Expected: PASS, significantly faster (8-15 tok/s with current kernel).

- [ ] **Step 7: Run 20/20 HF match**

Run: `cargo test -p moe-infer --test test_generation --release -- test_generation_matches_hf --ignored --nocapture`
Expected: PASS — 20/20 tokens match. **This is the critical gate.**

- [ ] **Step 8: Run tok/s benchmark**

Run: `cargo test -p moe-infer --test bench_tok_per_sec --release -- --ignored --nocapture`
Expected: 8-15 tok/s (up from 0.4). Print the number.

- [ ] **Step 9: Commit**

```bash
git add crates/moe-infer/src/generate.rs crates/moe-infer/tests/test_metal_decode.rs
git commit -m "perf: Metal GEMV for expert FFN — 0.4 → 8-15 tok/s"
```

---

## Task 4: Metal Kernel FMA Upgrade

**Files:**
- Modify: `crates/moe-kernels/src/dequant.rs` (shader string)

**Context:** The current shader does `fma(fma(nibble, scale, zero), x, sum)` — a dependent FMA chain. The flash-moe pattern pre-computes `scale*x` and `zero*x` per thread group, breaking the dependency chain. This takes the kernel from 38 GiB/s to 350+ GiB/s — a 9x improvement.

- [ ] **Step 1: Replace the `process_u32` function in the shader**

Replace the current dependent-chain FMA with the pre-factored version. Also add threadgroup x-caching:

```metal
// BEFORE (in DEQUANT_GEMV_SHADER):
sum = fma(fma(n0, scale, zero), xp[0], sum);

// AFTER:
float sx = scale * xp[0];
float bx = zero * xp[0];
sum += fma(float((pack) & 0xF), sx, bx);
```

Full replacement of the inner loop: pre-load x into threadgroup shared memory, compute `scale*x[col]` and `zero*x[col]` per group, use nibble as scalar multiplier.

- [ ] **Step 2: Run Metal vs CPU validation test**

Run: `cargo test -p moe-infer --test integ_metal_dequant --release`
Expected: PASS — Metal still matches CPU within 1e-3.

- [ ] **Step 3: Run 20/20 HF match**

Run: `cargo test -p moe-infer --test test_generation --release -- test_generation_matches_hf --ignored --nocapture`
Expected: PASS — tokens unchanged.

- [ ] **Step 4: Run tok/s benchmark**

Run: `cargo test -p moe-infer --test bench_tok_per_sec --release -- --ignored --nocapture`
Expected: **25-40 tok/s**. This is the target.

- [ ] **Step 5: Commit**

```bash
git add crates/moe-kernels/src/dequant.rs
git commit -m "perf: FMA kernel upgrade — 38 → 350+ GiB/s, 25-40 tok/s"
```

---

## Task 5: Batched Metal Dispatch

**Files:**
- Modify: `crates/moe-kernels/src/dequant.rs` (add `encode_into()`)
- Modify: `crates/moe-infer/src/generate.rs` (batch expert dispatches)

**Context:** Current Metal dispatch creates a new command buffer per expert GEMV call — 24 Metal round-trips per layer (3 matrices × 8 experts). Batching into 2 command buffer commits (gate+up, then down) reduces this to 2 round-trips per layer. Expected: +15-20% throughput.

- [ ] **Step 1: Add `encode_into()` to MetalDequantGemv**

```rust
/// Encode a GEMV dispatch into an existing command encoder (no commit).
/// Caller manages command buffer lifecycle for batching.
pub fn encode_into(
    &self,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    weights: &PackedWeights4Bit,
    x: &[f32],
    y_buf: &ProtocolObject<dyn MTLBuffer>,
) {
    let (packed_buf, scales_buf, zeros_buf, x_buf, _, in_feat_buf, group_buf) =
        self.create_buffers(weights, x, &vec![0.0; weights.out_features]);
    self.encode_dispatch(
        enc, &packed_buf, &scales_buf, &zeros_buf,
        &x_buf, y_buf, &in_feat_buf, &group_buf,
        weights.out_features,
    );
}
```

- [ ] **Step 2: Batch expert dispatches in moe_ffn()**

Replace the per-expert `metal.gemv()` calls with batched dispatch:

```rust
// Create one command buffer for all gate+up GEMVs
let cmd = metal.queue.commandBuffer().unwrap();
let enc = cmd.computeCommandEncoder().unwrap();
for &eid in &route.expert_ids {
    metal.encode_into(&enc, &gate_packed[eid], x, &gate_bufs[eid]);
    metal.encode_into(&enc, &up_packed[eid], x, &up_bufs[eid]);
}
enc.endEncoding();
cmd.commit();
cmd.waitUntilCompleted();
// ... SiLU on CPU (tiny) ...
// Second cmd_buf for all down GEMVs
```

- [ ] **Step 3: Run 20/20 HF match**

Run: `cargo test -p moe-infer --test test_generation --release -- test_generation_matches_hf --ignored --nocapture`
Expected: PASS.

- [ ] **Step 4: Run tok/s benchmark**

Expected: +15-20% over Task 4 baseline.

- [ ] **Step 5: Commit**

```bash
git add crates/moe-kernels/src/dequant.rs crates/moe-infer/src/generate.rs
git commit -m "perf: batched Metal dispatch — 2 commits per layer instead of 24"
```

---

## Task 6: ANE Prefill Attention

**Files:**
- Create: `crates/moe-kernels/src/attention_ane.rs`
- Modify: `crates/moe-kernels/src/lib.rs`
- Modify: `crates/moe-infer/src/generate.rs` (prefill branch)
- Create: `crates/moe-infer/tests/test_ane_prefill.rs`

**Context:** For prefill (processing the prompt), seq ≥ 64. The fused GQA attention graph on ANE has 53+ ops → 94% utilization → 2-5ms per layer. This is 7.5x faster than CPU prefill (which processes tokens sequentially). SRAM analysis: fits at seq ≤ 256 (16 MB working set, below 32 MB limit).

- [ ] **Step 1: Write the failing test**

```rust
// crates/moe-infer/tests/test_ane_prefill.rs
#[test]
fn ane_prefill_attention_matches_cpu() {
    // Process 64 tokens through CPU gqa_forward (sequential)
    // Process same 64 tokens through ANE build_gqa_prefill_graph
    // Assert outputs match within 1e-2 (fp16 tolerance)
}
```

- [ ] **Step 2: Implement `build_gqa_prefill_graph()`**

In `crates/moe-kernels/src/attention_ane.rs`, build the fused GQA attention graph following the spec from `stage2-wave2-gqa-ane-graph.md`:
- QKV projections via `matrix_multiplication`
- QK-norm via reshape + `reduce_mean` + `pow(-0.5)` (NOT rsqrt — ANE compiler bug)
- RoPE via slice-rotate-concat (neox-style, blobfile cos/sin tables)
- GQA broadcast via `tile(k, [1, 8, 1, 1])`
- SDPA: QK^T + scale + causal mask + softmax + @V
- Output: concat(attn_out, K, V) for KV cache write-back

- [ ] **Step 3: Add prefill branch to generate.rs**

In the `generate()` function, add a prefill path: if processing multiple input tokens and ANE is available, use the fused graph. Otherwise fall back to sequential CPU.

- [ ] **Step 4: Run test**

Run: `cargo test -p moe-infer --test test_ane_prefill --release -- --ignored --nocapture`
Expected: PASS — ANE output matches CPU within fp16 tolerance.

- [ ] **Step 5: Run 20/20 HF match**

Run: `cargo test -p moe-infer --test test_generation --release -- test_generation_matches_hf --ignored --nocapture`
Expected: PASS — tokens unchanged (decode path is the same, only prefill is faster).

- [ ] **Step 6: Commit**

```bash
git add crates/moe-kernels/src/attention_ane.rs crates/moe-kernels/src/lib.rs \
  crates/moe-infer/src/generate.rs crates/moe-infer/tests/test_ane_prefill.rs
git commit -m "feat: ANE prefill attention — fused GQA graph, TTFT 7x faster"
```

---

## Task 7: Batched ANE Expert FFN for Prefill

**Files:**
- Modify: `crates/moe-kernels/src/expert_ffn.rs`
- Modify: `crates/moe-infer/src/generate.rs` (prefill expert path)

**Context:** For prefill, expert FFN can run on ANE via batched conv1x1 graphs. One dispatch for all 8 experts' gate+up (6.2 MB working set, below 32 MB SRAM limit). Four dispatches for down (n=2 pairs, 25.6 MB each). Total: 5 dispatches per layer instead of 16.

- [ ] **Step 1: Add `build_batched_gate_up_conv()`**

Follow the spec from `stage2-wave2-batched-expert-graph.md`:
- Input: `[1, 2048, 1, seq + 2*8*768]` = all 8 experts' W1+W3 packed in spatial dim
- Output: `[1, 6144, 1, seq]` = all experts' SiLU(gate) * up interleaved
- Cap at seq=128 (SRAM analysis: fits at 6.2 MB)

- [ ] **Step 2: Add `build_batched_down_conv_n2()`**

- Input: `[1, 1536, 1, seq + 2*2048]` = 2 experts' W2 packed
- Output: `[1, 4096, 1, seq]` = 2 experts' down outputs
- 4 dispatches for all 8 experts

- [ ] **Step 3: Wire into prefill path in generate.rs**

After routing, stage top-8 expert weights to IOSurface, dispatch batched gate_up (1 call), then 4× batched down.

- [ ] **Step 4: Run 20/20 HF match**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/moe-kernels/src/expert_ffn.rs crates/moe-infer/src/generate.rs
git commit -m "feat: batched ANE expert FFN for prefill — 5 dispatches per layer"
```

---

## Task 8: CLI Timing Breakdown + Final Benchmark

**Files:**
- Modify: `crates/moe-infer/src/bin/infer.rs`
- Modify: `crates/moe-infer/src/generate.rs` (add timing instrumentation)

- [ ] **Step 1: Add per-component timing to generate()**

Track: prefill_ms, attention_ms, expert_ms, lm_head_ms, total_ms. Print breakdown after generation.

- [ ] **Step 2: Update CLI**

```
cargo run -p moe-infer --release --bin infer -- \
  --config configs/qwen3-moe-30b.toml --prompt "What is 2+2?"
```

Output should show:
```
Model: Qwen3-MoE-30B-A3B (Metal GPU + ANE)
Generated 50 tokens in 1.8s = 27.8 tok/s
  Prefill: 1.2s (ANE fused)
  Decode: 0.6s
    Attention: 12% (BLAS cblas_sgemv)
    Expert FFN: 82% (Metal GEMV batched)
    Other: 6%
```

- [ ] **Step 3: Run full benchmark**

Run: `cargo test -p moe-infer --test bench_tok_per_sec --release -- --ignored --nocapture`
Log result to `system/experiments-infer.tsv`.

- [ ] **Step 4: Run ALL acceptance tests**

```bash
cargo test -p moe-infer --release -- --ignored
cargo test -p moe-router --release
cargo test -p quantize --release
cargo test -p expert-pager --release
```

Expected: Zero failures.

- [ ] **Step 5: Commit**

```bash
git commit -m "Stage 3 complete: ANE prefill + Metal decode, XX tok/s"
```

---

## Exit Gate

Stage 3 is complete when ALL of these pass:

```bash
# Correctness
cargo test -p moe-infer --test test_generation --release -- --ignored       # 20/20 HF match
cargo test -p moe-infer --test test_metal_decode --release -- --ignored     # Metal vs CPU
cargo test -p moe-infer --test test_ane_prefill --release -- --ignored      # ANE vs CPU

# Performance
cargo test -p moe-infer --test bench_tok_per_sec --release -- --ignored --nocapture  # ≥ 25 tok/s

# No regressions
cargo test -p moe-infer --release
cargo test -p moe-router --release
cargo test -p moe-kernels --release
```

**Target:** 25-30 tok/s decode. If Metal kernel upgrade alone gets 25+, the ANE prefill tasks (6-7) improve TTFT but don't change decode throughput.

---

## Dispatch Summary

| Operation | Before (Stage 2) | After (Stage 3) | Speedup |
|-----------|------------------|-----------------|---------|
| Attention projections | CPU scalar loop | Accelerate cblas_sgemv | ~40x |
| Expert FFN (decode) | CPU dequant loop | Metal GEMV (FMA, batched) | ~60-100x |
| Expert FFN (prefill) | CPU sequential | ANE batched conv1x1 | ~10x |
| LM head | CPU scalar loop | ANE pre-staged conv1x1 | ~10x |
| **Total decode** | **0.4 tok/s** | **25-40 tok/s** | **60-100x** |
