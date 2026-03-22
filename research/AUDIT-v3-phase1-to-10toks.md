# Full Audit: V3 0.7 tok/s → 10 tok/s Gap Analysis

**Date:** 2026-03-22
**Auditors:** 5 parallel agents (compute, memory/IO, Metal GPU, code quality, architecture)
**Current:** 0.7 tok/s (22ms/layer, 1,340ms/token)
**Target:** 10 tok/s (1.6ms/layer, 100ms/token)
**Gap:** 14× speedup needed

---

## SHOWSTOPPER: Scalar Attention Loops (~10ms/layer wasted)

**The single biggest finding.** The attention score computation and value reconstruction in `mla_attention.rs` lines 237-296 use **triple-nested scalar loops with f64 accumulation**. No BLAS, no SIMD.

```rust
// Lines 247-258: SCALAR loop — 128 heads × seq_len × 512 multiply-adds in f64
for head in 0..h {
    for t in 0..seq_len {
        let mut dot_nope = 0.0f64;
        for d in 0..kv_rank { dot_nope += q_abs[d] as f64 * lat_t[d] as f64; }
```

At seq_len=100: 128 × 100 × 512 = 6.5M scalar f64 ops ≈ **~5ms**. The value reconstruction loop (lines 280-296) is equally slow: another **~5ms**.

**Fix:** Replace with `sgemm`:
- `scores_nope = q_absorbed[128, 512] @ latent_cache[100, 512]^T` → one sgemm call → **0.003ms**
- `v_latent = attn_weights[128, 100] @ latent_cache[100, 512]` → one sgemm call → **0.003ms**
- **Savings: ~10ms/layer → ~610ms/token → takes us from 0.7 to ~1.4 tok/s alone**

---

## CRITICAL BUGS (4 found)

| ID | Severity | Issue | File:Line |
|----|----------|-------|-----------|
| C1 | **UB** | `staging_ptr` creates `&mut [u8]` from `&self` (aliasing violation) | generate_v2.rs:761 |
| C2 | **OOB** | `wrap_mmap` pads beyond Vec allocation, Metal reads unowned memory | dequant.rs:647 |
| C3 | **UB** | `tensor_f16`/`tensor_f32` no alignment check on mmap pointers | weights.rs:149 |
| C4 | **Corruption** | `pread` short reads not handled (only checks < 0, not < expected) | loader.rs:45 |

**C1 is active UB in the current hot path** — the staging buffer aliasing is unsound and concurrent with rayon.

---

## Metal GPU: `x_cache[4096]` Too Small for V3

**Critical correctness bug in Metal shaders.** The fused gate+up kernel has `threadgroup float x_cache[4096]` but V3's `in_features = hidden_size = 7168`. The shader reads `x_cache[i]` for i up to 7167, but the array is only 4096 elements. **Out-of-bounds threadgroup memory access → garbage computation for expert gate/up projections.**

This means **V3 expert FFN results are currently wrong**. The model generates "Paris" despite this because the attention path is correct and the error is partially masked by the 8-expert averaging + residual connection.

---

## Architecture Gap: Where the 14× Must Come From

### Current Budget (22ms/layer)

| Component | ms/layer | % | Bound |
|-----------|----------|---|-------|
| f16→f32 conversion | 7.0 | 32% | Memory BW |
| Q LoRA sgemv (2x) | 0.5 | 2% | Memory BW |
| KV compress sgemv | 0.04 | 0% | Memory BW |
| **Attn scores (SCALAR)** | **5.0** | **23%** | **Compute (scalar f64)** |
| W_UK/W_UV absorption (128x serial sgemv) | 0.7 | 3% | Dispatch overhead |
| **Value recon (SCALAR)** | **5.0** | **23%** | **Compute (scalar f64)** |
| O projection sgemv | 1.2 | 5% | Memory BW |
| Shared FFN sgemv (3x) | 0.4 | 2% | Memory BW |
| Expert pread (8, parallel) | 3.0 | 14% | SSD/page cache |
| Metal dispatch | 3.0 | 14% | GPU + sync overhead |

### Target Budget (1.6ms/layer for 10 tok/s)

| Component | Target ms | Method |
|-----------|-----------|--------|
| Q LoRA | 0.24 | sgemv_f16 direct (no conversion) |
| KV compress | 0.02 | sgemv_f16 direct |
| Attn scores | 0.01 | **sgemm** (not scalar loops) |
| W_UK/W_UV | 0.16 | Single sgemm (batch 128 heads) |
| Value recon | 0.01 | **sgemm** |
| O projection | 0.59 | sgemv_f16 direct |
| Shared FFN | 0.22 | sgemv_f16 or Metal |
| Routed experts | 0.30 | In-memory pool (zero I/O) + Metal |
| **Total** | **~1.55** | |

### Staged Roadmap

| Stage | Change | Expected tok/s | Effort |
|-------|--------|----------------|--------|
| S1 | Replace scalar attn with sgemm | **1.4** | 1 hour |
| S2 | Switch to f16 decode path (`run_layer_f16`) | **3.0-3.5** | 30 min (code exists) |
| S3 | Batch W_UK/W_UV into sgemm | **4.0** | 1 hour |
| S4 | Integrate ExpertPool (in-memory, 3000+ experts) | **5.0-6.0** | 2 hours |
| S5 | Async GPU dispatch (remove waitUntilCompleted) | **6.0-7.0** | 2 hours |
| S6 | Fix x_cache[4096] bug, move shared FFN to Metal | **7.0-8.0** | 2 hours |
| S7 | Move backbone sgemv to Metal (unified GPU pipeline) | **8.0-10.0** | 4 hours |
| S8 | LM head optimization (f16 or INT8) | **10.0+** | 1 hour |

---

## Memory Architecture for 10 tok/s

| Component | GB | Notes |
|-----------|-----|-------|
| macOS + Metal | 10.0 | Fixed |
| Backbone mmap (f16, mlock'd) | 34.2 | mlock prevents eviction |
| Expert pool (in-RAM, 3300 experts) | 77.4 | 95% hit rate |
| KV cache (4K seq, MLA) | 0.6 | |
| Staging buffer | 0.2 | |
| **Total** | **~122.4** | Fits in 128 GB |

Key insight from memory agent: **77 GB is available for expert pool** if we skip f32 conversion buffers (use f16 path). 3,300 experts = 95% hit rate. At 95% hit, pread miss cost drops from 3ms/layer to ~0.15ms/layer.

---

## Code Quality Summary

| Severity | Count | Top Issues |
|----------|-------|------------|
| CRITICAL | 4 | staging_ptr UB, wrap_mmap OOB, mmap alignment, pread short reads |
| HIGH | 5 | Q LoRA fallthrough, final_norm.to_vec(), dead scaling_factor, hot-path allocs, rmsnorm allocs |
| MEDIUM | 5 | Unused ExpertPool import, dead f16 path, sgemv_f16 alloc, Sync justification, reserve logic |
| LOW | 3 | Magic numbers, duplicated stride calc, unnecessary clone |

### Hot-Path Allocations (Per Token)

| Allocation | Count | Total |
|------------|-------|-------|
| `rmsnorm` → Vec<f32> | 183 | ~5.3 MB |
| `v_latent` in attn loop | 7,808 | ~15 MB |
| `q`, `q_nope`, `q_pe`, etc. per layer | ~610 | ~37 MB |
| `final_norm.to_vec()` | 4 | 112 KB |
| Expert bufs (if using load_experts_parallel) | 464 | ~10.3 GB (!!) |
| **Total** | **~9,000** | **~57 MB + 10 GB transient** |

---

## Quick Wins (Do First)

1. **sgemm for attention scores + value recon** — 10ms/layer saved, ~1 hour work
2. **Switch to `run_layer_f16` in decode loop** — 7ms/layer saved, code exists, 30 min
3. **Fix `final_norm.to_vec()`** — remove 4 unnecessary copies per token, 5 min
4. **Fix staging_ptr UB** — use `UnsafeCell` or pass staging as `&mut`, 30 min
5. **Pre-allocate attn scratch buffers** — eliminate 9,000 allocs/token, 1 hour

---

## Physical Limits Assessment

At **546 GB/s** sustained memory bandwidth:
- Backbone f16 per token: 28.5 GB → 52ms (minimum, all operations fused)
- Expert INT4 per token: 10.3 GB → 19ms (100% cache hit)
- Total I/O floor: **71ms → max ~14 tok/s**

At **f32 weights** (current): 57 GB → 104ms → max 9.6 tok/s (already over budget)

**Conclusion: 10 tok/s requires f16 compute path.** The f32 conversion must be eliminated. The code for this (`run_layer_f16`, `mla_forward_decode_f16`) already exists but failed in Phase 1 due to the chunked sgemv overhead. The fix: use a properly vectorized f16→f32 conversion within sgemv (Neon intrinsics) or move to Metal f16 GEMV.
