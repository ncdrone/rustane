# M3 Ultra (512GB) Test Report — 2026-03-19

**Hardware:** Mac Studio M3 Ultra, 32-core CPU, 80-core GPU, 32-core Neural Engine, 512GB RAM
**OS:** macOS 26.3.1 (Tahoe) build 25D2128
**Rust:** 1.94.0
**Tester:** Jack Zampolin (amygdala project)

## Summary

6 of 8 ANE kernels compile and run correctly on M3 Ultra. The `sdpaFwd` kernel fails at ANE compilation due to `concat` MIL ops that M3 Ultra's ANE rejects. This blocks training but is fixable.

## Test Results

### Phase 0 — ANE Basics

| Test | Result | Notes |
|------|--------|-------|
| compile_graph_with_multiple_constants_on_ane | PASS | `(1+1)*2 = 4.0` correct |
| iosurface_raii_lock_guards_work | PASS | |
| iosurface_write_read_roundtrip | FAIL | fp16 precision: expected 0.01, got 0.010002136. Strict assert, not a real problem. |

### Phase 3 — Kernel Compilation

| Kernel | Compile | Run | Notes |
|--------|---------|-----|-------|
| DualDynMatmul | PASS | PASS | |
| woFwd | PASS | PASS | output sum=73927144.00 |
| ffnFused | PASS | PASS | out_ch=6912, sum=3756007.75 |
| ffnBwdW2t | PASS | PASS | output sum=197037440.00 |
| sdpaBwd1 | PASS | PASS | in_ch=3072, out_ch=6912, sum=5479.27 |
| sdpaBwd2 | PASS | PASS | in_ch=7680, out_ch=1536 |
| **sdpaFwd** | **FAIL** | — | `_ANECompiler : ANECCompile() FAILED` |
| all_forward_1000_iters | FAIL | — | Depends on sdpaFwd |

### Phase 4 — Training

| Test | Result | Notes |
|------|--------|-------|
| forward_produces_valid_loss | FAIL | sdpaFwd compile failure |
| six_layer_loss_decreases | FAIL | sdpaFwd compile failure |

## Root Cause

`sdpaFwd` uses `concat` MIL op in 3 places (`crates/engine/src/kernels/sdpa_fwd.rs`):

1. **Line 112** — RoPE rotation for Q: `g.concat(&[nq, q_e], 3)` (axis=width)
2. **Line 124** — RoPE rotation for K: `g.concat(&[nk, k_e], 3)` (axis=width)
3. **Line 158** — Output assembly: `g.concat(&[af, qrf, krf, vf, xn], 1)` (axis=channels)

**M4 Max accepts `concat` in MIL graphs. M3 Ultra does not.** This is a known ANE constraint ("concat op banned, must use multi-output programs") documented in `mechramc/Orion/docs/ane_constraints.md` — M4 appears to have relaxed this constraint.

## Suggested Fixes

### RoPE concat (lines 112, 124) — arithmetic replacement

The RoPE rotation `[-x_odd, x_even]` concat can be replaced with pure arithmetic:

```
// Instead of: concat([-odd, even], axis=3) → reshape
// Use: x * interleave_signs + rotate_left(x) * interleave_mask
// Or split the rotation into multiply + negate + add without concat
```

Specifically: `rotated_half(x) = x_even * cos - x_odd * sin` for even indices and `x_odd * cos + x_even * sin` for odd indices. This can be expressed as element-wise multiply + add with pre-computed sign-flip tensors, avoiding concat entirely.

### Output concat (line 158) — multi-output split

Split `sdpaFwd` into 2-3 separate ANE programs:
1. **sdpaFwd_qkv** — QKV projection + RoPE → outputs Q_rope, K_rope, V (3 separate IOSurfaces)
2. **sdpaFwd_attn** — attention scores + softmax + V matmul → output attn_out

This matches Orion's approach: "Must use multi-output programs" for the concat constraint.

## Environment

```
Build: cargo build --release — 25.01s, clean compile
Data: 631M training tokens prepared (climbmix-400B, 10 shards, uint16)
      63M validation tokens (1 shard)
      token_bytes.bin (8192 entries)
All data ready at /Users/admin/rustane/data/ on Studio-2.
```

## What We Can Test Once sdpaFwd Is Fixed

The M3 Ultra with 512GB is the ideal machine for pushing scale:
- **Training at 10-20B** (estimated 200-340GB RAM)
- **Forward pass at 50-100B+** (M4 Max hit 30B at 130GB)
- **ANE perf comparison**: M3 Ultra has 31.6 TFLOPS ANE vs M4 Max ~19 TFLOPS — potentially faster per-kernel

Data prep is complete and waiting. Happy to re-run the moment a fix lands.
