# Sharding Optimization Audit — Scale-Aware (v2)

**Scope:** 600M–30B training, 122B forward-only

---

## OPT-1: Use existing `ws.w1t`/`ws.w3t` for sharded FFN backward [P0 — TRAINING]

**Where:** [run_sharded_ffn_backward_worker L2704–2723](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L2704)

**Problem:** Each backward shard calls `stage_transposed_weight_columns` for W1^T and W3^T — ~8ms/shard cache-hostile gather.

**Fix:** `BackwardWorkspace` already has `w1t: Vec<f32>` and `w3t: Vec<f32>` ([L1266–1267](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L1266)), allocated once, reused across layers. The non-sharded backward path already transposes into them. The sharded path just needs to:
1. Transpose `weights.w1` → `ws.w1t` and `weights.w3` → `ws.w3t` **before** `thread::scope` (same as attention does with `ws.wot/wqt/wkt/wvt` at [L3377–3380](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L3377))
2. Each worker stages from contiguous rows of `ws.w1t`/`ws.w3t` via `stage_spatial` instead of `stage_transposed_weight_columns`

**Memory cost:** Zero. `w1t`/`w3t` already exist. They're transient per-layer scratch, not per-layer cache — reused across all 96 layers at 30B.

**Estimated impact:** FFN_SHARDS=4 backward from ~1014ms to ~500ms at 600M.

---

## OPT-6: Productize sharded forward-only for 122B [P0 — INFERENCE]

**Where:** Main inference path in [full_model.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/full_model.rs) vs existing bench paths in [parallel_bench.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/parallel_bench.rs) and [bench_fwd_only_scale.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/tests/bench_fwd_only_scale.rs)

**Problem:** At 122B the non-sharded forward hits ANE spatial width limits. The sharded forward machinery already exists in bench form but isn't integrated into the main inference runtime.

**Fix:** Unify or reuse the sharded forward path from `parallel_bench.rs` in the main forward-only runtime. Not invention — productization.

---

## OPT-4: Intra-worker ANE/staging overlap [P1 — TRAINING]

**Where:** FFN backward workers — W2 dispatch can overlap with W13t staging

In FFN backward, W2 staging uses `dffn` and `weights.w2` (available before the loop). After dispatching W2, the worker waits for ANE, then starts W13t staging. These could overlap: dispatch W2 → stage W13t while ANE runs W2 → wait for W2 → dispatch W13t.

**Estimated saving:** ~2–4ms/shard, ~40–80ms total.

**Constraint:** Requires async ANE dispatch or intra-worker threading. Incremental — should not come before OPT-1.

---

## OPT-7: Attention forward sharding for 122B [P1 — INFERENCE]

Fused `sdpaFwd` kernel may hit limits at 122B head counts / widths. Sharding attention forward would enable clean scaling beyond single-kernel capacity.

---

## OPT-3: Fuse W13+SiLU kernel [P2]

Eliminates CPU round-trip between W13 and W2 dispatches in FFN forward. Compiler-risk project. ~50ms forward saving if viable.

---

## OPT-8: Remove redundant `cache.h1/h3/gate.fill(0.0)` [P3]

[L1751–1753](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L1751) — shards cover full column range, zeroing is redundant. ~0.5ms.

---

## Demoted / Rejected

| Opt | Status | Reason |
| --- | --- | --- |
| OPT-2 (prepack weight IOSurface) | Demoted (P3) | Runtime shared across layers — staged weights overwritten by next layer immediately. Per-layer runtimes = memory explosion at 30B. |
| OPT-5 (skip unchanged W1/W3 staging) | Demoted (P3) | Same shared-runtime issue as OPT-2. |

---

## Priority by target

**30B training:** OPT-1 → re-measure → OPT-4 → OPT-3  
**122B forward-only:** OPT-6 → OPT-7 → OPT-3
