# Sharded FFN Forward Optimization

## Problem

`FFN_SHARDS=4` forward: 911ms vs 259ms baseline (3.5×). Per-layer: FFN wall ~51ms, ANE ~7.6ms (15%), **w2 staging ~8ms/shard** (dominant). Root cause: `stage_transposed_weight_columns` ([L1393](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L1393)) does element-by-element gather across 24MB `weights.w2`.

## Phases

| Phase | Goal | Per-layer FFN (est.) |
| --- | --- | --- |
| A | Cached W2 transpose for sharded staging | ~51ms → ~23ms |
| B | Parallelize forward shard loop | ~23ms → ~8ms |

Phase A first — eliminates cache-hostile staging before parallelization adds contention.

---

## Phase A — Use `cache.w2t_scratch` for sharded W2 staging

The non-sharded path already caches `vdsp::mtrans(weights.w2)` into per-layer `cache.w2t_scratch` ([L4152](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L4152)), keyed by `w2t_generation` (init `u64::MAX` at [L122](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L122)). The sharded path ignores this.

#### [MODIFY] [layer.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs)

In `run_sharded_ffn_forward_into` (L1542), before shard loop:

```rust
if cache.w2t_generation != weights.w2_generation {
    vdsp::mtrans(&weights.w2, hidden, &mut cache.w2t_scratch, dim, dim, hidden);
    cache.w2t_generation = weights.w2_generation;
}
```

Replace L1611–1620 per shard:
```rust
// BEFORE (~8ms): stage_transposed_weight_columns(buf, &weights.w2, ...)
// AFTER (~1ms):
let w2t_start = layout.col_start * dim;
let w2t_shard = &cache.w2t_scratch[w2t_start..w2t_start + shard_hidden * dim];
stage_spatial(buf, shard_hidden, w2_sp, w2t_shard, dim, seq);
```

### Phase A verification

> [!IMPORTANT]
> This is a **pure staging change** — same floats written to the same IOSurface positions, just via `stage_spatial` instead of `stage_transposed_weight_columns`. Outputs must be **bit-exact** between old and new staging within the sharded path.

**Test 1 (bit-exact):** sharded-with-cached-transpose vs `#[doc(hidden)]` serial reference (pre-optimization staging). `to_bits()` on `x_next`, `cache.h1/h3/gate`.

**Test 2 (toleranced):** sharded vs non-sharded `forward_into`. All fields. Max abs ≤ 2e-2, cosine ≥ 0.999.

Serial reference: `#[doc(hidden)] pub` wrapper matching `forward_into_with_training_ffn` ([L4232](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L4232)) signature, delegating to the pre-optimization serial loop internally. Integration tests call this directly.

---

## Phase B — Parallelize forward shard loop

After Phase A, each shard is ~4ms of cache-local work. Pattern: `thread::scope` + barrier per [bench_ffn_latency_parallel_full_model.rs L743](file:///Users/USER/RustRover-Projects/rustane/crates/engine/tests/bench_ffn_latency_parallel_full_model.rs#L743).

#### [MODIFY] [layer.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs)

```rust
fn run_sharded_ffn_forward_into(...) {
    // Pre-loop: transpose W2 if stale (Phase A)
    // ...
    let x2norm_ref: &[f32] = &cache.x2norm;
    let w2t_ref: &[f32] = &cache.w2t_scratch;
    thread::scope(|scope| {
        let barrier = Arc::new(Barrier::new(shard_count + 1));
        for (worker, layout) in workers.iter_mut().zip(layouts.iter()) {
            scope.spawn(move || {
                barrier.wait();
                // stage W13 → run W13 → read h1/h3 → silu_gate
                // stage W2 (from w2t_ref rows) → run W2
            });
        }
        barrier.wait();
    });
    // Post-join: merge ffn_out, store h1/h3/gate (deterministic order)
}
```

**Thread safety:** `x2norm_ref`, `w2t_ref`, `weights.w1/w3` — immutable shared. Worker IOSurfaces/buffers — worker-local. Cache/ffn_out — post-join only.

### Phase B verification

**Test 1 (bit-exact):** parallel vs serial reference. **Test 2 (toleranced):** sharded vs baseline. **Test 3 (idempotency).** All via `#[doc(hidden)]` whole-layer wrapper.

---

## Expected end state

Per-layer non-FFN floor is ~6ms (measured). Model-level overhead (embedding, final norm, logits, cross-entropy) adds ~20ms.

| Config | Before | After A (est.) | After A+B (est.) |
| --- | --- | --- | --- |
| Per-layer FFN | ~51ms | ~23ms | ~8ms |
| Per-layer total | ~57ms | ~29ms | ~14ms |
| **FFN_SHARDS=4 fwd** | **911ms** | **~600ms** | **~300ms** |
| Baseline fwd | 259ms | 259ms | 259ms |

> [!NOTE]
> ~300ms is the realistic floor, not ~180ms. 20 layers × ~6ms non-FFN = 120ms, plus 20 × ~8ms FFN = 160ms, plus ~20ms model overhead = 300ms. The gap vs baseline (259ms) reflects per-shard dispatch overhead and the merge step, which are hard to eliminate entirely.
