# Sharded Training Backward Timing Harness + Parallelization

Sharding regressions from `sweep-600m` show backward time ballooning from 710ms (baseline) to 1174ms (ATTN_SHARDS=2), 1978ms (FFN_SHARDS=4), and 2258ms (both). This plan instruments the sharded backward paths to isolate the bottleneck, then parallelizes the serial shard loops.

## Phased approach

| Phase | Goal | Deliverable | Touches training path? |
| --- | --- | --- | --- |
| 1 | Timing harness | New benchmark test | No |
| 2 | FFN backward parallelization | Production code change | Yes |
| 3 | Re-measure | Harness re-run + data | No |
| 4 | Attention backward parallelization | Production code change | Yes |

> [!IMPORTANT]
> Phase 2 should not start until Phase 1 data confirms the serial loop is the dominant contributor. Phase 4 should not start until Phase 3 data confirms FFN backward regression collapsed.

---

## Phase 1 — Sharded backward timing harness

### Timing structs

#### [MODIFY] [layer.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs)

All timing structs derive `Serialize, Deserialize` (following [bench_ffn_latency_parallel_full_model.rs L175](file:///Users/USER/RustRover-Projects/rustane/crates/engine/tests/bench_ffn_latency_parallel_full_model.rs#L175) pattern) for JSON output:

```rust
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DispatchTiming {
    pub kernel_name: String,        // "w2", "w13t", "wot", etc. (String for serde)
    pub staging_ms: f32,
    pub ane_wall_ms: f32,
    pub ane_hw_ns: Option<u64>,     // None in WallOnly mode
    pub readback_ms: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShardTiming {
    pub shard_idx: usize,
    pub dispatches: Vec<DispatchTiming>,
    pub cpu_grad_accum_ms: f32,
    pub cpu_silu_ms: f32,
    pub shard_total_ms: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShardedBackwardTimings {
    pub ffn_scale_dy_ms: f32,
    pub ffn_shards: Vec<ShardTiming>,
    pub ffn_grad_scatter_ms: f32,
    pub ffn_dx_merge_ms: f32,
    pub ffn_wall_ms: f32,
    pub ffn_sum_ane_wall_ms: f32,
    pub ffn_sum_cpu_ms: f32,
    pub ffn_sum_ane_hw_ms: Option<f32>,
    pub attn_transpose_ms: f32,
    pub attn_shards: Vec<ShardTiming>,
    pub attn_grad_scatter_ms: f32,
    pub attn_dx_merge_ms: f32,
    pub attn_wall_ms: f32,
    pub attn_sum_ane_wall_ms: f32,
    pub attn_sum_cpu_ms: f32,
    pub attn_sum_ane_hw_ms: Option<f32>,
    pub host_overhead_ms: Option<f32>,
}
```

### Dual-mode timing API

```rust
#[derive(Debug, Clone, Copy)]
pub enum TimingMode { WallOnly, WallAndHw }

pub fn backward_into_sharded_timed(
    cfg: &ModelConfig, kernels: &CompiledKernels, weights: &LayerWeights,
    cache: &ForwardCache, dy: &[f32], grads: &mut LayerGrads,
    ws: &mut BackwardWorkspace, dx_out: &mut [f32],
    sharded_attn: Option<&mut ShardedAttentionBackwardRuntime>,
    sharded_ffn: Option<&mut ShardedFfnBackwardRuntime>,
    mode: TimingMode,
) -> ShardedBackwardTimings { ... }
```

### Benchmark test

#### [NEW] [bench_sharded_backward_timing.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/tests/bench_sharded_backward_timing.rs)

`#[ignore]`d. 600M config. Explicit runtime construction (no env vars). 2 warmup + 5 timed.

> [!IMPORTANT]
> Each timed step calls the wrapper twice (`WallOnly` + `WallAndHw`). Each invocation must use **freshly zeroed** `LayerGrads` and `dx_out` buffers to avoid the second call running on mutated state from the first. Pre-allocate two sets of gradient/output buffers and `fill(0.0)` before each call, or allocate `LayerGrads::zeros()` per call.

JSON to `results/sharded_backward_timing/`.

```bash
cargo test -p engine --test bench_sharded_backward_timing --release -- --ignored --nocapture
```

---

## Phase 2 — Parallelize FFN backward shard loop

#### [MODIFY] [layer.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs)

**`ShardedFfnBackwardWorker` additions (L724):**

```rust
dw1_local: Vec<f32>,    // [dim * shard_hidden]
dw2_local: Vec<f32>,    // [dim * shard_hidden]
dw3_local: Vec<f32>,    // [dim * shard_hidden]
```

> [!CAUTION]
> `accumulate_dw` uses `beta=1.0`. Worker-local buffers must be zeroed before each backward call.

**Serial reference preservation:**

> [!IMPORTANT]
> `#[cfg(test)]` items in library crates are NOT visible to integration tests under `tests/` — those are separate compilation units that only see the public API. Use `#[doc(hidden)] pub` instead: always compiled, hidden from rustdoc, accessible from integration tests.

```rust
/// Serial reference for correctness tests. Identical to the pre-parallelization
/// loop. Not intended for production use — #[doc(hidden)] keeps it out of public docs.
#[doc(hidden)]
pub fn backward_into_sharded_serial(
    cfg: &ModelConfig, kernels: &CompiledKernels, weights: &LayerWeights,
    cache: &ForwardCache, dy: &[f32], grads: &mut LayerGrads,
    ws: &mut BackwardWorkspace, dx_out: &mut [f32],
    sharded_attn: Option<&mut ShardedAttentionBackwardRuntime>,
    sharded_ffn: Option<&mut ShardedFfnBackwardRuntime>,
) { /* delegates to run_sharded_ffn_backward_into_serial internally */ }
```

The serial loop body (current L1655–1768) is extracted into a private `run_sharded_ffn_backward_into_serial` called by this wrapper.

**Parallel loop:** `thread::scope` + barrier, worker-local `dw*_local`, post-loop `scatter_dw_columns` + `dx_ffn vadd`. (See v3 for pseudocode.)

### Phase 2 verification

#### [NEW] [test_sharded_ffn_backward_correctness.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/tests/test_sharded_ffn_backward_correctness.rs)

Non-`#[ignore]`d. Explicit runtime construction (no env vars).

**Test 1 — Bit-exact: parallel vs serial reference.** Calls production parallel path and `backward_into_sharded_serial`. `to_bits()` on all 9 gradient fields + `dx`. Schedule-only change → bit-exact.

**Test 2 — Toleranced: sharded vs non-sharded.** `backward_into_with_training_ffn(Some(...))` vs `backward_into(None)`. All 9 fields. Max abs ≤ 2e-2, cosine ≥ 0.999.

**Test 3 — Bit-exact idempotency.** Two runs of parallel path, fresh `LayerGrads` each → `to_bits()` identical.

> [!NOTE]
> No accumulation test. `scatter_dw_columns` ([L1499](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L1499)) is `copy_from_slice` (overwrites, not additive). Multi-call accumulation for sharded training is out of scope.

Sweep gate: `make sweep-600m FFN_SHARDS=4` + `make sweep-600m`.

---

## Phase 3 — Re-measure

Re-run Phase 1 harness. Key comparison:

| Metric | Phase 1 (serial) | Phase 3 (parallel) | Interpretation |
| --- | --- | --- | --- |
| `ffn_wall_ms` | ≈ sum(shards) | should drop | main success metric |
| `ffn_grad_scatter_ms` | small | may grow | new bottleneck? |
| `sum(ane_wall) / ffn_wall` | ≈ 1.0 | > 1.0 | overlap ratio |

---

## Phase 4 — Parallelize attention backward shard loop

Same pattern as Phase 2.

**Worker-local buffers:**
- `dwq_local` `[dim × q_dim_shard]` — column-sharded, scatter via `scatter_dw_columns`
- `dwk_local` `[dim × kv_dim_shard]` — column-sharded
- `dwv_local` `[dim × kv_dim_shard]` — column-sharded
- `dwo_local` `[q_dim_shard × dim]` — **row-sharded**, needs `scatter_dw_rows`

> [!IMPORTANT]
> Current sharded attention backward (L1792–2010) **never computes `dwo`**. Must add per-worker computation + `scatter_dw_rows`.

Serial reference: `#[doc(hidden)] pub backward_into_sharded_attn_serial`. Weight transposes before thread spawn. All buffers zeroed each call.

Same test structure: bit-exact parallel-vs-serial, toleranced sharded-vs-baseline, idempotency. Sweep: `make sweep-600m ATTN_SHARDS=2`.

---

## Data collection

| Config | Phase 0 | Phase 1 | Phase 2 | Phase 4 |
| --- | --- | --- | --- | --- |
| Baseline | 1046 ms/step, bwd=710 | — | — | — |
| ATTN_SHARDS=2 | 1541 ms/step, bwd=1174 | — | — | — |
| FFN_SHARDS=4 | 2993 ms/step, bwd=1978 | — | — | — |
| Both | 3297 ms/step, bwd=2258 | — | — | — |
