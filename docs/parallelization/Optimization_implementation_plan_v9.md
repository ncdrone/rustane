# Sharding Optimization Plan v5

**Targets:** 30B training, 122B forward-only

## Workspace reuse contract

`ModelForwardWorkspace` and `ModelBackwardWorkspace` may be reused across calls with different shard settings in the same process. Rules:

1. **Stored runtimes are keyed by resolved mode.** `CombinedForwardRuntime` stores its `ResolvedForwardMode`; on each call, if the requested mode differs from the stored mode, the runtime is rebuilt (recompile + reallocate). Same for backward runtimes. This extends the existing shard-count check at [full_model.rs L505](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/full_model.rs#L505) to cover the full resolved mode, not just one count.
2. **Transient buffers are not semantically persistent.** `LayerScratch` and `BackwardWorkspace` temps are fully overwritten each call — no invalidation needed. `ForwardCache` has two categories:
   - **Transient activation fields** (`x`, `xnorm`, `rms_inv1`, `q_rope`, `k_rope`, `v`, `attn_out`, `o_out`, `x2`, `x2norm`, `rms_inv2`, `h1`, `h3`, `gate`): overwritten each forward call, no invalidation.
   - **Persistent per-layer caches** (`w2t_scratch`, `w2t_generation`): keyed by weight generation, intentionally survive across calls. Must NOT be cleared on mode change — they are valid as long as the underlying weights haven't changed.
3. **Timing semantics:** Steady-state timing excludes runtime construction/rebuild. Any mode change invalidates stored runtimes before warmup. Timed samples run with fixed resolved mode and reusable workspace.

---

## P0-A: FFN backward transpose reuse

Transpose `weights.w1/w3` into existing `ws.w1t/ws.w3t` ([L1266–1267](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L1266)) once before `thread::scope` in `run_sharded_ffn_backward_into` ([L2827](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L2827)). Workers stage shard rows via `stage_spatial` instead of `stage_transposed_weight_columns` ([L2704](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L2704)). Zero new memory.

**Verify:** bit-exact parallel vs serial reference, `make sweep-600m` all four configs.

---

## P0-B: Shared resolution, runtime extraction, production layer API

### P0-B.1: Shared resolver with caller-selected policy

```rust
pub enum ShardPolicy {
    FailFast,
    AutoAdjustNearest,
}

pub enum ResolvedForwardMode { Baseline, AttentionOnly{..}, FfnOnly{..}, AttentionFfn{..} }
pub enum ResolvedBackwardMode { Baseline, AttentionOnly{..}, FfnOnly{..}, AttentionFfn{..} }

pub struct ModeResolution {
    pub forward: ResolvedForwardMode,
    pub backward: ResolvedBackwardMode,
    pub adjustments: Vec<ModeAdjustment>,  // empty under FailFast
}

pub struct ModeAdjustment {
    pub axis: &'static str,        // e.g. "FFN_FWD_SHARDS"
    pub requested: usize,
    pub applied: usize,
    pub reason: String,
}

pub fn resolve_modes(
    cfg: &ModelConfig,
    requested: ShardRequest,
    policy: ShardPolicy,
) -> Result<ModeResolution, String>;
```

- Training/sweeps: `FailFast` — invalid shard counts are hard errors
- Ladder/ceiling: `AutoAdjustNearest` — rounds to nearest valid, records adjustment
- `ModeResolution.adjustments` carries structured metadata: bench uses it for labels, suffixes, and reporting (replaces current `requested_*`/`applied_*`/`notes` in [parallel_bench.rs L47](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/parallel_bench.rs#L47)). Training callers print adjustments to stdout.
- `parallel_bench.rs` delegates to `resolve_modes` with `AutoAdjustNearest`, reads `adjustments` for machine-readable output
- Env parsing is a thin wrapper over `resolve_modes`; not the resolver itself

### P0-B.2: `CombinedForwardRuntime`

> [!IMPORTANT]
> FFN forward: **wrap the existing production `ShardedFfnForwardRuntime`** ([layer.rs L715](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L715)), not re-extracted from bench. Only extract the missing **attention-forward** and **composition** pieces from bench code. Two FFN-forward implementations must not coexist.

`LayerScratch` is **internal temporary only**, not the training state source of truth.

```rust
pub enum AttentionForwardRunner {
    Baseline(/* uses CompiledKernels sdpaFwd + woFwd */),
    Sharded(ShardedAttentionForwardRuntime),     // NEW: extracted from bench
}

pub enum FfnForwardRunner {
    Baseline(/* uses CompiledKernels w13Fwd + w2Fwd */),
    Sharded(ShardedFfnForwardRuntime),           // wraps existing production runtime
}

pub struct CombinedForwardRuntime {
    pub mode: ResolvedForwardMode,
    pub attn: AttentionForwardRunner,
    pub ffn: FfnForwardRunner,
    scratch: LayerScratch,
}
```

### P0-B.3: ForwardCache write contract

`ForwardCache` is the single write target for training. `forward_into_combined` must populate **all** fields backward depends on:

| Field | Source | Notes |
| --- | --- | --- |
| **`x`** | layer input | **Copy before any computation** ([L4769](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L4769)) — backward RMSNorm1 needs it ([L4216](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L4216)) |
| `xnorm`, `rms_inv1` | RMSNorm1 output | Direct write |
| `q_rope` | worker `sdpa.q_rope_out` | Scatter by `q_col_start` |
| `k_rope` | worker `sdpa.k_rope_out` | Scatter by `kv_col_start` |
| `v` | worker `sdpa.v_out` | Scatter by `kv_col_start` |
| `attn_out` | worker `sdpa.attn_out` | Scatter by `q_col_start` |
| `o_out` | merged wo outputs | Direct write |
| `x2`, `x2norm`, `rms_inv2` | post-attention | Direct write |
| `h1`, `h3`, `gate` | FFN shard outputs | `store_shard_channels` post-join |

### P0-B.4: Production layer API — sharded modes only

> [!IMPORTANT]
> Dispatch is keyed on `ResolvedForwardMode`, **not** on whether any env var is set. `ResolvedForwardMode::Baseline` → existing pipelined path. Any sharded variant → `forward_into_combined`. This prevents cases like `FFN_BWD_SHARDS=4` (backward-only) from unnecessarily routing forward through the combined path, and handles `AutoAdjustNearest` falling back to baseline correctly.

```rust
pub fn forward_into_combined(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,       // baseline attention fallback
    weights: &LayerWeights,
    x: &[f32],
    cache: &mut ForwardCache,        // THE write target
    x_next: &mut [f32],
    runtime: &mut CombinedForwardRuntime,
) {
    cache.x.copy_from_slice(x);      // preserve for backward
    // 1. RMSNorm1 → cache.xnorm, cache.rms_inv1
    // 2. Attention (baseline or sharded) → cache.{q_rope,k_rope,v,attn_out,o_out}
    // 3. Residual → cache.x2
    // 4. RMSNorm2 → cache.x2norm, cache.rms_inv2
    // 5. FFN (baseline or sharded) → cache.{h1,h3,gate}, x_next
}
```

Dispatch in `full_model.rs`:
- **`ResolvedForwardMode::Baseline`:** existing pipelined baseline — no change
- **Any sharded `ResolvedForwardMode`:** `forward_into_combined`

---

## P1-A: Wire into forward-only

Add `CombinedForwardRuntime` ownership to workspaces:

```rust
pub struct ModelForwardWorkspace {
    pub combined_forward: Option<CombinedForwardRuntime>,  // None = baseline pipelined
}
```

Add `forward_only_ws_with_options`. Runtime compiled once, stored persistently.

**Verify:**
- Toleranced sharded vs baseline
- `#[ignore]`d smoke at 30B and 122B
- **Workspace reuse tests (non-`#[ignore]`d):**
  - `baseline → ffn4 → baseline` on one workspace: second baseline must produce bit-exact output vs a fresh-workspace baseline, and `w2t_scratch` must survive the mode transitions
  - `FFN_BWD_SHARDS=4` with no forward sharding: forward must take pipelined path, not combined
  - `AutoAdjustNearest` resolving to baseline (e.g. invalid shard count with no valid neighbor): forward must take pipelined path, `ModeResolution.forward` must be `Baseline`

---

## P1-B: Wire into training forward + backward migration

1. Stop forcing `forward_attn_request: None` at [L133](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/full_model.rs#L133)
2. `forward_ws_with_options` dispatches: baseline → pipelined, sharded → `forward_into_combined` writing into per-layer `ForwardCache`
3. **Backward wiring migration:** `backward_ws_with_options` ([L572](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/full_model.rs#L572)) currently branches on `options.backward_ffn_request: Option<usize>` and `options.backward_attn_request: Option<usize>`. Migrate to branch on `ResolvedBackwardMode` from `TrainingParallelOptions`. Update runtime compile/reuse checks to match resolved mode variants, not raw `Option<usize>` fields.

**Verify:**
- Bit-exact + toleranced layer tests
- `make sweep-600m` all four combinations
- **Workspace reuse tests (non-`#[ignore]`d):**
  - `baseline → attn2+ffn4 → ffn4` on one workspace: runtime rebuilt on each mode change, outputs match fresh-workspace reference
  - `attn2 → attn2`: same mode twice, runtime NOT rebuilt (verify via shard-count match), bit-exact across calls
  - Forward+backward with mismatched shard counts (e.g. `FFN_FWD_SHARDS=4 FFN_BWD_SHARDS=2`): both paths resolve and execute independently, loss decreases

---

## P1-C: Research-gated ANE overlap spike

Go/no-go: prove async ANE dispatch yields measurable overlap in a microbench.

---

## Env var policy

| Env var | Direction | Default |
| --- | --- | --- |
| `ATTN_FWD_SHARDS` | forward attention | None (baseline) |
| `ATTN_BWD_SHARDS` | backward attention | None (baseline) |
| `FFN_FWD_SHARDS` | forward FFN | None (baseline) |
| `FFN_BWD_SHARDS` | backward FFN | None (baseline) |
| `ATTN_SHARDS` | shorthand: both attn directions | — |
| `FFN_SHARDS` | shorthand: both FFN directions | — |

If both a shorthand and a per-direction var are set for the same axis, **error immediately**: `"FFN_SHARDS=4 conflicts with FFN_FWD_SHARDS=2; use per-direction vars or the shorthand, not both"`.

`ResolvedForwardMode::Baseline` → pipelined path. Any sharded `ResolvedForwardMode` → combined runtime.
