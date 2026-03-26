# Experiment: Split Compile/Allocate Phases in parallel_bench.rs

## Hypothesis

> [!IMPORTANT]
> This is the **leading hypothesis**, not a confirmed root cause. Success criteria are empirical.

The ane debrief ([debrief-round-2.md](file:///Users/USER/other-projects/ane/debrief-round-2.md)) proved compile-payload IOSurface allocation works fine in isolation. The original failure is a **small compile-time MIL surface** (~11,865 bytes) inside `ane::client::compile_network` for `sdpaFwd`.

The hypothesis: `parallel_bench.rs` interleaves compilation and TensorData allocation, so later kernel compilations attempt their MIL payload IOSurface while earlier kernels' runtime TensorData IOSurfaces are already live. This overlap may exhaust a shared kernel resource.

> [!NOTE]
> **Caveat**: the baseline 7B compile path (`CompiledKernels::compile` in `layer.rs`) can also fail on the same small MIL allocation. That path compiles all executables first, then allocates buffers — the "good" pattern. This means the overlap theory may only explain the **increased failure frequency** in sharded/mode-runner paths, not the entire IOSurface problem. If the refactor reduces but does not eliminate failures, the remaining issue is likely system-state-sensitive and outside rustane's control.

## Observed Failure Data

- **Failing call**: `ane::client::compile_network` → `nsdata_on_surface("mil", ...)` → nil
- **Size**: 11,865 bytes (11.6 KB)
- **Error**: `IOSurfaceAlloc { context: "mil", byte_count: 11865, width: 11865, bpr: 11865 }`
- **Kernel**: `sdpaFwd`
- **Behavior**: nondeterministic — same code path sometimes passes, sometimes fails

## Status of Prior Work

- [x] `compile_forward_only()` is **truly forward-only** — compiles only forward kernels and uses `allocate_forward_only()` with tiny backward placeholders ([layer.rs L756–812](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs#L756)). Part of the surface-count reduction is already in place.
- [x] Retry wrapper — landed in [parallel_bench.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/parallel_bench.rs#L112)
- [x] ane-side `Result`-based `IOSurfaceAlloc` error — landed in ane `compile-fixed` branch
- [ ] **Split compile/allocate phases in `parallel_bench.rs`** — this experiment
- [ ] Fix `tensor_data_new_logged` byte count (fp32 → fp16)

## Proposed Changes

### [MODIFY] [parallel_bench.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/parallel_bench.rs)

Split each `compile_*_runner` into compile-only and allocate-only functions.

Restructure `compile_mode_runner` (L733) into two explicit sequential phases to prevent accidental per-worker allocate-while-compiling:

```rust
fn compile_mode_runner(cfg: &ModelConfig, mode: &ParallelMode) -> Result<(ModeRunner, f32), AneError> {
    let t0 = Instant::now();

    // ── Phase 1: compile ALL executables, no TensorData alive ──
    let attn_exes = match mode {
        ParallelMode::Attention { shards } | ParallelMode::AttentionFfn { attn_shards: shards, .. } => {
            let layouts = attention_shard_layouts(cfg, *shards);
            let exes: Vec<(AttentionShardLayout, ModelConfig, Executable, Executable)> = layouts
                .into_iter()
                .map(|layout| {
                    let wcfg = attention_group_cfg(cfg, layout.q_head_count, layout.kv_head_count);
                    let sdpa_exe = compile_sdpa_exe(&wcfg)?;
                    let wo_exe = compile_wo_exe(&wcfg)?;
                    Ok((layout, wcfg, sdpa_exe, wo_exe))
                })
                .collect::<Result<_, AneError>>()?;
            CompiledAttnExes::Sharded(exes)
        }
        _ => {
            let sdpa_exe = compile_sdpa_exe(cfg)?;
            let wo_exe = compile_wo_exe(cfg)?;
            CompiledAttnExes::Baseline(sdpa_exe, wo_exe)
        }
    };
    let ffn_exes = match mode {
        // ... similar: collect all FFN executables into tuples/vectors ...
    };

    // ── Phase 2: allocate ALL TensorData, assemble runners ──
    let attn = match attn_exes {
        CompiledAttnExes::Baseline(sdpa_exe, wo_exe) => {
            AttentionRunner::Baseline(BaselineAttentionRunner {
                cfg: cfg.clone(),
                sdpa: allocate_sdpa_buffers(sdpa_exe, cfg),
                wo: allocate_wo_buffers(wo_exe, cfg),
            })
        }
        CompiledAttnExes::Sharded(worker_exes) => {
            // ... allocate per-worker TensorData from pre-compiled exes ...
        }
    };
    // ... same for ffn ...

    Ok((ModeRunner { attn, ffn, scratch: LayerScratch::new(cfg) }, t0.elapsed().as_secs_f32()))
}
```

The key constraint: **no `TensorData::new()` / `tensor_data_new_logged()` calls may appear in phase 1.** All executables must be compiled into temporary tuples/vectors before any runtime surface is allocated.

### [MODIFY] [layer.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/layer.rs)

Fix `tensor_data_new_logged` (L35): change `elements * size_of::<f32>()` to `elements * 2`. This function is **env-var gated** (`RUSTANE_LOG_IOSURFACE_ALLOC`), not `cfg(debug_assertions)`, so the fix affects diagnostic output only when explicitly enabled.

## Verification Plan

### Primary Gates

```bash
make forward-ladder
make forward-ladder ATTN_SHARDS=10 FFN_SHARDS=10
make forward-ceiling   # 30B smoke, ~30s
```

### IOSurface Repro Path (optional)

```bash
# Run fwd_7b 5-10x to assess failure rate change
cargo test -p engine --test bench_fwd_only_scale --release -- --ignored --nocapture fwd_7b
```

### Existing Tests

```bash
cargo test -p engine --release
```

> [!IMPORTANT]
> Due to nondeterminism, a single passing run is not conclusive. Run `fwd_7b` repro path multiple times. If failure rate drops but doesn't reach zero, the overlap was a contributing factor but not the sole cause.
