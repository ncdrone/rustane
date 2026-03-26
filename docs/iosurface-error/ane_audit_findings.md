# ANE Audit Findings

## Context

This audit compared:

- the local Rustane repo state,
- the root note in `fp16-ane-changes.md`,
- the currently pinned `ane` dependency in Rustane,
- the local updated `ane` checkout at:
  `/Users/USER/Other projects/ane`
- the documented upstream update:
  commit `855c025` on branch `codex/fp16-ane-io-perf-stats`

No code changes were made as part of the audit itself.

## High-Level Conclusion

Rustane is not yet aligned to the `ane` revision described in `fp16-ane-changes.md`.

The repo is still pinned to an older `ane` commit in `Cargo.lock`, and while some fp16 support already exists in the currently pinned dependency, the full upstream update has not been adopted.

The biggest remaining gaps are:

1. dependency alignment to the newer `ane` commit,
2. adoption of the richer perf-stats API,
3. migration of engine call sites to the existing `ane-bridge::io` helper layer.

The upstream `ane` update does not appear to change graph semantics or MIL generation, so it should be treated as an ANE I/O and observability upgrade, not as a direct fix for the current 5B IOSurface/compile-pressure issue.

## What The Repo Is Pinned To Today

Rustane currently declares:

```toml
ane = { git = "https://github.com/ncdrone/ane", branch = "main" }
```

And the lockfile currently resolves that to:

- commit `016b754a7d6182d41144aee80eb83b572624bcff`

This does not match the documented updated commit:

- `855c0257f0afcfef86ea982b5560acc5c2d1713f`

## What The Updated Local `ane` Repo Adds

The local updated `ane` repo at `/Users/USER/Other projects/ane` does contain the documented work from `fp16-ane-changes.md`.

Confirmed additions include:

### 1. Richer fp16 tensor access

In the updated local `ane` checkout, `TensorData` includes:

- `with_f16_bits`
- `copy_from_f16_bits`
- `read_f16_bits`
- `as_f16_bits_slice`
- `as_f16_bits_slice_mut`

### 2. Richer perf-stats API

The updated local `ane` checkout includes:

- public `PerfStats`
- `Executable::run_cached_with_perf_stats(...)`

This is additive to the older:

- `run_cached_with_stats(...) -> Result<u64, Error>`

### 3. Perf counter decoding

The updated local `ane` repo contains:

- `crates/ane/src/perf_stats.rs`
- `crates/ane/tests/perf_stats_integration.rs`

That code decodes counter bytes into words and exposes a best-effort named counter map.

## What The Currently Pinned `ane` Already Has

The currently pinned `ane` commit is not completely pre-fp16.

It already includes:

- `TensorData::with_f16_bits`
- `TensorData::copy_from_f16_bits`
- fp16-backed `TensorData::new`
- NEON f32<->fp16 conversion on the f32 access path

So the missing upstream pieces are more specific:

- direct fp16 guard accessors
- `read_f16_bits`
- richer `PerfStats`
- `run_cached_with_perf_stats`
- stricter exact-length validation on `copy_from_f32`

## Hopper Corroboration

Hopper inspection of `AppleNeuralEngine2` corroborates the perf-stats assumptions in the markdown.

Observed strings include:

- `perfCounterData`
- `pStatsRawData`
- `hwExecutionTime`
- `kANE_FP16_CYCLES:`
- `kANE_NE_COMPUTE_CYCLES`
- `kANE_NE_INPUT_STALL_CYCLES`
- `kANE_NE_OUTPUT_STALL_CYCLES`
- multiple other `kANE_*` counter names

So the observability update is grounded in real framework symbols.

## What Rustane Has Already Built Locally

Rustane already has a substantial bridge layer in:

- `crates/ane-bridge/src/io.rs`

That file already implements most of what `fp16-ane-changes.md` recommends adding:

- f32 -> fp16 bit conversion helpers
- fp16 -> f32 conversion helpers
- direct tensor fp16-bit write helpers
- channel-major packers
- contiguous channel readback helpers
- named layout helpers for:
  - `sdpa_fwd`
  - `wo_fwd`
  - `ffn_fused`
  - `dyn_matmul`
  - `dual_dyn_matmul`
  - `dual_separate`

There are also direct bridge tests in:

- `crates/ane-bridge/tests/io_helpers.rs`

So the bridge module itself is not the missing piece anymore.

## What Rustane Is Still Not Doing

### 1. Engine call sites are not migrated to the bridge

Despite the existence of `ane-bridge::io`, the `engine` crate still mostly uses:

- handwritten staging loops,
- manual offset math,
- repeated `as_f32_slice_mut()` and `as_f32_slice()` logic,
- duplicated packing/readback behavior across runtime code and benchmarks.

This duplication is still present in files such as:

- `crates/engine/src/layer.rs`
- `crates/engine/src/parallel_forward.rs`
- `crates/engine/tests/bench_ffn_latency_parallel.rs`
- `crates/engine/tests/bench_ffn_latency_parallel_full_model.rs`
- `crates/engine/tests/bench_attn_ffn_parallel_full_model.rs`

### 2. Rustane has not adopted the richer perf-stats path

Current code still uses `run_cached_with_stats(...)` in benches and probes.

Notable examples:

- `crates/engine/tests/bench_forward_utilization.rs`
- `crates/engine/tests/bench_hw_execution_time.rs`
- `crates/engine/tests/bench_ffn_latency_parallel_full_model.rs`

And the repo still contains private-runtime swizzle probing in:

- `crates/engine/tests/bench_ane_perf_stats_probe.rs`

The new upstream `PerfStats` API is intended to replace a large part of that lower-level probing effort.

## Compatibility Risk To Watch

The updated upstream `ane` adds stricter exact-length validation to:

- `copy_from_f32`
- `copy_from_f16_bits`

That means any Rustane call site doing partial writes through those APIs would become a panic after the dependency update.

I checked the obvious `copy_from_f32` usages in Rustane, and they appear to be full-tensor writes, so this looks low-risk, but it is still a compatibility check worth preserving during dependency rollout.

## What Needs To Be Updated

### Required

1. Update Rustane to consume an `ane` revision that includes `855c025` or equivalent.
2. Pin that dependency by exact revision or local path rather than leaving it as a moving branch.
3. Refresh `Cargo.lock` so the repo actually resolves to the updated `ane`.
4. Migrate perf-sensitive benches and probes from `run_cached_with_stats` to `run_cached_with_perf_stats`.
5. Start moving engine staging/readback call sites onto `ane-bridge::io`.

### Recommended

1. After the dependency update, refactor `ane-bridge::io` to use upstream direct fp16 guard APIs instead of manual `surface()` locking where that improves clarity.
2. Reduce or replace Objective-C swizzle probes in favor of the new public `PerfStats` snapshot path.

## What Does Not Need To Be Rebuilt From Scratch

The following are already present locally and do not need fresh design work:

1. the bridge helper module,
2. the kernel layout formulas,
3. bridge-level layout tests.

The missing work is adoption and dependency alignment, not conceptual design.

## Important Limitation

The upstream `ane` update described by `855c025` should not be treated as a likely direct fix for Rustane's current 5B `IOSurface creation failed` issue.

The documented upstream changes are in tensor I/O and perf-stat reporting. They do not claim ANE graph-compile semantic changes, MIL generation changes, or compile-pressure fixes.

## Recommended First Step

The cleanest first step is:

1. point Rustane at the local updated `ane` checkout containing commit `855c025`,
2. refresh the lockfile,
3. verify the repo builds cleanly against that API surface,
4. only then begin call-site migration to `run_cached_with_perf_stats` and `ane-bridge::io`.
