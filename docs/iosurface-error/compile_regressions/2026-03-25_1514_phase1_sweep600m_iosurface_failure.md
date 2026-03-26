# Compile/IOSurface Regression Note

- Timestamp: `2026-03-25 15:14`
- Branch: `codex/latency-parallel-research`
- HEAD commit: `9e36f9a5bbb2d4304fbb914a1ea23b933f439550`
- Local `ane` dependency: path dependency to `/Users/USER/other-projects/ane/crates/ane`
- Local `ane` branch/rev: `compile-fixed` @ `900a70607eac249b1cfbf622439ab1db2589164f`
- Phase: Phase 1 extraction verification

## Last successful gates

- `cargo test -p engine --release -- --list`
- `make forward-ladder FFN_SHARDS=10`
  - `5B` completed successfully
  - `7B` completed successfully
  - `10B` compiled and entered allocation before manual interruption
- 30B smoke check for ~30 seconds
  - compiled and entered runtime
  - no early IOSurface/compile failure observed

## Failing command

```bash
make sweep-600m
```

## Requested/applied shard counts

- Requested `ATTN_SHARDS`: omitted
- Requested `FFN_SHARDS`: omitted
- Applied mode: baseline training benchmark path

## Failing shape

- Benchmark: `600m-A`
- Shape: `dim=1536 hidden=4096 heads=12 nlayers=20 seq=512`
- Printed estimate: `~579M params`

## Exact failure output

```text
============================================================
  600m-A — 1536d/4096h/20L/seq512 — ~579M params — h/d=2.67x
============================================================

thread 'sweep_600m_a' (997502) panicked at /Users/USER/other-projects/ane/crates/ane/src/io_surface.rs:55:14:
IOSurface creation failed
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
  [1/4] Compiling ANE kernels... test sweep_600m_a ... FAILED
```

## Worktree changes since HEAD

```text
Cargo.lock
Cargo.toml
Makefile
crates/engine/src/bin/theory_runner.rs
crates/engine/src/lib.rs
crates/engine/tests/bench_ane_perf_stats_probe.rs
crates/engine/tests/bench_forward_utilization.rs
crates/engine/tests/bench_fwd_only_scale.rs
crates/engine/tests/bench_hw_execution_time.rs
```

## Diff summary

```text
Cargo.lock                                        | 87 +++++++++++++----------
Cargo.toml                                        |  2 +-
Makefile                                          | 14 ++--
crates/engine/src/bin/theory_runner.rs            | 38 +++++-----
crates/engine/src/lib.rs                          |  1 +
crates/engine/tests/bench_ane_perf_stats_probe.rs | 60 +++-------------
crates/engine/tests/bench_forward_utilization.rs  |  6 +-
crates/engine/tests/bench_fwd_only_scale.rs       | 73 ++++++++++---------
crates/engine/tests/bench_hw_execution_time.rs    | 45 ++----------
9 files changed, 139 insertions(+), 187 deletions(-)
```

## Notes

- This is an IOSurface allocation/runtime failure, not an ANE compile failure.
- The failure happened on the baseline training benchmark path before the new extracted forward benchmark code would influence execution.
- The same local `ane` checkout allows `forward-ladder` to run through at least `5B` and `7B`, and allows the 30B smoke test to enter runtime, so the issue is not a blanket failure of all large-model benchmark paths.
- Per the rollout stop rule, implementation should remain paused until the local `ane` + training benchmark path incompatibility is understood.
