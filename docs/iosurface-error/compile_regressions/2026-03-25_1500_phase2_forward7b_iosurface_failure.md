# Compile/IOSurface Regression Note

- Timestamp: `2026-03-25 15:00`
- Branch: `codex/latency-parallel-research`
- HEAD commit: `9e36f9a5bbb2d4304fbb914a1ea23b933f439550`
- Local `ane` dependency: path dependency to `/Users/USER/other-projects/ane/crates/ane`
- Local `ane` branch/rev: `compile-fixed` @ `900a70607eac249b1cfbf622439ab1db2589164f`
- Phase: forward benchmark rollout verification

## Last successful gate

- `cargo test -p engine --release -- --list`
- Result: passed on the current worktree with the local `ane` path dependency

## Failing command

```bash
make forward-7b
```

## Requested/applied shard counts

- Requested `ATTN_SHARDS`: omitted
- Requested `FFN_SHARDS`: omitted
- Applied mode: baseline

## Failing shape

- Benchmark: `7B`
- Shape: `dim=4096 hidden=11008 heads=32 nlayers=32 seq=512`
- Printed estimate: `26.8GB`

## Exact failure output

```text
======================================================================
  7B — 4096d/11008h/32L/seq512 — 6.51B params — est. 26.8GB
======================================================================

thread 'main' (955869) panicked at /Users/USER/other-projects/ane/crates/ane/src/io_surface.rs:55:14:
IOSurface creation failed
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace

thread 'theory_forward_7b_isolated' (955868) panicked at crates/engine/tests/theory_runner_supervisor.rs:52:5:
isolated 7B forward should succeed
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
test theory_forward_7b_isolated ... FAILED
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
Makefile                                          |  6 ++
crates/engine/src/bin/theory_runner.rs            | 38 +++++-----
crates/engine/src/lib.rs                          |  1 +
crates/engine/tests/bench_ane_perf_stats_probe.rs | 60 +++-------------
crates/engine/tests/bench_forward_utilization.rs  |  6 +-
crates/engine/tests/bench_fwd_only_scale.rs       | 73 ++++++++++---------
crates/engine/tests/bench_hw_execution_time.rs    | 45 ++----------
9 files changed, 135 insertions(+), 183 deletions(-)
```

## Notes

- This is an IOSurface allocation/runtime failure, not an ANE compile failure.
- The failure happened on the baseline `make forward-7b` path before any sharded forward mode was exercised.
- Because the local `ane` path dependency was changed in the same worktree, that dependency change is a primary suspect and should be isolated before continuing the forward benchmark rollout.
