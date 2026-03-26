# Compile/IOSurface Regression Note

- Timestamp: `2026-03-25 16:26`
- Branch: `codex/latency-parallel-research`
- HEAD commit: `40e73017ccaeb9ea88f3a5b193d1559e65b9714d`
- Local `ane` dependency: path dependency to `/Users/USER/other-projects/ane/crates/ane`
- Local `ane` branch/rev: `compile-fixed` @ `900a70607eac249b1cfbf622439ab1db2589164f`
- Phase: pre-Phase-3 / Phase-2 verification battery

## Passing checks immediately before failure

- `cargo test -p engine --release -- --list`
- `cargo test -p engine --test parallel_bench_forward --release -- --ignored --nocapture parallel_bench_modes_match_baseline`
- `cargo test -p engine --test bench_fwd_only_scale --release -- --ignored --nocapture fwd_7b`

## Failing command

```bash
FFN_SHARDS=8 cargo test -p engine --test bench_fwd_only_scale --release -- --ignored --nocapture fwd_7b
```

## Requested/applied shard counts

- Requested `ATTN_SHARDS`: omitted
- Requested `FFN_SHARDS`: `8`
- Intended applied mode: `ffn8`

## Failing shape

- Benchmark: `7B`
- Shape: `dim=4096 hidden=11008 heads=32 nlayers=32 seq=512`
- Printed estimate: `26.8GB`

## Exact failure output

```text
======================================================================
  7B — 4096d/11008h/32L/seq512 — 6.51B params — est. 26.8GB
======================================================================

thread 'fwd_7b' (...) panicked at /Users/USER/other-projects/ane/crates/ane/src/io_surface.rs:55:14:
IOSurface creation failed
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
  [1/3] Compiling ANE kernels... test fwd_7b ... FAILED
```

## Worktree state

- Clean relative to `HEAD`
- No uncommitted code changes at the time of failure

## Notes

- This failure happened on the sharded `fwd_7b` gate after the baseline `fwd_7b` gate passed in the same session.
- The failure occurred before the benchmark reached weight/workspace allocation; it failed during the ANE kernel compile/setup path.
- Per the rollout rule, Phase 3 should remain paused until this local-`ane` instability is understood or explicitly waived.
