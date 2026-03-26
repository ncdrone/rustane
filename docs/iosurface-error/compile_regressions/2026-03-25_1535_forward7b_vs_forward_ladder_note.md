# Forward Benchmark Path Note

- Timestamp: `2026-03-25 15:35`
- Branch: `codex/latency-parallel-research`
- Context: local `ane` path dependency on `/Users/USER/other-projects/ane/crates/ane`

## Observation

The isolated `make forward-7b` path and the multi-shape `make forward-ladder` path do **not** behave the same on this branch with the current local `ane` checkout.

## Commands and outcomes

### `make forward-7b`

- Entry path:
  - `Makefile`
  - `crates/engine/tests/theory_runner_supervisor.rs`
  - `crates/engine/src/bin/theory_runner.rs`
- Outcome:
  - fails immediately at runtime
  - panic: `IOSurface creation failed`

### `make forward-ladder`

- Entry path:
  - `Makefile`
  - `crates/engine/tests/bench_fwd_only_scale.rs`
- Outcome:
  - `5B` completed successfully
  - `7B` completed successfully
  - `10B` compiled and entered allocation before manual interruption

## Practical implication

For rollout verification on this branch, use `make forward-ladder` instead of `make forward-7b` as the representative forward benchmark gate until the `theory_runner` path is investigated separately.

This note is diagnostic only. It does **not** explain the root cause yet; it only establishes that the failure is path-specific rather than a blanket failure of the forward-only benchmark path.
