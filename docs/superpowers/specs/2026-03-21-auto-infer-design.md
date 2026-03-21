# Auto-Infer: Autonomous Inference Optimization System

> Design spec for an autonomous optimization loop targeting Qwen3-MoE-30B decode throughput on M4 Max 128GB.
> Adapted from the auto-max training system. Designed for Opus 1M context with 60min hard timeout.

## Overview

A single bash script (`system/optimize-infer.sh`) runs Claude Opus in a git worktree, one iteration at a time. Each iteration reads the full codebase + research corpus, picks the highest-leverage change (tuning or architectural), implements it, passes a metric matrix, and commits or reverts. The agent has full freedom to explore — small tweaks or large architectural changes — constrained only by the metric matrix and the 60min timeout.

```
┌─────────────────────────────────────────────┐
│              optimize-infer.sh              │
│                                             │
│  for i in 1..N:                             │
│    1. Create/reuse worktree on agent branch │
│    2. Build prompt (code + research + AGENTS)│
│    3. Launch claude -p (Opus 1M, 60min)     │
│    4. Watchdog hard-kills at 60min          │
│    5. Sync experiments-infer.tsv back       │
│    6. Cooldown, repeat                      │
│                                             │
│  Agent (inside worktree):                   │
│    1. Read everything                       │
│    2. Think — pick highest-leverage change   │
│    3. Implement                             │
│    4. Pass metric matrix (3 tiers)          │
│    5. Commit if improved, revert if not     │
│    6. Log to experiments-infer.tsv          │
└─────────────────────────────────────────────┘
```

## System Components

### 1. `system/optimize-infer.sh` — The Loop

Bash script that runs the outer iteration loop. Manages worktrees, timeouts, logging.

**Arguments:**
```bash
./system/optimize-infer.sh                        # 10 iters, opus, 60min timeout
./system/optimize-infer.sh --iters 1              # single iteration
./system/optimize-infer.sh --timeout 60           # 60 min hard timeout (default)
./system/optimize-infer.sh --dry-run              # print prompt, don't run
./system/optimize-infer.sh --model opus           # model selection
./system/optimize-infer.sh --status               # show current status
```

**Worktree lifecycle:**
1. Branch from `rustane-infer` (the current inference branch, not master)
2. Agent branch: `infer-opt/auto-{agent_id}` (e.g., `infer-opt/auto-alpha`)
3. Worktree at `/tmp/rustane-infer-opt-{agent_id}`
4. Copy `dev/CURRENT.md`, `dev/METHODOLOGY.md` into worktree (gitignored context)
5. Copy `research/stage5-plan-2026-03-21.md` into worktree
6. After iteration: sync `system/experiments-infer.tsv` back to main repo
7. Worktree preserved between iterations (agent branch accumulates wins)

**Timeout:**
- Hard kill at 60 minutes. No soft extension — keeps the system simple.
- Watchdog runs as background subshell, kills claude process tree at timeout.
- On kill: `git checkout -- .` to revert partial changes, log TIMEOUT to experiments-infer.tsv.
- Ideal iteration: 20-30 minutes. 60min is the safety net for architectural changes.

**Control files:**
- `/tmp/rustane-infer-STOP` — stop after current iteration
- `/tmp/rustane-infer-PAUSE-{id}` — pause between iterations
- `/tmp/rustane-infer-INJECT-{id}` — append extra instructions to next iteration's prompt
- `/tmp/rustane-infer-status-{id}` — agent writes current phase (READING, CODING, TESTING, BENCHMARKING, DONE)

**Logging:**
- Console output: tee to `/tmp/rustane-infer-opt-{id}.log`
- Per-iteration timing reported
- experiments-infer.tsv synced back after each iteration

### 2. `AGENTS-INFER.md` — Institutional Knowledge

Checked into the repo root. Read by the agent at the start of every iteration. Contains:

**Sections:**
1. **Branch Policy** — never push to rustane-infer directly, work on agent branch
2. **Build & Test Commands** — exact commands for build, correctness, benchmarks
3. **Metric Matrix** — the 3-tier gate (see below)
4. **Architecture Overview** — current decode pipeline, where time goes
5. **M4 Max Hardware Facts** — verified constants (bandwidth, dispatch overhead, etc.)
6. **What's Been Tried** — every optimization from Stage 3 + Stage 4 with results
7. **Proven Dead Ends** — things that DON'T work and why (with wasted iteration count)
8. **What DOES Work** — proven patterns for this codebase
9. **Research Corpus** — key findings from ane-infer, uzu, flash-moe
10. **Current Bottleneck Breakdown** — exact per-layer timing data from bench_decode_layer_breakdown

### 3. The Metric Matrix — Anti-Cheat Gate

Three tiers. Agent must pass ALL of Tier 1 and not regress on Tier 2 to commit.

**Tier 1 — Hard Gates (any failure = instant revert)**

| # | Test | Current | Command | Rule |
|---|------|---------|---------|------|
| 1 | HF greedy match | 20/20 | `cargo test -p moe-infer --test test_generation --release -- test_generation_matches_hf --ignored` | Must stay ≥18/20 |
| 2 | Full test suite | all pass | `cargo test -p moe-infer --release` | Zero failures |
| 3 | Metal kernel correctness | all pass | `cargo test -p moe-infer --test test_fused_metal --release` | max_diff < 1e-3 |
| 4 | Build clean | 0 errors | `cargo build -p moe-infer --release` | Must compile |
| 5 | Agent's own custom test | pass | `cargo test -p moe-infer --test auto_<name> --release` | Agent writes per-experiment test |

**Tier 2 — Performance Metrics (must improve or hold)**

| # | Metric | Current | Command | Rule |
|---|--------|---------|---------|------|
| 1 | Decode tok/s | 19.6 | `bench_tok_per_sec` (median of 3 runs) | Must not regress >3% |
| 2 | Prefill latency | 310ms | `bench_tok_per_sec` (prefill_secs field) | Must not regress >10% |
| 3 | Decode p95 latency | ~55ms | `bench_decode_layer_breakdown` p95 | Must not regress >10% |

**Tier 3 — Tracked Metrics (logged to TSV, not gated)**

| # | Metric | Current | Why |
|---|--------|---------|-----|
| 1 | Per-layer breakdown (metal_us, attn_us) | metal 31ms, attn 25ms | Lie detector — shows WHERE time went |
| 2 | Metal allocs in hot path | 0 | Detect allocation regression |
| 3 | Time-to-first-token | ~360ms | User-facing latency |
| 4 | Peak memory during generation | ~20GB | Catch memory leaks |

**Anti-cheat properties of the matrix:**
- Tier 1 #1 (HF match) prevents numerical drift
- Tier 1 #5 (custom test) proves the specific change is semantically equivalent
- Tier 2 prevents improving one metric at the cost of another
- Tier 3 breakdown is the lie detector — impossible to fake a speedup when the per-component numbers don't add up
- Agent CANNOT modify: test_generation.rs, bench_tok_per_sec.rs, bench_decode_layer_breakdown.rs, test_fused_metal.rs, configs/*, weights/references/*

### 4. `system/experiments-infer.tsv` — Experiment Log

Tab-separated, append-only. One row per experiment.

**Columns:**
```
date	experiment	variable	baseline	result	verdict	decode_toks	prefill_ms	p95_ms	metal_ms	attn_ms	notes
```

**Verdict values:** `IMPROVED`, `REVERTED`, `NO EFFECT`, `BROKEN`, `INTERESTING`, `TIMEOUT`

**Rules:**
- Never modify existing rows — append only
- Always log, even failures (prevents re-attempts)
- `decode_toks` must be median of 3 benchmark runs
- `notes` must include WHY the change worked or didn't (the lesson)

### 5. The Prompt

The prompt sent to Claude each iteration. Structured as steps.

```
STEP 1 — READ CONTEXT:
  - AGENTS-INFER.md (institutional knowledge, dead ends, metric matrix)
  - system/experiments-infer.tsv (every experiment tried)
  - research/stage5-plan-2026-03-21.md (ranked optimization targets with estimates)
  - crates/moe-infer/src/generate.rs (decode loop — the hot path)
  - crates/moe-kernels/src/dequant.rs (Metal shaders)
  - crates/moe-infer/src/attention.rs (CPU attention — 38% of decode time)
  - crates/moe-infer/src/blas.rs (BLAS bindings)
  - dev/research/02-ane-inference-landscape.md (ane-infer and uzu patterns)
  - CREDITS.md (reference implementations)

STEP 2 — THINK:
  Read experiments-infer.tsv. What's been tried? What worked? What's left?
  Read the Stage 5 plan. What's the highest-leverage change?
  Consider: is it a small tuning change or a larger architectural change?
  Both are valid. You have 60 minutes and 1M context — use them.
  Write your hypothesis to the status file before coding.

STEP 3 — IMPLEMENT:
  Make your change. Can be small (shader constant) or large (new Metal kernel).
  Read existing code before modifying it. Follow existing patterns.

STEP 3.5 — WRITE A CUSTOM TEST:
  Before running correctness, write a test that proves YOUR specific change
  is semantically equivalent to the original code path.
  File: crates/moe-infer/tests/auto_<experiment_name>.rs
  The test MUST:
  a) Capture output of the ORIGINAL code path
  b) Run the OPTIMIZED code path on same input
  c) Assert they match within tolerance
  d) Test at least 2 edge cases specific to your change

STEP 4 — PASS THE METRIC MATRIX:
  4a. Your custom test (step 3.5)
  4b. Tier 1 hard gates (HF match, full suite, Metal correctness, build)
  4c. Tier 2 performance (tok/s median of 3, prefill, p95)
  If ANY Tier 1 fails: revert ALL changes, skip to STEP 6.
  If Tier 2 regresses >3%: revert, skip to STEP 6.

STEP 5 — RECORD TIER 3 METRICS:
  Run bench_decode_layer_breakdown. Record per-component breakdown.
  This is the lie detector — your improvement must show up in the components.

STEP 6 — LOG RESULTS:
  Append ONE row to system/experiments-infer.tsv.
  The notes column MUST explain WHY (not just what).

STEP 7 — COMMIT OR REVERT:
  If IMPROVED: git add changed files + auto_*.rs test + experiments-infer.tsv, commit.
  If REVERTED/BROKEN: git checkout -- ., git add experiments-infer.tsv, commit log only.

RULES:
  - NEVER modify locked test files (test_generation.rs, bench_tok_per_sec.rs,
    bench_decode_layer_breakdown.rs, test_fused_metal.rs)
  - NEVER modify configs/ or weights/references/
  - NEVER weaken test assertions to make tests pass
  - NEVER claim improvement without median-of-3 benchmark data
  - Always log, even failures — they prevent wasted re-attempts
  - The custom test (step 3.5) is ALWAYS committed with improvements —
    it permanently protects the invariant
  - Write status to /tmp/rustane-infer-status-{agent_id} at each phase
```

### 6. Locked Files

These files are read-only for the agent. The optimize-infer.sh script verifies they haven't been modified after each iteration (git diff check).

```
crates/moe-infer/tests/test_generation.rs       # HF match gate
crates/moe-infer/tests/bench_tok_per_sec.rs      # THE benchmark
crates/moe-infer/tests/bench_decode_layer_breakdown.rs  # lie detector
crates/moe-infer/tests/test_fused_metal.rs       # kernel correctness
configs/qwen3-moe-30b.toml                       # model config
weights/references/greedy_generation.json         # reference outputs
```

Post-iteration check in optimize-infer.sh:
```bash
LOCKED_FILES="crates/moe-infer/tests/test_generation.rs crates/moe-infer/tests/bench_tok_per_sec.rs ..."
for f in $LOCKED_FILES; do
    if ! git diff --quiet "$f" 2>/dev/null; then
        log "VIOLATION: agent modified locked file $f — reverting ALL changes"
        git checkout -- .
        break
    fi
done
```

## File Inventory

| File | Action | Description |
|------|--------|-------------|
| `system/optimize-infer.sh` | Create | The outer loop script |
| `AGENTS-INFER.md` | Create | Institutional knowledge for inference agents |
| `system/experiments-infer.tsv` | Modify | Add proper column headers matching new format |

## Usage

```bash
# Single iteration (test the system)
./system/optimize-infer.sh --iters 1 --dry-run     # preview prompt
./system/optimize-infer.sh --iters 1                # run one iteration

# Autonomous loop in tmux
tmux new -s infer-opt './system/optimize-infer.sh --iters 20'
# Detach: Ctrl-B D
# Reattach: tmux attach -t infer-opt

# Monitor
tail -f /tmp/rustane-infer-opt-alpha.log
cat /tmp/rustane-infer-status-alpha
tail -20 system/experiments-infer.tsv

# Control
touch /tmp/rustane-infer-STOP                              # stop after current iter
echo "focus on Metal attention" > /tmp/rustane-infer-INJECT-alpha  # inject guidance
```

## Success Criteria

The system is working when:
1. An iteration completes autonomously within 60min
2. The metric matrix is enforced (violations revert changes)
3. Locked files are verified unmodified after each iteration
4. experiments-infer.tsv accumulates results with WHY explanations
5. The agent makes progress toward 25+ tok/s decode
