# V3 Auto-Optimize System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an autonomous optimization loop (`optimize-v3.sh`) that runs Claude Opus in pipe mode to find small performance wins for DeepSeek-V3 inference (currently 0.7 tok/s, target 5+), with a 15-minute cron supervisor in the parent session.

**Architecture:** Fork of `system/optimize-infer.sh` retargeted for V3. New `AGENTS-V3.md` knowledge base with V3-specific bottleneck data, dead ends, metric matrix, and memory budget. Research synthesis files (~193 KB, 4.8% of 1M context) copied from `rustane-research` to worktree at loop start. Experiments logged to new `experiments-v3.tsv`. Cron-based supervisor monitors progress and can inject/pause/stop.

**Tech Stack:** Bash (loop script), Claude Opus 4.6 1M (pipe mode agent), git worktrees, CronCreate (supervisor)

**Context recovery note:** Read `memory/project_v3_auto_system.md` for full design rationale. Read `system/optimize-infer.sh` as the template being forked. Read `AGENTS-INFER.md` as the knowledge base template.

---

## File Structure

```
Files to CREATE:
  system/optimize-v3.sh              — Main loop script (fork of optimize-infer.sh)
  system/experiments-v3.tsv          — V3-only experiment log (seeded from existing data)
  system/monitor-v3.sh               — Companion dashboard script
  AGENTS-V3.md                       — V3 knowledge base for the auto agent

Files to READ (not modify):
  system/optimize-infer.sh           — Template being forked
  AGENTS-INFER.md                    — Knowledge base template
  system/experiments-infer.tsv       — Source for V3 seed data
  research/POST-MORTEM-v3-phase1.md  — Phase 1 results
  research/AUDIT-v3-phase1-to-10toks.md — Bugs and roadmap
  system/v3-phase2-optimize.md       — Phase 2 optimization plan
  system/v3-auto-optimize.md         — Deep architecture session guide
  crates/moe-infer/tests/bench_v3_tok_per_sec.rs     — V3 benchmark
  crates/moe-infer/tests/test_v3_validation.rs        — V3 validation
  crates/moe-infer/tests/test_model_validation.rs     — V2-Lite regression

Research repo (read-only, copy synthesis to worktree):
  /Users/dan/Dev/rustane-research/mla-1t/              — Full research corpus
```

---

### Task 1: Create `experiments-v3.tsv`

**Files:**
- Create: `system/experiments-v3.tsv`
- Read: `system/experiments-infer.tsv`

- [ ] **Step 1: Create the V3-only experiments file**

Extract V3 rows from `experiments-infer.tsv` (lines 9-18, the `v3-*` prefixed rows) into a new file with the same header format but V3-appropriate columns.

```tsv
date	experiment	variable	baseline	result	verdict	decode_toks	notes
2026-03-22	v3-baseline	lazy-conversion-serial	0.03	serial f16→f32 per layer, clones per token	baseline	0.03	initial state
2026-03-22	v3-zero-copy	no-clone-borrows	0.03	eliminated .clone() epidemic	keep	0.03	100GB memcpy saved
2026-03-22	v3-pipelined	channel-pipeline	-	pipeline conversion+compute, alloc thrashing	reverted	-	DEAD END: alloc thrashing worse than serial
2026-03-22	v3-single-buf	buffer-reuse	0.03	single buffer reuse, mmap page faults	keep	0.03	zero allocs after warmup
2026-03-22	v3-pread-experts	expert-pager-pread	0.03	pread experts (21x faster), conv now bottleneck	keep	0.2	replaced 348GB mmap thrashing
2026-03-22	v3-rayon-conv	par-chunk-conversion	0.2	7.3ms/layer conv (8.5x), 28ms/layer compute	keep	0.5	saturate memory bandwidth across cores
2026-03-22	v3-par-pread	parallel-pread+cached-metal	0.5	12.8ms/layer compute, 7.3ms conv, need pipeline	keep	0.7	NVMe QD>1 for throughput
2026-03-22	v3-rayon-pread	rayon-direct-staging	0.7	22.4ms/layer, rayon pread, no alloc overhead	keep	0.7	current architecture baseline
2026-03-22	v3-f16-sgemv	chunked-f16-sgemv	0.7	86ms/layer, per-chunk overhead dominates	reverted	0.2	DEAD END: 4x slower than convert+sgemv_f32
2026-03-22	v3-final	convert+sgemv+pread+rayon	0.03	22ms/layer avg, 1000x from baseline	current-best	0.7	Phase 1 final state
```

- [ ] **Step 2: Verify the file is well-formed**

Run: `head -1 system/experiments-v3.tsv && wc -l system/experiments-v3.tsv`
Expected: header row + 10 data rows = 11 lines total.

- [ ] **Step 3: Commit**

```bash
git add system/experiments-v3.tsv
git commit -m "chore: create V3-only experiments log seeded from Phase 1 data"
```

---

### Task 2: Create `AGENTS-V3.md`

**Files:**
- Create: `AGENTS-V3.md`
- Read: `AGENTS-INFER.md` (template for structure)
- Read: `research/POST-MORTEM-v3-phase1.md` (Phase 1 findings)
- Read: `research/AUDIT-v3-phase1-to-10toks.md` (bugs, roadmap)
- Read: `system/v3-phase2-optimize.md` (Phase 2 targets)
- Read: `crates/moe-infer/tests/bench_v3_tok_per_sec.rs` (V3 benchmark)
- Read: `crates/moe-infer/tests/test_v3_validation.rs` (V3 validation)
- Read: `crates/moe-infer/tests/test_model_validation.rs` (V2-Lite regression)

This is the agent's knowledge base — the single most important file. It must contain everything a V3 optimization agent needs to know. Use the structure of `AGENTS-INFER.md` as a template but rewrite every section for V3.

- [ ] **Step 1: Read all reference files**

Read all files listed above to gather V3-specific data. Key data to extract:
- V3 bottleneck breakdown (22ms/layer: conversion 32%, MLA 23%, shared FFN 18%, expert pread 14%, Metal 14%)
- V3 dead ends (f16 sgemv 4x slower, pipeline overhead > overlap, mmap thrashing)
- V3 bugs (B1: x_cache OOB, B2: staging_ptr UB, B5: ExpertPool not wired, B6: scalar attention, B7: scaling factor)
- V3 memory budget (128 GB total, ~70 GB used, ~50 GB available)
- V3 metric matrix commands (bench_v3_tok_per_sec, test_v3_validation, test_model_validation)
- V3 architecture (MLA, Q LoRA, sigmoid routing, 61 layers, 256 experts/layer, pread from SSD)
- Phase 2 optimization targets from `v3-phase2-optimize.md`
- Physical limits from research (f32 max 8.4 tok/s, f16 max 12+ tok/s)

- [ ] **Step 2: Write AGENTS-V3.md**

The file MUST contain these sections (follow `AGENTS-INFER.md` structure):

1. **Header** — "Instructions for AI agents optimizing DeepSeek-V3 inference on M4 Max 128GB"
2. **Branch Policy** — `v3-optimize` is stable, work on `v3-opt/auto-{agent_id}`, never push to v3-optimize
3. **Build & Test** — V3-specific commands (cargo build, test, bench_v3_tok_per_sec, test_v3_validation, test_model_validation)
4. **Metric Matrix** — V3-specific 3-tier gate:
   - Tier 1 hard gates: build clean, unit tests pass, V2-Lite regression (test_model_validation), V3 validation (test_v3_validation), custom test pass, locked files untouched
   - Tier 2 performance: V3 decode tok/s median-of-3 (bench_v3_tok_per_sec), no regress >5% from baseline
   - Tier 3 lie detector: per-layer timing from generate_v2.rs stderr output, improvement must show up in components
5. **Locked Files** — bench_v3_tok_per_sec.rs, test_v3_validation.rs, test_model_validation.rs, configs/deepseek-v3.toml, AGENTS-V3.md
6. **Architecture Overview** — V3 decode pipeline (backbone f16 mmap → MLA attention with Q LoRA → absorbed attention → O proj → MoE routing → expert pread from SSD → Metal INT4 GEMV → shared expert)
7. **Current Bottleneck** — 22ms/layer × 61 layers = 1,340 ms/token = 0.7 tok/s. Breakdown: conversion 32%, MLA sgemv 23%, shared FFN 18%, expert pread 14%, Metal dispatch 14%
8. **M4 Max Hardware Facts** — copy from AGENTS-INFER.md (same hardware)
9. **What's Been Tried (V3 Phase 1)** — table from experiments-v3.tsv data
10. **Proven Dead Ends — DO NOT RETRY** — f16 sgemv (4x slower, dispatch overhead per chunk), pipeline double-buffer (alloc thrashing), mmap for 348GB experts (page fault storm), mlock backbone (no measurable benefit)
11. **What DOES Work** — expert pager pread, rayon parallel conversion, cached Metal staging buffer, buffer reuse, zero-copy borrows, parallel pread for NVMe QD
12. **Known Bugs (DO NOT FIX — avoid in optimizations)** — B1 x_cache OOB, B2 staging_ptr UB, B5 ExpertPool not wired, B6 scalar attention, B7 scaling factor. Note: these are documented for awareness. Bug fixes are manual work, not auto-agent work.
13. **Memory Budget** — 128 GB total breakdown: backbone mmap f16 ~34 GB, expert staging ~2 GB, KV cache ~1 GB, Metal ~1 GB, OS ~10 GB, available ~50 GB. WARNING: f32 backbone pre-load = 68 GB, does NOT fit. WARNING: f32 path max 8.4 tok/s (physical limit at 546 GB/s).
14. **Scope Constraint** — "You are making SMALL, SAFE, TESTABLE optimizations within the existing architecture. If a change touches more than ~100 lines or requires a new subsystem, log it as PLANNED and exit. Architecture changes, bug fixes, and new subsystems are done in manual sessions."
15. **Research Context** — "Read files in research-context/ for deep background. Key files: stage3/04-stage3-findings.md (double-buffer + expert pool), stage2/01-audit-full.md (bugs), stage1/FINAL.md (10 architecture corrections)"
16. **Code Conventions** — same as AGENTS-INFER.md but with V3 paths (generate_v2.rs, mla_attention.rs, experiments-v3.tsv, etc.)

- [ ] **Step 3: Verify AGENTS-V3.md reads well**

Read it back, confirm all sections are present and data is accurate. Check that:
- All benchmark commands are V3-specific (not Qwen3)
- All file paths exist on the v3-optimize branch
- Bottleneck numbers match POST-MORTEM-v3-phase1.md
- Dead ends match experiments-v3.tsv

- [ ] **Step 4: Commit**

```bash
git add AGENTS-V3.md
git commit -m "docs: AGENTS-V3.md — V3 knowledge base for auto-optimize agents"
```

---

### Task 3: Create `optimize-v3.sh`

**Files:**
- Create: `system/optimize-v3.sh`
- Read: `system/optimize-infer.sh` (template — fork this)

Fork `optimize-infer.sh` with these changes:

- [ ] **Step 1: Copy the template**

```bash
cp system/optimize-infer.sh system/optimize-v3.sh
chmod +x system/optimize-v3.sh
```

- [ ] **Step 2: Update config section**

Change the top-of-file config block:

```bash
# --- Config ---
MAX_ITERS=10
COOLDOWN=15
MODEL="claude-opus-4-6"
AGENT_ID="alpha"
DRY_RUN=false
SHOW_STATUS=false
ITER_TIMEOUT_MIN=60
WORKTREE_BASE="/Users/dan/Dev/rustane-v3-auto"
BASE_BRANCH="v3-optimize"
RESEARCH_REPO="/Users/dan/Dev/rustane-research"
```

Derived paths (keep from template pattern):
```bash
WORKTREE="${WORKTREE_BASE}-${AGENT_ID}"
```

Key changes from optimize-infer.sh:
- `WORKTREE_BASE` → adjacent to repo (not `/tmp`)
- `BASE_BRANCH` → `v3-optimize` (not `rustane-infer`)
- Add `RESEARCH_REPO` variable
- Branch prefix → `v3-opt/auto-` (not `infer-opt/auto-`)

- [ ] **Step 3: Update file paths**

Update all path-derived variables:

```bash
BRANCH="v3-opt/auto-${AGENT_ID}"
LOGFILE="/tmp/rustane-v3-opt-${AGENT_ID}.log"
STOPFILE="/tmp/rustane-v3-STOP"
PAUSEFILE="/tmp/rustane-v3-PAUSE-${AGENT_ID}"
INJECTFILE="/tmp/rustane-v3-INJECT-${AGENT_ID}"
STATUSFILE="/tmp/rustane-v3-status-${AGENT_ID}"
```

- [ ] **Step 4: Update locked files list**

Replace the locked files array:

```bash
LOCKED_FILES=(
    "crates/moe-infer/tests/bench_v3_tok_per_sec.rs"
    "crates/moe-infer/tests/test_v3_validation.rs"
    "crates/moe-infer/tests/test_model_validation.rs"
    "configs/deepseek-v3.toml"
    "AGENTS-V3.md"
)
```

- [ ] **Step 5: Update worktree setup (no _BASE pattern)**

Replace the worktree setup section. Keep the `${WORKTREE}` pattern (derived from `${WORKTREE_BASE}-${AGENT_ID}`):

```bash
# Clean up stale worktree
if [ -d "$WORKTREE" ]; then
    log "Removing stale worktree at ${WORKTREE}"
    cd "$REPO_ROOT"
    git worktree remove --force "$WORKTREE" 2>/dev/null || rm -rf "$WORKTREE"
fi
```

- [ ] **Step 6: Add research context copy**

**Keep the existing `dev/` and `research/` copy blocks from the template unchanged.** Add the research-context block AFTER them:

```bash
# Copy research synthesis from private research repo
if [ -d "${RESEARCH_REPO}/mla-1t" ]; then
    log "Copying research synthesis to worktree..."
    RC="${WORKTREE}/research-context"
    mkdir -p "${RC}/stage0" "${RC}/stage1" "${RC}/stage2" "${RC}/stage3" "${RC}/stage4"

    # Root synthesis
    for f in 01-internal-architecture.md 05-post-stage2-assessment.md \
             testing-framework.md precision-notes.md model-comparison.md; do
        cp "${RESEARCH_REPO}/mla-1t/$f" "${RC}/" 2>/dev/null || true
    done

    # Stage 0 foundations
    cp "${RESEARCH_REPO}/mla-1t/stage0-foundations-2026-03-21/"*.md "${RC}/stage0/" 2>/dev/null || true

    # Stage 1 synthesis only
    for f in FINAL.md synthesis-wave1.md synthesis-wave2.md; do
        cp "${RESEARCH_REPO}/mla-1t/stage1-external-2026-03-21/$f" "${RC}/stage1/" 2>/dev/null || true
    done

    # Stage 2 post-mortem + synthesis
    for f in POST-MORTEM.md REFLECTIONS.md; do
        cp "${RESEARCH_REPO}/mla-1t/stage2-deepseekv3-execution-2026-03-21-1246/$f" "${RC}/stage2/" 2>/dev/null || true
    done
    cp "${RESEARCH_REPO}/mla-1t/stage2-deepseekv3-execution-2026-03-21-1246/post-mortem/"*.md "${RC}/stage2/" 2>/dev/null || true
    for f in FINAL.md synthesis-wave1.md synthesis-wave2.md; do
        cp "${RESEARCH_REPO}/mla-1t/stage2-deepseekv3-execution-2026-03-21-1246/stage2-deepseekv3-execution-external-research-2026-03-21-1246/$f" "${RC}/stage2/" 2>/dev/null || true
    done

    # Stage 3 all (small, all actionable)
    cp "${RESEARCH_REPO}/mla-1t/stage3-v3-runtime-2026-03-22/"*.md "${RC}/stage3/" 2>/dev/null || true

    # Stage 4 findings only
    cp "${RESEARCH_REPO}/mla-1t/stage4-heterogeneous-pipeline-2026-03-22/06-stage4-findings.md" "${RC}/stage4/" 2>/dev/null || true

    log "Research context copied ($(du -sh "${RC}" | awk '{print $1}'))"
else
    log "WARNING: research repo not found at ${RESEARCH_REPO}"
fi
```

- [ ] **Step 7: Replace the prompt**

Replace the entire PROMPT heredoc (lines 193-294 in the template) with the V3-specific prompt:

```
You are an autonomous inference optimization agent for DeepSeek-V3 (671B) on Apple M4 Max 128GB.
Your agent ID is: %%AGENT_ID%%
Your model: Opus with 1M token context. Use it — read everything, think deeply.

GOAL: Increase DeepSeek-V3 decode tok/s (currently 0.7, target 5+).
ANY measurable improvement is worth committing. Small wins compound.

TIME LIMIT: You have %%TIMEOUT%% minutes. Ideal iteration: 20-30 min. 60 min is the hard cap.
If implementation is taking too long, simplify or log as PLANNED and exit.

SCOPE: You are making SMALL, SAFE, TESTABLE optimizations within the existing architecture.
NOT architecture changes. NOT bug fixes. NOT new subsystems.
If a change touches >100 lines or requires a new subsystem, log as PLANNED and exit.
Think: buffer sizes, loop ordering, BLAS call batching, threading parameters,
allocation elimination, memory layout, constant tuning.

STEP 1 — READ CONTEXT (do this first, do not skip):
  - AGENTS-V3.md (THE source of truth — metric matrix, dead ends, bottlenecks, memory budget)
  - system/experiments-v3.tsv (every V3 experiment tried — do NOT repeat)
  - system/v3-phase2-optimize.md (ranked Phase 2 optimization targets)
  - crates/moe-infer/src/generate_v2.rs (THE hot path — V3 decode loop)
  - crates/moe-infer/src/mla_attention.rs (MLA forward pass)
  - crates/moe-infer/src/weights.rs (weight loading + buffer reuse)
  - crates/moe-infer/src/blas.rs (Accelerate BLAS FFI)
  - crates/expert-pager/src/pool.rs (expert pool — built but NOT wired)
  - crates/expert-pager/src/loader.rs (pread expert loader)

  Research synthesis (read for deep background):
  - research-context/stage3/04-stage3-findings.md (double-buffer design, expert pool sizing)
  - research-context/stage2/01-audit-full.md (bugs, showstoppers, 14x gap analysis)
  - research-context/stage2/03-architecture-10toks.md (10 tok/s blueprint, physical limits)
  - research-context/stage1/FINAL.md (10 architecture corrections from 9 research agents)
  - research-context/01-internal-architecture.md (MLA math, compute budget, weight tensors)

STEP 2 — THINK:
  What's the highest-leverage SMALL change? Consider:
  - Remaining targets from AGENTS-V3.md
  - Profiling data from experiments-v3.tsv (where is time actually spent?)
  - Research findings that suggest a small tweak (buffer size, chunk size, dispatch pattern)
  - Your own analysis of the code (unnecessary copies? suboptimal loop order? missed BLAS batch?)
  Write your plan to the status file:
    echo "THINKING: <hypothesis>" > /tmp/rustane-v3-status-%%AGENT_ID%%

STEP 3 — IMPLEMENT:
  Make your change. Read existing code before modifying.
  Keep changes focused on ONE optimization. Under 100 lines changed.
  Update status: echo "CODING: <what>" > /tmp/rustane-v3-status-%%AGENT_ID%%

STEP 3.5 — WRITE A CUSTOM TEST:
  File: crates/moe-infer/tests/auto_<experiment_name>.rs
  The test MUST:
  a) Capture output of the ORIGINAL code path
  b) Run the OPTIMIZED code path on same input
  c) Assert they match within tolerance:
     - Pure refactor: exact match
     - Reordered float ops: tolerance 1e-5
     - Precision change: tolerance 1e-3
     - Algorithmic change: tolerance 1e-2, justify in comment
  d) Test at least 2 edge cases specific to your change
  e) Run in under 10 seconds
  f) Doc comment explaining: what was optimized, what invariant, what failure means

STEP 4 — PASS THE METRIC MATRIX:
  Update status: echo "TESTING: metric matrix" > /tmp/rustane-v3-status-%%AGENT_ID%%

  4a. Your custom test:
      cargo test -p moe-infer --test auto_<name> --release -- --nocapture
      If fails: revert, skip to STEP 6.

  4b. Tier 1 hard gates:
      cargo build -p moe-infer --release
      cargo test -p moe-infer --release
      cargo test -p moe-infer --test test_v3_validation --release -- --ignored --nocapture
      cargo test -p moe-infer --test test_model_validation --release -- --ignored --nocapture
      If ANY fails: revert ALL changes, skip to STEP 6.

  4c. Tier 2 performance (MUST run 3 times, take median):
      Update status: echo "BENCHMARKING: tok/s (run 1/3)" > /tmp/rustane-v3-status-%%AGENT_ID%%
      cargo test -p moe-infer --test bench_v3_tok_per_sec --release -- --ignored --nocapture
      (repeat 2 more times, record warm decode tok/s from each)
      If warm decode tok/s regresses >5% from baseline in experiments-v3.tsv: revert, skip to STEP 6.

STEP 5 — LIE DETECTOR:
  Re-read the benchmark output from STEP 4c. The warm run prints per-phase timing.
  Your improvement MUST show up in a specific phase (prefill_ms or decode per-token time).
  If the tok/s change doesn't correspond to a visible component change, it's noise — revert.

STEP 6 — LOG RESULTS:
  Append ONE row to system/experiments-v3.tsv (tab-separated):
  date<TAB>experiment<TAB>variable<TAB>baseline<TAB>result<TAB>verdict<TAB>decode_toks<TAB>notes
  Verdict: IMPROVED, REVERTED, NO EFFECT, BROKEN, PLANNED, TIMEOUT
  The notes column MUST explain WHY (not just what).
  Always log, even failures — failed experiments are valuable data.

STEP 7 — COMMIT OR REVERT:
  If IMPROVED and all gates pass:
    git add changed source + auto_*.rs test + system/experiments-v3.tsv
    git commit -m "perf: <what> — <before> → <after> tok/s (<X>%)"
  If REVERTED/BROKEN/NO EFFECT:
    git checkout -- . (revert code + test)
    git add system/experiments-v3.tsv
    git commit -m "Log experiment: <name> (<verdict>)"

  Update status: echo "DONE: <verdict> <tok/s>" > /tmp/rustane-v3-status-%%AGENT_ID%%

RULES:
  - NEVER modify locked files (bench_v3_tok_per_sec.rs, test_v3_validation.rs, test_model_validation.rs, configs/deepseek-v3.toml, AGENTS-V3.md)
  - NEVER weaken test assertions to make tests pass
  - NEVER claim improvement without median-of-3 benchmark data
  - NEVER skip the metric matrix
  - NEVER modify existing experiments-v3.tsv rows — append only
  - NEVER make changes >100 lines — log as PLANNED instead
  - If stuck, log a PLANNED row with your hypothesis and exit cleanly
  - Before writing your test, read existing auto_*.rs tests for patterns
```

- [ ] **Step 8: Update the experiments sync path**

Change the sync line near end of loop from:
```bash
if [ -f "${WORKTREE}/system/experiments-infer.tsv" ]; then
    cp "${WORKTREE}/system/experiments-infer.tsv" "${REPO_ROOT}/system/experiments-infer.tsv" 2>/dev/null || true
fi
```
To:
```bash
if [ -f "${WORKTREE}/system/experiments-v3.tsv" ]; then
    cp "${WORKTREE}/system/experiments-v3.tsv" "${REPO_ROOT}/system/experiments-v3.tsv" 2>/dev/null || true
fi
```

- [ ] **Step 9: Add revert counter with auto-pause + macOS notification**

After the experiments sync and before the cooldown, add:

```bash
# Check for consecutive reverts
CONSECUTIVE_REVERTS=0
while IFS=$'\t' read -r _ _ _ _ _ verdict _ _; do
    case "$verdict" in
        reverted|REVERTED|BROKEN) CONSECUTIVE_REVERTS=$((CONSECUTIVE_REVERTS + 1)) ;;
        IMPROVED|keep|current-best) CONSECUTIVE_REVERTS=0 ;;
    esac
done < <(tail -5 "${WORKTREE}/system/experiments-v3.tsv" 2>/dev/null)

if [ "$CONSECUTIVE_REVERTS" -ge 3 ]; then
    log "AUTO-PAUSED: $CONSECUTIVE_REVERTS consecutive reverts — small optimizations may be exhausted"
    touch "$PAUSEFILE"
    osascript -e 'display notification "3 consecutive reverts — review needed" with title "rustane v3-auto" sound name "Ping"' 2>/dev/null || true
fi
```

- [ ] **Step 10: Update status mode for V3 paths**

In the `--status` block, update the experiments tail to use `experiments-v3.tsv` and update the title to "V3 Optimization Status".

- [ ] **Step 11: Update header comment**

Replace the file header comment with V3-specific usage:

```bash
#!/bin/bash
# optimize-v3.sh — Autonomous DeepSeek-V3 inference optimization loop
#
# Runs Claude Opus (1M context) in a git worktree. One experiment per iteration.
# Agent reads AGENTS-V3.md + research synthesis, picks small optimization,
# implements it, passes V3 metric matrix, commits or reverts.
#
# Usage:
#   ./system/optimize-v3.sh                          # 10 iters, opus, 60min timeout
#   ./system/optimize-v3.sh --iters 1                # single test iteration
#   ./system/optimize-v3.sh --iters 1 --dry-run      # print prompt, don't run
#   ./system/optimize-v3.sh --timeout 60             # 60 min hard timeout (default)
#   ./system/optimize-v3.sh --id beta                # agent ID (default: alpha)
#   ./system/optimize-v3.sh --status                 # show status and exit
#
# tmux (recommended):
#   tmux new -s v3-opt './system/optimize-v3.sh --iters 20'
#   Detach:  Ctrl-B then D
#   Reattach: tmux attach -t v3-opt
#
# Monitor:
#   tail -f /tmp/rustane-v3-opt-alpha.log
#   cat /tmp/rustane-v3-status-alpha
#   tail -20 system/experiments-v3.tsv
#
# Controls:
#   touch /tmp/rustane-v3-STOP                         # stop after current iter
#   touch /tmp/rustane-v3-PAUSE-alpha                  # pause after current iter
#   rm /tmp/rustane-v3-PAUSE-alpha                     # resume
#   echo "focus on X" > /tmp/rustane-v3-INJECT-alpha   # inject instructions
```

- [ ] **Step 12: Verify with dry-run**

Run: `./system/optimize-v3.sh --iters 1 --dry-run`

Expected output: the full prompt printed to stdout, worktree path, branch name, timeout. Verify:
- Prompt mentions DeepSeek-V3 (not Qwen3)
- Prompt references AGENTS-V3.md (not AGENTS-INFER.md)
- Prompt references experiments-v3.tsv (not experiments-infer.tsv)
- Prompt references bench_v3_tok_per_sec (not bench_tok_per_sec)
- Worktree path is `/Users/dan/Dev/rustane-v3-auto-alpha`
- Branch is `v3-opt/auto-alpha`

The dry-run creates then removes the worktree, so also verify:
- Research-context files were copied (check log output for "Research context copied")

- [ ] **Step 13: Commit**

```bash
git add system/optimize-v3.sh
git commit -m "feat: optimize-v3.sh — autonomous V3 inference optimization loop"
```

---

### Task 4: Create `monitor-v3.sh`

**Files:**
- Create: `system/monitor-v3.sh`

A simple dashboard script for a companion tmux pane.

- [ ] **Step 1: Write the monitor script**

```bash
#!/bin/bash
# monitor-v3.sh — Live dashboard for V3 auto-optimization
#
# Usage: ./system/monitor-v3.sh
# Or in tmux: tmux split-window './system/monitor-v3.sh'

AGENT_ID="${1:-alpha}"
STATUSFILE="/tmp/rustane-v3-status-${AGENT_ID}"
LOGFILE="/tmp/rustane-v3-opt-${AGENT_ID}.log"
WORKTREE="/Users/dan/Dev/rustane-v3-auto-${AGENT_ID}"
EXPERIMENTS="${WORKTREE}/system/experiments-v3.tsv"

watch -n 15 "
echo '=== DeepSeek-V3 Auto-Optimize Dashboard ==='
echo ''
echo 'Agent: '\"${AGENT_ID}\"
echo 'Status: '\$(cat ${STATUSFILE} 2>/dev/null || echo 'offline')
echo ''

# Check if loop is running
if pgrep -f 'optimize-v3.sh' > /dev/null 2>&1; then
    echo 'Loop: RUNNING'
else
    echo 'Loop: NOT RUNNING'
fi
if pgrep -f 'claude.*dangerously-skip-permissions' > /dev/null 2>&1; then
    echo 'Claude: ACTIVE'
else
    echo 'Claude: idle'
fi
echo ''

echo '--- Last 5 Experiments ---'
tail -5 ${EXPERIMENTS} 2>/dev/null | cut -f1-4,6-8 | column -t -s\$'\\t' || echo '(none)'
echo ''

echo '--- Recent Log (last 15 lines) ---'
tail -15 ${LOGFILE} 2>/dev/null || echo '(no log)'
"
```

- [ ] **Step 2: Make executable**

Run: `chmod +x system/monitor-v3.sh`

- [ ] **Step 3: Commit**

```bash
git add system/monitor-v3.sh
git commit -m "feat: monitor-v3.sh — live dashboard for V3 auto-optimize"
```

---

### Task 5: Dry-Run Validation

End-to-end verification that all pieces work together before launching for real.

**Files:**
- Read: `system/optimize-v3.sh`
- Read: `AGENTS-V3.md`
- Read: `system/experiments-v3.tsv`

- [ ] **Step 1: Dry-run the loop**

Run: `./system/optimize-v3.sh --iters 1 --dry-run`

Verify in the output:
1. Worktree was created at `/Users/dan/Dev/rustane-v3-auto-alpha`
2. Branch `v3-opt/auto-alpha` was created from `v3-optimize`
3. Research context was copied (log line shows size)
4. Prompt prints and contains all V3-specific content
5. Worktree was cleaned up after dry-run

- [ ] **Step 2: Verify research-context would be populated**

Before the dry-run cleans up, or run manually:

```bash
# Create worktree manually to test research copy
git worktree add /Users/dan/Dev/rustane-v3-auto-alpha v3-opt/auto-alpha 2>/dev/null || true

# Run just the copy section
RESEARCH_REPO="/Users/dan/Dev/rustane-research"
RC="/Users/dan/Dev/rustane-v3-auto-alpha/research-context"
mkdir -p "${RC}/stage0" "${RC}/stage1" "${RC}/stage2" "${RC}/stage3" "${RC}/stage4"
# (copy commands from optimize-v3.sh step 6)

# Verify
find "$RC" -name "*.md" | wc -l
# Expected: ~22 files

du -sh "$RC"
# Expected: ~190K

# Clean up
git worktree remove --force /Users/dan/Dev/rustane-v3-auto
```

- [ ] **Step 3: Verify AGENTS-V3.md references correct paths**

Grep for any Qwen3 references that shouldn't be there:

```bash
grep -i "qwen\|bench_tok_per_sec[^_v3]\|test_generation\|AGENTS-INFER\|experiments-infer" AGENTS-V3.md
```

Expected: zero matches. If any found, fix them.

- [ ] **Step 4: Verify experiments-v3.tsv is well-formed**

```bash
# Check column count is consistent
awk -F'\t' '{print NF}' system/experiments-v3.tsv | sort -u
# Expected: single number (8)

# Check no Qwen3 data leaked in
grep -i "qwen\|stage4-\|dequant\|pread-throughput" system/experiments-v3.tsv
# Expected: zero matches
```

---

### Task 6: Set Up Cron Supervisor

This runs AFTER launching the loop (Task 7). The cron fires every 15 minutes in the current session, reads the auto agent's experiments and log, and reports back in this conversation.

**Files:**
- Read: `system/experiments-v3.tsv` (in worktree)
- Read: `/tmp/rustane-v3-opt-alpha.log`
- Read: `/tmp/rustane-v3-status-alpha`

- [ ] **Step 1: Set up the cron**

Use `CronCreate` with a 15-minute schedule:

```
cron: "*/15 * * * *"
prompt: |
  You are supervising the V3 auto-optimize loop. Check its status:
  1. Read /tmp/rustane-v3-status-alpha (agent phase)
  2. Read the last 5 lines of /Users/dan/Dev/rustane-v3-auto-alpha/system/experiments-v3.tsv (recent experiments)
  3. Read the last 30 lines of /tmp/rustane-v3-opt-alpha.log (recent log)

  Report concisely:
  - Current agent phase (READING/CODING/TESTING/DONE/offline)
  - Last experiment result (name, verdict, tok/s)
  - Consecutive revert count
  - Any anomalies (errors, timeouts, stuck)

  If 3+ consecutive reverts: write "Small optimizations exhausted. Consider manual architecture session." to /tmp/rustane-v3-INJECT-alpha
  If agent has been in same phase >30 min: flag as potentially stuck
  If tok/s regressed below 0.5: write to /tmp/rustane-v3-PAUSE-alpha and alert
```

- [ ] **Step 2: Verify cron is registered**

The CronCreate tool returns a job ID. Note it for later deletion if needed.

---

### Task 7: Launch the Auto Loop

The final step — actually start the optimization loop.

- [ ] **Step 1: Launch in background**

Run in a background bash process (not tmux — we stay in this session as supervisor):

```bash
nohup ./system/optimize-v3.sh --iters 10 > /tmp/rustane-v3-launch.log 2>&1 &
```

Or if tmux is preferred for independent lifetime:
```bash
tmux new-session -d -s v3-opt './system/optimize-v3.sh --iters 10'
```

- [ ] **Step 2: Verify it started**

```bash
# Check process
pgrep -f "optimize-v3.sh"

# Check log
tail -5 /tmp/rustane-v3-opt-alpha.log

# Check worktree exists
ls /Users/dan/Dev/rustane-v3-auto-alpha/AGENTS-V3.md
```

- [ ] **Step 3: Verify research-context was copied**

```bash
ls /Users/dan/Dev/rustane-v3-auto-alpha/research-context/stage3/
# Expected: 04-stage3-findings.md, wave1-rq1..rq3.md
```

- [ ] **Step 4: Wait for first iteration status**

```bash
cat /tmp/rustane-v3-status-alpha
# Expected: READING (initially), then phases progress
```

The cron supervisor (Task 6) will monitor from here. Report status updates in this conversation as they come in.

---

## Post-Launch Checklist

After the loop is running and the cron is watching:

- [ ] First iteration completes without crash
- [ ] experiments-v3.tsv has a new row appended
- [ ] The new row has a reasonable verdict (IMPROVED, REVERTED, NO EFFECT, or PLANNED)
- [ ] If IMPROVED: verify the commit on `v3-opt/auto-alpha` looks sane
- [ ] If REVERTED: verify the worktree is clean (no uncommitted changes)
- [ ] Cron fires and reports status in this conversation
- [ ] Second iteration starts after cooldown
