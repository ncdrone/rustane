#!/bin/bash
# optimize-infer.sh — Autonomous inference optimization loop
#
# Runs Claude Opus (1M context) in a git worktree. One experiment per iteration.
# Agent reads full codebase + research corpus, picks highest-leverage change,
# implements it, passes metric matrix, commits or reverts.
#
# Usage:
#   ./system/optimize-infer.sh                          # 10 iters, opus, 60min timeout
#   ./system/optimize-infer.sh --iters 1                # single iteration (test the system)
#   ./system/optimize-infer.sh --iters 1 --dry-run      # print prompt, don't run
#   ./system/optimize-infer.sh --timeout 60             # 60 min hard timeout (default)
#   ./system/optimize-infer.sh --model opus             # model (default: opus)
#   ./system/optimize-infer.sh --id beta                # agent ID (default: alpha)
#   ./system/optimize-infer.sh --status                 # show status and exit
#
# tmux (recommended):
#   tmux new -s infer-opt './system/optimize-infer.sh --iters 20'
#   Detach:  Ctrl-B then D
#   Reattach: tmux attach -t infer-opt
#
# Monitor:
#   tail -f /tmp/rustane-infer-opt-alpha.log
#   cat /tmp/rustane-infer-status-alpha
#   tail -20 system/experiments-infer.tsv

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# --- Config ---
MAX_ITERS=10
COOLDOWN=15
MODEL="claude-opus-4-6"
AGENT_ID="alpha"
DRY_RUN=false
SHOW_STATUS=false
ITER_TIMEOUT_MIN=60
WORKTREE_BASE="/tmp/rustane-infer-opt"
BASE_BRANCH="rustane-infer"

# --- Parse Args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=true; shift ;;
        --status)    SHOW_STATUS=true; shift ;;
        --model)     MODEL="$2"; shift 2 ;;
        --iters)     MAX_ITERS="$2"; shift 2 ;;
        --timeout)   ITER_TIMEOUT_MIN="$2"; shift 2 ;;
        --id)        AGENT_ID="$2"; shift 2 ;;
        *)           echo "Unknown arg: $1"; exit 1 ;;
    esac
done

BRANCH="infer-opt/auto-${AGENT_ID}"
WORKTREE="${WORKTREE_BASE}-${AGENT_ID}"
LOGFILE="/tmp/rustane-infer-opt-${AGENT_ID}.log"
STOPFILE="/tmp/rustane-infer-STOP"
PAUSEFILE="/tmp/rustane-infer-PAUSE-${AGENT_ID}"
INJECTFILE="/tmp/rustane-infer-INJECT-${AGENT_ID}"
STATUSFILE="/tmp/rustane-infer-status-${AGENT_ID}"
ITER_TIMEOUT_SEC=$((ITER_TIMEOUT_MIN * 60))

# Locked files — agent CANNOT modify these
LOCKED_FILES=(
    "crates/moe-infer/tests/test_generation.rs"
    "crates/moe-infer/tests/bench_tok_per_sec.rs"
    "crates/moe-infer/tests/bench_decode_layer_breakdown.rs"
    "crates/moe-infer/tests/test_fused_metal.rs"
    "crates/moe-infer/tests/bench_metal_dispatch_latency.rs"
    "crates/moe-infer/tests/bench_blas_sizes.rs"
    "configs/qwen3-moe-30b.toml"
    "AGENTS-INFER.md"
)

# --- Status Mode ---
if $SHOW_STATUS; then
    echo "=== Rustane Inference Optimization Status ==="
    echo ""
    if pgrep -f "optimize-infer.sh" > /dev/null 2>&1; then
        echo "Loop: RUNNING"
    else
        echo "Loop: NOT RUNNING"
    fi
    if pgrep -f "claude.*dangerously-skip-permissions" > /dev/null 2>&1; then
        echo "Claude: RUNNING"
    else
        echo "Claude: idle"
    fi
    echo ""
    echo "--- Agent Status ---"
    cat "$STATUSFILE" 2>/dev/null || echo "(no status file)"
    echo ""
    echo "--- Latest Experiments ---"
    tail -5 "${WORKTREE}/system/experiments-infer.tsv" 2>/dev/null || \
        tail -5 "${REPO_ROOT}/system/experiments-infer.tsv" 2>/dev/null || \
        echo "(no experiments)"
    echo ""
    echo "--- Worktree ---"
    if [ -d "$WORKTREE" ]; then
        echo "Path: $WORKTREE"
        echo "Branch: $(git -C "$WORKTREE" branch --show-current 2>/dev/null || echo 'unknown')"
        echo "Last commit: $(git -C "$WORKTREE" log --oneline -1 2>/dev/null || echo 'none')"
    else
        echo "No worktree at $WORKTREE"
    fi
    exit 0
fi

# --- Functions ---
log() {
    local msg="[$(date '+%H:%M:%S')] [${AGENT_ID}] $*"
    echo "$msg" | tee -a "$LOGFILE"
}

check_locked_files() {
    local violations=0
    for f in "${LOCKED_FILES[@]}"; do
        if ! git -C "$WORKTREE" diff --quiet "$f" 2>/dev/null; then
            log "VIOLATION: agent modified locked file $f"
            violations=$((violations + 1))
        fi
    done
    if [ $violations -gt 0 ]; then
        log "REVERTING ALL CHANGES — $violations locked file violations"
        cd "$WORKTREE"
        git checkout -- . 2>/dev/null || true
        return 1
    fi
    return 0
}

cleanup_on_exit() {
    if [ -n "${WATCHDOG_PID:-}" ]; then
        kill "$WATCHDOG_PID" 2>/dev/null
        wait "$WATCHDOG_PID" 2>/dev/null || true
    fi
    if [ -n "${CLAUDE_PID:-}" ] && kill -0 "$CLAUDE_PID" 2>/dev/null; then
        log "Killing claude process..."
        kill "$CLAUDE_PID" 2>/dev/null
        sleep 1
        kill -9 "$CLAUDE_PID" 2>/dev/null || true
    fi
    pkill -f "claude.*dangerously-skip-permissions" 2>/dev/null || true
}

on_signal() {
    log "Signal received — shutting down."
    cleanup_on_exit
    exit 130
}
trap on_signal SIGINT SIGTERM

# --- Setup Worktree ---
log "=== optimize-infer.sh starting ==="
log "Model: ${MODEL} | Max: ${MAX_ITERS} iters | Timeout: ${ITER_TIMEOUT_MIN}min | Agent: ${AGENT_ID}"

# Clean up stale worktree
if [ -d "$WORKTREE" ]; then
    log "Removing stale worktree at ${WORKTREE}"
    cd "$REPO_ROOT"
    git worktree remove --force "$WORKTREE" 2>/dev/null || rm -rf "$WORKTREE"
fi

# Create or reset agent branch from base branch
if ! git -C "$REPO_ROOT" show-ref --quiet "refs/heads/${BRANCH}"; then
    git -C "$REPO_ROOT" branch "$BRANCH" "${BASE_BRANCH}"
    log "Created branch ${BRANCH} from ${BASE_BRANCH}"
else
    log "Reusing existing branch ${BRANCH}"
fi

# Create worktree
git -C "$REPO_ROOT" worktree add "$WORKTREE" "$BRANCH"
log "Worktree created at ${WORKTREE}"

# Copy context files (gitignored)
if [ -d "${REPO_ROOT}/dev" ]; then
    mkdir -p "${WORKTREE}/dev"
    for f in CURRENT.md METHODOLOGY.md; do
        [ -f "${REPO_ROOT}/dev/$f" ] && cp "${REPO_ROOT}/dev/$f" "${WORKTREE}/dev/$f"
    done
fi

# Copy research corpus
if [ -d "${REPO_ROOT}/research" ]; then
    cp -a "${REPO_ROOT}/research" "${WORKTREE}/research" 2>/dev/null || true
fi

cd "$WORKTREE"
log "Working directory: $(pwd)"

# --- The Prompt ---
read -r -d '' PROMPT << 'PROMPT_END' || true
You are an autonomous inference optimization agent for rustane on Apple M4 Max 128GB.
Your agent ID is: %%AGENT_ID%%
Your model: Opus with 1M token context. Use it — read everything, think deeply.

GOAL: Increase Qwen3-MoE-30B decode tok/s (currently 19.6) and/or reduce prefill latency (currently 310ms).
ANY measurable improvement is worth committing. Small wins compound.

TIME LIMIT: You have %%TIMEOUT%% minutes. Ideal iteration: 20-30 min. 60 min is the hard cap.
If implementation is taking too long, simplify or log as PLANNED and exit.

STEP 1 — READ CONTEXT (do this first, do not skip):
  - AGENTS-INFER.md (THE source of truth — metric matrix, dead ends, what works, hardware facts)
  - system/experiments-infer.tsv (every experiment tried — do NOT repeat)
  - research/stage5-plan-2026-03-21.md (ranked optimization targets with estimates)
  - crates/moe-infer/src/generate.rs (decode loop — the hot path)
  - crates/moe-kernels/src/dequant.rs (Metal shaders — fused kernel, ROWS_PER_TG=8)
  - crates/moe-infer/src/attention.rs (CPU attention — 38% of decode time)
  - crates/moe-infer/src/blas.rs (BLAS bindings — sgemv, sgemm)
  - dev/research/02-ane-inference-landscape.md (ane-infer and uzu patterns)
  - CREDITS.md (reference implementations)

STEP 2 — THINK:
  What's the highest-leverage change? Consider:
  - Remaining targets from AGENTS-INFER.md "Remaining Optimization Targets"
  - Could be a small tuning change (shader constant, buffer size, dispatch pattern)
  - Could be a large architectural change (Metal attention, batched QKV, new kernel)
  - You have 1M context — you can hold the entire codebase + research simultaneously
  Write your plan to the status file:
    echo "THINKING: <hypothesis>" > /tmp/rustane-infer-status-%%AGENT_ID%%

STEP 3 — IMPLEMENT:
  Make your change. Read existing code before modifying.
  Keep changes focused on ONE optimization.
  Update status: echo "CODING: <what>" > /tmp/rustane-infer-status-%%AGENT_ID%%

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
  Update status: echo "TESTING: metric matrix" > /tmp/rustane-infer-status-%%AGENT_ID%%

  4a. Your custom test:
      cargo test -p moe-infer --test auto_<name> --release -- --nocapture
      If fails: revert, skip to STEP 6.

  4b. Tier 1 hard gates:
      cargo build -p moe-infer --release
      cargo test -p moe-infer --release
      cargo test -p moe-infer --test test_fused_metal --release
      cargo test -p moe-infer --test test_generation --release -- test_generation_matches_hf --ignored
      If ANY fails: revert ALL changes, skip to STEP 6.

  4c. Tier 2 performance (MUST run 3 times, take median):
      Update status: echo "BENCHMARKING: tok/s (run 1/3)" > /tmp/rustane-infer-status-%%AGENT_ID%%
      cargo test -p moe-infer --test bench_tok_per_sec --release -- --ignored --nocapture
      (repeat 2 more times)
      If decode tok/s regresses >3% from baseline: revert, skip to STEP 6.

STEP 5 — RECORD TIER 3 (lie detector):
  cargo test -p moe-infer --test bench_decode_layer_breakdown --release -- --ignored --nocapture
  Your improvement MUST show up in per-component numbers.
  If the breakdown doesn't explain the improvement, it's noise — revert.

STEP 6 — LOG RESULTS:
  Append ONE row to system/experiments-infer.tsv (tab-separated):
  date<TAB>experiment<TAB>variable<TAB>baseline<TAB>result<TAB>verdict<TAB>decode_toks<TAB>prefill_ms<TAB>p95_ms<TAB>metal_ms<TAB>attn_ms<TAB>notes
  Verdict: IMPROVED, REVERTED, NO EFFECT, BROKEN, TIMEOUT
  The notes column MUST explain WHY (not just what).
  Always log, even failures.

STEP 7 — COMMIT OR REVERT:
  If IMPROVED and all gates pass:
    git add changed source + auto_*.rs test + system/experiments-infer.tsv
    git commit -m "perf: <what> — <before> → <after> tok/s (<X>%)"
  If REVERTED/BROKEN/NO EFFECT:
    git checkout -- . (revert code + test)
    git add system/experiments-infer.tsv
    git commit -m "Log experiment: <name> (<verdict>)"

  Update status: echo "DONE: <verdict> <tok/s>" > /tmp/rustane-infer-status-%%AGENT_ID%%

RULES:
  - NEVER modify locked files (test_generation.rs, bench_*.rs, configs/*, AGENTS-INFER.md)
  - NEVER weaken test assertions to make tests pass
  - NEVER claim improvement without median-of-3 benchmark data
  - NEVER skip the metric matrix
  - NEVER modify existing experiments-infer.tsv rows — append only
  - If stuck, log a PLANNED row with your hypothesis and exit cleanly
  - Before writing your test, read existing auto_*.rs tests for patterns
PROMPT_END

# Inject agent ID and timeout
PROMPT="${PROMPT//%%AGENT_ID%%/$AGENT_ID}"
PROMPT="${PROMPT//%%TIMEOUT%%/$ITER_TIMEOUT_MIN}"

# --- Main Loop ---
for i in $(seq 1 "$MAX_ITERS"); do
    ITER_START=$(date +%s)
    log "--- Iteration $i / $MAX_ITERS (timeout: ${ITER_TIMEOUT_MIN}min) ---"

    if $DRY_RUN; then
        echo ""
        echo "=== PROMPT ==="
        echo "$PROMPT"
        echo "==============="
        echo ""
        echo "Worktree: ${WORKTREE}"
        echo "Branch:   ${BRANCH}"
        echo "Timeout:  ${ITER_TIMEOUT_MIN} min"
        cd "$REPO_ROOT"
        git worktree remove --force "$WORKTREE" 2>/dev/null || true
        exit 0
    fi

    # Check stop/pause
    if [[ -f "$STOPFILE" ]]; then
        log "STOP file found. Exiting."
        rm -f "$STOPFILE"
        break
    fi
    if [[ -f "$PAUSEFILE" ]]; then
        log "PAUSED — remove $PAUSEFILE to resume"
        while [[ -f "$PAUSEFILE" ]]; do
            sleep 5
            [[ -f "$STOPFILE" ]] && { rm -f "$STOPFILE" "$PAUSEFILE"; break 2; }
        done
        log "RESUMED"
    fi

    # Check inject
    ITER_PROMPT="$PROMPT"
    if [[ -f "$INJECTFILE" ]]; then
        INJECT_CONTENT=$(cat "$INJECTFILE")
        rm -f "$INJECTFILE"
        log "INJECT: $INJECT_CONTENT"
        ITER_PROMPT="${PROMPT}

ADDITIONAL INSTRUCTIONS (from human):
${INJECT_CONTENT}"
    fi

    # Watchdog timer
    (
        sleep "$ITER_TIMEOUT_SEC"
        PIDS=$(pgrep -f "Your agent ID is: ${AGENT_ID}" 2>/dev/null || true)
        if [ -n "$PIDS" ]; then
            echo "[$(date '+%H:%M:%S')] [${AGENT_ID}] TIMEOUT: killed after ${ITER_TIMEOUT_MIN}min" | tee -a "$LOGFILE"
            for pid in $PIDS; do kill "$pid" 2>/dev/null; done
            sleep 2
            for pid in $PIDS; do kill -9 "$pid" 2>/dev/null || true; done
        fi
    ) &
    WATCHDOG_PID=$!

    # Run Claude
    log "Launching claude -p --model ${MODEL} ..."
    echo "READING" > "$STATUSFILE"
    set +e
    claude -p \
        --dangerously-skip-permissions \
        --model "$MODEL" \
        --effort max \
        "$ITER_PROMPT" 2>&1 | tee -a "$LOGFILE"
    CLAUDE_EXIT=${PIPESTATUS[0]}
    set -e

    # Cancel watchdog
    kill "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true
    WATCHDOG_PID=""

    # Post-iteration: verify locked files
    log "Checking locked files..."
    if ! check_locked_files; then
        log "Locked file violation detected — changes reverted"
    fi

    # Report timing
    ITER_END=$(date +%s)
    ITER_DURATION=$(( ITER_END - ITER_START ))
    ITER_MIN=$(( ITER_DURATION / 60 ))
    ITER_SEC=$(( ITER_DURATION % 60 ))
    log "Iteration $i took ${ITER_MIN}m${ITER_SEC}s (exit=$CLAUDE_EXIT)"

    if [ $CLAUDE_EXIT -ne 0 ]; then
        if [ $CLAUDE_EXIT -eq 137 ] || [ $CLAUDE_EXIT -eq 143 ]; then
            log "Claude was killed (timeout). Reverting partial changes."
            cd "$WORKTREE"
            git checkout -- . 2>/dev/null || true
        else
            log "Claude exited with code $CLAUDE_EXIT."
        fi
        sleep 10
        continue
    fi

    # Sync experiments-infer.tsv back
    if [ -f "${WORKTREE}/system/experiments-infer.tsv" ]; then
        cp "${WORKTREE}/system/experiments-infer.tsv" "${REPO_ROOT}/system/experiments-infer.tsv" 2>/dev/null || true
    fi

    log "Cooling down ${COOLDOWN}s..."
    sleep "$COOLDOWN"
done

log "=== optimize-infer.sh complete ==="
log "Worktree preserved at: ${WORKTREE}"
log "Branch: ${BRANCH}"
log "To merge wins: git merge ${BRANCH}"
