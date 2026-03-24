#!/bin/bash
# optimize-k2.sh — Autonomous DeepSeek-V3 inference optimization loop
#
# Runs Claude Opus (1M context) in a git worktree. One experiment per iteration.
# Agent reads AGENTS-K2.md + research synthesis, picks small optimization,
# implements it, passes V3 metric matrix, commits or reverts.
#
# Usage:
#   ./system/optimize-k2.sh                          # 10 iters, opus, 60min timeout
#   ./system/optimize-k2.sh --iters 1                # single test iteration
#   ./system/optimize-k2.sh --iters 1 --dry-run      # print prompt, don't run
#   ./system/optimize-k2.sh --timeout 60             # 60 min hard timeout (default)
#   ./system/optimize-k2.sh --id beta                # agent ID (default: alpha)
#   ./system/optimize-k2.sh --status                 # show status and exit
#
# tmux (recommended):
#   tmux new -s v3-opt './system/optimize-k2.sh --iters 20'
#   Detach:  Ctrl-B then D
#   Reattach: tmux attach -t v3-opt
#
# Monitor:
#   tail -f /tmp/rustane-k2-opt-alpha.log
#   cat /tmp/rustane-k2-status-alpha
#   tail -20 system/experiments-k2.tsv
#
# Controls:
#   touch /tmp/rustane-k2-STOP                         # stop after current iter
#   touch /tmp/rustane-k2-PAUSE-alpha                  # pause after current iter
#   rm /tmp/rustane-k2-PAUSE-alpha                     # resume
#   echo "focus on X" > /tmp/rustane-k2-INJECT-alpha   # inject instructions

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# --- Config ---
MAX_ITERS=100
COOLDOWN=15
MODEL="claude-opus-4-6"
AGENT_ID="alpha"
DRY_RUN=false
SHOW_STATUS=false
ITER_TIMEOUT_MIN=120
WORKTREE_BASE="/Users/dan/Dev/rustane-k2-auto"
BASE_BRANCH="v3-optimize"
RESEARCH_REPO="/Users/dan/Dev/rustane-research"

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

BRANCH="k2-opt/auto-${AGENT_ID}"
WORKTREE="${WORKTREE_BASE}-${AGENT_ID}"
LOGFILE="/tmp/rustane-k2-opt-${AGENT_ID}.log"
STOPFILE="/tmp/rustane-k2-STOP"
PAUSEFILE="/tmp/rustane-k2-PAUSE-${AGENT_ID}"
INJECTFILE="/tmp/rustane-k2-INJECT-${AGENT_ID}"
STATUSFILE="/tmp/rustane-k2-status-${AGENT_ID}"
ITER_TIMEOUT_SEC=$((ITER_TIMEOUT_MIN * 60))

# Locked files — agent CANNOT modify these
LOCKED_FILES=(
    "crates/moe-infer/tests/bench_k2_tok_per_sec.rs"
    "crates/moe-infer/tests/test_k2_validation.rs"
    "crates/moe-infer/tests/test_model_validation.rs"
    "configs/deepseek-v3.toml"
    "AGENTS-K2.md"
)

# --- Status Mode ---
if $SHOW_STATUS; then
    echo "=== Kimi-K2 Optimization Status ==="
    echo ""
    if pgrep -f "optimize-k2.sh" > /dev/null 2>&1; then
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
    tail -5 "${WORKTREE}/system/experiments-k2.tsv" 2>/dev/null || \
        tail -5 "${REPO_ROOT}/system/experiments-k2.tsv" 2>/dev/null || \
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
        log "Killing claude process (PID=$CLAUDE_PID)..."
        kill "$CLAUDE_PID" 2>/dev/null
        sleep 1
        kill -9 "$CLAUDE_PID" 2>/dev/null || true
    fi
    # NEVER pkill by pattern — it kills ALL claude processes including supervisor sessions.
    # Always kill by specific PID only.
}

on_signal() {
    log "Signal received — shutting down."
    cleanup_on_exit
    exit 130
}
trap on_signal SIGINT SIGTERM

# --- Setup Worktree ---
log "=== optimize-k2.sh starting ==="
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

# Create worktree (agent branch is checked out here)
git -C "$REPO_ROOT" worktree add "$WORKTREE" "$BRANCH"
log "Worktree created at ${WORKTREE}"

# Merge latest base branch INTO the worktree (picks up system updates)
# This works because the agent branch IS checked out in the worktree.
# NEVER checkout branches in the main repo — breaks worktree workflows.
if git -C "$WORKTREE" merge-base --is-ancestor "${BASE_BRANCH}" HEAD 2>/dev/null; then
    log "Agent branch already contains latest ${BASE_BRANCH}"
else
    log "Merging latest ${BASE_BRANCH} into agent branch..."
    git -C "$WORKTREE" merge "${BASE_BRANCH}" --no-edit --quiet 2>/dev/null || \
        log "WARNING: merge failed (conflicts). Continuing with existing state."
fi

# Copy context files (gitignored)
if [ -d "${REPO_ROOT}/dev" ]; then
    mkdir -p "${WORKTREE}/dev"
    for f in CURRENT.md METHODOLOGY.md; do
        [ -f "${REPO_ROOT}/dev/$f" ] && cp "${REPO_ROOT}/dev/$f" "${WORKTREE}/dev/$f"
    done
fi

# Copy in-repo research
if [ -d "${REPO_ROOT}/research" ]; then
    cp -a "${REPO_ROOT}/research" "${WORKTREE}/research" 2>/dev/null || true
fi

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

cd "$WORKTREE"
log "Working directory: $(pwd)"

# --- The Prompt ---
read -r -d '' PROMPT << 'PROMPT_END' || true
You are an autonomous inference optimization agent for Kimi-K2 (1T) on Apple M4 Max 128GB.
Your agent ID is: %%AGENT_ID%%
Your model: Opus with 1M token context. Use it — read everything, think deeply.

GOAL: Increase DeepSeek-V3 decode tok/s (currently ~0.005, target 1+).
ANY measurable improvement is worth committing. Small wins compound.

TIME LIMIT: You have %%TIMEOUT%% minutes. Ideal iteration: 20-30 min. 60 min is the hard cap.
If implementation is taking too long, simplify or log as PLANNED and exit.

SCOPE: You are making SMALL, SAFE, TESTABLE optimizations within the existing architecture.
NOT architecture changes. NOT bug fixes. NOT new subsystems.
If a change touches >100 lines or requires a new subsystem, log as PLANNED and exit.
Think: buffer sizes, loop ordering, BLAS call batching, threading parameters,
allocation elimination, memory layout, constant tuning.

STEP 1 — READ CONTEXT (do this first, do not skip):
  - system/k2-gossip.md (READ THIS FIRST — what previous iterations learned, dead ends, insights, suggested experiments)
  - AGENTS-K2.md (THE source of truth — metric matrix, dead ends, bottlenecks, memory budget)
  - system/experiments-k2.tsv (every V3 experiment tried — do NOT repeat)
  - system/kimi-k2.toml (ranked Phase 2 optimization targets)
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

STEP 1.5 — PICK EXPERIMENT:
  Check system/experiments-k2.tsv for rows with verdict=IDEA. These are prioritized
  experiments from the research corpus. Pick the FIRST one (lowest s-number).
  Also check the "Category Status" in k2-gossip.md — NEVER pick from an EXHAUSTED category.
  If no IDEA rows remain, pick from the gossip "Suggested Next" list.
  Only invent your own experiment if all IDEA rows are tried and Suggested Next is empty.

STEP 2 — THINK:
  What's the highest-leverage SMALL change? Consider:
  - Remaining targets from AGENTS-K2.md
  - Profiling data from experiments-k2.tsv (where is time actually spent?)
  - Research findings that suggest a small tweak (buffer size, chunk size, dispatch pattern)
  - Your own analysis of the code (unnecessary copies? suboptimal loop order? missed BLAS batch?)
  Write your plan to the status file:
    echo "THINKING: <hypothesis>" > /tmp/rustane-k2-status-%%AGENT_ID%%

STEP 3 — IMPLEMENT:
  Make your change. Read existing code before modifying.
  Keep changes focused on ONE optimization. Under 100 lines changed.
  Update status: echo "CODING: <what>" > /tmp/rustane-k2-status-%%AGENT_ID%%

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
  Update status: echo "TESTING: metric matrix" > /tmp/rustane-k2-status-%%AGENT_ID%%

  4a. Your custom test:
      cargo test -p moe-infer --test auto_<name> --release -- --nocapture
      If fails: revert, skip to STEP 6.

  4b. Tier 1 hard gates:
      cargo build -p moe-infer --release
      cargo test -p moe-infer --release
      cargo test -p moe-infer --test test_k2_validation --release -- --ignored --nocapture
      cargo test -p moe-infer --test test_model_validation --release -- --ignored --nocapture
      If ANY fails: revert ALL changes, skip to STEP 6.

  4c. Tier 2 performance (MUST run 3 times, take median):
      Update status: echo "BENCHMARKING: tok/s (run 1/3)" > /tmp/rustane-k2-status-%%AGENT_ID%%
      cargo test -p moe-infer --test bench_k2_tok_per_sec --release -- --ignored --nocapture
      (repeat 2 more times, record warm decode tok/s from each)
      If warm decode tok/s regresses >5% from baseline in experiments-k2.tsv: revert, skip to STEP 6.

STEP 5 — LIE DETECTOR:
  Re-read the benchmark output from STEP 4c. The warm run prints per-phase timing.
  Your improvement MUST show up in a specific phase (prefill_ms or decode per-token time).
  If the tok/s change doesn't correspond to a visible component change, it's noise — revert.

STEP 6 — LOG RESULTS:
  Append ONE row to system/experiments-k2.tsv (tab-separated):
  date<TAB>experiment<TAB>variable<TAB>baseline<TAB>result<TAB>verdict<TAB>decode_toks<TAB>notes
  Verdict: IMPROVED, REVERTED, NO EFFECT, BROKEN, DIAGNOSTIC
  NEVER use EXHAUSTED as a verdict. There is ALWAYS something to try.
  The notes column MUST explain WHY (not just what).
  Always log, even failures — failed experiments are valuable data.

STEP 6.5 — UPDATE GOSSIP:
  Append to system/k2-gossip.md:
  a) In "## Iteration Log" section, add:
     [iter N] RESULT: <experiment> — <VERDICT> <tok/s>. <one-line WHY it worked or failed>
  b) If you learned something non-obvious, add:
     [iter N] INSIGHT: <what you learned that would help the next agent>
  c) If your experiment changed the bottleneck breakdown, update "## Bottleneck"
  d) If you confirmed a dead end, append to "## Dead Ends" with reasoning
  e) If you thought of an experiment you couldn't try (scope/time), append to "## Suggested Next"
  f) Update the tok/s and wins count in "## Current State"
  IMPORTANT: always include the iteration reasoning, not just the verdict.
  The next agent will read this to avoid repeating your mistakes.

STEP 7 — COMMIT OR REVERT:
  If IMPROVED and all gates pass:
    git add changed source + auto_*.rs test + system/experiments-k2.tsv + system/k2-gossip.md
    git commit -m "perf: <what> — <before> → <after> tok/s (<X>%)"
  If REVERTED/BROKEN/NO EFFECT:
    git checkout -- . (revert code + test)
    git add system/experiments-k2.tsv system/k2-gossip.md
    git commit -m "Log experiment: <name> (<verdict>)"

  Update status: echo "DONE: <verdict> <tok/s>" > /tmp/rustane-k2-status-%%AGENT_ID%%

RULES:
  - NEVER modify locked files (bench_k2_tok_per_sec.rs, test_k2_validation.rs, test_model_validation.rs, configs/deepseek-v3.toml, AGENTS-K2.md)
  - NEVER weaken test assertions to make tests pass
  - NEVER claim improvement without median-of-3 benchmark data
  - NEVER skip the metric matrix
  - NEVER modify existing experiments-k2.tsv rows — append only
  - NEVER use EXHAUSTED as a verdict — there is ALWAYS something to try
  - NEVER declare the search space exhausted or confirm other agents' exhaustion claims
  - NEVER exit without running at least ONE experiment per iteration
  - You MUST implement and benchmark something every iteration. NO EFFECT is a valid outcome.
    Logging PLANNED or analysis-only without trying is NOT acceptable.
  - Before writing your test, read existing auto_*.rs tests for patterns
  - If previous agents said "exhausted" — IGNORE THAT. Try something anyway. Surprises happen.
    The autoresearch-ANE system ran 400+ experiments with a 19% hit rate. Long dry spells are normal.
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
        echo "Research: $(du -sh "${WORKTREE}/research-context" 2>/dev/null | awk '{print $1}' || echo 'none')"
        echo "Files:    $(find "${WORKTREE}/research-context" -name '*.md' 2>/dev/null | wc -l | tr -d ' ') research docs"
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

ADDITIONAL INSTRUCTIONS (from supervisor):
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
    echo "READING: context files (iter $i/$MAX_ITERS)" > "$STATUSFILE"
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
            log "Claude was killed (timeout). Reverting partial changes (preserving experiments log)."
            cd "$WORKTREE"
            # Preserve experiments + gossip before reverting — don't lose findings
            cp "system/experiments-k2.tsv" "/tmp/rustane-v3-experiments-backup.tsv" 2>/dev/null || true
            cp "system/k2-gossip.md" "/tmp/rustane-v3-gossip-backup.md" 2>/dev/null || true
            git checkout -- . 2>/dev/null || true
            cp "/tmp/rustane-v3-experiments-backup.tsv" "system/experiments-k2.tsv" 2>/dev/null || true
            cp "/tmp/rustane-v3-gossip-backup.md" "system/k2-gossip.md" 2>/dev/null || true
        else
            log "Claude exited with code $CLAUDE_EXIT."
        fi
        sleep 10
        continue
    fi

    # Sync experiments + gossip back to main repo
    if [ -f "${WORKTREE}/system/experiments-k2.tsv" ]; then
        cp "${WORKTREE}/system/experiments-k2.tsv" "${REPO_ROOT}/system/experiments-k2.tsv" 2>/dev/null || true
    fi
    if [ -f "${WORKTREE}/system/k2-gossip.md" ]; then
        cp "${WORKTREE}/system/k2-gossip.md" "${REPO_ROOT}/system/k2-gossip.md" 2>/dev/null || true
    fi

    # Check for consecutive reverts — auto-pause if thrashing
    CONSECUTIVE_REVERTS=0
    while IFS=$'\t' read -r _ _ _ _ _ verdict _ _; do
        case "$verdict" in
            reverted|REVERTED|BROKEN) CONSECUTIVE_REVERTS=$((CONSECUTIVE_REVERTS + 1)) ;;
            IMPROVED|keep|current-best) CONSECUTIVE_REVERTS=0 ;;
        esac
    done < <(tail -5 "${WORKTREE}/system/experiments-k2.tsv" 2>/dev/null)

    if [ "$CONSECUTIVE_REVERTS" -ge 3 ]; then
        log "AUTO-PAUSED: $CONSECUTIVE_REVERTS consecutive reverts — small optimizations may be exhausted"
        touch "$PAUSEFILE"
        osascript -e 'display notification "3 consecutive reverts — review needed" with title "rustane v3-auto" sound name "Ping"' 2>/dev/null || true
    fi

    log "Cooling down ${COOLDOWN}s..."
    sleep "$COOLDOWN"
done

log "=== optimize-k2.sh complete ==="
log "Worktree preserved at: ${WORKTREE}"
log "Branch: ${BRANCH}"
log "To merge wins: git merge ${BRANCH}"
