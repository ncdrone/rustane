# System Lessons — Rustane Auto-Optimize

Operational lessons learned from running the autonomous optimization system.
Read this before modifying any system/ scripts. Updated after every incident.

---

## L1: Never pkill by pattern — kill by PID only

**Date:** 2026-03-22
**Severity:** CRITICAL
**Incident:** Ran `pkill -f "claude.*dangerously-skip-permissions"` to kill the auto-optimize agent. This killed ALL claude processes — including the supervisor session running in the same terminal. Total disruption.

**Root cause:** All Claude CLI sessions share the same process name. Pattern-based kill is a shotgun — it hits everything.

**Fix:**
- Removed `pkill -f` from `optimize-v3.sh` cleanup function
- Always kill by specific PID: `kill <pid>`
- To stop the loop gracefully: `touch /tmp/rustane-v3-STOP`
- To kill a stuck agent: find PID from log or `pgrep`, then `kill <pid>`

**Rule:** NEVER run `pkill -f` on any pattern that matches "claude". Use PID-based kills only.

---

## L2: Never git checkout in the main repo during worktree workflows

**Date:** 2026-03-22
**Severity:** HIGH
**Incident:** `optimize-v3.sh` setup tried to merge base branch into agent branch using `git checkout v3-opt/auto-alpha` in the main repo. This fails because: (a) git refuses to checkout a branch that's in a worktree, (b) the `checkout` / `checkout -` dance is fragile — if anything fails between them, the main repo is left on the wrong branch, (c) `set -euo pipefail` kills the script on the first failure.

**5 Whys:**
1. Script exits on relaunch → merge step crashes
2. Merge crashes → `git checkout` fails in main repo
3. Checkout used → merge logic runs BEFORE worktree exists
4. Merge needed → agent branch (perf commits) diverged from base (system commits)
5. Wrong location → should merge INSIDE the worktree where the branch IS checked out

**Fix:** Move merge to AFTER `git worktree add`. Inside the worktree, the agent branch is already checked out — `git merge v3-optimize` works naturally. Added `merge-base --is-ancestor` check to skip if already up-to-date.

**Rule:** In worktree workflows, NEVER modify branch state from the main repo. All branch operations (merge, commit, checkout) happen INSIDE the worktree.

---

## L3: 60-minute timeout is too short for V3 inference optimization

**Date:** 2026-03-22
**Severity:** MEDIUM
**Incident:** 2 of 6 iterations (33%) timed out at 60 minutes. V3 benchmarks require loading 34 GB of weights (~30s), then running cold+warm passes (20 tokens each at 0.7 tok/s = ~60s per run). With median-of-3 benchmarks, that's ~4-5 min of pure benchmark time. Add ~10 min for context reading, ~15-25 min for coding/testing, and 60 min is razor thin.

**Fix:** Changed default `ITER_TIMEOUT_MIN` from 60 to 120.

**Rule:** For V3 (671B model), use 120 min timeout minimum. Budget: 10 min read + 20 min code + 15 min test + 15 min benchmark + 60 min margin.

---

## L4: Timeout reverts lose experiment data

**Date:** 2026-03-22
**Severity:** MEDIUM
**Incident:** When the watchdog kills Claude and the script runs `git checkout -- .`, it reverts ALL changes including `experiments-v3.tsv` rows the agent partially wrote. The agent's findings are lost — not even logged as TIMEOUT.

**Fix:** Backup `experiments-v3.tsv` and `v3-gossip.md` before reverting, restore after. The code changes are reverted but the experiment log and gossip are preserved.

**Rule:** Any file that accumulates state across iterations (experiments TSV, gossip) must be backed up before `git checkout -- .` and restored after.

---

## L5: Agents forget between iterations — need a gossip file

**Date:** 2026-03-22
**Severity:** MEDIUM
**Incident:** Each `claude -p` invocation is stateless. The agent reads `experiments-v3.tsv` to know WHAT was tried, but the notes column is too small for WHY things failed. Result: 3 separate f16 bypass attempts, each rediscovering the same dead end (single-core conversion can't beat multi-core rayon).

**Fix:** Created `system/v3-gossip.md` — a living document the agent reads FIRST and appends to after each experiment. Contains: current bottleneck breakdown, confirmed dead ends with reasoning, key insights, suggested next experiments, bug reports.

**Rule:** For any stateless agent loop, provide a gossip/findings file that accumulates reasoning across iterations. The TSV logs what happened; the gossip explains why.

---

## L6: Branch divergence is expected — plan for merges, not fast-forwards

**Date:** 2026-03-22
**Severity:** LOW
**Incident:** Used `--ff-only` merge initially, which fails when branches diverge (agent has perf commits, base has system commits). Changed to regular merge.

**Root cause:** In a system where the operator (you) commits to the base branch AND the agent commits to the agent branch, divergence is the normal state. Fast-forward only works for one-way flows.

**Fix:** Use `git merge --no-edit` instead of `--ff-only`. Check `merge-base --is-ancestor` first to skip unnecessary merges.

**Rule:** When two actors commit to related branches, always use merge (not ff-only). Check ancestry first to avoid unnecessary merge commits.

---

## L7: Commit synced-back files before relaunching

**Date:** 2026-03-22
**Severity:** LOW
**Incident:** The script syncs `experiments-v3.tsv` and `v3-gossip.md` back to the main repo after each iteration, but these are unstaged changes. If you relaunch without committing them to `v3-optimize`, the worktree merge won't include them (they're not in any commit).

**Fix:** Before relaunching, always `git add system/experiments-v3.tsv system/v3-gossip.md && git commit`. Or add this to the script as a pre-launch step.

**Rule:** Check `git status` on v3-optimize before relaunching. Commit any synced-back files.

---

## L8: Check for merge conflict markers after worktree merge

**Date:** 2026-03-22
**Severity:** HIGH
**Incident:** The worktree merge of v3-optimize into v3-opt/auto-alpha produced conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`) in experiments-v3.tsv. Both branches had appended different rows to the same file. The script's `git merge --no-edit` silently left markers in the file. Agent would have read a corrupted TSV.

**Fix:** After merge, check for conflict markers: `grep -l '<<<<<<' system/*.tsv system/*.md`. Resolve manually if found. Could also add this check to the script after the merge step.

**Rule:** Always verify merge results in accumulating files (TSV, gossip). These are the most likely to conflict when both sides append.

---

## General Principles

1. **Stateless agents need filesystem memory.** Gossip files, experiment logs, and state files bridge the gap between invocations.
2. **Kill by PID, never by pattern.** Multiple Claude processes share the same name.
3. **Worktree operations stay in the worktree.** Never modify branch state from the main repo.
4. **Preserve accumulating state across reverts.** Backup before `git checkout -- .`, restore after.
5. **Budget timeouts for the full cycle.** Include model load time, benchmark runs, and margin.
6. **Test with `--dry-run` before real launches.** Catches setup issues without burning iteration time.
7. **Plan for branch divergence.** Two actors = merge, not fast-forward.
