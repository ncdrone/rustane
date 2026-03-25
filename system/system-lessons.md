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

## L9: ALWAYS profile before optimizing — measure, don't guess

**Date:** 2026-03-23
**Severity:** CRITICAL
**Incident:** Let K2 optimizer run ~10 iterations targeting CPU-side MLA tricks. Real profiling revealed MLA is only 12.5% of layer time — the bottleneck is FFN+Metal+pread (87.5%) which we hadn't instrumented. Also assumed V3's 90% expert cache hit rate would transfer to K2 — actual K2 hit rate is 54.1% (384 experts are much more sparse than V3's 256).

**Fix:** Before launching any optimizer on a new model:
1. Run RUSTANE_MLA_PROFILE=1 + RUSTANE_POOL_SIM=1 first
2. Instrument every code path (MLA, FFN, Metal dispatch, pread)
3. Get real per-component numbers, THEN optimize the biggest one
4. Never assume one model's profile transfers to another

**Rule:** The first experiment is always DIAGNOSTIC. Profile first, optimize second.

---

## General Principles

1. **Stateless agents need filesystem memory.** Gossip files, experiment logs, and state files bridge the gap between invocations.
2. **Kill by PID, never by pattern.** Multiple Claude processes share the same name.
3. **Worktree operations stay in the worktree.** Never modify branch state from the main repo.
4. **Preserve accumulating state across reverts.** Backup before `git checkout -- .`, restore after.
5. **Budget timeouts for the full cycle.** Include model load time, benchmark runs, and margin.
6. **Test with `--dry-run` before real launches.** Catches setup issues without burning iteration time.
7. **Plan for branch divergence.** Two actors = merge, not fast-forward.

## Lesson 10: Three-Layer Caching on Apple Silicon (K2)
**Problem:** K2 benchmark results varied wildly (0.30 to 1.68 tok/s) across sessions.
**Root cause:** Three cache layers, all sharing 128 GB unified memory:
1. OS page cache (RAM) — evicted by Claude agent processes (~1-2 GB)
2. SSD controller DRAM cache (1-4 GB on Apple Fabric) — needs warmup runs to populate
3. NAND flash (cold reads at 7.4 GB/s)

Cold SSD controller = 0.51 tok/s. Warm = 1.68 tok/s. 3.5x difference.
F_NOCACHE bypasses OS page cache (layer 1) but experts still benefit from SSD controller cache (layer 2).

**Fix:** Benchmark protocol:
- Kill Claude agent before benchmarking (bash loop handles bench after agent exits)
- 2 warmup runs to populate SSD controller cache
- 3 measured runs, take median
- Apple's internal SSD uses "Apple Fabric" protocol, not standard NVMe

## L11: NEVER push to remote without explicit user permission

**Date:** 2026-03-24

**Incident:** Claude pushed `1t-moe-infer` branch and two research commits to remote without asking. User did not authorize any remote pushes.

**Root cause:** Claude assumed pushing a new branch was safe because it didn't touch existing branches. But ANY push to remote is a visible action that affects shared state and should require explicit authorization.

**Fix:** NEVER run `git push` in any form — `push`, `push -u`, `push origin` — unless the user explicitly says "push it" or "push to remote." This applies to ALL repos (rustane, rustane-research, any repo). Local commits are fine. Remote pushes require explicit permission every time. No exceptions.
