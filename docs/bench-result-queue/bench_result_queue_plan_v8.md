# Bench Result Queue — Implementation Plan

## Goal

Replace single-file overwrite (`target/bench-result.json`) with directory queue (`target/bench-results/`). `make submit` submits all pending results in order, deletes each only after successful POST.

---

## Changes

### [MODIFY] [bench_result.rs](file:///Users/USER/RustRover-Projects/rustane/crates/engine/src/bench_result.rs)

Change `write_result` (L96):
- Filename: `<YYYYMMDD_HHMMSS>_<benchmark-slug>_<fingerprint[..8]>.json` (e.g. `20260326_111500_train-600m_a3f7c012.json`). The 8-char fingerprint prefix makes collisions extremely unlikely for same-second results (byte-identical results sharing a timestamp would collide, but that implies an idempotent duplicate).
- **Slug sanitization:** `benchmark` field mapped to `[A-Za-z0-9._-]` with all other characters replaced by `_`. Applied in `write_result` before constructing the filename.
- **Atomic write:** write to a temp file (`.tmp` extension) in the same directory, then `std::fs::rename` into the final `.json` name. A crash mid-write leaves only a `.tmp` file, which `make submit` ignores (only globs `*.json`).
- `create_dir_all("target/bench-results/")`
- Print: `"📊 Result queued in target/bench-results/  (N pending)"`

No changes to `BenchResult` struct, callers, fingerprint, or any other function. The 5 existing call sites (`bench_fwd_only_scale.rs`, `bench_forward_multistream.rs`, `bench_param_sweep.rs`, `theory_runner.rs`) call `write_result` unchanged.

---

### [MODIFY] [Makefile](file:///Users/USER/RustRover-Projects/rustane/Makefile) — `submit` target (L111–149)

Replace the current single-file submit with a loop:

```makefile
submit:
	@if [ -z "$$(ls target/bench-results/*.json 2>/dev/null)" ]; then \
		echo "No pending results. Run a benchmark first."; exit 1; \
	fi
	@mkdir -p target/bench-results/submitted
	@COUNT=$$(ls target/bench-results/*.json | wc -l | tr -d ' '); \
	echo ""; \
	echo "$$COUNT pending result(s) in target/bench-results/"; \
	echo ""; \
	read -p "Your name: " NAME; \
	read -p "X handle (optional): " XHANDLE; \
	echo ""; \
	SUBMITTED=0; \
	for f in $$(ls target/bench-results/*.json | sort); do \
		BENCH=$$(jq -r '.benchmark' "$$f"); \
		echo "Submitting $$BENCH ($$f)..."; \
		TMPFILE=$$(mktemp); \
		jq --arg name "$$NAME" --arg x "$$XHANDLE" \
			'.submitter = {name: $$name, x_handle: $$x}' "$$f" > "$$TMPFILE"; \
		RESPONSE=$$(curl -s -X POST $(LEADERBOARD_API)/api/submit \
			-H "Content-Type: application/json" -d @"$$TMPFILE"); \
		rm -f "$$TMPFILE"; \
		RESULT_ID=$$(echo "$$RESPONSE" | jq -r '.id // empty'); \
		if [ -n "$$RESULT_ID" ]; then \
			PARAMS=$$(jq -r '.config.params_m' "$$f"); \
			TOKS=$$(jq -r '.results.tok_per_s' "$$f"); \
			MS=$$(jq -r '.results.ms_per_step' "$$f"); \
			FWD=$$(jq -r '.results.ms_fwd' "$$f"); \
			BWD=$$(jq -r '.results.ms_bwd' "$$f"); \
			LOSS=$$(jq -r '"\(.results.loss_start) → \(.results.loss_end)"' "$$f"); \
			echo "  ✅ $${PARAMS}M | $${TOKS} tok/s | $${MS}ms/step (fwd=$${FWD} bwd=$${BWD}) | loss $${LOSS}"; \
			echo "     → https://bench.rustane.org/?id=$$RESULT_ID"; \
			mv "$$f" target/bench-results/submitted/; \
			SUBMITTED=$$((SUBMITTED + 1)); \
		else \
			echo "  ❌ $$BENCH failed:"; \
			echo "$$RESPONSE" | jq .; \
			echo "  Stopping. $$SUBMITTED submitted, remaining kept."; \
			exit 1; \
		fi; \
	done; \
	echo ""; \
	echo "$$SUBMITTED result(s) submitted."
```

Key behaviors:
- **Submitter prompt once**, reused for all files
- **Sorted by filename** (timestamp-grouped, deterministic within same second)
- **Summary per result:** params, tok/s, ms/step, fwd/bwd split, loss trajectory
- **Archive on success:** `mv` to `target/bench-results/submitted/` — move back to re-queue if needed
- **Stop on first failure**, keep remaining files for retry
- X share prompt removed from loop

**Submission semantics:** at-most-once local dequeue after confirmed success response. Duplicates possible if POST succeeds but response is lost. To recover: move files from `submitted/` back to `bench-results/` and re-run `make submit`.

---

## Backward compat

- Old `target/bench-result.json` is no longer written. If it exists from a prior run, `make submit` won't see it (only reads from `bench-results/`). Add a one-time migration note to README or print a warning if the old file exists.
- `.gitignore` already covers `target/` so no change needed.

## Verify

- Run `make sweep-600m` twice → two files in `target/bench-results/`
- `make submit` submits both in order, directory empty after
- Kill network mid-submit → first file deleted (if response received), second survives for retry
- `make submit` with no files → clean error message
- Benchmark name with spaces/special chars → slug sanitized in filename
- Interrupt bench mid-write → only `.tmp` file left, `make submit` ignores it

> [!TIP]
> For maximum shell defensiveness, the `ls ... | sort` loop could later switch to `find -print0 | sort -z | xargs -0`. Not a blocker given the sanitized filename contract.
