# Rustane — common commands
#
# Run `make help` to see all available targets.

.PHONY: help build test sweep-600m sweep-1b sweep-3b sweep-5b sweep-full \
	forward-ladder forward-ceiling forward-7b forward-10b train-600m submit

STREAMS ?= 1
USE_LEAN_WORKSPACE ?= 1
ATTN_SHARDS ?=
FFN_SHARDS ?=

help: ## Show this help
	@echo "  Rustane — available commands:"
	@echo ""
	@echo "  make build                Build all crates"
	@echo "  make test                 Run all unit + integration tests"
	@echo ""
	@echo "  Training validation:"
	@echo "  make sweep-600m           Validate at 600M (~17s)"
	@echo "  make sweep-1b             Validate at 1B (~35s)"
	@echo "  make sweep-3b             Validate at 3B (~80s, needs 55GB)"
	@echo "  make sweep-5b             Validate at 5B (~150s, needs 85GB)"
	@echo "  make sweep-full           All 25 configs, 600M-5B (~60 min, needs 85GB)"
	@echo ""
	@echo "  Forward-only probes:"
	@echo "  make forward-7b           Forward pass at 7B (~30s, needs 31GB)"
	@echo "  make forward-10b          Forward pass at 10B (~45s, needs 46GB)"
	@echo "  make forward-ladder       Forward pass 5B to 20B (~8 min, needs 93GB)"
	@echo "  make forward-ceiling      Forward pass 25B/30B (~10 min, needs 130GB)"
	@echo "  add STREAMS=<N>           Use multistream forward benchmark mode"
	@echo "  example: make forward-ladder STREAMS=4"
	@echo "  add ATTN_SHARDS=<1/2/4/8/10>   Attention head sharding for forward benchmarks"
	@echo "  add FFN_SHARDS=<2/4/6/8/10/12/16> FFN sharding for forward benchmarks"
	@echo "  examples: make forward-10b FFN_SHARDS=8"
	@echo "            make forward-ladder ATTN_SHARDS=10 FFN_SHARDS=10"
	@echo ""
	@echo "  Training on real data:"
	@echo "  make train-600m DATA=/path/to/train.bin"
	@echo ""
	@echo "  Leaderboard:"
	@echo "  make submit               Submit queued benchmark results to bench.rustane.org"

build: ## Build all crates
	cargo build

test: ## Run all unit + integration tests
	cargo test -p engine --release

# ── Training validation ─────────────────────────────────────────────

sweep-600m: ## Validate training pipeline at 600M (~17s)
	$(if $(ATTN_SHARDS),ATTN_SHARDS=$(ATTN_SHARDS) )$(if $(FFN_SHARDS),FFN_SHARDS=$(FFN_SHARDS) )cargo test -p engine --test bench_param_sweep --release -- --ignored --nocapture sweep_600m_a

sweep-1b: ## Validate training pipeline at 1B (~35s)
	$(if $(ATTN_SHARDS),ATTN_SHARDS=$(ATTN_SHARDS) )$(if $(FFN_SHARDS),FFN_SHARDS=$(FFN_SHARDS) )cargo test -p engine --test bench_param_sweep --release -- --ignored --nocapture sweep_1b_a

sweep-3b: ## Validate training pipeline at 3B (~80s, needs 55GB)
	$(if $(ATTN_SHARDS),ATTN_SHARDS=$(ATTN_SHARDS) )$(if $(FFN_SHARDS),FFN_SHARDS=$(FFN_SHARDS) )cargo test -p engine --test bench_param_sweep --release -- --ignored --nocapture sweep_3b_a

sweep-5b: ## Validate training pipeline at 5B (~150s, needs 85GB)
	$(if $(ATTN_SHARDS),ATTN_SHARDS=$(ATTN_SHARDS) )$(if $(FFN_SHARDS),FFN_SHARDS=$(FFN_SHARDS) )cargo test -p engine --test bench_param_sweep --release -- --ignored --nocapture sweep_5b_a

sweep-full: ## Full parameter sweep, 25 configs, 600M-5B (~60 min, needs 85GB)
	$(if $(ATTN_SHARDS),ATTN_SHARDS=$(ATTN_SHARDS) )$(if $(FFN_SHARDS),FFN_SHARDS=$(FFN_SHARDS) )cargo test -p engine --test bench_param_sweep --release -- --ignored --nocapture sweep_full

# ── Forward-only scale probes ────────────────────────────────────────

forward-ladder: ## Forward pass 5B to 20B (~8 min, needs 93GB)
	@if [ "$(STREAMS)" = "1" ]; then \
		$(if $(ATTN_SHARDS),ATTN_SHARDS=$(ATTN_SHARDS) )$(if $(FFN_SHARDS),FFN_SHARDS=$(FFN_SHARDS) )USE_LEAN_WORKSPACE=$(USE_LEAN_WORKSPACE) cargo test -p engine --test bench_fwd_only_scale --release -- --ignored --nocapture fwd_scale_ladder; \
	else \
		STREAMS=$(STREAMS) cargo test -p engine --test bench_forward_multistream --release -- --ignored --nocapture forward_ladder_multistream; \
	fi

forward-ceiling: ## Push forward pass to 25B/30B (~10 min, needs 130GB)
	@if [ "$(STREAMS)" = "1" ]; then \
		$(if $(ATTN_SHARDS),ATTN_SHARDS=$(ATTN_SHARDS) )$(if $(FFN_SHARDS),FFN_SHARDS=$(FFN_SHARDS) )USE_LEAN_WORKSPACE=$(USE_LEAN_WORKSPACE) cargo test -p engine --test bench_fwd_only_scale --release -- --ignored --nocapture fwd_find_ceiling; \
	else \
		STREAMS=$(STREAMS) cargo test -p engine --test bench_forward_multistream --release -- --ignored --nocapture forward_ceiling_multistream; \
	fi

forward-7b: ## Single forward pass at 7B (~30s, needs 31GB)
	@if [ "$(STREAMS)" = "1" ]; then \
		$(if $(ATTN_SHARDS),ATTN_SHARDS=$(ATTN_SHARDS) )$(if $(FFN_SHARDS),FFN_SHARDS=$(FFN_SHARDS) )USE_LEAN_WORKSPACE=$(USE_LEAN_WORKSPACE) cargo test -p engine --test bench_fwd_only_scale --release -- --ignored --nocapture fwd_7b; \
	else \
		STREAMS=$(STREAMS) cargo test -p engine --test bench_forward_multistream --release -- --ignored --nocapture forward_7b_multistream; \
	fi

forward-10b: ## Single forward pass at 10B (~45s, needs 46GB)
	@if [ "$(STREAMS)" = "1" ]; then \
		$(if $(ATTN_SHARDS),ATTN_SHARDS=$(ATTN_SHARDS) )$(if $(FFN_SHARDS),FFN_SHARDS=$(FFN_SHARDS) )USE_LEAN_WORKSPACE=$(USE_LEAN_WORKSPACE) cargo test -p engine --test bench_fwd_only_scale --release -- --ignored --nocapture fwd_10b; \
	else \
		STREAMS=$(STREAMS) cargo test -p engine --test bench_forward_multistream --release -- --ignored --nocapture forward_10b_multistream; \
	fi

# ── Real data training ───────────────────────────────────────────────

train-600m: ## Train 600M on real data (needs climbmix-400B data file)
	cargo run -p engine --release --bin train -- \
		--model custom:1536,4096,20,512 --data $(DATA) \
		--lr 3e-4 --accum 1 --warmup 100 \
		--embed-lr 1.0 --beta2 0.99 \
		--loss-scale 1 --grad-clip 1 \
		--steps 72000

# ── Leaderboard ──────────────────────────────────────────────────────

LEADERBOARD_API ?= https://api.bench.rustane.org

submit: ## Submit queued benchmark results to leaderboard
	@if [ -f target/bench-result.json ]; then \
		echo "Note: target/bench-result.json is from the old single-file flow and is ignored."; \
		echo "      Pending submissions now live in target/bench-results/."; \
		echo ""; \
	fi
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
