#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE_SHA="9a568eca5340492b9b5bac54fb4e5211225996b1"
HEAD_SHA="e9cbf0300df82352b37794da8a59a6a837542fa3"

BASE_WT="/tmp/rustane-pr22-base"
HEAD_WT="/tmp/rustane-pr22-head"

copy_results() {
  local from_root="$1"
  local side="$2"
  mkdir -p "$ROOT/results/pr22_compare/$side"
  if [[ -d "$from_root/results/pr22_compare/$side" ]]; then
    cp "$from_root/results/pr22_compare/$side"/*.json "$ROOT/results/pr22_compare/$side/" 2>/dev/null || true
    cp "$from_root/results/pr22_compare/$side"/*.md "$ROOT/results/pr22_compare/$side/" 2>/dev/null || true
  fi
}

if [[ ! -d "$BASE_WT/.git" ]]; then
  git -C "$ROOT" worktree add --detach "$BASE_WT" "$BASE_SHA"
fi

if [[ ! -d "$HEAD_WT/.git" ]]; then
  git -C "$ROOT" worktree add --detach "$HEAD_WT" "$HEAD_SHA"
fi

cp "$ROOT/crates/engine/tests/bench_forward_utilization.rs" \
   "$BASE_WT/crates/engine/tests/bench_forward_utilization.rs"
cp "$ROOT/crates/engine/tests/bench_forward_utilization.rs" \
   "$HEAD_WT/crates/engine/tests/bench_forward_utilization.rs"
cp "$ROOT/crates/engine/tests/bench_forward_utilization_head.rs" \
   "$HEAD_WT/crates/engine/tests/bench_forward_utilization_head.rs"

rm -rf "$ROOT/results/pr22_compare/base" "$ROOT/results/pr22_compare/head" "$ROOT/results/pr22_compare/head_lean"

(cd "$BASE_WT" && cargo test -p engine --test bench_forward_utilization --release -- --ignored --nocapture pr22_compare_single_stream)
(cd "$BASE_WT" && cargo test -p engine --test bench_forward_utilization --release -- --ignored --nocapture pr22_compare_layer_breakdown)
(cd "$BASE_WT" && cargo test -p engine --test bench_forward_utilization --release -- --ignored --nocapture pr22_compare_kernel_overhead)

(cd "$HEAD_WT" && cargo test -p engine --test bench_forward_utilization --release -- --ignored --nocapture pr22_compare_single_stream)
(cd "$HEAD_WT" && cargo test -p engine --test bench_forward_utilization --release -- --ignored --nocapture pr22_compare_layer_breakdown)
(cd "$HEAD_WT" && cargo test -p engine --test bench_forward_utilization --release -- --ignored --nocapture pr22_compare_kernel_overhead)
(cd "$HEAD_WT" && cargo test -p engine --test bench_forward_utilization_head --release -- --ignored --nocapture pr22_head_lean_forward_only)

copy_results "$BASE_WT" base
copy_results "$HEAD_WT" head
copy_results "$HEAD_WT" head_lean

python3 "$ROOT/system/compare_pr22_forward.py"

echo "$ROOT/results/pr22_compare/compare.md"
