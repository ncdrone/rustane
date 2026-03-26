# Latency-Oriented Parallel Forward Design

## Goal

Reduce wall-clock latency for a **single** forward pass of a large model.

This is explicitly different from the current multistream path:

- current multistream: improves **aggregate throughput** by serving multiple full requests concurrently
- target design: improves **single-request latency** by splitting one request into concurrently executable shards

The immediate target is a 30B-class forward pass on the `m3-ultra` SDPA branch.

## Why The Current Multistream Path Is Not Enough

The current multistream implementation in [crates/engine/src/multistream.rs](../crates/engine/src/multistream.rs) shares one `Arc<ModelWeights>` and runs multiple worker-local forward passes with separate compiled kernels and workspaces.

That is useful for request concurrency, but it does not reduce the amount of work needed to complete one request.

In practice:

- aggregate `tok/s` increases
- per-stream wall time usually gets worse

This is the right tradeoff for throughput serving, but the wrong tradeoff if the real KPI is:

> make one forward pass finish faster than the single-stream baseline

## Current Performance Picture

The forward utilization results in [results/forward_utilization/summary.md](../results/forward_utilization/summary.md) are the best starting point.

At larger scales:

- ANE dominates total layer wall time
- staging and readback matter, but they are secondary
- CPU-side math is small

Examples from the existing measurements:

- 5B: ANE is about 80.5% of layer wall time
- 13B: ANE is about 88.1% of layer wall time

The dominant kernels at scale are:

- `ffn_w2_fwd`
- `sdpa_fwd`
- `ffn_w1_fwd`
- `ffn_w3_fwd`

That leads to an important conclusion:

> If we want single-request latency wins, the first good theories are ANE-facing sharding theories, not more host-side request concurrency.

## Compiler Evidence From Hopper

This design is not based only on source-level intuition. `ANECompiler` exposes compiler-side concepts that strongly suggest the compiler already reasons about intra-graph parallelization and subgraph splitting.

### Relevant ANECompiler strings

The currently opened `ANECompiler` document contains strings for:

- `EnableGlobalChannelSplitting`
- `EnableTaskSchedulerExp`
- `Parallel execution is materialized:`
- `Fail finding the parallel execution layers.`
- `scaled_dot_product_attention`
- `FindSubdividedClustersWithMinimumLatencyOverhead`
- `PressureBasedSubgraphIdentification`
- `DetermineSubgraphSplitInfo`
- `SplitByInputChannel failed`

This is strong evidence that:

- the compiler has explicit support for discovering independent parallel execution regions
- channel/spatial splitting are first-class internal concepts
- the scheduler scores and materializes parallel layer sets rather than treating the graph as strictly linear

### `FindParallelExecutionList`

Hopper pseudo-code for `__ZL25FindParallelExecutionList...` shows:

- the compiler computes perf for candidate layers
- it builds a list of layers that may be run in parallel
- candidates are filtered by layer type and DMA-read constraints
- there is a profitability threshold before more layers are added

Interpretation:

- parallel execution is selective, not universal
- layers with heavy DMA coupling are less attractive parallel candidates
- concurrency only helps when the compiler believes the overlap beats the coordination overhead

This is a good sign for **targeted sharding of the biggest kernels**, and a bad sign for naive “run everything everywhere” designs.

### `ParallelScoreCalculator::GetParallelScore`

Hopper pseudo-code for `PartitionGraph::ParallelScoreCalculator::GetParallelScore` shows:

- the compiler maintains a ready set
- it repeatedly schedules ready nodes
- it distinguishes different scheduling group sizes
- it computes a final score based on grouped execution

Interpretation:

- the compiler is not blind to parallel scheduling quality
- it appears to explicitly model whether the graph structure exposes useful parallel groups

This supports the idea that if we present the compiler with a graph that has clean, independent shards, it may schedule them more effectively than if we just launch duplicate full-model forwards from the host.

### `DetermineSubgraphSplitInfo`

Hopper pseudo-code for `ZinMirGraphSplitterBase::DetermineSubgraphSplitInfo` shows:

- tile count is a first-class output
- kernel-read size is estimated during split planning
- the split path calls `TileSubgraph(...)`

Interpretation:

- the compiler already expects subgraphs to be tiled and split
- kernel-read volume matters to whether a split is attractive

This matters because many of our best latency theories involve shrinking the per-shard kernel-read footprint, not just dividing arithmetic.

## Design Principle

The next step should be:

> Split one request into independent shards whose outputs can be merged cheaply, and whose per-shard geometry is more compiler-friendly than the monolith.

That suggests three candidate directions:

1. FFN hidden-dimension sharding
2. SDPA head-group sharding
3. layer/pipeline parallelism

I think the correct order is:

1. FFN sharding first
2. SDPA sharding second
3. full layer pipeline last, if at all

## Candidate A: FFN Hidden-Dimension Sharding

### Why this is the best first experiment

The FFN path is the single biggest forward-time bucket at large scale.

For decomposed FFN on this branch, the forward path is already structurally separated in [crates/engine/src/layer.rs](../crates/engine/src/layer.rs):

- `W1` projection
- `W3` projection
- SiLU gate
- `W2` projection back to `dim`

That makes FFN the cleanest place to introduce tensor-parallel sharding.

### Proposed shard geometry

For hidden width `H`, shard into `N` equal hidden blocks:

- shard `i` owns:
  - `W1[:, H_i]`
  - `W3[:, H_i]`
  - `W2[:, H_i]`

Each shard computes:

- `h1_i = xnorm @ W1_i`
- `h3_i = xnorm @ W3_i`
- `gate_i = silu(h1_i) * h3_i`
- `ffn_i = gate_i @ W2_i^T`

Final FFN output:

- `ffn_out = sum_i(ffn_i)`

Residual path remains unchanged:

- `x_next = x2 + alpha * ffn_out`

### Why this may beat the monolith

It does more than “split the work across workers.”

It changes the actual tensor geometry.

Example: 30B on this branch uses:

- `dim = 5120`
- `hidden = 13824`

With `4` shards:

- each shard hidden size becomes `3456`

That creates three possible wins:

1. smaller per-shard kernel reads
2. smaller IOSurface staging windows
3. better compiler-selected tile shapes or more favorable split planning

This is exactly the kind of geometry-sensitive improvement suggested by the `DetermineSubgraphSplitInfo` and split/tile machinery visible in `ANECompiler`.

### Merge cost

The merge is cheap:

- each shard returns one `dim x seq` partial
- final merge is an elementwise sum of `N` partials

That is dramatically cheaper than pipelining whole layer ranges.

### Practical first version

Do not shard the full model first.

Instead:

1. implement a single-layer FFN-sharded forward path
2. compare only the FFN section wall time against the current decomposed FFN path
3. if that wins, compose it back into the full layer forward

## Candidate B: SDPA Head-Group Sharding

### Why it is plausible

Attention heads are naturally partitionable.

This branch’s split-SDPA I/O in [crates/engine/src/kernels/sdpa_fwd.rs](../crates/engine/src/kernels/sdpa_fwd.rs) is an enabling change, because it already breaks SDPA into explicit inputs/outputs instead of one packed buffer contract.

For 30B:

- `heads = 40`

With `4` shards:

- each shard gets `10` heads

### Proposed shard geometry

Per head-group shard:

- shard `Wq`, `Wk`, `Wv` by output channels for that head group
- run SDPA only for that head group
- run the corresponding shard of `Wo`
- sum the partial `dim x seq` outputs across groups

### Why it may help

This reduces:

- QKV width per shard
- attention score tensor size per shard
- output-projection staging footprint per shard

It may also better align with compiler-side “parallel execution layers” if each head-group shard is a clean, mostly independent subgraph.

### Why it is second, not first

Compared to FFN sharding:

- more graph surgery
- more places where shape/plumbing bugs can occur
- more dependence on SDPA-specific compiler behavior

The split-SDPA branch makes it possible, but FFN sharding is still the simpler latency experiment.

## Candidate C: Layer / Pipeline Parallelism

### Why it is lower priority

For a single request, transformer layers are still serial:

- layer `l+1` depends on the full output of layer `l`

This branch already does the easy overlap:

- deferred FFN readback
- overlapped staging during ANE work

See:

- [crates/engine/src/full_model.rs](../crates/engine/src/full_model.rs)
- [crates/engine/src/layer.rs](../crates/engine/src/layer.rs)

A true layer pipeline is usually much better for many in-flight microbatches than for one latency-sensitive request.

### Likely result

You would pay:

- more synchronization
- more bubbles
- more host orchestration

without getting enough extra parallelism to beat a good FFN or SDPA sharding strategy.

So I do not recommend this as the next experiment.

## A More Concrete 30B Latency Plan

### Experiment 1: FFN-only 2-way and 4-way sharding

Target:

- current 30B shape on this branch

Compare:

- baseline decomposed FFN
- 2-way hidden sharded FFN
- 4-way hidden sharded FFN

Measure:

- FFN-only wall time
- full layer wall time
- full model wall time
- aggregate ANE dispatch count
- staging time
- merge time

Success criterion:

- lower full-layer wall time than baseline

Failure criterion:

- shard merge + extra dispatches erase the benefit

### Experiment 2: SDPA-only 2-way and 4-way head-group sharding

Compare:

- baseline SDPA
- head-group sharded SDPA

Measure:

- SDPA-only wall time
- QKV staging time
- readback time
- merge time

Success criterion:

- lower SDPA wall time and lower layer wall time

### Experiment 3: Combine FFN and SDPA sharding in one layer

Only do this if experiment 1 or 2 already wins on its own.

Otherwise the combined orchestration cost will be noise on top of two non-winning ideas.

## Important Warning About The Current `compile_forward_only`

The branch currently has a conceptual forward-only compile path in [crates/engine/src/layer.rs](../crates/engine/src/layer.rs), but `compile_forward_only` is still a stub that falls back to the full compile path.

That matters for latency-oriented experiments because:

- shard workers will carry unnecessary backward-side kernel-buffer overhead
- memory and compile costs will look worse than they need to

Before doing serious latency-parallel experiments, it is worth wiring the actual forward-only buffer reduction properly.

## What The Compiler Evidence Suggests

Given the Hopper evidence, I think the strongest current theory is:

- the compiler is already prepared to reason about subgraph splits
- it already has parallel-execution scoring and task scheduling
- it already understands channel/spatial split families

That means the best design is probably **not** host-side manual pipelining of full layers.

The best design is more likely:

> present the compiler with well-formed, independent subgraphs whose geometry is more favorable than the monolith, and let its scheduler decide how aggressively to materialize parallel execution.

That is exactly what FFN hidden sharding and SDPA head-group sharding would do.

## Current Recommendation

If we are optimizing for **single-request latency**, the next step should be:

1. FFN hidden sharding prototype at 30B
2. SDPA head-group sharding prototype at 30B
3. compare each against the current single-request baseline
4. only then consider a combined sharded-forward path

I do **not** recommend full-model layer pipelining as the next move.

## Open Questions For The Next Hopper Pass

When Hopper is stable again, the next binary-side questions should be:

1. How exactly does `RunTaskScheduler` decide when to materialize parallel execution groups?
2. What layer/property combinations cause `FindParallelExecutionList` to reject candidates?
3. Are there explicit limits on concurrent ANE task groups or NE cluster subdivision?
4. Is `EnableGlobalChannelSplitting` only for certain layer families, or can it be forced more broadly?
5. Is `scaled_dot_product_attention` lowered through a path that naturally accepts head-group subdivision, or does it get canonicalized too early?

Those answers will tell us whether FFN sharding or SDPA sharding is more likely to pay off first.
