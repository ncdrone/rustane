# Hopper Validation Pass 22: `ZinMirSpatialSplitter::TileWithGlobalRefinement`

This pass targets:

- `ZinMirSpatialSplitter::TileWithGlobalRefinement(...)`

and the closely related helpers:

- `OptimizeTileCountByInsertingResetLayers(...)`
- `IsWorthTile(...)`
- `GatherLatencyInfoOnLayer(...)`

The goal is to understand how Apple’s more sophisticated spatial-splitting path
works beyond the thin `TrySplitBySpace()` front door.

## 1. High-level result

This is the real “smarter” spatial split path.

`TileWithGlobalRefinement(...)` is not just a wrapper around subgraph tiling.
It layers together:

- pressure-based subgraph identification
- repeated graph traversals
- postprocessing
- CSE and convolution merges
- optional revalidation passes
- and, crucially, tile-count optimization using reset-layer insertion and
  latency heuristics

So Apple’s spatial split strategy is materially more sophisticated than the
repo’s current manual split/chunking logic.

## 2. `TileWithGlobalRefinement(...)`

Size:

- length: `808` bytes
- basic blocks: `21`

### 2.1 What Hopper shows

The main sequence is:

1. record current op-layer count
2. run a first CFG traversal with one function object
   - pressure-based spatial-split identification
3. call:
   - `ZinMirSpatialSplitUtils::PostprocessForPressureBasedSubgraphIdentification(...)`
4. run a second CFG traversal with another function object
5. run one or more forward traversals with a generic visitor
6. run common cleanup/optimization passes:
   - `MirOpt::CSE(...)`
   - optional `MirOpt::MergeConvolutions(...)`
   - optional `MirOpt::MergeFanoutConvolutions(...)`
7. optionally re-run a traversal when an internal splitter bit at `+0x26` is enabled
8. return `0` on success or `3` on any failure path

### 2.2 What this means

The “global refinement” path is not one heuristic toggle.

It is an alternate orchestration mode with:

- more passes
- more graph cleanup
- more opportunities to reconsider the tiling outcome before returning

That makes it look much closer to a mini optimization pipeline than to a single
split transform.

## 3. `IsWorthTile(...)`

Size:

- length: `652` bytes
- basic blocks: `25`

This helper answers one of the most important questions:

- how does Apple decide whether a candidate tiled subgraph is actually worth it?

### 3.1 What Hopper shows

The function combines several signals:

- current layer cost / latency
- latency info for the original graph
- latency info for the tiled graph
- sets of tensors associated with input/output pressure
- special handling for copy layers and ANE-layer status

The numeric structure visible in the decompiler suggests it computes:

- relative latency deltas
- normalized pressure/cost adjustments
- and then sets an output boolean through the `bool*` parameter

It also checks whether the candidate layer is an ANE layer, and whether
neighboring copy structure affects the tradeoff.

### 3.2 What this means

Apple is not splitting solely because a graph is technically legal to split.

It is making an economic decision:

- does the tiling improve or at least not hurt the latency/pressure budget
  enough to be worth keeping?

That is a big difference from the repo’s current approach, which is mostly
compile-admissibility driven.

## 4. `OptimizeTileCountByInsertingResetLayers(...)`

Size:

- length: `2008` bytes
- basic blocks: `60`

This is one of the strongest findings in the whole reverse-engineering effort.

### 4.1 What Hopper shows

The function:

1. checks whether all reset layers are output nodes
2. logs:
   - `INFO:: (SpatialSplit) ---OptimizeTileCountByInsertingResetLayers:BEGIN---`
3. computes a baseline pressure/latency aggregate across candidate layers
4. iterates candidate tiled subgraphs
5. repeatedly:
   - gathers latency info
   - reserves split branches
   - reserves tiled tensor regions
   - tiles the subgraph
   - computes resulting pressure/latency
   - accumulates candidate statistics
6. compares candidate totals against the baseline
7. if a refined choice is better, records it into the chosen split set
8. logs:
   - `INFO:: (SpatialSplit) ---OptimizeTileCountByInsertingResetLayers:END---`

Important direct callees:

- `GatherLatencyInfoOnLayer(...)`
- `ZinMirGraphSplitterBase::TileSubgraph(...)`
- `SplitInfo::ReserveBranch(...)`
- `SplitInfo::ReserveTiledLayerTensorRegions(...)`

### 4.2 What this means

This is not just “try a few tile counts.”

Apple’s global-refinement path is willing to:

- synthesize or use reset-layer structure
- tile candidate subgraphs
- measure latency effects
- compare multiple alternatives
- and pick the better one

That is a much more advanced split-selection mechanism than a static chunk size.

For `rustane`, this is probably the strongest evidence yet that:

- Apple treats splitting as an optimization/search problem
- while the repo currently treats it mostly as a correctness workaround

## 5. `GatherLatencyInfoOnLayer(...)`

Size:

- length: `3440` bytes
- basic blocks: `161`

This helper is large and clearly important.

### 5.1 What Hopper shows

It gathers latency-related information for candidate layers/subgraphs by:

- walking the current subgraph and its boundaries
- inspecting root outputs / outside-subgraph relationships
- building tiled tensor regions
- invoking per-layer callbacks to estimate latency impact
- accumulating latency values into a `LatencyInfo` structure

Strings and surrounding structure suggest it is also aware of:

- reset layers
- root outputs
- outside-subgraph dependencies
- multiple candidate tilings

### 5.2 What this means

Latency estimation is not an afterthought bolted onto the splitter.

It is a first-class part of deciding:

- whether to tile
- how much to tile
- and whether reset-layer insertion is worth it

## 6. What this changes in our understanding

### 6.1 Apple’s spatial split path is optimization-driven, not just legality-driven

This is the biggest result of the pass.

The compiler is not merely asking:

- “can I split this illegal graph?”

It is also asking:

- “is this split worthwhile under latency/pressure heuristics?”
- “can reset layers make a better split possible?”
- “should I keep or discard the refined split?”

### 6.2 The repo’s current split work is probably at the “minimum viable” level relative to Apple

That is not a criticism; it is just a clearer comparison now.

`rustane` currently does:

- targeted decomposition to get compilation working

Apple’s internal framework does:

- subgraph identification
- split admissibility
- candidate tiling
- latency modeling
- reset-layer-assisted refinement
- choice among alternatives

### 6.3 Global refinement is likely one of the highest-value unmapped areas left

Because this path touches:

- subgraph structure
- latency
- memory/pressure
- tiling heuristics
- reset layers

it is one of the most central places for understanding how Apple gets from
illegal graphs to performant legalized graphs.

## 7. Repo-relevant implications

For `rustane`, this pass suggests:

- there is real value in documenting split **strategy**, not just split
  mechanisms
- future decomposition work could benefit from:
  - candidate evaluation
  - simple cost modeling
  - maybe later, reset-layer-aware refinement

I would not jump to implementing Apple-like refinement immediately, but this is
strong evidence that a shared split/optimization subsystem would be a more
Apple-like direction than a growing pile of one-off kernel decompositions.

## 8. Best next targets from here

The best next targets are:

1. `ZinMirGraphSplitterBase::TileSubgraph(...)`
   - to see the actual graph rewrite once a split decision is made
2. the first and second CFG visitors used by `TileWithGlobalRefinement(...)`
   - to better understand pressure-based subgraph identification
3. `LegalizerSubgraphIdentification::AddSubgraph(...)`
   - to understand how candidate legalization regions are represented

For `rustane`, the highest-value next target is probably:

- `ZinMirGraphSplitterBase::TileSubgraph(...)`

because that should show the concrete graph surgery Apple performs once the
split strategy has chosen a plan.
