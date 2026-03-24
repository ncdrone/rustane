# Hopper Validation Pass 21: `ZinMirLayerSplitterBase::TrySplitBySpace`

This pass targets:

- `ZinMirLayerSplitterBase::TrySplitBySpace()`

and the immediately related helpers:

- `AnalyzeSplitBySpace(...)`
- `IsSpaceSplitable(...)`
- `ZinMirSpatialSplitter::Tile(...)`

The goal is to understand how Apple’s shared legalization machinery actually
attempts spatial splitting after illegal tensor dimensions are detected.

## 1. High-level result

`TrySplitBySpace()` is a thin front door.

The real split strategy lives in:

- `ZinMirSpatialSplitter::Tile(...)`

and the eligibility logic lives in:

- `AnalyzeSplitBySpace(...)`
- `IsSpaceSplitable(...)`

So the split stack is:

1. identify candidate illegal subgraphs
2. test whether a layer/subgraph is space-splittable
3. add that subgraph to the collected legalization set
4. call `TrySplitBySpace()`
5. which delegates to `ZinMirSpatialSplitter::Tile(...)`

This fits very well with the earlier tensor-dimension-legalizer pass.

## 2. `TrySplitBySpace()`

Size:

- length: `108` bytes
- basic blocks: `3`

Callers:

- `ZinMirTensorDimensionLegalizer::Execute(...)`
- `ZinMirL2Legalizer::Execute(...)`
- `ZinMirLatencyLegalizer::Execute(...)`

Callee:

- `ZinMirSpatialSplitter::Tile(...)`

### 2.1 What Hopper shows

The body is simple:

- read the collected-subgraph state from the splitter object
- call:
  - `ZinMirSpatialSplitter::Tile(shared_subgraph_splits, collected_ane_layers, flag)`
- clean up the temporary tree/set state
- return the resulting status

### 2.2 What this means

This is not where the split policy lives.

`TrySplitBySpace()` is the shared integration point that several legalization
passes use to invoke the spatial splitter.

That is important architecturally because it confirms spatial splitting is not
special to only one pass:

- tensor-dimension legalization uses it
- L2 legalization uses it
- latency legalization uses it

So “split by space” is a general compiler remediation strategy, not a one-off
hack.

## 3. `AnalyzeSplitBySpace(...)`

Size:

- length: `96` bytes
- basic blocks: `3`

What Hopper shows:

- calls `IsSpaceSplitable(graph, layer, hal_params)`
- if true:
  - `LegalizerSubgraphIdentification::AddSubgraph(...)`
- if that succeeds, returns success

### 3.1 What this means

This function is the bridge between:

- local layer eligibility
- and subgraph-level legalization collection

So the legalizer does not immediately split a single offending layer. It first
records the relevant subgraph for later splitting.

## 4. `IsSpaceSplitable(...)`

Size:

- length: `496` bytes
- basic blocks: `13`

This function is more informative than the thin wrappers.

### 4.1 What Hopper shows

It does at least these checks:

1. construct a one-element `vector<ZinIrPoolingMode>`
2. call:
   - `IsSupportedForReductionSplitting(layer, pooling_modes)`
3. if supported:
   - obtain current tensor layout/info from the layer
   - compute tensor-format size in bytes via `_ZinTensorFormatGetSizeInBytes(...)`
   - use HAL parameters and tensor shape information to decide whether space
     splitting is admissible
4. if tensor-format size lookup fails:
   - assert:
     - `Error in getting tensor format size in bytes`

It also checks for outgoing CW transpose structure through:

- `RawOrShared<ZinTextureLayer>::unwrap_const_ptr(...)`

and uses that information in the admissibility decision.

### 4.2 What this means

Eligibility is not just:

- “layer is too wide, therefore split”

It depends on:

- whether the layer is supported for reduction splitting
- tensor format size
- HAL-level limits
- neighboring transpose/layout structure

This reinforces the earlier point that width-related failures are a mixture of:

- global legalization strategy
- and local layout-specific admissibility rules

## 5. `CalculateSplitsForLayerLegalization(...)`

From the prior pass, this helper already showed that the legalizer:

- computes split information for ANE layers only
- may propagate split effects across outgoing CW transpose structure
- rewrites tensor dimensions based on those split decisions

That fits tightly with `IsSpaceSplitable(...)`:

- first check whether space splitting is admissible
- later compute how the dimensions should actually be rewritten

## 6. `ZinMirSpatialSplitter::Tile(...)`

Size:

- length: `712` bytes
- basic blocks: `17`

This is the first real strategy function in the chain.

### 6.1 What Hopper shows

`Tile(...)` has two distinct modes.

If a splitter configuration bit at `+0x23` is enabled:

- it calls:
  - `TileWithGlobalRefinement(...)`

Otherwise it uses a more explicit staged pipeline:

1. traverse the CFG with a first function object
   - pressure-based subgraph identification
2. call:
   - `ZinMirSpatialSplitUtils::PostprocessForPressureBasedSubgraphIdentification(...)`
3. traverse again with a second function object
4. traverse forward with a generic graph visitor
5. possibly traverse again depending on another splitter bit at `+0x26`
6. if any traversal fails, return status `3`
7. otherwise return success

The string surface around this area reinforces the interpretation:

- `GlobalRefinementInSpatialSplit`
- `EnableSpatialSplitInX`
- `EnableCircularBufferInSpatialSplit`
- `INFO:: (SpatialSplit) Can't tile subgraph b/c SIP > budget`
- `SpatialSplitPressureBasedSubgraphIdentification`
- `ZinMirSpatialSplitSimpleCostModel`

### 6.2 What this means

This is the most important finding of the pass.

Apple’s spatial splitting is not a single greedy “cut width in half” routine.
It has at least two distinct strategies:

- **global refinement**
- **pressure-based identification / splitting**

and it appears to be driven by:

- budget / pressure heuristics
- CFG traversals
- optional extra features like circular-buffer support and X-axis enabling

So the compiler’s split logic is much more sophisticated than the repo’s
current manual chunking.

## 7. What this changes in our understanding

### 7.1 Spatial splitting is a reusable framework service

Because `TrySplitBySpace()` is called from:

- tensor-dimension legalization
- L2 legalization
- latency legalization

Apple clearly treats spatial splitting as a general-purpose legalization tool.

### 7.2 The strategy is not purely local

The split decision path spans:

- local eligibility checks (`IsSpaceSplitable`)
- subgraph collection (`AnalyzeSplitBySpace`)
- global/pressure-based tiling (`Tile`)

That means the compiler is reasoning at:

- layer level
- neighbor/layout level
- subgraph level
- global CFG / pressure level

### 7.3 This is directly relevant to `rustane`

This is one of the strongest repo-relevant passes so far because the branch has
already been manually decomposing kernels to satisfy compiler ceilings.

Apple’s own stack appears to have:

- a more general split framework
- richer admissibility checks
- multiple split strategies

That suggests two practical future directions for the repo:

- keep doing narrow kernel-specific decompositions where needed
- but also think in terms of a more shared legalization framework, because that
  is how Apple appears to organize the problem

## 8. Best next targets from here

The best next targets are now:

1. `ZinMirSpatialSplitter::TileWithGlobalRefinement(...)`
   - to understand the more sophisticated split strategy
2. the first CFG visitor/lambda used by `Tile(...)`
   - to understand pressure-based subgraph identification
3. `LegalizerSubgraphIdentification::AddSubgraph(...)`
   - to understand exactly what gets collected as a legalization subgraph

For `rustane`, the highest-value next target is probably:

- `ZinMirSpatialSplitter::TileWithGlobalRefinement(...)`

because that is most likely where Apple’s split strategy departs furthest from
the repo’s current manual decompositions.
