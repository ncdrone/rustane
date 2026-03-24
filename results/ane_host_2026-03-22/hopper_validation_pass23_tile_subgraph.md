# Hopper Validation Pass 23: `ZinMirGraphSplitterBase::TileSubgraph`

This pass targets:

- `ZinMirGraphSplitterBase::TileSubgraph(...)`

and the immediately related helpers used during actual graph rewrite:

- `DetermineInputLayersForCurrentSplitLayer(...)`
- `DetermineOutputLayersForCurrentSplitLayer(...)`
- `BuildConcatsForOutputNodes(...)`

The goal is to move from “Apple has a sophisticated split strategy” to:

- “what concrete graph surgery does Apple perform once a split plan is chosen?”

## 1. High-level result

`TileSubgraph(...)` is the core graph-rewrite stage that turns a chosen split
plan into a transformed graph.

It is not a trivial helper. It:

- initializes a cost-model view of the chosen subgraph
- walks layers in the subgraph and back-propagates tiling constraints
- invokes deeper graph-split logic through a virtual splitter method
- uses input/output-layer determination helpers
- creates copy/view/bypass/concat plumbing as needed

So this is the point where Apple’s split strategy becomes actual graph surgery.

## 2. `ZinMirGraphSplitterBase::TileSubgraph(...)`

Size:

- length: `604` bytes
- basic blocks: `21`

### 2.1 What Hopper shows

The main structure is:

1. check a splitter-mode bit at `this + 0x1c`
2. inspect the split’s tile count via `TileCountType::Get(...)`
3. fetch split dimensions from `SplitInfo`
4. call a virtual helper at `this + 0x28`
   - this appears to build per-layer tiling helper state
5. initialize subgraph cost-model state:
   - `InitializeSubgraphForCostModel(...)`
6. iterate layers in the subgraph
7. for each layer:
   - call a virtual helper at `this + 0x68`
   - then call `BackPropTiling(...)`
8. once back-propagation succeeds:
   - call a virtual helper at `this + 0x60`
   - this appears to perform the actual final split rewrite
9. tear down temporary tiling-helper / tensor-region tables
10. return `0` on success or `3` on failure

### 2.2 What this means

This confirms that Apple’s split stage is not “cut tensor, insert concat, done.”

There is an explicit intermediate step:

- build local tiling-helper state
- back-propagate constraints across the subgraph
- only then perform final graph rewrite

That is a richer model than most repo-side manual decompositions.

## 3. `DetermineInputLayersForCurrentSplitLayer(...)`

Size:

- length: `4396` bytes
- basic blocks: `175`

This helper is large enough that it is clearly part of the real rewrite logic,
not just a lookup table.

### 3.1 What Hopper shows

The function:

- iterates incoming layers for the current split layer
- checks whether each incoming layer is already inside the current subgraph
- checks whether a view is needed for the current split dimension
  - via `ZinMirSpatialSplitUtils::ShouldCreateView(...)`
- if a direct reuse is not possible, it constructs new layers to feed the split:
  - `ZinBuilder::CreateView(...)`
  - `ZinBuilder::CreateNEBypass(...)`
- inserts those layers into the graph
- updates `Layer2TDMapper::SourceLayer`
- adds edges from the original producer into the new split-specific feeder
- updates an operation map keyed by basic block
- rewrites the input-layer arrays that the split branch should consume

Several internal error paths collapse to:

- `Spatial Splitting Internal Error`

### 3.2 What this means

Input handling in the split pipeline is already quite sophisticated:

- direct reuse when possible
- view creation when tensor regions line up
- bypass/copy layer insertion when they do not

So Apple’s split machinery is explicitly prepared to repair the graph around a
split boundary instead of requiring the original graph to line up perfectly.

## 4. `DetermineOutputLayersForCurrentSplitLayer(...)`

Size:

- length: `6700` bytes
- basic blocks: `197`

This is even larger than the input helper and looks like one of the most
complex graph-surgery routines we have touched so far.

### 4.1 What Hopper shows

The function handles many output-side cases, including:

- detecting whether the current layer is a root output or crosses subgraph
  boundaries
- synthesizing `ss_copy` and `ss_concat`-style names
- creating output-side bypass/copy/view structures:
  - `ZinBuilder::CreateNEBypass(...)`
  - view-related helpers
- adjusting dimensions for broadcast:
  - `ZinMirSpatialSplitUtils::AdjustDimensionsForBroadcast(...)`
- handling optional bonded-info / ane-index hints
- updating operation maps for the containing basic block
- rewriting the output layer arrays for the split branch

Again, many failure paths collapse to:

- `Spatial Splitting Internal Error`

### 4.2 What this means

Output handling is not symmetrical “just wire to concat.”

The splitter has to solve:

- root-output preservation
- cross-subgraph dependencies
- broadcast-correct dimension adjustments
- optional engine/bonding hints

That makes the output side of splitting a much bigger engineering problem than a
single CPU concat after ANE execution.

## 5. `BuildConcatsForOutputNodes(...)`

Size:

- length: `2548` bytes
- basic blocks: `112`

### 5.1 What Hopper shows

This function:

- iterates output nodes for the subgraph
- finds tiled tensor regions attached to each
- builds `concat_sssg`-style structures and names
- creates concat and bypass plumbing
- inserts the new layers into the graph
- rewrites outgoing node references to the new concat outputs

This confirms the split pipeline has an explicit “reassemble tiled outputs”
stage, not just split-time surgery on inputs.

### 5.2 What this means

Apple’s graph splitter manages both ends of the split:

- fan-out into split branches
- and fan-in / reassembly of outputs

That is the strongest evidence so far that the split system is a real subgraph
transformation framework, not a local layer transformer.

## 6. What this changes in our understanding

### 6.1 Apple’s split framework is end-to-end graph surgery

Combined with the previous passes, we now have a much clearer split pipeline:

1. identify illegal subgraphs
2. decide whether and how to split
3. optimize/refine the split plan
4. initialize cost-model state
5. back-propagate tiling through the subgraph
6. rewrite inputs
7. rewrite outputs
8. build concat/reassembly plumbing

That is much more than “chunk a big matmul.”

### 6.2 The repo’s current decomposition work is solving only one slice of the problem

`rustane` has been manually decomposing kernels to get compilation working.

Apple’s internal splitter is also solving:

- graph connectivity repair
- view vs bypass decisions
- output reassembly
- broadcast-aware output reshaping
- block-local operation remapping

So the repo’s current work is closer to:

- “manual local decomposition”

while Apple has:

- “full graph-level legalizing transformation”

### 6.3 This is one of the strongest framework-mapping results so far

If the long-term goal is documenting the framework itself, this pass is a major
step because it shows:

- where the split plan becomes graph edits
- which helper families perform those edits
- how much infrastructure exists around split rewrites

## 7. Best next targets from here

The best next targets are:

1. `LegalizerSubgraphIdentification::AddSubgraph(...)`
   - to understand how candidate subgraphs are represented before rewrite
2. the CFG visitors/lambdas used by `Tile(...)` / `TileWithGlobalRefinement(...)`
   - to understand pressure-based split identification
3. `BackPropTiling(...)`
   - to understand how tiling constraints are propagated before rewrite

For `rustane`, the highest-value next target is probably:

- `BackPropTiling(...)`

because that seems to be the bridge between the chosen split dimensions and the
actual layer-by-layer graph rewrite.
