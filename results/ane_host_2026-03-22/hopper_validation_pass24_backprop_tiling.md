# Hopper Validation Pass 24: `ZinMirGraphSplitterBase::BackPropTiling`

This pass targets:

- `ZinMirGraphSplitterBase::BackPropTiling(...)`

and the major helpers visible around it:

- `DetermineInputLayersForCurrentSplitLayer(...)`
- `DetermineOutputLayersForCurrentSplitLayer(...)`
- `BuildConcatsForOutputNodes(...)`

The goal is to understand how Apple propagates a chosen split plan backward
through a subgraph before the final graph rewrite is committed.

## 1. High-level result

`BackPropTiling(...)` is the bridge between:

- an already chosen split dimension / tile plan
- and the concrete rewiring of the surrounding subgraph

It is not a tiny bookkeeping helper. It:

- records tiled tensor regions per layer
- gathers latency info for candidate rewrites
- determines split-specific input and output feeders
- saves resulting input tiling back into `SplitInfo`
- and recursively propagates tiling constraints toward upstream producers

So this is the “constraint propagation” stage of Apple’s splitter.

## 2. `BackPropTiling(...)`

Size:

- length: `1736` bytes
- basic blocks: `66`

### 2.1 What Hopper shows

The broad structure is:

1. check whether the current layer is already within the subgraph’s tracked set
2. record or append tiled tensor regions for this layer in the per-layer map
3. if the layer participates in current split tracking:
   - gather latency info for the layer
   - call a virtual helper at `this + 0x10`
   - save resulting input tiling into `SplitInfo`
   - recurse into upstream layers
4. on failure, return status `3`
5. on success, return `0`

Important direct callees:

- `GatherLatencyInfoOnLayer(...)`
- `SaveInputTilingToSplitInfo(...)`
- recursive `BackPropTiling(...)`
- per-layer/per-graph maps of:
  - `LayerTilingHelper`
  - `vector<vector<ZinTensorRegion>>`

### 2.2 What this means

This is not just computing one layer’s split.

It is walking the dependency graph backward and building a consistent tiling
story that later stages can use to rewrite producers, consumers, and boundaries
together.

That explains why Apple’s splitter can do much more than the repo’s current
manual “split one kernel and concatenate later” approach.

## 3. `DetermineInputLayersForCurrentSplitLayer(...)`

Size:

- length: `4396` bytes
- basic blocks: `175`

### 3.1 What Hopper shows

This function handles the upstream side of the split.

It:

- iterates incoming layers for the current split layer
- decides whether the current producer is already part of the subgraph
- checks whether a direct view is possible for the tiled input region
- if not, synthesizes helper layers, including:
  - `CreateView(...)`
  - `CreateNEBypass(...)`
- inserts those helper layers into the graph
- rewrites the operation map keyed by basic block
- updates the incoming-layer array for the split branch

Repeated internal failure points collapse to:

- `Spatial Splitting Internal Error`

### 3.2 What this means

Input handling is already a full transformation problem:

- direct reuse when possible
- views when region alignment works
- bypass/copy when it does not

This is one of the clearest places where Apple’s splitter behaves like a graph
transformation engine rather than a tensor-chunking utility.

## 4. `DetermineOutputLayersForCurrentSplitLayer(...)`

Size:

- length: `6700` bytes
- basic blocks: `197`

### 4.1 What Hopper shows

This is even larger than the input-side helper and handles the downstream side
of the split.

It:

- determines whether an output is rooted inside or outside the current subgraph
- synthesizes `ss_copy` / `ss_concat`-style names
- creates bypass/copy/view structures for split outputs
- adjusts dimensions for broadcast when needed
- updates operation maps keyed by basic block
- rewrites output node references for the split branch

It also appears to be aware of:

- bonded-info / ANE index hints
- root outputs
- outside-subgraph dependencies

### 4.2 What this means

Output handling is not “just concatenate later”.

The splitter has to solve:

- how split outputs re-enter the graph
- how root outputs are preserved
- how broadcast semantics survive the split
- how operation/basic-block mappings remain coherent

That is a much more complete graph-rewrite system than the repo currently has.

## 5. `BuildConcatsForOutputNodes(...)`

Size:

- length: `2548` bytes
- basic blocks: `112`

### 5.1 What Hopper shows

This helper assembles the fan-in side of the split:

- iterates output nodes for the split subgraph
- finds tiled tensor regions associated with them
- creates `concat_sssg`-style wiring
- creates bypass/concat scaffolding
- inserts those layers into the graph
- rewrites the outgoing node references to the reassembled output

### 5.2 What this means

Apple’s splitter explicitly manages output reassembly as part of the same graph
transformation framework.

So the end-to-end split path is:

- tile subgraph
- rewrite inputs
- rewrite internal layers
- rewrite outputs
- reassemble outputs

not just a local split plus one external concat.

## 6. What this changes in our understanding

### 6.1 Split propagation is recursive and graph-aware

`BackPropTiling(...)` is recursively called as it walks upstream layers.

That means the compiler is not just deciding split parameters for each layer
independently. It is propagating a consistent tiling contract backward through
the subgraph.

### 6.2 Apple’s split framework is significantly more complete than the repo’s current decomposition logic

The repo’s current manual splits are solving:

- how to keep compilation under width/resource ceilings

Apple’s internal framework is additionally solving:

- how to represent tiled regions
- how to choose helper layers
- how to preserve graph connectivity
- how to rewrite basic-block operation mappings
- how to preserve output semantics

### 6.3 This is likely one of the most important architecture findings so far

For the “map the framework” goal, this pass shows that spatial splitting is a
major internal subsystem with:

- strategy
- constraint propagation
- graph rewrite
- output reassembly

all separated into different but cooperating stages.

## 7. Best next targets from here

The next most useful targets are:

1. `LegalizerSubgraphIdentification::AddSubgraph(...)`
   - to understand how illegal regions are represented before splitting
2. the CFG visitors/lambdas used in `Tile(...)` / `TileWithGlobalRefinement(...)`
   - to understand pressure-based subgraph identification
3. `SaveInputTilingToSplitInfo(...)`
   - if we want to understand the persisted tiling contract format

For `rustane`, the highest-value next target is probably:

- `LegalizerSubgraphIdentification::AddSubgraph(...)`

because that would close the loop on the splitter pipeline from:

- subgraph identification
- to strategy
- to propagation
- to graph rewrite
