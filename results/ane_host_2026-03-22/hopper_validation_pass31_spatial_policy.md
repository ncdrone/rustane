# Hopper Validation Pass 31: Spatial-Split Subclass Policy

This pass targets:

- `SpatialSplitPressureBasedSubgraphIdentification::IsClusterBoundary(...)`
- `SpatialSplitPressureBasedSubgraphIdentification::ValidateDimensionalConstraints(...)`
- `CostBasedSubgraphIdentification::ValidateTileCountForOutputTensorCompression(...)`
- `SpatialSplitPressureBasedSubgraphIdentification::PrioritizeSplittableDimensionsToCutCluster(...)`
- `SpatialSplitPressureBasedSubgraphIdentification::GetSplittableDimensionsForClusterLegalization(...)`
- `SpatialSplitPressureBasedSubgraphIdentification::LegalizeClusterAtSplittableDimensionsChangingBoundary(...)`
- `PressureBasedSubgraphIdentification::IsHighPressureLongLiverangeNode(...)`

The goal is to document the actual policy surface that the spatial-split
subclass adds on top of the generic pressure-based splitter.

## 1. High-level result

The spatial subclass does not just inherit generic pressure splitting. It adds
policy in three places:

- whether a layer should count as a cluster boundary
- whether candidate tile counts remain valid under compressed-output constraints
- how splittable dimensions are chosen and how the cluster boundary is shifted
  to make those dimensions legal

This is the clearest point yet where Apple’s split system starts looking like a
domain-specific optimizer for spatial/legalization behavior, not just a generic
memory-pressure tool.

## 2. `IsClusterBoundary(...)`

Size:

- length: `1108` bytes
- basic blocks: `55`

### 2.1 What Hopper shows

This function is large and highly branchy.

It mixes several signals:

- generic splitter virtual predicates at:
  - `this + 0x28`
  - `this + 0x38`
  - `this + 0x40`
  - `this + 0x50`
- PE-layer checks
- broadcast-dimension checks through:
  - `ZinIrBroadcastInfo::HasDimension(...)`
- non-resident tensor checks
- `IsLayerSplittable<Subgraph>(...)`
- split-pattern / dimension availability
- and fallback checks involving:
  - `PressureBasedSubgraphIdentification::IsHighPressureLongLiverangeNode(...)`

The broad behavior is:

- reject many candidate boundaries immediately
- accept only those that satisfy both generic boundary conditions and
  spatial/broadcast-specific constraints

### 2.2 What this means

This is the strongest single proof so far that Apple’s spatial splitter is
specialized.

It is not just using generic “graph cut” criteria. It is checking things like:

- PE-layer boundary behavior
- broadcast dimension compatibility
- non-resident tensor pressure
- splittability after pattern-aware dimension analysis

## 3. `ValidateDimensionalConstraints(...)`

Size:

- length: `8` bytes
- basic blocks: `2`

### 3.1 What Hopper shows

The spatial subclass method itself is just:

- `CostBasedSubgraphIdentification::ValidateTileCountForOutputTensorCompression(...)`

### 3.2 What this means

The real dimensional constraint policy here is not hidden in the spatial
subclass body. It is delegated into the shared cost-based helper.

So the interesting logic is in the callee, not the wrapper.

## 4. `ValidateTileCountForOutputTensorCompression(...)`

Size:

- length: `728` bytes
- basic blocks: `38`

### 4.1 What Hopper shows

This helper scans the subgraph’s outputs and looks for compressed tensor-family
cases where tile count could be invalid.

It:

- enumerates output-side layers
- queries tensor family / descriptor state
- looks specifically for family kind `0x1f`
- checks a flag at `tensor + 0xdb`
- gathers candidate outputs requiring extra alignment logic
- computes a split-alignment constraint with:
  - `ZinMirSpatialSplitUtils::CalculateSplitAlignmentConstraintInHOnCompressedTensor(...)`
- rejects the candidate if the proposed tile count is too small for the
  compressed output shape

### 4.2 What this means

Some tile-count failures are not caused by input legality at all. They happen
because the chosen output tiling is incompatible with compressed output tensor
layout.

That is a useful mental model for the repo: output-side layout can invalidate a
split even when the split looked plausible from the input side.

## 5. `GetSplittableDimensionsForClusterLegalization(...)`

Size:

- length: `412` bytes
- basic blocks: `14`

### 5.1 What Hopper shows

This helper:

1. calls a virtual function at `this + 0x50`
2. builds a temporary dimension set
3. inspects the resulting dimension tree
4. if a certain dimension range falls into a narrow acceptable window
   (`<= 3` / `<= 4` style checks in the tree walk), it erases one candidate
   dimension from the set

### 5.2 What this means

The spatial subclass is not blindly taking every dimension returned by the
generic split analysis. It post-filters the candidate dimensions before later
cluster-legalization steps use them.

## 6. `PrioritizeSplittableDimensionsToCutCluster(...)`

Size:

- length: `612` bytes
- basic blocks: `37`

### 6.1 What Hopper shows

This helper repeatedly calls:

- `GetSplittableDimensionsForClusterLegalization(...)`

and compares the resulting dimension trees across layers in the cluster.

It also branches on configuration state and on whether the current tile-count
maps are already populated.

The main effect is:

- if splittable-dimension sets differ across the cluster, it reports that the
  cluster needs prioritized dimension handling

### 6.2 What this means

The spatial subclass is explicitly trying to choose a dimension cut strategy
that is coherent across the whole cluster, not just locally valid for one
layer.

## 7. `LegalizeClusterAtSplittableDimensionsChangingBoundary(...)`

Size:

- length: `644` bytes
- basic blocks: `42`

### 7.1 What Hopper shows

This helper:

1. gathers splittable dimensions for each layer in the cluster
2. compares those dimension sets
3. measures each layer’s position relative to the cluster midpoint
4. picks a boundary-shift candidate that best preserves compatible splittable
   dimensions
5. finally calls:
   - `SubgraphIdentification::CutClusterAtLayer(...)`

If it cannot find a usable candidate, it asserts:

- `error - LegalizeClusterWithSplittableDimensions`

### 7.2 What this means

This is a real boundary-movement policy.

Apple is willing to change *where* the cluster is cut in order to make the
spatially splittable dimensions line up better across the cluster.

That is a far more sophisticated strategy than fixed midpoint or fixed-width
cutting.

## 8. `IsHighPressureLongLiverangeNode(...)`

Size:

- length: `1156` bytes
- basic blocks: `61`

### 8.1 What Hopper shows

This helper is one of the pressure-policy inputs used by `IsClusterBoundary`.

It checks:

- whether the layer is an ANE layer
- whether it has many incoming edges
- whether the incoming producers are heterogeneous
- whether the spatial-split mode has certain string/config values
- whether incoming op kinds include concat-like op kind `7`

In some branches it also consults:

- peak pressure from `ZinIrMemoryPressureAnalyzer`
- schedule information
- a non-resident tensor map

### 8.2 What this means

The splitter has a special category of “high-pressure long-live-range node” and
uses that category to bias boundary decisions.

So not all high-pressure nodes are treated equally. Some are explicitly flagged
as better boundary candidates than others.

## 9. What this changes in our understanding

### 9.1 The spatial subclass is doing real policy work

This pass makes it clear that the spatial subclass contributes more than a few
format checks. It is making decisions about:

- acceptable boundary shapes
- acceptable output compression tile counts
- coherent cluster-wide split dimensions
- where to move the cluster boundary when those dimensions do not line up

### 9.2 Output compression is a first-class tile-count constraint

This is a meaningful result for `rustane`.

Some split plans can fail not because of input size or memory pressure, but
because the output tensor compression rules invalidate the tile count.

### 9.3 Apple is explicitly willing to move the cut boundary

The existence of `LegalizeClusterAtSplittableDimensionsChangingBoundary(...)`
proves that Apple treats the cluster boundary itself as an optimization variable.

That is a major difference from the repo’s current manual decomposition work,
which mostly assumes the problematic boundary is already fixed.
