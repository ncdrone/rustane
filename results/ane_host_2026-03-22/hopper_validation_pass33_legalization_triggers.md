# Hopper Validation Pass 33: Recluster and Legalization Triggers

This pass targets the main decision hooks used by the shared cluster-construction
orchestrator:

- `SubgraphIdentification::NeedsReclusterAtWorkunitUtilizationLossBoundary(...)`
- `SubgraphIdentification::HasWorkUnitUtilizationLossAfterSplit(...)`
- `CostBasedSubgraphIdentification::IsMemoryFootprintReduced(...)`
- `CostBasedSubgraphIdentification::LegalizeCluster(...)`
- `SubgraphIdentification::CutClusterAtLayer(...)`
- `SubgraphIdentification::EstimateSizeOfKernelReads(...)`

The goal is to explain when the orchestrator decides that a cluster must be
reclustered or legalized instead of being accepted as-is.

## 1. High-level result

The shared cluster-construction path uses three major triggers:

- **work-unit utilization loss**
- **memory-footprint reduction failure**
- **absence of a legal split-dimension grouping**

If any of those go badly, it does not necessarily reject the cluster outright.
Instead it can run a legalization pipeline that cuts and reclusters the
candidate region.

## 2. `NeedsReclusterAtWorkunitUtilizationLossBoundary(...)`

Size:

- length: `248` bytes
- basic blocks: `15`

### 2.1 What Hopper shows

This helper walks the cluster’s output-side layers and calls:

- `HasWorkUnitUtilizationLossAfterSplit(...)`

for each one.

It returns `1` as soon as any layer reports a utilization-loss condition.

### 2.2 What this means

Work-unit utilization loss is treated as a cluster-level veto signal.
One bad output-side layer is enough to trigger the reclustering path.

## 3. `HasWorkUnitUtilizationLossAfterSplit(...)`

Size:

- length: `296` bytes
- basic blocks: `12`

### 3.1 What Hopper shows

This helper filters out several ineligible cases first, then checks:

- tensor format size in bytes
- spatial dimensions divided by the proposed tile count
- a threshold loaded from `*(*this->hal + 0x278)`

If the split makes the effective work unit too small relative to that threshold,
it reports utilization loss.

It also bypasses the check in several cases, including:

- certain configuration flags
- some layer categories marked through flag bits at `this + 0x4`

### 3.2 What this means

This is not a generic “small tensor bad” heuristic.
It is specifically a hardware/work-unit packing check against a HAL-derived
threshold.

That gives a concrete meaning to “utilization loss”: the split produces tiles
that underfill the hardware work unit.

## 4. `IsMemoryFootprintReduced(...)`

Size:

- length: `756` bytes
- basic blocks: `27`

### 4.1 What Hopper shows

This helper compares:

- extra non-resident boundary-tensor pressure introduced by the split
- against
- reduced kernel-read volume after tiling

It:

1. walks output-side layers
2. accumulates boundary tensor sizes, with special handling for concat-like
   output kind `7`
3. calls:
   - `EstimateSizeOfKernelReads(...)`
4. compares:
   - `original kernel-read estimate - tiled estimate`
   against
   - the added boundary-tensor cost (scaled in part by `*2`)

### 4.2 What this means

Apple is explicitly asking:

- “does the split actually reduce memory footprint once I account for extra
  boundary tensors?”

That is a much better decision rule than simply checking whether the split
reduced some local tensor size.

## 5. `EstimateSizeOfKernelReads(...)`

Size:

- length: `248` bytes
- basic blocks: `15`

### 5.1 What Hopper shows

This helper:

- walks ANE layers in the subgraph
- sums `EstimateKernelReadsPerNE(...)`
- optionally scales that value by tile-count terms

The scaling depends on:

- the chosen `TileCountType`
- a flag indicating whether to use the untiled or tiled estimate

### 5.2 What this means

Kernel-read volume is a first-class optimization metric in Apple’s split
decision process, not just a side detail.

## 6. `LegalizeCluster(...)`

Size:

- length: `1492` bytes
- basic blocks: `43`

### 6.1 What Hopper shows

This is the main fallback pipeline when the initial cluster is not acceptable.

It can:

- cut clusters at partial outputs
- remove border tensors with L2 allocation hints
- remove input/output no-op scaffolding
- recursively call the shared `+0xb8` cluster-construction slot on each refined
  cluster
- optionally route through other legalization callbacks via:
  - `this + 0xe0`
  - `this + 0xf8`
  - `this + 0x88`
  - `this + 0x90`

It also contains strong assertions like:

- `Splitted cluster must be smaller than the original cluster.`
- `Dram legalizer internal error`

### 6.2 What this means

`LegalizeCluster(...)` is a recursive cleanup-and-retry loop.
It does not try one alternative and stop. It keeps transforming the cluster
until:

- it finds refined fragments worth handing back to the shared slot
- or it hits one of its failure conditions

## 7. `CutClusterAtLayer(...)`

Size:

- length: `460` bytes
- basic blocks: `18`

### 7.1 What Hopper shows

This helper takes a concrete layer and splits the cluster into:

- layers before the cut
- layers after the cut

It then runs:

- `RemoveInputAndOutputNoopsOfClusters(...)`

on the resulting two-cluster vector.

### 7.2 What this means

When `LegalizeClusterAtSplittableDimensionsChangingBoundary(...)` picks a new
boundary layer, this is the primitive it uses to actually realize that boundary
move.

## 8. What this changes in our understanding

### 8.1 The shared slot has clear optimization vetoes

The cluster-construction pipeline is not just validating correctness. It has
explicit optimization vetoes for:

- underfilled work units
- poor memory-footprint tradeoffs

### 8.2 Legalization is an optimization repair path, not just a correctness fix

The existence of:

- `NeedsReclusterAtWorkunitUtilizationLossBoundary(...)`
- `IsMemoryFootprintReduced(...)`
- `LegalizeCluster(...)`

shows that legalization is being used to repair clusters that are *suboptimal*,
not only clusters that are outright illegal.

### 8.3 The cluster boundary is a tunable optimization variable

With `CutClusterAtLayer(...)` in the loop, Apple can move the cut point itself
when the current split boundary gives poor utilization or poor footprint
reduction.

That is one of the most important differences between the internal splitter and
the repo’s current manual decomposition strategy.
