# Hopper Validation Pass 32: Shared `ConstructSubGraphs` Virtual Slot

This pass targets the shared virtual slot we resolved manually from the vtables:

- `PressureBasedSubgraphIdentification` vtable slot `+0xb8`
- `BatchOrChannelSplitPressureBasedSubgraphIdentification` vtable slot `+0xb8`

Both point to:

- `0x1e4023324`

which sits inside the function Hopper currently mislabels as:

- `std::__1::map<...>::at(...)`
  with entrypoint `0x1e40232e4`

The goal is to document what that shared slot actually does.

## 1. High-level result

The unresolved virtual is now effectively resolved:

- it is **shared** between the pressure-based and batch/channel subclasses
- it is **not** a trivial wrapper
- it is a large shared cluster-to-subgraph orchestration function

Hopper’s symbol recovery is wrong here, but the call graph makes the function’s
role clear.

## 2. What the shared slot does

Despite the bad symbol, the function at `0x1e40232e4` clearly does the work we
were looking for.

Its major direct callees include:

- `SubgraphIdentification::ConstructSubGraph(...)`
- `SubgraphIdentification::NeedsReclusterAtWorkunitUtilizationLossBoundary(...)`
- `CostBasedSubgraphIdentification::LegalizeCluster(...)`
- `CostBasedSubgraphIdentification::IsMemoryFootprintReduced(...)`
- `ZinMirSpatialSplitUtils::BuildSplitIDGroupInfo(...)`
- the same `ConstructSubGraphs(...)` virtual at `this + 0xb8` for recursively
  handling refined clusters

That means this shared function is the real “cluster accepted as final subgraph,
or recurse/legalize/fail” orchestrator.

## 3. Main structure of the function

### 3.1 Initial checks

The function first:

- counts ANE layers in the cluster
- rejects clusters with exactly one ANE layer as too small / not worth keeping
- logs:
  - `Constructing Subgraph for Cluster [%zu,%zu]`

### 3.2 First construction attempt

It then:

1. calls `ConstructSubGraph(...)`
2. checks resulting subgraph size
3. logs:
   - `Too small of subgraph found`
   when the result is too small

### 3.3 Recluster / legalization branch

If the first construction succeeds, it then decides whether to keep it or
trigger more work:

- if `NeedsReclusterAtWorkunitUtilizationLossBoundary(...)` returns true:
  - it may call `LegalizeCluster(...)`
- otherwise it performs additional structural checks:
  - dominance relationship on the cluster
  - split-ID group consistency
  - memory-footprint reduction
  - availability of legal split dimensions

### 3.4 Recursive refinement

When legalization produces new cluster fragments, the function calls the same
shared `+0xb8` virtual again on each refined cluster.

So this slot is not only a constructor. It is also a recursion point for the
cluster-legalization pipeline.

## 4. Important visible outcomes / failure modes

The function can:

- reject tiny clusters
- reject illegal dominance relationships
- choose legalization because no legal split dimension was found
- accept the cluster and `emplace_back` a final `Subgraph`
- recurse into refined clusters after legalization/cutting

Visible log strings include:

- `Constructing Subgraph for Cluster [%zu,%zu]`
- `Too small of subgraph found`
- `Illegal Dominance Relationship`
- `Legalizing Cluster Due to no legal split dimension found`
- `Tile Count Found for Cluster [%zu,%zu]`

## 5. What this changes in our understanding

### 5.1 The “missing override” problem is practically solved

We still do not have a clean symbol name for the shared virtual target, but we
do now know:

- where the slot points
- that both subclasses share it
- and what it does structurally

That is enough for engineering understanding even if Hopper did not recover the
right C++ symbol.

### 5.2 Subgraph construction is a recursive decision engine

This shared slot is not just “construct subgraph and return.”

It is:

- initial construction
- evaluation
- optional reclustering
- optional legalization
- recursive processing of refined clusters
- final accept/reject

That is one of the strongest architectural results in the whole reversing
effort.
