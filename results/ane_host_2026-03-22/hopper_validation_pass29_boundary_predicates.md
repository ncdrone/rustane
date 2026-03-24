# Hopper Validation Pass 29: Boundary Predicates and Cluster Cut Cleanup

This pass targets:

- `SubgraphIdentification::IsPartialInput(...)`
- `SubgraphIdentification::IsPartialOutput(...)`
- `SubgraphIdentification::CutClusterAtPartialOutputs(...)`
- `SubgraphIdentification::CutClusterAtConcatWithPartialInputs(...)`
- `SubgraphIdentification::MinimizeNewPartialOutputs(...)`
- `SubgraphIdentification::FindOutputNoOps(...)`
- `SubgraphIdentification::FindUnsupportedConcats(...)`
- `SubgraphIdentification::IdentifyConnectedClusters(...)`
- `SubgraphIdentification::RemoveBorderTensorsWithL2AllocationHint(...)`
- `SubgraphIdentification::RemoveInputAndOutputNoopsOfCluster(...)`

The goal is to document the actual predicates and cleanup steps behind the
cluster cutters used by `RefineSubgraph(...)`.

## 1. High-level result

The cluster cutters are driven by simple but important topological predicates:

- **partial input** = some incoming producers are inside the cluster and some are
  outside
- **partial output** = some outgoing consumers are inside the cluster and some
  are outside

Apple then wraps those predicates in additional cleanup:

- minimize newly created partial outputs
- split connected components after cuts
- remove border tensors with L2 allocation hints
- remove input/output no-op scaffolding

So the “cut cluster” phase is not a single graph split; it is a sequence of
predicate-guided cleanup transforms.

## 2. `IsPartialInput(...)`

Size:

- length: `164` bytes
- basic blocks: `7`

### 2.1 What Hopper shows

The function:

- iterates incoming layers of the candidate layer
- tracks whether it has seen:
  - at least one producer inside the cluster
  - at least one producer outside the cluster
- returns true only if both conditions occur

### 2.2 What this means

The partial-input predicate is purely about mixed cluster membership among the
incoming edges. There is no extra semantic condition layered on top.

## 3. `IsPartialOutput(...)`

Size:

- length: `164` bytes
- basic blocks: `7`

### 3.1 What Hopper shows

This is the output analogue of `IsPartialInput(...)`.

It:

- iterates the layer’s consumers through the global layer table walk
- tracks whether any consumers are:
  - inside the cluster
  - outside the cluster
- returns true only when both are present

### 3.2 What this means

Apple’s partial-boundary logic is symmetric:

- mixed internal/external producers => partial input
- mixed internal/external consumers => partial output

## 4. `CutClusterAtPartialOutputs(...)`

Size:

- length: `1020` bytes
- basic blocks: `51`

### 4.1 What Hopper shows

This helper:

1. scans the cluster for partial-output layers
2. logs:
   - `Cutting Cluster [%zu,%zu] at Partial Outputs`
   - `Cutting Cluster at Partial Output %s : %zu`
3. builds a sequence of cluster fragments
4. repeatedly calls:
   - `MinimizeNewPartialOutputs(...)`
5. erases empty fragments
6. appends the remaining fragments to the output vector

### 4.2 What this means

Partial-output cutting is followed immediately by a local optimization pass to
reduce the number of new partial outputs introduced by the split.

So Apple is not satisfied with “any legal cut”; it tries to make the cut cleaner
before handing the fragments onward.

## 5. `MinimizeNewPartialOutputs(...)`

Size:

- length: `636` bytes
- basic blocks: `36`

### 5.1 What Hopper shows

This helper walks candidate output-side layers, builds a work deque, and tries
to move additional layers between adjacent fragments when doing so reduces the
number of resulting partial outputs.

The key operations are:

- testing `IsPartialOutput(...)`
- checking whether a moved layer would introduce new partial outputs
- migrating layers between adjacent sets when the move is favorable

### 5.2 What this means

This is a local refinement heuristic layered on top of the coarse cut. Apple is
explicitly optimizing the cleanliness of cluster boundaries, not just their
existence.

## 6. `CutClusterAtConcatWithPartialInputs(...)`

Size:

- length: `980` bytes
- basic blocks: `49`

### 6.1 What Hopper shows

This is the concat/partial-input analogue of the partial-output cutter.

It:

1. identifies layers that are both:
   - partial inputs
   - concat ops (op kind `7`)
2. logs:
   - `Cutting Cluster [%zu,%zu] at Concat With Partial Input`
   - `Cutting Cluster at Partial Input %s : %zu`
3. partitions the cluster around those boundaries
4. removes empty partitions
5. appends the surviving partitions to the output vector

### 6.2 What this means

Apple treats concat-with-partial-input as a special bad pattern worth its own
cutting pass. That lines up with the earlier sanitation passes that singled out
unsupported concat structure.

## 7. `FindOutputNoOps(...)` and `FindUnsupportedConcats(...)`

Sizes:

- `FindOutputNoOps(...)`: `288` bytes
- `FindUnsupportedConcats(...)`: `444` bytes

### 7.1 What Hopper shows

`FindOutputNoOps(...)`:

- scans cluster layers
- checks `ZinIrOpLayer::IsNoOp(...)`
- excludes concat-like op kind `7`
- marks no-op layers that sit at output boundaries

`FindUnsupportedConcats(...)`:

- scans concat-like layers (again op kind `7`)
- detects clusters where some concat neighbors are inside and some are outside
- records the offending concat and related neighboring layers

### 7.2 What this means

These are the prefilters feeding `RemoveIllegalInternalNodes(...)`. They explain
why concat/no-op structure keeps reappearing in the construction pipeline:
Apple is proactively cleaning those patterns out at several stages.

## 8. `IdentifyConnectedClusters(...)`

Size:

- length: `1188` bytes
- basic blocks: `47`

### 8.1 What Hopper shows

This helper rebuilds connected cluster fragments after cuts.

It:

- logs the cluster time span when debugging is on
- seeds a BFS/DFS-style work deque with layer pairs
- checks cluster-constraint compatibility with:
  - `ZinClusterConstraintPerAne::CanBeInCluster(...)`
- tracks visited nodes in a hash set
- adds connected neighbors through incoming-edge walks
- emits one or more connected cluster sets into the output vector

### 8.2 What this means

After cluster cuts, Apple does not assume the result is still a single connected
component. It recomputes connectivity explicitly before later construction.

## 9. `RemoveBorderTensorsWithL2AllocationHint(...)`

Size:

- length: `504` bytes
- basic blocks: `23`

### 9.1 What Hopper shows

This helper removes cluster border layers whose incoming tensors carry an L2
allocation hint:

- reads a tensor-side structure at `tensor + 0xa0`
- checks a field at offset `+0x18 == 1`
- erases those border-driving layers from the cluster
- then re-runs:
  - `IdentifyConnectedClusters(...)`

### 9.2 What this means

The refinement pipeline is explicitly willing to drop border layers when their
tensors imply an L2-sensitive allocation pattern. That is a strong hint that
some cluster boundaries are rejected because they would interact badly with L2
buffering, not because of pure graph semantics.

## 10. `RemoveInputAndOutputNoopsOfCluster(...)`

Size:

- length: `696` bytes
- basic blocks: `31`

### 10.1 What Hopper shows

This helper walks a cluster, identifies no-op layers on its boundary, and then:

- checks whether they are partial inputs
- treats some op kind `0x24` specially
- removes the bad boundary no-op layers
- recomputes connectivity through:
  - `FindConnectedAcyclicClusters(...)`

### 10.2 What this means

Apple considers some no-op boundary structure acceptable and some not. When it
removes those no-ops, it immediately rechecks connectedness, which again shows
that cluster refinement is tightly coupled to graph topology cleanup.

## 11. What this changes in our understanding

### 11.1 The cutter predicates are simple, but the cleanup around them is not

The actual “partial input/output” tests are straightforward mixed-membership
checks. The complexity comes afterward:

- minimize new boundary damage
- restore connected components
- remove L2-sensitive border layers
- strip no-op scaffolding

### 11.2 Concat is a first-class trouble spot for the splitter

Concat shows up repeatedly:

- unsupported concat discovery
- concat-with-partial-input cutting
- pure-input concat dropping

That makes concat one of the clearest operation families where Apple’s internal
split framework is doing real graph repair work that `rustane` currently does
not generalize.

### 11.3 L2 hints and connectivity both matter during refinement

The presence of:

- `RemoveBorderTensorsWithL2AllocationHint(...)`
- `IdentifyConnectedClusters(...)`

shows that cluster refinement is balancing:

- memory-placement constraints
- and plain graph connectedness

That is a strong explanation for why Apple’s internal split system looks more
stable than simple local decompositions.
