# Hopper Validation Pass 30: Cluster Constraints and Acyclic Boundary Construction

This pass targets:

- the unresolved subclass construction hook behind `ConstructSubGraphs(...)`
- `ZinClusterConstraintPerAne::CanBeInCluster(...)`
- `SubgraphIdentification::FindConnectedAcyclicClusters(...)`
- `SubgraphIdentification::AddBoundaryTensorsForAcyclicCluster(...)`
- `SubgraphIdentification::AddBoundaryTensorsByForkJoinPairs(...)`
- `SubgraphIdentification::AddBoundaryTensorBySchedule(...)`

The goal is to document the policy that decides whether a refined cluster can be
turned into an acyclic split candidate with valid boundary tensors.

## 1. High-level result

The “cluster accepted as subgraph” decision is constrained by three layers:

- a still-unresolved virtual construction hook in `ConstructSubGraphs(...)`
- a per-ANE cluster-constraint object
- an acyclic-cluster boundary-construction path that can fail if it cannot find
  valid fork/join or schedule cut points

So the compiler is not only asking whether a cluster is connected and legal to
split. It is also asking whether:

- the cluster can live on a coherent ANE/bonded placement
- the cluster can be made acyclic
- the cluster can be assigned valid boundary tensors

## 2. The `ConstructSubGraphs(...)` virtual override

### 2.1 What Hopper directly shows

We can directly confirm:

- `ConstructSubGraphs(...)` dispatches through a virtual call at `this + 0xb8`
- the base symbol:
  - `ConstructValidSubgraphsFromCluster(...)`
  is just a stub returning `5`
- `PressureBasedSubgraphIdentification` and
  `BatchOrChannelSplitPressureBasedSubgraphIdentification` install distinct
  vtable pointers in their constructors

### 2.2 What did not surface

Hopper did **not** surface a named concrete override for that slot on this
build.

So the best direct conclusion is:

- the real subclass implementation exists
- but Hopper did not recover it with a stable symbol name
- the generic `ConstructSubGraph(...)` sanitation/extraction path documented in
  the prior pass remains the clearest concrete construction logic we can see

That is a real limit of this pass, not something I can honestly resolve from
the current recovered symbol surface alone.

## 3. `ZinClusterConstraintPerAne::CanBeInCluster(...)`

Size:

- length: `180` bytes
- basic blocks: `10`

### 3.1 What Hopper shows

This helper enforces a simple but important rule:

- once the constraint object has been seeded from one ANE-layer’s bonded info,
  later ANE layers may join the cluster only if their bonded info is compatible

The function:

- checks whether the candidate layer is an ANE layer
- if the constraint object is uninitialized:
  - seeds it from `ZinBondedInfo::GetAneIndexHint()`
- if it is already initialized:
  - compares either:
    - explicit ANE index hint
    - or the bonded-info identity field

### 3.2 What this means

Connectedness alone is not enough.

Apple is refusing to put layers into the same connected cluster if their bonded
ANE placement hints are inconsistent.

That is one of the clearest policy constraints we have seen so far beyond pure
shape/legality checking.

## 4. `FindConnectedAcyclicClusters(...)`

Size:

- length: `512` bytes
- basic blocks: `21`

### 4.1 What Hopper shows

This helper repeatedly:

1. initializes a temporary cluster vector
2. calls:
   - `IdentifyConnectedClusters(...)`
3. then, for each connected component, calls:
   - `AddBoundaryTensorsForAcyclicCluster(...)`
4. if that succeeds, the component is kept as an acyclic cluster candidate
5. otherwise it is deferred or discarded

### 4.2 What this means

Apple treats “connected cluster” and “usable acyclic cluster” as different
things.

The acyclic-cluster filter is not just a connectivity test. It requires a
successful boundary-tensor construction step.

## 5. `AddBoundaryTensorsForAcyclicCluster(...)`

Size:

- length: `968` bytes
- basic blocks: `32`

### 5.1 What Hopper shows

This is one of the most informative cluster-construction helpers we have seen.

It:

1. logs the cluster span and, in debug mode, every layer in the cluster
2. calls a virtual helper at `this + 0x60` to derive:
   - input nodes
   - output nodes
3. builds temporary path / fork-join / schedule structures
4. calls:
   - `AddBoundaryTensorsByForkJoinPairs(...)`
5. if that does not settle the boundary set, it falls back to:
   - `AddBoundaryTensorBySchedule(...)`

It also asserts:

- `Cannot have a cluster without input nodes and output nodes`

### 5.2 What this means

Boundary construction is not a trivial “take cluster I/O tensors” step.
Apple tries a more semantic fork/join-based boundary construction first, and
only falls back to a schedule-based cut-point heuristic if needed.

## 6. `AddBoundaryTensorsByForkJoinPairs(...)`

Size:

- length: `1044` bytes
- basic blocks: `50`

### 6.1 What Hopper shows

This helper:

1. iterates cyclic path records
2. resolves a fork point and a join point for each path through a virtual helper
3. logs:
   - `Fork Point ...`
   - `Join Point ...`
4. if both exist, inserts the associated boundary tensors into the result set
5. if either is missing, it hard-fails with assertions like:
   - `Both fork and join point must be found`

### 6.2 What this means

Apple prefers semantically meaningful boundary tensors derived from graph fork /
join structure, not just arbitrary schedule cuts.

That is a much stronger notion of boundary quality than anything the repo’s
current manual decomposition logic attempts.

## 7. `AddBoundaryTensorBySchedule(...)`

Size:

- length: `800` bytes
- basic blocks: `39`

### 7.1 What Hopper shows

This is the fallback path.

It:

1. computes the cluster’s schedule midpoint
2. scores candidate layers near that midpoint
3. chooses a cut-point whose schedule distance is favorable
4. checks whether that cut-point yields a boundary tensor that is admissible
5. if successful, inserts the chosen tensor into the boundary set

Failure paths include:

- `Could not find cut-point`
- `Boundary Tensor Internal Error`
- `Cluster subdividing algorithm failure`

### 7.2 What this means

When Apple cannot find a clean fork/join-derived boundary tensor, it falls back
to a schedule-based heuristic cut. But that path is still heavily validated and
can fail outright.

## 8. What this changes in our understanding

### 8.1 Cluster construction is constrained by ANE placement, not just graph topology

`CanBeInCluster(...)` is a strong signal that cluster membership is tied to
bonded/ANE placement hints. This is likely one reason some graph fragments are
rejected even when they are otherwise connected and dimensionally legal.

### 8.2 Acyclic-cluster acceptance depends on boundary quality

The compiler is not satisfied with a connected component unless it can also
assign convincing boundary tensors to it. That is a structural quality test,
not just a legality test.

### 8.3 The concrete virtual override remains unresolved, but the effective
policy is still visible

Even though Hopper did not surface the concrete override symbol behind
`ConstructSubGraphs(...)`, the actual construction policy is already clear from
the helpers it relies on:

- cluster-constraint gating
- acyclic-cluster extraction
- fork/join boundary synthesis
- schedule cut fallback
