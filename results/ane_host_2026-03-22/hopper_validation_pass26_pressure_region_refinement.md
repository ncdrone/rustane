# Hopper Validation Pass 26: Pressure-Region Expansion and Refinement

This pass targets:

- `PressureBasedSubgraphIdentification::MergeContiguousHighPressureRegions(...)`
- `PressureBasedSubgraphIdentification::TryExpandHighPressureRegion(...)`
- `PressureBasedSubgraphIdentification::TryExpandHighPressureRegions(...)`
- `PressureBasedSubgraphIdentification::ComputeTileSize(...)`
- `PressureBasedSubgraphIdentification::RefineSubgraph(...)`
- `PressureBasedSubgraphIdentification::SplitClusterViaGuidance(...)`
- `SubgraphIdentification::ConstructSubGraphs(...)`

The goal is to document what Apple does after it has identified an initial
high-pressure region.

## 1. High-level result

Apple’s pressure-based splitter does not stop after saying “this range looks
bad.”

It then:

1. merges adjacent regions
2. tries to expand them left/right in schedule space
3. computes tile counts subject to budget and alignment
4. refines the extracted clusters with multiple cut/postprocess passes
5. finally re-materializes those clusters as `Subgraph` objects

This is the strongest evidence so far that the internal splitter is a true
subgraph-optimization framework rather than a local tensor-chunking patch.

## 2. `MergeContiguousHighPressureRegions(...)`

Size:

- length: `324` bytes
- basic blocks: `10`

### 2.1 What Hopper shows

This helper clones the region list, walks it in order, and merges adjacent
entries when:

- the current live-range end equals the next live-range start minus one

When that happens it:

- extends the previous range
- marks the merged result as valid
- clears the old subgraph vector payload from the superseded entry

### 2.2 What this means

Pressure regions are not treated as sacred one-shot detections.
Apple first normalizes the region list into larger schedule-contiguous regions
before later expansion/refinement.

## 3. `TryExpandHighPressureRegion(...)`

Size:

- length: `380` bytes
- basic blocks: `16`

### 3.1 What Hopper shows

This is the single-region expansion routine.

It:

1. receives a live range plus left/right schedule bounds
2. checks whether it should first expand backward or forward based on a config
   bit
3. repeatedly calls:
   - `SubgraphIdentification::ExtractSubgraphsInLiveRange(...)`
4. expands the candidate range geometrically
   - by doubling the step size on each attempt
5. tries at most a small fixed number of expansions per direction
6. records success by updating the live-range pair in-place
7. marks failure by setting the result-valid bit

### 3.2 What this means

Apple is not just growing regions one layer at a time. It is using a bounded
doubling search over the schedule range to find a better expansion quickly.

That is a notably more structured search policy than the repo’s current manual
splits.

## 4. `TryExpandHighPressureRegions(...)`

Size:

- length: `908` bytes
- basic blocks: `33`

### 4.1 What Hopper shows

This function manages the full list of high-pressure regions.

It:

1. clones the current region vector into a list
2. iterates each region entry
3. computes left/right bounds relative to neighboring regions
4. calls:
   - `TryExpandHighPressureRegion(...)`
5. if expansion succeeds, rewrites the current list entry in place
6. if expansion fails, it may erase the current list node and insert a newly
   expanded one elsewhere
7. if any list surgery becomes inconsistent, it sets the status flag
8. on success, copies the final list back into the original vector

### 4.2 What this means

This is a real region-management stage, not just a loop around a helper.
Apple is willing to reorder / replace the identified regions after expansion,
which means the pressure-region list is treated as an intermediate working
representation, not a final answer.

## 5. `ComputeTileSize(...)`

Size:

- length: `1160` bytes
- basic blocks: `39`

### 5.1 What Hopper shows

This is one of the most important pressure-side helpers.

The function:

1. computes the cluster time span
2. calls:
   - `PressureBasedSubgraphIdentification::GetPeakPressure(...)`
   - `PressureBasedSubgraphIdentification::ComputeSplitInvariantPressure(...)`
3. derives remaining pressure budget from:
   - the splitter budget field at `this + 0xad`
4. explicitly bails with:
   - `INFO:: (SpatialSplit) Can't tile subgraph b/c SIP > budget`
5. iterates candidate per-layer pressure records
6. may call:
   - `ResetLayerCausingTooMuchPressure(...)`
7. updates tile counts for:
   - bidirectional splitting
   - batch/channel splitting
8. optionally records a legalization tile count when
   `SplitConfiguration::UseLegalizeCluster()` is enabled

### 5.2 What this means

Tile-count choice is budgeted against a quantity Apple calls split-invariant
pressure, not just total tensor size.

That suggests Apple distinguishes:

- pressure you can reduce by tiling
- pressure that will remain no matter how you tile

That is a very useful mental model for `rustane`’s future decomposition work.

## 6. `RefineSubgraph(...)`

Size:

- length: `560` bytes
- basic blocks: `13`

### 6.1 What Hopper shows

This function refines the cluster set before final subgraph construction.

Depending on configuration, it may:

- call `SplitClusterViaGuidance(...)`
- cut clusters at reset layers
- cut clusters at concat nodes with partial inputs
- remove border tensors with L2 allocation hint
- cut clusters at partial outputs
- then call `SubgraphIdentification::ConstructSubGraphs(...)`

It loops until the cluster count stops shrinking.

### 6.2 What this means

Apple’s “refinement” stage is mostly about cleaning up cluster boundaries until
they are acceptable for later construction.

That is exactly the kind of graph-level hygiene missing from local
kernel-by-kernel chunking.

## 7. `SplitClusterViaGuidance(...)`

Size:

- length: `408` bytes
- basic blocks: `17`

### 7.1 What Hopper shows

This function walks a cluster and partitions it according to an externally
provided guidance set.

It:

- iterates the schedule-ordered cluster
- starts a new partition when it encounters a guided boundary layer
- calls:
  - `SubgraphIdentification::RemoveInputAndOutputNoopsOfCluster(...)`
- appends each finalized cluster partition into the output vector

### 7.2 What this means

The splitter supports externally guided cluster boundaries, not just automatic
pressure-derived ones. That matches the broader picture from
`TileWithGlobalRefinement(...)`: Apple keeps multiple ways to reshape a
candidate cluster before final tiling.

## 8. `SubgraphIdentification::ConstructSubGraphs(...)`

Size:

- length: `364` bytes
- basic blocks: `10`

### 8.1 What Hopper shows

This function is the final bridge from cluster sets to `Subgraph` objects.

It:

1. iterates the refined cluster vector
2. calls a virtual constructor/helper on the owning splitter object
3. passes:
   - the graph
   - the cluster set
   - the status object
   - two boolean-ish flags (`0`, `1`)
4. logs:
   - `SIP error during subgraph identification. cluster index: %zu`
   if construction fails

### 8.2 What this means

The pressure-based path does not construct `Subgraph` objects directly during
every intermediate step. It carries cluster sets around, refines them, then
only at the end converts them into concrete `Subgraph` records.

That separation likely makes the refinement logic cheaper and more flexible.

## 9. What this changes in our understanding

### 9.1 Apple’s splitter uses explicit search and refinement

We now have direct evidence for:

- merging
- geometric range expansion
- tile-size computation under pressure budget
- boundary-guided repartitioning
- repeated cluster cleanup

So “pressure-based splitting” is really a pipeline, not a heuristic toggle.

### 9.2 Split count is constrained by pressure math, not just legality

The `SIP > budget` bail-out in `ComputeTileSize(...)` is one of the clearest
resource-side constraints we have seen.

That makes it likely that some large `rustane` graphs fail not because they are
semantically unsupported, but because the internal split search decides no
budget-feasible tiling exists.

### 9.3 Apple separates region discovery from subgraph construction

The pipeline now looks like:

1. detect high-pressure ranges
2. merge / expand ranges
3. extract cluster sets
4. refine those cluster sets
5. compute tile counts
6. construct final `Subgraph` records

That separation is probably one reason the internal splitter can be reused by
multiple legalization/optimization passes.
