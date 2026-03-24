# Hopper Validation Pass 25: Legalizer and Pressure-Based Subgraph Identification

This pass targets:

- `LegalizerSubgraphIdentification::AddSubgraph(...)`
- `PressureBasedSubgraphIdentification::IdentifySubgraphs(...)`
- `PressureBasedSubgraphIdentification::ConstructMemoryPressureMap(...)`

The goal is to document how Apple chooses and records candidate spatial-split
subgraphs before any actual graph rewrite happens.

## 1. High-level result

There are two distinct identification paths feeding the splitter:

- a **legalizer-driven** path that seeds a `Subgraph` directly from an illegal
  layer and split dimension
- a **pressure-driven** path that scans the scheduled graph, builds a memory
  pressure model, and extracts high-pressure live-range regions into subgraphs

This is important for `rustane` because Apple is not relying on one trigger for
spatial splitting. It can split either because a shape is illegal or because
the graph is too expensive under internal memory-pressure heuristics.

## 2. `LegalizerSubgraphIdentification::AddSubgraph(...)`

Size:

- length: `1892` bytes
- basic blocks: `65`

### 2.1 What Hopper shows

`AddSubgraph(...)` is the legalizer-side constructor for a candidate subgraph.
The function:

1. initializes a stack-local `Subgraph` object and its internal containers
2. records:
   - the seed layer
   - the chosen split dimension
   - several schedule-ordered trees / vectors / maps
3. checks whether the seed tensor is resident:
   - `ZinIrTensor::IsResident()`
4. seeds the subgraph’s layer sets differently depending on that residency
5. calls:
   - `IsLayerSplittable<Subgraph>(...)`
6. fetches the layer’s incoming layers and symbol arrays
7. verifies that any required incoming producers for the chosen split
   dimension are already represented in the collected subgraph
8. consults:
   - `PressureBasedSubgraphIdentification::MinimumSplitAlignmentConstraint(...)`
9. computes a per-dimension `TileCountType`
10. stores the resulting `Subgraph` into:
    - a `map<ZinIrOpLayer const*, vector<Subgraph>>`
    - a `map<ZinIrOpLayerGraph*, unordered_set<ZinIrOpLayer const*>>`

It returns `0` on success and `3` on failure.

### 2.2 What this means

This is not just “mark the layer as split-worthy.”

Apple is building a structured candidate object with:

- participating layers
- dimension-specific tile counts
- split-alignment constraints
- graph-level membership tracking

So the legalizer path is already richer than the repo’s current kernel-local
chunking logic.

## 3. `PressureBasedSubgraphIdentification::ConstructMemoryPressureMap(...)`

Size:

- length: `604` bytes
- basic blocks: `25`

### 3.1 What Hopper shows

This function builds the pressure model that later identification passes use.

The main sequence is:

1. create a traversal callback over the CFG
2. call:
   - `ZinIrControlFlowGraph::TraverseForward(...)`
3. accumulate tensor allocations into:
   - `ZinIrMemoryPressureAnalyzer`
4. special-case chain allocations via:
   - `CpAllocUtils::IsChain(...)`
   - `ZinL2FootprintCalculator::GetChainBufferSize(...)`
5. fetch fallback tensor size via:
   - `PressureBasedSubgraphIdentification::GetTensorSize(...)`
6. add allocations with:
   - `ZinIrMemoryPressureAnalyzer::AddTensorAllocation(...)`

It also contains a hard assertion that chaining must be disabled in one L2
circular-buffer analysis mode:

- `The chaining should be disabled in spatial split analysis with L2-circular buffer. This is because we can't enable chaining in L2-circular buffer.`

### 3.2 What this means

Pressure-based splitting is built on an explicit allocation model, not on a
rough tensor-count heuristic.

Apple is distinguishing at least:

- normal tensor allocations
- chain allocations
- L2 circular-buffer cases

That makes the pressure side of the splitter meaningfully closer to an internal
allocator simulation than to a simple “big graph bad” heuristic.

## 4. `PressureBasedSubgraphIdentification::IdentifySubgraphs(...)`

Size:

- length: `2144` bytes
- basic blocks: `96`

### 4.1 What Hopper shows

This is the outer coordinator for pressure-based identification.

The function:

1. asserts the graph has already been scheduled
   - `Must run scheduler first`
2. emits pressure-debug logging when enabled:
   - `INFO:: (SpatialSplit) --mem pressure--`
3. initializes:
   - a visited-bit vector
   - a list of identified live-range regions
   - temporary cluster containers
4. walks scheduled layers in order
5. queries:
   - `ZinIrMemoryPressureAnalyzer::GetPeakPressure(...)`
6. compares pressure against an internal budget field at `this + 0xad`
7. uses multiple gating checks around:
   - chainability
   - root tensor size
   - splitter configuration bits
8. when a high-pressure region is found, it calls:
   - `SubgraphIdentification::ExtractSubgraphsInLiveRange(...)`
9. later postprocesses the identified regions with:
   - `MergeContiguousHighPressureRegions(...)`
   - `TryExpandHighPressureRegions(...)`
10. inserts the resulting `Subgraph` vectors into the caller-owned result

### 4.2 What this means

Pressure-based identification is schedule-aware and iterative.

The compiler is not merely looking for one oversized tensor. It is tracking:

- where pressure peaks occur in scheduled execution order
- whether the region is chain-sensitive
- whether the region can be expanded or merged
- whether the resulting cluster extraction succeeds

That is a much better match for the performance/resource ceilings `rustane`
hits than a single static width rule.

## 5. What this changes in our understanding

### 5.1 There is a real split-subgraph object model

Earlier passes showed that Apple can split a graph. This pass shows that it has
explicit data structures for:

- candidate subgraphs
- tile counts by dimension / usage
- per-graph subgraph membership
- high-pressure region records

### 5.2 Legalization and pressure are separate triggers

This matters directly for the repo.

Some splits in Apple’s compiler happen because a layer is dimensionally illegal.
Others happen because memory pressure exceeds budget even when the graph may
still be semantically valid.

That means a future `rustane` split framework should not assume every split is
driven by a single kernel legality constraint.

### 5.3 The pressure model is allocation-aware

The presence of:

- chain-aware allocation handling
- L2-footprint calculator integration
- explicit tensor-allocation insertion into a pressure analyzer

means that Apple’s splitter is reasoning about real runtime memory behavior.

That is probably one reason the internal split system is more robust than the
repo’s current manual chunking.
