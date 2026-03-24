# Hopper Validation Pass 27: Pressure Math Behind Tile Selection

This pass targets:

- `PressureBasedSubgraphIdentification::GetPeakPressure(...)`
- `PressureBasedSubgraphIdentification::ComputeSplitInvariantPressure(...)`

The goal is to understand the pressure arithmetic that sits underneath
`ComputeTileSize(...)`.

## 1. High-level result

Apple distinguishes between:

- **peak pressure** observed over a live range / candidate subgraph
- **split-invariant pressure** that remains even after tiling

That distinction is important because `ComputeTileSize(...)` compares the split
budget against:

- `peak pressure - split-invariant pressure`

not against raw tensor size alone.

So the splitter is explicitly reasoning about what tiling can and cannot buy
back.

## 2. `GetPeakPressure(...)`

Size:

- length: `2752` bytes
- basic blocks: `145`

### 2.1 What Hopper shows

This helper is large and clearly central.

It:

1. checks whether any layer in the subgraph has height-kernel support:
   - `ZinMirSpatialSplitUtils::HasKernelSupportOnHeight(...)`
2. calls two virtual helpers on the splitter object before walking the
   live-range payload
3. repeatedly asks:
   - `ZinIrMemoryPressureAnalyzer::GetPeakPressure(...)`
4. looks up mem-cache allocation metadata and per-layer TD instructions
5. computes tensor sizes with:
   - `ZinIrTensor::GetTensorSizeInBytesFromResidency(...)`
   - `PressureBasedSubgraphIdentification::GetTensorSize(...)`
6. uses:
   - `CostBasedSubgraphIdentification::TileComputationInputParameters::AggregatedNonSpatialFixedTileCount()`
7. accounts for:
   - over-computation pressure
   - copy pressure
   - chain-buffer effects via `ZinL2FootprintCalculator::GetChainBufferSize(...)`
   - DMA buffer pressure via `GetMinDMABufferSize(...)`
8. consults:
   - `PressureBasedSubgraphIdentification::IsSIPContributor(...)`
   - `PressureBasedSubgraphIdentification::ComputeOverComputationPressureForTensor(...)`

It materializes a vector of `MemoryPressureDivision` records containing at
least:

- baseline pressure
- additional pressure terms
- DMA/copy-related terms
- one trailing field that looks like split-eligible / contributor state

### 2.2 What this means

Peak pressure is not one scalar measured from a single tensor.

Apple is building it out of several interacting sources:

- direct tensor pressure
- chain/L2 pressure
- copy/DMA pressure
- over-computation due to tiling side effects

That matches the broader picture from the splitter passes: the compiler is
trying to model the cost of a tiling decision, not just the size of a layer.

## 3. `ComputeSplitInvariantPressure(...)`

Size:

- length: `1052` bytes
- basic blocks: `46`

### 3.1 What Hopper shows

This helper walks the live-range interval and accumulates pressure that remains
even if the cluster is tiled.

The function:

1. iterates scheduled layers across the live-range bounds
2. asks the memory-pressure analyzer for the peak-pressure contributors for each
   step
3. filters those contributors through:
   - `PressureBasedSubgraphIdentification::IsSIPContributor(...)`
4. treats some root tensors specially:
   - partial inputs
   - tensors not fully inside the subgraph
   - chained tensors
5. accounts for chain-buffer pressure via:
   - `ZinL2FootprintCalculator::GetChainBufferSize(...)`
6. uses:
   - `PressureBasedSubgraphIdentification::GetTensorSize(...)`
   - `ZinMirSpatialSplitUtils::IsChained(...)`
   - `SubgraphIdentification::IsPartialInput(...)`
   - `PressureBasedSubgraphIdentification::GetMinDMABufferSize(...)`
7. accumulates two totals:
   - a larger “all relevant pressure” total
   - a smaller split-invariant subtotal returned to the caller

### 3.2 What this means

The split-invariant subtotal is Apple’s answer to:

- “what pressure will still be there after I tile this cluster?”

That includes at least:

- tensors that remain structurally necessary
- DMA / chain-buffer pressure that tiling does not eliminate
- some partial-input and chained-root cases

So when `ComputeTileSize(...)` rejects a subgraph because:

- `SIP > budget`

it is specifically saying:

- even after discounting pressure that tiling could help with, the remaining
  unavoidable pressure is still too large

## 4. What this changes in our understanding

### 4.1 Apple’s tile budget is not based on gross tensor size

This is the biggest result of the pass.

The compiler is comparing budget against a pressure model that already factors
out some tiling-relievable pressure. That is much more nuanced than the repo’s
current hand-tuned chunk sizes.

### 4.2 Some split failures are likely “no worthwhile split exists”

Combined with `ComputeTileSize(...)`, this pass suggests that some compile-time
failures or non-splitting outcomes may mean:

- the graph is splittable in principle
- but every admissible split still leaves too much invariant pressure

That is a more useful mental model for `rustane` than assuming every bad case
is a hard semantic prohibition.

### 4.3 The splitter is already modeling the same categories the repo is
bumping into

The pressure math explicitly knows about:

- chained buffers
- partial inputs
- DMA buffers
- height-support asymmetries
- over-computation from tiling

Those are exactly the kinds of hidden costs that explain why a naive manual
split may compile but still perform badly or fail to generalize.
