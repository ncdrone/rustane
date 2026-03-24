# Hopper Validation Pass 20: `ZinMirTensorDimensionLegalizer::Execute`

This pass targets:

- `ZinMirTensorDimensionLegalizer::Execute(...)`

along with the two helper stages immediately around it:

- `CalculateSplitsForLegalization(...)`
- `ValidateLegalizedResults(...)`

The goal is to understand whether this pass is a real “shape ceiling /
legalization” stage relevant to `rustane`, and if so, how it behaves.

## 1. High-level result

Yes. This is a real shared legalization stage, and it looks directly relevant to
the kinds of shape/layout ceilings the repo has been hitting.

`ZinMirTensorDimensionLegalizer::Execute(...)` is not a giant per-op semantic
validator. It is a compact orchestration loop around:

- collecting subgraphs that need legalization
- attempting spatial splitting
- then validating the legalized result

That makes it one of the clearest internal “last structural gate” passes we’ve
seen so far.

## 2. `ZinMirTensorDimensionLegalizer::Execute(...)`

Size:

- length: `336` bytes
- basic blocks: `12`

Callers:

- `ZinValidationContext::ValidateGraphWithSavedLayerFusion(...)`
- `ZinMirPrepareLayers(...)`

Callees:

- `ZinMirLayerSplitterBase::ClearCollectedSubgraphs()`
- `ZinIrControlFlowGraph::TraverseForward(...)`
- `ZinMirLayerSplitterBase::HasCollectedSubgraphs()`
- `ZinMirLayerSplitterBase::TrySplitBySpace()`

### 2.1 What Hopper shows

The function loops like this:

1. clear collected subgraphs
2. traverse the control-flow graph forward with one legalization visitor
3. if that traversal reports an error:
   - return failure status `3`
4. if no subgraphs were collected:
   - run a second forward traversal to validate the result
   - if that validation fails:
     - assert:
       - `Validation after tensor dimension legalization result failed!\n`
   - otherwise return success
5. if subgraphs were collected:
   - call `TrySplitBySpace()`
   - if splitting fails, return failure status `3`
   - if splitting succeeds and no more subgraphs remain:
     - run the same validation traversal
     - assert on failure
   - otherwise loop and try again

### 2.2 What this means

This is a legalization loop, not just a single one-shot pass.

The compiler:

- identifies problematic subgraphs
- tries to legalize them by spatial splitting
- then re-validates the resulting graph
- repeating if necessary

That is highly relevant to the repo because it confirms there is an internal
compiler stage whose whole job is to “make illegal tensor dimensions legal,” and
it does so partly by splitting.

## 3. `CalculateSplitsForLegalization(...)`

Size:

- length: `240` bytes
- basic blocks: `5`

What Hopper shows:

- it first calls `CalculateSplitsForLayerLegalization(layer, hal_params)`
- for one specific layer kind (`0x5d` in the decomp), it also checks
  `HasOutgoingCWTranspose(...)`
- if present, it computes splits for the outgoing transpose target as well
- it then compares dimensions and may patch one dimension in the split result

### 3.1 What this means

The legalizer is not only considering one layer in isolation. It knows some
layers need companion/legalized handling across neighboring transpose structure.

That fits with the earlier observation that some width-limit failures are tied
to transpose-mapping and layout forms rather than only to top-level op shapes.

## 4. `CalculateSplitsForLayerLegalization(...)`

Size:

- length: `676` bytes
- basic blocks: `23`

This helper is more revealing than the name suggests.

What Hopper shows:

- it only works on ANE layers
  - otherwise:
    - `Tensor Dimension Legalizer works for ANE Layer only.`
- it collects incoming-layer node keys
- it computes split information once over symbols and once over incoming op layers
- it initializes an output tensor-dimension result
- then it applies dimension updates based on the split results
- it behaves differently depending on a layer property around `+0x240`
  - one branch uses `GetAllZinIrDimensions()` with special-case handling for
    certain dims
  - the other applies dimensions more directly

### 4.1 What this means

This is the stage where the compiler decides how a given ANE layer’s tensor
dimensions should be rewritten to fit legalization constraints.

It is not just “split counts” in the abstract. It is building a concrete new
dimension set for the layer after splitting/legalization decisions.

That makes it especially relevant to any repo-level question like:

- “what specific dimension forms does the compiler want after legalization?”

## 5. `ValidateLegalizedResults(...)`

Size:

- length: `1048` bytes
- basic blocks: `25`

What Hopper shows:

- it rebuilds tensor-info lists from the legalized graph inputs
- it collects input and output tensor format exceptions
- it collects input and output tensor dimension exceptions
- it runs `ZinLayerValidationUtils::ValidateTensorInfos(...)` on both sides
- it reconstructs a compact expected/output tensor-info object
- it compares the legalized result against the expected validation shape

So this is the proof stage after splitting/legalization:

- “did the transformed graph still satisfy the tensor-info contract?”

### 5.1 What this means

The legalizer does not just rewrite graph dimensions and trust them.

It explicitly re-validates the resulting graph against:

- format expectations
- dimension exceptions
- tensor-info consistency

That strengthens the idea that legalization is a major structural stage, not an
ad hoc patch.

## 6. Relevant failure strings

Directly tied strings include:

- `Tensor Dimension Legalization failure`
- `Validation after tensor dimension legalization result failed!\n`
- `TensorDimensionLegalizationFailure`
- `after_tensor_dimension_legalization`
- `Invalid concat for legalization`
- `JIT tile height legalization failed for ANE %d`
- `MIR Builder: Graph legalization for tensor based context switch failed!\n`
- `MIR Builder: Graph legalization for L2 failed!\n`
- `MIR Builder: Graph legalization for multi segment failed!\n`
- `MIR Builder: Graph legalization for latency failed!\n`

These strings make it clear that “tensor dimension legalization” is part of a
larger family of legalization passes, including:

- context-switch legalization
- L2 legalization
- multi-segment legalization
- latency legalization

So this pass sits in a broader internal ecosystem of graph-shape/resource
legalization.

## 7. What this changes in our understanding

### 7.1 Spatial splitting is not a speculative idea; it is part of Apple’s shared legalization loop

This is the biggest repo-relevant result.

`Execute(...)` explicitly:

- collects illegal subgraphs
- tries `TrySplitBySpace()`
- revalidates the result

That means the compiler’s own shared validation/lowering pipeline already uses
spatial splitting as a first-class remedy for illegal tensor dimensions.

### 7.2 Shape ceilings are not just one-off per-op errors

Earlier passes showed:

- transpose-specific width constraints
- ArgMinMax-specific split requirements

This pass shows there is also a higher-level shared mechanism that tries to
repair illegal tensor dimensions through subgraph-level legalization.

So the repo’s failures are likely a mix of:

- per-op semantic/layout constraints
- shared structural legalization failure

### 7.3 This is one of the most relevant framework passes for `rustane`

Because `rustane` is already fighting:

- width ceilings
- packed-layout limits
- shape-dependent compile failures

this pass is probably more immediately relevant than many of the lower-level
framework discoveries we’ve made.

## 8. Best next targets from here

The best next targets are:

1. `ZinMirLayerSplitterBase::TrySplitBySpace()`
   - to see exactly how spatial splitting is chosen/applied
2. the visitor used by the first `TraverseForward(...)` inside `Execute(...)`
   - to see how illegal subgraphs are identified
3. `ZinLayerValidationUtils::ValidateTensorInfos(...)`
   - to understand what exact tensor-info invariants must hold after legalization

For `rustane`, the highest-value next one is probably:

- `ZinMirLayerSplitterBase::TrySplitBySpace()`

because it is the most direct bridge between:

- the repo’s manual split/decomposition work
- Apple’s own shared legalization strategy.
