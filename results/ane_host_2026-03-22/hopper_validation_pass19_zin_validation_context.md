# Hopper Validation Pass 19: `ZinValidationContext::Validate`

This pass targets:

- `ZinValidationContext::Validate(...)`

with the goal of understanding whether this is the real shared validation engine
behind the many layer-specific validation wrappers.

## 1. High-level result

`ZinValidationContext::Validate(...)` itself is thin, but it is the entry point
into the shared unit-validation pipeline.

Its structure is:

1. `CreateContextWithGraph(unit)`
2. `ValidateGraphWithSavedLayerFusion(context, parameters, saved_layer_fusion)`

So the interesting logic lives in those two internal phases, not in the wrapper
itself.

That means the shared validation engine is best understood as:

- build a temporary graph/context from one unit
- run a standard graph-validation/lowering pipeline over that temporary graph

## 2. `ZinValidationContext::Validate(...)`

Size:

- length: `120` bytes
- basic blocks: `3`

Direct callers:

- many `ValidateLayer_Impl<...>` instantiations, including SDPA
- `ZinIrUnit::Validate(...)`
- several unit-specific `Validate(...)` methods
- a few internal constraint checks

Direct callees:

- `ZinValidationContext::CreateContextWithGraph(...)`
- `ZinValidationContext::ValidateGraphWithSavedLayerFusion(...)`

### 2.1 What this means

This confirms that layer-specific exported validators are not each running their
own bespoke full validation pipelines. They ultimately funnel into the same
context-based graph validation engine.

That is important for the “map the framework” goal because it identifies a real
shared center of gravity in the validation stack.

## 3. `CreateContextWithGraph(...)`

Size:

- length: `1508` bytes
- basic blocks: `36`

### 3.1 What Hopper shows

This function takes a single `ZinIrUnit` and builds a temporary validation
context/graph around it.

Key steps visible in the decompiler:

1. derive a name from the unit and append a suffix:
   - effectively `... + "_live_view"`
2. build a `ZinObjectNameFactory`
3. create a fresh `ZinIrContext(...)`
   - with the current compiler parameters
4. set a hardcoded version string:
   - `1.0.14`
5. require the control-flow graph to contain a single basic block
   - otherwise:
     - `Having more than one basic block in validation.`
6. obtain the destination symbol from `ZinCcdmaLayerMirInfo::GetDstSymbol()`
7. perform dimension propagation on the unit
   - otherwise:
     - `Could not perform dimension propagation for unit "%s"`
8. propagate axis type on the unit
   - otherwise:
     - `Could not propagate axis type for unit "%s"`
9. create an output tensor via `ZinIrTensor::CreateTensor(...)`
10. create a layer from the unit
    - otherwise:
      - `Could not create layer from unit "%s"`
11. insert that layer into the temporary graph via:
    - `CreateGraphFromLayer(...)`

Other direct assertions:

- `Graph cannot be nullptr`
- `Undefined output_format`

### 3.2 What this means

This is not just “wrap the unit in a tiny shell.”

The validation context is already doing meaningful normalization work before the
main validation pipeline runs:

- naming
- version stamping
- dimension propagation
- axis-type propagation
- output tensor materialization
- graph creation from the unit

So if a unit cannot even be expressed cleanly as a single-block graph with
propagated dimensions/axis types, validation fails before any later optimization
or lowering pass runs.

For `rustane`, this is relevant because it shows that some unit-level validation
failures may happen before the deeper graph passes we’ve been analyzing.

## 4. `ValidateGraphWithSavedLayerFusion(...)`

Size:

- length: `888` bytes
- basic blocks: `27`

### 4.1 What Hopper shows

This function runs a fixed graph-processing pipeline over the temporary graph.

Directly visible sequence:

1. allocate/configure a validation config object
2. get the control-flow graph from the validation context
3. insert DMA-conversion type casts:
   - `ZinIrPreprocess::ZinIrIOInsertTypeCastForDmaConversions(...)`
4. traverse forward for composite/layer creation
   - failure:
     - `Layer composites creation failure`
5. run dead-code elimination:
   - `ZinIrOptDCE(...)`
6. traverse forward for layer lowering
   - failure:
     - `Layer lowering failure`
7. run pad optimization:
   - `ZinMirPadOptimization::Execute()`
   - failure:
     - `Pad layer optimization failure`
8. run layer fusion:
   - `ZinMirLayerFusion::Run(...)`
   - failure:
     - `Layer cannot be fused on ANE`
9. traverse forward for DMA texture configuration
   - failure:
     - `Layer DMA texture configuration failure`
10. traverse forward for lower-engine assignment
    - failure:
      - `Layer lower engine failure`
11. optionally run PE transpose fusion
    - when a compiler-parameter bit is enabled
    - failure:
      - `ZinMirPETransposeFusion failure`
12. run tensor dimension legalization:
    - `ZinMirTensorDimensionLegalizer::Execute(...)`
    - failure:
      - `Tensor Dimension Legalization failure`

### 4.2 What this means

This is the real shared validation engine shape:

- validation is not just local semantic checking
- it is a miniature compiler/lowering pipeline run on a temporary graph

That has two important consequences:

1. many “validation” failures are really early pipeline failures in:
   - preprocessing
   - lowering
   - fusion
   - dimension legalization
2. layer/unit validation and compile-time graph transformation are much more
   intertwined than the names initially suggest

So for framework mapping, this is one of the more important architectural
findings we have.

## 5. What this changes in our understanding

### 5.1 There is a shared mini-compiler inside validation

The validator is not just checking descriptors or tensor shapes in isolation.
It builds a temporary graph and runs:

- preprocess
- DCE
- pad optimization
- fusion
- engine assignment
- dimension legalization

That means the same kinds of passes that affect real compile behavior are being
exercised here too.

### 5.2 Layer-specific validators are mostly front doors into this shared engine

This reinforces the earlier result for SDPA:

- `_ANECValidateSDPALayer` is thin
- `ValidateLayer_Impl<...>` is mostly plumbing
- `ZinValidationContext::Validate(...)` and its internal phases are where the
  common behavior lives

### 5.3 Some repo-relevant failures may come from validation-pipeline passes

For `rustane`, this means future failures may need to be interpreted as:

- graph-preprocess failure
- fusion failure
- dimension-legalization failure

not only as:

- “semantic mismatch”
- or “low-level ANEC IR failure”

This is especially relevant if the repo starts experimenting with fused SDPA or
other higher-level semantic ops.

## 6. What this suggests we should target next

The best next targets are now:

1. `ZinIrPreprocess::ZinIrIOInsertTypeCastForDmaConversions(...)`
   - if we want to understand validation-time ABI/type normalization
2. `ZinMirLayerFusion::Run(...)`
   - if we want to understand what “can be fused on ANE” really means
3. `ZinMirTensorDimensionLegalizer::Execute(...)`
   - if we want to understand the last structural gate before unit acceptance
4. `CreateGraphFromLayer(...)`
   - if we want to understand exactly how units are reified into temporary
     validation graphs

For `rustane`, the highest-value next target is probably:

- `ZinMirTensorDimensionLegalizer::Execute(...)`

because that is likely the closest shared validation pass to the shape/layout
ceilings the repo keeps running into.
