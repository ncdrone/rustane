# Hopper Validation Pass 17: SDPA Validation and Lowering

This pass targets the three SDPA-specific functions requested:

- `ZinSDPALayer::ValidateSemantics_Impl(...)`
- `_ANECValidateSDPALayer`
- `ZinSDPALayer::Lower(...)`

The goal is to understand:

- what semantic constraints Apple imposes on the fused SDPA unit
- what the exported layer-validation boundary looks like
- how the fused SDPA unit is lowered into actual graph/layer structure

## 1. High-level result

These functions confirm that Apple’s fused SDPA path is real and structured:

- `MILOpConverter::SDPA(...)` creates a dedicated `ZinIrSDPAUnitInfo`
- `ZinSDPALayer::ValidateSemantics_Impl(...)` enforces a concrete shape/layout
  contract on that unit
- `ZinSDPALayer::Lower(...)` decomposes the fused unit into a sequence of
  concrete lower-level layers
- `_ANECValidateSDPALayer` is only the exported wrapper around the generic
  descriptor-validation template

So the fused SDPA path exists, but it still has strict semantic constraints and
is ultimately lowered into more primitive internal layers.

## 2. `ZinSDPALayer::ValidateSemantics_Impl(...)`

Size:

- length: `416` bytes
- basic blocks: `23`

### 2.1 What Hopper shows

The function first calls the base layer semantics validator:

- `ZinIrOpLayer::ValidateSemantics_Impl(...)`

Then it enforces SDPA-specific conditions on the input tensor set.

Directly visible constraints:

- SDPA must have exactly:
  - `4` inputs, or
  - `5` inputs when an optional mask is present
- `Q` and `K` must have the same embedding size
  - asserted via:
    - `Q and K must have same embedding size i.e W dim`
- `K` and `V` must have the same sequence length
  - asserted via:
    - `K and V must have same sequence length i.e C dim`
- `Q`, `K`, and `V` must have the same tensor format
  - asserted via:
    - `Q, K and V must have same format`
- scale must be constant
  - asserted via:
    - `Scale is expected to be constant.`

If a mask is present, the following additional constraints are enforced:

- mask format must match `Q/K/V`
  - `Mask format must be same as Q, K and V`
- mask width axis must match the `K/V` channel axis
  - `Mask Width axis must match K and V Channel axis`
- mask channel axis must either:
  - match `Q` channel axis, or
  - be broadcastable
  - `Mask Channel axis must match Q Channel axis or broadcastable`

### 2.2 What this means

This is the clearest answer so far to “would Apple’s fused SDPA op help
`rustane` automatically?”

Not automatically.

The fused path exists, but it expects:

- a very specific semantic input contract
- compatible tensor formats
- a constant scale
- tightly constrained mask broadcasting

So any future experiment with fused SDPA in the repo would need to satisfy this
contract, not just rename the op.

## 3. `_ANECValidateSDPALayer`

This function is tiny:

- length: `20` bytes
- assembly shows it only rearranges arguments and jumps

Hopper assembly:

```asm
_ANECValidateSDPALayer:
    mov x5, x4
    mov x4, x3
    mov w3, #0x4
    mov x6, #0x0
    b __Z13ValidateLayerI17ANECSDPALayerDesc13ZinIrSDPAUnit17ZinIrSDPAUnitInfo26ANECSDPALayerDescAlternate...
```

The specialized template itself is also trivial:

- it just calls `ValidateLayer_Impl<ANECSDPALayerDesc, ZinIrSDPAUnit, ZinIrSDPAUnitInfo, ANECSDPALayerDescAlternate>(...)`

### 3.1 What this means

`_ANECValidateSDPALayer` adds almost no SDPA logic of its own.

Its role is:

- expose the SDPA layer descriptor to the generic ANEC layer-validation
  machinery
- fix the expected input count / argument layout for that template instantiation

So if we want real descriptor-level constraints, the useful target is not
`_ANECValidateSDPALayer` itself but the generic `ValidateLayer_Impl` path or the
semantic layer validator above it.

## 4. `ZinSDPALayer::Lower(...)`

Size:

- length: `3476` bytes
- basic blocks: `58`

This is the most revealing function of the three.

### 4.1 Input handling

The function first gathers incoming layers and enforces:

- exactly `4` or `5` bottoms
  - otherwise:
    - `4 or 5 bottoms must be present for SDPA`

It then binds:

- `query`
- `key`
- `value`
- `scale`
- optional `mask`

from those incoming layers.

### 4.2 Lowering structure

The decompilation shows the fused SDPA unit being lowered into a chain of
primitive layers:

1. create a transpose on one incoming path
   - via `ZinBuilder::CreateTranspose(...)`
2. create a matmul layer
   - this appears to be the `Q @ K^T`-style product
   - via `ZinBuilder::CreateMatMulLayer(...)`
3. read the constant scale value from `ZinIrConstData`
4. create a constant scale/bias GOC layer
   - via `ZinBuilder::CreateConstScaleAndBiasGOC(...)`
5. if a mask exists:
   - create an elementwise layer that applies the mask
   - via `ZinBuilder::CreateElementWiseLayer(...)`
6. create a softmax layer
   - via `ZinBuilder::CreateSoftmaxLayer(...)`
7. create another matmul layer
   - this appears to be the attention weights times `V`
   - via `ZinBuilder::CreateMatMulLayer(...)`
8. move outgoing edges from the original SDPA node to the final lowered node
   - `ZinIrOpLayerGraph::MoveOutgoingEdges(...)`
9. remove the original fused SDPA node
   - `ZinIrOpLayerGraph::RemoveNode(...)`

### 4.3 What this means

This is a crucial clarification:

- Apple has a fused semantic SDPA op in the validator/converter
- but the SDPA layer still gets decomposed into a lower-level internal graph

So the benefit of the fused path is not necessarily “one magical primitive all
the way down.” The benefit is that:

- Apple controls the decomposition
- Apple chooses the layouts
- Apple materializes the scale
- Apple enforces the mask contract

That is still materially different from the repo’s current hand-expanded MIL,
but it is not a proof that the final hardware layer is a single opaque SDPA
opcode.

## 5. What this changes in our understanding

### 5.1 Fused SDPA is a semantic convenience layer with strict constraints

The compiler has a dedicated SDPA semantic surface, but it is bounded by:

- exact input count
- constant scale
- specific format compatibility
- specific mask broadcasting rules

### 5.2 Lowering is still decomposition, but Apple-owned decomposition

`rustane` already decomposes attention manually.

Apple also decomposes, but at a later internal stage and under compiler-owned
rules. That difference is important:

- Apple can normalize layouts and constants before decomposition
- Apple can reject or rewrite cases based on SDPA-specific semantics

### 5.3 The most likely value of a fused-SDPA experiment in `rustane`

If the repo ever experiments with the fused op, the payoff would likely be:

- better semantic alignment with Apple’s validator/converter
- compiler-managed layout conversion and scale handling

not necessarily:

- a guaranteed bypass of all width/resource ceilings

## 6. Repo-relevant implications

For `rustane`, this suggests:

- a fused `scaled_dot_product_attention` experiment is now technically
  well-motivated
- but it should be treated as an experiment in semantic alignment, not assumed
  performance magic

The strongest immediate questions would be:

- can the repo produce MIL that satisfies the fused SDPA semantic contract?
- does that contract allow the repo’s current mask/broadcast shapes?
- does the Apple-managed decomposition avoid some of the compile failures seen
  with the hand-expanded graph?

## 7. Best next targets from here

If we keep pushing on this SDPA thread, the next best targets are:

1. `ValidateLayer_Impl<ANECSDPALayerDesc, ...>`
   - if we want descriptor-level validation details beyond the thin wrapper
2. `ZinBuilder::CreateConstScaleAndBiasGOC(...)`
   - if we want to understand how the SDPA scale is encoded materially
3. `ZinBuilder::CreateSoftmaxLayer(...)`
   - if we want to understand any special softmax constraints in this path

If we pivot back to the repo strategy question, the current evidence supports:

- fused SDPA is worth considering as a future experiment
- but compile-cache / descriptor identity still looks like the more immediate
  high-leverage path for the branch overall
