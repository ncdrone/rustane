# Hopper Validation Pass 16: `scaled_dot_product_attention` Handler

This pass targets the handler behind the MIL op registry entry:

- `scaled_dot_product_attention`

The most relevant function in `ANECompiler` is:

- `MILOpConverter::SDPA(...)`

with its dedicated builder path:

- `ZinMILUnitBuilder::CreateUnit<ZinIrSDPAUnitInfo>(...)`

## 1. High-level result

Apple’s compiler does have a **fully dedicated fused SDPA conversion path**.

This is not just a registry alias that eventually falls back to a generic
matmul/softmax/reshape decomposition. The handler:

- recognizes `scaled_dot_product_attention` as a first-class op
- builds a dedicated `ZinIrSDPAUnitInfo`
- injects its own layout-conversion steps
- materializes the scale constant itself
- finalizes through a specialized SDPA unit path

That is the strongest direct signal so far that Apple treats SDPA as a semantic
primitive in this compiler layer.

## 2. Why this is the right handler

From the previous pass:

- `GetMILConversionMaps()` contains the op name:
  - `scaled_dot_product_attention`

From this pass:

- the most directly relevant conversion routine in the image is:
  - `MILOpConverter::SDPA(...)`
- it has SDPA-specific unit types and helpers in the same binary:
  - `ZinIrSDPAUnitInfo`
  - `ZinIrSDPAUnit`
  - `ZinSDPALayer::Lower(...)`
  - `ZinSDPALayer::ValidateSemantics_Impl(...)`
  - `_ANECValidateSDPALayer`

There are no direct normal callers to `MILOpConverter::SDPA(...)`, which is
consistent with it being stored into the `std::function` registry built by
`GetMILConversionMaps()` rather than called statically.

## 3. `MILOpConverter::SDPA(...)`

Size:

- length: `3372` bytes
- basic blocks: `84`

This is large enough to be a real conversion path, but still focused enough to
understand structurally.

## 4. Inputs are declared explicitly as SDPA semantics

The handler distinguishes two shapes of SDPA op by input count.

If there are `4` inputs, it declares:

- `query`
- `key`
- `value`
- `attn_mask`

Otherwise it declares:

- `query`
- `key`
- `value`

This is direct evidence that the compiler sees SDPA in semantic input roles, not
just as an anonymous list of tensors.

## 5. Dedicated layout conversions for SDPA inputs

The handler builds explicit named conversion/layout steps for:

- `__@convert_query_layout`
- `__@convert_key_layout`
- `__@convert_value_layout`
- `__@convert_attn_mask_layout` (only when the 4th input exists)
- `__@convert_output_layout`

For the query/key/value paths, Hopper shows it constructing layout maps with:

- rank `3` -> `NCW`
- rank `4` -> `NHCW`

and then calling:

- `ZinMILUnitBuilder::DeclareCustomOperationInputLayout(...)`

So SDPA is not lowered with generic “whatever current layout is fine” rules.
It has dedicated expected input-layout conversion steps.

That is a strong clue that some SDPA-related compiler behavior may depend on
this fused path’s layout assumptions rather than only on the raw tensor shapes
that `rustane` currently manipulates.

## 6. The scale factor is materialized inside the handler

One of the clearest direct behaviors in the decompiler:

- it reads the output/input tensor shape
- takes the last relevant dimension
- computes:
  - `1 / sqrt(dim)`
- converts that to `MIL::Fp16`
- creates a constant info object via:
  - `MILOpConverter::CreateConstInfo<MIL::Fp16>(...)`
- then registers it with:
  - `MILOpConverter::AddConstInfo(...)`

So the canonical SDPA scaling factor is not assumed to already exist as a user
input. The compiler injects it itself as part of the fused SDPA lowering.

This matters because it means an Apple-style fused SDPA op is not just a
“shortcut syntax” for the graph `rustane` is currently emitting. It comes with
its own compiler-managed constant materialization.

## 7. The handler lowers into a dedicated SDPA unit

This is the most important direct structural finding.

`MILOpConverter::SDPA(...)` calls:

- `ZinMILUnitBuilder::CreateUnit<ZinIrSDPAUnitInfo>(...)`

That is a dedicated unit type, not a generic matmul or generic elementwise
unit.

So the compiler path is:

- MIL op `scaled_dot_product_attention`
- specialized converter `MILOpConverter::SDPA`
- specialized unit info `ZinIrSDPAUnitInfo`
- later specialized layer/unit validation and lowering:
  - `ZinIrSDPAUnit`
  - `ZinSDPALayer::Lower`
  - `_ANECValidateSDPALayer`

This is much stronger than simply seeing the op name in a registry.

## 8. Output is normalized back through a dedicated path

After creating the SDPA unit, the handler:

- uses `ConvertToDefaultLayout(...)` with `__@convert_output_layout`
- then calls `ZinMILUnitBuilder::Finalize(...)`

So the fused SDPA path has:

- explicit input-layout normalization
- explicit output-layout normalization
- specialized unit creation in between

That is a coherent end-to-end fused lowering path.

## 9. What this means for `rustane`

This is one of the most important reverse-engineering findings so far.

### 9.1 Apple has a native semantic target for SDPA

The compiler does not only know how to validate hand-expanded attention graphs.
It also has a dedicated semantic path for the fused op:

- `scaled_dot_product_attention`

### 9.2 Apple’s fused SDPA path is not equivalent to the repo’s current hand-expanded graph

It differs in at least these ways:

- semantic input roles are explicit
- input-layout conversion is compiler-managed
- output-layout conversion is compiler-managed
- the scale constant is compiler-created
- a dedicated unit type is produced

So any future experiment comparing:

- hand-expanded MIL SDPA
vs
- fused `scaled_dot_product_attention`

would be meaningful, not cosmetic.

### 9.3 This does not yet prove it would solve the current width-limit problems

Important caution:

- this pass proves the fused path exists
- it does **not** prove that using it would bypass the compiler/hardware limits
  the repo has been hitting

Those limits may still appear later at:

- layout conversion
- unit validation
- resource validation
- ANEC layer emission

But the existence of a dedicated fused SDPA path means there is now a concrete
Apple-side target if the repo later wants to test that representation.

## 10. Best next targets from here

If we keep pursuing the fused SDPA thread, the best next targets are:

1. `ZinSDPALayer::ValidateSemantics_Impl(...)`
   - to see the exact semantic constraints Apple imposes on the SDPA unit
2. `_ANECValidateSDPALayer`
   - to see the lower-level layer-desc validation boundary
3. `ZinSDPALayer::Lower(...)`
   - to understand how the unit becomes actual ANE-layer form

For `rustane`, the best next target is probably:

- `ZinSDPALayer::ValidateSemantics_Impl(...)`

because that should tell us whether the fused SDPA path has shape/layout
constraints that would matter before anyone considers trying to emit it.
