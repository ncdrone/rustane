# Hopper Validation Pass 9: `ANECPrepare`

This pass focuses specifically on:

- `ANECPrepare(ZinIrCompilerParameters&, ZinIrPlistCompilationStatus&)`

with the goal of understanding what it contributes between:

- compiler input / option parsing
- and the later lowering / codegen pipeline

## 1. Placement in the pipeline

Direct Hopper findings:

- `ANECPrepare` is called by:
  - `_ANECCompile`
  - `_ANECValidate`

That makes it a shared normalization/prepare stage for both compile and
validate flows, not a compile-only helper.

This is important because it means `ANECPrepare` is one of the last common
front-door stages before the compile path and the validation path diverge.

## 2. What `ANECPrepare` actually does

`ANECPrepare` is not huge by `ANECompiler` standards:

- length: `876` bytes
- basic blocks: `42`

But it is structurally important.

The high-level shape of the function is:

1. set the status/plist name
2. inspect the compiler input file format with `_ANECGetCompilerFileFormat(...)`
3. build a canonical “prepare info” representation from that format
4. walk each prepared network/procedure entry
5. validate dynamic input shapes when present
6. mark procedure properties and visibility attributes
7. return the normalized procedure/property vector to the caller

## 3. Format dispatch inside `ANECPrepare`

This is the strongest direct result from the pass.

`ANECPrepare` branches on `_ANECGetCompilerFileFormat(...)` and dispatches to:

- format `1`
  - `ANECCreatePrepareInfoFromANECIR(...)`
- format `2`
  - `ANECCreatePrepareInfoFromMILFile(...)`
- format `3`
  - `ANECCreatePrepareInfoFromMLIR(...)`
- format `4`
  - explicit error path inside `ANECPrepare`
- other / default case
  - expects pre-existing network dictionaries and walks them directly

What this tells us:

- `ANECPrepare` is the stage where different input representations are
  collapsed into one common internal representation
- the later compile path in `_ANECCompile` is operating on the output of this
  canonicalization, not directly on raw MIL/MLIR/ANECIR inputs

This is one of the clearest architectural boundaries we’ve recovered so far.

## 4. Dynamic-shape handling is explicit in `ANECPrepare`

For each prepared network entry, Hopper shows `ANECPrepare` doing:

- lookup of `Networks`
- pull the first function/network name
- resolve the corresponding function dictionary
- call `ZinHasDynamicInputShapes(...)`
- if dynamic shapes exist:
  - construct a `ZinIrNetworkStatus`
  - call `ZinValidateDynamicInputShapes(...)`
  - on success, mark the procedure properties entry at `+0xa0 = 1`
  - on failure, set plist error and zero the output result structure

If no dynamic shapes are present:

- it still calls `SetFunctionVisibilityAttribute(...)`

So `ANECPrepare` is doing two very concrete things:

- validating whether dynamic-input-shape metadata is acceptable
- attaching per-function/procedure properties before the main compile path

That lines up with the earlier strings in `ANECompiler` around:

- `DynamicShapesInitSymbolStart`
- `Unranked input types or dynamic shapes are not supported on ANEs`
- dynamic-shape validation errors

## 5. Function visibility is normalized here

One of the most useful direct callees is:

- `SetFunctionVisibilityAttribute(...)`

That means `ANECPrepare` is not only format-parsing and dynamic-shape checking.
It also mutates or annotates the per-function dictionaries with visibility
information before the rest of the pipeline runs.

This matters because it suggests later compile/codegen stages can assume a
normalized function dictionary shape:

- dynamic-shape status settled
- visibility attributes attached
- procedure-property bits ready

## 6. What `ANECPrepare` returns to callers

The decompiler shows it assembling a result structure that includes:

- a vector of network/function dictionaries
- a vector of `ANECProcedureInfo`
- per-procedure properties
- string/path fields copied out of the prepare-info builders

In other words, `ANECPrepare` is not just a boolean validator. It is the stage
that produces the canonical per-procedure metadata bundle used by the rest of
`_ANECCompile`.

That explains why `_ANECCompile` can immediately iterate procedures after
`ANECPrepare` and start enabling dynamic shapes, creating file backing, and
calling into the classic compiler core.

## 7. What `ANECPrepare` does *not* appear to do

This pass is also useful for narrowing scope.

`ANECPrepare` does **not** look like the place where the width/splitting
constraints themselves are enforced.

It appears to be responsible for:

- input-format canonicalization
- dynamic-shape admissibility checks
- per-function/procedure annotation

It does **not** appear to be where:

- transpose-width divisibility checks
- spatial split MIR lowering
- procedure compilation
- Mach-O generation

actually happen.

Those are still later-stage responsibilities.

## 8. What this changes in our understanding

### 8.1 `ANECPrepare` is the format-normalization choke point

This is the cleanest direct role assignment we have now:

- `ANECGetCompilerInputs` = gather raw file/path/config inputs
- `ANECGetCompilerOptions` = parse option/config dictionary
- `ANECPrepare` = normalize input representation + dynamic-shape/function metadata
- later `_ANECCompile` logic = actual lowering / codegen / object emission

So if the question is “where does the compiler stop thinking in terms of raw
MIL/MLIR/ANECIR files and start thinking in terms of a normalized per-procedure
compile plan?”, the answer is:

- `ANECPrepare`

### 8.2 `ANECPrepare` is a better next target than more top-level `_ANECCompile`

Because `_ANECCompile` is very large and mixes many later responsibilities,
`ANECPrepare` gives us a more tractable point to understand:

- input-format differences
- dynamic-shape admissibility
- function-level metadata shaping

without yet diving into the full codegen pipeline.

## 9. Best next targets from here

The highest-value next internal targets are now:

- `ANECCreatePrepareInfoFromMILFile(...)`
  - to see exactly how MIL input becomes the canonical prepare structure
- `ANECCreatePrepareInfoFromMLIR(...)`
  - to compare the MLIR path against MIL
- `SetFunctionVisibilityAttribute(...)`
  - to understand what visibility/function annotations the later pipeline expects
- `_ANECValidateNetworkCreate(...)`
  - to bridge from prepare-time normalization into validation-time network creation

If the goal stays tightly aligned with `rustane`, the best next one is probably:

- `ANECCreatePrepareInfoFromMILFile(...)`

because that is the closest direct bridge from the repo’s generated MIL into the
compiler’s normalized internal representation.
