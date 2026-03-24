# Hopper Validation Pass 8: `ANECompiler` Narrow Follow-Up

This pass narrows in on three specific questions:

- what `_ANECCompile` does immediately after `ANECGetCompilerOptions`
- how `ANECGetCompilerOptions` handles concrete spatial-split-related keys
- which functions own some of the width/splitting-related error strings that
  look relevant to `rustane`

## 1. `_ANECCompile` immediately hands off to `ANECPrepare`

Direct Hopper decompilation of `_ANECCompile` shows the early sequence clearly:

1. `ANECGetCompilerInputs(...)`
2. `ANECGetCompilerOptions(...)`
3. `ANECPrepare(...)`

That is the immediate helper after option parsing.

The relevant beginning of `_ANECCompile` decompiles as:

- initialize `ZinIrPlistCompilationStatus`
- initialize `ZinIrCompilerParameters`
- `ANECGetCompilerInputs(arg0, &var_490, &status)`
- if no errors:
  - `ANECGetCompilerOptions(arg1, &var_490, &status)`
- if still no errors:
  - `ANECPrepare(&var_490, &status)`

After `ANECPrepare`, `_ANECCompile` moves into the real compile/build path:

- logs `Start of compilation of network from file: %s`
- checks whether any procedure carries a dynamic-shape flag and, if so, enables
  dynamic shapes in compiler parameters
- computes dump/status/file-backing paths
- creates file backing via `ANECCreateFileBacking(...)`
- creates the target via `ZinIrTargetCreator::CreateTargetFromString(...)`
- iterates procedures and compiles them through `ZinCompilerCoreClassic::CompileProcedure(...)`
- accumulates compilation status into the plist status object
- builds the final Mach-O via `ZinIrConstManager::BuildMachO(...)`
- emits kernels/metadata to disk via `ANECHandleProgramKernelsAndMetadataToFile(...)`

### What this changes

The important clarification is:

- `ANECGetCompilerOptions` is not the last “front-door” step before codegen
- `ANECPrepare` is the immediate handoff after options parsing

So if we want to understand where compiler options actually start affecting the
network, `ANECPrepare` is now the best next target inside `ANECompiler`.

## 2. Exact `SpatialSplitMode` handling inside `ANECGetCompilerOptions`

We isolated the smaller lambda:

- `__ZZ22ANECGetCompilerOptions...$_3`

This lambda directly does:

- `CFDictionaryGetValue(..., @"SpatialSplitMode")`
- validates it is a CFString
- accepts these string values:
  - `Memory`
  - `Auto`
  - `Test`
  - `GenericDAG`
  - `GenericDAGExperimental`
  - `GenericDAGMemory`
  - `Disabled`
- converts the string through `CFStringRefToSpatialSplitMode(...)`
- passes the result into the supplied callback

So we now have a concrete, directly decompiled answer for one real option key:

- **key:** `SpatialSplitMode`
- **accepted values:** `Memory`, `Auto`, `Test`, `GenericDAG`,
  `GenericDAGExperimental`, `GenericDAGMemory`, `Disabled`

This is no longer inference from strings; Hopper shows the actual parser logic.

## 3. `ANECGetCompilerOptions` has a small CFArray-to-vector helper

We also isolated the companion lambda:

- `__ZZ22ANECGetCompilerOptions...$_2`

Its behavior is simple and useful:

- takes a `CFArray`
- converts each element to a C++ string via `ZinGetString(...)`
- pushes each string into one of two vector slots inside a `SubGraphSpec`
- chooses the destination vector offset based on a boolean:
  - offset `0x18` if `false`
  - offset `0x30` if `true`

That strongly suggests this helper is used to parse the two string-array fields
inside a spatial-split subgraph description, i.e. something very much like:

- `InputNodes`
- `OutputNodes`

The key point is that `ANECGetCompilerOptions` is not just reading scalar knobs.
It also parses structured subgraph specs.

## 4. Exact manual spatial-split structure from `_ANECCreateCompilerOptionDictionary`

To complement the parser side, we checked the exporter:

- `_ANECCreateCompilerOptionDictionary`

That function serializes manual spatial split configuration into a CFDictionary.
The relevant decompilation shows:

- it emits `SpatialSplitSubgraphs` as an array
- each subgraph entry is a CFDictionary with:
  - `HTileCount`
  - `InputNodes`
  - `OutputNodes`
- it uses `getSpatialSplitMode()` and treats `manual` specially

So the effective structure around manual spatial split is now clear:

- `SpatialSplitMode = Manual`
- `SpatialSplitSubgraphs = [ { HTileCount, InputNodes, OutputNodes }, ... ]`

That gives us a much more concrete picture of how the compiler expects manual
spatial splitting to be represented internally.

## 5. Concrete width/splitting codepaths

We resolved two strings to real procedures.

### 5.1 `Invalid input tensor width %ld, must be divisible by 2, 3, 4, 8.`

Containing function:

- `TransposeLayerUtils::ValidateTransposeMappings(...)`

What Hopper shows in that function:

- it validates transpose mappings against HAL parameters
- checks transpose-specific stride/format constraints
- checks channel alignment against tensor format size
- then reaches the width-divisibility assertion

This means the width-divisibility string is not some vague global compiler
error. It is specifically tied to transpose-mapping validation for a certain
class of lowered tensor layouts.

That is useful for `rustane` because it suggests some width failures may be
arising after lowering into transpose/view-based internal forms, not only from
the top-level MIL graph shape.

### 5.2 `It must be splitted into (1) multiple of 8 width tensor, and (2) remainder tensor`

Containing function:

- `ZinMirSpatialArgMinMax::InsertGenericPooling(...)`

What Hopper shows:

- it calls `ZinMirTensorTransform::Split(...)`
- expects a two-way split result
- if the split shape/layout does not match expectations, it raises exactly that
  assertion

So this string is not a generic compiler slogan. It is tied to a concrete MIR
lowering path for spatial ArgMin/ArgMax via generic pooling.

That gives us a second strong example that the compiler’s width logic is
implemented in many specific lowering passes, not just in a single global
validator.

## 6. Exact option keys now visible from the exporter

While the narrow request was mainly about the parser, `_ANECCreateCompilerOptionDictionary`
is helpful because it reveals many of the internal option names in one place.

Directly emitted keys include:

- `TargetArchitecture`
- `DisableContextSwitching`
- `DebugContextSwitchingDma`
- `SetIsSecureNetwork`
- `DisableMergeConstants`
- `FoldScale`
- `ForceHazardStallsBegin`
- `ForceHazardStallsEnd`
- `MaxTdCount`
- `MaxSegmentSize`
- `ProduceRelocatableObjects`
- `ProcedureName`
- `DramAllocatorType`
- `DramTensorPriorityType`
- `L2AllocatorType`
- `L3AllocatorType`
- `SpatialSplitSubgraphs`

For the allocator/priority knobs, Hopper shows concrete exported values such as:

- `FirstFitReuse`
- `BestFitReuse`
- `NoReuse`
- `costofreads`
- `orderofcreation`
- `sizebyliverange`
- `sizethenliverange`

So even where the parser side is still too large to fully decompile cleanly in
one pass, the exporter gives us an authoritative option vocabulary.

## 7. Narrow conclusions

### 7.1 Immediate helper after options parsing

The immediate helper after `ANECGetCompilerOptions` in `_ANECCompile` is:

- `ANECPrepare`

That is the best next place to dig if the goal is to connect option parsing to
actual compile-time network transformation.

### 7.2 Exact spatial-split mode values

We now directly know the accepted `SpatialSplitMode` values:

- `Memory`
- `Auto`
- `Test`
- `GenericDAG`
- `GenericDAGExperimental`
- `GenericDAGMemory`
- `Disabled`

### 7.3 Exact manual subgraph shape

We now directly know the internal/exported structure of manual spatial split:

- `SpatialSplitSubgraphs`
  - each entry has `HTileCount`, `InputNodes`, and `OutputNodes`

### 7.4 Width failures are localized to real lowering passes

We now have two concrete examples:

- transpose-mapping validation owns one width-divisibility constraint
- spatial ArgMin/ArgMax MIR lowering owns a specific “split into multiple of 8 +
  remainder” requirement

That makes the compiler behavior look even less like one monolithic width limit
and more like a collection of pass-specific layout constraints.

## 8. Best next targets from here

If we keep digging in `ANECompiler`, the highest-value next steps are:

- `ANECPrepare(...)`
  - to see how parsed options become prepared compiler/network state
- the helper that calls `_ANECValidateNetworkCreate(...)`
  - to connect MIL validation to the rest of normal compile
- more xref-driven passes on width/splitting strings
  - to map which failure messages belong to which lowering families
