# Hopper Validation Pass 7: `ANECompiler`

This pass focuses on `ANECompiler` as the low-level compile boundary beneath
`ANECompilerService` and below the `Espresso` runtime bridge.

Primary questions:

- what the exported compile entry points look like
- where compiler options are parsed
- where MIL validation sits in the pipeline
- what `_ANECCreateModelDictionary` actually packages
- whether there are direct signs of width/splitting/shape-limit logic relevant
  to `rustane`

## 1. Exported compile entry points

Validated exports:

- `_ANECCompile`
- `_ANECCompileJIT`
- `_ANECCompileOnline`
- `_ANECCreateModelDictionary`

### 1.1 `_ANECCompile`

`_ANECCompile` is real and large:

- length: `5832` bytes
- basic blocks: `277`

This looks like the normal host compiler front door, not a trivial shim.

### 1.2 `_ANECCompileJIT`

`_ANECCompileJIT` is also real and large:

- length: `4676` bytes
- basic blocks: `202`

The decompiler clearly shows a separate JIT pipeline rather than a thin wrapper
around `_ANECCompile`.

### 1.3 `_ANECCompileOnline`

`_ANECCompileOnline` is **not** a real implementation on this build. Hopper
shows it is only:

```asm
_ANECCompileOnline:
    mov w0, #0x1
    ret
```

So on this macOS build, the “online” entry point is effectively a stub that
returns success. The real work is happening in `_ANECCompile` and
`_ANECCompileJIT`.

## 2. Option parsing lives in `ANECGetCompilerOptions`

Validated function:

- `ANECGetCompilerOptions(...)`

This function is very large:

- length: `22192` bytes
- basic blocks: `1201`

Most important xref result:

- `ANECGetCompilerOptions` is called from both:
  - `_ANECCompile`
  - `_ANECCompileJIT`

So normal compile and JIT compile share the same broad compiler-option parsing
surface.

### 2.1 Direct option-related evidence

Strings present in `ANECompiler` and associated with this option layer include:

- `SpatialSplitSubgraphs`
- `EnableCircularBufferInSpatialSplit`
- `DynamicShapesInitSymbolStart`

This is direct evidence that the compiler front door has explicit support for:

- spatial splitting
- circular-buffer behavior inside spatial splitting
- dynamic-shape-related initialization

That matters to `rustane` because the branch has been running directly into
compiler width/shape ceilings. These strings show the compiler really does have
internal machinery for splitting and shape-special handling rather than simply
being a monolithic black box.

## 3. MIL validation sits behind `_ANECValidateNetworkCreate`

Validated functions:

- `ValidateMILProgram(...)`
- `_ANECValidateNetworkCreate(...)`

`ValidateMILProgram` itself is substantial:

- length: `4336` bytes
- basic blocks: `158`

Most important xref result:

- `ValidateMILProgram` is called from `_ANECValidateNetworkCreate`

That places MIL validation in a distinct helper stage before or during compiler
network creation, rather than as an incidental late-stage check.

### 3.1 Validation-related strings that matter to `rustane`

Representative strings in `ANECompiler`:

- `Unranked input types or dynamic shapes are not supported on ANEs`
- `Error: Multi layer procedure families are not supported`
- `Error: Could not validate the ANEC IR procedure.`
- `Error: Network is already a Dynamic Shapes model.`
- `Jit operation count needs to be 0 or match operation count`
- `Error: Could not retrieve JIT shapes dictionary version.`

This tells us the low-level compiler is explicitly policing:

- dynamic-shape usage
- procedure-family structure
- JIT-shape bookkeeping
- ANEC IR procedure validity

That is exactly the kind of logic that could explain why some `rustane` graphs
compile and others fail even when they are “close” structurally.

## 4. `_ANECCreateModelDictionary` is a compute-program introspection/export path

Validated function:

- `_ANECCreateModelDictionary`

This function is mid-sized:

- length: `1860` bytes
- basic blocks: `81`

The decompiler shows this is not “compile the model.” It is an export/introspection
helper that:

- calls `ZinComputeProgramMake(...)`
- builds a mutable CFDictionary
- creates `NetworkStatusList`
- walks per-network bindings
- packages live I/O names and attributes into CF containers

### 4.1 Concrete dictionary fields observed

Hopper shows it populating:

- `NetworkStatusList`
- `LiveInputList`
- `LiveInputParamList`
- `Name`

and building per-binding entries via helper routines such as:

- `CreateTiledIOAttributeDict`
- `CreateMultiPlaneLinearLiveIOAttributeDict`
- `CreateLiveInputParamAttributeDict`
- `CreateSinglePlaneCircularLiveIOAttributeDict`

It also branches across several binding/storage layouts, including:

- single-plane tiled compressed
- multi-plane tiled compressed
- multi-plane linear
- single-plane uncompressed
- single-plane circular

So `_ANECCreateModelDictionary` looks like the API Apple exposes for exporting
or inspecting compute-program I/O structure after compilation, not the routine
that is actually generating ANE code.

## 5. JIT compile path details

The decompiler for `_ANECCompileJIT` shows a concrete JIT-specific front door:

- calls `ANECGetJITCompilerInputs(...)`
- calls `ANECGetCompilerOptions(...)`
- logs `Start of JIT compilation of network from file: %s`
- checks file formats for both the base network and the JIT-shapes input
- creates a `ZinComputeProgram`
- later emits output through `ZinObjectGenHandleProgramToFile(...)`
- logs `End of JIT compilation of network from file: %s`

Relevant JIT strings:

- `NetworkJITShapesPath`
- `Start of JIT compilation of network from file: %s`
- `Error: Could not retrieve JIT shapes dictionary version.`
- `Jit operation count needs to be 0 or match operation count`
- `Error: JIT input \"%s\" has JIT rank %ld not equal to AOT rank %ld.`

This makes the JIT split very concrete:

- `_ANECCompileJIT` is not just “compile with extra options”
- it expects a second structured shapes input and checks it carefully against
  the base/AOT network

## 6. Evidence of shape, width, and splitting constraints

`ANECompiler` contains many strings that look directly relevant to the kinds of
issues `rustane` has been seeing:

- `Invalid input tensor width %ld, must be divisible by 2, 3, 4, 8.`
- `Pool input tensor width (%zd) must be a multiple of stride (%d)`
- `It must be splitted into (1) multiple of 8 width tensor, and (2) remainder tensor`
- `Spatial Splitting Intenral Error`
- `SpatialSplitSubgraphs`
- `EnableCircularBufferInSpatialSplit`
- `Error: large kernel stride decomposition fails`
- `Error: large kernel stride decomposition fails`
- `Error: could not add shared init block to RT graph`

Even without full disassembly of every call path, that is strong evidence that
the compiler has explicit:

- width-divisibility constraints
- width-splitting/decomposition logic
- spatial-splitting machinery
- architecture/family-specific lowering rules

This matters because it suggests the branch’s current “width ceiling” problems
are not just accidental artifacts of MIL generation. The compiler itself has a
rich body of width/splitting logic and explicit failure modes in this area.

## 7. What this changes in our understanding

### 7.1 `ANECompiler` is more useful than `MIL` for current repo blockers

`MIL Framework` tells us how MIL is parsed and serialized.

`ANECompiler` tells us:

- where compile options are parsed
- where MIL validation happens before ANEC IR generation
- that width/splitting logic is a first-class internal concern
- that JIT and normal compile are genuinely different entry paths

For current `rustane` work, that makes `ANECompiler` more relevant than MIL to
the practical question of “why does this graph fail to compile?”

### 7.2 The “online compile” path is probably not worth chasing

Since `_ANECCompileOnline` is just `mov w0, #1; ret` on this build, it does not
look like a promising path for further local investigation.

### 7.3 The best remaining ANECompiler targets

If we do another pass on `ANECompiler`, the best next targets are:

- `_ANECCompile` internals around:
  - `ANECGetCompilerOptions`
  - `_ANECValidateNetworkCreate`
  - the helper that calls `ValidateMILProgram`
- exact option-key handling inside `ANECGetCompilerOptions`
- the codepaths behind the width/splitting-related strings

That is the most likely place to directly connect Hopper findings to the
compile-shape behavior `rustane` is hitting.
