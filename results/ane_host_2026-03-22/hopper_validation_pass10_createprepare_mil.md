# Hopper Validation Pass 10: `ANECCreatePrepareInfoFromMILFile`

This pass targets the MIL-specific prepare bridge:

- `ANECCreatePrepareInfoFromMILFile(...)`

and its main internal helper:

- `CreateMILAndConvert(...)`

The goal is to answer the repo-relevant question:

- how does `ANECompiler` turn a MIL file into the normalized per-procedure
  representation later consumed by `ANECPrepare` / `_ANECCompile`?

## 1. High-level result

`ANECCreatePrepareInfoFromMILFile(...)` is mostly a gate-and-wrapper.

The real work happens in the internal helper:

- `CreateMILAndConvert(...)`

That helper is the actual MIL-to-internal-prepare bridge.

## 2. What `ANECCreatePrepareInfoFromMILFile(...)` itself does

The top-level function is small:

- length: `460` bytes
- basic blocks: `17`

Direct Hopper decompilation shows this sequence:

1. check `ANECSupportsMIL(target_arch_string)`
2. build a function object around the MIL parser path
3. call `CreateMILAndConvert(...)`
4. if conversion returns at least one `ANECProcedureInfo`, copy that vector into
   the output result
5. initialize `ZinIdentStrings` from the first procedure dictionary
6. otherwise set plist error and zero the output structure

So the exported MIL entrypoint does **not** itself parse MIL or lower ops. It:

- validates that MIL is supported for the current target architecture
- delegates to the deeper conversion helper
- copies the produced procedure metadata into the caller-visible prepare result

## 3. `ANECSupportsMIL(...)`

This support gate is also small:

- length: `304` bytes
- basic blocks: `17`

What it does:

- loads a static vector of supported strings
- compares the target-architecture string against that allowlist
- returns success/failure

So MIL support is explicitly gated by the target-architecture string in
compiler parameters, not assumed universally.

I did not recover the full symbolic list of supported targets in this pass, but
the gating behavior itself is direct.

## 4. `CreateMILAndConvert(...)` is the actual bridge

This helper is much larger:

- length: `3500` bytes
- basic blocks: `145`

This is where the meaningful MIL-specific work lives.

### 4.1 Early setup

At the start, Hopper shows it:

- hardcodes a version string:
  - `1.0.15`
- calls `ANECGetAdditionalWeightFileName(...)`
- deletes any pre-existing additional-weight file at that path
- constructs `MILProgramInfo(...)`
- creates a `MILContext` via `MIL::Opsets::Common::CreateMILContext()`
- registers:
  - `MIL::Opsets::Custom::ane::RegisteraneOpsets(...)`
  - `MIL::Opsets::Prototype::prototype::RegisterprototypeOpsets(...)`
- creates `MIL::ParserOptions`
- sets parser options using the input MIL path

This is direct evidence that `ANECompiler` does not consume plain MIL in a
vacuum. It constructs a MIL context with Apple’s custom ANE opsets and a parser
configuration around the file path first.

### 4.2 The parser/converter function object

The helper takes a callable that returns:

- `unique_ptr<MIL::IRProgram>`
- `shared_ptr<MIL::MILContext>`
- `MIL::ParserOptions`

That means the helper is structured to be generic over “how the MIL program was
obtained,” but in this path it is being used for direct MIL-file parsing.

So the division is:

- `ANECCreatePrepareInfoFromMILFile` chooses the MIL-file parser closure
- `CreateMILAndConvert` handles everything after the program exists

## 5. What happens after the MIL program is obtained

Once the program is available, Hopper shows `CreateMILAndConvert(...)` doing
several important MIL-specific things.

### 5.1 Mutable-weight and model-source extraction

The helper directly calls:

- `RetrieveMutableWeightToSymbol(...)`
- `RetrieveModelSourceInformation(...)`

This is strong confirmation that mutable-weight metadata and source/provenance
information are extracted at the MIL conversion stage, not bolted on later in
the runtime.

That is directly relevant to `rustane`, since the repo already depends on
dynamic weight staging.

### 5.2 Function enumeration

The helper walks the MIL program’s function map and converts functions one by
one into `ANECProcedureInfo` output.

So the normalized prepare structure after MIL conversion is already
procedure-oriented, which lines up with:

- `ANECPrepare`
- `procedureIndex`
- the request-packing model we saw in `AppleNeuralEngine` / `Espresso`

### 5.3 Flexible-shape handling is deep in the MIL path

This pass is especially useful here.

Hopper shows the helper using:

- `MIL::Attributes::FlexibleShapeInfo::Make(...)`
- `GetEnumeratedShapes()`
- `TryGetRangeShapes()`
- `MIL::Transform::ProgramTransformer`
- `MIL::Passes::PropagateInputTensorShapes(...)`

and then recursively calling `CreateMILAndConvert(...)` on transformed programs.

That means flexible-shape support is not just a metadata check in
`ANECPrepare`. The MIL conversion path actively:

- inspects enumerated shapes / range dims
- transforms the MIL program for those shapes
- reruns conversion on the transformed MIL program

This is one of the strongest direct confirmations we have that shape handling is
deeply baked into the MIL-to-compiler bridge.

## 6. MIL-specific errors visible in this path

Strings tied to this area include:

- `MIL Syntax Error: Program has FlexibleShapeInformation but doesn't have RangeDims is invalid`
- `MILFramework Error: failed to get live in shapes from RangeDims`
- `MILFramework Error: Could not infer MIL enumerated shape \"%s\": %s.`
- `MILFramework Error: Could not propagate AOT MIL MIN shapes:`
- `MILFramework Error: Could not propagate AOT MIL MAX shapes:`
- `MILFramework Error: Could not propagate AOT MIL liveins with same MIN and MAX shapes:`
- `ANEC doesn't support MIL program with RangeDims and multi-functions`
- `ANE internal error: ANECompiler cannot handle mutable weights - requires transform infrastructure.`
- `Symbols must be supplied for all mutable weights`

These are high-value because they show the MIL path has its own semantic failure
surface beyond generic codegen:

- flexible-shape consistency
- RangeDims requirements
- AOT shape propagation
- multi-function limitations with RangeDims
- mutable-weight support requirements

For `rustane`, this means that even a structurally valid MIL program can fail
before later ANEC IR/codegen stages if its flexible-shape or mutable-weight
semantics do not fit the compiler’s expectations.

## 7. Recursive conversion behavior matters

One of the most important structural findings is that
`CreateMILAndConvert(...)` recursively calls itself when handling transformed
MIL programs for flexible/enumerated shapes.

That means the prepare bridge is not a one-shot “parse once, lower once”
routine. It can:

- derive shape-specialized MIL variants
- re-enter the same conversion logic
- accumulate per-procedure results from those specialized variants

This matters because it helps explain why shape handling and AOT/JIT shape
support show up throughout the compiler in multiple places: the MIL bridge
itself is already shape-specializing.

## 8. What `ANECCreatePrepareInfoFromMILFile(...)` returns

On success, the MIL path ultimately produces:

- a vector of `ANECProcedureInfo`
- `ZinIdentStrings` initialized from the first produced procedure dictionary

So by the time control returns to `ANECPrepare`, the raw MIL file has already
been turned into:

- a procedure-oriented representation
- with mutable-weight/source info extracted
- with flexible-shape propagation already attempted

That makes `ANECPrepare` more of a final normalizer and validator than the
place where MIL semantics are first interpreted.

## 9. What this changes in our understanding

### 9.1 The real MIL bridge is `CreateMILAndConvert(...)`, not the wrapper

This is the key result from the pass.

If we want to understand how `rustane`’s emitted MIL is actually interpreted by
Apple’s compiler stack, the most relevant internal target is now:

- `CreateMILAndConvert(...)`

not the thinner wrapper `ANECCreatePrepareInfoFromMILFile(...)`.

### 9.2 Flexible-shape logic starts very early

We already knew dynamic-shape validation happens in `ANECPrepare`.

Now we know that shape-sensitive program transformation and recursive
conversion already happen in the MIL bridge itself.

So shape handling is layered:

- MIL bridge: shape-specialized MIL transformation and conversion
- `ANECPrepare`: dynamic-shape admissibility and procedure-property annotation
- later compile stages: lowering/codegen-time layout constraints

### 9.3 Mutable weights are a first-class MIL-bridge concern

Because this helper directly extracts mutable-weight symbol information and has
its own mutable-weight-related error strings, mutable-weight semantics are
clearly not incidental to Apple’s MIL compiler path.

That makes this area directly relevant to any future `rustane` experiments
around compile-cache reuse or Apple-like mutable-weight behavior.

## 10. Best next targets from here

If we continue with the MIL branch specifically, the best next targets are:

- `RetrieveMutableWeightToSymbol(...)`
  - to understand Apple’s expected symbol mapping for mutable weights
- `RetrieveModelSourceInformation(...)`
  - to understand what provenance/source fields are captured this early
- `MIL::Passes::PropagateInputTensorShapes(...)`
  - to understand why certain flexible/enumerated-shape programs fail

If we stay focused on `rustane`, the highest-value next one is probably:

- `RetrieveMutableWeightToSymbol(...)`

because it is the strongest remaining bridge between Apple’s MIL semantics and
the repo’s current dynamic-weight model.
