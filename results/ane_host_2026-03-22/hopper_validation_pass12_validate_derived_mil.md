# Hopper Validation Pass 12: `ValidateDerivedMILProgram`

This pass targets:

- `ValidateDerivedMILProgram(...)`

The goal is to understand how Apple validates the already-converted MIL
representation before later ANEC IR/codegen stages proceed.

## 1. Placement in the pipeline

Direct Hopper finding:

- `ValidateDerivedMILProgram(...)` is called by:
  - `ValidateMILProgram(...)`

So this function sits squarely inside the compiler’s MIL validation path, not in
the normal top-level compile wrapper.

That makes it the most relevant internal target we have looked at so far for
the question:

- what semantic conditions must a converted MIL function satisfy before the
  compiler treats it as a valid derived/internal program?

## 2. High-level structure

The function is medium-sized:

- length: `1616` bytes
- basic blocks: `63`

The decompiler shows this overall flow:

1. retrieve the target MIL function and block by function name
2. construct `MILProgramInfo`
3. build the mutable-weight path->symbol map
4. construct `MILFunctionInfo`
5. initialize tensor info
6. run `ValidateMILConversion(...)`
7. validate live-I/O memory footprint
8. validate units / operations
9. if dynamic-shape validation finds non-ANE-resident operations, mark all ops
   invalid with a descriptive message

So this is a real semantic validation stage, not just a parser sanity check.

## 3. Function and block lookup

One of the first things it does is:

- call `RetrieveFunctionAndBlock(...)`

If that fails, it raises:

- `ANE internal validation error: Cannot retrieve MILFramework function and block from function name \"%s\"`

That tells us the validation stage assumes the function-name identity produced
earlier by the MIL bridge is stable and recoverable.

This connects directly to the earlier findings that:

- procedure-oriented output is formed very early
- function/procedure naming is not incidental metadata

## 4. Mutable weights are validated as part of derived-MIL semantics

Directly after `MILProgramInfo(...)`, Hopper shows:

- `RetrieveMutableWeightToSymbol(...)`

being called inside this validator.

This confirms the mutable-weight path->symbol mapping is not just carried along
for later runtime use. It is part of the **validation** contract for the
derived MIL function.

Relevant strings tied to this area:

- `ANE internal error: ANECompiler cannot handle mutable weights - requires transform infrastructure.`
- `Symbols must be supplied for all mutable weights`

Taken together with the helper pass, this means:

- mutable weights must have symbol bindings
- the compiler has explicit failure paths if the required transform
  infrastructure for mutable weights is not available

For `rustane`, this is strong evidence that dynamic-weight behavior is expected
to be symbolically structured, not just positional.

## 5. `MILFunctionInfo` is the main per-function validation object

After building `MILProgramInfo` and the mutable-weight map, the function
constructs:

- `MILFunctionInfo(...)`

and then immediately does:

- `MILFunctionInfo::InitTensorInfoMap(...)`

This suggests `MILFunctionInfo` is the compiler’s main semantic view of one MIL
function:

- tensor inventory
- weight/source metadata
- per-function properties used by later validation and lowering

So if we continue down the MIL-validation branch later, `MILFunctionInfo` is a
strong target.

## 6. `ValidateMILConversion(...)` is the central semantic gate

One of the most important callees is:

- `ValidateMILConversion(...)`

That is the first strong indication we have of a distinct validation stage for
the *converted* MIL function, as opposed to generic parser/attribute checking.

From the surrounding structure, this validation runs after:

- function/block lookup
- mutable-weight symbol extraction
- tensor-info initialization

So its inputs are already rich and structured.

That means if `rustane` wants to understand why a MIL program is semantically
rejected before codegen, `ValidateMILConversion(...)` is a prime next target.

## 7. Live-I/O memory footprint validation is enforced here

The function calls:

- `ValidateMemoryFootprintLiveIO(...)` three times
- `ValidateLiveIOMemoryFootprint(...)`

and constructs a `ZinIrContext` / `ZinIrConstManager` around that process.

This is one of the clearest signs yet that the derived-MIL validator is not
just checking symbolic correctness. It is also enforcing hardware/resource
constraints around live I/O memory usage.

That is likely directly relevant to some of the practical scaling limits the
repo has been hitting.

## 8. Weight-file size enforcement is explicit

The decompiler also shows a separate debug/info-driven validation path that:

- walks debug-info paths
- sums file sizes
- compares the total against a maximum value from compiler/function state

and on overflow builds an error like:

- `Weight file size (%s bytes) exceeds the maximum (%s bytes)`

This is important because it shows the compiler has explicit post-conversion
resource validation for weight materialization, not just tensor-shape checks.

So even if MIL syntax and function semantics are valid, the derived MIL program
can still fail on materialized-weight size constraints.

## 9. Dynamic-shape fallback behavior is very concrete

After `ValidateUnits(...)`, Hopper shows a specific dynamic-shape fallback path:

- if one or more operations are not ANE-resident under dynamic shapes,
  construct the message:
  - `Dynamic Shapes: One or more network operations are not ANE-resident - Marking all operations as non ANE-resident.`
- then call `MarkAllOpsAsInvalid(...)`

This is one of the most actionable findings so far.

It means the compiler does not always fail hard when dynamic-shape validation
finds a mixed-residency situation. In at least some cases it:

- downgrades the whole function/program by marking all operations invalid for
  ANE residency

That suggests some compile outcomes may look like “the compiler accepted the
program, but nothing useful stayed ANE-resident” rather than a clean fatal
compile error.

For `rustane`, that is a strong hint that some future experiments should check
not just compile success/failure, but whether the derived program still contains
ANE-resident units after validation.

## 10. What this function tells us about the compiler’s validation contract

By this point in the pipeline, Apple’s compiler expects all of the following to
be coherent:

- function-name -> function/block lookup
- mutable-weight path -> symbol mapping
- per-function tensor-info initialization
- converted-MIL semantic validity
- live-I/O memory footprint admissibility
- weight-file size limits
- unit-level ANE residency, especially under dynamic shapes

That is a much richer contract than “parser accepted the MIL file.”

## 11. What this changes in our understanding

### 11.1 Mutable weights are not optional side metadata

This pass upgrades the earlier inference:

- mutable-weight metadata is used directly in validation, not just later
  runtime bookkeeping

So if `rustane` ever wants to align better with Apple’s expectations for dynamic
weights, it should treat symbolic weight identity as a core concern.

### 11.2 Resource validation starts before later codegen

Live-I/O memory footprint and weight-file size are checked here, in the
derived-MIL validator, not only in late lowering or object emission.

That means some “compile failures” may really be validation-stage resource
rejections rather than lower-level ANEC IR failures.

### 11.3 Dynamic-shape behavior can degrade residency instead of failing hard

The explicit `MarkAllOpsAsInvalid(...)` fallback is a major clue. It suggests a
binary “compile succeeded / compile failed” view is incomplete for dynamic-shape
cases.

## 12. Best next targets from here

The best follow-up targets, in order, are:

- `ValidateMILConversion(...)`
  - likely the central semantic validator for the converted function
- `ValidateLiveIOMemoryFootprint(...)`
  - likely the most direct resource-limit checker relevant to scaling
- `ValidateUnits(...)`
  - likely the stage that decides ANE residency / validity of the lowered units
- `MarkAllOpsAsInvalid(...)`
  - if we want to understand the downgrade path for dynamic-shape cases

For `rustane`, the highest-value next target is probably:

- `ValidateMILConversion(...)`

because it sits at the center of the semantic contract this function is
enforcing.
