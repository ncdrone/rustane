# Hopper Validation Pass 13: `ValidateMILConversion`

This pass targets:

- `ValidateMILConversion(...)`

The goal is to pin down whether this is the deep semantic validator itself or a
smaller dispatcher into more specialized validation stages.

## 1. High-level result

`ValidateMILConversion(...)` is **thin**.

It is not the giant all-in-one semantic validator. Instead, it is the
orchestration point for three concrete sub-stages:

1. `ConvertLiveInputs(...)`
2. `ValidateOpList(...)`
3. `ConvertLiveOutputs(...)`

So the central role of `ValidateMILConversion(...)` is to sequence those stages
and enforce that failure in op-list conversion stops later live-output
processing.

## 2. Direct decompilation

The function is tiny by comparison with the earlier targets:

- length: `188` bytes
- basic blocks: `5`

Hopper decompilation is straightforward:

- call `ConvertLiveInputs(function, MILFunctionInfo)`
- call `ValidateOpList(block->operations, MILFunctionInfo, validate_entry_map)`
- if that succeeds:
  - call `ConvertLiveOutputs(function, block, MILFunctionInfo, validate_entry_map)`
- otherwise:
  - raise
    - `ANE internal validation error: Cannot convert the MIL operation list.`

That means `ValidateMILConversion(...)` is basically the transaction boundary
for “live inputs -> operation list -> live outputs”.

## 3. Callers and callees

Direct caller:

- `ValidateDerivedMILProgram(...)`

Direct callees:

- `ConvertLiveInputs(...)`
- `ValidateOpList(...)`
- `ConvertLiveOutputs(...)`
- `ZinAssertImpl(...)`

So the earlier guess was slightly too broad:

- `ValidateDerivedMILProgram(...)` is the richer semantic/resource validator
- `ValidateMILConversion(...)` is the subroutine that validates the *converted
  function body itself* via those three sub-stages

## 4. What each sub-stage appears to own

### 4.1 `ConvertLiveInputs(...)`

This function is large:

- length: `2476` bytes
- basic blocks: `132`

From Hopper, it is responsible for:

- iterating MIL function live inputs
- converting them into `ZinIrIOInfo` / input-param structures
- recording tensor info into `MILFunctionInfo`
- handling state-type wrapping
- updating dynamic-shape pass type information

Its visible failure strings include:

- `Unsupported input param data type %s`
- `Could not convert all Live inputs.`
- `Unable to determine MIL dynamic shape pass type`

So live-input conversion is itself a substantial semantic validation stage, not
just a bookkeeping helper.

### 4.2 `ValidateOpList(...)`

This function is very large:

- length: `7164` bytes
- basic blocks: `301`

That makes it the heaviest callee by far and the most likely place where:

- operation-by-operation semantic constraints
- ANE-lowerability checks
- unit/format compatibility checks

actually live.

This is now the most likely “real center” of converted-MIL semantic validation.

### 4.3 `ConvertLiveOutputs(...)`

This function is also substantial:

- length: `3704` bytes
- basic blocks: `132`

So the output side is not trivial either. The converted MIL validator treats
live outputs as another first-class conversion/validation phase after op-list
validation passes.

## 5. Failure surface of `ValidateMILConversion(...)`

The direct failure string tied to the wrapper itself is:

- `ANE internal validation error: Cannot convert the MIL operation list.`

That means:

- failures inside `ValidateOpList(...)`
- or any condition that causes it to return false

are surfaced at this wrapper boundary as “cannot convert the MIL operation
list”.

This is useful because it tells us where one category of generic validation
error text originates.

## 6. Relationship to the dynamic-shape downgrade path

The dynamic-shape downgrade string we saw earlier:

- `Dynamic Shapes: One or more network operations are not ANE-resident - Marking all operations as non ANE-resident.`

does **not** come from `ValidateMILConversion(...)` directly.

That behavior lives one level up, in `ValidateDerivedMILProgram(...)`, after
`ValidateMILConversion(...)` returns.

So the layers are:

- `ValidateMILConversion(...)`
  - validate converted function body
- `ValidateDerivedMILProgram(...)`
  - apply higher-level policy, including possible downgrade of all ops to
    non-ANE-resident

That separation is important for interpreting future failures.

## 7. What this changes in our understanding

### 7.1 `ValidateMILConversion(...)` is a dispatcher, not the deepest validator

The name sounds like it might contain all the converted-MIL logic.

It does not.

It mainly sequences:

- live input conversion
- op list validation
- live output conversion

### 7.2 `ValidateOpList(...)` is now the most important remaining target

Because:

- it is by far the largest callee
- it sits exactly in the middle of the converted-MIL validation wrapper
- wrapper-level failure text points at inability to convert the op list

it is now the best place to keep digging if the goal is:

- “what semantic properties of a MIL op/function make the compiler reject it?”

### 7.3 Live input/output conversion are more important than they looked

Both `ConvertLiveInputs(...)` and `ConvertLiveOutputs(...)` are large and clearly
perform significant semantic work. They are not simple adapters.

That suggests some repo-level failures could come from:

- input/output ABI conversion
- tensor-info recording
- dynamic-shape pass typing

and not only from the op list itself.

## 8. Best next targets from here

The next targets are now much clearer:

1. `ValidateOpList(...)`
   - likely the main semantic validator for converted operations
2. `ConvertLiveInputs(...)`
   - especially if we want to understand input ABI and dynamic-shape pass typing
3. `ConvertLiveOutputs(...)`
   - for output ABI / live-output constraints

If the goal is to maximize signal for `rustane`, the best next target is:

- `ValidateOpList(...)`

because it is the deepest remaining semantic-validation stage on the direct
converted-MIL path.
