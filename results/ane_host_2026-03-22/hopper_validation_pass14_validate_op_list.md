# Hopper Validation Pass 14: `ValidateOpList`

This pass targets:

- `ValidateOpList(...)`

This is the function that now looks most likely to be the real semantic center
of converted-MIL validation inside `ANECompiler`.

## 1. High-level result

`ValidateOpList(...)` is indeed the deepest remaining semantic validator on the
direct converted-MIL path that we have inspected so far.

It is large:

- length: `7164` bytes
- basic blocks: `301`

and it does much more than a simple “iterate operations and check types” pass.

The main pattern Hopper shows is:

1. load conversion/validation maps
2. iterate MIL operations
3. retrieve a stable operation identifier
4. skip or specially handle certain ops/attributes
5. dispatch per-op validation/materialization rules
6. revalidate producers/indices for specific ops
7. fold results back into `MILFunctionInfo` and the validation-status map
8. finally remove constant-folded operations

So this is clearly not a wrapper. It is an actual op-level semantic and
lowerability pass.

## 2. First-stage structure

The function begins by:

- calling `GetMILConversionMaps()`
- taking the dereferenced vector of `MIL::IROperation` objects
- iterating operation-by-operation
- calling `RetrieveOpIdentifier(...)` on each op

If the operation identifier cannot be recovered, it ultimately surfaces:

- `MILFramework error: Could not retrieve operation identifier for operation %s.`

That means op identifiers are foundational to the rest of the pass. A lot of
the later dispatch logic is keyed by a stable operation name.

## 3. `constantFolded` is explicitly recognized

One of the first strings tied directly to this pass is:

- `constantFolded`

Hopper shows the function checking that attribute very early during operation
iteration. Combined with the direct callee:

- `RemoveConstantFoldedOps(...)`

the intent is clear:

- some operations are tracked as constant-folded during validation
- those are removed from the live operation set at the end

This is important because it means the validator is already mutating the
effective derived op list before later phases run.

## 4. Operation-specific semantic dispatch is map-driven

`ValidateOpList(...)` calls:

- `GetMILConversionMaps()`

and the body is full of lookups against string-keyed maps and trees. The
decompilation strongly suggests:

- the validator maintains named conversion/validation handlers keyed by op name

That matches the observed structure of:

- retrieve op identifier
- use that identifier to look up how the op should be validated or transformed

So the core semantics here are likely table-driven rather than one massive
switch over op enums.

## 5. Special-case validation families we can now see directly

Even without exhaustively reversing every branch, Hopper shows several concrete
semantic families inside this pass.

### 5.1 Terminal read-state / ring-buffer handling

Direct callees:

- `MILOpConverter::ReadStateTerminalRBW(...)`
- `RevalidateReadStateProducerOpsFromWriteStateOps(...)`

Direct tied string:

- `Error in materializing terminal RingBufferWriter: `

This shows that stateful read/write paths are not handled by the generic op
validation alone. They have dedicated revalidation logic for producer chains.

### 5.2 Revalidation of index-producing ops

Direct helper family:

- `RevalidateOpIndicesProducer(...)`

From the decompilation, it is explicitly applied to these op families:

- `gather`
- `gather_nd`
- `gather_along_axis`
- `crop_resize`
- `resample`

This is a strong result. It means there is a dedicated second-pass semantic
check around the producer of index/coordinate-like operands for these ops.

That makes the validator much richer than a one-op-at-a-time local check.

### 5.3 Dynamic-shape resize path

Direct string:

- `Missing dynamic shape resize shape input`

This string appears directly in `ValidateOpList(...)`, not just in some later
dynamic-shape policy layer.

That means at least part of dynamic-shape support is enforced at op-list
validation time, and missing shape-driving operands for resize-like behavior are
rejected here.

### 5.4 Precision-loss checks

Direct string:

- `Losing precision on ANE, cannot be consumed by `

This appears in a path where the function:

- compares producer/consumer type information
- builds a message involving the consuming op
- records validation status accordingly

So `ValidateOpList(...)` is explicitly policing lossy type transitions that
would not be acceptable for ANE consumption.

That is directly relevant to any repo experiments that change tensor formats or
rely on implicit casts.

## 6. Live operation status bookkeeping is built here

The function directly calls:

- `SetValidationStatus(...)`

many times while walking operations.

This means `ValidateOpList(...)` is not only deciding pass/fail. It is building
the per-operation validation-status record that later stages can use.

That fits well with the earlier `ValidateDerivedMILProgram(...)` finding that
some dynamic-shape cases do not fail hard, but instead mark operations or whole
functions invalid for ANE residency.

## 7. Relationship with `MILFunctionInfo`

Throughout the function, Hopper shows repeated use of:

- `MILFunctionInfo::ContainOp(...)`
- `MILFunctionInfo::AtOp(...)`
- `MILFunctionInfo::InsertOp(...)`
- `MILFunctionInfo::GetTensorName(...)`
- `MILFunctionInfo::IsRootFunction(...)`

So `ValidateOpList(...)` is operating against the persistent semantic view of
the function built earlier in `ValidateDerivedMILProgram(...)`, not a fresh
temporary snapshot.

This makes `MILFunctionInfo` look like the backbone data structure for:

- op identity
- tensor naming
- conversion state
- validation results

## 8. What `ValidateOpList(...)` does not appear to be

It is very important, but it is not the only validator.

It does **not** appear to own:

- live input ABI conversion
- live output ABI conversion
- top-level mutable-weight extraction
- top-level dynamic-shape downgrade policy

Those still live in:

- `ConvertLiveInputs(...)`
- `ConvertLiveOutputs(...)`
- `RetrieveMutableWeightToSymbol(...)`
- `ValidateDerivedMILProgram(...)`

So `ValidateOpList(...)` is best understood as:

- the central per-operation semantic validator and revalidator

not the whole derived-MIL contract on its own.

## 9. What this changes in our understanding

### 9.1 The most likely source of semantic MIL rejection is now clearly op-list validation

We already knew `ValidateMILConversion(...)` raised:

- `ANE internal validation error: Cannot convert the MIL operation list.`

This pass clarifies what that probably means in practice:

- some op identifier could not be resolved
- a special revalidation rule failed
- a dynamic-shape resize input was missing
- a precision-loss check failed
- a stateful read/write producer chain failed to materialize correctly

That is much more concrete than “some conversion failed.”

### 9.2 Several important ANE constraints are per-op-family, not global

The presence of dedicated revalidation for:

- gather-family ops
- crop/resize/resample
- read_state / write_state / ring-buffer flows

shows that Apple’s MIL validator has many operation-family-specific semantic
rules.

That matters for `rustane` because it suggests compile trouble is likely to be
clustered around certain op families rather than explained by a single global
rule.

### 9.3 Constant-folding changes the effective validation surface

Because `constantFolded` is explicitly recognized and
`RemoveConstantFoldedOps(...)` runs at the end, the semantic op list seen by
later stages is already pruned/rewritten here.

That means any future attempts to compare “input MIL ops” to “what the compiler
really sees” need to account for this pass.

## 10. Best next targets from here

The best next deep targets are now:

1. `GetMILConversionMaps()`
   - to recover the op-name -> handler mapping directly
2. `RevalidateOpIndicesProducer(...)`
   - because it governs several important indexed op families
3. `ConvertLiveInputs(...)`
   - if we want to understand input ABI / dynamic-shape pass typing more deeply
4. `SetValidationStatus(...)`
   - if we want to reconstruct the meaning of validation statuses more precisely

If the goal is maximum signal for `rustane`, the best next target is probably:

- `GetMILConversionMaps()`

because that is the cleanest path to understanding which op families Apple
gives bespoke validation/lowering handlers at this stage.
