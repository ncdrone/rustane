# Hopper Validation Pass 18: Resource, Status, and SDPA Descriptor Validation

This pass targets the three functions requested:

- `ValidateLiveIOMemoryFootprint(...)`
- `SetValidationStatus(...)`
- `ValidateLayer_Impl<ANECSDPALayerDesc, ...>`

The goal is to understand:

- where live-I/O resource ceilings are enforced
- what the validator records as per-op status
- what the generic SDPA descriptor-validation path actually does

## 1. High-level result

These three functions sit at different layers of the validation stack:

- `ValidateLiveIOMemoryFootprint(...)`
  - resource-limit enforcement for live I/O tensors
- `SetValidationStatus(...)`
  - per-operation validation-result recording
- `ValidateLayer_Impl<ANECSDPALayerDesc, ...>`
  - generic descriptor -> unit -> validation plumbing for exported SDPA layer

Together they make the validation model clearer:

- semantic checks produce per-op status entries
- resource checks can reject otherwise-valid units/functions
- exported SDPA layer validation is mostly generic infrastructure around the SDPA unit

## 2. `ValidateLiveIOMemoryFootprint(...)`

Size:

- length: `488` bytes
- basic blocks: `33`

Callers:

- `ZinIrFactory::CheckLiveIOSize()`
- `ValidateDerivedMILProgram(...)`

Callees:

- `ZinIrTensor::IsLiveOut()`
- `ZinIrTensor::IsLiveState()`
- `ZinIrTensor::IsLiveInLiveStateOrConstTensor()`
- `ZinIrTensor::GetTensorSizeInBytesFromResidency(...)`
- `DimensionOrderHint::DimensionOrderHint(DefaultDimensionOrder)`

### 2.1 What Hopper shows

The function walks three tensor collections:

- live-ins
- live-outs
- live-state tensors

For each tensor it:

- optionally enforces that the tensor belongs to the expected class
  - live-in / live-out / live-state
- computes its byte footprint via:
  - `GetTensorSizeInBytesFromResidency(..., AllocationHint=2, DefaultDimensionOrder, ...)`
- accumulates the total

At the end it compares the aggregate size against the compiler-parameter BSS
limit and raises:

- `Error: the live io tensor memory footprint (%zd bytes) exceeds the bss limit (%lld bytes)`

Relevant tied strings:

- `Error: the tensor should be a live-in tensor`
- `Error: the tensor should be a live-out tensor`
- `Error: the tensor should be a live-state tensor`
- `Error: the live io tensor memory footprint (%zd bytes) exceeds the bss limit (%lld bytes)`

### 2.2 What this means

This is a concrete resource gate.

By the time validation reaches this function, the compiler is already enforcing:

- that tensors are classified into the right live-I/O buckets
- that their aggregate memory footprint fits the BSS budget

For `rustane`, this is directly relevant because it means some large-shape
failures may be caused by live-I/O memory budgeting before later codegen or
execution stages.

This is one of the clearest repo-relevant resource checks we’ve recovered so far.

## 3. `SetValidationStatus(...)`

Size:

- length: `404` bytes
- basic blocks: `21`

Callers include:

- `ValidateOpList(...)`
- `ValidateUnits(...)`
- `MarkAllOpsAsInvalid(...)`
- `RevalidateReadStateProducerOpsFromWriteStateOps(...)`
- `ValidateMLIRProgram(...)`

Callees are small:

- tree/map insertion
- string copy helpers

### 3.1 What Hopper shows

This function takes:

- a `map<uint64_t, ValidateEntry>&`
- a pointer to an operation identifier / key
- a boolean validation result
- two strings
- a trailing boolean flag

It then:

- inserts or updates the `ValidateEntry` for that operation key
- copies the boolean result
- copies both strings into the stored entry

So `ValidateEntry` is at least:

- keyed by op identifier
- contains a pass/fail-style boolean
- contains two strings of status/detail text
- also tracks one additional boolean-ish flag from the caller path

### 3.2 What this means

This is the central storage point for per-operation validation outcomes.

That matters because many higher-level validation functions do not just return
success/failure. They are populating a persistent status map that can later be
used to:

- explain which ops failed
- explain why they failed
- distinguish “invalid / downgraded / precision-loss / unsupported” kinds of outcomes

For `rustane`, this suggests a future dynamic experiment should not stop at
“did compile fail?” The Apple stack itself preserves much more granular per-op
validation state.

## 4. `ValidateLayer_Impl<ANECSDPALayerDesc, ...>`

The exported wrapper `_ANECValidateSDPALayer` is tiny and just forwards into:

- `ValidateLayer_Impl<ANECSDPALayerDesc, ZinIrSDPAUnit, ZinIrSDPAUnitInfo, ANECSDPALayerDescAlternate>(...)`

### 4.1 What Hopper shows

The generic SDPA layer validator does the following:

1. null-check all required validation parameters
   - otherwise:
     - `Error: All validation parameters must not be nullptr.`
2. create a `ZinIrPlistCompilationStatus`
3. obtain validation backing from `ZinValidator`
4. create file backing via `ANECCreateFileBacking(...)`
5. convert each `ANECTensorDesc` into `ZinIrTensorInfo`
   - via `ANECTensorDescToZinTensorInfo(...)`
6. if tensor value descriptors are present, convert them via
   `ANECTensorValueDescToValues(...)`
7. translate the layer descriptor into `ZinIrSDPAUnitInfo`
   - via `ANECDescToUnitInfo<ANECSDPALayerDesc, ZinIrSDPAUnitInfo>(...)`
8. construct a `ZinIrSDPAUnit`
9. attach bottoms via `ZinIrSDPAUnit::SetBottoms(...)`
10. build a `ZinValidationContext`
11. call `ZinValidationContext::Validate(unit)`

### 4.2 What this means

The exported SDPA descriptor validation path is mostly generic plumbing.

The real SDPA-specific logic is not here. It lives in:

- `ANECDescToUnitInfo<ANECSDPALayerDesc, ZinIrSDPAUnitInfo>(...)`
- the `ZinIrSDPAUnit` / `ZinSDPALayer` validation and semantics code
- `ZinValidationContext::Validate(...)`

So if we want descriptor-level SDPA semantics beyond “can the descriptor be
wrapped into a unit and passed through validation?”, this generic layer function
is not the richest target.

It is useful mainly for showing how the exported C-facing layer validator plugs
into the internal unit-validation machinery.

## 5. What this changes in our understanding

### 5.1 Resource ceilings are enforced earlier and more explicitly than expected

`ValidateLiveIOMemoryFootprint(...)` confirms a hard, explicit live-I/O BSS
budget check.

So some compile failures or downgrades relevant to `rustane` may come from:

- aggregate live-I/O byte usage

not only from op semantics or ANEC layer lowering.

### 5.2 Apple preserves per-op validation detail

`SetValidationStatus(...)` makes it clear the compiler tracks:

- per-op identity
- per-op status
- at least two strings of explanatory context

That is a much richer internal diagnostic model than a single compiler return code.

### 5.3 Exported SDPA validation is a generic shell

The generic `ValidateLayer_Impl` path for SDPA does not override the earlier
conclusion:

- the interesting SDPA-specific logic is in the unit / semantics / lowering path

not in the exported wrapper or the generic descriptor validator itself.

## 6. Best next targets from here

Given these results, the best next targets are:

1. `ZinValidationContext::Validate(...)`
   - to understand how generic unit validation drives pass/fail and alternates
2. `ANECDescToUnitInfo<ANECSDPALayerDesc, ZinIrSDPAUnitInfo>(...)`
   - to see exactly how SDPA descriptor fields are interpreted
3. `CheckLiveIOSize()` / other callers around `ValidateLiveIOMemoryFootprint(...)`
   - if the goal shifts toward hard resource ceilings

For `rustane`, the highest-value next target is probably:

- `ZinValidationContext::Validate(...)`

because it likely connects:

- per-unit semantics
- resource checks
- validation-status recording

into one shared validation engine.
