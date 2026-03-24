# Hopper Validation Pass 42: Request, Direct Evaluation, and Chaining Paths

This pass targets the AppleNeuralEngine runtime submission path:

- `-[_ANEClient doEvaluateDirectWithModel:options:request:qos:error:]`
- `-[_ANEClient doPrepareChainingWithModel:options:chainingReq:qos:error:]`
- `-[_ANERequest initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:]`
- `-[_ANERequest validate]`
- `-[_ANEChainingRequest initWithInputs:outputs:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:]`
- `-[_ANEChainingRequest validate]`
- fallback:
  - `-[_ANEVirtualClient doEvaluateWithModel:options:request:qos:completionEvent:error:]`
  - `-[_ANEVirtualClient(Private) doEvaluateWithModelLegacy:options:request:qos:completionEvent:error:]`

The goal is to clarify which dispatch-reduction ideas are realistically suggested
by the current runtime path.

## 1. High-level result

The runtime path splits cleanly into three layers:

- `_ANERequest` / `_ANEChainingRequest`
  - data carriers plus validation
- `_ANEClient`
  - direct host-side submission / prepare-chaining orchestration
- `_ANEVirtualClient`
  - a much heavier fallback/VM path with explicit IOSurface packing and IOUserClient calls

This matters for `rustane` because direct dispatch reduction work should focus on
the `_ANEClient` path first. The virtual client is much more complex and is not
the right baseline for a first optimization attempt.

## 2. `_ANERequest`

### 2.1 Constructor layout

`-[_ANERequest initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:]`

The initializer stores:

- `inputArray` at `+0x08`
- `inputIndexArray` at `+0x10`
- `outputArray` at `+0x18`
- `outputIndexArray` at `+0x20`
- `weightsBuffer` at `+0x28`
- `procedureIndex` at `+0x30`
- `transactionHandle` at `+0x38`
- `sharedEvents` at `+0x40`
- `perfStatsArray` at `+0x50`

This matches and refines the earlier request-layout work:
- the constructor’s trailing “perfStats” parameter is really populating the
  `perfStatsArray` slot
- `perfStats` itself is a separate later-set field

### 2.2 Validation rules

`-[_ANERequest validate]` confirms:

- `inputArray.count > 0`
- `outputArray.count > 0`
- `inputArray.count == inputIndexArray.count`
- `outputArray.count == outputIndexArray.count`
- each input/output symbol index must be `< 0xff`
- `procedureIndex < 0x81`
- perf stat entries must have valid stat types
- shared-event signal/wait counts are bounded

The request validator is strict and preserves detailed error reporting, but it
does **not** imply any built-in multi-procedure batching beyond the explicit
`procedureIndex` field.

## 3. `_ANEChainingRequest`

### 3.1 Constructor layout

`-[_ANEChainingRequest initWithInputs:outputs:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:]`

The initializer stores:

- `inputBuffer`
- `outputSets`
- `loopbackInputSymbolIndex`
- `loopbackOutputSymbolIndex`
- `signalEvents`
- `procedureIndex`
- `transactionHandle`
- `fwEnqueueDelay`
- `memoryPoolId`

This is the concrete runtime shape behind the chaining APIs we had previously
only inferred from strings and partial decompilation.

### 3.2 Validation rules

`-[_ANEChainingRequest validate]` confirms:

- `inputBuffer.count > 0`
- `outputSets.count > 0`
- each input symbol index `< 0xff`
- each output set must have at least one output buffer
- output-buffer symbol indices `< 0xff`
- `procedureIndex < 0x81`
- loopback input/output arrays must have the same count
- loopback symbol count is bounded
- signal event count is bounded

So chaining is real, but it is also heavily constrained and explicitly modeled
as a separate request type rather than an overloaded `_ANERequest`.

## 4. `_ANEClient` direct evaluation

### 4.1 `doEvaluateDirectWithModel`

This is the clearest direct-dispatch path.

The function:

- rejects nil model early
- falls back to `virtualClient` if present
- otherwise:
  - computes `queueIndexForQoS`
  - inspects request input/output IOSurface IDs for tracing
  - waits on a queue-specific semaphore
  - calls `processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:`
    on the model’s `program`
  - handles completion handler / shared events
  - emits trace/signpost events

This is strong evidence that the direct path still submits **one request at a
time** and uses queue selection plus semaphore coordination rather than any
obvious built-in request coalescing.

### 4.2 What this means

The direct path can plausibly save daemon/XPC overhead, but Hopper does not
suggest that it inherently batches multiple logical kernel executions into a
single submission. Any dispatch-reduction win from direct eval is likely to come
from lower per-request overhead, not hidden multi-op batching.

## 5. `_ANEClient` chaining preparation

`-[_ANEClient doPrepareChainingWithModel:options:chainingReq:qos:error:]`

This path:

- rejects virtual-client mode
- rejects VM/host-too-old cases
- traces using `model.string_id`
- validates the chaining request
- picks the loading connection / queue for the requested QoS
- dispatches synchronously through that queue

This confirms:

- chaining is treated as a model-loading/runtime-preparation operation, not just
  a special evaluate flag
- `string_id` and queue selection are part of the real chaining path

That makes chaining a credible future optimization target, but it still needs a
separate request type and preparation call rather than a trivial extension of
the current request submission path.

## 6. `_ANEVirtualClient`

### 6.1 `doEvaluateWithModel`

The virtual path is large and clearly much heavier:

- it serializes the model/options/request state into explicit structs
- copies model/request metadata to IOSurfaces
- assigns input/output/perfStats/shared-event arrays into packed buffers
- creates extra control IOSurfaces
- calls an IOUserClient method directly
- updates errors and perf stats after return

### 6.2 `doEvaluateWithModelLegacy`

The legacy path does similar explicit serialization and dictionary-based VM
submission, with many more string-keyed dictionary fields and IOSurface copies.

### 6.3 What this means

This path is not where we should expect an easy dispatch-reduction win for the
repo. It is clearly a fallback/compat path with a lot of packing overhead.

## 7. What this changes in our understanding

### 7.1 The best dispatch-reduction target is still chaining

The direct path is leaner than the daemon path, but it still looks one-request /
one-submission oriented. The only concrete runtime path that obviously points
toward dispatch reduction beyond per-request overhead is the separate chaining
API family.

### 7.2 `_ANERequest` and `_ANEChainingRequest` are distinct execution models

This is useful for `rustane` planning:

- `_ANERequest` is the normal single-evaluation request shape
- `_ANEChainingRequest` is the explicit structured path for loopback/signal/memory-pool driven execution

So if the repo ever wants to prototype Apple-like chaining, it likely needs new
FFI/request builders, not just extra fields on the current request path.
