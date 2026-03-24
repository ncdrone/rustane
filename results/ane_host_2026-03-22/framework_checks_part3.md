# Framework Checks Part 3: Targeted `AppleNeuralEngine` and `MLCompilerOSXPC` Passes

This pass focused on three questions:

1. what does `prepareChaining` really imply?
2. how explicit is the real-time path?
3. what do model-descriptor symbols suggest about cache identity and MIL usage?

It also included a focused check on `MLCompilerOSXPC`.

## 1. `AppleNeuralEngine.framework` Focused Symbol Pass

An extracted copy of `AppleNeuralEngine` was created from the dyld cache with:

- `ipsw-safe dyld extract ... AppleNeuralEngine`

That extraction is marked experimental by `ipsw-safe`, but it was good enough to
run focused `macho info -n` and `macho info -c` passes.

### Chaining findings

Relevant symbols:

- `prepareChainingWithModel:options:chainingReq:qos:error:`
- `prepareChainingWithModel:options:chainingReq:qos:withReply:`
- `chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:`
- `requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:`
- `requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:`
- `requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:`
- `requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:`
- `requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:`

Relevant strings:

- `loopbackInputs`
- `loopbackOutputs`
- `signalEvents`
- `signalEventsCount`
- `memoryPoolId`
- `procedureIndex`
- `executionDelay`

### Interpretation

This is stronger than a generic “maybe it supports chaining” guess.

The Apple runtime appears to have:

- explicit procedure indexing,
- loopback input/output symbol indices,
- optional event signaling,
- transaction handles,
- an enqueue delay,
- a memory-pool ID.

That suggests the ANE host runtime can coordinate more complex multi-stage or
low-overhead execution flows than the simple one-request/one-output mental model.

## 2. Real-time path findings

Relevant symbols:

- `beginRealTimeTask`
- `endRealTimeTask`
- `beginRealTimeTaskWithReply:`
- `endRealTimeTaskWithReply:`
- `loadRealTimeModel:options:qos:error:`
- `evaluateRealTimeWithModel:options:request:error:`
- `unloadRealTimeModel:options:qos:error:`
- `aneRealTimeTaskQoS`

### Interpretation

The host stack distinguishes:

- normal compile/load/evaluate/unload
- real-time begin/load/evaluate/unload/end

That is not just a QoS flag on a normal evaluate call. It looks like a separate
mode or session boundary in the framework/daemon protocol.

## 3. Model-descriptor semantics

Relevant symbols:

- `modelWithMILText:weights:optionsPlist:`
- `modelWithNetworkDescription:weights:optionsPlist:`
- `initWithNetworkText:weights:optionsPlist:isMILModel:`
- `networkTextHash`
- `weightsHash`
- `optionsPlistHash`
- `hexStringIdentifier`
- `descriptor`
- `isMILModel`
- `string_id`

Relevant strings:

- `networkText`
- `networkTextHash`
- `weightsHash`
- `optionsPlist`
- `optionsPlistHash`

### Interpretation

The descriptor object appears to encode model identity from:

- network text,
- weights,
- options plist,
- a MIL/non-MIL flag.

This strongly implies cache identity is content-based, not just path-based.

For `rustane`, that means stable descriptor construction may matter if later
work tries to exploit compile-cache reuse or compare Apple's cache behavior to
the project’s current direct compilation flow.

## 4. `MLCompilerOSXPC`

Target:

- `/System/Library/PrivateFrameworks/MLCompilerServices.framework/Versions/A/XPCServices/MLCompilerOSXPC.xpc/Contents/MacOS/MLCompilerOSXPC`

### What was visible

Info plist:

- bundle id: `com.apple.mlcompiler.services.compiler`

Symbol pass:

- `xpc_compile(...)`
- `compile_thread_handler(void*)`
- `conection_handler(_xpc_connection_s*)`
- `mlc::xpc_dispatch_t::handle_xpc_call(void*) const`
- imported symbol `_mlc_model_compile`

String pass:

- `com.apple.mlcompiler.service.compiler`

### Interpretation

This looks like a thin XPC wrapper over an internal `MLCompilerOS` library.
Unlike `AppleNeuralEngine.framework`, it does not currently expose a rich ObjC
surface through `class-dump`.

So the current conclusion is:

- `MLCompilerServices` is real and likely important in Apple’s broader ML
  compile stack,
- but for `rustane` specifically, it is still less directly useful than the
  `AppleNeuralEngine` + `ANECompilerService` path we already mapped.

## Practical Takeaway

The most important new insight from this pass is not another class name. It is
that Apple’s host ANE runtime appears to support:

- descriptor-hash-based model identity,
- real-time execution mode,
- multi-procedure chained execution with loopback and event signaling.

That is enough to justify future `rustane` research in three directions:

1. descriptor-stable compile-cache reuse
2. lower-overhead chained execution models
3. real-time / QoS differentiated execution paths
