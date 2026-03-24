# Framework Checks Part 4: Request Packing, Virtual Client, and Cache Identity

This pass focused specifically on the extracted
`/tmp/ipsw-ane-extract/AppleNeuralEngine` binary.

The goal was to go deeper on:

- request packing,
- virtual-client fallback,
- model/cache identifier behavior.

## 1. Request Packing

The clearest new result from this pass is the presence of a dedicated request
object with multiple packing variants.

### `_ANERequest`

Observed constructor signatures:

- `requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:`
- `requestWithInputs:inputIndices:outputs:outputIndices:perfStats:procedureIndex:`
- `requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:`
- `requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:`
- `requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:`
- `requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:`

Observed request fields / ivars:

- `inputIndexArray`
- `outputIndexArray`
- `weightsBuffer`
- `perfStats`
- `perfStatsArray`
- `sharedEvents`
- `transactionHandle`
- `procedureIndex`

Observed validation strings:

- `At least one input or output name required...`
- `inputArray and inputIndexArray size mismatch...`
- `outputArray and outputIndexArray size mismatch...`
- `request.inputIndexArray[%u]=%u is invalid`
- `request.outputIndexArray[%u]=%u is invalid`
- `kANERequestMaxSymbolIndex`

### Interpretation

This suggests the runtime’s execution request format is:

- symbol-indexed,
- capable of carrying a separate weights buffer,
- capable of requesting perf stats,
- capable of attaching shared events,
- optionally tied to a transaction handle,
- tied to a particular `procedureIndex`.

That is a more structured execution contract than a plain `inputs[] -> outputs[]`
call.

## 2. Cache Identity and Model Identity

The `_ANEModel` and `_ANEInMemoryModelDescriptor` signals became clearer.

### `_ANEModel`

Observed constructors:

- `modelWithCacheURLIdentifier:`
- `modelWithCacheURLIdentifier:UUID:`
- `modelAtURLWithCacheURLIdentifier:key:cacheURLIdentifier:`
- `modelAtURLWithSourceURL:sourceURL:key:cacheURLIdentifier:`
- `modelAtURLWithSourceURL:sourceURL:key:identifierSource:cacheURLIdentifier:`
- `modelAtURLWithSourceURL:sourceURL:key:identifierSource:cacheURLIdentifier:UUID:`

Observed fields:

- `identifierSource`
- `cacheURLIdentifier`
- `sourceURL`
- `string_id`
- `perfStatsMask`

Observed validation strings:

- `identifierSource is _ANEModelCacheURLIdentifierSource but cacheURLIdentifier is nil`
- `cacheURLIdentifier(%@) contains .., hence invalid`
- `cacheURLIdentifier is set already!`
- `identifierSource is _ANEModelIdentifierSourceURLAndKey but sourceURL is nil`
- `model cacheURLIdentifier for new instance should be nil`

### `_ANEInMemoryModelDescriptor`

Observed symbols:

- `modelWithMILText:weights:optionsPlist:`
- `modelWithNetworkDescription:weights:optionsPlist:`
- `networkTextHash`
- `weightsHash`
- `optionsPlistHash`
- `hexStringIdentifier`

### Interpretation

There appear to be two overlapping identity layers:

1. **Descriptor identity**
   - based on `networkTextHash`, `weightsHash`, `optionsPlistHash`,
   - used for content-derived identity.

2. **Model/cache identity**
   - based on `cacheURLIdentifier`, `sourceURL`, `identifierSource`, and
     `string_id`,
   - used for persistent cache lookup / lifecycle tracking.

This is important because it means Apple’s stack likely separates:

- “what model is this, structurally?”
- “what cached artifact / source path / source mode does this model belong to?”

## 3. Compiled-Model Existence and Purge by Hash

Observed methods:

- `compiledModelExistsFor:`
- `compiledModelExistsMatchingHash:`
- `purgeCompiledModel:`
- `purgeCompiledModelMatchingHash:`

These exist on:

- `_ANEClient`
- `_ANEDaemonConnection`
- `_ANEVirtualClient`
- `_ANEInMemoryModel` (at least the simple existence/purge forms)

### Interpretation

The runtime clearly supports:

- checking cache presence by ordinary model identity,
- checking cache presence by hash,
- purging by ordinary identity,
- purging by hash.

That strengthens the earlier conclusion that compiled-model caching is a major,
first-class behavior rather than a side effect.

## 4. Virtual Client Path

The `_ANEVirtualClient` path is much richer than just a few stubs.

### Key observed methods

- `compileModel:options:qos:error:`
- `loadModel:options:qos:error:`
- `loadModelNewInstance:options:modelInstParams:qos:error:`
- `loadModelNewInstanceLegacy:options:modelInstParams:qos:error:`
- `unloadModel:options:qos:error:`
- `evaluateWithModel:options:request:qos:error:`
- `doEvaluateWithModel:options:request:qos:completionEvent:error:`
- `compiledModelExistsFor:`
- `compiledModelExistsMatchingHash:`
- `purgeCompiledModel:`
- `purgeCompiledModelMatchingHash:`
- `mapIOSurfacesWithModel:request:cacheInference:error:`
- `doMapIOSurfacesWithModel:request:cacheInference:error:`
- `validateEnvironmentForPrecompiledBinarySupport`
- `validateNetworkCreateMLIR:validation_params:`
- `validateNetworkCreate:uuid:function:directoryPath:scratchPadPath:milTextData:`
- `transferFileToHostWithPath:withChunkSize:withUUID:withModelInputPath:overWriteFileNameWith:`

### Key observed strings

- `AppleVirtIONeuralEngineDevice`
- `ANEVirtualClient Found a device of class AppleVirtIONeuralEngineDevice`
- `Calling functions from userspace to kernel space...`
- `Chunking required for file transfer and guest-to-host interface is incompatible`
- `loadModel dictionary Model Cache URL from Host %@`
- `ANEVirtualClient compileModel dictionary call succeeded`
- `ANEVirtualClient evaluateWithModel dictionary call succeeded`

### Interpretation

The virtual client looks like a guest/host transport implementation that:

- serializes model files, options, metadata, and requests,
- transports them via IOSurfaces and dictionary-style command packets,
- talks to a kernel/userclient path tied to a virtual ANE device,
- has its own compatibility and chunk-transfer constraints.

This is probably not the path `rustane` uses today, but it reveals a lot about
how Apple itself thinks about model packaging at the transport layer.

## 5. Why This Matters for `rustane`

These findings suggest three concrete research directions:

1. **Request packing**
   - `rustane` currently uses straightforward positional tensor arrays.
   - The Apple runtime suggests a richer symbol-indexed request model with
     optional request-local weights, perf stats, and event metadata.

2. **Cache identity**
   - if `rustane` ever wants compile reuse, it may need more stable descriptor
     identity and a clearer distinction between descriptor hash and cache URL
     identity.

3. **Chaining**
   - Apple’s runtime likely supports lower-overhead, multi-procedure execution
     patterns that could matter for dispatch-heavy workloads.
