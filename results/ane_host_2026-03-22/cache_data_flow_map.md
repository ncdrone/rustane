# Cache and Data-Flow Map

This is a host-side model of how the Apple ANE stack appears to move models from
client request to compiled artifact, based on:

- XPC protocol names
- ObjC selectors
- imported symbols
- C string diagnostics

## End-to-End Flow

```text
client
  -> _ANEClient / _ANEInMemoryModel
  -> _ANEDaemonConnection XPC
  -> sandbox extension + csIdentity + model path + options + temp dirs
  -> model-type specific frontend compiler
       CoreML / MIL / MLIR / CVAIR / ANECIR / LLIR bundle
  -> Espresso translation / plan build when needed
  -> ANECompiler.framework low-level compile call
       ANECCompile / ANECCompileJIT / ANECCompileOnline
  -> cache / retain / source-model bookkeeping
  -> compiled output path + optional AOT model binary path
```

## Service Entry Points

### Compiler service

Protocol method:

- `compileModelAt:csIdentity:sandboxExtension:options:tempDirectory:cloneDirectory:outputURL:aotModelBinaryPath:withReply:`

This suggests the service expects:

- a model path,
- a code-signing identity or client identity,
- a sandbox extension,
- one or more temporary directories,
- an output URL,
- optionally an AOT output path.

### Direct framework daemon protocol

From `AppleNeuralEngine.framework` itself:

- `compileModel:sandboxExtension:options:qos:withReply:`
- `compiledModelExistsFor:withReply:`
- `compiledModelExistsMatchingHash:withReply:`
- `loadModel:sandboxExtension:options:qos:withReply:`
- `loadModelNewInstance:options:modelInstParams:qos:withReply:`
- `prepareChainingWithModel:options:chainingReq:qos:withReply:`
- `purgeCompiledModel:withReply:`
- `purgeCompiledModelMatchingHash:withReply:`
- `unloadModel:options:qos:withReply:`

This is the clearest host-side evidence yet that the real runtime flow is:

- client-side framework object model,
- daemon/XPC transport,
- compiler/load/evaluate lifecycle managed by the framework,
- not just a single raw compile primitive.

### Storage maintenance

Protocol method:

- `purgeDanglingModelsAt:withReply:`

This looks like a separate service responsible for cache hygiene and stale or
orphaned compiled model cleanup.

## Model-Type Routing

Relevant strings and keys:

- `kANEFModelTypeKey`
- `kANEFModelCoreMLValue`
- `kANEFModelMILValue`
- `kANEFModelMLIRValue`
- `kANEFModelANECIRValue`
- `kANEFModelLLIRBundleValue`
- `No model type set`

Likely meaning:

- the compiler service uses a dictionary-like model descriptor containing a
  `modelType`,
- different frontend compiler classes are selected from that model type,
- MIL and MLIR are first-class host-side formats, not artifacts invented only
  by outside reverse engineering.

## Translation / Frontend Layer

Relevant classes:

- `_ANECoreMLModelCompiler`
- `_ANEMILCompiler`
- `_ANEMLIRCompiler`
- `_ANECVAIRCompiler`
- `_ANEEspressoIRTranslator`

Relevant strings:

- `model.espresso.net`
- `model.llir.bundle`
- `defaultMILFileName`
- `defaultANECIRFileName`
- `ModelFailsEspressoCompilation`
- `ModelFailsToCompileANECIR`

Most likely interpretation:

- some model inputs are translated into Espresso plans or networks first,
- some paths generate MIL/MLIR/ANECIR artifacts on disk,
- compilation may be staged through several IR formats before a final ANE binary
  is produced.

## Low-Level Compiler Boundary

Imported from `ANECompiler.framework`:

- `_ANECCompile`
- `_ANECCompileJIT`
- `_ANECCompileOnline`
- `_ANECCreateModelDictionary`

Relevant strings:

- `Calling ANECCompileOnline...`
- `Calling ANE compiler`
- `Calling ANE compiler done ret(%d)`
- `_ANECompiler : ANECCompile() FAILED`
- `ANECCompile(%@) FAILED: err=%@`

This is the most important host-side boundary for `rustane`:

- Apple's own service shell eventually funnels into the same lower-level ANE
  compiler functions we are effectively reaching through reverse-engineered APIs.
- When `rustane` sees an ANE compile rejection, it is likely a real compiler
  limitation, not just a bug in our wrapper shape.

## Cache and Retention Layer

### Identity and cache keys

Relevant methods:

- `cacheURLIdentifierForModel:useSourceURL:withReply:`
- `cachedModelPathFor:csIdentity:`
- `cachedModelPathFor:csIdentity:useSourceURL:`
- `cachedModelPathMatchingHash:csIdentity:`
- `cachedModelAllSegmentsPathFor:csIdentity:`
- `cachedSourceModelStoreNameFor:`
- `cachedModelRetainNameFor:`
- `getModelBinaryPathFromURLIdentifier:bundleID:`

Relevant strings:

- `kANEFModelIdentityStrKey`
- `kANEFModelCacheIdentifierUsingSourceURLKey`
- `kANEFModelHasCacheURLIdentifierKey`
- `kANEFAOTCacheUrlIdentifierKey`
- `kANEFBaseModelIdentifierKey`
- `kANEModelKeyAllSegmentsValue`
- `kANEModelKeyNoSegmentsValue`

Interpretation:

- cache identity is probably not just “path to model”;
- it may incorporate:
  - source URL,
  - code-signing identity,
  - bundle ID,
  - model hash,
  - whether the output is all-segments vs standard,
  - AOT cache identifiers.

### Retain files and source-model preservation

Relevant methods and strings:

- `saveSourceModelPath:outputModelDirectory:`
- `createModelCacheRetain:`
- `cachedModelRetainNameFor:`
- `retainModelCache=%d`
- `modelCacheRetainPath=%@`
- `modelRetainFileExists=%d`
- `kANEFRetainModelsWithoutSourceURLKey`

Interpretation:

- Apple likely keeps a “retain” sidecar or marker around compiled outputs,
- source-model availability matters to cache behavior,
- retaining compiled models without a stable source URL is a configurable path.

### Access-time and GC behavior

Relevant methods:

- `setAccessTime:forModelFilePath:`
- `getAccessTimeForFilePath:`
- `updateAccessTimeForFilePath:`
- `removeFilePath:ifDate:olderThanSecond:`
- `removeStaleModels`
- `removeStaleModelsAtPath:`
- `garbageCollectDanglingModels`
- `garbageCollectDanglingModelsAtPath:`
- `purgeDanglingModelsAt:withReply:`

Relevant strings:

- `kANEAccessSeconds`
- `_ANED_MODELCACHE_GC`
- `_ANED_PURGE_COMPILED_MODEL`

Interpretation:

- cache eviction is age-based, not just size-based,
- there is explicit dangling-model garbage collection,
- the maintenance service likely runs independently of compile requests.

### Disk space and purgeability

Relevant methods:

- `getDiskSpaceItemizedByBundleIDAndPurge:`
- `getDiskSpaceForBundleID:`
- `sizeOfModelCacheAtPath:purgeSubdirectories:`
- `markPathAndDirectParentPurgeable:error:`
- `enableApfsPurging`

Relevant strings:

- `Mark %s as purgeable`
- `Fail to mark %s as purgeable`

Interpretation:

- Apple treats compiled ANE artifacts as purgeable cache entries on APFS.
- This is another sign that compiled model outputs are intended to be reused, not
  rebuilt from scratch every time.

## Sandbox Flow

Relevant selectors:

- `sandboxExtensionPathForModelURL:`
- `issueSandboxExtensionForPath:error:`
- `issueSandboxExtensionForModel:error:`
- `consumeSandboxExtension:forModel:error:`
- `consumeSandboxExtension:forPath:error:`
- `releaseSandboxExtension:handle:`

Relevant imports:

- `sandbox_extension_issue_file`
- `sandbox_extension_consume`
- `sandbox_extension_release`
- `sandbox_init_with_parameters`

Relevant strings:

- `Failed to enter sandbox: %s`
- `%@: sandbox_extension_issue_file() returned NULL. path=%@`
- `%@: model=%@ sandboxExtension=%@`

Interpretation:

- file access is intentionally delegated through sandbox extensions,
- the compiler service appears designed to run with restricted file access,
- clients are expected to hand off explicit access grants for model files.

## Runtime / Evaluation Flow

From `AppleNeuralEngine.framework` classes:

- `_ANEInMemoryModel compileWithQoS:options:error:`
- `_ANEInMemoryModel compiledModelExists`
- `_ANEInMemoryModel loadWithQoS:options:error:`
- `_ANEInMemoryModel unloadWithQoS:error:`
- `_ANEInMemoryModel evaluateWithQoS:options:request:error:`
- `_ANEInMemoryModel mapIOSurfacesWithRequest:cacheInference:error:`
- `_ANEInMemoryModel unmapIOSurfacesWithRequest:`
- `_ANEClient evaluateWithModel:options:request:qos:error:`
- `_ANEClient evaluateRealTimeWithModel:options:request:error:`
- `_ANEClient buffersReadyWithModel:inputBuffers:options:qos:error:`
- `_ANEClient enqueueSetsWithModel:outputSet:options:qos:error:`
- `_ANEClient prepareChainingWithModel:options:chainingReq:qos:error:`

Interpretation:

- compilation and evaluation are separate phases,
- models can be loaded and unloaded independently of compilation,
- IOSurface mapping is an explicit step,
- there is a notion of “buffers ready” and “enqueue sets” beyond a simple
  synchronous evaluate call,
- chaining / loopback / procedure index support likely exists to reduce dispatch
  overhead or support more complex multi-stage pipelines.

That is directly relevant to `rustane`, which currently treats each kernel
dispatch much more simply.

## Chaining and Procedure-Level Flow

The most specific host-side evidence for chained execution comes from
`AppleNeuralEngine.framework` symbols and strings.

### Chaining request object

Relevant methods:

- `chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:`
- `initWithInputs:outputs:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:`
- `loopbackInputSymbolIndex`
- `loopbackOutputSymbolIndex`
- `signalEvents`
- `procedureIndex`
- `transactionHandle`
- `fwEnqueueDelay`
- `memoryPoolId`

Relevant strings:

- `loopbackInputs`
- `loopbackOutputs`
- `signalEvents`
- `signalEventsCount`
- `signalEvents%dSymbolIndex`
- `signalEvents%dtype`
- `signalEvents%dport`
- `signalEvents%dValue`
- `signalEvents%dAgentMask`
- `memoryPoolId`
- `procedureIndex`
- `executionDelay`

Interpretation:

- the runtime supports more than one procedure per compiled model,
- buffers are identified by symbol / index rather than only by position,
- loopback outputs can feed later procedure inputs,
- shared events / signal events are part of the execution contract,
- there is explicit support for a firmware enqueue delay and a memory-pool ID.

This is the closest host-side hint yet that Apple has a real multi-stage
execution/chaining API below the higher-level compile/evaluate surface.

## Request Packing Semantics

The deeper symbol/string pass exposed a concrete request object family:

### `_ANERequest`

Relevant constructor signatures:

- `requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:`
- `requestWithInputs:inputIndices:outputs:outputIndices:perfStats:procedureIndex:`
- `requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:`
- `requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:`
- `requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:`
- `requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:`

Relevant ivars / properties inferred from strings:

- `inputIndexArray`
- `outputIndexArray`
- `weightsBuffer`
- `perfStats`
- `perfStatsArray`
- `sharedEvents`
- `transactionHandle`
- `procedureIndex`

Validation/error strings:

- `At least one input or output name required. inputIndexArray.count = %lu : outputIndexArray.count = %lu`
- `inputArray and inputIndexArray size mismatch...`
- `outputArray and outputIndexArray size mismatch...`
- `inputIndexArray[%u]... exceeds kANERequestMaxSymbolIndex=%d`
- `outputIndexArray[%u]... exceeds kANERequestMaxSymbolIndex=%d`

Interpretation:

- requests are not just “list of buffers”; they are explicitly symbol-indexed,
- `weightsBuffer` can be carried as part of the request itself,
- perf stats collection is part of request packing,
- shared events and transaction handles are optional request-level controls,
- there are explicit hard limits like `kANERequestMaxSymbolIndex`.

For `rustane`, this suggests the underlying Apple runtime contract is closer to
a symbol-indexed program I/O map than a simple ordered tensor array.

## Real-Time Flow

Relevant methods:

- `beginRealTimeTask`
- `endRealTimeTask`
- `beginRealTimeTaskWithReply:`
- `endRealTimeTaskWithReply:`
- `loadRealTimeModel:options:qos:error:`
- `evaluateRealTimeWithModel:options:request:error:`
- `unloadRealTimeModel:options:qos:error:`

Relevant symbols/strings:

- `aneRealTimeTaskQoS`

Interpretation:

- the framework distinguishes a real-time execution path from ordinary evaluate,
- real-time appears to be a task context entered and exited explicitly,
- this may help explain why the framework has both normal and “priority/queue”
  plumbing rather than a single stateless evaluate API.

For `rustane`, this suggests there may be an unexplored performance/control path
below the current direct-eval usage.

## Model Descriptor Semantics

Relevant methods:

- `modelWithMILText:weights:optionsPlist:`
- `modelWithNetworkDescription:weights:optionsPlist:`
- `initWithNetworkText:weights:optionsPlist:isMILModel:`
- `networkTextHash`
- `weightsHash`
- `optionsPlistHash`
- `hexStringIdentifier`
- `isMILModel`

Relevant strings:

- `networkText`
- `networkTextHash`
- `weightsHash`
- `optionsPlist`
- `optionsPlistHash`
- `descriptor`
- `string_id`

Interpretation:

- descriptor identity is hash-based across at least:
  - network text,
  - weights,
  - options plist.
- MIL is not inferred heuristically; there is an explicit `isMILModel` bit.
- the descriptor likely drives both compile deduplication and cache identity.

That is important for `rustane` because it means:

- stable MIL text and stable options/weights hashing may matter to cache reuse,
- path-only identity is likely not the whole story,
- changing “equivalent” model text formatting may still alter cache identity.

## Cache Identifier Semantics

The deeper pass also exposed a richer `_ANEModel` identity model.

Relevant constructors:

- `modelWithCacheURLIdentifier:`
- `modelWithCacheURLIdentifier:UUID:`
- `modelAtURLWithCacheURLIdentifier:key:cacheURLIdentifier:`
- `modelAtURLWithSourceURL:sourceURL:key:cacheURLIdentifier:`
- `modelAtURLWithSourceURL:sourceURL:key:identifierSource:cacheURLIdentifier:`
- `modelAtURLWithSourceURL:sourceURL:key:identifierSource:cacheURLIdentifier:UUID:`

Relevant fields / selectors:

- `identifierSource`
- `cacheURLIdentifier`
- `sourceURL`
- `string_id`
- `hexStringIdentifier`
- `compiledModelExistsMatchingHash`
- `purgeCompiledModelMatchingHash`

Relevant validation strings:

- `identifierSource is _ANEModelCacheURLIdentifierSource but cacheURLIdentifier is nil`
- `cacheURLIdentifier(%@) contains .., hence invalid`
- `cacheURLIdentifier is set already!`
- `identifierSource is _ANEModelIdentifierSourceURLAndKey but sourceURL is nil`
- `model cacheURLIdentifier for new instance should be nil`

Interpretation:

- model identity appears to support several explicit source modes,
- cache URL identifiers are treated as security-/validity-sensitive strings,
- source URL and cache URL identifier are not interchangeable,
- “new instance” loading has its own cache-identifier rules.

This is likely highly relevant to any future `rustane` work on compile reuse.

## Virtual-Client Fallback / Virtualization Path

The deeper pass surfaced a substantial `_ANEVirtualClient` implementation.

Relevant methods:

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
- `beginRealTimeTask`
- `endRealTimeTask`
- `mapIOSurfacesWithModel:request:cacheInference:error:`
- `doMapIOSurfacesWithModel:request:cacheInference:error:`
- `validateEnvironmentForPrecompiledBinarySupport`
- `validateNetworkCreateMLIR:validation_params:`
- `validateNetworkCreate:uuid:function:directoryPath:scratchPadPath:milTextData:`
- `transferFileToHostWithPath:withChunkSize:withUUID:withModelInputPath:overWriteFileNameWith:`

Relevant strings:

- `AppleVirtIONeuralEngineDevice`
- `ANEVirtualClient Found a device of class AppleVirtIONeuralEngineDevice`
- `Chunking required for file transfer and guest-to-host interface is incompatible`
- `loadModel dictionary Model Cache URL from Host %@`
- `Calling functions from userspace to kernel space...`

Interpretation:

- Apple has a virtualization / guest-host transport path for ANE,
- that path transfers model files, options, metadata, and IOSurface IDs across a
  boundary,
- chunking and interface-version compatibility matter in that path,
- it is distinct enough to have its own compile/load/evaluate implementations.

This is probably not immediately useful for ordinary local `rustane` execution,
but it is very useful for understanding why the framework has a larger API
surface than the local daemon/XPC path alone would suggest.

## Relevance to `rustane`

The biggest likely connections to `rustane` are:

1. **Compile failures are probably real compiler boundary failures**
   - Apple itself ultimately calls `_ANECCompile*`.

2. **Compile-cache reuse may matter**
   - If `ane-bridge` recompiles from fresh temp paths every run, it may be
     missing host-side reuse patterns Apple depends on.

3. **Model format vocabulary is now less speculative**
   - MIL / MLIR / ANECIR / LLIR / Espresso are explicit host-side concepts.

4. **Sandbox behavior matters if we ever reproduce Apple's service route**
   - Especially if the project later tries to compare direct private API usage
     against service-mediated compilation.

5. **The private framework surface strongly validates the current project
   direction**
   - `_ANEClient`, `_ANEInMemoryModel`, `_ANEInMemoryModelDescriptor`, and
     IOSurface-backed buffer wrappers are all real host-side concepts.
   - The README claim that `rustane` is working with `_ANEClient` and
     `_ANEInMemoryModel` is consistent with what is present on this machine.
