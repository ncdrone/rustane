# Interesting Symbols and Strings

This file collects the highest-signal symbols and strings extracted from the ANE
host services that look actionable for further `rustane` work.

## Imported Frameworks and Libraries

From `ANECompilerService`:

- `/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/Versions/A/AppleNeuralEngine`
- `/System/Library/PrivateFrameworks/ANECompiler.framework/Versions/A/ANECompiler`
- `/System/Library/PrivateFrameworks/MIL.framework/Versions/A/MIL`
- `/System/Library/PrivateFrameworks/Espresso.framework/Versions/A/Espresso`
- `/System/Library/Frameworks/IOSurface.framework/Versions/A/IOSurface`
- `/System/Library/Frameworks/IOKit.framework/Versions/A/IOKit`

From `ANEStorageMaintainer`:

- `/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/Versions/A/AppleNeuralEngine`
- `/System/Library/PrivateFrameworks/ANECompiler.framework/Versions/A/ANECompiler`
- `/System/Library/PrivateFrameworks/MIL.framework/Versions/A/MIL`

## Low-Level Compiler Imports

Imported from `ANECompiler.framework`:

- `_ANECCompile`
- `_ANECCompileJIT`
- `_ANECCompileOnline`
- `_ANECCreateModelDictionary`

Why these matter:

- these are the host-side compile calls likely nearest to the failures `rustane`
  observes,
- if later instrumentation or logging is possible, these are prime hook points.

## Espresso Imports

Imported from `Espresso.framework`:

- `_espresso_create_context`
- `_espresso_create_plan`
- `_espresso_plan_add_network`
- `_espresso_plan_build`
- `_espresso_plan_get_error_info`
- `_espresso_network_compiler_set_metadata_key`
- `_espresso_dump_ir`
- `_espresso_context_set_int_option`
- `_espresso_context_destroy`

Why these matter:

- they strongly suggest Apple's ANE compiler service uses Espresso as a real IR
  planner / frontend, not just as a historical leftover.

## Sandbox Imports

Imported from `libSystem.B.dylib`:

- `_sandbox_extension_issue_file`
- `_sandbox_extension_consume`
- `_sandbox_extension_release`
- `_sandbox_free_error`
- `_sandbox_init_with_parameters`

Why these matter:

- this gives a concrete host-side explanation for why the service contract
  includes a sandbox extension argument.

## Compiler Format / Option Keys

High-value constants:

- `kANEFModelTypeKey`
- `kANEFModelCoreMLValue`
- `kANEFModelMILValue`
- `kANEFModelMLIRValue`
- `kANEFModelANECIRValue`
- `kANEFModelLLIRBundleValue`
- `kANEModelKeyEspressoTranslationOptions`
- `kANEFCompilerOptionsFilenameKey`
- `kANEFNetPlistFilenameKey`
- `kANEFModelDescriptionKey`
- `kANEFModelIdentityStrKey`
- `kANEFModelIsEncryptedKey`
- `kANEFSkipPreparePhaseKey`
- `kANEFKeepModelMemoryWiredKey`
- `kANEFEnablePowerSavingKey`
- `kANEFEnableLateLatchKey`
- `kANEFDisableIOFencesUseSharedEventsKey`
- `kANEFEnableFWToFWSignal`
- `kANEFPerformanceStatsMaskKey`
- `kANEFMemoryPoolIDKey`
- `kANEFIntermediateBufferHandleKey`

Why these matter:

- these are likely the exact option-dictionary keys Apple’s host compiler stack
  reads,
- several may correspond to runtime/performance toggles `rustane` does not
  currently expose or even know about.

## Cache / Retention Keys

- `kANEFModelHasCacheURLIdentifierKey`
- `kANEFModelCacheIdentifierUsingSourceURLKey`
- `kANEFAOTCacheUrlIdentifierKey`
- `kANEFBaseModelIdentifierKey`
- `kANEFRetainModelsWithoutSourceURLKey`
- `kANEModelKeyAllSegmentsValue`
- `kANEModelKeyNoSegmentsValue`
- `kANEDModelCacheDetailsKey`
- `kANEDModelDirectorySizeKey`
- `kANEAccessSeconds`

Why these matter:

- they provide the internal vocabulary for compiled-model caching,
- these are likely useful clues if we want to study whether `rustane` is
  defeating host-side compile reuse.

## Model Artifact Names

- `defaultMILFileName`
- `defaultANECIRFileName`
- `model.espresso.net`
- `model.llir.bundle`

Why these matter:

- they are the clearest host-side hints we have for how intermediate artifacts
  are named on disk.

## High-Signal Diagnostics

Compiler-side diagnostics:

- `_ANECompiler : ANECCompile() FAILED`
- `ANECCompile(%@) FAILED: err=%@`
- `Calling ANECCompileOnline... modelFilename=%@ compilerInput:%@ compilerOptions:%@`
- `Calling ANE compiler`
- `Calling ANE compiler done ret(%d)`
- `_ANEMILCompiler: for %@ FAILED: lAttr=%@ : lErr=%@`
- `_ANEMLIRCompiler: for %@ FAILED: lAttr=%@ : lErr=%@`
- `_ANECVAIRCompiler: for %@ FAILED: lAttr=%@ : lErr=%@`
- `_ANECoreMLModelCompiler : error %@`
- `_ANEEspressoIRTranslator : error %s`
- `ModelFailsEspressoCompilation`
- `ModelFailsToCompileANECIR`
- `No compiler options at optionsFilePath (%@)`
- `No model type set`

Sandbox/cache diagnostics:

- `Failed to enter sandbox: %s`
- `%@: sandbox_extension_issue_file() returned NULL. path=%@`
- `%@: model=%@ sandboxExtension=%@`
- `%@: Remove modelDirURL=%@ : %@`
- `%@: Removing compiled JIT models associated with =%@ :`
- `Mark %s as purgeable`
- `Fail to mark %s as purgeable`
- `retainModelCache=%d`
- `modelRetainFileExists=%d`

Why these matter:

- these strings tell us what Apple’s own internal error taxonomy and logging
  surface look like,
- if later experiments can trigger or capture host logs, these are the exact
  messages to search for.

## Other Interesting Imported Classes

Imported from `AppleNeuralEngine.framework`:

- `_ANECloneHelper`
- `_ANEDataReporter`
- `_ANEDeviceInfo`
- `_ANEErrors`
- `_ANEHashEncoding`
- `_ANELog`
- `_ANEStrings`

These look like framework-level helpers rather than service-local classes, and
they may be worth targeting next if we keep drilling into the host stack.

## AppleNeuralEngine Framework Surface

Most relevant framework classes and protocols seen directly in
`AppleNeuralEngine.framework`:

- `_ANEDaemonProtocol`
- `_ANEDaemonProtocol_Private`
- `_ANEClient`
- `_ANEDaemonConnection`
- `_ANEInMemoryModel`
- `_ANEInMemoryModelDescriptor`
- `_ANEIOSurfaceObject`
- `_ANEIOSurfaceOutputSets`
- `_ANEBuffer`
- `_ANEInputBuffersReady`
- `_ANEChainingRequest`
- `_ANEDeviceInfo`
- `_ANEErrors`
- `_ANECloneHelper`
- `_ANELog`

Most relevant selectors:

- `compileModel:sandboxExtension:options:qos:withReply:`
- `loadModel:sandboxExtension:options:qos:withReply:`
- `loadModelNewInstance:options:modelInstParams:qos:withReply:`
- `evaluateWithQoS:options:request:error:`
- `prepareChainingWithModel:options:chainingReq:qos:error:`
- `buffersReadyWithModel:inputBuffers:options:qos:error:`
- `mapIOSurfacesWithRequest:cacheInference:error:`
- `unmapIOSurfacesWithRequest:`
- `purgeCompiledModel`
- `compiledModelExists`
- `modelWithMILText:weights:optionsPlist:`
- `modelWithNetworkDescription:weights:optionsPlist:`

These are the most direct host-side symbols/verbs corresponding to what
`rustane` is trying to drive.

## Chaining / Real-Time / Descriptor Signals

High-value `AppleNeuralEngine.framework` symbols and strings from the focused
pass:

### Chaining

- `prepareChainingWithModel:options:chainingReq:qos:error:`
- `prepareChainingWithModel:options:chainingReq:qos:withReply:`
- `chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:`
- `requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:`
- `requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:`
- `requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:`
- `requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:`
- `requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:`

Strings:

- `loopbackInputs`
- `loopbackOutputs`
- `procedureIndex`
- `signalEvents`
- `signalEventsCount`
- `memoryPoolId`
- `executionDelay`

### Real-time

- `beginRealTimeTask`
- `endRealTimeTask`
- `beginRealTimeTaskWithReply:`
- `endRealTimeTaskWithReply:`
- `loadRealTimeModel:options:qos:error:`
- `evaluateRealTimeWithModel:options:request:error:`
- `unloadRealTimeModel:options:qos:error:`
- `aneRealTimeTaskQoS`

### Model-descriptor semantics

- `modelWithMILText:weights:optionsPlist:`
- `modelWithNetworkDescription:weights:optionsPlist:`
- `initWithNetworkText:weights:optionsPlist:isMILModel:`
- `networkTextHash`
- `weightsHash`
- `optionsPlistHash`
- `hexStringIdentifier`
- `string_id`
- `descriptor`

These are especially useful because they tell us:

- the runtime has explicit multi-procedure / loopback semantics,
- there is a distinct real-time API family,
- model identity is tied to descriptor hashes, not just filesystem paths.

## Request-Packing Signals

High-value request-related symbols:

- `requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:`
- `requestWithInputs:inputIndices:outputs:outputIndices:perfStats:procedureIndex:`
- `requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:`
- `requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:`
- `requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:`
- `requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:`

High-value request-related strings:

- `inputIndexArray`
- `outputIndexArray`
- `weightsBuffer`
- `perfStats`
- `perfStatsArray`
- `sharedEvents`
- `transactionHandle`
- `At least one input or output name required...`
- `inputArray and inputIndexArray size mismatch...`
- `outputArray and outputIndexArray size mismatch...`
- `kANERequestMaxSymbolIndex`

This is the cleanest evidence yet that ANE runtime requests are symbol-indexed
and carry more than just input/output tensors.

## Cache-Identifier Signals

High-value cache/identity symbols:

- `modelWithCacheURLIdentifier:`
- `modelAtURLWithCacheURLIdentifier:key:cacheURLIdentifier:`
- `modelAtURLWithSourceURL:sourceURL:key:cacheURLIdentifier:`
- `modelAtURLWithSourceURL:sourceURL:key:identifierSource:cacheURLIdentifier:`
- `identifierSource`
- `cacheURLIdentifier`
- `compiledModelExistsMatchingHash:`
- `purgeCompiledModelMatchingHash:`

High-value strings:

- `identifierSource is _ANEModelCacheURLIdentifierSource but cacheURLIdentifier is nil`
- `cacheURLIdentifier(%@) contains .., hence invalid`
- `cacheURLIdentifier is set already!`
- `identifierSource is _ANEModelIdentifierSourceURLAndKey but sourceURL is nil`
- `model cacheURLIdentifier for new instance should be nil`

These suggest Apple distinguishes multiple identity modes for models and treats
cache URL identifiers as a first-class, validated concept.

## Virtual-Client Signals

High-value virtual-client symbols:

- `_ANEVirtualClient`
- `transferFileToHostWithPath:withChunkSize:withUUID:withModelInputPath:overWriteFileNameWith:`
- `validateEnvironmentForPrecompiledBinarySupport`
- `validateNetworkCreateMLIR:validation_params:`
- `validateNetworkCreate:uuid:function:directoryPath:scratchPadPath:milTextData:`
- `doEvaluateWithModel:options:request:qos:completionEvent:error:`
- `doMapIOSurfacesWithModel:request:cacheInference:error:`

High-value strings:

- `AppleVirtIONeuralEngineDevice`
- `ANEVirtualClient Found a device of class AppleVirtIONeuralEngineDevice`
- `Chunking required for file transfer and guest-to-host interface is incompatible`
- `loadModel dictionary Model Cache URL from Host %@`
- `calling dictionary compileModel method`
- `calling dictionary doEvaluateWithModel method`

This clearly points to a virtualization / guest-host ANE path in Apple’s
framework, separate from the ordinary local daemon connection.

## Additional Service IDs

- `com.apple.ANECompilerService`
- `com.apple.private.ANEStorageMaintainer`
- `com.apple.mlcompiler.services.compiler`

The first two are clearly ANE-specific. The third appears to be a neighboring ML
compiler service rather than the ANE runtime itself.

## MLCompilerServices / MLCompilerOSXPC

The `MLCompilerOSXPC` executable is not ObjC- or Swift-rich, but symbol output
shows:

- `xpc_compile(...)`
- `compile_thread_handler(void*)`
- `mlc::xpc_dispatch_t::handle_xpc_call(void*) const`
- imported symbol `_mlc_model_compile`

and a service id string:

- `com.apple.mlcompiler.service.compiler`

This suggests a separate ML compiler XPC stack exists, but based on what is
visible so far it looks adjacent to the ANE runtime path rather than the main
ANE execution API itself.

## Most Actionable Next Targets

If we continue using only `ipsw-safe`, the most promising next binaries or
frameworks to inspect are:

- `ANECompiler.framework`
- `MIL.framework`
- `Espresso.framework`
- `ANECompilerService`
- `ANEStorageMaintainer`

That order is chosen to move from service shell toward actual compiler boundary.
