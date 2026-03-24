# Selector and Class Inventory

Source binaries:

- `ANECompilerService.xpc/Contents/MacOS/ANECompilerService`
- `ANEStorageMaintainer.xpc/Contents/MacOS/ANEStorageMaintainer`

Method source:

- `ipsw-safe class-dump`

## XPC Services

### `com.apple.ANECompilerService`

- Bundle executable: `ANECompilerService`
- XPC service type: `Application`
- Multiple instances: true

### `com.apple.private.ANEStorageMaintainer`

- Bundle executable: `ANEStorageMaintainer`
- XPC service type: `Application`

## Protocols

### Compiler service protocols

- `_ANECompilerServiceProtocol`
  - `compileModelAt:csIdentity:sandboxExtension:options:tempDirectory:cloneDirectory:outputURL:aotModelBinaryPath:withReply:`
- `_ANEMaintenanceProtocol`
  - `scheduleMaintenanceWithName:directoryPaths:`
- `_ANEStorageMaintainerProtocol`
  - `purgeDanglingModelsAt:withReply:`

### Common protocol glue

- `NSXPCListenerDelegate`
  - `listener:shouldAcceptNewConnection:`

## Compiler-side classes

### `_ANECompilerService`

- `initialize`
- `compileModelAt:csIdentity:sandboxExtension:options:tempDirectory:cloneDirectory:outputURL:aotModelBinaryPath:withReply:`

This appears to be the main XPC boundary between host clients and the private
ANE compiler stack.

### `_ANECompiler`

- `initialize`
- `createNetworkFromModelAtPath:modelFilename:`
- `createJITNetworkFromModelAtPath:modelFilename:aotModelAtPath:aotModelFilename:`
- `createErrorWithUnderlyingError:`
- `compileModelJIT:ok:error:`
- `compileModel:options:ok:error:`
- `createInMemoryConstants:`

This appears to be the direct wrapper around the lower-level `ANECompiler`
framework functions.

### `_ANECoreMLModelCompiler`

- `initialize`
- `createErrorWithString:`
- `pathsForModelURL:`
- `compileModelAt:csIdentity:key:optionsFilename:tempDirectory:outputURL:saveSourceModelPath:aotModelBinaryPath:isEncryptedModel:options:ok:error:`

This looks like the high-level CoreML-oriented compile entrypoint.

### `_ANEEspressoIRTranslator`

- `createErrorForPlan:status:`
- `translateModelAt:key:outputPath:isEncryptedModel:translationOptions:error:`
- `destroyEspresso:ctx:`

This strongly suggests Espresso is used to turn a model into a lower-level ANE
compiler input format before the final compile call.

### `_ANEMILCompiler`

- `compileModelAt:modelName:csIdentity:optionsFilename:outputURL:saveSourceURL:aotModelBinaryPath:isEncryptedModel:options:ok:error:`

### `_ANEMLIRCompiler`

- `compileModelAt:modelName:csIdentity:optionsFilename:outputURL:saveSourceURL:aotModelBinaryPath:isEncryptedModel:options:mpsConstants:ok:error:`

### `_ANECVAIRCompiler`

- `compileModelAt:csIdentity:plistFilename:optionsFilename:outputURL:saveSourceURL:aotModelBinaryPath:isEncryptedModel:options:ok:error:`

These classes suggest the compiler stack handles multiple frontends / IR flavors
under a shared service shell.

## Cache and storage classes

### `_ANEModelCacheManager`

Class methods:

- `initialize`
- `new`
- `isSystemModelPath:`
- `cachedSourceModelStoreNameFor:`
- `saveSourceModelPath:outputModelDirectory:`
- `cachedModelRetainNameFor:`
- `createModelCacheRetain:`
- `removeIfStaleBinary:forModelPath:`

Instance methods:

- `init`
- `initWithURL:createDirectory:`
- `initWithURL:`
- `URLForModel:bundleID:`
- `URLForModel:bundleID:useSourceURL:`
- `URLForModel:bundleID:aotCacheUrlIdentifier:`
- `URLForModel:bundleID:forAllSegments:`
- `URLForModel:bundleID:useSourceURL:aotCacheUrlIdentifier:`
- `URLForModel:bundleID:useSourceURL:forAllSegments:aotCacheUrlIdentifier:`
- `URLForModel:bundleID:forAllSegments:aotCacheUrlIdentifier:`
- `cacheURLIdentifierForModel:useSourceURL:withReply:`
- `getModelBinaryPathFromURLIdentifier:bundleID:`
- `cachedModelPathFor:csIdentity:useSourceURL:`
- `cachedModelPathFor:csIdentity:`
- `cachedModelAllSegmentsPathFor:csIdentity:`
- `cachedSourceModelStoreNameFor:csIdentity:`
- `cachedModelRetainNameFor:csIdentity:`
- `URLForBundleID:`
- `removeAllModelsForBundleID:`
- `filePathForModel:bundleID:`
- `getDiskSpaceItemizedByBundleIDAndPurge:`
- `getDiskSpaceForBundleID:`
- `shouldEnforceSizeLimits`
- `garbageCollectDanglingModels`
- `startDanglingModelGC`
- `scheduleMaintenanceWithName:directoryPaths:`
- `scanAllPartitionsForModel:csIdentity:expunge:`
- `cacheDir`

### `_ANEInMemoryModelCacheManager`

Class methods:

- `new`
- `removeFilesFromDirectory:notAccessedInSeconds:`
- `notRecentlyUsedSecondsThreshold`
- `removeStaleModelsAtPath:`

Instance methods:

- `init`
- `initWithURL:createDirectory:`
- `initWithURL:`
- `URLForBundleID:`
- `URLForModelHash:bundleID:`
- `cachedModelPathMatchingHash:csIdentity:`
- `removeAllModelsForBundleID:`
- `getDiskSpaceItemizedByBundleIDAndPurge:`
- `getDiskSpaceForBundleID:`
- `shouldEnforceSizeLimits`
- `removeStaleModels`
- `scheduleMaintenanceWithName:directoryPaths:`
- `cacheDir`

### `_ANEStorageHelper`

- `initialize`
- `removeDirectoryAtPath:`
- `removeShapesDirectoryAtPath:`
- `memoryMapModelAtPath:modelAttributes:`
- `memoryMapModelAtPath:isPrecompiled:modelAttributes:`
- `memoryMapWeightAtPath:`
- `setAccessTime:forModelFilePath:`
- `getAccessTimeForFilePath:`
- `updateAccessTimeForFilePath:`
- `removeFilePath:ifDate:olderThanSecond:`
- `garbageCollectDanglingModelsAtPath:`
- `uniqueFirstLevelSubdirectories:`
- `sizeOfDirectoryAtPath:recursionLevel:`
- `addSubdirectoryDetails:directoryPath:size:`
- `createModelCacheDictionary`
- `sizeOfModelCacheAtPath:purgeSubdirectories:`
- `mergeModelCacheStorageInformation:with:`
- `_markPurgeablePath:error:`
- `markPathAndDirectParentPurgeable:error:`
- `enableApfsPurging`

### `_ANEStorageMaintainer`

- `initialize`
- `purgeDanglingModelsAt:withReply:`

## Sandbox and scheduling classes

### `_ANESandboxingHelper`

- `initialize`
- `canAccessPathAt:methodName:error:`
- `sandboxExtensionPathForModelURL:`
- `issueSandboxExtensionForPath:error:`
- `issueSandboxExtensionForModel:error:`
- `consumeSandboxExtension:forModel:error:`
- `consumeSandboxExtension:forPath:error:`
- `releaseSandboxExtension:handle:`

### `_ANETask`

- `new`
- `taskWithName:period:handler:`
- `init`
- `initWithName:period:handler:`
- `periodSeconds`
- `name`
- `handler`
- `queue`
- `executionCriteria`

### `_ANETaskManager`

- `registerTask:`
- `unregisterTask:`

## Immediate Implications

- The host compiler interface is explicitly XPC-based and file-path-based.
- Sandbox extension handling is a normal part of compilation, not a rare edge
  case.
- Apple distinguishes multiple model representations and frontend compilers:
  CoreML, MIL, MLIR, CVAIR, ANECIR, LLIR bundle.
- There is a persistent cache / maintenance subsystem around compilation, not
  just a transient “compile and run” API.

## AppleNeuralEngine Framework Surface

The dyld-cached `AppleNeuralEngine.framework` is the closest match yet to the
API surface `rustane` claims to target.

### Daemon protocols

- `_ANEDaemonProtocol`
  - `compileModel:sandboxExtension:options:qos:withReply:`
  - `compiledModelExistsFor:withReply:`
  - `compiledModelExistsMatchingHash:withReply:`
  - `loadModel:sandboxExtension:options:qos:withReply:`
  - `loadModelNewInstance:options:modelInstParams:qos:withReply:`
  - `prepareChainingWithModel:options:chainingReq:qos:withReply:`
  - `purgeCompiledModel:withReply:`
  - `purgeCompiledModelMatchingHash:withReply:`
  - `reportTelemetryToPPS:playload:`
  - `unloadModel:options:qos:withReply:`
- `_ANEDaemonProtocol_Private`
  - `echo:withReply:`
  - `beginRealTimeTaskWithReply:`
  - `endRealTimeTaskWithReply:`

### Client-side entry points

#### `_ANEClient`

Class methods:

- `new`
- `initialize`
- `sharedConnection`
- `sharedPrivateConnection`

Instance methods:

- `compileModel:options:qos:error:`
- `compiledModelExistsFor:`
- `compiledModelExistsMatchingHash:`
- `loadModel:options:qos:error:`
- `loadModelNewInstance:options:modelInstParams:qos:error:`
- `unloadModel:options:qos:error:`
- `purgeCompiledModel:`
- `purgeCompiledModelMatchingHash:`
- `evaluateWithModel:options:request:qos:error:`
- `evaluateRealTimeWithModel:options:request:error:`
- `mapIOSurfacesWithModel:request:cacheInference:error:`
- `unmapIOSurfacesWithModel:request:`
- `prepareChainingWithModel:options:chainingReq:qos:error:`
- `enqueueSetsWithModel:outputSet:options:qos:error:`
- `buffersReadyWithModel:inputBuffers:options:qos:error:`
- `sessionHintWithModel:hint:options:report:error:`
- `beginRealTimeTask`
- `endRealTimeTask`

This is the most direct host-side evidence yet that `rustane`'s README is
describing a real private ANE interface family rather than a made-up wrapper
layer.

#### `_ANEDaemonConnection`

- `daemonConnection`
- `daemonConnectionRestricted`
- `userDaemonConnection`
- `compileModel:sandboxExtension:options:qos:withReply:`
- `loadModel:sandboxExtension:options:qos:withReply:`
- `loadModelNewInstance:options:modelInstParams:qos:withReply:`
- `prepareChainingWithModel:options:chainingReq:qos:withReply:`
- `purgeCompiledModel:withReply:`
- `purgeCompiledModelMatchingHash:withReply:`
- `unloadModel:options:qos:withReply:`
- `beginRealTimeTaskWithReply:`
- `endRealTimeTaskWithReply:`

### Model and descriptor objects

#### `_ANEInMemoryModel`

- `inMemoryModelWithDescriptor:`
- `compileWithQoS:options:error:`
- `compiledModelExists`
- `loadWithQoS:options:error:`
- `unloadWithQoS:error:`
- `evaluateWithQoS:options:request:error:`
- `mapIOSurfacesWithRequest:cacheInference:error:`
- `unmapIOSurfacesWithRequest:`
- `purgeCompiledModel`
- `compilerOptionsWithOptions:isCompiledModelCached:`
- `isMILModel`
- `localModelPath`
- `hexStringIdentifier`
- `string_id`
- `modelAttributes`
- `perfStatsMask`
- `intermediateBufferHandle`
- `programHandle`

#### `_ANEInMemoryModelDescriptor`

- `modelWithNetworkDescription:weights:optionsPlist:`
- `modelWithMILText:weights:optionsPlist:`
- `hexStringIdentifier`
- `networkTextHash`
- `optionsPlistHash`
- `weightsHash`
- `isMILModel`
- `networkText`
- `optionsPlist`
- `weights`

This is directly relevant to `rustane` because it confirms there are at least
two model-descriptor creation paths:

- generic network description
- explicit MIL text

### Buffer / request helpers

#### `_ANEIOSurfaceObject`

- `createIOSurfaceWithWidth:pixel_size:height:`
- `createIOSurfaceWithWidth:pixel_size:height:bytesPerElement:`
- `objectWithIOSurface:`
- `objectWithIOSurface:startOffset:`
- `objectWithIOSurfaceNoRetain:startOffset:`

#### `_ANEBuffer`

- `bufferWithIOSurfaceObject:symbolIndex:source:`
- `ioSurfaceObject`
- `symbolIndex`
- `source`

#### `_ANEInputBuffersReady`

- `inputBuffersWithProcedureIndex:inputBufferInfoIndex:inputFreeValue:executionDelay:`
- `procedureIndex`
- `inputBufferInfoIndex`
- `inputFreeValue`
- `executionDelay`

#### `_ANEChainingRequest`

- `chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:`
- `loopbackInputSymbolIndex`
- `loopbackOutputSymbolIndex`
- `procedureIndex`
- `signalEvents`
- `transactionHandle`
- `fwEnqueueDelay`
- `memoryPoolId`

These types strongly suggest the underlying runtime contract is:

- symbol-indexed IOSurface-backed buffers,
- optional multi-step chaining with loopback symbols,
- per-procedure execution and event signaling,
- optional real-time / queue-oriented dispatch behavior.

### Device and error helpers

#### `_ANEDeviceInfo`

- `hasANE`
- `numANEs`
- `numANECores`
- `aneArchitectureType`
- `aneSubType`
- `aneBoardType`
- `precompiledModelChecksDisabled`
- `isVirtualMachine`
- `productName`
- `buildVersion`

#### `_ANEErrors`

Includes error builders for:

- bad arguments
- entitlement failures
- file access / missing file
- invalid model / invalid key / invalid model instance
- missing code signing
- program creation / load / new-instance load failures
- program inference failures
- chaining prepare failures
- timeout
- virtualization host / kernel / data failures
- system-model purge not allowed

This error taxonomy is useful for interpreting future host logs or matching
Apple-side failure modes to `rustane` failures.
