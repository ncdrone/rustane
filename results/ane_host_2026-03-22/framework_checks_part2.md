# Framework Checks Part 2: `AppleNeuralEngine`, `ANEServices`, `MLCompilerServices`

These follow the same method as the earlier framework pass:

- `ipsw-safe dyld image`
- `ipsw-safe class-dump`
- `ipsw-safe plist`
- targeted cstring inspection where useful

## 1. `AppleNeuralEngine.framework`

### What it gave us

This is the most relevant framework checked so far.

From `dyld image`:

- image is ObjC-heavy
- it depends on:
  - `MIL.framework`
  - `ANECompiler.framework`
  - `ANEServices.framework`
  - `IOSurface`
  - `IOKit`
  - `libsandbox`
  - `Security`
  - `Foundation`

From `class-dump`, it exposes the core private runtime surface:

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

### Why it matters

This framework is the strongest host-side confirmation so far that `rustane`
really is orbiting the same private interface family it claims to use.

The most important confirmations are:

- there is a real `_ANEClient` class,
- there is a real `_ANEInMemoryModel` class,
- model descriptors can be created from MIL text,
- compile/load/evaluate/purge/chaining are all first-class framework methods,
- IOSurface-backed buffer objects are part of the normal contract.

### Most relevant methods

`_ANEClient`

- `compileModel:options:qos:error:`
- `loadModel:options:qos:error:`
- `loadModelNewInstance:options:modelInstParams:qos:error:`
- `unloadModel:options:qos:error:`
- `evaluateWithModel:options:request:qos:error:`
- `evaluateRealTimeWithModel:options:request:error:`
- `mapIOSurfacesWithModel:request:cacheInference:error:`
- `unmapIOSurfacesWithModel:request:`
- `prepareChainingWithModel:options:chainingReq:qos:error:`
- `enqueueSetsWithModel:outputSet:options:qos:error:`
- `buffersReadyWithModel:inputBuffers:options:qos:error:`

`_ANEInMemoryModel`

- `compileWithQoS:options:error:`
- `compiledModelExists`
- `loadWithQoS:options:error:`
- `unloadWithQoS:error:`
- `evaluateWithQoS:options:request:error:`
- `mapIOSurfacesWithRequest:cacheInference:error:`
- `unmapIOSurfacesWithRequest:`
- `purgeCompiledModel`
- `isMILModel`

`_ANEInMemoryModelDescriptor`

- `modelWithMILText:weights:optionsPlist:`
- `modelWithNetworkDescription:weights:optionsPlist:`

### Practical interpretation

For `rustane`, this means the current project direction is not just “inspired by”
Apple private APIs. It is meaningfully aligned with actual framework objects and
verbs present on this system.

## 2. `ANEServices.framework`

### What it gave us

This framework is much thinner than `AppleNeuralEngine.framework`.

From `dyld image`:

- ObjC-enabled image
- depends mostly on:
  - `IOSurface`
  - `IOKit`
  - `Foundation`
  - `CoreFoundation`

From `class-dump`:

- `ANEServicesLog`
  - `verbose`
  - `test`
  - `services`
  - `handle`

### Practical interpretation

- At least from the surface visible through `class-dump`, this looks more like a
  support/logging framework than the core model compiler/runtime interface.
- It is relevant, but lower priority than `AppleNeuralEngine.framework`,
  `ANECompiler.framework`, and `Espresso.framework`.

## 3. `MLCompilerServices.framework`

### Framework image

From `dyld image`:

- image is present in cache
- depends on:
  - `MLCompilerRuntime.framework`
  - `Accelerate`
  - `IOSurface`
  - `Foundation`
  - `SoftLinking`
- `class-dump` found no ObjC data in the framework image itself

### XPC service

The framework ships:

- `MLCompilerOSXPC.xpc`

Info plist:

- bundle id: `com.apple.mlcompiler.services.compiler`

From `macho info --arch arm64e`:

- executable links:
  - `/AppleInternal/Library/Frameworks/MLCompilerOS.framework/Versions/A/MLCompilerOS`
  - `libc++.1`
  - `libSystem.B`

From `class-dump --arch arm64e`:

- no ObjC info

From a focused cstring pass:

- `com.apple.mlcompiler.service.compiler`

### Practical interpretation

- `MLCompilerServices.framework` appears adjacent to the ANE stack, but not the
  same thing.
- It may represent a broader host ML compilation service layer that Espresso can
  weak-link against.
- Without deeper tooling, it does not currently look as directly actionable for
  `rustane` as `AppleNeuralEngine.framework`.

## Priority After This Pass

Based on `ipsw-safe` alone, the current order of value is:

1. `AppleNeuralEngine.framework`
2. `ANECompilerService.xpc`
3. `ANEStorageMaintainer.xpc`
4. `Espresso.framework`
5. `ANECompiler.framework`
6. `MIL.framework`
7. `MLCompilerServices.framework`
8. `ANEServices.framework`

The reason is simple:

- `AppleNeuralEngine.framework` exposes the most directly relevant private API
  surface,
- `ANECompilerService` and `StorageMaintainer` explain host compile/cache flow,
- `Espresso` explains the higher-level frontend layer,
- the others are either lower-level but opaque or adjacent but less obviously
  useful with `ipsw-safe` alone.
