# Framework Checks: `ANECompiler`, `MIL`, `Espresso`

These checks were run with `ipsw-safe` against the live arm64e dyld shared cache:

- `/System/Volumes/Preboot/Cryptexes/OS/System/Library/dyld/dyld_shared_cache_arm64e`

Target images:

- `/System/Library/PrivateFrameworks/ANECompiler.framework/Versions/A/ANECompiler`
- `/System/Library/PrivateFrameworks/MIL.framework/Versions/A/MIL`
- `/System/Library/PrivateFrameworks/Espresso.framework/Versions/A/Espresso`

## 1. `ANECompiler.framework`

### What `ipsw-safe` could see

From `dyld image`:

- image is present in the dyld cache
- VM size: `0x3fa98000`
- dependents:
  - `Accelerate`
  - `MIL.framework`
  - `CoreFoundation`
  - `CoreAnalytics`
  - `libc++.1`
  - `libSystem.B`

From `class-dump`:

- `no ObjC data found in dylib 'ANECompiler'`

### Interpretation

- This strongly suggests `ANECompiler.framework` is primarily a C / C++ style
  library rather than an ObjC framework with a rich reflected runtime surface.
- That matches the `ANECompilerService` imports:
  - `_ANECCompile`
  - `_ANECCompileJIT`
  - `_ANECCompileOnline`
  - `_ANECCreateModelDictionary`

### Relevance

- This is probably the lowest host-side framework boundary that still matters to
  `rustane`.
- `ipsw-safe` can confirm image presence and dependency shape, but without Hopper
  or disassembly we should not expect much more than symbol- and string-level
  clues from this framework.

## 2. `MIL.framework`

### What `ipsw-safe` could see

From `dyld image`:

- image is present in the dyld cache
- VM size: `0x73814000`
- dependents:
  - `libc++.1`
  - `libSystem.B`

From `class-dump`:

- `no ObjC data found in dylib 'MIL'`

### Interpretation

- `MIL.framework` also looks more like a C++ / core-library layer than an ObjC
  API surface.
- This fits the current `rustane` understanding that MIL is an intermediate
  representation and compiler/runtime substrate, not just a front-end framework.

### Relevance

- The fact that Apple’s own stack still ships `MIL.framework` supports
  `rustane`’s choice to work with MIL directly rather than treat it as an
  accidental implementation detail.

## 3. `Espresso.framework`

### What `ipsw-safe` could see

From `dyld image`:

- image is present in the dyld cache
- VM size: `0x866a4000`
- the image is ObjC-heavy
- weak dependencies include:
  - `ANECompiler.framework`
  - `AppleNeuralEngine.framework`
  - `MIL.framework`
  - `MLCompilerServices.framework`
  - `ANEServices.framework`
- regular dependencies include:
  - `Metal`
  - `MetalPerformanceShaders`
  - `MetalPerformanceShadersGraph`
  - `IOSurface`
  - `ImageIO`
  - `CoreGraphics`
  - `Foundation`
  - `CoreAnalytics`

From `class-dump`, sample surface includes:

- protocols:
  - `ETDataProvider`
  - `ETDataSource`
  - `ETTaskContext`
  - `ExternalDetectedObject`
  - `MPSCNNConvolutionDataSource`
- classes:
  - `ETDataPoint`
  - `ETDataPointDictionary`
  - `ETDataSourceBlobF4`
  - `ETDataSourceBuf`
  - `ETDataSourceFromFolderData`
  - `ETDataSourceWithCache`
  - `ETDataSourceWithExtractor`
  - `ETDataTensor`
  - `ETImageDescriptorExtractor`
  - `ETImageFolderDataProvider`
  - `ETImagePreprocessParams`
  - `ETImagePreprocessor`
  - `ETLayerInitializationParameters`
  - `ETLossConfig`
  - `ETLossDefinition`
  - `ETModelDef`
  - `ETModelDefLeNet`
  - `ETModelDefMLP`
  - `ETModelDefinition`

Useful methods:

- `-[ETTaskContext setTensorNamed:withValue:error:]`
- `-[ETTaskContext getTensorNamed:]`
- `-[ETTaskContext doInferenceOnData:error:]`
- `-[ETTaskContext saveNetwork:inplace:error:]`
- `-[ETModelDefinition initWithInferenceNetworkPath:inferenceInputs:inferenceOutputs:error:]`
- `-[ETModelDefinition initWithInferenceNetworkPath:error:]`

### Interpretation

- `Espresso.framework` is much richer at the ObjC layer than `ANECompiler` or
  `MIL`.
- It appears to provide higher-level data, tensor, preprocessing, model, and
  task abstractions.
- Because it weak-links `ANECompiler`, `AppleNeuralEngine`, and `MIL`, it looks
  like a high-level planning / model-management layer that can route into the
  ANE path when available.

### Relevance

- This reinforces the earlier finding from `ANECompilerService` that Apple uses
  Espresso as a meaningful frontend / translation layer in the host compiler
  stack.
- It may also explain some of the terminology seen in the compiler-service
  strings like `model.espresso.net` and `_ANEEspressoIRTranslator`.

## Overall Takeaway

For `rustane`, the three frameworks split into distinct roles:

- `ANECompiler.framework`
  - likely the low-level compile boundary
  - mostly opaque to `class-dump`
- `MIL.framework`
  - likely the core IR / runtime substrate
  - also mostly opaque to `class-dump`
- `Espresso.framework`
  - the highest-level, most inspectable host-side ML layer
  - clearly connected to ANE/MIL/compiler services

So if we keep mining with `ipsw-safe` alone, `Espresso.framework` is the best
place to extract more semantic structure, while `ANECompiler.framework` and
`MIL.framework` likely require a lower-level symbol/disassembly toolchain to go
much further.
