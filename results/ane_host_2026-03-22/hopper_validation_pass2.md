# Hopper Validation Pass 2

This pass focused on:

1. `_ANERequest` constructors and ivar layout
2. `_ANEModel` / `identifierSource` reconstruction
3. `_ANECompiler compileModelJIT:ok:error:` and the `ANECCompileOnline` branch

Targets:

- `AppleNeuralEngine.hop`
- `ANECompilerService.hop`

## 1. `_ANERequest` constructors and logical ivar layout

Validated procedures:

- `-[_ANERequest initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:]`
- `+[_ANERequest requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:]`
- `+[_ANERequest requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:]`
- `-[_ANERequest validate]`

## 1.1 Constructor behavior

The designated initializer is simple and direct:

- retain all incoming objects
- call `super init`
- store the retained values into fixed object slots

From the decompiled constructor, the logical layout is:

- `+0x08` -> `inputArray`
- `+0x10` -> `inputIndexArray`
- `+0x18` -> `outputArray`
- `+0x20` -> `outputIndexArray`
- `+0x28` -> `weightsBuffer`
- `+0x30` -> `procedureIndex`
- `+0x38` -> `transactionHandle`
- `+0x40` -> `sharedEvents`
- `+0x50` -> `perfStats`

This matches the accessors seen in validation and symbol output:

- `inputArray`
- `inputIndexArray`
- `outputArray`
- `outputIndexArray`
- `weightsBuffer`
- `procedureIndex`
- `transactionHandle`
- `sharedEvents`
- `perfStats`
- `perfStatsArray`

## 1.2 Constructor families

The class constructors are just convenience wrappers over the full initializer.

Examples:

- `requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:`
  - passes `weightsBuffer = nil`
  - passes `perfStats = nil`
  - passes `sharedEvents = nil`
  - passes `transactionHandle = nil`

- `requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:`
  - forwards everything directly

## 1.3 What this confirms

This is now directly confirmed:

- request packing is **not** just positional tensors
- the runtime has a first-class concept of:
  - symbol-index arrays
  - optional per-request weight buffer
  - optional perf stats object
  - optional shared events
  - optional transaction handle
  - procedure-scoped dispatch

For `rustane`, this is the strongest evidence so far that Apple’s execution
model is richer than the simple `run_cached_direct(&[inputs], &[outputs])`
interface currently exposed to the engine.

## 2. `_ANEModel` and `identifierSource`

Validated procedures:

- `-[_ANEModel initWithModelAtURL:sourceURL:UUID:key:identifierSource:cacheURLIdentifier:modelAttributes:standardizeURL:string_id:generateNewStringId:mpsConstants:]`
- `+[_ANEModel modelAtURLWithSourceURL:sourceURL:key:identifierSource:cacheURLIdentifier:]`
- `+[_ANEModel modelWithCacheURLIdentifier:]`

## 2.1 Directly confirmed rules

From the decompiled initializer:

- if `cacheURLIdentifier` contains `..`, initialization fails
- if `identifierSource == 3`, `cacheURLIdentifier` must be non-nil
- if `identifierSource == 2`, `sourceURL` must be non-nil
- `standardizeURL` controls whether URLs are canonicalized via
  `URLByStandardizingPath`
- the object stores:
  - model URL
  - source URL
  - key
  - identifier source
  - cache URL identifier
  - model attributes
  - UUID
  - optional `mpsConstants`
  - `string_id`

## 2.2 Reconstructed enum semantics

This is still partly inferential, but much stronger now.

### `identifierSource == 3`

Strongly indicates:

- **cache-URL-identifier source mode**

Why:

- the constructor explicitly rejects `identifierSource == 3` when
  `cacheURLIdentifier == nil`
- prior string evidence named this mode:
  - `_ANEModelCacheURLIdentifierSource`

### `identifierSource == 2`

Strongly indicates:

- **source-URL-and-key source mode**

Why:

- the constructor explicitly rejects `identifierSource == 2` when `sourceURL ==
  nil`
- prior string evidence named this mode:
  - `_ANEModelIdentifierSourceURLAndKey`

### Other values

Not fully reconstructed yet.

Most likely:

- one mode is the ordinary model URL + key path
- one mode may be a pure model-identifier / cache-only recovery mode

But Hopper validation in this pass is not enough to name those values with high
confidence.

## 2.3 Why this matters

This confirms the cache-identity model is real and typeful.

`rustane` cannot assume there is only one notion of model identity. Apple’s
runtime clearly distinguishes at least:

- a source-URL-based identity mode
- a cache-URL-identifier-based identity mode

That makes the compile-cache reuse hypothesis much stronger.

## 3. `compileModelJIT:ok:error:` and `ANECCompileOnline`

Validated procedures:

- `+[_ANECompiler compileModelJIT:ok:error:]`
- `+[_ANECompiler createJITNetworkFromModelAtPath:modelFilename:aotModelAtPath:aotModelFilename:]`
- `+[_ANECompiler createNetworkFromModelAtPath:modelFilename:]`
- `+[_ANECompiler compileModel:options:ok:error:]`

## 3.1 JIT network construction

`createJITNetworkFromModelAtPath:modelFilename:aotModelAtPath:aotModelFilename:`
builds an array containing one dictionary with:

- `NetworkSourceFileName`
- `NetworkSourcePath`
- `NetworkJITShapesName`
- `NetworkJITShapesPath`

By contrast, the non-JIT helper `createNetworkFromModelAtPath:modelFilename:`
only builds:

- `NetworkSourceFileName`
- `NetworkSourcePath`

So JIT compilation really is a separate compiler input shape, not just a flag on
normal compile.

## 3.2 `compileModelJIT:ok:error:`

Directly confirmed behavior:

- collects JIT-specific model pieces from the compiler request struct
- logs a JIT-specific signpost:
  - `_ANEF_JIT_ANEC_COMPILE`
- constructs the `InputNetworks` payload via `createJITNetwork...`
- builds compiler dictionary keys similar to normal compile:
  - `InputNetworks`
  - `OutputFilePath`
  - `OutputFileName`
  - `TargetArchitecture`
  - optional `OptionsFilePath`
  - `BSSLimit`
- calls `ANECCompileJIT()`

On success it follows the same general post-compile pattern as the normal path:

- remove existing output if needed
- move compiled temp artifact into place
- mark artifact + parent purgeable on APFS
- propagate file dates
- save source model path into the output directory
- optionally create retain files depending on options

On failure:

- writes `ok = false`
- formats a compiler-domain error
- reports `ModelFailsToCompileANECIR`

## 3.3 `ANECCompileOnline` call path

This is not part of the JIT path. It lives in normal
`+[_ANECompiler compileModel:options:ok:error:]`.

Directly confirmed branch:

- if model filename is exactly `model.llir.bundle`
  - log `Calling ANECCompileOnline...`
  - call `ANECCompileOnline()`
- else
  - call `ANECCompile()`

So the online compiler path is specifically tied to the `LLIR bundle` model
format, not to JIT generally.

## 3.4 Why this matters

This gives `rustane` three concrete format-path facts:

1. **JIT is separate**
   - separate input-network dictionary shape
   - separate low-level call: `ANECCompileJIT`

2. **LLIR bundle is special**
   - routed to `ANECCompileOnline`

3. **Normal non-JIT compile is different again**
   - routed to `ANECCompile`

So Apple’s compiler stack is at least a three-way split:

- normal compile
- JIT compile
- online compile for LLIR bundle

## 4. Immediate `rustane` Implications

### Highest-confidence

- `rustane` should treat descriptor stability as important if it wants cache
  reuse
- `rustane` should not assume all compile paths in Apple’s runtime are one thing
  with a flag; JIT and LLIR-online are distinct
- request packing in Apple’s runtime is objectively richer than the engine’s
  current abstraction

### Still open

- exact semantics of `identifierSource` values other than `2` and `3`
- whether `rustane` can exploit any of these cache or request-structure concepts
  through the currently used `ane` crate layer without invasive changes
