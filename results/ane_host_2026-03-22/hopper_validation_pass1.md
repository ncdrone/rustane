# Hopper Validation Pass 1

This pass used Hopper MCP directly against the three open binaries:

- `AppleNeuralEngine.hop`
- `ANECompilerService.hop`
- `ANEStorageMaintainer.hop`

Priority was:

1. `AppleNeuralEngine`
2. `ANECompilerService`
3. `ANEStorageMaintainer`

The goal was to validate the highest-value hypotheses directly:

- compile-cache reuse
- request-packing assumptions
- chaining support
- service-layer behavior

## 1. `AppleNeuralEngine`

## 1.1 `_ANEInMemoryModelDescriptor` really is content-hash based

Validated procedure:

- `-[_ANEInMemoryModelDescriptor initWithNetworkText:weights:optionsPlist:isMILModel:]`

### Confirmed behavior

- requires non-nil `networkText`
- requires non-nil `weights`
- copies network text and stores it
- computes `networkTextHash` using `_ANEHashEncoding hexStringForData:`
- iterates weight keys in **sorted order**
- pulls per-key weight payloads and computes `weightsHash` using
  `_ANEHashEncoding hexStringForDataArray:`
- copies `optionsPlist`
- computes `optionsPlistHash` using `_ANEHashEncoding hexStringForData:`
- stores `isMILModel`

### Why this matters

This is direct confirmation that descriptor identity is content-based and that
weight ordering is normalized by sorting keys before hashing.

For `rustane`, this means:

- stable weight-key ordering matters
- stable network text matters
- stable options plist serialization matters

If compile-cache reuse exists at this layer, unstable descriptor construction
will defeat it.

## 1.2 `_ANERequest` is more structured than plain positional buffers

Validated procedure:

- `-[_ANERequest validate]`

### Confirmed behavior

The request validator explicitly checks:

- `inputArray` non-empty
- `outputArray` non-empty
- `inputIndexArray` non-empty
- `outputIndexArray` non-empty
- `inputArray.count == inputIndexArray.count`
- `outputArray.count == outputIndexArray.count`
- input and output buffer counts are bounded by `kANEMaxBuffers` (`0xff`)
- each input and output symbol index is `< 0xff`
- `procedureIndex < 0x81` (`kANEMaxProcedures = 128`)
- `perfStatsArray` stat types are validated against a small enum range
- `sharedEvents.signalEvents.count <= 0x40`
- `sharedEvents.waitEvents.count <= 0x40`

### Why this matters

This directly validates that Apple’s runtime request contract is:

- symbol-indexed
- procedure-scoped
- optionally perf-stats-aware
- optionally shared-event-aware

That is materially richer than the simpler `run_cached_direct(&[inputs], &[outputs])`
model `rustane` currently uses.

## 1.3 `_ANEChainingRequest` is real and strongly validated

Validated procedure:

- `-[_ANEChainingRequest validate]`

### Confirmed behavior

The chaining validator explicitly checks:

- `inputBuffer` non-empty
- `outputSets` non-empty
- input buffer count `< 0xff`
- every input buffer symbol index `< 0xff`
- output set count `< 0x0c` (strict upper bound around 12)
- each output set must have a non-empty output buffer
- output buffer counts are bounded
- every output symbol index `< 0xff`
- `loopbackInputSymbolIndex.count == loopbackOutputSymbolIndex.count`
- loopback counts are bounded
- `procedureIndex < 0x81`
- `signalEvents.count < 0x101`

### Why this matters

This confirms chaining is not speculative. It is a first-class runtime feature
with:

- loopback input/output indices
- per-procedure semantics
- signal events
- transaction handles
- memory-pool metadata

`rustane` does not currently model any of this.

## 1.4 `_ANEClient doPrepareChaining...` does real work and is not supported on the virtual path

Validated procedure:

- `-[_ANEClient doPrepareChainingWithModel:options:chainingReq:qos:error:]`

### Confirmed behavior

- if `virtualClient` exists, the method logs and returns failure
- if running as a VM without the right host support, it returns a `hostTooOld`
  error
- otherwise it:
  - validates the chaining request
  - finds the connection used for loading the model, or falls back to `conn`
  - selects a queue based on QoS
  - executes the real work inside `dispatch_sync(...)`
  - propagates errors through the passed error pointer

### Why this matters

This confirms:

- chaining is a daemon/local-runtime feature, not just a virtual-client feature
- chaining depends on the connection associated with model loading
- chaining is bound to QoS/queue selection

So if `rustane` ever wants to target chaining, it likely needs to think in terms
of model lifecycle and queue selection, not just one-off kernel dispatches.

## 1.5 `_ANEClient compileModel...` has three distinct branches

Validated procedure:

- `-[_ANEClient compileModel:options:qos:error:]`

### Confirmed behavior

Branch 1:

- if `virtualClient` exists, delegate compile to the virtual client

Branch 2:

- if running inside a VM without the needed host support, return `hostTooOld`

Branch 3:

- normal local path:
  - select a queue from `priorityQ` using QoS
  - do the actual compile work inside `dispatch_sync(...)`
  - propagate errors through block-owned state

The procedure also logs model perf stats mask around compile scheduling.

### Why this matters

This confirms the local compile path is queue-based and synchronously dispatched
onto a QoS-derived queue, not just a naked direct compiler call.

## 1.6 `_ANEModel` identity rules are stricter than expected

Validated procedure:

- `-[_ANEModel initWithModelAtURL:sourceURL:UUID:key:identifierSource:cacheURLIdentifier:modelAttributes:standardizeURL:string_id:generateNewStringId:mpsConstants:]`

### Confirmed behavior

- rejects `cacheURLIdentifier` strings containing `..`
- if `identifierSource == 3`, requires non-nil `cacheURLIdentifier`
- if `identifierSource == 2`, requires non-nil `sourceURL`
- optionally standardizes URLs before storing them
- stores:
  - model URL
  - source URL
  - key
  - identifier source
  - cache URL identifier
  - model attributes
  - UUID / constants
- if `generateNewStringId` is set:
  - derives `string_id` from the model path via `kdebug_trace_string(...)`
- otherwise:
  - uses the caller-provided `string_id`

### Why this matters

This confirms model identity is not only descriptor-hash-based. There is also a
distinct `_ANEModel` layer with:

- source URL semantics
- identifier-source semantics
- cache URL identifier validation
- runtime-visible `string_id`

That strongly supports the earlier hypothesis that compile reuse in `rustane`
will depend on more than just “same MIL text”.

## 2. `ANECompilerService`

## 2.1 The service method is a queueing shell around lower-level compiler logic

Validated procedure:

- `-[_ANECompilerService compileModelAt:csIdentity:sandboxExtension:options:tempDirectory:cloneDirectory:outputURL:aotModelBinaryPath:withReply:]`

### Confirmed behavior

- retains all input arguments
- creates an autorelease pool
- emits signposts / kdebug trace markers using `model.string_id`
- dispatches the real work synchronously on a compiler queue
- returns through the reply block after the queued work completes

### Why this matters

This confirms the service itself is mostly orchestration:

- signposting
- queueing
- lifetime handling
- reply propagation

The real compile decisions happen below it.

## 2.2 `+[_ANECompiler compileModel:options:ok:error:]` really selects between JIT and normal compile

Validated procedure:

- `+[_ANECompiler compileModel:options:ok:error:]`

### Confirmed behavior

- if the model is flagged as JIT, it routes to `compileModelJIT:ok:error:`
- otherwise it:
  - creates a network from model path + model filename
  - builds a compiler dictionary with keys like:
    - `InputNetworks`
    - `OutputFileName`
    - `OutputFilePath`
    - `TargetArchitecture`
    - `OptionsFilePath`
    - `mpsConstants`
    - `BSSLimit`
  - calls:
    - `ANECCompileOnline()` when model filename is `model.llir.bundle`
    - otherwise `ANECCompile()`
- after success it:
  - moves the compiled temp output into place
  - tries to mark the output purgeable on APFS
  - propagates source file dates to the compiled artifact
  - saves the source model path into the output directory
  - optionally creates a retain marker if `kANEFRetainModelsWithoutSourceURLKey`
    is enabled

### Why this matters

This directly confirms several things we only suspected from strings:

- JIT is a real separate compile path
- `LLIR bundle` is special-cased to `ANECCompileOnline`
- output artifacts are intentionally treated as purgeable cache items
- source-model preservation and retain files are part of compile output handling

That is highly relevant to `rustane`’s compile-cache hypotheses.

## 2.3 `+[_ANEEspressoIRTranslator translateModelAt:...]` really builds and dumps Espresso IR

Validated procedure:

- `+[_ANEEspressoIRTranslator translateModelAt:key:outputPath:isEncryptedModel:translationOptions:error:]`

### Confirmed behavior

- creates an Espresso context
- iterates translation option keys and applies them as integer options
- creates an Espresso plan
- adds a network to the plan
- sets compiler metadata key(s)
- builds the plan
- dumps IR to the requested output path

### Why this matters

This is direct confirmation that Espresso is not just loosely adjacent to the
ANE path. It is actively used as a translation/planning layer.

## 2.4 `_ANEModelCacheManager cacheURLIdentifierForModel:useSourceURL:withReply:`

Validated procedure:

- `-[_ANEModelCacheManager cacheURLIdentifierForModel:useSourceURL:withReply:]`

### Confirmed behavior

- chooses either:
  - `modelURL`, or
  - `sourceURL`
  based on `useSourceURL` and `identifierSource`
- gets the chosen path as a string
- computes `hexStringForString(path)`
- computes `hexStringForString(model.key)`
- replies with both hex strings

### Why this matters

This is one of the strongest findings in the whole Hopper pass.

It means cache URL identification is not magical:

- one component is the selected path string
- one component is the model key string

That gives `rustane` a much clearer starting point for compile-cache reuse
experiments.

## 3. `ANEStorageMaintainer`

## 3.1 The maintainer is a thin wrapper over storage GC

Validated procedure:

- `-[_ANEStorageMaintainer purgeDanglingModelsAt:withReply:]`

### Confirmed behavior

- logs START
- calls `_ANEStorageHelper garbageCollectDanglingModelsAtPath:`
- replies with the boolean result
- logs END

### Why this matters

This confirms the real maintenance logic lives in `_ANEStorageHelper`, not in
the XPC wrapper.

## 3.2 `_ANEStorageHelper garbageCollectDanglingModelsAtPath:` matches the cache-retain hypothesis

Validated procedure:

- `+[_ANEStorageHelper garbageCollectDanglingModelsAtPath:]`

### Confirmed behavior

- enumerates model-cache directories under the supplied path
- looks specifically for entries whose last path component matches
  `modelBinaryName`
- finds the corresponding `modelSourceStoreName`
- reads the stored source-path string
- if the source path still exists, the compiled cache stays
- if the source path is missing:
  - checks for a retain marker (`modelCacheRetainName`)
  - if retain marker exists, checks access time of the compiled binary
  - keeps the cache if access time is recent (threshold is `604800` seconds,
    i.e. 7 days)
  - otherwise removes the whole model directory
- logs attempted and removed counts

### Why this matters

This directly confirms the earlier compile-cache hypothesis:

- Apple keeps compiled caches alive if the source still exists
- if source is gone, retain markers + recent access can still preserve the cache
- otherwise stale compiled outputs are deleted

That is the clearest cache-lifecycle evidence we have so far.

## 3.3 `+[_ANEStorageHelper sizeOfModelCacheAtPath:purgeSubdirectories:]`

Validated procedure:

- `+[_ANEStorageHelper sizeOfModelCacheAtPath:purgeSubdirectories:]`

### Confirmed behavior

- builds a cache-detail dictionary
- computes directory sizes recursively
- tracks first-level subdirectory details
- if `purgeSubdirectories` is true, it actually removes subdirectories while
  accumulating the result

### Why this matters

This confirms there is an official size-accounting and optional purge path for
compiled model caches.

## 3.4 `+[_ANEStorageHelper markPathAndDirectParentPurgeable:error:]`

Validated procedure:

- `+[_ANEStorageHelper markPathAndDirectParentPurgeable:error:]`

### Confirmed behavior

- marks the file/path purgeable
- then marks its direct parent purgeable

### Why this matters

This confirms Apple intentionally makes both the artifact and its container
eligible for APFS purge behavior.

## 3.5 `+[_ANEStorageHelper removeFilePath:ifDate:olderThanSecond:]`

Validated procedure:

- `+[_ANEStorageHelper removeFilePath:ifDate:olderThanSecond:]`

### Confirmed behavior

- compares file date against the threshold date
- removes the file when the date is older than or equal to the threshold
- logs failures if deletion fails

### Why this matters

This rounds out the cache-age eviction story with direct code, not just strings.

## Implications for `rustane`

These Hopper-validated findings sharpen the earlier memo:

### Highest-confidence opportunities

1. **Compile-cache reuse**
   - strongly worth targeted experimentation now
   - we now know cache identifiers depend on path and key, and descriptor hashes
     depend on network text, weights, and options plist

2. **Descriptor stability**
   - likely prerequisite for meaningful reuse
   - especially weight ordering and options serialization

3. **Request-packing understanding**
   - symbol-indexed requests are confirmed
   - worth using as a guide for future optimization work

### Still probably second-wave

4. **Chaining**
   - definitely real
   - but still likely lower priority than cache/identity experiments unless
     dispatch orchestration becomes the dominant bottleneck
