# Hopper Validation Pass 3

This pass focused on:

1. reconstructing the remaining `identifierSource` enum semantics
2. inspecting `_ANEModelCacheManager` beyond `cacheURLIdentifierForModel`
3. inspecting `_ANERequest` / `_ANEChainingRequest` accessors and setters for
   additional fields

Targets:

- `AppleNeuralEngine.hop`
- `ANECompilerService.hop`

## 1. `identifierSource` enum reconstruction

This pass moved the enum story further, but not all the way to full certainty.

## 1.1 Directly confirmed values

From decompiled `_ANEModel` factory methods:

### `identifierSource == 1`

Directly used by:

- `+[_ANEModel modelAtURL:key:modelAttributes:]`
- `+[_ANEModel modelAtURL:key:mpsConstants:]`
- `+[_ANEModel modelAtURLWithSourceURL:sourceURL:key:cacheURLIdentifier:]`

Interpretation:

- this is the **default normal model identity mode**
- it works even when a `sourceURL` is present
- it does **not** require `cacheURLIdentifier`

So this appears to be the “normal model URL + key” style identity, with
`sourceURL` acting as auxiliary metadata when present.

### `identifierSource == 3`

Directly used by:

- `+[_ANEModel modelAtURLWithCacheURLIdentifier:key:cacheURLIdentifier:]`
- `+[_ANEModel modelWithCacheURLIdentifier:]`

And directly validated by the designated initializer:

- if `identifierSource == 3` and `cacheURLIdentifier == nil`, initialization
  fails

Interpretation:

- this is the **cache-URL-identifier source mode**

## 1.2 Strongly inferred value

### `identifierSource == 2`

Not directly observed in a small factory method during this pass, but strongly
supported by two independent code paths:

1. designated initializer:
   - if `identifierSource == 2` and `sourceURL == nil`, initialization fails

2. `_ANEModelCacheManager cacheURLIdentifierForModel:useSourceURL:withReply:`
   - if `useSourceURL == false` and `model.identifierSource != 2`, it hashes
     `modelURL`
   - otherwise it hashes `sourceURL`

Interpretation:

- `identifierSource == 2` is very likely the
  **source-URL-and-key identity mode**

This matches the earlier runtime string:

- `identifierSource is _ANEModelIdentifierSourceURLAndKey but sourceURL is nil`

## 1.3 Remaining uncertainty

What is still **not** directly confirmed:

- whether there are any additional enum values beyond `1`, `2`, and `3`
- the exact symbolic names Apple uses for value `1`
- whether value `0` exists in production code or is just unused/invalid

So the best current reconstruction is:

- `1` = default model URL / key identity mode
- `2` = source URL + key identity mode
- `3` = cache URL identifier identity mode

with value `2` still marked as **strong inference**, not fully direct proof.

## 2. `_ANEModelCacheManager` deeper inspection

Validated procedures:

- `-[_ANEModelCacheManager URLForModel:bundleID:useSourceURL:forAllSegments:aotCacheUrlIdentifier:]`
- `-[_ANEModelCacheManager cachedModelPathFor:csIdentity:useSourceURL:]`
- `-[_ANEModelCacheManager getModelBinaryPathFromURLIdentifier:bundleID:]`
- `-[_ANEModelCacheManager cachedSourceModelStoreNameFor:csIdentity:]`
- `-[_ANEModelCacheManager cachedModelRetainNameFor:csIdentity:]`
- `+[_ANEModelCacheManager removeIfStaleBinary:forModelPath:]`

## 2.1 Cache root selection

`URLForModel:bundleID:useSourceURL:forAllSegments:aotCacheUrlIdentifier:`
directly confirms:

- it checks the current XPC connection for a sharing entitlement
- if the client has the sharing entitlement **or** the model is a system model,
  it uses `systemModelsCacheDirectory`
- otherwise it uses the caller’s bundle-ID directory

This is stronger than the earlier “there is a cache layer” conclusion. It shows
the cache root itself is policy-controlled.

## 2.2 Cache path construction

The same method shows two path-building modes.

### Mode A: explicit `cacheURLIdentifier` already exists on the model

- take `model.getCacheURLIdentifier`
- replace `_` with `/`
- append that path to the chosen bundle/system cache root

### Mode B: no explicit cache URL identifier on the model

- call `cacheURLIdentifierForModel:useSourceURL:withReply:`
- that produces two hex-string components:
  - a path-derived hash
  - a key-derived hash

Then:

- if `aotCacheUrlIdentifier` is present:
  - append transformed AOT cache ID
  - append `shapes`
  - append path-derived hash
- otherwise:
  - append path-derived hash directly

Finally:

- if `forAllSegments == true`
  - return that directory
- else
  - append the key-derived hash as the final component

### Practical interpretation

This strongly suggests a cache hierarchy like:

- `root / pathHash / keyHash`

or, for AOT/shape variants:

- `root / aotCacheId / shapes / pathHash / keyHash`

That is far more concrete than we had before.

## 2.3 `cachedModelPathFor`

`cachedModelPathFor:csIdentity:useSourceURL:` is thin:

- calls `URLForModel:bundleID:useSourceURL:`
- then appends `modelBinaryName`

So the real cache semantics live in `URLForModel...`, and the actual compiled
artifact path is just the resolved cache directory plus the standard model-binary
leaf name.

## 2.4 `getModelBinaryPathFromURLIdentifier`

This method:

- takes a cache URL identifier
- replaces `_` with `/`
- appends it to the bundle-ID cache root
- returns the resulting path

This confirms cache identifiers are not just opaque comparison keys; they are
used as reversible path encodings.

## 2.5 `cachedSourceModelStoreNameFor` and `cachedModelRetainNameFor`

These methods:

- resolve the model’s cache directory via `URLForModel...`
- then call class helpers to derive the sidecar file names from that path

This reinforces the earlier conclusion that:

- source-store files and retain files are tightly coupled to cache-directory
  identity, not just global temp files.

## 2.6 `removeIfStaleBinary:forModelPath:`

This method directly confirms:

- compare the compiled binary’s creation/modification dates with the model
  source path’s dates
- if the compiled binary is not newer, treat it as stale
- if source-path attributes cannot be read, log and force removal
- stale removal is done by deleting the binary/directory path

This matters because it shows the cache is not only content-addressed. It is also
guarded by ordinary filesystem freshness checks.

## 3. `_ANERequest` / `_ANEChainingRequest` extra fields

## 3.1 `_ANERequest` accessors resolve the logical layout

Directly confirmed accessors:

- `inputArray` -> slot `0x08`
- `inputIndexArray` -> slot `0x10`
- `outputArray` -> slot `0x18`
- `outputIndexArray` -> slot `0x20`
- `weightsBuffer` -> slot `0x28`
- `sharedEvents` -> slot `0x30`
- `transactionHandle` -> slot `0x38`
- `procedureIndex` -> slot `0x40`
- `perfStats` -> slot `0x48`
- `perfStatsArray` -> slot `0x50`
- `completionHandler` -> slot `0x58`

### New field confirmed

The biggest new addition is:

- `completionHandler`

which was not part of the earlier simplified layout summary.

## 3.2 Constructor/layout caveat

There is one unresolved detail:

- the designated initializer stores the `x7` argument into slot `0x50`
- but the accessor names indicate:
  - `0x48` = `perfStats`
  - `0x50` = `perfStatsArray`

So one of these is true:

1. Hopper’s recovered argument label `perfStats` on the constructor is slightly
   wrong and that argument is really the perf-stats array
2. the initializer populates `perfStatsArray` directly and leaves `perfStats`
   unset for later
3. there is some aliasing/translation step we have not yet traced

The layout itself is clear. The constructor naming around `perfStats` versus
`perfStatsArray` is the remaining ambiguity.

## 3.3 `_ANERequest initWithVirtualModel:`

This method is minimal:

- just `super init`

So it does **not** reveal a second hidden layout family by itself.

That suggests the virtual path probably populates the same request object shape
through other setters/builders rather than through a distinct “virtual request”
layout.

## 3.4 `_ANERequest ioSurfacesCount`

This helper confirms:

- count input index array
- count output index array
- add one more if `weightsBuffer` is present

So the runtime treats weights as a distinct IOSurface-bearing participant in the
request.

## 3.5 `_ANEChainingRequest`

No new hidden fields were discovered beyond the first Hopper pass. The accessors
still line up with:

- input buffer
- output sets
- loopback input symbol index
- loopback output symbol index
- signal events
- transaction handle
- procedure index
- firmware enqueue delay
- memory pool ID

## 4. Bottom Line

This pass sharpened three things:

1. **Cache-manager behavior is now quite concrete**
   - root selection
   - path hashing
   - AOT/shape subdirectory behavior
   - sidecar naming
   - stale-binary removal rules

2. **The request object has one more confirmed field than before**
   - `completionHandler`

3. **The remaining uncertainty is now narrow**
   - exact symbolic name of `identifierSource == 1`
   - full direct proof of `identifierSource == 2`
   - constructor semantics of `perfStats` vs `perfStatsArray`
