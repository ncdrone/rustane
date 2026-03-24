# Hopper Validation Pass 4

This pass targeted the remaining ambiguities from the previous Hopper work:

1. exact symbolic name for `identifierSource == 1`
2. fully direct proof of `identifierSource == 2`
3. `perfStats` versus `perfStatsArray` constructor semantics

Targets:

- `AppleNeuralEngine.hop`
- `ANECompilerService.hop`

## 1. `identifierSource` values

## 1.1 What is directly confirmed

From decompiled `_ANEModel` factory methods:

### `identifierSource == 1`

Directly used by:

- `+[_ANEModel modelAtURL:key:modelAttributes:]`
- `+[_ANEModel modelAtURL:key:mpsConstants:]`
- `+[_ANEModel modelAtURLWithSourceURL:sourceURL:key:cacheURLIdentifier:]`

So `1` is definitely the standard/default constructor mode.

### `identifierSource == 3`

Directly used by:

- `+[_ANEModel modelAtURLWithCacheURLIdentifier:key:cacheURLIdentifier:]`
- `+[_ANEModel modelWithCacheURLIdentifier:]`

and enforced by the designated initializer:

- if `identifierSource == 3` and `cacheURLIdentifier == nil`, initialization
  fails

So `3` is definitely the cache-URL-identifier mode.

## 1.2 What is strongly inferred but still not fully direct

### `identifierSource == 2`

Still not directly observed in a small factory method in this pass, but we now
have two strong code-level clues:

1. designated initializer:
   - if `identifierSource == 2` and `sourceURL == nil`, initialization fails

2. `_ANEModelCacheManager URLForModel:bundleID:useSourceURL:forAllSegments:aotCacheUrlIdentifier:`
   - when `useSourceURL == false` and `model.identifierSource != 2`, it starts
     from `modelURL`
   - otherwise it starts from `sourceURL`

This means value `2` is the source-aware identity mode. Combined with the
earlier runtime string:

- `identifierSource is _ANEModelIdentifierSourceURLAndKey but sourceURL is nil`

the best current conclusion remains:

- `2` = `_ANEModelIdentifierSourceURLAndKey`

This is still marked as **strong inference**, not full direct proof, because we
did not recover a caller that passes the literal value `2` into a factory in
clear decompiled form.

## 1.3 What is still unresolved

### Exact symbolic name for `identifierSource == 1`

This remains unresolved.

What we know:

- `1` is the default path used by `modelAtURL...`
- it works with or without a `sourceURL`
- it does not require a `cacheURLIdentifier`

What we do **not** yet have:

- a recovered Apple-side symbolic string naming value `1`

So the exact symbolic name for `identifierSource == 1` is still open.

Best current semantic label:

- **default model URL / key identity mode**

but that is our label, not Apple’s recovered string.

## 2. `_ANERequest` layout and perf stats ambiguity

This pass tightened the request layout substantially.

## 2.1 Directly confirmed `_ANERequest` fields

From accessors:

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

This is stronger than the earlier pass because the getter methods line up
cleanly and remove most of the layout ambiguity.

## 2.2 Assembly-confirmed constructor stores

From the assembly of
`-[_ANERequest initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:]`:

- x2 -> `+0x08`
- x3 -> `+0x10`
- x4 -> `+0x18`
- x5 -> `+0x20`
- x6 -> `+0x28`
- stack arg `arg_8` -> `+0x30`
- stack arg `arg_0` -> `+0x38`
- x7 -> `+0x50`
- no store to `+0x48` in the constructor body

## 2.3 What that means

This resolves the earlier ambiguity in one important direction:

- the constructor’s decompiler label `perfStats` on the seventh register
  argument is misleading
- the object stored from that position lands in the slot later exposed by
  `perfStatsArray`

So the best current reading is:

- the designated initializer argument that Hopper labeled as `perfStats`
  is functionally the **perf-stats array** field
- the separate `perfStats` object at `+0x48` is **not** initialized by this
  constructor

## 2.4 Remaining ambiguity

What is still unresolved:

- where `perfStats` (`+0x48`) is populated in the normal request lifecycle
- whether it is lazily derived from `perfStatsArray`
- whether another constructor / helper sets it later

But the core question is now answered:

- `perfStatsArray` is what the initializer populates
- `perfStats` is a distinct field, not the same slot under another name

## 2.5 Extra request metadata confirmed

This pass also directly confirmed:

- `completionHandler` exists as a real field at `+0x58`
- `initWithVirtualModel:` does almost nothing beyond `super init`
- `ioSurfacesCount` counts:
  - `inputIndexArray.count`
  - `outputIndexArray.count`
  - plus one more if `weightsBuffer` is present

That reinforces the idea that weights are treated as a separate IOSurface-bearing
participant in request packing.

## 3. Bottom Line

This pass closes one of the three open issues and narrows the other two.

### Resolved enough to act on

- `_ANERequest` layout
- `perfStatsArray` vs `perfStats` constructor behavior

### Strong but still not absolute

- `identifierSource == 2` as `_ANEModelIdentifierSourceURLAndKey`

### Still unresolved

- exact Apple symbolic name for `identifierSource == 1`

So the best current enum reconstruction is:

- `1` = default model URL / key identity mode
- `2` = source URL + key identity mode
- `3` = cache URL identifier identity mode

with only value `1` still lacking a recovered Apple-side symbolic string name.
