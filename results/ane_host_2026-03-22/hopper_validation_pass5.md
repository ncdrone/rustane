# Hopper Validation Pass 5

This pass targeted the last three ambiguities:

1. exact symbolic name for `identifierSource == 1`
2. direct proof of `identifierSource == 2`
3. `perfStats` vs `perfStatsArray` constructor semantics

## 1. `identifierSource == 2` is now strong enough to treat as directly proven

The strongest direct proof comes from the designated initializer:

- `-[_ANEModel initWithModelAtURL:sourceURL:UUID:key:identifierSource:cacheURLIdentifier:modelAttributes:standardizeURL:string_id:generateNewStringId:mpsConstants:]`

### Direct behavior

- if `identifierSource == 3` and `cacheURLIdentifier == nil`, initialization fails
- else, if `sourceURL == nil` and `identifierSource == 2`, initialization fails

The failure path for the `identifierSource == 2` branch uses the concrete error
string we already extracted:

- `identifierSource is _ANEModelIdentifierSourceURLAndKey but sourceURL is nil`

### Conclusion

This is now good enough to upgrade the earlier status:

- `identifierSource == 2` = `_ANEModelIdentifierSourceURLAndKey`

That is no longer just a “strong inference from surrounding behavior”. The
combination of the explicit numeric compare and the specific error string is
direct enough for practical purposes.

## 2. `identifierSource == 1` is semantically clear, but its exact symbolic name is still not recoverable

Directly observed factories:

- `+[_ANEModel modelAtURL:key:modelAttributes:]`
- `+[_ANEModel modelAtURL:key:mpsConstants:]`
- `+[_ANEModel modelAtURLWithSourceURL:sourceURL:key:cacheURLIdentifier:]`

All of these pass:

- `identifierSource = 1`

### What that proves

`identifierSource == 1` is the default constructor mode used for ordinary
model-at-URL creation.

It also proves something subtle:

- having a `sourceURL` present does **not** automatically imply
  `identifierSource == 2`

because `modelAtURLWithSourceURL:sourceURL:key:cacheURLIdentifier:` still passes
`identifierSource = 1`.

### What remains unresolved

No Apple-side symbolic string name for value `1` was recovered in this pass.

We searched the relevant strings and code paths and found:

- `_ANEModelIdentifierSourceURLAndKey` for `2`
- `_ANEModelCacheURLIdentifierSource` for `3`

but nothing similarly explicit for `1`.

### Best current statement

- `identifierSource == 1` = default model-URL identity mode

That is semantically accurate, but still **our** descriptive label rather than a
recovered Apple symbol name.

## 3. `perfStats` vs `perfStatsArray` is now resolved

This pass checked:

- `_ANERequest` getters
- `_ANERequest` setters
- the assembly of
  `-[_ANERequest initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:]`

### Directly confirmed layout

Accessors show:

- `inputArray` -> `+0x08`
- `inputIndexArray` -> `+0x10`
- `outputArray` -> `+0x18`
- `outputIndexArray` -> `+0x20`
- `weightsBuffer` -> `+0x28`
- `sharedEvents` -> `+0x30`
- `transactionHandle` -> `+0x38`
- `procedureIndex` -> `+0x40`
- `perfStats` -> `+0x48`
- `perfStatsArray` -> `+0x50`
- `completionHandler` -> `+0x58`

### Constructor assembly

The constructor stores:

- x2 -> `+0x08`
- x3 -> `+0x10`
- x4 -> `+0x18`
- x5 -> `+0x20`
- x6 -> `+0x28`
- stack arg `arg_8` -> `+0x30`
- stack arg `arg_0` -> `+0x38`
- x7 -> `+0x50`

There is **no store to `+0x48`** in that constructor body.

### Setter/accessor cross-check

Observed setters:

- `setSharedEvents:` writes `+0x30`
- `setTransactionHandle:` exists
- `setPerfStats:` writes `+0x48`
- `setCompletionHandler:` uses property storage at `+0x58`

No `setPerfStatsArray:` method was recovered in this pass.

### Conclusion

The ambiguity is now resolved:

- the constructor’s recovered argument label `perfStats` is misleading
- the constructor actually initializes the field later exposed as
  `perfStatsArray` (`+0x50`)
- `perfStats` (`+0x48`) is a distinct field that is set later through
  `setPerfStats:`

So the best current model is:

- `perfStatsArray` = constructor-supplied stats request/specification data
- `perfStats` = separate mutable runtime stats object / handle

Even if the exact semantic names could still be refined, the field split itself
is no longer ambiguous.

## 4. `_ANERequest` extra field status

This pass also confirmed there is no hidden second layout family needed to
explain the virtual path:

- `initWithVirtualModel:` does almost nothing beyond `super init`

So the normal `_ANERequest` layout remains the right mental model even for the
paths that later interact with the virtual client.

## 5. Net Result

### Closed enough

- `identifierSource == 2`
- `perfStats` vs `perfStatsArray`

### Still open

- exact Apple symbolic name for `identifierSource == 1`

### Best current enum reconstruction

- `1` = default model-URL identity mode
- `2` = `_ANEModelIdentifierSourceURLAndKey`
- `3` = `_ANEModelCacheURLIdentifierSource`

with only the symbolic label for `1` still missing.
