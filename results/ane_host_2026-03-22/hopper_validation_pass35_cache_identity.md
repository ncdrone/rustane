# Hopper Validation Pass 35: AppleNeuralEngine Cache and Model Identity

This pass targets the core identity-bearing objects in `AppleNeuralEngine`:

- `-[_ANEInMemoryModelDescriptor initWithNetworkText:weights:optionsPlist:isMILModel:]`
- `-[_ANEModel initWithModelAtURL:sourceURL:UUID:key:identifierSource:cacheURLIdentifier:modelAttributes:standardizeURL:string_id:generateNewStringId:mpsConstants:]`
- the associated property surface:
  - `networkTextHash`
  - `weightsHash`
  - `optionsPlistHash`
  - `hexStringIdentifier`
  - `identifierSource`
  - `cacheURLIdentifier`
  - `sourceURL`
  - `string_id`

The goal is to pin down how Apple constructs model identity before Espresso or
the compiler runtime try to cache anything.

## 1. High-level result

Apple keeps **two layers of identity**:

- a **descriptor/content identity**
  - derived from network text, weight contents, and options plist
- a **model/cache identity**
  - derived from model URL / source URL / cache URL identifier / string id

That split matches what we were already inferring from strings, but Hopper now
makes the structure much clearer.

## 2. `_ANEInMemoryModelDescriptor`

### 2.1 `initWithNetworkText:weights:optionsPlist:isMILModel:`

Size:

- length: `1072` bytes
- basic blocks: `22`

### 2.2 What Hopper shows

The initializer:

1. retains:
   - network text
   - weights dictionary
   - options plist
2. copies network text into the descriptor
3. computes:
   - `networkTextHash`
   using `hexStringForData:`
4. stores the raw weights dictionary
5. extracts and sorts the weight keys
6. walks those keys in sorted order
7. pulls the first value from each per-key array/dictionary payload
8. appends those values into a flat array
9. computes:
   - `weightsHash`
   using `hexStringForDataArray:`
10. copies the options plist
11. computes:
   - `optionsPlistHash`
   using `hexStringForData:`
12. stores the `isMILModel` flag

### 2.3 What this means

Two important points for `rustane`:

- weight hashing is **order-stabilized** by sorting keys first
- the descriptor hash layer is independent of file path / source URL identity

So Apple clearly wants a deterministic content-derived identity before it
reaches the cache layer.

## 3. `_ANEModel`

### 3.1 `initWithModelAtURL:sourceURL:UUID:key:identifierSource:cacheURLIdentifier:modelAttributes:standardizeURL:string_id:generateNewStringId:mpsConstants:`

Size:

- length: `888` bytes
- basic blocks: `24`

### 3.2 What Hopper shows

This initializer is the policy gate for model identity.

It:

1. rejects cache URL identifiers containing `..`
2. checks `identifierSource`
3. enforces:
   - if `identifierSource == 3`, `cacheURLIdentifier` must be non-nil
   - if `identifierSource == 2`, `sourceURL` must be non-nil
4. optionally standardizes both model URL and source URL
5. stores:
   - model URL
   - source URL
   - key
   - identifierSource
   - cacheURLIdentifier
   - model attributes
   - UUID
   - mps constants
6. sets `string_id`
   - either from caller-supplied value
   - or by generating it from the model path via `kdebug_trace_string()`

### 3.3 What this means

This is the real bridge between:

- content identity
- source/path identity
- cache-key identity

It also shows that `string_id` is not just some opaque runtime handle. It can be
derived from the path when requested.

## 4. `identifierSource` semantics

What remains true after this pass:

- `1` = default model URL / key identity mode
- `2` = source-URL-and-key mode
- `3` = cache-URL-identifier mode

The exact Apple symbol for `1` is still not recovered from codegen, but the
operational meaning is clear from the initializer and factory methods.

## 5. What this changes in our understanding

### 5.1 Apple’s cache layer is not purely content-addressed

The descriptor hashes are content-based, but `_ANEModel` adds a separate source
/ cache URL / string-id layer on top.

That means a `rustane` compile-cache experiment needs both:

- deterministic MIL / weights / options
- and stable model/source/cache identity at the runtime API boundary

### 5.2 Stable weight-key ordering is a concrete actionable clue

This is one of the most actionable findings for the repo.

If `rustane` emits weights in nondeterministic key order, it will not match
Apple’s descriptor hashing behavior even if the underlying tensors are the same.

### 5.3 `string_id` is part of the identity story

The generated-or-supplied `string_id` keeps showing up across AppleNeuralEngine
runtime logs and virtual-client paths. It is not the whole cache identity, but
it is clearly part of the runtime model identity surface.
