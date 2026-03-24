# Hopper Validation Pass 36: Espresso Compile-to-Cache Flow

This pass targets the Espresso side of runtime/cache identity:

- `Espresso::ANERuntimeEngine::compiler::compile_network_to_cache_url_identifier()`

and correlates it with the AppleNeuralEngine model factories it calls.

The goal is to explain how Espresso turns a network path plus segment key into
the `cacheURLIdentifier` that later ANE runtime calls use.

## 1. High-level result

Espresso’s compile-to-cache path is straightforward and very relevant to the
repo:

1. build an `_ANEModel`
2. compile it through the shared ANE connection
3. read back the `cacheURLIdentifier`
4. treat nil cache identifiers as hard failure

That means the runtime cache identity is not some separate daemon-only concept.
It is directly produced by the same compile call that Espresso makes in-process.

## 2. `compile_network_to_cache_url_identifier()`

Size:

- length: `1468` bytes
- basic blocks: `35`

### 2.1 What Hopper shows

The function:

1. verifies it is running in the supported ANE compiler context
2. checks the compiler instance and segment count
3. rejects:
   - multihead multiprocess models
   - in-memory model cases
4. creates an `NSMutableDictionary` of compile options
5. computes the segment key with:
   - `Espresso::ANERuntimeEngine::compiler::key_for_segment(...)`
6. logs:
   - `input net url=%s key=%s`
7. gets the source URL if one exists:
   - `get_original_url_if_exists(...)`
8. derives the parent directory of the model URL
9. constructs an `_ANEModel` with:
   - `modelAtURLWithSourceURL:sourceURL:key:cacheURLIdentifier:`
   where `cacheURLIdentifier` is initially nil
10. obtains the shared ANE connection
11. calls:
   - `compileModel:options:qos:error:`
12. on success, calls:
   - `getCacheURLIdentifier`
13. if that identifier is nil, throws:
   - `ANERuntimeCompiler: ANEF returned cacheURLIdentifier as nil.`
14. on success, returns the cache URL identifier string to the caller

### 2.2 What this means

This is exactly the kind of path `rustane` should care about if it wants to test
compile-cache reuse.

Espresso is not doing anything magical after compile. It is simply:

- constructing the right `_ANEModel`
- compiling it
- and asking for the cache identifier that ANEF assigned

## 3. What Espresso contributes to identity

The important inputs are:

- model file URL
- source URL if available
- segment key
- compile options dictionary
- QoS

The most important output is:

- `cacheURLIdentifier`

This is the bridge between Espresso’s segment model and AppleNeuralEngine’s
model/cache layer.

## 4. What this changes in our understanding

### 4.1 Cache identity is compile-produced, not just cache-manager-produced

Earlier work on `_ANEModelCacheManager` showed how paths are derived and looked
up. This pass shows the other side:

- the compile call itself produces the cache URL identifier that later lookup
  uses

### 4.2 `sourceURL` really matters in a live compile path

Espresso explicitly tries to recover and pass through the original source URL.
That reinforces the earlier AppleNeuralEngine-side finding that
`identifierSource == 2` is a real operational mode, not a dead field.

### 4.3 This is the clearest runtime experiment target for `rustane`

If the repo wants to test cache reuse instead of just reverse-engineering it,
the closest hypothesis now is:

- stable model URL
- stable source URL
- stable key / segment identity
- deterministic descriptor contents

should yield more stable or reusable `cacheURLIdentifier` behavior across
repeated compiles.
