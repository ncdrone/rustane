# Hopper Validation Pass 41: `_ANEModelCacheManager` Front Door, Path Lookup, and Stale-Hit Behavior

This pass targets the cache-manager side of runtime identity.

Because the `_ANEModelCacheManager` methods are less symbol-friendly in Hopper on
this build, this pass combines:

- direct front-door methods we can resolve now:
  - `-[_ANEDaemonConnection compiledModelExistsFor:withReply:]`
  - `-[_ANEDaemonConnection compiledModelExistsMatchingHash:withReply:]`
- with the `_ANEModelCacheManager` path/stale behavior already directly
  recovered in earlier Hopper work:
  - `cacheURLIdentifierForModel:useSourceURL:withReply:`
  - `URLForModel:bundleID:useSourceURL:forAllSegments:aotCacheUrlIdentifier:`
  - `removeIfStaleBinary:forModelPath:`

## 1. High-level result

There are two cache-lookup modes exposed all the way out to the daemon/client
boundary:

- lookup by full model identity
  - `compiledModelExistsFor:`
- lookup by matching hash
  - `compiledModelExistsMatchingHash:`

Underneath that, `_ANEModelCacheManager` derives a concrete cache path from:

- model key
- optionally source URL
- optionally cache URL identifier
- bundle/system cache root choice

and then applies stale-binary / source-freshness rules before returning a hit.

## 2. Direct front doors: `_ANEDaemonConnection`

### 2.1 `compiledModelExistsFor:withReply:`

This method is a thin XPC front door:

1. retain model and reply block
2. get synchronous remote object proxy
3. invoke:
   - `compiledModelExistsFor:withReply:`

### 2.2 `compiledModelExistsMatchingHash:withReply:`

Same structure, but routed to:

- `compiledModelExistsMatchingHash:withReply:`

### 2.3 What this means

Apple explicitly supports both:

- exact-identity cache existence
- hash-based matching existence

at the daemon connection boundary.

That is a stronger signal than just seeing internal cache-manager helpers.

## 3. Earlier Hopper-confirmed `_ANEModelCacheManager` behavior

From the earlier direct Hopper pass on `_ANEModelCacheManager`:

### 3.1 `cacheURLIdentifierForModel:useSourceURL:withReply:`

Confirmed behavior:

- chooses model URL vs source URL depending on `identifierSource` and
  `useSourceURL`
- combines a hex string of the selected path with a hex string of the model key
- returns a cache URL identifier built from those pieces

### 3.2 `URLForModel:bundleID:useSourceURL:forAllSegments:aotCacheUrlIdentifier:`

Confirmed behavior:

- chooses bundle cache root vs system cache root
- supports decoding explicit `cacheURLIdentifier`
  - `_` mapped back to `/`
- otherwise computes path hierarchy from:
  - path hash
  - key hash
- handles special AOT / shapes directory cases

### 3.3 `removeIfStaleBinary:forModelPath:`

Confirmed behavior:

- checks filesystem freshness / source timestamps
- removes stale compiled binaries when they no longer match the source state

## 4. What this means for cache hits

The cache-hit story is now fairly clear:

1. compile or caller creates `_ANEModel`
2. `_ANEModelCacheManager` derives or decodes the cache path
3. daemon/client can ask:
   - exact model exists?
   - matching hash exists?
4. stale-binary logic may suppress a hit even if a compiled artifact is present

So cache reuse is not just “same hash means hit.”

It is at least:

- identifier-mode dependent
- source-URL dependent in some modes
- filesystem freshness dependent

## 5. Why this matters for `rustane`

This is probably the most actionable runtime/caching result so far.

If the repo wants Apple-like compile-cache reuse, it likely needs to control:

- deterministic descriptor contents
- stable `_ANEModel` identity mode
- stable model/source paths
- and possibly avoid needless source-path churn that would make
  `removeIfStaleBinary` treat the cache as outdated

The presence of both:

- `compiledModelExistsFor:`
- `compiledModelExistsMatchingHash:`

also suggests two experiment shapes:

- strict exact-identity reuse
- looser hash-oriented reuse

Those are both worth testing from `ane-bridge` if the runtime API surface is
reachable.
