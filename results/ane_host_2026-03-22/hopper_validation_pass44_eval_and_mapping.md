# Hopper Validation Pass 44: Evaluate Path and IOSurface Mapping

This pass targets the remaining runtime submission questions relevant to
dispatch reduction:

- `-[_ANEClient evaluateWithModel:options:request:qos:error:]`
- `-[_ANEClient doEvaluateDirectWithModel:options:request:qos:error:]`
- `-[_ANEClient mapIOSurfacesWithModel:request:cacheInference:error:]`
- `-[_ANEProgramIOSurfacesMapper prepareANEMemoryMappingParams:request:]`
- `-[_ANEProgramIOSurfacesMapper mapIOSurfacesWithModel:request:cacheInference:error:]`

The goal is to understand whether the normal client path differs materially from
direct eval, and where IOSurface mapping overhead actually lives.

## 1. High-level result

The biggest surprise in this pass is simple:

- `-[_ANEClient evaluateWithModel:options:request:qos:error:]`
  is just a thin wrapper over
- `-[_ANEClient doEvaluateDirectWithModel:options:request:qos:error:]`

So, at least on this build, there is **no separate normal host-side evaluate
path** hiding behind that selector. The real differences in request overhead are
elsewhere:

- request construction
- IOSurface mapping
- virtual-client fallback
- Espresso-side request reuse / segment preparation

## 2. `_ANEClient evaluateWithModel`

### 2.1 What Hopper shows

The body is trivial:

- it immediately calls `doEvaluateDirectWithModel:options:request:qos:error:`

### 2.2 What this means

For `rustane`, this substantially narrows the direct-vs-normal question:

- if we are already driving `_ANEClient evaluateWithModel:...`, we are not
  missing some richer non-direct host path
- the meaningful comparison is instead:
  - `_ANEClient` direct path
  - `_ANEVirtualClient` path
  - and any daemon/other wrappers above that

## 3. `_ANEClient doEvaluateDirectWithModel`

### 3.1 What Hopper confirms

This is the real host submission path:

- retain model/options/request
- choose `queueIndexForQoS`
- inspect request input/output IOSurface IDs for tracing
- wait on a queue-specific semaphore
- call:
  - `processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:`
  on the model’s `program`
- signal completion / trace end

### 3.2 What this means

The direct path still looks one-request-at-a-time. Hopper still does not show
hidden batching here.

So any dispatch reduction beyond shaving pure request overhead likely needs:

- request reuse
- multi-procedure grouping
- or explicit chaining

not just switching selectors.

## 4. `_ANEClient mapIOSurfacesWithModel`

### 4.1 What Hopper shows

This is the host-side mapping front door.

It:

- returns immediately through `virtualClient` if present
- rejects VM mode on the normal path
- fetches `model.mapper`
- calls:
  - `mapIOSurfacesWithModel:request:cacheInference:error:`
  on that mapper
- traces with `model.string_id`

### 4.2 What this means

Mapping overhead is factored out into the mapper object, not entangled directly
with evaluate. That makes the mapper path an independent optimization target.

## 5. `_ANEProgramIOSurfacesMapper prepareANEMemoryMappingParams`

### 5.1 What Hopper shows

This helper prepares a packed mapping struct by:

- zeroing a fixed-size buffer
- setting:
  - `ioSurfacesCount`
  - `procedureIndex`
  - `programHandle`
- iterating request inputs and outputs
- copying:
  - IOSurface refs
  - symbol indices
  - a type tag distinguishing input/output/weights
- optionally adding `weightsBuffer`

### 5.2 What this means

This is the lowest cleanly surfaced place where per-request IOSurface packing is
happening. If request reuse avoids rerunning this path, that would be a concrete
dispatch-overhead win.

## 6. `_ANEProgramIOSurfacesMapper mapIOSurfacesWithModel`

### 6.1 What Hopper shows

This method:

- validates the request/model pair
- calls `prepareANEMemoryMappingParams:request:`
- dispatches synchronously on `gANEMemoryMapperQueue`
- stores success/error through block-captured state

The mapper path therefore adds:

- validation
- parameter packing
- one synchronous dispatch onto the memory-mapper queue

before evaluation can proceed.

### 6.2 What this means

This is likely a meaningful chunk of per-request overhead, especially for small
kernels. It also reinforces that request reuse is not just about avoiding ObjC
allocations: it may also avoid repeated mapping/setup work.

## 7. `_ANEVirtualClient` mapper/eval path

Although not the main target of this pass, the virtual path confirms the normal
client path is the only reasonable optimization baseline:

- `_ANEVirtualClient mapIOSurfacesWithModel...` says “No support for
  VirtualClient yet.” and routes through much heavier compatibility logic
- `_ANEVirtualClient doEvaluateWithModel...` / legacy path performs explicit
  struct packing, dictionary building, IOSurface copies, and IOUserClient calls

That path is clearly not where early repo-side dispatch optimizations should
focus.

## 8. What this changes in our understanding

### 8.1 The client “normal path” vs “direct path” delta is basically gone

That question is now settled:

- the public evaluate selector is effectively the direct path on this build

### 8.2 The meaningful runtime overhead lives in request setup and mapping

The main remaining runtime-overhead buckets now look like:

- request object construction
- request validation
- IOSurface mapping param packing
- synchronous mapping queue work
- program `processRequest(...)`

That is a much more concrete breakdown than we had before.
