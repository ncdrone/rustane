# Hopper Validation Pass 43: Espresso Request Construction and Segment Submission

This pass targets the Espresso side of runtime request creation:

- `Espresso::ANERuntimeEngine::compiler::create_ane_request(...)`
- `Espresso::ANERuntimeEngine::compiler::create_ane_request_for_runtime_segment_combination(...)`
- `Espresso::ANERuntimeEngine::compiler::build_segment(...)`

The goal is to understand how Apple’s high-level runtime actually groups work
into `_ANERequest`s before submitting them.

## 1. High-level result

Espresso is not doing anything magical that looks like hidden generic batching.
What it is doing is:

- computing a procedure name / segment key
- using `procedure_io_symbol_mapper_t` to fetch symbol indices
- collecting input/output IOSurfaces from named blob containers
- building the right `_ANERequest` variant
- storing those requests per segment / procedure key

This is useful because it narrows the dispatch-reduction story:
- Apple seems to reduce overhead by preparing structured per-segment requests and
  model state, not by auto-fusing arbitrary requests behind the scenes.

## 2. `create_ane_request(...)`

This is the standard per-segment request builder.

What it does:

- derives a procedure name for the segment/configuration
- stores procedure index and symbol-index mappings into internal maps
- materializes input IOSurfaces by:
  - looking up named blob containers
  - pulling `ane_io_surfaceForMultiBufferFrame(...)`
- materializes output IOSurfaces the same way
- collects output and input symbol-index arrays from
  `procedure_io_symbol_mapper_t`
- optionally collects extra model/net constants into a perf-stats-like array
- chooses the `_ANERequest` constructor form:
  - with `weightsBuffer` when `modelHasWeightsBuffer` style flags are set
  - without it otherwise

The key direct result is that Espresso is very explicitly using:

- `requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:`
- or
- `requestWithInputs:inputIndices:outputs:outputIndices:perfStats:procedureIndex:`

depending on segment/runtime state.

## 3. `create_ane_request_for_runtime_segment_combination(...)`

This is the multi-procedure / runtime-combination request builder.

What it does:

- derives a combined procedure name
- builds combined input/output symbol-index maps
- gathers input/output IOSurfaces across the segment combination
- finally builds a normal `_ANERequest` using:
  - `requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:`

The key point is:
- even the runtime-segment-combination path still lands on the standard request
  family, not on `_ANEChainingRequest`

So Apple has at least two distinct higher-level combination mechanisms:
- multi-procedure combined request construction in Espresso
- explicit chaining request construction in AppleNeuralEngine

## 4. `build_segment(...)`

This is the broader segment orchestration path and it ties together many of the
earlier findings.

Key behaviors confirmed in this pass:

- creates or reuses per-segment dictionaries keyed by `key_for_segment(...)`
- builds compile/load options dictionaries
- recognizes precompiled paths
- creates `_ANEModel`
- records:
  - `Created ANE in-memory model identifier: %@`
  - `Purge ANE in-memory model identifier: %@`
- stores input/output symbol maps and request objects keyed by procedure name
- calls into the request-construction helpers above

This makes the runtime organization much clearer:
- segment identity
- model identity
- procedure identity
- request identity

are all tracked separately and then stitched together by Espresso.

## 5. What this changes in our understanding

### 5.1 Espresso is preparing reusable per-segment requests

This is the strongest new runtime-side result.

Espresso is not reconstructing the abstract submission model from scratch at the
last second. It keeps:

- procedure-name maps
- input symbol-index maps
- output symbol-index maps
- per-segment request objects

That suggests the closest repo-side dispatch-reduction analogue may be:
- more aggressive request reuse / preparation around stable kernel shapes
- not just lower-level FFI tricks

### 5.2 Multi-procedure grouping exists, but it is not the same as chaining

This matters for future `rustane` design:

- Espresso’s “runtime segment combination” still uses normal `_ANERequest`
- AppleNeuralEngine’s chaining uses `_ANEChainingRequest`

So there are at least two different ways Apple reduces or structures runtime
submission overhead, and they should not be conflated.

### 5.3 The best next dispatch-reduction hypotheses are now narrower

The Hopper work now points most strongly at:

- request reuse and stable request preparation
- segment/procedure grouping where the repo can express it
- explicit chaining only if we are willing to add new request/FFI surface

That is more concrete than the earlier generic “maybe batching exists” idea.
