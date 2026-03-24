# Hopper Validation Pass 45: Espresso Request Reuse and Segment-State Retention

This pass targets Espresso’s reuse of already-constructed request state:

- `Espresso::ANERuntimeEngine::compiler::create_ane_request(...)`
- `Espresso::ANERuntimeEngine::compiler::create_ane_request_for_runtime_segment_combination(...)`
- `Espresso::ANERuntimeEngine::compiler::build_segment(...)`

The goal is to answer whether Espresso appears to retain request objects and
symbol maps across segment builds, which is directly relevant to dispatch
reduction ideas in `rustane`.

## 1. High-level result

Yes, Espresso clearly retains per-segment and per-procedure request-related
state.

The strongest direct evidence is in `build_segment(...)`, which populates and
reuses internal maps keyed by segment/procedure names for:

- procedure index
- input symbol-index arrays
- output symbol-index arrays
- NSArray-backed request input/output index objects
- compiled/request state dictionaries

This is not full proof of long-lived cross-run request reuse, but it is strong
evidence that Espresso is structured around **prepared segment request state**
rather than rebuilding everything from scratch each time.

## 2. `create_ane_request(...)`

### 2.1 What Hopper confirms

The standard request builder does all of the following:

- derive a procedure name for the segment/configuration
- record a procedure index in a string-keyed map
- populate maps of:
  - input symbol names
  - output symbol names
  - NSArray symbol-index arrays
- gather input/output IOSurfaces from named blob containers
- build one `_ANERequest` using the standard request constructors
- store the resulting request-associated arrays/maps under the procedure key

### 2.2 Why this matters

This means Espresso’s unit of reuse is not just “compiled model.”
It is also:

- procedure naming
- symbol-index materialization
- prepared ObjC collections that are ready to feed `_ANERequest`

That is much closer to the kind of dispatch reduction `rustane` could emulate
than trying to mirror Apple’s hidden runtime internals directly.

## 3. `create_ane_request_for_runtime_segment_combination(...)`

### 3.1 What Hopper confirms

The runtime-segment-combination path:

- derives a combined procedure name
- gathers input and output IOSurfaces across the combination
- assembles combined input/output index arrays
- then constructs a standard `_ANERequest`

Critically, it still stores the associated request metadata under the combined
procedure key.

### 3.2 Why this matters

This is the clearest Hopper evidence so far that Apple sometimes reduces
submission overhead by **grouping work at the Espresso segment/procedure layer**
while still using the normal request family.

That is probably a more reachable optimization model for `rustane` than
chaining in the near term.

## 4. `build_segment(...)`

### 4.1 What Hopper confirms

`build_segment(...)` now looks like the main retention point for reusable
request/segment state.

In the code we can see repeated population of maps at fixed offsets holding:

- procedure index by procedure name
- input symbol-index arrays by procedure name
- output symbol-index arrays by procedure name
- NSArray-backed index collections by procedure name
- segment/request dictionaries keyed by `key_for_segment(...)`

It also stores:

- created ANE in-memory model identifiers
- purge information
- segment tracing / analytics metadata

### 4.2 Why this matters

This is the strongest static evidence yet for a dispatch-reduction strategy
short of chaining:

- retain stable request state per segment/procedure
- reuse those prepared arrays/maps when the graph shape is unchanged

That aligns with what our Phase 1 cache experiments suggested at a higher level:
- same-process warm-state reuse is real

## 5. What this changes in our understanding

### 5.1 Request reuse looks like a better near-term target than hidden batching

The current evidence suggests:

- `_ANEClient` direct eval is still one request at a time
- Espresso reduces overhead by retaining **prepared request state**

So the most realistic near-term runtime optimization for `rustane` is probably:

- persistent request preparation / reuse for stable kernel shapes

not:

- hoping the client runtime will secretly batch requests for us

### 5.2 Multi-procedure grouping is real, but it happens above the client layer

The grouping logic we found is on the Espresso side, not in `_ANEClient`.

That means the repo’s closest analogue would be:

- reducing the number of logical per-token kernel submissions by grouping work
  at a higher graph/procedure layer

if that can be expressed through the current graph/runtime model.

### 5.3 The dispatch-reduction roadmap is now clearer

Static evidence now points to three progressively harder runtime ideas:

1. request reuse for stable kernel shapes
2. segment/procedure grouping where graph structure allows it
3. explicit chaining through `_ANEChainingRequest` if the repo is willing to add
   new FFI/runtime surface

That is a much sharper target list than we had before this pass.
