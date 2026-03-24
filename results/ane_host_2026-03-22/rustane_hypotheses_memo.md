# Rustane Hypotheses Memo

This memo translates the host-side ANE reverse-engineering findings into
`rustane`-specific hypotheses and priorities.

Scope:

- compile-cache reuse
- request-packing assumptions
- whether chaining is worth targeting

The goal is not to claim any of these are proven. The goal is to identify the
highest-value experiments suggested by the Apple host stack we observed on this
machine.

## Executive Summary

Three hypotheses are strong enough to matter immediately:

1. **`rustane` is probably leaving compile-cache reuse on the table.**
   Apple’s host stack has explicit model-cache identifiers, descriptor hashes,
   retain files, stale-model GC, and purge hooks. `rustane` appears to compile
   kernels as ephemeral standalone programs rather than as stable, reusable model
   identities.

2. **`rustane` likely models request submission too simply.**
   Apple’s runtime has a structured `_ANERequest` object with explicit symbol
   indices, optional weights buffers, perf-stats arrays, shared events, and
   transaction handles. `rustane` currently treats ANE dispatches much more like
   positional input/output tensor arrays.

3. **Chaining is interesting, but not yet the highest-ROI target.**
   Apple clearly has multi-procedure chaining with loopback inputs/outputs and
   signal events. That is potentially valuable for dispatch-heavy workloads.
   But today `rustane`’s biggest practical issues are still compile behavior and
   scale ceilings, not dispatch orchestration sophistication. Chaining is worth
   investigating after cache identity and compiler-surface experiments, not
   before.

## 1. Compile-Cache Reuse Hypothesis

### Hypothesis

`rustane` is compiling too many effectively-new models because its descriptor
identity is unstable or too local to each run, so Apple-side compiled-model
reuse is not being exercised.

### Why this hypothesis exists

Observed host-side signals:

- `_ANEModelCacheManager`
- `_ANEInMemoryModelCacheManager`
- `_ANEStorageMaintainer`
- `_ANEStorageHelper`
- `compiledModelExistsFor:`
- `compiledModelExistsMatchingHash:`
- `purgeCompiledModel:`
- `purgeCompiledModelMatchingHash:`
- `cacheURLIdentifier`
- `identifierSource`
- `sourceURL`
- `string_id`
- `networkTextHash`
- `weightsHash`
- `optionsPlistHash`
- `retainModelCache=%d`
- `modelRetainFileExists=%d`

Observed `rustane` shape:

- 'lib.rs' describes `ane-bridge` as a thin wrapper around dynamic weight pipeline behavior
- 'layer.rs' shows the engine repeatedly compiling/using many kernel executables as isolated programs
- README positions the project around direct MIL compilation rather than a persistent model service in 'README.md'

### What this implies

Apple’s stack appears to distinguish:

- descriptor/content identity:
  - network text
  - weights
  - options plist
- cache/source identity:
  - source URL
  - cache URL identifier
  - identifier source

That means a model can be “the same” in more than one sense. If `rustane`
produces equivalent MIL but varies temporary paths, file names, or descriptor
packaging, it may still miss the host cache path Apple uses.

### Confidence

Medium.

The cache machinery is clearly real. What is not yet proven is whether the
current direct private-API path used by `ane-bridge` benefits from it in the
same way as Apple’s daemon/service route.

### Best experiments

1. Fix model/program identity across repeated compile runs.
   Use stable MIL text, stable weight ordering, and stable options serialization.

2. Measure compile time for identical repeated compiles under controlled naming.
   If the second compile time collapses, there is likely reuse.

3. Vary only one dimension at a time:
   - same MIL text, different temp path
   - same MIL text, different options plist order
   - same MIL text, same weights, different descriptor filename

4. Search for persistent compiled artifacts or timestamps after repeated runs.
   The host-side storage helpers strongly suggest persisted artifacts exist.

### Recommendation

This is worth targeting soon.

Not because it will solve shape-limit failures directly, but because it may:

- reduce repeated compile overhead,
- clarify whether Apple’s model identity rules affect behavior,
- give `rustane` a more realistic picture of how the host ANE stack wants to be used.

## 2. Request-Packing Assumptions Hypothesis

### Hypothesis

`rustane`’s current mental model of an ANE dispatch as “input tensors in fixed
order, output tensors in fixed order” is functionally workable, but likely below
the abstraction level Apple’s runtime actually uses internally.

### Why this hypothesis exists

Observed host-side signals:

- `_ANERequest`
- `requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:`
- variants with:
  - `weightsBuffer`
  - `perfStats`
  - `sharedEvents`
  - `transactionHandle`
- validation strings for:
  - input/output array count mismatch
  - invalid symbol indices
  - `kANERequestMaxSymbolIndex`

Observed `rustane` shape:

- 'layer.rs' submits kernels through `run_cached_direct(&[...], &[...])`
- The current engine code stages buffers manually and assumes explicit packed or
  per-buffer positional ordering

### What this implies

Apple’s runtime appears to think in terms of:

- **symbol-indexed program I/O**
- **procedure-level execution**
- optional **request-local weights**
- optional **perf stats collection**
- optional **shared-event synchronization**

That is a more expressive execution contract than raw positional tensors.

### What does *not* follow from this

This does **not** mean `rustane` is wrong today.

It may just mean:

- the `ane` crate is exposing a simpler lower-level facade,
- or `rustane` is using the direct path beneath this richer request layer,
- or the richer layer exists mostly for Apple’s daemon and virtualization paths.

### Confidence

Medium-high that Apple’s runtime is richer than `rustane`’s current abstraction.

Low-medium that `rustane` should immediately change its internal request model.

### Best experiments

1. Compare performance/stat reporting between current dispatches and any path
   that exposes hardware/perf stats more explicitly.

2. Investigate whether the `ane` crate can already expose symbol-indexed request
   behavior without major surgery.

3. Check whether any current pathologies in `rustane` line up with request model
   limitations:
   - repeated weight staging
   - inability to express loopback without extra kernels
   - difficulty coordinating multi-step work with shared events

### Recommendation

Do **not** redesign `rustane` around this immediately.

Treat it as a framework for interpreting future optimization work:

- if compile/cache experiments pay off first, keep request-packing changes small;
- if later performance work hits staging/dispatch coordination limits, revisit
  this with more urgency.

## 3. Chaining Worth-Targeting Hypothesis

### Hypothesis

Apple’s chaining model is real and potentially valuable, but it is probably a
second-wave optimization target for `rustane`, not a first-wave survival fix.

### Why this hypothesis exists

Observed host-side signals:

- `_ANEChainingRequest`
- `prepareChainingWithModel:options:chainingReq:qos:error:`
- loopback input/output symbol indices
- `signalEvents`
- `transactionHandle`
- `memoryPoolId`
- `fwEnqueueDelay`
- `procedureIndex`

This looks like a genuine mechanism for:

- multi-procedure execution,
- reusing outputs as future inputs,
- synchronizing through shared events,
- reducing host orchestration between steps.

### Potential upside for `rustane`

If `rustane` eventually uses a richer chained execution model, it might reduce:

- per-dispatch host overhead,
- repeated host-side staging,
- repeated readback/repack between logically connected stages,
- some of the CPU-side glue now needed between ANE kernels.

That is especially interesting because the README already highlights dispatch
overhead as an ANE gotcha in 'README.md'.

### Why it is not top priority yet

Current branch reality:

- the most painful issues have been compile constraints and shape ceilings
- the recent work has focused on making kernels compile and scale at all
- many bottlenecks are still at compile/layout/kernel-boundary level, not at the
  orchestration boundary

Chaining will not fix:

- illegal spatial-width kernels
- unsupported compiler ops
- FFN decomposition needs
- descriptor/cache instability

### Confidence

Medium that chaining would matter eventually.

Low-medium that it is the best next optimization target right now.

### Best experiments

1. Quantify dispatch-heavy sections where chaining could plausibly help.
   Focus on forward paths with many small kernels or repeated weight/buffer
   transitions.

2. Identify one narrow prototype:
   - e.g. a two-stage or three-stage forward-only chain
   - not a full-model rewrite

3. Only pursue it after compile-cache and identity experiments establish a more
   stable base.

### Recommendation

Worth targeting later, not first.

Suggested priority:

1. compile/cache identity experiments
2. framework/descriptor parity experiments
3. only then chaining prototype work

## 4. Virtual-Client Hypothesis

### Hypothesis

The `_ANEVirtualClient` path is important for understanding Apple’s packaging and
transport assumptions, but probably not the right immediate target for local
`rustane` optimization.

### Why

Observed signals:

- `AppleVirtIONeuralEngineDevice`
- guest/host transfer functions
- chunking and interface-version compatibility logic
- model/file/IOSurface serialization routines

This likely explains why Apple’s private framework surface is broader than the
local daemon path alone.

### Recommendation

Use it as a conceptual reference only for now:

- it teaches us what Apple serializes,
- it teaches us that model transport includes cache IDs and file chunking,
- but it is not likely the local performance path `rustane` should target next.

## Priority Recommendation

If this memo drives actual engineering work, the order I would choose is:

1. **Compile-cache reuse investigation**
   - highest likely payoff per unit of effort
   - directly grounded in strong host-side signals

2. **Descriptor identity hygiene**
   - likely prerequisite to any meaningful cache experiment

3. **Targeted request-model study**
   - only where current staging/dispatch behavior becomes an optimization limit

4. **Chaining prototype**
   - only after the above

5. **Virtual-client study**
   - mostly for completeness and long-term architectural understanding

## Bottom Line

The host-side ANE findings suggest:

- `rustane` is pointed at a real private ANE API surface,
- Apple has richer compile-cache identity handling than the project currently
  appears to model,
- Apple’s request/execution layer is more structured than plain positional
  buffers,
- chaining is promising but should come after cache/identity work.

So the most useful immediate hypothesis is:

**`rustane` should test whether stable descriptor construction and stable model
identity materially improve compile reuse before attempting a larger execution
model redesign.**
