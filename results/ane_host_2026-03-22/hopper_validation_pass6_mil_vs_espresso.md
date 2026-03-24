# Hopper Validation Pass 6: MIL vs Espresso

This pass treats the two dyld-cache-backed framework documents as different
layers of the host ML stack:

- `MIL Framework` = IR format / parser / serializer layer
- `Espresso` = runtime / ANE integration / request construction layer

It also closes the earlier “are these both just the whole dyld cache?”
question: they are not. Hopper shows distinct segment maps, symbol families,
and string sets for each document.

## 1. Document sanity check

### `MIL Framework`

- compact segment map with the expected `__TEXT`, `__DATA_CONST`, `__DATA`,
  `__AUTH`, `__AUTH_CONST`, `__LINKEDIT`, and `External Symbols`
- procedure names are overwhelmingly `MIL::...` and C++ parser/serializer/blob
  infrastructure
- strings are overwhelmingly `CoreML.Specification.MILSpec...` plus MIL blob
  writer errors

### `Espresso`

- much larger ObjC-heavy image with `__objc_methlist`, `__objc_classname`,
  `__objc_methname`, `__objc_stubs`, and many imported dyld-cache ranges
- procedure names are overwhelmingly `Espresso::MILTranslator::...`,
  `Espresso::ANECompilerEngine::...`, and `Espresso::ANERuntimeEngine::...`
- strings include `model.espresso.net`, `fragment.mil`,
  `evaluateRealTimeWithModel:options:request:error:`, request-construction
  diagnostics, and cache-identifier logging

Conclusion: `MIL Framework` is the MIL image; `Espresso` is the Espresso image.
They are not duplicate “whole-cache” views.

## 2. `MIL Framework`: IR format / parser / serializer

### 2.1 Parsing MIL text is a first-class MIL responsibility

Validated procedures:

- `MIL::Text::ParseProgramFromFile`
- `MIL::Text::TryParseProgram`
- `MIL::Text::TryParseProgramView`
- `MIL::Text::ParseProgramView`

What Hopper shows in `ParseProgramFromFile`:

- copies the model path out of parser options
- creates an mmap-backed reader via `MIL::Blob::MakeMMapFileReader`
- reads file contents through the blob reader
- allocates a `MIL::ParserContext`
- constructs `MIL::Text::Parser::Program`
- calls `MIL::Text::Parser::Program::Parse`

That is direct evidence that MIL itself owns text parsing and scanner/parser
machinery. Espresso is not the first place where textual MIL becomes IR.

### 2.2 Serialization to MILSpec protobuf is also in MIL

Validated procedures:

- `MIL::Proto::SerializeProgram`
- `MIL::Proto::SerializeFunction`
- `MIL::Proto::SerializeOperation`

What Hopper shows in `SerializeProgram`:

- allocates a `CoreML::Specification::MILSpec::Program`
- iterates the program’s functions
- serializes each through `SerializeFunction`
- copies them into the protobuf map on the output program
- serializes IR attributes into the MILSpec object

Representative MIL strings:

- `CoreML.Specification.MILSpec.Program`
- `CoreML.Specification.MILSpec.Function`
- `CoreML.Specification.MILSpec.Operation`
- `CoreML.Specification.MILSpec.Value.BlobFileValue`
- `CoreML.Specification.MILSpec.Value.BlobFileValue.fileName`

This confirms the split we suspected earlier:

- `MIL Framework` owns the IR/text/protobuf representation
- `Espresso` consumes and translates around that representation for runtime use

### 2.3 Model path/name/owner are explicit MIL metadata

Validated procedures:

- `ParserOptionsImpl::SetModelPath`
- `ParserOptionsImpl::GetModelPath`
- `BuildInfo::SetModelPath`
- `BuildInfo::GetModelPath`
- `BuildInfo::SetModelName`
- `BuildInfo::GetModelName`
- `BuildInfo::SetModelOwner`
- `BuildInfo::GetModelOwner`

This matters because path/provenance are not invented only in
`AppleNeuralEngine.framework`. MIL already tracks model path, name, and owner at
its own layer, which fits the later framework-level cache identity behavior.

### 2.4 Blob/file layout is strict, aligned, and format-aware

Validated procedures:

- `MIL::Blob::FileWriter::WriteData`
- `MIL::Blob::StorageWriter::Impl::StorageWriter(...)`
- `MIL::Blob::StorageWriter::WriteData<...>`

What Hopper shows:

- `FileWriter::WriteData` explicitly rejects offsets that are not `0x40`
  aligned
- `StorageWriter::Impl` either writes a fresh header or validates an existing
  header before appending metadata/data
- the storage path is treated as a structured file format, not an arbitrary
  byte dump

Representative strings:

- `[MIL FileWriter]: Provided offset not aligned. offset=`
- `[MIL StorageWriter]: Incorrect file header, please use truncateFile=true`
- `[MIL StorageWriter]: dataOffset is expected to be 64 bits aligned.`
- `[MIL StorageWriter]: Metadata written to different offset than expected.`

For `rustane`, this reinforces that deterministic weight/blob layout is likely
important if compile/cache reuse is the goal.

### 2.5 Mutable weight paths are visible at the MIL boundary

Relevant strings:

- `MutableMILWeightPaths`
- `ValidateMutableMILWeightPaths`
- `All mutable weights on BlobFileMutabilityInfo attribute must be provided by SetMutableMILWeightPaths API.`

This is important because `rustane` already relies on dynamic weight staging.
Apple’s stack appears to have an explicit, validated mutable-weight model rather
than treating dynamic weights as an ad hoc exception.

### 2.6 Bottom line on MIL

`MIL Framework` is the right target for:

- MIL text parsing
- MILSpec protobuf serialization
- blob-file and external-weight layout
- model path/name/owner metadata
- mutable weight path semantics

It does **not** look like the main target for request orchestration, cache URL
identifier generation, or ANE runtime submission.

## 3. `Espresso`: runtime / ANE integration / request construction

### 3.1 Espresso contains both translation and runtime layers

Validated procedure families:

- `Espresso::MILTranslator::...`
- `Espresso::MILTranslator::MILProgramBuilder::...`
- `Espresso::ANECompilerEngine::...`
- `Espresso::ANERuntimeEngine::...`

This is the cleanest layering split visible in Hopper:

- `MILTranslator` / `MILProgramBuilder`: translate higher-level/EIR/network
  forms into MIL-shaped program structures
- `ANECompilerEngine`: compile-time lowering and ANE-targeted segment/kernel
  logic
- `ANERuntimeEngine`: segment execution, request creation, cache-ID-based model
  loading, runtime evaluation

### 3.2 Espresso is the runtime bridge into ANE, not just a neighboring framework

Representative strings:

- `model.espresso.net`
- `/model.espresso.net`
- `fragment.mil`
- `compile_network_to_cache_url_identifier`
- `Created ANE in-memory model identifier`
- `Purge ANE in-memory model identifier`
- `evaluateRealTimeWithModel:options:request:error:`

Representative procedures:

- `Espresso::ANERuntimeEngine::compiler::compile_network_to_cache_url_identifier`
- `Espresso::ANERuntimeEngine::compiler::create_ane_request`
- `Espresso::ANERuntimeEngine::compiler::build_segment`
- `Espresso::ANERuntimeEngine::compiler::query_each_segment`
- `Espresso::ANERuntimeEngine::compiler::add_ane_eval_profiling_options`

This is the strongest evidence so far that Espresso is the bridge layer where
MIL-derived network artifacts turn into ANE runtime objects and requests.

### 3.3 `compile_network_to_cache_url_identifier` is a concrete ANE compile/cache path

Hopper decompilation of
`Espresso::ANERuntimeEngine::compiler::compile_network_to_cache_url_identifier`
shows it:

- requires the E5 compiler context
- expects exactly one segment and rejects multihead multiprocedure networks
- derives a segment key with `key_for_segment`
- constructs a file URL from the segment’s network path
- recovers an original source URL if present
- creates an `_ANEModel` through the `modelAtURLWithSourceURL:sourceURL:key:cacheURLIdentifier:` path
- obtains the shared ANE connection
- calls `compileModel:options:qos:error:`
- on success, calls `getCacheURLIdentifier`
- logs and returns the resulting cache identifier

This directly ties Espresso into the host-side ANE cache identity flow we
already mapped in `AppleNeuralEngine.framework` and `ANECompilerService`.

### 3.4 `create_ane_request` confirms Espresso is assembling real `_ANERequest` objects

Hopper decompilation of
`Espresso::ANERuntimeEngine::compiler::create_ane_request(...)` shows it:

- derives a procedure name for the segment/configuration
- creates per-procedure bookkeeping maps keyed by that name
- obtains input symbol indices from a procedure IO symbol mapper
- obtains output symbol indices from the same mapper
- resolves input blob names to concrete ANE I/O surfaces
- resolves output blob names to concrete ANE I/O surfaces
- builds a mutable perf-stats array keyed by segment name and/or `net`
- finally calls one of the `_ANERequest` constructors:
  - `requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:`
  - or `requestWithInputs:inputIndices:outputs:outputIndices:perfStats:procedureIndex:`

This is a direct bridge from Espresso’s segment/runtime bookkeeping into the
`_ANERequest` structure we already reconstructed in `AppleNeuralEngine`.

That matters for `rustane` because it shows Apple is not operating on raw
positional arrays alone. There is an explicit layer of:

- procedure naming
- symbol-index mapping
- surface resolution
- optional weights buffer attachment
- perf-stats array attachment

### 3.5 `build_segment` looks like the model-build / runtime-load coordinator

`Espresso::ANERuntimeEngine::compiler::build_segment(...)` is much larger
(`230` basic blocks) than the request and cache-ID helpers, but its strings and
call graph are enough to place it:

- logs creation and purge of ANE in-memory model identifiers
- composes per-segment build/load decisions
- sits between compile-time segment generation and runtime request execution

This is likely the main Espresso-side coordinator for:

- producing the cache-URL-identified model artifact
- associating it with an in-memory ANE model identifier
- preparing the segment for later request submission

### 3.6 Espresso also contains the mutable-weight validation boundary

Representative strings:

- `MutableMILWeightPaths`
- `ValidateMutableMILWeightPaths`
- `The base model has \`BlobFileMutabilityInfo\` containing mutable weights, but they are not provided by SetMutableMILWeightPaths API.`

This suggests the mutable-weight model is enforced at both:

- the MIL/serialization boundary
- the Espresso/runtime-compiler boundary

So if `rustane` ever wants Apple-like dynamic-weight semantics, Espresso is at
least as relevant as MIL.

### 3.7 Bottom line on Espresso

`Espresso` looks like the right target for:

- how MIL-derived artifacts turn into ANE-targeted segments
- how cache URL identifiers are requested/generated
- how ANE requests are constructed from surfaces + symbol indices
- how per-segment perf/profiling data is attached
- how real-time and normal runtime evaluation paths are surfaced

It is more directly relevant than MIL to:

- request packing
- compile-cache reuse
- ANE runtime submission behavior

## 4. Updated synthesis for the repo

The refined split for `rustane` is:

- `MIL Framework` tells us how Apple represents, parses, serializes, and stores
  MIL programs and weight blobs
- `Espresso` tells us how Apple turns those programs into ANE-oriented segment
  builds, cache identifiers, and `_ANERequest` objects

If we are prioritizing reverse-engineering effort by likely payoff:

1. `AppleNeuralEngine.framework`
   - exact `_ANEModel`, `_ANERequest`, and chaining/runtime semantics
2. `Espresso`
   - the practical bridge from MIL/network artifacts into ANE requests and cache IDs
3. `MIL Framework`
   - parser/serializer/blob semantics and deterministic model representation
4. `ANECompiler.framework`
   - lower-level compile internals once the higher-level object flow is pinned down

So the earlier working model still holds, but more sharply now:

- **MIL = format and serialization substrate**
- **Espresso = operational ANE bridge**

For `rustane`, the most actionable immediate insight is still on the Espresso
side: stable segment/model identity and Apple-like request construction are much
more likely to influence compile/cache behavior than further MIL parser work.
