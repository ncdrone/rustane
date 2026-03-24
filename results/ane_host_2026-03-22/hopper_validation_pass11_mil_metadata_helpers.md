# Hopper Validation Pass 11: MIL Metadata Helpers

This pass targets the two MIL-side metadata helpers reached from
`CreateMILAndConvert(...)`:

- `RetrieveMutableWeightToSymbol(...)`
- `RetrieveModelSourceInformation(...)`

The goal is to understand exactly what Apple extracts from the MIL program
before producing the normalized `ANECProcedureInfo` representation.

## 1. High-level result

These helpers are narrower and cleaner than most of the surrounding compiler
pipeline.

They confirm that the MIL bridge explicitly extracts:

- a map from mutable-weight file paths to symbol names
- a filtered map of source/provenance attributes from the MIL program

So Apple’s MIL path is preserving both:

- dynamic weight identity
- model/source metadata

very early, before later ANEC IR/codegen stages.

## 2. `RetrieveMutableWeightToSymbol(...)`

Signature:

- `RetrieveMutableWeightToSymbol(const MIL::IRProgram&, unordered_map<string, string>&)`

Size:

- length: `204` bytes
- basic blocks: `9`

### 2.1 What Hopper shows

The function does this:

1. constructs `MIL::Attributes::BlobFileMutabilityInfo` from the `IRProgram`
2. calls `GetAllPaths()`
3. iterates the returned entries
4. copies the stored path string from each entry
5. calls `RetrieveAbsolutePath(...)`
6. inserts an entry into the output unordered_map
7. assigns the mapped value from the metadata entry’s symbol/name field

The effective logic is:

- `output[absolute_mutable_weight_path] = mutable_weight_symbol_name`

### 2.2 What that means

This is stronger than “the compiler knows mutable weights exist.”

It shows the MIL bridge explicitly normalizes mutable weight file paths and
binds them to symbolic names extracted from MIL metadata.

That is directly relevant to `rustane` because it suggests Apple’s dynamic
weight handling is keyed by a stable symbol/path relationship, not just by raw
buffer order or one-off staging.

### 2.3 Caller placement

Callers:

- `CreateMILAndConvert(...)`
- `ValidateDerivedMILProgram(...)`

So the same mutable-weight map is important in both:

- prepare-time conversion
- MIL validation

That is a strong sign this mapping is semantically important, not just debug
metadata.

## 3. `RetrieveModelSourceInformation(...)`

Signature:

- `RetrieveModelSourceInformation(const MIL::IRProgram&, map<string, string>&)`

Size:

- length: `604` bytes
- basic blocks: `33`

### 3.1 What Hopper shows

The function starts by:

1. reading the top-level program attributes via `MIL::IRObject::GetAttributes()`
2. constructing an exclusion set containing:
   - `BlobFileMutabilityInfo`
   - `ANEBinaryPoint`

Then it iterates all top-level attributes.

For each attribute not in the exclusion set:

- if the value is a dictionary:
  - iterate the dictionary entries
  - extract string scalars for key/value
  - insert them into the output `map<string, string>`
- otherwise:
  - extract the attribute name
  - extract the scalar string value
  - insert `map[attr_name] = scalar_value`

### 3.2 What that means

This helper is effectively doing:

- flatten selected top-level MIL source/provenance metadata into a plain string
  map

It explicitly excludes:

- `BlobFileMutabilityInfo`
  - handled separately by `RetrieveMutableWeightToSymbol(...)`
- `ANEBinaryPoint`
  - treated as a special attribute, not generic source metadata

That means Apple is conceptually separating three things:

1. mutable-weight metadata
2. binary-point / compiler-specific metadata
3. general source/provenance metadata

This separation is useful for `rustane` because it suggests the compiler
expects different metadata classes to remain distinct, even if they all live in
MIL attributes.

## 4. What the pair means together

Taken together, these helpers imply that `CreateMILAndConvert(...)` is not just
parsing/lowering MIL syntax. It is also building two distinct metadata products:

- **mutable-weight binding map**
  - absolute mutable weight path -> symbol name
- **source/provenance metadata map**
  - selected top-level MIL attributes flattened to string pairs

That makes the MIL bridge much richer than a plain syntax-to-IR converter.

## 5. Repo-relevant implications

### 5.1 Dynamic weights likely need stable symbolic identity

Because Apple explicitly builds a path-to-symbol map for mutable weights, the
compiler/runtime stack probably expects dynamic weights to have:

- stable symbolic names
- stable file/path identity

That strengthens the earlier hypothesis that `rustane` should care about stable
descriptor/model identity and deterministic weight materialization if it wants
to behave more like Apple’s own stack.

### 5.2 Source metadata is preserved separately from weight mutability

Because `RetrieveModelSourceInformation(...)` excludes
`BlobFileMutabilityInfo`, model/source identity is not being collapsed into the
mutable-weight map.

That suggests future `rustane` experiments around cache reuse should treat:

- source/provenance metadata
- mutable-weight metadata
- runtime request packing

as separate levers, not one merged bucket of “MIL metadata.”

### 5.3 `ANEBinaryPoint` is special

The helper explicitly excludes `ANEBinaryPoint` from generic source extraction.

That reinforces the earlier impression that `ANEBinaryPoint` is a special
compiler/runtime attribute family, not ordinary descriptive metadata.

## 6. What this changes in our understanding

Before this pass, we knew the MIL bridge handled mutable weights and source
information in some form.

Now we know more precisely:

- mutable weights are normalized into an absolute-path -> symbol mapping
- top-level source/provenance attributes are flattened separately into a string
  map
- `BlobFileMutabilityInfo` and `ANEBinaryPoint` are intentionally excluded from
  the generic source-info path

That is enough to treat these as distinct internal concepts rather than just
“whatever MIL attributes happen to be present.”

## 7. Best next targets from here

The next useful internal targets, if we stay on this thread, are:

- `ValidateDerivedMILProgram(...)`
  - because it also consumes `RetrieveMutableWeightToSymbol(...)` and should
    show how the compiler validates the mutable-weight mapping
- `RetrieveAbsolutePath(...)`
  - if we want to know exactly how relative MIL weight paths are normalized
- any xref target for `ANEBinaryPoint`
  - to understand why it is treated specially and excluded from generic source
    extraction

For `rustane`, the highest-value next target is probably:

- `ValidateDerivedMILProgram(...)`

because that is the most likely place to see how these extracted MIL metadata
structures are enforced semantically.
