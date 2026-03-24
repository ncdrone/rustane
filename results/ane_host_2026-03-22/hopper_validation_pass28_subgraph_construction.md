# Hopper Validation Pass 28: Cluster-to-Subgraph Construction

This pass targets:

- `SubgraphIdentification::ConstructSubGraphs(...)`
- `SubgraphIdentification::ConstructSubGraph(...)`
- `SubgraphIdentification::RemoveInputNodeDrivingBothExternalAndInternalNodes(...)`
- `SubgraphIdentification::RemoveIllegalInternalNodes(...)`
- `SubgraphIdentification::IdentifyInputOutputNodes(...)`
- `SubgraphIdentification::ExtractSubgraph(...)`
- `SubgraphIdentification::DropPureInputConcats(...)`
- `SubgraphIdentification::RemovePureInOutNodes(...)`

The goal is to document how Apple turns a refined cluster into a concrete
`Subgraph`.

## 1. High-level result

The cluster-to-subgraph bridge is a two-layer mechanism:

- `ConstructSubGraphs(...)` dispatches cluster-by-cluster through a virtual
  construction hook on the splitter object
- the generic fallback / helper path is `ConstructSubGraph(...)`, which performs
  repeated sanitation before extracting the final `Subgraph`

The important point for `rustane` is that Apple does not treat cluster
construction as a trivial container conversion. It performs a substantial amount
of graph cleanup before accepting a subgraph as valid.

## 2. `ConstructSubGraphs(...)`

Size:

- length: `364` bytes
- basic blocks: `10`

### 2.1 What Hopper shows

`ConstructSubGraphs(...)`:

1. iterates the refined cluster vector
2. calls a virtual at `this + 0xb8` with:
   - graph
   - cluster set
   - output subgraph vector
   - two boolean-ish flags (`0`, `1`)
3. if construction fails, sets the status flag and logs:
   - `SIP error during subgraph identification. cluster index: %zu`

There is also a symbolized base helper:

- `SubgraphIdentification::ConstructValidSubgraphsFromCluster(...)`

but on this build it is just a stub returning `5`.

### 2.2 What this means

The actual splitter-specific construction logic likely lives in a subclass
override that Hopper did not surface with a distinct public symbol. What we can
see directly is:

- a generic dispatcher
- a dead/simple base stub
- and the generic sanitation/extraction routine used lower in the stack

That is enough to establish the structure even if the exact override symbol is
missing.

## 3. `ConstructSubGraph(...)`

Size:

- length: `152` bytes
- basic blocks: `5`

### 3.1 What Hopper shows

This routine repeatedly sanitizes a cluster until its size stabilizes, then
extracts the subgraph:

1. loop:
   - `RemoveInputNodeDrivingBothExternalAndInternalNodes(...)`
   - `RemoveIllegalInternalNodes(...)`
2. once cluster size stops changing:
   - `IdentifyInputOutputNodes(...)`
   - `ExtractSubgraph(...)`
   - `DropPureInputConcats(...)`
   - `RemovePureInOutNodes(...)`

### 3.2 What this means

The construction path is intentionally iterative.

Apple expects cluster cleanup to reveal new cleanup opportunities, so it runs
the sanitation loop until it reaches a fixed point before doing final
subgraph extraction.

## 4. `RemoveInputNodeDrivingBothExternalAndInternalNodes(...)`

Size:

- length: `756` bytes
- basic blocks: `38`

### 4.1 What Hopper shows

This helper looks for an input-side layer whose influence crosses both:

- inside the candidate cluster
- and outside it

It:

- inspects incoming layers of each cluster layer
- checks whether those producers are inside/outside the cluster
- compares dominance relationships with:
  - `ZinIrNgraph<...>::IsDominanceRelationship(...)`
- removes offending layers from the cluster

### 4.2 What this means

Apple is explicitly forbidding ambiguous “shared boundary” input nodes inside a
candidate subgraph. This is another example of the framework preferring clean
cut boundaries over heroic later repair.

## 5. `RemoveIllegalInternalNodes(...)`

Size:

- length: `968` bytes
- basic blocks: `49`

### 5.1 What Hopper shows

This helper first gathers several categories of problematic layers:

- output no-ops
  - `FindOutputNoOps(...)`
- unsupported concats
  - `FindUnsupportedConcats(...)`

It then builds multiple temporary sets and removes layers that violate the
internal-cluster assumptions, including unsupported no-op / concat structure.

### 5.2 What this means

Some cluster cleanup is not about edges or schedule shape at all.

Apple is also filtering out operations that are structurally awkward for a clean
subgraph representation, especially no-op and concat patterns.

## 6. `IdentifyInputOutputNodes(...)`

Size:

- length: `272` bytes
- basic blocks: `18`

### 6.1 What Hopper shows

The function computes two boundary sets inside the `Subgraph`:

- inputs: cluster layers with at least one incoming producer outside the cluster
- outputs: cluster layers with at least one outgoing consumer outside the cluster

It does this by checking:

- incoming layers directly
- external users via the debug-info string table walk

### 6.2 What this means

The subgraph boundary definition is purely graph-topological here:

- outside producer => input boundary
- outside consumer => output boundary

This gives the later extraction pass a clean separation between internal nodes
and boundary nodes.

## 7. `ExtractSubgraph(...)`

Size:

- length: `572` bytes
- basic blocks: `29`

### 7.1 What Hopper shows

`ExtractSubgraph(...)` converts the now-sanitized cluster plus its boundary sets
into a full `Subgraph` record.

It:

- copies the cluster set
- iterates output-side and non-output-side layers separately
- builds additional bookkeeping around layers not present in the output map
- populates the `Subgraph`’s internal trees / maps / hash sets

Failure paths return `3` and log through the standard ANE logging path.

### 7.2 What this means

This is the actual materialization step. By the time execution reaches here, the
cluster has already been normalized enough that extraction is mostly about
bookkeeping rather than semantic triage.

## 8. `DropPureInputConcats(...)`

Size:

- length: `484` bytes
- basic blocks: `27`

### 8.1 What Hopper shows

This pass removes concat nodes from the extracted subgraph when they are pure
input-side scaffolding:

- concat op kind check via op-type `7`
- input lookups against the subgraph’s tensor/layer maps
- erase from:
  - all-layers set
  - input-boundary set
  - output-boundary set

### 8.2 What this means

Apple does not want concat nodes that only serve as inbound packaging cluttering
the final subgraph representation. Those are normalized away after extraction.

## 9. `RemovePureInOutNodes(...)`

Size:

- length: `364` bytes
- basic blocks: `23`

### 9.1 What Hopper shows

This pass removes nodes that are simultaneously trivial inputs and trivial
outputs.

The pattern it checks is:

- layer in input-boundary set
- layer in output-boundary set
- all incoming producers outside the output set

Those nodes are then erased from the main layer set and both boundary sets.

### 9.2 What this means

The extracted `Subgraph` is meant to represent meaningful internal work, not
degenerate one-hop boundary stubs. This pass trims those degenerate nodes away.

## 10. What this changes in our understanding

### 10.1 Cluster construction is a cleanup pipeline

The pressure-based splitter does not hand clusters directly to the graph rewrite
stage. There is a dedicated construction pipeline that:

- sanitizes boundary-sharing inputs
- removes illegal internal patterns
- identifies explicit boundary nodes
- extracts the subgraph
- removes trivial concat and pure in/out scaffolding

### 10.2 Apple separates “find a cluster” from “accept a subgraph”

This is a useful conceptual distinction for `rustane`.

The internal compiler first finds a region worth splitting, then separately
checks whether that region can be represented as a clean subgraph boundary.

That helps explain why some apparently plausible split regions may still be
discarded or heavily rewritten before use.
