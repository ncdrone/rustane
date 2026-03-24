# Hopper Validation Pass 39: `ZinIrMatrixMultUnitInfo` and Layer Bridge

This pass targets the unit/layer code behind `ZinIrMatrixMultUnitInfo`:

- `ZinIrMatrixMultUnitC2(...)`
- `ZinIrMatrixMultUnit::CreateLayer(...)`
- related matrix-mult unit types surfaced in `ANECompiler`

The goal is to document how the matrix-mult unit-info object turns into a real
layer.

## 1. High-level result

The matrix-mult unit bridge is fairly direct:

- `ZinIrMatrixMultUnitInfo` carries the metadata
- `ZinIrMatrixMultUnit` wraps it into a generic unit object
- `CreateLayer(...)` materializes a `ZinMatrixMultLayer`
- then immediately runs semantic validation on that layer

So the unit-info path is structurally thin. The real acceptance logic still
lands in the layer builder / validator rather than being deeply hidden in the
unit wrapper.

## 2. `ZinIrMatrixMultUnitC2(...)`

Size:

- length: `248` bytes
- basic blocks: `4`

### 2.1 What Hopper shows

The constructor:

- initializes the generic `ZinIrUnit` base
- installs the matrix-mult unit vtable
- copies unit name / string payload
- copies a format-like field from `unit_info[0x8]`
- copies a vector of strings / metadata from `unit_info[0x5..0x6]`
- copies additional fields including one at `unit_info[0x14]`

### 2.2 What this means

The unit object mostly packages already-computed metadata. It does not appear to
be where hard semantic gating happens.

## 3. `ZinIrMatrixMultUnit::CreateLayer(...)`

Size:

- length: `296` bytes
- basic blocks: `10`

### 3.1 What Hopper shows

The function:

1. calls:
   - `CreateOpcode(...)`
2. if opcode creation fails:
   - zeros the out-layer handle
3. otherwise:
   - allocates `ZinMatrixMultLayer`
   - passes in:
     - tensor handle
     - name / metadata from the unit
     - created kernel/opcode
4. immediately calls:
   - `ZinIrOpLayer::ValidateSemantics(...)`
5. wraps the created layer in the returned `RawOrShared` carrier

### 3.2 What this means

There is no long hidden validation pipeline between matrix-mult unit creation
and the final layer object.

Once the unit exists, the compiler quickly commits to a `ZinMatrixMultLayer` and
lets the normal layer semantics validator decide whether the result is legal.

## 4. Why this matters for `rustane`

This pass narrows the direct causes of matmul failures:

- unit-info packaging is probably not the main source of rejection
- the real gates are still:
  - matmul conversion
  - output-axis / output-dimension derivation
  - `ZinMatrixMultLayer::ValidateSemantics_Impl(...)`

So if the repo wants to influence compiler acceptance, it is better to target:

- the shape/layout of emitted MIL
- the builder/validator expectations

rather than assuming a large hidden unit-info transformation is happening here.
