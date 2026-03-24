# Hopper Validation Pass 34: Compiler-Side Matmul Path

This pass targets the direct compiler-side matmul path in `ANECompiler`:

- `MILOpConverter::Matmul(...)`
- `MILOpConverter::NEMatMul(...)`
- `MILOpConverter::FillNEMatMulUnitInfo(...)`
- `MILOpConverter::Einsum(...)`
- `MILOpConverter::EinsumEquation::ParseEquation(...)`
- `ZinBuilder::CreateMatMulLayer(...)`
- `ZinMatrixMultLayer::ValidateSemantics_Impl(...)`

It also distinguishes those from a later optimizer path:

- `ZinIrOpt::ReplaceMatmulWithConv::Execute(...)`

The goal is to explain how Apple handles matmul at the compiler boundary that is
closest to the repo’s actual failures.

## 1. High-level result

Apple has **multiple** matmul-related compiler paths:

- a general `matmul` converter
- a restricted `einsum` converter with an allowlist of equations
- a dedicated `ane.ne_matmul` / `NEMatMul` path
- a later optimizer that can rewrite some matmuls into convs

The most important repo-specific conclusion is:

- the accepted matmul shapes are **much more constrained** than “any two tensors
  with compatible inner dimensions”
- rank, layout conversion, output-channel rules, depth rules, compressed output
  handling, and const-input format rules all show up explicitly

So the repo’s matmul pain is not just about one width ceiling. It sits on top
of a fairly opinionated compiler path.

## 2. `MILOpConverter::Matmul(...)`

Size:

- length: `5456` bytes
- basic blocks: `181`

### 2.1 What Hopper shows

The main steps are:

1. initialize MIL unit-builder state
2. retrieve operation parameters:
   - `x`
   - `y`
   - `transpose_x`
   - `transpose_y`
3. retrieve tensor shapes for `x` and `y`
4. compute input ranks
5. reject any input rank outside `1..4`
   - `ANE can only support matmul with input tensors rank between 1 and 4`
6. branch based on shape/rank structure

The function clearly has at least two major lowering branches:

- a **linear-like** branch
  - visible via strings such as:
    - `__@linear_output_layout_convert`
  - and calls:
    - `ZinMILUnitBuilder::InsertLinear(...)`
    - `ZinMILUnitBuilder::ConvertToDefaultLayout(...)`
- a more general **matmul path**
  - with explicit reshape / squeeze / finalize behavior
  - visible through strings such as:
    - `__@squeeze`

### 2.2 What this means

Apple is not treating all matmuls the same.

Some rank/layout patterns are canonicalized into a linear-style path, while
others stay on the fuller matmul route. That matters for `rustane` because small
graph changes could move a case from one path to another.

## 3. `MILOpConverter::NEMatMul(...)`

Size:

- length: `1076` bytes
- basic blocks: `20`

### 3.1 What Hopper shows

This path builds a dedicated `ZinIrNEMatMulUnitInfo` and then emits an
operation named:

- `__@nematmul`

The function:

1. zero-initializes `ZinIrNEMatMulUnitInfo`
2. calls:
   - `MILOpConverter::FillNEMatMulUnitInfo(...)`
3. declares operation inputs `x` and `y`
4. builds the NE matmul unit through:
   - `ZinMILUnitBuilder::CreateUnit<ZinIrNEMatMulUnitInfo>(...)`
5. finalizes via:
   - `ZinMILUnitBuilder::Finalize(...)`

### 3.2 What this means

`ane.ne_matmul` is a real dedicated unit path, not just a naming alias layered
on top of the generic matmul converter.

That makes it a good future target if the repo wants to understand whether its
generated MIL can be nudged into Apple’s more specialized NE matmul path.

## 4. `MILOpConverter::FillNEMatMulUnitInfo(...)`

Size:

- length: `84` bytes
- basic blocks: `1`

### 4.1 What Hopper shows

This helper is small but revealing.

It:

- sets one unit-info field to `0x45`
- calls:
  - `FillOutputFormatAndChannel(...)`
  - `FillNEUnitInfo(...)`
- sets another field to `0x12`

### 4.2 What this means

The dedicated NE matmul path has its own explicit unit typing/configuration,
even if Hopper does not recover the semantic names of those constants.

## 5. `MILOpConverter::Einsum(...)`

Size:

- length: `1372` bytes
- basic blocks: `43`

### 5.1 What Hopper shows

This path is much stricter than general matmul.

It:

1. declares a single operation input named:
   - `values`
2. retrieves the `equation` parameter
3. parses it through:
   - `MILOpConverter::EinsumEquation::ParseEquation(...)`
4. checks against a static allowlist of supported equations
5. rejects unsupported equations with:
   - `Unsupported einsum equation: %s`
6. validates expected input ranks against the chosen equation
7. then emits a `ZinIrMatrixMultUnitInfo` unit and finalizes it

The visible supported example hardcoded in the function is:

- `chk,khq->chq`

### 5.2 What this means

Apple’s einsum support here is not generic. It is a constrained equation-to-
matrix-mult lowering path with explicit supported-equation enumeration.

That means `rustane` should not treat `einsum` as a free-form escape hatch.

## 6. `EinsumEquation::ParseEquation(...)`

Size:

- length: `548` bytes
- basic blocks: `29`

### 6.1 What Hopper shows

The parser explicitly checks for:

- missing `,`
- missing `->`
- too many indexes
- unknown indexes in the output

Corresponding strings include:

- `invalid einsum equation: missing ',' in the input`
- `invalid einsum equation: missing '->'`
- `invalid einsum equation: too many indexes`
- `invalid einsum equation: unknown index in the output`

### 6.2 What this means

The einsum path is parse-validated before any meaningful lowering work begins,
which is consistent with the allowlist-based design above.

## 7. `ZinBuilder::CreateMatMulLayer(...)`

Size:

- length: `932` bytes
- basic blocks: `30`

### 7.1 What Hopper shows

This is the lower builder that materializes the actual matrix-mult layer.

It:

1. requires exactly two bottoms
   - `Matrix mult. layer can only have two bottoms`
2. computes output axis types
3. computes output dimensions
4. allocates the output tensor
5. constructs the `ZinMatrixMultLayer`
6. validates its semantics

The direct failure strings are:

- `ComputeOutputAxisType in CreateMatMulLayer failed`
- `ComputeOutputDimensions in CreateMatMulLayer failed`

### 7.2 What this means

Even after MIL-level conversion, matmul still has a dedicated builder phase with
its own axis/dimension failure modes. So some repo failures could plausibly be
coming from this lower stage rather than from MIL conversion proper.

## 8. `ZinMatrixMultLayer::ValidateSemantics_Impl(...)`

Size:

- length: `576` bytes
- basic blocks: `25`

### 8.1 What Hopper shows

This is one of the most directly relevant functions in the whole pass.

It explicitly checks:

- exactly two inputs
- output channel of the result must equal input A’s channel dimension
  - `Error: invalid output channel = %zd. Expecting it to be input A's channel dimension = %zd for MatMult`
- depth dimension must not exceed `1`
  - `Error: depth > 1 is not supported for MatMult inputs but get dim_A.d = %zd, dim_B.d = %zd`
- selected dimensions of A and B must be either `1` or equal
  - `Error: invalid dim %s for input tensor A (%zd) and B (%zd), must be 1 or equal to one another`
- output channel of input A must fit in KMEM
  - `Error: the output channel of input A (%zu kB) can not fit the Kmem (%zu kB)`

### 8.2 What this means

This function directly explains several kinds of repo-relevant shape failures:

- depth > 1 rejection
- output-channel mismatch rejection
- dimension-broadcast mismatch rejection
- KMEM capacity rejection

These are not vague downstream symptoms. They are explicit semantic guards in
the matrix-mult layer validator.

## 9. `ReplaceMatmulWithConv` is not the primary compiler path

The strings:

- `constIn should be present for Matmul`
- `Incompatible Dims for Matmul with bias for %s`

land in:

- `ZinIrOpt::ReplaceMatmulWithConv::Execute(...)`
- `ZinIrOpt::ReplaceMatmulWithConv::GetConstDataIndexStrides(...)`

Those are later optimization/rewrite routines, not the primary MIL conversion
path.

That distinction matters:

- if a `rustane` graph dies in the main compiler-side matmul path, these strings
  may be irrelevant
- if it survives into matmul-to-conv rewriting, then const-input format and
  bias compatibility become relevant

## 10. What this changes in our understanding

### 10.1 The main matmul path is more constrained than expected

The strongest practical result is that Apple’s compiler-side matmul path is
guarded by explicit checks on:

- input rank `1..4`
- output-channel identity
- depth dimension `<= 1`
- dimension equality/broadcast rules
- KMEM capacity
- output axis/dimension derivation

That is directly useful for `rustane`.

### 10.2 There are multiple internal matmul flavors

We now have direct evidence for:

- generic matmul
- equation-restricted einsum
- dedicated `ane.ne_matmul`
- later matmul-to-conv rewrite

So “matmul path” in Apple’s compiler really means a family of paths, not one.

### 10.3 The deepest remaining matmul-specific questions are now narrower

The best next targets from here are probably:

- `FillOutputFormatAndChannel(...)`
- `FillNEUnitInfo(...)`
- the lower/unit validators behind `ZinIrMatrixMultUnitInfo`
- any direct users of:
  - `ComputeOutputAxisType in CreateMatMulLayer failed`
  - `ComputeOutputDimensions in CreateMatMulLayer failed`

Those would tell us more about how to steer `rustane` MIL toward accepted
matmul layouts rather than just explaining failures after the fact.
