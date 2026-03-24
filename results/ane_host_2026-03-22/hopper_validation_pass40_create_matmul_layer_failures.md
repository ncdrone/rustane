# Hopper Validation Pass 40: Direct Owners of `CreateMatMulLayer` Axis/Dimension Failures

This pass targets the direct owners of the two builder failures we already found:

- `ComputeOutputAxisType in CreateMatMulLayer failed`
- `ComputeOutputDimensions in CreateMatMulLayer failed`

The relevant functions are:

- `ZinIrMatrixMultInfo::ComputeOutputAxisType(...)`
- `ZinIrMatrixMultInfo::ComputeOutputDimensions(...)`
- `ZinBuilder::CreateMatMulLayer(...)`

## 1. High-level result

These failures are owned by very small, concrete functions.

That is good news for the repo:

- the rules are not buried in an enormous opaque pass
- the output-axis and output-dimension contracts are comparatively readable

## 2. `ZinBuilder::CreateMatMulLayer(...)`

Size:

- length: `932` bytes
- basic blocks: `30`

### 2.1 What Hopper shows

The builder:

1. requires exactly two input tensors
   - `Matrix mult. layer can only have two bottoms`
2. computes output axis types through the matrix-mult info object
3. computes output dimensions through the matrix-mult info object
4. allocates the output tensor and layer
5. validates the resulting layer

This is the direct owner of the two builder failure strings.

## 3. `ZinIrMatrixMultInfo::ComputeOutputAxisType(...)`

Size:

- length: `632` bytes
- basic blocks: `26`

### 3.1 What Hopper shows

This function expects exactly two input axis-type/dimension records.

It then:

1. checks tensor-format / axis-type flags on both inputs
2. requires compatible axis packing for the participating dimensions
3. requires selected dimensions to be:
   - equal
   - or broadcast-compatible
4. if successful, writes an output axis-type pack with:
   - one axis type copied from input A
   - one copied from input B
   - and symbolic dimension propagation when necessary

If any of those tests fail, it returns `3`.

### 3.2 What this means

The axis-type failure is mostly about:

- two-input expectation
- compatible axis packing
- broadcast/equality semantics on specific axes

So `CreateMatMulLayer` can fail even before raw dimensions are considered if
the axis-type layer metadata is not coherent.

## 4. `ZinIrMatrixMultInfo::ComputeOutputDimensions(...)`

Size:

- length: `76` bytes
- basic blocks: `3`

### 4.1 What Hopper shows

This function is simple.

It also expects exactly two input dimension records, then computes output dims
as:

- output dim 0 = max of one pair of input dims
- output dim 1 = one preserved dim from input A
- output dim 2 = max of another pair of input dims
- output dim 3 = one preserved dim from input B
- output dim 4 = one preserved tail dim from input A

If the input record count is wrong, it returns `3`.

### 4.2 What this means

The output-dimension failure is not doing fancy algebra. It is mostly enforcing
a fixed matrix-mult shape contract over two structured tensor descriptors.

That makes it practical to reason about from the repo side.

## 5. Why this matters for `rustane`

This is one of the more directly actionable passes in the whole reverse-
engineering effort.

It means that if `rustane` wants to avoid builder-side matmul failures, the
relevant questions are now concrete:

- are there exactly two bottoms?
- are the axis-type packs compatible?
- are the expected dimensions equal or broadcast-compatible?
- do the output channels / preserved dims line up with Apple’s fixed output-shape
  contract?

That is much easier to test experimentally than a vague “the compiler rejected
matmul.”
