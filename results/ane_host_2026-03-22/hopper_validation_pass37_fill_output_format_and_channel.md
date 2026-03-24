# Hopper Validation Pass 37: `FillOutputFormatAndChannel(...)`

This pass targets the helper named by Hopper’s decompiler inside the NE matmul
path:

- `MILOpConverter::FillOutputFormatAndChannel(...)`

The important caveat is:

- Hopper did **not** surface this as a standalone recoverable procedure symbol on
  this build

So this pass is based on triangulation from the direct callers and the matrix-
mult unit structures it feeds.

## 1. High-level result

`FillOutputFormatAndChannel(...)` is clearly a real logical helper in the
decompiled NE matmul path, but Hopper did not recover it as an independently
named procedure.

What we can say with confidence is:

- it runs before `FillNEUnitInfo(...)`
- it is part of `FillNEMatMulUnitInfo(...)`
- it must be responsible for populating the unit-info fields that later drive:
  - output format
  - output channel count
  - matrix-mult layer creation / validation

## 2. Direct evidence from `FillNEMatMulUnitInfo(...)`

In `MILOpConverter::FillNEMatMulUnitInfo(...)`, Hopper decompiles:

1. set `arg2[0x8] = 0x45`
2. call `FillOutputFormatAndChannel(...)`
3. call `FillNEUnitInfo(...)`
4. set `arg2[0x64] = 0x12`

Even though `procedure_callees` only surfaced `FillNEUnitInfo(...)` explicitly,
the decompiler clearly identified a logical helper here.

## 3. Why the helper matters

The direct matrix-mult path later depends on:

- `CreateMatMulLayer(...)`
- `ZinMatrixMultLayer::ValidateSemantics_Impl(...)`

and those downstream stages explicitly care about:

- output axis type
- output dimensions
- output channel matching
- KMEM fit

That makes `FillOutputFormatAndChannel(...)` the earliest likely point where the
NE matmul unit is being aligned to the expected output layout/channel contract.

## 4. What we can infer from downstream structures

From `ZinIrMatrixMultUnit` and `ZinIrMatrixMultUnitInfo` construction:

- the unit stores a name/opcode payload
- a format-like field at unit-info offset `0x8`
- one additional configuration field at offset `0x64`
- output tensor semantics later flow into:
  - `ZinIrMatrixMultInfo::ComputeOutputAxisType(...)`
  - `ZinIrMatrixMultInfo::ComputeOutputDimensions(...)`
  - `ZinMatrixMultLayer::ValidateSemantics_Impl(...)`

So the most plausible role of the missing helper is:

- derive output format and output-channel metadata from the MIL op and stash it
  into the NE matmul unit-info before the generic NE-unit filler adds activation
  / binary-point metadata

## 5. What remains unresolved

This is one of the few places where Hopper recovery is still imperfect.

What is **not** directly proven yet:

- the exact code address of the helper
- the exact field offsets it writes
- the exact branching logic inside it

So this pass should be read as:

- a confirmed logical compiler step
- with its role triangulated from caller/callee structure
- but without a clean standalone symbol body yet

## 6. Why this still matters for `rustane`

Even with the symbol gap, this helper is still worth calling out because it is
the earliest output-format/channel-specific step in the dedicated NE matmul
path.

If future repo work tries to steer MIL toward `ane.ne_matmul`, this is one of
the key early places where Apple’s compiler is likely deciding:

- what output layout is expected
- and what output-channel semantics the later layer builder will enforce
