# Hopper Validation Pass 38: `FillNEUnitInfo(...)`

This pass targets:

- `MILOpConverter::FillNEUnitInfo(...)`

The goal is to document the generic NE-unit metadata that Apple fills after the
matmul-specific output-format/channel step.

## 1. High-level result

`FillNEUnitInfo(...)` is a small shared helper that fills generic NE-unit
metadata, not matmul-specific shape semantics.

It is shared by at least:

- `FillNEMatMulUnitInfo(...)`
- `FillNEConvUnitInfo(...)`
- `FillNEPoolUnitInfo(...)`

So this is part of Apple’s generic “NE unit” substrate rather than a matmul-only
decision point.

## 2. What Hopper shows

Size:

- length: `196` bytes
- basic blocks: `7`

The function:

1. calls:
   - `MILOpConverter::FillNEGOCInfo(...)`
2. reads optional `activation`
3. calls:
   - `MILOpConverter::FillActivationInfo(...)`
4. reads optional `binary_point`
5. if present:
   - stores the scalar integer value into `unit_info[0x5a]`
   - sets a presence flag at `unit_info[0x16c] = 1`

## 3. What this means

The helper is filling:

- GOC-related metadata
- activation metadata
- optional binary-point metadata

That makes it clearly separate from the output-format / output-channel decisions
that matter most for matmul layout legality.

## 4. Why this matters for `rustane`

This pass narrows the search space.

If the repo is failing on:

- output layout
- output channel expectations
- rank/dimension legality

then `FillNEUnitInfo(...)` is probably **not** the root cause.

Those problems are more likely in:

- the missing `FillOutputFormatAndChannel(...)` helper
- `ZinIrMatrixMultInfo::*`
- `ZinMatrixMultLayer::ValidateSemantics_Impl(...)`

What `FillNEUnitInfo(...)` *is* likely to affect is:

- activation handling
- quant/binary-point behavior
- generic NE-unit metadata shared across multiple op families
