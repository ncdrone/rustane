# V3 Optimization Gossip
# Each iteration reads this FIRST and appends findings at the end.
# This is your memory between iterations — the experiments TSV has what was tried,
# this file has WHY things worked or failed.

## Current State
tok/s: 1.13 | ms/layer: 15.2 | baseline: 0.7 | wins: 3 | iterations: 9

## Bottleneck (update after each win)
MLA attention:    ~5ms  (31%)  — memory-bound AMX sgemv, sgemm ready but invisible at seq_len=5
Metal dispatch:   ~3ms  (19%)  — GPU compute + waitUntilCompleted sync
Residual compute: ~3ms  (19%)  — RMSNorm, routing, residuals
Conversion:       ~2ms  (13%)  — hidden by pipeline overlap, deferred to FFN phase only (was 7ms serial)
Expert pread:     ~2ms  (13%)  — hidden by shared FFN overlap (was 3ms serial)

## Dead Ends (append-only — DO NOT RETRY these or variations of them)
- [iter 2] manual NEON sgemv: 3x slower than AMX BLAS for [128,512]. AMX dispatch overhead only 1.3μs/call.
- [iter 4] f16 shared expert bypass: sgemv_f16 uses scalar to_f32() not NEON FCVTL. 34% regression (0.53 tok/s).
- [iter 4] f16 direct path with FCVTL: 1-core conversion (80 GB/s) loses to 12-core rayon (300 GB/s) + pipeline overlap. Even vectorized, halving traffic can't compensate for losing parallelism.
- [iter 5] alloc elimination (rmsnorm_into + final_norm borrow): correct refactor, 126 allocs/tok = ~50μs, invisible at 1300ms/tok. Already committed as code quality improvement.
- ALL f16 CPU compute paths are dead ends. f16 requires Metal GPU, not CPU tricks.
- [iter 7] overlap Metal dispatch with convert instead of shared FFN: no improvement. M4 Max unified memory handles concurrent AMX+rayon without BW contention. CPU-side BW scheduling rearrangements are dead ends.
- [iter 8] rayon per-head W_UK/W_UV parallelization: <1% improvement. Per-head sgemv on [128,512] matrices is ~1μs (L2/L3 resident, hardware prefetch), total ~260μs/layer. Parallelizing tiny sequential BLAS calls doesn't help.
- [iter 9] overlap convert(layer 0) with last layer's FFN: no improvement. Last layer (60) is dense FFN (CPU sgemv), not MoE. Rayon convert threads compete for same CPU memory bandwidth as dense sgemv. The ~2.5ms saved from skipping serial convert(0) is eaten by BW contention during dense FFN.
- GENERAL: overlap only works when the two tasks use DIFFERENT hardware resources. Dense FFN + rayon convert = both CPU memory bandwidth = contention. This closes the "overlap within same resource" family of ideas.

## What Works (proven patterns — build on these)
- [iter 2] thread::scope overlap: 21μs overhead per scope, can hide up to 7ms of work. Key: the overlapped work must use a DIFFERENT hardware resource than the main thread.
- [iter 5] pread || shared_FFN overlap: SSD I/O (NVMe controller) and memory BW (DRAM) don't compete. 5.8ms/layer saved.
- GENERAL PRINCIPLE: look for operations that use different hardware resources and run them simultaneously. Resources: P-cores (AMX/NEON), E-cores, GPU, NVMe, DRAM bandwidth.

## Bugs Found (for manual sessions — do not fix, just document)
- test_v3_validation L2 failure: layers_f32 empty in lazy mode (pre-existing, not caused by optimizations)

## Suggested Next (ideas agents couldn't try — pick from here if relevant)
- Batch multiple sgemv calls in MLA attention into fewer sgemm calls (Q LoRA W_qa + W_qb back-to-back)
- Move RMSNorm to NEON intrinsics (avoid vDSP call overhead for small vectors)
- Wire ExpertPool (pool.rs) to cache hot experts in RAM instead of pread every token
- Overlap lm_head sgemv with next token's embedding lookup (lm_head is [151936,7168] = 4.35 GB, takes ~11ms)
- Profile actual per-component timing within MLA (instrument inside mla_forward_decode, measure W_qa/W_qb/W_kva/W_UK/W_UV/O individually) to find the real bottleneck breakdown

## Iteration Log
[iter 1] TIMEOUT at 60min — spent entire budget on diagnostic V3 benchmark. Never coded.
[iter 2] RESULT: v3-pipeline-decode — KEPT, 0.7→0.8 tok/s (14%). thread::scope overlaps convert(N+1) with compute(N). 1.4ms/layer saved.
[iter 2] INSIGHT: thread::scope (21μs) is far lighter than channels. Pipeline works when overlapped work < main work and they use different resources.
[iter 3] TIMEOUT at 60min — sgemm attention work incomplete. Benchmark cycle too slow for 60min budget.
[iter 4] RESULT: v3-manual-gemv — REVERTED 0.67 tok/s. Manual NEON 3x slower than AMX. Don't try to beat hardware accelerators.
[iter 4] RESULT: v3-sgemm-attention — NO EFFECT 0.81 tok/s. Replaced scalar f64 loops with sgemm. Correct but scalar loops only 0.1ms/layer at seq_len=5. Will matter at seq_len>100.
[iter 4] RESULT: v3-f16-shared-expert — REVERTED 0.53 tok/s. sgemv_f16 scalar conversion in hot path.
[iter 5] RESULT: v3-elim-norm-allocs — NO EFFECT 0.79 tok/s. rmsnorm_into + final_norm borrow. 126 allocs eliminated but ~50μs total.
[iter 5] RESULT: v3-overlap-pread-ffn — KEPT, 0.8→1.06 tok/s (34%). Overlap expert pread with shared FFN. SSD I/O and memory BW use different hardware.
[iter 5] INSIGHT: the remaining bottleneck is memory-bandwidth-bound (MLA attention + Metal dispatch + residual). CPU tricks exhausted — need Metal f16 GEMV or ExpertPool for next jump.
[iter 6] RESULT: v3-deferred-convert — KEPT, 1.06→1.13 tok/s (6.6%). Deferred conversion thread to overlap with FFN phase only (not MLA). MLA sgemv is memory-bandwidth-bound; concurrent conversion stole ~15% BW. Deferring to FFN phase (Metal GPU + SSD pread, no BW contention) recovers 1.0ms/layer.
[iter 6] INSIGHT: conversion overlap must be SELECTIVE — only overlap with operations that don't compete for DRAM bandwidth. MLA attention uses AMX which saturates DRAM. FFN uses Metal GPU + NVMe which don't.
[iter 7] RESULT: v3-overlap-metal-convert — NO EFFECT 1.11 tok/s. Split moe_ffn into prepare(shared+pread) + dispatch(Metal), overlap convert with Metal only. Both pipelines = 7ms: max(3ms convert, 7ms FFN)=7ms vs 4ms shared+max(3ms convert, 3ms Metal)=7ms.
[iter 7] INSIGHT: M4 Max unified memory handles concurrent AMX sgemv + rayon conversion WITHOUT measurable BW contention. The hypothesis that shared FFN BW degrades during concurrent conversion was wrong — Apple Silicon's memory controller distributes bandwidth efficiently across cores. CPU-side BW contention is a dead end for optimization.
[iter 8] RESULT: v3-rayon-perhead-sgemv — NO EFFECT 1.14 tok/s. Parallelized 128 per-head W_UK/W_UV sgemv calls with rayon par_chunks_mut. Hypothesis: 128 sequential BLAS calls × 1.3μs dispatch overhead = 166μs/call-site × 2 × 61 layers = 20ms. Reality: per-head sgemv on [128,512] matrices takes ~1μs total (not 4.5μs) because data stays in L2 cache from sequential access + hardware prefetch eliminates dispatch stalls. Actual W_UK+W_UV time is ~260μs/layer, not 1.2ms.
[iter 8] INSIGHT: Small sgemv calls ([128,512] = 256 KB) are MUCH faster than bandwidth model predicts because: (1) hardware prefetcher pre-loads next head's matrix during current head's compute, (2) 33 MB total W_UK data fits in L3 so no DRAM round-trips after first head, (3) AMX dispatch overhead ~1.3μs is amortized when data is L2/L3 resident. Parallelizing small sequential BLAS is a dead end — the benefit is <1% at this matrix size.
[iter 9] RESULT: v3-overlap-convert0-lastlayer — NO EFFECT 1.1 tok/s. Overlap convert(layer 0) with last layer (60) FFN via thread::scope, pre-warming buf_a for next token to skip serial convert(0). Hypothesis: save ~2.5ms per token. Reality: layer 60 is dense FFN (CPU sgemv, not MoE Metal+pread), so rayon convert threads compete for CPU memory bandwidth with dense sgemv. Net effect zero.
[iter 9] INSIGHT: The overlap pattern ONLY works when the two concurrent tasks use different hardware resources. Dense FFN layers use CPU memory bandwidth (same as rayon conversion). MoE layers use Metal GPU + NVMe (different from rayon conversion). This definitively closes the "rearrange CPU overlap timing" family of optimizations — all remaining improvements require architectural changes (ExpertPool, Metal attention kernel, f16 compute path).
