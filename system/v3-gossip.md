# V3 Optimization Gossip
# Each iteration reads this FIRST and appends findings at the end.
# This is your memory between iterations — the experiments TSV has what was tried,
# this file has WHY things worked or failed.

## Current State
tok/s: 1.06 | ms/layer: 16.2 | baseline: 0.7 | wins: 2 | iterations: 6

## Bottleneck (update after each win)
MLA attention:    ~5ms  (31%)  — memory-bound AMX sgemv, sgemm ready but invisible at seq_len=5
Metal dispatch:   ~3ms  (19%)  — GPU compute + waitUntilCompleted sync
Residual compute: ~3ms  (19%)  — RMSNorm, routing, residuals
Conversion:       ~3ms  (19%)  — hidden by pipeline overlap (was 7ms serial)
Expert pread:     ~2ms  (12%)  — hidden by shared FFN overlap (was 3ms serial)

## Dead Ends (append-only — DO NOT RETRY these or variations of them)
- [iter 2] manual NEON sgemv: 3x slower than AMX BLAS for [128,512]. AMX dispatch overhead only 1.3μs/call.
- [iter 4] f16 shared expert bypass: sgemv_f16 uses scalar to_f32() not NEON FCVTL. 34% regression (0.53 tok/s).
- [iter 4] f16 direct path with FCVTL: 1-core conversion (80 GB/s) loses to 12-core rayon (300 GB/s) + pipeline overlap. Even vectorized, halving traffic can't compensate for losing parallelism.
- [iter 5] alloc elimination (rmsnorm_into + final_norm borrow): correct refactor, 126 allocs/tok = ~50μs, invisible at 1300ms/tok. Already committed as code quality improvement.
- ALL f16 CPU compute paths are dead ends. f16 requires Metal GPU, not CPU tricks.

## What Works (proven patterns — build on these)
- [iter 2] thread::scope overlap: 21μs overhead per scope, can hide up to 7ms of work. Key: the overlapped work must use a DIFFERENT hardware resource than the main thread.
- [iter 5] pread || shared_FFN overlap: SSD I/O (NVMe controller) and memory BW (DRAM) don't compete. 5.8ms/layer saved.
- GENERAL PRINCIPLE: look for operations that use different hardware resources and run them simultaneously. Resources: P-cores (AMX/NEON), E-cores, GPU, NVMe, DRAM bandwidth.

## Bugs Found (for manual sessions — do not fix, just document)
- test_v3_validation L2 failure: layers_f32 empty in lazy mode (pre-existing, not caused by optimizations)

## Suggested Next (ideas agents couldn't try — pick from here if relevant)
- Overlap Metal expert dispatch with next layer's conversion (Metal GPU || CPU rayon — different resources)
- Batch multiple sgemv calls in MLA attention into fewer sgemm calls (Q LoRA W_qa + W_qb back-to-back)
- Pre-compute RMSNorm scales for all layers at token start (tiny tensors, eliminate per-layer overhead)
- Move RMSNorm to NEON intrinsics (avoid vDSP call overhead for small vectors)

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
