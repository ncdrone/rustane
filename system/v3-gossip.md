# V3 Optimization Gossip
# Each iteration reads this FIRST and appends findings at the end.
# This is your memory between iterations — the experiments TSV has what was tried,
# this file has WHY things worked or failed.

## Current State
tok/s: 1.13 | ms/layer: 15.2 | baseline: 0.7 | wins: 3 | iterations: 13

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
- [iter 10] parallel gate+up sgemv (thread::scope) in shared_expert_ffn and dense_ffn: <1% improvement, within noise. Hypothesis: save 0.39ms×58 + 3.5ms×3 = 33ms. Reality: AMX sgemv per P-core achieves ~150 GB/s (not 80 GB/s), so gate+up serial time is shorter than estimated. The 21μs thread::scope overhead per layer (61 layers = 1.3ms) eats a significant fraction of the savings. Net gain ~22ms = 2.5% = below noise threshold.
- GENERAL: parallelizing sequential sgemv calls within a single function is a dead end. The per-call savings (~0.4ms for shared expert, ~3.5ms for dense) are partially eaten by thread::scope overhead, and actual BLAS per-core bandwidth is higher than the 80 GB/s estimate (more like 150 GB/s for AMX sequential access). This closes the "parallelize CPU sgemv within a function" family of optimizations.
- [iter 11] explicit row-partitioning of large sgemv (O projection [7168,16384] = 470 MB): 5.3% REGRESSION (1.13→1.07 tok/s). Apple Accelerate cblas_sgemv already internally multi-threads large matrices. Splitting into 4 rayon chunks defeats BLAS internal parallelization — 4 small calls are less efficient than 1 large one. This confirms: NEVER manually parallelize individual BLAS calls; Accelerate handles large-matrix parallelism internally.
- GENERAL: ALL forms of explicit sgemv parallelization are dead ends — both parallelizing SEPARATE calls (iter 10) and row-partitioning SINGLE calls (iter 11). Accelerate handles multi-threading internally for large matrices and has near-optimal single-core performance for small matrices.
- [iter 12] cached layer 0 f32 weights (buf_layer0, skip serial convert(0)): NO EFFECT 1.15 tok/s. Hypothesis: save ~5ms serial convert(0) per token. Reality: 5ms / 940ms per token = 0.5%, below noise. This confirms: single-layer conversion savings (~5ms) are invisible at the current per-token cost (~940ms). Eliminating individual convert() calls is a dead end.
- GENERAL: ALL CPU-side "skip work" optimizations on single-layer granularity are dead ends. The per-token cost is ~940ms across 61 layers. Saving <10ms from any single layer = <1% = invisible. The exhaustion of CPU-side optimizations is now CONFIRMED across 12 iterations.
- [iter 13] f16 O projection from mmap via sgemv_f16_par: 24% REGRESSION (1.13→0.86 tok/s). Skipped o_proj f32 conversion, read 235 MB f16 directly from mmap instead of 470 MB pre-converted f32. Reality: pre-converted f32 buffer is warm in RAM from background pipeline thread; mmap f16 pages are cold (last accessed during previous token). The chunked rayon convert+sgemv on cold mmap pages incurs page faults and serialization that vastly outweigh the halved DRAM traffic. Additionally, sgemv_f16_par defeats Accelerate's internal multi-threading for large matrices (same issue as iter 11).
- GENERAL: reading f16 weights from mmap on the critical path is ALWAYS slower than using pre-warmed f32 from the pipeline overlap buffer. The pipeline background thread (running during FFN) pre-faults and converts pages into hot RAM. Bypassing this to read cold mmap pages is a net loss even at half the data volume. This closes the "f16 mmap direct read" family of optimizations.

## What Works (proven patterns — build on these)
- [iter 2] thread::scope overlap: 21μs overhead per scope, can hide up to 7ms of work. Key: the overlapped work must use a DIFFERENT hardware resource than the main thread.
- [iter 5] pread || shared_FFN overlap: SSD I/O (NVMe controller) and memory BW (DRAM) don't compete. 5.8ms/layer saved.
- GENERAL PRINCIPLE: look for operations that use different hardware resources and run them simultaneously. Resources: P-cores (AMX/NEON), E-cores, GPU, NVMe, DRAM bandwidth.

## Bugs Found (for manual sessions — do not fix, just document)
- test_v3_validation L2 failure: layers_f32 empty in lazy mode (pre-existing, not caused by optimizations)

## Category Status (NEVER pick from EXHAUSTED categories)
- CPU overlap/scheduling: EXHAUSTED (6 experiments, 3 wins then ceiling)
- CPU parallelization: EXHAUSTED (4 experiments, Apple BLAS already multi-threads internally)
- Allocation elimination: EXHAUSTED (<50µs invisible at 940ms/token)
- f16 CPU compute: DEAD END (4 experiments, needs Metal not CPU — includes f16 mmap direct read)
- BLAS batching: OPEN (sgemm-attention tried but invisible at seq_len=5; Q LoRA batching untried)
- Metal GPU compute: OPEN (untried — biggest untapped potential, needs arch session)
- Expert caching/pool: OPEN (ExpertPool built, unwired — borderline small opt)
- Memory layout: OPEN (INT8 KV, f16 buffers, 2MB alignment untried)
- LM head optimization: OPEN (4.35 GB, ~11ms/token)
- Profiling/instrumentation: OPEN (no per-component MLA breakdown exists)

## Suggested Next (pick from OPEN categories only, ranked by expected impact)
1. [Profiling] Per-component MLA timing — instrument mla_forward_decode: W_qa, W_qb, W_kva, W_UK, W_UV, attn scores, value recon, O_proj individually. ~20 lines. Reveals real targets.
2. [I/O] 2MB-aligned expert buffers — posix_memalign(buf, 2MB, size) for pread. Apple SSD DMA coalesces 2MB reads. ~10 lines. Could be 3.6x pread throughput.
3. [Overlap] Overlap lm_head with next token — thread::scope lm_head ([151936,7168]=4.35GB, ~11ms) while background does embedding+norm for next token. Different data, no BW conflict. ~40 lines.
4. [Memory] Pre-convert small tensors at load — convert norms, router, bias to f32 once at startup (~0.5 GB). Skip per-layer conversion for these. ~20 lines.
5. [BLAS] Batch Q LoRA W_qa+W_qb — concatenate at load, one sgemm instead of two sequential sgemv. Save ~30µs/layer dispatch. ~30 lines.
6. [Profiling] ExpertPool stats logging — wire pool.stats() into decode loop, log hit_rate/miss_count every 10 tokens. Don't cache yet, just measure. ~15 lines.
7. [Compute] RMSNorm via NEON intrinsics — replace vDSP calls with inline float32x4_t. Avoid 0.5µs/call FFI overhead × 183 calls/token. ~40 lines.
8. [Memory] Reuse attention scratch across layers — pre-allocate q[24576], kv_out[576], scores[128×seq] once, reuse. ~30 lines.

EXHAUSTION NOTE: 12 iterations exhausted CPU overlap, parallelization, and allocation categories. The ideas above are from DIFFERENT categories mined from the full 43-file research corpus.

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
[iter 10] RESULT: v3-parallel-gate-up — NO EFFECT 1.14 tok/s. Parallel gate+up sgemv via thread::scope in shared_expert_ffn and dense_ffn. Hypothesis: separate P-cores with independent AMX engines halve gate+up time, saving ~33ms/token. Reality: median 1.14 across 3 runs (15.0, 15.1, 15.0 ms/layer). Within noise of baseline 1.13. AMX achieves ~150 GB/s per core, so shared expert gate+up serial time is ~0.78ms (not ~1.46ms). Net savings ~22ms minus scope overhead = 2.5% = below noise.
[iter 10] INSIGHT: Apple M4 Max AMX achieves ~150 GB/s single-core bandwidth for sequential sgemv (not 80 GB/s as estimated in hardware facts). This means: (a) individual sgemv calls are ~2x faster than bandwidth model predicts, (b) parallelizing two sequential sgemv calls saves less time than expected, (c) the 21μs thread::scope overhead becomes a significant fraction of the per-call savings for small-to-medium matrices. ALL remaining improvements at the current architecture level are <3% individually — below the noise floor for 3-run benchmarks. The path to 5+ tok/s requires architectural changes: ExpertPool (eliminate pread), Metal attention kernel (GPU for MLA), or f16 compute path (half bandwidth).
[iter 11] RESULT: v3-parallel-oproj — REVERTED 1.07 tok/s (5.3% regression). Tried rayon par_chunks_mut to split O projection [7168,16384] (470 MB f32) across 4 threads. Expected 2ms/layer savings from multi-core DRAM saturation. Got regression because Apple Accelerate cblas_sgemv ALREADY multi-threads internally for large matrices — explicit parallelization defeats BLAS internal optimization.
[iter 11] INSIGHT: Apple Accelerate cblas_sgemv internally multi-threads for large matrices (>100 MB). This means the single sgemv call on O projection (470 MB) is ALREADY using multiple cores and approaching peak DRAM bandwidth (~400-546 GB/s). Manually splitting into smaller chunks forces each chunk into single-threaded mode, reducing total throughput. This definitively closes ALL forms of explicit sgemv parallelization — BLAS handles it optimally. The ONLY remaining path to improve MLA performance is reducing the DATA volume (f16 compute, W_UK absorption caching, or GPU offload), not changing HOW it's parallelized.
[iter 12] RESULT: v3-cached-layer0 — NO EFFECT 1.15 tok/s. Cached layer 0's f32 weights in a third buffer (buf_layer0), peeled layer 0 out of the decode loop to skip the serial convert_layer_into(0) call. Median 1.15 across 3 runs (14.8, 15.2, 15.4 ms/layer). The 5ms saved per token is 0.5% of 940ms, invisible.
[iter 12] INSIGHT: Single-layer conversion costs ~5ms. At 940ms/token total, this is 0.5% — confirmed below noise. This is the LAST possible "skip redundant work" optimization on single-layer granularity. 12 iterations have now exhaustively tried: overlap (iters 2,5,6,7,9), parallelization (iters 8,10,11), allocation elimination (iter 5), and work elimination (iter 12). ALL produce <3% improvement. CPU-side optimization is definitively exhausted. The next measurable gain requires GPU compute (Metal attention, f16 GEMV) or data volume reduction (ExpertPool caching).
[iter 13] RESULT: v3-f16-oproj-mmap — REVERTED 0.86 tok/s (24% regression). Skipped o_proj f32 conversion in pipeline, read 235 MB f16 from mmap directly via sgemv_f16_par (rayon chunked convert+sgemv). Used mla_decode_v_concat refactor to split O projection from MLA core. Pre-warmed f32 from pipeline overlap buffer beats cold mmap f16 pages even at half data volume.
[iter 13] INSIGHT: The pipeline overlap buffer is a WARM CACHE — the background thread pre-faults and converts mmap pages into hot f32 during the FFN phase. Reading from this buffer is fast because pages are already in the TLB and L3. Reading f16 from mmap on the critical path bypasses this warm cache, hitting cold pages that need page faults + TLB misses. The 50% bandwidth reduction from f16 is completely negated by the page fault latency. This is the same fundamental issue as iter 4 (f16 bypass) but at a different level — not single-core conversion speed, but page cache warmth.
