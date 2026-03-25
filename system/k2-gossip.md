# K2 Optimization Gossip

## Current State
tok/s: 1.68 (default top_k=8) | AUTO POOL (cap=3000 on 128GB): 1.13 (+31% over old default) | POOL+top_k=4: 1.41 (+176%) | wins: 10 | experiments: 42
F_NOCACHE on expert fds: direct SSD DMA bypasses page cache, eliminates 10 GB/token cache pollution

## BENCHMARK PROTOCOL (CRITICAL — READ THIS)
Benchmarks are now run by the bash loop AFTER you exit. Do NOT run bench_k2_tok_per_sec yourself.
Just pass Tier 1 gates (build + correctness), commit, write experiment marker, and exit.
The loop does: 2 warmup runs (SSD controller cache) + 3 measured runs (median).
This eliminates page cache pollution from Claude (~1 GB RAM) and ensures consistent results.
Three-layer cache: OS page cache (RAM) → SSD controller DRAM (1-4 GB) → NAND flash.
Cold baseline: 0.51 tok/s. Warm baseline: 1.68 tok/s. Difference = SSD controller cache state.
Apple SSD uses "Apple Fabric" protocol, not standard NVMe.

## Model Facts
- Kimi-K2: 1 trillion parameters, 61 layers
- 64 attention heads (MLA with Q LoRA)
- 384 experts per MoE layer, top-8 routed, INT4 quantized
- 1 dense layer (layer 0), 60 MoE layers
- Backbone: 23.4 GB f16, Expert files: 9 GB each (524 GB total)
- Tokenizer: tiktoken-based, vocab 163840

## Hardware
- M4 Max 128 GB unified memory
- CPU (AMX): 3 TFLOPS — currently handles MLA attention + shared FFN
- Metal GPU: 15 TFLOPS — currently handles expert INT4 dispatch only
- ANE: 17.8 TFLOPS — currently UNUSED (ane-bridge crate exists)
- NVMe SSD: 17.5 GB/s pread

## Per-Layer Breakdown (MEASURED, warm, with F_NOCACHE)
- **MLA: ~1.9ms/layer** (19%) — o_proj 925µs (48% of MLA), q_proj 586µs (30%), w_uv 166µs, w_uk 119µs, kv 117µs
- **FFN: ~7.1ms/layer** (71%) — max(pread ~3-5ms, shared_ffn ~1.5-2.7ms) + Metal ~3ms
- **convert_wait: ~38ms total** (~0.6ms/layer) — hidden behind FFN
- **lm_head: 19ms** (3%)
- **other: ~1.5ms total** (0.2%)

## Metal GPU Profiling (2026-03-24, GPU-side timestamps)
- **Expert gate+up [7168→2048]**: CPU 486µs, GPU 245µs → 50% CPU overhead!
- **Expert down [2048→7168]**: CPU 567µs, GPU 347µs → 39% CPU overhead!
- **GPU bandwidth: 22-32 GB/s = 5-8% of M4 Max 400 GB/s peak**
- Root cause: expert matrices too small (7.8 MB) to saturate 40 GPU EUs
- FIX: batch 8 experts into SINGLE dispatch → eliminate 7× CPU overhead (241µs each)
- Projected savings: 103ms/token from CPU overhead + higher GPU bandwidth = 17-25% speedup

## MLA Component Breakdown (per-layer, RUSTANE_MLA_PROFILE=1)
- o_proj: 925µs (48%) — #1 MLA target, bandwidth-bound [8192→7168]
- q_proj (Q LoRA): 586µs (30%) — two sgemv calls [7168→1536] + [1536→12288]
- value_recon+w_uv: 166µs (9%)
- w_uk_absorb: 119µs (6%) — 64 sequential tiny sgemv
- kv_compress: 117µs (6%)
- attn_scores+softmax: 2µs (0.1%) — negligible

## ANE Findings (2026-03-24, comprehensive profiling)
- ANE for single-token decode: SLOWER than CPU (proven across all approaches)
- ANE crossover at seq=2 tokens for all projections (HUGE for prefill/spec-decode)
- ANE at seq=64: 5-35x faster than CPU on every projection
- 119 compile limit GONE on macOS 26.3 (150+ compiles succeed)
- Baked matmul weights compile at all K2 dims (even 117 MB O projection)
- Conv1x1 fails on asymmetric dims (compiler tiling bug) — use matmul instead
- IOSurface spatial width must be ≥ 64 (not just mod-16), silent garbage below
- ANE shared gate [7168→2048] is 1.75x faster than CPU (only projection where ANE wins at seq=1)

## The Goal
Get tok/s as high as possible. Theoretical max ~5 tok/s.

## PRIORITY: K2-FFN — Metal Expert Dispatch Optimization (71% of decode)
This is the K2-FFN optimization variant. The #1 target is METAL EXPERT DISPATCH.
MLA and ANE have been exhaustively profiled and the remaining wins are marginal.
Metal FFN is 71% of decode and running at 5-8% GPU bandwidth utilization.

### K2-FFN Experiment Queue (in priority order)
1. **Batch 8 experts into single Metal dispatch** — eliminate 7× CPU overhead (241µs each = 1.7ms/layer). GPU at 5-8% bandwidth because expert matrices (7.8 MB) too small for 40 EUs. Batching → 62.4 MB → closer to peak.
2. **Restore x_cache[7168] in fused_gate_up_silu kernel** — 28 KB fits 32 KB TG memory. Currently reading x from device memory (removed due to old x_cache[4096] OOB). 8 rows share same x = 7× redundant device reads eliminated.
3. **Single command buffer for fused+down** — currently 2 separate commit+wait per expert. Single cmdbuf eliminates 1 roundtrip per expert.
4. **Fuse fused+down into single kernel** — eliminate intermediate buffer write/read between gate+up+SiLU and down projection.
5. **BF16 x vector** — half the x bandwidth (Metal 4 supports BF16 natively).

### Key files for Metal kernel work
- `crates/moe-kernels/src/dequant.rs` — Metal shader source + dispatch logic
- `fused_gate_up_silu` kernel (line ~197) — gate+up+SiLU fused, ROWS_PER_TG=8
- `dequant_4bit_gemv_v2` kernel (line ~121) — down projection
- `MetalDequantGemv::fused_and_down_single_cmdbuf` (line ~529) — dispatch orchestration
- New: `gemv_gpu_timed()` — GPU-side timing via GPUStartTime/GPUEndTime

### Metal profiling infrastructure (already on 1t-moe-infer branch)
- `crates/moe-kernels/tests/test_metal_gpu_timing.rs` — GPU-side timing test
- `crates/moe-infer/tests/profile_full_system.rs` — comprehensive system profiler
- `gemv_gpu_timed()` on MetalDequantGemv — returns (result, cpu_secs, gpu_secs)

## Iteration Log

### Iteration 1: f16 direct decode path — REVERTED
- **Experiment**: Replace f32 double-buffer pipeline with f16 inline decode (run_layer_f16)
- **Result**: 0.32 tok/s (scalar conversion), 0.75 tok/s (SIMD convert_to_f32_slice) vs 1.54 baseline
- **Verdict**: REVERTED — f16 inline conversion adds to critical path (no pipelining)
- **Insight**: The f32 double-buffer hides ~1.6ms/layer conversion behind 8ms FFN compute. f16 path puts conversion on critical path. Also: scalar `f16::to_f32()` in a loop is 40x slower than `convert_to_f32_slice()` (SIMD FCVTL) — always use bulk SIMD conversion.
- **Infrastructure kept**: SIMD fix in blas.rs (sgemv_f16 + sgemv_f16_trans now use convert_to_f32_slice). No effect on current f32 path.

### Iteration 2: lm_head parallel sgemv — NO EFFECT
- **Experiment**: Parallel rayon sgemv_f32 for lm_head (163840×7168 = 4.7 GB), also tried sgemv_f16_par and sgemv_f16
- **Result**: 1.54 tok/s (identical to baseline).
- **Verdict**: NO EFFECT — lm_head is only 24ms (3.3% of total). Not a bottleneck.
- **Insight**: cblas_sgemv already saturates M4 Max memory bandwidth single-threaded. f16 conversion throughput (FCVTL) is SLOWER than DRAM read.

### Iteration 3: Disable pool write-back — WIN (+0.03 tok/s)
- **Experiment**: Disable ExpertPool write-back that copies 23 MB per expert miss to pool buffers
- **Result**: 1.57 tok/s (median warm: 1.58, 1.57, 1.57) vs 1.54 baseline
- **Verdict**: WIN — pool write-back was causing 3x regression (0.49 tok/s with default 3000 cap)
- **Root cause**: 480 expert misses/token × 23 MB alloc+copy = 11 GB memory churn per token. This evicts OS page cache entries for expert files, making pread slower. Net effect: write-back costs more than the cache hits save.
- **Fix**: Disabled write-back, OS page cache handles expert caching natively. Pool tracking still runs for statistics.
- **Profiling data**: MLA=2.2ms/layer (18%), FFN=9.0ms/layer (75%), convert_wait≈0, lm_head=24ms, other=1.6ms total.

### Iteration 4: Pool disable default (pool_cap=0) — NO EFFECT
- **Experiment**: Change default pool_cap from 3000→0 (pool=None) since write-back is disabled
- **Result**: 1.55 tok/s (median warm: 1.37, 1.55, 1.57) vs 1.57 baseline
- **Verdict**: NO EFFECT — -1.3%, within noise. Committed as code cleanup.
- **Insight**: Pool HashMap tracking overhead (~0.3ms/layer = ~18ms/token) is real but too small to reliably measure against 650ms/token total. The 1.37 outlier shows page cache state dominates run-to-run variance. RUSTANE_POOL_CAP env var preserved for future pool experiments.

### Iteration 5: Split pread + pipelined Metal fused/down — NO EFFECT (REVERTED)
- **Experiment**: Split expert pread into gate+up (phase 1, overlaps shared_ffn) then down (phase 2, overlaps GPU fused dispatch). Two Metal command buffers instead of one.
- **Result**: 1.56 tok/s (warm) vs 1.57 baseline
- **Verdict**: NO EFFECT — pread_dn (1.5-2.5ms) > GPU fused (~1.75ms), so GPU finishes before pread. No overlap benefit.
- **Detailed profiling** (warm, layers 1-3): pread_gu=2-3ms, shared_ffn=2.3-4ms, pread_dn=1.5-2.5ms, Metal_total=3.7-4ms
- **Insight**: The GPU expert dispatch (fused gate+up+SiLU) completes faster than a single pread of 62 MB down data. Splitting the Metal dispatch into two command buffers adds overhead (~0.3ms/layer) without pipeline benefit. The fundamental bottleneck is pread_total + Metal_total = serial, and neither can be hidden behind the other because pread feeds Metal.
- **Infrastructure kept**: const u32 buffer caching in MetalDequantGemv (eliminates 360 Metal buffer allocs/token).

### Iteration 6: Cached constant u32 Metal buffers — NO EFFECT
- **Experiment**: Pre-cache u32 constant Metal buffers (hidden=7168, moe_inter=2048, group_size=128) in MetalDequantGemv struct, eliminating all per-dispatch HashMap creation and newBufferWithBytes_length_options calls.
- **Result**: 1.56 tok/s (median warm: 1.50, 1.56, 1.56) vs 1.57 baseline
- **Verdict**: NO EFFECT — eliminated ~360 Metal buffer allocs/token but each was only ~1μs (4-byte buffer). Total savings ~0.18ms/token, unmeasurable against 650ms total.
- **Insight**: Metal constant buffer CPU-side overhead is negligible. The 3.5ms/layer Metal dispatch time is GPU kernel execution, not buffer setup. Committed as code cleanup — removes 3 HashMap allocations per dispatch call.

### Iteration 7: Parallel per-head W_UK/W_UV sgemv — NO EFFECT
- **Experiment**: Parallelized 64 per-head W_UK absorption and W_UV projection sgemv calls using rayon par_chunks_mut. Each head's sgemv is independent ([128,512] matrix, 256 KB), GPU idle during MLA.
- **Result**: 1.51 tok/s (median warm: 1.48, 1.51, 1.53) vs 1.57 baseline
- **Verdict**: NO EFFECT — synthetic timing shows only 315µs→286µs (1.10x) for both loops combined. Over 61 layers: ~1.8ms savings, unmeasurable against 650ms total.
- **Insight**: Per-head sgemv on [128,512] is ~5µs/call (compute-bound, data fits L2). 64 sequential calls = 320µs/layer. Rayon parallel dispatch overhead (~15µs) nearly cancels the parallelism benefit. AMX sgemv on tiny matrices has per-call overhead that dominates; parallelism doesn't help because the BLAS function-call overhead is already the bottleneck, not the sequential execution time.
- **Infrastructure**: Fixed pre-existing build error in make_attn_weights (removed dead f16 field assignments that referenced non-existent struct fields).

### Iteration 8: rayon::join for pread+shared_ffn overlap — NO EFFECT
- **Experiment**: Replace std::thread::scope with rayon::join for the pread+shared_ffn overlap in moe_ffn_v2. Eliminates ~60 pthread_create/join per token by reusing rayon's warm thread pool.
- **Result**: 1.53 tok/s (median warm: 1.53, 1.53, 1.51) vs 1.57 baseline
- **Verdict**: NO EFFECT — -2.5%, within noise. REVERTED.
- **Insight**: rayon::join with nested par_iter (pread uses into_par_iter inside join's second closure) causes work-stealing contention. Rayon's work-stealing scheduler can steal pread subtasks onto the thread running shared_ffn, partially serializing work. thread::scope avoids this because the spawned OS thread is independent of rayon's pool. The ~60 pthread_create/join calls cost ~2-6ms/token total but rayon contention costs a similar amount, netting zero.
- **Infrastructure kept**: auto_rayon_join_ffn.rs test file (validates rayon::join concurrency properties).

### Iteration 9: Fix fused shader x_cache OOB (half[7168]) — CORRECTNESS WIN
- **Experiment**: fused_gate_up_silu shader had `threadgroup float x_cache[4096]` but K2 in_features=7168. Columns 4096-7167 accessed OOB threadgroup memory. Fixed to `threadgroup half x_cache[7168]` (14KB, fits 32KB TG limit). Half precision saves threadgroup memory vs float[7168] (28KB) which would halve GPU occupancy.
- **Result**: 1.15 tok/s (median warm: 1.15, 1.14, 1.15) vs 1.57* inflated baseline
- **Verdict**: CORRECTNESS WIN — model output changed from degenerate repetition ("a transformer, a transformer is a transformer") to correct ("a static electrical device which works on the principle of electromagnetic induction and can transfer electrical energy from one circuit").
- **Root cause of previous "fast" speed**: OOB x_cache zeroed cols 4096-7167, making expert FFN outputs degenerate → model repeated same tokens → same experts accessed every token → page cache warm → pread fast. The 1.57 measurement was an artifact of broken output.
- **With correct output**: diverse token generation → diverse expert routing → more page cache misses → slower pread → true speed is 1.15 tok/s.
- **Intermediate finding**: float[7168] (28KB) produced 1.10 tok/s. half[7168] (14KB) produced 1.15 tok/s. The 5% improvement from half confirms occupancy matters (28KB allows only 1 concurrent TG, 14KB allows 2).
- **Precision**: max_diff between CPU ref and GPU fused at K2 dims: 0.088 (half) vs 0.0005 (float). Both well within INT4 quantization noise.
- **New correct baseline: 1.15 tok/s**. All previous experiments measured against inflated baseline need re-evaluation.

### Iteration 10: Split-pread pipeline v2 (2-cmd-buf overlap) — NO EFFECT
- **Experiment**: Split pread into gate+up (phase 1, overlaps shared_ffn) + down (phase 3, overlaps Metal fused). Two Metal command buffers: dispatch_fused_phase (commit, don't wait) then dispatch_down_phase (wait). PendingGpuWork wrapper for opaque cmd buffer handle. load_expert_partial for sub-expert pread.
- **Result**: 1.16 tok/s (median warm: 1.13, 1.16, 1.20) vs 1.15 baseline
- **Verdict**: NO EFFECT — +0.4%, within noise. REVERTED.
- **Profiling** (warm): pread_gu: 2-5ms, pread_dn: ~1-2ms (inferred), Metal_fused: ~1.75ms, Metal_down: ~1.75ms. Overlap window ≈ 0 because pread_dn ≈ Metal_fused.
- **Root cause**: pread_dn time (1/3 of total pread) roughly equals Metal_fused time. When pread_dn ≥ Metal_fused, savings = 0. Additionally, 2 command buffers add ~0.1ms overhead × 60 layers = 6ms/token. The split trades one large pread for two smaller ones without reducing total serial time.
- **Correctness**: split dispatch is bit-identical to single-cmdbuf (test verified). Output unchanged.

### Iteration 15: Remove dead MLA weight lookups — NO EFFECT (committed for code quality)
- **Experiment**: Remove dead `mla_layer_weights()` calls in `run_mla_only` and `run_layer_compute`. The result was passed to `make_attn_weights` which ignored the `_f16w` parameter — all fields came from pre-converted f32 `MlaLayerF32`. Also removed unnecessary `final_norm.to_vec()` (rmsnorm accepts `&[f32]` directly).
- **Result**: 1.34 tok/s (median warm: 1.34, 1.35, 1.33) vs 1.34* baseline
- **Verdict**: NO EFFECT — as expected. Committed for code quality.
- **Note**: Session baseline 1.34 tok/s vs historical 1.68 tok/s due to system state (Claude Code overhead, thermal state, page cache state). Baseline verified at 1.34 by reverting changes — identical result.
- **Dead work removed**: ~30 HashMap lookups + ~30 string format! allocations per token (15 per layer × 2 hot path functions). Plus 4× unnecessary 28 KB Vec allocation per token from final_norm.to_vec(). Total overhead was ~0.2ms/token — unmeasurable at 600ms/token.
- **Insight**: mla_layer_weights() lookups are only needed in convert_layer_into (f16→f32 conversion) and the f16 inference path. The f32 decode path already has all weights pre-converted in MlaLayerF32 — the HashMap lookups were pure dead work introduced when make_attn_weights was refactored to remove f16 field assignments.

### Iteration 14: Remove x_cache from GPU shaders — NO EFFECT (committed for correctness)
- **Experiment**: Remove `threadgroup half x_cache[7168]` from fused_gate_up_silu and `threadgroup float x_cache[4096]` from dequant_4bit_gemv_v2. x now read directly from device memory (GPU L2-cached).
- **Result**: 1.71 tok/s (median warm: 1.72, 1.71, 1.70) vs 1.68 baseline
- **Verdict**: NO EFFECT for speed (+1.8%, within noise). COMMITTED for:
  1. **Correctness**: max_diff 0.088→0.000488 (180x improvement). f32 x reads replace the f32→f16 truncation in x_cache.
  2. **Code quality**: removes hardcoded x_cache sizes that were the source of the OOB bug (iter 9). No more threadgroup barrier.
  3. **Occupancy**: theoretical 2 TGs/EU → 4+ TGs/EU, but real improvement masked by pread variance.
- **Insight**: For bandwidth-bound INT4 GEMV on M4 Max, threadgroup memory x_cache doesn't improve performance because GPU L2 cache (32 MB) is large enough to hold x (28 KB) and serves as an effective shared cache across TGs. The threadgroup barrier overhead is ~the same as the L2 latency.

### Iteration 13: F_NOCACHE on expert file fds — WIN (+46%, 1.15 → 1.68 tok/s)
- **Experiment**: `fcntl(fd, F_NOCACHE, 1)` on expert file fds in ExpertLoader::open(). Bypasses OS page cache for expert pread, using direct SSD-to-user-buffer DMA.
- **Result**: 1.68 tok/s (median warm: 1.71, 1.68, 1.68) vs 1.15 baseline
- **Verdict**: WIN — +46% improvement. 1 line of code.
- **Root cause of improvement**: Expert pread without F_NOCACHE pollutes the OS page cache with 176 MB/layer × 61 layers = ~10 GB per token of rarely-reused expert weight data. This evicts shared FFN weights (176 MB) and other working set data from DRAM, forcing shared_ffn sgemv to compete for memory bandwidth with page cache management. F_NOCACHE eliminates this:
  1. **Reduced bandwidth contention**: shared_ffn sgemv runs at full DRAM bandwidth (drops from 2.4-4.4ms to 1.5-2.7ms)
  2. **No memcpy overhead**: pread goes SSD→user buffer directly, skipping page cache→user buffer memcpy
  3. **NVMe SSD controller DRAM** (1-4 GB) handles frequently-accessed experts natively
- **Trade-off**: Cold decode slightly slower (0.67 vs 0.93 tok/s) — no OS page cache warming between cold and warm runs. Warm decode benefits enormously.
- **Key insight**: For working sets >> physical memory (524 GB experts vs 128 GB RAM), OS page cache is net negative: it wastes bandwidth tracking pages that will be evicted before reuse. Direct DMA + SSD controller cache is a better caching strategy.

### Iteration 12: f16 o_proj via sgemv_f16_par from mmap — REVERTED (regression)
- **Experiment**: Read o_proj weights as f16 directly from backbone mmap via sgemv_f16_par (parallel chunked f16→f32 convert+sgemv), bypassing the pre-converted f32 in the double-buffer. Halves DRAM traffic for o_proj (118 MB f16 vs 235 MB f32 for [7168, 8192]).
- **Result**: 0.91 tok/s (median warm: 0.90, 0.95, 0.91) vs 1.15 baseline
- **Verdict**: REVERTED — -21% regression.
- **Implementation**: Added `o_proj_f16: Option<&[half::f16]>` to MlaLayerWeights, conditional dispatch in mla_forward_decode. Plumbed f16 backbone weights through make_attn_weights in generate_v2.rs. ~15 lines of changes. Correctness perfect (max_diff=4.84e-8).
- **Root cause**: The double-buffer pipeline already converts o_proj f16→f32 during the PREVIOUS layer's FFN phase. When MLA runs, the f32 o_proj is warm in DRAM/L3 cache. The f16_par path bypasses this warm data and reads from backbone mmap, which may not be page-cached (competing with expert pread I/O for OS page cache). Reading cold mmap pages during MLA adds latency rather than saving it.
- **Insight**: This is the same lesson as iteration 1 (f16 inline decode): ANY path that reads from mmap during the MLA critical path will be slower than reading pre-staged f32 from the double-buffer, because the double-buffer conversion is already fully hidden behind FFN compute (convert_wait ≈ 0). The only way to benefit from f16 o_proj would be to ALSO remove it from the double-buffer pre-conversion, saving conversion time — but conversion is free (hidden behind FFN), so there's nothing to save.

### Iteration 11: TG=512, ROWS_PER_TG=16 for fused+V2 shaders — NO EFFECT (regression)
- **Experiment**: Changed both fused_gate_up_silu and dequant_4bit_gemv_v2 shaders from TG=256/ROWS_PER_TG=8 to TG=512/ROWS_PER_TG=16. THREADS_PER_ROW=32 preserved (one SIMD group per row). Updated all 5 Rust dispatch sites. Hypothesis: halving TG count amortizes x_cache load over 2× more rows.
- **Result**: 1.05 tok/s (median decode of 1.05, 1.05, 1.12) vs 1.15 baseline
- **Verdict**: NO EFFECT — -8.7% regression. REVERTED.
- **Correctness**: TG=512 is numerically correct (max_diff=0.004 for fused+down at K2 dims).
- **Root cause**: Larger threadgroups (512 threads) reduce GPU occupancy. M4 Max has 40 execution units; with TG=256, each EU can run more concurrent TGs to hide memory latency. With TG=512, fewer TGs fit per EU, reducing latency hiding. The x_cache load cost saved by larger TGs (~0.5ms/token) is overwhelmed by occupancy loss (~5ms/token).
- **Insight**: For bandwidth-bound INT4 GEMV on M4 Max, TG=256 (8 SIMD groups) is better than TG=512 (16 SIMD groups). Smaller TGs = more concurrent work = better latency hiding.

### Iteration 16: Split-pread pipeline v3 (F_NOCACHE) — NO EFFECT (REVERTED)
- **Experiment**: Re-test split-pread pipeline under F_NOCACHE conditions. Split pread into gate+up (overlaps shared_ffn in thread::scope) → dispatch_fused_phase (non-blocking GPU commit) → pread down (overlaps GPU fused) → dispatch_down_phase_and_wait. Added load_expert_partial to ExpertLoader, dispatch_fused_phase/dispatch_down_phase_and_wait to MetalDequantGemv.
- **Result**: 1.37 tok/s (median warm: 1.35, 1.37, 1.38) vs 1.34* session baseline
- **Verdict**: NO EFFECT — +2.2%, within noise. REVERTED hot path, kept infra (load_expert_partial, split dispatch methods).
- **Root cause**: Same as iter 5 and 10. With F_NOCACHE, pread uses direct SSD→user DMA which is fast (~1.5ms for down portion). Metal_fused is ~1.5ms. The overlap window is ≈ 0 because they complete in roughly the same time. Extra command buffer overhead (~0.1ms × 60 layers = 6ms) further negates any micro-gain.
- **Insight**: This is the 3rd attempt at split-pread (iters 5, 10, 16). Confirmed dead end across all conditions: pre-F_NOCACHE (iter 5), post-correctness-fix (iter 10), and post-F_NOCACHE (iter 16). The fundamental issue is that pread_dn ≈ Metal_fused, so there's nothing to overlap.

### Iteration 17: Overlap conversion with MLA — REVERTED (regression)
- **Experiment**: Move convert_layer_into to overlap with MLA + FFN (instead of FFN-only). Conversion thread starts before MLA, giving it an extra 2.2ms to complete. Reduces convert_wait from 24ms to 0.4ms.
- **Result**: 1.24 tok/s (warm: 1.24, 1.24) vs 1.34* session baseline
- **Verdict**: REVERTED — -9.5% regression.
- **Root cause**: MLA sgemv operations (especially o_proj: 235 MB read) need full DRAM bandwidth (~273 GB/s). Concurrent conversion (reading f16 mmap + writing f32 buf_b) steals bandwidth, causing MLA to double from 135ms to 270ms (+135ms). The 24ms saved from eliminating convert_wait is dwarfed by the 135ms MLA slowdown.
- **Key insight**: MLA is a DRAM bandwidth-critical section. ANY concurrent memory-intensive work during MLA degrades performance. The current design (MLA alone, then conversion overlaps FFN) is optimal because FFN's pread uses SSD DMA (not DRAM bandwidth) and Metal GPU dispatch uses GPU memory controller (also not CPU DRAM bandwidth).

### Iteration 18: Pre-cache all 61 layers as f32 (~54 GB) — REVERTED (massive regression)
- **Experiment**: Replace double-buffer pipeline with upfront pre-cache of all layers' MLA weights as f32. Eliminates per-token conversion, thread::scope, convert_wait, and DRAM bandwidth contention during FFN.
- **Result**: 0.6 tok/s (median warm: 0.6, 0.6, 0.6) vs 1.68 baseline
- **Verdict**: REVERTED — -64% regression. The worst regression of all experiments.
- **Implementation**: Vec of 61 MlaLayerF32 (~54 GB total), pre-converted at load time in 3.7s. Simplified decode loop: no thread::scope, no buf swap, direct `cached_all[layer]`. MLA unchanged at 2.1ms/layer. FFN regressed from 7ms to 27ms/layer (4x worse).
- **Root cause**: 54 GB of heap-allocated f32 weights exhausts available DRAM on 128 GB system. Backbone mmap pages (shared FFN weights, ~176 MB per layer) are evicted from page cache → every layer's shared_ffn faults from SSD. The double-buffer approach only pins ~1.8 GB at a time, leaving ample room for backbone pages in page cache.
- **Key insight**: On a 128 GB system running a 524 GB model, DRAM is the scarcest resource. Any optimization that trades DRAM for compute savings will backfire. The double-buffer pipeline (~1.8 GB) is near-optimal for memory footprint. Large pre-caches (>10 GB) will evict working set pages.

### Iteration 19: RUSTANE_TOP_K=6 sweep — WIN (+25%, opt-in)
- **Experiment**: Added RUSTANE_TOP_K env var to override num_experts_per_tok. Tested top_k=6 vs default 8.
- **Result**: 0.64 tok/s (median warm: 0.64, 0.64, 0.64) vs 0.51* session baseline
- **Verdict**: WIN — +25.5% improvement. Opt-in via env var; default behavior unchanged.
- **Profiling**: FFN dropped from 1899ms to 1449ms/token (-24%). MLA unchanged at 133ms. Savings entirely from 2 fewer expert pread + Metal dispatches per MoE layer × 60 layers. Per-layer FFN: 31ms → 24ms (-7ms/layer).
- **Output quality**: top_k=6 produces coherent text ("a transformer is a type of neural network architecture that was introduced in"). Routing weights properly renormalized. top_k=6 selects a strict subset of the top_k=8 experts.
- **Implementation**: ~15 lines. `effective_top_k` field on ModelV2, checked at all routing call sites. Env var RUSTANE_TOP_K=N overrides config. Staging buffer, Metal scratch, and router all use effective value.
- **Insight**: Reducing top_k is a first-order lever for pread-dominated workloads. With 524 GB experts on SSD, each expert pread costs ~4ms. Cutting 2 experts/layer saves ~8ms/layer × 60 = 480ms/token. Quality impact is acceptable for INT4 quantized model. Future: try top_k=4 for another 33% reduction.

### Iteration 20: Expert speculation with speculative pread — REVERTED (massive regression)
- **Experiment**: Pre-load next layer's predicted experts during current layer's Metal dispatch. Cross-layer routing similarity >95% means most predictions are useful. madvise(MADV_WILLNEED) incompatible with F_NOCACHE, so pivoted to actual pread into separate staging buffer.
- **Result**: 0.30 tok/s (warm) vs 0.51* session baseline. -41% regression.
- **Three implementations tried**:
  1. thread::scope wrapping Metal — scope blocks until ALL threads done, adding ~36ms/layer for sequential spec pread
  2. thread::spawn + sequential pread — 8×23MB sequential reads take ~37ms, overlap window only ~3ms
  3. thread::spawn + parallel pread (inner scope with 8 threads) — NVMe can serve 8 concurrent reads in ~5ms, but overlap window still only ~3ms, join blocks ~2ms/layer minimum
- **Root cause**: Speculative pread adds 184 MB/layer of F_NOCACHE SSD I/O (8 experts × 23 MB) that competes for NVMe bandwidth. Even with >95% hit rate reducing normal pread to ~0.5 experts, the total SSD throughput increases. Thread spawn overhead (9600+ spawns/decode) and NVMe controller cache thrashing compound the regression.
- **Key insight**: With F_NOCACHE, pread is already near-optimal — SSD serves direct DMA at ~5 GB/s. There's no "cold miss latency spike" to eliminate (unlike page-cache mode). Speculation only helps when there's a cache layer to warm ahead of time. F_NOCACHE eliminates the cache, making speculation pointless.

### Iteration 21: GPU o_proj (Metal f32 GEMV) — REVERTED
- **Experiment**: Move o_proj [7168, 8192] sgemv from CPU cblas_sgemv to Metal GPU f32_gemv shader
- **Result**: 0.48 tok/s median (of 0.48, 0.48, 0.50) vs 0.50* session baseline. MLA: 180-274ms (vs CPU 128-153ms).
- **Three approaches tested**:
  1. **Per-layer wrap_mmap**: Creates new 235 MB Metal buffer each layer → 12ms/layer GPU page table setup. Unusable.
  2. **Persistent Metal buffer + memcpy**: Allocate once, copy 235 MB/layer via contents(). MLA=477ms (3.7x). memcpy at ~100 GB/s = 2.3ms/layer dominates.
  3. **Pre-cached zero-copy buffers**: Large Vec allocs on macOS use mmap → page-aligned (confirmed: 0xf2e800000). wrap_mmap uses newBufferWithBytesNoCopy. Page tables set up once at warmup. MLA: 180-274ms — GPU dispatch overhead (~1.5ms/layer) still slower than CPU AMX sgemv.
- **Root cause**: On UMA (M4 Max), CPU and GPU share ~273 GB/s memory bus. For bandwidth-bound sgemv, GPU can't beat CPU's AMX coprocessor. GPU adds overhead: command buffer creation/encoding (~0.5ms), waitUntilCompleted synchronization (~0.5ms), result copy. These overheads negate any potential bandwidth advantage.
- **Key insight for UMA**: GPU offload only wins for compute-bound kernels (INT4 dequant+GEMV = ALU-bound). For pure f32 GEMV (bandwidth-bound), CPU AMX is faster due to zero dispatch overhead.
- **Infrastructure kept**: oproj_gpu method, ensure_oproj_scratch(out_features, v_total), auto_oproj_metal.rs test (4 tests, max_diff<1e-3). Correctness confirmed: GPU produces identical results to CPU.

### Iteration 22: Batch W_UK/W_UV as sgemm — NOT FEASIBLE
- **Experiment**: Reshape W_UK [64,128,512] to [8192,512], dispatch as single cblas_sgemm instead of 64 sequential cblas_sgemv.
- **Result**: NOT FEASIBLE — mathematically impossible. Each of 64 heads has a DIFFERENT weight matrix. sgemm(Q[64,128], W[8192,512]) sums across heads (wrong). Block-diagonal trick requires 64x more FLOPs. Per-head weights are inherent to MLA architecture.
- **Time analysis**: 64 sgemv × 5µs × 2 loops = 640µs/layer = 39ms for 61 layers = 2.5% of total decode. Not worth pursuing even if feasible.

### Iteration 23: top_k=4 sweep — WIN (+79%)
- **Experiment**: RUSTANE_TOP_K=4 vs default top_k=8
- **Result**: 0.84 tok/s median warm (of 0.88, 0.84, 0.84) vs 0.47* session baseline. +79%.
- **FFN**: 960ms vs 1900ms at top_k=8 (-50%). 4 fewer expert pread (92 MB less I/O) + 4 fewer Metal dispatches per layer × 60 layers.
- **MLA**: 148ms (unchanged, expected).
- **Quality**: Degraded — "more more" repetition, garbled syntax ("circuits through.") vs top_k=8 clean output ("a static electrical device which works on the principle of electromagnetic induction"). Still generating related content but noticeably worse.
- **Comparison with top_k=6**: top_k=6 gave +25% with good quality. top_k=4 gives +79% but with quality trade-off. Diminishing returns on quality vs speed.
- **Opt-in**: RUSTANE_TOP_K=4 (default unchanged).

### Iteration 25: Pool staging-copy + LRU eviction — WIN (+45%, 0.51 → 0.74 tok/s)
- **Experiment**: Re-enable expert DRAM pool with two fixes: (1) copy pool→staging on hits (DRAM memcpy replaces SSD pread), (2) LRU eviction (monotonic clock) replaces Least-Stale (layer-based).
- **Result**: 0.74 tok/s (median warm: 0.73, 0.74, 0.74) vs 0.51 baseline. Cold: 0.86.
- **Verdict**: WIN — +45% improvement. Opt-in via RUSTANE_POOL_CAP=500 RUSTANE_POOL_WRITEBACK=1.
- **Two bugs fixed**:
  1. **Stale staging on pool hit**: Pool hits skipped pread but left staging data stale. Metal would process wrong expert weights. Never triggered because write-back was disabled (iter 3). Fixed: copy pool→staging on hit.
  2. **Least-Stale eviction bias**: `last_used_layer = layer` meant L01 entries always had lowest priority. During sequential layer processing, L01 entries from previous tokens got evicted first — exactly the entries needed soonest. Fixed: monotonic clock gives true LRU ordering.
- **Why write-back is safe now**: iter 3 disabled write-back because 23 MB memcpy per miss evicted OS page cache entries, making pread slower. F_NOCACHE (iter 13) bypasses page cache entirely — memcpy no longer causes page cache eviction.
- **Pool mechanics**: 500 slots × 23.4 MB = 11.7 GB DRAM. Hit rate builds over tokens: 0% (token 0), ~25% (token 1), ~50%+ (token 2+). Per-layer: L02 with 4/8 hits → pread drops from 28ms to 14ms. DRAM memcpy: 23 MB at ~200 GB/s = 0.12ms vs pread ~28ms = 233× faster.
- **Memory impact**: 11.7 GB pool + 23.4 GB backbone + 0.6 GB KV + 4.7 GB lm_head = ~40 GB. 128 GB system has ~88 GB headroom. No page cache eviction observed.

### Iteration 26: Pool capacity sweep (500 vs 1000) — WIN (+69% at cap=1000)
- **Experiment**: Sweep RUSTANE_POOL_CAP from 500 to 1000 with write-back enabled.
- **Result**: cap=1000 gives 0.86 tok/s median warm (0.86, 0.86, 0.86) vs 0.51 baseline.
- **Verdict**: WIN — +69% improvement. cap=1000 is +16% over cap=500 (0.74).
- **Pool scaling**: cap=1000 (23.4 GB) holds ~2 tokens' expert sets (480/token). Reduces eviction pressure from 4% miss-per-token (cap=500) to <1%. First warm token: 1032ms (0.97 tok/s instantaneous).
- **Memory**: 23.4 GB pool + 23.4 GB backbone + 4.7 GB lm_head + 0.6 GB KV = ~52 GB of 128 GB.
- **No code change** — config only. Recommended config: RUSTANE_POOL_CAP=1000 RUSTANE_POOL_WRITEBACK=1.

### Iteration 24: Fuse MLA scores (sgemm_nt_add + in-place softmax) — NO EFFECT (committed for code quality)
- **Experiment**: Replace two-buffer MLA score computation (sgemm_nt + sgemm_nt + element-wise add + separate softmax) with sgemm_nt_add (beta=1.0 accumulate rope scores in-place) + in-place softmax. Eliminates scores_rope_buf and attn_weights allocations (~13KB/layer at seq_len=26).
- **Result**: 0.51 tok/s (median warm: 0.51, 0.51, 0.51) vs 0.47* session baseline
- **Verdict**: NO EFFECT — MLA dropped from ~134ms to ~126ms (~6% of MLA phase), but MLA is only 6% of total token time. Net savings ~0.4% of decode, within noise.
- **Implementation**: Added sgemm_nt_add to blas.rs (sgemm_nt with beta=1.0). Modified mla_decode_v_concat in mla_attention.rs. ~20 lines changed. 4 correctness tests pass (max_diff < 3.3e-7).
- **Insight**: At current SSD-dominated decode times (~2000ms/token), MLA micro-optimizations (saving <10ms) are unmeasurable. Code is cleaner: fewer allocs, one fewer BLAS call, one fewer intermediate buffer.

### Iteration 28: Pool capacity extended sweep — WIN (+116% at cap=3000, REGRESSION at 4000)
- **Experiment**: Extended pool capacity sweep: cap=1500, 2000, 3000, 4000.
- **Results**:
  - cap=1500 (35.1 GB): 0.97 tok/s (+90%)
  - cap=2000 (46.8 GB): 1.04 tok/s (+104%)
  - cap=3000 (70.2 GB): 1.10 tok/s (+116%)
  - cap=4000 (93.6 GB): 0.52 tok/s **REGRESSION** (-2%)
- **Verdict**: WIN up to 3000, REGRESSION at 4000. Memory ceiling established.
- **Root cause of 4000 regression**: 93.6 GB pool + 23.4 GB backbone + 4.7 GB lm_head + 0.6 GB KV = ~122 GB. Only 6 GB free on 128 GB system. macOS evicts backbone mmap pages → shared_ffn faults from SSD every layer. FFN balloons from ~7ms to ~30ms/layer.
- **Scaling curve**: Diminishing returns — 500→1000 (+16%), 1000→1500 (+13%), 1500→2000 (+7%), 2000→3000 (+6%). Sweet spot: 2000 slots (46.8 GB) for 128 GB systems.
- **Memory budget formula**: pool_max_gb ≈ total_ram - 30 GB (backbone + lm_head + KV) - 15% safety margin. On 128 GB: 128 - 30 - 19 = 79 GB → ~3376 slots. Empirically 3000 works, 4000 fails.

### Iteration 29: Auto-adaptive pool cap — WIN (+31%, 0.86 → 1.13 tok/s)
- **Experiment**: Replace fixed `pool_cap=1000` default with auto-detection via `sysctl hw.memsize`.
- **Formula**: `pool_cap = (total_ram - 30 GB model overhead) × 0.85 / expert_stride`, capped at 3000.
- **Result**: 1.13 tok/s (median warm: 1.13, 1.13, 1.13) vs 0.86 at old cap=1000.
- **Verdict**: WIN — +31% throughput. Zero-config improvement for all users.
- **Scaling**: 128 GB → 3000 slots (70.2 GB), 64 GB → 1404 slots (32.9 GB), 36 GB → 313 slots (7.3 GB), <30 GB → disabled.
- **Implementation**: `auto_pool_cap()` + `system_memory_bytes()` in generate_v2.rs. 30 lines. `RUSTANE_POOL_CAP` env var still overrides. 8 unit tests verify formula for all RAM sizes.
- **Key insight**: On 128 GB, the old fixed default of 1000 slots was leaving 47 GB of usable DRAM on the table. Auto-adaptive uses the full memory budget safely.

### Iteration 30: Overlap pool write-back with Metal dispatch — PENDING BENCHMARK
- **Experiment**: Overlap pool write-back (~0.3-0.5ms/layer) with Metal dispatch (~3ms/layer) via thread::scope. Write-back reads staging (shared), Metal reads staging_metal (same underlying buffer). No write conflict.
- **Implementation**: Moved Metal ops build before write-back block. New thread::scope: spawned thread does pool write-back, main thread does Metal dispatch. MTLBuffer invalidation deferred to after scope (MTLBuffer is !Send). ~30 lines changed.
- **Expected**: ~0.3-0.5ms/layer × 60 = 18-30ms savings (~2-3%). Likely NO EFFECT in benchmark — write-back is already short and Metal dominates the scope.
- **Key insight**: This is a structural improvement — write-back no longer blocks Metal dispatch. Even if unmeasurable now, it removes a serial dependency that could matter with larger pool sizes or slower memcpy.

## Dead Ends (do not retry)
- **lm_head optimization**: Only 3.3% of total time. cblas_sgemv saturates bandwidth single-threaded.
- **f16 inline decode**: Puts conversion on critical path. f32 double-buffer hides it behind FFN compute.
- **Pool write-back on critical path**: 23 MB alloc+copy per miss causes page cache eviction. Must be async or deferred.
- **Split pread + pipeline Metal fused/down (3 attempts)**: pread_dn ≈ GPU fused (~1.5ms each). No overlap benefit — they complete in the same time. Extra cmd buffer overhead cancels any micro-gain. Tested pre-F_NOCACHE (iter 5), post-correctness-fix (iter 10), and post-F_NOCACHE (iter 16). All NO EFFECT. Dead end confirmed.
- **Metal constant buffer caching**: 4-byte u32 buffers are ~1μs each via newBufferWithBytes. ~360/token = 0.18ms, unmeasurable.
- **Parallel per-head sgemv (W_UK/W_UV)**: 64×[128,512] sgemv = 320µs/layer total. Each call ~5µs, dominated by BLAS function-call overhead, not compute. Rayon parallelism saves <30µs/layer = 1.8ms/token.
- **rayon::join for pread+shared_ffn**: Nested par_iter inside rayon::join causes work-stealing contention. thread::scope's independent OS thread avoids this. pthread overhead (~2-6ms) ≈ rayon contention cost.
- **V2 down shader half x_cache**: Down pass with in_features=2048 is bandwidth-bound, not occupancy-limited. float[4096]=16KB already allows 2 concurrent TGs. half saves TG memory but no perf impact.
- **fcntl(F_RDADVISE) prefetch before pread**: pread itself triggers DMA immediately. The hint-to-read window is microseconds — kernel can't start DMA before pread does it.
- **Split-pread pipeline (3 attempts: iters 5, 10, 16)**: Confirmed dead across all conditions. pread_dn ≈ Metal_fused, overlap window ≈ 0. Extra cmd buf overhead negates any gain. Do not retry.
- **TG=512/ROWS_PER_TG=16 for fused+V2 shaders**: -8.7% regression (1.05 vs 1.15). Larger TGs reduce occupancy on M4 Max (40 EUs). TG=256 is optimal for bandwidth-bound INT4 GEMV.
- **f16 o_proj via sgemv_f16_par from mmap**: -21% regression (0.91 vs 1.15). Double-buffer already pre-converts o_proj f16→f32 overlapped with FFN. f16_par bypasses warm DRAM, reads cold mmap during MLA, competes with expert pread for page cache. Same lesson as f16 inline decode: anything on MLA critical path that touches mmap loses to pre-staged f32.
- **Dead MLA code removal (mla_layer_weights + final_norm.to_vec)**: ~0.2ms/token dead overhead. Unmeasurable. Committed as code cleanup.
- **Overlap conversion with MLA**: -9.5% regression. MLA sgemv needs full 273 GB/s DRAM bandwidth. Concurrent conversion (f16 mmap read + f32 write) steals half, doubling MLA from 135ms to 270ms. The 24ms convert_wait savings is dwarfed. MLA must run alone.
- **Pre-cache all layers as f32 (~54 GB)**: -64% regression (0.6 vs 1.68). 54 GB heap exhausts DRAM, evicting backbone mmap pages → shared_ffn faults from SSD every layer. FFN 7ms→27ms/layer. On 128 GB system with 524 GB model, DRAM is scarce — never pin >10 GB of converted weights.
- **Expert speculation (speculative pread for next layer)**: -41% regression (0.30 vs 0.51*). Three variants tried (scope, sequential spawn, parallel spawn). Incompatible with F_NOCACHE: direct DMA is already fast, no cache layer to pre-warm. 184 MB/layer spec I/O doubles SSD bandwidth usage. Thread spawn overhead (9600 spawns/decode) compounds. Dead end with F_NOCACHE.
- **Batch W_UK/W_UV as single sgemm**: NOT FEASIBLE. Each of 64 MLA heads has a different weight matrix — sgemm uses one shared matrix. Block-diagonal trick requires 64× more FLOPs. Total time (640µs/layer) is only 2.5% of decode. Not a bottleneck.
- **GPU o_proj (Metal f32 GEMV for o_proj)**: -4% regression (0.48 vs 0.50*). Three approaches: per-layer wrap_mmap (12ms page table), persistent memcpy (3.7x MLA), zero-copy pre-cached (still +40-100% MLA). On UMA, f32 GEMV is bandwidth-bound — GPU shares same 273 GB/s memory bus as CPU AMX. GPU dispatch overhead (~1.5ms/layer encode+sync) exceeds any bandwidth gain. GPU offload only viable for ALU-bound kernels (INT4 dequant).
- **Fuse MLA scores (sgemm_nt_add + in-place softmax)**: NO EFFECT. Saves ~8ms total MLA (134→126ms), but MLA is only 6% of token time. Net savings ~0.4%, unmeasurable. Committed for code quality (fewer allocs, cleaner code).
- **Pool cap>3000 on 128 GB systems**: REGRESSION. cap=4000 (93.6 GB) leaves only 6 GB free → macOS evicts backbone mmap pages → shared_ffn faults from SSD. Memory ceiling: ~70 GB pool (3000 slots) on 128 GB. Formula: max_pool_gb = total_ram - 30 GB model overhead - 15% safety.
- **Prefill expert skipping (stride 0/2/4)**: MLA KV cache too sensitive. K2 has no delta-net layers like Qwen3.5. All MLA layers need routed experts for quality.
- **Prefill layer-by-layer loop reorder**: 7% cold, 0% warm. With 6-8 tokens, backbone mmap stays hot. Needs 200+ tokens to matter.

## Suggested Next Experiments
NOTE: Fresh profiling from iter 10 (warm, correct shader):
- pread_gu: 2-5ms/layer, pread_dn: ~1-2ms (inferred from timing gaps)
- Metal (fused+down): 3.5-4.5ms/layer total
- shared_ffn: 2.4-4.4ms/layer
- LAYER_TOTAL (FFN): 7-10ms warm, 14-25ms cold
- Bottleneck: pread is still dominant (56%+ of FFN time), but highly variable due to OS page cache state.

1. **Async Metal dispatch overlap with MLA** — Metal dispatch and MLA use different hardware (GPU vs CPU/AMX). Currently serial. Overlap layer N's Metal with layer N+1's MLA. Requires: double-buffer staging, restructure decode loop to pipeline across layers. ~150 lines. Potential: save ~2ms/layer of MLA time that currently runs while GPU idles.
2. **Reduce Metal GPU kernel time** — TG=512 tried and regressed (-8.7%). Try batching multiple experts into single dispatch, or compute-aware scheduling. Metal fused+down = 3.5-4.5ms/layer is 30-40% of warm FFN time.
3. **madvise(MADV_SEQUENTIAL) on expert files** — hint kernel that expert file reads are sequential within each file. May improve readahead for pread.
4. **Larger pread I/O size** — instead of 8 separate pread calls (one per expert, ~23MB each), single contiguous pread if experts are contiguous in file. Reduces syscall overhead.
5. **ANE for norms/activations** — 17.8 TFLOPS sitting idle. Could offload RMSNorm + SiLU.
6. **top_k=4 sweep** — top_k=6 gave +25%. top_k=4 would save another ~33% of pread+Metal per layer. Quality might degrade — test empirically. Use RUSTANE_TOP_K=4.

## ANE Research Integration (NEW — from rustane-research/ANE/)
READ: research-context/ane-synthesis.md, research-context/ane-unified-plan.md, research-context/ane-mla-attention.md

### Key finding: 0% ANE utilization during decode
M4 Max has 56.5 TFLOPS total silicon. We use 2.5% of it. The ANE (19 TFLOPS at 2.8W) is completely idle.

### Immediately actionable (no ANE integration needed):

1. ~~**Expert speculation with madvise prefault**~~ — **DEAD END** (iter 20). Tested as speculative pread (madvise incompatible with F_NOCACHE). -41% regression. See dead ends list.

2. **o_proj on Metal GPU** — o_proj is 1.2ms/layer on CPU (cblas_sgemv). K2 o_proj is [7168, 4096] at f32. Metal already has F32_GEMV_SHADER compiled in dequant.rs. Dispatch o_proj on GPU overlapped with next layer's convert. Saves ~73ms/token (1.2ms × 61 layers). ~50 lines. **CAVEAT**: iter 12 showed that f16 o_proj from mmap regressed -21% because it bypassed warm f32 DRAM. GPU o_proj would need to use the pre-converted f32 buffer, not mmap.

3. **Batch W_UK + W_UV as sgemm** — currently 64 sequential sgemv calls (already tried rayon, didn't help). Instead: reshape to single sgemm call. W_UK is [64, 128, 512] = [64×128, 512] = [8192, 512]. One sgemm replaces 64 sgemv. Same for W_UV. Potential: 2× faster absorption. **CAVEAT**: Accelerate's cblas_sgemm doesn't support batched/block-diagonal. Would need to reshape and scatter-gather, which may negate BLAS gains.

### Medium-term (ANE integration — bigger lift):
4. **ANE Conv1x1 for MLA projections** — Q LoRA + KV compression + W_UK absorption as compiled ANE graph. Conv1x1 is 3x faster than matmul on ANE. Saves ~6ms/layer. Needs: ane-bridge FFI, IOSurface weight staging, compilation. ~500 lines.

5. **Heterogeneous pipeline** — ANE does projections while Metal does expert FFN while CPU does attention scoring. Three accelerators running simultaneously. This is the path to 5+ tok/s.

### Constraints (do not violate):
- Expert FFN (dim=7168) MUST stay on Metal — exceeds ANE's 32 MB SRAM cliff (4.7x slower on ANE)
- IOSurface spatial width must be multiple of 16 (silent data corruption)
- No rsqrt after reduce on ANE — use pow(-0.5)
- MLA must run without concurrent DRAM-intensive work (confirmed by iter 17: bandwidth contention doubles MLA time)
