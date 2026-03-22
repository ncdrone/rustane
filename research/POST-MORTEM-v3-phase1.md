# Post-Mortem: DeepSeek-V3 Phase 1 Optimization (0.03 → 0.7 tok/s)

**Date:** 2026-03-22
**Branch:** v3-optimize (16 commits from rustane-infer)
**Hardware:** M4 Max, 128 GB unified, NVMe SSD
**Model:** DeepSeek-V3, 671B params, 61 layers, 128 heads, 256 experts/layer

## Result Summary

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Decode tok/s | 0.03 | 0.7 | **23×** |
| Prefill tok/s | ~0 | 1.0 | ∞ |
| Per-layer time | ~500ms | ~22ms | **23×** |
| Memory (RSS) | ~34 GB | ~17 GB | -50% |
| Correctness | V2-Lite 4/4 | V2-Lite 4/4 | maintained |
| Output quality | "Paris" | "Paris" | correct |

## What Worked

### 1. Expert Pager (pread replaces 348 GB mmap) — **21× speedup, the biggest win**
- **Problem:** All 58 expert files (348 GB total) wrapped as Metal buffers at load time. When Metal dispatched, it touched pages across all files, causing catastrophic page cache thrashing.
- **Fix:** Use `ExpertLoader::load_expert()` with pread to read only the 8 selected experts (~178 MB) per MoE layer. Pack into a staging Metal buffer.
- **Why it worked:** pread reads exactly what's needed. No 348 GB virtual memory pressure. Page cache can focus on recently-used expert data.

### 2. Rayon Parallel Conversion — **8.5× speedup on conversion**
- **Problem:** f16→f32 conversion was single-threaded at ~100 GB/s, taking ~60ms/layer.
- **Fix:** `rayon::par_chunks_mut` splits large tensors across CPU cores.
- **Result:** ~7ms/layer (approaching memory bandwidth limit).

### 3. Zero-Copy Borrows — eliminated ~100 GB/token of memcpy
- **Problem:** `MlaLayerWeights` used `Vec<f32>` requiring `.clone()` of ALL weight vectors per layer per token. O_proj alone was 469 MB × 61 layers = 28.6 GB of useless memcpy.
- **Fix:** Changed to `&'a [f32]` borrows. Zero-copy from MlaLayerF32 buffer.

### 4. Buffer Reuse (`convert_layer_into`) — eliminated page fault thrashing
- **Problem:** `convert_layer_f32()` allocated new `Vec<f32>` for every field of every layer of every token. 61 layers × ~1.8 GB = ~110 GB of malloc/free per token. The mmap/munmap cycle caused TLB shootdowns and page faults — each layer took ~1.9s just for memory management!
- **Fix:** `convert_layer_into()` reuses pre-allocated Vecs via `clear() + extend()`. After warmup (~3 layers), zero allocations.
- **Why it worked:** Vec capacity persists after clear(). No mmap/munmap, no page faults.

### 5. Cached Metal Staging Buffer — **30% speedup on MoE dispatch**
- **Problem:** `m.wrap_mmap(staging_slice)` created a new Metal buffer for every MoE layer dispatch (58× per token). Metal buffer creation involves kernel calls.
- **Fix:** Create the staging Metal buffer once at load time, reuse for all dispatches. The pread writes to the same underlying memory — Metal sees the updated data.

### 6. Backbone Warmup + madvise — eliminated cold-start penalty
- **Problem:** First token took ~120s because backbone mmap pages were cold (SSD page faults).
- **Fix:** `madvise(MADV_WILLNEED)` at load + explicit warmup pass (convert all 61 layers once, ~1s). All backbone pages pre-faulted into page cache.

## What Didn't Work

### 1. Channel-Based Double-Buffer Pipeline — REVERTED
- **Idea:** Converter thread converts layer N+1 while main thread computes layer N. Use channels for buffer ownership transfer.
- **Problem:** Pipeline overhead (channel send/recv + thread wakeup + contention between converter's rayon workers and main thread's expert pread threads) exceeded the conversion time being hidden. Result: 27ms/layer pipelined vs 20ms/layer sequential.
- **Lesson:** Pipeline overhead matters when the overlapped work (7ms conversion) is small relative to compute (13ms). The crossover point is when conversion ≈ compute.

### 2. Chunked f16 sgemv — **4× SLOWER, reverted**
- **Idea:** Convert 64 rows at a time into L2-resident buffer, then AMX sgemv on the chunk. This would halve main memory traffic (read f16 instead of f32).
- **Problem:** 112 cblas_sgemv calls for o_proj (7168/64 chunks) with per-call overhead. The scalar `f16::to_f32()` conversion loop didn't auto-vectorize well. Result: 86ms/layer vs 22ms/layer.
- **Lesson:** Chunked approaches add function call overhead that dominates small matrices. A fused NEON kernel (convert + multiply in same instruction stream) would work, but `cblas_sgemv` per-chunk doesn't.

### 3. Serial Expert pread — **25% SLOWER than parallel**
- **Idea:** Load experts serially directly into staging buffer (simpler code, no alloc).
- **Problem:** NVMe SSDs need queue depth >1 for full throughput. Serial pread at QD=1: ~3 GB/s. Parallel at QD=4-8: ~5 GB/s.
- **Lesson:** Always use parallel I/O on NVMe. Even `std::thread::scope` with 4 threads helped.

### 4. mlock on Backbone — no measurable improvement
- **Idea:** Lock 34 GB backbone in memory to prevent eviction by expert page cache.
- **Problem:** With warmup already loading pages, and the expert pager not using mmap anymore, there was no eviction pressure. mlock just consumed address space.
- **Lesson:** mlock is useful when competing mappings exist. After switching to pread for experts, the backbone pages are stable.

## Key Insights

1. **Memory management dominated compute.** The initial 0.03 tok/s was >99% memory overhead (clones, allocs, page faults), not compute. The actual MLA attention + FFN compute was ~13ms/layer all along.

2. **mmap for large sparse files is an anti-pattern.** The 348 GB expert files caused the OS page cache to thrash. pread with targeted reads was 100× better.

3. **Allocation churn is the silent killer.** Each `Vec::collect()` in the hot path costs milliseconds in page faults. Buffer reuse is essential for large model inference.

4. **Profiling was critical.** The per-layer conv/compute timing split revealed that 95% of time was in expert dispatch (page faults), NOT conversion or attention compute. Without profiling, I would have optimized the wrong thing.

5. **Simple > clever for threading.** The sequential single-buffer approach beat the pipelined double-buffer approach because the pipeline's overhead exceeded its benefit.

## Current Bottleneck Breakdown (22ms/layer)

| Component | Time | % | Bound By |
|-----------|------|---|----------|
| f16→f32 conversion | ~7ms | 32% | Memory BW |
| MLA attention (Q LoRA + KV + scores + O proj) | ~5ms | 23% | Memory BW (AMX sgemv) |
| Shared expert FFN (3× sgemv) | ~4ms | 18% | Memory BW |
| Expert pread (8 experts, parallel) | ~3ms | 14% | SSD + page cache |
| Metal expert dispatch | ~3ms | 14% | GPU compute + dispatch overhead |

## Path to Higher Performance

### To reach 1 tok/s (16ms/layer):
- Eliminate f16→f32 conversion via native f16 compute or pre-loaded f32 weights
- Need ~5ms savings per layer

### To reach 3-5 tok/s (3-5ms/layer):
- Move backbone compute to Metal GPU (f16 GEMV natively)
- In-memory expert pool (eliminate all pread I/O)
- Pipeline GPU backbone + GPU experts in single command buffer

### To reach 10 tok/s (1.6ms/layer):
- **Physically impossible at f32** — backbone alone is 57 GB/token at 546 GB/s = 104ms
- **Requires f16 everywhere** — 28.5 GB/token at 546 GB/s = 52ms → ~19 tok/s theoretical max
- **Or requires pre-loaded f32 in RAM** — 68 GB backbone + 45 GB expert pool = 113 GB. Fits in 128 GB.
- Metal GPU for ALL compute (both backbone sgemv and expert GEMV)
- Zero I/O during decode (everything in RAM)
- AMX peak decode at 10 TFLOPS with 233M FLOPs/layer = 0.023ms/layer compute-bound → memory-bound reality: ~0.9 GB/layer at 546 GB/s = 1.6ms/layer → theoretically achievable at f32 if weights are pre-loaded!

## Files Changed (16 commits)

- `crates/moe-infer/src/generate_v2.rs` — major restructure: buffer reuse, expert pager, f16 path (reverted)
- `crates/moe-infer/src/mla_attention.rs` — zero-copy borrows, f16 forward pass
- `crates/moe-infer/src/blas.rs` — sgemv_f16 (chunked, reverted to simple), sgemv_f16_trans
- `crates/moe-infer/src/weights.rs` — madvise, mlock (removed)
- `crates/expert-pager/src/lib.rs` — export ExpertFileLayout
- `crates/expert-pager/src/loader.rs` — unsafe impl Sync for ExpertLoader
- `crates/moe-infer/tests/` — all tests updated for borrowed MlaLayerWeights
