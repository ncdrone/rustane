# DeepSeek-V3 671B: Architecture for 10 tok/s on M4 Max 128GB

**Date:** 2026-03-22
**Author:** Architecture audit (Claude)
**Current:** 0.7 tok/s (1,430 ms/token)
**Target:** 10 tok/s (100 ms/token)
**Hardware:** M4 Max, 128GB unified memory, 546 GB/s bandwidth, ~5 GB/s NVMe SSD

---

## 1. Physical Limits Analysis

### 1.1 Backbone Weight I/O Budget

The backbone (non-expert) weights must be read once per token per layer.

**Per-layer backbone weight sizes (from deepseek-v3.toml + architecture):**

| Tensor | Shape | f16 bytes | f32 bytes |
|--------|-------|-----------|-----------|
| q_a_proj | [1536, 7168] | 22.0 MB | 44.0 MB |
| q_a_layernorm | [1536] | 6 KB | 6 KB |
| q_b_proj | [24576, 1536] | 75.5 MB | 151.0 MB |
| kv_a_proj | [576, 7168] | 8.3 MB | 16.5 MB |
| kv_a_layernorm | [512] | 2 KB | 2 KB |
| w_uk (from kv_b_proj) | [128, 128, 512] | 16.8 MB | 33.6 MB |
| w_uv (from kv_b_proj) | [128, 128, 512] | 16.8 MB | 33.6 MB |
| o_proj | [7168, 16384] | 234.9 MB | 469.8 MB |
| **Per-layer total (MLA only)** | | **374.3 MB** | **748.5 MB** |

**Shared expert FFN (58 MoE layers):**

| Tensor | Shape | f16 bytes |
|--------|-------|-----------|
| shared_gate | [2048, 7168] | 29.4 MB |
| shared_up | [2048, 7168] | 29.4 MB |
| shared_down | [7168, 2048] | 29.4 MB |
| router | [256, 7168] | 3.7 MB |
| **Per-layer shared** | | **91.9 MB** |

**Dense FFN (3 layers, 0-2):**

| Tensor | Shape | f16 bytes |
|--------|-------|-----------|
| dense_gate | [18432, 7168] | 264.2 MB |
| dense_up | [18432, 7168] | 264.2 MB |
| dense_down | [7168, 18432] | 264.2 MB |
| **Per dense layer** | | **792.6 MB** |

**Total backbone weight reads per token:**

| Component | Layers | f16/layer | Total f16 | Total f32 |
|-----------|--------|-----------|-----------|-----------|
| MLA attention | 61 | 374 MB | 22.8 GB | 45.7 GB |
| Shared expert FFN | 58 | 92 MB | 5.3 GB | 10.7 GB |
| Dense FFN | 3 | 793 MB | 2.4 GB | 4.8 GB |
| Norms/biases | 61 | ~0.1 MB | ~6 MB | ~6 MB |
| LM head | 1 | 1.85 GB | 1.85 GB | 3.7 GB |
| **TOTAL** | | | **32.4 GB** | **64.9 GB** |

### 1.2 Memory Bandwidth Wall

At 546 GB/s (M4 Max measured):

| Scenario | Data read | Min time | Max tok/s |
|----------|-----------|----------|-----------|
| f32 from RAM | 64.9 GB | 119 ms | 8.4 |
| f16 from RAM | 32.4 GB | 59 ms | **16.9** |
| f32 (current) | 64.9 GB + conversion | ~160 ms | 6.3 |

**VERDICT: f32 compute CANNOT reach 10 tok/s.** The backbone alone at f32 is 64.9 GB per token. At 546 GB/s, reading it takes 119 ms -- and that is JUST the reads with zero compute overhead. The 100 ms budget is exceeded by pure I/O.

**f16 compute CAN reach 10 tok/s.** At 32.4 GB per token, the read takes 59 ms, leaving 41 ms for compute + expert I/O. This is tight but physically possible.

### 1.3 Expert Weight I/O Budget

Per token: 58 MoE layers x 8 experts x 22.3 MB (INT4) = **10.3 GB**

| Source | Throughput | Time for 10.3 GB |
|--------|-----------|-------------------|
| RAM (page cache hit) | 546 GB/s | 19 ms |
| SSD (cold, 5 GB/s) | 5 GB/s | 2,060 ms |
| 90% hit rate | mix | 19 + 206 = 225 ms |
| 95% hit rate | mix | 19 + 103 = 122 ms |
| 100% hit rate (all cached) | 546 GB/s | 19 ms |

**VERDICT: 10 tok/s requires near-100% expert cache hit rate.** Even 95% hit rate adds 103 ms from SSD misses, blowing the 100 ms budget. This means ALL active experts must be resident in RAM.

### 1.4 Combined Physical Floor

| Component | f16 path (ms) | f32 path (ms) |
|-----------|---------------|---------------|
| Backbone read | 59 | 119 |
| Expert read (100% cached) | 19 | 19 |
| Compute (80.7 GFLOP) | ~6 | ~12 |
| **Total floor** | **84 ms** | **150 ms** |
| **Theoretical max tok/s** | **11.9** | **6.7** |

The f16 path gives ~16 ms of headroom above the physical floor. This is achievable only with a perfectly pipelined architecture that overlaps backbone reads with compute.

---

## 2. The Architecture: All-Metal Unified Pipeline

### 2.1 The Key Insight

The current architecture has a fatal flaw: **it converts f16 to f32, then feeds f32 to Accelerate BLAS on CPU.** This doubles the memory traffic. The f16-to-f32 conversion itself takes ~15 ms/layer, but worse, the converted f32 weights are 2x larger, consuming 2x the memory bandwidth for every GEMV.

The fix is NOT "faster conversion." The fix is **eliminate conversion entirely** by doing all compute in f16 on the GPU.

### 2.2 Why Metal, Not CPU

| Path | Throughput | f16 native? | Notes |
|------|-----------|-------------|-------|
| CPU + Accelerate cblas_sgemv | ~3 TFLOPS f32 | NO -- requires f16-to-f32 | Reads 2x data |
| CPU + cblas_hgemm | Unavailable | N/A | Apple has not exposed f16 BLAS in Accelerate |
| CPU + sgemv_f16 (chunked) | ~86 ms/layer (measured) | Sort of -- chunk convert in L2 | 12x too slow |
| Metal f16 GEMV | ~15 TFLOPS f16 | YES | Native f16 compute, reads half the data |
| Metal + fused kernels | ~15 TFLOPS f16 | YES | Amortize dispatch overhead |

**Apple's Accelerate framework does NOT provide cblas_hgemm or any native f16 BLAS.** The `vDSP_vflt16` function does not exist. The only way to do f16 compute at full bandwidth is Metal.

The experiment `v3-f16-sgemv` confirmed this: chunked f16-to-f32 convert+sgemv on CPU gave 86 ms/layer (WORSE than the current 22 ms/layer f32 path), because the per-chunk overhead dominates when the chunk size is small enough to fit L2.

### 2.3 Architecture: GPU-First Pipeline

```
                    DECODE TOKEN PIPELINE (target: 100ms total)
                    =============================================

  CPU Thread 0 (orchestrator)          Metal GPU Command Queue
  ===========================          =======================

  for each token:
    embed(token_id)  ─────────>  CMD_EMBED: embedding lookup (f16)
                                       |
    for layer 0..60:                   v
      [encode all layer ops]    CMD_LAYER[i]: (single command buffer)
                                  ├─ RMSNorm (fused, f16 in/out)
                                  ├─ Q LoRA: q_a_proj GEMV [1536,7168] f16
                                  ├─ Q LoRA: RMSNorm + q_b_proj GEMV [24576,1536] f16
                                  ├─ KV compress: kv_a GEMV [576,7168] f16
                                  ├─ KV norm + cache write
                                  ├─ W_UK absorption: 128x batched GEMV f16
                                  ├─ SDPA attention (fused nope+rope, online softmax)
                                  ├─ V reconstruction (W_UV, absorbed)
                                  ├─ O projection: GEMV [7168,16384] f16
                                  ├─ Residual add
                                  ├─ RMSNorm
                                  ├─ Router GEMV [256,7168] f16
                                  ├─ [GPU commit, CPU reads routing logits]
                                  │
      CPU: sigmoid + grouped     │
      topK routing (0.003ms) ────│
                                  │
      CPU: parallel pread 8      │  (overlapped with routing)
      experts into staging ──────│
                                  │
                                  ├─ Shared expert: fused gate+up+SiLU+down f16
                                  ├─ 8x routed expert: fused INT4 dequant GEMV (existing)
                                  ├─ Expert combine (weighted sum)
                                  └─ Residual add
                                       |
                                       v
    CMD_FINAL:
      ├─ Final RMSNorm
      ├─ LM head GEMV [129280,7168] f16
      └─ [readback logits]

  CPU: sample next token
```

### 2.4 Required Metal Kernels

| Kernel | Status | Notes |
|--------|--------|-------|
| f16 GEMV (row-major, NoTrans) | NEW | Core workhorse. For all projections. |
| f16 GEMV (transposed) | NEW | For W_UK absorption, o_proj |
| RMSNorm (f16 in/out) | NEW | Fused with next op where possible |
| SDPA vector (fused nope+rope) | NEW | Adapted from MLX sdpa_vector.h |
| SiLU (fused with gate+up) | NEW | For shared expert f16 path |
| INT4 dequant GEMV (existing) | EXISTS | Expert path, already works |
| Fused gate+up+SiLU (existing) | EXISTS | Expert path, already works |
| Expert down + combine | EXISTS | Expert path, already works |

The f16 GEMV kernel is the most critical new piece. It handles ~80% of the computation.

---

## 3. Memory Layout

### 3.1 What Must Be Resident in RAM

```
COMPONENT                          SIZE        SOURCE       ACCESS PATTERN
==========================================================================
Backbone weights (f16, mmap)      ~32 GB       backbone.bin  Sequential scan, 61 layers
  - Pre-loaded into page cache                               MUST be fully cached
  - Zero-copy Metal buffer wrap                              GPU reads directly

Expert pool (INT4, page cache)    ~90 GB       layer_XX.bin  Random, 8 per layer
  - 4000 experts (~90 GB)                                    95% hit rate minimum
  - OS page cache manages eviction                           pread for misses

KV cache (f16)                    ~0.3 GB      Allocated     Random writes, sequential reads
  - In Metal buffer                                          GPU-resident

Metal scratch buffers             ~1 GB        Allocated     Reused per layer
  - Activation intermediates
  - Expert staging buffer

macOS + overhead                  ~8 GB        System        Wired

TOTAL                             ~131 GB      OVER BUDGET by ~3 GB
```

**Problem: 32 + 90 + 0.3 + 1 + 8 = 131 GB.** This exceeds 128 GB by 3 GB.

### 3.2 Resolution: Backbone Stays f16, Expert Pool Sized to Fit

The backbone at f16 is ~32 GB and MUST be resident (streaming from SSD at 5 GB/s would take 6.4 seconds per token). There is no alternative.

The expert pool must be sized to fit the remaining RAM:

```
Available for experts = 128 - 32 (backbone) - 1 (scratch) - 0.3 (KV) - 8 (OS) = 86.7 GB
Expert pool size = 86.7 / 0.0223 = ~3,888 experts
Hit rate at 3,888 experts = ~95%  (from research model)
```

At 95% hit rate: 464 accesses/token x 5% miss x 0.37 ms/miss = **8.6 ms from SSD misses.**

This is within budget (100 - 59 - 19 - 6 = 16 ms headroom).

### 3.3 Revised Memory Layout

```
M4 Max 128 GB Unified Memory
================================================================
| macOS + Metal framework          |    8.0 GB |  Wired       |
| Backbone weights (f16, mmap)     |   32.4 GB |  Page-locked |
|   Wrapped as Metal buffer        |           |  GPU-readable|
| Expert pool (OS page cache)      |   86.0 GB |  ~3,850 exp  |
| KV cache (f16, Metal buffer)     |    0.3 GB |  GPU-resident|
| Metal scratch + staging          |    1.0 GB |  GPU-resident|
| Headroom                         |    0.3 GB |              |
================================================================
                              TOTAL:  128.0 GB
```

---

## 4. Compute Pipeline Detail

### 4.1 Per-Layer Time Budget (f16 Metal path)

At 100 ms/token for 61 layers: **1.64 ms per layer budget.**

Current (f32 CPU): 22 ms/layer.
Required speedup: **13.4x**

Is this achievable? Let's compute:

**Memory bandwidth model (f16 weights, GPU reads):**

| Operation | Weight size (f16) | At 546 GB/s | FLOPs | At 15 TFLOPS |
|-----------|-------------------|-------------|-------|--------------|
| q_a_proj GEMV [1536,7168] | 22.0 MB | 0.040 ms | 22.0M | 0.001 ms |
| q_b_proj GEMV [24576,1536] | 75.5 MB | 0.138 ms | 75.5M | 0.005 ms |
| kv_a GEMV [576,7168] | 8.3 MB | 0.015 ms | 8.3M | 0.001 ms |
| W_UK 128x GEMV [128,512] | 16.8 MB | 0.031 ms | 8.4M | 0.001 ms |
| W_UV 128x GEMV [128,512] | 16.8 MB | 0.031 ms | 8.4M | 0.001 ms |
| o_proj GEMV [7168,16384] | 234.9 MB | 0.430 ms | 234.9M | 0.016 ms |
| SDPA attention (seq=1K) | ~1.1 MB | 0.002 ms | 73.7M | 0.005 ms |
| RMSNorm x2 | ~28 KB | ~0 ms | ~14K | ~0 ms |
| **Subtotal MLA** | **374.4 MB** | **0.686 ms** | | |

| Operation | Weight size | At 546 GB/s | FLOPs | At 15 TFLOPS |
|-----------|-------------|-------------|-------|--------------|
| Router GEMV [256,7168] | 3.7 MB | 0.007 ms | 3.7M | ~0 ms |
| Shared gate GEMV [2048,7168] | 29.4 MB | 0.054 ms | 29.4M | 0.002 ms |
| Shared up GEMV [2048,7168] | 29.4 MB | 0.054 ms | 29.4M | 0.002 ms |
| Shared down GEMV [7168,2048] | 29.4 MB | 0.054 ms | 29.4M | 0.002 ms |
| 8x expert INT4 GEMV | 178 MB | 0.326 ms | 797M | 0.053 ms |
| **Subtotal FFN** | **269.9 MB** | **0.495 ms** | | |

| Component | Memory time | Compute time | Dominant |
|-----------|-------------|--------------|----------|
| MLA attention | 0.686 ms | 0.029 ms | Memory |
| FFN (shared + routed) | 0.495 ms | 0.059 ms | Memory |
| Dispatch overhead | 0.050 ms | -- | Fixed |
| **Layer total** | **1.23 ms** | **0.09 ms** | **Memory** |

**61 layers x 1.23 ms = 75 ms. Plus LM head (1.85 GB, 3.4 ms) = 78 ms.**

Expert SSD misses at 95% hit rate: 8.6 ms.
Sampling + overhead: ~2 ms.
**Predicted total: ~89 ms = 11.2 tok/s**

THIS IS PHYSICALLY ACHIEVABLE. The f16 Metal path, at realistic memory bandwidth utilization (~85% of peak), gives ~95 ms/token = 10.5 tok/s.

### 4.2 Sensitivity Analysis

| Bandwidth utilization | Layer time | Total | tok/s |
|----------------------|-----------|-------|-------|
| 100% (546 GB/s) | 1.23 ms | 89 ms | 11.2 |
| 85% (464 GB/s) | 1.45 ms | 102 ms | 9.8 |
| 75% (410 GB/s) | 1.64 ms | 114 ms | 8.8 |
| 65% (355 GB/s) | 1.89 ms | 130 ms | 7.7 |

At 85% bandwidth utilization (realistic for well-optimized Metal kernels), we get 9.8 tok/s. At 75%, we get 8.8 tok/s. The margin is thin but achievable with careful kernel design.

---

## 5. Threading Model

### 5.1 Thread Roles

```
Thread 0 (main):     Orchestrator
  - Encodes Metal command buffers
  - Reads routing logits from GPU
  - Runs sigmoid + grouped topK (0.003 ms, CPU-only)
  - Dispatches expert pread
  - Manages token sampling

Threads 1-8:         Expert I/O pool (rayon or pthreads)
  - 8 parallel pread() calls per MoE layer
  - Writes directly into Metal staging buffer
  - Pre-warmed, no thread creation overhead

GPU:                 All compute
  - f16 GEMV for all backbone projections
  - SDPA attention (fused kernel)
  - INT4 expert dequant GEMV (existing)
  - RMSNorm, SiLU activations
```

### 5.2 Pipeline Overlap

The critical overlap is between GPU compute and CPU I/O:

```
Time ─────────────────────────────────────────────────>

GPU:  [──CMD_attn──][commit]           [──CMD_expert──]
CPU:                 [route][──pread──]
                     0.003   ~0.3 ms (cached) / ~8 ms (miss)

Layer N:   GPU attn ────> CPU route + pread ────> GPU expert FFN
Layer N+1:                GPU attn ────> CPU route + pread ────> ...
```

For cached experts (95% of cases), the pread takes ~0.3 ms which overlaps with GPU attn encode for the next layer. For cold experts, the 4.46 ms per expert is the dominant cost -- but this only happens 5% of the time.

### 5.3 CPU Work Remaining

After moving projections to Metal, the only CPU work is:

| Task | Time | Can overlap? |
|------|------|-------------|
| Routing sigmoid + topK | 0.003 ms | No (needs GPU logits) |
| Expert pread (cached) | 0.3 ms | Yes (overlap with GPU) |
| Expert pread (cold) | 4.5 ms | Yes (overlap with GPU) |
| Token sampling | 0.01 ms | No (needs GPU logits) |
| Command buffer encoding | ~0.05 ms/layer | Yes (partial) |

Total non-overlapped CPU: < 0.1 ms/layer. CPU is NOT the bottleneck.

---

## 6. I/O Strategy

### 6.1 Backbone: Zero-Copy Metal Buffer

```rust
// At load time: mmap backbone.bin, wrap as Metal buffer
let backbone_mmap = memmap2::MmapOptions::new().map(&backbone_file)?;
let backbone_metal = metal.device.newBufferWithBytesNoCopy(
    backbone_mmap.as_ptr() as *mut _,
    backbone_mmap.len() as u64,
    MTLResourceStorageModeShared,  // Unified memory: CPU + GPU access
    None,
);
```

The backbone stays as f16 on disk, mmap'd, and wrapped as a Metal buffer. The GPU reads f16 directly -- no conversion. Pre-fault all pages at startup (madvise WILLNEED).

### 6.2 Expert Loading: pread Into Metal Staging Buffer

```rust
// Pre-allocated Metal staging buffer: 8 experts x 22.3 MB = 178 MB
// Created once at startup, reused every layer
let staging = metal.device.newBuffer(8 * expert_stride, MTLResourceStorageModeShared);

// Per MoE layer:
// 1. GPU computes routing logits
// 2. CPU reads logits, runs topK
// 3. Parallel pread into staging buffer
// 4. GPU dispatches INT4 GEMV from staging buffer
```

### 6.3 Expert Cache Warmup

At startup, pre-load the ~2000 most likely experts (based on frequency statistics or first-token routing). This takes 2000 x 22.3 MB / 5 GB/s = ~9 seconds. Subsequent tokens benefit from near-100% hit rate during warmup.

### 6.4 No Custom Cache

Per flash-moe research: trust the OS page cache. No custom LRU. The OS page cache on macOS with 86 GB of headroom naturally retains ~3,850 experts. F_NOCACHE is explicitly rejected.

---

## 7. Metal f16 GEMV Kernel Design

This is the single most critical new component. Every backbone projection uses it.

### 7.1 Kernel Specification

```metal
// f16 matrix-vector multiply: y = W @ x
// W: [M, K] row-major half (f16)
// x: [K] half (f16)
// y: [M] half (f16)
//
// One threadgroup per tile of output rows.
// Each threadgroup: BM rows, each simdgroup handles one row.
// Within simdgroup: 32 threads split K dimension.

kernel void f16_gemv(
    const device half* W [[buffer(0)]],   // [M, K] f16
    const device half* x [[buffer(1)]],   // [K] f16
    device half* y [[buffer(2)]],         // [M] f16
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    // Each simdgroup computes one output row
    uint row = tgid * SIMDGROUPS_PER_TG + simd_gid;
    if (row >= M) return;

    float acc = 0.0;
    const device half* w_row = W + row * K;

    // Each thread accumulates K/32 elements
    for (uint k = simd_lid; k < K; k += 32) {
        acc += float(w_row[k]) * float(x[k]);
    }

    // Reduce across simdgroup
    acc = simd_sum(acc);

    if (simd_lid == 0) {
        y[row] = half(acc);
    }
}
```

### 7.2 Expected Performance

For o_proj [7168, 16384] f16:
- Weight data: 234.9 MB
- At 546 GB/s: 0.43 ms (memory bound)
- FLOPs: 234.9M. At 15 TFLOPS: 0.016 ms
- Arithmetic intensity: 1 FLOP/byte (deeply memory-bound)
- **Expected: ~0.5 ms** (including dispatch overhead)

This is 40x faster than the current f32 CPU path for o_proj (~20 ms).

---

## 8. Activation Data Flow

### 8.1 Precision Strategy

```
Input:     f16 (embedding table is f16)
Backbone:  f16 weights, f32 accumulation inside GEMV, f16 output
Attention:  f16 Q/K/V, f32 scores (for softmax stability), f16 output
Expert:    INT4 weights, f16 activation, f32 accumulation, f16 output
Residual:  f16 (accumulated in f32 inside kernel, stored as f16)
Output:    f16 logits -> f32 for sampling
```

Using f16 activations halves the residual stream memory traffic (7168 x 2 = 14 KB per layer instead of 28 KB). This is negligible compared to weight I/O but simplifies the pipeline.

### 8.2 KV Cache in f16

Currently the KV cache stores f32 (61 layers x 4096 x 576 x 4 = 548 MB). Switching to f16 halves this to 274 MB and allows the SDPA kernel to read it at half the bandwidth cost. The compressed latent [512] + rope [64] = 576 values per position per layer. At f16 and 4K context:

```
KV cache = 61 layers x 4096 positions x 576 dims x 2 bytes = 274 MB
```

---

## 9. Migration Path: Current -> 10 tok/s

### Phase 1: Metal f16 GEMV (0.7 -> 5 tok/s)

**Goal:** Replace CPU sgemv_f32 with Metal f16 GEMV for ALL backbone projections.

1. Write Metal f16 GEMV kernel (Section 7)
2. Wrap backbone mmap as Metal buffer (zero-copy)
3. Replace `convert_layer_into` + `sgemv_f32` with single Metal dispatch
4. Activations flow as f16 between kernels (no GPU-CPU roundtrip)
5. Keep expert path unchanged (already Metal INT4)

**Expected:** 22 ms/layer -> ~3 ms/layer = 183 ms/token = 5.5 tok/s

### Phase 2: Fused SDPA Attention Kernel (5 -> 7 tok/s)

**Goal:** Replace CPU attention (dot products + softmax) with fused Metal SDPA.

1. Implement MLA-specific SDPA kernel (adapted from MLX sdpa_vector)
2. Fused nope + rope score computation
3. Online softmax with value accumulation
4. KV cache in Metal buffer (f16)
5. No CPU-GPU synchronization during attention

**Expected:** Eliminates CPU attention bottleneck, saves ~0.5 ms/layer

### Phase 3: Full Layer Fusion (7 -> 10 tok/s)

**Goal:** Encode entire layer as one Metal command buffer with zero CPU sync.

1. Fuse RMSNorm + GEMV (avoid separate norm kernel launch)
2. Fuse shared expert gate+up+SiLU+down (all f16)
3. Pipeline: encode layer N+1 while GPU executes layer N
4. Only CPU sync point: routing (must read GPU logits for expert selection)
5. Batch all 8 expert dispatches into single command buffer

**Expected:** Reduces dispatch overhead from ~3 ms total to ~0.5 ms total

### Phase 4: Expert Warmup + Prefetch (reliability)

**Goal:** Ensure 95%+ expert cache hit rate.

1. Warmup: pre-load 2000 most-frequent experts at startup
2. Monitor hit rate per layer
3. Adaptive pool sizing based on available memory (Section 3.2)
4. Memory pressure monitoring (HOST_VM_INFO64)

---

## 10. Realistic Assessment

### What's achievable:

| Phase | tok/s | Timeline | Risk |
|-------|-------|----------|------|
| Current | 0.7 | Done | -- |
| Phase 1 (Metal f16 GEMV) | 4-6 | 2-3 days | Medium (new kernel) |
| Phase 2 (Fused SDPA) | 6-8 | 2-3 days | Medium (complex kernel) |
| Phase 3 (Full fusion) | 8-11 | 3-5 days | High (many moving parts) |
| Phase 4 (Expert warmup) | +0.5 | 1 day | Low |

### What could go wrong:

1. **Metal f16 GEMV bandwidth < 85% utilization.** If we only hit 65% utilization (355 GB/s), we get 7.7 tok/s instead of 10. Mitigation: tune threadgroup geometry, use simdgroup operations, ensure coalesced memory access.

2. **GPU dispatch overhead.** Each Metal command buffer commit has overhead. If we need 3-5 commits per layer (attn, route-sync, expert), that is 183-305 commits per token at ~0.01 ms each = 1.8-3.0 ms total. Mitigated by batching encoders into fewer command buffers.

3. **Expert SSD misses.** At 95% hit rate, misses add 8.6 ms. At 90%, they add 17 ms. The solution is aggressive warmup and large pool size. With 86 GB available for the pool, we can hold ~3,850 experts (26% of total) for ~95% hit rate.

4. **Routing sync point.** Every MoE layer requires reading GPU routing logits back to CPU for topK selection, then issuing preads. This CPU-GPU sync is unavoidable with the current SSD-based expert loading. At ~0.05 ms per sync, 58 MoE layers = 2.9 ms. Not a dealbreaker but significant.

### The honest answer:

**8-10 tok/s is achievable with Metal f16 GEMV + fused SDPA + expert warmup.** Getting above 10 requires near-perfect bandwidth utilization and may not be reliably achievable. The physical floor at 85% bandwidth utilization is 9.8 tok/s.

The transformation from 0.7 to 10 tok/s is a **14x speedup**, of which:
- ~2x comes from eliminating f16-to-f32 conversion (halving data read)
- ~5x comes from GPU vs CPU bandwidth advantage (546 vs ~100 GB/s effective)
- ~1.4x comes from pipelining and fusion (eliminating sync points)

---

## 11. Questions Answered

**Q1: What must be in RAM vs streamed from SSD?**
- **RAM (mandatory):** Backbone weights f16 (32.4 GB), KV cache (0.3 GB), Metal scratch (1 GB)
- **RAM (expert pool):** ~86 GB for ~3,850 experts, managed by OS page cache
- **SSD (on miss):** Expert weights for the ~74% of experts not in pool

**Q2: What compute should be CPU (AMX) vs GPU (Metal)?**
- **GPU:** ALL matrix-vector multiplies, attention, RMSNorm, SiLU, expert dequant
- **CPU:** Routing topK only (0.003 ms/layer), token sampling, pread I/O dispatch

**Q3: Optimal thread/pipeline architecture?**
- 1 CPU thread for orchestration + routing
- 8 I/O threads for parallel expert pread
- GPU command queue with 1-3 command buffers per layer
- Pipeline: encode layer N+1 while GPU executes layer N

**Q4: Minimum expert pool size?**
- Minimum for 10 tok/s: ~3,850 experts (95% hit rate, 8.6 ms SSD budget)
- Minimum for 5 tok/s: ~2,000 experts (90% hit rate, 17 ms SSD budget)
- Absolute minimum: ~1,000 experts (80% hit rate, 34 ms SSD budget)

**Q5: Is Accelerate's f16 BLAS available?**
- **No.** `cblas_hgemm` is not available in Apple's Accelerate framework. `vDSP_vflt16` does not exist. The only f16 compute path is Metal GPU shaders.

**Q6: Could we use Metal for ALL compute?**
- **Yes, and we must.** This is the only architecture that can reach 10 tok/s. The CPU path is fundamentally limited by the f16-to-f32 conversion overhead and lower effective bandwidth. Metal reads f16 natively at 546 GB/s with f16 ALUs.
