# Stage 3 Research Prompt: V3 Runtime on M4 Max 128GB

> Give this to your research orchestrator. Dispatch 2 waves × 3 agents = 6 agents.
> Results should be saved to: research/mla-1t/04-stage3-findings.md
> Date: 2026-03-22

---

## Context

We have DeepSeek-V3 (671B MoE, MLA attention, 256 experts/layer) running through a Rust inference engine on Apple M4 Max 128GB. The full infrastructure is built and validated:

- **FP8→f16 converter proven** — cosine=1.000000 against Python ml_dtypes on real V3 weights (q_a_proj 1536×7168)
- **V2-Lite (15.7B MLA) validated** — 4-level validation passes, same code path as V3
- **Qwen3-MoE-30B (GQA)** — 19.6 tok/s decode, 20/20 HF match, Metal 4-bit expert dispatch
- **Expert pager** — Least-Stale eviction (by layer index), pread loader at 59.9 GB/s
- **Full V3 config** — 7168 hidden, 61 layers, 128 heads, 256 experts, Q LoRA (rank 1536), sigmoid grouped routing with e_score_correction_bias

**Current blocker:** The runtime pre-converts ALL layer weights from f16 to f32 Vec at model load. For V3, this would consume ~54 GB of RAM for f32 copies alone (900 MB/layer × 61 layers), plus 22 GB backbone mmap, leaving only ~52 GB for the expert pool (vs 96 GB budgeted). We need a lazy/streaming weight strategy.

**Target:** 3-5 tok/s warm decode, fitting in 128 GB with maximum expert pool capacity.

---

## Wave 1: Memory Architecture (3 agents in parallel)

### RQ1: Lazy f16→f32 Conversion Strategy

The current code (generate_v2.rs) pre-converts all layers to `MlaLayerF32` (Vec<f32>) at startup. This worked for V2-Lite (2048 hidden, 27 layers, ~2 GB f32 total) but won't work for V3 (~54 GB f32 total).

Research these alternatives:

1. **On-the-fly conversion per layer per token**: Keep weights as f16 in the mmap'd backbone.bin. For each token, convert the current layer's weights to f32 in a reusable scratch buffer before the BLAS call. Questions:
   - What is the actual throughput of `f16::to_f32()` on M4 Max? (Neon VCVT instruction, should be ~50 GB/s)
   - For V3's largest per-layer matrix (o_proj: 7168×16384 = 117M elements), how long does f16→f32 take? Is it < 1ms?
   - Can we use Accelerate's `vDSP_vflt16` or similar for SIMD f16→f32 conversion?
   - Does Apple's Accelerate BLAS (cblas_sgemv) support f16 input directly? Check both vecLib and AMX documentation.
   - Does the `half` crate's `to_f32()` auto-vectorize with rustc + `-C target-cpu=native` on Apple Silicon?

2. **Partial pre-conversion**: Pre-convert only the small matrices (norms, router, shared expert) to f32 and leave the large projections (q_a, q_b, kv_a, w_uk, w_uv, o_proj, dense FFN) as f16. How much RAM does this save?

3. **Layer-streaming with double buffer**: Allocate two f32 scratch buffers (one being used, one being converted). While layer N runs on the current buffer, convert layer N+1's weights into the other buffer on a background thread. Does the conversion overlap with compute?

For each approach: estimate RAM usage, conversion latency per token, and implementation complexity.

### RQ2: Expert Pool Capacity Optimization

We budgeted 96 GB for the expert pool (4,314 experts @ 22.3 MB each in INT4). With the lazy weight strategy, we may have more or less headroom.

Research:

1. **What expert pool capacity maximizes hit rate for V3?** The model has 58 MoE layers × 256 experts = 14,848 total experts, selecting 8 per layer per token. During a single decode step, 58 × 8 = 464 expert slots are needed. With Least-Stale eviction:
   - What steady-state hit rate do we expect at pool sizes of 500, 1000, 2000, 4000 experts?
   - What's the theoretical minimum pool size for >90% hit rate? (Consider that experts at adjacent layers may overlap)
   - Is there empirical data from DeepSeek-V3 on expert routing patterns? (e.g., do certain experts dominate across tokens?)

2. **Expert file layout for NVMe performance**: Our expert files are per-layer (layer_XX_experts.bin, 256 experts contiguous). For pread:
   - Is per-layer or per-expert-across-layers better for the OS page cache?
   - What's the optimal read size for Apple's NVMe controller? (Research suggests 2MB DMA alignment)
   - Should we pad expert data to 2MB boundaries even if the actual expert is 22.3 MB?

3. **Memory pressure monitoring**: On macOS, how do we detect when we're approaching the memory limit? Can we dynamically shrink the expert pool to avoid swap? Research `vm_statistics`, `host_statistics`, or `os_proc_available_memory()`.

### RQ3: W_UK Absorbed Attention at V3 Scale

V3 has 128 heads (vs V2-Lite's 16). The W_UK absorption step computes `q_absorbed[h] = W_UK[h]^T @ q_nope[h]` for each head — 128 serial `sgemv_f32_trans` calls of [128, 512] × [128] → [512].

Research:

1. **Rayon parallel W_UK**: Split 128 heads across Rayon worker threads. Each head's sgemv is independent.
   - What's the optimal chunk size? (heads per thread)
   - What's the expected speedup on M4 Max (12 P-cores + 4 E-cores)?
   - Does Accelerate's sgemv use AMX? If so, do parallel sgemv calls contend for AMX units?

2. **Batched sgemv via sgemm**: Instead of 128 separate sgemv calls, reshape into one sgemm: [128 × 128, 512] × [512, 1] → [128 × 128, 1], then reshape back. Is sgemm more efficient than 128 sgemv for these dimensions?

3. **Metal kernel**: Write a single Metal compute kernel that processes all 128 heads in one GPU dispatch. Each threadgroup handles one head's [128, 512] × [128] dot products. Expected throughput?

4. **Actually measure the current serial cost**: On M4 Max, what is the wall-clock time for 128 × sgemv_f32_trans([128, 512] × [128] → [512])? This tells us whether W_UK is even a bottleneck or if we should focus elsewhere.

---

## Wave 2: Performance Modeling (3 agents in parallel)

### RQ4: V3 Decode Latency Budget

Build a per-component latency model for V3 decode (single token, seq_len=100):

For each of the 61 layers, estimate:
1. **RMSNorm**: 7168-dim vector, 2 passes (input + post_attn). Time on CPU?
2. **Q LoRA projection**: sgemv [1536, 7168] + RMSNorm [1536] + sgemv [24576, 1536]. Total time?
3. **KV compression**: sgemv [576, 7168]. Time?
4. **W_UK absorption**: 128 × sgemv_trans [128, 512]. Time? (This is RQ3)
5. **Attention scores**: 128 heads × seq_len dot products of [512] + [64]. Time?
6. **V reconstruction**: 128 heads × weighted sum + sgemv [128, 512]. Time?
7. **O projection**: sgemv [7168, 16384]. Time?
8. **MoE routing**: sgemv [256, 7168] + sigmoid + grouped top-k. Time?
9. **Shared expert FFN**: 3 × sgemv (gate [2048, 7168], up [2048, 7168], down [7168, 2048]) + SiLU. Time?
10. **Routed expert FFN**: 8 × Metal 4-bit GEMV (gate+up fused [2048, 7168], down [7168, 2048]). Time?
11. **Expert loading (on miss)**: pread 22.3 MB at 59.9 GB/s. Time?

Sum all components. What's the predicted ms/token? What's the predicted tok/s?

Reference Qwen3 measurements for calibration:
- Qwen3 (2048 hidden, 48 layers, 128 experts, top-8): 19.6 tok/s = 51ms/token
- Metal MoE dispatch: ~31ms (60%)
- CPU attention: ~25ms (38%)

### RQ5: llama.cpp and mlx-lm V3 Performance

Search for actual measured V3 performance numbers on M4 Max:

1. **llama.cpp**: Does ggml support DeepSeek-V3? If so, what tok/s on M4 Max 128GB? What quantization format? How do they handle expert streaming?
2. **mlx-lm**: Does MLX support DeepSeek-V3? Performance? How do they handle the 641 GB weight set?
3. **Any other framework** running V3 on Apple Silicon with measured tok/s?

We need realistic reference points. Our 3-5 tok/s target may be too high or too low.

### RQ6: Optimization Opportunities Specific to V3

Given V3's architecture, what optimizations are uniquely valuable?

1. **Speculative decoding with V2-Lite as draft model**: V2-Lite fits entirely in RAM (~4 GB). Can it serve as a draft model for V3? They share the same tokenizer and vocabulary. Expected acceptance rate? tok/s multiplier?

2. **Expert caching across tokens**: In autoregressive generation, do expert routing patterns show temporal locality? (i.e., token N uses similar experts to token N-1). If so, the Least-Stale policy naturally exploits this — but by how much?

3. **Quantization-aware expert tiering**: Keep "hot" experts at INT4 (22.3 MB) but store "cold" experts at INT2 (11.2 MB). This doubles effective pool capacity. What's the quality impact? Research DeepSeek's own analysis of expert importance distribution.

4. **KV cache compression**: V3's MLA KV cache is already compressed (512-dim latent vs full KV). At 4K context, it's only ~300 MB. At 32K context, it's ~2.4 GB. Is this a concern, or is the expert pool the binding constraint?

---

## Output Format

For each research question, provide:
1. **Direct answer** with specific numbers (latency in ms, throughput in GB/s, memory in GB)
2. **Source** (link to code, paper, benchmark result)
3. **Confidence level** (measured / estimated from similar hardware / theoretical)
4. **Recommendation** for our implementation

Save results to: `research/mla-1t/04-stage3-findings.md`

## Priority

**RQ1 is highest priority** — if lazy weight conversion adds > 5ms/token, we need to rethink the memory layout before running V3. RQ4 gives us the predicted baseline to compare against. RQ5 tells us what's achievable.
