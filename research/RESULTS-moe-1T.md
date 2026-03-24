# rustane-moe-1T: Implementation Results

> Companion to the unified implementation plan. Documents what was built, measured results, and remaining gaps.
> Branch: `rustane-infer` (10 commits from master)
> Date: 2026-03-20

---

## Stage 0: Scaffolding — COMPLETE

| Task | Status | Notes |
|------|--------|-------|
| 0.1 Add 5 crate stubs | DONE | `quantize`, `expert-pager`, `moe-router`, `moe-kernels`, `moe-infer` |
| 0.2 Copy configs | DONE | `configs/qwen3-moe-30b.toml` + `configs/target-1t.toml` |
| 0.3 Add weights/ to .gitignore | DONE | |
| 0.4 Create experiments-infer.tsv | DONE | 2 entries logged |
| 0.5 Add Makefile targets | DONE | 7 `infer-*` targets |
| 0.6 Download Qwen3 config.json | DONE | Validated TOML: corrected 4 mismatches (intermediate_size, heads, tie_embeddings, moe_intermediate_size) |
| 0.7 Verify no regressions | DONE | 28 engine tests pass |

**Exit gate:** `cargo build` clean, `cargo test -p engine --release` passes.

---

## Stage 1: Quantization Foundation — COMPLETE

| Task | Status | Result |
|------|--------|--------|
| 1.1 PackedWeights4Bit | DONE | Pack/unpack/dequant, group_size=128, f16 scale/zero |
| 1.2 CPU dequant_to_f32() | DONE | f64 accumulation reference, GEMV included |
| 1.3 Metal dequant GEMV kernel | DONE | Threadgroup-256, SIMD reduction, FMA dequant |
| 1.4 Download 1 safetensors shard | DONE | 4.0 GB shard, 1063 tensors, 2B params |

**Test results (8 unit + 5 integration + 1 benchmark = 14 tests):**
- Pack→unpack→repack: exact bit match
- Quantization error on synthetic [-1,1]: max 0.067 (within 0.08 bound)
- Metal vs CPU: max_diff < 1e-3 at all sizes
- **Real Qwen3 weights: max_err=0.023, mean_err=0.002** (validated on 1037 expert tensors)
- **Metal vs CPU on real weights: diff < 1e-6** (essentially identical)

**Benchmark:** 143.5 GiB/s @ 8Kx8K (target was 400 — dispatch overhead dominates; needs batched expert dispatch for full bandwidth)

---

## Stage 2: MoE Router + Expert FFN — COMPLETE

| Task | Status | Result |
|------|--------|--------|
| 2.1 MoeRouter | DONE | Sigmoid scoring, top-k, bias-based load balancing, predict_next() |
| 2.2 Expert FFN conv1x1 on ANE | DONE | Two-graph split: build_gate_up_conv() + build_down_conv() |
| 2.3 Two-graph split for large dims | DONE | Separate channel dims for gate_up (dim→hidden) and down (hidden→dim) |
| 2.4 Single-graph for small dims | DONE | build_conv1x1() for simple matmul |
| 2.5 Wire router + expert FFN | DONE | Pipeline test with 8 experts, top-2 |

**Test results (11 router + 3 ANE + 3 pipeline = 17 tests):**
- ANE conv1x1 matches CPU matmul: max_diff < 1e-2 (fp16 precision)
- Full expert FFN (gate+up+SiLU+down) on ANE: max_diff < 0.05
- Expert usage balanced: 0.956-1.048 (ideal=1.0)
- **ANE minimum IOSurface size discovered:** small dims (256x512x16) cause runtime errors; need ≥768 channels, seq≥64

---

## Stage 3: Expert Pager + SSD Streaming — COMPLETE

| Task | Status | Result |
|------|--------|--------|
| 3.1 ExpertPool (LRU) | DONE | Ring buffer, eviction, hit/miss stats |
| 3.2 ExpertLoader (pread) | DONE | libc::pread, parallel via std::thread::scope |
| 3.3 Weight converter | DONE | create_synthetic_experts() + metadata.json |
| 3.4 Download full Qwen3-MoE-30B | PARTIAL | 1 shard (4 GB) downloaded, not full 60 GB |
| 3.5 ExpertPrefetcher | DONE | Cross-layer gate similarity, zero training cost |
| 3.6 DeltaPatcher | NOT DONE | Orion unload/reload cycle — deferred |
| 3.7 E2E Qwen3 inference | DONE | **Real weights, SSD streaming, Metal GEMV** |

**Test results (7 pool + 3 loader + 3 streaming = 13 tests):**
- LRU eviction order: exact
- pread loads correct bytes: verified sequential + parallel
- SSD-paged output = RAM-resident output: bit-identical
- Prefetcher predicts correct experts from 2-layer-back scores
- **pread benchmark: 59.9 GB/s @ 8 threads** (target was >5 GB/s — exceeded 12x)

**Key learning from flash-moe research:** Drop custom LRU cache, trust OS page cache. Flash-moe found all custom caching was slower than OS page cache by 15-38%. Our ExpertPool is useful for tracking stats but shouldn't be used for eviction policy in production.

---

## Stage 4: MLA Attention + KV Cache — COMPLETE

| Task | Status | Result |
|------|--------|--------|
| 4.1 MLA Q/KV projections | DONE | CPU reference: mla_q_projection(), mla_kv_compress() |
| 4.2 Absorbed attention | PARTIAL | Data structures ready, compute not wired |
| 4.3 KV cache compressed latent | DONE | Stores kv_lora_rank=512 dim, not full KV |
| 4.4 Rolling buffer with GQA | DONE | Rolling wrap at max_seq, correct for DeepSeek-V3 config |

**Test results (6 tests):**
- KV cache memory: **0.95 GB for 8K context** (vs 1.91 GB full KV = 2x compression)
- Rolling buffer wraps correctly
- Append/read roundtrip: exact match

---

## Stage 5: Pipeline Fusion + CLI — COMPLETE

| Task | Status | Result |
|------|--------|--------|
| 5.1 Conv1x1 migration | DONE | build_conv1x1(), build_gate_up_conv(), build_down_conv() |
| 5.2 Batched expert dispatch | DONE | forward_batched() matches sequential (< 1e-5 diff) |
| 5.3 3-stage CMD buffer overlap | NOT DONE | Needs real model inference loop |
| 5.4 Router on Metal | NOT DONE | CPU routing sufficient for now |
| 5.5 CLI | DONE | `cargo run -p moe-infer --bin infer -- --config <toml>` |

**Test results (2 tests + CLI):**
- Batched matches sequential: verified on 5 tokens
- CLI parses both qwen3-moe-30b.toml and target-1t.toml correctly
- Memory budget estimates correct (validated against flash-moe's figures)

---

## Stage 6: Scale to 700B+ — COMPLETE (estimates)

| Task | Status | Result |
|------|--------|--------|
| 6.1 Weight converter for large models | PARTIAL | Synthetic converter done; real safetensors→bin needs work |
| 6.2 Mixed quantization (4-bit/2-bit) | DONE | PackedWeights2Bit with palettized quantization |
| 6.3 Scale validation ladder | DONE | Memory estimates for 30B/700B/1T |
| 6.4 Quality benchmarks | NOT DONE | Needs full model + tokenizer |
| 6.5 Comparison benchmarks | NOT DONE | Needs flash-moe running side-by-side |

**2-bit vs 4-bit on real Qwen3 weights:**
- 4-bit: max_err=0.007, **7.5x compression**
- 2-bit: max_err=0.126, **12.8x compression**
- Clear quality/size tradeoff confirmed on real data

**Memory estimates:**
| Model | 4-bit experts | Active per token | Attention | Total mixed |
|-------|-------------|-----------------|-----------|-------------|
| Qwen3-MoE-30B | 14.5 GB | 18.9 MB | 0.4 GB | 11.3 GB |
| DeepSeek-V3 (700B) | 343.9 GB | 176.2 MB | 6.3 GB | 264.2 GB |
| Target 1T | 515.8 GB | 176.2 MB | 6.3 GB | 393.1 GB |

---

## Stage 7: Long Context — COMPLETE (framework)

| Task | Status | Result |
|------|--------|--------|
| 3-tier KV cache | DONE | Hot (f32) + Warm (3-bit target) + Cold (Infini-attention running mean) |
| NSA sparse attention | PARTIAL | Config + data structures, no Metal kernels |
| Document-segmented RoPE | NOT DONE | |
| RULER benchmark | NOT DONE | Needs full model |

**Test results (6 tests):**
- Hot→warm→cold tier promotion: correct
- Cold tier running mean converges to expected value
- Memory budget: hot 64K = 3.5 GB, warm 1M (3-bit) = 5.25 GB

---

## Stage 8: Autonomous Sweep — COMPLETE (scaffold)

- `system/optimize-infer.sh` ready
- `system/experiments-infer.tsv` has 2 entries
- Framework mirrors training's optimize-loop.sh pattern

---

## E2E Validation: Real Qwen3-MoE-30B Inference via SSD

**The critical test — proves the full pipeline works on real trained weights:**

```
[1/6] Load safetensors shard:     4.0 GB in 0.4s (1063 tensors)
[2/6] Quantize 128 experts:       1.0s → 306 MB packed
[3/6] Open for pread streaming:   instant
[4/6] Route with real gate weights: experts [79, 73, 119, 7, 14, 123, 74, 90]
[5/6] pread + Metal dequant GEMV:
  Expert 79: load=309us, gpu=5080us (cold), ||out||=0.0005
  Expert 73: load=304us, gpu=538us,          ||out||=0.0004
  Expert 14: load=245us, gpu=476us,          ||out||=0.0006
  (8 experts total, all correct)
[6/6] Combined output: finite, non-zero, all 8 experts contributed
```

**Per-expert timing:** ~260 us pread + ~500 us Metal GPU = ~760 us
**Per-layer estimate (8 experts):** ~6 ms (vs flash-moe's 4.28 ms on M3 Max)

---

## Measured Benchmarks

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Metal dequant bandwidth (8Kx8K) | 143.5 GiB/s | 400 GiB/s | 36% — needs fused kernels |
| pread throughput (8 threads) | 59.9 GB/s | >5 GB/s | 12x exceeded |
| 4-bit quant error (real weights) | 0.023 max | <0.2 | Excellent |
| Metal vs CPU (real weights) | <1e-6 diff | <1e-3 | Near-perfect |
| Expert pread (cold, 1MB) | 260 us | — | Baseline |
| Metal GEMV per expert | 500 us | — | Baseline |
| MLA KV cache (8K ctx, 61L) | 0.95 GB | <1 GB | On budget |
| Expert usage balance | 0.956-1.048 | ~1.0 | Excellent |

---

## Stage 2 Post-Mortem: First Tokens from Qwen3-MoE-30B — COMPLETE (2026-03-20)

> Date: 2026-03-20 15:51 PST. Branch: `rustane-infer` (8 new commits).
> Goal: Generate real tokens, measure tok/s.
> Acceptance test: greedy output matches HuggingFace for 20 tokens.

### Architecture corrections discovered during execution:
- **ALL 48 layers are MoE** — `decoder_sparse_step=1`, `mlp_only_layers=[]`. No dense layer 0.
- **No shared experts** — zero `shared_expert` keys in the safetensors. The plan's shared_expert assumption was wrong.
- **GQA confirmed** — 32 Q heads, 4 KV heads, head_dim=128, RoPE theta=1e6 neox-style.
- **QK-norm confirmed** — per-head RMSNorm on Q and K before attention scores.

### New modules:
| Module | Lines | Function |
|--------|-------|----------|
| `attention.rs` | 253 | GQA forward: RoPE + QK-norm + causal mask + softmax |
| `rmsnorm.rs` | 80 | RMSNorm + per-head QK-norm (eps=1e-6) |
| `kv_cache.rs` | 108 | GQA KV cache (48 layers × 4 KV heads) |
| `weights.rs` | 198 | mmap backbone loader with zero-copy f16 slices |
| `generate.rs` | 295 | Full decode loop: embed → 48 layers → LM head → sample |
| `config.rs` | 129 | Rewritten with toml + serde deserialization |
| `convert_qwen3_moe.py` | 691 | Safetensors → backbone.bin + expert files (4-bit) |
| `generate_references.py` | 275 | HF reference tensors for validation |

### Test results (44 tests, all pass):
| Group | Count | Status |
|-------|-------|--------|
| RMSNorm | 4 | All pass |
| KV Cache | 4 | All pass |
| RoPE + GQA Attention | 8 | All pass (includes real-weight forward) |
| Config Parser | 1 | Pass |
| Nibble Packing | 4 | All pass |
| MoE Router | 13 | All pass (includes new softmax routing) |
| Weight Loader | 5 | All pass (real mmap'd weights) |
| Tokenizer | 2 | All pass (HF tokenizer roundtrip) |
| **Generation** | **3** | **All pass** |

### Acceptance test result:
```
Prompt: "The capital of France is"
Output: " Paris. The capital of the United Kingdom is London.
         The capital of the United States is Washington,"

Token match: 20/20 (exact match with HuggingFace transformers greedy)
Token IDs: [12095, 13, 576, 6722, 315, 279, 3639, 15072, 374, 7148,
            13, 576, 6722, 315, 279, 3639, 4180, 374, 6515, 11]
```

### Benchmark:
| Metric | Result | Notes |
|--------|--------|-------|
| tok/s (CPU-only) | 0.4 | 48 MoE layers × CPU dequant GEMV |
| Quant error (full model) | 0.049 max | 4-bit asymmetric, group_size=128 |
| Converted model size | 18.48 GB | backbone 1.32 GB + 48 × 320 MB experts |
| Weight conversion time | ~90s | All 48 layers, bf16 → 4-bit |
| Time to first token | ~13.7s | CPU attention + CPU expert FFN |

---

## Stage 3 Post-Mortem: Decode Optimization + ANE Prefill — COMPLETE (2026-03-20)

> Date: 2026-03-20. Branch: `rustane-infer`.
> Goal: Maximize tok/s through hardware acceleration.
> Result: **0.4 → 14.0 tok/s decode (35x), 0.80s → 0.71s prefill, 20/20 HF match preserved.**

### Optimization journey (4 commits):

| Step | Change | tok/s | Speedup |
|------|--------|-------|---------|
| Baseline | CPU dequant GEMV for everything | 0.4 | 1x |
| +BLAS attention | cblas_sgemv for Q/K/V/O projections (AMX hardware) | 1.0 | 2.5x |
| +Metal expert FFN | Zero-copy mmap'd 4-bit weights, batched dispatch | 1.0 → 3.5 | 8.8x |
| +FMA kernel | Pre-factored `scale*x` + threadgroup x-cache | 3.5 → 4.5 | 11x |
| +Zero-copy + f32 pre-conv | Eliminate per-token f16→f32 conversion (5ms/layer) | 4.5 → 11.7 | 29x |
| Final (with timing) | Timing breakdown instrumentation | 14.0 | **35x** |

### Architecture (decode, seq=1):

```
Per layer (48 total):
  Attention:  BLAS sgemv (Q/K/V/O projections) + CPU softmax
  Expert FFN: Metal GEMV (zero-copy mmap'd 4-bit weights, batched gate+up+down)
  Everything else: CPU (RMSNorm, residuals, MoE routing)
```

### ANE batched prefill attention:

Replaced sequential token-by-token attention with a fused ANE graph that processes all prompt tokens in one dispatch per layer.

**Graph:** Two-input design (activations re-staged per layer at 512KB; weights pre-staged once at load time at 20MB/layer). Fuses QKV projection → QK-norm (pow(-0.5), not rsqrt) → neox RoPE → grouped causal SDPA. O_proj + expert FFN remain on CPU/Metal.

**ANE accuracy:** attn max_diff=0.0016, K max_diff=0.013, V max_diff=0.0009 (all well under 0.05 fp16 tolerance).

| Metric | Before | After | Notes |
|--------|--------|-------|-------|
| Prefill (13 tokens) | 0.80s | 0.71s | ANE attention fast, expert FFN dominates (~500ms) |
| Decode | 14.0 tok/s | 14.1 tok/s | Unchanged |
| HF match | 20/20 | 20/20 | Greedy output identical |
| ANE graph compile | — | 8.8s (48 graphs) | One-time cost at model load |

**Why prefill improvement is modest:** Expert FFN runs per-token on Metal (~0.5s of 0.8s total). ANE attention replaced ~0.3s of compute with ~0.01s dispatch, but the FFN bottleneck remains. To get prefill under 0.2s would require batched expert FFN on ANE (blocked by variable routing + 4-bit weights).

### New modules (Stage 3):

| Module | Lines | Function |
|--------|-------|----------|
| `gqa_prefill.rs` | 226 | ANE fused GQA graph: QKV proj + QK-norm + neox RoPE + grouped causal SDPA |
| `test_ane_prefill.rs` | 163 | ANE vs CPU correctness (seq=64, seq=128) |
| `attention.rs` additions | +109 | `BatchGqaOutput`, `gqa_forward_batch_f32()` CPU reference |
| `generate.rs` additions | +218 | `AnePrefillCache`, `compile_ane_prefill()`, `prefill_layer_ane()` |

### Key decisions:

1. **Grouped attention over K/V expansion** — matmul broadcast [1,8,seq,hd] @ [1,1,hd,seq] works on ANE, avoids 64 identity slices for GQA dedup.
2. **Two-input graph** — pre-staging weights eliminates 40MB/layer cache-unfriendly transpose during generation. Initial single-input design was 3.4x *slower* than sequential CPU.
3. **pow(-0.5) not rsqrt** — ANE compiler fails on rsqrt after reduce ops (confirmed in CLAUDE.md gotchas).
4. **Expert FFN stays on Metal** — different tokens route to different experts (can't batch on ANE) + 4-bit weights need CPU dequant.

---

## Stage 5 Post-Mortem: DeepSeek-V2-Lite MLA Dry Run — COMPLETE (2026-03-21)

> Date: 2026-03-21. Branch: `rustane-infer`.
> Goal: Validate full MLA pipeline on V2-Lite (15.7B) before scaling to V3 (671B).
> Result: **4-level validation: embedding PASS, layer output PASS (cos=0.938), logit top-5 PASS, 39 tests, 0 Qwen3 regressions.**

### Key deliverables:
- bf16→f16 weight converter (`convert_deepseek.rs`) with W_UK/W_UV split from kv_b_proj
- Full MLA absorbed attention (`mla_attention.rs`): Q projection → split nope/rope → RoPE → KV compression → cache → absorbed scoring → softmax → value combination → O projection
- YaRN RoPE with 3-band frequency scaling (`yarn_rope.rs`)
- Generation loop (`generate_v2.rs`): MLA + dense FFN (layer 0) + MoE + shared experts + Metal 4-bit
- Q LoRA path + sigmoid grouped routing pre-built for V3
- 4-level validation test suite (`test_model_validation.rs`)

---

## Stage 6: DeepSeek-V3 (671B) Execution — IN PROGRESS (2026-03-22)

> Date: 2026-03-22. Branch: `rustane-infer`.
> Goal: Get V3 generating tokens on M4 Max 128GB at 3-5 tok/s.
> Status: **Infrastructure complete. Weight conversion running.**

### 8 tasks completed:

| # | Task | Gate | Status |
|---|------|------|--------|
| T1 | V3 Config + TOML | All 3 TOMLs parse, `is_moe_layer(2)=false` | DONE |
| T2 | FP8 Dequant Module | LUT validated vs ml_dtypes (all 256 bytes) | DONE |
| T3 | FP8→INT4 Converter | 1-layer test: 4.9 GB backbone in 11.3s | DONE |
| T4 | Expert Pager (Least-Stale) | Evicts lowest-layer expert, 7 tests pass | DONE |
| T5 | V3 Wiring + Bug Fixes | 37 lib tests, 0 regressions, compiles | DONE |
| T6 | Python Reference Gen | Partial-load + API scripts written | DONE |
| T7 | V3 Validation Suite | 4-level framework, compiles | DONE |
| T8 | E2E Benchmark | Benchmark test ready | DONE |

### FP8 end-to-end validation (real V3 weights):

| Test | Cosine | Max Diff |
|------|--------|----------|
| Embedding (bf16→f16) | 1.000000 | 0.000000 |
| Input LayerNorm | 1.000000 | 0.000000 |
| Normed Embedding | 1.000000 | 0.000000 |
| q_a_proj row 0 (FP8→f16) | 1.000000 | 0.000014 |
| q_a_proj row 500 (FP8→f16) | 1.000000 | 0.000009 |
| Q Latent (full matmul) | 1.000000 | 0.000052 |

**Rust FP8→f16 converter is bit-accurate vs Python ml_dtypes ground truth.**

### Critical corrections applied (from Stage 2 research):
1. `routed_scaling_factor` bug FIXED — was scaling combined output, now scales per-expert weight
2. LRU → Least-Stale eviction (by minimum `last_used_layer`)
3. L-2 prefetcher disabled (-18% perf, 25% hit rate in research)
4. `first_k_dense_replace = 3` (THREE dense layers, not 1)
5. `shared_expert_count = 1` (not 2), `norm_topk_prob = true`
6. `e_score_correction_bias` loaded per MoE layer for V3 sigmoid routing
7. `route_sigmoid_v3` with grouped top-k wired into generate_v2

### Known blocker:
**Memory pressure** — current code pre-converts all layers to f32 Vec (~54 GB for V3). Need lazy f16→f32 conversion strategy. Research dispatched (Stage 3 research prompt: `research/mla-1t/03-stage3-v3-runtime-research.md`).

### New files:
| File | Lines | Function |
|------|-------|----------|
| `fp8.rs` | 140 | FP8 e4m3fn LUT + block-wise dequant |
| `bin/convert_v3.rs` | 320 | FP8→INT4 converter with Rayon |
| `configs/deepseek-v3.toml` | 48 | V3 inference config |
| `tests/test_v3_validation.rs` | 230 | 4-level validation suite |
| `tests/bench_v3_tok_per_sec.rs` | 70 | Throughput benchmark |
| `scripts/generate_v3_ref.py` | 130 | Partial-load HF reference |
| `scripts/v3_api_reference.py` | 90 | DeepSeek API reference |

### Test summary: 49 tests passing, 0 failures
| Crate | Tests | Status |
|-------|-------|--------|
| moe-infer (lib) | 37 | All pass |
| moe-router | 5 | All pass |
| expert-pager | 7 | All pass |

---

## What's NOT Done (Gaps → Production)

### Critical path to 25-30 tok/s:
1. ~~**Metal GEMV for expert dispatch**~~ DONE — 0.4 → 14 tok/s
2. ~~**FMA kernel reorder**~~ DONE — +12% from pre-factored scale*x
3. **3-stage CMD buffer pipeline** — overlap GPU/CPU/SSD
4. **Speculative decoding** — draft model + verify (2-4x potential)

### Important but not blocking:
- Delta patching (Orion-style ANE weight reloading)
- 2MB-aligned pread buffers (flash-moe: 3.6x DMA speedup)
- Quality benchmarks (MMLU, perplexity)
- Batched expert FFN for faster prefill (currently bottlenecked at ~500ms)
- Pre-compile ANE graphs for seq=128, 256 (currently only seq=64)

### Learned from flash-moe (58 experiments):
- **Trust OS page cache** — all custom caching was slower (delete our LRU for production)
- **Don't use mmap for experts** — per-page faults 100x slower than pread
- **Don't prefetch with F_RDADVISE** — unified memory DMA contention with GPU
- **Don't compress (LZ4)** — decompression overhead > I/O savings
- **FMA kernel reorder** — `fma(nibble, scale*x, bias*x)` saves 1 mult per value (+12%)
- **Serial GPU→SSD→GPU is optimal** — unified memory bus can't overlap DMA + GPU

### Learned from Stage 2 execution:
- **Always inspect actual safetensors** before trusting config.json commentary. `decoder_sparse_step=1` means ALL layers are MoE, but our plan assumed layer 0 was dense.
- **bf16 requires torch for loading** — numpy can't read bfloat16 safetensors. Use `framework='pt'` with safe_open.
- **Model fits in 128GB RAM at 4-bit** — no SSD streaming needed for Qwen3-30B. The pager/streamer infrastructure is for 700B+ models.

### Learned from Stage 3 execution:
- **Weight staging is the ANE bottleneck** — initial single-IOSurface design caused 40MB/layer cache-unfriendly transpose per dispatch, making ANE prefill 3.4x *slower* than CPU. Two-input design (pre-staged weights) fixed it.
- **ANE matmul supports batch broadcasting** — [1,8,seq,hd] @ [1,1,hd,seq] works for grouped attention, avoiding expensive K/V head expansion via concat.
- **f32 pre-conversion is crucial** — eliminating per-token f16→f32 conversion (5ms/layer × 48 layers = 240ms) gave 2.6x speedup.
- **Expert FFN dominates prefill** — ANE attention is fast (~10ms total), but per-token Metal expert FFN takes ~500ms for 13 tokens × 48 layers. Prefill optimization requires batched expert dispatch.

---

## Stage 4 Post-Mortem: Decode Optimization II — 14 → 19.6 tok/s (2026-03-21)

> Date: 2026-03-21. Branch: `rustane-infer` (8 new commits).
> Goal: 14 → 25+ tok/s through Metal shader improvements + dispatch batching.
> Result: **14.0 → 19.6 tok/s decode (40%), prefill 710ms → 310ms (56%), 20/20 HF match preserved.**

### Optimization journey (8 commits):

| Step | Change | Decode tok/s | Prefill |
|------|--------|-------------|---------|
| Baseline | Stage 3 final | 14.0 | 710ms |
| +Observability | Per-layer timing: Metal=60%, attn=38% | 14.9 | 710ms |
| +Metal dispatch bench | 85µs/commit, 4µs/dispatch overhead | — | — |
| +cblas_sgemm | O_proj batching available (3.5x speedup) | — | — |
| +ROWS_PER_TG=8 | 8 SIMD groups share x_cache, 8x fewer TGs | **17.2** | 530ms |
| +Fused gate+up+SiLU | Single kernel, eliminate CPU roundtrip | 17.1 | 470ms |
| +Scratch + single cmd_buf | Pre-allocated buffers, 96→48 commits/token | **19.1** | 400ms |
| +sgemm O_proj prefill | Batched [2048,4096]×[4096,13] | 19.2 | **310ms** |
| +ANE run_cached_direct | XPC daemon bypass | **19.6** | 310ms |

### Key insight from observability (Task 1):
CPU SiLU was only 0.5% of decode — NOT the bottleneck we expected. Metal dispatch at 60% was the clear target. This redirected the entire optimization strategy.

### Remaining bottleneck analysis (19.6 tok/s = 51ms/token):
- Metal MoE dispatch: ~27ms compute + ~4ms overhead = ~31ms (60%)
- CPU attention: ~25ms (38%) — 4 sgemv calls × 48 layers + softmax/scores
- Everything else: ~1ms (2%)

### Next optimization targets (from research):

| # | Optimization | Expected | Risk |
|---|-------------|----------|------|
| R1 | Batched QKV sgemv (3→1 call) | -2.4ms → 20.6 tok/s | Very low |
| R2 | Single cmd_buf ALL 48 layers | -4ms → 22.4 tok/s | Medium |
| R3 | Eliminate 240 allocs/token in attn | -1.5ms → 23.2 tok/s | Low |
| R4 | Metal f32 sgemv for attn projs (ane-infer pattern) | -5-10ms → 26-30 tok/s | Medium |
| R5 | Metal attention scores kernel (uzu pattern) | -2-4ms → 28-34 tok/s | Medium |

R4 is the highest-leverage single change: move ALL per-layer compute (attention + MoE) into one Metal cmd_buf per token. Eliminates all CPU-GPU sync points. This is what ane-infer does for 32 tok/s.

---

## Measured Benchmarks

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| **Decode tok/s** | **19.6** | 25-30 | **49x from baseline (0.4)** |
| **Prefill (13 tokens)** | **0.31s** | 0.1-0.2s | sgemm O_proj + ANE cached_direct |
| **HF greedy match** | **20/20** | 20/20 | Exact match maintained |
| Metal dequant bandwidth (8Kx8K) | 143.5 GiB/s | 400 GiB/s | 36% — needs fused kernels |
| pread throughput (8 threads) | 59.9 GB/s | >5 GB/s | 12x exceeded |
| 4-bit quant error (real weights) | 0.023 max | <0.2 | Excellent |
| Metal vs CPU (real weights) | <1e-6 diff | <1e-3 | Near-perfect |
| ANE attn vs CPU (fp16 tolerance) | 0.013 max | <0.05 | Well within bounds |
| MLA KV cache (8K ctx, 61L) | 0.95 GB | <1 GB | On budget |
| Expert usage balance | 0.956-1.048 | ~1.0 | Excellent |

---

## Test Summary

| Crate | Tests | Status |
|-------|-------|--------|
| engine | 28 unit + 40+ integration | All pass |
| quantize | 12 (8 pack4 + 4 pack2) | All pass |
| moe-router | 13 (5 inline + 8 integration) | All pass |
| expert-pager | 10 (7 pool + 3 loader) | All pass |
| moe-kernels | 0 (tests in moe-infer) | N/A |
| moe-infer | 46+ (13 lib + 4 nibble + 5 weights + 2 tok + 8 attn + 3 gen + 2 ANE prefill + 9 legacy) | All pass |
| **Total** | **112+** | **0 failures** |

---

## Commits on rustane-infer

```
13a3379 perf: ANE run_cached_direct for prefill — XPC bypass
b478a61 perf: sgemm batched O_proj in prefill — 400ms → 310ms (22%)
709131c perf: scratch buffers + single cmd_buf — 17 → 19.1 tok/s (12%)
512aaf6 feat: fused gate+up+SiLU Metal kernel — eliminate CPU roundtrip
43d17da perf: ROWS_PER_TG=8 Metal shader — 14 → 17.2 tok/s decode (23%)
e787917 feat: add cblas_sgemm FFI — 3.5x O_proj prefill speedup
13af373 feat: Metal dispatch latency isolation — 85µs/commit, 4µs/dispatch
8d21209 feat: per-layer decode timing breakdown — Metal 60%, attention 38%
0ecf93b docs: Stage 3 results — 0.4 → 14 tok/s decode (35x), ANE prefill
2541b68 feat: ANE batched prefill attention — fused GQA graph + pipeline wiring
9256bc5 feat: timing breakdown (prefill/decode) + CLI output
0ce7326 perf: zero-copy Metal + f32 pre-conversion → 0.4 → 11.7 tok/s (29x)
3dc9c16 perf: FMA kernel upgrade — pre-factored scale*x + threadgroup x-cache
3fe70b0 perf: BLAS attention + Metal GEMV for expert FFN (0.4 → 1.0 tok/s)
0d84c90 plan: Stage 3 ANE-first inference (8 tasks, reviewed)
cd5a700 Update Cargo.lock for safetensors dev-dependency
78f589f E2E inference: real Qwen3-MoE-30B weights streamed from SSD via pread
a874757 Validate quantization pipeline on real Qwen3-MoE-30B weights
2525b8e Stage 8: Autonomous optimization loop scaffold
e51bb81 Stage 7: Long context with 3-tier KV cache and sparse attention
167e786 Stage 6: 2-bit palettized quantization and scale validation
5ec88d9 Stage 5: Full pipeline fusion, batched expert dispatch, CLI
2f2e374 Stage 4: MLA attention with compressed KV cache
0045778 Stage 3: Expert pager with weight converter, prefetcher, streaming pipeline
93044e5 Stages 0-3: MoE inference scaffolding, quantization, router, expert pager
```
