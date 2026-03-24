# Implementation Plan: rustane-moe-1T

> Step-by-step plan to bring rustane from 579M dense training to 1T MoE inference on M4 Max 128GB.
> See `rustane-moe-1T.md` for full research backing every decision.

---

## Phase 1: 4-bit Dequant + Small MoE (Weeks 1-3)

### Goal
Prove 4-bit quantized inference and basic MoE routing on the existing rustane pipeline.

### Steps

- [ ] **1.1** Create `crates/quantize/` crate with 4-bit weight packing
  - `PackedWeights4Bit`: pack/unpack nibbles into u32, group_size=128
  - CPU dequant fallback: `dequant_to_f32()`
  - Roundtrip tests: pack -> unpack == original (within quantization error)

- [ ] **1.2** Write Metal dequant kernel: `expert_dequant_4bit_gemv`
  - Fused 4-bit dequant + GEMV in single Metal compute shader
  - FMA optimization: precompute `scale*x`, `bias*x` per group
  - Target: >400 GiB/s bandwidth utilization
  - Benchmark against CPU dequant baseline

- [ ] **1.3** Add 4-bit support to existing forward pass
  - Load weights as `PackedWeights4Bit`, dequant to fp32 before staging
  - Verify: 4-bit dense inference matches fp16 within 0.5 perplexity on val set
  - All existing engine tests still pass

- [ ] **1.4** Create `crates/moe-router/` crate
  - `BasicMoERouter`: sigmoid scoring, top-k selection
  - Bias-based load balancing (DeepSeek-V3/Kimi-K2 style)
  - Unit tests: verify top-k selection, gate normalization, bias updates

- [ ] **1.5** Small MoE integration test
  - Convert 48.8M gpt_karpathy FFN layers into 8 experts, top-2
  - Verify: MoE forward produces valid logits
  - Verify: expert usage is roughly balanced (within 2x)

### Exit Criteria
- 4-bit dense inference works end-to-end
- Metal dequant shader benchmarked
- MoE routing produces correct outputs
- No regressions in existing tests

---

## Phase 2: Expert Pager + SSD Streaming (Weeks 4-7)

### Goal
Stream expert weights from SSD on demand, matching flash-moe's approach.

### Steps

- [ ] **2.1** Create `crates/expert-pager/` crate
  - `ExpertPager`: opens fds, pre-allocates 2MB-aligned buffers
  - `pread()` via `libc::pread` (thread-safe, no seeking)
  - Parallel loading via `std::thread::scope` (simpler than GCD FFI initially)
  - Buffer wrapping as `MTLBuffer::newBufferWithBytesNoCopy` for Metal zero-copy

- [ ] **2.2** Expert weight file format
  - `layer_XX.bin`: 384 experts concatenated, pread at `expert_id * EXPERT_SIZE`
  - Conversion tool: `cargo run -p expert-pager --bin convert`
  - Support both 4-bit and 2-bit packed formats

- [ ] **2.3** Page cache trust validation
  - Benchmark: measure pread throughput with 4/8 parallel threads
  - Verify: no F_NOCACHE, no madvise, no app-level caching needed
  - Target: >5 GB/s sustained sequential read

- [ ] **2.4** ExpertPrefetcher
  - 3-layer MLP predictor (hidden_dim -> 128 -> 128 -> n_experts)
  - 2-layer lookahead: predict layer L+2 experts from layer L routing
  - Background thread for async pread (non-blocking)

- [ ] **2.5** Orion delta patching (`DeltaPatcher`)
  - Implement unload -> write BLOBFILE -> reload cycle
  - Transfer tmp_dir ownership to prevent race
  - Benchmark: <10ms per expert kernel reload

- [ ] **2.6** Integration: 100B MoE from converted weights
  - Download or synthesize 100B MoE weights (e.g., Mixtral-8x7B at 47B)
  - Convert to rustane expert file format
  - Run inference end-to-end with SSD streaming
  - Target: >2 tok/s

### Exit Criteria
- Expert pager streams weights at >5 GB/s
- Delta patching works in <10ms
- 100B-class MoE runs with SSD streaming
- Page cache achieves >60% hit rate

---

## Phase 3: Long Context (MSA + Sparse Attention) (Weeks 8-11)

### Goal
Support 100M+ token context with <11 GB KV cache memory.

### Steps

- [ ] **3.1** Three-tier KV cache
  - Hot: 64K recent tokens, full precision (2.2 GB)
  - Warm: 1M heavy-hitter tokens, 3-bit quantized (5.7 GB)
  - Cold: Infini-attention compressive memory (10 MB fixed)
  - KV eviction: H2O heavy-hitter oracle for warm tier selection

- [ ] **3.2** KV cache quantization (Metal kernel)
  - 3-bit per-channel quantization (pre-RoPE, following KVQuant)
  - Metal shaders for quantize/dequantize
  - Verify: <1% quality loss vs full-precision KV at 100K context

- [ ] **3.3** NSA 3-branch sparse attention (Metal)
  - Branch 1: Compressed (conv pooling KV to 256-token summaries)
  - Branch 2: Selected (MoBA routes query to top-16 1K-token blocks)
  - Branch 3: Sliding window (last 4096 tokens)
  - Gated combination with learned weights
  - Target: <3ms per layer

- [ ] **3.4** Document-segmented RoPE
  - RoPE position resets at document boundaries
  - Enables treating documents as independent context units
  - Compatible with YaRN scaling for extended context

- [ ] **3.5** Long-context validation
  - RULER benchmark at 100K, 1M, 10M tokens
  - Needle-in-haystack at 1M tokens
  - Memory usage verification: <11 GB for 100M tokens

### Exit Criteria
- 100K context with <2% quality loss vs full attention
- 1M context functional with 3-tier cache
- KV cache memory <11 GB at 100M tokens

---

## Phase 4: Conv1x1 + Full Kernel Fusion (Weeks 12-15)

### Goal
3x ANE throughput via conv1x1 conversion; batched expert execution.

### Steps

- [ ] **4.1** Conv1x1 migration for all ANE kernels
  - Convert matmul formulation to conv1x1 in MIL generation
  - Weight reshape: [out, in] -> [out, in, 1, 1] at load time
  - Verify: identical outputs, 3x throughput on ANE benchmarks

- [ ] **4.2** Batched expert kernel
  - Single MIL program processes K=8 experts in one ANE dispatch
  - Pack 8 experts into spatial dimension
  - Reduces dispatch overhead from 52ms to 6.5ms per layer

- [ ] **4.3** MLA (Multi-Latent Attention) kernels
  - Q LoRA projection: x -> q_a (down) -> RMSNorm -> q_b (up)
  - KV LoRA projection: x -> kv_a (down) -> kv_b (up) -> split K, V
  - Absorbed attention: avoid explicit KV decompression
  - 10x KV cache reduction vs standard MHA

- [ ] **4.4** Full pipeline overlap
  - Overlap: SSD pread || shared expert ANE || attention Metal
  - CMD1 (attention) -> CMD2 (routing + shared) -> pread || CMD3 (routed experts)
  - Match flash-moe's deferred CMD3 pattern

- [ ] **4.5** End-to-end benchmark on 400B MoE
  - Target: >4 tok/s with conv1x1 + batched experts
  - Compare vs flash-moe on same hardware

### Exit Criteria
- Conv1x1 gives 3x measured ANE throughput improvement
- Batched experts reduce dispatch overhead 8x
- 400B MoE runs at >4 tok/s
- Full pipeline overlap working (I/O hidden behind compute)

---

## Phase 5: 1T Production + Sweep Agent (Weeks 16-20)

### Goal
Full 1T MoE inference at target quality and speed.

### Steps

- [ ] **5.1** Weight converter for Kimi-K2 / DeepSeek-V3
  - HuggingFace safetensors -> rustane expert file format
  - Support: MLA attention weights, 384 experts per layer, shared experts
  - Mixed quantization: 4-bit hot experts, 2-bit cold experts

- [ ] **5.2** Create `crates/moe-infer/` top-level inference crate
  - Full inference loop (Section 2.3 pseudocode)
  - CLI: `--model-dir`, `--quant`, `--experts`, `--top-k`, `--context`, etc.
  - Interactive chat mode
  - Benchmark mode (automated tok/s, RAM, power measurements)

- [ ] **5.3** Autonomous sweep agent
  - Extend system/optimize-loop.sh for MoE configs
  - Sweep: n_experts, top_k, quant_bits, prefetch_depth, cache_strategy
  - Record: tok/s, quality (perplexity), peak RAM, power
  - Output: Pareto-optimal configurations

- [ ] **5.4** Quality validation
  - MMLU 5-shot: target >80
  - HumanEval pass@1
  - Tool-calling accuracy: >90%
  - RULER long-context at 1M tokens

- [ ] **5.5** Production hardening
  - Graceful OOM handling (evict experts before crashing)
  - Thermal monitoring (reduce parallelism if approaching 100C)
  - Progress reporting (tokens generated, cache hit rate, power draw)
  - Error recovery (corrupted expert file detection)

### Exit Criteria
- 1T MoE runs at >3 tok/s on M4 Max 128GB
- Quality metrics meet targets
- Tool-calling works reliably
- System is stable for sustained multi-hour sessions

---

## Summary Timeline

```
Week  1-3:   Phase 1  (4-bit + basic MoE)
Week  4-7:   Phase 2  (expert pager + SSD streaming)
Week  8-11:  Phase 3  (long context + sparse attention)
Week 12-15:  Phase 4  (conv1x1 + full fusion)
Week 16-20:  Phase 5  (1T production + sweep)
```

## Key Dependencies

| Dependency | Source | Purpose |
|-----------|--------|---------|
| ane crate | github.com/ncdrone/ane | ANE private API FFI |
| objc2 + objc2-metal | crates.io | Metal GPU + Objective-C runtime |
| half | crates.io | fp16 type |
| libc | crates.io | pread, posix_memalign |
| memmap2 | crates.io | Memory-mapped I/O |
| 1T MoE weights | HuggingFace (Kimi-K2 or DeepSeek-V3) | Model weights |
| 4TB+ SSD | Hardware | Expert storage |

## Success Metrics

| Metric | Target | Stretch |
|--------|--------|---------|
| tok/s (warm) | 5 | 12+ |
| tok/s (cold) | 2 | 5 |
| Peak RAM | <100 GB | <80 GB |
| Context length | 100K | 100M |
| MMLU 5-shot | >80 | >85 |
| Tool-calling | >90% | >95% |
| ANE power | <5W | <3W |
| Expert cache hit | >70% | >85% |
