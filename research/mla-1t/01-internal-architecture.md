# MLA Internal Architecture Research

> Internal research: what we have, what needs to change, exact compute costs.
> Date: 2026-03-21. Target: DeepSeek-V3 (671B) → Kimi-K2 (1T).

## The Core Finding

MLA replaces 4 simple matmuls (Q/K/V/O projections) with a 2-stage LoRA pipeline + absorbed attention. The payoff: KV cache drops 57x (from 9.57 MB/token with full MHA to 137 KB/token with MLA). The cost: more complex attention math, but fewer total FLOPs.

## MLA Decode Path (Single Token)

For hidden state `x [7168]`, position `pos`, `seq_len` cached tokens:

```
STEP 1: Q projection (two-stage LoRA)
  q_latent = x @ W_qa^T           [7168] → [1536]        22.0M FLOPs
  q_latent = RMSNorm(q_latent)    [1536] → [1536]        ~3K FLOPs
  q = q_latent @ W_qb^T           [1536] → [24576]       75.5M FLOPs
  q_nope, q_pe = split(q)         → [128 heads, 128] + [128 heads, 64]
  q_pe = apply_rope(q_pe, pos)

STEP 2: KV compression
  kv_out = x @ W_kva^T            [7168] → [576]         8.3M FLOPs
  kv_latent, k_pe = split(kv_out) → [512] + [64]
  kv_latent = RMSNorm(kv_latent)  [512] → [512]
  k_pe = apply_rope(k_pe, pos)
  CACHE: store kv_latent [512] + k_pe [64]  ← this is the 57x saving

STEP 3: Absorbed attention (no K/V reconstruction)
  W_UK = W_kvb[:, :128, :]        [128 heads, 128 nope, 512 lora]
  W_UV = W_kvb[:, 128:, :]        [128 heads, 128 v_dim, 512 lora]

  q_absorbed = q_nope @ W_UK      [128, 128] @ [128, 128, 512] → [128, 512]  8.4M FLOPs
  scores_nope = q_absorbed @ cache_latent^T    [128, 512] @ [seq, 512]^T     128*512*seq
  scores_rope = q_pe @ cache_rope^T            [128, 64] @ [seq, 64]^T       128*64*seq
  scores = (scores_nope + scores_rope) * scale
  weights = softmax(scores)

STEP 4: Value combination (absorbed)
  v_latent = weights @ cache_latent            [128, seq] @ [seq, 512]        128*seq*512
  v = v_latent @ W_UV              [128, 512] @ [128, 128, 512]^T → [128, 128]  8.4M FLOPs

STEP 5: Output projection
  output = concat(v) @ W_o^T      [16384] → [7168]       234.9M FLOPs
```

**Total fixed FLOPs per layer: 357.5M** (dominated by O_proj at 234.9M)
**Seq-dependent: 139K × seq_len** per layer

## Compute Budget (Full Token)

| Component | Layers | FLOPs/layer | Total | % |
|-----------|--------|-------------|-------|---|
| MLA attention | 61 | 497M | 30.3G | 38% |
| MoE FFN (8 experts) | 58 | 797M | 46.2G | 57% |
| Dense FFN (layers 0-2) | 3 | 791M | 2.4G | 3% |
| LM head | 1 | 1.85G | 1.85G | 2% |
| **TOTAL** | | | **80.7G** | |

## Theoretical Performance

| Scenario | Compute | I/O | Estimated tok/s |
|----------|---------|-----|-----------------|
| All in RAM (impossible — 352GB) | 160ms | 0 | 6.3 |
| SSD, 70% cache hit | 160ms | 174ms | 3-4 |
| SSD, cold start | 160ms | 580ms | ~1.4 |
| Flash-moe-ane (M3 Max 48GB) | — | — | 4.36 (measured) |

## Weight Tensors (New vs Qwen3-MoE-30B)

### New MLA tensors per layer:
| Tensor | Shape | Bytes (fp8) |
|--------|-------|-------------|
| q_a_proj | [1536, 7168] | 11.0M |
| q_a_layernorm | [1536] | 6K |
| q_b_proj | [24576, 1536] | 37.7M |
| kv_a_proj_with_mqa | [576, 7168] | 4.1M |
| kv_a_layernorm | [512] | 2K |
| kv_b_proj | [32768, 512] | 16.8M |
| o_proj | [7168, 16384] | 117.4M |

### New MoE tensors per layer:
| Tensor | Shape | Notes |
|--------|-------|-------|
| gate.e_score_correction_bias | [256] | Router bias |
| shared_experts.{gate,up,down}_proj | [2048,7168] etc | Always-resident shared expert |

### Dense FFN (layers 0-2 only):
| Tensor | Shape | Notes |
|--------|-------|-------|
| mlp.{gate,up,down}_proj | [18432,7168] etc | Standard SwiGLU, no experts |

## KV Cache Memory

| Context | f32 | f16 | vs GQA MHA (f32) |
|---------|-----|-----|-------------------|
| 8K | 1.09 GB | 0.55 GB | 57x smaller than MHA |
| 32K | 4.38 GB | 2.19 GB | |
| 128K | 17.5 GB | 8.76 GB | fits in target-1t.toml 8GB budget at f16 |

## What Our Existing Code Handles Already

- Metal 4-bit dequant GEMV → works for DeepSeek-V3 expert dims [2048, 7168]
- Fused gate+up+SiLU kernel → works (same SwiGLU pattern)
- Single cmd_buf per layer → works
- Scratch buffer pattern → works (just larger buffers)
- MoeRouter with softmax → needs sigmoid variant + bias correction
- MlaKvCache → already scaffolded in mla.rs
- RMSNorm → already have it
- RoPE → already have it (need partial rotary for 64/256 dims)

## What Needs to Be Built

1. **MLA forward pass** — absorbed attention (the math above)
2. **Weight converter** — FP8 safetensors → 4-bit rustane format
3. **Shared expert** — always-resident FFN alongside routed experts
4. **Dense FFN layers** — layers 0-2 standard SwiGLU (no MoE)
5. **Sigmoid routing** — replace softmax with sigmoid + bias correction
6. **Partial RoPE** — apply to only 64 of 256 dims (qk_rope_head_dim)
7. **SSD expert streaming** — wire expert-pager pread into generate loop

## Flash-moe-ane: No MLA

Flash-moe-ane runs Qwen3.5-397B which uses GatedDeltaNet, not MLA. No MLA implementation exists there. But their Metal attention shaders (attn_scores_batched, attn_softmax_batched, attn_values_batched) and 3-CMD pipeline design are templates for the MLA absorbed attention kernel.

## Key Architectural Decision

**Should MLA projection weights be quantized to 4-bit or kept in f16?**

The backbone (attention weights) is only ~8.5GB at 4-bit (~17GB at f16). Both fit in 128GB RAM. Keeping f16 gives higher quality + uses Accelerate BLAS at ~3 TFLOPS (vs 0.5 TFLOPS for our 4-bit Metal path). The 4-bit path is only needed for expert FFN weights that stream from SSD.

**Recommendation:** Keep MLA projection weights in f16, use BLAS. Quantize only expert FFN weights to 4-bit for SSD streaming.
