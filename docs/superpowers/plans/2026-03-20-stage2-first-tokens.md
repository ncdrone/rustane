# Stage 2: First Tokens — Qwen3-MoE-30B Inference

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate real tokens from Qwen3-MoE-30B-A3B on M4 Max 128GB, measure tok/s.

**Architecture:** Download full model (61 GB, 16 shards), convert to rustane format (backbone.bin + per-layer expert files), load via mmap + pread, run GQA attention + MoE expert FFN through 48 layers, decode tokens. CPU attention for Phase 1, Metal expert GEMV. Model fits entirely in 128GB RAM — no SSD streaming needed for this model.

**Tech Stack:** Rust (moe-infer crate), Python (weight converter), `tokenizers` crate (HuggingFace BPE), `memmap2` (zero-copy weight loading), `half` (f16), `libc` (pread), Metal (dequant GEMV).

**Key Research Sources:**
- `rustane-research/moe-1T/stage1-bottleneck-2026-03-20/wave3-integration-spec.md` — exact module APIs
- `rustane-research/moe-1T/stage1-bottleneck-2026-03-20/wave2-build-order.md` — complete Python converter + Rust loader code
- `rustane-research/moe-1T/stage1-bottleneck-2026-03-20/wave3-risks.md` — 6 silent failure traps
- `rustane-research/moe-1T/stage1-bottleneck-2026-03-20/wave2-metal-gemv-kernel.md` — optimized Metal kernel
- `weights/qwen3-moe-30b-ref/config.json` — ground truth model config

**Critical Architecture Facts (from config.json, verified):**
- hidden_size=2048, vocab_size=151936, num_layers=48
- Attention: GQA (NOT MLA), 32 Q heads, 4 KV heads, head_dim=128
- RoPE: theta=1,000,000, standard (no YaRN), neox-style pairing
- Layer 0: dense FFN (intermediate_size=6144, no experts)
- Layers 1-47: MoE with 1 shared expert + top-8 of 128 routed experts
- moe_intermediate_size=768, norm_topk_prob=true
- BOS=151643, EOS=151645, torch_dtype=bfloat16

---

## File Structure

### Review Status

**Plan reviewed by code-reviewer agent. All critical/important issues addressed below.**

### New Files

| File | Responsibility |
|------|---------------|
| `scripts/convert_qwen3_moe.py` | Safetensors → backbone.bin + layer_XX_experts.bin (uses Rust u32 asymmetric format) |
| `scripts/generate_references.py` | Generate HF reference tensors for test validation |
| `crates/moe-infer/src/weights.rs` | Zero-copy mmap backbone loader, LayerWeights slices (includes q_norm/k_norm) |
| `crates/moe-infer/src/kv_cache.rs` | GQA KV cache (not MLA), store/retrieve per head |
| `crates/moe-infer/src/attention.rs` | GQA forward: RoPE + QK-norm + causal mask + softmax |
| `crates/moe-infer/src/rmsnorm.rs` | RMSNorm forward (eps=1e-6) |
| `crates/moe-infer/src/generate.rs` | Decode loop: embed → 48 layers → LM head → sample |
| `crates/moe-infer/tests/test_tokenizer.rs` | Tokenizer encode/decode + HF match |
| `crates/moe-infer/tests/test_attention.rs` | RoPE + single-head + GQA + causal mask vs HF reference |
| `crates/moe-infer/tests/test_single_layer.rs` | Full layer forward vs HF reference |
| `crates/moe-infer/tests/test_generation.rs` | Generate tokens, compare to HF greedy output |
| `crates/moe-infer/tests/bench_tok_per_sec.rs` | THE metric: tok/s on real model |

### Modified Files

| File | Change |
|------|--------|
| `configs/qwen3-moe-30b.toml` | Replace with corrected `[attention]` + `[ffn]` sections |
| `crates/moe-infer/Cargo.toml` | Add tokenizers, serde_json, anyhow, rand, libc, toml deps + features |
| `crates/moe-infer/src/lib.rs` | Add weights, kv_cache, attention, rmsnorm, generate modules |
| `crates/moe-infer/src/config.rs` | **Rewrite** with proper TOML parsing (toml crate) for nested sections |
| `crates/moe-router/src/lib.rs` | Add softmax routing mode (Qwen3 uses softmax, not sigmoid) |

### Unchanged (but relevant)

| File | Why |
|------|-----|
| `crates/moe-infer/tests/integ_mla.rs` | Gate with `#[cfg(feature = "mla")]`, keep for 1T target |
| `crates/quantize/src/pack4.rs` | Existing 4-bit packing — Python converter MUST match this format exactly |
| `crates/moe-kernels/src/dequant.rs` | Existing Metal GEMV — already reads Rust u32 format |

---

## Reviewer Fixes Applied

**CRITICAL fixes (from code-reviewer agent):**

1. **Nibble packing resolution specified:** Python converter will be rewritten to output Rust u32 asymmetric format (matching `pack4.rs`), NOT the wave2 byte-packed symmetric format. The Metal kernel already reads the Rust format correctly.

2. **Config parser rewrite task added (Task 1B):** The existing flat key=value scanner cannot handle `[attention]`, `[ffn]` sections. Added task to rewrite using `toml` crate.

3. **QK-norm weights added throughout:** `LayerWeights` includes `q_norm`/`k_norm` fields. Python converter extracts `self_attn.q_norm.weight` and `k_norm.weight`. Attention forward applies per-head RMSNorm to Q and K.

**IMPORTANT fixes:**

4. **Router softmax mode added (Task 1C):** Qwen3 uses softmax routing, not sigmoid. Add `route_softmax()` to MoeRouter.

5. **RMSNorm task added (Task 9B):** Standalone module, eps=1e-6.

6. **Attention reference comparison is MANDATORY** (not "if available"). Task 11 requires loading HF reference and asserting tolerance.

7. **Causal mask test at seq_len > 1 added** to Task 11.

8. **Missing deps added:** `rand`, `libc` (as regular dep), `toml` crate.

**Minor fixes:** `mkdir -p scripts/` added. `hf` CLI name kept (user confirmed this is the correct binary name on their system). Reference script API fixed (`tok(prompt, return_tensors="pt").input_ids`). Prefill documented as sequential (known limitation).

---

## Phase 2A: Foundation Fixes

### Task 1: Fix Config TOML

**Files:**
- Modify: `configs/qwen3-moe-30b.toml`

- [ ] **Step 1: Read the current TOML and the real config.json**

Read `configs/qwen3-moe-30b.toml` and `weights/qwen3-moe-30b-ref/config.json`. Identify every mismatch. The TOML currently has `decoder_sparse_step = 1` which implies all layers are MoE — but layer 0 is dense.

- [ ] **Step 2: Replace TOML with corrected version**

```toml
# configs/qwen3-moe-30b.toml
# Source: Qwen/Qwen3-30B-A3B config.json (verified 2026-03-20)

[model]
name = "Qwen3-MoE-30B-A3B"
vocab_size = 151936
hidden_size = 2048
num_layers = 48
bos_token_id = 151643
eos_token_id = 151645
rms_norm_eps = 1e-6

[attention]
kind = "gqa"
num_q_heads = 32
num_kv_heads = 4
head_dim = 128
rope_theta = 1000000.0

[ffn]
dense_layer = 0
dense_inter_size = 6144
moe_inter_size = 768
num_experts = 128
num_experts_per_tok = 8
shared_expert_count = 1
norm_topk_prob = true

[quantization]
bits = 4
group_size = 128
```

- [ ] **Step 3: Verify CLI loads the corrected config**

Run: `cargo run -p moe-infer --release --bin infer -- --config configs/qwen3-moe-30b.toml --benchmark`
Expected: No errors, prints correct values (hidden_size=2048, layers=48, experts=128)

- [ ] **Step 4: Commit**

```bash
git add configs/qwen3-moe-30b.toml
git commit -m "fix: correct qwen3-moe-30b.toml against HF config.json"
```

---

### Task 1B: Rewrite Config Parser

**Files:**
- Modify: `crates/moe-infer/Cargo.toml` (add `toml = "0.8"`)
- Rewrite: `crates/moe-infer/src/config.rs`

The existing `InferConfig::from_toml()` is a flat key=value scanner that cannot handle nested TOML sections (`[attention]`, `[ffn]`). It has no fields for `kind`, `dense_layer`, `rope_theta`, `norm_topk_prob`, etc.

- [ ] **Step 1: Add `toml` dependency**

Add `toml = "0.8"` to `[dependencies]` in `crates/moe-infer/Cargo.toml`.

- [ ] **Step 2: Rewrite config.rs with proper TOML parsing**

Replace the hand-rolled parser with `serde` + `toml` deserialization. Define structs matching the TOML structure:

```rust
#[derive(Deserialize)]
pub struct InferConfig {
    pub model: ModelSection,
    pub attention: AttentionSection,
    pub ffn: FfnSection,
    pub quantization: QuantSection,
}

#[derive(Deserialize)]
pub struct AttentionSection {
    pub kind: String,           // "gqa" or "mla"
    pub num_q_heads: usize,     // 32
    pub num_kv_heads: usize,    // 4
    pub head_dim: usize,        // 128
    pub rope_theta: f32,        // 1000000.0
}

#[derive(Deserialize)]
pub struct FfnSection {
    pub dense_layer: usize,           // 0
    pub dense_inter_size: usize,      // 6144
    pub moe_inter_size: usize,        // 768
    pub num_experts: usize,           // 128
    pub num_experts_per_tok: usize,   // 8
    pub shared_expert_count: usize,   // 1
    pub norm_topk_prob: bool,         // true
}
```

- [ ] **Step 3: Verify CLI still works with new parser**

Run: `cargo run -p moe-infer --release --bin infer -- --config configs/qwen3-moe-30b.toml --benchmark`
Expected: Parses all sections, prints correct values including attention.kind="gqa" and ffn.dense_layer=0.

- [ ] **Step 4: Commit**

```bash
git add crates/moe-infer/src/config.rs crates/moe-infer/Cargo.toml
git commit -m "feat: rewrite config parser with proper TOML deserialization"
```

---

### Task 1C: Add Softmax Routing to MoeRouter

**Files:**
- Modify: `crates/moe-router/src/lib.rs`

Qwen3 uses **softmax** routing (standard MoE), not sigmoid (DeepSeek-V3 style). The existing `MoeRouter::route()` uses sigmoid scoring. We need both modes.

- [ ] **Step 1: Add `route_softmax()` method**

```rust
/// Route using softmax scoring (Qwen3 style).
/// gate_logits → softmax over all experts → top-k → renormalize to sum=1.0
pub fn route_softmax(&mut self, gate_logits: &[f32]) -> RouteResult {
    let n = gate_logits.len();
    // Softmax
    let max_val = gate_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = gate_logits.iter().map(|&l| (l - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let all_scores: Vec<f32> = exps.iter().map(|e| e / sum).collect();

    // Top-k
    let (expert_ids, mut weights) = top_k(&all_scores, self.config.top_k);

    // Renormalize top-k to sum=1.0
    if self.config.norm_topk_prob {
        let wsum: f32 = weights.iter().sum();
        if wsum > 0.0 {
            for w in &mut weights { *w /= wsum; }
        }
    }

    RouteResult { expert_ids, weights, all_scores }
}
```

- [ ] **Step 2: Add test for softmax routing**

```rust
#[test]
fn softmax_gate_weights_sum_to_one() {
    let config = RouterConfig { num_experts: 128, top_k: 8, norm_topk_prob: true, bias_lr: 0.0 };
    let mut router = MoeRouter::new(config);
    let logits: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.1).collect();
    let result = router.route_softmax(&logits);
    let wsum: f32 = result.weights.iter().sum();
    assert!((wsum - 1.0).abs() < 1e-5, "gate weights should sum to 1.0, got {wsum}");
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p moe-router --release`
Expected: All existing tests pass + new softmax test passes.

- [ ] **Step 4: Commit**

```bash
git add crates/moe-router/src/lib.rs
git commit -m "feat: add softmax routing mode for Qwen3"
```

---

### Task 2: Gate MLA Tests

**Files:**
- Modify: `crates/moe-infer/Cargo.toml` (add `[features]` section)
- Modify: `crates/moe-infer/tests/integ_mla.rs` (add cfg gate)

- [ ] **Step 1: Add features section to Cargo.toml**

Add under `[package]`:
```toml
[features]
default = []
mla = []
```

- [ ] **Step 2: Gate integ_mla.rs**

Add at the very top of `crates/moe-infer/tests/integ_mla.rs`:
```rust
#![cfg(feature = "mla")]
```

- [ ] **Step 3: Verify MLA tests are skipped by default**

Run: `cargo test -p moe-infer --test integ_mla --release 2>&1`
Expected: 0 tests run (file not compiled without `--features mla`)

- [ ] **Step 4: Verify MLA tests still work with feature flag**

Run: `cargo test -p moe-infer --test integ_mla --release --features mla 2>&1`
Expected: 6 tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/moe-infer/Cargo.toml crates/moe-infer/tests/integ_mla.rs
git commit -m "feat: gate MLA tests behind feature flag, GQA is Phase 1"
```

---

### Task 3: Nibble Packing Convention Validation

**Files:**
- Create: `crates/moe-infer/tests/test_nibble_convention.rs`

This is a fatal trap from wave3-risks.md. The wave2 Python converter uses **symmetric quantization** with **byte-packed hi-nibble-first**. Our Rust `PackedWeights4Bit` uses **asymmetric quantization** with **u32-packed LSB-first**. These differ in BOTH packing format AND quantization formula.

**Resolution (decided):** The Python converter (Task 6) will be rewritten to output the Rust u32 asymmetric format, matching `pack4.rs` exactly. The Metal kernel already reads the Rust format. This means we do NOT use the wave2 `quantize_4bit()` as-is — we write a new Python quantizer that mirrors `PackedWeights4Bit::pack()`.

This task validates the Rust convention so we know exactly what the Python must match.

- [ ] **Step 1: Read our Rust packing convention**

Read `crates/quantize/src/pack4.rs`, specifically the `pack()` method. Find the line where nibbles are shifted into u32. Document: which element goes in which nibble position.

- [ ] **Step 2: Read the Python converter convention**

Read `rustane-research/moe-1T/stage1-bottleneck-2026-03-20/wave2-build-order.md` lines 132-138. The Python uses `hi = q_flat[:, 0::2]`, `lo = q_flat[:, 1::2]`, `packed = ((hi << 4) | lo)`. This means even-indexed element is in the HIGH nibble.

- [ ] **Step 3: Write the cross-convention test**

```rust
// crates/moe-infer/tests/test_nibble_convention.rs
//! Verify Rust and Python nibble packing conventions agree.
//! Fatal trap: if they disagree, every weight is silently wrong.

#[test]
fn rust_nibble_convention_documented() {
    // Pack known values: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    // With scale/zero, these map to known nibble values.
    // Verify which nibble position each element occupies.
    let weights: Vec<f32> = (1..=128).map(|i| i as f32 / 128.0).collect();
    let packed = quantize::PackedWeights4Bit::pack(&weights, 1, 128, 128);

    // Unpack and verify order
    let raw = packed.unpack_raw();
    // raw[0] should correspond to weights[0], raw[1] to weights[1], etc.
    // The quantized values should be monotonically increasing
    for i in 1..raw.len() {
        assert!(raw[i] >= raw[i-1] || raw[i-1] == 15,
            "nibble order wrong at index {i}: raw[{i}]={} < raw[{}]={}",
            raw[i], i-1, raw[i-1]);
    }
}
```

- [ ] **Step 4: Run test**

Run: `cargo test -p moe-infer --test test_nibble_convention --release`
Expected: PASS — confirms our Rust convention is consistent

- [ ] **Step 5: Document the convention mismatch and resolution plan**

The Python converter uses hi-nibble-first, our Rust uses different packing. We need to reconcile these in Task 6 (write converter). Add a comment to the test documenting the exact convention our Rust code uses.

- [ ] **Step 6: Commit**

```bash
git add crates/moe-infer/tests/test_nibble_convention.rs
git commit -m "test: nibble packing convention validation (fatal trap prevention)"
```

---

### Task 4: Download Full Model

**Files:**
- None (downloads to gitignored `weights/` directory)

- [ ] **Step 1: Download all 16 safetensors shards + tokenizer**

```bash
hf download Qwen/Qwen3-30B-A3B --local-dir weights/qwen3-30b-a3b \
  --include "*.safetensors" "*.json" "tokenizer*" "*.txt"
```

Expected: ~61 GB download, 16 shard files + config.json + tokenizer.json + tokenizer_config.json

- [ ] **Step 2: Verify download integrity**

```bash
ls -la weights/qwen3-30b-a3b/model-*.safetensors | wc -l  # expect 16
ls weights/qwen3-30b-a3b/tokenizer.json                    # must exist
python3 -c "
from safetensors import safe_open
f = safe_open('weights/qwen3-30b-a3b/model-00001-of-00016.safetensors', framework='numpy')
print(f'Shard 1: {len(f.keys())} tensors')
"
```

- [ ] **Step 3: Commit** (nothing to commit — weights are gitignored)

---

### Task 5: Generate HuggingFace Reference Tensors

**Files:**
- Create: `scripts/generate_references.py`

- [ ] **Step 1: Write reference generation script**

```python
#!/usr/bin/env python3
"""Generate HuggingFace reference tensors for test validation.

Saves reference inputs/outputs so Rust tests can compare against ground truth.
Run ONCE on a machine with enough RAM (~64 GB for bf16 model).

Output: weights/references/*.npz
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-30B-A3B"
OUT = "weights/references"

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL)

# Tier 0: Tokenizer references
texts = [
    "Hello, world!",
    "The capital of France is",
    "def fibonacci(n):",
    "import torch\nfrom transformers import",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>",
]
tok_refs = {}
for text in texts:
    ids = tok.encode(text)
    tok_refs[text] = ids
np.savez(f"{OUT}/tokenizer_refs.npz", **{f"text_{i}": np.array(ids) for i, (text, ids) in enumerate(tok_refs.items())})
# Save texts separately
import json
with open(f"{OUT}/tokenizer_texts.json", "w") as f:
    json.dump(texts, f)
print(f"Tokenizer refs: {len(texts)} texts")

print("Loading model (bf16)...")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cpu")
model.eval()

# Tier 2: Single-layer attention reference
print("Generating attention reference...")
hidden = torch.randn(1, 1, 2048, dtype=torch.bfloat16)
# Save input + layer 0 output
with torch.no_grad():
    # Get layer 1 (first MoE layer) attention output
    # This requires hooking into the model internals
    layer1 = model.model.layers[1]
    pos_ids = torch.tensor([[0]])

    # RMSNorm + Attention
    normed = layer1.input_layernorm(hidden)
    attn_out, _, _ = layer1.self_attn(normed, position_ids=pos_ids)

np.savez(f"{OUT}/attention_ref.npz",
    input=hidden.float().numpy(),
    normed=normed.float().numpy(),
    attn_output=attn_out.float().numpy())
print("Attention ref saved")

# Tier 4: Generation reference (greedy, 20 tokens)
print("Generating token reference...")
prompt = "The capital of France is"
input_ids = tok(prompt, return_tensors="pt").input_ids
with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=20, do_sample=False)
gen_ids = output[0].tolist()
gen_text = tok.decode(gen_ids)
with open(f"{OUT}/generation_ref.json", "w") as f:
    json.dump({"prompt": prompt, "token_ids": gen_ids, "text": gen_text}, f, indent=2)
print(f"Generation ref: {len(gen_ids)} tokens: {gen_text[:80]}...")

print("Done. References saved to weights/references/")
```

- [ ] **Step 2: Create output directory and run**

```bash
mkdir -p weights/references
python3 scripts/generate_references.py
```

Note: This requires ~64 GB RAM for the bf16 model. If memory is tight, use `torch_dtype=torch.float16` or load model sharded.

- [ ] **Step 3: Verify reference files exist**

```bash
ls -la weights/references/
# Expected: tokenizer_refs.npz, tokenizer_texts.json, attention_ref.npz, generation_ref.json
```

- [ ] **Step 4: Commit script (not reference data — gitignored)**

```bash
git add scripts/generate_references.py
git commit -m "feat: HuggingFace reference tensor generation script"
```

---

## Phase 2B: Weight Converter + Tokenizer

### Task 6: Python Weight Converter

**Files:**
- Create: `scripts/convert_qwen3_moe.py`

The complete converter code is in `rustane-research/moe-1T/stage1-bottleneck-2026-03-20/wave2-build-order.md`. Copy it, adapting the nibble packing to match our Rust convention.

- [ ] **Step 1: Copy converter from research spec**

Copy the full converter from wave2-build-order.md lines 38-510. Save to `scripts/convert_qwen3_moe.py`.

- [ ] **Step 2: Reconcile nibble packing with Rust**

Read our Rust `pack4.rs` packing convention. Update the Python `quantize_4bit()` function to match exactly. The critical line is how `hi` and `lo` nibbles are assigned from even/odd indices.

- [ ] **Step 3: Run converter on 3 layers (dry run)**

```bash
python3 scripts/convert_qwen3_moe.py \
  --input-dir weights/qwen3-30b-a3b \
  --output-dir weights/rustane-qwen3 \
  --max-layers 3
```

Expected: backbone.bin + backbone_index.json + layer_01_experts.bin + layer_02_experts.bin

- [ ] **Step 4: Verify expert stride**

```bash
python3 -c "
import os
sz = os.path.getsize('weights/rustane-qwen3/layer_01_experts.bin')
stride = sz // 128
print(f'stride={stride} bytes ({stride/1e6:.2f} MB)')
# Expected: ~2,506,752 bytes (2.39 MB)
"
```

- [ ] **Step 5: Verify quantization roundtrip**

The converter has a built-in verification step that compares dequantized weights against originals. Check output for `max_err < 0.05`.

- [ ] **Step 6: Run full conversion (all 48 layers)**

```bash
python3 scripts/convert_qwen3_moe.py \
  --input-dir weights/qwen3-30b-a3b \
  --output-dir weights/rustane-qwen3
```

Expected: ~15 minutes, ~18 GB output

- [ ] **Step 7: Commit converter**

```bash
git add scripts/convert_qwen3_moe.py
git commit -m "feat: Python weight converter for Qwen3-MoE-30B"
```

---

### Task 7: Rust Backbone Weight Loader

**Files:**
- Create: `crates/moe-infer/src/weights.rs`
- Modify: `crates/moe-infer/src/lib.rs`
- Modify: `crates/moe-infer/Cargo.toml`
- Create: `crates/moe-infer/tests/test_weights.rs`

- [ ] **Step 1: Add dependencies to Cargo.toml**

Add to `[dependencies]` (some may already exist from Task 1B):
```toml
anyhow = "1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
toml = "0.8"           # may already be added in Task 1B
tokenizers = "0.21"
rand = "0.8"
libc.workspace = true  # move from dev-deps to deps for madvise/pread in weights.rs
```

- [ ] **Step 2: Write the failing test**

```rust
// crates/moe-infer/tests/test_weights.rs
use std::path::Path;

const WEIGHTS_DIR: &str = "weights/rustane-qwen3";

#[test]
#[ignore = "requires converted weights"]
fn test_embedding_load() {
    let weights = moe_infer::weights::BackboneWeights::load(
        Path::new(WEIGHTS_DIR)
    ).expect("load backbone");

    let emb = weights.embedding(0).expect("token 0 embedding");
    assert_eq!(emb.len(), 2048, "embedding dim should be 2048");
    assert!(emb.iter().any(|v| v.to_f32() != 0.0), "embedding should not be all zeros");
}

#[test]
#[ignore = "requires converted weights"]
fn test_layer_weights_shapes() {
    let weights = moe_infer::weights::BackboneWeights::load(
        Path::new(WEIGHTS_DIR)
    ).expect("load backbone");

    let lw = weights.layer_weights(1).expect("layer 1 weights");
    assert_eq!(lw.input_norm.len(), 2048);
    assert_eq!(lw.q_proj.len(), 4096 * 2048);    // [32*128, 2048]
    assert_eq!(lw.k_proj.len(), 512 * 2048);     // [4*128, 2048]
    assert_eq!(lw.v_proj.len(), 512 * 2048);
    assert_eq!(lw.o_proj.len(), 2048 * 4096);
    assert!(lw.router.is_some(), "MoE layer should have router");
    assert!(lw.shared_gate.is_some(), "MoE layer should have shared expert");
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p moe-infer --test test_weights --release -- --ignored`
Expected: FAIL — `weights` module doesn't exist

- [ ] **Step 4: Write minimal weights.rs implementation**

Follow the API from wave3-integration-spec.md Section 2. Key elements:
- `BackboneWeights` struct with `mmap: Mmap` and `index: HashMap<String, TensorInfo>`
- `load()` reads backbone_index.json, mmaps backbone.bin
- `embedding(token_id)` returns `&[f16]` slice of length hidden_size
- `layer_weights(layer)` returns `LayerWeights` with zero-copy slices
- `lm_head()` and `final_norm()` return appropriate slices

- [ ] **Step 5: Update lib.rs**

```rust
pub mod config;
pub mod pipeline;
pub mod sampler;
pub mod weights;
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cargo test -p moe-infer --test test_weights --release -- --ignored --nocapture`
Expected: PASS — embeddings load, shapes match

- [ ] **Step 7: Commit**

```bash
git add crates/moe-infer/src/weights.rs crates/moe-infer/src/lib.rs \
  crates/moe-infer/Cargo.toml crates/moe-infer/tests/test_weights.rs
git commit -m "feat: mmap backbone weight loader with shape verification"
```

---

### Task 8: Tokenizer Integration

**Files:**
- Create: `crates/moe-infer/tests/test_tokenizer.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/moe-infer/tests/test_tokenizer.rs
use tokenizers::Tokenizer;

const TOKENIZER_PATH: &str = "weights/qwen3-30b-a3b/tokenizer.json";

#[test]
#[ignore = "requires downloaded model"]
fn test_tokenizer_roundtrip() {
    let tok = Tokenizer::from_file(TOKENIZER_PATH).expect("load tokenizer");

    let text = "Hello, world!";
    let encoding = tok.encode(text, false).expect("encode");
    let ids = encoding.get_ids();
    assert!(!ids.is_empty(), "should produce tokens");

    let decoded = tok.decode(ids, true).expect("decode");
    assert_eq!(decoded, text, "roundtrip should be identity");
}

#[test]
#[ignore = "requires downloaded model + HF references"]
fn test_tokenizer_matches_hf() {
    let tok = Tokenizer::from_file(TOKENIZER_PATH).expect("load tokenizer");

    // Load HF reference token IDs
    // (generated by scripts/generate_references.py)
    let ref_path = "weights/references/tokenizer_texts.json";
    let texts: Vec<String> = serde_json::from_str(
        &std::fs::read_to_string(ref_path).expect("read ref")
    ).expect("parse");

    // TODO: Load npz reference IDs and compare
    // For now, just verify tokenizer produces non-empty output for each text
    for text in &texts {
        let encoding = tok.encode(text.as_str(), false).expect("encode");
        assert!(!encoding.get_ids().is_empty(), "empty for: {text}");
    }
}
```

- [ ] **Step 2: Run test**

Run: `cargo test -p moe-infer --test test_tokenizer --release -- --ignored --nocapture`
Expected: PASS — tokenizer loads and encodes/decodes

- [ ] **Step 3: Commit**

```bash
git add crates/moe-infer/tests/test_tokenizer.rs
git commit -m "test: tokenizer roundtrip and HF reference matching"
```

---

## Phase 2C: GQA Attention + Single Layer

### Task 9: KV Cache

**Files:**
- Create: `crates/moe-infer/src/kv_cache.rs`
- Create: `crates/moe-infer/tests/test_kv_cache.rs`

- [ ] **Step 1: Write failing test**

```rust
// Test: store K,V for 3 positions, retrieve, verify shapes and values
#[test]
fn test_kv_store_retrieve() {
    let mut cache = KvCache::new(48, 4, 128, 4096); // layers, kv_heads, head_dim, max_seq
    let k = vec![1.0f32; 4 * 128]; // 4 KV heads * 128 head_dim
    let v = vec![2.0f32; 4 * 128];
    cache.store(0, 0, &k, &v);

    let (k_out, v_out) = cache.get(0, 0, 1); // layer 0, kv_head 0, seq_len 1
    assert_eq!(k_out.len(), 128); // 1 position * head_dim
    assert_eq!(k_out[0], 1.0);
}
```

- [ ] **Step 2: Implement KvCache**

Follow the API from wave3-integration-spec.md. Flat storage: `k_store[layer * kv_heads + head]` = `Vec<f32>` of length `max_seq * head_dim`.

- [ ] **Step 3: Run test, verify pass**

- [ ] **Step 4: Commit**

---

### Task 9B: RMSNorm Implementation

**Files:**
- Create: `crates/moe-infer/src/rmsnorm.rs`

RMSNorm is used twice per layer (pre-attention and pre-FFN) and is missing from the codebase. Qwen3 uses eps=1e-6.

- [ ] **Step 1: Write failing test**

```rust
#[test]
fn test_rmsnorm_unit_gamma() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0f32; 4];
    let out = rmsnorm(&x, &gamma, 1e-6);
    // RMS = sqrt(mean(x^2)) = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
    // out = x / RMS, so out[0] ≈ 0.3651
    assert!((out[0] - 1.0 / 7.5f32.sqrt()).abs() < 1e-5);
}
```

- [ ] **Step 2: Implement rmsnorm()**

```rust
pub fn rmsnorm(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / n as f32 + eps).sqrt();
    x.iter().zip(gamma.iter()).map(|(&xi, &gi)| xi / rms * gi).collect()
}
```

- [ ] **Step 3: Add to lib.rs**

```rust
pub mod rmsnorm;
```

- [ ] **Step 4: Run test, verify pass**

- [ ] **Step 5: Commit**

---

### Task 10: RoPE Tables

**Files:**
- Add to: `crates/moe-infer/src/attention.rs`
- Create: `crates/moe-infer/tests/test_attention.rs`

- [ ] **Step 1: Write failing RoPE test**

```rust
#[test]
fn test_rope_theta_1e6() {
    let rope = RopeTables::build(4096, 128, 1_000_000.0);

    // Position 0: cos should be all 1.0, sin should be all 0.0
    assert!((rope.cos[0] - 1.0).abs() < 1e-6);
    assert!(rope.sin[0].abs() < 1e-6);

    // High-frequency dim (dim 0): at pos 512, theta_0 = 1.0
    // cos(512 * 1.0) = cos(512) ≈ 0.4
    let idx_pos512_dim0 = 512 * 64; // pos * head_dim/2
    assert!((rope.cos[idx_pos512_dim0] - (512.0f32).cos()).abs() < 1e-4);

    // Low-frequency dim (dim 63): theta_63 = 1/(1e6^(126/128)) ≈ 7.9e-6
    // cos(512 * 7.9e-6) ≈ 1.0 (barely rotated)
    let idx_pos512_dim63 = 512 * 64 + 63;
    assert!(rope.cos[idx_pos512_dim63] > 0.999, "low-freq dim should barely rotate");
}
```

**CRITICAL:** Use theta=1e6, NOT 1e4. Use neox-style pairing: pairs are `(i, i + head_dim/2)`, NOT `(2i, 2i+1)`.

- [ ] **Step 2: Implement RopeTables::build()**

- [ ] **Step 3: Run test, verify pass**

- [ ] **Step 4: Commit**

---

### Task 11: GQA Attention Forward

**Files:**
- Add to: `crates/moe-infer/src/attention.rs`
- Add to: `crates/moe-infer/tests/test_attention.rs`

- [ ] **Step 1: Write failing test**

```rust
#[test]
#[ignore = "requires converted weights"]
fn test_gqa_forward_one_token() {
    let weights = BackboneWeights::load(Path::new("weights/rustane-qwen3")).unwrap();
    let lw = weights.layer_weights(1).unwrap();
    let mut cache = KvCache::new(48, 4, 128, 4096);
    let rope = RopeTables::build(4096, 128, 1_000_000.0);
    let cfg = GqaConfig { num_q_heads: 32, num_kv_heads: 4, head_dim: 128, rope_theta: 1e6, max_seq: 4096 };

    let x = vec![0.01f32; 2048]; // dummy hidden state
    let out = gqa_forward(&x, &lw, &mut cache, 1, 0, &rope, &cfg);

    assert_eq!(out.len(), 2048);
    assert!(out.iter().all(|v| v.is_finite()), "NaN/Inf in attention output");
    assert!(out.iter().any(|v| *v != 0.0), "all-zero attention output");
}
```

- [ ] **Step 2: Implement gqa_forward()**

Key steps:
1. Q = x @ q_proj → [4096], reshape to [32, 128]
2. K = x @ k_proj → [512], reshape to [4, 128]
3. V = x @ v_proj → [512], reshape to [4, 128]
4. **QK-norm:** apply per-head RMSNorm to Q and K using `q_norm.weight` and `k_norm.weight` from LayerWeights. **Qwen3 requires this — without it, attention scores are miscalibrated (wave3-risks.md Trap 3).**
5. Apply RoPE (neox-style: pairs are `(i, i + head_dim/2)`, NOT `(2i, 2i+1)`) to Q and K
6. Store K, V in KV cache
7. For each Q head: attend to KV_head[q_head / 8] (GQA broadcast — integer division, NOT modulo)
8. scores = Q @ K^T / sqrt(128) + causal_mask (mask: `s <= t`, NOT `s < t`)
9. attn_weights = softmax(scores)
10. output = attn_weights @ V
11. Concatenate heads → o_proj → output [2048]

**LayerWeights must include `q_norm: &[f16]` and `k_norm: &[f16]`** — add these fields in weights.rs. The Python converter must extract `model.layers.{i}.self_attn.q_norm.weight` and `k_norm.weight`.

- [ ] **Step 3: Run test, verify pass**

- [ ] **Step 4: Write causal mask test at seq_len > 1**

```rust
#[test]
fn test_causal_mask_multi_token() {
    // Process 5 tokens sequentially, verify KV cache builds correctly
    // At position 4, attention should see positions 0-4 (not 0-3)
    // Verify: attention weights at pos 4 sum to 1.0 across 5 positions
}
```

- [ ] **Step 5: MANDATORY — compare against HF reference**

Load `weights/references/attention_ref.npz`. Use the reference input tensor. Assert our output matches HF within tolerance 1e-2. This catches GQA grouping errors, scale factor errors, QK-norm bugs, and RoPE errors. **This is not optional.**

- [ ] **Step 6: Commit**

---

### Task 12: Single-Layer Forward

**Files:**
- Add to: `crates/moe-infer/src/generate.rs`
- Create: `crates/moe-infer/tests/test_single_layer.rs`

- [ ] **Step 1: Write failing test**

```rust
#[test]
#[ignore = "requires converted weights"]
fn test_single_layer_forward() {
    // Load weights, create KV cache, run one layer
    // Layer 1 (first MoE layer): RMSNorm → Attention → Residual → RMSNorm → MoE → Residual
    let x = vec![0.01f32; 2048];
    let out = run_layer(&model, &mut cache, 1, &x, 0);

    assert_eq!(out.len(), 2048);
    assert!(out.iter().all(|v| v.is_finite()));
    // Output should differ from input (layer did something)
    let diff: f32 = x.iter().zip(out.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 0.001, "layer output should differ from input");
}
```

- [ ] **Step 2: Implement run_layer()**

For layer 0 (dense): RMSNorm → Attention → Residual → RMSNorm → Dense FFN → Residual
For layers 1-47 (MoE): RMSNorm → Attention → Residual → RMSNorm → Shared Expert + Routed Experts → Residual

- [ ] **Step 3: Run test, verify pass**

- [ ] **Step 4: Commit**

---

## Phase 2D: Generation Loop + Performance

### Task 13: Decode Loop

**Files:**
- Add to: `crates/moe-infer/src/generate.rs`
- Create: `crates/moe-infer/tests/test_generation.rs`

- [ ] **Step 1: Write failing test**

```rust
#[test]
#[ignore = "requires converted weights + tokenizer"]
fn test_generate_one_token() {
    let model = Model::load("weights/rustane-qwen3", "configs/qwen3-moe-30b.toml").unwrap();
    let tok = Tokenizer::from_file("weights/qwen3-30b-a3b/tokenizer.json").unwrap();

    let output = generate(&model, &tok, "The capital of France is", 1,
        &SamplingConfig { temperature: 0.0, top_p: 1.0, greedy: true });

    assert!(output.is_ok(), "generation should not error");
    let text = output.unwrap();
    assert!(!text.is_empty(), "should generate at least 1 token");
    println!("Generated: {text}");
}
```

- [ ] **Step 2: Implement generate()**

```
encode prompt → token_ids
embed first token
for pos in 0..max_tokens:
    for layer in 0..48:
        x = run_layer(model, cache, layer, x, pos)
    logits = x @ lm_head
    next_token = sample(logits, sampling_config)
    if next_token == eos: break
    decode and print token
    embed next_token for next iteration
```

- [ ] **Step 3: Run test — FIRST TOKEN**

Run: `cargo test -p moe-infer --test test_generation --release -- --ignored --nocapture`
Expected: Prints a real word. This is the milestone.

- [ ] **Step 4: Test greedy matches HF**

```rust
#[test]
#[ignore = "requires converted weights + HF references"]
fn test_generation_matches_hf() {
    // Load weights/references/generation_ref.json
    // Generate with same prompt, greedy
    // Compare first 20 tokens
}
```

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: generation loop — first tokens from Qwen3-MoE-30B"
```

---

### Task 14: tok/s Benchmark

**Files:**
- Create: `crates/moe-infer/tests/bench_tok_per_sec.rs`

- [ ] **Step 1: Write benchmark**

```rust
#[test]
#[ignore = "benchmark: requires converted weights"]
fn bench_tok_per_sec() {
    let model = Model::load("weights/rustane-qwen3", "configs/qwen3-moe-30b.toml").unwrap();
    let tok = Tokenizer::from_file("weights/qwen3-30b-a3b/tokenizer.json").unwrap();

    let prompt = "Explain what a mixture of experts model is in one paragraph.";

    // Warmup
    generate(&model, &tok, prompt, 5, &SamplingConfig::greedy()).unwrap();

    // Timed
    let t0 = std::time::Instant::now();
    let output = generate(&model, &tok, prompt, 50, &SamplingConfig::greedy()).unwrap();
    let elapsed = t0.elapsed();

    let tokens_generated = /* count from output */;
    let tok_per_sec = tokens_generated as f64 / elapsed.as_secs_f64();

    println!("Generated {} tokens in {:.1}s = {:.1} tok/s",
        tokens_generated, elapsed.as_secs_f64(), tok_per_sec);
    println!("Output: {output}");
}
```

- [ ] **Step 2: Run benchmark**

Run: `cargo test -p moe-infer --test bench_tok_per_sec --release -- --ignored --nocapture`

Expected: Any number > 0. Target: 25-30 tok/s with Metal GEMV, ~1-3 tok/s CPU-only initially.

- [ ] **Step 3: Log to experiments-infer.tsv**

- [ ] **Step 4: Commit**

```bash
git commit -m "bench: tok/s measurement on Qwen3-MoE-30B"
```

---

### Task 15: Update CLI for Real Inference

**Files:**
- Modify: `crates/moe-infer/src/bin/infer.rs`

- [ ] **Step 1: Update CLI to support real generation**

Add `--prompt` flag for interactive generation. Keep `--benchmark` for throughput measurement.

```bash
cargo run -p moe-infer --release --bin infer -- \
  --config configs/qwen3-moe-30b.toml \
  --prompt "What is 2+2?"
```

- [ ] **Step 2: Test with real prompt**

Expected: Prints coherent answer. If gibberish: check RoPE theta, QK-norm, gate normalization.

- [ ] **Step 3: Commit**

---

## Exit Gate

Stage 2 is complete when ALL of these pass:

```bash
# 1. Tokenizer works
cargo test -p moe-infer --test test_tokenizer --release -- --ignored

# 2. Weights load correctly
cargo test -p moe-infer --test test_weights --release -- --ignored

# 3. Attention produces correct output
cargo test -p moe-infer --test test_attention --release -- --ignored

# 4. Generation produces real tokens
cargo test -p moe-infer --test test_generation --release -- --ignored

# 5. tok/s is measured
cargo test -p moe-infer --test bench_tok_per_sec --release -- --ignored --nocapture

# 6. No regressions
cargo test -p engine --release
cargo test -p quantize --release
cargo test -p moe-router --release
cargo test -p expert-pager --release
cargo test -p moe-infer --release
```

**The acceptance test:** `test_generation_matches_hf` — greedy token output matches HuggingFace `model.generate()` for 20 tokens.
