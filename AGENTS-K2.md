# AGENTS-K2.md — Kimi-K2 (1T) Inference Optimization

Optimize Kimi-K2 (1 trillion parameters) inference on Apple M4 Max 128GB.
Current: 1.3 tok/s. Goal: as fast as possible.

## Build & Test

```bash
cargo build -p moe-infer --release
cargo test -p moe-infer --release
cargo test -p moe-infer --test bench_k2_tok_per_sec --release -- --ignored --nocapture
```

Run inference:
```bash
cargo run -p moe-infer --release --bin infer -- \
  --config configs/kimi-k2.toml \
  --weights weights/rustane-k2 \
  --tokenizer weights/kimi-k2/tokenizer.json \
  --prompt "The capital of France is" --max-tokens 10
```

## Metric Matrix

### Tier 1 — Hard Gates (any failure = instant revert)
| Test | Command |
|------|---------|
| Build clean | `cargo build -p moe-infer --release` |
| Full test suite | `cargo test -p moe-infer --release` |
| Your custom test | `cargo test -p moe-infer --test auto_<name> --release` |

### Tier 2 — Performance (median of 3, must not regress >5%)
| Metric | Baseline | Command |
|--------|----------|---------|
| K2 warm decode tok/s | 1.3 | `bench_k2_tok_per_sec` |

## Locked Files
```
crates/moe-infer/tests/bench_k2_tok_per_sec.rs
configs/kimi-k2.toml
AGENTS-K2.md
```

## K2 Architecture

61 layers, 64 MLA attention heads, 384 MoE experts per layer (top-8 routed).
Backbone: 23.4 GB f16. Expert files: 9 GB each, 524 GB total. INT4 quantized.

Three compute units available:
- **CPU (AMX)**: 3 TFLOPS — currently handles MLA attention + shared FFN + conversion
- **Metal GPU**: 15 TFLOPS — currently handles expert INT4 dispatch only
- **ANE**: 17.8 TFLOPS — currently **UNUSED** (ane-bridge crate exists, was used for Qwen3 prefill)

Current decode per layer (~12.4 ms):
```
MoE FFN:     ~8 ms   (experts pread + Metal dispatch + shared FFN)
MLA attn:    ~0.4 ms (64 heads, Q LoRA, sgemm scores, O proj)
Other:       ~4 ms   (conversion, norms, Metal overhead, L2 misses)
```

## Memory Budget
```
macOS + Metal:           ~10 GB
Backbone mmap (f16):     ~23 GB
Expert staging:           ~2 GB
KV cache:                 ~1 GB
Available:               ~91 GB
```

## Key Source Files
```
crates/moe-infer/src/generate_v2.rs    ← decode loop
crates/moe-infer/src/mla_attention.rs  ← MLA forward (64 heads)
crates/moe-infer/src/blas.rs           ← BLAS FFI
crates/moe-infer/src/weights.rs        ← weight loading
crates/expert-pager/src/pool.rs        ← expert pool (built, NOT wired into decode)
crates/expert-pager/src/loader.rs      ← pread loader
crates/moe-router/src/lib.rs           ← sigmoid routing
crates/ane-bridge/src/                 ← ANE private API (UNUSED for decode)
configs/kimi-k2.toml                   ← K2 config
```

## Rules
- NEVER use EXHAUSTED as a verdict. There is ALWAYS something to try.
- You MUST implement and benchmark something every iteration.
- One variable per experiment. Write a custom test. Log to experiments-k2.tsv.
- Read system/k2-gossip.md for what previous iterations learned.
- Commit wins: `"perf: <what> — <before> → <after> tok/s (<X>%)"`
