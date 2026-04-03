# rustane

Rust training + inference engine for Apple Neural Engine (ANE) + Metal GPU.

Training pipeline validated from 48M to 5B parameters. Forward pass confirmed to 30B. All on M4 Max 128GB using reverse-engineered private ANE APIs.

## What This Does

Rustane is the first training-capable, memory-safe Rust engine for the Apple Neural Engine. It uses reverse-engineered private APIs (`_ANEClient`, `_ANEInMemoryModel`) to compile and evaluate MIL kernels directly on ANE hardware — no CoreML, no black-box scheduler.

The engine trains transformer models at 3-5W power draw, leaving the GPU completely free. Trained weights export via SafeTensors for inference anywhere.

## Benchmarking

Run `make help` to see all available commands.

## Checkpoint Inference

Rustane can now load its own training checkpoints and run a minimal autoregressive
generation loop from the `engine` crate.

```bash
# raw token IDs
cargo run -p engine --bin generate --release -- \
  --checkpoint /path/to/ckpt_001000.bin \
  --prompt-ids 1,2,3,4 \
  --steps 32

# text prompt with a compatible tokenizer.json
cargo run -p engine --bin generate --release -- \
  --checkpoint /path/to/ckpt_001000.bin \
  --tokenizer /path/to/tokenizer.json \
  --prompt "Hello from Rustane" \
  --steps 32

# force the Metal-backed KV-cache decode path
cargo run -p engine --bin generate --release -- \
  --checkpoint /path/to/ckpt_001000.bin \
  --prompt-ids 1,2,3,4 \
  --steps 32 \
  --kv-cache

# explicitly choose the decode backend
cargo run -p engine --bin generate --release -- \
  --checkpoint /path/to/ckpt_001000.bin \
  --prompt-ids 1,2,3,4 \
  --steps 32 \
  --decode-backend auto

# sample multiple continuations from one shared prompt prefill
cargo run -p engine --bin generate --release -- \
  --checkpoint /path/to/ckpt_001000.bin \
  --prompt "Hello from Rustane" \
  --tokenizer /path/to/tokenizer.json \
  --decode-backend metal \
  --temperature 0.8 \
  --top-k 8 \
  --samples 4 \
  --steps 64

# keep one prompt/session hot and serve repeated JSONL requests over stdin/stdout
printf '%s\n%s\n' \
  '{"steps":32,"temperature":0.8,"top_k":8,"samples":2,"seed":11}' \
  '{"steps":16,"decode_backend":"naive","temperature":0.0,"top_k":1,"samples":1,"seed":12}' \
  | cargo run -p engine --bin generate --release -- \
      --checkpoint /path/to/ckpt_001000.bin \
      --prompt-ids 1,2,3,4 \
      --decode-backend metal \
      --jsonl-session
```

The CLI reports separate `prefill` and `decode` timings. Current decode mode is
selected automatically. `auto` prefers `naive_full_context` for small checkpoints
and `kv_cache_metal` for larger ones when the requested decode window stays within
`cfg.seq`. `--kv-cache` is a shorthand for `--decode-backend metal`. When sampling
is greedy (`temperature <= 0` or `top_k == 1`), the Metal KV-cache path also uses
an on-GPU embedding lookup plus on-GPU argmax step, so it does not need to upload
a full token embedding or read back a full logits vector each token. For bounded
sampling (`top_k > 1`), Rustane also uses a GPU top-k candidate path for cached
decode when `top_k <= 32`, and only reads back those candidates for CPU sampling.
For full-vocab sampling (`top_k == 0`), cached decode uses an exact GPU block-stats
path and only materializes one sampled vocab block instead of the whole logits row.
`--samples N` reuses one prompt prefill across multiple continuations in a single
run, and reports amortized decode / overall throughput across all samples. For a
persistent process, `--jsonl-session` keeps one `InferenceSession` alive and reads
one JSON request per stdin line. Startup metadata is written to stderr; stdout is
reserved for JSONL responses so the process can be scripted cleanly.

Tokenizer note: token IDs produced by the tokenizer must fit within the model
checkpoint vocabulary. Rustane checkpoints currently use the training vocab baked
into the checkpoint config.

## HTTP Serving

Rustane also has a minimal HTTP server for checkpoint-backed inference:

```bash
cargo run -p engine --bin serve --release -- \
  --checkpoint /path/to/ckpt_001000.bin \
  --tokenizer /path/to/tokenizer.json \
  --bind 127.0.0.1:8080 \
  --model-name rustane-local \
  --max-cache-sessions 32
```

Available routes:
- `GET /healthz`
- `GET /v1/models`
- `POST /v1/completions`
- `POST /v1/chat/completions`

Example completion request:

```bash
curl http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "rustane-local",
    "prompt_ids": [1, 2, 3, 4],
    "max_tokens": 16,
    "temperature": 0.8,
    "top_k": 8,
    "n": 2
  }'
```

Example chat request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "rustane-local",
    "messages": [
      {"role": "system", "content": "Be terse."},
      {"role": "user", "content": "Hello"}
    ],
    "max_tokens": 16,
    "temperature": 0.0,
    "top_k": 1
  }'
```

Streaming is also supported with Server-Sent Events for `n=1`:

```bash
curl -N http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "rustane-local",
    "prompt_ids": [1, 2, 3, 4],
    "max_tokens": 8,
    "temperature": 0.0,
    "top_k": 1,
    "stream": true
  }'
```

Responses include a `rustane` metadata block with:
- `effective_prompt_tokens`
- `prompt_truncated`
- `cache_prefix_tokens`
- `base_prefill_ms`
- `request_prefill_ms`
- `decode_ms_total`

The server keeps an in-memory prompt-session cache and reuses the deepest cached
prefix for subsequent requests. This is intentionally minimal: non-streaming only,
single-process, and best suited for local experimentation.

### Three Types of Tests

| Test | What It Measures | Command | Time |
|------|-----------------|---------|------|
| **Training validation** | Full pipeline: compile + forward + backward + Adam + loss decrease | `make sweep-600m` | ~17s |
| **Architecture sweep** | 5 variants per scale, finds optimal depth/width | `make sweep-full` | ~60 min |
| **Forward-only probe** | Forward pass only (no backward/optimizer), tests scale ceiling | `make forward-ladder` | ~8 min |

### Quick Start Benchmarking

```bash
# Build and run unit tests first
make build
make test

# Validate training pipeline at 600M (any Mac with 18GB+)
make sweep-600m

# Validate at each scale individually
make sweep-1b        # needs ~25GB
make sweep-3b        # needs ~55GB
make sweep-5b        # needs ~85GB

# Full sweep: 25 configs across 600M-5B (needs 85GB, ~60 min)
make sweep-full

# Forward-only: how far can your hardware go?
make forward-7b      # needs ~31GB
make forward-10b     # needs ~46GB
make forward-ladder  # 5B to 20B, needs ~93GB
make forward-ceiling # 25B to 30B, needs ~130GB
```

### What Results Look Like

Training validation output:
```
============================================================
  600m-A — 1536d/4096h/20L/seq512 — ~579M params — h/d=2.67x
============================================================
  [1/4] Compiling ANE kernels... 0.5s
  [2/4] Forward pass... loss=9.0108
  [3/4] Training 10 steps... 9.0108 → 8.9857 (delta=-0.0251)
  [4/4] Timing 5 steps... 958ms/step (fwd=241 bwd=645 upd=55) = 535 tok/s
```

Forward-only output:
```
======================================================================
  7B — 4096d/11008h/32L/seq512 — 6.51B params — est. 31.1GB
======================================================================
  [1/3] Compiling ANE kernels... 1.5s
  [2/3] Allocating 31.1GB... 5.3s
  [3/3] Forward pass (1 warmup + 3 timed)... 3132ms (loss=9.3228)
```

### Sharing Your Results

Run a test, copy the output, open an issue with your hardware info. We collect results across chips — every data point matters.

Include: chip model, RAM, macOS version, full test output.

## Results

### Training Pipeline Validation (M4 Max 128GB)

Best config at each scale from 25 architecture variants (5 per scale). Each config validates: ANE kernel compilation, forward pass, backward pass, Adam weight update, loss decrease.

| Scale | Params | Best Config | ms/step | tok/s | RAM |
|-------|--------|-------------|---------|-------|-----|
| 600M | 579M | wide+shallow (14L) | 865 | 592 | ~12GB |
| 1B | 1.3B | wide+shallow (20L) | 2,012 | 254 | ~25GB |
| 1.5B | 1.9B | wide+shallow (24L) | 2,775 | 184 | ~30GB |
| 3B | 3.2B | baseline (40L) | 4,639 | 110 | ~55GB |
| **5B** | **5.0B** | **deep+narrow (60L)** | **6,893** | **74** | **~85GB** |
| 7B | 6.5B | baseline (32L) | 94,000 | 5 | ~112GB (swap) |

### Forward-Only Scale Probes (M4 Max 128GB)

No backward pass or optimizer — tests ANE kernel compilation and forward pass at extreme scale.

| Scale | Forward Time | RAM |
|-------|-------------|-----|
| 5B | 2.1s | 25GB |
| 7B | 3.1s | 31GB |
| 10B | 4.7s | 46GB |
| 13B | 22s | 59GB |
| 15B | 27s | 70GB |
| 20B | 41s | 93GB |
| **30B** | **75s** | **130GB** |

No ANE compilation ceiling found. The limit is RAM, not the chip.

### Forward-Only: M5 Max (community)

Results from [Anemll](https://github.com/Anemll) testing on M5 Max 128GB (forward-only):

| Scale | M4 Max | M5 Max | Speedup |
|-------|--------|--------|---------|
| 5B | 2,064ms | 1,910ms | 8% |
| 7B | 3,132ms | 2,878ms | 8% |
| 10B | 4,696ms | 4,329ms | 8% |
| 13B | 21,962ms | 20,270ms | 8% |
| 15B | 26,740ms | 24,303ms | 9% |
| 20B | 40,933ms | 32,380ms | **21%** |

Steady 8% faster at 5B-15B, 21% at 20B. The dim=5120 efficiency cliff is present on both chips.

### Key Findings

- **Architecture crossover at 3B**: wide+shallow wins below (fewer ANE dispatches), deep+narrow wins above (smaller matmuls more efficient)
- **Efficiency cliff at dim=5120**: forward time jumps 4.7x per layer. Keep dim at or below 4096 for ANE.
- **Practical training ceiling**: ~5B on 128GB. An M3 Ultra with 512GB could reach ~20B.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                      engine crate                    │
│  Training loop | Forward/Backward | Adam optimizer   │
│  ANE kernels (10) | CPU ops (vDSP) | Metal Adam     │
└───┬─────────────────────────────────────────┬───────┘
    │                                         │
┌───▼──────────────┐           ┌──────────────▼───────┐
│   ane-bridge     │           │   metal-decode       │
│   ANE private    │           │   Metal compute      │
│   API FFI        │           │   shaders (planned)  │
│   IOSurface I/O  │           │   Single-token       │
│   MIL generation │           │   decode path        │
└──────────────────┘           └──────────────────────┘
```

### Crates

- **ane-bridge** — Safe Rust FFI to ANE private APIs via objc2/dlopen. MIL kernel generation, IOSurface weight packing, dynamic weight pipeline.
- **metal-decode** — Metal shaders for single-token decode (planned).
- **engine** — Training orchestrator: ANE forward (10 fused kernels), CPU backward (Accelerate sgemm), Metal/CPU Adam optimizer.

## Hardware Requirements

Any Apple Silicon Mac with 18GB+ RAM. The ANE is the same 16-core design across M1-M5. Only RAM differs.

Tested on M4 Max 128GB. Other configs are estimates based on RAM scaling.

| Hardware | Memory | Training Ceiling | Forward Ceiling |
|----------|--------|-----------------|-----------------|
| M1/M2/M3 Pro 18GB | 18 GB | ~300M | ~3B |
| M1/M2/M3 Pro 36GB | 36 GB | ~1B | ~7B |
| M1/M2/M3/M4/M5 Max 64GB | 64 GB | ~3B | ~15B |
| M3/M4/M5 Max 96GB | 96 GB | ~5B | ~20B |
| **M3/M4/M5 Max 128GB** | **128 GB** | **~5B** (tested) | **~30B** (tested) |
| M3 Ultra 192GB | 192 GB | ~10B | ~40B+ |
| M3 Ultra 512GB | 512 GB | ~20B | ~100B+ |

## ANE Gotchas

- IOSurface spatial width must be multiple of 16 (silent data corruption otherwise)
- ANE compiler fails on rsqrt/sqrt after reduce ops — use pow(-0.5)
- Per-ANE-dispatch overhead: ~0.095ms (XPC + IOKit round-trip)
- IOSurface stores fp32, ANE casts to fp16 internally
- dim must be divisible by 128 (heads = dim/128, hd=128)
- hidden must be divisible by 16

## Sister Projects

- [autoresearch-ANE](https://github.com/ncdrone/autoresearch-ANE) — ANE training in native Obj-C (private APIs). The research foundation.
- [autoresearch-mlx](https://github.com/ncdrone/autoresearch-mlx) — MLX training in Python. Architecture exploration (241 experiments).

## Credits

See [CREDITS.md](CREDITS.md) for the full list. Key acknowledgments:

- **maderix** — ANE private API reverse engineering, the foundational work everything builds on
- **ane crate** (computer-graphics-tools) — Rust FFI to ANE, our ane-bridge base
- **thebasedcapital/ane-infer** — First Rust + ANE + Metal prototype
- **karpathy/llm.c** — Training architecture reference, climbmix-400B dataset

## License

MIT
