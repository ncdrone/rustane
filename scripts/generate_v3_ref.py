#!/usr/bin/env python3
"""Generate DeepSeek-V3 reference outputs for L1/L2 validation.

Partial-load approach: only loads ~200 MB of weights (embedding + layer 0)
to generate golden reference intermediates without requiring full model in RAM.

Usage:
    python3 scripts/generate_v3_ref.py \
        --weights-dir weights/deepseek-v3 \
        --output-dir weights/references/
"""

import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np

try:
    import ml_dtypes
except ImportError:
    print("ERROR: pip install ml_dtypes safetensors")
    raise


def load_safetensors_index(weights_dir: Path) -> dict:
    """Load the safetensors index to find which shard contains each tensor."""
    index_path = weights_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)
    return {}


def load_tensor_from_shards(weights_dir: Path, name: str, index: dict) -> np.ndarray:
    """Load a single tensor from the correct shard."""
    from safetensors import safe_open

    # Find which shard has this tensor
    if "weight_map" in index:
        shard_name = index["weight_map"].get(name)
        if shard_name:
            shard_path = weights_dir / shard_name
            with safe_open(str(shard_path), framework="numpy") as f:
                return f.get_tensor(name)

    # Fallback: search all shards
    for shard in sorted(weights_dir.glob("*.safetensors")):
        try:
            with safe_open(str(shard), framework="numpy") as f:
                if name in f.keys():
                    return f.get_tensor(name)
        except Exception:
            continue

    raise ValueError(f"Tensor not found: {name}")


def dequant_fp8(tensor: np.ndarray, scale_inv: np.ndarray, block_size: int = 128) -> np.ndarray:
    """Dequantize FP8 e4m3fn tensor with block-wise scales."""
    # View raw bytes as fp8_e4m3fn
    if tensor.dtype == np.uint8:
        fp8_vals = tensor.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    elif tensor.dtype == ml_dtypes.float8_e4m3fn:
        fp8_vals = tensor.astype(np.float32)
    else:
        return tensor.astype(np.float32)

    m, k = fp8_vals.shape
    num_row_blocks = (m + block_size - 1) // block_size
    num_col_blocks = (k + block_size - 1) // block_size

    out = np.zeros((m, k), dtype=np.float32)
    for rb in range(num_row_blocks):
        r_start = rb * block_size
        r_end = min(r_start + block_size, m)
        for cb in range(num_col_blocks):
            c_start = cb * block_size
            c_end = min(c_start + block_size, k)
            scale = scale_inv[rb, cb]
            out[r_start:r_end, c_start:c_end] = fp8_vals[r_start:r_end, c_start:c_end] * scale

    return out


def load_fp8_tensor_raw(weights_dir: Path, name: str, index: dict) -> np.ndarray:
    """Load an FP8 tensor using raw byte parsing (avoids numpy FP8 dtype issue)."""
    import struct as st_mod

    # Find which shard
    shard_name = index.get("weight_map", {}).get(name)
    scale_name = f"{name}_scale_inv"
    scale_shard_name = index.get("weight_map", {}).get(scale_name, shard_name)

    shard_path = weights_dir / shard_name
    with open(shard_path, 'rb') as f:
        header_size = st_mod.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_size).decode('utf-8'))
        data_start = 8 + header_size

        # Load weight bytes
        info = header[name]
        f.seek(data_start + info['data_offsets'][0])
        size = info['data_offsets'][1] - info['data_offsets'][0]
        w_bytes = np.frombuffer(f.read(size), dtype=np.uint8).reshape(info['shape'])

        # Load scale (may be in same shard)
        if scale_name in header:
            s_info = header[scale_name]
            f.seek(data_start + s_info['data_offsets'][0])
            s_size = s_info['data_offsets'][1] - s_info['data_offsets'][0]
            scale = np.frombuffer(f.read(s_size), dtype=np.float32).reshape(s_info['shape'])
        else:
            # Scale in different shard
            s_shard_path = weights_dir / scale_shard_name
            with open(s_shard_path, 'rb') as sf:
                sh_size = st_mod.unpack('<Q', sf.read(8))[0]
                sh = json.loads(sf.read(sh_size).decode('utf-8'))
                sd_start = 8 + sh_size
                s_info = sh[scale_name]
                sf.seek(sd_start + s_info['data_offsets'][0])
                s_size = s_info['data_offsets'][1] - s_info['data_offsets'][0]
                scale = np.frombuffer(sf.read(s_size), dtype=np.float32).reshape(s_info['shape'])

    # Dequant via ml_dtypes + block-wise scale
    w_fp8 = w_bytes.view(ml_dtypes.float8_e4m3fn)
    w_f32 = w_fp8.astype(np.float32)

    # Apply block-wise scale (dequant_fp8 expects uint8 or float8 input,
    # but we already have f32 — apply scale directly)
    M, K = info['shape']
    block_size = 128
    out = np.zeros((M, K), dtype=np.float32)
    for rb in range((M + block_size - 1) // block_size):
        rs, re = rb * block_size, min((rb + 1) * block_size, M)
        for cb in range((K + block_size - 1) // block_size):
            cs, ce = cb * block_size, min((cb + 1) * block_size, K)
            out[rs:re, cs:ce] = w_f32[rs:re, cs:ce] * scale[rb, cb]
    return out


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMS Layer Normalization."""
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return (x / rms) * weight


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU activation."""
    return x / (1.0 + np.exp(-x))


def main():
    parser = argparse.ArgumentParser(description="Generate V3 reference outputs")
    parser.add_argument("--weights-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--token-id", type=int, default=9707, help="Token ID for reference (default: 'The')")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = load_safetensors_index(args.weights_dir)

    print("Loading embedding...")
    embed = load_tensor_from_shards(args.weights_dir, "model.embed_tokens.weight", index)
    embed_f32 = embed.astype(np.float32) if embed.dtype != np.float32 else embed
    print(f"  embed shape: {embed_f32.shape}, dtype: {embed.dtype}")

    # Token embedding
    token_id = args.token_id
    token_emb = embed_f32[token_id]
    print(f"  token {token_id} embedding: norm={np.linalg.norm(token_emb):.6f}")

    # Layer 0 norms
    print("Loading layer 0 weights...")
    input_norm = load_tensor_from_shards(args.weights_dir, "model.layers.0.input_layernorm.weight", index).astype(np.float32)
    post_attn_norm = load_tensor_from_shards(args.weights_dir, "model.layers.0.post_attention_layernorm.weight", index).astype(np.float32)

    # L1: Embedding + norm
    normed_emb = rmsnorm(token_emb, input_norm)
    print(f"  normed embedding: norm={np.linalg.norm(normed_emb):.6f}")

    # Load Q LoRA weights for layer 0 (raw byte parsing to avoid numpy FP8 dtype issue)
    q_a_f32 = load_fp8_tensor_raw(args.weights_dir, "model.layers.0.self_attn.q_a_proj.weight", index)
    q_a_norm = load_tensor_from_shards(args.weights_dir, "model.layers.0.self_attn.q_a_layernorm.weight", index).astype(np.float32)

    # Q LoRA step 1: x @ W_qa^T
    q_latent = normed_emb @ q_a_f32.T
    q_latent_normed = rmsnorm(q_latent, q_a_norm)
    print(f"  q_latent: shape={q_latent.shape}, norm={np.linalg.norm(q_latent):.6f}")
    print(f"  q_latent_normed: norm={np.linalg.norm(q_latent_normed):.6f}")

    # Save intermediates
    out_path = args.output_dir / "deepseek_v3_intermediates.npz"
    np.savez(
        str(out_path),
        token_id=np.array([token_id]),
        token_embedding=token_emb,
        normed_embedding=normed_emb,
        q_latent=q_latent,
        q_latent_normed=q_latent_normed,
        input_norm_weights=input_norm,
    )
    print(f"\nSaved: {out_path}")
    print(f"  Size: {os.path.getsize(out_path) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
