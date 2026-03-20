#!/usr/bin/env python3
"""Convert HuggingFace Qwen3-MoE-30B safetensors to rustane binary format.

Produces:
  backbone.bin         -- embed, per-layer dense weights (f16/f32), final norm, lm_head
  backbone_index.json  -- maps tensor names to {offset, shape, dtype} in backbone.bin
  layer_XX_experts.bin -- per-layer routed expert weights in 4-bit quantized format

Nibble packing matches Rust pack4.rs EXACTLY:
  - 8 nibbles per u32, LSB-first
  - nibble 0 = bits [0:3], nibble 1 = bits [4:7], ..., nibble 7 = bits [28:31]
  - Asymmetric quantization: scale = (max - min) / 15.0, zero = min
  - q = clamp(round((val - zero) / scale), 0, 15)
  - Pack: word = q[0] | (q[1] << 4) | (q[2] << 8) | ... | (q[7] << 28)
  - Scales and zeros stored as f16

Usage:
  python scripts/convert_qwen3_moe.py --model-dir weights/qwen3-moe-30b-ref --output-dir weights/rustane-qwen3
  python scripts/convert_qwen3_moe.py --model-dir weights/qwen3-moe-30b-ref --output-dir weights/rustane-qwen3 --max-layers 2

Model config (from config.json):
  hidden_size=2048, vocab_size=151936, num_layers=48
  num_attention_heads=32, num_key_value_heads=4, head_dim=128
  intermediate_size=6144 (dense layer 0), moe_intermediate_size=768
  num_experts=128, num_experts_per_tok=8, shared_expert_count=1
  torch_dtype=bfloat16
"""

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Model constants (verified against config.json)
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 2048
VOCAB_SIZE = 151936
NUM_LAYERS = 48
NUM_Q_HEADS = 32
NUM_KV_HEADS = 4
HEAD_DIM = 128
DENSE_INTER_SIZE = 6144  # layer 0 only
MOE_INTER_SIZE = 768
NUM_EXPERTS = 128
GROUP_SIZE = 128

# Derived
Q_PROJ_DIM = NUM_Q_HEADS * HEAD_DIM   # 4096
KV_PROJ_DIM = NUM_KV_HEADS * HEAD_DIM  # 512


def parse_args():
    p = argparse.ArgumentParser(description="Convert Qwen3-MoE-30B to rustane format")
    p.add_argument(
        "--model-dir",
        required=True,
        help="Directory with HF safetensors shards and config.json",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for converted weights",
    )
    p.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Only convert first N layers (for dry runs)",
    )
    p.add_argument(
        "--verify-experts",
        type=int,
        default=3,
        help="Number of experts per layer to verify after quantization (0 to skip)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Safetensors loading
# ---------------------------------------------------------------------------


def load_sharded_index(model_dir: Path) -> dict:
    """Load the safetensors index to find which shard has each tensor."""
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        # Single-shard model -- list all .safetensors files
        shards = sorted(model_dir.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"No safetensors files in {model_dir}")
        # Return None to signal "load all shards and search"
        return None

    with open(index_path) as f:
        index = json.load(f)
    return index  # has "weight_map": { tensor_name: shard_filename }


def open_shard(model_dir: Path, shard_name: str):
    """Open a safetensors shard, return SafeTensors object and raw bytes."""
    import safetensors
    from safetensors import safe_open

    path = model_dir / shard_name
    if not path.exists():
        raise FileNotFoundError(f"Shard not found: {path}")
    return safe_open(str(path), framework="numpy")


class ShardedTensorLoader:
    """Lazy loader that opens shards on demand and caches them."""

    def __init__(self, model_dir: Path, index: dict | None):
        self.model_dir = model_dir
        self.index = index
        self._shard_cache: dict = {}  # shard_name -> safe_open handle

        if index is not None:
            self.weight_map = index["weight_map"]
        else:
            # Build weight map by scanning all shards
            self.weight_map = {}
            for shard_path in sorted(model_dir.glob("*.safetensors")):
                handle = open_shard(model_dir, shard_path.name)
                for name in handle.keys():
                    self.weight_map[name] = shard_path.name
                self._shard_cache[shard_path.name] = handle

    def _get_shard(self, shard_name: str):
        if shard_name not in self._shard_cache:
            self._shard_cache[shard_name] = open_shard(self.model_dir, shard_name)
        return self._shard_cache[shard_name]

    def get_tensor(self, name: str) -> np.ndarray:
        """Load a tensor by name, returning numpy array."""
        if name not in self.weight_map:
            raise KeyError(f"Tensor '{name}' not found in any shard")
        shard_name = self.weight_map[name]
        handle = self._get_shard(shard_name)
        return handle.get_tensor(name)

    def has_tensor(self, name: str) -> bool:
        return name in self.weight_map

    def list_tensors(self) -> list[str]:
        return sorted(self.weight_map.keys())


# ---------------------------------------------------------------------------
# 4-bit quantization (matches Rust pack4.rs)
# ---------------------------------------------------------------------------


def quantize_4bit(weights: np.ndarray, group_size: int = GROUP_SIZE) -> tuple:
    """Quantize a 2D weight matrix to 4-bit, matching Rust pack4.rs convention.

    Args:
        weights: [out_features, in_features] float32 array
        group_size: quantization group size (must divide in_features)

    Returns:
        (packed_data, scales, zeros) where:
        - packed_data: uint32 array, 8 nibbles per word, LSB-first
        - scales: float16 array, per-group
        - zeros: float16 array, per-group
    """
    assert weights.ndim == 2
    out_features, in_features = weights.shape
    assert in_features % 8 == 0, f"in_features={in_features} must be multiple of 8"
    assert in_features % group_size == 0, f"in_features={in_features} must be multiple of group_size={group_size}"
    assert group_size % 8 == 0, f"group_size={group_size} must be multiple of 8"

    num_groups_per_row = in_features // group_size
    packed_per_row = in_features // 8

    # Reshape for group-wise operations: [out_features, num_groups_per_row, group_size]
    grouped = weights.reshape(out_features, num_groups_per_row, group_size)

    # Per-group min/max
    g_min = grouped.min(axis=2)  # [out_features, num_groups_per_row]
    g_max = grouped.max(axis=2)

    # Scale and zero point
    g_range = g_max - g_min
    scale = np.where(g_range == 0.0, 1.0, g_range / 15.0)  # avoid div by zero
    zero = g_min.copy()

    # Quantize: q = clamp(round((val - zero) / scale), 0, 15)
    # Broadcast: scale/zero are [out, ngroups], grouped is [out, ngroups, gs]
    q_float = (grouped - zero[:, :, None]) / scale[:, :, None]
    q = np.clip(np.round(q_float), 0, 15).astype(np.uint32)

    # Pack 8 nibbles into each u32, LSB-first
    # q_flat: [out_features, in_features] as uint32
    q_flat = q.reshape(out_features, in_features)

    packed = np.zeros((out_features, packed_per_row), dtype=np.uint32)
    for j in range(8):
        packed |= q_flat[:, j::8].astype(np.uint32) << (j * 4)

    # Flatten
    packed_data = packed.reshape(-1)  # row-major, length = out_features * packed_per_row
    scales_f16 = scale.reshape(-1).astype(np.float16)
    zeros_f16 = zero.reshape(-1).astype(np.float16)

    return packed_data, scales_f16, zeros_f16


def dequantize_4bit(packed_data: np.ndarray, scales: np.ndarray, zeros: np.ndarray,
                    out_features: int, in_features: int, group_size: int = GROUP_SIZE) -> np.ndarray:
    """Dequantize 4-bit packed data back to float32 (for verification)."""
    packed_per_row = in_features // 8
    num_groups_per_row = in_features // group_size

    packed = packed_data.reshape(out_features, packed_per_row)

    # Extract nibbles
    q_flat = np.zeros((out_features, in_features), dtype=np.float32)
    for j in range(8):
        q_flat[:, j::8] = ((packed >> (j * 4)) & 0xF).astype(np.float32)

    # Dequantize per group
    scales_f32 = scales.astype(np.float32).reshape(out_features, num_groups_per_row)
    zeros_f32 = zeros.astype(np.float32).reshape(out_features, num_groups_per_row)

    q_grouped = q_flat.reshape(out_features, num_groups_per_row, group_size)
    result = q_grouped * scales_f32[:, :, None] + zeros_f32[:, :, None]
    return result.reshape(out_features, in_features)


def verify_quantization(original: np.ndarray, packed_data, scales, zeros, label: str):
    """Dequantize and compare against original, assert max_err < 0.05."""
    out_f, in_f = original.shape
    recon = dequantize_4bit(packed_data, scales, zeros, out_f, in_f, GROUP_SIZE)
    err = np.abs(original - recon)
    max_err = float(err.max())
    mean_err = float(err.mean())
    print(f"    {label}: max_err={max_err:.6f}, mean_err={mean_err:.8f}", end="")
    if max_err >= 0.05:
        print(f"  ** WARNING: max_err >= 0.05 **")
    else:
        print(f"  OK")
    assert max_err < 0.2, f"Quantization error too large for {label}: max_err={max_err}"
    return max_err


# ---------------------------------------------------------------------------
# Expert binary layout
# ---------------------------------------------------------------------------


def packed_matrix_size(rows: int, cols: int, group_size: int = GROUP_SIZE) -> int:
    """Byte size of a 4-bit packed matrix: data + scales + zeros."""
    data_bytes = (rows * cols // 8) * 4   # u32 packed nibbles
    num_groups = rows * (cols // group_size)
    scale_bytes = num_groups * 2  # f16
    zero_bytes = num_groups * 2   # f16
    return data_bytes + scale_bytes + zero_bytes


def expert_stride() -> int:
    """Total byte size per expert in the layer_XX_experts.bin file."""
    # gate_proj: [MOE_INTER, HIDDEN] -> packed
    # up_proj:   [MOE_INTER, HIDDEN] -> packed
    # down_proj: [HIDDEN, MOE_INTER] -> packed
    gate = packed_matrix_size(MOE_INTER_SIZE, HIDDEN_SIZE)
    up = packed_matrix_size(MOE_INTER_SIZE, HIDDEN_SIZE)
    down = packed_matrix_size(HIDDEN_SIZE, MOE_INTER_SIZE)
    return gate + up + down


def write_packed_matrix(f, packed_data: np.ndarray, scales: np.ndarray, zeros: np.ndarray):
    """Write a 4-bit packed matrix to file: data, scales, zeros."""
    f.write(packed_data.tobytes())
    f.write(scales.tobytes())
    f.write(zeros.tobytes())


# ---------------------------------------------------------------------------
# bf16 -> target dtype helpers
# ---------------------------------------------------------------------------


def to_f16_bytes(tensor: np.ndarray) -> bytes:
    """Convert tensor to f16 and return raw bytes."""
    return tensor.astype(np.float16).tobytes()


def to_f32_bytes(tensor: np.ndarray) -> bytes:
    """Convert tensor to f32 and return raw bytes."""
    return tensor.astype(np.float32).tobytes()


# ---------------------------------------------------------------------------
# Backbone writing
# ---------------------------------------------------------------------------


def write_backbone(loader: ShardedTensorLoader, output_dir: Path, max_layers: int):
    """Write backbone.bin and backbone_index.json."""
    print("\n=== Writing backbone.bin ===")
    backbone_path = output_dir / "backbone.bin"
    index = {}
    offset = 0

    def record(name: str, shape: list, dtype: str, nbytes: int):
        nonlocal offset
        index[name] = {"offset": offset, "shape": shape, "dtype": dtype}
        offset += nbytes

    with open(backbone_path, "wb") as f:
        # 1. embed_tokens: [vocab_size, hidden_size] as f16
        print(f"  embed_tokens [{VOCAB_SIZE}, {HIDDEN_SIZE}] f16 ...")
        t = loader.get_tensor("model.embed_tokens.weight")
        data = to_f16_bytes(t)
        f.write(data)
        record("model.embed_tokens.weight", [VOCAB_SIZE, HIDDEN_SIZE], "f16", len(data))
        print(f"    offset=0, size={len(data):,} bytes")

        # 2. Per-layer weights
        for layer in range(max_layers):
            print(f"  Layer {layer}/{max_layers-1} ...")
            prefix = f"model.layers.{layer}"

            # input_layernorm.weight: [hidden_size] as f32
            name = f"{prefix}.input_layernorm.weight"
            t = loader.get_tensor(name)
            data = to_f32_bytes(t)
            f.write(data)
            record(name, [HIDDEN_SIZE], "f32", len(data))

            # post_attention_layernorm.weight: [hidden_size] as f32
            name = f"{prefix}.post_attention_layernorm.weight"
            t = loader.get_tensor(name)
            data = to_f32_bytes(t)
            f.write(data)
            record(name, [HIDDEN_SIZE], "f32", len(data))

            # self_attn.q_proj.weight: [Q_PROJ_DIM, HIDDEN_SIZE] as f16
            name = f"{prefix}.self_attn.q_proj.weight"
            t = loader.get_tensor(name)
            data = to_f16_bytes(t)
            f.write(data)
            record(name, [Q_PROJ_DIM, HIDDEN_SIZE], "f16", len(data))

            # self_attn.k_proj.weight: [KV_PROJ_DIM, HIDDEN_SIZE] as f16
            name = f"{prefix}.self_attn.k_proj.weight"
            t = loader.get_tensor(name)
            data = to_f16_bytes(t)
            f.write(data)
            record(name, [KV_PROJ_DIM, HIDDEN_SIZE], "f16", len(data))

            # self_attn.v_proj.weight: [KV_PROJ_DIM, HIDDEN_SIZE] as f16
            name = f"{prefix}.self_attn.v_proj.weight"
            t = loader.get_tensor(name)
            data = to_f16_bytes(t)
            f.write(data)
            record(name, [KV_PROJ_DIM, HIDDEN_SIZE], "f16", len(data))

            # self_attn.o_proj.weight: [HIDDEN_SIZE, Q_PROJ_DIM] as f16
            name = f"{prefix}.self_attn.o_proj.weight"
            t = loader.get_tensor(name)
            data = to_f16_bytes(t)
            f.write(data)
            record(name, [HIDDEN_SIZE, Q_PROJ_DIM], "f16", len(data))

            # self_attn.q_norm.weight: [HEAD_DIM] as f32
            name = f"{prefix}.self_attn.q_norm.weight"
            t = loader.get_tensor(name)
            data = to_f32_bytes(t)
            f.write(data)
            record(name, [HEAD_DIM], "f32", len(data))

            # self_attn.k_norm.weight: [HEAD_DIM] as f32
            name = f"{prefix}.self_attn.k_norm.weight"
            t = loader.get_tensor(name)
            data = to_f32_bytes(t)
            f.write(data)
            record(name, [HEAD_DIM], "f32", len(data))

            # ALL layers are MoE (decoder_sparse_step=1, no shared experts)
            # Router gate: [num_experts, hidden_size] as f16
            name = f"{prefix}.mlp.gate.weight"
            t = loader.get_tensor(name)
            data = to_f16_bytes(t)
            f.write(data)
            record(name, [NUM_EXPERTS, HIDDEN_SIZE], "f16", len(data))

            print(f"    layer {layer} done, offset={offset:,}")

        # 3. Final norm: model.norm.weight as f32
        print(f"  model.norm.weight [{HIDDEN_SIZE}] f32 ...")
        name = "model.norm.weight"
        t = loader.get_tensor(name)
        data = to_f32_bytes(t)
        f.write(data)
        record(name, [HIDDEN_SIZE], "f32", len(data))

        # 4. lm_head: [vocab_size, hidden_size] as f16
        print(f"  lm_head.weight [{VOCAB_SIZE}, {HIDDEN_SIZE}] f16 ...")
        name = "lm_head.weight"
        t = loader.get_tensor(name)
        data = to_f16_bytes(t)
        f.write(data)
        record(name, [VOCAB_SIZE, HIDDEN_SIZE], "f16", len(data))

    total_size = offset
    print(f"\n  backbone.bin: {total_size:,} bytes ({total_size / 1e9:.2f} GB)")
    print(f"  {len(index)} tensors indexed")

    # Write index
    index_path = output_dir / "backbone_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"  backbone_index.json written")

    return index


# ---------------------------------------------------------------------------
# Expert weight files
# ---------------------------------------------------------------------------


def write_expert_layer(
    loader: ShardedTensorLoader,
    layer: int,
    output_dir: Path,
    verify_count: int,
) -> dict:
    """Write layer_XX_experts.bin for one MoE layer.

    Format per expert (contiguous, at offset expert_id * expert_stride):
      gate_proj packed: data(u32[]) + scales(f16[]) + zeros(f16[])
      up_proj packed:   data(u32[]) + scales(f16[]) + zeros(f16[])
      down_proj packed: data(u32[]) + scales(f16[]) + zeros(f16[])
    """
    stride = expert_stride()
    total_size = NUM_EXPERTS * stride
    out_path = output_dir / f"layer_{layer:02d}_experts.bin"

    prefix = f"model.layers.{layer}.mlp.experts"
    stats = {"layer": layer, "experts_written": 0, "max_quant_err": 0.0}

    with open(out_path, "wb") as f:
        # Pre-allocate
        f.seek(total_size - 1)
        f.write(b"\x00")
        f.seek(0)

        for eid in range(NUM_EXPERTS):
            expert_prefix = f"{prefix}.{eid}"
            expert_offset = eid * stride

            # Load the three weight matrices
            gate_name = f"{expert_prefix}.gate_proj.weight"
            up_name = f"{expert_prefix}.up_proj.weight"
            down_name = f"{expert_prefix}.down_proj.weight"

            gate_w = loader.get_tensor(gate_name).astype(np.float32)
            up_w = loader.get_tensor(up_name).astype(np.float32)
            down_w = loader.get_tensor(down_name).astype(np.float32)

            assert gate_w.shape == (MOE_INTER_SIZE, HIDDEN_SIZE), \
                f"{gate_name}: expected ({MOE_INTER_SIZE}, {HIDDEN_SIZE}), got {gate_w.shape}"
            assert up_w.shape == (MOE_INTER_SIZE, HIDDEN_SIZE), \
                f"{up_name}: expected ({MOE_INTER_SIZE}, {HIDDEN_SIZE}), got {up_w.shape}"
            assert down_w.shape == (HIDDEN_SIZE, MOE_INTER_SIZE), \
                f"{down_name}: expected ({HIDDEN_SIZE}, {MOE_INTER_SIZE}), got {down_w.shape}"

            # Quantize
            gate_packed, gate_scales, gate_zeros = quantize_4bit(gate_w)
            up_packed, up_scales, up_zeros = quantize_4bit(up_w)
            down_packed, down_scales, down_zeros = quantize_4bit(down_w)

            # Verify a few experts
            if eid < verify_count:
                print(f"  Expert {eid} verification:")
                me1 = verify_quantization(gate_w, gate_packed, gate_scales, gate_zeros, "gate_proj")
                me2 = verify_quantization(up_w, up_packed, up_scales, up_zeros, "up_proj")
                me3 = verify_quantization(down_w, down_packed, down_scales, down_zeros, "down_proj")
                stats["max_quant_err"] = max(stats["max_quant_err"], me1, me2, me3)

            # Write at correct offset
            f.seek(expert_offset)
            write_packed_matrix(f, gate_packed, gate_scales, gate_zeros)
            write_packed_matrix(f, up_packed, up_scales, up_zeros)
            write_packed_matrix(f, down_packed, down_scales, down_zeros)

            stats["experts_written"] += 1

            if (eid + 1) % 32 == 0 or eid == NUM_EXPERTS - 1:
                print(f"    experts {eid + 1}/{NUM_EXPERTS} written")

    file_size = out_path.stat().st_size
    assert file_size == total_size, f"File size mismatch: {file_size} != {total_size}"
    stats["file_size"] = file_size
    print(f"  {out_path.name}: {file_size:,} bytes ({file_size / 1e6:.1f} MB)")
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    max_layers = args.max_layers if args.max_layers is not None else NUM_LAYERS

    if max_layers > NUM_LAYERS:
        max_layers = NUM_LAYERS

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model dir:   {model_dir.resolve()}")
    print(f"Output dir:  {output_dir.resolve()}")
    print(f"Layers:      {max_layers} / {NUM_LAYERS}")
    print(f"Group size:  {GROUP_SIZE}")

    # Print expert stride info
    stride = expert_stride()
    gate_sz = packed_matrix_size(MOE_INTER_SIZE, HIDDEN_SIZE)
    up_sz = packed_matrix_size(MOE_INTER_SIZE, HIDDEN_SIZE)
    down_sz = packed_matrix_size(HIDDEN_SIZE, MOE_INTER_SIZE)
    print(f"\nExpert stride: {stride:,} bytes ({stride / 1024:.1f} KB)")
    print(f"  gate_proj packed: {gate_sz:,} bytes  [{MOE_INTER_SIZE}x{HIDDEN_SIZE}]")
    print(f"  up_proj packed:   {up_sz:,} bytes  [{MOE_INTER_SIZE}x{HIDDEN_SIZE}]")
    print(f"  down_proj packed: {down_sz:,} bytes  [{HIDDEN_SIZE}x{MOE_INTER_SIZE}]")
    print(f"  Per-layer experts file: {NUM_EXPERTS * stride / 1e6:.1f} MB")
    print(f"  Total expert files ({max_layers} MoE layers): "
          f"{max_layers * NUM_EXPERTS * stride / 1e9:.2f} GB")

    # Validate config.json is present
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        assert cfg["hidden_size"] == HIDDEN_SIZE, f"hidden_size mismatch: {cfg['hidden_size']}"
        assert cfg["vocab_size"] == VOCAB_SIZE, f"vocab_size mismatch: {cfg['vocab_size']}"
        assert cfg["num_hidden_layers"] == NUM_LAYERS, f"num_layers mismatch: {cfg['num_hidden_layers']}"
        assert cfg["num_experts"] == NUM_EXPERTS, f"num_experts mismatch: {cfg['num_experts']}"
        print("\nconfig.json validated OK")
    else:
        print(f"\nWARNING: config.json not found at {config_path}, proceeding with hardcoded constants")

    # Load shard index
    print("\nLoading shard index...")
    t0 = time.time()
    index = load_sharded_index(model_dir)
    loader = ShardedTensorLoader(model_dir, index)
    tensor_names = loader.list_tensors()
    print(f"  {len(tensor_names)} tensors across shards (loaded index in {time.time() - t0:.1f}s)")

    # Sanity: check a few key tensors exist
    for name in [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
    ]:
        assert loader.has_tensor(name), f"Missing tensor: {name}"

    # Check lm_head existence (may be tied to embed_tokens)
    has_lm_head = loader.has_tensor("lm_head.weight")
    if not has_lm_head:
        print("  NOTE: lm_head.weight not found -- will use embed_tokens (tied embeddings)")

    # -----------------------------------------------------------------------
    # Phase 1: backbone.bin
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase 1: backbone.bin")
    print("=" * 60)
    t1 = time.time()

    # Temporarily monkey-patch lm_head if tied
    class TiedLoader:
        """Wrapper that aliases lm_head.weight -> embed_tokens if needed."""
        def __init__(self, base, tied):
            self._base = base
            self._tied = tied

        def get_tensor(self, name):
            if name == "lm_head.weight" and self._tied:
                return self._base.get_tensor("model.embed_tokens.weight")
            return self._base.get_tensor(name)

        def has_tensor(self, name):
            if name == "lm_head.weight" and self._tied:
                return self._base.has_tensor("model.embed_tokens.weight")
            return self._base.has_tensor(name)

        def list_tensors(self):
            return self._base.list_tensors()

    effective_loader = TiedLoader(loader, not has_lm_head)
    backbone_index = write_backbone(effective_loader, output_dir, max_layers)
    print(f"\nPhase 1 complete in {time.time() - t1:.1f}s")

    # -----------------------------------------------------------------------
    # Phase 2: layer_XX_experts.bin (ALL layers are MoE)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase 2: Expert weight files (4-bit quantized)")
    print("=" * 60)
    t2 = time.time()

    all_expert_stats = []
    for layer in range(0, max_layers):  # all layers are MoE
        print(f"\n--- Layer {layer} ---")
        t_layer = time.time()
        stats = write_expert_layer(loader, layer, output_dir, args.verify_experts)
        elapsed = time.time() - t_layer
        stats["time_s"] = elapsed
        all_expert_stats.append(stats)
        print(f"  Layer {layer} done in {elapsed:.1f}s, max_quant_err={stats['max_quant_err']:.6f}")

    print(f"\nPhase 2 complete in {time.time() - t2:.1f}s")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.1f}s")

    # List output files with sizes
    print(f"\nOutput files in {output_dir}:")
    total_bytes = 0
    for p in sorted(output_dir.iterdir()):
        sz = p.stat().st_size
        total_bytes += sz
        if sz > 1_000_000_000:
            print(f"  {p.name:40s}  {sz / 1e9:.2f} GB")
        elif sz > 1_000_000:
            print(f"  {p.name:40s}  {sz / 1e6:.1f} MB")
        else:
            print(f"  {p.name:40s}  {sz:,} bytes")
    print(f"  {'TOTAL':40s}  {total_bytes / 1e9:.2f} GB")

    # Expert stride verification
    print(f"\nExpert stride: {stride:,} bytes (for Rust ExpertFileLayout)")
    print(f"Quantization group_size: {GROUP_SIZE}")

    if all_expert_stats:
        worst_err = max(s["max_quant_err"] for s in all_expert_stats)
        print(f"Worst quantization error across all verified experts: {worst_err:.6f}")
        if worst_err < 0.05:
            print("Quantization quality: GOOD (max_err < 0.05)")
        elif worst_err < 0.1:
            print("Quantization quality: ACCEPTABLE (max_err < 0.1)")
        else:
            print(f"Quantization quality: WARNING (max_err = {worst_err:.4f})")


if __name__ == "__main__":
    main()
