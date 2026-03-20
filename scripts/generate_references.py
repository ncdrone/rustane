#!/usr/bin/env python3
"""Generate HuggingFace reference tensors for Rust inference test validation.

Produces reference data from the canonical HuggingFace Qwen3-30B-A3B model:
  1. Tokenizer IDs for test prompts
  2. Attention layer 1 input/output pair
  3. Greedy generation (20 tokens from "The capital of France is")

Output: weights/references/*.{npy,json}

Usage:
  python scripts/generate_references.py
  python scripts/generate_references.py --output-dir weights/references
  python scripts/generate_references.py --model-id Qwen/Qwen3-30B-A3B
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description="Generate HF reference tensors")
    p.add_argument(
        "--model-id",
        default="Qwen/Qwen3-30B-A3B",
        help="HuggingFace model ID (default: Qwen/Qwen3-30B-A3B)",
    )
    p.add_argument(
        "--output-dir",
        default="weights/references",
        help="Output directory for reference files",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Device to run on (cpu or mps). bf16 model is ~60 GB, cpu is safest.",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
        help="Max tokens to generate for greedy reference",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# 1. Tokenizer references
# ---------------------------------------------------------------------------

TEST_TEXTS = [
    "The capital of France is",
    "Hello, world!",
    "Rust is a systems programming language.",
    "",  # empty string edge case
    "A" * 200,  # long repeated char
]


def generate_tokenizer_refs(tokenizer, output_dir: Path):
    """Save tokenizer IDs for test texts."""
    print("\n[1/3] Generating tokenizer references...")

    results = {}
    for i, text in enumerate(TEST_TEXTS):
        ids = tokenizer.encode(text, add_special_tokens=False)
        label = f"text_{i}"
        results[label] = {
            "text": text[:80] + ("..." if len(text) > 80 else ""),
            "ids": ids,
            "num_tokens": len(ids),
        }
        print(f"  {label}: {len(ids)} tokens  ids[:8]={ids[:8]}")

    # Save as JSON (token IDs are small integers, JSON is fine)
    out_path = output_dir / "tokenizer_ids.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_path}")

    # Also save the vocab size for sanity checking
    meta = {
        "vocab_size": tokenizer.vocab_size,
        "model_id": tokenizer.name_or_path,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    meta_path = output_dir / "tokenizer_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: {meta_path}")


# ---------------------------------------------------------------------------
# 2. Attention layer reference
# ---------------------------------------------------------------------------


def generate_attention_ref(model, tokenizer, output_dir: Path, device: str):
    """Save layer 1 input and output tensors for a test prompt."""
    print("\n[2/3] Generating attention layer 1 reference...")

    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"  Prompt: '{prompt}'  ->  {input_ids.shape[1]} tokens")

    # Register hooks on layer 1 to capture input/output
    layer1_input = {}
    layer1_output = {}

    def hook_fn(module, inp, out):
        # inp is a tuple; first element is the hidden states
        layer1_input["hidden"] = inp[0].detach().cpu().float()
        # out is typically (hidden_states, ...) or a tuple
        if isinstance(out, tuple):
            layer1_output["hidden"] = out[0].detach().cpu().float()
        else:
            layer1_output["hidden"] = out.detach().cpu().float()

    hook = model.model.layers[1].register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(input_ids)

    hook.remove()

    if "hidden" not in layer1_input:
        print("  WARNING: hook did not fire, skipping attention reference")
        return

    inp_tensor = layer1_input["hidden"].numpy()  # [1, seq_len, hidden_size]
    out_tensor = layer1_output["hidden"].numpy()

    print(f"  Layer 1 input:  shape={inp_tensor.shape}, dtype={inp_tensor.dtype}")
    print(f"  Layer 1 output: shape={out_tensor.shape}, dtype={out_tensor.dtype}")
    print(f"  Input  norm: {np.linalg.norm(inp_tensor):.4f}")
    print(f"  Output norm: {np.linalg.norm(out_tensor):.4f}")

    np.save(output_dir / "layer1_input.npy", inp_tensor)
    np.save(output_dir / "layer1_output.npy", out_tensor)

    # Save metadata
    meta = {
        "prompt": prompt,
        "input_ids": input_ids[0].tolist(),
        "layer_index": 1,
        "input_shape": list(inp_tensor.shape),
        "output_shape": list(out_tensor.shape),
        "input_norm": float(np.linalg.norm(inp_tensor)),
        "output_norm": float(np.linalg.norm(out_tensor)),
    }
    with open(output_dir / "layer1_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: layer1_input.npy, layer1_output.npy, layer1_meta.json")


# ---------------------------------------------------------------------------
# 3. Greedy generation reference
# ---------------------------------------------------------------------------


def generate_greedy_ref(model, tokenizer, output_dir: Path, device: str, max_new: int):
    """Greedy-decode from a prompt, save token IDs and text."""
    print("\n[3/3] Generating greedy generation reference...")

    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"  Prompt: '{prompt}'  ({input_ids.shape[1]} tokens)")
    print(f"  Max new tokens: {max_new}")

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new,
            do_sample=False,  # greedy
            temperature=None,
            top_p=None,
            # Disable thinking/special features if present
            pad_token_id=tokenizer.eos_token_id,
        )

    output_ids = output[0].tolist()
    prompt_ids = input_ids[0].tolist()
    generated_ids = output_ids[len(prompt_ids):]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    print(f"  Generated {len(generated_ids)} tokens")
    print(f"  Generated IDs: {generated_ids}")
    print(f"  Generated text: '{generated_text}'")
    print(f"  Full output: '{full_text}'")

    result = {
        "prompt": prompt,
        "prompt_ids": prompt_ids,
        "generated_ids": generated_ids,
        "full_ids": output_ids,
        "generated_text": generated_text,
        "full_text": full_text,
        "max_new_tokens": max_new,
        "decoding": "greedy",
    }

    out_path = output_dir / "greedy_generation.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_path}")

    # Also save the full logits for the last prompt token (for argmax validation)
    with torch.no_grad():
        logits = model(input_ids).logits  # [1, seq_len, vocab_size]
    last_logits = logits[0, -1, :].cpu().float().numpy()
    np.save(output_dir / "last_prompt_logits.npy", last_logits)
    print(f"  Saved: last_prompt_logits.npy  (shape={last_logits.shape})")
    print(f"  Argmax token: {np.argmax(last_logits)} = '{tokenizer.decode([int(np.argmax(last_logits))])}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model_id}")
    print(f"Output: {output_dir.resolve()}")
    print(f"Device: {args.device}")

    # Load tokenizer first (fast, no GPU needed)
    print("\nLoading tokenizer...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  BOS: {tokenizer.bos_token_id}, EOS: {tokenizer.eos_token_id}")

    generate_tokenizer_refs(tokenizer, output_dir)

    # Load model (this is the big one -- ~60 GB in bf16)
    print("\nLoading model (bf16)... this may take a while and use ~60 GB RAM")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    generate_attention_ref(model, tokenizer, output_dir, args.device)
    generate_greedy_ref(model, tokenizer, output_dir, args.device, args.max_new_tokens)

    print("\n=== All references generated ===")
    print(f"Output directory: {output_dir.resolve()}")
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size
        if size > 1_000_000:
            print(f"  {f.name}  ({size / 1e6:.1f} MB)")
        else:
            print(f"  {f.name}  ({size:,} bytes)")


if __name__ == "__main__":
    main()
