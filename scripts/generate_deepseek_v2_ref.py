#!/usr/bin/env python3
"""Generate HF golden reference tensors for DeepSeek-V2-Lite MLA validation.

Requires: transformers==4.39.3 (compatible with V2-Lite custom code)

Outputs:
  weights/references/deepseek_v2_lite_intermediates.npz  — per-layer MLA tensors
  weights/references/deepseek_v2_lite_greedy.json        — 20-token greedy generation
"""

import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "weights/deepseek-v2-lite"


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    print("Loading model (bf16 on CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    model.eval()

    # ── Part 1: Per-layer MLA intermediates ──────────────────────────
    print("\n=== Part 1: Extracting MLA intermediates ===")
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"Prompt: {prompt!r}  →  {input_ids.shape[1]} tokens: {input_ids[0].tolist()}")

    intermediates = {}

    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            with torch.no_grad():
                # hidden_states may be in args or kwargs
                if len(args) > 0:
                    hidden = args[0].float()
                else:
                    hidden = kwargs['hidden_states'].float()
                bsz, seq_len, _ = hidden.shape

                nope_dim = module.qk_nope_head_dim
                rope_dim = module.qk_rope_head_dim
                num_heads = module.num_heads
                v_head_dim = module.v_head_dim
                kv_lora_rank = module.kv_lora_rank

                # Q projection (V2-Lite: direct q_proj, no LoRA)
                q = torch.nn.functional.linear(hidden, module.q_proj.weight.float())
                q = q.view(bsz, seq_len, num_heads, nope_dim + rope_dim)
                q_nope = q[:, :, :, :nope_dim].contiguous()
                q_pe = q[:, :, :, nope_dim:].contiguous()

                # KV compression
                kv_a = torch.nn.functional.linear(hidden, module.kv_a_proj_with_mqa.weight.float())
                compressed_kv = kv_a[:, :, :kv_lora_rank]
                k_pe_raw = kv_a[:, :, kv_lora_rank:]

                # kv_a_layernorm (CRITICAL: must happen before cache)
                kv_normed = module.kv_a_layernorm(
                    compressed_kv.to(module.kv_a_layernorm.weight.dtype)
                ).float()

                # kv_b_proj → K_nope + V
                kv_b = torch.nn.functional.linear(
                    kv_normed.to(module.kv_b_proj.weight.dtype),
                    module.kv_b_proj.weight
                ).float()
                kv_b = kv_b.view(bsz, seq_len, num_heads, nope_dim + v_head_dim)
                k_nope = kv_b[:, :, :, :nope_dim]
                v_from_kv = kv_b[:, :, :, nope_dim:]

                prefix = f"layer{layer_idx}"
                intermediates[f"{prefix}_hidden_input"] = hidden[0, -1].cpu().numpy()
                intermediates[f"{prefix}_q_nope"] = q_nope[0, -1].cpu().numpy()
                intermediates[f"{prefix}_q_pe"] = q_pe[0, -1].cpu().numpy()
                intermediates[f"{prefix}_compressed_kv"] = compressed_kv[0, -1].cpu().numpy()
                intermediates[f"{prefix}_kv_normed"] = kv_normed[0, -1].cpu().numpy()
                intermediates[f"{prefix}_k_pe_raw"] = k_pe_raw[0, -1].cpu().numpy()
                intermediates[f"{prefix}_k_nope"] = k_nope[0, -1].cpu().numpy()
                intermediates[f"{prefix}_v_from_kv"] = v_from_kv[0, -1].cpu().numpy()

                # W_UK / W_UV split from kv_b_proj
                w_kv_b = module.kv_b_proj.weight.float()
                w_kv_b_reshaped = w_kv_b.view(num_heads, nope_dim + v_head_dim, kv_lora_rank)
                w_uk = w_kv_b_reshaped[:, :nope_dim, :].contiguous()
                w_uv = w_kv_b_reshaped[:, nope_dim:, :].contiguous()
                intermediates[f"{prefix}_w_uk"] = w_uk.cpu().numpy()
                intermediates[f"{prefix}_w_uv"] = w_uv.cpu().numpy()

            return output
        return hook_fn

    hooks = []
    for li in [0, 1]:
        layer = model.model.layers[li]
        h = layer.self_attn.register_forward_hook(make_hook(li), with_kwargs=True)
        hooks.append(h)

    layer_outputs = {}
    def make_layer_output_hook(layer_idx):
        def hook_fn(module, args, output):
            if isinstance(output, tuple):
                layer_outputs[layer_idx] = output[0][0, -1].float().detach().cpu().numpy()
            return output
        return hook_fn

    for li in [0, 1]:
        h = model.model.layers[li].register_forward_hook(make_layer_output_hook(li))
        hooks.append(h)

    with torch.no_grad():
        out = model(input_ids)

    for h in hooks:
        h.remove()

    for li in [0, 1]:
        if li in layer_outputs:
            intermediates[f"layer{li}_output"] = layer_outputs[li]

    out_path = "weights/references/deepseek_v2_lite_intermediates.npz"
    np.savez(out_path, **intermediates)
    print(f"Saved intermediates to {out_path}")
    for k, v in sorted(intermediates.items()):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    # ── Part 2: Greedy generation reference ──────────────────────────
    print("\n=== Part 2: Greedy generation ===")
    prompt2 = "The quick brown fox"
    inputs2 = tokenizer(prompt2, return_tensors="pt")
    input_ids2 = inputs2["input_ids"]
    prompt_tokens = input_ids2[0].tolist()
    print(f"Prompt: {prompt2!r}  →  {len(prompt_tokens)} tokens: {prompt_tokens}")

    with torch.no_grad():
        gen_out = model.generate(
            input_ids2,
            max_new_tokens=20,
            do_sample=False,
        )

    gen_ids = gen_out[0].tolist()
    new_ids = gen_ids[len(prompt_tokens):]
    gen_text = tokenizer.decode(new_ids)
    print(f"Generated {len(new_ids)} tokens: {new_ids}")
    print(f"Text: {gen_text!r}")

    ref = {
        "prompt": prompt2,
        "prompt_token_ids": prompt_tokens,
        "generated_token_ids": new_ids,
        "generated_text": gen_text,
        "all_token_ids": gen_ids,
        "model": "deepseek-ai/DeepSeek-V2-Lite",
        "method": "greedy",
        "max_new_tokens": 20,
    }
    ref_path = "weights/references/deepseek_v2_lite_greedy.json"
    with open(ref_path, "w") as f:
        json.dump(ref, f, indent=2)
    print(f"Saved greedy reference to {ref_path}")

    # ── Part 3: Last-token logits ──
    print("\n=== Part 3: Last-token logits for 'The capital of France is' ===")
    with torch.no_grad():
        logits_out = model(input_ids)
        last_logits = logits_out.logits[0, -1].float().cpu().numpy()
    np.save("weights/references/deepseek_v2_lite_last_logits.npy", last_logits)
    top5_ids = np.argsort(last_logits)[-5:][::-1]
    print(f"Top-5 logit token IDs: {top5_ids.tolist()}")
    for tid in top5_ids:
        print(f"  {tid}: {tokenizer.decode([tid])!r} = {last_logits[tid]:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
