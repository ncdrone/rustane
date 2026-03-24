#!/usr/bin/env python3
"""Generate DeepSeek-V3 reference outputs via DeepSeek API for L3 validation.

Calls the DeepSeek API with logprobs enabled to get top-20 token probabilities
for validation against our local inference engine.

Usage:
    DEEPSEEK_API_KEY=<key> python3 scripts/v3_api_reference.py \
        --output-dir weights/references/
"""

import argparse
import json
import os
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: pip install requests")
    raise


def query_deepseek(prompt: str, api_key: str, max_tokens: int = 20) -> dict:
    """Query DeepSeek API with logprobs."""
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 20,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Generate V3 API reference outputs")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prompt", default="The capital of France is", help="Test prompt")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: Set DEEPSEEK_API_KEY environment variable")
        print("  Get one at https://platform.deepseek.com/")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Prompt: {args.prompt!r}")
    print("Querying DeepSeek API...")

    result = query_deepseek(args.prompt, api_key)

    # Extract logprobs
    choice = result["choices"][0]
    generated_text = choice["message"]["content"]
    logprobs_content = choice.get("logprobs", {}).get("content", [])

    print(f"Generated: {generated_text!r}")
    print(f"Tokens with logprobs: {len(logprobs_content)}")

    # Save full response
    out_path = args.output_dir / "deepseek_v3_api_reference.json"
    reference = {
        "prompt": args.prompt,
        "generated_text": generated_text,
        "model": result.get("model", "deepseek-chat"),
        "tokens": [],
    }

    for i, tok_info in enumerate(logprobs_content):
        token_entry = {
            "position": i,
            "token": tok_info["token"],
            "logprob": tok_info["logprob"],
            "top_logprobs": [
                {"token": lp["token"], "logprob": lp["logprob"]}
                for lp in tok_info.get("top_logprobs", [])
            ],
        }
        reference["tokens"].append(token_entry)
        top5 = tok_info.get("top_logprobs", [])[:5]
        top5_str = ", ".join(f"{lp['token']!r}({lp['logprob']:.3f})" for lp in top5)
        print(f"  [{i}] {tok_info['token']!r} ({tok_info['logprob']:.3f}) | top5: {top5_str}")

    with open(out_path, "w") as f:
        json.dump(reference, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
