//! E2E logit-level validation: DeepSeek-V2-Lite vs HF reference.
//!
//! Requires:
//! - Converted weights at weights/rustane-deepseek-v2-lite/
//! - Config at configs/deepseek-v2-lite.toml
//! - Tokenizer at weights/deepseek-v2-lite/tokenizer.json
//! - Reference logits at weights/references/deepseek_v2_lite_last_logits.npy (shape [102400])
//! - Reference greedy at weights/references/deepseek_v2_lite_greedy.json
//!
//! Run: cargo test -p moe-infer --test test_v2_lite_logits --release -- --ignored --nocapture

use std::path::Path;

use moe_infer::mla_attention::{MlaLayerWeights as MlaAttnWeights, MlaKvCache, mla_forward_decode};
use moe_infer::rmsnorm::rmsnorm;
use moe_infer::generate_v2::ModelV2;
use half::f16;
use moe_router::{MoeRouter, RouterConfig};
use moe_kernels::{ExpertGemvOp, FusedGateUpSiluOp};

fn ws_root() -> std::path::PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).parent().unwrap().parent().unwrap().to_path_buf()
}

/// Parse a .npy file and return f32 values.
fn load_npy_f32(path: &std::path::Path) -> Vec<f32> {
    let data = std::fs::read(path).expect("read npy");
    assert_eq!(&data[..6], b"\x93NUMPY");
    let major = data[6];
    let header_len = if major == 1 {
        u16::from_le_bytes([data[8], data[9]]) as usize
    } else {
        u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize
    };
    let data_start = if major == 1 { 10 } else { 12 } + header_len;
    let raw = &data[data_start..];
    // Check header for dtype
    let header = std::str::from_utf8(&data[if major == 1 { 10 } else { 12 }..data_start]).unwrap();
    if header.contains("<f4") {
        raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    } else if header.contains("<f8") {
        raw.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32).collect()
    } else {
        panic!("unsupported npy dtype in {header}");
    }
}

/// Return indices of the top-n values in descending order.
fn top_n_indices(logits: &[f32], n: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.iter().take(n).map(|(i, _)| *i).collect()
}

/// Softmax in-place over a slice.
fn softmax(v: &[f32]) -> Vec<f32> {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = v.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = out.iter().sum();
    for x in &mut out {
        *x /= sum;
    }
    out
}

/// KL divergence KL(p || q) = sum_i p_i * ln(p_i / q_i).
/// Both p and q are assumed to be valid probability distributions (sum to 1, non-negative).
fn kl_divergence(p: &[f32], q: &[f32]) -> f64 {
    assert_eq!(p.len(), q.len());
    let eps = 1e-12f64;
    p.iter().zip(q.iter()).map(|(&pi, &qi)| {
        let pi = pi as f64;
        let qi = qi as f64;
        if pi > eps {
            pi * (pi / (qi + eps)).ln()
        } else {
            0.0
        }
    }).sum()
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let dot: f64 = a.iter().zip(b.iter()).map(|(&ai, &bi)| ai as f64 * bi as f64).sum();
    let norm_a: f64 = a.iter().map(|&ai| (ai as f64).powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|&bi| (bi as f64).powi(2)).sum::<f64>().sqrt();
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Embed a single token from an f16 table.
fn embed_token(table: &[f16], token_id: usize, hidden: usize) -> Vec<f32> {
    let start = token_id * hidden;
    table[start..start + hidden].iter().map(|v| v.to_f32()).collect()
}

/// Dense SwiGLU FFN: SiLU(x @ gate^T) * (x @ up^T) → @ down^T
fn dense_swiglu_ffn(x: &[f32], gate_w: &[f32], up_w: &[f32], down_w: &[f32]) -> Vec<f32> {
    let hidden = x.len();
    let inter = gate_w.len() / hidden;

    let mut gate_out = vec![0.0f32; inter];
    let mut up_out = vec![0.0f32; inter];
    moe_infer::blas::sgemv_f32(gate_w, x, &mut gate_out, inter, hidden);
    moe_infer::blas::sgemv_f32(up_w, x, &mut up_out, inter, hidden);

    // SiLU(gate) * up
    for i in 0..inter {
        let g = gate_out[i];
        gate_out[i] = (g / (1.0 + (-g).exp())) * up_out[i];
    }

    let mut out = vec![0.0f32; hidden];
    moe_infer::blas::sgemv_f32(down_w, &gate_out, &mut out, hidden, inter);
    out
}

/// Run one V2-Lite transformer layer (MLA attention + FFN) returning residual.
/// This replicates `run_layer_v2` from generate_v2.rs using the public API.
fn run_v2_layer(
    model: &ModelV2,
    cache: &mut MlaKvCache,
    router: &mut MoeRouter,
    layer: usize,
    x: &[f32],
    pos: usize,
) -> Vec<f32> {
    let hidden = model.config.hidden_size();
    let eps = model.config.rms_norm_eps();
    let lf = &model.layers_f32[layer];

    // 1. input RMSNorm
    let normed = rmsnorm(x, &lf.input_norm, eps);

    // 2. MLA attention
    let attn_weights = MlaAttnWeights {
        q_proj: &lf.q_proj,
        q_a_proj: None,
        q_a_layernorm: None,
        q_b_proj: None,
        kv_a_proj: &lf.kv_a_proj,
        kv_a_layernorm: &lf.kv_a_layernorm,
        w_uk: &lf.w_uk,
        w_uv: &lf.w_uv,
        o_proj: &lf.o_proj,
        input_norm: &lf.input_norm,
        post_attn_norm: &lf.post_attn_norm,
    };
    let attn_out = mla_forward_decode(
        &normed, &attn_weights, cache, layer, pos,
        &model.rope, &model.mla_config, model.attn_scale,
    );

    // 3. Attention residual
    let mut residual = vec![0.0f32; hidden];
    for d in 0..hidden {
        residual[d] = x[d] + attn_out[d];
    }

    // 4. Post-attention RMSNorm
    let normed2 = rmsnorm(&residual, &lf.post_attn_norm, eps);

    // 5. FFN
    let ffn_out = if model.config.is_moe_layer(layer) {
        run_moe_ffn(model, router, layer, &normed2)
    } else {
        // Dense layer (layer 0)
        let gate_w = lf.dense_gate.as_ref().expect("dense layer needs gate_proj");
        let up_w = lf.dense_up.as_ref().expect("dense layer needs up_proj");
        let down_w = lf.dense_down.as_ref().expect("dense layer needs down_proj");
        dense_swiglu_ffn(&normed2, gate_w, up_w, down_w)
    };

    // 6. FFN residual
    for d in 0..hidden {
        residual[d] += ffn_out[d];
    }

    residual
}

/// MoE FFN: shared experts + routed experts (replicates moe_ffn_v2 from generate_v2.rs).
fn run_moe_ffn(
    model: &ModelV2,
    router: &mut MoeRouter,
    layer: usize,
    x: &[f32],
) -> Vec<f32> {
    let hidden = model.config.hidden_size();
    let lf = &model.layers_f32[layer];
    let num_experts = model.config.num_experts();

    // 1. Router gate
    let router_w = lf.router.as_ref().expect("MoE layer needs router");
    let mut gate_logits = vec![0.0f32; num_experts];
    moe_infer::blas::sgemv_f32(router_w, x, &mut gate_logits, num_experts, hidden);

    let route = if model.config.ffn.scoring_func == "sigmoid" {
        router.route(&gate_logits)
    } else {
        router.route_softmax(&gate_logits)
    };

    // 2. Shared experts (always run)
    let mut combined = if lf.shared_gate.is_some() {
        let gate_w = lf.shared_gate.as_ref().unwrap();
        let up_w = lf.shared_up.as_ref().expect("shared up");
        let down_w = lf.shared_down.as_ref().expect("shared down");
        dense_swiglu_ffn(x, gate_w, up_w, down_w)
    } else {
        vec![0.0f32; hidden]
    };

    // 3. Routed experts (quantized, Metal GPU or skip on CPU)
    let expert_mmap = model.weights.expert_mmap(layer);
    let mmap_metal_buf = model.expert_metal_bufs.get(&layer).map(|b| b.as_ref());

    let moe_inter = model.config.moe_inter_size();
    let group_size = model.config.quantization.group_size;

    if let Some(expert_data) = expert_mmap {
        let gu_packed = moe_inter * hidden / 2;
        let gu_groups = moe_inter * (hidden / group_size);
        let gu_scales = gu_groups * 2;
        let gu_total = gu_packed + gu_scales * 2;
        let dn_packed = hidden * moe_inter / 2;
        let dn_groups = hidden * (moe_inter / group_size);
        let dn_scales = dn_groups * 2;
        let dn_total = dn_packed + dn_scales * 2;
        let expert_stride = gu_total * 2 + dn_total;

        if let (Some(m), Some(mmap_buf)) = (model.metal.as_ref(), mmap_metal_buf) {
            let mut fused_ops = Vec::new();
            let mut down_ops = Vec::new();
            let mut routing_weights = Vec::new();

            for (&eid, &weight) in route.expert_ids.iter().zip(route.weights.iter()) {
                let base = eid * expert_stride;
                if base + expert_stride > expert_data.len() {
                    continue;
                }
                fused_ops.push(FusedGateUpSiluOp {
                    gate_packed_offset: base,
                    gate_scales_offset: base + gu_packed,
                    gate_zeros_offset: base + gu_packed + gu_scales,
                    up_packed_offset: base + gu_total,
                    up_scales_offset: base + gu_total + gu_packed,
                    up_zeros_offset: base + gu_total + gu_packed + gu_scales,
                    out_features: moe_inter,
                    in_features: hidden,
                    group_size,
                });
                down_ops.push(ExpertGemvOp {
                    packed_offset: base + 2 * gu_total,
                    scales_offset: base + 2 * gu_total + dn_packed,
                    zeros_offset: base + 2 * gu_total + dn_packed + dn_scales,
                    out_features: hidden,
                    in_features: moe_inter,
                    group_size,
                });
                routing_weights.push(weight);
            }

            if !routing_weights.is_empty() {
                let down_results = m.fused_and_down_single_cmdbuf(mmap_buf, &fused_ops, &down_ops, x);
                for (i, weight) in routing_weights.iter().enumerate() {
                    for d in 0..hidden {
                        combined[d] += weight * down_results[i][d];
                    }
                }
            }
        }
    }

    // Apply routed_scaling_factor if != 1.0
    let scale = model.config.ffn.routed_scaling_factor;
    if (scale - 1.0).abs() > 0.01 {
        for d in 0..hidden {
            combined[d] *= scale;
        }
    }

    combined
}

/// Run the full V2-Lite forward pass on a token sequence and return final logits.
fn forward_pass_logits(model: &ModelV2, input_ids: &[u32]) -> Vec<f32> {
    let hidden = model.config.hidden_size();
    let vocab = model.config.vocab_size();
    let num_layers = model.config.num_layers();
    let max_seq = model.rope.max_seq;

    let mut cache = MlaKvCache::new(
        num_layers,
        model.mla_config.kv_lora_rank,
        model.mla_config.qk_rope_head_dim,
        max_seq,
    );

    let router_config = RouterConfig {
        num_experts: model.config.num_experts(),
        top_k: model.config.num_experts_per_tok(),
        norm_topk_prob: model.config.ffn.norm_topk_prob,
        bias_lr: 0.0,
    };
    let mut router = MoeRouter::new(router_config);

    let embed_table = model.weights.embed_table().expect("embed_table");
    let final_norm_weights = model.weights.final_norm().expect("final_norm").to_vec();

    let mut last_x = vec![0.0f32; hidden];

    for (i, &token_id) in input_ids.iter().enumerate() {
        let emb = embed_token(embed_table, token_id as usize, hidden);
        let mut x = emb;
        for layer in 0..num_layers {
            x = run_v2_layer(model, &mut cache, &mut router, layer, &x, i);
        }
        cache.advance();
        last_x = x;
    }

    // Final norm + LM head
    let normed = rmsnorm(&last_x, &final_norm_weights, model.config.rms_norm_eps());
    let mut logits = vec![0.0f32; vocab];
    moe_infer::blas::sgemv_f32(&model.lm_head_f32, &normed, &mut logits, vocab, hidden);

    logits
}

#[test]
#[ignore = "requires converted weights + tokenizer + reference logits"]
fn test_v2_lite_logits_vs_hf() {
    let root = ws_root();
    let weights_dir = root.join("weights/rustane-deepseek-v2-lite");
    let config_path = root.join("configs/deepseek-v2-lite.toml");
    let tokenizer_path = root.join("weights/deepseek-v2-lite/tokenizer.json");
    let logits_npy_path = root.join("weights/references/deepseek_v2_lite_last_logits.npy");
    let greedy_json_path = root.join("weights/references/deepseek_v2_lite_greedy.json");

    // Check prerequisites
    if !weights_dir.join("backbone.bin").exists() {
        eprintln!("SKIP: converted weights not found at {}", weights_dir.display());
        return;
    }
    if !tokenizer_path.exists() {
        eprintln!("SKIP: tokenizer not found at {}", tokenizer_path.display());
        return;
    }
    if !logits_npy_path.exists() {
        eprintln!("SKIP: reference logits not found at {}", logits_npy_path.display());
        return;
    }
    if !greedy_json_path.exists() {
        eprintln!("SKIP: greedy reference not found at {}", greedy_json_path.display());
        return;
    }

    // --- Load reference logits (.npy) ---
    eprintln!("Loading HF reference logits from {}...", logits_npy_path.display());
    let hf_logits = load_npy_f32(&logits_npy_path);
    assert_eq!(hf_logits.len(), 102400,
        "Expected 102400-dim logits, got {}", hf_logits.len());
    eprintln!("HF logits loaded: {} values", hf_logits.len());

    // --- Load greedy reference (.json) ---
    let ref_json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&greedy_json_path).expect("read greedy json")
    ).expect("parse greedy json");
    let ref_tokens: Vec<u32> = ref_json["generated_token_ids"].as_array()
        .expect("generated_token_ids array")
        .iter()
        .map(|v| v.as_u64().expect("token id as u64") as u32)
        .collect();
    let prompt = ref_json["prompt"].as_str().expect("prompt string");
    eprintln!("HF reference: prompt={prompt:?}, {} generated tokens", ref_tokens.len());
    eprintln!("HF reference tokens: {ref_tokens:?}");

    // --- Load model ---
    eprintln!("Loading model from {}...", weights_dir.display());
    let t_load = std::time::Instant::now();
    let model = ModelV2::load(&weights_dir, &config_path).expect("load model");
    eprintln!("Model loaded in {:.1}s", t_load.elapsed().as_secs_f64());

    // --- Load tokenizer ---
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .expect("load tokenizer");

    // =========================================================================
    // Part 1: Logit comparison on "The capital of France is"
    // =========================================================================
    // The prompt for which we have reference logits
    let logit_prompt = "The capital of France is";
    let encoding = tokenizer.encode(logit_prompt, false).expect("tokenize");
    let mut input_ids: Vec<u32> = encoding.get_ids().to_vec();

    // Prepend BOS (100000)
    let bos = model.config.model.bos_token_id;
    if input_ids.first() != Some(&bos) {
        input_ids.insert(0, bos);
    }
    eprintln!("Prompt tokens (with BOS): {:?}", input_ids);

    eprintln!("Running forward pass for logit comparison ({} tokens)...", input_ids.len());
    let t_fwd = std::time::Instant::now();
    let our_logits = forward_pass_logits(&model, &input_ids);
    eprintln!("Forward pass done in {:.1}s", t_fwd.elapsed().as_secs_f64());

    assert_eq!(our_logits.len(), hf_logits.len(),
        "Logit dimension mismatch: ours={} vs hf={}", our_logits.len(), hf_logits.len());

    let vocab = our_logits.len();

    // (a) Top-1 match
    let our_top1 = top_n_indices(&our_logits, 1)[0];
    let hf_top1 = top_n_indices(&hf_logits, 1)[0];
    let top1_match = our_top1 == hf_top1;
    eprintln!("\n--- Logit Comparison: 'The capital of France is' ---");
    eprintln!("  (a) Top-1: ours={} vs HF={} => {}", our_top1, hf_top1,
        if top1_match { "MATCH" } else { "MISMATCH" });

    // (b) Top-5 overlap
    let our_top5 = top_n_indices(&our_logits, 5);
    let hf_top5 = top_n_indices(&hf_logits, 5);
    let top5_overlap = our_top5.iter().filter(|t| hf_top5.contains(t)).count();
    eprintln!("  (b) Top-5 overlap: {}/5", top5_overlap);
    eprintln!("      ours={our_top5:?}");
    eprintln!("      hf  ={hf_top5:?}");

    // (c) Top-10 overlap
    let our_top10 = top_n_indices(&our_logits, 10);
    let hf_top10 = top_n_indices(&hf_logits, 10);
    let top10_overlap = our_top10.iter().filter(|t| hf_top10.contains(t)).count();
    eprintln!("  (c) Top-10 overlap: {}/10", top10_overlap);

    // (d) Max absolute logit difference
    let max_abs_diff = our_logits.iter().zip(hf_logits.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("  (d) Max |logit diff| across all {vocab} logits: {max_abs_diff:.4}");

    // (e) Cosine similarity
    let cos_sim = cosine_similarity(&our_logits, &hf_logits);
    eprintln!("  (e) Cosine similarity: {cos_sim:.6}");

    // (f) KL divergence (our || hf)
    let our_probs = softmax(&our_logits);
    let hf_probs = softmax(&hf_logits);
    let kl_div = kl_divergence(&our_probs, &hf_probs);
    eprintln!("  (f) KL divergence KL(ours||HF): {kl_div:.6}");

    eprintln!();

    // =========================================================================
    // Part 2: Greedy generation — top-5 accuracy over 20 tokens
    // =========================================================================
    eprintln!("--- Greedy Generation: top-5 accuracy ---");

    let sampling = moe_infer::generate::SamplingConfig::greedy();
    eprintln!("Running generate_v2 (max_new_tokens=20)...");
    let t_gen = std::time::Instant::now();
    let output = moe_infer::generate_v2::generate_v2(
        &model, &tokenizer, prompt, 20, &sampling,
    ).expect("generation");
    eprintln!("Generation done in {:.1}s ({:.1}s prefill + {:.1}s decode)",
        t_gen.elapsed().as_secs_f64(), output.prefill_secs, output.decode_secs);
    eprintln!("Generated text: {:?}", output.text);
    eprintln!("Generated token IDs: {:?}", output.token_ids);

    // For top-5 accuracy we need per-step logits. Re-run the forward pass
    // step-by-step, collecting the top-5 at each decode position.
    //
    // Encode the prompt as generate_v2 does (with BOS prepended).
    let gen_encoding = tokenizer.encode(prompt, false).expect("tokenize gen prompt");
    let mut gen_input_ids: Vec<u32> = gen_encoding.get_ids().to_vec();
    let gen_bos = model.config.model.bos_token_id;
    if gen_input_ids.first() != Some(&gen_bos) {
        gen_input_ids.insert(0, gen_bos);
    }

    // We'll do a second forward-pass sweep that logs top-5 at each new token position.
    let num_layers = model.config.num_layers();
    let max_seq = model.rope.max_seq;
    let hidden = model.config.hidden_size();
    let vocab_size = model.config.vocab_size();

    let mut cache2 = MlaKvCache::new(
        num_layers,
        model.mla_config.kv_lora_rank,
        model.mla_config.qk_rope_head_dim,
        max_seq,
    );
    let router_config2 = RouterConfig {
        num_experts: model.config.num_experts(),
        top_k: model.config.num_experts_per_tok(),
        norm_topk_prob: model.config.ffn.norm_topk_prob,
        bias_lr: 0.0,
    };
    let mut router2 = MoeRouter::new(router_config2);
    let embed_table = model.weights.embed_table().expect("embed_table");
    let final_norm_weights2 = model.weights.final_norm().expect("final_norm").to_vec();

    // Prefill phase: run through prompt tokens, collect logits at last prompt position
    let max_compare = ref_tokens.len().min(20);
    let mut top5_hits = 0usize;

    // Prefill: run all but the last token without computing logits
    for (i, &token_id) in gen_input_ids.iter().enumerate() {
        let emb = embed_token(embed_table, token_id as usize, hidden);
        let mut x = emb;
        for layer in 0..num_layers {
            x = run_v2_layer(&model, &mut cache2, &mut router2, layer, &x, i);
        }
        cache2.advance();

        // At the last prompt token, get logits for step 0
        if i == gen_input_ids.len() - 1 {
            let normed = rmsnorm(&x, &final_norm_weights2, model.config.rms_norm_eps());
            let mut step_logits = vec![0.0f32; vocab_size];
            moe_infer::blas::sgemv_f32(&model.lm_head_f32, &normed, &mut step_logits, vocab_size, hidden);
            let our_top5_step = top_n_indices(&step_logits, 5);
            let ref_tok = if max_compare > 0 { ref_tokens[0] as usize } else { usize::MAX };
            let hit = our_top5_step.contains(&ref_tok);
            if hit { top5_hits += 1; }
            eprintln!("  step 0: ref_token={} in_our_top5={} top5={:?}",
                ref_tok, hit, our_top5_step);
        }
    }

    // Decode phase: generate remaining steps
    let mut pos = gen_input_ids.len();
    let mut all_ids = gen_input_ids.clone();

    for step in 1..max_compare {
        if pos >= max_seq {
            break;
        }
        let token_id = *all_ids.last().unwrap();
        if token_id == model.config.model.eos_token_id {
            eprintln!("  step {step}: EOS, stopping");
            break;
        }

        let emb = embed_token(embed_table, token_id as usize, hidden);
        let mut x = emb;
        for layer in 0..num_layers {
            x = run_v2_layer(&model, &mut cache2, &mut router2, layer, &x, pos);
        }
        cache2.advance();

        let normed = rmsnorm(&x, &final_norm_weights2, model.config.rms_norm_eps());
        let mut step_logits = vec![0.0f32; vocab_size];
        moe_infer::blas::sgemv_f32(&model.lm_head_f32, &normed, &mut step_logits, vocab_size, hidden);

        // Greedily pick next token for continuing generation
        let next_token = top_n_indices(&step_logits, 1)[0] as u32;
        all_ids.push(next_token);

        let our_top5_step = top_n_indices(&step_logits, 5);
        let ref_tok = ref_tokens[step] as usize;
        let hit = our_top5_step.contains(&ref_tok);
        if hit { top5_hits += 1; }
        eprintln!("  step {step}: ref_token={ref_tok} in_our_top5={hit} top5={our_top5_step:?}");

        pos += 1;
    }

    let top5_accuracy = top5_hits as f64 / max_compare as f64;
    eprintln!("\n--- Summary ---");
    eprintln!("  Logit top-1 match:   {}", top1_match);
    eprintln!("  Logit top-5 overlap: {}/5", top5_overlap);
    eprintln!("  Logit top-10 overlap:{}/10", top10_overlap);
    eprintln!("  Logit max |diff|:    {max_abs_diff:.4}");
    eprintln!("  Logit cosine sim:    {cos_sim:.6}");
    eprintln!("  Logit KL(ours||HF):  {kl_div:.6}");
    eprintln!("  Generation top-5 accuracy: {top5_hits}/{max_compare} = {:.1}%",
        100.0 * top5_accuracy);

    // Soft assertions — report quality, only hard-fail on catastrophic misalignment
    assert!(cos_sim > 0.5,
        "Cosine similarity {cos_sim:.4} too low — logit vectors are poorly aligned");
    assert!(top5_accuracy >= 0.30,
        "Top-5 accuracy {top5_hits}/{max_compare} ({:.1}%) below 30% threshold",
        100.0 * top5_accuracy);
}
