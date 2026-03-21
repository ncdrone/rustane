//! Per-layer decode timing breakdown.
//! Shows where the 71ms/token budget goes across 48 layers.

use std::path::Path;

fn ws_root() -> std::path::PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).parent().unwrap().parent().unwrap().to_path_buf()
}

#[test]
#[ignore = "benchmark: requires converted weights + tokenizer"]
fn bench_decode_layer_breakdown() {
    use moe_infer::generate::{Model, SamplingConfig, LayerTimings, run_layer, run_layer_timed};
    use moe_infer::kv_cache::KvCache;
    use moe_router::{MoeRouter, RouterConfig};
    use half::f16;

    let root = ws_root();
    let model = Model::load(
        &root.join("weights/rustane-qwen3"),
        &root.join("configs/qwen3-moe-30b.toml"),
    ).expect("load model");

    let tok = tokenizers::Tokenizer::from_file(root.join("weights/qwen3-30b-a3b/tokenizer.json"))
        .expect("load tokenizer");

    let hidden = model.config.hidden_size();
    let num_layers = model.config.num_layers();
    let max_seq = model.gqa_config.max_seq;

    let mut cache = KvCache::new(
        num_layers, model.config.num_kv_heads(),
        model.config.head_dim(), max_seq,
    );
    let router_config = RouterConfig {
        num_experts: model.config.num_experts(),
        top_k: model.config.num_experts_per_tok(),
        norm_topk_prob: model.config.ffn.norm_topk_prob,
        bias_lr: 0.0,
    };
    let mut router = MoeRouter::new(router_config);

    let embed_table = model.weights.embed_table().unwrap();

    // Prefill with a prompt to warm KV cache
    let prompt = "Explain what a mixture of experts model is in one paragraph.";
    let encoding = tok.encode(prompt, false).unwrap();
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();

    // Simple sequential prefill (not timed)
    for (i, &token_id) in input_ids.iter().enumerate() {
        let start = token_id as usize * hidden;
        let emb: Vec<f32> = embed_table[start..start + hidden].iter().map(|v| v.to_f32()).collect();
        let mut x = emb;
        for layer in 0..num_layers {
            x = run_layer(&model, &mut cache, &mut router, layer, &x, i).unwrap();
        }
    }

    // Decode with timing: generate 50 tokens
    let num_decode = 50;
    let mut pos = input_ids.len();

    // all_timings[token][layer]
    let mut all_timings: Vec<Vec<LayerTimings>> = Vec::new();
    let mut token_times_ms: Vec<f64> = Vec::new();

    // Start with a dummy "first token" from prefill
    let mut next_token = 151643u32; // BOS as placeholder, real generation would use the prefill output
    // Actually, let's use a simple generate to get the first token, then switch to timed
    let prefill_out = moe_infer::generate::generate(
        &model, &tok, prompt, 1, &SamplingConfig::greedy(),
    ).unwrap();
    // Reload everything for the timed decode
    let mut cache = KvCache::new(
        num_layers, model.config.num_kv_heads(),
        model.config.head_dim(), max_seq,
    );
    let mut router = MoeRouter::new(RouterConfig {
        num_experts: model.config.num_experts(),
        top_k: model.config.num_experts_per_tok(),
        norm_topk_prob: model.config.ffn.norm_topk_prob,
        bias_lr: 0.0,
    });

    // Re-prefill
    for (i, &token_id) in input_ids.iter().enumerate() {
        let start = token_id as usize * hidden;
        let emb: Vec<f32> = embed_table[start..start + hidden].iter().map(|v| v.to_f32()).collect();
        let mut x = emb;
        for layer in 0..num_layers {
            x = run_layer(&model, &mut cache, &mut router, layer, &x, i).unwrap();
        }
    }

    // Now do timed decode
    let first_token = prefill_out.token_ids[0];
    let mut current_token = first_token;
    pos = input_ids.len();

    for _step in 0..num_decode {
        let start = current_token as usize * hidden;
        let emb: Vec<f32> = embed_table[start..start + hidden].iter().map(|v| v.to_f32()).collect();
        let mut x = emb;

        let t0 = std::time::Instant::now();
        let mut layer_timings = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            let (out, lt) = run_layer_timed(&model, &mut cache, &mut router, layer, &x, pos).unwrap();
            x = out;
            layer_timings.push(lt);
        }
        let token_ms = t0.elapsed().as_secs_f64() * 1e3;
        token_times_ms.push(token_ms);
        all_timings.push(layer_timings);

        // Simple greedy: use LM head to pick next token
        let final_norm = model.weights.final_norm().unwrap();
        let normed = moe_infer::rmsnorm::rmsnorm(&x, &final_norm.to_vec(), model.config.rms_norm_eps());
        let vocab = model.config.vocab_size();
        let mut logits = vec![0.0f32; vocab];
        moe_infer::blas::sgemv_f32(&model.lm_head_f32, &normed, &mut logits, vocab, hidden);
        current_token = logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap().0 as u32;
        pos += 1;
    }

    // Aggregate: sum across 48 layers per token, then median across tokens
    let n = all_timings.len();
    let mut per_token_sums: Vec<LayerTimings> = Vec::new();
    for token_timings in &all_timings {
        let mut sum = LayerTimings::default();
        for lt in token_timings {
            sum.rmsnorm1_us += lt.rmsnorm1_us;
            sum.attn_proj_us += lt.attn_proj_us;
            sum.rmsnorm2_us += lt.rmsnorm2_us;
            sum.router_us += lt.router_us;
            sum.metal_dispatch_us += lt.metal_dispatch_us;
            sum.cpu_silu_us += lt.cpu_silu_us;
            sum.combine_us += lt.combine_us;
        }
        per_token_sums.push(sum);
    }

    // Sort each field and report p5/median/p95
    fn percentile(vals: &mut Vec<f64>, p: f64) -> f64 {
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((p / 100.0) * (vals.len() - 1) as f64).round() as usize;
        vals[idx.min(vals.len() - 1)]
    }

    let fields = [
        ("rmsnorm1", per_token_sums.iter().map(|s| s.rmsnorm1_us).collect::<Vec<_>>()),
        ("attn_total", per_token_sums.iter().map(|s| s.attn_proj_us).collect::<Vec<_>>()),
        ("rmsnorm2", per_token_sums.iter().map(|s| s.rmsnorm2_us).collect::<Vec<_>>()),
        ("router", per_token_sums.iter().map(|s| s.router_us).collect::<Vec<_>>()),
        ("metal_dispatch", per_token_sums.iter().map(|s| s.metal_dispatch_us).collect::<Vec<_>>()),
        ("cpu_silu", per_token_sums.iter().map(|s| s.cpu_silu_us).collect::<Vec<_>>()),
        ("combine", per_token_sums.iter().map(|s| s.combine_us).collect::<Vec<_>>()),
    ];

    println!("\n=== DECODE TIMING BREAKDOWN (summed across {} layers, {} tokens) ===", num_layers, n);
    println!("{:<16} {:>10} {:>10} {:>10}", "operation", "p5 (µs)", "median", "p95 (µs)");
    println!("{}", "-".repeat(50));

    let mut total_median = 0.0;
    for (name, vals) in &fields {
        let mut v = vals.clone();
        let p5 = percentile(&mut v, 5.0);
        let med = percentile(&mut v, 50.0);
        let p95 = percentile(&mut v, 95.0);
        total_median += med;
        println!("{:<16} {:>10.0} {:>10.0} {:>10.0}", name, p5, med, p95);
    }

    let mut tok_ms = token_times_ms.clone();
    let wall_median = percentile(&mut tok_ms, 50.0);
    println!("{}", "-".repeat(50));
    println!("{:<16} {:>10.0} {:>10.0}", "sum(median)", "", total_median);
    println!("{:<16} {:>10.1} ms", "wall median", wall_median);
    println!("{:<16} {:>10.1} tok/s", "decode rate", 1000.0 / wall_median);

    // Per-layer CSV for detailed analysis
    println!("\n=== PER-LAYER MEDIAN (µs) ===");
    println!("layer,rmsnorm1,attn,rmsnorm2,router,metal,silu,combine,total");
    for layer in 0..num_layers {
        let mut rn1: Vec<f64> = all_timings.iter().map(|t| t[layer].rmsnorm1_us).collect();
        let mut att: Vec<f64> = all_timings.iter().map(|t| t[layer].attn_proj_us).collect();
        let mut rn2: Vec<f64> = all_timings.iter().map(|t| t[layer].rmsnorm2_us).collect();
        let mut rtr: Vec<f64> = all_timings.iter().map(|t| t[layer].router_us).collect();
        let mut mtl: Vec<f64> = all_timings.iter().map(|t| t[layer].metal_dispatch_us).collect();
        let mut sil: Vec<f64> = all_timings.iter().map(|t| t[layer].cpu_silu_us).collect();
        let mut cmb: Vec<f64> = all_timings.iter().map(|t| t[layer].combine_us).collect();

        let m_rn1 = percentile(&mut rn1, 50.0);
        let m_att = percentile(&mut att, 50.0);
        let m_rn2 = percentile(&mut rn2, 50.0);
        let m_rtr = percentile(&mut rtr, 50.0);
        let m_mtl = percentile(&mut mtl, 50.0);
        let m_sil = percentile(&mut sil, 50.0);
        let m_cmb = percentile(&mut cmb, 50.0);
        let total = m_rn1 + m_att + m_rn2 + m_rtr + m_mtl + m_sil + m_cmb;

        println!("{},{:.0},{:.0},{:.0},{:.0},{:.0},{:.0},{:.0},{:.0}",
            layer, m_rn1, m_att, m_rn2, m_rtr, m_mtl, m_sil, m_cmb, total);
    }
    println!("=========================");
}
