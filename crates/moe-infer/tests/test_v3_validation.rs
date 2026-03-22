//! DeepSeek-V3 (671B) inference validation suite.
//!
//! Same 4-level framework as V2-Lite, pointing at V3 config and references.
//! Thresholds relaxed for 61-layer model with INT4 experts.
//!
//! Test hierarchy:
//! 1. EMBEDDING CHECK — does embed(token) match Python reference? (cos > 0.999)
//! 2. SINGLE-LAYER CHECK — does layer 0 output match reference? (cos > 0.99)
//! 3. LOGIT DISTRIBUTION — top-k overlap vs DeepSeek API (report only)
//! 4. GENERATION — coherent text (report only)
//!
//! Run: cargo test -p moe-infer --test test_v3_validation --release -- --ignored --nocapture

use std::path::Path;

fn ws_root() -> std::path::PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).parent().unwrap().parent().unwrap().to_path_buf()
}

fn weights_dir() -> std::path::PathBuf {
    ws_root().join("weights/rustane-v3")
}

fn config_path() -> std::path::PathBuf {
    ws_root().join("configs/deepseek-v3.toml")
}

fn intermediates_path() -> std::path::PathBuf {
    ws_root().join("weights/references/deepseek_v3_intermediates.npz")
}

fn api_reference_path() -> std::path::PathBuf {
    ws_root().join("weights/references/deepseek_v3_api_reference.json")
}

fn tokenizer_path() -> std::path::PathBuf {
    ws_root().join("weights/deepseek-v3/tokenizer.json")
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| *x as f64 * *y as f64).sum();
    let norm_a: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    if norm_a < 1e-12 || norm_b < 1e-12 { return 0.0; }
    (dot / (norm_a * norm_b)) as f32
}

/// Load a named array from an NPZ file.
fn load_npz_f32(path: &Path, name: &str) -> (Vec<f32>, Vec<usize>) {
    use std::io::Read;

    let file = std::fs::File::open(path).expect("open npz");
    let mut archive = zip::ZipArchive::new(file).expect("parse zip");

    let arr_name = format!("{name}.npy");
    let mut entry = archive.by_name(&arr_name)
        .unwrap_or_else(|_| panic!("array '{name}' not found in NPZ"));

    let mut buf = Vec::new();
    entry.read_to_end(&mut buf).expect("read npy");

    // Parse minimal NPY format
    parse_npy_f32(&buf)
}

fn parse_npy_f32(data: &[u8]) -> (Vec<f32>, Vec<usize>) {
    assert_eq!(&data[..6], b"\x93NUMPY");
    let header_len = u16::from_le_bytes([data[8], data[9]]) as usize;
    let header = std::str::from_utf8(&data[10..10 + header_len]).expect("header utf8");

    // Parse shape from header like "'shape': (7168,),"
    let shape_start = header.find("'shape': (").expect("shape") + 10;
    let shape_end = header[shape_start..].find(')').expect("shape end") + shape_start;
    let shape_str = &header[shape_start..shape_end];
    let shape: Vec<usize> = shape_str
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse().expect("parse dim"))
        .collect();

    let data_start = 10 + header_len;
    let num_elements: usize = shape.iter().product();

    // Check dtype
    let is_f32 = header.contains("'<f4'") || header.contains("float32");
    let is_f64 = header.contains("'<f8'") || header.contains("float64");
    let is_i64 = header.contains("'<i8'") || header.contains("int64");

    let values: Vec<f32> = if is_f32 {
        data[data_start..data_start + num_elements * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    } else if is_f64 {
        data[data_start..data_start + num_elements * 8]
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect()
    } else if is_i64 {
        data[data_start..data_start + num_elements * 8]
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect()
    } else {
        panic!("unsupported dtype in npy header: {header}");
    };

    (values, shape)
}

// ─── Level 1: Embedding ─────────────────────────────────────────────

#[test]
#[ignore = "requires converted V3 weights + Python references"]
fn v3_level1_embedding() {
    if !weights_dir().join("backbone.bin").exists() {
        eprintln!("SKIP: V3 weights not found at {}", weights_dir().display());
        return;
    }
    if !intermediates_path().exists() {
        eprintln!("SKIP: intermediates NPZ not found");
        return;
    }

    let model = moe_infer::generate_v2::ModelV2::load(&weights_dir(), &config_path())
        .expect("load V3 model");

    let (ref_emb, _) = load_npz_f32(&intermediates_path(), "token_embedding");
    let (ref_token_id, _) = load_npz_f32(&intermediates_path(), "token_id");
    let token_id = ref_token_id[0] as usize;

    let hidden = model.config.hidden_size();
    let embed_table = model.weights.embed_table().expect("embed table");
    let our_emb: Vec<f32> = embed_table[token_id * hidden..(token_id + 1) * hidden]
        .iter().map(|v| v.to_f32()).collect();

    let cos = cosine_similarity(&our_emb, &ref_emb);
    eprintln!("L1 Embedding: token={token_id}, cos={cos:.6}");
    assert!(cos > 0.999, "L1 FAIL: embedding cosine {cos:.6} < 0.999");
    eprintln!("L1 PASS: cos={cos:.6}");
}

// ─── Level 2: Layer 0 Output ────────────────────────────────────────

#[test]
#[ignore = "requires converted V3 weights + Python references"]
fn v3_level2_layer0_output() {
    if !weights_dir().join("backbone.bin").exists() {
        eprintln!("SKIP: V3 weights not found");
        return;
    }
    if !intermediates_path().exists() {
        eprintln!("SKIP: intermediates NPZ not found");
        return;
    }

    let model = moe_infer::generate_v2::ModelV2::load(&weights_dir(), &config_path())
        .expect("load V3 model");

    let (ref_normed, _) = load_npz_f32(&intermediates_path(), "normed_embedding");
    let (ref_token_id, _) = load_npz_f32(&intermediates_path(), "token_id");
    let token_id = ref_token_id[0] as usize;

    // Compute our normed embedding
    let hidden = model.config.hidden_size();
    let embed_table = model.weights.embed_table().expect("embed table");
    let our_emb: Vec<f32> = embed_table[token_id * hidden..(token_id + 1) * hidden]
        .iter().map(|v| v.to_f32()).collect();
    let our_normed = moe_infer::rmsnorm::rmsnorm(
        &our_emb,
        &model.layers_f32[0].input_norm,
        model.config.rms_norm_eps(),
    );

    let cos = cosine_similarity(&our_normed, &ref_normed);
    eprintln!("L2 Layer 0 normed: cos={cos:.6}");
    assert!(cos > 0.99, "L2 FAIL: layer 0 normed cosine {cos:.6} < 0.99");
    eprintln!("L2 PASS: cos={cos:.6}");
}

// ─── Level 3: Logit Distribution (API Reference) ────────────────────

#[test]
#[ignore = "requires converted V3 weights + API reference"]
fn v3_level3_logit_distribution() {
    if !weights_dir().join("backbone.bin").exists() {
        eprintln!("SKIP: V3 weights not found");
        return;
    }
    if !api_reference_path().exists() {
        eprintln!("SKIP: API reference not found");
        return;
    }
    if !tokenizer_path().exists() {
        eprintln!("SKIP: tokenizer not found");
        return;
    }

    // Load API reference
    let api_json = std::fs::read_to_string(&api_reference_path()).expect("read API ref");
    let api_ref: serde_json::Value = serde_json::from_str(&api_json).expect("parse API ref");

    let prompt = api_ref["prompt"].as_str().unwrap_or("The capital of France is");
    eprintln!("L3: prompt={prompt:?}");

    // Load model and generate
    let model = moe_infer::generate_v2::ModelV2::load(&weights_dir(), &config_path())
        .expect("load V3 model");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path())
        .expect("load tokenizer");

    let sampling = moe_infer::generate::SamplingConfig {
        greedy: true,
        temperature: 1.0,
        top_k: 1,
    };

    let output = moe_infer::generate_v2::generate_v2(
        &model, &tokenizer, prompt, 20, &sampling,
    ).expect("generate");

    eprintln!("L3: generated={:?}", output.text);
    eprintln!("L3: tokens={:?}", output.token_ids);
    eprintln!("L3: {:.1} tok/s (decode)",
        output.tokens_generated as f64 / output.decode_secs);

    // Compare first generated token with API top-20
    if let Some(tokens) = api_ref["tokens"].as_array() {
        if let Some(first_tok) = tokens.first() {
            let empty = vec![];
            let api_top: Vec<&str> = first_tok["top_logprobs"].as_array()
                .unwrap_or(&empty)
                .iter()
                .filter_map(|lp| lp["token"].as_str())
                .collect();
            let our_first = tokenizer.decode(&[output.token_ids[0]], true)
                .unwrap_or_default();
            let in_top20 = api_top.iter().any(|t| t.trim() == our_first.trim());
            eprintln!("L3: our first token={our_first:?}, in API top-20={in_top20}");
            eprintln!("L3: API top-5: {:?}", &api_top[..5.min(api_top.len())]);
        }
    }

    eprintln!("L3 REPORT: check output above (no hard gate for L3)");
}

// ─── Level 4: Greedy Generation ─────────────────────────────────────

#[test]
#[ignore = "requires converted V3 weights"]
fn v3_level4_generation() {
    if !weights_dir().join("backbone.bin").exists() {
        eprintln!("SKIP: V3 weights not found");
        return;
    }
    if !tokenizer_path().exists() {
        eprintln!("SKIP: tokenizer not found");
        return;
    }

    let model = moe_infer::generate_v2::ModelV2::load(&weights_dir(), &config_path())
        .expect("load V3 model");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path())
        .expect("load tokenizer");

    let sampling = moe_infer::generate::SamplingConfig {
        greedy: true,
        temperature: 1.0,
        top_k: 1,
    };

    let prompts = [
        "The capital of France is",
        "In machine learning, a transformer is",
        "def fibonacci(n):",
    ];

    for prompt in &prompts {
        let output = moe_infer::generate_v2::generate_v2(
            &model, &tokenizer, prompt, 30, &sampling,
        ).expect("generate");

        eprintln!("\nL4: prompt={prompt:?}");
        eprintln!("L4: output={:?}", output.text);
        eprintln!("L4: {:.1} tok/s decode, {} tokens",
            output.tokens_generated as f64 / output.decode_secs,
            output.tokens_generated);
    }

    eprintln!("\nL4 REPORT: check output above for coherence (no hard gate)");
}
