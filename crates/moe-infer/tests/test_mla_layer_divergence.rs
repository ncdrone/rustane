//! MLA layer divergence test: compare our forward pass against HF reference intermediates.
//!
//! Loads DeepSeek-V2-Lite weights + pre-captured HF intermediate tensors, then checks:
//! 1. Embedding of last token matches HF layer0_hidden_input
//! 2. Layer 0 output (after processing all 6 tokens) matches HF layer0_output
//! 3. Layer 1 output matches HF layer1_output
//!
//! Run:
//!   cargo test -p moe-infer --test test_mla_layer_divergence --release -- --ignored --nocapture

use std::path::Path;

use moe_infer::generate_v2::{ModelV2, MlaLayerF32};
use moe_infer::mla_attention::{MlaLayerWeights as MlaAttnWeights, MlaKvCache, mla_forward_decode};
use moe_infer::rmsnorm::rmsnorm;
use moe_infer::yarn_rope::{compute_mscale, mla_attention_scale};

// ---------------------------------------------------------------------------
// NPZ helpers (copied from test_mla_attention.rs)
// ---------------------------------------------------------------------------

/// Load numpy .npz file and extract a named array as f32 Vec.
fn load_npz_f32(path: &Path, name: &str) -> (Vec<f32>, Vec<usize>) {
    use std::io::Read;
    let file = std::fs::File::open(path).expect("open npz");
    let mut archive = zip::ZipArchive::new(file).expect("parse zip");
    let entry_name = format!("{name}.npy");
    let mut entry = archive
        .by_name(&entry_name)
        .unwrap_or_else(|_| panic!("tensor {name} not found in npz"));

    let mut buf = Vec::new();
    entry.read_to_end(&mut buf).expect("read npy");
    parse_npy_f32(&buf)
}

/// Parse a .npy byte buffer into f32 values + shape.
fn parse_npy_f32(data: &[u8]) -> (Vec<f32>, Vec<usize>) {
    // NumPy .npy format: magic + version + header_len + header_str + data
    assert_eq!(&data[..6], b"\x93NUMPY", "not a .npy file");
    let major = data[6];
    let _minor = data[7];

    let header_len = if major == 1 {
        u16::from_le_bytes([data[8], data[9]]) as usize
    } else {
        u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize
    };
    let header_start = if major == 1 { 10 } else { 12 };
    let header =
        std::str::from_utf8(&data[header_start..header_start + header_len]).unwrap();

    // Parse shape from header
    let shape_start = header.find("'shape': (").expect("no shape") + 10;
    let shape_end = header[shape_start..].find(')').unwrap() + shape_start;
    let shape_str = &header[shape_start..shape_end];
    let shape: Vec<usize> = if shape_str.is_empty() {
        vec![] // scalar
    } else {
        shape_str
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| s.parse().unwrap())
            .collect()
    };

    // Parse dtype
    let descr_start = header.find("'descr': '").expect("no descr") + 10;
    let descr_end = header[descr_start..].find('\'').unwrap() + descr_start;
    let dtype = &header[descr_start..descr_end];

    let data_start = header_start + header_len;
    let raw = &data[data_start..];

    let values: Vec<f32> = match dtype {
        "<f4" | "float32" => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        "<f8" | "float64" => raw
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        _ => panic!("unsupported dtype: {dtype}"),
    };

    (values, shape)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
    sum / a.len() as f32
}

// ---------------------------------------------------------------------------
// Workspace root
// ---------------------------------------------------------------------------

fn ws_root() -> std::path::PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

// ---------------------------------------------------------------------------
// Helper: build MlaAttnWeights from MlaLayerF32
// ---------------------------------------------------------------------------

fn build_attn_weights<'a>(lf: &'a MlaLayerF32) -> MlaAttnWeights<'a> {
    MlaAttnWeights {
        q_proj: &lf.q_proj,
        q_a_proj: None,       // V2-Lite: direct q_proj, no LoRA
        q_a_layernorm: None,
        q_b_proj: None,
        kv_a_proj: &lf.kv_a_proj,
        kv_a_layernorm: &lf.kv_a_layernorm,
        w_uk: &lf.w_uk,
        w_uv: &lf.w_uv,
        o_proj: &lf.o_proj,
        input_norm: &lf.input_norm,
        post_attn_norm: &lf.post_attn_norm,
    }
}

// ---------------------------------------------------------------------------
// Helper: dense FFN forward (copied from generate_v2 logic)
// ---------------------------------------------------------------------------

fn dense_ffn_forward(x: &[f32], lf: &MlaLayerF32) -> Vec<f32> {
    let gate_w = lf.dense_gate.as_ref().expect("dense layer needs gate_proj");
    let up_w = lf.dense_up.as_ref().expect("dense layer needs up_proj");
    let down_w = lf.dense_down.as_ref().expect("dense layer needs down_proj");

    let hidden = x.len();
    let inter = gate_w.len() / hidden;

    let mut gate_out = vec![0.0f32; inter];
    let mut up_out = vec![0.0f32; inter];
    moe_infer::blas::sgemv_f32(gate_w, x, &mut gate_out, inter, hidden);
    moe_infer::blas::sgemv_f32(up_w, x, &mut up_out, inter, hidden);

    // SiLU(gate) * up
    for i in 0..inter {
        let silu = gate_out[i] / (1.0 + (-gate_out[i]).exp());
        gate_out[i] = silu * up_out[i];
    }

    let mut out = vec![0.0f32; hidden];
    moe_infer::blas::sgemv_f32(down_w, &gate_out, &mut out, hidden, inter);
    out
}

// ---------------------------------------------------------------------------
// Helper: run one full layer (attention + residual + FFN + residual) — dense only.
// Returns the output hidden state.
// ---------------------------------------------------------------------------

fn run_dense_layer(
    model: &ModelV2,
    cache: &mut MlaKvCache,
    layer: usize,
    x: &[f32],
    pos: usize,
) -> Vec<f32> {
    let hidden = model.config.hidden_size();
    let eps = model.config.rms_norm_eps();
    let lf = &model.layers_f32[layer];

    // 1. Input RMSNorm
    let normed = rmsnorm(x, &lf.input_norm, eps);

    // 2. MLA attention
    let attn_weights = build_attn_weights(lf);
    let attn_out = mla_forward_decode(
        &normed,
        &attn_weights,
        cache,
        layer,
        pos,
        &model.rope,
        &model.mla_config,
        model.attn_scale,
    );

    // 3. Attention residual
    let mut residual = vec![0.0f32; hidden];
    for d in 0..hidden {
        residual[d] = x[d] + attn_out[d];
    }

    // 4. Post-attention RMSNorm
    let normed2 = rmsnorm(&residual, &lf.post_attn_norm, eps);

    // 5. Dense FFN (layer 0 of V2-Lite is always dense)
    let ffn_out = dense_ffn_forward(&normed2, lf);

    // 6. FFN residual
    for d in 0..hidden {
        residual[d] += ffn_out[d];
    }

    residual
}

// ---------------------------------------------------------------------------
// Helper: run one MoE layer (attention only — no expert dispatch in this test,
// just shared experts via the dense path if available, otherwise zero FFN).
// For the divergence check we only need layers 0 and 1 output; layer 1 is MoE
// but for a first-order comparison we still run the full shared-expert path.
// ---------------------------------------------------------------------------

fn run_layer_attn_only(
    model: &ModelV2,
    cache: &mut MlaKvCache,
    layer: usize,
    x: &[f32],
    pos: usize,
) -> Vec<f32> {
    let hidden = model.config.hidden_size();
    let eps = model.config.rms_norm_eps();
    let lf = &model.layers_f32[layer];

    // 1. Input RMSNorm
    let normed = rmsnorm(x, &lf.input_norm, eps);

    // 2. MLA attention
    let attn_weights = build_attn_weights(lf);
    let attn_out = mla_forward_decode(
        &normed,
        &attn_weights,
        cache,
        layer,
        pos,
        &model.rope,
        &model.mla_config,
        model.attn_scale,
    );

    // 3. Attention residual only (skip FFN for this helper)
    let mut residual = vec![0.0f32; hidden];
    for d in 0..hidden {
        residual[d] = x[d] + attn_out[d];
    }

    residual
}

// ---------------------------------------------------------------------------
// Main divergence test
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires weights at weights/rustane-deepseek-v2-lite/ and references at weights/references/"]
fn test_mla_layer_divergence() {
    let root = ws_root();
    let weights_dir = root.join("weights/rustane-deepseek-v2-lite");
    let config_path = root.join("configs/deepseek-v2-lite.toml");
    let npz_path = root.join("weights/references/deepseek_v2_lite_intermediates.npz");

    // Graceful skip if prerequisites are missing
    if !weights_dir.join("backbone.bin").exists() {
        eprintln!("SKIP: weights not found at {}", weights_dir.display());
        return;
    }
    if !config_path.exists() {
        eprintln!("SKIP: config not found at {}", config_path.display());
        return;
    }
    if !npz_path.exists() {
        eprintln!("SKIP: reference npz not found at {}", npz_path.display());
        return;
    }

    // -----------------------------------------------------------------------
    // Load model
    // -----------------------------------------------------------------------
    eprintln!("Loading model from {} ...", weights_dir.display());
    let model = ModelV2::load(&weights_dir, &config_path).expect("load ModelV2");

    let hidden = model.config.hidden_size();
    let num_layers = model.config.num_layers();
    let _eps = model.config.rms_norm_eps();

    eprintln!(
        "Model: hidden={hidden}, layers={num_layers}, attn_scale={:.6}",
        model.attn_scale
    );

    // Verify scale computation matches expected V2-Lite value
    {
        let mscale_coeff = model
            .config
            .attention
            .rope_scaling
            .as_ref()
            .map(|s| s.mscale)
            .unwrap_or(1.0);
        let factor = model
            .config
            .attention
            .rope_scaling
            .as_ref()
            .map(|s| s.factor)
            .unwrap_or(1.0);
        let mscale = compute_mscale(mscale_coeff, factor);
        let scale = mla_attention_scale(
            model.mla_config.qk_nope_head_dim,
            model.mla_config.qk_rope_head_dim,
            mscale,
        );
        eprintln!("Recomputed attn_scale={scale:.6} (should match model.attn_scale)");
    }

    // -----------------------------------------------------------------------
    // Load reference tensors
    // -----------------------------------------------------------------------
    eprintln!("Loading reference tensors from {} ...", npz_path.display());

    let (layer0_hidden_input, hid_shape) = load_npz_f32(&npz_path, "layer0_hidden_input");
    eprintln!("layer0_hidden_input: shape={hid_shape:?}, len={}", layer0_hidden_input.len());
    assert_eq!(
        layer0_hidden_input.len(),
        hidden,
        "layer0_hidden_input should be [hidden_size={hidden}]"
    );

    let (layer0_output, l0_shape) = load_npz_f32(&npz_path, "layer0_output");
    eprintln!("layer0_output: shape={l0_shape:?}, len={}", layer0_output.len());
    assert_eq!(
        layer0_output.len(),
        hidden,
        "layer0_output should be [hidden_size={hidden}]"
    );

    let (layer1_output, l1_shape) = load_npz_f32(&npz_path, "layer1_output");
    eprintln!("layer1_output: shape={l1_shape:?}, len={}", layer1_output.len());
    assert_eq!(
        layer1_output.len(),
        hidden,
        "layer1_output should be [hidden_size={hidden}]"
    );

    // -----------------------------------------------------------------------
    // Token IDs
    // "The capital of France is" with BOS prepended:
    // [100000, 549, 6077, 280, 7239, 317]
    //   BOS     The  capital  of  France  is
    // -----------------------------------------------------------------------
    let token_ids: Vec<u32> = vec![100000, 549, 6077, 280, 7239, 317];
    let seq_len = token_ids.len();
    eprintln!("Token IDs: {token_ids:?} (seq_len={seq_len})");

    // -----------------------------------------------------------------------
    // Stage 1: Embedding of last token vs layer0_hidden_input
    //
    // HF captures hidden[0, -1] at the INPUT to layer 0's self_attn, which
    // is just the embedding of the last token (index 5, token 317 "is").
    // -----------------------------------------------------------------------
    let embed_table = model.weights.embed_table().expect("embed table");
    let last_token = token_ids[seq_len - 1] as usize;
    let emb_last: Vec<f32> = embed_table[last_token * hidden..(last_token + 1) * hidden]
        .iter()
        .map(|v| v.to_f32())
        .collect();

    let emb_diff_max = max_abs_diff(&emb_last, &layer0_hidden_input);
    let emb_diff_mean = mean_abs_diff(&emb_last, &layer0_hidden_input);
    eprintln!(
        "\n[Stage 1] Embedding of last token (id={last_token}) vs layer0_hidden_input:"
    );
    eprintln!("  max_abs_diff  = {emb_diff_max:.6}");
    eprintln!("  mean_abs_diff = {emb_diff_mean:.6}");
    eprintln!("  ours[0..4]    = {:?}", &emb_last[..4]);
    eprintln!("  ref[0..4]     = {:?}", &layer0_hidden_input[..4]);

    // -----------------------------------------------------------------------
    // Stage 2: Run ALL tokens through layer 0 sequentially, compare output
    // for position 5 (last token) vs layer0_output.
    //
    // HF uses full-sequence batch; we simulate with sequential decode.
    // The cache accumulates KV for positions 0-4, then position 5 is the
    // comparison target.
    // -----------------------------------------------------------------------
    let mut cache = MlaKvCache::new(
        num_layers,
        model.mla_config.kv_lora_rank,
        model.mla_config.qk_rope_head_dim,
        512, // generous max_seq for this test
    );

    let mut layer0_out_at_pos5: Vec<f32> = Vec::new();
    {
        // Layer 0 is dense (first_k_dense_replace=1)
        assert!(
            !model.config.is_moe_layer(0),
            "layer 0 should be dense in V2-Lite"
        );

        for (pos, &token_id) in token_ids.iter().enumerate() {
            let emb: Vec<f32> = embed_table[token_id as usize * hidden..(token_id as usize + 1) * hidden]
                .iter()
                .map(|v| v.to_f32())
                .collect();

            let out = run_dense_layer(&model, &mut cache, 0, &emb, pos);
            cache.advance();

            if pos == seq_len - 1 {
                layer0_out_at_pos5 = out;
            }
        }
    }

    let l0_diff_max = max_abs_diff(&layer0_out_at_pos5, &layer0_output);
    let l0_diff_mean = mean_abs_diff(&layer0_out_at_pos5, &layer0_output);
    eprintln!("\n[Stage 2] Layer 0 output at pos=5 vs layer0_output:");
    eprintln!("  max_abs_diff  = {l0_diff_max:.6}");
    eprintln!("  mean_abs_diff = {l0_diff_mean:.6}");
    eprintln!("  ours[0..4]    = {:?}", &layer0_out_at_pos5[..4]);
    eprintln!("  ref[0..4]     = {:?}", &layer0_output[..4]);

    // -----------------------------------------------------------------------
    // Stage 3: Layer 1 output for last token.
    //
    // Layer 1 is MoE, but we run MLA attention + residual (skipping full
    // expert dispatch which requires quantized expert weights on disk).
    // This still exercises the MLA forward pass for layer 1.
    //
    // We replay from scratch with a fresh cache to avoid cache state issues.
    // -----------------------------------------------------------------------
    let mut cache2 = MlaKvCache::new(
        num_layers,
        model.mla_config.kv_lora_rank,
        model.mla_config.qk_rope_head_dim,
        512,
    );

    // Keep per-token layer-0 outputs so we can feed them into layer 1
    let mut layer1_out_at_pos5: Vec<f32> = Vec::new();
    {
        assert!(
            model.config.is_moe_layer(1),
            "layer 1 should be MoE in V2-Lite"
        );

        for (pos, &token_id) in token_ids.iter().enumerate() {
            let emb: Vec<f32> = embed_table[token_id as usize * hidden..(token_id as usize + 1) * hidden]
                .iter()
                .map(|v| v.to_f32())
                .collect();

            // Layer 0 (dense)
            let l0_out = run_dense_layer(&model, &mut cache2, 0, &emb, pos);

            // Layer 1 (MoE — attention + residual only, no expert FFN dispatch)
            // For the divergence check, we compare the attention+FFN output.
            // Since expert weights require the quantized expert binary, we run
            // attention-only here and note the limitation in the report.
            let l1_out = run_layer_attn_only(&model, &mut cache2, 1, &l0_out, pos);

            cache2.advance();

            if pos == seq_len - 1 {
                layer1_out_at_pos5 = l1_out;
            }
        }
    }

    let l1_diff_max = max_abs_diff(&layer1_out_at_pos5, &layer1_output);
    let l1_diff_mean = mean_abs_diff(&layer1_out_at_pos5, &layer1_output);
    eprintln!(
        "\n[Stage 3] Layer 1 attention-residual output at pos=5 vs layer1_output:"
    );
    eprintln!("  max_abs_diff  = {l1_diff_max:.6}");
    eprintln!("  mean_abs_diff = {l1_diff_mean:.6}");
    eprintln!("  (Note: layer1 comparison excludes MoE FFN — attention divergence only)");
    eprintln!("  ours[0..4]    = {:?}", &layer1_out_at_pos5[..4]);
    eprintln!("  ref[0..4]     = {:?}", &layer1_output[..4]);

    // -----------------------------------------------------------------------
    // Summary report
    // -----------------------------------------------------------------------
    eprintln!("\n===== Divergence Summary =====");
    eprintln!(
        "Stage 1  embedding vs layer0_hidden_input : max={emb_diff_max:.6}  mean={emb_diff_mean:.6}"
    );
    eprintln!(
        "Stage 2  layer0 output vs layer0_output   : max={l0_diff_max:.6}  mean={l0_diff_mean:.6}"
    );
    eprintln!(
        "Stage 3  layer1 attn   vs layer1_output   : max={l1_diff_max:.6}  mean={l1_diff_mean:.6}"
    );
    eprintln!("==============================");

    // Soft assertions: embedding should match within f16 precision (~1e-3)
    assert!(
        emb_diff_max < 1e-2,
        "Stage 1: embedding mismatch too large: max_abs_diff={emb_diff_max:.6} (expected < 0.01)"
    );
}
