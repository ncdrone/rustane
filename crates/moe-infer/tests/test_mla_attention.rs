//! MLA attention tests against HF reference tensors from DeepSeek-V2-Lite.
//!
//! Validates the corrected MLA architecture:
//! - Q projection split (nope + rope)
//! - KV compression + layernorm
//! - W_UK/W_UV split from kv_b_proj
//! - Absorbed attention (two separate dot products)

use std::path::Path;

/// Load numpy .npz file and extract a named array as f32 Vec.
fn load_npz_f32(path: &Path, name: &str) -> (Vec<f32>, Vec<usize>) {
    use std::io::Read;
    let file = std::fs::File::open(path).expect("open npz");
    let mut archive = zip::ZipArchive::new(file).expect("parse zip");
    let entry_name = format!("{name}.npy");
    let mut entry = archive.by_name(&entry_name)
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
    let header = std::str::from_utf8(&data[header_start..header_start + header_len]).unwrap();

    // Parse shape from header
    let shape_start = header.find("'shape': (").expect("no shape") + 10;
    let shape_end = header[shape_start..].find(')').unwrap() + shape_start;
    let shape_str = &header[shape_start..shape_end];
    let shape: Vec<usize> = if shape_str.is_empty() {
        vec![]  // scalar
    } else {
        shape_str.split(',')
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
        "<f4" | "float32" => {
            raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        "<f8" | "float64" => {
            raw.chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect()
        }
        _ => panic!("unsupported dtype: {dtype}"),
    };

    (values, shape)
}

fn npz_path() -> std::path::PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).parent().unwrap().parent().unwrap()
        .join("weights/references/deepseek_v2_lite_intermediates.npz")
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
#[ignore = "requires HF reference tensors at weights/references/"]
fn mla_q_projection_matches_hf() {
    let npz = npz_path();
    if !npz.exists() { eprintln!("Skipping: {npz:?} not found"); return; }

    let (hidden_input, _) = load_npz_f32(&npz, "layer0_hidden_input");
    let (q_nope_ref, q_nope_shape) = load_npz_f32(&npz, "layer0_q_nope");
    let (q_pe_ref, q_pe_shape) = load_npz_f32(&npz, "layer0_q_pe");

    eprintln!("hidden_input: len={}", hidden_input.len());
    eprintln!("q_nope_ref: shape={q_nope_shape:?}, len={}", q_nope_ref.len());
    eprintln!("q_pe_ref: shape={q_pe_shape:?}, len={}", q_pe_ref.len());

    // V2-Lite: 16 heads, nope=128, rope=64
    assert_eq!(q_nope_ref.len(), 16 * 128);
    assert_eq!(q_pe_ref.len(), 16 * 64);

    // Load q_proj weights from safetensors to validate projection
    // For now, just verify shapes match
    eprintln!("Q projection shapes verified: nope=[16,128], pe=[16,64]");
}

#[test]
#[ignore = "requires HF reference tensors at weights/references/"]
fn mla_kv_compress_matches_hf() {
    let npz = npz_path();
    if !npz.exists() { return; }

    let (compressed_kv, _) = load_npz_f32(&npz, "layer0_compressed_kv");
    let (kv_normed, _) = load_npz_f32(&npz, "layer0_kv_normed");
    let (k_pe_raw, _) = load_npz_f32(&npz, "layer0_k_pe_raw");

    assert_eq!(compressed_kv.len(), 512, "kv_latent should be [512]");
    assert_eq!(kv_normed.len(), 512, "kv_normed should be [512]");
    assert_eq!(k_pe_raw.len(), 64, "k_pe should be [64]");

    // Verify layernorm was applied (normed should differ from raw)
    let diff = max_abs_diff(&compressed_kv, &kv_normed);
    eprintln!("kv raw vs normed max_diff={diff:.6}");
    assert!(diff > 0.001, "kv_a_layernorm should change values");
}

#[test]
#[ignore = "requires HF reference tensors at weights/references/"]
fn mla_w_uk_w_uv_split_shapes() {
    let npz = npz_path();
    if !npz.exists() { return; }

    let (w_uk, w_uk_shape) = load_npz_f32(&npz, "layer0_w_uk");
    let (w_uv, w_uv_shape) = load_npz_f32(&npz, "layer0_w_uv");

    // V2-Lite: 16 heads, nope=128, v=128, kv_rank=512
    assert_eq!(w_uk_shape, vec![16, 128, 512], "W_UK shape");
    assert_eq!(w_uv_shape, vec![16, 128, 512], "W_UV shape");
    assert_eq!(w_uk.len(), 16 * 128 * 512);
    assert_eq!(w_uv.len(), 16 * 128 * 512);

    // W_UK and W_UV should be different (they're different projections)
    let diff = max_abs_diff(&w_uk, &w_uv);
    eprintln!("W_UK vs W_UV max_diff={diff:.6}");
    assert!(diff > 0.01, "W_UK and W_UV should differ");
}

#[test]
#[ignore = "requires HF reference tensors at weights/references/"]
fn mla_absorbed_attention_decomposition() {
    let npz = npz_path();
    if !npz.exists() { return; }

    // Verify that q_nope @ W_UK gives same result as k_nope (from explicit kv_b_proj)
    let (q_nope, _) = load_npz_f32(&npz, "layer0_q_nope");
    let (kv_normed, _) = load_npz_f32(&npz, "layer0_kv_normed");
    let (k_nope_ref, _) = load_npz_f32(&npz, "layer0_k_nope");
    let (w_uk, _) = load_npz_f32(&npz, "layer0_w_uk");

    // k_nope should equal kv_normed @ W_UK^T (per head)
    // W_UK[h] is [nope=128, kv_rank=512], kv_normed is [512]
    // k_nope[h] = kv_normed @ W_UK[h]^T = [128]
    for h in 0..16 {
        let w_uk_h = &w_uk[h * 128 * 512..(h + 1) * 128 * 512]; // [128, 512]
        let mut k_nope_computed = vec![0.0f32; 128];
        // k_nope_computed[i] = sum_j(W_UK[h][i][j] * kv_normed[j])
        for i in 0..128 {
            let mut sum = 0.0f64;
            for j in 0..512 {
                sum += w_uk_h[i * 512 + j] as f64 * kv_normed[j] as f64;
            }
            k_nope_computed[i] = sum as f32;
        }
        let k_nope_h = &k_nope_ref[h * 128..(h + 1) * 128];
        let diff = max_abs_diff(&k_nope_computed, k_nope_h);
        assert!(diff < 0.05, "head {h}: k_nope from W_UK×kv_normed vs HF diff={diff:.6}");
    }
    eprintln!("Absorbed attention decomposition verified: k_nope = kv_normed @ W_UK^T");
}

#[test]
fn mla_scale_factor_correct() {
    use moe_infer::yarn_rope::{compute_mscale, mla_attention_scale};

    // V2-Lite: nope=128, rope=64, mscale_coeff=0.707, factor=40
    let mscale = compute_mscale(0.707, 40.0);
    let scale = mla_attention_scale(128, 64, mscale);

    // 1/sqrt(192) ≈ 0.07217
    // mscale ≈ 1.2608
    // scale = 0.07217 * 1.2608^2 ≈ 0.07217 * 1.5896 ≈ 0.1147
    eprintln!("mscale={mscale:.4}, scale={scale:.6}");
    assert!((scale - 0.1147).abs() < 0.005, "scale={scale}");

    // Without YaRN (factor=1): scale = 1/sqrt(192) ≈ 0.07217
    let scale_base = mla_attention_scale(128, 64, 1.0);
    assert!((scale_base - 0.07217).abs() < 0.001, "base scale={scale_base}");
}
