//! Smoke test: ANE MLA on real K2 weights (2-token generation).
//!
//! What: Loads K2 model with ANE enabled, generates 2 tokens.
//! Not a benchmark — just verifies ANE dispatch doesn't crash on real weights.
//!
//! Invariant: ANE path produces coherent output (no crash, no garbage).
//! Failure means: IOSurface staging bug with real weight dimensions/values.

use std::path::Path;

fn ws_root() -> std::path::PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).parent().unwrap().parent().unwrap().to_path_buf()
}

#[test]
#[ignore = "requires K2 weights + ANE hardware"]
fn ane_k2_smoke_2_tokens() {
    let root = ws_root();
    let weights_dir = root.join("weights/rustane-k2");
    let config_path = root.join("configs/kimi-k2.toml");
    let tokenizer_path = root.join("weights/kimi-k2/tokenizer.json");

    if !weights_dir.join("backbone.bin").exists() {
        eprintln!("SKIP: K2 weights not found");
        return;
    }
    if !tokenizer_path.exists() {
        eprintln!("SKIP: tokenizer not found");
        return;
    }

    // Force ANE MLA on
    // Safety: single-threaded test, no other threads reading this var
    unsafe { std::env::set_var("RUSTANE_ANE_MLA", "1"); }

    eprintln!("Loading K2 with ANE MLA enabled...");
    let t = std::time::Instant::now();
    let model = moe_infer::generate_v2::ModelV2::load(&weights_dir, &config_path)
        .expect("load K2");
    eprintln!("Loaded in {:.1}s", t.elapsed().as_secs_f64());

    // Check ANE compiled
    assert!(model.ane_mla.is_some(), "ANE MLA should have compiled on M4 Max");
    eprintln!("ANE MLA kernels: compiled");
    if model.ane_shared_ffn.is_some() {
        eprintln!("ANE Shared FFN: compiled");
    }

    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .expect("tokenizer");

    let sampling = moe_infer::generate::SamplingConfig {
        greedy: true,
        temperature: 1.0,
        top_k: 1,
    };

    // Generate just 2 tokens — enough to verify MLA dispatch works
    eprintln!("Generating 2 tokens with ANE MLA...");
    let t = std::time::Instant::now();
    let output = moe_infer::generate_v2::generate_v2(
        &model, &tokenizer, "Hello", 2, &sampling,
    ).expect("generate with ANE");
    let elapsed = t.elapsed().as_secs_f64();

    eprintln!("Generated {} tokens in {:.1}s", output.tokens_generated, elapsed);
    eprintln!("Output: {:?}", output.text);
    eprintln!("Prefill: {:.2}s, Decode: {:.2}s", output.prefill_secs, output.decode_secs);

    // Verify we got tokens (not empty or crashed)
    assert!(output.tokens_generated >= 1, "Should generate at least 1 token");
    assert!(!output.text.is_empty(), "Output text should not be empty");
    eprintln!("ANE K2 smoke test PASSED");
}
