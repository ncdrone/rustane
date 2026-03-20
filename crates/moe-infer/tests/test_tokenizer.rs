//! Tokenizer integration tests.

use tokenizers::Tokenizer;

const TOKENIZER_PATH: &str = "weights/qwen3-30b-a3b/tokenizer.json";

fn ws_root() -> std::path::PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    std::path::Path::new(&manifest).parent().unwrap().parent().unwrap().to_path_buf()
}

#[test]
#[ignore = "requires downloaded model"]
fn test_tokenizer_roundtrip() {
    let path = ws_root().join(TOKENIZER_PATH);
    let tok = Tokenizer::from_file(&path).expect("load tokenizer");

    let texts = [
        "Hello, world!",
        "The capital of France is",
        "def fibonacci(n):",
        "fn main() { println!(\"hello\"); }",
    ];

    for text in &texts {
        let encoding = tok.encode(*text, false).expect("encode");
        let ids = encoding.get_ids();
        assert!(!ids.is_empty(), "should produce tokens for: {text}");

        let decoded = tok.decode(ids, true).expect("decode");
        assert_eq!(&decoded, text, "roundtrip failed for: {text}");
    }
}

#[test]
#[ignore = "requires downloaded model"]
fn test_special_tokens() {
    let path = ws_root().join(TOKENIZER_PATH);
    let tok = Tokenizer::from_file(&path).expect("load tokenizer");

    // BOS = 151643, EOS = 151645
    let eos_text = tok.decode(&[151645], true).expect("decode EOS");
    println!("EOS decodes to: '{eos_text}'");

    let bos_text = tok.decode(&[151643], true).expect("decode BOS");
    println!("BOS decodes to: '{bos_text}'");
}
