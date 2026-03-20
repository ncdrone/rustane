//! Weight conversion: HuggingFace safetensors → rustane expert weight files.
//!
//! Converts model shards into per-layer expert weight files for pread streaming.
//! Format: layer_XX.bin = N experts concatenated, each at expert_id * EXPERT_SIZE.
//!
//! Usage: cargo run -p expert-pager --bin convert -- --model-dir weights/qwen3-moe-30b --output-dir weights/converted

use std::fs;
use std::io::{self, Write};
use std::path::Path;

/// Metadata for a converted model.
#[derive(Clone, Debug)]
pub struct ConvertedModelLayout {
    pub num_layers: usize,
    pub num_experts: usize,
    pub expert_size: usize,
    /// Whether experts are stored as 4-bit packed.
    pub quantized: bool,
}

impl ConvertedModelLayout {
    /// Write metadata to a JSON file alongside the weight files.
    pub fn write_metadata(&self, output_dir: &Path) -> io::Result<()> {
        let json = format!(
            "{{\n  \"num_layers\": {},\n  \"num_experts\": {},\n  \"expert_size\": {},\n  \"quantized\": {}\n}}\n",
            self.num_layers, self.num_experts, self.expert_size, self.quantized
        );
        fs::write(output_dir.join("metadata.json"), json)
    }
}

/// Create a synthetic expert weight file for testing.
/// Each expert gets deterministic data based on (layer, expert_id).
pub fn create_synthetic_experts(
    output_dir: &Path,
    num_layers: usize,
    num_experts: usize,
    expert_size: usize,
) -> io::Result<ConvertedModelLayout> {
    fs::create_dir_all(output_dir)?;

    for layer in 0..num_layers {
        let path = output_dir.join(format!("layer_{layer:02}.bin"));
        let mut file = fs::File::create(&path)?;

        for expert_id in 0..num_experts {
            // Deterministic fill: seed from layer + expert_id
            let seed = (layer as u64) * 1000 + expert_id as u64;
            let data = deterministic_bytes(expert_size, seed);
            file.write_all(&data)?;
        }
        file.flush()?;
    }

    let layout = ConvertedModelLayout {
        num_layers,
        num_experts,
        expert_size,
        quantized: false,
    };
    layout.write_metadata(output_dir)?;

    Ok(layout)
}

/// Generate deterministic bytes for testing.
fn deterministic_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 56) as u8
        })
        .collect()
}
