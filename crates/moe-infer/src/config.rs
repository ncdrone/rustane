//! TOML-based model configuration loading.

use std::fs;
use std::path::Path;

/// Top-level inference model configuration.
#[derive(Clone, Debug)]
pub struct InferConfig {
    pub model_name: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,

    // MoE params
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,

    // Quantization
    pub bits: usize,
    pub group_size: usize,

    // MLA (optional)
    pub kv_lora_rank: Option<usize>,
}

impl InferConfig {
    /// Load from a TOML file (simplified parser — just key=value lines).
    pub fn from_toml(path: &Path) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;

        let get = |key: &str| -> Option<String> {
            for line in content.lines() {
                let line = line.trim();
                if line.starts_with('#') || line.starts_with('[') { continue; }
                if let Some((k, v)) = line.split_once('=') {
                    if k.trim() == key {
                        // Strip inline comments and quotes
                        let v = v.split('#').next().unwrap_or(v).trim();
                        return Some(v.trim_matches('"').to_string());
                    }
                }
            }
            None
        };

        let get_usize = |key: &str, default: usize| -> usize {
            get(key).and_then(|v| v.parse().ok()).unwrap_or(default)
        };

        Ok(Self {
            model_name: get("name").unwrap_or_else(|| "unknown".to_string()),
            hidden_size: get_usize("hidden_size", 2048),
            intermediate_size: get_usize("intermediate_size", 6144),
            num_hidden_layers: get_usize("num_hidden_layers", 48),
            num_attention_heads: get_usize("num_attention_heads", 32),
            num_key_value_heads: get_usize("num_key_value_heads", 4),
            head_dim: get_usize("head_dim", 128),
            vocab_size: get_usize("vocab_size", 151936),
            max_position_embeddings: get_usize("max_position_embeddings", 40960),
            num_experts: get_usize("num_experts", 128),
            num_experts_per_tok: get_usize("num_experts_per_tok", 8),
            moe_intermediate_size: get_usize("moe_intermediate_size", 768),
            bits: get_usize("bits", 4),
            group_size: get_usize("group_size", 128),
            kv_lora_rank: get("kv_lora_rank").and_then(|v| v.parse().ok()),
        })
    }
}
