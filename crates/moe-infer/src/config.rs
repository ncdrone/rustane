//! TOML-based model configuration with proper nested section parsing.

use serde::Deserialize;
use std::fs;
use std::path::Path;

/// Top-level inference config, matching the TOML structure.
#[derive(Clone, Debug, Deserialize)]
pub struct InferConfig {
    pub model: ModelSection,
    pub attention: AttentionSection,
    pub ffn: FfnSection,
    pub quantization: QuantSection,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ModelSection {
    pub name: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_bos")]
    pub bos_token_id: u32,
    #[serde(default = "default_eos")]
    pub eos_token_id: u32,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct AttentionSection {
    /// "gqa" or "mla"
    pub kind: String,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    /// MLA-only: KV LoRA rank
    #[serde(default)]
    pub kv_lora_rank: Option<usize>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct FfnSection {
    /// If true, all layers are MoE (decoder_sparse_step=1).
    #[serde(default)]
    pub all_moe: bool,
    /// Per-expert MoE intermediate size.
    pub moe_inter_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    /// Number of shared experts (0 = none, as in Qwen3-MoE-30B).
    #[serde(default)]
    pub shared_expert_count: usize,
    #[serde(default)]
    pub norm_topk_prob: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct QuantSection {
    pub bits: usize,
    pub group_size: usize,
}

fn default_max_pos() -> usize { 40960 }
fn default_bos() -> u32 { 151643 }
fn default_eos() -> u32 { 151645 }
fn default_rms_eps() -> f32 { 1e-6 }

impl InferConfig {
    /// Load from a TOML file with nested [model], [attention], [ffn], [quantization] sections.
    pub fn from_toml(path: &Path) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
        toml::from_str(&content)
            .map_err(|e| format!("Failed to parse {}: {e}", path.display()))
    }

    /// Whether a given layer is MoE (has experts).
    pub fn is_moe_layer(&self, _layer: usize) -> bool {
        // Qwen3-MoE-30B: all_moe=true, all 48 layers are MoE
        self.ffn.all_moe
    }

    // Convenience accessors that flatten the nested structure
    pub fn model_name(&self) -> &str { &self.model.name }
    pub fn hidden_size(&self) -> usize { self.model.hidden_size }
    pub fn num_layers(&self) -> usize { self.model.num_layers }
    pub fn vocab_size(&self) -> usize { self.model.vocab_size }
    pub fn num_experts(&self) -> usize { self.ffn.num_experts }
    pub fn num_experts_per_tok(&self) -> usize { self.ffn.num_experts_per_tok }
    pub fn moe_inter_size(&self) -> usize { self.ffn.moe_inter_size }
    pub fn head_dim(&self) -> usize { self.attention.head_dim }
    pub fn num_q_heads(&self) -> usize { self.attention.num_q_heads }
    pub fn num_kv_heads(&self) -> usize { self.attention.num_kv_heads }
    pub fn rope_theta(&self) -> f32 { self.attention.rope_theta }
    pub fn rms_norm_eps(&self) -> f32 { self.model.rms_norm_eps }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_qwen3_toml() {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let ws_root = Path::new(&manifest).parent().unwrap().parent().unwrap();
        let config = InferConfig::from_toml(&ws_root.join("configs/qwen3-moe-30b.toml"))
            .expect("parse config");
        assert_eq!(config.model.hidden_size, 2048);
        assert_eq!(config.model.num_layers, 48);
        assert_eq!(config.model.vocab_size, 151936);
        assert_eq!(config.attention.kind, "gqa");
        assert_eq!(config.attention.num_q_heads, 32);
        assert_eq!(config.attention.num_kv_heads, 4);
        assert_eq!(config.attention.head_dim, 128);
        assert_eq!(config.attention.rope_theta, 1_000_000.0);
        assert!(config.ffn.all_moe);
        assert_eq!(config.ffn.moe_inter_size, 768);
        assert_eq!(config.ffn.num_experts, 128);
        assert_eq!(config.ffn.num_experts_per_tok, 8);
        assert_eq!(config.ffn.shared_expert_count, 0);
        assert!(config.ffn.norm_topk_prob);
        assert!(config.is_moe_layer(0));
        assert!(config.is_moe_layer(47));
    }
}
