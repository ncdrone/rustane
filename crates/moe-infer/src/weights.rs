//! Zero-copy mmap backbone weight loader.
//!
//! Reads backbone.bin (flat binary) + backbone_index.json (tensor locations).
//! All weight slices are zero-copy views into the mmap'd file.

use anyhow::{Context, Result, bail};
use half::f16;
use memmap2::Mmap;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Tensor descriptor from backbone_index.json.
#[derive(Clone, Debug, Deserialize)]
pub struct TensorInfo {
    pub offset: usize,
    /// "f16" or "f32"
    pub dtype: String,
    pub shape: Vec<usize>,
}

impl TensorInfo {
    /// Compute byte size from shape and dtype.
    pub fn size_bytes(&self) -> usize {
        let elems: usize = self.shape.iter().product();
        let elem_size = match self.dtype.as_str() {
            "f16" => 2,
            "f32" => 4,
            _ => panic!("unsupported dtype: {}", self.dtype),
        };
        elems * elem_size
    }
}

/// Memory-mapped backbone weights.
pub struct BackboneWeights {
    mmap: Mmap,
    index: HashMap<String, TensorInfo>,
    /// Mmap'd expert files per layer (only for MoE layers).
    expert_mmaps: HashMap<usize, Mmap>,
}

/// Weights for a single GQA transformer layer (zero-copy slices).
/// Used by Qwen3-MoE (decoder_sparse_step=1, all layers MoE).
pub struct LayerWeights<'a> {
    pub input_norm: &'a [f32],
    pub post_attn_norm: &'a [f32],
    pub q_proj: &'a [f16],
    pub k_proj: &'a [f16],
    pub v_proj: &'a [f16],
    pub o_proj: &'a [f16],
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
    // MoE router (all layers)
    pub router: &'a [f16],
}

/// Weights for a single MLA transformer layer (zero-copy slices).
/// Used by DeepSeek-V2/V3.
pub struct MlaLayerWeights<'a> {
    pub input_norm: &'a [f32],
    pub post_attn_norm: &'a [f32],
    // MLA attention
    pub q_proj: &'a [f16],
    pub kv_a_proj: &'a [f16],       // kv_a_proj_with_mqa
    pub kv_a_layernorm: &'a [f32],
    pub w_uk: &'a [f16],            // pre-split from kv_b_proj
    pub w_uv: &'a [f16],            // pre-split from kv_b_proj
    pub o_proj: &'a [f16],
    // V3 Q LoRA (None for V2-Lite)
    pub q_a_proj: Option<&'a [f16]>,
    pub q_a_layernorm: Option<&'a [f32]>,
    pub q_b_proj: Option<&'a [f16]>,
    // FFN: either dense or MoE
    pub router: Option<&'a [f16]>,   // None for dense layers
    // e_score_correction_bias (V3 sigmoid routing, None for V2-Lite)
    pub e_score_correction_bias: Option<&'a [f32]>,
    // Shared expert weights (for MoE layers)
    pub shared_gate_proj: Option<&'a [f16]>,
    pub shared_up_proj: Option<&'a [f16]>,
    pub shared_down_proj: Option<&'a [f16]>,
    // Dense FFN weights (for dense layers, layer 0)
    pub dense_gate_proj: Option<&'a [f16]>,
    pub dense_up_proj: Option<&'a [f16]>,
    pub dense_down_proj: Option<&'a [f16]>,
}

impl BackboneWeights {
    /// Load backbone weights from a directory containing backbone.bin and backbone_index.json.
    pub fn load(dir: &Path) -> Result<Self> {
        Self::load_with_layers(dir, 48)
    }

    /// Load backbone weights, scanning up to `max_layers` for expert files.
    pub fn load_with_layers(dir: &Path, max_layers: usize) -> Result<Self> {
        let index_path = dir.join("backbone_index.json");
        let bin_path = dir.join("backbone.bin");

        let index_content = fs::read_to_string(&index_path)
            .with_context(|| format!("reading {}", index_path.display()))?;
        let index: HashMap<String, TensorInfo> = serde_json::from_str(&index_content)
            .with_context(|| format!("parsing {}", index_path.display()))?;

        let file = fs::File::open(&bin_path)
            .with_context(|| format!("opening {}", bin_path.display()))?;
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("mmap {}", bin_path.display()))?;

        // Pre-fault backbone pages into page cache (async kernel readahead).
        #[cfg(target_os = "macos")]
        unsafe {
            libc::madvise(mmap.as_ptr() as *mut _, mmap.len(), libc::MADV_WILLNEED);
        }

        // Try to load expert files
        let mut expert_mmaps = HashMap::new();
        for layer in 0..max_layers {
            let expert_path = dir.join(format!("layer_{layer:02}_experts.bin"));
            if expert_path.exists() {
                let f = fs::File::open(&expert_path)
                    .with_context(|| format!("opening {}", expert_path.display()))?;
                let m = unsafe { Mmap::map(&f) }
                    .with_context(|| format!("mmap {}", expert_path.display()))?;
                expert_mmaps.insert(layer, m);
            }
        }

        Ok(Self { mmap, index, expert_mmaps })
    }

    /// Get a tensor's raw bytes by name.
    fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let info = self.index.get(name)
            .with_context(|| format!("tensor not found: {name}"))?;
        let size = info.size_bytes();
        if info.offset + size > self.mmap.len() {
            bail!("tensor {name} out of bounds: offset={} size={size} file_len={}",
                info.offset, self.mmap.len());
        }
        Ok(&self.mmap[info.offset..info.offset + size])
    }

    /// Get tensor as f16 slice.
    fn tensor_f16(&self, name: &str) -> Result<&[f16]> {
        let bytes = self.tensor_bytes(name)?;
        let info = &self.index[name];
        assert_eq!(info.dtype, "f16", "expected f16 for {name}");
        Ok(unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const f16,
                bytes.len() / 2,
            )
        })
    }

    /// Get tensor as f32 slice.
    fn tensor_f32(&self, name: &str) -> Result<&[f32]> {
        let bytes = self.tensor_bytes(name)?;
        let info = &self.index[name];
        assert_eq!(info.dtype, "f32", "expected f32 for {name}");
        Ok(unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const f32,
                bytes.len() / 4,
            )
        })
    }

    /// Get embedding vector for a token ID (parameterized by hidden_size).
    pub fn embedding_with_hidden(&self, token_id: usize, hidden: usize) -> Result<&[f16]> {
        let emb = self.tensor_f16("model.embed_tokens.weight")?;
        let start = token_id * hidden;
        if start + hidden > emb.len() {
            bail!("token_id {token_id} out of range");
        }
        Ok(&emb[start..start + hidden])
    }

    /// Get embedding vector for a token ID (assumes hidden=2048).
    pub fn embedding(&self, token_id: usize) -> Result<&[f16]> {
        self.embedding_with_hidden(token_id, 2048)
    }

    /// Get full embedding table.
    pub fn embed_table(&self) -> Result<&[f16]> {
        self.tensor_f16("model.embed_tokens.weight")
    }

    /// Get final layer norm weights.
    pub fn final_norm(&self) -> Result<&[f32]> {
        self.tensor_f32("model.norm.weight")
    }

    /// Get LM head weights.
    pub fn lm_head(&self) -> Result<&[f16]> {
        self.tensor_f16("lm_head.weight")
    }

    /// Get GQA layer weights (Qwen3 path).
    pub fn layer_weights(&self, layer: usize) -> Result<LayerWeights<'_>> {
        let prefix = format!("model.layers.{layer}");

        let input_norm = self.tensor_f32(&format!("{prefix}.input_layernorm.weight"))?;
        let post_attn_norm = self.tensor_f32(&format!("{prefix}.post_attention_layernorm.weight"))?;
        let q_proj = self.tensor_f16(&format!("{prefix}.self_attn.q_proj.weight"))?;
        let k_proj = self.tensor_f16(&format!("{prefix}.self_attn.k_proj.weight"))?;
        let v_proj = self.tensor_f16(&format!("{prefix}.self_attn.v_proj.weight"))?;
        let o_proj = self.tensor_f16(&format!("{prefix}.self_attn.o_proj.weight"))?;
        let q_norm = self.tensor_f32(&format!("{prefix}.self_attn.q_norm.weight"))?;
        let k_norm = self.tensor_f32(&format!("{prefix}.self_attn.k_norm.weight"))?;

        // All layers are MoE — router gate weight
        let router = self.tensor_f16(&format!("{prefix}.mlp.gate.weight"))?;

        Ok(LayerWeights {
            input_norm,
            post_attn_norm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            router,
        })
    }

    /// Get MLA layer weights (DeepSeek-V2/V3 path).
    pub fn mla_layer_weights(&self, layer: usize, is_moe: bool) -> Result<MlaLayerWeights<'_>> {
        let prefix = format!("model.layers.{layer}");

        let input_norm = self.tensor_f32(&format!("{prefix}.input_layernorm.weight"))?;
        let post_attn_norm = self.tensor_f32(&format!("{prefix}.post_attention_layernorm.weight"))?;

        // MLA attention — try Q LoRA first (V3), fall back to direct q_proj (V2-Lite)
        let q_a_proj = self.tensor_f16(&format!("{prefix}.self_attn.q_a_proj.weight")).ok();
        let q_a_layernorm = self.tensor_f32(&format!("{prefix}.self_attn.q_a_layernorm.weight")).ok();
        let q_b_proj = self.tensor_f16(&format!("{prefix}.self_attn.q_b_proj.weight")).ok();
        let q_proj = self.tensor_f16(&format!("{prefix}.self_attn.q_proj.weight"))
            .unwrap_or_else(|_| &[]); // empty if V3 Q LoRA path

        let kv_a_proj = self.tensor_f16(&format!("{prefix}.self_attn.kv_a_proj_with_mqa.weight"))?;
        let kv_a_layernorm = self.tensor_f32(&format!("{prefix}.self_attn.kv_a_layernorm.weight"))?;
        let w_uk = self.tensor_f16(&format!("{prefix}.self_attn.w_uk"))?;
        let w_uv = self.tensor_f16(&format!("{prefix}.self_attn.w_uv"))?;
        let o_proj = self.tensor_f16(&format!("{prefix}.self_attn.o_proj.weight"))?;

        // Validate critical tensor shapes (catches V3/K2 head count mismatches)
        if let Some(info) = self.tensor_info(&format!("{prefix}.self_attn.w_uk")) {
            if info.shape.len() == 3 && info.shape[0] != 0 {
                let expected_heads = w_uk.len() / (info.shape[1] * info.shape[2]);
                if layer == 0 {
                    eprintln!("  w_uk shape: {:?} ({} heads)", info.shape, info.shape[0]);
                }
            }
        }
        if let Some(info) = self.tensor_info(&format!("{prefix}.self_attn.o_proj.weight")) {
            if layer == 0 {
                eprintln!("  o_proj shape: {:?}", info.shape);
            }
        }

        if is_moe {
            let router = self.tensor_f16(&format!("{prefix}.mlp.gate.weight"))?;
            let e_bias = self.tensor_f32(&format!("{prefix}.mlp.gate.e_score_correction_bias")).ok();
            let shared_gate = self.tensor_f16(&format!("{prefix}.mlp.shared_experts.gate_proj.weight")).ok();
            let shared_up = self.tensor_f16(&format!("{prefix}.mlp.shared_experts.up_proj.weight")).ok();
            let shared_down = self.tensor_f16(&format!("{prefix}.mlp.shared_experts.down_proj.weight")).ok();

            Ok(MlaLayerWeights {
                input_norm, post_attn_norm,
                q_proj, kv_a_proj, kv_a_layernorm, w_uk, w_uv, o_proj,
                q_a_proj, q_a_layernorm, q_b_proj,
                router: Some(router),
                e_score_correction_bias: e_bias,
                shared_gate_proj: shared_gate,
                shared_up_proj: shared_up,
                shared_down_proj: shared_down,
                dense_gate_proj: None,
                dense_up_proj: None,
                dense_down_proj: None,
            })
        } else {
            // Dense FFN layer
            let gate = self.tensor_f16(&format!("{prefix}.mlp.gate_proj.weight"))?;
            let up = self.tensor_f16(&format!("{prefix}.mlp.up_proj.weight"))?;
            let down = self.tensor_f16(&format!("{prefix}.mlp.down_proj.weight"))?;

            Ok(MlaLayerWeights {
                input_norm, post_attn_norm,
                q_proj, kv_a_proj, kv_a_layernorm, w_uk, w_uv, o_proj,
                q_a_proj, q_a_layernorm, q_b_proj,
                router: None,
                e_score_correction_bias: None,
                shared_gate_proj: None,
                shared_up_proj: None,
                shared_down_proj: None,
                dense_gate_proj: Some(gate),
                dense_up_proj: Some(up),
                dense_down_proj: Some(down),
            })
        }
    }

    /// Get expert file mmap for a layer (for quantized expert dispatch).
    pub fn expert_mmap(&self, layer: usize) -> Option<&Mmap> {
        self.expert_mmaps.get(&layer)
    }

    /// Check if a tensor exists in the index.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// Get tensor info (for debugging/verification).
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name)
    }

    /// List all tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.index.keys().map(|s| s.as_str()).collect()
    }
}
