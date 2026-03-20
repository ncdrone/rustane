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

/// Weights for a single transformer layer (zero-copy slices).
/// All 48 layers are MoE in Qwen3-MoE-30B (decoder_sparse_step=1).
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

impl BackboneWeights {
    /// Load backbone weights from a directory containing backbone.bin and backbone_index.json.
    pub fn load(dir: &Path) -> Result<Self> {
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

        // Try to load expert files
        let mut expert_mmaps = HashMap::new();
        for layer in 0..48 {
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

    /// Get embedding vector for a token ID.
    pub fn embedding(&self, token_id: usize) -> Result<&[f16]> {
        let emb = self.tensor_f16("model.embed_tokens.weight")?;
        let hidden = 2048; // TODO: derive from index shape
        let start = token_id * hidden;
        if start + hidden > emb.len() {
            bail!("token_id {token_id} out of range");
        }
        Ok(&emb[start..start + hidden])
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

    /// Get weights for a specific layer.
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
