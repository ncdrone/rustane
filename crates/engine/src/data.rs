//! Data loading: mmap'd uint16 token files + token_bytes for val_bpb.
//!
//! Binary format: flat array of uint16 little-endian tokens.
//! token_bytes.bin: flat array of int32 little-endian (byte count per token ID).

use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Memory-mapped token data.
pub struct TokenData {
    _mmap: memmap2::Mmap,
    n_tokens: usize,
}

impl TokenData {
    /// Open and mmap a uint16 binary token file.
    pub fn open(path: &Path) -> Self {
        let file = File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .unwrap_or_else(|e| panic!("mmap {}: {e}", path.display()));
        let n_tokens = mmap.len() / 2;
        Self {
            _mmap: mmap,
            n_tokens,
        }
    }

    /// Number of tokens in the file.
    pub fn len(&self) -> usize {
        self.n_tokens
    }

    /// Get token at position (uint16 → u32).
    pub fn token(&self, pos: usize) -> u32 {
        let bytes = &self._mmap[pos * 2..pos * 2 + 2];
        u16::from_le_bytes([bytes[0], bytes[1]]) as u32
    }

    /// Extract a slice of tokens as u32.
    pub fn tokens(&self, start: usize, len: usize) -> Vec<u32> {
        (start..start + len).map(|i| self.token(i)).collect()
    }
}

/// Token byte lengths for val_bpb computation.
pub struct TokenBytes {
    bytes: Vec<i32>, // [VOCAB] — byte length per token ID (0 for special tokens)
}

impl TokenBytes {
    /// Load token_bytes.bin (int32 little-endian array).
    pub fn load(path: &Path) -> Self {
        let mut file = File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
        let mut raw = Vec::new();
        file.read_to_end(&mut raw).unwrap();
        let n = raw.len() / 4;
        let bytes: Vec<i32> = (0..n)
            .map(|i| {
                i32::from_le_bytes([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]])
            })
            .collect();
        Self { bytes }
    }

    /// Byte length for a given token ID (0 for special tokens).
    pub fn byte_len(&self, token_id: u32) -> i32 {
        self.bytes[token_id as usize]
    }
}

/// Compute val_bpb (bits per byte) from per-token losses and token byte lengths.
/// `losses`: per-token cross-entropy losses (nats), `targets`: target token IDs.
/// Returns (val_bpb, total_nats, total_bytes).
pub fn compute_bpb(losses: &[f32], targets: &[u32], token_bytes: &TokenBytes) -> (f32, f32, usize) {
    let mut total_nats = 0.0f32;
    let mut total_bytes = 0usize;
    for (&loss, &tok) in losses.iter().zip(targets.iter()) {
        let bl = token_bytes.byte_len(tok);
        if bl > 0 {
            total_nats += loss;
            total_bytes += bl as usize;
        }
    }
    let bpb = if total_bytes > 0 {
        total_nats / (std::f32::consts::LN_2 * total_bytes as f32)
    } else {
        0.0
    };
    (bpb, total_nats, total_bytes)
}

/// Simple PRNG for position sampling (same LCG as Obj-C reference).
pub fn random_position(step: u64, micro: u64, max_pos: u64) -> usize {
    let seed = step
        .wrapping_mul(7919)
        .wrapping_add(micro.wrapping_mul(104729));
    (seed % max_pos) as usize
}
