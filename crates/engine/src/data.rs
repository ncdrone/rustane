//! Data loading for both Rustane flat token files and Parameter Golf FineWeb shards.
//!
//! Rustane binary format: flat array of uint16 little-endian tokens.
//! token_bytes.bin: flat array of int32 little-endian (byte count per token ID).
//! FineWeb shard format: 256 x int32 header followed by uint16 little-endian tokens.

use memmap2::Mmap;
use serde::Deserialize;
use std::fs::{self, File};
use std::io::{self, ErrorKind, Read};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Memory-mapped token data.
pub struct TokenData {
    _mmap: Mmap,
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

struct FineWebShard {
    _mmap: Mmap,
    n_tokens: usize,
}

impl FineWebShard {
    const HEADER_WORDS: usize = 256;
    const HEADER_BYTES: usize = Self::HEADER_WORDS * 4;
    const MAGIC: i32 = 20240520;
    const VERSION: i32 = 1;

    fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file) }?;
        let n_tokens = validate_fineweb_shard_bytes(path, &mmap)?;
        Ok(Self {
            _mmap: mmap,
            n_tokens,
        })
    }

    fn len(&self) -> usize {
        self.n_tokens
    }

    fn token(&self, pos: usize) -> u32 {
        let off = Self::HEADER_BYTES + pos * 2;
        let bytes = &self._mmap[off..off + 2];
        u16::from_le_bytes([bytes[0], bytes[1]]) as u32
    }
}

fn fineweb_shard_paths(dataset_dir: &Path, prefix: &str) -> io::Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = fs::read_dir(dataset_dir)?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            if name.starts_with(prefix) && name.ends_with(".bin") {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(io::Error::new(
            ErrorKind::NotFound,
            format!("no {prefix}*.bin shards found in {}", dataset_dir.display()),
        ));
    }
    Ok(files)
}

pub struct FineWebTrainStream {
    shards: Vec<FineWebShard>,
    shard_idx: usize,
    pos_in_shard: usize,
    total_tokens: usize,
    epochs_completed: u64,
}

impl FineWebTrainStream {
    pub fn open_dir(dataset_dir: &Path) -> io::Result<Self> {
        let paths = fineweb_shard_paths(dataset_dir, "fineweb_train_")?;
        let mut shards = Vec::with_capacity(paths.len());
        let mut total_tokens = 0usize;
        for path in paths {
            let shard = FineWebShard::open(&path)?;
            total_tokens += shard.len();
            shards.push(shard);
        }
        Ok(Self {
            shards,
            shard_idx: 0,
            pos_in_shard: 0,
            total_tokens,
            epochs_completed: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.total_tokens
    }

    pub fn epochs_completed(&self) -> u64 {
        self.epochs_completed
    }

    pub fn next_tokens(&mut self, n: usize) -> Vec<u32> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let shard = &self.shards[self.shard_idx];
            if self.pos_in_shard >= shard.len() {
                self.advance_shard();
                continue;
            }
            let available = shard.len() - self.pos_in_shard;
            let take = (n - out.len()).min(available);
            for i in 0..take {
                out.push(shard.token(self.pos_in_shard + i));
            }
            self.pos_in_shard += take;
        }
        out
    }

    fn advance_shard(&mut self) {
        self.shard_idx += 1;
        self.pos_in_shard = 0;
        if self.shard_idx >= self.shards.len() {
            self.shard_idx = 0;
            self.epochs_completed += 1;
        }
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

/// Fixed FineWeb validation split exported by Parameter Golf.
pub struct FineWebValidationData {
    tokens: Vec<u32>,
}

impl FineWebValidationData {
    /// Load all `fineweb_val_*.bin` shards from a dataset directory.
    pub fn load_dir(dataset_dir: &Path) -> io::Result<Self> {
        let files = fineweb_shard_paths(dataset_dir, "fineweb_val_")?;

        let mut tokens = Vec::new();
        for file in &files {
            tokens.extend(load_fineweb_shard(file)?);
        }
        if tokens.len() <= 1 {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("validation split in {} is too short", dataset_dir.display()),
            ));
        }
        Ok(Self { tokens })
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }
}

fn load_fineweb_shard(path: &Path) -> io::Result<Vec<u32>> {
    let mut file = File::open(path)?;
    let mut raw = Vec::new();
    file.read_to_end(&mut raw)?;
    let num_tokens = validate_fineweb_shard_bytes(path, &raw)?;
    let body = &raw[FineWebShard::HEADER_BYTES..];
    let mut tokens = Vec::with_capacity(num_tokens);
    for chunk in body.chunks_exact(2) {
        tokens.push(u16::from_le_bytes([chunk[0], chunk[1]]) as u32);
    }
    Ok(tokens)
}

fn validate_fineweb_shard_bytes(path: &Path, raw: &[u8]) -> io::Result<usize> {
    if raw.len() < FineWebShard::HEADER_BYTES {
        return Err(io::Error::new(
            ErrorKind::UnexpectedEof,
            format!("FineWeb shard too small: {}", path.display()),
        ));
    }

    let header_word = |idx: usize| -> i32 {
        let off = idx * 4;
        i32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]])
    };

    if header_word(0) != FineWebShard::MAGIC || header_word(1) != FineWebShard::VERSION {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            format!("unexpected FineWeb shard header in {}", path.display()),
        ));
    }

    let num_tokens = usize::try_from(header_word(2)).map_err(|_| {
        io::Error::new(
            ErrorKind::InvalidData,
            format!("negative token count in {}", path.display()),
        )
    })?;
    let expected_size = FineWebShard::HEADER_BYTES + num_tokens * 2;
    if raw.len() != expected_size {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "FineWeb shard size mismatch for {}: expected {} bytes, got {}",
                path.display(),
                expected_size,
                raw.len()
            ),
        ));
    }
    Ok(num_tokens)
}

/// PG-compatible tokenizer byte-count lookup tables for SentencePiece BPB evaluation.
pub struct SentencePieceBpbLut {
    base_bytes: Vec<i16>,
    has_leading_space: Vec<bool>,
    is_boundary_token: Vec<bool>,
}

#[derive(Deserialize)]
struct SentencePieceBpbLutJson {
    base_bytes: Vec<i16>,
    has_leading_space: Vec<bool>,
    is_boundary_token: Vec<bool>,
}

impl SentencePieceBpbLut {
    /// Build the same byte-count lookup tables as parameter-golf by shelling out to Python's
    /// `sentencepiece` package. This keeps the logic byte-for-byte aligned with their reference.
    pub fn from_python(tokenizer_path: &Path, vocab_size: usize) -> io::Result<Self> {
        let script = r#"
import json
import sys
import sentencepiece as spm

tokenizer_path = sys.argv[1]
vocab_size = int(sys.argv[2])
sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
sp_vocab_size = int(sp.vocab_size())
table_size = max(sp_vocab_size, vocab_size)
base_bytes = [0] * table_size
has_leading_space = [False] * table_size
is_boundary_token = [True] * table_size
for token_id in range(sp_vocab_size):
    if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
        continue
    is_boundary_token[token_id] = False
    if sp.is_byte(token_id):
        base_bytes[token_id] = 1
        continue
    piece = sp.id_to_piece(token_id)
    if piece.startswith("▁"):
        has_leading_space[token_id] = True
        piece = piece[1:]
    base_bytes[token_id] = len(piece.encode("utf-8"))
json.dump(
    {
        "base_bytes": base_bytes,
        "has_leading_space": has_leading_space,
        "is_boundary_token": is_boundary_token,
    },
    sys.stdout,
)
"#;
        let output = Command::new("python3")
            .arg("-c")
            .arg(script)
            .arg(tokenizer_path)
            .arg(vocab_size.to_string())
            .output()?;
        if !output.status.success() {
            return Err(io::Error::new(
                ErrorKind::Other,
                format!(
                    "python3 sentencepiece LUT build failed for {}: {}",
                    tokenizer_path.display(),
                    String::from_utf8_lossy(&output.stderr)
                ),
            ));
        }

        let parsed: SentencePieceBpbLutJson =
            serde_json::from_slice(&output.stdout).map_err(|e| {
                io::Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "failed to parse sentencepiece LUT JSON for {}: {e}",
                        tokenizer_path.display()
                    ),
                )
            })?;
        Ok(Self {
            base_bytes: parsed.base_bytes,
            has_leading_space: parsed.has_leading_space,
            is_boundary_token: parsed.is_boundary_token,
        })
    }

    pub fn byte_len(&self, prev_id: u32, target_id: u32) -> usize {
        let tgt = target_id as usize;
        let prev = prev_id as usize;
        let mut bytes = self.base_bytes.get(tgt).copied().unwrap_or_default().max(0) as usize;
        let has_space = self.has_leading_space.get(tgt).copied().unwrap_or(false);
        let prev_is_boundary = self.is_boundary_token.get(prev).copied().unwrap_or(true);
        if has_space && !prev_is_boundary {
            bytes += 1;
        }
        bytes
    }
}

/// Compute tokenizer-agnostic BPB using the same byte-count logic as Parameter Golf's
/// sentencepiece-based validation loop.
pub fn compute_sentencepiece_bpb(
    losses: &[f32],
    prev_ids: &[u32],
    targets: &[u32],
    lut: &SentencePieceBpbLut,
) -> (f32, f32, usize) {
    let mut total_nats = 0.0f32;
    let mut total_bytes = 0usize;
    for ((&loss, &prev), &target) in losses.iter().zip(prev_ids.iter()).zip(targets.iter()) {
        let bytes = lut.byte_len(prev, target);
        if bytes > 0 {
            total_nats += loss;
            total_bytes += bytes;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn sentencepiece_bpb_adds_leading_space_when_previous_token_is_not_boundary() {
        let lut = SentencePieceBpbLut {
            base_bytes: vec![0, 2, 3],
            has_leading_space: vec![false, false, true],
            is_boundary_token: vec![true, false, false],
        };
        assert_eq!(lut.byte_len(0, 2), 3);
        assert_eq!(lut.byte_len(1, 2), 4);
    }

    #[test]
    fn sentencepiece_bpb_matches_expected_bits_per_byte() {
        let lut = SentencePieceBpbLut {
            base_bytes: vec![0, 2, 3],
            has_leading_space: vec![false, false, true],
            is_boundary_token: vec![true, false, false],
        };
        let losses = [std::f32::consts::LN_2 * 4.0, std::f32::consts::LN_2 * 6.0];
        let prev = [0, 1];
        let target = [1, 2];
        let (bpb, _, total_bytes) = compute_sentencepiece_bpb(&losses, &prev, &target, &lut);
        assert_eq!(total_bytes, 6);
        assert!((bpb - (10.0 / 6.0)).abs() < 1e-6);
    }

    #[test]
    fn fineweb_train_stream_wraps_across_shards_in_order() {
        let root = std::env::temp_dir().join(format!(
            "rustane-fineweb-stream-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();

        write_test_fineweb_shard(&root.join("fineweb_train_000000.bin"), &[1, 2, 3]).unwrap();
        write_test_fineweb_shard(&root.join("fineweb_train_000001.bin"), &[4, 5]).unwrap();

        let mut stream = FineWebTrainStream::open_dir(&root).unwrap();
        assert_eq!(stream.len(), 5);
        assert_eq!(stream.next_tokens(4), vec![1, 2, 3, 4]);
        assert_eq!(stream.next_tokens(4), vec![5, 1, 2, 3]);
        assert_eq!(stream.epochs_completed(), 1);

        let _ = fs::remove_dir_all(&root);
    }

    fn write_test_fineweb_shard(path: &Path, tokens: &[u32]) -> io::Result<()> {
        let mut raw = vec![0u8; FineWebShard::HEADER_BYTES];
        raw[0..4].copy_from_slice(&FineWebShard::MAGIC.to_le_bytes());
        raw[4..8].copy_from_slice(&FineWebShard::VERSION.to_le_bytes());
        raw[8..12].copy_from_slice(&(tokens.len() as i32).to_le_bytes());
        for &tok in tokens {
            raw.extend_from_slice(&(tok as u16).to_le_bytes());
        }
        fs::write(path, raw)
    }
}
