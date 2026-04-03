//! Rustane checkpoint loading for inference and eval tools.
//!
//! Checkpoints are written by `src/bin/train.rs` as:
//! magic("RSTK"), version(u32), step(u32), dim(u32), nlayers(u32), vocab(u32), seq(u32),
//! optionally activation(u32) for version >= 2,
//! followed by f32 weights in little-endian order.

use crate::full_model::ModelWeights;
use crate::layer::LayerWeights;
use crate::model::{FfnActivation, ModelConfig};
use std::io::{self, ErrorKind};
use std::path::Path;

const MAGIC: &[u8; 4] = b"RSTK";
const VERSION: u32 = 2;
const HEADER_BYTES_V1: usize = 4 + 6 * 4;
const HEADER_BYTES_V2: usize = 4 + 7 * 4;

pub struct Checkpoint {
    pub cfg: ModelConfig,
    pub weights: ModelWeights,
    pub step: u32,
}

pub fn load_checkpoint(path: &Path) -> io::Result<Checkpoint> {
    let bytes = std::fs::read(path)?;
    parse_checkpoint(&bytes)
}

fn parse_checkpoint(bytes: &[u8]) -> io::Result<Checkpoint> {
    if bytes.len() < HEADER_BYTES_V1 {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "checkpoint too small for header",
        ));
    }

    if &bytes[..4] != MAGIC {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "invalid checkpoint magic",
        ));
    }

    let mut cursor = 4;
    let version = read_u32(bytes, &mut cursor)?;
    if !(1..=VERSION).contains(&version) {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            format!("unsupported checkpoint version {version}"),
        ));
    }
    if version >= 2 && bytes.len() < HEADER_BYTES_V2 {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "checkpoint too small for v2 header",
        ));
    }

    let step = read_u32(bytes, &mut cursor)?;
    let dim = read_u32(bytes, &mut cursor)? as usize;
    let nlayers = read_u32(bytes, &mut cursor)? as usize;
    let vocab = read_u32(bytes, &mut cursor)? as usize;
    let seq = read_u32(bytes, &mut cursor)? as usize;
    let ffn_activation = if version >= 2 {
        let code = read_u32(bytes, &mut cursor)?;
        FfnActivation::from_checkpoint_code(code).ok_or_else(|| {
            io::Error::new(
                ErrorKind::InvalidData,
                format!("unsupported checkpoint activation code {code}"),
            )
        })?
    } else {
        FfnActivation::SwiGlu
    };

    if dim == 0 || nlayers == 0 || vocab == 0 || seq == 0 {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "checkpoint config contains zero dimension",
        ));
    }
    if dim % 128 != 0 {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            format!("checkpoint dim {dim} is not divisible by 128"),
        ));
    }
    if (bytes.len() - cursor) % 4 != 0 {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "checkpoint payload is not aligned to f32 values",
        ));
    }

    let payload_floats = (bytes.len() - cursor) / 4;
    let heads = dim / 128;
    let hidden = infer_hidden(dim, nlayers, vocab, payload_floats)?;
    let cfg = ModelConfig {
        dim,
        hidden,
        heads,
        kv_heads: heads,
        hd: 128,
        seq,
        nlayers,
        vocab,
        q_dim: dim,
        kv_dim: dim,
        gqa_ratio: 1,
        ffn_activation,
    };

    let embed = read_f32_vec(bytes, &mut cursor, vocab * dim)?;
    let gamma_final = read_f32_vec(bytes, &mut cursor, dim)?;

    let mut layers = Vec::with_capacity(nlayers);
    for _ in 0..nlayers {
        let wq = read_f32_vec(bytes, &mut cursor, cfg.dim * cfg.q_dim)?;
        let wk = read_f32_vec(bytes, &mut cursor, cfg.dim * cfg.kv_dim)?;
        let wv = read_f32_vec(bytes, &mut cursor, cfg.dim * cfg.kv_dim)?;
        let wo = read_f32_vec(bytes, &mut cursor, cfg.q_dim * cfg.dim)?;
        let w1 = read_f32_vec(bytes, &mut cursor, cfg.dim * cfg.hidden)?;
        let w3 = read_f32_vec(bytes, &mut cursor, cfg.dim * cfg.hidden)?;
        let w2 = read_f32_vec(bytes, &mut cursor, cfg.dim * cfg.hidden)?;
        let gamma1 = read_f32_vec(bytes, &mut cursor, cfg.dim)?;
        let gamma2 = read_f32_vec(bytes, &mut cursor, cfg.dim)?;
        let mut layer = LayerWeights {
            wq,
            wk,
            wv,
            wo,
            w1,
            w3,
            w2,
            wqt: vec![0.0; cfg.q_dim * cfg.dim],
            wkt: vec![0.0; cfg.kv_dim * cfg.dim],
            wvt: vec![0.0; cfg.kv_dim * cfg.dim],
            wot: vec![0.0; cfg.dim * cfg.q_dim],
            w1t: vec![0.0; cfg.hidden * cfg.dim],
            w3t: vec![0.0; cfg.hidden * cfg.dim],
            gamma1,
            gamma2,
            generation: 0,
        };
        layer.refresh_transposes(&cfg);
        layers.push(layer);
    }

    if cursor != bytes.len() {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "checkpoint had trailing bytes",
        ));
    }

    Ok(Checkpoint {
        cfg,
        weights: ModelWeights {
            embed,
            layers,
            gamma_final,
        },
        step,
    })
}

fn infer_hidden(
    dim: usize,
    nlayers: usize,
    vocab: usize,
    payload_floats: usize,
) -> io::Result<usize> {
    let fixed_floats = vocab
        .checked_mul(dim)
        .and_then(|x| x.checked_add(dim))
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "checkpoint dimensions overflow"))?;
    let layer_fixed = 4usize
        .checked_mul(dim)
        .and_then(|x| x.checked_mul(dim))
        .and_then(|x| x.checked_add(2 * dim))
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "checkpoint dimensions overflow"))?;
    if payload_floats < fixed_floats + nlayers * layer_fixed {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "checkpoint payload too small for fixed-size tensors",
        ));
    }

    let hidden_total = payload_floats - fixed_floats - nlayers * layer_fixed;
    let hidden_denom = nlayers
        .checked_mul(3)
        .and_then(|x| x.checked_mul(dim))
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "checkpoint dimensions overflow"))?;
    if hidden_denom == 0 || hidden_total % hidden_denom != 0 {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "checkpoint payload does not imply an integer hidden size",
        ));
    }
    let hidden = hidden_total / hidden_denom;
    if hidden == 0 {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "checkpoint hidden size inferred as zero",
        ));
    }
    Ok(hidden)
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> io::Result<u32> {
    let next = cursor
        .checked_add(4)
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "checkpoint cursor overflow"))?;
    let chunk = bytes
        .get(*cursor..next)
        .ok_or_else(|| io::Error::new(ErrorKind::UnexpectedEof, "unexpected EOF reading u32"))?;
    *cursor = next;
    Ok(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
}

fn read_f32_vec(bytes: &[u8], cursor: &mut usize, count: usize) -> io::Result<Vec<f32>> {
    let byte_len = count
        .checked_mul(4)
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "checkpoint tensor size overflow"))?;
    let next = cursor
        .checked_add(byte_len)
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "checkpoint cursor overflow"))?;
    let chunk = bytes.get(*cursor..next).ok_or_else(|| {
        io::Error::new(
            ErrorKind::UnexpectedEof,
            format!("unexpected EOF reading tensor with {count} f32 values"),
        )
    })?;
    *cursor = next;

    let mut out = Vec::with_capacity(count);
    for raw in chunk.chunks_exact(4) {
        out.push(f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_f32_vec(buf: &mut Vec<u8>, values: &[f32]) {
        for &value in values {
            buf.extend_from_slice(&value.to_le_bytes());
        }
    }

    #[test]
    fn parses_rustane_checkpoint_and_rebuilds_transposes() {
        let cfg = ModelConfig {
            dim: 256,
            hidden: 640,
            heads: 2,
            kv_heads: 2,
            hd: 128,
            seq: 64,
            nlayers: 2,
            vocab: 32,
            q_dim: 256,
            kv_dim: 256,
            gqa_ratio: 1,
            ffn_activation: FfnActivation::SwiGlu,
        };
        let weights = ModelWeights::random(&cfg);

        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&123u32.to_le_bytes());
        bytes.extend_from_slice(&(cfg.dim as u32).to_le_bytes());
        bytes.extend_from_slice(&(cfg.nlayers as u32).to_le_bytes());
        bytes.extend_from_slice(&(cfg.vocab as u32).to_le_bytes());
        bytes.extend_from_slice(&(cfg.seq as u32).to_le_bytes());
        bytes.extend_from_slice(&cfg.ffn_activation.checkpoint_code().to_le_bytes());
        write_f32_vec(&mut bytes, &weights.embed);
        write_f32_vec(&mut bytes, &weights.gamma_final);
        for layer in &weights.layers {
            for tensor in [
                &layer.wq,
                &layer.wk,
                &layer.wv,
                &layer.wo,
                &layer.w1,
                &layer.w3,
                &layer.w2,
                &layer.gamma1,
                &layer.gamma2,
            ] {
                write_f32_vec(&mut bytes, tensor);
            }
        }

        let ckpt = parse_checkpoint(&bytes).expect("checkpoint should parse");
        assert_eq!(ckpt.step, 123);
        assert_eq!(ckpt.cfg.dim, cfg.dim);
        assert_eq!(ckpt.cfg.hidden, cfg.hidden);
        assert_eq!(ckpt.cfg.nlayers, cfg.nlayers);
        assert_eq!(ckpt.cfg.vocab, cfg.vocab);
        assert_eq!(ckpt.cfg.seq, cfg.seq);
        assert_eq!(ckpt.cfg.ffn_activation, cfg.ffn_activation);
        assert_eq!(ckpt.weights.embed, weights.embed);
        assert_eq!(ckpt.weights.gamma_final, weights.gamma_final);
        assert_eq!(ckpt.weights.layers.len(), weights.layers.len());
        assert_eq!(ckpt.weights.layers[0].w1, weights.layers[0].w1);
        assert_eq!(ckpt.weights.layers[0].w1t, weights.layers[0].w1t);
        assert_eq!(ckpt.weights.layers[1].wot, weights.layers[1].wot);
    }
}
