//! 4-bit and 2-bit weight quantization for MoE inference.
//!
//! Group-wise asymmetric quantization with f16 scale/zero per group.
//! Pack 8 nibbles per u32, LSB-first.

pub mod pack4;

pub use pack4::PackedWeights4Bit;
