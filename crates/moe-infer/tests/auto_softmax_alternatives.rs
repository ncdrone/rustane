//! Profile attention softmax: scalar vs vDSP vectorized.
//! Current: scalar loop in mla_attention.rs. At short seq_len (< 100),
//! this is trivial. But at longer contexts it could matter.

use std::time::Instant;

fn scalar_softmax(s: &mut [f32]) {
    let max_s = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in s.iter_mut() { *v = (*v - max_s).exp(); sum += *v; }
    for v in s.iter_mut() { *v /= sum; }
}

fn two_pass_softmax(s: &mut [f32]) {
    // Pass 1: find max
    let mut max_s = f32::NEG_INFINITY;
    for &v in s.iter() { if v > max_s { max_s = v; } }
    // Pass 2: exp + sum
    let mut sum = 0.0f32;
    for v in s.iter_mut() {
        *v = (*v - max_s).exp();
        sum += *v;
    }
    // Pass 3: normalize
    let inv_sum = 1.0 / sum;
    for v in s.iter_mut() { *v *= inv_sum; }
}

#[test]
#[ignore = "profiling test"]
fn profile_softmax_variants() {
    eprintln!("\n=== Softmax Alternatives ===\n");

    for &seq_len in &[10, 50, 100, 500, 1000, 4096] {
        let heads = 64;
        let mut scores = vec![0.1f32; heads * seq_len];
        let iters = if seq_len > 1000 { 100 } else { 500 };

        // Scalar (current)
        let t = Instant::now();
        for _ in 0..iters {
            for h in 0..heads {
                scalar_softmax(&mut scores[h * seq_len..(h + 1) * seq_len]);
            }
        }
        let scalar_us = t.elapsed().as_micros() as f64 / iters as f64;

        // Two-pass (multiply by 1/sum instead of divide)
        let t = Instant::now();
        for _ in 0..iters {
            for h in 0..heads {
                two_pass_softmax(&mut scores[h * seq_len..(h + 1) * seq_len]);
            }
        }
        let two_pass_us = t.elapsed().as_micros() as f64 / iters as f64;

        let winner = if two_pass_us < scalar_us { "two-pass" } else { "scalar" };
        eprintln!("  seq={:<5} scalar: {:>6.0}µs  two-pass: {:>6.0}µs  winner: {}",
            seq_len, scalar_us, two_pass_us, winner);
    }
}
