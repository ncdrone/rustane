//! RMSNorm — forward and backward passes using vDSP.
//!
//! Forward:  y = (x / rms(x)) * gamma
//!   where rms(x) = sqrt(mean(x²) + eps)
//!
//! Backward: dx = (gamma / rms) * (dy - x * dot(dy, x*gamma) / (n * rms²))

use super::vdsp;

const EPS: f32 = 1e-5;

/// RMSNorm forward pass.
/// `x`: input [dim], `gamma`: learned scale [dim], `out`: normalized output [dim].
/// Returns the inverse RMS (1/rms) needed for backward.
pub fn forward(x: &[f32], gamma: &[f32], out: &mut [f32]) -> f32 {
    let dim = x.len();
    assert_eq!(gamma.len(), dim);
    assert_eq!(out.len(), dim);

    // x² → out (reuse buffer temporarily)
    vdsp::vmul(x, x, out);

    // mean(x²)
    let mean_sq = vdsp::sve(out) / dim as f32;

    // 1 / sqrt(mean_sq + eps)
    let rms_inv = 1.0 / (mean_sq + EPS).sqrt();

    // out = x * rms_inv
    vdsp::vsmul(x, rms_inv, out);

    // out = out * gamma  (need scratch to avoid aliasing)
    let mut tmp = vec![0.0f32; dim];
    vdsp::vmul(out, gamma, &mut tmp);
    out.copy_from_slice(&tmp);

    rms_inv
}

/// RMSNorm backward pass.
/// `dy`: gradient from upstream [dim], `x`: original input [dim],
/// `gamma`: scale weights [dim], `rms_inv`: from forward pass.
/// Writes `dx` [dim] and accumulates into `dgamma` [dim].
pub fn backward(
    dy: &[f32],
    x: &[f32],
    gamma: &[f32],
    rms_inv: f32,
    dx: &mut [f32],
    dgamma: &mut [f32],
) {
    let dim = x.len();
    assert_eq!(dy.len(), dim);
    assert_eq!(gamma.len(), dim);
    assert_eq!(dx.len(), dim);
    assert_eq!(dgamma.len(), dim);

    // x_hat = x * rms_inv (normalized input, stored in dx temporarily)
    vdsp::vsmul(x, rms_inv, dx);

    // dgamma += dy * x_hat  (accumulate)
    let mut scratch = vec![0.0f32; dim];
    vdsp::vmul(dy, dx, &mut scratch); // scratch = dy * x_hat
    let mut tmp = vec![0.0f32; dim];
    vdsp::vadd(dgamma, &scratch, &mut tmp);
    dgamma.copy_from_slice(&tmp);

    // dx = rms_inv * (dy * gamma - x_hat * mean(dy * gamma * x_hat))
    vdsp::vmul(dy, gamma, &mut scratch); // scratch = dy * gamma
    vdsp::vmul(&scratch, dx, &mut tmp); // tmp = dy * gamma * x_hat
    let dot = vdsp::sve(&tmp) / dim as f32;

    vdsp::vsmul(dx, dot, &mut tmp); // tmp = x_hat * dot
    vdsp::vsub(&tmp, &scratch, dx); // dx = (dy * gamma) - (x_hat * dot)
    scratch.copy_from_slice(dx);
    vdsp::vsmul(&scratch, rms_inv, dx); // dx *= rms_inv
}

/// Batch RMSNorm forward on position-contiguous data.
/// `x_t` is [seq, dim] row-major (position-contiguous), `gamma` is [dim].
/// Writes `xnorm_t` [seq, dim] and `rms_inv` [seq].
/// Uses a single scratch allocation for all positions.
pub fn forward_batch(
    x_t: &[f32],
    gamma: &[f32],
    xnorm_t: &mut [f32],
    rms_inv: &mut [f32],
    dim: usize,
    seq: usize,
) {
    let mut scratch = vec![0.0f32; dim];
    for s in 0..seq {
        let x_pos = &x_t[s * dim..(s + 1) * dim];
        let out = &mut xnorm_t[s * dim..(s + 1) * dim];

        vdsp::vmul(x_pos, x_pos, &mut scratch);
        let mean_sq = vdsp::sve(&scratch) / dim as f32;
        let inv = 1.0 / (mean_sq + EPS).sqrt();
        rms_inv[s] = inv;

        vdsp::vsmul(x_pos, inv, out);
        vdsp::vmul(out, gamma, &mut scratch);
        out.copy_from_slice(&scratch);
    }
}

/// Batch RMSNorm backward on position-contiguous data.
/// All inputs/outputs are [seq, dim] row-major. Accumulates into `dgamma`.
pub fn backward_batch(
    dy_t: &[f32],
    x_t: &[f32],
    gamma: &[f32],
    rms_inv: &[f32],
    dx_t: &mut [f32],
    dgamma: &mut [f32],
    dim: usize,
    seq: usize,
) {
    let mut scratch = vec![0.0f32; dim];
    let mut tmp = vec![0.0f32; dim];
    for s in 0..seq {
        let dy_pos = &dy_t[s * dim..(s + 1) * dim];
        let x_pos = &x_t[s * dim..(s + 1) * dim];
        let dx_pos = &mut dx_t[s * dim..(s + 1) * dim];
        let inv = rms_inv[s];

        // x_hat = x * inv (stored in dx temporarily)
        vdsp::vsmul(x_pos, inv, dx_pos);

        // dgamma += dy * x_hat
        vdsp::vmul(dy_pos, dx_pos, &mut scratch);
        vdsp::vadd(dgamma, &scratch, &mut tmp);
        dgamma.copy_from_slice(&tmp);

        // dx = inv * (dy * gamma - x_hat * mean(dy * gamma * x_hat))
        vdsp::vmul(dy_pos, gamma, &mut scratch);
        vdsp::vmul(&scratch, dx_pos, &mut tmp);
        let dot = vdsp::sve(&tmp) / dim as f32;
        vdsp::vsmul(dx_pos, dot, &mut tmp);
        vdsp::vsub(&tmp, &scratch, dx_pos);
        scratch.copy_from_slice(dx_pos);
        vdsp::vsmul(&scratch, inv, dx_pos);
    }
}

/// Channel-first RMSNorm forward on [dim, seq] data.
/// Operates directly on channel-first layout — no transpose needed.
/// `x` is [dim, seq], `gamma` is [dim], writes `out` [dim, seq] and `rms_inv` [seq].
pub fn forward_channel_first(
    x: &[f32],
    gamma: &[f32],
    out: &mut [f32],
    rms_inv: &mut [f32],
    dim: usize,
    seq: usize,
) {
    // Step 1: sum_sq[s] = sum_d x[d*seq+s]^2, accumulated into rms_inv
    for s in 0..seq {
        rms_inv[s] = 0.0;
    }
    for d in 0..dim {
        let row = &x[d * seq..(d + 1) * seq];
        for s in 0..seq {
            rms_inv[s] += row[s] * row[s];
        }
    }

    // Step 2: rms_inv[s] = 1/sqrt(sum_sq[s]/dim + eps)
    let inv_dim = 1.0 / dim as f32;
    for s in 0..seq {
        rms_inv[s] = 1.0 / (rms_inv[s] * inv_dim + EPS).sqrt();
    }

    // Step 3: out[d,s] = x[d,s] * rms_inv[s] * gamma[d]
    for d in 0..dim {
        let x_row = &x[d * seq..(d + 1) * seq];
        let out_row = &mut out[d * seq..(d + 1) * seq];
        let g = gamma[d];
        for s in 0..seq {
            out_row[s] = x_row[s] * rms_inv[s] * g;
        }
    }
}

/// Channel-first RMSNorm backward on [dim, seq] data.
/// Operates directly on channel-first layout — no transpose needed.
/// `dy`, `x` are [dim, seq], `gamma` [dim], `rms_inv` [seq].
/// Writes `dx` [dim, seq], accumulates into `dgamma` [dim].
/// `dot_buf` is scratch of size [seq].
pub fn backward_channel_first(
    dy: &[f32],
    x: &[f32],
    gamma: &[f32],
    rms_inv: &[f32],
    dx: &mut [f32],
    dgamma: &mut [f32],
    dim: usize,
    seq: usize,
    dot_buf: &mut [f32],
) {
    // Pass 1: compute x_hat (stored in dx), dgamma, and dot_per_pos
    for s in 0..seq {
        dot_buf[s] = 0.0;
    }

    for d in 0..dim {
        let dy_row = &dy[d * seq..(d + 1) * seq];
        let x_row = &x[d * seq..(d + 1) * seq];
        let dx_row = &mut dx[d * seq..(d + 1) * seq];
        let g = gamma[d];

        let mut dg_accum = 0.0f32;
        for s in 0..seq {
            let x_hat = x_row[s] * rms_inv[s];
            dx_row[s] = x_hat; // store x_hat temporarily
            dg_accum += dy_row[s] * x_hat;
            dot_buf[s] += dy_row[s] * g * x_hat;
        }
        dgamma[d] += dg_accum;
    }

    // Pass 2: dx = rms_inv * (dy * gamma - x_hat * dot/dim)
    let inv_dim = 1.0 / dim as f32;
    for s in 0..seq {
        dot_buf[s] *= inv_dim;
    }

    for d in 0..dim {
        let dy_row = &dy[d * seq..(d + 1) * seq];
        let g = gamma[d];
        let dx_row = &mut dx[d * seq..(d + 1) * seq];
        // dx_row currently contains x_hat from pass 1
        for s in 0..seq {
            let x_hat = dx_row[s];
            dx_row[s] = rms_inv[s] * (dy_row[s] * g - x_hat * dot_buf[s]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_unit_gamma() {
        // gamma=1 everywhere → pure normalization
        let x = [3.0, 4.0]; // rms = sqrt((9+16)/2) = sqrt(12.5)
        let gamma = [1.0, 1.0];
        let mut out = [0.0f32; 2];
        let rms_inv = forward(&x, &gamma, &mut out);

        let expected_rms = (12.5f32 + EPS).sqrt();
        assert!((rms_inv - 1.0 / expected_rms).abs() < 1e-5);
        assert!((out[0] - 3.0 / expected_rms).abs() < 1e-5);
        assert!((out[1] - 4.0 / expected_rms).abs() < 1e-5);
    }

    #[test]
    fn forward_with_gamma() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let gamma = [0.5, 0.5, 0.5, 0.5];
        let mut out = [0.0f32; 4];
        let rms_inv = forward(&x, &gamma, &mut out);

        let mean_sq = (1.0 + 4.0 + 9.0 + 16.0) / 4.0; // 7.5
        let expected_rms_inv = 1.0 / (mean_sq + EPS).sqrt();
        assert!((rms_inv - expected_rms_inv).abs() < 1e-5);

        for i in 0..4 {
            let expected = x[i] * expected_rms_inv * 0.5;
            assert!(
                (out[i] - expected).abs() < 1e-5,
                "out[{i}] = {} vs {expected}",
                out[i]
            );
        }
    }

    #[test]
    fn backward_numerical_gradient() {
        // Numerical gradient check for dx
        let x = [1.0f32, -2.0, 3.0, -0.5];
        let gamma = [1.0, 0.5, 2.0, 1.5];
        let dy = [1.0, 1.0, 1.0, 1.0];
        let dim = x.len();

        let mut out = [0.0f32; 4];
        let rms_inv = forward(&x, &gamma, &mut out);

        let mut dx = [0.0f32; 4];
        let mut dgamma = [0.0f32; 4];
        backward(&dy, &x, &gamma, rms_inv, &mut dx, &mut dgamma);

        // Numerical gradient
        let eps = 1e-4;
        for i in 0..dim {
            let mut x_plus = x;
            let mut x_minus = x;
            x_plus[i] += eps;
            x_minus[i] -= eps;

            let mut out_plus = [0.0f32; 4];
            let mut out_minus = [0.0f32; 4];
            forward(&x_plus, &gamma, &mut out_plus);
            forward(&x_minus, &gamma, &mut out_minus);

            // loss = sum(out * dy) = sum(out) when dy=1
            let loss_plus: f32 = out_plus.iter().sum();
            let loss_minus: f32 = out_minus.iter().sum();
            let numerical = (loss_plus - loss_minus) / (2.0 * eps);

            assert!(
                (dx[i] - numerical).abs() < 1e-2,
                "dx[{i}]: analytical={} vs numerical={numerical}",
                dx[i]
            );
        }
    }

    #[test]
    fn channel_first_matches_batch() {
        // Verify channel-first forward matches the transpose+batch+transpose path
        let dim = 4;
        let seq = 3;
        let gamma = [1.5, 0.5, 2.0, 0.8];
        // Channel-first [dim, seq]: x[d*seq + s]
        let x_cf: Vec<f32> = (0..dim * seq)
            .map(|i| ((i * 17 + 3) % 100) as f32 * 0.01 - 0.5)
            .collect();

        // Channel-first forward
        let mut out_cf = vec![0.0f32; dim * seq];
        let mut rms_inv_cf = vec![0.0f32; seq];
        forward_channel_first(&x_cf, &gamma, &mut out_cf, &mut rms_inv_cf, dim, seq);

        // Transpose to [seq, dim], run batch, transpose back
        let mut x_t = vec![0.0f32; seq * dim];
        for d in 0..dim {
            for s in 0..seq {
                x_t[s * dim + d] = x_cf[d * seq + s];
            }
        }
        let mut out_t = vec![0.0f32; seq * dim];
        let mut rms_inv_batch = vec![0.0f32; seq];
        forward_batch(&x_t, &gamma, &mut out_t, &mut rms_inv_batch, dim, seq);
        let mut out_batch_cf = vec![0.0f32; dim * seq];
        for d in 0..dim {
            for s in 0..seq {
                out_batch_cf[d * seq + s] = out_t[s * dim + d];
            }
        }

        for s in 0..seq {
            assert!(
                (rms_inv_cf[s] - rms_inv_batch[s]).abs() < 1e-5,
                "rms_inv[{s}]: cf={} vs batch={}",
                rms_inv_cf[s],
                rms_inv_batch[s]
            );
        }
        for i in 0..dim * seq {
            assert!(
                (out_cf[i] - out_batch_cf[i]).abs() < 1e-5,
                "out[{i}]: cf={} vs batch={}",
                out_cf[i],
                out_batch_cf[i]
            );
        }
    }

    #[test]
    fn channel_first_backward_matches_batch() {
        let dim = 4;
        let seq = 3;
        let gamma = [1.5, 0.5, 2.0, 0.8];
        let x_cf: Vec<f32> = (0..dim * seq)
            .map(|i| ((i * 17 + 3) % 100) as f32 * 0.01 - 0.5)
            .collect();
        let dy_cf: Vec<f32> = (0..dim * seq)
            .map(|i| ((i * 13 + 7) % 100) as f32 * 0.01 - 0.5)
            .collect();

        // Forward (channel-first) to get rms_inv
        let mut out_cf = vec![0.0f32; dim * seq];
        let mut rms_inv = vec![0.0f32; seq];
        forward_channel_first(&x_cf, &gamma, &mut out_cf, &mut rms_inv, dim, seq);

        // Channel-first backward
        let mut dx_cf = vec![0.0f32; dim * seq];
        let mut dgamma_cf = vec![0.0f32; dim];
        let mut dot_buf = vec![0.0f32; seq];
        backward_channel_first(
            &dy_cf,
            &x_cf,
            &gamma,
            &rms_inv,
            &mut dx_cf,
            &mut dgamma_cf,
            dim,
            seq,
            &mut dot_buf,
        );

        // Transpose, batch backward, transpose back
        let mut dy_t = vec![0.0f32; seq * dim];
        let mut x_t = vec![0.0f32; seq * dim];
        for d in 0..dim {
            for s in 0..seq {
                dy_t[s * dim + d] = dy_cf[d * seq + s];
                x_t[s * dim + d] = x_cf[d * seq + s];
            }
        }
        let mut dx_t = vec![0.0f32; seq * dim];
        let mut dgamma_batch = vec![0.0f32; dim];
        backward_batch(
            &dy_t,
            &x_t,
            &gamma,
            &rms_inv,
            &mut dx_t,
            &mut dgamma_batch,
            dim,
            seq,
        );
        let mut dx_batch_cf = vec![0.0f32; dim * seq];
        for d in 0..dim {
            for s in 0..seq {
                dx_batch_cf[d * seq + s] = dx_t[s * dim + d];
            }
        }

        for d in 0..dim {
            assert!(
                (dgamma_cf[d] - dgamma_batch[d]).abs() < 1e-4,
                "dgamma[{d}]: cf={} vs batch={}",
                dgamma_cf[d],
                dgamma_batch[d]
            );
        }
        for i in 0..dim * seq {
            assert!(
                (dx_cf[i] - dx_batch_cf[i]).abs() < 1e-4,
                "dx[{i}]: cf={} vs batch={}",
                dx_cf[i],
                dx_batch_cf[i]
            );
        }
    }
}
