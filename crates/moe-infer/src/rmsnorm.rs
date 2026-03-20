//! RMSNorm: Root Mean Square Layer Normalization.
//! Used pre-attention and pre-FFN in Qwen3. eps=1e-6.

/// RMSNorm forward: out[i] = x[i] / rms(x) * gamma[i]
/// where rms(x) = sqrt(mean(x^2) + eps)
pub fn rmsnorm(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    debug_assert_eq!(x.len(), gamma.len());
    let n = x.len() as f32;
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let rms = (ss / n + eps).sqrt();
    let inv_rms = 1.0 / rms;
    x.iter()
        .zip(gamma.iter())
        .map(|(&xi, &gi)| xi * inv_rms * gi)
        .collect()
}

/// In-place RMSNorm: overwrites `x`.
pub fn rmsnorm_inplace(x: &mut [f32], gamma: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), gamma.len());
    let n = x.len() as f32;
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (ss / n + eps).sqrt();
    for (xi, &gi) in x.iter_mut().zip(gamma.iter()) {
        *xi *= inv_rms * gi;
    }
}

/// Per-head RMSNorm for QK-norm in Qwen3 attention.
/// x shape: [num_heads * head_dim], gamma shape: [head_dim] (shared across heads)
pub fn per_head_rmsnorm(x: &mut [f32], gamma: &[f32], num_heads: usize, eps: f32) {
    let head_dim = gamma.len();
    debug_assert_eq!(x.len(), num_heads * head_dim);
    for h in 0..num_heads {
        let head = &mut x[h * head_dim..(h + 1) * head_dim];
        rmsnorm_inplace(head, gamma, eps);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_gamma() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0f32; 4];
        let out = rmsnorm(&x, &gamma, 1e-6);
        // RMS = sqrt(mean([1,4,9,16])) = sqrt(30/4) = sqrt(7.5)
        let rms = 7.5f32.sqrt();
        assert!((out[0] - 1.0 / rms).abs() < 1e-5);
        assert!((out[1] - 2.0 / rms).abs() < 1e-5);
        assert!((out[2] - 3.0 / rms).abs() < 1e-5);
        assert!((out[3] - 4.0 / rms).abs() < 1e-5);
    }

    #[test]
    fn scaled_gamma() {
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let gamma = vec![2.0, 3.0, 4.0, 5.0];
        let out = rmsnorm(&x, &gamma, 1e-6);
        // RMS = sqrt(4/4) = 1.0, so out = gamma
        assert!((out[0] - 2.0).abs() < 1e-5);
        assert!((out[3] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn inplace_matches() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![0.5, 1.0, 1.5, 2.0];
        let expected = rmsnorm(&x, &gamma, 1e-6);
        let mut x2 = x.clone();
        rmsnorm_inplace(&mut x2, &gamma, 1e-6);
        for (a, b) in expected.iter().zip(x2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn per_head() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = vec![1.0f32; 4]; // head_dim=4, 2 heads
        per_head_rmsnorm(&mut x, &gamma, 2, 1e-6);

        // Head 0: [1,2,3,4], RMS = sqrt(30/4)
        let rms0 = (30.0f32 / 4.0).sqrt();
        assert!((x[0] - 1.0 / rms0).abs() < 1e-5);

        // Head 1: [5,6,7,8], RMS = sqrt(174/4)
        let rms1 = (174.0f32 / 4.0).sqrt();
        assert!((x[4] - 5.0 / rms1).abs() < 1e-5);
    }
}
