//! SiLU (Swish) activation and its derivative.
//!
//! Forward: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//! Backward: silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
//!                     = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))

/// SiLU forward pass: out[i] = x[i] * sigmoid(x[i])
pub fn forward(x: &[f32], out: &mut [f32]) {
    let n = x.len();
    assert_eq!(out.len(), n);

    // sigmoid(x) = 1 / (1 + exp(-x)) = exp(x) / (1 + exp(x))
    // For numerical stability, compute exp(-|x|) based approach via scalar loop
    // vDSP doesn't have sigmoid, and negating + exp + div is awkward
    for i in 0..n {
        let sig = 1.0 / (1.0 + (-x[i]).exp());
        out[i] = x[i] * sig;
    }
}

/// SiLU backward pass: d_x[i] = dy[i] * silu'(x[i])
/// where silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
pub fn backward(dy: &[f32], x: &[f32], dx: &mut [f32]) {
    let n = x.len();
    assert_eq!(dy.len(), n);
    assert_eq!(dx.len(), n);

    for i in 0..n {
        let sig = 1.0 / (1.0 + (-x[i]).exp());
        let deriv = sig * (1.0 + x[i] * (1.0 - sig));
        dx[i] = dy[i] * deriv;
    }
}

/// Fused SiLU-gate: out = silu(gate) * up
/// Used in SwiGLU FFN: output = silu(W_gate * x) * (W_up * x)
pub fn silu_gate(gate: &[f32], up: &[f32], out: &mut [f32]) {
    let n = gate.len();
    assert_eq!(up.len(), n);
    assert_eq!(out.len(), n);

    for i in 0..n {
        let sig = 1.0 / (1.0 + (-gate[i]).exp());
        out[i] = gate[i] * sig * up[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_zero() {
        let x = [0.0f32];
        let mut out = [0.0f32; 1];
        forward(&x, &mut out);
        // silu(0) = 0 * 0.5 = 0
        assert!((out[0]).abs() < 1e-6);
    }

    #[test]
    fn forward_positive() {
        let x = [2.0f32];
        let mut out = [0.0f32; 1];
        forward(&x, &mut out);
        let sig = 1.0 / (1.0 + (-2.0f32).exp());
        assert!((out[0] - 2.0 * sig).abs() < 1e-6);
    }

    #[test]
    fn backward_numerical() {
        let x = [-1.0f32, 0.0, 0.5, 2.0];
        let dy = [1.0f32; 4];
        let mut dx = [0.0f32; 4];
        backward(&dy, &x, &mut dx);

        let eps = 1e-4;
        for i in 0..x.len() {
            let mut x_p = x;
            let mut x_m = x;
            x_p[i] += eps;
            x_m[i] -= eps;
            let mut out_p = [0.0f32; 4];
            let mut out_m = [0.0f32; 4];
            forward(&x_p, &mut out_p);
            forward(&x_m, &mut out_m);
            let numerical = (out_p[i] - out_m[i]) / (2.0 * eps);
            assert!(
                (dx[i] - numerical).abs() < 1e-3,
                "dx[{i}]: analytical={} vs numerical={numerical}",
                dx[i]
            );
        }
    }

    #[test]
    fn silu_gate_matches_manual() {
        let gate = [1.0f32, 2.0];
        let up = [3.0f32, 4.0];
        let mut out = [0.0f32; 2];
        silu_gate(&gate, &up, &mut out);

        let mut silu_out = [0.0f32; 2];
        forward(&gate, &mut silu_out);
        for i in 0..2 {
            let expected = silu_out[i] * up[i];
            assert!((out[i] - expected).abs() < 1e-6);
        }
    }
}
