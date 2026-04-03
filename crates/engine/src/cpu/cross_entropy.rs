//! Cross-entropy loss with numerically stable log-softmax.
//!
//! Forward: loss = -log(softmax(logits)[target])
//!   where log_softmax = logits - max(logits) - log(sum(exp(logits - max)))
//!
//! Backward: d_logits[i] = softmax[i] - (i == target)

use super::vdsp;

/// Cross-entropy forward pass (single token).
/// `logits`: raw scores [vocab], `target`: correct class index.
/// Returns (loss, log_softmax buffer) — log_softmax cached for backward.
pub fn forward(logits: &[f32], target: usize) -> (f32, Vec<f32>) {
    let vocab = logits.len();
    assert!(target < vocab);

    // Numerically stable log-softmax:
    // log_softmax = (logits - max) - log(sum(exp(logits - max)))
    let max_val = vdsp::maxv(logits);

    // shifted = logits - max
    let mut shifted = vec![0.0f32; vocab];
    vdsp::vsadd(logits, -max_val, &mut shifted);

    // exp_shifted = exp(shifted)
    let mut exp_shifted = vec![0.0f32; vocab];
    vdsp::expf(&shifted, &mut exp_shifted);

    let sum_exp = vdsp::sve(&exp_shifted);
    let log_sum_exp = sum_exp.ln();

    // log_softmax = shifted - log_sum_exp
    let mut log_sm = vec![0.0f32; vocab];
    vdsp::vsadd(&shifted, -log_sum_exp, &mut log_sm);

    let loss = -log_sm[target];
    (loss, log_sm)
}

/// Cross-entropy backward pass (single token).
/// `log_softmax`: from forward, `target`: correct class.
/// Writes gradient into `d_logits` [vocab].
pub fn backward(log_softmax: &[f32], target: usize, d_logits: &mut [f32]) {
    let vocab = log_softmax.len();
    assert_eq!(d_logits.len(), vocab);
    assert!(target < vocab);

    // d_logits[i] = softmax[i] - (i == target)
    // softmax[i] = exp(log_softmax[i])
    vdsp::expf(log_softmax, d_logits);
    d_logits[target] -= 1.0;
}

/// Batched cross-entropy forward+backward (single alloc for all positions).
/// `logits`: [seq * vocab] row-major, `targets`: [seq] target class indices.
/// Writes gradients into `dlogits` [seq * vocab], scaled by `inv_seq`.
/// Returns total loss (sum over positions, NOT averaged).
pub fn forward_backward_batch(
    logits: &[f32],
    targets: &[u32],
    vocab: usize,
    dlogits: &mut [f32],
    inv_seq: f32,
) -> f32 {
    let seq = targets.len();
    let mut shifted = vec![0.0f32; vocab];
    let mut exp_vals = vec![0.0f32; vocab];
    let mut total_loss = 0.0f32;

    for s in 0..seq {
        let tok_logits = &logits[s * vocab..(s + 1) * vocab];
        let target = targets[s] as usize;

        // log-softmax: shifted = logits - max
        let max_val = vdsp::maxv(tok_logits);
        vdsp::vsadd(tok_logits, -max_val, &mut shifted);

        // exp(shifted) → softmax numerator
        vdsp::expf(&shifted, &mut exp_vals);
        let sum_exp = vdsp::sve(&exp_vals);

        // loss = -(shifted[target] - log(sum_exp))
        total_loss -= shifted[target] - sum_exp.ln();

        // backward: d_tok = softmax - one_hot = exp_vals/sum_exp - one_hot
        // Reuse exp_vals directly (avoids second expf call per position)
        let d_tok = &mut dlogits[s * vocab..(s + 1) * vocab];
        vdsp::vsmul(&exp_vals, 1.0 / sum_exp, d_tok);
        d_tok[target] -= 1.0;
    }

    // Single-pass scale over full dlogits array (1 call vs 512)
    vdsp::sscal(dlogits, inv_seq);

    total_loss
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loss_is_positive() {
        let logits = [1.0, 2.0, 3.0, 4.0];
        let (loss, _) = forward(&logits, 3);
        assert!(loss > 0.0);
        // target=3 has highest logit, so loss should be small
        assert!(loss < 2.0);
    }

    #[test]
    fn loss_decreases_with_confidence() {
        let low_conf = [1.0, 1.0, 1.0, 1.0];
        let high_conf = [0.0, 0.0, 0.0, 10.0];
        let (loss_low, _) = forward(&low_conf, 3);
        let (loss_high, _) = forward(&high_conf, 3);
        assert!(loss_high < loss_low);
    }

    #[test]
    fn gradients_sum_to_zero() {
        let logits = [1.0, 2.0, 3.0];
        let (_, log_sm) = forward(&logits, 1);
        let mut d_logits = [0.0f32; 3];
        backward(&log_sm, 1, &mut d_logits);

        // softmax sums to 1, minus 1 for target → gradients sum to 0
        let sum: f32 = d_logits.iter().sum();
        assert!(sum.abs() < 1e-5, "gradient sum = {sum}");
    }

    #[test]
    fn numerical_gradient() {
        let logits = [2.0f32, -1.0, 0.5, 1.5];
        let target = 2;
        let (_, log_sm) = forward(&logits, target);
        let mut d_logits = [0.0f32; 4];
        backward(&log_sm, target, &mut d_logits);

        let eps = 1e-4;
        for i in 0..logits.len() {
            let mut plus = logits;
            let mut minus = logits;
            plus[i] += eps;
            minus[i] -= eps;
            let (loss_plus, _) = forward(&plus, target);
            let (loss_minus, _) = forward(&minus, target);
            let numerical = (loss_plus - loss_minus) / (2.0 * eps);
            assert!(
                (d_logits[i] - numerical).abs() < 5e-3,
                "d_logits[{i}]: analytical={} vs numerical={numerical}",
                d_logits[i]
            );
        }
    }
}
