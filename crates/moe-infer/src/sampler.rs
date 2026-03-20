//! Token sampling: greedy, top-k, top-p, temperature.

/// Greedy: return the index with the highest logit.
pub fn sample_greedy(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Top-k sampling with temperature.
pub fn sample_top_k(logits: &[f32], k: usize, temperature: f32, seed: u64) -> usize {
    if temperature <= 0.0 {
        return sample_greedy(logits);
    }

    // Apply temperature
    let scaled: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();

    // Get top-k indices
    let mut indexed: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
    let k = k.min(indexed.len());
    indexed.select_nth_unstable_by(k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    indexed.truncate(k);

    // Softmax over top-k
    let max_val = indexed.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = indexed.iter().map(|(_, v)| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

    // Sample from distribution
    let mut rng = seed;
    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let r = (rng >> 33) as f32 / (u32::MAX >> 1) as f32;

    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return indexed[i].0;
        }
    }
    indexed.last().map(|(i, _)| *i).unwrap_or(0)
}
