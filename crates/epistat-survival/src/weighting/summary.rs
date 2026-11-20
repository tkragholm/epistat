//! The summary of a set of weights.

/// The summary of the weights of one group, as `summary.weightit` reports it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightSummary {
    /// The number of weights.
    pub n: usize,
    /// The sum of the weights.
    pub sum: f64,
    /// The smallest weight.
    pub min: f64,
    /// The largest weight.
    pub max: f64,
    /// The mean weight.
    pub mean: f64,
    /// The standard deviation, with divisor `n - 1`, over the mean. `None`
    /// for fewer than two weights.
    pub coefficient_of_variation: Option<f64>,
    /// Kish's effective sample size `(sum w)^2 / sum w^2`, the size of an
    /// unweighted sample with the same variance of its mean.
    pub effective_sample_size: f64,
}

impl WeightSummary {
    /// The summary of the weights, or `None` when there are none.
    #[must_use]
    pub fn new(weights: impl IntoIterator<Item = f64>) -> Option<Self> {
        let weights: Vec<f64> = weights.into_iter().collect();
        let n = weights.len();
        if n == 0 {
            return None;
        }
        let sum = epistat_stats::sum::compensated(weights.iter().copied());
        let sum_sq = epistat_stats::sum::compensated(weights.iter().map(|w| w * w));
        let mean = sum / n as f64;
        let coefficient_of_variation = (n > 1).then(|| {
            let ss =
                epistat_stats::sum::compensated(weights.iter().map(|w| (w - mean) * (w - mean)));
            (ss / (n - 1) as f64).sqrt() / mean
        });
        Some(Self {
            n,
            sum,
            min: weights.iter().copied().fold(f64::INFINITY, f64::min),
            max: weights.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            mean,
            coefficient_of_variation,
            effective_sample_size: sum * sum / sum_sq,
        })
    }
}
