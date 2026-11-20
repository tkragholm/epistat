//! Quantiles of a survival curve and of its confidence limits.

/// The `sqrt(eps)` that decides when a flat part of the curve is at a
/// requested level.
pub(super) const TOLERANCE: f64 = 1.490_116_119_384_765_6e-8;

/// The index of the first node at or above `v` among the nodes `y + shift`,
/// taken in order of value, or `None` when `v` lies outside the range of the
/// nodes or no node is defined.
fn first_at_or_above(y: &[f64], shift: f64, v: f64) -> Option<usize> {
    let mut best: Option<(f64, usize)> = None;
    let mut low = f64::INFINITY;
    let mut high = f64::NEG_INFINITY;
    for (i, &value) in y.iter().enumerate() {
        if value.is_nan() {
            continue;
        }
        let node = value + shift;
        low = low.min(node);
        high = high.max(node);
        if node >= v && best.is_none_or(|(b, _)| node < b) {
            best = Some((node, i));
        }
    }
    if v < low || v > high {
        return None;
    }
    best.map(|(_, i)| i)
}

/// The quantiles `probs` of a distribution function given at the points `x`
/// as `y`, which is `1 - S` and starts at 0.
///
/// `y` may hold NaN where a confidence limit is undefined. A level that the
/// curve never reaches gives `None`. Where the curve is flat at exactly the
/// level, within the tolerance, the answer is the midpoint of the flat part.
pub(super) fn find(x: &[f64], y: &[f64], probs: &[f64]) -> Vec<Option<f64>> {
    let tol = TOLERANCE;
    let max_y = y
        .iter()
        .copied()
        .filter(|v| !v.is_nan())
        .fold(f64::NEG_INFINITY, f64::max);
    let min_p = probs.iter().copied().fold(f64::INFINITY, f64::min);
    if max_y < min_p {
        return vec![None; probs.len()];
    }
    let x_max = x[x.len() - 1];
    // Keep the first of each run of equal levels. A level that is undefined
    // counts as equal to the other undefined ones.
    let mut xs = Vec::with_capacity(x.len());
    let mut ys: Vec<f64> = Vec::with_capacity(y.len());
    let mut seen_nan = false;
    let mut distinct = std::collections::HashSet::new();
    for (&xv, &yv) in x.iter().zip(y) {
        let fresh = if yv.is_nan() {
            !std::mem::replace(&mut seen_nan, true)
        } else {
            distinct.insert(yv.to_bits())
        };
        if fresh {
            xs.push(xv);
            ys.push(yv);
        }
    }
    let last = ys[ys.len() - 1];
    probs
        .iter()
        .map(|&p| {
            let i1 = first_at_or_above(&ys, tol, p);
            let i2 = first_at_or_above(&ys, -tol, p);
            let mut quantile = match (i1, i2) {
                (Some(a), Some(b)) => Some(f64::midpoint(xs[a], xs[b])),
                _ => None,
            };
            if p == 0.0 {
                quantile = Some(xs[0]);
            }
            if !last.is_nan() && (p - last).abs() < tol {
                quantile = i1.map(|a| f64::midpoint(xs[a], x_max));
            }
            quantile
        })
        .collect()
}
