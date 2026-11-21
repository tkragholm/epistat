//! The two-sample Kolmogorov-Smirnov test, with the exact distribution given
//! ties in the pooled sample.
//!
//! The statistic is the largest absolute difference between the two empirical
//! distribution functions, taken at the end of each run of tied values. The
//! exact p-value is the share of the arrangements of the pooled sample into
//! groups of the two sizes whose statistic is at least the observed one. It
//! is counted on the lattice of paths from `(0, 0)` to `(m, n)`: a step to the
//! right is a unit of the first sample, a step up a unit of the second, and a
//! path is an arrangement. A path is excluded when, at the end of a run of
//! ties, it is at distance `q` or more from the diagonal. The count follows
//! Schroer and Trenkler (1995), and is kept as a probability, so no binomial
//! coefficient is formed. The asymptotic p-value is the limiting distribution
//! of Kolmogorov (1933).
//!
//! The test is that of R's `ks.test`: exact when `m * n < 10000`, asymptotic
//! otherwise.

/// The statistic of two unsorted samples, or `None` when either is empty.
pub(crate) fn ks_statistic(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.is_empty() || y.is_empty() {
        return None;
    }
    let mut xs = x.to_vec();
    let mut ys = y.to_vec();
    xs.sort_by(f64::total_cmp);
    ys.sort_by(f64::total_cmp);
    Some(ks_d_sorted(&xs, &ys))
}

/// The statistic of two sorted samples.
#[expect(
    clippy::float_cmp,
    reason = "runs of tied values are runs of equal floats"
)]
pub(crate) fn ks_d_sorted(xs: &[f64], ys: &[f64]) -> f64 {
    let nx = xs.len() as f64;
    let ny = ys.len() as f64;
    let (mut i, mut j) = (0_usize, 0_usize);
    let mut d: f64 = 0.0;
    while i < xs.len() || j < ys.len() {
        let value = match (xs.get(i), ys.get(j)) {
            (Some(&a), Some(&b)) => a.min(b),
            (Some(&a), None) => a,
            (None, Some(&b)) => b,
            (None, None) => break,
        };
        while i < xs.len() && xs[i] == value {
            i += 1;
        }
        while j < ys.len() && ys[j] == value {
            j += 1;
        }
        d = d.max((i as f64 / nx - j as f64 / ny).abs());
    }
    d
}

/// The statistic and p-value of a two-sample test, exact below `m * n = 10000`
/// and asymptotic from there on, for unsorted samples. The last entry says
/// whether the p-value is exact.
#[expect(
    clippy::float_cmp,
    reason = "runs of tied values are runs of equal floats"
)]
pub(crate) fn ks_test(x: &[f64], y: &[f64]) -> Option<(f64, f64, bool)> {
    let d = ks_statistic(x, y)?;
    let (m, n) = (x.len(), y.len());
    if (m as f64) * (n as f64) < 10_000.0 {
        let mut pooled: Vec<f64> = x.iter().chain(y).copied().collect();
        pooled.sort_by(f64::total_cmp);
        let boundary: Vec<bool> = std::iter::once(true)
            .chain(pooled.windows(2).map(|pair| pair[0] != pair[1]))
            .chain(std::iter::once(true))
            .collect();
        Some((d, exact_upper(d, m, n, &boundary), true))
    } else {
        Some((d, asymptotic_upper(d, m, n), false))
    }
}

/// `P(D >= q)` for sample sizes `m` and `n`. `boundary[k]` says whether `k`
/// units into the pooled sample is the end of a run of ties, for `k` from 0 to
/// `m + n`.
pub(crate) fn exact_upper(q: f64, m: usize, n: usize, boundary: &[bool]) -> f64 {
    if q <= 0.0 {
        return 1.0;
    }
    if q > 1.0 {
        return 0.0;
    }
    // The statistic is a multiple of 1 / (m n), so `q` is read to the nearest
    // multiple below it, with a margin for rounding: a path is out when
    // |i n - j m| is at least `limit`.
    let mn = (m as f64) * (n as f64);
    let limit = (q * mn - 1e-7).floor() + 1.0;
    let out = |i: usize, j: usize| -> bool {
        boundary[i + j] && ((i * n) as f64 - (j * m) as f64).abs() >= limit
    };
    // p[j] is the probability that a path to (i, j) has stayed inside.
    let mut p = vec![0.0_f64; n + 1];
    p[0] = if out(0, 0) { 0.0 } else { 1.0 };
    for j in 1..=n {
        p[j] = if out(0, j) { 0.0 } else { p[j - 1] };
    }
    for i in 1..=m {
        let first = if out(i, 0) { 0.0 } else { p[0] };
        p[0] = first;
        for j in 1..=n {
            let total = (i + j) as f64;
            p[j] = if out(i, j) {
                0.0
            } else {
                (i as f64 / total) * p[j] + (j as f64 / total) * p[j - 1]
            };
        }
    }
    (1.0 - p[n]).clamp(0.0, 1.0)
}

/// The limiting upper tail of the two-sided statistic,
/// `1 - K(sqrt(m n / (m + n)) q)`.
pub(crate) fn asymptotic_upper(q: f64, m: usize, n: usize) -> f64 {
    let (m, n) = (m as f64, n as f64);
    let x = (m * n / (m + n)).sqrt() * q;
    kolmogorov_upper(x).clamp(0.0, 1.0)
}

/// `1 - K(x)`, where `K` is the Kolmogorov distribution function.
///
/// Below 1 the theta-function form `K(x) = sqrt(2 pi) / x * sum exp(-(2k-1)^2
/// pi^2 / (8 x^2))` converges fastest. From 1 the alternating form `1 - K(x) =
/// 2 sum (-1)^(k-1) exp(-2 k^2 x^2)` does, and is summed on the upper tail
/// directly so that small tails keep their relative precision. Terms are added
/// until they fall below `1e-17`.
pub(crate) fn kolmogorov_upper(x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if x < 1.0 {
        let z = -(std::f64::consts::PI * std::f64::consts::PI / 8.0) / (x * x);
        let mut sum = 0.0;
        let mut k = 1.0_f64;
        loop {
            let term = (k * k * z).exp();
            sum += term;
            if term < 1e-17 {
                break;
            }
            k += 2.0;
        }
        let k = sum * (2.0 * std::f64::consts::PI).sqrt() / x;
        return 1.0 - k;
    }
    let z = -2.0 * x * x;
    let mut sum = 0.0;
    let mut sign = 1.0;
    let mut k = 1.0_f64;
    loop {
        let term = (z * k * k).exp();
        sum += sign * term;
        if term < 1e-17 {
            break;
        }
        sign = -sign;
        k += 1.0;
    }
    2.0 * sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statistic_counts_ties_once() {
        let d = ks_statistic(&[1.0, 2.0, 2.0], &[2.0, 3.0, 3.0]).unwrap();
        // At 1: 1/3 against 0. At 2: 1 against 1/3. At 3: 1 against 1.
        assert!((d - 2.0 / 3.0).abs() < 1e-15);
    }

    #[test]
    fn identical_samples_give_p_one() {
        let x = [1.0, 2.0, 3.0, 4.0];
        assert!((ks_test(&x, &x).unwrap().1 - 1.0).abs() < 1e-15);
    }

    #[test]
    fn exact_matches_complete_enumeration() {
        // m = 3, n = 3, no ties: 20 arrangements. D takes the value 1 in 2 of
        // them, so P(D >= 1) = 0.1.
        let boundary = vec![true; 7];
        assert!((exact_upper(1.0, 3, 3, &boundary) - 0.1).abs() < 1e-15);
    }

    #[test]
    fn kolmogorov_forms_agree_at_one() {
        let below = kolmogorov_upper(1.0 - 1e-12);
        let above = kolmogorov_upper(1.0);
        assert!((below - above).abs() < 1e-10);
    }
}
