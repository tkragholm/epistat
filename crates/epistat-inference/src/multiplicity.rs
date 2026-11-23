//! Adjustment of a vector of p-values for several tests.
//!
//! [`adjust`] gives the Bonferroni, Holm \[1\], Hochberg \[2\], Hommel \[3\],
//! Benjamini-Hochberg \[4\] and Benjamini-Yekutieli \[5\] adjusted p-values,
//! with the arithmetic of `stats::p.adjust` in R. With `n` the number of
//! comparisons, `p_(1) <= ... <= p_(k)` the sorted p-values and `k` their
//! number, the adjusted value of the `i`th smallest is
//!
//! ```text
//! Bonferroni            min(1, n p_(i))
//! Holm                  min(1, max_{j <= i} (n + 1 - j) p_(j))
//! Hochberg              min(1, min_{j >= i} (n + 1 - (k + 1 - j)) p_(j))
//! Benjamini-Hochberg    min(1, min_{j >= i} (n / j) p_(j))
//! Benjamini-Yekutieli   min(1, min_{j >= i} (q n / j) p_(j)),    q = sum_{m=1}^{n} 1/m
//! ```
//!
//! Holm is a step-down and Hochberg a step-up procedure, so one is a running
//! maximum from the smallest p-value and the other a running minimum from the
//! largest. Hommel's procedure has no short form. It pads the p-values with
//! ones to `n`, and for `m` from `n - 1` down to 2 takes the smallest of
//! `m p_i / i` over the `m - 1` largest p-values for the `n - m + 1` smallest,
//! as the loop of `p.adjust` does, and returns the larger of the running
//! maximum and `p`. A set of one comparison is returned unchanged and Hommel
//! with two comparisons is Hochberg's, as in R.
//!
//! The p-values are sorted stably, so ties keep their input order, and tied
//! p-values receive equal adjusted values under Holm, Hochberg and the two
//! false discovery rate methods. The result is in the order of the input.
//! `n` may exceed the number of p-values, for tests whose p-values are not
//! in the vector.
//!
//! # Missing values
//!
//! R drops a missing p-value and the number of comparisons with it. A NaN here
//! is an error, since dropping it changes `n` without the caller saying so.
//!
//! # Cost
//!
//! The sort is a radix sort on the bits of the p-values, so Bonferroni, Holm,
//! Hochberg and the two false discovery rate methods are `O(k)`. Hommel's
//! loop is `O(n^2)`. Its steps are independent up to a running maximum, which
//! is exact, so they are spread over the threads and the result does not
//! depend on their number. It takes about 0.01 s at 10^4 tests and 1.3 s at
//! 10^5 on six cores and 12 threads, and is unsuitable beyond about 10^6.
//!
//! # Accuracy
//!
//! Every method but Benjamini-Yekutieli is the same sequence of products and
//! comparisons as R's and agrees to the bit. The harmonic sum `q` is added in
//! order from 1, as R adds it, and differs from R's only if R accumulates in
//! extended precision, by at most `n` units in the last place of `q`.
//!
//! # References
//!
//! 1. Holm S. A simple sequentially rejective multiple test procedure. Scand J
//!    Stat 1979;6:65-70.
//! 2. Hochberg Y. A sharper Bonferroni procedure for multiple tests of
//!    significance. Biometrika 1988;75:800-802.
//! 3. Hommel G. A stagewise rejective multiple test procedure based on a
//!    modified Bonferroni test. Biometrika 1988;75:383-386.
//! 4. Benjamini Y, Hochberg Y. Controlling the false discovery rate: a
//!    practical and powerful approach to multiple testing. J R Stat Soc B
//!    1995;57:289-300.
//! 5. Benjamini Y, Yekutieli D. The control of the false discovery rate in
//!    multiple testing under dependency. Ann Stat 2001;29:1165-1188.

use rayon::prelude::*;

use crate::InferenceError;

/// A method of adjusting p-values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Adjustment {
    /// Multiplication by the number of comparisons.
    Bonferroni,
    /// Holm's step-down procedure.
    Holm,
    /// Hochberg's step-up procedure.
    Hochberg,
    /// Hommel's closed procedure.
    Hommel,
    /// The false discovery rate of Benjamini and Hochberg.
    BenjaminiHochberg,
    /// The false discovery rate of Benjamini and Yekutieli, valid under any
    /// dependence.
    BenjaminiYekutieli,
}

/// The adjusted p-values of `p`, in the order of `p`, for `n` comparisons, or
/// for `p.len()` comparisons when `n` is `None`.
///
/// # Errors
///
/// [`InferenceError::PValue`] for a p-value that is NaN or outside `[0, 1]`
/// and [`InferenceError::Comparisons`] when `n` is below the number of
/// p-values.
pub fn adjust(p: &[f64], method: Adjustment, n: Option<usize>) -> Result<Vec<f64>, InferenceError> {
    if let Some(&bad) = p.iter().find(|v| !(0.0..=1.0).contains(*v)) {
        return Err(InferenceError::PValue(bad));
    }
    let lp = p.len();
    let n = n.unwrap_or(lp);
    if n < lp {
        return Err(InferenceError::Comparisons {
            comparisons: n,
            p_values: lp,
        });
    }
    if n <= 1 {
        return Ok(p.to_vec());
    }
    let method = if n == 2 && method == Adjustment::Hommel {
        Adjustment::Hochberg
    } else {
        method
    };
    let nf = n as f64;
    Ok(match method {
        Adjustment::Bonferroni => p.iter().map(|&v| (nf * v).min(1.0)).collect(),
        Adjustment::Holm => {
            let mut running = f64::NEG_INFINITY;
            let mut out = vec![0.0; lp];
            for (rank, &(key, index)) in sorted_keys(p, false).iter().enumerate() {
                running = running.max((nf + 1.0 - (rank + 1) as f64) * key_value(key));
                out[index] = running.min(1.0);
            }
            out
        }
        Adjustment::Hochberg => step_up(p, |i| nf + 1.0 - i),
        Adjustment::BenjaminiHochberg => step_up(p, |i| nf / i),
        Adjustment::BenjaminiYekutieli => {
            let mut q = 0.0;
            for k in 1..=n {
                q += 1.0 / k as f64;
            }
            step_up(p, |i| q * nf / i)
        }
        Adjustment::Hommel => hommel(p, n),
    })
}

/// A p-value in the order of [`f64::total_cmp`], as an unsigned integer.
fn order_key(v: f64) -> u64 {
    let bits = v.to_bits();
    if bits >> 63 == 1 {
        !bits
    } else {
        bits | 1 << 63
    }
}

/// The p-value of an [`order_key`].
fn key_value(key: u64) -> f64 {
    f64::from_bits(if key >> 63 == 1 {
        key & !(1 << 63)
    } else {
        !key
    })
}

/// Bits of one pass of the radix sort.
const DIGIT_BITS: u32 = 11;
const BUCKETS: usize = 1 << DIGIT_BITS;

/// The keys of `p` with their positions, sorted by key ascending, or by key
/// descending when `descending`, ties in input order.
///
/// A least-significant-digit radix sort on the key is stable, so ties keep
/// their input order for either direction (a descending sort sorts the
/// complemented key). It does `O(k)` work in sequential passes where a
/// comparison sort on the positions reads `p` at random; a pass is skipped
/// when every key has the same digit, as for the high bits of values in
/// `[0, 1]`.
fn sorted_keys(p: &[f64], descending: bool) -> Vec<(u64, usize)> {
    let mut items: Vec<(u64, usize)> = p
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let key = order_key(v);
            (if descending { !key } else { key }, i)
        })
        .collect();
    let passes = u64::BITS.div_ceil(DIGIT_BITS) as usize;
    let digit =
        |key: u64, pass: usize| ((key >> (pass as u32 * DIGIT_BITS)) as usize) & (BUCKETS - 1);
    let mut counts = vec![[0usize; BUCKETS]; passes];
    for &(key, _) in &items {
        for (pass, count) in counts.iter_mut().enumerate() {
            count[digit(key, pass)] += 1;
        }
    }
    let mut buffer = vec![(0u64, 0usize); items.len()];
    for (pass, count) in counts.iter_mut().enumerate() {
        if count.iter().any(|&c| c == items.len()) {
            continue;
        }
        let mut total = 0;
        for c in count.iter_mut() {
            (*c, total) = (total, total + *c);
        }
        for &item in &items {
            let slot = &mut count[digit(item.0, pass)];
            buffer[*slot] = item;
            *slot += 1;
        }
        std::mem::swap(&mut items, &mut buffer);
    }
    if descending {
        for item in &mut items {
            item.0 = !item.0;
        }
    }
    items
}

/// A running minimum from the largest p-value of `factor(i) * p`, with `i`
/// running from the number of p-values down to 1 as R's `i <- lp:1`, capped
/// at 1.
fn step_up(p: &[f64], factor: impl Fn(f64) -> f64) -> Vec<f64> {
    let lp = p.len();
    let mut running = f64::INFINITY;
    let mut out = vec![0.0; lp];
    for (step, &(key, index)) in sorted_keys(p, true).iter().enumerate() {
        let i = (lp - step) as f64;
        running = running.min(factor(i) * key_value(key));
        out[index] = running.min(1.0);
    }
    out
}

/// The steps of Hommel's loop for `j` from `n - 1` down to 2 that the
/// range `steps` covers, as a running maximum into `pa`.
///
/// Step `j` takes `q1`, the smallest of `j p_i / k` over the `j - 1` largest
/// p-values with `k` from 2, sets the adjusted value of the `n - j + 1`
/// smallest to `min(j p_i, q1)` and of the others to the value of the last of
/// those, and raises `pa` to it. A maximum is exact and does not depend on
/// the order of the steps, so ranges can be done apart and combined.
fn hommel_steps(sorted: &[f64], steps: std::ops::Range<usize>, pa: &mut [f64]) {
    let n = sorted.len();
    for j in steps.rev() {
        let jf = j as f64;
        let q1 = sorted[n - j + 1..]
            .iter()
            .enumerate()
            .map(|(k, &v)| jf * v / (k + 2) as f64)
            .fold(f64::INFINITY, f64::min);
        for (slot, &v) in pa[..=n - j].iter_mut().zip(sorted) {
            *slot = slot.max((jf * v).min(q1));
        }
        let fill = (jf * sorted[n - j]).min(q1);
        for slot in &mut pa[n - j + 1..] {
            *slot = slot.max(fill);
        }
    }
}

/// Below this many comparisons Hommel's loop runs on one thread.
const HOMMEL_PARALLEL_FROM: usize = 512;

fn hommel(p: &[f64], n: usize) -> Vec<f64> {
    let lp = p.len();
    let mut padded = p.to_vec();
    padded.resize(n, 1.0);
    let order: Vec<usize> = sorted_keys(&padded, false)
        .iter()
        .map(|&(_, i)| i)
        .collect();
    let sorted: Vec<f64> = order.iter().map(|&i| padded[i]).collect();
    let nf = n as f64;
    let start = sorted
        .iter()
        .enumerate()
        .map(|(k, &v)| nf * v / (k + 1) as f64)
        .fold(f64::INFINITY, f64::min);
    let mut pa = vec![start; n];
    if n < HOMMEL_PARALLEL_FROM {
        hommel_steps(&sorted, 2..n, &mut pa);
    } else {
        let chunks = 4 * rayon::current_num_threads();
        let size = (n - 2).div_ceil(chunks);
        let partial: Vec<Vec<f64>> = (0..chunks)
            .into_par_iter()
            .map(|c| {
                let from = 2 + c * size;
                let mut local = vec![start; n];
                hommel_steps(&sorted, from.min(n)..(from + size).min(n), &mut local);
                local
            })
            .collect();
        for local in &partial {
            for (a, &b) in pa.iter_mut().zip(local) {
                *a = a.max(b);
            }
        }
    }
    let mut out = vec![0.0; n];
    for (rank, &index) in order.iter().enumerate() {
        out[index] = pa[rank].max(sorted[rank]);
    }
    out.truncate(lp);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: [f64; 5] = [0.01, 0.04, 0.03, 0.005, 0.2];

    #[test]
    fn bonferroni_and_holm_by_hand() {
        let b = adjust(&P, Adjustment::Bonferroni, None).unwrap();
        assert_eq!(b, vec![0.05, 0.2, 0.15, 0.025, 1.0]);
        let h = adjust(&P, Adjustment::Holm, None).unwrap();
        // Sorted: 0.005 * 5, 0.01 * 4, 0.03 * 3, 0.04 * 2, 0.2 * 1.
        assert_eq!(h, vec![0.04, 0.09, 0.09, 0.025, 0.2]);
    }

    #[test]
    fn single_p_value_is_unchanged_and_ties_stay_tied() {
        for m in [
            Adjustment::Holm,
            Adjustment::Hommel,
            Adjustment::BenjaminiYekutieli,
        ] {
            assert_eq!(adjust(&[0.3], m, None).unwrap(), vec![0.3]);
        }
        let tied = adjust(&[0.02, 0.02, 0.5], Adjustment::BenjaminiHochberg, None).unwrap();
        assert_eq!(tied[0].to_bits(), tied[1].to_bits());
    }

    /// Hommel's loop as `p.adjust` writes it, step by step on whole vectors.
    fn hommel_reference(p: &[f64], n: usize) -> Vec<f64> {
        let lp = p.len();
        let mut padded = p.to_vec();
        padded.resize(n, 1.0);
        let order: Vec<usize> = reference_order(&padded, false);
        let sorted: Vec<f64> = order.iter().map(|&i| padded[i]).collect();
        let nf = n as f64;
        let start = sorted
            .iter()
            .enumerate()
            .map(|(k, &v)| nf * v / (k + 1) as f64)
            .fold(f64::INFINITY, f64::min);
        let mut q = vec![start; n];
        let mut pa = vec![start; n];
        for j in (2..n).rev() {
            let jf = j as f64;
            // 0-based positions: ij = 0..n-j+1, i2 = n-j+1..n.
            let q1 = (n - j + 1..n)
                .enumerate()
                .map(|(k, pos)| jf * sorted[pos] / (k + 2) as f64)
                .fold(f64::INFINITY, f64::min);
            for pos in 0..=(n - j) {
                q[pos] = (jf * sorted[pos]).min(q1);
            }
            let fill = q[n - j];
            for slot in &mut q[n - j + 1..n] {
                *slot = fill;
            }
            for (a, &b) in pa.iter_mut().zip(&q) {
                *a = a.max(b);
            }
        }
        let mut out = vec![0.0; n];
        for (rank, &index) in order.iter().enumerate() {
            out[index] = pa[rank].max(sorted[rank]);
        }
        out.truncate(lp);
        out
    }

    /// A stable comparison sort of the positions, the order `order(p)` gives.
    fn reference_order(p: &[f64], descending: bool) -> Vec<usize> {
        let mut order: Vec<usize> = (0..p.len()).collect();
        if descending {
            order.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
        } else {
            order.sort_by(|&a, &b| p[a].total_cmp(&p[b]));
        }
        order
    }

    #[test]
    fn radix_sort_is_the_stable_comparison_sort() {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for len in [0, 1, 2, 7, 300, 5000] {
            for kind in 0..4 {
                let p: Vec<f64> = (0..len)
                    .map(|_| {
                        let u = (next() >> 11) as f64 / (1u64 << 53) as f64;
                        match kind {
                            0 => u,
                            // Many ties, including both zeros and one.
                            1 => [0.0, -0.0, 1.0, 0.5, 0.25][(next() % 5) as usize],
                            2 => (u * 20.0).floor() / 20.0,
                            _ => u.powi(8) * f64::MIN_POSITIVE.max(1e-300) * 1e300,
                        }
                    })
                    .collect();
                for descending in [false, true] {
                    let got = sorted_keys(&p, descending);
                    let expected = reference_order(&p, descending);
                    assert_eq!(got.iter().map(|&(_, i)| i).collect::<Vec<_>>(), expected);
                    for &(key, i) in &got {
                        assert_eq!(key_value(key).to_bits(), p[i].to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn hommel_is_the_loop_of_p_adjust() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for (len, extra) in [(3, 0), (40, 5), (600, 0), (700, 90), (1500, 0)] {
            for kind in 0..3 {
                let p: Vec<f64> = (0..len)
                    .map(|_| {
                        let u = (next() >> 11) as f64 / (1u64 << 53) as f64;
                        match kind {
                            0 => u,
                            1 => u.powi(5),
                            _ => (u * 25.0).floor() / 25.0,
                        }
                    })
                    .collect();
                let n = len + extra;
                let got = hommel(&p, n);
                let expected = hommel_reference(&p, n);
                assert!(
                    got.iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            }
        }
    }

    #[test]
    fn refuses_bad_input() {
        assert!(matches!(
            adjust(&[0.1, f64::NAN], Adjustment::Holm, None),
            Err(InferenceError::PValue(v)) if v.is_nan()
        ));
        assert!(adjust(&[1.5], Adjustment::Holm, None).is_err());
        assert!(adjust(&[0.1, 0.2, 0.3], Adjustment::Holm, Some(2)).is_err());
    }
}
