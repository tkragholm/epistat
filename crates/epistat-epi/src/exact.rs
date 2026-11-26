//! The two-sided exact tests of R's `poisson.test` and `binom.test`.
//!
//! # Algorithm
//!
//! The p-value of an observed count `x` is the total probability of the
//! outcomes whose point probability is not above that of `x`, within a
//! relative tolerance of `1e-7`, as R collects them. On the side of the mean
//! opposite `x`, those outcomes form a tail beyond some count, which is found
//! by counting the outcomes up to the mean (or from the mean up to a point
//! where the probabilities have fallen below that of `x`). The p-value is the
//! tail of `x` plus that other tail, capped at 1, with both tails from
//! [`epistat_stats::dist`].

use epistat_stats::dist::{binomial, poisson};

/// The relative tolerance on the point probability that the exact tests of R
/// allow when they collect the outcomes that are not more likely than the
/// observed one.
pub(crate) const REL_ERR: f64 = 1.0 + 1e-7;

/// The two-sided p-value of an exact Poisson test of `x` against the mean
/// `m > 0`, as `stats::poisson.test`: the total probability of the outcomes
/// with a probability not above that of `x`, within [`REL_ERR`].
#[expect(
    clippy::float_cmp,
    reason = "R takes the observed count at the mean as the tie of the test"
)]
pub(crate) fn poisson_two_sided(x: f64, m: f64) -> f64 {
    let d = poisson::pmf(x, m);
    if x == m {
        return 1.0;
    }
    let p = if x < m {
        let mut top = (2.0 * m - x).ceil();
        while poisson::pmf(top, m) > d {
            top *= 2.0;
        }
        let mut count = 0.0;
        let mut i = m.ceil();
        while i <= top {
            if poisson::pmf(i, m) <= d * REL_ERR {
                count += 1.0;
            }
            i += 1.0;
        }
        poisson::cdf(x, m) + poisson::sf(top - count, m)
    } else {
        let mut count = 0.0;
        let mut i = 0.0;
        while i <= m.floor() {
            if poisson::pmf(i, m) <= d * REL_ERR {
                count += 1.0;
            }
            i += 1.0;
        }
        poisson::cdf(count - 1.0, m) + poisson::sf(x - 1.0, m)
    };
    p.min(1.0)
}

/// The two-sided p-value of an exact binomial test of `x` successes in `n`
/// trials against the probability `p` in `(0, 1)`, as `stats::binom.test`.
#[expect(
    clippy::float_cmp,
    reason = "R takes the observed count at the mean as the tie of the test"
)]
pub(crate) fn binomial_two_sided(x: f64, n: f64, p: f64) -> f64 {
    let d = binomial::pmf(x, n, p);
    let m = n * p;
    if x == m {
        return 1.0;
    }
    let value = if x < m {
        let mut count = 0.0;
        let mut i = m.ceil();
        while i <= n {
            if binomial::pmf(i, n, p) <= d * REL_ERR {
                count += 1.0;
            }
            i += 1.0;
        }
        binomial::cdf(x, n, p) + binomial::sf(n - count, n, p)
    } else {
        let mut count = 0.0;
        let mut i = 0.0;
        while i <= m.floor() {
            if binomial::pmf(i, n, p) <= d * REL_ERR {
                count += 1.0;
            }
            i += 1.0;
        }
        binomial::cdf(count - 1.0, n, p) + binomial::sf(x - 1.0, n, p)
    };
    value.min(1.0)
}
