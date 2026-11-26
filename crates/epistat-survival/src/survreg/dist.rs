//! The four standardised error distributions of the location-scale models
//! and the derivatives of a row's log likelihood in them.

use super::tdist;
use epistat_stats::dist::normal;

/// `ln(sqrt(2 pi))`.
const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_8;
/// `1 / sqrt(2 pi)`.
const FRAC_1_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

/// The standardised error distribution: standard minimum extreme value,
/// standard logistic, standard normal, or Student t with the degrees of
/// freedom given.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Base {
    ExtremeValue,
    Logistic,
    Gaussian,
    StudentT(f64),
}

/// The distribution function `F`, the survival function `S = 1 - F` computed
/// without cancellation, the density `f` and its derivative, at one point.
pub(super) struct Tail {
    pub cdf: f64,
    pub sf: f64,
    pub pdf: f64,
    pub dpdf: f64,
}

impl Base {
    /// `ln f(z)` and its first two derivatives.
    pub(super) fn log_density(self, z: f64) -> (f64, f64, f64) {
        match self {
            Self::ExtremeValue => {
                let w = z.exp();
                (z - w, 1.0 - w, -w)
            }
            Self::Logistic => {
                // With a = exp(-|z|): ln f = -|z| - 2 ln(1 + a), the slope is
                // -tanh(z / 2) and the curvature -2 a / (1 + a)^2.
                let a = (-z.abs()).exp();
                let slope = (1.0 - a) / (1.0 + a);
                (
                    -z.abs() - 2.0 * a.ln_1p(),
                    if z > 0.0 { -slope } else { slope },
                    -2.0 * a / ((1.0 + a) * (1.0 + a)),
                )
            }
            Self::Gaussian => (-0.5 * z * z - LN_SQRT_2PI, -z, -1.0),
            Self::StudentT(nu) => tdist::log_density(z, nu),
        }
    }

    pub(super) fn tail(self, z: f64) -> Tail {
        match self {
            Self::ExtremeValue => {
                let w = z.exp();
                if w.is_infinite() {
                    return Tail {
                        cdf: 1.0,
                        sf: 0.0,
                        pdf: 0.0,
                        dpdf: 0.0,
                    };
                }
                let sf = (-w).exp();
                let pdf = w * sf;
                Tail {
                    cdf: -(-w).exp_m1(),
                    sf,
                    pdf,
                    dpdf: pdf * (1.0 - w),
                }
            }
            Self::Logistic => {
                // The exponential is taken of a non-positive number only.
                let a = (-z.abs()).exp();
                let (small, large) = (a / (1.0 + a), 1.0 / (1.0 + a));
                let (cdf, sf) = if z > 0.0 {
                    (large, small)
                } else {
                    (small, large)
                };
                let pdf = a / ((1.0 + a) * (1.0 + a));
                Tail {
                    cdf,
                    sf,
                    pdf,
                    dpdf: pdf * (sf - cdf),
                }
            }
            Self::Gaussian => {
                let pdf = (-0.5 * z * z).exp() * FRAC_1_SQRT_2PI;
                Tail {
                    cdf: normal::cdf(z),
                    sf: normal::cdf(-z),
                    pdf,
                    dpdf: -z * pdf,
                }
            }
            Self::StudentT(nu) => {
                let (cdf, sf) = tdist::cdf_sf(z, nu);
                if z.is_infinite() {
                    return Tail {
                        cdf,
                        sf,
                        pdf: 0.0,
                        dpdf: 0.0,
                    };
                }
                let (h, h1, _) = tdist::log_density(z, nu);
                let pdf = h.exp();
                Tail {
                    cdf,
                    sf,
                    pdf,
                    dpdf: pdf * h1,
                }
            }
        }
    }

    /// The quantile function, for `p` in `(0, 1)`.
    pub(super) fn quantile(self, p: f64) -> f64 {
        match self {
            Self::ExtremeValue => (-(-p).ln_1p()).ln(),
            Self::Logistic => (p / (1.0 - p)).ln(),
            Self::Gaussian => normal::quantile(p),
            Self::StudentT(nu) => tdist::quantile(p, nu),
        }
    }

    /// The location and variance a method-of-moments start takes from the mean
    /// and variance of the times: the mean of the standard distribution
    /// removed, and the variance divided by the variance of the standard
    /// distribution, rounded as `survreg` rounds it (1.64 for pi^2 / 6, 3.2
    /// for pi^2 / 3).
    pub(super) fn start(self, mean: f64, variance: f64) -> (f64, f64) {
        match self {
            Self::ExtremeValue => (mean + 0.572, variance / 1.64),
            Self::Logistic => (mean, variance / 3.2),
            Self::Gaussian => (mean, variance),
            Self::StudentT(nu) => (mean, variance * (nu - 2.0) / nu),
        }
    }

    /// The saturated log likelihood of a row and the centre it is attained at,
    /// for the deviance and response residuals, on the transformed scale.
    /// `sigma` is the row's scale.
    pub(super) fn saturated(self, status: u8, y1: f64, y2: f64, sigma: f64) -> (f64, f64) {
        let interval = status == 3;
        match self {
            Self::ExtremeValue => {
                let width = if interval { (y2 - y1) / sigma } else { 1.0 };
                let temp = width / width.exp_m1();
                let center = if interval { y1 - temp.ln() } else { y1 };
                let loglik = match status {
                    1 => -(1.0 + sigma.ln()),
                    3 => -temp + (-(-width.exp()).exp_m1()).ln(),
                    _ => 0.0,
                };
                (center, loglik)
            }
            Self::Logistic => {
                let width = if interval { (y2 - y1) / sigma } else { 0.0 };
                let center = if interval { 0.5 * (y1 + y2) } else { y1 };
                let loglik = match status {
                    1 => -(4.0 * sigma).ln(),
                    3 => {
                        let e = (0.5 * width).exp();
                        ((e - 1.0) / (e + 1.0)).ln()
                    }
                    _ => 0.0,
                };
                (center, loglik)
            }
            Self::Gaussian => {
                let width = if interval { (y2 - y1) / sigma } else { 0.0 };
                let center = if interval { 0.5 * (y1 + y2) } else { y1 };
                let loglik = match status {
                    1 => -(LN_SQRT_2PI + sigma.ln()),
                    3 => (2.0 * normal::cdf(0.5 * width) - 1.0).ln(),
                    _ => 0.0,
                };
                (center, loglik)
            }
            Self::StudentT(nu) => {
                let width = if interval { (y2 - y1) / sigma } else { 0.0 };
                let center = if interval { 0.5 * (y1 + y2) } else { y1 };
                let loglik = match status {
                    1 => -(tdist::log_density(0.0, nu).0 + sigma.ln()),
                    3 => (2.0 * tdist::cdf_sf(0.5 * width, nu).0 - 1.0).ln(),
                    _ => 0.0,
                };
                (center, loglik)
            }
        }
    }
}

/// The log likelihood of one row and its derivatives in the location `eta`
/// and in `ls`, the log of the scale.
#[derive(Debug, Clone, Copy)]
pub(super) struct RowDerivs {
    pub g: f64,
    pub g_eta: f64,
    pub g_ls: f64,
    pub g_ee: f64,
    pub g_el: f64,
    pub g_ll: f64,
}

impl RowDerivs {
    /// A row with no probability, whose log likelihood is `-inf` and which
    /// has no derivatives.
    const IMPOSSIBLE: Self = Self {
        g: f64::NEG_INFINITY,
        g_eta: 0.0,
        g_ls: 0.0,
        g_ee: 0.0,
        g_el: 0.0,
        g_ll: 0.0,
    };
}

/// The log likelihood of a row with status 1 (exact), 0 (right), 2 (left) or 3
/// (interval from `y1` to `y2`) and its derivatives.
///
/// With `z = (y - eta) / sigma` and `sigma = exp(ls)`, an exact time
/// contributes `ln f(z) - ls`. A censored time contributes `ln T`, with `T`
/// the probability of the censoring set: `S(z)`, `F(z)`, or for an interval
/// `S(z_lower) - S(z_upper)` when the lower bound is above the centre of the
/// distribution and `F(z_upper) - F(z_lower)` otherwise, which keeps the
/// difference of two probabilities from cancelling in a tail. Every `T` is a
/// signed sum of terms `c P(z_j)` with `P` one of `F` and `S`, and the chain
/// rule through `z_j` gives
///
/// ```text
/// T_eta   = -sum c P'(z) / sigma
/// T_ls    = -sum c z P'(z)
/// T_ee    =  sum c P''(z) / sigma^2
/// T_el    =  sum c (z P''(z) + P'(z)) / sigma
/// T_ll    =  sum c (z^2 P''(z) + z P'(z)),
/// ```
///
/// and `g_x = T_x / T`, `g_xy = T_xy / T - g_x g_y`. For an exact time the
/// same chain rule applies to `h(z) = ln f(z)`.
#[expect(
    clippy::similar_names,
    reason = "t_e, t_l and the like are the derivatives of T in eta and ln sigma"
)]
pub(super) fn row_derivs(base: Base, status: u8, y1: f64, y2: f64, eta: f64, ls: f64) -> RowDerivs {
    let sigma = ls.exp();
    let z = (y1 - eta) / sigma;
    if status == 1 {
        let (h, h1, h2) = base.log_density(z);
        return RowDerivs {
            g: h - ls,
            g_eta: -h1 / sigma,
            g_ls: -h1 * z - 1.0,
            g_ee: h2 / (sigma * sigma),
            g_el: (h2 * z + h1) / sigma,
            g_ll: h2 * z * z + h1 * z,
        };
    }
    // Terms of the probability: sign, whether it is F (else S), point.
    let zu = (y2 - eta) / sigma;
    let (terms, count): ([(f64, bool, f64); 2], usize) = match status {
        0 => ([(1.0, false, z), (0.0, false, 0.0)], 1),
        2 => ([(1.0, true, z), (0.0, false, 0.0)], 1),
        _ if z > 0.0 => ([(1.0, false, z), (-1.0, false, zu)], 2),
        _ => ([(1.0, true, zu), (-1.0, true, z)], 2),
    };
    let (mut t, mut t_e, mut t_l, mut t_ee, mut t_el, mut t_ll) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for &(c, is_cdf, at) in &terms[..count] {
        let tail = base.tail(at);
        let (p, p1, p2) = if is_cdf {
            (tail.cdf, tail.pdf, tail.dpdf)
        } else {
            (tail.sf, -tail.pdf, -tail.dpdf)
        };
        t += c * p;
        t_e += -c * p1 / sigma;
        t_l += -c * at * p1;
        t_ee += c * p2 / (sigma * sigma);
        t_el += c * (at * p2 + p1) / sigma;
        t_ll += c * (at * at * p2 + at * p1);
    }
    // `!(t > 0.0)` also catches NaN.
    #[expect(
        clippy::neg_cmp_op_on_partial_ord,
        reason = "NaN must take the same branch as zero"
    )]
    if !(t > 0.0) {
        return RowDerivs::IMPOSSIBLE;
    }
    let g_eta = t_e / t;
    let g_ls = t_l / t;
    RowDerivs {
        g: t.ln(),
        g_eta,
        g_ls,
        g_ee: t_ee / t - g_eta * g_eta,
        g_el: t_el / t - g_eta * g_ls,
        g_ll: t_ll / t - g_ls * g_ls,
    }
}

#[cfg(test)]
mod tests {
    use super::{Base, row_derivs};

    /// Every derivative of a row against central differences of its log
    /// likelihood. The step `h` makes the truncation error of a central
    /// difference, about `h^2 f'''/6`, and the rounding error, about
    /// `eps f / h` for a first and `eps f / h^2` for a second derivative,
    /// equal near `h = eps^(1/4)` for the second derivatives, so the check
    /// allows `1e-5` relative to the larger of the value and 1.
    #[test]
    fn derivatives_match_central_differences() {
        let h = 1e-4;
        for base in [
            Base::ExtremeValue,
            Base::Logistic,
            Base::Gaussian,
            Base::StudentT(4.0),
            Base::StudentT(2.5),
            Base::StudentT(30.0),
        ] {
            for (status, y1, y2) in [
                (1_u8, 1.3, 0.0),
                (0, 1.3, 0.0),
                (2, 1.3, 0.0),
                (3, 0.4, 1.9),
                (3, -2.1, -0.5),
                (3, 1.0, 3.5),
            ] {
                let (eta, ls) = (0.7, -0.2);
                let g = |e: f64, l: f64| row_derivs(base, status, y1, y2, e, l).g;
                let d = row_derivs(base, status, y1, y2, eta, ls);
                let numeric = [
                    (g(eta + h, ls) - g(eta - h, ls)) / (2.0 * h),
                    (g(eta, ls + h) - g(eta, ls - h)) / (2.0 * h),
                    (g(eta + h, ls) - 2.0 * g(eta, ls) + g(eta - h, ls)) / (h * h),
                    (g(eta + h, ls + h) - g(eta + h, ls - h) - g(eta - h, ls + h)
                        + g(eta - h, ls - h))
                        / (4.0 * h * h),
                    (g(eta, ls + h) - 2.0 * g(eta, ls) + g(eta, ls - h)) / (h * h),
                ];
                let exact = [d.g_eta, d.g_ls, d.g_ee, d.g_el, d.g_ll];
                for (n, e) in numeric.iter().zip(exact) {
                    assert!(
                        (n - e).abs() <= 1e-5 * e.abs().max(1.0),
                        "{base:?} status {status}: {n} against {e}"
                    );
                }
            }
        }
    }
}
