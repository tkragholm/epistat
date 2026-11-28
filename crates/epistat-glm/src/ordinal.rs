//! Cumulative link models for an ordered outcome, with the proportional odds
//! model as the main case, and the Brant test of proportional odds.
//!
//! The outcome `y` takes the categories `0, ..., K - 1` in order. With a
//! distribution function `F` (the link), thresholds `ζ_0 < ... < ζ_{K-2}` and
//! the linear predictor `η = x'β + offset`,
//!
//! ```text
//! P(y <= k | x) = F(ζ_k - η),        k = 0, ..., K - 2
//! P(y = k | x)  = F(ζ_k - η) - F(ζ_{k-1} - η),   ζ_{-1} = -∞, ζ_{K-1} = ∞
//! ```
//!
//! [`OrdinalLink::Logit`] is the proportional odds model, whose `exp(β)` is
//! the odds ratio of a higher category and the same at every threshold \[1\].
//! [`OrdinalLink::Probit`] has the normal, [`OrdinalLink::Cloglog`] the
//! extreme-value distribution `F(z) = 1 - exp(-exp(z))` and
//! [`OrdinalLink::Loglog`] `F(z) = exp(-exp(-z))`. The parametrisation is that
//! of `MASS::polr` and `ordinal::clm`.
//!
//! Category-specific columns, a second matrix `W` given to
//! [`OrdinalData`]'s builder, have a coefficient for each threshold, `η_k = x'β + w'τ_k`, the partial
//! proportional odds model of Peterson and Harrell \[2\].
//!
//! # Algorithm
//!
//! The log-likelihood with case weights `w` is `l = Σ w ln P(y_i | x_i)`. For
//! the logit, probit and extreme-value links the probability of an interval
//! under a log-concave density is log-concave in its end points, so `l` is
//! concave in `(ζ, β)` \[3\]. Writing `u = ζ_y - η_y` and `v = ζ_{y-1} -
//! η_{y-1}` for the ends of the interval of the observed category and `f = F'`,
//!
//! ```text
//! ∂l/∂u = f(u) / P          ∂l/∂v = -f(v) / P
//! ∂²l/∂u²  = f'(u) / P - (f(u) / P)²
//! ∂²l/∂v²  = -f'(v) / P - (f(v) / P)²
//! ∂²l/∂u∂v = f(u) f(v) / P²
//! ```
//!
//! and `u` and `v` are linear in the parameters, so the score and Hessian are
//! sums of these terms times outer products of sparse vectors. Newton's method
//! starts from the thresholds that reproduce the marginal cumulative
//! proportions, `ζ_k = F^{-1}(cumulative share up to k)`, and `β = 0`. A step
//! that does not raise `l`, or that gives a non-positive probability to any
//! observed category, which includes thresholds that are not increasing, is
//! halved, at most 30 times. The fit has converged when the largest step is
//! below `tolerance * (1 + largest |parameter|)`, or when no step raises `l`
//! and the Newton step is below `sqrt(eps) * (1 + largest |parameter|)`. The
//! partial model starts from the proportional odds fit with `τ = 0`.
//!
//! `P` is the difference of two distribution values, computed where it
//! cannot cancel: for the symmetric logit and probit as `F(-v) - F(-u)` when
//! `v >= 0`, and for the extreme-value links as `exp(-e^v) (-expm1(e^v -
//! e^u))` and its mirror image.
//!
//! The covariance is the inverse information at the maximum, thresholds
//! first, then the coefficients, then the category-specific coefficients
//! threshold by threshold. With `cluster` given, the robust covariance
//! `I^{-1} (Σ_g s_g s_g') I^{-1}` is formed from the score `s_g` summed over
//! the rows of cluster `g`, without a small-sample factor. The scores of all
//! rows are held in memory for it, `n (K - 1 + p)` values.
//!
//! # Edge cases
//!
//! A design with a constant column is an error: the thresholds are the
//! intercepts. A column that is a linear combination of the columns before
//! it, with the constant counted, is aliased: its coefficient and covariance
//! are `None` as in [`Glm`]. A category with no row of positive
//! weight is an error, since its threshold would diverge. Complete or
//! quasi-complete separation sends a coefficient to infinity: the information
//! becomes singular, which is [`Error::Separation`] when an observed
//! category has probability within ten machine epsilons of 1 or 0, or the
//! iteration reaches its limit with [`OrdinalFit::converged`] false. The null
//! log-likelihood is that of the thresholds alone, without covariates or
//! offset. In the partial model the probabilities of a category can be
//! negative for covariate values away from the data, and
//! [`OrdinalFit::ordering_violations`] counts the rows of the data where it
//! happens.
//!
//! # Brant test
//!
//! [`brant`] fits the `K - 1` binary logistic models of `y > k` and tests that
//! each covariate has the same coefficient in all of them \[4\]. The
//! covariance of the stacked estimates `b_k` uses the model-based moments,
//! for `j <= k` with `π_k = P(y > k)` from the binary fit of `k`
//!
//! ```text
//! Cov(b_j, b_k) = (X'W_j X)^-1 (X' W_jk X) (X'W_k X)^-1
//! W_j = diag(w π_j (1 - π_j))        W_jk = diag(w π_k (1 - π_j))
//! ```
//!
//! and the Wald statistic of the `K - 2` differences `b_k - b_{k+1}` of a
//! covariate has `K - 2` degrees of freedom, that of all covariates together
//! `p (K - 2)`. It is defined for the logit link only.
//!
//! # Determinism and cost
//!
//! Sums run over fixed blocks of rows added in block order, so results do not
//! depend on the number of threads. A Newton iteration costs
//! `O(n (K + p)²)` for the proportional odds model.
//!
//! The fit is checked against `ordinal::clm` 2026.7.26 (Newton with analytic
//! derivatives), `MASS::polr` 7.3.66, `rms::orm` 8.2.1 and `VGAM::vglm` 1.1.14,
//! and the Brant test against a transcription of Brant's formulas.
//!
//! # References
//!
//! 1. McCullagh P. Regression models for ordinal data. J R Stat Soc Ser B.
//!    1980;42(2):109-42.
//! 2. Peterson B, Harrell FE Jr. Partial proportional odds models for ordinal
//!    response variables. J R Stat Soc Ser C. 1990;39(2):205-17.
//! 3. Pratt JW. Concavity of the log likelihood. J Am Stat Assoc.
//!    1981;76(373):103-6.
//! 4. Brant R. Assessing proportionality in the proportional odds model for
//!    ordinal logistic regression. Biometrics. 1990;46(4):1171-8.

#![expect(clippy::float_cmp, reason = "a constant column holds one exact value")]

use epistat_core::Covariance;
use epistat_inference::Clusters;
use epistat_inference::sandwich::{meat, sandwich_with_inverse};
use epistat_inference::wald::{Wald, wald};
use epistat_linalg::design::independent_columns;
use epistat_linalg::{Attempt, Factor, Mat, MatRef, Method, weighted_gram};
use epistat_stats::dist::normal;
use rayon::prelude::*;

use crate::fit::covariance;
use crate::{Error, Family, Glm, GlmData, Link, OrdinalData, expit};

/// Rows per block of the sums.
const ROWS: usize = 1024;

/// The distribution function `F` of the cumulative link model.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrdinalLink {
    /// The logistic distribution: the proportional odds model.
    Logit,
    /// The standard normal distribution.
    Probit,
    /// `F(z) = 1 - exp(-exp(z))`.
    Cloglog,
    /// `F(z) = exp(-exp(-z))`.
    Loglog,
}

impl OrdinalLink {
    fn cdf(self, z: f64) -> f64 {
        match self {
            Self::Logit => expit(z),
            Self::Probit => normal::cdf(z),
            Self::Cloglog => -(-z.exp()).exp_m1(),
            Self::Loglog => (-(-z).exp()).exp(),
        }
    }

    /// The density `f = F'`, 0 at infinite arguments.
    fn density(self, z: f64) -> f64 {
        if z.is_infinite() {
            return 0.0;
        }
        match self {
            Self::Logit => {
                let e = (-z.abs()).exp();
                e / ((1.0 + e) * (1.0 + e))
            }
            Self::Probit => normal::pdf(z),
            Self::Cloglog => (z - z.exp()).exp(),
            Self::Loglog => (-z - (-z).exp()).exp(),
        }
    }

    /// `f'`, 0 at infinite arguments.
    fn density_derivative(self, z: f64) -> f64 {
        if z.is_infinite() {
            return 0.0;
        }
        let f = self.density(z);
        match self {
            Self::Logit => f * (1.0 - 2.0 * expit(z)),
            Self::Probit => -z * f,
            Self::Cloglog => f * (1.0 - z.exp()),
            Self::Loglog => f * ((-z).exp() - 1.0),
        }
    }

    fn quantile(self, p: f64) -> f64 {
        match self {
            Self::Logit => (p / (1.0 - p)).ln(),
            Self::Probit => normal::quantile(p),
            Self::Cloglog => (-(1.0 - p).ln()).ln(),
            Self::Loglog => -(-p.ln()).ln(),
        }
    }

    /// `F(u) - F(v)`, without cancellation in the tails.
    fn interval(self, v: f64, u: f64) -> f64 {
        match self {
            Self::Logit | Self::Probit => {
                if v >= 0.0 {
                    self.cdf(-v) - self.cdf(-u)
                } else {
                    self.cdf(u) - self.cdf(v)
                }
            }
            Self::Cloglog => (-v.exp()).exp() * -(v.exp() - u.exp()).exp_m1(),
            Self::Loglog => (-(-u).exp()).exp() * -((-u).exp() - (-v).exp()).exp_m1(),
        }
    }
}

/// Fitting controls for the cumulative link model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ordinal {
    /// The distribution function.
    pub link: OrdinalLink,
    /// The largest step, relative to `1 + max |parameter|`, at which the fit
    /// has converged.
    pub tolerance: f64,
    /// The most Newton iterations.
    pub max_iterations: usize,
}

impl Ordinal {
    /// A model with tolerance `1e-10` and 100 iterations.
    #[must_use]
    pub const fn new(link: OrdinalLink) -> Self {
        Self {
            link,
            tolerance: 1e-10,
            max_iterations: 100,
        }
    }
}

/// A fitted cumulative link model.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct OrdinalFit {
    /// The link of the fit.
    pub link: OrdinalLink,
    /// The number of categories `K`.
    pub classes: usize,
    /// The `K - 1` thresholds.
    pub thresholds: Vec<f64>,
    /// The coefficients of the columns of `x`, `None` for an aliased column.
    pub coefficients: Vec<Option<f64>>,
    /// The category-specific coefficients `τ`, row-major `(K - 1)` by the
    /// number of columns of `W`, empty for the proportional odds model.
    pub nominal: Vec<f64>,
    /// The covariance of the parameters, with the thresholds first, then the
    /// coefficients and then `τ`, `None` in the rows and columns of an aliased
    /// coefficient.
    pub covariance: Covariance,
    /// The cluster-robust covariance, in the same layout, when clusters were
    /// given.
    pub robust_covariance: Option<Covariance>,
    /// The log-likelihood at the fit.
    pub log_likelihood: f64,
    /// The log-likelihood of the thresholds alone.
    pub null_log_likelihood: f64,
    /// The number of Newton iterations.
    pub iterations: usize,
    /// Whether the largest step fell below the tolerance.
    pub converged: bool,
    /// Whether an observed category has probability within ten machine
    /// epsilons of 0 or 1, which marks separation.
    pub fitted_at_boundary: bool,
    /// The number of rows for which some category has a negative
    /// probability, which can happen in the partial model only.
    pub ordering_violations: usize,
}

impl OrdinalFit {
    /// The standard errors of all parameters in the order of
    /// [`covariance`](Self::covariance), `None` for an aliased coefficient.
    #[must_use]
    pub fn standard_errors(&self) -> Vec<Option<f64>> {
        self.covariance.standard_errors()
    }

    /// The probabilities of the categories for the rows of `x`, row-major `n`
    /// by `K`, with `nominal` the category-specific columns of a partial model.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when `x` has another number of columns than the fit,
    /// `nominal` is missing for a partial model or has the wrong shape, or
    /// `offset` has another number of rows than `x`.
    pub fn probabilities(
        &self,
        x: MatRef<'_, f64>,
        nominal: Option<MatRef<'_, f64>>,
        offset: Option<&[f64]>,
    ) -> Result<Vec<f64>, Error> {
        let k1 = self.classes - 1;
        let q = self.nominal.len().checked_div(k1).unwrap_or(0);
        crate::error::length("x columns", self.coefficients.len(), x.ncols())?;
        if let Some(o) = offset {
            crate::error::length("offset", x.nrows(), o.len())?;
        }
        if q > 0 {
            let w = nominal.ok_or(epistat_core::InputError::Empty { what: "nominal" })?;
            crate::error::length("nominal", x.nrows(), w.nrows())?;
            crate::error::length("nominal columns", q, w.ncols())?;
        }
        let mut out = Vec::with_capacity(x.nrows() * self.classes);
        for i in 0..x.nrows() {
            let mut xb = offset.map_or(0.0, |o| o[i]);
            for (j, b) in self.coefficients.iter().enumerate() {
                if let Some(b) = b {
                    xb += x[(i, j)] * b;
                }
            }
            let eta = |k: usize| {
                let mut e = xb;
                if let (true, Some(w)) = (q > 0, nominal) {
                    for l in 0..q {
                        e += w[(i, l)] * self.nominal[k * q + l];
                    }
                }
                e
            };
            for c in 0..self.classes {
                let u = if c < k1 {
                    self.thresholds[c] - eta(c)
                } else {
                    f64::INFINITY
                };
                let v = if c > 0 {
                    self.thresholds[c - 1] - eta(c - 1)
                } else {
                    f64::NEG_INFINITY
                };
                out.push(self.link.interval(v, u));
            }
        }
        Ok(out)
    }
}

/// The model with the identifiable columns, rows in input order.
struct Spec<'a> {
    link: OrdinalLink,
    classes: usize,
    /// Column-major `n * p`.
    x: Vec<f64>,
    p: usize,
    /// Column-major `n * q`.
    w_nominal: Vec<f64>,
    q: usize,
    y: &'a [u32],
    weight: &'a [f64],
    offset: &'a [f64],
}

impl Spec<'_> {
    fn n(&self) -> usize {
        self.y.len()
    }

    fn dimension(&self) -> usize {
        let k1 = self.classes - 1;
        k1 + self.p + k1 * self.q
    }
}

/// The terms of one row, in the arguments `u` and `v` of its interval.
struct Row {
    class: usize,
    has_upper: bool,
    has_lower: bool,
    /// `w ∂l/∂u`, `w ∂l/∂v`.
    gu: f64,
    gv: f64,
    /// `w ∂²l/∂u²`, `w ∂²l/∂v²`, `w ∂²l/∂u∂v`.
    a: f64,
    b: f64,
    m: f64,
    log_probability: f64,
}

impl Spec<'_> {
    /// The terms of row `i` at `theta`, `None` when the observed category
    /// has a probability that is not positive.
    fn row(&self, i: usize, theta: &[f64], derivatives: bool) -> Option<Row> {
        let k1 = self.classes - 1;
        let c = self.y[i] as usize;
        let n = self.n();
        let beta = &theta[k1..k1 + self.p];
        let tau = &theta[k1 + self.p..];
        let mut xb = self.offset[i];
        for (j, b) in beta.iter().enumerate() {
            xb += self.x[j * n + i] * b;
        }
        let eta = |k: usize| {
            let mut e = xb;
            for l in 0..self.q {
                e += self.w_nominal[l * n + i] * tau[k * self.q + l];
            }
            e
        };
        let has_upper = c < k1;
        let has_lower = c > 0;
        let u = if has_upper {
            theta[c] - eta(c)
        } else {
            f64::INFINITY
        };
        let v = if has_lower {
            theta[c - 1] - eta(c - 1)
        } else {
            f64::NEG_INFINITY
        };
        let p = self.link.interval(v, u);
        if !(p > 0.0 && p.is_finite()) {
            return None;
        }
        let w = self.weight[i];
        let mut row = Row {
            class: c,
            has_upper,
            has_lower,
            gu: 0.0,
            gv: 0.0,
            a: 0.0,
            b: 0.0,
            m: 0.0,
            log_probability: w * p.ln(),
        };
        if derivatives {
            let (fu, fv) = (self.link.density(u), self.link.density(v));
            let (du, dv) = (
                self.link.density_derivative(u),
                self.link.density_derivative(v),
            );
            let lu = fu / p;
            let lv = -fv / p;
            row.gu = w * lu;
            row.gv = w * lv;
            row.a = w * (du / p - lu * lu);
            row.b = w * (-dv / p - lv * lv);
            row.m = w * (-lu * lv);
        }
        Some(row)
    }

    /// The gradient of the log-likelihood contribution of a row, added into
    /// `out` with the row's `x` and `w` columns.
    fn add_gradient(&self, i: usize, row: &Row, out: &mut [f64]) {
        let k1 = self.classes - 1;
        let n = self.n();
        let c = row.class;
        if row.has_upper {
            out[c] += row.gu;
        }
        if row.has_lower {
            out[c - 1] += row.gv;
        }
        let both = row.gu + row.gv;
        for j in 0..self.p {
            out[k1 + j] -= both * self.x[j * n + i];
        }
        for l in 0..self.q {
            let wl = self.w_nominal[l * n + i];
            if row.has_upper {
                out[k1 + self.p + c * self.q + l] -= row.gu * wl;
            }
            if row.has_lower {
                out[k1 + self.p + (c - 1) * self.q + l] -= row.gv * wl;
            }
        }
    }

    /// Adds the Hessian of the log-likelihood contribution of a row to the
    /// upper triangle of `h`.
    fn add_hessian(&self, i: usize, row: &Row, h: &mut [f64]) {
        let k1 = self.classes - 1;
        let (p, q, n) = (self.p, self.q, self.n());
        let t = self.dimension();
        let c = row.class;
        let (a, b, m) = (row.a, row.b, row.m);
        let beta = |j: usize| k1 + j;
        let tau = |k: usize, l: usize| k1 + p + k * q + l;
        let xj = |j: usize| self.x[j * n + i];
        let wl = |l: usize| self.w_nominal[l * n + i];
        let (hu, hv) = (row.has_upper, row.has_lower);
        if hu {
            h[c * t + c] += a;
        }
        if hv {
            h[(c - 1) * t + c - 1] += b;
        }
        if hu && hv {
            h[(c - 1) * t + c] += m;
        }
        for j in 0..p {
            if hu {
                h[c * t + beta(j)] -= (a + m) * xj(j);
            }
            if hv {
                h[(c - 1) * t + beta(j)] -= (b + m) * xj(j);
            }
        }
        let coefficient = a + b + 2.0 * m;
        for j in 0..p {
            let cj = coefficient * xj(j);
            for k in j..p {
                h[beta(j) * t + beta(k)] += cj * xj(k);
            }
        }
        for l in 0..q {
            let w = wl(l);
            if hu {
                h[c * t + tau(c, l)] -= a * w;
            }
            if hv {
                h[(c - 1) * t + tau(c - 1, l)] -= b * w;
            }
            if hu && hv {
                h[c * t + tau(c - 1, l)] -= m * w;
                h[(c - 1) * t + tau(c, l)] -= m * w;
            }
            for j in 0..p {
                if hu {
                    h[beta(j) * t + tau(c, l)] += (a + m) * xj(j) * w;
                }
                if hv {
                    h[beta(j) * t + tau(c - 1, l)] += (b + m) * xj(j) * w;
                }
            }
            for l2 in l..q {
                let w2 = wl(l2);
                if hu {
                    h[tau(c, l) * t + tau(c, l2)] += a * w * w2;
                }
                if hv {
                    h[tau(c - 1, l) * t + tau(c - 1, l2)] += b * w * w2;
                }
            }
            if hu && hv {
                for l2 in 0..q {
                    h[tau(c - 1, l) * t + tau(c, l2)] += m * w * wl(l2);
                }
            }
        }
    }

    /// The log-likelihood at `theta`, `None` when a probability is not
    /// positive.
    fn log_likelihood(&self, theta: &[f64]) -> Option<f64> {
        let n = self.n();
        let blocks = n.div_ceil(ROWS);
        let parts: Vec<Option<f64>> = (0..blocks)
            .into_par_iter()
            .map(|b| {
                let mut sum = 0.0;
                for i in b * ROWS..((b + 1) * ROWS).min(n) {
                    if self.weight[i] > 0.0 {
                        sum += self.row(i, theta, false)?.log_probability;
                    }
                }
                Some(sum)
            })
            .collect();
        let mut total = 0.0;
        for part in parts {
            total += part?;
        }
        Some(total)
    }

    /// The gradient and the information matrix, `-H`, with the upper triangle
    /// mirrored.
    fn information(&self, theta: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
        let n = self.n();
        let t = self.dimension();
        let blocks = n.div_ceil(ROWS);
        let parts: Vec<Option<(Vec<f64>, Vec<f64>)>> = (0..blocks)
            .into_par_iter()
            .map(|b| {
                let mut gradient = vec![0.0; t];
                let mut hessian = vec![0.0; t * t];
                for i in b * ROWS..((b + 1) * ROWS).min(n) {
                    if self.weight[i] > 0.0 {
                        let row = self.row(i, theta, true)?;
                        self.add_gradient(i, &row, &mut gradient);
                        self.add_hessian(i, &row, &mut hessian);
                    }
                }
                Some((gradient, hessian))
            })
            .collect();
        let mut gradient = vec![0.0; t];
        let mut hessian = vec![0.0; t * t];
        for part in parts {
            let (g, h) = part?;
            for (a, v) in gradient.iter_mut().zip(&g) {
                *a += v;
            }
            for (a, v) in hessian.iter_mut().zip(&h) {
                *a += v;
            }
        }
        for a in 0..t {
            for b in 0..a {
                hessian[a * t + b] = hessian[b * t + a];
            }
        }
        for v in &mut hessian {
            *v = -*v;
        }
        Some((gradient, hessian))
    }
}

/// The inverse of a symmetric positive definite matrix.
fn inverse_of(information: &[f64], t: usize) -> Result<Mat<f64>, Error> {
    let a = Mat::from_fn(t, t, |i, j| information[i * t + j]);
    epistat_linalg::inverse(a.as_ref(), &[Attempt::plain(Method::Cholesky)])
        .map(|solution| solution.x)
        .map_err(|_| Error::Singular)
}

impl Ordinal {
    /// Fits the cumulative link model of the categories of `data` (numbered
    /// `0..K`) on the columns of its design, which must not hold an
    /// intercept. With category-specific columns in the data it is the
    /// partial proportional odds model, in which those columns have a
    /// coefficient for each threshold. Case weights are 1 when the data have
    /// none, the offset is added to the linear predictor, and clusters, if
    /// given, sum the scores of the robust covariance.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] for a tolerance that is not positive and finite,
    /// [`Error::Invalid`] for fewer than two categories,
    /// [`Error::EmptyClass`] for a category without a row of positive
    /// weight, [`Error::Intercept`] for a constant column, [`Error::Aliased`]
    /// for a category-specific column that is a combination of the other
    /// columns, and [`Error::Separation`] or [`Error::Singular`] when the
    /// information is singular.
    pub fn fit(&self, data: &OrdinalData<'_>) -> Result<OrdinalFit, Error> {
        let (x, nominal, y) = (data.x(), data.nominal(), data.y());
        let (weights, offset, cluster) = (data.weights(), data.offset(), data.clusters());
        let n = x.nrows();
        let ones = vec![1.0; n];
        let zeros = vec![0.0; n];
        let w = weights.unwrap_or(&ones);
        let o = offset.unwrap_or(&zeros);
        crate::error::tolerance("tolerance", self.tolerance)?;
        let classes = y.iter().copied().max().map_or(0, |m| m as usize + 1);
        if classes < 2 {
            return Err(Error::Invalid("the outcome has fewer than two categories"));
        }
        let mut totals = vec![0.0; classes];
        for (&c, &wi) in y.iter().zip(w) {
            totals[c as usize] += wi;
        }
        if let Some(class) = totals.iter().position(|&t| t <= 0.0) {
            return Err(Error::EmptyClass { class });
        }
        let constant = |m: MatRef<'_, f64>, j: usize| {
            let first = (0..n).find(|&i| w[i] > 0.0);
            first.is_some_and(|f| {
                m[(f, j)] != 0.0 && (0..n).all(|i| w[i] == 0.0 || m[(i, j)] == m[(f, j)])
            })
        };
        for j in 0..x.ncols() {
            if constant(x, j) {
                return Err(Error::Intercept { column: j });
            }
        }
        if let Some(m) = nominal {
            for j in 0..m.ncols() {
                if constant(m, j) {
                    return Err(Error::Intercept { column: j });
                }
            }
        }

        // Aliased columns, with the constant counted.
        let p_in = x.ncols();
        let q = nominal.map_or(0, |m| m.ncols());
        let joint = Mat::from_fn(n, 1 + p_in + q, |i, j| match j {
            0 => 1.0,
            j if j <= p_in => x[(i, j - 1)],
            j => nominal.map_or(0.0, |m| m[(i, j - 1 - p_in)]),
        });
        let keep = independent_columns(joint.as_ref(), w, None, 1e-8).map_err(Error::Design)?;
        if let Some(l) = keep[1 + p_in..].iter().position(|&k| !k) {
            return Err(Error::Aliased { column: p_in + l });
        }
        let kept: Vec<usize> = (0..p_in).filter(|&j| keep[1 + j]).collect();
        let p = kept.len();
        let mut xs = Vec::with_capacity(n * p);
        for &j in &kept {
            xs.extend((0..n).map(|i| x[(i, j)]));
        }
        let mut ws = Vec::with_capacity(n * q);
        if let Some(m) = nominal {
            for l in 0..q {
                ws.extend((0..n).map(|i| m[(i, l)]));
            }
        }
        let spec = Spec {
            link: self.link,
            classes,
            x: xs,
            p,
            w_nominal: ws,
            q,
            y,
            weight: w,
            offset: o,
        };
        let k1 = classes - 1;

        // Proportional odds first, then the partial model from it.
        let base = Spec {
            link: self.link,
            classes,
            x: spec.x.clone(),
            p,
            w_nominal: Vec::new(),
            q: 0,
            y,
            weight: w,
            offset: o,
        };
        let total: f64 = totals.iter().sum();
        let mut theta = vec![0.0; k1 + p];
        let mut cumulative = 0.0;
        for k in 0..k1 {
            cumulative += totals[k];
            theta[k] = self.link.quantile(cumulative / total);
            if !theta[k].is_finite() {
                return Err(Error::Undefined("the starting thresholds"));
            }
        }
        let first = self.newton(&base, theta)?;
        let (theta, iterations, converged) = if q == 0 {
            first
        } else {
            let mut start = first.0;
            start.extend(std::iter::repeat_n(0.0, k1 * q));
            self.newton(&spec, start)?
        };

        let log_likelihood = spec.log_likelihood(&theta).ok_or(Error::Singular)?;
        let (_, information) = spec.information(&theta).ok_or(Error::Singular)?;
        let t = spec.dimension();
        let inverse = inverse_of(&information, t).map_err(|e| {
            if boundary(&spec, &theta) {
                Error::Separation
            } else {
                e
            }
        })?;
        let robust = match cluster {
            None => None,
            Some(labels) => {
                let clusters = Clusters::from_labels_first_seen(labels)?;
                let mut scores = Mat::<f64>::zeros(n, t);
                let mut row_score = vec![0.0; t];
                for i in 0..n {
                    if w[i] > 0.0 {
                        row_score.fill(0.0);
                        let row = spec.row(i, &theta, true).ok_or(Error::Singular)?;
                        spec.add_gradient(i, &row, &mut row_score);
                        for (j, v) in row_score.iter().enumerate() {
                            scores[(i, j)] = *v;
                        }
                    }
                }
                let m = meat(scores.as_ref(), |_| 1.0, Some(&clusters))?;
                Some(sandwich_with_inverse(inverse.as_ref(), m.as_ref()))
            }
        };

        // Expand to the layout with every input column.
        let full = k1 + p_in + k1 * q;
        let position: Vec<Option<usize>> = (0..full)
            .map(|index| {
                if index < k1 {
                    Some(index)
                } else if index < k1 + p_in {
                    kept.iter().position(|&j| j == index - k1).map(|r| k1 + r)
                } else {
                    Some(k1 + p + (index - k1 - p_in))
                }
            })
            .collect();
        let expand = |m: &Mat<f64>| -> Result<Covariance, Error> {
            let mut out = vec![None; full * full];
            for a in 0..full {
                for b in 0..full {
                    if let (Some(i), Some(j)) = (position[a], position[b]) {
                        out[a * full + b] = Some(m[(i, j)]);
                    }
                }
            }
            covariance(full, out)
        };
        let mut coefficients = vec![None; p_in];
        for (r, &j) in kept.iter().enumerate() {
            coefficients[j] = Some(theta[k1 + r]);
        }
        let null = totals.iter().map(|&c| c * (c / total).ln()).sum::<f64>();
        let mut violations = 0;
        if q > 0 {
            violations = count_violations(&spec, &theta);
        }
        Ok(OrdinalFit {
            link: self.link,
            classes,
            thresholds: theta[..k1].to_vec(),
            coefficients,
            nominal: theta[k1 + p..].to_vec(),
            covariance: expand(&inverse)?,
            robust_covariance: robust.as_ref().map(expand).transpose()?,
            log_likelihood,
            null_log_likelihood: null,
            iterations,
            converged,
            fitted_at_boundary: boundary(&spec, &theta),
            ordering_violations: violations,
        })
    }

    /// Newton's method with step halving from `theta`, returning the
    /// parameters, the iterations and whether it converged.
    fn newton(
        &self,
        spec: &Spec<'_>,
        mut theta: Vec<f64>,
    ) -> Result<(Vec<f64>, usize, bool), Error> {
        let t = spec.dimension();
        let mut loglik = spec
            .log_likelihood(&theta)
            .ok_or(Error::Undefined("the starting thresholds"))?;
        let mut converged = false;
        let mut iterations = 0;
        for iteration in 1..=self.max_iterations {
            iterations = iteration;
            let (gradient, information) = spec.information(&theta).ok_or(Error::Singular)?;
            let a = Mat::from_fn(t, t, |i, j| information[i * t + j]);
            let factor =
                Factor::new(a.as_ref(), &[Attempt::plain(Method::Cholesky)]).map_err(|_| {
                    if boundary(spec, &theta) {
                        Error::Separation
                    } else {
                        Error::Singular
                    }
                })?;
            let rhs = Mat::from_fn(t, 1, |i, _| gradient[i]);
            let step = factor.solve(rhs.as_ref())?;
            let mut scale = 1.0;
            let mut accepted = None;
            for _ in 0..30 {
                let trial: Vec<f64> = (0..t).map(|i| theta[i] + scale * step[(i, 0)]).collect();
                if let Some(l) = spec.log_likelihood(&trial)
                    && l >= loglik
                {
                    accepted = Some((trial, l));
                    break;
                }
                scale /= 2.0;
            }
            let largest_beta = theta.iter().map(|b| b.abs()).fold(0.0_f64, f64::max);
            let Some((trial, l)) = accepted else {
                let largest_step = (0..t).map(|i| step[(i, 0)].abs()).fold(0.0_f64, f64::max);
                converged = largest_step <= f64::EPSILON.sqrt() * (1.0 + largest_beta);
                break;
            };
            let largest_step = (0..t)
                .map(|i| (trial[i] - theta[i]).abs())
                .fold(0.0_f64, f64::max);
            let largest = trial.iter().map(|b| b.abs()).fold(0.0_f64, f64::max);
            theta = trial;
            loglik = l;
            if largest_step <= self.tolerance * (1.0 + largest) {
                converged = true;
                break;
            }
        }
        Ok((theta, iterations, converged))
    }
}

/// Whether an observed category has a probability within ten machine epsilons
/// of 0 or 1.
fn boundary(spec: &Spec<'_>, theta: &[f64]) -> bool {
    (0..spec.n()).any(|i| {
        spec.weight[i] > 0.0
            && match spec.row(i, theta, false) {
                None => true,
                Some(row) => {
                    let p = (row.log_probability / spec.weight[i]).exp();
                    !(p > 10.0 * f64::EPSILON && p < 1.0 - 10.0 * f64::EPSILON)
                }
            }
    })
}

/// The rows at which some category has a negative probability.
fn count_violations(spec: &Spec<'_>, theta: &[f64]) -> usize {
    let k1 = spec.classes - 1;
    let n = spec.n();
    let mut count = 0;
    for i in 0..n {
        let mut xb = spec.offset[i];
        for j in 0..spec.p {
            xb += spec.x[j * n + i] * theta[k1 + j];
        }
        let eta = |k: usize| {
            let mut e = xb;
            for l in 0..spec.q {
                e += spec.w_nominal[l * n + i] * theta[k1 + spec.p + k * spec.q + l];
            }
            e
        };
        let bad = (0..spec.classes).any(|c| {
            let u = if c < k1 {
                theta[c] - eta(c)
            } else {
                f64::INFINITY
            };
            let v = if c > 0 {
                theta[c - 1] - eta(c - 1)
            } else {
                f64::NEG_INFINITY
            };
            spec.link.interval(v, u) < 0.0
        });
        count += usize::from(bad);
    }
    count
}

/// The result of [`brant`].
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct BrantTest {
    /// The slopes of the binary logistic models of `y > k`, row-major
    /// `(K - 1)` by `p`.
    pub coefficients: Vec<f64>,
    /// The Wald test that every covariate has the same slope in all models,
    /// with `p (K - 2)` degrees of freedom.
    pub omnibus: Wald,
    /// The Wald test of each covariate, with `K - 2` degrees of freedom.
    pub per_covariate: Vec<Wald>,
}

/// The Brant test of the proportional odds assumption for the categories of
/// `data` (numbered `0..K`, `K >= 3`) on the columns of its design, which must
/// not hold an intercept, with the case weights of the data.
///
/// # Errors
///
/// [`Error::Unsupported`] for data with category-specific columns, an offset
/// or clusters, which the test does not take, [`Error::Invalid`] for fewer
/// than three categories, and the errors of the binary fits: a binary model
/// that cannot be fitted, such as one with separation, is an error of
/// [`Glm::fit`] or [`Error::Singular`].
pub fn brant(data: &OrdinalData<'_>) -> Result<BrantTest, Error> {
    if data.nominal().is_some() || data.offset().is_some() || data.clusters().is_some() {
        return Err(Error::Unsupported {
            what: "the Brant test with category-specific columns, an offset or clusters",
        });
    }
    let (x, y, weights) = (data.x(), data.y(), data.weights());
    let (n, p) = (x.nrows(), x.ncols());
    let ones = vec![1.0; n];
    let w = weights.unwrap_or(&ones);
    let classes = y.iter().copied().max().map_or(0, |m| m as usize + 1);
    if classes < 3 {
        return Err(Error::Invalid("the Brant test needs three categories"));
    }
    let k1 = classes - 1;
    let z = Mat::from_fn(n, p + 1, |i, j| if j == 0 { 1.0 } else { x[(i, j - 1)] });
    let mut model = Glm::new(Family::Binomial, Link::Logit);
    model.tolerance = 1e-12;
    model.max_iterations = 100;
    let mut pis = Vec::with_capacity(k1);
    let mut slopes = vec![0.0; k1 * p];
    let mut inverses = Vec::with_capacity(k1);
    for k in 0..k1 {
        let response: Vec<f64> = y
            .iter()
            .map(|&c| f64::from(u8::from(c as usize > k)))
            .collect();
        let fit = model.fit(&GlmData::from_parts(
            z.as_ref(),
            &response,
            Some(w),
            None,
            None,
        )?)?;
        for j in 0..=p {
            let b = fit.coefficients[j].ok_or(Error::Singular)?;
            if j > 0 {
                slopes[k * p + j - 1] = b;
            }
        }
        let weight_k: Vec<f64> = (0..n)
            .map(|i| w[i] * fit.fitted[i] * (1.0 - fit.fitted[i]))
            .collect();
        let info = weighted_gram(z.as_ref(), &weight_k).map_err(Error::Design)?;
        let factor = Factor::new(info.as_ref(), &[Attempt::plain(Method::Cholesky)])
            .map_err(|_| Error::Singular)?;
        inverses.push(factor.solve(Mat::<f64>::identity(p + 1, p + 1).as_ref())?);
        pis.push(fit.fitted);
    }
    // The covariance of the stacked slopes.
    let size = k1 * p;
    let mut stacked = Mat::<f64>::zeros(size, size);
    for j in 0..k1 {
        for k in j..k1 {
            let pair: Vec<f64> = (0..n)
                .map(|i| w[i] * pis[k][i] * (1.0 - pis[j][i]))
                .collect();
            let middle = weighted_gram(z.as_ref(), &pair).map_err(Error::Design)?;
            let left = Mat::from_fn(p + 1, p + 1, |a, b| {
                (0..=p)
                    .map(|c| inverses[j][(a, c)] * middle[(c, b)])
                    .sum::<f64>()
            });
            for a in 1..=p {
                for b in 1..=p {
                    let v: f64 = (0..=p).map(|c| left[(a, c)] * inverses[k][(c, b)]).sum();
                    stacked[(j * p + a - 1, k * p + b - 1)] = v;
                    stacked[(k * p + b - 1, j * p + a - 1)] = v;
                }
            }
        }
    }
    // The differences b_k - b_{k+1}, for the covariates in `which`.
    let test = |which: &[usize]| -> Result<Wald, Error> {
        let rows: Vec<(usize, usize)> = which
            .iter()
            .flat_map(|&j| (0..k1 - 1).map(move |k| (j, k)))
            .collect();
        let m = rows.len();
        let estimates: Vec<f64> = rows
            .iter()
            .map(|&(j, k)| slopes[k * p + j] - slopes[(k + 1) * p + j])
            .collect();
        let at = |r: usize, c: usize| -> f64 {
            let ((j1, k1_), (j2, k2)) = (rows[r], rows[c]);
            stacked[(k1_ * p + j1, k2 * p + j2)]
                - stacked[(k1_ * p + j1, (k2 + 1) * p + j2)]
                - stacked[((k1_ + 1) * p + j1, k2 * p + j2)]
                + stacked[((k1_ + 1) * p + j1, (k2 + 1) * p + j2)]
        };
        let covariance = Mat::from_fn(m, m, at);
        Ok(wald(&estimates, covariance.as_ref())?)
    };
    let all: Vec<usize> = (0..p).collect();
    let omnibus = test(&all)?;
    let per_covariate = (0..p).map(|j| test(&[j])).collect::<Result<Vec<_>, _>>()?;
    Ok(BrantTest {
        coefficients: slopes,
        omnibus,
        per_covariate,
    })
}
