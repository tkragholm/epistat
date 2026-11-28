//! Generalised linear mixed models with a random intercept for one grouping
//! factor, as `lme4::glmer` fits them with `nAGQ = 1` (the Laplace
//! approximation) and `nAGQ > 1` (adaptive Gauss-Hermite quadrature).
//!
//! The model for row `j` of cluster `i` is
//!
//! ```text
//! g(E[y_ij | u_i]) = η_ij = x_ij'β + o_ij + θ u_i,     u_i ~ N(0, 1)
//! ```
//!
//! for the binomial family with the logit link (`y` a proportion and the
//! prior weight its number of trials) and the Poisson family with the log
//! link. `θ` is the standard deviation of the random intercept `b_i = θ u_i`.
//!
//! # Likelihood
//!
//! Cluster `i` contributes `L_i = ∫ Π_j f(y_ij | η_ij) φ(u) du`. With
//! `h_i(u) = Σ_j w_ij (y_ij η_ij - b(η_ij)) - u²/2`, `b` the cumulant function
//! (`ln(1 + e^η)` or `e^η`), the conditional mode `û_i` maximises `h_i`, and
//! `a_i = -h_i''(û_i) = 1 + θ² Σ_j w_ij b''(η_ij)` is its curvature. Adaptive
//! Gauss-Hermite quadrature with the nodes `z_k` and weights `w_k` of
//! [`gauss_hermite`] for the standard normal density (Liu and Pierce \[4\])
//! places the nodes at `u_ik = û_i + s_i z_k`, `s_i = a_i^{-1/2}`:
//!
//! ```text
//! ln L_i ≈ c_i + ln s_i + ln Σ_k w_k exp(z_k²/2 + h_i(u_ik))
//! ```
//!
//! with the constant `c_i` the log binomial coefficients or `-Σ w ln y!`. One
//! node, `z = 0`, is the Laplace approximation `c_i + h_i(û_i) - ln(a_i)/2`.
//! These are `lme4`'s formulas: its Laplace deviance is `-2 Σ ln L_i`, and its
//! deviance for `nAGQ > 1` leaves out `Σ c_i`, as does the `logLik` it
//! reports then. [`GlmmFit::log_likelihood`] always includes it, so values
//! for different numbers of nodes can be compared.
//!
//! # Algorithm
//!
//! The mode of each cluster comes from Newton's method on `u`, from the mode at
//! the last evaluation, with the step halved until `h_i` does not decrease.
//! `h_i` is strictly concave, because `b'' > 0` and the prior adds `-u²/2`, so
//! the mode exists and is unique even when all outcomes of a cluster are
//! equal. Newton stops after a step below `1e-11 (1 + |u|)`, which leaves the
//! mode exact to rounding since the convergence is quadratic.
//!
//! The gradient of `ln L_i` with respect to `ψ = (θ, β)` is closed form. The
//! nodes move with `ψ` through `û_i` and `s_i`, so with `π_k` the normalised
//! terms of the sum,
//!
//! ```text
//! d ln L_i/dψ = s_i^{-1} ds_i/dψ + Σ_k π_k [∂h_i/∂ψ (u_ik) + h_i'(u_ik) (dû_i/dψ + z_k ds_i/dψ)]
//! dû_i/dψ     = (∂h_i'/∂ψ)(û_i) / a_i                       (implicit function)
//! ds_i/dψ     = -(s_i / (2 a_i)) da_i/dψ
//! da_i/dβ     = θ² Σ_j w_ij b'''(η_ij) (x_ij + θ dû_i/dβ)
//! da_i/dθ     = 2θ Σ_j w_ij b''(η_ij) + θ² Σ_j w_ij b'''(η_ij) (û_i + θ dû_i/dθ)
//! ```
//!
//! at `η` of the mode, with `b''' = b''(1 - 2μ)` for the logit and `μ` for the
//! log link. The `β` part is `X'c` for a coefficient `c_ij` per row, one pass
//! over the design. The criterion `-2 Σ ln L_i / N` is minimised over `(θ, β)`
//! by BFGS (`epistat-optim`), from the fit without random effects
//! (`epistat-glm`) and `θ = 1`, `lme4`'s start, with the inverse Hessian
//! started at `N/2` times the covariance of that fit for `β` and at 1 for
//! `θ`. A search that stops without meeting its gradient tolerance is
//! restarted from where it stopped, at most three times. The likelihood is
//! even in `θ`, so the optimisation runs over all real `θ` and the estimate is
//! `|θ|`; an estimate below `1e-4`, `lme4`'s tolerance for a singular fit, is
//! replaced by the fit at `θ = 0` (the model without random effects) when
//! that is not worse by more than `1e-10` of the criterion. Newton steps with
//! the Hessian below follow, until the step is below [`Glmm::tolerance`] times
//! `max(1, |(θ, β)|)`; that last step is not taken, and the fit has converged.
//! A Newton step is taken only when it lowers the norm of the gradient.
//!
//! The covariance of `(θ, β)` is `2 H^{-1}` with `H` the Hessian of
//! `-2 Σ ln L_i`, taken by central differences of the closed-form gradient
//! with steps `1e-5 max(1, |ψ_j|)`, which balance the truncation error, of
//! the order of the step squared, against the rounding of the gradient
//! divided by the step. `lme4`'s `vcov` uses the same Hessian, from
//! differences of the deviance itself. At `θ = 0` the Hessian is over `β`
//! alone and `θ` has no standard error.
//!
//! The conditional modes are `b_i = θ û_i` and their conditional variances
//! `θ² / a_i` (`ranef(condVar = TRUE)`). For the logit link the latent-scale
//! intraclass correlation is `θ² / (θ² + π²/3)`.
//!
//! The fit is checked against `lme4` 2.0.6 at fixed parameters (deviance,
//! modes, conditional variances and the Hessian), at the optimum, and, for
//! more than one node, against the marginal likelihood integrated by R's
//! `integrate()`.
//!
//! # Hazards
//!
//! The Laplace approximation is poor for binary outcomes in small clusters,
//! such as sibling pairs, and the quadrature then needs more nodes. Comparing
//! [`Glmm::evaluate`] at the estimate for several numbers of nodes shows how
//! much the approximation moves the likelihood.
//!
//! # Scope
//!
//! A random intercept for one grouping factor, with the canonical links of the
//! binomial and Poisson families. Random slopes, other links and families,
//! and several grouping factors are not fitted: `lme4` allows `nAGQ > 1` only
//! for a single scalar random effect, and the others need its sparse
//! formulation.
//!
//! # Cost
//!
//! `O(N (k + c))` exponentials per evaluation, with `k` nodes and `c` Newton
//! steps per cluster, and `O(N p)` for `Xβ` and `X'c`. Memory `O(N + G)`
//! besides the design.
//!
//! # Determinism
//!
//! Clusters are processed in blocks of 512 clusters, each block in order, and
//! the blocks summed in order, so results do not depend on the number of
//! threads. The block size is fixed and part of the bits of a fit.
//!
//! # References
//!
//! 1. Pinheiro JC, Bates DM. Mixed-Effects Models in S and S-PLUS. New York:
//!    Springer; 2000.
//! 2. Bates D, Mächler M, Bolker B, Walker S. Fitting linear mixed-effects
//!    models using lme4. J Stat Softw. 2015;67(1):1-48.
//! 3. Breslow NE, Clayton DG. Approximate inference in generalized linear
//!    mixed models. J Am Stat Assoc. 1993;88(421):9-25.
//! 4. Liu Q, Pierce DA. A note on Gauss-Hermite quadrature. Biometrika.
//!    1994;81(3):624-9.

use epistat_core::{Covariance, check};
use epistat_glm::{Family, Glm, GlmData, Link};
use epistat_linalg::cholesky::{Pivot, cholesky_lower};
use epistat_linalg::small::{cholesky_inverse, cholesky_solve};
use epistat_linalg::{Mat, MatRef, matvec_into, xtv};
use epistat_optim::bfgs::Bfgs;
use epistat_stats::dist::gamma::ln_gamma;
use rayon::prelude::*;

use crate::error::{iterations, tolerance};
use crate::groups::Groups;
use crate::lmm::{BLOCK, check_size, covariance};
use crate::quadrature::gauss_hermite;
use crate::{Convergence, Error, MixedData};

/// `lme4`'s tolerance below which `θ` counts as zero.
const SINGULAR: f64 = 1e-4;

/// A generalised linear mixed model with a random intercept and its fitting
/// controls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Glmm {
    /// The family: [`Family::Binomial`] or [`Family::Poisson`].
    pub family: Family,
    /// The link: [`Link::Logit`] for the binomial, [`Link::Log`] for the
    /// Poisson family.
    pub link: Link,
    /// Quadrature nodes, `nAGQ`: 1 is the Laplace approximation.
    pub nodes: usize,
    /// The fit has converged when the Newton step from the returned
    /// `(θ, β)` is below this times `max(1, |(θ, β)|)`, Euclidean norms. BFGS
    /// also stops when the norm of the gradient of `-2 ln L / N` is below it.
    pub tolerance: f64,
    /// The most BFGS iterations of one search.
    pub max_iterations: usize,
}

impl Glmm {
    /// A model with `nodes` quadrature nodes, tolerance `1e-8` and 200
    /// iterations.
    #[must_use]
    pub const fn new(family: Family, link: Link, nodes: usize) -> Self {
        Self {
            family,
            link,
            nodes,
            tolerance: 1e-8,
            max_iterations: 200,
        }
    }
}

/// A fitted generalised linear mixed model.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct GlmmFit {
    /// The fixed effects.
    pub beta: Vec<f64>,
    /// Their covariance, `p x p`, from the Hessian of the likelihood in
    /// `(θ, β)`; `None` when that Hessian is not positive definite, and an
    /// entry that is not finite is `None`.
    pub beta_covariance: Option<Covariance>,
    /// The standard deviation `θ` of the random intercept.
    pub theta: f64,
    /// The standard error of `θ`; `None` at `θ = 0` or when the Hessian is
    /// not positive definite.
    pub theta_se: Option<f64>,
    /// The approximated marginal log likelihood, with the constants of the
    /// family included.
    pub log_likelihood: f64,
    /// The gradient of `-2` times the log likelihood with respect to
    /// `(θ, β)`.
    pub gradient: Vec<f64>,
    /// The group labels, increasing; the order of the modes.
    pub groups: Vec<u32>,
    /// The conditional modes `b_i = θ û_i`.
    pub modes: Vec<f64>,
    /// Their conditional variances `θ² / a_i`.
    pub mode_variances: Vec<f64>,
    /// The latent-scale intraclass correlation `θ² / (θ² + π²/3)` of the
    /// logit link; `None` for the Poisson family.
    pub icc_latent: Option<f64>,
    /// Whether `θ` is 0.
    pub singular: bool,
    /// The number of quadrature nodes.
    pub nodes: usize,
    /// How the optimisation ended, `None` for [`Glmm::evaluate`].
    pub convergence: Option<Convergence>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Binomial,
    Poisson,
}

impl Kind {
    /// `(y η - b(η), μ, b''(η), b'''(η))`.
    #[inline]
    fn terms(self, eta: f64, y: f64) -> (f64, f64, f64, f64) {
        match self {
            Self::Binomial => {
                let (mu, softplus) = if eta >= 0.0 {
                    let e = (-eta).exp();
                    (1.0 / (1.0 + e), eta + e.ln_1p())
                } else {
                    let e = eta.exp();
                    (e / (1.0 + e), e.ln_1p())
                };
                let v = mu * (1.0 - mu);
                (y * eta - softplus, mu, v, v * (1.0 - 2.0 * mu))
            }
            Self::Poisson => {
                let mu = eta.exp();
                (y * eta - mu, mu, mu, mu)
            }
        }
    }

    /// `y η - b(η)` and `μ`.
    #[inline]
    fn kernel(self, eta: f64, y: f64) -> (f64, f64) {
        match self {
            Self::Binomial => {
                if eta >= 0.0 {
                    let e = (-eta).exp();
                    (y * eta - (eta + e.ln_1p()), 1.0 / (1.0 + e))
                } else {
                    let e = eta.exp();
                    (y * eta - e.ln_1p(), e / (1.0 + e))
                }
            }
            Self::Poisson => {
                let mu = eta.exp();
                (y * eta - mu, mu)
            }
        }
    }
}

/// The value and gradient of `-2 Σ ln L_i`, without the constants, at one
/// point.
struct Evaluation {
    value: f64,
    gradient: Vec<f64>,
    /// `a_i` at the modes.
    curvature: Vec<f64>,
}

struct Problem<'a> {
    n: usize,
    p: usize,
    kind: Kind,
    x: MatRef<'a, f64>,
    offset: Vec<f64>,
    groups: Groups,
    /// Response and prior weights in cluster order.
    y: Vec<f64>,
    w: Vec<f64>,
    nodes: Vec<f64>,
    /// `ln w_k + z_k² / 2`.
    log_weights: Vec<f64>,
    /// `Σ c_i`.
    constant: f64,
}

/// Buffers reused from cluster to cluster: the node terms, the node sums
/// of `w (y - μ)`, and the means at the nodes with `b''` and `b'''` at the
/// mode.
#[derive(Default)]
struct Scratch {
    terms: Vec<f64>,
    totals: Vec<f64>,
    mus: Vec<f64>,
}

/// What one cluster contributes.
struct Contribution {
    log_l: f64,
    d_theta: f64,
    mode: f64,
    curvature: f64,
}

impl<'a> Problem<'a> {
    fn new(model: &Glmm, data: &MixedData<'a>) -> Result<Self, Error> {
        let kind = match (model.family, model.link) {
            (Family::Binomial, Link::Logit) => Kind::Binomial,
            (Family::Poisson, Link::Log) => Kind::Poisson,
            _ => {
                return Err(Error::Unsupported {
                    what: "a family and link other than binomial-logit and Poisson-log",
                });
            }
        };
        if data.slope().is_some() {
            return Err(Error::Unsupported {
                what: "a random slope in a generalised linear mixed model",
            });
        }
        let (x, y, offset) = (data.x(), data.y(), data.offset());
        let (n, p) = (x.nrows(), x.ncols());
        check_size(n, p)?;
        let w: Vec<f64> = data.weights().map_or_else(|| vec![1.0; n], <[f64]>::to_vec);
        let valid = |v: f64| match kind {
            Kind::Binomial => (0.0..=1.0).contains(&v),
            Kind::Poisson => v >= 0.0,
        };
        if let Some((row, &value)) = y.iter().enumerate().find(|&(_, &v)| !valid(v)) {
            return Err(Error::Response { row, value });
        }
        tolerance(model.tolerance)?;
        iterations(model.max_iterations)?;
        let (nodes, node_weights) = gauss_hermite(model.nodes)?;
        let log_weights = nodes
            .iter()
            .zip(&node_weights)
            .map(|(z, w)| w.ln() + z * z / 2.0)
            .collect();
        let mut constant = 0.0;
        for i in 0..n {
            constant += match kind {
                Kind::Binomial => {
                    let (m, s) = (w[i], w[i] * y[i]);
                    ln_gamma(m + 1.0) - ln_gamma(s + 1.0) - ln_gamma(m - s + 1.0)
                }
                Kind::Poisson => {
                    if w[i] == 0.0 {
                        0.0
                    } else {
                        -w[i] * ln_gamma(y[i] + 1.0)
                    }
                }
            };
        }
        let groups = Groups::new(data.groups());
        Ok(Self {
            n,
            p,
            kind,
            x,
            offset: offset.map_or_else(|| vec![0.0; n], <[f64]>::to_vec),
            y: groups.gather(y),
            w: groups.gather(&w),
            groups,
            nodes,
            log_weights,
            constant,
        })
    }

    /// `(h(u), h'(u), a(u))` for one cluster.
    fn mode_terms(&self, theta: f64, eta: &[f64], y: &[f64], w: &[f64], u: f64) -> (f64, f64, f64) {
        let (mut h, mut score, mut info) = (0.0, 0.0, 0.0);
        for j in 0..eta.len() {
            let (k, mu, v, _) = self.kind.terms(eta[j] + theta * u, y[j]);
            h += w[j] * k;
            score += w[j] * (y[j] - mu);
            info += w[j] * v;
        }
        (
            h - u * u / 2.0,
            theta * score - u,
            1.0 + theta * theta * info,
        )
    }

    /// The conditional mode from `u`, with `h` and `a` there.
    fn mode(
        &self,
        theta: f64,
        eta: &[f64],
        y: &[f64],
        w: &[f64],
        mut u: f64,
    ) -> Option<(f64, f64, f64)> {
        if !u.is_finite() {
            u = 0.0;
        }
        let (mut h, mut score, mut a) = self.mode_terms(theta, eta, y, w, u);
        for _ in 0..200 {
            if !(h.is_finite() && score.is_finite() && a.is_finite()) {
                return None;
            }
            let step = score / a;
            if step.abs() <= 1e-11 * (1.0 + u.abs()) {
                u += step;
                let (h, _, a) = self.mode_terms(theta, eta, y, w, u);
                return Some((u, h, a));
            }
            let mut t = 1.0;
            loop {
                let next = u + t * step;
                let (hn, sn, an) = self.mode_terms(theta, eta, y, w, next);
                // Below a step of 1e-6 Newton converges quadratically, and a
                // decrease of `h` is rounding.
                if (hn.is_finite() && hn >= h) || (t * step).abs() < 1e-6 * (1.0 + u.abs()) {
                    u = next;
                    h = hn;
                    score = sn;
                    a = an;
                    break;
                }
                t /= 2.0;
            }
        }
        None
    }

    /// One cluster's log likelihood and gradient, writing the row
    /// coefficients of the `β` gradient to `coef` when given.
    #[expect(clippy::too_many_arguments, reason = "the parts of one cluster")]
    fn cluster(
        &self,
        theta: f64,
        eta: &[f64],
        y: &[f64],
        w: &[f64],
        start: f64,
        coef: Option<&mut [f64]>,
        scratch: &mut Scratch,
    ) -> Option<Contribution> {
        let (u_hat, h_hat, a) = self.mode(theta, eta, y, w, start)?;
        let s = 1.0 / a.sqrt();
        let m = eta.len();
        let k = self.nodes.len();
        let gradient = coef.is_some();
        let Scratch { terms, totals, mus } = scratch;
        if gradient {
            mus.clear();
            mus.resize(m * (k + 2), 0.0);
        }
        terms.clear();
        terms.resize(k, 0.0);
        totals.clear();
        totals.resize(k, 0.0);
        for (node, &z) in self.nodes.iter().enumerate() {
            let u = u_hat + s * z;
            let mut total = 0.0;
            let h = if z == 0.0 && !gradient {
                h_hat
            } else {
                let mut sum = 0.0;
                for j in 0..m {
                    let (kern, mu) = self.kind.kernel(eta[j] + theta * u, y[j]);
                    sum += w[j] * kern;
                    total += w[j] * (y[j] - mu);
                    if gradient {
                        mus[node * m + j] = mu;
                    }
                }
                if z == 0.0 { h_hat } else { sum - u * u / 2.0 }
            };
            terms[node] = self.log_weights[node] + h;
            totals[node] = total;
        }
        let top = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for t in terms.iter_mut() {
            *t = (*t - top).exp();
            sum += *t;
        }
        let log_l = s.ln() + top + sum.ln();
        let mut d_theta = 0.0;
        if let Some(coef) = coef {
            for t in terms.iter_mut() {
                *t /= sum;
            }
            let pi = &*terms;
            // The second and third derivatives of b at the mode.
            let (mut big_v, mut big_vp, mut t_hat) = (0.0, 0.0, 0.0);
            for j in 0..m {
                let (_, mu, v, vp) = self.kind.terms(eta[j] + theta * u_hat, y[j]);
                big_v += w[j] * v;
                big_vp += w[j] * vp;
                t_hat += w[j] * (y[j] - mu);
                mus[k * m + j] = v;
                mus[(k + 1) * m + j] = vp;
            }
            let (mut g, mut gz, mut pu_t) = (0.0, 0.0, 0.0);
            for (node, &z) in self.nodes.iter().enumerate() {
                let u = u_hat + s * z;
                let score = theta * totals[node] - u;
                g += pi[node] * score;
                gz += pi[node] * score * z;
                pu_t += pi[node] * u * totals[node];
            }
            let factor = (1.0 + s * gz) / (2.0 * a);
            for j in 0..m {
                let (v, vp) = (mus[k * m + j], mus[(k + 1) * m + j]);
                let alpha = -(theta / a) * w[j] * v;
                let delta = theta * theta * w[j] * vp + theta * theta * theta * big_vp * alpha;
                let mut mean = 0.0;
                for node in 0..k {
                    mean += pi[node] * mus[node * m + j];
                }
                coef[j] = w[j] * (y[j] - mean) + g * alpha - factor * delta;
            }
            let du = (t_hat - theta * u_hat * big_v) / a;
            let da = 2.0 * theta * big_v + theta * theta * big_vp * (u_hat + theta * du);
            let ds = -(s / (2.0 * a)) * da;
            d_theta = pu_t + g * du + (1.0 / s + gz) * ds;
        }
        Some(Contribution {
            log_l,
            d_theta,
            mode: u_hat,
            curvature: a,
        })
    }

    /// `-2 Σ ln L_i` without the constants, its gradient when asked, from the
    /// modes in `modes`, which are replaced by the new ones.
    fn evaluate(
        &self,
        params: &[f64],
        modes: &mut [f64],
        gradient: bool,
    ) -> Result<Evaluation, Error> {
        let (n, p) = (self.n, self.p);
        let theta = params[0];
        let beta = Mat::from_fn(p, 1, |j, _| params[1 + j]);
        let mut eta = vec![0.0; n];
        matvec_into(&mut eta, self.x, beta.as_ref());
        for (e, o) in eta.iter_mut().zip(&self.offset) {
            *e += o;
        }
        let eta = self.groups.gather(&eta);
        let count = self.groups.count();
        let blocks = count.div_ceil(BLOCK);
        let starts = &self.groups.starts;
        let mut coef = vec![0.0; if gradient { n } else { 0 }];
        let mut slices: Vec<&mut [f64]> = Vec::with_capacity(blocks);
        {
            let mut rest: &mut [f64] = &mut coef;
            for b in 0..blocks {
                let len = if gradient {
                    starts[((b + 1) * BLOCK).min(count)] - starts[b * BLOCK]
                } else {
                    0
                };
                let (head, tail) = rest.split_at_mut(len);
                slices.push(head);
                rest = tail;
            }
        }
        let mut curvature = vec![0.0; count];
        let parts: Vec<Result<(f64, f64), usize>> = slices
            .into_par_iter()
            .zip(modes.par_chunks_mut(BLOCK))
            .zip(curvature.par_chunks_mut(BLOCK))
            .enumerate()
            .map(|(b, ((coef, modes), curv))| {
                let (mut log_l, mut d_theta) = (0.0, 0.0);
                let mut scratch = Scratch::default();
                let base = starts[b * BLOCK];
                for (local, (mode, c)) in modes.iter_mut().zip(curv.iter_mut()).enumerate() {
                    let i = b * BLOCK + local;
                    let (lo, hi) = (starts[i], starts[i + 1]);
                    let out = self
                        .cluster(
                            theta,
                            &eta[lo..hi],
                            &self.y[lo..hi],
                            &self.w[lo..hi],
                            *mode,
                            gradient.then(|| &mut coef[lo - base..hi - base]),
                            &mut scratch,
                        )
                        .ok_or(i)?;
                    log_l += out.log_l;
                    d_theta += out.d_theta;
                    *mode = out.mode;
                    *c = out.curvature;
                }
                Ok((log_l, d_theta))
            })
            .collect();
        let (mut log_l, mut d_theta) = (0.0, 0.0);
        for part in parts {
            let (l, d) = part.map_err(|cluster| Error::Mode { cluster })?;
            log_l += l;
            d_theta += d;
        }
        let mut grad = Vec::new();
        if gradient {
            let mut ordered = vec![0.0; n];
            self.groups.scatter(&coef, &mut ordered);
            let xc = xtv(self.x, &ordered);
            grad.push(-2.0 * d_theta);
            grad.extend((0..p).map(|j| -2.0 * xc[(j, 0)]));
        }
        Ok(Evaluation {
            value: -2.0 * log_l,
            gradient: grad,
            curvature,
        })
    }

    /// The Hessian of `-2 Σ ln L_i` over the free parameters, by central
    /// differences of the gradient, with the modes started from `modes` at
    /// every point; `None` when an evaluation fails.
    fn hessian(&self, params: &[f64], modes: &[f64], free: &[usize]) -> Option<Vec<f64>> {
        let k = free.len();
        let mut hessian = vec![0.0; k * k];
        for (a, &j) in free.iter().enumerate() {
            let h = 1e-5 * params[j].abs().max(1.0);
            let mut grads = [Vec::new(), Vec::new()];
            for (side, sign) in [1.0, -1.0].into_iter().enumerate() {
                let mut point = params.to_vec();
                point[j] += sign * h;
                let mut m = modes.to_vec();
                grads[side] = self.evaluate(&point, &mut m, true).ok()?.gradient;
            }
            for (b, &l) in free.iter().enumerate() {
                hessian[a * k + b] = (grads[0][l] - grads[1][l]) / (2.0 * h);
            }
        }
        for a in 0..k {
            for b in 0..a {
                let v = f64::midpoint(hessian[a * k + b], hessian[b * k + a]);
                hessian[a * k + b] = v;
                hessian[b * k + a] = v;
            }
        }
        Some(hessian)
    }

    /// Newton steps on the free parameters with the Hessian of
    /// [`Problem::hessian`], until the step is below the tolerance, which is
    /// then not taken. A step is taken only when it lowers the norm of the
    /// gradient. Returns whether the step fell below the tolerance, the
    /// Newton steps taken and the Hessian at the returned point.
    fn polish(
        &self,
        tolerance: f64,
        params: &mut [f64],
        modes: &mut Vec<f64>,
        free: &[usize],
    ) -> Result<(bool, usize, Option<Vec<f64>>), Error> {
        let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let pick = |g: &[f64]| free.iter().map(|&j| g[j]).collect::<Vec<f64>>();
        let mut gradient = pick(&self.evaluate(params, modes, true)?.gradient);
        let k = free.len();
        let mut steps = 0;
        for _ in 0..8 {
            let Some(hessian) = self.hessian(params, modes, free) else {
                return Ok((false, steps, None));
            };
            let Some(factor) = cholesky_lower(&hessian, k, Pivot::Strict) else {
                return Ok((false, steps, Some(hessian)));
            };
            let mut step = gradient.clone();
            cholesky_solve(&factor, k, &mut step)?;
            if norm(&step) <= tolerance * norm(&pick(params)).max(1.0) {
                return Ok((true, steps, Some(hessian)));
            }
            let mut trial = params.to_vec();
            for (&j, s) in free.iter().zip(&step) {
                trial[j] -= s;
            }
            let mut trial_modes = modes.clone();
            let next = match self.evaluate(&trial, &mut trial_modes, true) {
                Ok(e) if e.value.is_finite() => pick(&e.gradient),
                _ => return Ok((false, steps, Some(hessian))),
            };
            if norm(&next).partial_cmp(&norm(&gradient)) != Some(std::cmp::Ordering::Less) {
                return Ok((false, steps, Some(hessian)));
            }
            params.copy_from_slice(&trial);
            *modes = trial_modes;
            gradient = next;
            steps += 1;
        }
        Ok((false, steps, self.hessian(params, modes, free)))
    }

    /// Everything a fit reports, at `params`, from the modes `modes`, with
    /// the Hessian over the free parameters when it is known and computed
    /// otherwise.
    fn at(
        &self,
        params: &[f64],
        modes: &mut [f64],
        nodes: usize,
        hessian: Option<Vec<f64>>,
    ) -> Result<GlmmFit, Error> {
        let e = self.evaluate(params, modes, true)?;
        let p = self.p;
        let theta = params[0];
        let singular = theta == 0.0;
        let free: Vec<usize> = if singular {
            (1..=p).collect()
        } else {
            (0..=p).collect()
        };
        let hessian = hessian.or_else(|| self.hessian(params, modes, &free));
        let k = free.len();
        let factor = hessian.and_then(|h| cholesky_lower(&h, k, Pivot::Strict));
        let inverse = match factor {
            Some(r) => Some(
                cholesky_inverse(&r, k)?
                    .iter()
                    .map(|v| 2.0 * v)
                    .collect::<Vec<f64>>(),
            ),
            None => None,
        };
        let (beta_covariance, theta_se) = match inverse {
            Some(c) if singular => (Some(covariance(p, &c)?), None),
            Some(c) => {
                let k = p + 1;
                let beta_cov: Vec<f64> =
                    (0..p * p).map(|i| c[(1 + i / p) * k + 1 + i % p]).collect();
                (Some(covariance(p, &beta_cov)?), Some(c[0].max(0.0).sqrt()))
            }
            None => (None, None),
        };
        let modes_b: Vec<f64> = modes.iter().map(|u| theta * u).collect();
        let mode_variances = e.curvature.iter().map(|a| theta * theta / a).collect();
        let icc_latent = (self.kind == Kind::Binomial).then(|| {
            let t2 = theta * theta;
            t2 / (t2 + std::f64::consts::PI * std::f64::consts::PI / 3.0)
        });
        Ok(GlmmFit {
            beta: params[1..].to_vec(),
            beta_covariance,
            theta: theta.abs(),
            theta_se,
            log_likelihood: -e.value / 2.0 + self.constant,
            gradient: e.gradient,
            groups: self.groups.labels.clone(),
            modes: modes_b,
            mode_variances,
            icc_latent,
            singular,
            nodes,
            convergence: None,
        })
    }

    /// Minimises over the free parameters from `params`, in place, with up to
    /// three restarts; returns the criterion, the iterations and whether the
    /// tolerance was met.
    fn optimise(
        &self,
        controls: &Glmm,
        params: &mut [f64],
        modes: &mut Vec<f64>,
        free: &[usize],
        inverse_hessian: &[f64],
    ) -> Result<(f64, usize, bool), Error> {
        let scale = self.n as f64;
        let bfgs = Bfgs {
            max_iterations: controls.max_iterations,
            gradient_tolerance: controls.tolerance,
            ..Bfgs::default()
        };
        let k = free.len();
        let mut h0 = vec![0.0; k * k];
        for (a, &i) in free.iter().enumerate() {
            for (b, &j) in free.iter().enumerate() {
                h0[a * k + b] = inverse_hessian[i * (self.p + 1) + j];
            }
        }
        let (mut iterations, mut converged, mut value) = (0, false, f64::INFINITY);
        for _ in 0..4 {
            let fixed = params.to_vec();
            let start: Vec<f64> = free.iter().map(|&j| params[j]).collect();
            let mut cache = modes.clone();
            let full = |point: &[f64]| {
                let mut t = fixed.clone();
                for (&j, &v) in free.iter().zip(point) {
                    t[j] = v;
                }
                t
            };
            let minimum = bfgs
                .minimize(
                    &start,
                    |point| {
                        let mut trial = cache.clone();
                        match self.evaluate(&full(point), &mut trial, true) {
                            Ok(e) if e.value.is_finite() => {
                                cache = trial;
                                (
                                    e.value / scale,
                                    free.iter().map(|&j| e.gradient[j] / scale).collect(),
                                )
                            }
                            _ => (f64::INFINITY, vec![0.0; k]),
                        }
                    },
                    Some(h0.clone()),
                )
                .map_err(|_| Error::NotFiniteStart)?;
            iterations += minimum.iterations;
            let improved = minimum.value * scale < value;
            for (&j, &v) in free.iter().zip(&minimum.x) {
                params[j] = v;
            }
            value = minimum.value * scale;
            // The modes of the returned point, from those of the last point
            // evaluated.
            self.evaluate(params, &mut cache, false)?;
            *modes = cache;
            converged = minimum.converged;
            if converged || !improved {
                break;
            }
        }
        Ok((value, iterations, converged))
    }
}

impl Glmm {
    /// Fits the model to `data`: the response is a proportion for the
    /// binomial family, with the number of trials as the prior weight, the
    /// weights are not negative and the offset is a known part of the linear
    /// predictor.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for a family and link other than binomial-logit
    /// and Poisson-log, or data with a random slope; [`Error::Input`] for a
    /// tolerance that is not positive and finite, an iteration limit of 0 or
    /// a number of nodes outside `1..=100`; [`Error::Response`] for a
    /// response outside the family's support; [`Error::TooFewRows`] and
    /// [`Error::RankDeficient`] for a design whose fixed effects are not
    /// identified; [`Error::Start`] when the fit without random effects
    /// fails; [`Error::Mode`] and [`Error::NotFiniteStart`] when the
    /// likelihood cannot be evaluated.
    pub fn fit(&self, data: &MixedData<'_>) -> Result<GlmmFit, Error> {
        let problem = Problem::new(self, data)?;
        let p = problem.p;
        let mut start = GlmData::builder(data.x(), data.y());
        if let Some(weights) = data.weights() {
            start = start.weights(weights);
        }
        if let Some(offset) = data.offset() {
            start = start.offset(offset);
        }
        let start = Glm::new(self.family, self.link).fit(&start.build()?)?;
        let mut params = Vec::with_capacity(p + 1);
        params.push(1.0);
        for c in &start.coefficients {
            params.push(c.ok_or(Error::RankDeficient)?);
        }
        let k = p + 1;
        let scale = problem.n as f64 / 2.0;
        let mut h0 = vec![0.0; k * k];
        h0[0] = 1.0;
        for i in 0..p {
            for j in 0..p {
                h0[(1 + i) * k + 1 + j] =
                    scale * start.covariance.as_slice()[i * p + j].ok_or(Error::RankDeficient)?;
            }
        }
        let mut modes = vec![0.0; problem.groups.count()];
        let all: Vec<usize> = (0..k).collect();
        let (value, mut iterations, _) =
            problem.optimise(self, &mut params, &mut modes, &all, &h0)?;
        params[0] = params[0].abs();
        let mut free = all;
        if params[0] < SINGULAR {
            let mut candidate = params.clone();
            candidate[0] = 0.0;
            let mut candidate_modes = vec![0.0; modes.len()];
            let beta_only: Vec<usize> = (1..k).collect();
            let (v, it, _) =
                problem.optimise(self, &mut candidate, &mut candidate_modes, &beta_only, &h0)?;
            iterations += it;
            if v <= value + 1e-10 * value.abs().max(1.0) {
                params = candidate;
                modes = candidate_modes;
                free = beta_only;
            }
        }
        let (converged, steps, hessian) =
            problem.polish(self.tolerance, &mut params, &mut modes, &free)?;
        iterations += steps;
        let mut fit = problem.at(&params, &mut modes, self.nodes, hessian)?;
        fit.convergence = Some(Convergence {
            iterations,
            converged,
        });
        Ok(fit)
    }

    /// The fit at given `θ` and `β`: the log likelihood, its gradient, the
    /// modes and their variances, and the covariance from the Hessian there.
    /// The sign of `θ` is ignored, since the likelihood is even in it.
    ///
    /// # Errors
    ///
    /// As [`Glmm::fit`] for the model and the data, and [`Error::Input`] when
    /// `beta` does not have one entry per column of the design or a parameter
    /// is not finite.
    pub fn evaluate(
        &self,
        data: &MixedData<'_>,
        theta: f64,
        beta: &[f64],
    ) -> Result<GlmmFit, Error> {
        let problem = Problem::new(self, data)?;
        check::same_length("beta", problem.p, beta.len())?;
        check::finite("theta", &[theta])?;
        check::finite("beta", beta)?;
        let mut params = vec![theta.abs()];
        params.extend_from_slice(beta);
        let mut modes = vec![0.0; problem.groups.count()];
        problem.at(&params, &mut modes, self.nodes, None)
    }
}
