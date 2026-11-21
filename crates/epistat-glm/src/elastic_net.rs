//! Elastic-net regression for the Gaussian and binomial families, by cyclic
//! coordinate descent as in glmnet.
//!
//! [`ElasticNet::fit`] takes a design matrix, a response, and optional prior
//! weights, offset, per-column penalty factors and starting coefficients. It
//! returns an [`ElasticNetFit`] with the coefficients, whether the iteration
//! converged, and the number of IRLS steps and coordinate passes it took.
//!
//! # Objective
//!
//! Let `w̄` be the prior weights scaled to mean 1, so that `Σ w̄ = n`, `x_j` the
//! columns of the design, `o` the offset and `η = Xβ + o`. The fit minimises
//!
//! ```text
//! (1/n) Σ w̄ ℓ(y, η)  +  λ Σ_j f_j ( α |β_j| + (1 - α)/2 β_j² )
//! ```
//!
//! with the loss `ℓ = (y - η)² / 2` for the Gaussian family and the log-loss
//! `ln(1 + e^η) - y η` for the binomial. `λ` is per observation, `α` is the
//! mixing parameter (0 is ridge, 1 is lasso) and `f_j` is the penalty factor of
//! column `j`, 1 by default. The design is used as given and is not
//! standardised, which is glmnet's `standardize = FALSE`. There is no implicit
//! intercept: a constant column in the design is the intercept, and a column
//! with penalty factor 0 is not penalised. A column with no weighted variation
//! is skipped.
//!
//! # Algorithm
//!
//! For the Gaussian family the loss is quadratic, and the working response is
//! `z = y - o`. For the binomial family each IRLS step takes the quadratic
//! approximation of the log-loss at the current `η`, with `p = 1/(1 + e^-η)`
//! clipped to `[c, 1 - c]`,
//!
//! ```text
//! V = max(p (1 - p), c),      z - (η - o) = (y - p) / V,      W = w̄ V
//! ```
//!
//! and solves the penalised weighted least-squares problem
//! `(1/2n) Σ W (z - Xβ)² + penalty` by cyclic coordinate descent. With
//! `v_j = (1/n) Σ W x_ij²`, the residual `r = z - Xβ` and the soft-threshold
//! `S(a, t) = sign(a) max(|a| - t, 0)`, coordinate `j` is updated to
//!
//! ```text
//! β_j = S( v_j β_j + (1/n) Σ W x_ij r_i , λ α f_j ) / ( v_j + λ (1 - α) f_j )
//! ```
//!
//! and the residual is updated in place, so a pass costs `O(n p)`. This is the
//! form used for designs with more than 32 columns. With
//! 32 or fewer, the fit works from the weighted Gram matrix, see below. The binomial
//! step is halved towards the previous coefficients, up to 30 times, until the
//! penalised objective no longer rises, which glmnet does not do and which
//! makes the step safe far from the solution. If no halving helps, the fit
//! returns the coefficients of the previous step, not converged. Passes, steps
//! and halvings run in a fixed order on one thread, so a fit does not depend on
//! the number of threads.
//!
//! # Covariance updates
//!
//! With at most 32 columns, glmnet's covariance
//! updating is used. One pass over the rows at the current coefficients
//! computes `η`, the probability, the log-loss, the working weights and
//! response, and from them the weighted moments `Σ W x_j`, `Σ W x_j x_k` and
//! `Σ W x_j r`. Coordinate descent then runs on those `p x p` moments alone. The
//! gradient `g_j = (1/n) Σ W x_ij r_i` is kept up to date by
//! `g_i -= C_ij Δβ_j` after each change, with `C` the weighted covariance of the
//! columns, so a pass of coordinate descent costs `O(p²)` whatever `n` is. The
//! same row pass at the next coefficients, which the objective check of the
//! step needs anyway, produces the moments of the next IRLS step, so an IRLS
//! step costs one pass over the rows, `O(n p²)` for the moments and `O(n)`
//! exponentials and logarithms, plus `O(p²)` per coordinate pass. The rows are
//! read once per step, in order, and nothing of size `n` is allocated besides
//! the outputs.
//!
//! The columns are shifted by their prior-weighted mean once, before any pass,
//! and the moments are taken about the shift, so that centring at the
//! working-weighted means, `C_jk = M_jk - S_j S_k / T`, does not subtract two
//! nearly equal large numbers when a column has a large mean and a small spread.
//! A column whose variance is no more than `16 ε` times its second moment about
//! the shift has no variation that the arithmetic can resolve and is skipped.
//!
//! glmnet also keeps an active set of the coordinates that are not zero and
//! screens the others with strong rules, which saves passes over the rows when
//! `p` is large. With the moments in hand a pass costs `O(p²)` with `p` at most
//! 32, which is below the cost of one row, so neither is
//! used here. With more columns the moments would cost more than the passes they
//! replace, and the residual form above is used.
//!
//! # Convergence
//!
//! A pass stops the coordinate descent when
//!
//! ```text
//! max_j v_j (Δβ_j)²  <=  tolerance
//! ```
//!
//! which is glmnet's `thresh`, with a column's change weighted by its
//! variance, so that the rule does not depend on the scale of a column. The
//! default is glmnet's `1e-7`. A binomial fit stops when the same quantity,
//! measured over a whole IRLS step, is at most `tolerance` and the last
//! coordinate descent converged. `converged` is `false` when the pass limit or
//! the IRLS limit is reached first, and [`ElasticNetFit::passes`] and
//! [`ElasticNetFit::iterations`] report what was used. Nothing is capped
//! silently.
//!
//! With an unpenalised constant column in the design, the other columns are
//! centred at their working-weighted means for each solve and the constant
//! column's coefficient is recovered afterwards, as glmnet does with its
//! implicit intercept. The objective is unchanged, and a column with a large
//! mean and a small spread no longer fights the intercept. `v_j` is then the
//! weighted variance of the column. Without such a column nothing is centred
//! and `v_j` is the weighted second moment. Near the solution the coefficient
//! error is at most about `sqrt(tolerance / v_j) / (1 - ρ)` for column `j`,
//! where `ρ < 1` is the contraction of a pass and grows with the correlation of
//! the columns.
//!
//! # Agreement with glmnet
//!
//! For the binomial family the objective is glmnet's. For the Gaussian family
//! glmnet standardises the response to unit weighted standard deviation `s`,
//! about its weighted mean with weights summing to 1, and solves for the same
//! `λ` on that scale. In the original units this leaves the lasso term alone
//! and divides the ridge term by `s`:
//!
//! ```text
//! (1/n) Σ w̄ (y - η)² / 2  +  λ Σ_j f_j ( α |β_j| + (1 - α)/(2 s) β_j² )
//! ```
//!
//! [`ResponseScaling::Glmnet`] does this, and the fit then agrees with
//! `glmnet(family = "gaussian", standardize = FALSE)` to the tolerance of the
//! stopping rule. [`ResponseScaling::None`], the default, uses the objective
//! above without `s`, so that `λ` means the same in both families and does not
//! change with the units of `y`. With ridge only, the two differ by a factor
//! `s` in `λ`. glmnet also rescales `penalty.factor` to sum to the number of
//! columns, which is not done here, so factors that sum to the number of
//! columns agree with glmnet and others differ by that factor in `λ`.
//!
//! # Warm starts
//!
//! `start` gives the coefficients to begin from, for a path of decreasing `λ`
//! where each fit starts from the last. The default is 0.
//!
//! # References
//!
//! 1. Friedman J, Hastie T, Tibshirani R. Regularization paths for generalized
//!    linear models via coordinate descent. J Stat Softw. 2010;33(1):1-22.
//!    doi:10.18637/jss.v033.i01
//! 2. Zou H, Hastie T. Regularization and variable selection via the elastic
//!    net. J R Stat Soc Ser B. 2005;67(2):301-20.
//!    doi:10.1111/j.1467-9868.2005.00503.x

use epistat_linalg::MatRef;

use crate::Family;

/// The largest number of columns for which the fit works from the weighted
/// moments of the design. A pass over the rows then costs about `p²/2`
/// multiply-adds per row for the moments, against `2 p` per coordinate pass for
/// the residual form, which needs about ten passes: the two break even near 32.
const COVARIANCE_MAX_COLUMNS: usize = 32;
/// Passes of coordinate descent allowed per IRLS step, by default.
const DEFAULT_MAX_PASSES: usize = 10_000;
/// Step halvings allowed when an IRLS step raises the penalised objective.
const MAX_HALVINGS: usize = 30;

/// How the Gaussian fit treats the spread of the response, see the module
/// documentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ResponseScaling {
    /// The objective as written, whatever the units of the response.
    #[default]
    None,
    /// glmnet's: the ridge term divided by the weighted standard deviation of
    /// the response.
    Glmnet,
}

/// An elastic-net model and its fitting controls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElasticNet {
    /// [`Family::Gaussian`] or [`Family::Binomial`].
    pub family: Family,
    /// The penalty per observation, `λ`.
    pub lambda: f64,
    /// The mixing parameter in `[0, 1]`, 0 for ridge and 1 for lasso.
    pub alpha: f64,
    /// The bound on the largest variance-weighted squared coefficient change,
    /// glmnet's `thresh`.
    pub tolerance: f64,
    /// Passes of coordinate descent allowed per IRLS step.
    pub max_passes: usize,
    /// IRLS steps allowed for the binomial family.
    pub max_iterations: usize,
    /// The binomial probabilities are kept within `[c, 1 - c]`, and the working
    /// weight is at least `c`.
    pub probability_clip: f64,
    /// The treatment of the response's spread, for the Gaussian family.
    pub scaling: ResponseScaling,
}

impl ElasticNet {
    /// A model with glmnet's `thresh` of `1e-7`, 10,000 passes, 100 IRLS
    /// steps, a probability clip of `1e-8` and no response scaling.
    #[must_use]
    pub const fn new(family: Family, lambda: f64, alpha: f64) -> Self {
        Self {
            family,
            lambda,
            alpha,
            tolerance: 1e-7,
            max_passes: DEFAULT_MAX_PASSES,
            max_iterations: 100,
            probability_clip: 1e-8,
            scaling: ResponseScaling::None,
        }
    }
}

/// Why an elastic net could not be fitted.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ElasticNetError {
    /// An input has the wrong length.
    #[error("{what} has {found} entries where {expected} were expected")]
    Length {
        /// The name of the input.
        what: &'static str,
        /// The length it needed.
        expected: usize,
        /// The length it had.
        found: usize,
    },
    /// The family is neither Gaussian nor binomial.
    #[error("the elastic net is defined for the Gaussian and binomial families")]
    Family,
    /// `lambda` is negative or not finite, or `alpha` is outside `[0, 1]`.
    #[error("lambda must be finite and not negative, and alpha must lie in [0, 1]")]
    Penalty,
    /// A penalty factor is negative or not finite.
    #[error("penalty factor {index} is {value}, which is negative or not finite")]
    PenaltyFactor {
        /// The position of the factor.
        index: usize,
        /// Its value.
        value: f64,
    },
    /// A prior weight is negative or not finite.
    #[error("prior weight {index} is {value}, which is negative or not finite")]
    Weight {
        /// The position of the weight.
        index: usize,
        /// Its value.
        value: f64,
    },
    /// Every prior weight is 0.
    #[error("no row has a positive prior weight")]
    NoWeight,
    /// A response lies outside the support of the family.
    #[error("response {index} is {value}, outside the family's support")]
    Response {
        /// The position of the response.
        index: usize,
        /// Its value.
        value: f64,
    },
    /// The design, offset or starting coefficients hold a value that is not
    /// finite.
    #[error("{what} holds a value that is not finite")]
    NotFinite {
        /// The name of the input.
        what: &'static str,
    },
    /// A control is out of range: the tolerance must be positive and finite,
    /// the probability clip in `(0, 0.5)`, and the limits at least 1.
    #[error("a fitting control is out of range")]
    Control,
    /// The response does not vary, so glmnet's scaling is undefined.
    #[error("the response has no weighted variation to scale by")]
    ConstantResponse,
}

/// A fitted elastic net.
#[derive(Clone, Debug)]
pub struct ElasticNetFit {
    /// The coefficients, one per column of the design.
    pub coefficients: Vec<f64>,
    /// The linear predictor, offset included.
    pub linear_predictor: Vec<f64>,
    /// The fitted means: the linear predictor for the Gaussian family, the
    /// clipped probabilities for the binomial.
    pub fitted: Vec<f64>,
    /// The penalised objective at the coefficients, as in the module
    /// documentation.
    pub objective: f64,
    /// The IRLS steps taken, 1 for the Gaussian family.
    pub iterations: usize,
    /// The passes of coordinate descent over all steps.
    pub passes: usize,
    /// Whether the stopping rule was met within the limits.
    pub converged: bool,
}

/// Soft-thresholding, `S(a, t)`.
fn soft_threshold(value: f64, threshold: f64) -> f64 {
    if value > threshold {
        value - threshold
    } else if value < -threshold {
        value + threshold
    } else {
        0.0
    }
}

/// The probability at `eta`, kept within `[clip, 1 - clip]`.
fn sigmoid_clipped(eta: f64, clip: f64) -> f64 {
    let probability = if eta >= 0.0 {
        1.0 / (1.0 + (-eta).exp())
    } else {
        let e = eta.exp();
        e / (1.0 + e)
    };
    probability.clamp(clip, 1.0 - clip)
}

/// What stays fixed across the passes: the design by columns, the scaled
/// prior weights and the per-column penalties.
struct Problem<'a> {
    columns: Vec<f64>,
    x: MatRef<'a, f64>,
    n: usize,
    p: usize,
    n_inv: f64,
    prior: Vec<f64>,
    offset: Vec<f64>,
    l1: Vec<f64>,
    l2: Vec<f64>,
    /// The first unpenalised constant column and its value.
    intercept: Option<(usize, f64)>,
}

impl Problem<'_> {
    /// `η = Xβ + o` and the probabilities clipped at `clip`.
    fn predict(&self, beta: &[f64], clip: f64, eta: &mut [f64], probability: &mut [f64]) {
        for i in 0..self.n {
            let linear: f64 = (0..self.p).map(|j| self.x[(i, j)] * beta[j]).sum::<f64>();
            eta[i] = linear + self.offset[i];
            probability[i] = sigmoid_clipped(eta[i], clip);
        }
    }

    /// The penalty `Σ_j (l1_j |β_j| + l2_j β_j² / 2)`.
    fn penalty(&self, beta: &[f64]) -> f64 {
        (0..self.p)
            .map(|j| self.l1[j] * beta[j].abs() + 0.5 * self.l2[j] * beta[j] * beta[j])
            .sum()
    }

    /// The prior-weighted mean log-loss plus the penalty.
    fn binomial_objective(&self, eta: &[f64], y: &[f64], beta: &[f64]) -> f64 {
        let loss: f64 = (0..self.n)
            .map(|i| {
                self.prior[i] * (eta[i].max(0.0) + (-eta[i].abs()).exp().ln_1p() - y[i] * eta[i])
            })
            .sum::<f64>()
            * self.n_inv;
        loss + self.penalty(beta)
    }

    /// Cyclic coordinate descent on weights `w` and residual `r = z - Xβ`,
    /// updating `beta` and `r`. Returns whether it converged and the passes.
    fn descend(
        &self,
        columns: &[f64],
        w: &[f64],
        variances: &[f64],
        r: &mut [f64],
        beta: &mut [f64],
        tolerance: f64,
        max_passes: usize,
    ) -> (bool, usize) {
        for pass in 1..=max_passes {
            let mut max_change = 0.0_f64;
            for j in 0..self.p {
                let variance = variances[j];
                if variance <= 0.0 {
                    continue;
                }
                let column = &columns[j * self.n..(j + 1) * self.n];
                let previous = beta[j];
                let gradient = column
                    .iter()
                    .zip(w)
                    .zip(r.iter())
                    .map(|((x, w), r)| w * x * r)
                    .sum::<f64>()
                    * self.n_inv;
                let target = variance * previous + gradient;
                let updated = soft_threshold(target, self.l1[j]) / (variance + self.l2[j]);
                let change = updated - previous;
                if change != 0.0 {
                    for (r, x) in r.iter_mut().zip(column) {
                        *r -= change * x;
                    }
                    beta[j] = updated;
                }
                max_change = max_change.max(variance * change * change);
            }
            if max_change <= tolerance {
                return (true, pass);
            }
        }
        (false, max_passes)
    }

    /// `v_j = (1/n) Σ W x_ij²` over `columns`.
    fn variances(&self, columns: &[f64], w: &[f64]) -> Vec<f64> {
        (0..self.p)
            .map(|j| {
                columns[j * self.n..(j + 1) * self.n]
                    .iter()
                    .zip(w)
                    .map(|(x, w)| w * x * x)
                    .sum::<f64>()
                    * self.n_inv
            })
            .collect()
    }

    /// One penalised weighted least-squares solve at weights `w` and residual
    /// `r = z - Xβ`, updating `beta` and `r`. With an unpenalised constant
    /// column the other columns are centred at their `w`-weighted means, which
    /// leaves the objective unchanged, and the constant column's coefficient is
    /// recovered afterwards. Returns whether it converged, the passes, and the
    /// largest `v_j (Δβ_j)²`.
    fn solve(
        &self,
        w: &[f64],
        r: &mut [f64],
        beta: &mut [f64],
        tolerance: f64,
        max_passes: usize,
    ) -> (bool, usize, f64) {
        let Some((k, constant)) = self.intercept else {
            let variances = self.variances(&self.columns, w);
            let before = beta.to_vec();
            let (converged, passes) =
                self.descend(&self.columns, w, &variances, r, beta, tolerance, max_passes);
            let change = (0..self.p)
                .map(|j| {
                    let d = beta[j] - before[j];
                    variances[j] * d * d
                })
                .fold(0.0_f64, f64::max);
            return (converged, passes, change);
        };
        let n = self.n;
        let total: f64 = w.iter().sum();
        let mut centred = self.columns.clone();
        let means: Vec<f64> = (0..self.p)
            .map(|j| {
                if j == k {
                    return 0.0;
                }
                let m = centred[j * n..(j + 1) * n]
                    .iter()
                    .zip(w)
                    .map(|(x, w)| w * x)
                    .sum::<f64>()
                    / total;
                for x in &mut centred[j * n..(j + 1) * n] {
                    *x -= m;
                }
                m
            })
            .collect();
        let shift = |beta: &[f64]| -> f64 {
            (0..self.p).map(|j| means[j] * beta[j]).sum::<f64>() / constant
        };
        let variances = self.variances(&centred, w);
        let mut moved = beta.to_vec();
        moved[k] += shift(beta);
        let before = moved.clone();
        let (converged, passes) = self.descend(
            &centred, w, &variances, r, &mut moved, tolerance, max_passes,
        );
        let change = (0..self.p)
            .map(|j| {
                let d = moved[j] - before[j];
                variances[j] * d * d
            })
            .fold(0.0_f64, f64::max);
        moved[k] -= shift(&moved);
        beta.copy_from_slice(&moved);
        (converged, passes, change)
    }
}

/// The weighted moments of the design at one set of IRLS weights, all divided by
/// `n` and taken about the column shift: `total = Σ W`, `residual = Σ W r`,
/// `first_j = Σ W x_j`, `cross_j = Σ W x_j r` and `second_jk = Σ W x_j x_k` for
/// `j <= k`.
struct Moments {
    total: f64,
    residual: f64,
    first: Vec<f64>,
    cross: Vec<f64>,
    second: Vec<f64>,
}

/// The loss at a set of coefficients, and the moments there when asked for.
struct Evaluation {
    loss: f64,
    moments: Option<Moments>,
}

/// The fit from the weighted moments of the design, for few columns.
struct Covariance<'a> {
    x: MatRef<'a, f64>,
    n: usize,
    p: usize,
    n_inv: f64,
    y: &'a [f64],
    prior: &'a [f64],
    offset: &'a [f64],
    l1: Vec<f64>,
    l2: Vec<f64>,
    binomial: bool,
    clip: f64,
    /// The first unpenalised constant column and its value.
    intercept: Option<(usize, f64)>,
    /// What is taken off each column before the moments: its prior-weighted
    /// mean when there is an intercept, and 0 otherwise.
    shift: Vec<f64>,
}

impl Covariance<'_> {
    /// Coefficients of the shifted design that give the same `η` as `beta` on
    /// the design as given.
    fn shifted(&self, beta: &[f64]) -> Vec<f64> {
        let mut u = beta.to_vec();
        if let Some((k, constant)) = self.intercept {
            u[k] += self.dot_excluding(k, &self.shift, beta) / constant;
        }
        u
    }

    /// The inverse of [`Covariance::shifted`].
    fn unshifted(&self, u: &[f64]) -> Vec<f64> {
        let mut beta = u.to_vec();
        if let Some((k, constant)) = self.intercept {
            beta[k] -= self.dot_excluding(k, &self.shift, u) / constant;
        }
        beta
    }

    /// `Σ_{j != k} a_j b_j`.
    fn dot_excluding(&self, k: usize, a: &[f64], b: &[f64]) -> f64 {
        (0..self.p)
            .filter(|&j| j != k)
            .map(|j| a[j] * b[j])
            .sum::<f64>()
    }

    /// The penalty `Σ_j (l1_j |β_j| + l2_j β_j² / 2)`. The unpenalised constant
    /// column is the only one whose coefficient differs between the shifted
    /// and the given design, so the penalty is the same on both.
    fn penalty(&self, u: &[f64]) -> f64 {
        (0..self.p)
            .map(|j| self.l1[j] * u[j].abs() + 0.5 * self.l2[j] * u[j] * u[j])
            .sum()
    }

    /// One pass over the rows at the shifted coefficients `u`. It writes the
    /// linear predictor and the fitted mean of every row, and returns the
    /// prior-weighted mean loss and, when `want`, the moments at the working
    /// weights and response there. The rows are visited, and the sums formed,
    /// in order.
    fn evaluate(
        &self,
        u: &[f64],
        want: bool,
        eta_out: &mut [f64],
        fitted_out: &mut [f64],
    ) -> Evaluation {
        let (n, p) = (self.n, self.p);
        let mut xs = [0.0_f64; COVARIANCE_MAX_COLUMNS];
        let xs = &mut xs[..p];
        let mut loss = 0.0_f64;
        let mut moments = want.then(|| Moments {
            total: 0.0,
            residual: 0.0,
            first: vec![0.0; p],
            cross: vec![0.0; p],
            second: vec![0.0; p * p],
        });
        for i in 0..n {
            for (j, (cell, shift)) in xs.iter_mut().zip(&self.shift).enumerate() {
                *cell = self.x[(i, j)] - shift;
            }
            let linear: f64 = xs.iter().zip(u).map(|(x, b)| x * b).sum::<f64>();
            let eta = linear + self.offset[i];
            let prior = self.prior[i];
            let (weight, residual) = if self.binomial {
                let decay = (-eta.abs()).exp();
                let probability = if eta >= 0.0 {
                    1.0 / (1.0 + decay)
                } else {
                    decay / (1.0 + decay)
                }
                .clamp(self.clip, 1.0 - self.clip);
                loss += prior * (eta.max(0.0) + decay.ln_1p() - self.y[i] * eta);
                fitted_out[i] = probability;
                let variance = (probability * (1.0 - probability)).max(self.clip);
                (prior * variance, (self.y[i] - probability) / variance)
            } else {
                let d = self.y[i] - eta;
                loss += prior * d * d;
                fitted_out[i] = eta;
                (prior, d)
            };
            eta_out[i] = eta;
            if let Some(m) = moments.as_mut() {
                if weight == 0.0 {
                    continue;
                }
                m.total += weight;
                let weighted_residual = weight * residual;
                m.residual += weighted_residual;
                for a in 0..p {
                    let wa = weight * xs[a];
                    m.first[a] += wa;
                    m.cross[a] += weighted_residual * xs[a];
                    let row = &mut m.second[a * p + a..(a + 1) * p];
                    for (cell, xb) in row.iter_mut().zip(&xs[a..p]) {
                        *cell += wa * xb;
                    }
                }
            }
        }
        let loss = if self.binomial {
            loss * self.n_inv
        } else {
            loss * (0.5 * self.n_inv)
        };
        if let Some(m) = moments.as_mut() {
            m.total *= self.n_inv;
            m.residual *= self.n_inv;
            for v in m
                .first
                .iter_mut()
                .chain(m.cross.iter_mut())
                .chain(m.second.iter_mut())
            {
                *v *= self.n_inv;
            }
        }
        Evaluation { loss, moments }
    }

    /// One penalised weighted least-squares solve on the moments `m`,
    /// updating the shifted coefficients `u`. With an unpenalised constant
    /// column the other columns are centred at their working-weighted means,
    /// which leaves the objective unchanged and is done on the moments. Returns
    /// whether it converged, the passes, and the largest `v_j (Δβ_j)²` over the
    /// solve.
    fn solve(
        &self,
        m: &Moments,
        u: &mut [f64],
        tolerance: f64,
        max_passes: usize,
    ) -> (bool, usize, f64) {
        let p = self.p;
        // The covariance C, the gradient g at the current coefficients, the
        // second moments about the shift, and the coordinates of the solve.
        let mut c = vec![0.0_f64; p * p];
        let mut g = vec![0.0_f64; p];
        let mut raw = vec![0.0_f64; p];
        let mut b = u.to_vec();
        let mut means = vec![0.0_f64; p];
        for a in 0..p {
            for k in a..p {
                c[a * p + k] = m.second[a * p + k];
            }
            raw[a] = m.second[a * p + a];
            g[a] = m.cross[a];
        }
        if let Some((k, constant)) = self.intercept {
            for (j, mean) in means.iter_mut().enumerate() {
                if j != k {
                    *mean = m.first[j] / m.total;
                }
            }
            for a in 0..p {
                for l in a..p {
                    c[a * p + l] = if a == k || l == k {
                        0.0
                    } else {
                        m.second[a * p + l] - m.first[a] * m.first[l] / m.total
                    };
                }
                g[a] = if a == k {
                    constant * m.residual
                } else {
                    m.cross[a] - means[a] * m.residual
                };
            }
            c[k * p + k] = constant * constant * m.total;
            raw[k] = c[k * p + k];
            b[k] += self.dot_excluding(k, &means, u) / constant;
        }
        for a in 0..p {
            for l in 0..a {
                c[a * p + l] = c[l * p + a];
            }
        }
        let before = b.clone();
        let usable: Vec<bool> = (0..p)
            .map(|j| c[j * p + j] > 0.0 && c[j * p + j] > 16.0 * f64::EPSILON * raw[j])
            .collect();
        let mut converged = false;
        let mut passes = max_passes;
        for pass in 1..=max_passes {
            let mut max_change = 0.0_f64;
            for j in 0..p {
                if !usable[j] {
                    continue;
                }
                let variance = c[j * p + j];
                let previous = b[j];
                let target = variance * previous + g[j];
                let updated = soft_threshold(target, self.l1[j]) / (variance + self.l2[j]);
                let change = updated - previous;
                if change != 0.0 {
                    for (gi, cij) in g.iter_mut().zip(&c[j * p..(j + 1) * p]) {
                        *gi -= cij * change;
                    }
                    b[j] = updated;
                }
                max_change = max_change.max(variance * change * change);
            }
            if max_change <= tolerance {
                converged = true;
                passes = pass;
                break;
            }
        }
        let change = (0..p)
            .map(|j| {
                let d = b[j] - before[j];
                c[j * p + j] * d * d
            })
            .fold(0.0_f64, f64::max);
        u.copy_from_slice(&b);
        if let Some((k, constant)) = self.intercept {
            u[k] -= self.dot_excluding(k, &means, &b) / constant;
        }
        (converged, passes, change)
    }

    /// The fit, from the coefficients `start` on the design as given.
    fn run(mut self, model: &ElasticNet, start: Option<&[f64]>) -> ElasticNetFit {
        let (n, p) = (self.n, self.p);
        if self.intercept.is_some() {
            let total: f64 = self.prior.iter().sum();
            self.shift = (0..p)
                .map(|j| {
                    if self.intercept.is_some_and(|(k, _)| k == j) {
                        0.0
                    } else {
                        (0..n).map(|i| self.prior[i] * self.x[(i, j)]).sum::<f64>() / total
                    }
                })
                .collect();
        } else {
            self.shift = vec![0.0; p];
        }
        let mut u = self.shifted(&start.map_or_else(|| vec![0.0; p], <[f64]>::to_vec));
        let mut eta = vec![0.0; n];
        let mut fitted = vec![0.0; n];
        let mut evaluation = self.evaluate(&u, true, &mut eta, &mut fitted);

        if !self.binomial {
            let moments = evaluation.moments.take().expect("moments were asked for");
            let (converged, passes, _) =
                self.solve(&moments, &mut u, model.tolerance, model.max_passes);
            let last = self.evaluate(&u, false, &mut eta, &mut fitted);
            return ElasticNetFit {
                objective: last.loss + self.penalty(&u),
                coefficients: self.unshifted(&u),
                linear_predictor: eta,
                fitted,
                iterations: 1,
                passes,
                converged,
            };
        }

        // The objective is a sum of `n` terms, and a recursive sum of `n` terms
        // has a relative error of at most `n ε`. A candidate that is higher by no
        // more than that is a tie, not a rise.
        let noise = n as f64 * f64::EPSILON;
        let mut current = evaluation.loss + self.penalty(&u);
        let (mut converged, mut iterations, mut passes) = (false, 0, 0);
        for step in 1..=model.max_iterations {
            iterations = step;
            let moments = match evaluation.moments.take() {
                Some(moments) => moments,
                None => self
                    .evaluate(&u, true, &mut eta, &mut fitted)
                    .moments
                    .expect("moments were asked for"),
            };
            let before = u.clone();
            let (inner_converged, used, change) =
                self.solve(&moments, &mut u, model.tolerance, model.max_passes);
            passes += used;
            let done = change <= model.tolerance && inner_converged;

            // Halve towards the previous coefficients until the penalised
            // objective does not rise. The moments for the next step come with
            // the evaluation, unless the fit has converged.
            let full = u.clone();
            let mut accepted = false;
            let mut scale = 1.0_f64;
            for _ in 0..=MAX_HALVINGS {
                for j in 0..p {
                    u[j] = before[j] + scale * (full[j] - before[j]);
                }
                let candidate = self.evaluate(&u, !done, &mut eta, &mut fitted);
                let objective = candidate.loss + self.penalty(&u);
                if objective <= current + noise * current.abs().max(1.0) {
                    current = objective;
                    evaluation = candidate;
                    accepted = true;
                    break;
                }
                scale *= 0.5;
            }
            if !accepted {
                u.copy_from_slice(&before);
                self.evaluate(&u, false, &mut eta, &mut fitted);
                break;
            }
            if done {
                converged = true;
                break;
            }
        }
        ElasticNetFit {
            objective: current,
            coefficients: self.unshifted(&u),
            linear_predictor: eta,
            fitted,
            iterations,
            passes,
            converged,
        }
    }
}

fn check(what: &'static str, expected: usize, found: usize) -> Result<(), ElasticNetError> {
    if expected == found {
        Ok(())
    } else {
        Err(ElasticNetError::Length {
            what,
            expected,
            found,
        })
    }
}

impl ElasticNet {
    /// Fits the model to `y` on the columns of `x`.
    ///
    /// `weights` are prior weights (1 when `None`), `offset` is added to the
    /// linear predictor (0 when `None`), `penalty_factor` multiplies the
    /// penalty of each column (1 when `None`, 0 for an unpenalised column such
    /// as the intercept) and `start` gives the starting coefficients (0 when
    /// `None`).
    ///
    /// A fit that does not meet the stopping rule within its limits is
    /// returned with `converged` false and the coefficients it reached.
    ///
    /// # Errors
    ///
    /// See [`ElasticNetError`].
    pub fn fit(
        &self,
        x: MatRef<'_, f64>,
        y: &[f64],
        weights: Option<&[f64]>,
        offset: Option<&[f64]>,
        penalty_factor: Option<&[f64]>,
        start: Option<&[f64]>,
    ) -> Result<ElasticNetFit, ElasticNetError> {
        let covariance = x.ncols() <= COVARIANCE_MAX_COLUMNS;
        self.fit_with(covariance, x, y, weights, offset, penalty_factor, start)
    }

    /// [`ElasticNet::fit`] with the solver named: the weighted moments of the
    /// design when `covariance`, the residual form otherwise.
    fn fit_with(
        &self,
        covariance: bool,
        x: MatRef<'_, f64>,
        y: &[f64],
        weights: Option<&[f64]>,
        offset: Option<&[f64]>,
        penalty_factor: Option<&[f64]>,
        start: Option<&[f64]>,
    ) -> Result<ElasticNetFit, ElasticNetError> {
        let (n, p) = (x.nrows(), x.ncols());
        let binomial = match self.family {
            Family::Gaussian => false,
            Family::Binomial => true,
            _ => return Err(ElasticNetError::Family),
        };
        if !(self.lambda.is_finite() && self.lambda >= 0.0 && (0.0..=1.0).contains(&self.alpha)) {
            return Err(ElasticNetError::Penalty);
        }
        if !(self.tolerance.is_finite() && self.tolerance > 0.0)
            || self.max_passes == 0
            || self.max_iterations == 0
            || !(self.probability_clip > 0.0 && self.probability_clip < 0.5)
        {
            return Err(ElasticNetError::Control);
        }
        check("y", n, y.len())?;
        for (what, values) in [("weights", weights), ("offset", offset)] {
            if let Some(values) = values {
                check(what, n, values.len())?;
            }
        }
        for (what, values) in [("penalty_factor", penalty_factor), ("start", start)] {
            if let Some(values) = values {
                check(what, p, values.len())?;
            }
        }
        if n == 0 {
            return Err(ElasticNetError::NoWeight);
        }
        if let Some((index, &value)) = y
            .iter()
            .enumerate()
            .find(|&(_, &v)| !self.family.valid_response(v))
        {
            return Err(ElasticNetError::Response { index, value });
        }
        if let Some((index, &value)) = weights.and_then(|w| {
            w.iter()
                .enumerate()
                .find(|&(_, &v)| !(v.is_finite() && v >= 0.0))
        }) {
            return Err(ElasticNetError::Weight { index, value });
        }
        if let Some((index, &value)) = penalty_factor.and_then(|f| {
            f.iter()
                .enumerate()
                .find(|&(_, &v)| !(v.is_finite() && v >= 0.0))
        }) {
            return Err(ElasticNetError::PenaltyFactor { index, value });
        }
        if offset.is_some_and(|o| o.iter().any(|v| !v.is_finite())) {
            return Err(ElasticNetError::NotFinite { what: "offset" });
        }
        if start.is_some_and(|s| s.iter().any(|v| !v.is_finite())) {
            return Err(ElasticNetError::NotFinite { what: "start" });
        }
        if (0..p).any(|j| (0..n).any(|i| !x[(i, j)].is_finite())) {
            return Err(ElasticNetError::NotFinite { what: "x" });
        }

        // Prior weights scaled to mean 1. Unit weights stay exactly 1.
        let prior: Vec<f64> = match weights {
            None => vec![1.0; n],
            Some(w) => {
                let total: f64 = w.iter().sum();
                if total <= 0.0 || !total.is_finite() {
                    return Err(ElasticNetError::NoWeight);
                }
                let scale = n as f64 / total;
                w.iter().map(|v| v * scale).collect()
            }
        };
        let offset: Vec<f64> = offset.map_or_else(|| vec![0.0; n], <[f64]>::to_vec);
        let n_inv = 1.0 / n as f64;

        let mut l2_scale = 1.0;
        if !binomial && self.scaling == ResponseScaling::Glmnet {
            // Weighted standard deviation of y - o, weights summing to 1.
            let total: f64 = prior.iter().sum();
            let mean: f64 = (0..n).map(|i| prior[i] * (y[i] - offset[i])).sum::<f64>() / total;
            let variance: f64 = (0..n)
                .map(|i| {
                    let d = y[i] - offset[i] - mean;
                    prior[i] * d * d
                })
                .sum::<f64>()
                / total;
            let scale = variance.sqrt();
            if !(scale > 0.0 && scale.is_finite()) {
                return Err(ElasticNetError::ConstantResponse);
            }
            l2_scale = 1.0 / scale;
        }
        let l1_base = self.lambda * self.alpha;
        let l2_base = self.lambda * (1.0 - self.alpha) * l2_scale;
        let factor = |j: usize| penalty_factor.map_or(1.0, |f| f[j]);

        if covariance {
            assert!(p <= COVARIANCE_MAX_COLUMNS);
            let intercept = (0..p).find(|&j| {
                factor(j) == 0.0
                    && x[(0, j)] != 0.0
                    && (1..n).all(|i| x[(i, j)].to_bits() == x[(0, j)].to_bits())
            });
            let moments = Covariance {
                x,
                n,
                p,
                n_inv,
                y,
                prior: &prior,
                offset: &offset,
                l1: (0..p).map(|j| l1_base * factor(j)).collect(),
                l2: (0..p).map(|j| l2_base * factor(j)).collect(),
                binomial,
                clip: self.probability_clip,
                intercept: intercept.map(|j| (j, x[(0, j)])),
                shift: Vec::new(),
            };
            return Ok(moments.run(self, start));
        }

        let mut columns = vec![0.0_f64; n * p];
        for j in 0..p {
            for i in 0..n {
                columns[j * n + i] = x[(i, j)];
            }
        }
        let problem = Problem {
            columns,
            x,
            n,
            p,
            n_inv,
            prior,
            offset,
            l1: (0..p).map(|j| l1_base * factor(j)).collect(),
            l2: (0..p).map(|j| l2_base * factor(j)).collect(),
            intercept: None,
        };
        let intercept = (0..p).find(|&j| {
            factor(j) == 0.0
                && x[(0, j)] != 0.0
                && (1..n).all(|i| x[(i, j)].to_bits() == x[(0, j)].to_bits())
        });
        let problem = Problem {
            intercept: intercept.map(|j| (j, x[(0, j)])),
            ..problem
        };
        let mut beta = start.map_or_else(|| vec![0.0; p], <[f64]>::to_vec);
        let mut eta = vec![0.0; n];
        let mut probability = vec![0.0; n];
        problem.predict(&beta, self.probability_clip, &mut eta, &mut probability);

        if !binomial {
            let mut r: Vec<f64> = (0..n).map(|i| y[i] - eta[i]).collect();
            let (converged, passes, _) = problem.solve(
                &problem.prior,
                &mut r,
                &mut beta,
                self.tolerance,
                self.max_passes,
            );
            problem.predict(&beta, self.probability_clip, &mut eta, &mut probability);
            let loss: f64 = (0..n)
                .map(|i| {
                    let d = y[i] - eta[i];
                    problem.prior[i] * d * d
                })
                .sum::<f64>()
                * (0.5 * n_inv);
            return Ok(ElasticNetFit {
                objective: loss + problem.penalty(&beta),
                coefficients: beta,
                fitted: eta.clone(),
                linear_predictor: eta,
                iterations: 1,
                passes,
                converged,
            });
        }

        let clip = self.probability_clip;
        let mut current = problem.binomial_objective(&eta, y, &beta);
        let mut w = vec![0.0; n];
        let mut r = vec![0.0; n];
        let (mut converged, mut iterations, mut passes) = (false, 0, 0);
        for step in 1..=self.max_iterations {
            iterations = step;
            for i in 0..n {
                let variance = (probability[i] * (1.0 - probability[i])).max(clip);
                w[i] = problem.prior[i] * variance;
                r[i] = (y[i] - probability[i]) / variance;
            }
            let before = beta.clone();
            let (inner_converged, used, change) =
                problem.solve(&w, &mut r, &mut beta, self.tolerance, self.max_passes);
            passes += used;

            // Halve towards the previous coefficients until the penalised
            // objective does not rise.
            let full = beta.clone();
            let mut accepted = false;
            let mut scale = 1.0_f64;
            for _ in 0..=MAX_HALVINGS {
                for j in 0..p {
                    beta[j] = before[j] + scale * (full[j] - before[j]);
                }
                problem.predict(&beta, clip, &mut eta, &mut probability);
                let candidate = problem.binomial_objective(&eta, y, &beta);
                if candidate <= current + 1e-14 * current.abs().max(1.0) {
                    current = candidate;
                    accepted = true;
                    break;
                }
                scale *= 0.5;
            }
            if !accepted {
                beta.copy_from_slice(&before);
                problem.predict(&beta, clip, &mut eta, &mut probability);
                break;
            }
            if change <= self.tolerance && inner_converged {
                converged = true;
                break;
            }
        }
        Ok(ElasticNetFit {
            objective: problem.binomial_objective(&eta, y, &beta),
            coefficients: beta,
            linear_predictor: eta,
            fitted: probability,
            iterations,
            passes,
            converged,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use epistat_linalg::Mat;

    fn data(n: usize) -> (Mat<f64>, Vec<f64>) {
        let x = Mat::from_fn(n, 3, |i, j| {
            let t = i as f64;
            match j {
                0 => 1.0,
                1 => (t * 0.7).sin() * 2.0,
                _ => (t * 0.3).cos() + 0.5 * (t * 0.7).sin(),
            }
        });
        let y = (0..n)
            .map(|i| 1.0 + 0.8 * x[(i, 1)] - 0.5 * x[(i, 2)] + (i as f64 * 1.3).sin() * 0.4)
            .collect();
        (x, y)
    }

    #[test]
    fn no_penalty_is_least_squares() {
        let (x, y) = data(50);
        let enet = ElasticNet {
            tolerance: 1e-24,
            ..ElasticNet::new(Family::Gaussian, 0.0, 0.5)
        };
        let fit = enet.fit(x.as_ref(), &y, None, None, None, None).unwrap();
        let glm = crate::Glm::new(Family::Gaussian, crate::Link::Identity)
            .fit(x.as_ref(), &y, None, None, None)
            .unwrap();
        assert!(fit.converged);
        for (a, b) in fit.coefficients.iter().zip(&glm.coefficients) {
            assert!((a - b.unwrap()).abs() < 1e-10, "{a} {b:?}");
        }
    }

    #[test]
    fn ridge_has_a_closed_form() {
        // (X'WX / n + lambda D) b = X'W y / n with D zero for the intercept.
        let (x, y) = data(40);
        let lambda = 0.3;
        let enet = ElasticNet {
            tolerance: 1e-26,
            ..ElasticNet::new(Family::Gaussian, lambda, 0.0)
        };
        let factors = [0.0, 1.0, 1.0];
        let fit = enet
            .fit(x.as_ref(), &y, None, None, Some(&factors), None)
            .unwrap();
        let mut a = Mat::from_fn(3, 3, |j, k| {
            (0..40).map(|i| x[(i, j)] * x[(i, k)]).sum::<f64>() / 40.0
        });
        for j in 0..3 {
            a[(j, j)] += lambda * factors[j];
        }
        let b = Mat::from_fn(3, 1, |j, _| {
            (0..40).map(|i| x[(i, j)] * y[i]).sum::<f64>() / 40.0
        });
        let solved =
            epistat_linalg::solve(a.as_ref(), b.as_ref(), &epistat_linalg::Attempt::SYMMETRIC)
                .unwrap();
        for j in 0..3 {
            assert!((fit.coefficients[j] - solved.x[(j, 0)]).abs() < 1e-11);
        }
    }

    #[test]
    fn a_large_lasso_penalty_zeroes_the_penalised_columns() {
        let (x, y) = data(50);
        let factors = [0.0, 1.0, 1.0];
        let fit = ElasticNet::new(Family::Gaussian, 100.0, 1.0)
            .fit(x.as_ref(), &y, None, None, Some(&factors), None)
            .unwrap();
        assert_eq!(fit.coefficients[1].to_bits(), 0.0_f64.to_bits());
        assert_eq!(fit.coefficients[2].to_bits(), 0.0_f64.to_bits());
        let mean = y.iter().sum::<f64>() / 50.0;
        assert!((fit.coefficients[0] - mean).abs() < 1e-12);
    }

    #[test]
    fn a_pass_limit_is_reported() {
        let (x, y) = data(50);
        let enet = ElasticNet {
            max_passes: 1,
            tolerance: 1e-30,
            ..ElasticNet::new(Family::Gaussian, 0.1, 0.5)
        };
        let fit = enet.fit(x.as_ref(), &y, None, None, None, None).unwrap();
        assert!(!fit.converged);
        assert_eq!(fit.passes, 1);
    }

    #[test]
    fn a_warm_start_at_the_solution_takes_one_pass() {
        let (x, y) = data(50);
        let enet = ElasticNet::new(Family::Gaussian, 0.05, 0.5);
        let cold = enet.fit(x.as_ref(), &y, None, None, None, None).unwrap();
        let warm = enet
            .fit(x.as_ref(), &y, None, None, None, Some(&cold.coefficients))
            .unwrap();
        assert!(warm.converged && warm.passes <= 2 && warm.passes < cold.passes);
    }

    #[test]
    fn binomial_without_penalty_is_the_logistic_fit() {
        let (x, _) = data(80);
        let y: Vec<f64> = (0..80)
            .map(|i| {
                f64::from(u8::from(
                    x[(i, 1)] * 0.8 + (i as f64 * 2.1).sin() * 3.0 > 0.3,
                ))
            })
            .collect();
        let enet = ElasticNet {
            tolerance: 1e-18,
            ..ElasticNet::new(Family::Binomial, 0.0, 0.5)
        };
        let fit = enet.fit(x.as_ref(), &y, None, None, None, None).unwrap();
        let glm = crate::Glm::new(Family::Binomial, crate::Link::Logit)
            .fit(x.as_ref(), &y, None, None, None)
            .unwrap();
        assert!(fit.converged);
        for (a, b) in fit.coefficients.iter().zip(&glm.coefficients) {
            assert!((a - b.unwrap()).abs() < 1e-7, "{a} {b:?}");
        }
    }

    #[test]
    fn zero_prior_weights_drop_rows() {
        let (x, y) = data(50);
        let mut weights = vec![1.0; 50];
        for w in &mut weights[40..] {
            *w = 0.0;
        }
        let enet = ElasticNet {
            tolerance: 1e-24,
            ..ElasticNet::new(Family::Gaussian, 0.05, 0.5)
        };
        let weighted = enet
            .fit(x.as_ref(), &y, Some(&weights), None, None, None)
            .unwrap();
        let head = Mat::from_fn(40, 3, |i, j| x[(i, j)]);
        let dropped = enet
            .fit(head.as_ref(), &y[..40], None, None, None, None)
            .unwrap();
        for (a, b) in weighted.coefficients.iter().zip(&dropped.coefficients) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    /// A small deterministic generator, so the cases do not depend on a crate.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 11) as f64) / ((1_u64 << 53) as f64)
        }
    }

    struct Case {
        x: Mat<f64>,
        y: Vec<f64>,
        weights: Vec<f64>,
        offset: Vec<f64>,
        factors: Vec<f64>,
    }

    /// `n` rows and `p` columns: an unpenalised constant, one column with a mean
    /// of 1000 and a spread of 1, one with a spread of 1e-3, one with a spread of
    /// 1e2, and correlated noise columns.
    fn case(seed: u64, n: usize, p: usize, binomial: bool, intercept: bool) -> Case {
        let mut rng = Lcg(seed);
        let mut x = Mat::from_fn(n, p, |_, _| 0.0);
        for i in 0..n {
            let common = rng.next() - 0.5;
            for j in 0..p {
                let noise = rng.next() - 0.5;
                x[(i, j)] = match j {
                    0 if intercept => 1.0,
                    1 => 1000.0 + 3.0 * noise,
                    2 => 1e-3 * noise,
                    3 => 1e2 * noise,
                    _ => noise + 0.6 * common,
                };
            }
        }
        let y = (0..n)
            .map(|i| {
                let signal = 0.4 * x[(i, p - 1)] - 0.3 * x[(i, p.min(5) - 1)]
                    + 0.01 * (x[(i, 3.min(p - 1))]);
                let noise = rng.next() - 0.5;
                if binomial {
                    f64::from(u8::from(
                        rng.next() < 1.0 / (1.0 + (-(signal + noise)).exp()),
                    ))
                } else {
                    signal + noise
                }
            })
            .collect();
        let weights = (0..n).map(|_| 0.5 + 2.0 * rng.next()).collect();
        let offset = (0..n).map(|_| 0.3 * (rng.next() - 0.5)).collect();
        let factors = (0..p)
            .map(|j| if intercept && j == 0 { 0.0 } else { 1.0 })
            .collect();
        Case {
            x,
            y,
            weights,
            offset,
            factors,
        }
    }

    /// The solver on the moments of the design and the one on the residual
    /// agree on four sizes, both families, with and without an intercept, with
    /// and without prior weights, offset and starting coefficients, and from
    /// ridge to lasso. Both stop at `thresh = 1e-20` on `max_j v_j (Δβ_j)²`, so
    /// a coefficient of a column with weighted variance `v` is determined to
    /// about `sqrt(thresh / v)`. The two differ by at most twice that, times 10
    /// for the correlation of the columns. The prior weights are at least 0.5
    /// and the linear predictors stay within 1.5 of 0, so the working weights
    /// are at least 0.1 and `v` is at least `0.05` times the variance of the
    /// column, which the bound uses. The linear predictor
    /// differs by at most the sum over columns of the half-range times that
    /// bound. The columns have means from 0 to 1000 and ranges from 3e-3 to 100.
    #[test]
    fn the_moment_solver_agrees_with_the_residual_solver() {
        let mut worst = 0.0_f64;
        for (seed, n, p) in [(1, 60, 6), (2, 200, 8), (3, 500, 5), (4, 150, 12)] {
            for binomial in [false, true] {
                for intercept in [false, true] {
                    for (lambda, alpha) in [(0.0, 0.5), (0.01, 1.0), (0.05, 0.5), (0.3, 0.0)] {
                        for variant in 0..4 {
                            let c = case(seed, n, p, binomial, intercept);
                            let family = if binomial {
                                Family::Binomial
                            } else {
                                Family::Gaussian
                            };
                            let enet = ElasticNet {
                                tolerance: 1e-20,
                                max_iterations: 200,
                                scaling: if variant == 3 && !binomial {
                                    ResponseScaling::Glmnet
                                } else {
                                    ResponseScaling::None
                                },
                                ..ElasticNet::new(family, lambda, alpha)
                            };
                            let weights = (variant % 2 == 1).then_some(c.weights.as_slice());
                            let offset = (variant >= 2).then_some(c.offset.as_slice());
                            let start: Vec<f64> = (0..p).map(|j| 0.01 * j as f64).collect();
                            let start = (variant == 1).then_some(start.as_slice());
                            let run = |covariance: bool| {
                                enet.fit_with(
                                    covariance,
                                    c.x.as_ref(),
                                    &c.y,
                                    weights,
                                    offset,
                                    Some(&c.factors),
                                    start,
                                )
                                .unwrap()
                            };
                            let (a, b) = (run(true), run(false));
                            let label = format!(
                                "seed {seed} binomial {binomial} intercept {intercept} lambda {lambda} variant {variant}"
                            );
                            assert_eq!(a.converged, b.converged, "{label}");
                            // Half the range of each column about its mean.
                            let half = |j: usize| match j {
                                1 => 1.5,
                                2 => 1.5e-3,
                                3 => 50.0,
                                _ => 0.8,
                            };
                            let mut eta_bound = 0.0;
                            for j in usize::from(intercept)..p {
                                let spread = half(j) / 3.0_f64.sqrt();
                                let bound = 10.0 * 2.0 * 1e-10 / (0.05 * spread * spread).sqrt();
                                eta_bound += half(j) * bound;
                                let dev = (a.coefficients[j] - b.coefficients[j]).abs();
                                worst = worst.max(dev / bound);
                                assert!(dev <= bound, "{label} column {j}: {dev} against {bound}");
                            }
                            for i in 0..n {
                                let dev = (a.linear_predictor[i] - b.linear_predictor[i]).abs();
                                assert!(
                                    dev <= eta_bound,
                                    "{label} eta {i}: {dev} against {eta_bound}"
                                );
                            }
                            assert!(
                                (a.objective - b.objective).abs()
                                    <= 1e-12 * b.objective.abs().max(1.0),
                                "{label} objective"
                            );
                        }
                    }
                }
            }
        }
        assert!(worst <= 1.0);
    }

    #[test]
    fn a_fit_repeats_to_the_bit() {
        let (x, y) = data(60);
        let enet = ElasticNet::new(Family::Gaussian, 0.05, 0.3);
        let a = enet.fit(x.as_ref(), &y, None, None, None, None).unwrap();
        let b = enet.fit(x.as_ref(), &y, None, None, None, None).unwrap();
        let bits = |f: &ElasticNetFit| -> Vec<u64> {
            f.coefficients.iter().map(|v| v.to_bits()).collect()
        };
        assert_eq!(bits(&a), bits(&b));
    }

    #[test]
    fn invalid_input_is_an_error() {
        let (x, y) = data(20);
        let ok = ElasticNet::new(Family::Gaussian, 0.1, 0.5);
        let run = |enet: ElasticNet, y: &[f64], w: Option<&[f64]>, f: Option<&[f64]>| {
            enet.fit(x.as_ref(), y, w, None, f, None).unwrap_err()
        };
        assert_eq!(
            run(ElasticNet::new(Family::Poisson, 0.1, 0.5), &y, None, None),
            ElasticNetError::Family
        );
        assert_eq!(
            run(ElasticNet::new(Family::Gaussian, -1.0, 0.5), &y, None, None),
            ElasticNetError::Penalty
        );
        assert_eq!(
            run(ElasticNet::new(Family::Gaussian, 1.0, 1.5), &y, None, None),
            ElasticNetError::Penalty
        );
        assert!(matches!(
            run(ok, &y[..5], None, None),
            ElasticNetError::Length { what: "y", .. }
        ));
        assert!(matches!(
            run(ok, &y, None, Some(&[1.0, -1.0, 1.0])),
            ElasticNetError::PenaltyFactor { index: 1, .. }
        ));
        let mut weights = vec![1.0; 20];
        weights[2] = f64::NAN;
        assert!(matches!(
            run(ok, &y, Some(&weights), None),
            ElasticNetError::Weight { index: 2, .. }
        ));
        assert_eq!(
            run(ok, &y, Some(&[0.0; 20]), None),
            ElasticNetError::NoWeight
        );
        assert!(matches!(
            run(ElasticNet::new(Family::Binomial, 0.1, 0.5), &y, None, None),
            ElasticNetError::Response { .. }
        ));
        let scaled = ElasticNet {
            scaling: ResponseScaling::Glmnet,
            ..ok
        };
        assert_eq!(
            run(scaled, &[1.0; 20], None, None),
            ElasticNetError::ConstantResponse
        );
    }
}
