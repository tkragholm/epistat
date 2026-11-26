//! The covariate balancing propensity score of Imai and Ratkovic \[1\].
//!
//! # Model and moment conditions
//!
//! The propensity score is logistic, `p_i = 1 / (1 + exp(-x_i' b))`, and `b` is
//! found by the generalised method of moments from two sets of conditions.
//! With `T_i` the treatment indicator, `s_i` the sampling weight scaled to
//! mean 1, `n = sum s_i` and `n_t = sum s_i T_i`, the moment vector is
//!
//! ```text
//! gbar(b) = (1/n) sum s_i ( x_i (T_i - p_i) ,  x_i w_i )
//! ```
//!
//! The first block is the score of the logistic likelihood. The second block is
//! the covariate balance of the weighted groups, and `w_i` depends on the
//! estimand:
//!
//! ```text
//! ATT   w_i = (n / n_t) (T_i - p_i) / (1 - p_i)     treated 1, controls -odds
//! ATE   w_i = 1 / (p_i - 1 + T_i)                   T/p - (1-T)/(1-p)
//! ```
//!
//! The treated are the anchors. With the ATT weights the balance conditions set
//! the weighted mean of every covariate in the controls to its mean in the
//! treated. With the ATE weights they set both groups' weighted means to the
//! full-sample mean.
//!
//! # Estimators
//!
//! The exact (just-identified) estimator uses the balance block only and
//! minimises `m' M m` with `m = (1/n) sum s_i x_i w_i` and `M = (X' S X)^-`,
//! which is zero when the balance conditions hold exactly. The over-identified
//! estimator minimises `J(b) = gbar' W gbar` over both blocks, where `W` is the
//! generalised inverse of the model covariance of the moments.
//!
//! The covariance is evaluated under the model, with `p` for the probability of
//! treatment, as the paper does:
//!
//! ```text
//! ATT   V = (1/n) [ X' S P(1-P) X         (n/n_t) X' S P X        ]
//!                 [ (n/n_t) X' S P X      (n/n_t)^2 X' S P/(1-P) X ]
//! ATE   V = (1/n) [ X' S P(1-P) X         X' S X                  ]
//!                 [ X' S X                X' S 1/(P(1-P)) X       ]
//! ```
//!
//! `W` is `V^-`, with the eigenvalues at most `sqrt(eps)` times the largest
//! dropped, which is the rule of `MASS::ginv`. The two-step form that CBPS uses
//! by default evaluates `V` once, at a starting value, and keeps it fixed while
//! `J` is minimised, so the objective is a quadratic form in `gbar` with a
//! fixed matrix and its gradient is `2 D' W gbar`, with `D` the derivative of
//! `gbar`.
//!
//! # Steps
//!
//! 1. The design is standardised. Each column other than the intercept is
//!    centred and divided by its standard deviation, and the standardised
//!    columns with the intercept are replaced by the left singular vectors `U`
//!    of that matrix, so that `U' U = I`. `X b = U g` with `g = D V' b` for the
//!    singular value decomposition `X = U D V'`, and the coefficients on the
//!    scale of the covariates are recovered from `g` by the inverse of this map.
//!    The decomposition is computed from the eigenvectors of `X' X`. A design
//!    whose smallest singular value is at most 1e-7 of the largest is not of
//!    full rank, and is an error.
//! 2. The logistic regression of `T` on `U` gives `g0`. The scale factor `a` in
//!    `[0.8, 1.1]` that minimises the over-identified `J(a g0)`, with `V`
//!    recomputed at each `a`, is found by Brent's method with tolerance
//!    `eps^(1/4)`, and `g_init = a g0`. `V` at `g_init` is the fixed `W`.
//! 3. BFGS minimises the balance loss from `g_init`, which gives the exact
//!    estimate `g_bal`. For the exact estimator that is the result.
//! 4. For the over-identified estimator BFGS minimises `J` with the fixed `W`
//!    from `g_init` and from `g_bal`, and the fit with the lower `J` is kept.
//!
//! The probabilities are bounded to `[1e-6, 1 - 1e-6]` in the moment conditions
//! and in the returned scores, as CBPS does. The gradient does not differentiate
//! the bound, which has no effect unless a score reaches it. A fit with a score
//! at the bound is reported through `fitted_at_boundary`.
//!
//! # Stopping rule
//!
//! [`epistat_optim::bfgs::Bfgs`] stops when the gradient norm is below
//! [`CbpsConfig::gradient_tolerance`], when the loss changes by less than
//! four times the estimated rounding error of the loss at the start of the
//! search, or after [`CbpsConfig::max_iter`] iterations. The rounding error
//! of the exact estimator's loss, which is zero at the solution, is `eps` times
//! its value. That of `J` is `2 eps ||W|| |gbar|^2`, which is large when the
//! covariance of the moments is nearly singular. A fit that stops by none of the first two rules, because the
//! iteration limit is reached or a step cannot decrease the loss, is reported
//! as not converged and is not repaired.
//!
//! # References
//!
//! 1. Imai K, Ratkovic M. Covariate balancing propensity score. J R Stat Soc
//!    Ser B. 2014;76(1):243-263. doi:10.1111/rssb.12027
//! 2. Brent RP. Algorithms for Minimization without Derivatives. Prentice-Hall;
//!    1973. Chapter 5.
//! 3. Hansen LP. Large sample properties of generalized method of moments
//!    estimators. Econometrica. 1982;50(4):1029-1054.

#![expect(
    deprecated,
    reason = "the fits take their data as separate arguments, the epistat-glm 0.2 form"
)]

use super::EstimationError;
use super::config::{CbpsConfig, CbpsEstimand, CbpsMethod};
use super::encoding::CovariateMatrix;
use super::linalg::pseudo_inverse;
use super::logistic::LogisticFit;
use epistat_glm::{Family, Glm, Link};
use epistat_linalg::eigen::symmetric_eigen_jacobi;
use epistat_linalg::{Mat, MatRef, matvec_into, weighted_gram, xtv};
use epistat_optim::bfgs::Bfgs;

/// The bounds of a probability in the moment conditions.
const PROBABILITY_BOUND: f64 = 1e-6;

/// The smallest singular value of the design, relative to the largest, below
/// which the design is not of full rank. It is the tolerance of R's `qr()`.
const RANK_TOLERANCE: f64 = 1e-7;

/// The multiple of the estimated rounding error of the loss below which a
/// change of the loss is taken as rounding.
const ROUNDING_FACTOR: f64 = 4.0;

/// The bracket of the scale factor of the logistic fit.
const SCALE_BRACKET: (f64, f64) = (0.8, 1.1);

/// The statistics of a CBPS fit besides the scores.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CbpsFit {
    /// `J`, the over-identified objective `gbar' W gbar` at the fitted
    /// coefficients, with the weight matrix `W` fixed at the starting value.
    /// Without the factor `n` of the test statistic. It is also reported for
    /// the exact estimator.
    pub j_statistic: f64,
    /// `J` at the logistic fit, with the same weight matrix.
    pub logistic_j_statistic: f64,
    /// The scale factor of the logistic fit that gave the starting value.
    pub start_scale: f64,
    /// The gradient norm of the minimised loss at the fit, in the
    /// orthogonalised coordinates.
    pub gradient_norm: f64,
    /// The Euclidean norm of the balance moments `(1/n) U' S w` at the fit,
    /// in the orthogonalised coordinates. It is zero when the balance
    /// conditions hold exactly.
    pub balance_residual_norm: f64,
}

/// The design after standardisation and orthogonalisation, with what is needed
/// to map the coefficients back.
struct Orthogonal {
    /// `n x k`, `U' U = I`.
    u: Mat<f64>,
    /// Singular values, largest first.
    d: Vec<f64>,
    /// Right singular vectors, row-major `k x k`, columns in the order of `d`.
    v: Vec<f64>,
    /// Which design columns entered, the intercept first.
    columns: Vec<usize>,
    mean: Vec<f64>,
    sd: Vec<f64>,
}

fn orthogonalise(matrix: &CovariateMatrix) -> Result<Orthogonal, EstimationError> {
    let n = matrix.n_rows;
    let intercept = matrix
        .column_names
        .iter()
        .position(|name| name == "intercept")
        .ok_or(EstimationError::CbpsNeedsIntercept)?;
    let mut columns = vec![intercept];
    let mut mean = Vec::new();
    let mut sd = Vec::new();
    for column in 0..matrix.n_cols {
        if column == intercept {
            continue;
        }
        let values = (0..n).map(|row| matrix.values[row * matrix.n_cols + column]);
        let m = values.clone().sum::<f64>() / n as f64;
        let ss = values.map(|x| (x - m) * (x - m)).sum::<f64>();
        let s = (ss / (n as f64 - 1.0)).sqrt();
        // A constant column carries no information and is dropped, as it is
        // aliased with the intercept.
        if s > 0.0 && s.is_finite() {
            columns.push(column);
            mean.push(m);
            sd.push(s);
        }
    }
    let k = columns.len();
    if n <= k {
        return Err(EstimationError::CbpsNotFullRank);
    }
    let mut z = Mat::<f64>::zeros(n, k);
    for row in 0..n {
        z[(row, 0)] = 1.0;
        for (j, &column) in columns.iter().enumerate().skip(1) {
            z[(row, j)] = (matrix.values[row * matrix.n_cols + column] - mean[j - 1]) / sd[j - 1];
        }
    }
    let ones = vec![1.0; n];
    let gram = weighted_gram(z.as_ref(), &ones).map_err(|_| EstimationError::CbpsNotFullRank)?;
    let flat: Vec<f64> = (0..k)
        .flat_map(|i| (0..k).map(move |j| (i, j)))
        .map(|(i, j)| gram[(i, j)])
        .collect();
    let (values, vectors) = symmetric_eigen_jacobi(&flat, k);
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
    let d: Vec<f64> = order.iter().map(|&i| values[i].max(0.0).sqrt()).collect();
    if !(d[0].is_finite() && d[k - 1] > RANK_TOLERANCE * d[0]) {
        return Err(EstimationError::CbpsNotFullRank);
    }
    let mut v = vec![0.0; k * k];
    for (new, &old) in order.iter().enumerate() {
        for g in 0..k {
            v[g * k + new] = vectors[g * k + old];
        }
    }
    let mut u = Mat::<f64>::zeros(n, k);
    for row in 0..n {
        for new in 0..k {
            let mut sum = 0.0;
            for g in 0..k {
                sum += z[(row, g)] * v[g * k + new];
            }
            u[(row, new)] = sum / d[new];
        }
    }
    Ok(Orthogonal {
        u,
        d,
        v,
        columns,
        mean,
        sd,
    })
}

impl Orthogonal {
    /// The coefficients on the scale of the covariates for the coefficients
    /// `g` on the orthogonal columns, as `None` for a dropped constant column.
    fn coefficients(&self, g: &[f64], n_cols: usize) -> Vec<Option<f64>> {
        let k = g.len();
        let mut standardised = vec![0.0; k];
        for (row, entry) in standardised.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (col, (d, g)) in self.d.iter().zip(g).enumerate() {
                sum += self.v[row * k + col] / d * g;
            }
            *entry = sum;
        }
        let mut out = vec![None; n_cols];
        let mut intercept = standardised[0];
        for j in 1..k {
            let b = standardised[j] / self.sd[j - 1];
            intercept -= self.mean[j - 1] * b;
            out[self.columns[j]] = Some(b);
        }
        out[self.columns[0]] = Some(intercept);
        out
    }
}

/// The fixed pieces of a CBPS problem in the orthogonal coordinates.
struct Problem<'a> {
    u: MatRef<'a, f64>,
    treated: Vec<bool>,
    /// Sampling weights with mean 1.
    s: Vec<f64>,
    n: f64,
    n_t: f64,
    estimand: CbpsEstimand,
    k: usize,
}

type Dense = Vec<f64>;

impl Problem<'_> {
    fn probabilities(&self, g: &[f64]) -> Vec<f64> {
        let mut theta = vec![0.0; self.treated.len()];
        matvec_into(
            &mut theta,
            self.u,
            MatRef::from_column_major_slice(g, self.k, 1),
        );
        theta
            .into_iter()
            .map(|t| epistat_glm::expit(t).clamp(PROBABILITY_BOUND, 1.0 - PROBABILITY_BOUND))
            .collect()
    }

    /// The balance weight of each unit.
    fn balance_weights(&self, p: &[f64]) -> Vec<f64> {
        (0..p.len())
            .map(|i| match self.estimand {
                CbpsEstimand::Att => {
                    self.n / self.n_t * (f64::from(u8::from(self.treated[i])) - p[i]) / (1.0 - p[i])
                }
                CbpsEstimand::Ate => 1.0 / (p[i] - 1.0 + f64::from(u8::from(self.treated[i]))),
            })
            .collect()
    }

    /// `U' (s v) / n`.
    fn project(&self, v: impl Fn(usize) -> f64) -> Vec<f64> {
        let values: Vec<f64> = (0..self.s.len()).map(|i| self.s[i] * v(i)).collect();
        let column = xtv(self.u, &values);
        (0..self.k).map(|j| column[(j, 0)] / self.n).collect()
    }

    /// `(1/n) U' diag(s q) U` for `q >= 0`, row-major.
    fn gram(&self, q: impl Fn(usize) -> f64) -> Dense {
        let weights: Vec<f64> = (0..self.s.len()).map(|i| self.s[i] * q(i)).collect();
        let gram = weighted_gram(self.u, &weights).expect("weights are finite and non-negative");
        let k = self.k;
        (0..k * k)
            .map(|idx| gram[(idx / k, idx % k)] / self.n)
            .collect()
    }

    fn is_treated(&self, i: usize) -> f64 {
        f64::from(u8::from(self.treated[i]))
    }

    /// The moments `gbar`, `2k` values.
    fn moments(&self, p: &[f64]) -> Vec<f64> {
        let w = self.balance_weights(p);
        let mut out = self.project(|i| self.is_treated(i) - p[i]);
        out.extend(self.project(|i| w[i]));
        out
    }

    /// The model covariance `V` of the moments, `2k x 2k` row-major.
    fn covariance(&self, p: &[f64]) -> Dense {
        let k = self.k;
        let a = self.gram(|i| p[i] * (1.0 - p[i]));
        let (b, c) = match self.estimand {
            CbpsEstimand::Att => {
                let r = self.n / self.n_t;
                let b = self.gram(|i| p[i]);
                let c = self.gram(|i| p[i] / (1.0 - p[i]));
                (
                    b.iter().map(|x| x * r).collect::<Vec<_>>(),
                    c.iter().map(|x| x * r * r).collect::<Vec<_>>(),
                )
            }
            CbpsEstimand::Ate => (
                self.gram(|_| 1.0),
                self.gram(|i| 1.0 / (p[i] * (1.0 - p[i]))),
            ),
        };
        let mut v = vec![0.0; 4 * k * k];
        for i in 0..k {
            for j in 0..k {
                v[i * 2 * k + j] = a[i * k + j];
                v[i * 2 * k + k + j] = b[i * k + j];
                v[(k + i) * 2 * k + j] = b[i * k + j];
                v[(k + i) * 2 * k + k + j] = c[i * k + j];
            }
        }
        v
    }

    /// The generalised inverse of the covariance, with the rule of
    /// `MASS::ginv`.
    fn weight_matrix(&self, p: &[f64]) -> Result<Dense, EstimationError> {
        pseudo_inverse(&self.covariance(p), f64::EPSILON.sqrt()).map(|(inverse, _)| inverse)
    }

    /// The derivative of the balance block of `gbar`, `k x k`, symmetric.
    fn balance_derivative(&self, p: &[f64]) -> Dense {
        match self.estimand {
            CbpsEstimand::Att => {
                let r = self.n / self.n_t;
                let g = self.gram(|i| {
                    if self.treated[i] {
                        0.0
                    } else {
                        p[i] / (1.0 - p[i])
                    }
                });
                g.iter().map(|x| -r * x).collect()
            }
            CbpsEstimand::Ate => {
                let t = |i: usize| self.is_treated(i);
                let g = self.gram(|i| (t(i) - p[i]).powi(2) / (p[i] * (1.0 - p[i])));
                g.iter().map(|x| -x).collect()
            }
        }
    }

    /// The size of the rounding error of `J = gbar' W gbar` at `g`.
    ///
    /// Each moment carries a relative rounding error of about `eps`, so
    /// `gbar` has an error of about `eps |gbar|` and `J` an error of at most
    /// `2 ||W||_F eps |gbar|^2`, with `||W||_F >= ||W||_2`. It is large when
    /// the covariance of the moments is nearly singular and `W` is large.
    fn j_rounding(&self, g: &[f64], w: &[f64]) -> f64 {
        let gbar = self.moments(&self.probabilities(g));
        let w_norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        let m_squared = gbar.iter().map(|x| x * x).sum::<f64>();
        2.0 * f64::EPSILON * w_norm * m_squared
    }

    /// `J = gbar' W gbar` and its gradient `2 D' W gbar`.
    fn j_with_gradient(&self, g: &[f64], w: &[f64]) -> (f64, Vec<f64>) {
        let k = self.k;
        let p = self.probabilities(g);
        let gbar = self.moments(&p);
        let wg: Vec<f64> = (0..2 * k)
            .map(|r| (0..2 * k).map(|c| w[r * 2 * k + c] * gbar[c]).sum())
            .collect();
        let value = gbar.iter().zip(&wg).map(|(a, b)| a * b).sum();
        let score = self.gram(|i| p[i] * (1.0 - p[i]));
        let balance = self.balance_derivative(&p);
        let gradient = (0..k)
            .map(|j| {
                2.0 * (0..k)
                    .map(|a| -score[j * k + a] * wg[a] + balance[j * k + a] * wg[k + a])
                    .sum::<f64>()
            })
            .collect();
        (value, gradient)
    }

    fn j_value(&self, g: &[f64], w: &[f64]) -> f64 {
        let p = self.probabilities(g);
        let gbar = self.moments(&p);
        let k2 = 2 * self.k;
        (0..k2)
            .map(|r| gbar[r] * (0..k2).map(|c| w[r * k2 + c] * gbar[c]).sum::<f64>())
            .sum()
    }

    /// The exact estimator's loss `m' M m` and its gradient `2 D M m`.
    fn balance_loss_with_gradient(&self, g: &[f64], m_inv: &[f64]) -> (f64, Vec<f64>) {
        let k = self.k;
        let p = self.probabilities(g);
        let w = self.balance_weights(&p);
        let m = self.project(|i| w[i]);
        let mm: Vec<f64> = (0..k)
            .map(|r| (0..k).map(|c| m_inv[r * k + c] * m[c]).sum())
            .collect();
        let value = m.iter().zip(&mm).map(|(a, b)| a * b).sum();
        let d = self.balance_derivative(&p);
        let gradient = (0..k)
            .map(|j| 2.0 * (0..k).map(|a| d[j * k + a] * mm[a]).sum::<f64>())
            .collect();
        (value, gradient)
    }
}

/// Fits the covariate balancing propensity score to the design `matrix` and
/// the treatment `labels` (1 for the anchors), with prior weights `weights`
/// (1 when `None`).
///
/// # Errors
///
/// [`EstimationError::CbpsNeedsIntercept`] when the design has no column named
/// `intercept`, [`EstimationError::CbpsNotFullRank`] when it is not of full
/// rank, [`EstimationError::InvalidCbpsConfig`] for an unusable configuration
/// or a group with no weight, [`EstimationError::GlmFitFailed`] when the
/// logistic starting fit fails, and [`EstimationError::CbpsFitFailed`] when the
/// objective is not finite at the starting value or a search direction is not
/// a descent direction. A fit that does not reach the gradient tolerance is
/// returned with `converged` false.
pub(super) fn fit_cbps(
    matrix: &CovariateMatrix,
    labels: &[f64],
    weights: Option<&[f64]>,
    config: &CbpsConfig,
) -> Result<LogisticFit, EstimationError> {
    let n = matrix.n_rows;
    if labels.len() != n
        || weights.is_some_and(|w| w.len() != n)
        || config.max_iter == 0
        || !(config.gradient_tolerance.is_finite() && config.gradient_tolerance > 0.0)
    {
        return Err(EstimationError::InvalidCbpsConfig);
    }
    let orthogonal = orthogonalise(matrix)?;
    let k = orthogonal.d.len();
    let mut s: Vec<f64> = weights.map_or_else(|| vec![1.0; n], <[f64]>::to_vec);
    let total: f64 = s.iter().sum();
    if !(total.is_finite() && total > 0.0) {
        return Err(EstimationError::InvalidCbpsConfig);
    }
    for value in &mut s {
        *value *= n as f64 / total;
    }
    let treated: Vec<bool> = labels.iter().map(|&label| label > 0.5).collect();
    let n_t: f64 = s
        .iter()
        .zip(&treated)
        .filter(|(_, t)| **t)
        .map(|(w, _)| w)
        .sum();
    let n_total: f64 = s.iter().sum();
    if n_t <= 0.0 || n_t >= n_total {
        return Err(EstimationError::InvalidCbpsConfig);
    }
    let problem = Problem {
        u: orthogonal.u.as_ref(),
        treated,
        s,
        n: n_total,
        n_t,
        estimand: config.estimand,
        k,
    };

    let mut logistic = Glm::new(Family::Binomial, Link::Logit);
    logistic.max_iterations = 25;
    let start = logistic
        .fit_slices(problem.u, labels, Some(&problem.s), None, None)
        .map_err(EstimationError::GlmFitFailed)?;
    let g0: Vec<f64> = start
        .coefficients
        .iter()
        .map(|c| c.unwrap_or(0.0))
        .collect();

    // The scale of the logistic fit that minimises J, with the weight matrix
    // recomputed at each trial.
    let scaled = |a: f64| -> Vec<f64> { g0.iter().map(|g| g * a).collect() };
    let objective = |a: f64| -> Result<f64, EstimationError> {
        let g = scaled(a);
        let p = problem.probabilities(&g);
        let w = problem.weight_matrix(&p)?;
        Ok(problem.j_value(&g, &w))
    };
    let mut failure = None;
    let scale = epistat_optim::brent::minimize(
        |a| match objective(a) {
            Ok(value) => value,
            Err(error) => {
                failure.get_or_insert(error);
                f64::INFINITY
            }
        },
        SCALE_BRACKET.0,
        SCALE_BRACKET.1,
        f64::EPSILON.powf(0.25),
    )
    .expect("the scale bracket and the tolerance are finite and positive")
    .x;
    if let Some(error) = failure {
        return Err(error);
    }
    let g_init = scaled(scale);
    let w_fixed = problem.weight_matrix(&problem.probabilities(&g_init))?;

    // A decrease of the loss below its rounding error cannot be told from
    // none, and the rounding error is relative to the size of the loss at the
    // start of the search, which for the exact estimator falls to zero at the
    // solution.
    let bfgs_with = |cost_tolerance: f64| Bfgs {
        max_iterations: config.max_iter,
        gradient_tolerance: config.gradient_tolerance,
        cost_tolerance,
        ..Bfgs::default()
    };

    // `gram` carries the factor 1/n, so the generalised inverse of
    // `U' S U` is that of `gram` divided by `n`.
    let m_inv: Vec<f64> = pseudo_inverse(&problem.gram(|_| 1.0), f64::EPSILON.sqrt())?
        .0
        .iter()
        .map(|x| x / problem.n)
        .collect();
    let balance = bfgs_with(
        ROUNDING_FACTOR * f64::EPSILON * problem.balance_loss_with_gradient(&g_init, &m_inv).0,
    )
    .minimize(
        &g_init,
        |g| problem.balance_loss_with_gradient(g, &m_inv),
        None,
    )
    .map_err(EstimationError::CbpsFitFailed)?;

    let chosen = match config.method {
        CbpsMethod::Exact => balance,
        CbpsMethod::Over => {
            let from_logistic = bfgs_with(ROUNDING_FACTOR * problem.j_rounding(&g_init, &w_fixed))
                .minimize(&g_init, |g| problem.j_with_gradient(g, &w_fixed), None)
                .map_err(EstimationError::CbpsFitFailed)?;
            let from_balance =
                bfgs_with(ROUNDING_FACTOR * problem.j_rounding(&balance.x, &w_fixed))
                    .minimize(&balance.x, |g| problem.j_with_gradient(g, &w_fixed), None)
                    .map_err(EstimationError::CbpsFitFailed)?;
            if from_logistic.value < from_balance.value {
                from_logistic
            } else {
                from_balance
            }
        }
    };

    let g = chosen.x;
    let probabilities = problem.probabilities(&g);
    let mut linear_predictor = vec![0.0; n];
    matvec_into(
        &mut linear_predictor,
        problem.u,
        MatRef::from_column_major_slice(&g, k, 1),
    );
    let balance = problem.balance_weights(&probabilities);
    let at_bound = probabilities
        .iter()
        .any(|&p| p <= PROBABILITY_BOUND || p >= 1.0 - PROBABILITY_BOUND);
    let coefficients = orthogonal.coefficients(&g, matrix.n_cols);
    let aliased_columns = coefficients
        .iter()
        .zip(&matrix.column_names)
        .filter(|(c, _)| c.is_none())
        .map(|(_, name)| name.clone())
        .collect();
    Ok(LogisticFit {
        coefficients,
        linear_predictor,
        probabilities,
        converged: chosen.converged,
        iterations: chosen.iterations,
        aliased_columns,
        lambda: None,
        cross_validation: None,
        cbps: Some(CbpsFit {
            j_statistic: problem.j_value(&g, &w_fixed),
            logistic_j_statistic: problem.j_value(&g0, &w_fixed),
            start_scale: scale,
            balance_residual_norm: problem
                .project(|i| balance[i])
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt(),
            gradient_norm: chosen.gradient.iter().map(|x| x * x).sum::<f64>().sqrt(),
        }),
        at_boundary: at_bound,
    })
}
