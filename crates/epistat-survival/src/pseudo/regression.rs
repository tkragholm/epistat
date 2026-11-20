//! Regression on pseudo-observations: generalised estimating equations with
//! independence working correlation and a clustered sandwich variance.
//!
//! # Algorithm
//!
//! The pseudo-values `y_ij` of subject `i` at time `j` are modelled by
//! `mu_ij = g^-1(x_ij' beta + o_ij)`, with the link `g` identity, log or
//! complementary log-log. With independence between and within the
//! pseudo-values of a subject and a constant variance, the estimating
//! equation of `beta` is
//!
//! ```text
//! sum over i, j of  w_ij  mu'_ij  x_ij  (y_ij - mu_ij) = 0,
//! ```
//!
//! with `mu' = d mu / d eta` and `w` a prior weight \[1\]\[2\]. This is the
//! equation `geese` solves with `corstr = "independence"`, and the one of a
//! generalised linear model with constant variance, which is how `eventglm`
//! starts. Under the identity link it is least squares and one step solves it.
//! The other links are solved by Fisher scoring, which is Gauss-Newton here,
//! started from the mean of the pseudo-values as `glm.fit` starts and with the
//! step halved while the weighted sum of squares increases by more than its
//! rounding error. A step smaller than
//! `tolerance` times the larger of 1 and the largest coefficient ends the
//! iteration, after it is applied. Away from the identity link the iteration
//! converges linearly, at a rate that grows with the size of the residuals, so
//! the coefficients are within about the tolerance, times the rate over one less
//! the rate, of the solution of the equation.
//!
//! The covariance is the sandwich of Liang and Zeger \[3\],
//!
//! ```text
//! V = A^-1 M A^-1,   A = sum w mu'^2 x x',   M = sum over subjects of s_i s_i',
//! s_i = sum over j of w_ij mu'_ij x_ij (y_ij - mu_ij),
//! ```
//!
//! with the subject as the cluster, so the several times of one subject are
//! allowed to be correlated. No small-sample correction is made, as
//! `geese` does none unless `jack` is asked for. The variance of the constant
//! cancels, so it is not estimated. `M` is formed by
//! [`epistat_inference::sandwich::meat`].
//!
//! The pseudo-values are estimated, and a variance that treats them as data is
//! the one `geese` gives and `eventglm` gives for pseudo-values made by
//! the jackknife. It is consistent when the pseudo-values come from
//! the jackknife \[1\].
//!
//! # Differences from R
//!
//! - A design that is rank deficient is an error. `glm.fit` reports `NA` for the
//!   aliased coefficients, and `geese` fails.
//! - A mean of the pseudo-values outside the range of the link, which is
//!   not positive for the log link and not in `(0, 1)` for complementary
//!   log-log, has no start and is an error.
//! - `eventglm` reports for one time the covariance of `sandwich::vcovHC` at
//!   its default type, HC3, which divides each score by one less its leverage. The
//!   covariance here is HC0, which is what `geese` gives and `vcovHC(type =
//!   "HC0")` gives for a `glm`.
//! - Fisher scoring halves a step that increases the sum of squares. `geese` takes
//!   the step.
//!
//! # References
//!
//! 1. Andersen P. K., Perme M. P. (2010) Pseudo-observations in survival
//!    analysis, Stat Methods Med Res 19: 71-99.
//! 2. Overgaard M., Parner E. T., Pedersen J. (2017) Asymptotic theory of
//!    generalized estimating equations based on jack-knife pseudo-observations,
//!    Ann Stat 45: 1988-2015.
//! 3. Liang K.-Y., Zeger S. L. (1986) Longitudinal data analysis using
//!    generalized linear models, Biometrika 73: 13-22.

use epistat_inference::Clusters;
use epistat_inference::sandwich::meat;
use faer::{Mat, MatRef};

use crate::error::{Error, Result};

/// The link between the mean of the pseudo-values and the linear predictor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Link {
    /// `mu = eta`, a difference of the target.
    Identity,
    /// `mu = exp(eta)`, a ratio.
    Log,
    /// `mu = 1 - exp(-exp(eta))`, a log hazard ratio when the target is a
    /// survival probability.
    Cloglog,
}

impl Link {
    fn link(self, mu: f64) -> f64 {
        match self {
            Self::Identity => mu,
            Self::Log => mu.ln(),
            Self::Cloglog => (-(-mu).ln_1p()).ln(),
        }
    }

    fn inverse(self, eta: f64) -> f64 {
        match self {
            Self::Identity => eta,
            Self::Log => eta.exp(),
            Self::Cloglog => -(-eta.exp()).exp_m1(),
        }
    }

    fn derivative(self, eta: f64) -> f64 {
        match self {
            Self::Identity => 1.0,
            Self::Log => eta.exp(),
            Self::Cloglog => (eta - eta.exp()).exp(),
        }
    }

    /// Whether `mu` is a mean the link can start from.
    fn admits(self, mu: f64) -> bool {
        match self {
            Self::Identity => mu.is_finite(),
            Self::Log => mu > 0.0 && mu.is_finite(),
            Self::Cloglog => mu > 0.0 && mu < 1.0,
        }
    }
}

/// What [`fit`] does besides the data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeeOptions {
    /// The link. It has no default.
    pub link: Link,
    /// A step of Fisher scoring smaller than this, relative to the larger of 1
    /// and the largest coefficient, ends the iteration.
    pub tolerance: f64,
    /// The most steps taken.
    pub max_iterations: usize,
}

impl GeeOptions {
    /// Options for a link with the tolerance of `glm.control`, `1e-8`, and its
    /// 25 iterations.
    #[must_use]
    pub const fn new(link: Link) -> Self {
        Self {
            link,
            tolerance: 1e-8,
            max_iterations: 25,
        }
    }
}

/// A fitted regression.
#[derive(Debug, Clone, PartialEq)]
pub struct GeeFit {
    /// The coefficients, one per column of the design.
    pub coefficients: Vec<f64>,
    /// The sandwich covariance, row-major, `p` by `p`.
    pub covariance: Vec<f64>,
    /// The fitted means.
    pub fitted: Vec<f64>,
    /// Whether the iteration stopped on its tolerance.
    pub converged: bool,
    /// The number of steps taken.
    pub iterations: usize,
    /// The number of observations, which are the rows of the design.
    pub n_obs: usize,
    /// The number of clusters.
    pub n_clusters: usize,
    /// The link.
    pub link: Link,
    /// The information matrix `A`, row-major, `p` by `p`.
    pub information: Vec<f64>,
    /// The condition number of the information matrix `A` in the 1-norm.
    /// Coefficients lose about this many times the rounding unit to the
    /// solve.
    pub information_condition: f64,
}

impl GeeFit {
    /// The standard errors, the square roots of the diagonal of the
    /// covariance.
    #[must_use]
    pub fn standard_errors(&self) -> Vec<f64> {
        let p = self.coefficients.len();
        (0..p)
            .map(|i| self.covariance[i * p + i].max(0.0).sqrt())
            .collect()
    }
}

/// A square matrix factorised by Gaussian elimination with partial pivoting.
struct Lu {
    a: Vec<f64>,
    pivot: Vec<usize>,
    n: usize,
}

impl Lu {
    fn new(matrix: &[f64], n: usize) -> Result<Self> {
        let mut a = matrix.to_vec();
        let scale = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let mut pivot = vec![0; n];
        #[expect(
            clippy::cast_precision_loss,
            reason = "a design has far fewer columns than 2^53"
        )]
        let floor = f64::EPSILON * n as f64 * scale;
        for k in 0..n {
            let mut best = k;
            for i in k + 1..n {
                if a[i * n + k].abs() > a[best * n + k].abs() {
                    best = i;
                }
            }
            let magnitude = a[best * n + k].abs();
            if magnitude.is_nan() || magnitude <= floor {
                return Err(Error::input(
                    "the design is rank deficient, a column is a combination of the others",
                ));
            }
            pivot[k] = best;
            if best != k {
                for j in 0..n {
                    a.swap(k * n + j, best * n + j);
                }
            }
            for i in k + 1..n {
                let f = a[i * n + k] / a[k * n + k];
                a[i * n + k] = f;
                for j in k + 1..n {
                    a[i * n + j] -= f * a[k * n + j];
                }
            }
        }
        Ok(Self { a, pivot, n })
    }

    fn solve(&self, b: &mut [f64]) {
        let n = self.n;
        for k in 0..n {
            b.swap(k, self.pivot[k]);
            for i in k + 1..n {
                b[i] -= self.a[i * n + k] * b[k];
            }
        }
        for k in (0..n).rev() {
            for j in k + 1..n {
                b[k] -= self.a[k * n + j] * b[j];
            }
            b[k] /= self.a[k * n + k];
        }
    }
}

/// `sum w mu'^2 x x'`, summed in row order.
fn information(x: MatRef<'_, f64>, w: &[f64]) -> Vec<f64> {
    let p = x.ncols();
    let mut a = vec![0.0; p * p];
    for i in 0..x.nrows() {
        for j in 0..p {
            let wxj = w[i] * x[(i, j)];
            for l in j..p {
                a[j * p + l] += wxj * x[(i, l)];
            }
        }
    }
    for j in 0..p {
        for l in 0..j {
            a[j * p + l] = a[l * p + j];
        }
    }
    a
}

/// Fits the regression of `y` on the rows of `x`, the rows of one cluster
/// being the observations of one subject.
///
/// `weights` are prior weights, 1 when absent, `offset` is added to the linear
/// predictor, and `start` replaces the start from the mean. The columns of
/// `x` include the intercept if there is one. See the
/// [module documentation](self) for the equations and the covariance.
///
/// # Errors
///
/// [`Error::Input`] when the inputs differ in length, `y`, `x`, the weights or
/// the offset are not finite, a weight is negative, there are fewer rows than
/// columns, the tolerance is not positive and finite, the mean of `y` is
/// outside the range of the link and no start is given, the design is rank
/// deficient, or Fisher scoring cannot reduce the sum of squares.
pub fn fit(
    x: MatRef<'_, f64>,
    y: &[f64],
    cluster: &[u32],
    weights: Option<&[f64]>,
    offset: Option<&[f64]>,
    start: Option<&[f64]>,
    opts: &GeeOptions,
) -> Result<GeeFit> {
    let (n, p) = (x.nrows(), x.ncols());
    for (name, length) in [("y", y.len()), ("cluster", cluster.len())]
        .into_iter()
        .chain(weights.map(|w| ("weights", w.len())))
        .chain(offset.map(|o| ("offset", o.len())))
    {
        if length != n {
            return Err(Error::input(format!(
                "{name} has {length} entries and the design has {n} rows"
            )));
        }
    }
    if let Some(start) = start {
        if start.len() != p {
            return Err(Error::input(format!(
                "start has {} entries and the design has {p} columns",
                start.len()
            )));
        }
    }
    if p == 0 || n < p {
        return Err(Error::input(format!(
            "a design of {n} rows and {p} columns cannot be fitted"
        )));
    }
    if !(opts.tolerance.is_finite() && opts.tolerance > 0.0) {
        return Err(Error::input("the tolerance must be positive and finite"));
    }
    let finite = y.iter().all(|v| v.is_finite())
        && (0..n).all(|i| (0..p).all(|j| x[(i, j)].is_finite()))
        && weights.is_none_or(|w| w.iter().all(|v| v.is_finite()))
        && offset.is_none_or(|o| o.iter().all(|v| v.is_finite()));
    if !finite {
        return Err(Error::input(
            "y, the design, the weights and the offset must be finite",
        ));
    }
    if weights.is_some_and(|w| w.iter().any(|&v| v < 0.0)) {
        return Err(Error::input("a weight is negative"));
    }
    let w: Vec<f64> = weights.map_or_else(|| vec![1.0; n], <[f64]>::to_vec);
    let o: Vec<f64> = offset.map_or_else(|| vec![0.0; n], <[f64]>::to_vec);
    let link = opts.link;
    let total_weight: f64 = w.iter().sum();
    if total_weight <= 0.0 {
        return Err(Error::input("the weights sum to zero"));
    }

    let eta_of = |beta: &[f64]| -> Vec<f64> {
        (0..n)
            .map(|i| (0..p).map(|j| x[(i, j)] * beta[j]).sum::<f64>() + o[i])
            .collect()
    };
    let sum_of_squares = |eta: &[f64]| -> f64 {
        (0..n)
            .map(|i| {
                let r = y[i] - link.inverse(eta[i]);
                w[i] * r * r
            })
            .sum()
    };
    // A weighted least squares step: the solution of
    // sum w d^2 x x' b = sum w d x z for the working response z.
    let solve_step = |scale: &[f64], z: &[f64]| -> Result<Vec<f64>> {
        let weight: Vec<f64> = (0..n).map(|i| w[i] * scale[i] * scale[i]).collect();
        let a = information(x, &weight);
        let mut rhs = vec![0.0; p];
        for i in 0..n {
            for j in 0..p {
                rhs[j] += weight[i] * x[(i, j)] * z[i];
            }
        }
        Lu::new(&a, p)?.solve(&mut rhs);
        Ok(rhs)
    };

    let mut beta = if let Some(start) = start {
        start.to_vec()
    } else {
        let mean = y.iter().zip(&w).map(|(v, wi)| v * wi).sum::<f64>() / total_weight;
        if !link.admits(mean) {
            return Err(Error::input(format!(
                "the mean {mean} of the pseudo-values is outside the range of the link, \
                 so there is no start; supply one"
            )));
        }
        let eta0 = link.link(mean);
        let derivative = link.derivative(eta0);
        let scale = vec![derivative; n];
        let z: Vec<f64> = (0..n)
            .map(|i| eta0 - o[i] + (y[i] - mean) / derivative)
            .collect();
        solve_step(&scale, &z)?
    };
    let mut eta = eta_of(&beta);
    let mut current = sum_of_squares(&eta);
    let mut converged = false;
    let mut iterations = 0;
    for _ in 0..opts.max_iterations {
        iterations += 1;
        let derivative: Vec<f64> = eta.iter().map(|&e| link.derivative(e)).collect();
        let z: Vec<f64> = (0..n)
            .map(|i| (y[i] - link.inverse(eta[i])) / derivative[i])
            .collect();
        // The Gauss-Newton step solves the same equations for the change.
        let step = solve_step(&derivative, &z)?;
        let largest = beta.iter().fold(1.0f64, |m, b| m.max(b.abs()));
        let full_step_is_small = step.iter().all(|v| v.abs() <= opts.tolerance * largest);
        // The sum of squares is a sum of n terms, so a rise below its rounding
        // error is no rise.
        #[expect(clippy::cast_precision_loss, reason = "a row count is far below 2^53")]
        let rounding = 2.0 * (n as f64 + 2.0) * f64::EPSILON;
        let mut factor = 1.0f64;
        loop {
            let candidate: Vec<f64> = beta
                .iter()
                .zip(&step)
                .map(|(b, s)| b + factor * s)
                .collect();
            let candidate_eta = eta_of(&candidate);
            let ss = sum_of_squares(&candidate_eta);
            if full_step_is_small || ss <= current + rounding * current {
                converged = full_step_is_small;
                beta = candidate;
                eta = candidate_eta;
                current = ss;
                break;
            }
            factor *= 0.5;
            if factor < f64::EPSILON {
                return Err(Error::input(
                    "Fisher scoring cannot reduce the sum of squares from the current coefficients",
                ));
            }
        }
        if converged {
            break;
        }
    }

    let derivative: Vec<f64> = eta.iter().map(|&e| link.derivative(e)).collect();
    let fitted: Vec<f64> = eta.iter().map(|&e| link.inverse(e)).collect();
    let weight: Vec<f64> = (0..n)
        .map(|i| w[i] * derivative[i] * derivative[i])
        .collect();
    let a = information(x, &weight);
    let scored = Mat::from_fn(n, p, |i, j| w[i] * derivative[i] * x[(i, j)]);
    let clusters = Clusters::from_sorted_labels(cluster);
    let m = meat(scored.as_ref(), |i| y[i] - fitted[i], Some(&clusters))
        .map_err(|e| Error::input(format!("the sandwich could not be formed: {e}")))?;
    let lu = Lu::new(&a, p)?;
    let norm_a = (0..p)
        .map(|j| (0..p).map(|i| a[i * p + j].abs()).sum::<f64>())
        .fold(0.0f64, f64::max);
    let norm_inverse = (0..p)
        .map(|j| {
            let mut column = vec![0.0; p];
            column[j] = 1.0;
            lu.solve(&mut column);
            column.iter().map(|v| v.abs()).sum::<f64>()
        })
        .fold(0.0f64, f64::max);
    // V = A^-1 M A^-1: solve A L = M, then A R = L', and V = R'.
    let mut left = vec![0.0; p * p];
    for j in 0..p {
        let mut column: Vec<f64> = (0..p).map(|i| m[(i, j)]).collect();
        lu.solve(&mut column);
        for i in 0..p {
            left[i * p + j] = column[i];
        }
    }
    let mut covariance = vec![0.0; p * p];
    for i in 0..p {
        let mut row: Vec<f64> = (0..p).map(|j| left[i * p + j]).collect();
        lu.solve(&mut row);
        for j in 0..p {
            covariance[i * p + j] = row[j];
        }
    }
    for i in 0..p {
        for j in 0..i {
            let mid = covariance[i * p + j].midpoint(covariance[j * p + i]);
            covariance[i * p + j] = mid;
            covariance[j * p + i] = mid;
        }
    }
    Ok(GeeFit {
        coefficients: beta,
        covariance,
        fitted,
        converged,
        iterations,
        n_obs: n,
        n_clusters: clusters.count(),
        link,
        information_condition: norm_a * norm_inverse,
        information: a,
    })
}
