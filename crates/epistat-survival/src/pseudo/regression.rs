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
//! starts. Each iteration is one step of [`epistat_glm::Glm`] with the Gaussian
//! family, whose variance is constant, and the link: `glm.fit`'s iteratively
//! reweighted least squares, which is Fisher scoring. The iteration starts from
//! the mean of the pseudo-values as `glm.fit` starts: the first working response
//! is formed at the mean, and one weighted least squares step gives the starting
//! coefficients. Under the identity link this is least squares and one step
//! solves it. A step smaller than `tolerance` times the larger of 1 and the
//! largest coefficient ends the iteration, after it is applied. Away from the
//! identity link the iteration converges linearly, at a rate that grows with the
//! size of the residuals, so the coefficients are within about the tolerance,
//! times the rate over one less the rate, of the solution of the equation. The
//! stopping rule is on the step and not on the change of the deviance, as in
//! `glm.control`, because the deviance changes with the square of the step near
//! the solution and so stops the iteration at a step of about the square root of
//! its tolerance.
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
//! With [`SandwichType::Hc3`] each score is divided by one less the leverage of
//! its observation, `h_i = w_i mu'_i^2 x_i' A^-1 x_i`, so that
//!
//! ```text
//! V = A^-1 ( sum over i of s_i s_i' / (1 - h_i)^2 ) A^-1,
//! ```
//!
//! which is `sandwich::vcovHC(type = "HC3")` of the `glm` fit \[4\], and what
//! `eventglm` reports for a single time. It is defined for one observation in
//! each cluster, as `vcovHC` is, and is an error otherwise.
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
//!   its default type, HC3, and for several times that of `geese`, which has no
//!   leverage correction. The default here is HC0, which is what `geese` gives
//!   and `vcovHC(type = "HC0")` gives for a `glm`, so that one time and several
//!   are the same estimator. [`SandwichType::Hc3`] is `eventglm`'s for one time.
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
//! 4. MacKinnon J. G., White H. (1985) Some heteroskedasticity-consistent
//!    covariance matrix estimators with improved finite sample properties,
//!    J Econometrics 29: 305-325.

use epistat_core::check;
use epistat_glm::robust::SandwichType;
use epistat_glm::{Family, Glm, Link};
use epistat_inference::Clusters;
use epistat_inference::sandwich::{meat, sandwich};
use epistat_linalg::{Attempt, Mat, MatRef, inverse, weighted_gram};

use crate::check::{refuse_non_finite, refuse_non_finite_matrix};
use crate::covariance;
use crate::error::{Error, Result};

/// Whether the mean `mu` lies in the range of `link`, for the links the
/// regression takes: identity, log, logit and complementary log-log.
fn admits(link: Link, mu: f64) -> Option<bool> {
    match link {
        Link::Identity => Some(mu.is_finite()),
        Link::Log => Some(mu > 0.0 && mu.is_finite()),
        Link::Logit | Link::Cloglog => Some(mu > 0.0 && mu < 1.0),
        _ => None,
    }
}

/// The covariance of a regression on pseudo-values, under its name in 0.3.
#[deprecated(since = "0.4.0", note = "use `epistat_glm::robust::SandwichType`")]
pub type Covariance = SandwichType;

/// What [`fit`] does besides the data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PseudoRegressionOptions {
    /// The link: identity, log, logit or complementary log-log, as
    /// `eventglm`'s `link`. It has no default.
    pub link: Link,
    /// The sandwich: [`SandwichType::Hc0`], Liang and Zeger's with no
    /// small-sample correction, the one of `geese`, or [`SandwichType::Hc3`],
    /// each score divided by one less the leverage of its observation, as
    /// `sandwich::vcovHC(type = "HC3")`, which needs one observation in each
    /// cluster. The default is `Hc0`.
    pub covariance: SandwichType,
    /// A step of Fisher scoring smaller than this, relative to the larger of 1
    /// and the largest coefficient, ends the iteration.
    pub tolerance: f64,
    /// The most steps taken.
    pub max_iterations: usize,
}

impl PseudoRegressionOptions {
    /// Options for a link with the tolerance of `glm.control`, `1e-8`, and its
    /// 25 iterations.
    #[must_use]
    pub const fn new(link: Link) -> Self {
        Self {
            link,
            covariance: SandwichType::Hc0,
            tolerance: 1e-8,
            max_iterations: 25,
        }
    }
}

/// A fitted regression.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct PseudoRegression {
    /// The coefficients, one per column of the design.
    pub coefficients: Vec<f64>,
    /// The sandwich covariance, `p` by `p`.
    pub covariance: epistat_core::Covariance,
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
    /// The leverage of each observation, `w mu'^2 x' A^-1 x`, when the
    /// covariance is [`SandwichType::Hc3`], which divides each score by one less it.
    pub leverage: Option<Vec<f64>>,
}

/// The options of [`fit`], under their name in 0.3.
#[deprecated(since = "0.4.0", note = "renamed `PseudoRegressionOptions`")]
pub type GeeOptions = PseudoRegressionOptions;

/// The result of [`fit`], under its name in 0.3.
#[deprecated(since = "0.4.0", note = "renamed `PseudoRegression`")]
pub type GeeFit = PseudoRegression;

impl PseudoRegression {
    /// The standard errors, the square roots of the diagonal of the
    /// covariance, `None` where a variance is undefined.
    #[must_use]
    pub fn se(&self) -> Vec<Option<f64>> {
        self.covariance
            .variances()
            .into_iter()
            .map(|v| v.map(|v| v.max(0.0).sqrt()))
            .collect()
    }

    /// The standard errors, under their name in 0.3.
    #[deprecated(since = "0.4.0", note = "use `se`")]
    #[must_use]
    pub fn standard_errors(&self) -> Vec<f64> {
        self.se()
            .into_iter()
            .map(|s| s.unwrap_or(f64::NAN))
            .collect()
    }
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
/// An input error when the inputs differ in length, `y`, `x`, the weights or
/// the offset are not finite, a weight is negative, there are fewer rows than
/// columns, the tolerance is not positive and finite, no iteration is allowed, the mean of `y` is
/// outside the range of the link and no start is given, the design is rank
/// deficient, the iteration leaves the range of the link and cannot return, or
/// [`SandwichType::Hc3`] is asked with a cluster of more than one observation or an
/// observation of leverage 1; [`Error::Unsupported`] for a link other than
/// identity, log, logit and complementary log-log, or a sandwich other than
/// `Hc0` and `Hc3`.
pub fn fit(
    x: MatRef<'_, f64>,
    y: &[f64],
    cluster: &[u32],
    weights: Option<&[f64]>,
    offset: Option<&[f64]>,
    start: Option<&[f64]>,
    opts: &PseudoRegressionOptions,
) -> Result<PseudoRegression> {
    let (n, p) = (x.nrows(), x.ncols());
    check::non_empty("x", n)?;
    for (name, length) in [("y", y.len()), ("cluster", cluster.len())]
        .into_iter()
        .chain(weights.map(|w| ("weights", w.len())))
        .chain(offset.map(|o| ("offset", o.len())))
    {
        check::non_empty(name, length)?;
        check::same_length(name, n, length)?;
    }
    if let Some(start) = start {
        check::same_length("start", p, start.len())?;
    }
    if p == 0 || n < p {
        return Err(Error::no_information(format!(
            "a design of {n} rows and {p} columns cannot be fitted"
        )));
    }
    if !(opts.tolerance.is_finite() && opts.tolerance > 0.0) {
        return Err(Error::options("the tolerance must be positive and finite"));
    }
    if opts.max_iterations == 0 {
        return Err(Error::options("at least one iteration must be allowed"));
    }
    refuse_non_finite(y, "y")?;
    refuse_non_finite_matrix(x, "x")?;
    if let Some(w) = weights {
        refuse_non_finite(w, "weights")?;
    }
    if let Some(o) = offset {
        refuse_non_finite(o, "offset")?;
    }
    if let Some(w) = weights {
        check::weights("weights", w)?;
    }
    let w: Vec<f64> = weights.map_or_else(|| vec![1.0; n], <[f64]>::to_vec);
    let o: Vec<f64> = offset.map_or_else(|| vec![0.0; n], <[f64]>::to_vec);
    let link = opts.link;
    let glm_link = link;
    if admits(link, 0.5).is_none() {
        return Err(Error::unsupported(
            "link",
            format!("link: {link:?}; the regression takes identity, log, logit and cloglog"),
        ));
    }
    if !matches!(opts.covariance, SandwichType::Hc0 | SandwichType::Hc3) {
        return Err(Error::unsupported(
            "covariance",
            format!(
                "covariance: {:?}; the regression has Hc0 and Hc3",
                opts.covariance
            ),
        ));
    }
    let total_weight: f64 = w.iter().sum();
    if total_weight <= 0.0 {
        return Err(Error::no_information("the weights sum to zero"));
    }
    let rank_deficient = || {
        Error::no_information(
            "the design is rank deficient, a column is a combination of the others",
        )
    };

    let beta0: Vec<f64> = if let Some(start) = start {
        start.to_vec()
    } else {
        // The working response of the first iteration at the mean of the
        // pseudo-values, and one weighted least squares step on it.
        let mean = y.iter().zip(&w).map(|(v, wi)| v * wi).sum::<f64>() / total_weight;
        if admits(link, mean) != Some(true) {
            return Err(Error::value(format!(
                "the mean {mean} of the pseudo-values is outside the range of the link, \
                 so there is no start; supply one"
            )));
        }
        let eta0 = glm_link.link(mean);
        let derivative = glm_link.derivative(eta0);
        let z: Vec<f64> = (0..n)
            .map(|i| eta0 - o[i] + (y[i] - mean) / derivative)
            .collect();
        let step = Glm::new(Family::Gaussian, epistat_glm::Link::Identity)
            .fit(&crate::logistic::data(x, &z, Some(&w), None, None)?)
            .map_err(Error::model("the starting regression"))?;
        step.coefficients
            .into_iter()
            .collect::<Option<Vec<f64>>>()
            .ok_or_else(rank_deficient)?
    };

    // One step of Fisher scoring is one iteration of the model, which also
    // detects an aliased column and halves a step that leaves the range of
    // the link. The stopping rule on the step is applied here.
    let model = Glm {
        max_iterations: 1,
        ..Glm::new(Family::Gaussian, glm_link)
    };
    let mut beta = beta0;
    let mut solved = None;
    let (mut converged, mut iterations) = (false, 0);
    for _ in 0..opts.max_iterations {
        iterations += 1;
        let step = model
            .fit(&crate::logistic::data(
                x,
                y,
                Some(&w),
                Some(&o),
                Some(&beta),
            )?)
            .map_err(Error::model("the regression"))?;
        let next = step
            .coefficients
            .iter()
            .copied()
            .collect::<Option<Vec<f64>>>()
            .ok_or_else(rank_deficient)?;
        let largest = beta.iter().fold(1.0f64, |m, b| m.max(b.abs()));
        converged = next
            .iter()
            .zip(&beta)
            .all(|(a, b)| (a - b).abs() <= opts.tolerance * largest);
        beta = next;
        solved = Some(step);
        if converged {
            break;
        }
    }
    let solved = solved.ok_or_else(|| Error::internal("no iteration ran"))?;
    let fitted = solved.fitted;
    let derivative: Vec<f64> = solved
        .linear_predictor
        .iter()
        .map(|&e| glm_link.derivative(e))
        .collect();
    let weight: Vec<f64> = (0..n)
        .map(|i| w[i] * derivative[i] * derivative[i])
        .collect();
    let a = weighted_gram(x, &weight)?;
    let scored = Mat::from_fn(n, p, |i, j| w[i] * derivative[i] * x[(i, j)]);
    let clusters = Clusters::from_sorted_labels(cluster)?;
    let a_inverse = inverse(a.as_ref(), &Attempt::SYMMETRIC)?.x;
    // The factor of each score: 1, or 1 / (1 - h) with h the leverage.
    let (factor, leverage): (Vec<f64>, Option<Vec<f64>>) = match opts.covariance {
        SandwichType::Hc3 => {
            if clusters.count() != n {
                return Err(Error::not_applicable(
                    "HC3 is defined for one observation in each cluster",
                ));
            }
            let leverage: Vec<f64> = (0..n)
                .map(|i| {
                    let mut quad = 0.0f64;
                    for r in 0..p {
                        let mut inner = 0.0f64;
                        for c in 0..p {
                            inner += a_inverse[(r, c)] * x[(i, c)];
                        }
                        quad += x[(i, r)] * inner;
                    }
                    weight[i] * quad
                })
                .collect();
            let factor = leverage
                .iter()
                .enumerate()
                .map(|(i, &h)| {
                    if h < 1.0 {
                        Ok(1.0 / (1.0 - h))
                    } else {
                        Err(Error::value(format!("observation {i} has leverage 1")))
                    }
                })
                .collect::<Result<_>>()?;
            (factor, Some(leverage))
        }
        // Hc0, the one other sandwich the options were checked to hold.
        _ => (vec![1.0; n], None),
    };
    let m = meat(
        scored.as_ref(),
        |i| factor[i] * (y[i] - fitted[i]),
        Some(&clusters),
    )?;
    let covariance = sandwich(a.as_ref(), m.as_ref(), &Attempt::SYMMETRIC)?;
    let one_norm = |m: &Mat<f64>| {
        (0..p)
            .map(|j| (0..p).map(|i| m[(i, j)].abs()).sum::<f64>())
            .fold(0.0f64, f64::max)
    };
    let row_major = |m: &Mat<f64>| -> Vec<f64> {
        (0..p)
            .flat_map(|i| (0..p).map(move |j| (i, j)))
            .map(|(i, j)| m[(i, j)])
            .collect()
    };
    Ok(PseudoRegression {
        coefficients: beta,
        covariance: covariance::from_values(p, &row_major(&covariance))?,
        fitted,
        converged,
        iterations,
        n_obs: n,
        n_clusters: clusters.count(),
        link,
        information_condition: one_norm(&a) * one_norm(&a_inverse),
        leverage,
        information: row_major(&a),
    })
}
