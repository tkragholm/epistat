//! The attributable fraction of an exposure from a logistic model, with a
//! standard error from the sandwich of the stacked estimating equations, as the
//! functions `AFglm` of the R package AF.
//!
//! # Estimators
//!
//! The model is a logistic regression fitted to the sample, with the exposure
//! in some columns of the design `x`. `x_unexposed` is the same design with the
//! exposure set to its unexposed value in every row, so that an interaction
//! with a confounder moves with it. With `eta_i` the linear predictor,
//! `delta_i = (x_i - x0_i)' beta` is the part of it due to the exposure and
//! `p0_i = expit(eta_i - delta_i)` the risk of the person unexposed. Rows have
//! the weight `w_i`, 1 when none is given.
//!
//! - [`Design::CrossSectional`], for a cohort or a cross-sectional sample, the
//!   g-formula
//!   `AF = 1 - P0 / P`, with `P = sum w_i y_i / sum w_i` the observed risk and
//!   `P0 = sum w_i p0_i / sum w_i` the risk had nobody been exposed \[1, 2\].
//!   With an intercept in the model `P` equals the mean fitted risk.
//! - [`Design::CaseControl`], for a sample in which the cases are over-sampled
//!   and the odds ratios are valid but the intercept is not, the estimator of
//!   Bruzzi and of Greenland and Drescher \[3, 4\], over the cases only:
//!   `AF = 1 - sum w_i y_i exp(-delta_i) / sum w_i y_i`.
//!
//! The attributable fraction is the proportion of the cases that would not
//! have occurred without the exposure, under the model, which assumes that the
//! confounders in the model are sufficient and that removing the exposure
//! changes nothing else. It is negative for a protective exposure and is not
//! a sum of parts for several exposures \[5, 6\].
//!
//! # Variance
//!
//! The estimates are solutions of estimating equations stacked with the score of
//! the logistic model, `s_i = w_i x_i (y_i - p_i)`. For the cross-sectional
//! design the parameters are `(P, P0, beta)` and the equations
//! `w_i (y_i - P)`, `w_i (p0_i - P0)` and `s_i`; for the case-control design
//! `(AF, beta)` with `w_i y_i (exp(-delta_i) - AF)` and `s_i`. The influence
//! of row `i` on `AF` is, writing `u_i = n V s_i` with `V` the covariance of the
//! fit, `n` the number of rows and `wbar` the mean weight,
//!
//! ```text
//! cross-sectional:  Z_i = [ P0 w_i (y_i - P) / P^2 - (w_i (p0_i - P0) + g' u_i) / P ] / wbar
//!                   g = (1/n) sum w_i p0_i (1 - p0_i) x0_i
//! case-control:     Z_i = [ w_i y_i (exp(-delta_i) - AF) + h' u_i ] / ybar
//!                   h = (1/n) sum w_i y_i exp(-delta_i) (x0_i - x_i),  ybar = (1/n) sum w_i y_i
//! ```
//!
//! and the variance is `sum (Z_i - Zbar)^2 / (n - 1) / n`. With clusters the `Z_i`
//! are summed within each of `C` clusters and the variance is `C / n^2` times
//! the sample variance of the sums, with divisor `C - 1`, as `AFglm` with
//! `clusterid`. The terms `g' u_i` and `h' u_i` carry the uncertainty of the
//! coefficients, which is why a delta method with the covariance of `beta` alone
//! is too small: the means over the sample vary as well. The coefficients are
//! taken from the fit and `V` is its covariance with the dispersion divided
//! out, which is the inverse of the information. The sums are plain loops in
//! row order and do not depend on the number of threads.
//!
//! The limits are `AF +- z se` on the identity scale, and on the scale of
//! `log(1 - AF)`, which keeps them below 1,
//! `1 - (1 - AF) exp(+- z se / (1 - AF))`, the log transformation of
//! Dahlqwist et al. for the complement.
//!
//! # Results that are not numbers
//!
//! `AF` is `None` when no row has an event (and a positive weight) and, for the
//! cross-sectional design, when the observed risk is 0. The standard error is
//! `None` for fewer than two rows or clusters, and the limits on the log scale
//! are `None` when `AF` is 1 or more or `se` is `None`. An aliased column is
//! left out of the sums, as the fit leaves it out of the model.
//!
//! # Agreement with R
//!
//! The tests compare the estimate and the variance with `AFglm` of AF 0.1.5
//! run from its source, for the cross-sectional design with an interaction,
//! clustered, and for the case-control design, and with a sandwich of the
//! stacked equations whose Jacobian is numerical (`numDeriv`) for weighted fits,
//! which `AFglm` does not take.
//!
//! # References
//!
//! 1. Greenland S, Drescher K. Maximum likelihood estimation of the attributable
//!    fraction from logistic models. Biometrics 1993;49(3):865-72.
//! 2. Sjolander A, Vansteelandt S. Doubly robust estimation of attributable
//!    fractions in survival analysis. Stat Methods Med Res 2017;26(2):948-69.
//! 3. Bruzzi P, Green SB, Byar DP, Brinton LA, Schairer C. Estimating the
//!    population attributable risk for multiple risk factors using case-control
//!    data. Am J Epidemiol 1985;122(5):904-14.
//! 4. Dahlqwist E, Zetterqvist J, Pawitan Y, Sjolander A. Model-based estimation
//!    of the attributable fraction for cross-sectional, case-control and cohort
//!    studies using the R package AF. Eur J Epidemiol 2016;31(6):575-82.
//! 5. Rockhill B, Newman B, Weinberg C. Use and misuse of population
//!    attributable fractions. Am J Public Health 1998;88(1):15-9.
//! 6. Rowe AK, Powell KE, Flanders WD. Why population attributable fractions can
//!    sum to more than one. Am J Prev Med 2004;26(3):243-9.

use epistat_linalg::MatRef;
use epistat_stats::dist::normal;

use crate::fit::Fit;

/// The sampling that the model was fitted to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Design {
    /// A cohort or cross-sectional sample: the g-formula over everybody.
    CrossSectional,
    /// A case-control sample: the formula over the cases, which does not use
    /// the intercept.
    CaseControl,
}

/// The scale of the confidence limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntervalScale {
    /// `AF +- z se`.
    Identity,
    /// The limits of `log(1 - AF)`, back-transformed, so they stay below 1.
    LogComplement,
}

/// What [`attributable_logistic`] takes besides the model and the data.
#[derive(Debug, Clone, Copy)]
pub struct AfOptions<'a> {
    /// The sampling of the data.
    pub design: Design,
    /// The confidence level, in `(0, 1)`.
    pub level: f64,
    /// The scale of the limits.
    pub scale: IntervalScale,
    /// Case weights, one per row, not negative. They must be the prior weights
    /// of the fit.
    pub weights: Option<&'a [f64]>,
    /// Cluster codes, one per row, for a cluster-robust variance.
    pub cluster: Option<&'a [u32]>,
}

impl Default for AfOptions<'_> {
    fn default() -> Self {
        Self {
            design: Design::CrossSectional,
            level: 0.95,
            scale: IntervalScale::LogComplement,
            weights: None,
            cluster: None,
        }
    }
}

/// An attributable fraction with its standard error and confidence limits.
#[derive(Debug, Clone, PartialEq)]
pub struct Attributable {
    /// The attributable fraction.
    pub af: Option<f64>,
    /// Its standard error.
    pub se: Option<f64>,
    /// The lower confidence limit.
    pub lower: Option<f64>,
    /// The upper confidence limit.
    pub upper: Option<f64>,
    /// The rows of the data.
    pub n_rows: usize,
    /// The rows with an event and a positive weight.
    pub n_cases: usize,
    /// The clusters, 0 when none were given.
    pub n_clusters: usize,
}

/// Why an attributable fraction could not be formed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AfError {
    /// The input cannot be used: columns of different length, a response that
    /// is not 0 or 1, a weight that is negative or not finite, a level outside
    /// `(0, 1)`, a fit that is not a logistic fit, or a covariance the fit does
    /// not have.
    #[error("{0}")]
    Input(String),
}

fn invalid(message: impl Into<String>) -> AfError {
    AfError::Input(message.into())
}

/// The attributable fraction of the exposure that distinguishes `x` from
/// `x_unexposed`, from the logistic `fit` of `y` on `x`.
///
/// `x` is the design the model was fitted to and `x_unexposed` the same design
/// with the exposure removed, as the module documentation describes. The fit
/// must be binomial with the logit link; the fitted means are checked against
/// the linear predictor.
///
/// # Errors
///
/// [`AfError::Input`] as described there.
pub fn attributable_logistic(
    fit: &Fit,
    x: MatRef<'_, f64>,
    x_unexposed: MatRef<'_, f64>,
    y: &[f64],
    opts: &AfOptions<'_>,
) -> Result<Attributable, AfError> {
    let n = y.len();
    let p = fit.coefficients.len();
    if fit.fitted.len() != n || fit.linear_predictor.len() != n {
        return Err(invalid("the fit and y have different numbers of rows"));
    }
    if x.nrows() != n || x_unexposed.nrows() != n || x.ncols() != p || x_unexposed.ncols() != p {
        return Err(invalid(format!(
            "x and x_unexposed must have {n} rows and {p} columns, the rows and coefficients of the fit"
        )));
    }
    if !(opts.level > 0.0 && opts.level < 1.0) {
        return Err(invalid("level must be in (0, 1)"));
    }
    if let Some(i) = y.iter().position(|&v| !matches!(v, 0.0 | 1.0)) {
        return Err(invalid(format!("y[{i}] is {}, not 0 or 1", y[i])));
    }
    let ones = vec![1.0; if opts.weights.is_some() { 0 } else { n }];
    let w: &[f64] = opts.weights.unwrap_or(&ones);
    if w.len() != n {
        return Err(invalid("weights must have one entry per row"));
    }
    if let Some(i) = w.iter().position(|&v| !(v.is_finite() && v >= 0.0)) {
        return Err(invalid(format!("weight {i} is negative or not finite")));
    }
    let event: Vec<bool> = y.iter().map(|&v| matches!(v, 1.0)).collect();
    if let Some(c) = opts.cluster
        && c.len() != n
    {
        return Err(invalid("cluster must have one entry per row"));
    }
    for i in 0..n {
        let expected = crate::family::expit(fit.linear_predictor[i]);
        if (fit.fitted[i] - expected).abs() > 1e-8 * expected.max(1e-300) + 1e-12 {
            return Err(invalid(
                "the fitted means are not the logistic function of the linear predictor",
            ));
        }
    }
    let kept: Vec<usize> = (0..p).filter(|&j| fit.coefficients[j].is_some()).collect();
    let q = kept.len();
    let dispersion = fit
        .dispersion
        .ok_or_else(|| invalid("the fit has no dispersion, so no covariance"))?;
    let mut v = vec![0.0; q * q];
    for (a, &ja) in kept.iter().enumerate() {
        for (b, &jb) in kept.iter().enumerate() {
            let c =
                fit.covariance[ja * p + jb].ok_or_else(|| invalid("the fit has no covariance"))?;
            v[a * q + b] = c / dispersion;
        }
    }
    let beta: Vec<f64> = kept
        .iter()
        .map(|&j| fit.coefficients[j].unwrap_or(0.0))
        .collect();
    let n_f = n as f64;
    let sum_w: f64 = w.iter().sum();
    let wbar = sum_w / n_f;
    let cases = (0..n).filter(|&i| event[i] && w[i] > 0.0).count();
    let mut result = Attributable {
        af: None,
        se: None,
        lower: None,
        upper: None,
        n_rows: n,
        n_cases: cases,
        n_clusters: 0,
    };
    if n == 0 || sum_w <= 0.0 {
        return Ok(result);
    }
    // delta_i, the part of the linear predictor due to the exposure.
    let delta: Vec<f64> = (0..n)
        .map(|i| {
            kept.iter()
                .zip(&beta)
                .map(|(&j, &b)| (x[(i, j)] - x_unexposed[(i, j)]) * b)
                .sum()
        })
        .collect();
    // u_i = n V s_i and the gradient of the estimating equation of the
    // quantity with respect to beta.
    let case_control = opts.design == Design::CaseControl;
    let p0: Vec<f64> = (0..n)
        .map(|i| crate::family::expit(fit.linear_predictor[i] - delta[i]))
        .collect();
    let (af, grad, scale_a, scale_b);
    let mut ybar = 0.0;
    let mut p_obs = 0.0;
    let mut p_unexposed = 0.0;
    if case_control {
        let (mut num, mut den) = (0.0, 0.0);
        for i in 0..n {
            if event[i] {
                num += w[i] * (-delta[i]).exp();
                den += w[i];
            }
        }
        if den <= 0.0 {
            return Ok(result);
        }
        ybar = den / n_f;
        af = 1.0 - num / den;
        let mut h = vec![0.0; q];
        for i in 0..n {
            if event[i] {
                let e = w[i] * (-delta[i]).exp();
                for (a, &j) in kept.iter().enumerate() {
                    h[a] += e * (x_unexposed[(i, j)] - x[(i, j)]);
                }
            }
        }
        for v in &mut h {
            *v /= n_f;
        }
        grad = h;
        scale_a = 0.0;
        scale_b = 0.0;
    } else {
        p_obs = (0..n).map(|i| w[i] * y[i]).sum::<f64>() / sum_w;
        p_unexposed = (0..n).map(|i| w[i] * p0[i]).sum::<f64>() / sum_w;
        if p_obs == 0.0 {
            return Ok(result);
        }
        af = 1.0 - p_unexposed / p_obs;
        let mut g = vec![0.0; q];
        for i in 0..n {
            let r = w[i] * p0[i] * (1.0 - p0[i]);
            for (a, &j) in kept.iter().enumerate() {
                g[a] += r * x_unexposed[(i, j)];
            }
        }
        for v in &mut g {
            *v /= n_f;
        }
        grad = g;
        scale_a = p_unexposed / (p_obs * p_obs);
        scale_b = 1.0 / p_obs;
    }
    result.af = Some(af);
    // The influence of each row.
    let mut z = Vec::with_capacity(n);
    let mut s = vec![0.0; q];
    for i in 0..n {
        for (a, &j) in kept.iter().enumerate() {
            s[a] = w[i] * x[(i, j)] * (y[i] - fit.fitted[i]);
        }
        // g' (n V s_i) or h' (n V s_i)
        let mut gu = 0.0;
        for a in 0..q {
            let mut vs = 0.0;
            for b in 0..q {
                vs += v[a * q + b] * s[b];
            }
            gu += grad[a] * n_f * vs;
        }
        z.push(if case_control {
            (w[i] * y[i] * ((-delta[i]).exp() - af) + gu) / ybar
        } else {
            (scale_a * w[i] * (y[i] - p_obs) - scale_b * (w[i] * (p0[i] - p_unexposed) + gu)) / wbar
        });
    }
    // Sums within clusters, in order of first appearance.
    let (sums, n_clusters) = match opts.cluster {
        None => (z, 0),
        Some(codes) => {
            let mut order: Vec<u32> = Vec::new();
            let mut index = std::collections::HashMap::new();
            let mut sums: Vec<f64> = Vec::new();
            for (i, &c) in codes.iter().enumerate() {
                let k = *index.entry(c).or_insert_with(|| {
                    order.push(c);
                    sums.push(0.0);
                    sums.len() - 1
                });
                sums[k] += z[i];
            }
            let c = sums.len();
            (sums, c)
        }
    };
    result.n_clusters = n_clusters;
    let m = sums.len();
    if m >= 2 {
        let mean = sums.iter().sum::<f64>() / m as f64;
        let ss: f64 = sums.iter().map(|t| (t - mean) * (t - mean)).sum();
        let factor = if n_clusters == 0 {
            1.0 / n_f
        } else {
            n_clusters as f64 / (n_f * n_f)
        };
        let var = ss / (m - 1) as f64 * factor;
        let se = var.max(0.0).sqrt();
        result.se = Some(se);
        let zq = normal::two_sided_z(opts.level);
        match opts.scale {
            IntervalScale::Identity => {
                result.lower = Some(af - zq * se);
                result.upper = Some(af + zq * se);
            }
            IntervalScale::LogComplement => {
                let r = 1.0 - af;
                if r > 0.0 {
                    let half = zq * se / r;
                    result.lower = Some(1.0 - r * half.exp());
                    result.upper = Some(1.0 - r * (-half).exp());
                }
            }
        }
    }
    Ok(result)
}
