//! Rubin's rules for pooling an estimate over multiply imputed data sets.
//!
//! [`rubin`] pools the `m` estimates `Q_i` of one quantity and their
//! variances `U_i`, one pair per imputed data set, as Rubin \[1\] describes.
//! With
//!
//! ```text
//! Qbar = mean(Q_i)                      the pooled estimate
//! Ubar = mean(U_i)                      the within-imputation variance
//! B    = sum (Q_i - Qbar)^2 / (m - 1)   the between-imputation variance
//! T    = Ubar + (1 + 1/m) B             the total variance
//! ```
//!
//! the standard error of the pooled estimate is `sqrt(T)`. The statistics
//! reported with it are
//!
//! ```text
//! riv    = (1 + 1/m) B / Ubar           the relative increase in variance
//! lambda = (1 + 1/m) B / T              the proportion of variance due to
//!                                       the missing data
//! fmi    = (riv + 2/(df + 3)) / (riv + 1)  the fraction of missing information
//! ```
//!
//! # Degrees of freedom
//!
//! The reference distribution of `(Qbar - Q) / sqrt(T)` is a t distribution
//! with `df` degrees of freedom. Rubin's own value is `(m - 1) / lambda^2`.
//! Barnard and Rubin \[2\] correct it for a complete-data analysis on a small
//! sample, whose degrees of freedom `dfcom` the caller gives:
//!
//! ```text
//! df_old = (m - 1) / lambda^2
//! df_obs = (dfcom + 1) / (dfcom + 3) * dfcom * (1 - lambda)
//! df     = df_old * df_obs / (df_old + df_obs)
//! ```
//!
//! With `dfcom = None`, the complete-data degrees of freedom are infinite and
//! `df = df_old`. With `B = 0` and so `lambda = 0`, `df_old` is infinite and
//! `df = df_obs`, or infinite when `dfcom` is. `mice::pool()` raises a
//! `lambda` below `1e-4` to `1e-4`, which turns an infinite `df_old` into a
//! very large one, so its `df` differs from this one by less than a part in
//! a million in that case and is finite where this one is infinite.
//!
//! An undefined statistic is `None`: `riv` when `Ubar` is zero, and `lambda`,
//! `fmi` and `df` when `T` is zero. An infinite `df` is `f64::INFINITY`.
//!
//! The pooling is checked against `mice::pool()` and `mice::pool.scalar()`
//! 3.19.0.
//!
//! # Vectors and tests of several parameters
//!
//! [`rubin_vector`] pools `k` estimates with their covariance matrices,
//! given as an `m x k` matrix of estimates, one row per data set, and `m`
//! covariance matrices:
//! `Qbar` is the mean of the estimates, `Ubar` the mean of the covariances,
//! `B = sum (Q_i - Qbar)(Q_i - Qbar)' / (m - 1)` and `T = Ubar + (1 + 1/m) B`.
//! The average relative increase in variance is
//! `r = (1 + 1/m) tr(B Ubar^-1) / k`, with `tr(B Ubar^-1)` from the solution of
//! `Ubar X = B` and not from an inverse.
//!
//! [`d1`] tests that `k` estimates equal a null value, by the statistic of Li,
//! Raghunathan and Rubin \[3\],
//!
//! ```text
//! D1 = (Qbar - Q0)' [(1 + r) Ubar]^-1 (Qbar - Q0) / k  ~  F(k, nu)
//! t  = k (m - 1)
//! nu = 4 + (t - 4) [1 + (1 - 2/t) / r]^2           t > 4
//! nu = t (1 + 1/k) (1 + 1/r)^2 / 2                  otherwise
//! ```
//!
//! With a finite complete-data degrees of freedom `dfcom`, `nu` is Reiter's \[5\]
//! small-sample value, in the arithmetic of `mitml`,
//!
//! ```text
//! a = r t / (t - 2),     v* = (dfcom + 1) dfcom / (dfcom + 3)
//! c0 = 1 / (t - 4),  c1 = v* - 2 (1 + a),  c2 = v* - 4 (1 + a)
//! 1/z = 1/c2 + c0 a^2 c1 / ((1 + a)^2 c2) + c0 (8 a^2 c1 / ((1 + a) c2^2) + 4 a^2 / ((1 + a) c2))
//!       + c0 (4 a^2 / (c2 c1) + 16 a^2 c1 / c2^3) + c0 (8 a^2 / c2^2),     nu = 4 + 1/z
//! ```
//!
//! which needs `t > 4` and is refused otherwise: `mitml` returns a value there
//! that does not mean anything. With `r = 0` the sets agree and `nu` is
//! infinite, or `v*` with `dfcom`.
//!
//! [`d2`] combines the Wald statistics `d_i` of the `m` data sets \[3\]:
//!
//! ```text
//! dbar = mean(d_i),   r2 = (1 + 1/m) var(sqrt(d_i))        (divisor m - 1)
//! D2 = (dbar / k - (m + 1)/(m - 1) r2) / (1 + r2)  ~  F(k, nu2),   nu2 = k^(-3/m) (m - 1) (1 + 1/r2)^2
//! ```
//!
//! A negative `D2` is reported as 0, with `truncated` set, and has the p-value 1
//! as with `mitml`. [`d3`] combines likelihood ratio statistics, `dbar` at the
//! estimates of each data set and `dtilde` at the pooled estimates \[4\]:
//!
//! ```text
//! ariv = (m + 1) / (k (m - 1)) (dbar - dtilde),   D3 = dtilde / (k (1 + ariv))  ~  F(k, nu),   r = ariv in nu
//! ```
//!
//! An `ariv` below 0 is noise in a quantity that is a variance. With
//! [`NegativeRiv::Keep`] it enters as it is, as `mice::D3` has it, which can
//! give a negative statistic. With [`NegativeRiv::Truncate`] it is set to 0,
//! so that `nu` is infinite, and the result says so in `truncated`.
//!
//! The p-values are the upper tail of the F distribution, which is the
//! regularised incomplete beta function and for infinite `nu` the chi-square
//! distribution of `k D`.
//!
//! # Pooling on another scale
//!
//! [`rubin_transformed`] pools estimates that are not normal on their own
//! scale after a transformation `g`, with the variance `se^2 g'(q)^2` of the
//! delta method, and gives the estimate and limits on the original scale by
//! `g^-1` \[6\]. Log suits ratios, logit proportions and the c-statistic,
//! Fisher's z correlations, the complementary log-log cumulative incidences and
//! the square root a Brier score that is nearly normal on that scale. The
//! standard error on the original scale is not given, since it would need
//! another delta method.
//!
//! # References
//!
//! 1. Rubin DB. Multiple Imputation for Nonresponse in Surveys. New York:
//!    Wiley; 1987.
//! 2. Barnard J, Rubin DB. Small-sample degrees of freedom with multiple
//!    imputation. Biometrika. 1999;86(4):948-55.
//! 3. Li KH, Raghunathan TE, Rubin DB. Large-sample significance levels from
//!    multiply imputed data using moment-based statistics and an F reference
//!    distribution. J Am Stat Assoc. 1991;86(416):1065-73.
//! 4. Meng XL, Rubin DB. Performing likelihood ratio tests with
//!    multiply-imputed data sets. Biometrika. 1992;79(1):103-11.
//! 5. Reiter JP. Small-sample degrees of freedom for multi-component
//!    significance tests with multiple imputation for missing data.
//!    Biometrika. 2007;94(2):502-8.
//! 6. Marshall A, Altman DG, Holder RL, Royston P. Combining estimates of
//!    interest in prognostic modelling studies after multiple imputation:
//!    current practice and guidelines. BMC Med Res Methodol. 2009;9:57.

use epistat_core::{Covariance, InputError, Interval, Level, Reference, Test, check};
use epistat_linalg::{Attempt, Mat, MatRef, Method, solve};
use epistat_stats::dist::f::sf as f_sf;
use epistat_stats::dist::t::quantile as t_quantile;

use crate::{Error, check_length, empty};

/// The first entry of `values` that is not finite, as an input error.
fn finite(what: &'static str, values: &[f64]) -> Result<(), Error> {
    Ok(check::finite(what, values)?)
}

/// The first entry of `values` that is negative or not finite, as an input
/// error.
fn non_negative(what: &'static str, values: &[f64]) -> Result<(), Error> {
    finite(what, values)?;
    match values.iter().find(|v| **v < 0.0) {
        Some(&value) => Err(InputError::Domain {
            what,
            value,
            domain: "[0, inf)",
            row: None,
        }
        .into()),
        None => Ok(()),
    }
}

/// A complete-data degrees of freedom that is `None` or positive.
fn check_dfcom(dfcom: Option<f64>, allow_infinite: bool) -> Result<(), Error> {
    match dfcom {
        Some(d) if d.is_nan() || d <= 0.0 || (!allow_infinite && d.is_infinite()) => {
            Err(InputError::Domain {
                what: "dfcom",
                value: d,
                domain: if allow_infinite {
                    "(0, inf]"
                } else {
                    "(0, inf)"
                },
                row: None,
            }
            .into())
        }
        _ => Ok(()),
    }
}

/// At least two data sets, so that the between variance is defined.
fn at_least_two(what: &'static str, m: usize) -> Result<(), Error> {
    if m < 2 {
        Err(InputError::Length {
            what,
            expected: 2,
            found: m,
        }
        .into())
    } else {
        Ok(())
    }
}

/// An estimate pooled over imputed data sets.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pooled {
    /// The number of imputed data sets.
    pub m: usize,
    /// The pooled estimate.
    pub estimate: f64,
    /// The within-imputation variance.
    pub within: f64,
    /// The between-imputation variance.
    pub between: f64,
    /// The total variance.
    pub total: f64,
    /// The relative increase in variance, `None` when the within variance is
    /// zero.
    pub riv: Option<f64>,
    /// The proportion of the total variance due to the missing data, `None`
    /// when the total variance is zero.
    pub lambda: Option<f64>,
    /// The fraction of missing information, `None` when `lambda` is.
    pub fmi: Option<f64>,
    /// The degrees of freedom of the t reference distribution, `None` when
    /// the total variance is zero.
    pub df: Option<f64>,
}

impl Pooled {
    /// The standard error of the pooled estimate, `None` when the total
    /// variance is zero.
    #[must_use]
    pub fn se(&self) -> Option<f64> {
        (self.total > 0.0).then(|| self.total.sqrt())
    }

    /// The two-sided p-value of the null `estimate = null`, from the t
    /// distribution on `df` degrees of freedom as `mice::pool()` gives it, and
    /// the normal distribution when `df` is infinite. `None` when the total
    /// variance is zero or the distribution function does not converge.
    ///
    /// # Errors
    ///
    /// [`InputError::NotFinite`] when `null` is not finite.
    pub fn p_value(&self, null: f64) -> Result<Option<f64>, Error> {
        finite("null", &[null])?;
        let (Some(se), Some(df)) = (self.se(), self.df) else {
            return Ok(None);
        };
        let z = ((self.estimate - null) / se).abs();
        // P(|T| > z) on df degrees of freedom is the upper tail of F(1, df) at z^2.
        let p = if df.is_infinite() {
            2.0 * epistat_stats::dist::normal::cdf(-z)
        } else {
            f_sf(z * z, 1.0, df)
        };
        Ok(p.is_finite().then_some(p))
    }

    /// The confidence interval `estimate -/+ t * se` at `level`, with `t` the
    /// quantile of the t distribution at the level's upper probability,
    /// `1 - alpha / 2`, on `df` degrees of freedom. The limits are `None` when
    /// the total variance is zero or the quantile does not converge.
    #[must_use]
    pub fn interval(&self, level: Level) -> Interval {
        let none = Interval::new(None, None, level);
        let (Some(se), Some(df)) = (self.se(), self.df) else {
            return none;
        };
        let t = if df.is_infinite() {
            epistat_stats::dist::normal::quantile(level.upper_probability())
        } else {
            match t_quantile(level.upper_probability(), df) {
                t if t.is_nan() => return none,
                t => t,
            }
        };
        Interval::new(
            Some(self.estimate - t * se),
            Some(self.estimate + t * se),
            level,
        )
    }
}

/// Pools `estimates` and their `variances`, one of each per imputed data set,
/// by Rubin's rules with the Barnard-Rubin degrees of freedom for a
/// complete-data analysis on `dfcom` degrees of freedom (`None` for infinite).
///
/// # Errors
///
/// [`InputError::Empty`] for no data set, [`InputError::Length`] when the two
/// lists differ in length, and with `expected` 2 for a single data set, whose
/// between variance is not defined, [`InputError::NotFinite`] for an estimate
/// or a variance that is not finite, and [`InputError::Domain`] for a
/// negative variance or a `dfcom` that is not positive.
pub fn rubin(estimates: &[f64], variances: &[f64], dfcom: Option<f64>) -> Result<Pooled, Error> {
    let m = estimates.len();
    if m == 0 {
        return Err(empty("estimates"));
    }
    check_length("variances", m, variances.len())?;
    at_least_two("estimates", m)?;
    finite("estimates", estimates)?;
    non_negative("variances", variances)?;
    check_dfcom(dfcom, true)?;
    let mf = m as f64;
    let mut sum = 0.0;
    for q in estimates {
        sum += q;
    }
    let estimate = sum / mf;
    let mut usum = 0.0;
    for u in variances {
        usum += u;
    }
    let within = usum / mf;
    let mut ss = 0.0;
    for q in estimates {
        ss += (q - estimate) * (q - estimate);
    }
    let between = ss / (mf - 1.0);
    let inflated = (1.0 + 1.0 / mf) * between;
    let total = within + inflated;
    let riv = (within > 0.0).then(|| inflated / within);
    let lambda = (total > 0.0).then(|| inflated / total);
    let df = lambda.map(|lambda| barnard_rubin(mf, lambda, dfcom));
    let fmi = match (riv, df) {
        (Some(riv), Some(df)) => Some((riv + 2.0 / (df + 3.0)) / (riv + 1.0)),
        // A zero within variance with a positive total has an infinite riv,
        // so every unit of variance is due to the missing data.
        (None, Some(_)) => Some(1.0),
        _ => None,
    };
    Ok(Pooled {
        m,
        estimate,
        within,
        between,
        total,
        riv,
        lambda,
        fmi,
        df,
    })
}

fn barnard_rubin(m: f64, lambda: f64, dfcom: Option<f64>) -> f64 {
    let old = if lambda > 0.0 {
        (m - 1.0) / (lambda * lambda)
    } else {
        f64::INFINITY
    };
    let Some(dfcom) = dfcom else {
        return old;
    };
    let observed = (dfcom + 1.0) / (dfcom + 3.0) * dfcom * (1.0 - lambda);
    if old.is_infinite() {
        observed
    } else {
        old * observed / (old + observed)
    }
}

/// The complete-data degrees of freedom of a model fitted to `rows`
/// observations with `coefficients` parameters, `rows - coefficients`, at least
/// 1, as `mice` takes it for a regression.
#[must_use]
pub fn dfcom_residual(rows: usize, coefficients: usize) -> f64 {
    rows.saturating_sub(coefficients).max(1) as f64
}

/// The complete-data degrees of freedom of a Cox model, the number of events
/// less the number of coefficients, at least 1, as `mice::get.dfcom` takes it.
/// The information of a Cox model is in the events, not the rows.
#[must_use]
pub fn dfcom_events(events: usize, coefficients: usize) -> f64 {
    events.saturating_sub(coefficients).max(1) as f64
}

/// A vector of estimates pooled over imputed data sets.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct PooledVector {
    /// The number of imputed data sets.
    pub m: usize,
    /// The number of estimates.
    pub k: usize,
    /// The pooled estimates.
    pub estimate: Vec<f64>,
    /// The within-imputation covariance, `k x k`.
    pub within: Covariance,
    /// The between-imputation covariance, `k x k`.
    pub between: Covariance,
    /// The total covariance, `k x k`.
    pub total: Covariance,
    /// The average relative increase in variance, `(1 + 1/m) tr(B Ubar^-1) / k`.
    pub riv: f64,
}

/// How far a covariance matrix may be from symmetric, relative to the
/// geometric mean of its two diagonal entries. A covariance that comes from
/// inverting an information matrix is symmetric to about `kappa eps`, with
/// `kappa` the condition number, so `1e-8` allows `kappa` up to about `1e7`.
const SYMMETRY: f64 = 1e-8;

/// The estimates of each data set and its row-major covariance.
type Sets = (Vec<Vec<f64>>, Vec<Vec<f64>>);

/// The sets as row-major slices: the estimates of each data set and its
/// covariance, after the checks of [`rubin_vector`].
fn check_sets(estimates: MatRef<'_, f64>, covariances: &[MatRef<'_, f64>]) -> Result<Sets, Error> {
    let (m, k) = (estimates.nrows(), estimates.ncols());
    check::non_empty("estimates", m)?;
    check::non_empty("estimates columns", k)?;
    check_length("covariances", m, covariances.len())?;
    at_least_two("estimates", m)?;
    let mut rows = Vec::with_capacity(m);
    let mut matrices = Vec::with_capacity(m);
    for (set, u) in covariances.iter().enumerate() {
        check_length("covariance rows", k, u.nrows())?;
        check_length("covariance columns", k, u.ncols())?;
        let q: Vec<f64> = (0..k).map(|j| estimates[(set, j)]).collect();
        let u: Vec<f64> = (0..k * k).map(|e| u[(e / k, e % k)]).collect();
        finite("estimates", &q)?;
        finite("covariance", &u)?;
        for i in 0..k {
            if u[i * k + i] < 0.0 {
                return Err(InputError::Domain {
                    what: "covariance diagonal",
                    value: u[i * k + i],
                    domain: "[0, inf)",
                    row: None,
                }
                .into());
            }
            for j in 0..i {
                let scale = (u[i * k + i] * u[j * k + j]).sqrt();
                if (u[i * k + j] - u[j * k + i]).abs() > SYMMETRY * scale {
                    return Err(Error::NotSymmetric);
                }
            }
        }
        rows.push(q);
        matrices.push(u);
    }
    Ok((rows, matrices))
}

/// The pooled covariance parts of `m` sets of `k` estimates, symmetrised.
fn vector_parts(
    estimates: &[Vec<f64>],
    covariances: &[Vec<f64>],
    k: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mf = estimates.len() as f64;
    let mut qbar = vec![0.0; k];
    for q in estimates {
        for (acc, v) in qbar.iter_mut().zip(q.iter()) {
            *acc += v;
        }
    }
    for v in &mut qbar {
        *v /= mf;
    }
    let mut ubar = vec![0.0; k * k];
    for u in covariances {
        for (acc, v) in ubar.iter_mut().zip(u.iter()) {
            *acc += v;
        }
    }
    for v in &mut ubar {
        *v /= mf;
    }
    for i in 0..k {
        for j in 0..i {
            let mean = f64::midpoint(ubar[i * k + j], ubar[j * k + i]);
            ubar[i * k + j] = mean;
            ubar[j * k + i] = mean;
        }
    }
    let mut between = vec![0.0; k * k];
    for q in estimates {
        for i in 0..k {
            for j in 0..k {
                between[i * k + j] += (q[i] - qbar[i]) * (q[j] - qbar[j]);
            }
        }
    }
    for v in &mut between {
        *v /= mf - 1.0;
    }
    (qbar, ubar, between)
}

/// `tr(B U^-1)`, from the solution of `U X = B`.
fn trace_solve(ubar: &[f64], between: &[f64], k: usize) -> Result<f64, Error> {
    let u = Mat::from_fn(k, k, |i, j| ubar[i * k + j]);
    let b = Mat::from_fn(k, k, |i, j| between[i * k + j]);
    let x = solve(u.as_ref(), b.as_ref(), &[Attempt::plain(Method::FullPivLu)])?;
    let trace: f64 = (0..k).map(|i| x.x[(i, i)]).sum();
    if trace.is_finite() {
        Ok(trace)
    } else {
        Err(Error::NotFinite)
    }
}

/// Pools `k` estimates and their covariance matrices: row `i` of `estimates`
/// (`m x k`) and `covariances[i]` (`k x k`) come from imputed data set `i`.
///
/// # Errors
///
/// [`InputError::Empty`] for no estimates, [`InputError::Length`] for a set of
/// the wrong size or fewer than two sets, [`InputError::NotFinite`] for a
/// value that is not finite, [`InputError::Domain`] for a negative variance,
/// [`Error::NotSymmetric`] for a covariance that is not symmetric within
/// `1e-8` of its scale, and [`Error::Solve`] when the pooled within
/// covariance is singular, which means an aliased parameter in the models.
pub fn rubin_vector(
    estimates: MatRef<'_, f64>,
    covariances: &[MatRef<'_, f64>],
) -> Result<PooledVector, Error> {
    let (rows, matrices) = check_sets(estimates, covariances)?;
    let (m, k) = (rows.len(), estimates.ncols());
    let (estimate, within, between) = vector_parts(&rows, &matrices, k);
    let inflation = 1.0 + 1.0 / m as f64;
    let riv = inflation * trace_solve(&within, &between, k)? / k as f64;
    let total: Vec<f64> = within
        .iter()
        .zip(&between)
        .map(|(u, b)| u + inflation * b)
        .collect();
    Ok(PooledVector {
        m,
        k,
        estimate,
        within: Covariance::from_values(k, &within)?,
        between: Covariance::from_values(k, &between)?,
        total: Covariance::from_values(k, &total)?,
        riv,
    })
}

/// The method of a test of several parameters.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiMethod {
    /// The statistic of Li, Raghunathan and Rubin on pooled estimates.
    D1,
    /// The combination of Wald statistics.
    D2,
    /// The combination of likelihood ratio statistics of Meng and Rubin.
    D3,
}

/// How a negative estimate of the relative increase in variance is handled by
/// [`d3`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NegativeRiv {
    /// Use it as it is, as `mice::D3` does.
    Keep,
    /// Set it to 0 and say so in the result.
    Truncate,
}

/// A test of `k` parameters at once, with the F reference distribution.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiTest {
    /// The method.
    pub method: MultiMethod,
    /// The F test: its statistic, `None` when it is not finite; `df`, the
    /// numerator degrees of freedom, the number of parameters; the
    /// denominator degrees of freedom in [`Reference::F`], infinite when the
    /// data sets agree; and the upper tail probability, `None` when the F
    /// distribution function does not converge.
    pub test: Test,
    /// The relative increase in variance that went into `df2`.
    pub riv: f64,
    /// Whether a negative statistic or relative increase was set to 0.
    pub truncated: bool,
}

impl MultiTest {
    /// The denominator degrees of freedom of the F test, infinite when the
    /// data sets agree.
    #[must_use]
    pub const fn df2(&self) -> f64 {
        match self.test.distribution {
            Reference::F { df2 } => df2,
            _ => f64::INFINITY,
        }
    }
}

fn multi_test(
    method: MultiMethod,
    statistic: f64,
    k: usize,
    df2: f64,
    riv: f64,
    truncated: bool,
) -> MultiTest {
    let df1 = k as f64;
    let finite = statistic.is_finite().then_some(statistic);
    let p_value = finite.and_then(|statistic| {
        let p = f_sf(statistic, df1, df2);
        p.is_finite().then_some(p)
    });
    MultiTest {
        method,
        test: Test::new(finite, Some(df1), p_value, Reference::F { df2 }),
        riv,
        truncated,
    }
}

/// The degrees of freedom of the F distribution of D1 and D3 from the relative
/// increase in variance `r`, for `t = k (m - 1)`.
fn li_df(t: f64, k: f64, r: f64) -> f64 {
    if r == 0.0 {
        return f64::INFINITY;
    }
    if t > 4.0 {
        let inner = 1.0 + (1.0 - 2.0 / t) / r;
        4.0 + (t - 4.0) * inner * inner
    } else {
        let inner = 1.0 + 1.0 / r;
        t * (1.0 + 1.0 / k) * inner * inner / 2.0
    }
}

/// Reiter's degrees of freedom for a finite complete-data `dfcom`, with
/// `t = k (m - 1) > 4`.
fn reiter_df(t: f64, r: f64, dfcom: f64) -> f64 {
    let vstar = (dfcom + 1.0) / (dfcom + 3.0) * dfcom;
    if r == 0.0 {
        return vstar;
    }
    let a = r * t / (t - 2.0);
    let c0 = 1.0 / (t - 4.0);
    let c1 = vstar - 2.0 * (1.0 + a);
    let c2 = vstar - 4.0 * (1.0 + a);
    let z = 1.0 / c2
        + c0 * (a * a * c1 / ((1.0 + a) * (1.0 + a) * c2))
        + c0 * (8.0 * a * a * c1 / ((1.0 + a) * c2 * c2) + 4.0 * a * a / ((1.0 + a) * c2))
        + c0 * (4.0 * a * a / (c2 * c1) + 16.0 * a * a * c1 / (c2 * c2 * c2))
        + c0 * (8.0 * a * a / (c2 * c2));
    4.0 + 1.0 / z
}

/// The test that `k` parameters equal `null` (zero when `None`), by D1 on the
/// pooled estimates and covariances of `m` imputed data sets. `dfcom` is the
/// complete-data degrees of freedom for Reiter's small-sample reference
/// distribution, `None` for infinite; see [`dfcom_residual`] and
/// [`dfcom_events`].
///
/// # Errors
///
/// As [`rubin_vector`], [`InputError::Length`] for a `null` that is not `k`
/// long, [`InputError::NotFinite`] for a `null` that is not finite,
/// [`InputError::Domain`] for a `dfcom` that is not positive and finite, and
/// [`Error::Imputations`] when `dfcom` is given and `k (m - 1)` is not above
/// 4.
pub fn d1(
    estimates: MatRef<'_, f64>,
    covariances: &[MatRef<'_, f64>],
    null: Option<&[f64]>,
    dfcom: Option<f64>,
) -> Result<MultiTest, Error> {
    let (rows, matrices) = check_sets(estimates, covariances)?;
    let (m, k) = (rows.len(), estimates.ncols());
    if let Some(null) = null {
        check_length("null", k, null.len())?;
        finite("null", null)?;
    }
    check_dfcom(dfcom, false)?;
    let kf = k as f64;
    let t = kf * (m as f64 - 1.0);
    if dfcom.is_some() && t <= 4.0 {
        return Err(Error::Imputations {
            what: "D1 with dfcom",
            k,
            m,
        });
    }
    let (qbar, ubar, between) = vector_parts(&rows, &matrices, k);
    let r = (1.0 + 1.0 / m as f64) * trace_solve(&ubar, &between, k)? / kf;
    let diff: Vec<f64> = qbar
        .iter()
        .enumerate()
        .map(|(i, q)| q - null.map_or(0.0, |n| n[i]))
        .collect();
    let u = Mat::from_fn(k, k, |i, j| ubar[i * k + j]);
    let b = Mat::from_fn(k, 1, |i, _| diff[i]);
    let x = solve(u.as_ref(), b.as_ref(), &[Attempt::plain(Method::FullPivLu)])?;
    let quad: f64 = (0..k).map(|i| diff[i] * x.x[(i, 0)]).sum();
    let statistic = quad / (1.0 + r) / kf;
    if !statistic.is_finite() {
        return Err(Error::NotFinite);
    }
    let df2 = match dfcom {
        Some(d) => reiter_df(t, r, d),
        None => li_df(t, kf, r),
    };
    Ok(multi_test(MultiMethod::D1, statistic, k, df2, r, false))
}

/// The test that `k` parameters are zero, by D2 from the Wald statistic of each
/// imputed data set. A negative statistic is reported as 0 with `truncated`
/// set.
///
/// # Errors
///
/// [`InputError::Empty`] for no statistic or `k = 0`, [`InputError::Length`]
/// for fewer than two data sets, [`InputError::NotFinite`] for a statistic
/// that is not finite and [`InputError::Domain`] for one that is negative.
pub fn d2(wald_statistics: &[f64], k: usize) -> Result<MultiTest, Error> {
    let m = wald_statistics.len();
    check::non_empty("statistics", m)?;
    check::non_empty("parameters", k)?;
    at_least_two("statistics", m)?;
    non_negative("statistics", wald_statistics)?;
    let mf = m as f64;
    let kf = k as f64;
    let mut sum = 0.0;
    for d in wald_statistics {
        sum += d;
    }
    let dbar = sum / mf;
    let mut roots = 0.0;
    for d in wald_statistics {
        roots += d.sqrt();
    }
    let root_mean = roots / mf;
    let mut ss = 0.0;
    for d in wald_statistics {
        ss += (d.sqrt() - root_mean) * (d.sqrt() - root_mean);
    }
    let r = (1.0 + 1.0 / mf) * (ss / (mf - 1.0));
    let raw = (dbar / kf - (mf + 1.0) / (mf - 1.0) * r) / (1.0 + r);
    let df2 = if r == 0.0 {
        f64::INFINITY
    } else {
        kf.powf(-3.0 / mf) * (mf - 1.0) * (1.0 + 1.0 / r) * (1.0 + 1.0 / r)
    };
    let truncated = raw < 0.0;
    let statistic = if truncated { 0.0 } else { raw };
    Ok(multi_test(MultiMethod::D2, statistic, k, df2, r, truncated))
}

/// The test that `k` parameters are zero, by D3 from the likelihood ratio
/// statistic of each imputed data set at its own estimates, `lr_own`, and at
/// the pooled estimates, `lr_pooled`. Each is the deviance of the null model
/// less that of the full model.
///
/// # Errors
///
/// [`InputError::Empty`] for no data set or `k = 0`, [`InputError::Length`]
/// for lists of different length or fewer than two sets, and
/// [`InputError::NotFinite`] for a statistic that is not finite.
pub fn d3(
    lr_own: &[f64],
    lr_pooled: &[f64],
    k: usize,
    negative: NegativeRiv,
) -> Result<MultiTest, Error> {
    let m = lr_own.len();
    check::non_empty("statistics", m)?;
    check::non_empty("parameters", k)?;
    check_length("pooled statistics", m, lr_pooled.len())?;
    at_least_two("statistics", m)?;
    finite("statistics", lr_own)?;
    finite("pooled statistics", lr_pooled)?;
    let mf = m as f64;
    let kf = k as f64;
    let mut own = 0.0;
    for v in lr_own {
        own += v;
    }
    let mut pooled = 0.0;
    for v in lr_pooled {
        pooled += v;
    }
    let (dbar, dtilde) = (own / mf, pooled / mf);
    let mut ariv = (mf + 1.0) / (kf * (mf - 1.0)) * (dbar - dtilde);
    let truncated = ariv < 0.0 && negative == NegativeRiv::Truncate;
    if truncated {
        ariv = 0.0;
    }
    let statistic = dtilde / (kf * (1.0 + ariv));
    let t = kf * (mf - 1.0);
    let df2 = li_df(t, kf, ariv);
    Ok(multi_test(
        MultiMethod::D3,
        statistic,
        k,
        df2,
        ariv,
        truncated,
    ))
}

/// A transformation for pooling an estimate that is not normal on its own
/// scale.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transform {
    /// `ln q`, for ratios of risks, odds or hazards, with `q > 0`.
    Log,
    /// `ln(q / (1 - q))`, for proportions and the c-statistic, with `0 < q < 1`.
    Logit,
    /// `atanh q`, for correlations, with `-1 < q < 1`.
    FisherZ,
    /// `ln(-ln(1 - q))`, for cumulative incidences, with `0 < q < 1`.
    Cloglog,
    /// `sqrt q`, for scores that are nearly normal on that scale, with `q > 0`.
    Sqrt,
}

impl Transform {
    fn name(self) -> &'static str {
        match self {
            Self::Log => "log",
            Self::Logit => "logit",
            Self::FisherZ => "Fisher z",
            Self::Cloglog => "complementary log-log",
            Self::Sqrt => "square root",
        }
    }

    fn domain(self) -> &'static str {
        match self {
            Self::Log | Self::Sqrt => "(0, inf)",
            Self::Logit | Self::Cloglog => "(0, 1)",
            Self::FisherZ => "(-1, 1)",
        }
    }

    /// `g(q)` and `g'(q)`, or the input error for a `q` outside the domain.
    fn checked(self, q: f64) -> Result<(f64, f64), Error> {
        self.forward(q).ok_or_else(|| {
            InputError::Domain {
                what: self.name(),
                value: q,
                domain: self.domain(),
                row: None,
            }
            .into()
        })
    }

    /// `g(q)` and `g'(q)`, `None` outside the domain.
    fn forward(self, q: f64) -> Option<(f64, f64)> {
        match self {
            Self::Log => (q > 0.0).then(|| (q.ln(), 1.0 / q)),
            Self::Logit => {
                (q > 0.0 && q < 1.0).then(|| ((q / (1.0 - q)).ln(), 1.0 / (q * (1.0 - q))))
            }
            Self::FisherZ => (q > -1.0 && q < 1.0).then(|| (q.atanh(), 1.0 / (1.0 - q * q))),
            Self::Cloglog => (q > 0.0 && q < 1.0).then(|| {
                let l = -(-q).ln_1p();
                (l.ln(), 1.0 / ((1.0 - q) * l))
            }),
            Self::Sqrt => (q > 0.0).then(|| (q.sqrt(), 0.5 / q.sqrt())),
        }
    }

    fn inverse(self, x: f64) -> f64 {
        match self {
            Self::Log => x.exp(),
            Self::Logit => epistat_stats::logistic::expit(x),
            Self::FisherZ => x.tanh(),
            Self::Cloglog => -(-x.exp()).exp_m1(),
            Self::Sqrt => x * x,
        }
    }
}

/// An estimate pooled on a transformed scale.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PooledOnScale {
    /// The transformation.
    pub transform: Transform,
    /// The pooled estimate and variances on the transformed scale.
    pub pooled: Pooled,
    /// The pooled estimate on the original scale, the inverse transformation of
    /// the pooled estimate.
    pub estimate: f64,
}

impl PooledOnScale {
    /// The confidence interval at `level` on the transformed scale, as
    /// [`Pooled::interval`], carried back to the original scale. The limits
    /// are `None` where those on the transformed scale are.
    #[must_use]
    pub fn interval(&self, level: Level) -> Interval {
        let on_scale = self.pooled.interval(level);
        let back = |x: Option<f64>| x.map(|x| self.transform.inverse(x));
        Interval::new(back(on_scale.lower), back(on_scale.upper), level)
    }

    /// The two-sided p-value of the null `q = null` on the original scale, tested
    /// on the transformed scale.
    ///
    /// # Errors
    ///
    /// [`InputError::Domain`] for a `null` outside the domain of the
    /// transformation.
    pub fn p_value(&self, null: f64) -> Result<Option<f64>, Error> {
        let (g, _) = self.transform.checked(null)?;
        self.pooled.p_value(g)
    }
}

/// Pools `estimates` with the standard errors `ses`, one pair per imputed data
/// set, after the transformation `transform`. The variance on the transformed
/// scale is `se^2 g'(q)^2`. `dfcom` is as in [`rubin`].
///
/// # Errors
///
/// [`InputError::Domain`] for an estimate outside the domain of the
/// transformation or a negative standard error, [`InputError::NotFinite`]
/// for a standard error that is not finite, and the errors of [`rubin`].
pub fn rubin_transformed(
    estimates: &[f64],
    ses: &[f64],
    transform: Transform,
    dfcom: Option<f64>,
) -> Result<PooledOnScale, Error> {
    check_length("standard errors", estimates.len(), ses.len())?;
    non_negative("standard errors", ses)?;
    let mut on_scale = Vec::with_capacity(estimates.len());
    let mut variances = Vec::with_capacity(estimates.len());
    for (&q, &se) in estimates.iter().zip(ses) {
        let (g, slope) = transform.checked(q)?;
        on_scale.push(g);
        variances.push(se * se * slope * slope);
    }
    let pooled = rubin(&on_scale, &variances, dfcom)?;
    Ok(PooledOnScale {
        transform,
        estimate: transform.inverse(pooled.estimate),
        pooled,
    })
}
