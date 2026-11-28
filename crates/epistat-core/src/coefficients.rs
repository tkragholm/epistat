//! The coefficient table of a fitted model and the summary of the fit.
//!
//! Every regression fit reports the same things: named coefficients with
//! their standard errors, test statistics, p-values and intervals, the
//! covariance they came from, and a few numbers about the fit as a whole.
//! [`Coefficients`] is that table, as R's `summary(fit)$coefficients` and
//! broom's `tidy(fit, conf.int = TRUE)`; [`Glance`] is the one-row summary,
//! as broom's `glance(fit)`. A fit implements [`Tidy`] to hand out both.
//!
//! # Scales
//!
//! The table is built on the scale of the linear predictor: a log hazard
//! ratio for a Cox model, a log odds ratio for a logistic regression. The
//! [`EffectScale`] says which. [`Coefficients::exp`] gives the ratios, with
//! each estimate and interval exponentiated and the standard error kept on
//! the log scale, as [`Scale::Log`] defines it. An
//! identity-scale table has no `exp`, so a difference in means cannot be
//! exponentiated by accident.
//!
//! # Variance
//!
//! The standard errors come from one covariance, model-based or robust, as
//! the caller asked through [`VarianceKind`] and the table records.
//!
//! ```
//! use epistat_core::{
//!     Coefficients, Covariance, EffectScale, Estimate, Interval, Level, Scale, Term,
//!     VarianceKind,
//! };
//!
//! // A Cox fit's log hazard ratio for one covariate, as a fit's crate builds it.
//! let b = -0.673;
//! let se = 0.0893;
//! let z = 1.959_963_984_540_054;
//! let term = Term::new(
//!     "treated",
//!     Some(
//!         Estimate::point(b)
//!             .with_se(Some(se))
//!             .with_interval(Some(Interval::new(Some(b - z * se), Some(b + z * se), Level::P95)))
//!             .with_p_value(Some(4.9e-14)),
//!     ),
//!     Some(b / se),
//! );
//! let table = Coefficients::new(
//!     vec![term],
//!     Covariance::from_values(1, &[se * se])?,
//!     Level::P95,
//!     VarianceKind::ModelBased,
//!     EffectScale::LogHazardRatio,
//! )?;
//! let hr = table.exp().expect("a log scale");
//! let treated = hr.get("treated").and_then(|t| t.estimate).expect("defined");
//! assert_eq!(treated.scale, Scale::Log);
//! assert_eq!(treated.to_string(), "0.510 (95% CI 0.428 to 0.608), p = 4.9e-14");
//! # Ok::<(), epistat_core::InputError>(())
//! ```

use core::fmt;

use crate::check::{self, InputError};
use crate::covariance::Covariance;
use crate::estimate::{Estimate, Interval, Scale};
use crate::format;
use crate::hypothesis::Reference;
use crate::level::Level;

/// Which covariance a table's standard errors come from.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum VarianceKind {
    /// The inverse information of the model, R's `vcov(fit)` for `glm` and
    /// the naive variance of `coxph`.
    ModelBased,
    /// The sandwich, clustered where the fit has clusters, as `coxph`'s
    /// `robust` variance or `sandwich::vcovCL`.
    Robust,
}

impl VarianceKind {
    /// `"model-based"` or `"robust"`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ModelBased => "model-based",
            Self::Robust => "robust",
        }
    }
}

impl fmt::Display for VarianceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The scale of a model's coefficients, which decides what `exp` of one is.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum EffectScale {
    /// The scale of the outcome itself, as for a linear model; there is no
    /// ratio.
    Identity,
    /// Log hazard ratios, as a Cox model's.
    LogHazardRatio,
    /// Log subdistribution hazard ratios, as a Fine-Gray model's.
    LogSubdistributionHazardRatio,
    /// Log odds ratios, as a logistic regression's.
    LogOddsRatio,
    /// Log risk ratios, as a binomial regression with log link.
    LogRiskRatio,
    /// Log rate ratios, as a Poisson regression's.
    LogRateRatio,
    /// Log ratios of means, as a gamma or log-link Gaussian regression's.
    LogMeanRatio,
    /// Log time ratios, as an accelerated failure time model's.
    LogTimeRatio,
}

impl EffectScale {
    /// Whether the coefficients are logs of ratios, so that
    /// [`Coefficients::exp`] applies.
    #[must_use]
    pub const fn is_log(self) -> bool {
        !matches!(self, Self::Identity)
    }

    /// The short name of `exp` of a coefficient, `None` on the identity
    /// scale: `HR`, `SHR`, `OR`, `RR`, `IRR`, `MR` or `TR`.
    #[must_use]
    pub const fn ratio_label(self) -> Option<&'static str> {
        match self {
            Self::Identity => None,
            Self::LogHazardRatio => Some("HR"),
            Self::LogSubdistributionHazardRatio => Some("SHR"),
            Self::LogOddsRatio => Some("OR"),
            Self::LogRiskRatio => Some("RR"),
            Self::LogRateRatio => Some("IRR"),
            Self::LogMeanRatio => Some("MR"),
            Self::LogTimeRatio => Some("TR"),
        }
    }
}

/// One row of a coefficient table: a name, its estimate and its test
/// statistic.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Term {
    /// The name of the term: the column name the data view was given, or
    /// `x0`, `x1`, ... by position.
    pub name: String,
    /// The coefficient with its standard error, interval and p-value, `None`
    /// for a term the fit could not estimate, such as an aliased column.
    pub estimate: Option<Estimate>,
    /// The Wald statistic `b / se` on the coefficient scale, a z or t value
    /// as the table's [`Reference`] says, `None` where undefined.
    pub statistic: Option<f64>,
}

impl Term {
    /// The term `name` with its estimate and statistic.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        estimate: Option<Estimate>,
        statistic: Option<f64>,
    ) -> Self {
        Self {
            name: name.into(),
            estimate,
            statistic,
        }
    }
}

/// A coefficient table: named estimates with standard errors on one
/// covariance, at one level, on one effect scale.
///
/// Built once by the fit's crate and checked when built, so its fields are
/// private and read through methods.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(try_from = "CoefficientsParts", into = "CoefficientsParts")
)]
pub struct Coefficients {
    terms: Vec<Term>,
    covariance: Covariance,
    level: Level,
    variance: VarianceKind,
    effect: EffectScale,
    reference: Reference,
    df: Option<f64>,
    exponentiated: bool,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct CoefficientsParts {
    terms: Vec<Term>,
    covariance: Covariance,
    level: Level,
    variance: VarianceKind,
    effect: EffectScale,
    reference: Reference,
    df: Option<f64>,
    exponentiated: bool,
}

impl TryFrom<CoefficientsParts> for Coefficients {
    type Error = InputError;

    fn try_from(parts: CoefficientsParts) -> Result<Self, InputError> {
        if parts.exponentiated && !parts.effect.is_log() {
            return Err(InputError::Domain {
                what: "exponentiated",
                value: 1.0,
                domain: "a table on a log effect scale",
                row: None,
            });
        }
        let scale = if parts.exponentiated {
            Scale::Log
        } else {
            Scale::Identity
        };
        validate(&parts.terms, &parts.covariance, parts.level, scale)?;
        Self {
            terms: parts.terms,
            covariance: parts.covariance,
            level: parts.level,
            variance: parts.variance,
            effect: parts.effect,
            reference: Reference::Normal,
            df: None,
            exponentiated: parts.exponentiated,
        }
        .with_reference(parts.reference, parts.df)
    }
}

/// The checks of [`Coefficients::new`], with the scale every estimate must
/// be on: the identity for coefficients, the log for an exponentiated table.
fn validate(
    terms: &[Term],
    covariance: &Covariance,
    level: Level,
    scale: Scale,
) -> Result<(), InputError> {
    check::same_length("covariance", terms.len(), covariance.dim())?;
    for (row, term) in terms.iter().enumerate() {
        if terms[..row].iter().any(|earlier| earlier.name == term.name) {
            return Err(InputError::Duplicate {
                what: "term names",
                row,
            });
        }
        let Some(estimate) = term.estimate else {
            continue;
        };
        let refuse = |value: f64, domain: &'static str| {
            Err(InputError::Domain {
                what: "coefficient",
                value,
                domain,
                row: Some(row),
            })
        };
        if !estimate.estimate.is_finite() {
            return refuse(estimate.estimate, "finite numbers");
        }
        if estimate.scale != scale {
            return refuse(
                estimate.estimate,
                "the coefficient scale, Scale::Identity; exp gives the ratios",
            );
        }
        if let Some(interval) = estimate.interval {
            if !interval.level.same_coverage(level) {
                return refuse(
                    interval.level.confidence_value(),
                    "intervals at the table's level",
                );
            }
            if interval.coverage.is_simultaneous() && estimate.p_value.is_some() {
                return refuse(
                    estimate.estimate,
                    "no p-value beside a simultaneous interval",
                );
            }
        }
    }
    Ok(())
}

impl From<Coefficients> for CoefficientsParts {
    fn from(table: Coefficients) -> Self {
        Self {
            terms: table.terms,
            covariance: table.covariance,
            level: table.level,
            variance: table.variance,
            effect: table.effect,
            reference: table.reference,
            df: table.df,
            exponentiated: table.exponentiated,
        }
    }
}

impl Coefficients {
    /// The table of `terms` on the coefficient scale, their `covariance`,
    /// intervals at `level`, standard errors from the `variance` asked for,
    /// on the scale `effect`. The statistics are referred to the standard
    /// normal; [`with_reference`](Self::with_reference) sets a t reference.
    ///
    /// # Errors
    ///
    /// [`InputError::Length`] when the covariance's dimension differs from
    /// the number of terms, [`InputError::Duplicate`] for a name that an
    /// earlier term has, with the later term's position, and
    /// [`InputError::Domain`] for a term whose estimate is not finite, is on
    /// the log scale (the table holds coefficients, `exp` gives the ratios),
    /// has an interval at another level than `level`, or has a p-value beside
    /// a simultaneous interval.
    pub fn new(
        terms: Vec<Term>,
        covariance: Covariance,
        level: Level,
        variance: VarianceKind,
        effect: EffectScale,
    ) -> Result<Self, InputError> {
        validate(&terms, &covariance, level, Scale::Identity)?;
        Ok(Self {
            terms,
            covariance,
            level,
            variance,
            effect,
            reference: Reference::Normal,
            df: None,
            exponentiated: false,
        })
    }

    /// The same table with its statistics referred to `reference` on `df`
    /// degrees of freedom, as a t reference on the residual degrees of
    /// freedom for a Gaussian or quasi model.
    ///
    /// # Errors
    ///
    /// [`InputError::Domain`] for a reference other than
    /// [`Reference::Normal`] and [`Reference::StudentT`], for degrees of
    /// freedom with a normal reference, and for a t reference without
    /// degrees of freedom that are finite and above 0.
    pub fn with_reference(
        mut self,
        reference: Reference,
        df: Option<f64>,
    ) -> Result<Self, InputError> {
        match (reference, df) {
            (Reference::Normal, None) => {}
            (Reference::StudentT, Some(df)) => check::positive("df", df)?,
            (Reference::Normal, Some(df)) => {
                return Err(InputError::Domain {
                    what: "df",
                    value: df,
                    domain: "no degrees of freedom with a normal reference",
                    row: None,
                });
            }
            _ => {
                return Err(InputError::Domain {
                    what: "reference",
                    value: df.unwrap_or(0.0),
                    domain: "the normal or a t distribution",
                    row: None,
                });
            }
        }
        self.reference = reference;
        self.df = df;
        Ok(self)
    }

    /// The names `x0`, `x1`, ... of `k` terms, for a fit whose data view was
    /// given no column names.
    #[must_use]
    pub fn positional_names(k: usize) -> Vec<String> {
        (0..k).map(|j| format!("x{j}")).collect()
    }

    /// The table with each estimate and interval exponentiated: hazard
    /// ratios, odds ratios and the like, on [`Scale::Log`], whose standard
    /// errors are those of the logs. `None` for a table on the identity scale
    /// or one already exponentiated. The covariance stays that of the
    /// coefficients.
    #[must_use]
    pub fn exp(&self) -> Option<Self> {
        if !self.effect.is_log() || self.exponentiated {
            return None;
        }
        let terms = self
            .terms
            .iter()
            .map(|term| Term {
                name: term.name.clone(),
                estimate: term.estimate.map(|e| {
                    Estimate::new(
                        e.estimate.exp(),
                        e.se,
                        e.interval.map(|i| {
                            Interval::new(i.lower.map(f64::exp), i.upper.map(f64::exp), i.level)
                                .with_coverage(i.coverage)
                        }),
                        e.p_value,
                        Scale::Log,
                    )
                }),
                statistic: term.statistic,
            })
            .collect();
        Some(Self {
            terms,
            exponentiated: true,
            ..self.clone()
        })
    }

    /// The terms in the order of the design's columns.
    #[must_use]
    pub fn terms(&self) -> &[Term] {
        &self.terms
    }

    /// The term named `name`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Term> {
        self.terms.iter().find(|term| term.name == name)
    }

    /// The term at position `index`.
    #[must_use]
    pub fn term(&self, index: usize) -> Option<&Term> {
        self.terms.get(index)
    }

    /// The position of the term named `name`.
    #[must_use]
    pub fn position(&self, name: &str) -> Option<usize> {
        self.terms.iter().position(|term| term.name == name)
    }

    /// The names of the terms, in order.
    pub fn names(&self) -> impl ExactSizeIterator<Item = &str> + '_ {
        self.terms.iter().map(|term| term.name.as_str())
    }

    /// The number of terms.
    #[must_use]
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Whether the table has no terms.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// The covariance of the coefficients, on the coefficient scale.
    #[must_use]
    pub const fn covariance(&self) -> &Covariance {
        &self.covariance
    }

    /// The covariance of the terms at `positions`, in that order, for a joint
    /// test of them; `None` for a position outside the table.
    #[must_use]
    pub fn sub_covariance(&self, positions: &[usize]) -> Option<Covariance> {
        if positions.iter().any(|&p| p >= self.terms.len()) {
            return None;
        }
        let values = positions
            .iter()
            .flat_map(|&i| positions.iter().map(move |&j| (i, j)))
            .map(|(i, j)| self.covariance.get(i, j))
            .collect();
        Covariance::new(positions.len(), values).ok()
    }

    /// The level of the intervals.
    #[must_use]
    pub const fn level(&self) -> Level {
        self.level
    }

    /// The covariance the standard errors come from.
    #[must_use]
    pub const fn variance(&self) -> VarianceKind {
        self.variance
    }

    /// The scale of the coefficients.
    #[must_use]
    pub const fn effect(&self) -> EffectScale {
        self.effect
    }

    /// The distribution the statistics are referred to.
    #[must_use]
    pub const fn reference(&self) -> Reference {
        self.reference
    }

    /// The degrees of freedom of a t reference.
    #[must_use]
    pub const fn df(&self) -> Option<f64> {
        self.df
    }

    /// Whether [`exp`](Self::exp) made this table, so its estimates are
    /// ratios.
    #[must_use]
    pub const fn is_exponentiated(&self) -> bool {
        self.exponentiated
    }

    /// The table as broom's tidy rows, one per term.
    #[must_use]
    pub fn tidy(&self) -> Vec<TidyRow> {
        self.terms
            .iter()
            .map(|term| {
                let e = term.estimate;
                TidyRow {
                    term: term.name.clone(),
                    estimate: e.map(|e| e.estimate),
                    std_error: e.and_then(|e| e.se),
                    statistic: term.statistic,
                    p_value: e.and_then(|e| e.p_value),
                    conf_low: e.and_then(|e| e.lower()),
                    conf_high: e.and_then(|e| e.upper()),
                }
            })
            .collect()
    }
}

/// One row of [`Coefficients::tidy`], broom's columns.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TidyRow {
    /// The term's name.
    pub term: String,
    /// The coefficient, or its `exp` in an exponentiated table.
    pub estimate: Option<f64>,
    /// The standard error, of the coefficient on the coefficient scale.
    pub std_error: Option<f64>,
    /// The z or t value.
    pub statistic: Option<f64>,
    /// The p-value.
    pub p_value: Option<f64>,
    /// The lower limit of the interval.
    pub conf_low: Option<f64>,
    /// The upper limit of the interval.
    pub conf_high: Option<f64>,
}

impl fmt::Display for Coefficients {
    /// The table as R's `printCoefmat` prints it, with the interval beside
    /// each row and a line saying the variance and level:
    ///
    /// ```text
    ///          Estimate  Std. Error  z value  Pr(>|z|)    2.5 %   97.5 %
    /// treated   -0.6730     0.08930   -7.536   4.8e-14  -0.8480  -0.4980
    ///
    /// Variance: model-based; 95% pointwise intervals.
    /// ```
    ///
    /// An exponentiated table heads its first column with the ratio, such as
    /// `HR`, and its second `SE (log)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let decimals = f.precision();
        let (stat, p) = match self.reference {
            Reference::StudentT => ("t value", "Pr(>|t|)"),
            _ => ("z value", "Pr(>|z|)"),
        };
        let first = if self.exponentiated {
            self.effect.ratio_label().unwrap_or("exp(coef)")
        } else {
            "Estimate"
        };
        let second = if self.exponentiated {
            "SE (log)"
        } else {
            "Std. Error"
        };
        let lower = format!(
            "{} %",
            format::Trimmed(self.level.lower_probability() * 100.0, 12)
        );
        let upper = format!(
            "{} %",
            format::Trimmed(self.level.upper_probability() * 100.0, 12)
        );
        let header = [first, second, stat, p, lower.as_str(), upper.as_str()];
        let rows: Vec<(String, Vec<String>)> = self
            .tidy()
            .into_iter()
            .map(|row| {
                let cell = |value| format::optional(value, 4, decimals);
                let cells = vec![
                    cell(row.estimate),
                    cell(row.std_error),
                    cell(row.statistic),
                    format::p_value(row.p_value),
                    cell(row.conf_low),
                    cell(row.conf_high),
                ];
                (row.term, cells)
            })
            .collect();
        format::table(f, &header, &rows)?;
        let coverage = self
            .terms
            .iter()
            .filter_map(|t| t.estimate?.interval)
            .any(|i| i.coverage.is_simultaneous());
        let coverage = if coverage {
            "simultaneous"
        } else {
            "pointwise"
        };
        write!(
            f,
            "\n\nVariance: {}; {} {coverage} intervals",
            self.variance, self.level
        )?;
        if let (Reference::StudentT, Some(df)) = (self.reference, self.df) {
            write!(f, "; t on {} df", format::Trimmed(df, 6))?;
        }
        f.write_str(".")
    }
}

/// The one-row summary of a fit, as broom's `glance`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Glance {
    /// The rows the data view held.
    pub n: usize,
    /// The rows that entered the fit, after rows with a weight of 0 or an
    /// undefined value were left out.
    pub n_used: usize,
    /// The number of events, for a survival or count model.
    pub n_events: Option<usize>,
    /// The log likelihood, or log partial likelihood, at the estimate.
    pub log_likelihood: Option<f64>,
    /// The log likelihood of the model without covariates.
    pub null_log_likelihood: Option<f64>,
    /// Akaike's information criterion.
    pub aic: Option<f64>,
    /// The Bayesian information criterion.
    pub bic: Option<f64>,
    /// The deviance.
    pub deviance: Option<f64>,
    /// The residual degrees of freedom.
    pub df_residual: Option<f64>,
    /// The iterations the fit took, `None` for a fit in closed form.
    pub iterations: Option<usize>,
    /// Whether the fit converged; a fit in closed form has.
    pub converged: bool,
}

impl Glance {
    /// The summary of a fit on `n` rows, `n_used` of them used, that did or
    /// did not converge; every other number undefined until set.
    #[must_use]
    pub const fn new(n: usize, n_used: usize, converged: bool) -> Self {
        Self {
            n,
            n_used,
            n_events: None,
            log_likelihood: None,
            null_log_likelihood: None,
            aic: None,
            bic: None,
            deviance: None,
            df_residual: None,
            iterations: None,
            converged,
        }
    }

    /// The same summary with `n_events` events.
    #[must_use]
    pub const fn with_events(mut self, n_events: Option<usize>) -> Self {
        self.n_events = n_events;
        self
    }

    /// The same summary with the log likelihoods of the fit and of the null
    /// model.
    #[must_use]
    pub const fn with_log_likelihood(mut self, fit: Option<f64>, null: Option<f64>) -> Self {
        self.log_likelihood = fit;
        self.null_log_likelihood = null;
        self
    }

    /// The same summary with the information criteria.
    #[must_use]
    pub const fn with_information(mut self, aic: Option<f64>, bic: Option<f64>) -> Self {
        self.aic = aic;
        self.bic = bic;
        self
    }

    /// The same summary with the deviance and residual degrees of freedom.
    #[must_use]
    pub const fn with_deviance(mut self, deviance: Option<f64>, df_residual: Option<f64>) -> Self {
        self.deviance = deviance;
        self.df_residual = df_residual;
        self
    }

    /// The same summary with the number of iterations.
    #[must_use]
    pub const fn with_iterations(mut self, iterations: Option<usize>) -> Self {
        self.iterations = iterations;
        self
    }
}

impl fmt::Display for Glance {
    /// `n = 100 (98 used), 40 events; log-likelihood -123.4 (null -130.2);
    /// AIC 250.8; converged in 5 iterations`, leaving out what is undefined.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "n = {}", self.n)?;
        if self.n_used != self.n {
            write!(f, " ({} used)", self.n_used)?;
        }
        if let Some(events) = self.n_events {
            write!(f, ", {events} events")?;
        }
        let number = |x| format::Significant(x, 5);
        if let Some(ll) = self.log_likelihood {
            write!(f, "; log-likelihood {}", number(ll))?;
            if let Some(null) = self.null_log_likelihood {
                write!(f, " (null {})", number(null))?;
            }
        }
        if let Some(aic) = self.aic {
            write!(f, "; AIC {}", number(aic))?;
        }
        if let Some(bic) = self.bic {
            write!(f, "; BIC {}", number(bic))?;
        }
        if let Some(deviance) = self.deviance {
            write!(f, "; deviance {}", number(deviance))?;
            if let Some(df) = self.df_residual {
                write!(f, " on {} df", format::Trimmed(df, 6))?;
            }
        }
        f.write_str(if self.converged {
            "; converged"
        } else {
            "; did not converge"
        })?;
        if let Some(iterations) = self.iterations {
            let noun = if iterations == 1 {
                "iteration"
            } else {
                "iterations"
            };
            let word = if self.converged { "in" } else { "after" };
            write!(f, " {word} {iterations} {noun}")?;
        }
        Ok(())
    }
}

/// A fitted model that hands out its coefficient table and summary.
///
/// Each fit's crate implements it, so every fit reads the same way:
///
/// ```
/// use epistat_core::{
///     Coefficients, Covariance, EffectScale, Glance, InputError, Level, Tidy, VarianceKind,
/// };
///
/// struct Fit;
///
/// impl Tidy for Fit {
///     type Error = InputError;
///
///     fn coefficients(
///         &self,
///         level: Level,
///         variance: VarianceKind,
///     ) -> Result<Coefficients, InputError> {
///         Coefficients::new(
///             Vec::new(),
///             Covariance::new(0, Vec::new())?,
///             level,
///             variance,
///             EffectScale::Identity,
///         )
///     }
///
///     fn default_variance(&self) -> VarianceKind {
///         VarianceKind::ModelBased
///     }
///
///     fn glance(&self) -> Glance {
///         Glance::new(10, 10, true)
///     }
/// }
///
/// let table = Fit.tidy()?;
/// assert_eq!(table.variance(), VarianceKind::ModelBased);
/// assert_eq!(table.level(), Level::P95);
/// assert!(Fit.glance().converged);
/// # Ok::<(), InputError>(())
/// ```
pub trait Tidy {
    /// The error of the fit's crate.
    type Error;

    /// The coefficient table with intervals at `level` and standard errors
    /// from the covariance `variance` names.
    ///
    /// # Errors
    ///
    /// When the fit has no such covariance, such as a robust variance for a
    /// fit without one, or the table cannot be built.
    fn coefficients(
        &self,
        level: Level,
        variance: VarianceKind,
    ) -> Result<Coefficients, Self::Error>;

    /// The covariance this fit reports by default, the one R's `summary`
    /// prints for the same model; each fit documents its choice.
    fn default_variance(&self) -> VarianceKind;

    /// The one-row summary of the fit.
    fn glance(&self) -> Glance;

    /// The coefficient table at the default level, 95%, with the default
    /// variance.
    ///
    /// # Errors
    ///
    /// As [`coefficients`](Self::coefficients).
    fn tidy(&self) -> Result<Coefficients, Self::Error> {
        self.coefficients(Level::default(), self.default_variance())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Classify;

    const Z: f64 = 1.959_963_984_540_054;

    fn wald(b: f64, se: f64, p: f64) -> (Option<Estimate>, Option<f64>) {
        let interval = Interval::new(Some(b - Z * se), Some(b + Z * se), Level::P95);
        (
            Some(
                Estimate::point(b)
                    .with_se(Some(se))
                    .with_interval(Some(interval))
                    .with_p_value(Some(p)),
            ),
            Some(b / se),
        )
    }

    fn table() -> Coefficients {
        let (a, za) = wald(-0.673, 0.0893, 4.8e-14);
        let (b, zb) = wald(0.02, 0.01, 0.0455);
        Coefficients::new(
            vec![
                Term::new("treated", a, za),
                Term::new("age", b, zb),
                Term::new("aliased", None, None),
            ],
            Covariance::new(
                3,
                vec![
                    Some(0.0893 * 0.0893),
                    Some(0.0001),
                    None,
                    Some(0.0001),
                    Some(0.0001),
                    None,
                    None,
                    None,
                    None,
                ],
            )
            .unwrap(),
            Level::P95,
            VarianceKind::Robust,
            EffectScale::LogHazardRatio,
        )
        .unwrap()
    }

    #[test]
    fn a_table_reads_by_name_and_position() {
        let t = table();
        assert_eq!(t.len(), 3);
        assert!(!t.is_empty());
        assert_eq!(t.names().collect::<Vec<_>>(), ["treated", "age", "aliased"]);
        assert_eq!(t.position("age"), Some(1));
        assert_eq!(
            t.get("age").and_then(|t| t.estimate).map(|e| e.estimate),
            Some(0.02)
        );
        assert_eq!(t.term(2).map(|t| t.estimate), Some(None));
        assert_eq!(t.get("missing"), None);
        assert_eq!(t.variance(), VarianceKind::Robust);
        assert_eq!(t.effect(), EffectScale::LogHazardRatio);
        assert_eq!(t.reference(), Reference::Normal);
        assert_eq!(t.level(), Level::P95);
        assert_eq!(t.df(), None);
        assert!(!t.is_exponentiated());
        let joint = t.sub_covariance(&[1, 0]).unwrap();
        assert_eq!(joint.get(0, 0), Some(0.0001));
        assert_eq!(joint.get(1, 1), Some(0.0893 * 0.0893));
        assert_eq!(t.sub_covariance(&[3]), None);
        assert_eq!(Coefficients::positional_names(3), ["x0", "x1", "x2"]);
    }

    #[test]
    fn exp_gives_ratios_with_the_log_standard_error() {
        let t = table();
        let hr = t.exp().unwrap();
        assert!(hr.is_exponentiated());
        assert_eq!(hr.exp(), None);
        let treated = hr.get("treated").unwrap().estimate.unwrap();
        assert_eq!(treated.estimate.to_bits(), (-0.673_f64).exp().to_bits());
        assert_eq!(treated.se, Some(0.0893));
        assert_eq!(treated.se_log(), Some(0.0893));
        assert_eq!(
            treated.lower().map(f64::to_bits),
            Some((-0.673 - Z * 0.0893_f64).exp().to_bits())
        );
        assert_eq!(treated.scale, Scale::Log);
        assert_eq!(hr.covariance(), t.covariance());
        let identity = Coefficients::new(
            Vec::new(),
            Covariance::new(0, Vec::new()).unwrap(),
            Level::P95,
            VarianceKind::ModelBased,
            EffectScale::Identity,
        )
        .unwrap();
        assert_eq!(identity.exp(), None);
    }

    #[test]
    fn tidy_rows_have_broom_columns() {
        let rows = table().tidy();
        assert_eq!(rows[0].term, "treated");
        assert_eq!(rows[0].std_error, Some(0.0893));
        assert_eq!(rows[0].statistic, Some(-0.673 / 0.0893));
        assert_eq!(rows[1].p_value, Some(0.0455));
        assert_eq!(rows[2].estimate, None);
        assert_eq!(rows[2].conf_low, None);
    }

    #[test]
    fn a_table_prints_as_r_prints_one() {
        let text = table().to_string();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(
            lines[0],
            "         Estimate  Std. Error  z value  Pr(>|z|)      2.5 %   97.5 %"
        );
        assert_eq!(
            lines[1],
            "treated   -0.6730     0.08930   -7.536   4.8e-14    -0.8480  -0.4980"
        );
        assert_eq!(
            lines[3],
            "aliased        NA          NA       NA        NA         NA       NA"
        );
        assert_eq!(lines[5], "Variance: robust; 95% pointwise intervals.");
        let hr = table().exp().unwrap().to_string();
        assert!(hr.starts_with("             HR  SE (log)"), "{hr}");
    }

    #[test]
    fn inconsistent_tables_are_refused() {
        let (a, za) = wald(0.1, 0.1, 0.3);
        let cov = |k| Covariance::new(k, vec![Some(1.0); k * k]).unwrap();
        let build = |terms, k| {
            Coefficients::new(
                terms,
                cov(k),
                Level::P95,
                VarianceKind::ModelBased,
                EffectScale::LogOddsRatio,
            )
        };
        assert_eq!(
            build(vec![Term::new("a", a, za)], 2).unwrap_err().code(),
            "input.length"
        );
        assert_eq!(
            build(vec![Term::new("a", a, za), Term::new("a", a, za)], 2),
            Err(InputError::Duplicate {
                what: "term names",
                row: 1
            })
        );
        let log = a.map(|e| e.with_scale(Scale::Log));
        assert!(build(vec![Term::new("a", log, za)], 1).is_err());
        let other_level =
            a.map(|e| e.with_interval(Some(Interval::new(Some(0.0), Some(0.2), Level::P90))));
        assert!(build(vec![Term::new("a", other_level, za)], 1).is_err());
        let band = a.map(|e| {
            e.with_interval(Some(
                Interval::simultaneous(Some(0.0), Some(0.2), Level::P95, 2.5).unwrap(),
            ))
        });
        assert!(build(vec![Term::new("a", band, za)], 1).is_err());
        let nan = Some(Estimate::point(f64::NAN));
        assert!(build(vec![Term::new("a", nan, None)], 1).is_err());
        let alpha = a.map(|e| {
            e.with_interval(Some(Interval::new(
                Some(0.0),
                Some(0.2),
                Level::alpha(0.05).unwrap(),
            )))
        });
        assert!(build(vec![Term::new("a", alpha, za)], 1).is_ok());
    }

    #[test]
    fn a_t_reference_needs_degrees_of_freedom() {
        let t = table();
        let with_t = t
            .clone()
            .with_reference(Reference::StudentT, Some(30.0))
            .unwrap();
        assert_eq!(with_t.df(), Some(30.0));
        assert!(with_t.to_string().contains("t value"));
        assert!(with_t.to_string().ends_with("; t on 30 df."));
        assert!(t.clone().with_reference(Reference::StudentT, None).is_err());
        assert!(
            t.clone()
                .with_reference(Reference::Normal, Some(3.0))
                .is_err()
        );
        assert!(t.with_reference(Reference::ChiSquare, Some(3.0)).is_err());
    }

    #[test]
    fn a_glance_prints_what_is_defined() {
        let g = Glance::new(100, 98, true)
            .with_events(Some(40))
            .with_log_likelihood(Some(-123.4), Some(-130.2))
            .with_information(Some(250.8), None)
            .with_iterations(Some(5));
        assert_eq!(
            g.to_string(),
            "n = 100 (98 used), 40 events; log-likelihood -123.40 (null -130.20); AIC 250.80; converged in 5 iterations"
        );
        let stuck = Glance::new(10, 10, false)
            .with_deviance(Some(12.5), Some(8.0))
            .with_iterations(Some(1));
        assert_eq!(
            stuck.to_string(),
            "n = 10; deviance 12.500 on 8 df; did not converge after 1 iteration"
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn a_table_round_trips_and_is_validated() {
        let t = table()
            .with_reference(Reference::StudentT, Some(12.0))
            .unwrap();
        let text = serde_json::to_string(&t).unwrap();
        assert_eq!(serde_json::from_str::<Coefficients>(&text).unwrap(), t);
        let duplicated = text.replace(r#""name":"age""#, r#""name":"treated""#);
        assert!(serde_json::from_str::<Coefficients>(&duplicated).is_err());
        let hr = t.exp().unwrap();
        let exponentiated = serde_json::to_string(&hr).unwrap();
        assert_eq!(
            serde_json::from_str::<Coefficients>(&exponentiated).unwrap(),
            hr
        );
        let mislabelled =
            exponentiated.replace(r#""exponentiated":true"#, r#""exponentiated":false"#);
        assert!(serde_json::from_str::<Coefficients>(&mislabelled).is_err());
        let g = Glance::new(3, 3, true);
        let back: Glance = serde_json::from_str(&serde_json::to_string(&g).unwrap()).unwrap();
        assert_eq!(back, g);
        let row = &t.tidy()[0];
        let back: TidyRow = serde_json::from_str(&serde_json::to_string(row).unwrap()).unwrap();
        assert_eq!(&back, row);
    }
}
