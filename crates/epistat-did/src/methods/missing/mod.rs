//! Two-period difference-in-differences with the pre-treatment outcome missing
//! at random, the estimators of Testa, Kennedy and Reimherr \[1\].
//!
//! # Setting
//!
//! Each unit has covariates `X`, a treatment indicator `A`, a post-treatment
//! outcome `Y1` that is always observed, and a pre-treatment outcome `Y0` that
//! is observed when `R0 = 1`. The target is the ATT,
//! `E[Y1(1) - Y1(0) | A = 1]`, under conditional parallel trends
//! (`Y1(0) - Y0` independent of `A` given `X`), consistency and positivity of
//! `pi(x) = P(A = 1 | X = x)`, and one of two missing-at-random assumptions:
//!
//! - [`MissingAssumption::MarX`]: `Y0` independent of `R0` given `X` and `A`,
//!   with `gamma(x, a) = P(R0 = 1 | X = x, A = a)` in `(0, 1)` (their
//!   Assumption 2.3);
//! - [`MissingAssumption::MarXY1`]: `Y0` independent of `R0` given `X`, `Y1`
//!   and `A`, with `gamma(x, y1, a) = P(R0 = 1 | X, Y1, A)` in `(0, 1)`, so that
//!   whether the baseline is recorded may depend on the later outcome (their
//!   Assumption 2.4).
//!
//! # Formulas
//!
//! With `mu_t(x, a) = E[Y_t | X = x, A = a]`, where `mu_0` is fitted among the
//! units with `R0 = 1`, the efficient influence function under `MarX` is
//! (their equation 5)
//!
//! ```text
//! phi = A / E[A] * ( Y1 - mu0(X,1) - R0 / gamma(X,1) * (Y0 - mu0(X,1))
//!                    - mu1(X,0) + mu0(X,0) )
//!     - (1 - A) pi(X) / ((1 - pi(X)) E[A])
//!                  * ( Y1 - mu1(X,0) - R0 / gamma(X,0) * (Y0 - mu0(X,0)) )
//!     - A / E[A] * theta.
//! ```
//!
//! Under `MarXY1` the pre-treatment regression conditions on `Y1` as well,
//! `mu0(x, y1, a) = E[Y0 | X = x, Y1 = y1, A = a, R0 = 1]`, and the nested
//! regression `eta0(x) = E[mu0(x, Y1, 0) | X = x, A = 0]` takes the place of
//! `mu0(X, 0)` (their equation 6):
//!
//! ```text
//! phi = A / E[A] * ( Y1 - mu0(X,Y1,1) - R0 / gamma(X,Y1,1) * (Y0 - mu0(X,Y1,1))
//!                    - mu1(X,0) + eta0(X) )
//!     - (1 - A) pi(X) / ((1 - pi(X)) E[A])
//!         * ( Y1 - mu0(X,Y1,0) - R0 / gamma(X,Y1,0) * (Y0 - mu0(X,Y1,0))
//!             - mu1(X,0) + eta0(X) )
//!     - A / E[A] * theta.
//! ```
//!
//! With `Y0` always observed and `gamma = 1` both reduce to the efficient
//! influence function of the panel DiD of Sant'Anna and Zhao \[2\] (their
//! Remark 2.9), with the outcome regression of the change `Y1 - Y0` among the
//! controls equal to `mu1(X,0) - mu0(X,0)`.
//!
//! # Algorithm
//!
//! The estimator solves the sample mean of `phi` = 0 with the nuisance
//! functions replaced by estimates and `E[A]` by the share treated:
//!
//! ```text
//! theta = sum_i [ A_i t1_i - (1 - A_i) pi_i / (1 - pi_i) t0_i ] / sum_i A_i,
//! ```
//!
//! with `t1` and `t0` the brackets of the treated and the control term above.
//! The weights of the controls are `pi / (1 - pi)` divided by the number
//! treated, as in the paper's equations 12 and 13; they are not normalised
//! to sum to one as in `DRDID::drdid_panel`, so with complete data the
//! estimate differs from [`estimate_drdid_panel`](crate::estimate_drdid_panel)'s
//! by the factor `sum (1 - A) pi / (1 - pi) / sum A` on the control term,
//! which tends to 1 when the propensity model is right.
//!
//! [`att_missing_pre`] fits the nuisance models itself:
//!
//! - `pi`: logistic regression of `A` on the propensity covariates, all units;
//! - `gamma(., a)`: logistic regression of `R0` on the missingness covariates,
//!   with `Y1` under `MarXY1`, among the units of arm `a`. An arm in which
//!   every `Y0` is observed gets `gamma = 1`, the limit of the maximum
//!   likelihood fit, without a model;
//! - `mu0(., a)`: least squares of `Y0` on the outcome covariates, with `Y1`
//!   under `MarXY1`, among the units of arm `a` with `R0 = 1`;
//! - `mu1(., 0)`: least squares of `Y1` on the outcome covariates among the
//!   controls;
//! - `eta0` (`MarXY1` only): least squares of a pseudo-outcome on the nested
//!   covariates among the controls. The pseudo-outcome is `mu0(X, Y1, 0)`
//!   ([`NestedOutcome::Plain`]) or the augmented
//!   `mu0(X, Y1, 0) + R0 (Y0 - mu0(X, Y1, 0)) / gamma(X, Y1, 0)`
//!   ([`NestedOutcome::Augmented`], the default), which keeps the nested
//!   regression consistent when `gamma` is right and `mu0` is not (their
//!   Theorem 3.4).
//!
//! Every model has an intercept. The covariates are centred and scaled by
//! their mean and standard deviation over all units before the fits, which
//! leaves the fitted values unchanged and keeps the normal equations well
//! conditioned. The logistic models are fitted by Newton's method with step
//! halving, stopping after a step whose largest coefficient change (on the
//! scaled covariates) is at most [`MissingConfig::tolerance`]; Newton's
//! quadratic convergence leaves an error of the order of the square of that
//! step. A fit that does not stop within [`MissingConfig::max_iterations`]
//! steps, typically because of separation, is an error, and so is a design
//! whose Cholesky pivot leaves less than `1e-10` of a column's own squared
//! norm, a column that is collinear with the others among the rows of the
//! fit.
//!
//! With [`MissingFolds::CrossFit`] the units (or the clusters, when the data have
//! them) are put in a random order from the seed and dealt to the folds in
//! turn. For each fold the models are fitted on the other folds and predict
//! for the fold, as in the paper's equation 10, and `theta` solves the
//! estimating equation pooled over the folds. The paper writes the estimate
//! as the mean of the per-fold estimates (its equation 11), which is the same
//! when every fold has the same number of treated units. Under `MarXY1` the
//! paper fits `eta0` on a subsample independent of the models in its
//! pseudo-outcome: within each fold's training units, the order above also
//! deals every unit to one of two halves, `mu0(., Y1, 0)` and `gamma(., Y1, 0)`
//! of the pseudo-outcome are fitted on the first, and `eta0` on the
//! controls of the second. The models that enter `phi` are fitted on all the
//! training units. Without cross-fitting every model, `eta0` included, is
//! fitted on all units, as in the authors' simulations.
//!
//! [`att_missing_pre_from_predictions`] takes the predictions of any learner
//! instead, which is how random forests or other flexible models enter. For
//! the inference to hold, each unit's predictions must come from models that
//! were not fitted on that unit, by cross-fitting: split the units into `K`
//! folds, fit every model on the other folds, predict for the fold, and pass
//! the out-of-fold predictions of all units. Under `MarXY1` the nested
//! regression needs its own split within the training folds, as above.
//!
//! # Variance
//!
//! The influence function returned is `phi_i` with `E[A]` replaced by the
//! share treated, which sums to zero over the units at the estimate. The
//! standard error is the standard deviation of the `phi_i` (divisor `n`)
//! over `sqrt(n)`, the paper's `V[phi] / n`. With clusters it is the norm of
//! the cluster sums of [`did_cluster_sums`](crate::did_cluster_sums), as for
//! the other estimators of the crate. The variance takes the nuisance
//! predictions as fixed, so it is the efficient variance; it is the right one
//! for cross-fitted predictions and for parametric models that are all
//! correctly specified. With parametric models fitted on the same data and
//! only some of them right, the estimate stays consistent where the paper's
//! multiple robustness says so, but this variance does not account for the
//! estimation of the wrong models. The interval is `theta +- z se`.
//!
//! # Edge cases
//!
//! A propensity outside `(0, propensity_ceiling)` for a control, or a
//! `gamma` outside `(gamma_floor, 1]` for a unit with `Y0` observed, is an
//! error: the weights `1 / (1 - pi)` and `1 / gamma` are not trimmed or
//! clipped. [`MissingAtt`] reports the ranges of both where they enter, so
//! that overlap can be judged. An arm without units, an arm without any `Y0`
//! observed, values that are not finite, and inputs of different lengths are
//! errors. The authors' code clips both probabilities to `[0.01, 0.99]`,
//! which changes the estimand where it binds.
//!
//! # References
//!
//! 1. Testa L, Kennedy EH, Reimherr M. Efficient difference-in-differences
//!    estimation when outcomes are missing at random. arXiv:2509.25009
//!    \[stat.ME\]; 2025 (version 2, 24 January 2026).
//! 2. Sant'Anna PHC, Zhao J. Doubly robust difference-in-differences
//!    estimators. J Econom 2020;219(1):101-22.
//!    doi:10.1016/j.jeconom.2020.06.003
//! 3. Chernozhukov V, Chetverikov D, Demirer M, Duflo E, Hansen C, Newey W,
//!    Robins J. Double/debiased machine learning for treatment and structural
//!    parameters. Econom J 2018;21(1):C1-68. doi:10.1111/ectj.12097

mod fit;

use epistat_inference::Clusters;
use epistat_linalg::MatRef;

use crate::inference::did_clusters::standard_error as clustered_standard_error;
use crate::inference::{normal_interval, standard_error_from_influence, validate_confidence_level};

/// The missing-at-random assumption for the pre-treatment outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingAssumption {
    /// Whether `Y0` is observed depends on the covariates and the treatment.
    MarX,
    /// Whether `Y0` is observed depends on the covariates, the treatment and
    /// the post-treatment outcome.
    MarXY1,
}

/// The outcomes and treatment of each unit.
#[derive(Debug, Clone, Copy)]
pub struct MissingPreData<'a> {
    /// Whether each unit is treated.
    pub treated: &'a [bool],
    /// The pre-treatment outcome, `None` where it is missing.
    pub outcome_pre: &'a [Option<f64>],
    /// The post-treatment outcome, observed for every unit.
    pub outcome_post: &'a [f64],
    /// Clusters for the standard error, and for the folds of cross-fitting;
    /// every unit its own cluster when `None`.
    pub clusters: Option<&'a Clusters>,
}

/// The covariates of each nuisance model, as column indices into the
/// covariate matrix. Every model also has an intercept, and under
/// [`MissingAssumption::MarXY1`] the missingness model and the regression of
/// `Y0` also have `Y1`.
#[derive(Debug, Clone, Copy)]
pub struct MissingModels<'a> {
    /// The propensity score `pi(x)`.
    pub propensity: &'a [usize],
    /// The probability `gamma` that `Y0` is observed, one model per arm.
    pub missingness: &'a [usize],
    /// The outcome regressions `mu0(., 1)`, `mu1(., 0)` and `mu0(., 0)`.
    pub outcome: &'a [usize],
    /// The nested regression `eta0`, used under
    /// [`MissingAssumption::MarXY1`] only.
    pub nested: &'a [usize],
}

/// Whether and how the nuisance models are cross-fitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingFolds {
    /// Every model is fitted on all units.
    None,
    /// The models are fitted on all folds but one and predict for that one.
    CrossFit {
        /// The number of folds, at least 2 and at most the number of units
        /// (or clusters).
        folds: usize,
        /// The seed of the random order in which units are dealt to folds.
        seed: u64,
    },
}

/// The response of the nested regression under [`MissingAssumption::MarXY1`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NestedOutcome {
    /// The fitted `mu0(X, Y1, 0)`.
    Plain,
    /// `mu0(X, Y1, 0) + R0 (Y0 - mu0(X, Y1, 0)) / gamma(X, Y1, 0)`.
    #[default]
    Augmented,
}

/// What the estimator does besides the data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MissingConfig {
    /// The level of the confidence interval, default 0.95.
    pub confidence_level: f64,
    /// A `gamma` at or below this, for a unit whose `Y0` is observed, is an
    /// error. Default 0, any positive `gamma` is accepted.
    pub gamma_floor: f64,
    /// A propensity at or above this, for a control, is an error. Default 1.
    pub propensity_ceiling: f64,
    /// The response of the nested regression, default augmented.
    pub nested_outcome: NestedOutcome,
    /// The logistic fits stop after a Newton step no larger than this in any
    /// coefficient, on the scaled covariates. Default `1e-10`.
    pub tolerance: f64,
    /// The largest number of Newton steps of a logistic fit. Default 50.
    pub max_iterations: usize,
}

impl Default for MissingConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            gamma_floor: 0.0,
            propensity_ceiling: 1.0,
            nested_outcome: NestedOutcome::Augmented,
            tolerance: 1e-10,
            max_iterations: 50,
        }
    }
}

/// Nuisance predictions for every unit, as [`att_missing_pre_from_predictions`]
/// takes them. Every value must be finite; the ranges are checked only where a
/// value enters the estimator.
#[derive(Debug, Clone, Copy)]
pub struct MissingPredictions<'a> {
    /// `pi(X)`. Enters for the controls.
    pub propensity: &'a [f64],
    /// `gamma(X, 1)`, or `gamma(X, Y1, 1)`. Enters for treated units with
    /// `Y0` observed.
    pub gamma_treated: &'a [f64],
    /// `gamma(X, 0)`, or `gamma(X, Y1, 0)`. Enters for controls with `Y0`
    /// observed.
    pub gamma_control: &'a [f64],
    /// `mu0(X, 1)`, or `mu0(X, Y1, 1)`. Enters for the treated.
    pub pre_treated: &'a [f64],
    /// `mu1(X, 0)`. Enters for every unit.
    pub post_control: &'a [f64],
    /// `mu0(X, 0)`, or `mu0(X, Y1, 0)`. Enters for every unit under `MarX`
    /// and for the controls under `MarXY1`.
    pub pre_control: &'a [f64],
    /// `eta0(X)`: required under `MarXY1`, `None` under `MarX`, where it is
    /// `mu0(X, 0)`.
    pub nested_control: Option<&'a [f64]>,
}

/// The nuisance predictions an estimate used, one per unit, out of fold when
/// cross-fitted. The fields are those of [`MissingPredictions`].
#[derive(Debug, Clone, PartialEq)]
pub struct MissingNuisance {
    /// `pi(X)`.
    pub propensity: Vec<f64>,
    /// `gamma(X, 1)` or `gamma(X, Y1, 1)`.
    pub gamma_treated: Vec<f64>,
    /// `gamma(X, 0)` or `gamma(X, Y1, 0)`.
    pub gamma_control: Vec<f64>,
    /// `mu0(X, 1)` or `mu0(X, Y1, 1)`.
    pub pre_treated: Vec<f64>,
    /// `mu1(X, 0)`.
    pub post_control: Vec<f64>,
    /// `mu0(X, 0)` or `mu0(X, Y1, 0)`.
    pub pre_control: Vec<f64>,
    /// `eta0(X)`, under `MarXY1`.
    pub nested_control: Option<Vec<f64>>,
}

/// The ATT with its influence function and the overlap of the nuisances.
#[derive(Debug, Clone, PartialEq)]
pub struct MissingAtt {
    /// The ATT.
    pub att: f64,
    /// Its standard error, clustered when the data have clusters.
    pub se: f64,
    /// The lower end of the confidence interval.
    pub ci_low: f64,
    /// The upper end of the confidence interval.
    pub ci_high: f64,
    /// `phi_i` of every unit; see the module documentation for the scale.
    pub influence: Vec<f64>,
    /// The number of treated units.
    pub n_treated: usize,
    /// The number of controls.
    pub n_control: usize,
    /// The number of treated units with `Y0` observed.
    pub observed_pre_treated: usize,
    /// The number of controls with `Y0` observed.
    pub observed_pre_control: usize,
    /// The smallest and largest propensity among the controls.
    pub propensity_range: (f64, f64),
    /// The smallest and largest `gamma` among the treated with `Y0`
    /// observed.
    pub gamma_treated_range: (f64, f64),
    /// The smallest and largest `gamma` among the controls with `Y0`
    /// observed.
    pub gamma_control_range: (f64, f64),
    /// The number of cross-fitting folds, 1 without cross-fitting.
    pub folds: usize,
    /// The predictions the estimate used.
    pub nuisance: MissingNuisance,
}

/// Why an estimate could not be formed.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MissingError {
    /// No units.
    #[error("no units")]
    Empty,
    /// An input has another length than the treatment indicator.
    #[error("{what} has {found} entries where {expected} were expected")]
    Length {
        /// The input.
        what: &'static str,
        /// The number of units.
        expected: usize,
        /// Its length.
        found: usize,
    },
    /// A value is NaN or infinite.
    #[error("{what} of unit {unit} is not finite")]
    NotFinite {
        /// The input.
        what: &'static str,
        /// The unit.
        unit: usize,
    },
    /// A covariate index is outside the covariate matrix.
    #[error("covariate column {column} does not exist; there are {columns}")]
    Column {
        /// The index.
        column: usize,
        /// The number of columns.
        columns: usize,
    },
    /// A covariate, or `Y1` where it is a covariate, has the same value for
    /// every unit, which the intercept already is.
    #[error("{what} is constant")]
    Constant {
        /// The covariate.
        what: String,
    },
    /// No treated units, or no controls.
    #[error("there are no {0}")]
    EmptyArm(&'static str),
    /// No unit of an arm has `Y0` observed.
    #[error("no {0} has the pre-treatment outcome observed")]
    NoObservedPre(&'static str),
    /// A propensity is outside `(0, propensity_ceiling)` for a control.
    #[error("the propensity of control {unit} is {value}, outside (0, {ceiling})")]
    Propensity {
        /// The unit.
        unit: usize,
        /// The propensity.
        value: f64,
        /// [`MissingConfig::propensity_ceiling`].
        ceiling: f64,
    },
    /// A `gamma` is outside `(gamma_floor, 1]` for a unit with `Y0` observed.
    #[error("gamma of unit {unit} is {value}, outside ({floor}, 1]")]
    Gamma {
        /// The unit.
        unit: usize,
        /// The probability.
        value: f64,
        /// [`MissingConfig::gamma_floor`].
        floor: f64,
    },
    /// The nested regression is given under `MarX` or missing under `MarXY1`.
    #[error("the nested regression is required under MarXY1 and not used under MarX")]
    Nested,
    /// A nuisance model could not be fitted.
    #[error("the {model} model: {reason}")]
    Fit {
        /// The model.
        model: &'static str,
        /// What went wrong.
        reason: &'static str,
    },
    /// The configuration or the folds are invalid.
    #[error("invalid configuration: {0}")]
    Config(&'static str),
}

/// The ATT with nuisance models fitted here: logistic regressions for the
/// propensity and the missingness, least squares for the outcome regressions
/// and the nested regression, with or without cross-fitting. See the module
/// documentation.
///
/// `covariates` holds one row per unit and no intercept; `models` picks the
/// columns of each model.
///
/// # Errors
///
/// [`MissingError`] for invalid input, a model that cannot be fitted, and
/// predictions out of range.
pub fn att_missing_pre(
    data: &MissingPreData<'_>,
    covariates: MatRef<'_, f64>,
    assumption: MissingAssumption,
    models: &MissingModels<'_>,
    folds: MissingFolds,
    config: &MissingConfig,
) -> Result<MissingAtt, MissingError> {
    validate_config(config)?;
    let n = validate_data(data)?;
    if covariates.nrows() != n {
        return Err(MissingError::Length {
            what: "covariates",
            expected: n,
            found: covariates.nrows(),
        });
    }
    let (nuisance, fold_count) =
        fit::fit_nuisance(data, covariates, assumption, models, folds, config)?;
    let predictions = MissingPredictions {
        propensity: &nuisance.propensity,
        gamma_treated: &nuisance.gamma_treated,
        gamma_control: &nuisance.gamma_control,
        pre_treated: &nuisance.pre_treated,
        post_control: &nuisance.post_control,
        pre_control: &nuisance.pre_control,
        nested_control: nuisance.nested_control.as_deref(),
    };
    let core = estimate(data, assumption, &predictions, config)?;
    Ok(core.finish(fold_count, nuisance))
}

/// The ATT from nuisance predictions made elsewhere, for instance out-of-fold
/// predictions of random forests. See the module documentation for how they
/// should be made, and [`MissingPredictions`] for what enters where.
///
/// # Errors
///
/// [`MissingError`] for invalid input and predictions out of range.
pub fn att_missing_pre_from_predictions(
    data: &MissingPreData<'_>,
    assumption: MissingAssumption,
    predictions: &MissingPredictions<'_>,
    config: &MissingConfig,
) -> Result<MissingAtt, MissingError> {
    validate_config(config)?;
    validate_data(data)?;
    let core = estimate(data, assumption, predictions, config)?;
    let nuisance = MissingNuisance {
        propensity: predictions.propensity.to_vec(),
        gamma_treated: predictions.gamma_treated.to_vec(),
        gamma_control: predictions.gamma_control.to_vec(),
        pre_treated: predictions.pre_treated.to_vec(),
        post_control: predictions.post_control.to_vec(),
        pre_control: predictions.pre_control.to_vec(),
        nested_control: predictions.nested_control.map(<[f64]>::to_vec),
    };
    Ok(core.finish(1, nuisance))
}

fn validate_config(config: &MissingConfig) -> Result<(), MissingError> {
    if !validate_confidence_level(config.confidence_level) {
        return Err(MissingError::Config("confidence_level must be in (0, 1)"));
    }
    if !(config.gamma_floor.is_finite() && (0.0..1.0).contains(&config.gamma_floor)) {
        return Err(MissingError::Config("gamma_floor must be in [0, 1)"));
    }
    if !(config.propensity_ceiling > 0.0 && config.propensity_ceiling <= 1.0) {
        return Err(MissingError::Config("propensity_ceiling must be in (0, 1]"));
    }
    if !(config.tolerance.is_finite() && config.tolerance > 0.0) {
        return Err(MissingError::Config("tolerance must be positive"));
    }
    if config.max_iterations == 0 {
        return Err(MissingError::Config("max_iterations must be positive"));
    }
    Ok(())
}

/// Checks lengths and values, and that both arms have units; returns `n`.
fn validate_data(data: &MissingPreData<'_>) -> Result<usize, MissingError> {
    let n = data.treated.len();
    if n == 0 {
        return Err(MissingError::Empty);
    }
    for (what, found) in [
        ("outcome_pre", data.outcome_pre.len()),
        ("outcome_post", data.outcome_post.len()),
        ("clusters", data.clusters.map_or(n, Clusters::len)),
    ] {
        if found != n {
            return Err(MissingError::Length {
                what,
                expected: n,
                found,
            });
        }
    }
    for unit in 0..n {
        if !data.outcome_post[unit].is_finite() {
            return Err(MissingError::NotFinite {
                what: "outcome_post",
                unit,
            });
        }
        if data.outcome_pre[unit].is_some_and(|v| !v.is_finite()) {
            return Err(MissingError::NotFinite {
                what: "outcome_pre",
                unit,
            });
        }
    }
    let treated = data.treated.iter().filter(|&&a| a).count();
    if treated == 0 {
        return Err(MissingError::EmptyArm("treated units"));
    }
    if treated == n {
        return Err(MissingError::EmptyArm("controls"));
    }
    Ok(n)
}

/// The estimate before the fold count and the nuisance are attached.
struct Core {
    att: f64,
    se: f64,
    ci: (f64, f64),
    influence: Vec<f64>,
    counts: [usize; 4],
    propensity_range: (f64, f64),
    gamma_ranges: [(f64, f64); 2],
}

impl Core {
    fn finish(self, folds: usize, nuisance: MissingNuisance) -> MissingAtt {
        MissingAtt {
            att: self.att,
            se: self.se,
            ci_low: self.ci.0,
            ci_high: self.ci.1,
            influence: self.influence,
            n_treated: self.counts[0],
            n_control: self.counts[1],
            observed_pre_treated: self.counts[2],
            observed_pre_control: self.counts[3],
            propensity_range: self.propensity_range,
            gamma_treated_range: self.gamma_ranges[0],
            gamma_control_range: self.gamma_ranges[1],
            folds,
            nuisance,
        }
    }
}

/// A sum with Neumaier's compensation, in input order.
pub(super) fn compensated_sum(values: impl Iterator<Item = f64>) -> f64 {
    let (mut sum, mut carry) = (0.0_f64, 0.0_f64);
    for v in values {
        let t = sum + v;
        if sum.abs() >= v.abs() {
            carry += (sum - t) + v;
        } else {
            carry += (v - t) + sum;
        }
        sum = t;
    }
    sum + carry
}

fn widen(range: &mut (f64, f64), value: f64) {
    range.0 = range.0.min(value);
    range.1 = range.1.max(value);
}

/// The estimator of equations 12 and 13 on given predictions, with the
/// influence function and its standard error.
fn estimate(
    data: &MissingPreData<'_>,
    assumption: MissingAssumption,
    p: &MissingPredictions<'_>,
    config: &MissingConfig,
) -> Result<Core, MissingError> {
    let n = data.treated.len();
    let nested = match (assumption, p.nested_control) {
        (MissingAssumption::MarX, None) => None,
        (MissingAssumption::MarXY1, Some(eta)) => Some(eta),
        _ => return Err(MissingError::Nested),
    };
    let mut columns: Vec<(&'static str, &[f64])> = vec![
        ("propensity", p.propensity),
        ("gamma_treated", p.gamma_treated),
        ("gamma_control", p.gamma_control),
        ("pre_treated", p.pre_treated),
        ("post_control", p.post_control),
        ("pre_control", p.pre_control),
    ];
    if let Some(eta) = nested {
        columns.push(("nested_control", eta));
    }
    for (what, values) in &columns {
        if values.len() != n {
            return Err(MissingError::Length {
                what,
                expected: n,
                found: values.len(),
            });
        }
        if let Some(unit) = values.iter().position(|v| !v.is_finite()) {
            return Err(MissingError::NotFinite { what, unit });
        }
    }

    let empty = (f64::INFINITY, f64::NEG_INFINITY);
    let mut propensity_range = empty;
    let mut gamma_ranges = [empty; 2];
    let mut counts = [0_usize; 4];
    let mut terms = Vec::with_capacity(n);
    for i in 0..n {
        let y1 = data.outcome_post[i];
        let y0 = data.outcome_pre[i];
        let term = if data.treated[i] {
            counts[0] += 1;
            let m01 = p.pre_treated[i];
            let augmentation = match y0 {
                Some(y0) => {
                    counts[2] += 1;
                    let gamma = p.gamma_treated[i];
                    check_gamma(i, gamma, config)?;
                    widen(&mut gamma_ranges[0], gamma);
                    (y0 - m01) / gamma
                }
                None => 0.0,
            };
            let eta = nested.map_or(p.pre_control[i], |eta| eta[i]);
            y1 - (m01 + augmentation) - p.post_control[i] + eta
        } else {
            counts[1] += 1;
            let pi = p.propensity[i];
            if !(pi > 0.0 && pi < config.propensity_ceiling) {
                return Err(MissingError::Propensity {
                    unit: i,
                    value: pi,
                    ceiling: config.propensity_ceiling,
                });
            }
            widen(&mut propensity_range, pi);
            let m00 = p.pre_control[i];
            let augmentation = match y0 {
                Some(y0) => {
                    counts[3] += 1;
                    let gamma = p.gamma_control[i];
                    check_gamma(i, gamma, config)?;
                    widen(&mut gamma_ranges[1], gamma);
                    (y0 - m00) / gamma
                }
                None => 0.0,
            };
            let bracket = match nested {
                None => y1 - p.post_control[i] - augmentation,
                Some(eta) => y1 - (m00 + augmentation) - p.post_control[i] + eta[i],
            };
            -(pi / (1.0 - pi)) * bracket
        };
        terms.push(term);
    }
    let n_treated = counts[0] as f64;
    let att = compensated_sum(terms.iter().copied()) / n_treated;
    let scale = n as f64 / n_treated;
    let influence: Vec<f64> = terms
        .iter()
        .zip(data.treated)
        .map(|(&t, &a)| scale * (if a { t - att } else { t }))
        .collect();
    let se = match data.clusters {
        None => standard_error_from_influence(&influence),
        Some(clusters) => clustered_standard_error(&influence, clusters),
    };
    let ci = normal_interval(att, se, config.confidence_level);
    Ok(Core {
        att,
        se,
        ci,
        influence,
        counts,
        propensity_range,
        gamma_ranges,
    })
}

fn check_gamma(unit: usize, gamma: f64, config: &MissingConfig) -> Result<(), MissingError> {
    if gamma > config.gamma_floor && gamma > 0.0 && gamma <= 1.0 {
        Ok(())
    } else {
        Err(MissingError::Gamma {
            unit,
            value: gamma,
            floor: config.gamma_floor,
        })
    }
}

#[cfg(test)]
mod tests;
