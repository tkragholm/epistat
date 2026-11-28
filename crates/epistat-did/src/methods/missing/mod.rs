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
//! estimate differs from the panel doubly robust
//! [`two_period::estimate`](crate::two_period::estimate)'s
//! by the factor `sum (1 - A) pi / (1 - pi) / sum A` on the control term,
//! which tends to 1 when the propensity model is right.
//!
//! [`estimate`] fits the nuisance models itself:
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
//! scaled covariates) is at most [`MissingOptions::tolerance`]; Newton's
//! quadratic convergence leaves an error of the order of the square of that
//! step. A fit that does not stop within [`MissingOptions::max_iterations`]
//! steps, typically because of separation, is an error, and so is a design
//! whose Cholesky pivot leaves less than `1e-10` of a column's own squared
//! norm, a column that is collinear with the others among the rows of the
//! fit.
//!
//! With [`MissingFolds::CrossFit`] the units (or the clusters, when the data
//! have them) are put in a random order from the seed and dealt to the folds
//! in turn. For each fold the models are fitted on the other folds and
//! predict for the fold, as in the paper's equation 10, and `theta` solves
//! the estimating equation pooled over the folds. The paper writes the
//! estimate as the mean of the per-fold estimates (its equation 11), which is
//! the same when every fold has the same number of treated units. Under
//! `MarXY1` the paper fits `eta0` on a subsample independent of the models in
//! its pseudo-outcome: within each fold's training units, the order above
//! also deals every unit to one of two halves, `mu0(., Y1, 0)` and
//! `gamma(., Y1, 0)` of the pseudo-outcome are fitted on the first, and
//! `eta0` on the controls of the second. The models that enter `phi` are
//! fitted on all the training units. [`cross_fit_folds`] gives the
//! folds and halves. Without cross-fitting every model, `eta0` included, is
//! fitted on all units, as in the authors' simulations.
//!
//! [`from_predictions`] takes the predictions of any learner
//! instead, which is how random forests or other flexible models enter. For
//! the inference to hold, each unit's predictions must come from models that
//! were not fitted on that unit, by cross-fitting: split the units into `K`
//! folds, fit every model on the other folds, predict for the fold, and pass
//! the out-of-fold predictions of all units. Under `MarXY1` the nested
//! regression needs its own split within the training folds, as above. The
//! documentation of [`from_predictions`] lists the steps.
//!
//! # Variance
//!
//! The influence function returned is `phi_i` with `E[A]` replaced by the
//! share treated, which sums to zero over the units at the estimate, as an
//! [`Influence`] keyed by row under [`ClusterRule::NoSmallSample`]. The
//! standard error is the standard deviation of the `phi_i` (divisor `n`)
//! over `sqrt(n)`, the paper's `V[phi] / n`. With clusters it is the norm of
//! the cluster sums of [`did_cluster_sums`](crate::did_cluster_sums), as for
//! the other estimators of the crate, and [`Influence::variance`] with the
//! same clusters gives it to rounding. The variance takes the nuisance
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
//! errors.
//!
//! # The authors' code
//!
//! The replication code of the paper (github.com/testalorenzo/MissingDiD,
//! `dgp.py`) computes the same estimator, with three differences, where this
//! module follows the paper. It regresses the nested pseudo-outcome on the
//! covariates of all units rather than of the controls, which is a different
//! function when the treatment shifts `Y1` and matters when the propensity
//! model is wrong. It clips the fitted propensity and `gamma` to
//! `[0.01, 0.99]`, which changes the estimand where it binds; here they are
//! used as fitted and refused at the caller's limits. Its simulations fit
//! every model on all units, as [`MissingFolds::None`] does, without the
//! cross-fitting the paper's theory assumes. On the reference's own
//! predictions the estimate, influence function and standard error of
//! [`from_predictions`] agree with the code's to rounding.
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

use epistat_core::{Estimate, InputError, Level, check};
use epistat_inference::{ClusterRule, Clusters, Influence};
use epistat_linalg::MatRef;

use crate::error::Error;
use crate::inference::did_clusters::standard_error as clustered_standard_error;
use crate::inference::{estimate_from, normal_interval, standard_error_from_influence};

/// The missing-at-random assumption for the pre-treatment outcome.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MissingAssumption {
    /// Whether `Y0` is observed depends on the covariates and the treatment,
    /// the default.
    #[default]
    MarX,
    /// Whether `Y0` is observed depends on the covariates, the treatment and
    /// the post-treatment outcome.
    MarXY1,
}

/// A validated sample of units, one entry per unit: the treatment, both
/// outcomes, and optionally covariates and clusters.
///
/// [`MissingData::builder`] takes the required columns, the builder's
/// methods add the optional ones, and [`MissingDataBuilder::build`] checks
/// every column once:
///
/// | column | check |
/// |---|---|
/// | every column | one entry per unit, at least one unit, at most `u32::MAX` units |
/// | `outcome_post` | finite |
/// | `outcome_pre` | finite where observed |
/// | `covariates` | finite |
///
/// The covariates are given without an intercept: every nuisance model adds
/// it. Cluster codes, when given, cluster the standard error and are dealt
/// to the folds of cross-fitting whole; they are numbered in increasing code
/// ([`Clusters::from_codes`]).
///
/// ```
/// use epistat_did::missing::{self, MissingData, MissingOptions, NuisanceModels};
/// use epistat_linalg::MatRef;
///
/// let treated = [true, true, true, false, false, false, false, false];
/// let pre = [Some(1.0), None, Some(1.4), Some(0.9), Some(1.1), None, Some(1.0), Some(1.3)];
/// let post = [3.2, 2.9, 3.6, 1.0, 1.3, 1.2, 1.1, 1.5];
/// let age = [41.0, 37.0, 45.0, 40.0, 38.0, 44.0, 39.0, 43.0];
/// let data = MissingData::builder(&treated, &pre, &post)
///     .covariates(MatRef::from_column_major_slice(&age, 8, 1))
///     .build()?;
/// // Age in the propensity and outcome models, the missingness by arm only.
/// let options = MissingOptions {
///     models: NuisanceModels {
///         missingness: Some(vec![]),
///         ..NuisanceModels::default()
///     },
///     ..MissingOptions::default()
/// };
/// let fit = missing::estimate(&data, &options)?;
/// assert!(fit.estimate.estimate > 1.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct MissingData<'a> {
    treated: &'a [bool],
    outcome_pre: &'a [Option<f64>],
    outcome_post: &'a [f64],
    covariates: Option<MatRef<'a, f64>>,
    clusters: Option<Clusters>,
}

/// The builder of a [`MissingData`].
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct MissingDataBuilder<'a> {
    treated: &'a [bool],
    outcome_pre: &'a [Option<f64>],
    outcome_post: &'a [f64],
    covariates: Option<MatRef<'a, f64>>,
    clusters: Option<&'a [u32]>,
}

impl<'a> MissingData<'a> {
    /// Whether each unit is treated, its pre-treatment outcome (`None` where
    /// it is missing) and its post-treatment outcome, which is observed for
    /// every unit.
    pub const fn builder(
        treated: &'a [bool],
        outcome_pre: &'a [Option<f64>],
        outcome_post: &'a [f64],
    ) -> MissingDataBuilder<'a> {
        MissingDataBuilder {
            treated,
            outcome_pre,
            outcome_post,
            covariates: None,
            clusters: None,
        }
    }

    /// Whether each unit is treated.
    #[must_use]
    pub const fn treated(&self) -> &'a [bool] {
        self.treated
    }

    /// The pre-treatment outcome, `None` where it is missing.
    #[must_use]
    pub const fn outcome_pre(&self) -> &'a [Option<f64>] {
        self.outcome_pre
    }

    /// The post-treatment outcome.
    #[must_use]
    pub const fn outcome_post(&self) -> &'a [f64] {
        self.outcome_post
    }

    /// The covariates, without the intercept the models add; `None` without
    /// covariates.
    #[must_use]
    pub const fn covariates(&self) -> Option<MatRef<'a, f64>> {
        self.covariates
    }

    /// The clusters of the units, numbered in increasing code; `None` when
    /// each unit is its own cluster.
    #[must_use]
    pub const fn clusters(&self) -> Option<&Clusters> {
        self.clusters.as_ref()
    }

    /// The number of units.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.treated.len()
    }

    /// Whether there are no units, which the builder refuses, so always false.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.treated.is_empty()
    }

    /// The covariates, or a matrix without columns.
    pub(crate) fn covariate_matrix(&self) -> MatRef<'a, f64> {
        self.covariates
            .unwrap_or_else(|| MatRef::from_column_major_slice(&[], self.len(), 0))
    }
}

impl<'a> MissingDataBuilder<'a> {
    /// Covariates, one row per unit, without an intercept: the models add it.
    pub const fn covariates(mut self, covariates: MatRef<'a, f64>) -> Self {
        self.covariates = Some(covariates);
        self
    }

    /// Cluster codes, one per unit.
    pub const fn clusters(mut self, clusters: &'a [u32]) -> Self {
        self.clusters = Some(clusters);
        self
    }

    /// The validated data.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] for the first check of [`MissingData`] that fails:
    /// [`InputError::Empty`] without units, [`InputError::TooLarge`] for more
    /// units than `u32` numbers, [`InputError::Length`] for a column of the
    /// wrong length and [`InputError::NotFinite`] for an outcome or covariate
    /// that is not finite.
    pub fn build(self) -> Result<MissingData<'a>, Error> {
        let n = self.treated.len();
        check::non_empty("treated", n)?;
        check::fits_u32("units", n)?;
        check::same_length("outcome_pre", n, self.outcome_pre.len())?;
        check::same_length("outcome_post", n, self.outcome_post.len())?;
        check::finite("outcome_post", self.outcome_post)?;
        if let Some(row) = self
            .outcome_pre
            .iter()
            .position(|v| v.is_some_and(|v| !v.is_finite()))
        {
            return Err(InputError::NotFinite {
                what: "outcome_pre",
                row,
                column: None,
            }
            .into());
        }
        if let Some(x) = self.covariates {
            check::same_length("covariates", n, x.nrows())?;
            for j in 0..x.ncols() {
                if let Some(row) = (0..n).find(|&i| !x[(i, j)].is_finite()) {
                    return Err(InputError::NotFinite {
                        what: "covariates",
                        row,
                        column: Some(j),
                    }
                    .into());
                }
            }
        }
        if let Some(codes) = self.clusters {
            check::same_length("clusters", n, codes.len())?;
        }
        Ok(MissingData {
            treated: self.treated,
            outcome_pre: self.outcome_pre,
            outcome_post: self.outcome_post,
            covariates: self.covariates,
            clusters: self.clusters.map(Clusters::from_codes),
        })
    }
}

/// The outcomes and treatment of each unit, as [`att_missing_pre`] took
/// them.
#[deprecated(
    since = "0.1.0",
    note = "use MissingData::builder, with the covariates and cluster codes in the data"
)]
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
/// covariates of [`MissingData`]. Every model also has an intercept, and
/// under [`MissingAssumption::MarXY1`] the missingness model and the
/// regression of `Y0` also have `Y1`. `None`, the default for every model,
/// is every covariate; `Some(vec![])` is the intercept alone.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NuisanceModels {
    /// The propensity score `pi(x)`.
    pub propensity: Option<Vec<usize>>,
    /// The probability `gamma` that `Y0` is observed, one model per arm.
    pub missingness: Option<Vec<usize>>,
    /// The outcome regressions `mu0(., 1)`, `mu1(., 0)` and `mu0(., 0)`.
    pub outcome: Option<Vec<usize>>,
    /// The nested regression `eta0`, used under
    /// [`MissingAssumption::MarXY1`] only.
    pub nested: Option<Vec<usize>>,
}

/// The covariates of each nuisance model, as [`att_missing_pre`] took them.
#[deprecated(since = "0.1.0", note = "use NuisanceModels in MissingOptions::models")]
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
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MissingFolds {
    /// Every model is fitted on all units, the default, as in the authors'
    /// simulations.
    #[default]
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
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NestedOutcome {
    /// The fitted `mu0(X, Y1, 0)`.
    Plain,
    /// `mu0(X, Y1, 0) + R0 (Y0 - mu0(X, Y1, 0)) / gamma(X, Y1, 0)`.
    #[default]
    Augmented,
}

/// What the estimator does besides the data, built with
/// `..MissingOptions::default()`.
#[derive(Debug, Clone, PartialEq)]
pub struct MissingOptions {
    /// The missing-at-random assumption, default [`MissingAssumption::MarX`].
    pub assumption: MissingAssumption,
    /// The covariates of each nuisance model that [`estimate`] fits, by
    /// default every covariate in every model. Not used by
    /// [`from_predictions`].
    pub models: NuisanceModels,
    /// Whether [`estimate`] cross-fits its models, by default not. Not used
    /// by [`from_predictions`].
    pub folds: MissingFolds,
    /// The level of the confidence interval, default 95%.
    pub level: Level,
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

impl Default for MissingOptions {
    fn default() -> Self {
        Self {
            assumption: MissingAssumption::MarX,
            models: NuisanceModels::default(),
            folds: MissingFolds::None,
            level: Level::default(),
            gamma_floor: 0.0,
            propensity_ceiling: 1.0,
            nested_outcome: NestedOutcome::Augmented,
            tolerance: 1e-10,
            max_iterations: 50,
        }
    }
}

/// Nuisance predictions for every unit, as [`from_predictions`] takes them. Every value must be finite; the ranges are checked only where a
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
#[non_exhaustive]
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
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct MissingAtt {
    /// The ATT with its standard error, clustered when the data have
    /// clusters, the normal interval at [`MissingOptions::level`] and the
    /// normal p-value; a part that is not finite is `None`.
    pub estimate: Estimate,
    /// `phi_i` of every unit on the per-unit scale, keyed by row, under
    /// [`ClusterRule::NoSmallSample`], the rule of the standard error: its
    /// variance with the clusters of the data (or none) gives the standard
    /// error to rounding. See the module documentation.
    pub influence: Influence,
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
    /// observed, `None` when no treated unit has `Y0`.
    pub gamma_treated_range: Option<(f64, f64)>,
    /// The smallest and largest `gamma` among the controls with `Y0`
    /// observed, `None` when no control has `Y0`.
    pub gamma_control_range: Option<(f64, f64)>,
    /// The number of cross-fitting folds, 1 without cross-fitting.
    pub folds: usize,
    /// The predictions the estimate used.
    pub nuisance: MissingNuisance,
}

/// The folds of cross-fitting, as [`cross_fit_folds`] deals them.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrossFitFolds {
    /// The fold of each unit, from 0 to the number of folds less 1.
    pub fold: Vec<usize>,
    /// The half of the training units each unit belongs to for the nested
    /// regression: `false` for the half on which `mu0(., Y1, 0)` and
    /// `gamma(., Y1, 0)` of the pseudo-outcome are fitted, `true` for the
    /// half on whose controls `eta0` is fitted.
    pub nested_half: Vec<bool>,
}

/// The folds that [`estimate`] uses with [`MissingFolds::CrossFit`]: the
/// units, or the clusters, are put in a random order from `seed` and dealt
/// to the folds in turn, and within each fold alternately to the two halves
/// of the nested regression. For cross-fitting other learners on the same
/// folds before [`from_predictions`].
///
/// # Errors
///
/// [`Error::Input`] for fewer than 2 folds or more folds than units (or
/// clusters).
pub fn cross_fit_folds(
    data: &MissingData<'_>,
    folds: usize,
    seed: u64,
) -> Result<CrossFitFolds, Error> {
    let (fold, nested_half) = fit::fold_assignment(data.len(), data.clusters(), folds, seed)?;
    Ok(CrossFitFolds { fold, nested_half })
}

/// The ATT with nuisance models fitted here: logistic regressions for the
/// propensity and the missingness, least squares for the outcome regressions
/// and the nested regression, with or without cross-fitting, under
/// [`MissingOptions::assumption`]. See the module documentation.
///
/// [`MissingOptions::models`] picks the covariates of each model.
///
/// # Errors
///
/// - [`Error::Input`] for options that are invalid, and a model column
///   outside the covariates.
/// - [`Error::NoTreated`] and [`Error::NoComparison`] for an arm without
///   units, and [`Error::NoObservedPre`] for an arm without `Y0`.
/// - [`Error::ConstantCovariate`] for a model covariate that does not vary.
/// - [`Error::NuisanceFit`] for a nuisance model that cannot be fitted.
/// - [`Error::PropensityRange`] and [`Error::MissingnessRange`] for fitted
///   predictions out of range.
/// - [`Error::NotFinite`] for an influence function that is not finite.
pub fn estimate(data: &MissingData<'_>, options: &MissingOptions) -> Result<MissingAtt, Error> {
    validate_options(options)?;
    check_arms(data)?;
    let (nuisance, fold_count) = fit::fit_nuisance(data, options)?;
    let predictions = MissingPredictions {
        propensity: &nuisance.propensity,
        gamma_treated: &nuisance.gamma_treated,
        gamma_control: &nuisance.gamma_control,
        pre_treated: &nuisance.pre_treated,
        post_control: &nuisance.post_control,
        pre_control: &nuisance.pre_control,
        nested_control: nuisance.nested_control.as_deref(),
    };
    let core = estimate_core(data, &predictions, options)?;
    core.finish(fold_count, nuisance)
}

/// The ATT from nuisance predictions made elsewhere, for instance out-of-fold
/// predictions of random forests, under [`MissingOptions::assumption`].
/// [`MissingPredictions`] lists what enters where.
///
/// For the standard error to hold, the predictions of a unit must come from
/// models that did not see it. With folds from [`cross_fit_folds`] (or any
/// other split into `K` folds), for each fold `k`:
///
/// ```text
/// train = units not in fold k
/// pi            classifier of A on X, fitted on train
/// gamma(., 1)   classifier of R0 on X (and Y1 under MarXY1), on the treated of train
/// gamma(., 0)   the same on the controls of train
/// mu0(., 1)     regression of Y0 on X (and Y1 under MarXY1), on the treated of train with Y0
/// mu1(., 0)     regression of Y1 on X, on the controls of train
/// mu0(., 0)     regression of Y0 on X (and Y1 under MarXY1), on the controls of train with Y0
/// predict all six for the units of fold k
/// ```
///
/// Under MarXY1 the nested regression comes from the two halves of `train`
/// given by `nested_half`: fit `mu0(., Y1, 0)` and `gamma(., Y1, 0)` again on
/// the first half, form the pseudo-outcome `m + R0 (Y0 - m) / g` (or `m`)
/// with their predictions for the controls of the second half, regress it on
/// `X` among those controls, and predict for fold `k`.
///
/// Probabilities from a classifier such as a random forest can be exactly 0
/// or 1, which is an error here where they enter. Clipping them, as the
/// authors' code does, changes the estimand where it binds and should be a
/// visible choice of the caller's.
///
/// # Errors
///
/// - [`Error::Input`] for predictions or options that are invalid.
/// - [`Error::NoTreated`] and [`Error::NoComparison`] for an arm without
///   units.
/// - [`Error::NestedRegression`] for a nested regression given under
///   [`MissingAssumption::MarX`] or missing under
///   [`MissingAssumption::MarXY1`].
/// - [`Error::PropensityRange`] and [`Error::MissingnessRange`] for
///   predictions out of range.
/// - [`Error::NotFinite`] for an influence function that is not finite.
pub fn from_predictions(
    data: &MissingData<'_>,
    predictions: &MissingPredictions<'_>,
    options: &MissingOptions,
) -> Result<MissingAtt, Error> {
    validate_options(options)?;
    check_arms(data)?;
    let core = estimate_core(data, predictions, options)?;
    let nuisance = MissingNuisance {
        propensity: predictions.propensity.to_vec(),
        gamma_treated: predictions.gamma_treated.to_vec(),
        gamma_control: predictions.gamma_control.to_vec(),
        pre_treated: predictions.pre_treated.to_vec(),
        post_control: predictions.post_control.to_vec(),
        pre_control: predictions.pre_control.to_vec(),
        nested_control: predictions.nested_control.map(<[f64]>::to_vec),
    };
    core.finish(1, nuisance)
}

/// The view of the old data, with the clusters given by their numbers, which
/// [`Clusters::from_codes`] numbers again unchanged.
#[expect(deprecated, reason = "the deprecated entry points take the old data")]
fn view<'a>(
    data: &MissingPreData<'a>,
    covariates: Option<MatRef<'a, f64>>,
) -> Result<MissingData<'a>, Error> {
    let builder = MissingData::builder(data.treated, data.outcome_pre, data.outcome_post);
    let builder = match covariates {
        Some(x) => builder.covariates(x),
        None => builder,
    };
    let built = match data.clusters {
        Some(clusters) => builder.clusters(clusters.of()).build()?,
        None => builder.build()?,
    };
    Ok(built)
}

/// The folds of [`att_missing_pre`] with [`MissingFolds::CrossFit`].
///
/// # Errors
///
/// [`Error::Input`] for fewer than 2 folds or more folds than units (or
/// clusters), and for clusters of another length.
#[deprecated(since = "0.1.0", note = "use missing::cross_fit_folds")]
pub fn missing_cross_fit_folds(
    units: usize,
    clusters: Option<&Clusters>,
    folds: usize,
    seed: u64,
) -> Result<CrossFitFolds, Error> {
    let (fold, nested_half) = fit::fold_assignment(units, clusters, folds, seed)?;
    Ok(CrossFitFolds { fold, nested_half })
}

/// [`estimate`] with the covariates, the assumption, the models and the
/// folds as arguments; the same fields of `options` are not read.
///
/// # Errors
///
/// As [`estimate`], and [`Error::Input`] for data that are invalid.
#[deprecated(
    since = "0.1.0",
    note = "use missing::estimate, with the covariates in MissingData and the assumption, \
            models and folds in MissingOptions"
)]
#[expect(deprecated, reason = "the deprecated entry point takes the old data")]
pub fn att_missing_pre(
    data: &MissingPreData<'_>,
    covariates: MatRef<'_, f64>,
    assumption: MissingAssumption,
    models: &MissingModels<'_>,
    folds: MissingFolds,
    options: &MissingOptions,
) -> Result<MissingAtt, Error> {
    let data = view(data, Some(covariates))?;
    let options = MissingOptions {
        assumption,
        models: NuisanceModels {
            propensity: Some(models.propensity.to_vec()),
            missingness: Some(models.missingness.to_vec()),
            outcome: Some(models.outcome.to_vec()),
            nested: Some(models.nested.to_vec()),
        },
        folds,
        ..options.clone()
    };
    estimate(&data, &options)
}

/// [`from_predictions`] with the assumption as an argument; the field of
/// `options` is not read.
///
/// # Errors
///
/// As [`from_predictions`], and [`Error::Input`] for data that are invalid.
#[deprecated(
    since = "0.1.0",
    note = "use missing::from_predictions, with the assumption in MissingOptions"
)]
#[expect(deprecated, reason = "the deprecated entry point takes the old data")]
pub fn att_missing_pre_from_predictions(
    data: &MissingPreData<'_>,
    assumption: MissingAssumption,
    predictions: &MissingPredictions<'_>,
    options: &MissingOptions,
) -> Result<MissingAtt, Error> {
    let data = view(data, None)?;
    let options = MissingOptions {
        assumption,
        ..options.clone()
    };
    from_predictions(&data, predictions, &options)
}

fn domain(what: &'static str, value: f64, domain: &'static str) -> Error {
    InputError::Domain {
        what,
        value,
        domain,
        row: None,
    }
    .into()
}

fn validate_options(options: &MissingOptions) -> Result<(), Error> {
    if !(options.gamma_floor.is_finite() && (0.0..1.0).contains(&options.gamma_floor)) {
        return Err(domain("gamma_floor", options.gamma_floor, "[0, 1)"));
    }
    if !(options.propensity_ceiling > 0.0 && options.propensity_ceiling <= 1.0) {
        return Err(domain(
            "propensity_ceiling",
            options.propensity_ceiling,
            "(0, 1]",
        ));
    }
    if !(options.tolerance.is_finite() && options.tolerance > 0.0) {
        return Err(domain("tolerance", options.tolerance, "(0, inf)"));
    }
    if options.max_iterations == 0 {
        return Err(domain("max_iterations", 0.0, "[1, inf)"));
    }
    Ok(())
}

/// Checks that both arms have units.
fn check_arms(data: &MissingData<'_>) -> Result<(), Error> {
    let treated = data.treated.iter().filter(|&&a| a).count();
    if treated == 0 {
        return Err(Error::NoTreated);
    }
    if treated == data.len() {
        return Err(Error::NoComparison);
    }
    Ok(())
}

/// The estimate before the fold count and the nuisance are attached.
struct Core {
    estimate: Estimate,
    influence: Vec<f64>,
    counts: [usize; 4],
    propensity_range: (f64, f64),
    gamma_ranges: [Option<(f64, f64)>; 2],
}

impl Core {
    fn finish(self, folds: usize, nuisance: MissingNuisance) -> Result<MissingAtt, Error> {
        if self.influence.iter().any(|value| !value.is_finite()) {
            return Err(Error::NotFinite);
        }
        let rows = (0..self.influence.len())
            .map(|i| check::fits_u32("units", i))
            .collect::<Result<_, _>>()?;
        let influence = Influence::over_rows(self.influence, rows, 1, ClusterRule::NoSmallSample)?;
        Ok(MissingAtt {
            estimate: self.estimate,
            influence,
            n_treated: self.counts[0],
            n_control: self.counts[1],
            observed_pre_treated: self.counts[2],
            observed_pre_control: self.counts[3],
            propensity_range: self.propensity_range,
            gamma_treated_range: self.gamma_ranges[0],
            gamma_control_range: self.gamma_ranges[1],
            folds,
            nuisance,
        })
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

fn widen(range: &mut Option<(f64, f64)>, value: f64) {
    *range = Some(range.map_or((value, value), |(low, high)| {
        (low.min(value), high.max(value))
    }));
}

/// The estimator of equations 12 and 13 on given predictions, with the
/// influence function and its standard error.
fn estimate_core(
    data: &MissingData<'_>,
    p: &MissingPredictions<'_>,
    options: &MissingOptions,
) -> Result<Core, Error> {
    let n = data.treated.len();
    let nested = match (options.assumption, p.nested_control) {
        (MissingAssumption::MarX, None) => None,
        (MissingAssumption::MarXY1, Some(eta)) => Some(eta),
        _ => return Err(Error::NestedRegression),
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
    for &(what, values) in &columns {
        epistat_core::check::same_length(what, n, values.len())?;
        epistat_core::check::finite(what, values)?;
    }

    let mut propensity_range = None;
    let mut gamma_ranges = [None; 2];
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
                    check_gamma(i, gamma, options)?;
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
            if !(pi > 0.0 && pi < options.propensity_ceiling) {
                return Err(Error::PropensityRange {
                    unit: i,
                    value: pi,
                    ceiling: options.propensity_ceiling,
                });
            }
            widen(&mut propensity_range, pi);
            let m00 = p.pre_control[i];
            let augmentation = match y0 {
                Some(y0) => {
                    counts[3] += 1;
                    let gamma = p.gamma_control[i];
                    check_gamma(i, gamma, options)?;
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
    let se = match data.clusters() {
        None => standard_error_from_influence(&influence),
        Some(clusters) => clustered_standard_error(&influence, clusters),
    };
    let ci = normal_interval(att, se, options.level.confidence_value());
    let Some(propensity_range) = propensity_range else {
        return Err(Error::NoComparison);
    };
    Ok(Core {
        estimate: estimate_from(att, se, ci, options.level),
        influence,
        counts,
        propensity_range,
        gamma_ranges,
    })
}

fn check_gamma(unit: usize, gamma: f64, options: &MissingOptions) -> Result<(), Error> {
    if gamma > options.gamma_floor && gamma > 0.0 && gamma <= 1.0 {
        Ok(())
    } else {
        Err(Error::MissingnessRange {
            unit,
            value: gamma,
            floor: options.gamma_floor,
        })
    }
}

#[cfg(test)]
mod tests;
