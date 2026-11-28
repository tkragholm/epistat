//! A match described as data, and one function that runs it.
//!
//! [`MatchJob`] changes its type with every strategy and
//! constraint added, which suits Rust and not a caller that reads its options
//! at run time, such as a binding to R or Python. A [`MatchSpec`] names every
//! option as a plain value, with serde, and [`run_spec`] checks it and builds
//! the typed job. The spec is a layer over the job and adds no matching of
//! its own: a spec and the job written out by hand give the same outcome.
//!
//! Invalid options are errors here. The typed job reports an invalid
//! configuration as a count in the diagnostics of an empty outcome, and
//! [`run_spec`] turns that into a [`Error`] as well, so a caller never
//! mistakes it for a match that found nothing.
//!
//! # Algorithm
//!
//! [`run_spec`] reads the method, the distance, the estimator and the
//! constraints from the spec, builds the distance configuration from the
//! covariates when the spec names a model or a covariance, and calls the
//! matcher of the method. Each option is read by the methods that use it, and
//! an option that the chosen method does not read is an error. The names of
//! the options follow those of `matchit()`.
//!
//! # References
//!
//! 1. Ho DE, Imai K, King G, Stuart EA. MatchIt: nonparametric preprocessing
//!    for parametric causal inference. Journal of Statistical Software
//!    2011;42(8).

mod wiring;

pub use wiring::{CardinalityMatch, CardinalitySpec, FitFuncSpec, GeneticSpec, LossSpec};

use crate::Error;
use crate::TrialMatching;
use crate::constraints::{
    AntiExact, Chained, ExactMatchConstraint, MustBeAlive, MustBeResident, caliper_on_field,
};
use crate::matching::{
    Assignment, CbpsConfig, CbpsMethod, CemBinning, CemRule, CemVariable, CovariateEncodingConfig,
    CovariateRecord, DeterministicSelection, DistanceConfig, ElasticNetCvConfig,
    ElasticNetLogisticConfig, EuclideanPreparationConfig, EuclideanScale, FineBalance,
    FullMatchConfig, GeneticOutcome, LambdaRule, LogisticRegressionConfig,
    MahalanobisCovarianceStrategy, MahalanobisPreparationConfig, MahalanobisTransform, MatchJob,
    MatchOrder, MatchingRecord, MissingValuePolicy, NearestBirthDateSelection, NearestSelection,
    OptimalRequest, PairDistanceMatrix, PropensityEstimator, PropensityLink, PropensityScoreConfig,
    PropensityScoreEstimate, PropensityScoreOutputScale, QuickMatchOptions, RandomSelection,
    Retain, RoleIndexedRecord, Rounds, ScoreTable, SeedMethod, SelectionStrategy,
    SubclassReferenceGroup, SubclassedUnits, SubclassificationConfig, SubclassificationOutcome,
    VariableRatio, apply_transposed, cem_k2k_pairs, cem_strata, cem_subclasses,
    estimate_propensity_and_match, exact_match_subclasses, match_fine_balance, match_full,
    match_optimal, prepare_euclidean_distance_config, prepare_mahalanobis_distance_config,
    prepare_propensity_distance_config, quick_match_subclasses,
    subclassify_by_propensity_score_map,
};
use crate::role_transition::IdOrder;
use crate::role_transition::{ExposureDensityOutcome, exposure_density_sets};
use crate::trials::{TrialOptions, TrialOutcome, TrialPanel, match_trials, trial_seed};
use crate::types::reject_invalid;
use crate::types::{
    AgeLimitYears, BirthDateWindowDays, CommonSupport, DistanceCaliper, Estimand, MatchOutcome,
    MatchRatio, MatchingCriteria, RoleTransitionRecord,
};
use chrono::NaiveDate;
use epistat_linalg::cholesky::{Pivot, cholesky_lower};
use rapidhash::RapidHashMap;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Standard matching of a set of cases to a set of controls, or risk-set
/// matching within one cohort.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchMode {
    /// Fixed cases and controls.
    #[default]
    Standard,
    /// Each case takes controls from the units still at risk on its index
    /// date. The units are one cohort, and a unit with a transition date is a
    /// case.
    RiskSet,
    /// Exposure density sampling: risk-set matching in which the transition
    /// date is the date of exposure. The output carries the index date of
    /// each set and the date on which each control is later exposed.
    ExposureDensity,
    /// Sequential trial emulation: in each period at which people are
    /// eligible, those who start treatment are matched to eligible people who
    /// do not, by risk-set matching on the date of the period. The output
    /// carries the sets of every trial, the start of follow-up of each person
    /// and the period at which the person deviates from the assigned strategy.
    SequentialTrials,
}

/// How controls are chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchMethod {
    /// A random eligible control, from the spec's seed.
    #[default]
    Random,
    /// The eligible control that comes first in the data.
    LowestIndex,
    /// The eligible control whose birth date is nearest the case's.
    NearestBirthDate,
    /// The nearest control by the distance, as `matchit(method = "nearest")`.
    /// Standard mode only.
    Nearest,
    /// Everyone in a shared stratum of the `exact` keys, with weights, as
    /// `matchit(method = "exact")`. Standard mode only.
    Exact,
    /// Subclasses of the propensity score or another score, as
    /// `matchit(method = "subclass")`. Standard mode only.
    Subclass,
    /// Coarsened exact matching on the variables of `cem`, with weights, or
    /// one to one within strata when `cem.k2k` is set, as
    /// `matchit(method = "cem")`. Standard mode only.
    Cem,
    /// Generalised full matching on the distance, with weights, as
    /// `matchit(method = "quick")`. Standard mode only.
    Quick,
    /// Pairs that minimise the total distance, as
    /// `matchit(method = "optimal")`. Standard mode only. The cases are the
    /// focal group.
    Optimal,
    /// Subclasses of one case with several controls or the reverse that
    /// minimise the total distance, as `matchit(method = "full")`. Standard
    /// mode only. The cases are the focal group.
    Full,
    /// The weights of a genetic search, then nearest neighbour matching on the
    /// weighted distance, as `matchit(method = "genetic")`. Standard mode only.
    /// The cases are the focal group.
    Genetic,
    /// The largest subset that satisfies balance constraints, selected by an
    /// integer program, as `matchit(method = "cardinality")`. Standard mode
    /// only.
    Cardinality,
}

/// The order in which cases take their controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssignmentSpec {
    /// Each case takes all its controls before the next case starts.
    #[default]
    Sequential,
    /// Every case takes a first control before any takes a second.
    Rounds,
}

impl From<AssignmentSpec> for Assignment {
    fn from(spec: AssignmentSpec) -> Self {
        match spec {
            AssignmentSpec::Sequential => Self::Sequential,
            AssignmentSpec::Rounds => Self::Rounds,
        }
    }
}

/// The order in which cases are matched, `m.order` of `matchit()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrderSpec {
    /// Largest distance score first.
    Largest,
    /// Smallest distance score first.
    Smallest,
    /// The order of the data.
    Data,
    /// An order drawn from the seed.
    Random,
    /// Nearest pair first.
    Closest,
    /// Farthest pair first.
    Farthest,
}

impl From<OrderSpec> for MatchOrder {
    fn from(spec: OrderSpec) -> Self {
        match spec {
            OrderSpec::Largest => Self::Largest,
            OrderSpec::Smallest => Self::Smallest,
            OrderSpec::Data => Self::Data,
            OrderSpec::Random => Self::Random,
            OrderSpec::Closest => Self::Closest,
            OrderSpec::Farthest => Self::Farthest,
        }
    }
}

/// What to do with a missing covariate value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MissingSpec {
    /// Stop with an error.
    #[default]
    Error,
    /// Impute the mean of a numeric covariate and give a categorical one a
    /// level for missing.
    Impute,
}

impl From<MissingSpec> for MissingValuePolicy {
    fn from(spec: MissingSpec) -> Self {
        match spec {
            MissingSpec::Error => Self::Error,
            MissingSpec::Impute => Self::Impute,
        }
    }
}

/// The link of the propensity model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkSpec {
    /// Logistic regression.
    #[default]
    Logit,
    /// Probit regression.
    Probit,
    /// Complementary log-log regression.
    Cloglog,
}

/// The scale of the propensity score the distance is taken on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScaleSpec {
    /// The probability.
    #[default]
    Probability,
    /// The linear predictor, `link = "linear.logit"` and its siblings.
    Linear,
}

/// Which cross-validated penalty is chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LambdaSpec {
    /// The penalty with the smallest cross-validated deviance.
    Min,
    /// The largest penalty within one standard error of the smallest, which
    /// MatchIt uses.
    #[default]
    OneStandardError,
}

/// How the propensity model is fitted.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum EstimatorSpec {
    /// A generalised linear model with no penalty, as R's `glm()`.
    #[default]
    Glm,
    /// An elastic net at a given penalty.
    ElasticNet {
        /// Mixing of lasso and ridge, 1 for lasso and 0 for ridge.
        alpha: f64,
        /// Penalty strength.
        lambda: f64,
        /// Scale the columns to unit standard deviation before fitting. False
        /// by default, unlike the cross-validated elastic net.
        #[serde(default)]
        standardize: bool,
    },
    /// An elastic net with the penalty chosen by cross-validation, folds drawn
    /// from the spec's seed.
    ElasticNetCv {
        /// Mixing of lasso and ridge, 1 for lasso and 0 for ridge.
        alpha: f64,
        /// Number of folds, at least 3.
        #[serde(default = "default_folds")]
        nfolds: usize,
        /// Number of penalties tried.
        #[serde(default = "default_nlambda")]
        nlambda: usize,
        /// Smallest penalty as a fraction of the largest, `lambda.min.ratio`
        /// of `glmnet`. The default is 1e-2 when there are fewer units than
        /// covariates and 1e-4 otherwise.
        #[serde(default)]
        lambda_min_ratio: Option<f64>,
        /// The rule that picks the penalty.
        #[serde(default)]
        rule: LambdaSpec,
        /// Scale the columns to unit standard deviation before fitting, as
        /// MatchIt's `cv.glmnet` call does. True by default.
        #[serde(default = "default_true")]
        standardize: bool,
    },
    /// The covariate balancing propensity score, as MatchIt's
    /// `distance = "cbps"` with `CBPS::CBPS()`. The link of the distance is
    /// logit, and the model needs an intercept. The estimand whose balance
    /// conditions the fit satisfies is the spec's `estimand`, which must be
    /// the ATT or the ATE.
    Cbps {
        /// Score and balance conditions together, CBPS's `method = "over"`.
        /// `false` is `method = "exact"`, the balance conditions alone.
        #[serde(default = "default_true")]
        over: bool,
    },
}

const fn default_folds() -> usize {
    10
}

const fn default_nlambda() -> usize {
    100
}

const fn default_true() -> bool {
    true
}

/// The covariates of a distance and how they are read.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CovariatesSpec {
    /// The covariates to use. Empty means every covariate the units carry.
    pub names: Vec<String>,
    /// What to do with a missing value.
    pub missing: MissingSpec,
}

impl CovariatesSpec {
    fn encoding(&self, intercept: bool) -> CovariateEncodingConfig {
        CovariateEncodingConfig {
            covariate_keys: self.names.clone(),
            include_intercept: intercept,
            missing_value_policy: self.missing.into(),
            ..CovariateEncodingConfig::default()
        }
    }
}

/// The distance between an case and a control.
///
/// The `caliper` of each variant is the widest distance a pair may have, in
/// the units of the distance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DistanceSpec {
    /// The difference of the birth dates in days. With no caliper the birth
    /// date window of the spec is the only limit.
    Date {
        /// Widest difference in days.
        #[serde(default)]
        caliper: Option<f64>,
    },
    /// The propensity score, fitted from the covariates.
    Propensity {
        /// The covariates of the model.
        #[serde(default)]
        covariates: CovariatesSpec,
        /// The link of the model.
        #[serde(default)]
        link: LinkSpec,
        /// How the model is fitted.
        #[serde(default)]
        estimator: EstimatorSpec,
        /// The scale of the score.
        #[serde(default)]
        scale: ScaleSpec,
        /// Sampling weights by unit id, entered as prior weights in the fit.
        #[serde(default)]
        sampling_weights: Option<BTreeMap<String, f64>>,
        /// Widest difference of the score.
        #[serde(default)]
        caliper: Option<f64>,
    },
    /// MatchIt's Mahalanobis distance of the covariates.
    Mahalanobis {
        /// The covariates of the distance.
        #[serde(default)]
        covariates: CovariatesSpec,
        /// The covariance matrix. The default is the pooled within-group
        /// covariance MatchIt uses.
        #[serde(default)]
        covariance: CovarianceSpec,
        /// Rank-transform the covariates first.
        #[serde(default)]
        transform: TransformSpec,
        /// Widest distance.
        #[serde(default)]
        caliper: Option<f64>,
    },
    /// The Euclidean distance of the covariates.
    Euclidean {
        /// The covariates of the distance.
        #[serde(default)]
        covariates: CovariatesSpec,
        /// Divide each column by its pooled within-group standard deviation.
        #[serde(default)]
        scaled: bool,
        /// Widest distance.
        #[serde(default)]
        caliper: Option<f64>,
    },
    /// A score for every unit, given by the caller, as `distance` given as a
    /// numeric vector in `matchit()`.
    Scores {
        /// The score of each unit by id.
        scores: BTreeMap<String, f64>,
        /// Widest difference of the score.
        #[serde(default)]
        caliper: Option<f64>,
    },
    /// A distance for every pair, given by the caller, as `distance` given as
    /// a matrix in `matchit()`. Rows are cases and columns controls. An
    /// entry of `None` forbids the pair, as an infinite entry does in R.
    Matrix {
        /// Ids of the rows.
        case_ids: Vec<String>,
        /// Ids of the columns.
        control_ids: Vec<String>,
        /// The distances, one row per case.
        values: Vec<Vec<Option<f64>>>,
        /// Widest distance.
        #[serde(default)]
        caliper: Option<f64>,
    },
}

impl Default for DistanceSpec {
    fn default() -> Self {
        Self::Date { caliper: None }
    }
}

/// The covariance matrix of the Mahalanobis distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CovarianceSpec {
    /// Pooled within-group covariance over the number of units less two.
    #[default]
    PooledByDegreesOfFreedom,
    /// The mean of the two groups' covariance matrices.
    PooledWithinGroups,
    /// Covariance of the whole sample.
    FullSample,
}

/// The transformation of the covariates before the Mahalanobis distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransformSpec {
    /// No transformation.
    #[default]
    Raw,
    /// Ranks.
    Rank,
    /// Ranks with the covariance rescaled for ties, Rosenbaum's robust
    /// Mahalanobis distance.
    RankTieAdjusted,
}

/// The part of the common support to trim, `discard` of `matchit()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiscardSpec {
    /// Cases outside the overlap of the scores.
    Treated,
    /// Controls outside the overlap of the scores.
    Control,
    /// Both.
    Both,
}

impl From<DiscardSpec> for CommonSupport {
    fn from(spec: DiscardSpec) -> Self {
        match spec {
            DiscardSpec::Treated => Self::Treated,
            DiscardSpec::Control => Self::Control,
            DiscardSpec::Both => Self::Both,
        }
    }
}

/// A variable number of controls per case, `ratio`, `min.controls` and
/// `max.controls` of `matchit()`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VariableRatioSpec {
    /// The mean number of controls per case.
    pub average: f64,
    /// The fewest controls an case gets.
    pub min: usize,
    /// The most controls an case gets.
    pub max: usize,
}

/// Which calipers are read in standard deviations, `std.caliper` of
/// `matchit()`.
///
/// A single flag applies to the distance caliper and every caliper on a field.
/// The other form says it for each caliper by name, as the vector of
/// `std.caliper` does. A caliper that is not named is in its own units.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StdCaliperSpec {
    /// All calipers or none.
    All(bool),
    /// Each caliper by name.
    Each(StdCaliperEach),
}

impl Default for StdCaliperSpec {
    fn default() -> Self {
        Self::All(false)
    }
}

impl From<bool> for StdCaliperSpec {
    fn from(all: bool) -> Self {
        Self::All(all)
    }
}

/// The calipers read in standard deviations, by name.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StdCaliperEach {
    /// The caliper of the distance.
    pub distance: bool,
    /// The calipers on fields, by field name. Each name must have an entry in
    /// the spec's `calipers`.
    pub fields: BTreeMap<String, bool>,
}

impl StdCaliperSpec {
    /// Whether the distance caliper is in standard deviations.
    #[must_use]
    pub const fn distance(&self) -> bool {
        match self {
            Self::All(all) => *all,
            Self::Each(each) => each.distance,
        }
    }

    /// Whether the caliper on a field is in standard deviations.
    #[must_use]
    pub fn field(&self, name: &str) -> bool {
        match self {
            Self::All(all) => *all,
            Self::Each(each) => each.fields.get(name).copied().unwrap_or(false),
        }
    }

    /// Whether any caliper is in standard deviations.
    #[must_use]
    pub fn any(&self) -> bool {
        match self {
            Self::All(all) => *all,
            Self::Each(each) => each.distance || each.fields.values().any(|set| *set),
        }
    }
}

/// The options of the nearest method.
///
/// Options that only the nearest method has are fields of this struct.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NearestSpec {
    /// The order in which cases are matched. The default is `matchit()`'s:
    /// largest score first for a score distance, and the order of the data
    /// otherwise.
    pub m_order: Option<OrderSpec>,
    /// Rank the controls by this distance, usually Mahalanobis, and use the
    /// spec's distance as a caliper and for the order of cases, as
    /// `mahvars` does.
    pub mahvars: Option<DistanceSpec>,
    /// A variable number of controls per case in place of `ratio`. Needs a
    /// score distance and no fallback ratios.
    pub variable_ratio: Option<VariableRatioSpec>,
}

/// How a numeric variable is cut into bins.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum BinningSpec {
    /// A number of bins from a rule.
    Rule {
        /// The rule.
        rule: BinRuleSpec,
    },
    /// This many bins of equal width between the smallest and largest value.
    Bins {
        /// Number of bins.
        n: usize,
    },
    /// Cut points. A value on a cut point goes to the bin above.
    Cutpoints {
        /// The cut points.
        values: Vec<f64>,
    },
    /// Groups of equal share, cut at the quantiles of all values.
    Quantiles {
        /// Number of groups, at least 2.
        n: usize,
    },
    /// No binning: units must share the value.
    Exact,
}

impl Default for BinningSpec {
    fn default() -> Self {
        Self::Rule {
            rule: BinRuleSpec::default(),
        }
    }
}

impl From<&BinningSpec> for CemBinning {
    fn from(spec: &BinningSpec) -> Self {
        match spec {
            BinningSpec::Rule { rule } => Self::Rule(match rule {
                BinRuleSpec::Sturges => CemRule::Sturges,
                BinRuleSpec::FreedmanDiaconis => CemRule::FreedmanDiaconis,
                BinRuleSpec::Scott => CemRule::Scott,
            }),
            BinningSpec::Bins { n } => Self::Bins(*n),
            BinningSpec::Cutpoints { values } => Self::Cutpoints(values.clone()),
            BinningSpec::Quantiles { n } => Self::Quantiles(*n),
            BinningSpec::Exact => Self::Exact,
        }
    }
}

/// A rule that gives a number of bins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinRuleSpec {
    /// Sturges.
    #[default]
    Sturges,
    /// Freedman and Diaconis.
    FreedmanDiaconis,
    /// Scott.
    Scott,
}

/// A variable of coarsened exact matching.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CemVariableSpec {
    /// A numeric field of the units, binned. The units' numeric fields hold
    /// the values.
    Numeric {
        /// The name of the field.
        name: String,
        /// How it is binned. The default is Sturges.
        #[serde(default)]
        binning: BinningSpec,
    },
    /// A strata key of the units, whose values may be merged into groups.
    Categorical {
        /// The strata key.
        name: String,
        /// Groups of values that count as the first value of the group.
        #[serde(default)]
        groups: Vec<Vec<String>>,
    },
}

/// The pairing of k:k matching within the strata of coarsened exact matching.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct K2kSpec {
    /// The distance that ranks the controls of a stratum. Without one the
    /// controls are taken in an order drawn from the seed. A caliper is not
    /// used.
    pub distance: Option<DistanceSpec>,
    /// The order in which cases take their control. The default is the
    /// order of the data.
    pub m_order: Option<OrderSpec>,
}

/// The options of coarsened exact matching.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CemSpec {
    /// The variables to match on.
    pub variables: Vec<CemVariableSpec>,
    /// Pair the units of each stratum one to one.
    pub k2k: Option<K2kSpec>,
}

/// The options of the subclass method.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SubclassSpec {
    /// Number of subclasses.
    pub n: usize,
    /// Fewest units of each group a subclass may hold.
    pub min_n: usize,
    /// The group whose scores give the cutpoints. The default follows the
    /// estimand.
    pub reference: Option<ReferenceSpec>,
}

impl Default for SubclassSpec {
    fn default() -> Self {
        let config = SubclassificationConfig::default();
        Self {
            n: config.subclasses,
            min_n: config.min_n,
            reference: None,
        }
    }
}

/// The group whose scores give the subclass cutpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceSpec {
    /// The cases.
    Cases,
    /// The controls.
    Controls,
    /// Both groups together.
    Pooled,
}

impl From<ReferenceSpec> for SubclassReferenceGroup {
    fn from(spec: ReferenceSpec) -> Self {
        match spec {
            ReferenceSpec::Cases => Self::Cases,
            ReferenceSpec::Controls => Self::Controls,
            ReferenceSpec::Pooled => Self::Pooled,
        }
    }
}

/// The options of the quick method.
///
/// The caliper is the one of the distance, the widest distance between two
/// units of a subclass.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct QuickSpec {
    /// Fewest cases in a subclass. `min_anchors` in 0.2.
    #[serde(alias = "min_anchors")]
    pub min_cases: usize,
    /// Fewest controls in a subclass. `min_candidates` in 0.2.
    #[serde(alias = "min_candidates")]
    pub min_controls: usize,
    /// Fewest units in a subclass. The default is the two minimums together.
    pub size: Option<usize>,
    /// The order in which units are tried as seeds.
    pub seed_method: SeedMethod,
}

impl Default for QuickSpec {
    fn default() -> Self {
        let options = QuickMatchOptions::default();
        Self {
            min_cases: options.min_cases,
            min_controls: options.min_controls,
            size: options.size,
            seed_method: options.seed_method,
        }
    }
}

/// Everything a match can be configured with.
///
/// The defaults are those of [`MatchJob`]: random choice among the controls
/// within 30 days of birth, one control per case, no replacement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each flag is one named option of matchit() or of the typed job"
)]
pub struct MatchSpec {
    /// Standard or risk-set matching.
    pub mode: MatchMode,
    /// How controls are chosen.
    pub method: MatchMethod,
    /// The distance between an case and a control.
    pub distance: DistanceSpec,
    /// Options of the nearest method.
    pub nearest: NearestSpec,
    /// Options of the subclass method.
    pub subclass: SubclassSpec,
    /// Options of coarsened exact matching.
    pub cem: CemSpec,
    /// Options of the quick method.
    pub quick: QuickSpec,
    /// Options of the full method.
    pub full: FullSpec,
    /// Options of the genetic method.
    pub genetic: GeneticSpec,
    /// Options of the cardinality method.
    pub cardinality: CardinalitySpec,
    /// Fine, near-fine or refined balance on nominal keys. Optimal method only.
    pub fine_balance: Option<FineBalance>,
    /// Options of [`MatchMode::SequentialTrials`]. The options of risk-set
    /// matching (ratio, exact, birth window, age limit, alive, resident and
    /// the rest) apply inside each trial. Every trial draws with a seed
    /// derived from `seed` and the period of the trial.
    pub trials: TrialOptions,
    /// How cases share the controls when each takes more than one. The
    /// default is sequential, and rounds for the nearest method as in
    /// `matchit()`. Risk-set matching is always sequential.
    pub assignment: Option<AssignmentSpec>,
    /// Controls per case.
    pub ratio: usize,
    /// Descending ratios to fall back to when an case finds fewer controls
    /// than the ratio. Not available with the nearest method or rounds.
    pub ratio_fallback: Vec<usize>,
    /// Let a control serve several cases.
    pub replace: bool,
    /// Widest difference in days between the birth dates of a pair.
    pub birth_window_days: i32,
    /// Calipers on numeric fields of the units, by field name, as the named
    /// calipers of `matchit()`.
    pub calipers: BTreeMap<String, f64>,
    /// Read calipers in standard deviations, as `std.caliper` does: the
    /// distance caliper in standard deviations of the distance, and a caliper
    /// on a field in standard deviations of the field over the units that remain
    /// after `discard`, with `reestimate` as well. One flag covers all
    /// calipers, or each is named. Nearest method only.
    pub std_caliper: StdCaliperSpec,
    /// Trim the units outside the common support of the scores before
    /// matching, `discard` of `matchit()`. Needs a propensity score or a score
    /// of your own as distance, and one of the random, lowest_index,
    /// nearest_birth_date, nearest, subclass, full, optimal, quick and genetic
    /// methods. The exact and cem methods do not read it. Subclass cutpoints
    /// are taken over the reference group including the trimmed units, as
    /// `matchit()` does, except with `reestimate`, where they are taken over
    /// the kept units because `matchit()` mixes two fits there.
    pub discard: Option<DiscardSpec>,
    /// Fit the propensity model again on the units that remain after
    /// `discard`, `reestimate` of `matchit()`. Needs `discard` and a propensity
    /// distance.
    pub reestimate: bool,
    /// Strata keys on which a pair must agree, `exact` in `matchit()`.
    pub exact: Vec<String>,
    /// Strata keys on which a pair must differ, `antiexact` in `matchit()`.
    pub antiexact: Vec<String>,
    /// Require the same value of the sex key in a pair.
    pub match_sex: bool,
    /// The strata key that holds sex.
    pub sex_key: String,
    /// Strata key of which each control value may be used once.
    pub unique_by_key: Option<String>,
    /// Risk-set matching: the age in whole years below which a transition
    /// makes a case. The default is `None`, no age limit.
    pub age_limit_years: Option<u8>,
    /// Risk-set matching: controls must be alive on the case's index date.
    pub require_alive: bool,
    /// Risk-set matching: controls must be resident on the case's index date.
    pub require_resident: bool,
    /// The target estimand.
    pub estimand: Estimand,
    /// Seed of every random choice. There is no default: [`MatchSpec::validate`]
    /// refuses a spec without one, whatever its method.
    pub seed: Option<u64>,
}

impl Default for MatchSpec {
    fn default() -> Self {
        let criteria = MatchingCriteria::default();
        Self {
            mode: MatchMode::Standard,
            method: MatchMethod::Random,
            distance: DistanceSpec::default(),
            nearest: NearestSpec::default(),
            subclass: SubclassSpec::default(),
            cem: CemSpec::default(),
            quick: QuickSpec::default(),
            full: FullSpec::default(),
            genetic: GeneticSpec::default(),
            cardinality: CardinalitySpec::default(),
            fine_balance: None,
            trials: TrialOptions::default(),
            assignment: None,
            ratio: criteria.match_ratio,
            ratio_fallback: Vec::new(),
            replace: criteria.allow_replacement,
            birth_window_days: criteria.birth_date_window_days,
            calipers: BTreeMap::new(),
            std_caliper: StdCaliperSpec::default(),
            discard: None,
            reestimate: false,
            exact: Vec::new(),
            antiexact: Vec::new(),
            match_sex: false,
            sex_key: "sex".to_string(),
            unique_by_key: None,
            age_limit_years: None,
            require_alive: false,
            require_resident: false,
            estimand: criteria.estimand,
            seed: None,
        }
    }
}

/// Options of the full method, those of `optmatch::fullmatch()`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FullSpec {
    /// Fewest controls per case in a subclass. A number below 1 allows
    /// several cases per control.
    pub min_controls: f64,
    /// Most controls per case in a subclass. Absent for no limit.
    pub max_controls: Option<f64>,
    /// Fraction of the controls (positive) or cases (negative) to leave
    /// out. Not with `mean_controls`.
    pub omit_fraction: Option<f64>,
    /// Controls per case to use on average. Not with `omit_fraction`.
    pub mean_controls: Option<f64>,
}

impl Default for FullSpec {
    fn default() -> Self {
        Self {
            min_controls: 0.0,
            max_controls: None,
            omit_fraction: None,
            mean_controls: None,
        }
    }
}

/// The units a spec is run on.
///
/// New kinds of input may be added, so a `match` on it outside this crate
/// needs a wildcard arm:
///
/// ```compile_fail
/// use epistat_matching::{BaseRecord, SpecInput};
///
/// fn kind(input: SpecInput<'_, BaseRecord>) -> &'static str {
///     match input {
///         SpecInput::Standard { .. } => "standard",
///         SpecInput::RiskSet { .. } => "risk set",
///         SpecInput::Trials { .. } => "trials",
///     }
/// }
/// ```
///
/// With the wildcard arm it compiles:
///
/// ```
/// use epistat_matching::{BaseRecord, SpecInput};
///
/// fn kind(input: SpecInput<'_, BaseRecord>) -> &'static str {
///     match input {
///         SpecInput::Standard { .. } => "standard",
///         SpecInput::RiskSet { .. } => "risk set",
///         SpecInput::Trials { .. } => "trials",
///         _ => "other",
///     }
/// }
/// assert_eq!(kind(SpecInput::Standard { cases: &[], controls: &[] }), "standard");
/// ```
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum SpecInput<'a, R> {
    /// Cases and controls, for [`MatchMode::Standard`].
    Standard {
        /// The cases.
        cases: &'a [R],
        /// The controls.
        controls: &'a [R],
    },
    /// One cohort, for [`MatchMode::RiskSet`] and [`MatchMode::ExposureDensity`].
    RiskSet {
        /// The cohort. A unit with a transition date is a case on that date.
        cohort: &'a [RoleTransitionRecord<R>],
    },
    /// The people of a panel, for [`MatchMode::SequentialTrials`].
    Trials {
        /// One entry per person, with the periods in which the person is
        /// observed.
        panel: &'a TrialPanel<'a, R>,
    },
}

/// What a spec produced.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SpecOutput {
    /// Pairs of cases and controls.
    Pairs {
        /// The pairs and their diagnostics.
        outcome: Box<MatchOutcome>,
        /// The propensity model, when the distance was a fitted score.
        propensity: Option<Box<PropensityScoreEstimate>>,
    },
    /// Exposure density sets, from [`MatchMode::ExposureDensity`].
    ExposureDensity(Box<ExposureDensityOutcome>),
    /// The trials of a sequential trial emulation, from
    /// [`MatchMode::SequentialTrials`].
    Trials(Box<TrialOutcome>),
    /// Strata with weights, from the exact method and from coarsened exact
    /// matching without k:k pairing.
    Exact(Box<SubclassedUnits>),
    /// Subclasses with weights, from the quick method.
    Quick(Box<SubclassedUnits>),
    /// Subclasses with the weights of each estimand, from the full method.
    Full(Box<SubclassedUnits>),
    /// The pairs of genetic matching and the search that found their weights.
    /// The pairs are those of [`SpecOutput::Pairs`], so the outputs and
    /// balance functions that take a [`MatchOutcome`] take them.
    Genetic {
        /// The pairs and their diagnostics.
        outcome: Box<MatchOutcome>,
        /// The best weights of the search, their loss and the history.
        search: Box<GeneticOutcome>,
    },
    /// The selected units of cardinality or profile matching, with their
    /// weights and, when `mahvars` was given, their pairs.
    Cardinality(Box<CardinalityMatch>),
    /// Subclasses with weights, from the subclass method.
    Subclass {
        /// The subclasses and their weights.
        outcome: Box<SubclassificationOutcome>,
        /// The propensity model, when the scores were fitted.
        propensity: Option<Box<PropensityScoreEstimate>>,
    },
}

impl DistanceSpec {
    const fn caliper(&self) -> Option<f64> {
        match self {
            Self::Date { caliper }
            | Self::Propensity { caliper, .. }
            | Self::Mahalanobis { caliper, .. }
            | Self::Euclidean { caliper, .. }
            | Self::Scores { caliper, .. }
            | Self::Matrix { caliper, .. } => *caliper,
        }
    }

    /// Whether the distance is one number per unit, which cases can be
    /// ordered by.
    const fn is_score(&self) -> bool {
        matches!(
            self,
            Self::Date { .. } | Self::Propensity { .. } | Self::Scores { .. }
        )
    }

    /// Whether this is the default distance.
    fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

fn typed_caliper(
    option: &'static str,
    value: Option<f64>,
) -> Result<Option<DistanceCaliper>, Error> {
    value
        .map(|width| {
            DistanceCaliper::new(width).ok_or_else(|| {
                Error::option(
                    option,
                    format!(
                        "the caliper {width} is negative or not finite; use a width of 0 or more"
                    ),
                )
            })
        })
        .transpose()
}

impl MatchSpec {
    /// The seed of the spec.
    ///
    /// # Errors
    ///
    /// [`Error::Option`] naming `seed` when the spec has none.
    pub(crate) fn root_seed(&self) -> Result<u64, Error> {
        self.seed.ok_or_else(|| {
            Error::option(
                "seed",
                "the spec has no seed; give the seed of its random choices",
            )
        })
    }

    /// Check the options against each other, before any unit is read.
    ///
    /// [`run_spec`] does this first.
    ///
    /// # Errors
    ///
    /// A [`Error`] naming the option that is out of range or conflicts
    /// with another, and how to fix it.
    pub fn validate(&self) -> Result<(), Error> {
        self.root_seed()?;
        self.criteria().validate()?;
        for ratio in &self.ratio_fallback {
            if MatchRatio::new(*ratio).is_none() {
                return Err(Error::option(
                    "ratio_fallback",
                    "a fallback ratio is 0; use ratios of 1 or more",
                ));
            }
        }
        for (field, width) in &self.calipers {
            if !(width.is_finite() && *width >= 0.0) {
                return Err(Error::option(
                    "calipers",
                    format!("the caliper on {field} is {width}; use a width of 0 or more"),
                ));
            }
        }
        typed_caliper("distance", self.distance.caliper())?;
        if let DistanceSpec::Scores { scores, .. } = &self.distance
            && let Some((id, score)) = scores.iter().find(|(_, score)| !score.is_finite())
        {
            return Err(Error::option(
                "distance",
                format!("the score of {id} is {score}; scores must be finite numbers"),
            ));
        }
        if let Some(mahvars) = &self.nearest.mahvars {
            typed_caliper("nearest.mahvars", mahvars.caliper())?;
        }
        if self.subclass.n == 0 {
            return Err(Error::option(
                "subclass.n",
                "the number of subclasses is 0; use 1 or more",
            ));
        }

        if self.cem != CemSpec::default() && self.method != MatchMethod::Cem {
            return Err(Error::option(
                "cem",
                "the cem options belong to the cem method; set the method to cem or drop them",
            ));
        }
        if self.quick != QuickSpec::default() && self.method != MatchMethod::Quick {
            return Err(Error::option(
                "quick",
                "the size, minimums and seed method belong to the quick method; \
                 set the method to quick or drop them",
            ));
        }
        if self.full != FullSpec::default() && self.method != MatchMethod::Full {
            return Err(Error::option(
                "full",
                "min_controls, max_controls, omit_fraction and mean_controls belong to the \
                 full method; set the method to full or drop them",
            ));
        }
        if self.genetic != GeneticSpec::default() && self.method != MatchMethod::Genetic {
            return Err(Error::option(
                "genetic",
                "the genetic options belong to the genetic method; \
                 set the method to genetic or drop them",
            ));
        }
        if self.cardinality != CardinalitySpec::default() && self.method != MatchMethod::Cardinality
        {
            return Err(Error::option(
                "cardinality",
                "the cardinality options belong to the cardinality method; \
                 set the method to cardinality or drop them",
            ));
        }
        let nearest_options = self.nearest != NearestSpec::default() || self.std_caliper.any();
        if nearest_options && self.method != MatchMethod::Nearest {
            return Err(Error::option(
                "nearest",
                "m_order, mahvars, variable_ratio and std_caliper belong to the nearest method; \
                 set the method to nearest or drop them",
            ));
        }
        self.validate_support()?;
        self.validate_read_by_method()?;
        match self.method {
            MatchMethod::Exact => self.validate_exact()?,
            MatchMethod::Subclass => self.validate_subclass()?,
            MatchMethod::Cem => self.validate_cem()?,
            MatchMethod::Quick => self.validate_quick()?,
            MatchMethod::Optimal | MatchMethod::Full => self.validate_optimal()?,
            MatchMethod::Nearest => self.validate_nearest()?,
            MatchMethod::Genetic => self.validate_genetic()?,
            MatchMethod::Cardinality => self.validate_cardinality()?,
            MatchMethod::Random | MatchMethod::LowestIndex | MatchMethod::NearestBirthDate => {
                if self.assignment == Some(AssignmentSpec::Rounds)
                    && !self.ratio_fallback.is_empty()
                {
                    return Err(Error::option(
                        "assignment",
                        "rounds cannot be combined with ratio_fallback; \
                         use sequential assignment or drop the fallback",
                    ));
                }
            }
        }
        match self.mode {
            MatchMode::Standard => self.validate_standard(),
            MatchMode::RiskSet | MatchMode::ExposureDensity | MatchMode::SequentialTrials => {
                self.validate_risk_set()
            }
        }
    }

    /// Options that only some methods read. A method that does not read one
    /// would ignore it, so it is an error naming the option.
    fn validate_read_by_method(&self) -> Result<(), Error> {
        if self.trials != TrialOptions::default() && self.mode != MatchMode::SequentialTrials {
            return Err(Error::option(
                "trials",
                "the trial options belong to sequential trial emulation; \
                 set the mode to sequential_trials or drop them",
            ));
        }
        if self.fine_balance.is_some() && self.method != MatchMethod::Optimal {
            return Err(Error::option(
                "fine_balance",
                "fine balance belongs to the optimal method; set the method to optimal or drop it",
            ));
        }
        let method = match self.method {
            MatchMethod::Exact => "exact",
            MatchMethod::Subclass => "subclass",
            MatchMethod::Quick => "quick",
            MatchMethod::Cem => "cem",
            _ => return Ok(()),
        };
        let reads_estimand = self.method == MatchMethod::Subclass
            || (self.method == MatchMethod::Cem && self.cem.k2k.is_some());
        if self.method == MatchMethod::Subclass && self.distance.caliper().is_some() {
            return Err(Error::option(
                "distance",
                "the subclass method cuts the score into classes and takes no caliper; \
                 drop the caliper",
            ));
        }
        let unread: [(&'static str, bool); 2] = [
            (
                "birth_window_days",
                self.birth_window_days != MatchingCriteria::default().birth_date_window_days,
            ),
            (
                "estimand",
                !reads_estimand && self.estimand != MatchingCriteria::default().estimand,
            ),
        ];
        match unread.iter().find(|(_, set)| *set) {
            Some((option, _)) => Err(Error::option(
                option,
                format!("the {method} method does not use it; drop it"),
            )),
            None => Ok(()),
        }
    }

    fn criteria(&self) -> MatchingCriteria {
        MatchingCriteria {
            birth_date_window_days: self.birth_window_days,
            match_ratio: self.ratio,
            required_strata: self.exact.clone(),
            unique_by_key: self.unique_by_key.clone(),
            allow_replacement: self.replace,
            estimand: self.estimand,
            common_support: self.discard.map(Into::into),
        }
    }

    fn validate_nearest(&self) -> Result<(), Error> {
        if self.mode != MatchMode::Standard {
            return Err(Error::option(
                "method",
                "the nearest method ranks candidates by a distance and runs in standard mode; \
                 use random, lowest_index or nearest_birth_date for risk-set matching",
            ));
        }
        if !self.ratio_fallback.is_empty() {
            return Err(Error::option(
                "ratio_fallback",
                "the nearest method has no fallback ratios; drop them or use another method",
            ));
        }
        if let Some(variable) = self.nearest.variable_ratio {
            VariableRatio::new(variable.average, variable.min, variable.max)
                .map_err(|error| Error::option("nearest.variable_ratio", error.to_string()))?;
            if !self.distance.is_score() {
                return Err(Error::option(
                    "nearest.variable_ratio",
                    "a variable ratio is set by the scores of the units; use a propensity \
                     score or a score of your own as the distance",
                ));
            }
            if self.ratio != 1 {
                return Err(Error::option(
                    "ratio",
                    "a variable ratio sets the number of controls; drop the ratio",
                ));
            }
        }
        let assignment = self.assignment.unwrap_or(AssignmentSpec::Rounds);
        if matches!(
            self.nearest.m_order,
            Some(OrderSpec::Closest | OrderSpec::Farthest)
        ) && assignment == AssignmentSpec::Sequential
        {
            return Err(Error::option(
                "nearest.m_order",
                "closest and farthest need rounds; set assignment to rounds",
            ));
        }
        if matches!(
            self.nearest.m_order,
            Some(OrderSpec::Largest | OrderSpec::Smallest)
        ) && !self.distance.is_score()
        {
            return Err(Error::option(
                "nearest.m_order",
                "largest and smallest order the anchors by a score, which a Mahalanobis or \
                 Euclidean distance does not have; use data, random, closest or farthest",
            ));
        }
        if self.std_caliper.distance()
            && !self.distance.is_score()
            && self.distance.caliper().is_some()
        {
            return Err(Error::option(
                "std_caliper",
                "a caliper in standard deviations needs a score distance; \
                 give the caliper in the units of the distance",
            ));
        }
        Ok(())
    }

    fn validate_optimal(&self) -> Result<(), Error> {
        let name = if self.method == MatchMethod::Full {
            "full"
        } else {
            "optimal"
        };
        let conflicts: [(&'static str, bool); 4] = [
            ("replace", self.replace),
            ("ratio_fallback", !self.ratio_fallback.is_empty()),
            ("unique_by_key", self.unique_by_key.is_some()),
            ("assignment", self.assignment.is_some()),
        ];
        if let Some((option, _)) = conflicts.iter().find(|(_, set)| *set) {
            return Err(Error::option(
                option,
                format!("the {name} method does not use it; drop it"),
            ));
        }
        if self.method == MatchMethod::Full && self.ratio != 1 {
            return Err(Error::option(
                "ratio",
                "the full method sizes its subclasses by min_controls and max_controls; \
                 drop the ratio",
            ));
        }
        if self.full.omit_fraction.is_some() && self.full.mean_controls.is_some() {
            return Err(Error::option(
                "full",
                "omit_fraction and mean_controls cannot both be set",
            ));
        }
        Ok(())
    }

    fn validate_exact(&self) -> Result<(), Error> {
        if self.exact.is_empty() {
            return Err(Error::option(
                "exact",
                "the exact method needs the strata keys to match on; name at least one",
            ));
        }
        self.reject_pair_options("exact")
    }

    fn validate_cem(&self) -> Result<(), Error> {
        if self.mode != MatchMode::Standard {
            return Err(Error::option(
                "method",
                "coarsened exact matching needs anchors and candidates in standard mode",
            ));
        }
        if self.cem.variables.is_empty() {
            return Err(Error::option(
                "cem.variables",
                "coarsened exact matching needs the variables to match on; name at least one",
            ));
        }
        for variable in &self.cem.variables {
            let CemVariableSpec::Numeric { name, binning } = variable else {
                continue;
            };
            let problem = match binning {
                BinningSpec::Bins { n: 0 } => Some("0 bins; use 1 or more, or the exact binning"),
                BinningSpec::Quantiles { n } if *n < 2 => {
                    Some("fewer than 2 quantile groups; use 2 or more")
                }
                BinningSpec::Cutpoints { values }
                    if values.is_empty() || values.iter().any(|v| !v.is_finite()) =>
                {
                    Some("cut points must be a non-empty list of finite numbers")
                }
                _ => None,
            };
            if let Some(problem) = problem {
                return Err(Error::option(
                    "cem.variables",
                    format!("the binning of {name} has {problem}"),
                ));
            }
        }
        if !self.exact.is_empty() {
            return Err(Error::option(
                "exact",
                "coarsened exact matching takes its variables from cem.variables; drop the keys",
            ));
        }
        if let Some(k2k) = &self.cem.k2k {
            if matches!(k2k.m_order, Some(OrderSpec::Largest | OrderSpec::Smallest)) {
                return Err(Error::option(
                    "cem.k2k.m_order",
                    "largest and smallest need a score; use data, random, closest or farthest",
                ));
            }
            if let Some(distance) = &k2k.distance {
                if distance.caliper().is_some() {
                    return Err(Error::option(
                        "cem.k2k.distance",
                        "k:k matching pairs within strata and takes no caliper; drop it",
                    ));
                }
                if matches!(distance, DistanceSpec::Date { .. }) {
                    return Err(Error::option(
                        "cem.k2k.distance",
                        "give a covariate distance, a score or a matrix",
                    ));
                }
            }
        }
        self.reject_pair_options("cem")
    }

    fn validate_subclass(&self) -> Result<(), Error> {
        if !matches!(
            self.distance,
            DistanceSpec::Propensity { .. } | DistanceSpec::Scores { .. }
        ) {
            return Err(Error::option(
                "distance",
                "the subclass method needs a propensity score or a score of your own; \
                 set the distance to propensity or scores",
            ));
        }
        if !self.exact.is_empty() {
            return Err(Error::option(
                "exact",
                "the subclass method does not take exact keys; drop them",
            ));
        }
        self.reject_pair_options("subclass")
    }

    fn validate_quick(&self) -> Result<(), Error> {
        if matches!(self.distance, DistanceSpec::Date { .. }) {
            return Err(Error::option(
                "distance",
                "the quick method needs a propensity score, a score of your own, or a \
                 Euclidean or Mahalanobis distance; set the distance",
            ));
        }
        if self.std_caliper.any() {
            return Err(Error::option(
                "std_caliper",
                "the quick method takes its caliper in the units of the distance; drop it",
            ));
        }
        self.reject_pair_options("quick")
    }

    /// The options that pair formation reads, which a method that forms no
    /// pairs would silently ignore.
    fn reject_pair_options(&self, method: &str) -> Result<(), Error> {
        let conflicts: [(&'static str, bool); 8] = [
            ("ratio", self.ratio != 1),
            ("ratio_fallback", !self.ratio_fallback.is_empty()),
            ("replace", self.replace),
            ("calipers", !self.calipers.is_empty()),
            ("antiexact", !self.antiexact.is_empty()),
            ("match_sex", self.match_sex),
            ("unique_by_key", self.unique_by_key.is_some()),
            ("assignment", self.assignment.is_some()),
        ];
        if let Some((option, _)) = conflicts.iter().find(|(_, set)| *set) {
            return Err(Error::option(
                option,
                format!("the {method} method forms no pairs and does not use it; drop it"),
            ));
        }
        if method == "exact" && !self.distance.is_default() {
            return Err(Error::option(
                "distance",
                "the exact method does not use a distance; drop it",
            ));
        }
        if method == "cem" && !self.distance.is_default() {
            return Err(Error::option(
                "distance",
                "coarsened exact matching takes its distance from cem.k2k.distance; drop it",
            ));
        }
        Ok(())
    }

    /// The options on calipers in standard deviations and on common support.
    fn validate_support(&self) -> Result<(), Error> {
        if let StdCaliperSpec::Each(each) = &self.std_caliper {
            let unnamed = each
                .fields
                .keys()
                .find(|name| !self.calipers.contains_key(*name));
            if let Some(name) = unnamed {
                return Err(Error::option(
                    "std_caliper",
                    format!("{name} has no caliper in calipers; add one or drop the entry"),
                ));
            }
        }
        if self.reestimate && self.discard.is_none() {
            return Err(Error::option(
                "reestimate",
                "the model is fitted again on the units that discard leaves; set discard",
            ));
        }
        if self.discard.is_none() {
            return Ok(());
        }
        let supported = self.mode == MatchMode::RiskSet
            || (self.mode == MatchMode::Standard
                && matches!(
                    self.method,
                    MatchMethod::Random
                        | MatchMethod::LowestIndex
                        | MatchMethod::NearestBirthDate
                        | MatchMethod::Nearest
                        | MatchMethod::Subclass
                        | MatchMethod::Full
                        | MatchMethod::Optimal
                        | MatchMethod::Quick
                        | MatchMethod::Genetic
                ));
        if !supported {
            return Err(Error::option(
                "discard",
                "common support is trimmed in risk-set mode and in standard mode with the \
                 random, lowest_index, nearest_birth_date, nearest, subclass, full, optimal, \
                 quick and genetic methods; drop it",
            ));
        }
        let score_distance = match self.mode {
            MatchMode::RiskSet => matches!(self.distance, DistanceSpec::Scores { .. }),
            _ => matches!(
                self.distance,
                DistanceSpec::Propensity { .. } | DistanceSpec::Scores { .. }
            ),
        };
        if !score_distance {
            return Err(Error::option(
                "discard",
                "common support is read from a propensity score or a score of your own; \
                 set the distance to propensity or scores",
            ));
        }
        if self.reestimate && !matches!(self.distance, DistanceSpec::Propensity { .. }) {
            return Err(Error::option(
                "reestimate",
                "only a propensity model can be fitted again; set the distance to propensity",
            ));
        }
        Ok(())
    }

    fn validate_standard(&self) -> Result<(), Error> {
        for (option, set) in [
            ("require_alive", self.require_alive),
            ("require_resident", self.require_resident),
            ("age_limit_years", self.age_limit_years.is_some()),
        ] {
            if set {
                return Err(Error::option(
                    option,
                    "this option is read by risk-set matching only; \
                     set the mode to risk_set or drop it",
                ));
            }
        }
        Ok(())
    }

    fn validate_risk_set(&self) -> Result<(), Error> {
        if !matches!(
            self.method,
            MatchMethod::Random | MatchMethod::LowestIndex | MatchMethod::NearestBirthDate
        ) {
            return Err(risk_set_method_error());
        }
        if !matches!(
            self.distance,
            DistanceSpec::Date { .. } | DistanceSpec::Scores { .. }
        ) {
            return Err(Error::option(
                "distance",
                "risk-set matching takes a birth date or a score of your own as distance; \
                 covariate distances need anchors and candidates in standard mode",
            ));
        }
        if self
            .assignment
            .is_some_and(|a| a != AssignmentSpec::Sequential)
        {
            return Err(Error::option(
                "assignment",
                "risk-set matching assigns sequentially; drop the option",
            ));
        }
        if self
            .age_limit_years
            .is_some_and(|years| AgeLimitYears::new(years).is_none())
        {
            return Err(Error::option(
                "age_limit_years",
                "the age limit is 0; use 1 year or more",
            ));
        }
        Ok(())
    }
}

/// The propensity model of a propensity distance, from its spec.
fn propensity_config(
    spec: &DistanceSpec,
    estimand: Estimand,
    seed: u64,
    caliper: Option<DistanceCaliper>,
    reestimate: bool,
) -> Result<PropensityScoreConfig, Error> {
    let DistanceSpec::Propensity {
        covariates,
        link,
        estimator,
        scale,
        sampling_weights,
        ..
    } = spec
    else {
        return Err(Error::option(
            "distance",
            "reestimate needs a propensity score distance",
        ));
    };
    let estimator = match *estimator {
        EstimatorSpec::Glm => PropensityEstimator::GlmLogit,
        EstimatorSpec::ElasticNet {
            alpha,
            lambda,
            standardize,
        } => PropensityEstimator::ElasticNetLogit(ElasticNetLogisticConfig {
            lambda,
            alpha,
            standardize,
            ..ElasticNetLogisticConfig::default()
        }),
        EstimatorSpec::Cbps { over } => PropensityEstimator::Cbps(CbpsConfig {
            estimand: match estimand {
                Estimand::Att => Estimand::Att,
                Estimand::Ate => Estimand::Ate,
                _ => {
                    return Err(Error::option(
                        "estimand",
                        "the covariate balancing propensity score is defined for the \
                             ATT and the ATE only",
                    ));
                }
            },
            method: if over {
                CbpsMethod::Over
            } else {
                CbpsMethod::Exact
            },
            ..CbpsConfig::default()
        }),
        EstimatorSpec::ElasticNetCv {
            alpha,
            nfolds,
            nlambda,
            lambda_min_ratio,
            rule,
            standardize,
        } => PropensityEstimator::ElasticNetCv(ElasticNetCvConfig {
            alpha,
            nfolds,
            nlambda,
            lambda_min_ratio,
            rule: match rule {
                LambdaSpec::Min => LambdaRule::Min,
                LambdaSpec::OneStandardError => LambdaRule::OneStandardError,
            },
            standardize,
            ..ElasticNetCvConfig::new(seed)
        }),
    };
    let logistic = LogisticRegressionConfig {
        link: match link {
            LinkSpec::Logit => PropensityLink::Logit,
            LinkSpec::Probit => PropensityLink::Probit,
            LinkSpec::Cloglog => PropensityLink::Cloglog,
        },
        ..LogisticRegressionConfig::default()
    };
    Ok(PropensityScoreConfig {
        encoding: covariates.encoding(true),
        estimator,
        logistic,
        output_scale: match scale {
            ScaleSpec::Probability => PropensityScoreOutputScale::Probability,
            ScaleSpec::Linear => PropensityScoreOutputScale::LinearPredictor,
        },
        caliper,
        sampling_weights: sampling_weights.as_ref().map(|weights| {
            weights
                .iter()
                .map(|(id, weight)| (id.clone(), *weight))
                .collect::<RapidHashMap<_, _>>()
        }),
        reestimate,
    })
}

/// The distance of a spec, built from the units, and the model behind it.
struct BuiltDistance {
    config: Option<DistanceConfig>,
    propensity: Option<PropensityScoreEstimate>,
}

fn build_distance<R: CovariateRecord>(
    spec: &DistanceSpec,
    anchors: &[R],
    candidates: &[R],
    estimand: Estimand,
    seed: u64,
    mahvars: bool,
) -> Result<BuiltDistance, Error> {
    let caliper = typed_caliper(
        if mahvars {
            "nearest.mahvars"
        } else {
            "distance"
        },
        spec.caliper(),
    )?;
    Ok(match spec {
        DistanceSpec::Date { .. } => BuiltDistance {
            config: caliper.map(|caliper| DistanceConfig::date(Some(caliper))),
            propensity: None,
        },
        DistanceSpec::Scores { scores, .. } => BuiltDistance {
            config: Some(DistanceConfig::propensity_score_map(
                scores
                    .iter()
                    .map(|(id, score)| (id.clone(), *score))
                    .collect(),
                caliper,
            )),
            propensity: None,
        },
        DistanceSpec::Propensity { .. } => {
            let config = propensity_config(spec, estimand, seed, caliper, false)?;
            let prepared = prepare_propensity_distance_config(anchors, candidates, &config, None)?;
            BuiltDistance {
                config: Some(prepared.distance_config),
                propensity: Some(prepared.estimate),
            }
        }
        DistanceSpec::Mahalanobis {
            covariates,
            covariance,
            transform,
            ..
        } => {
            let config = MahalanobisPreparationConfig {
                encoding: covariates.encoding(false),
                covariance_strategy: match covariance {
                    CovarianceSpec::PooledByDegreesOfFreedom => {
                        MahalanobisCovarianceStrategy::PooledByDegreesOfFreedom
                    }
                    CovarianceSpec::PooledWithinGroups => {
                        MahalanobisCovarianceStrategy::PooledWithinGroups
                    }
                    CovarianceSpec::FullSample => MahalanobisCovarianceStrategy::FullSample,
                },
                transform: match transform {
                    TransformSpec::Raw => MahalanobisTransform::Raw,
                    TransformSpec::Rank => MahalanobisTransform::Rank,
                    TransformSpec::RankTieAdjusted => MahalanobisTransform::RankTieAdjusted,
                },
            };
            BuiltDistance {
                config: Some(
                    prepare_mahalanobis_distance_config(anchors, candidates, caliper, &config)?
                        .distance_config,
                ),
                propensity: None,
            }
        }
        DistanceSpec::Matrix {
            case_ids,
            control_ids,
            values,
            ..
        } => {
            let flat = values
                .iter()
                .flat_map(|row| row.iter().map(|value| value.unwrap_or(f64::INFINITY)))
                .collect::<Vec<_>>();
            if values.iter().any(|row| row.len() != control_ids.len()) {
                return Err(Error::option(
                    "distance",
                    "every row of the distance matrix needs one value for each candidate id",
                ));
            }
            let matrix = PairDistanceMatrix::new(case_ids.clone(), control_ids.clone(), flat)
                .map_err(|error| Error::option("distance", error.to_string()))?;
            BuiltDistance {
                config: Some(DistanceConfig::matrix(matrix, caliper)),
                propensity: None,
            }
        }
        DistanceSpec::Euclidean {
            covariates, scaled, ..
        } => {
            let config = EuclideanPreparationConfig {
                encoding: covariates.encoding(false),
                scale: if *scaled {
                    EuclideanScale::PooledWithinGroupSd
                } else {
                    EuclideanScale::None
                },
            };
            BuiltDistance {
                config: Some(
                    prepare_euclidean_distance_config(anchors, candidates, caliper, &config)?
                        .distance_config,
                ),
                propensity: None,
            }
        }
    })
}

/// Generalised full matching on the points of a built distance. A
/// Mahalanobis distance is made Euclidean first.
fn run_quick<R: MatchingRecord>(
    config: &DistanceConfig,
    anchors: &[R],
    candidates: &[R],
    exact: &[String],
    options: &QuickMatchOptions,
) -> Result<SubclassedUnits, Error> {
    match config {
        DistanceConfig::PropensityScoreMap { scores, .. } => quick_match_subclasses(
            anchors,
            candidates,
            |row: &R| scores.get(row.id()).map(std::slice::from_ref),
            1,
            exact,
            options,
        ),
        DistanceConfig::EuclideanMap { vectors, .. } => quick_match_subclasses(
            anchors,
            candidates,
            |row: &R| vectors.get(row.id()),
            vectors.dimension(),
            exact,
            options,
        ),
        DistanceConfig::MahalanobisMap {
            vectors,
            inverse_covariance,
            ..
        } => {
            let dimension = vectors.dimension();
            let factor =
                cholesky_lower(inverse_covariance, dimension, Pivot::Strict).ok_or_else(|| {
                    Error::option(
                        "distance",
                        "the inverse covariance matrix is not positive definite; \
                     drop covariates that are collinear or constant",
                    )
                })?;
            let embedded: Vec<f64> = (0..vectors.len())
                .flat_map(|position| apply_transposed(&factor, vectors.row(position), dimension))
                .collect();
            quick_match_subclasses(
                anchors,
                candidates,
                |row: &R| {
                    vectors
                        .position(row.id())
                        .map(|position| &embedded[position * dimension..(position + 1) * dimension])
                },
                dimension,
                exact,
                options,
            )
        }
        DistanceConfig::Date { .. } => Err(Error::option(
            "distance",
            "the quick method needs a score or a covariate distance",
        )),
        DistanceConfig::Matrix { .. } => Err(Error::option(
            "distance",
            "the quick method clusters points and cannot take a distance matrix; \
             give it a score or the covariates",
        )),
    }
}

/// The scores of a distance for subclassification.
fn scores_of(config: &DistanceConfig) -> Option<&ScoreTable> {
    match config {
        DistanceConfig::PropensityScoreMap { scores, .. } => Some(scores),
        DistanceConfig::Date { .. }
        | DistanceConfig::MahalanobisMap { .. }
        | DistanceConfig::EuclideanMap { .. }
        | DistanceConfig::Matrix { .. } => None,
    }
}

/// Calipers on fields with their widths, in standard deviations if asked.
///
/// A caliper in standard deviations is scaled by the standard deviation of the
/// field over the units that take part in the matching, so over the units that
/// common-support trimming leaves, as `matchit()` does. The trimming is read
/// from `scores`, the scores of the distance.
fn field_calipers<R: MatchingRecord>(
    spec: &MatchSpec,
    anchors: &[R],
    candidates: &[R],
    scores: Option<&ScoreTable>,
) -> Result<Vec<(String, f64)>, Error> {
    let kept = match (spec.discard, scores) {
        (Some(policy), Some(scores)) => {
            units_after_trimming(anchors, candidates, policy.into(), scores)
        }
        _ => (anchors.iter().collect(), candidates.iter().collect()),
    };
    spec.calipers
        .iter()
        .map(|(field, width)| {
            if !spec.std_caliper.field(field) {
                return Ok((field.clone(), *width));
            }
            let values: Vec<f64> = kept
                .0
                .iter()
                .chain(&kept.1)
                .filter_map(|record| record.numeric(field))
                .filter(|value| value.is_finite())
                .collect();
            let sd = crate::constraints::sample_sd(&values).ok_or_else(|| {
                Error::option(
                    "std_caliper",
                    format!(
                        "fewer than two units have a value of {field}, so its standard \
                         deviation is undefined; give the caliper in the units of the field"
                    ),
                )
            })?;
            Ok((field.clone(), width * sd))
        })
        .collect()
}

/// The cases and controls that common-support trimming leaves, by the
/// engine's own rule (`standard_common_support`), so the units a field caliper
/// is scaled over are the units the matcher keeps. When the ranges do not
/// overlap, or a group has no finite score, nothing is trimmed here, and the
/// matcher reports the failure.
fn units_after_trimming<'a, R: MatchingRecord>(
    anchors: &'a [R],
    candidates: &'a [R],
    policy: CommonSupport,
    scores: &ScoreTable,
) -> (Vec<&'a R>, Vec<&'a R>) {
    let Ok(trimmed) = crate::matching::standard_common_support(anchors, candidates, policy, scores)
    else {
        return (anchors.iter().collect(), candidates.iter().collect());
    };
    let keep = |rows: &'a [R], out: &std::collections::HashSet<usize>| -> Vec<&'a R> {
        rows.iter()
            .enumerate()
            .filter(|(idx, _)| !out.contains(idx))
            .map(|(_, row)| row)
            .collect()
    };
    (
        keep(anchors, &trimmed.anchors),
        keep(candidates, &trimmed.candidates),
    )
}

/// What a typed standard job needs beyond the units.
struct StandardPlan<'a, R> {
    anchors: &'a [R],
    candidates: &'a [R],
    spec: &'a MatchSpec,
    distance: Option<DistanceConfig>,
    calipers: Vec<(String, f64)>,
    rounds: bool,
}

fn typed_ratio(spec: &MatchSpec) -> Result<MatchRatio, Error> {
    MatchRatio::new(spec.ratio).ok_or(Error::ZeroMatchRatio)
}

fn typed_window(spec: &MatchSpec) -> Result<BirthDateWindowDays, Error> {
    BirthDateWindowDays::new(spec.birth_window_days).ok_or(Error::NegativeBirthDateWindow)
}

fn typed_fallback(spec: &MatchSpec) -> Vec<MatchRatio> {
    spec.ratio_fallback
        .iter()
        .filter_map(|ratio| MatchRatio::new(*ratio))
        .collect()
}

fn run_standard_with<R, S>(
    plan: StandardPlan<'_, R>,
    strategy: S,
) -> Result<(MatchOutcome, Option<PropensityScoreEstimate>), Error>
where
    R: MatchingRecord + CovariateRecord,
    S: SelectionStrategy<R> + Clone + Send + Sync,
{
    let spec = plan.spec;
    let sex: Vec<ExactMatchConstraint> = if spec.match_sex {
        vec![sex_match(&spec.sex_key)]
    } else {
        Vec::new()
    };
    let anti: Vec<AntiExact> = spec
        .antiexact
        .iter()
        .map(|key| AntiExact::on_key(key.clone()))
        .collect();
    let calipers: Vec<_> = plan
        .calipers
        .iter()
        .map(|(field, width)| caliper_on_field::<R>(field, *width))
        .collect();
    if spec.reestimate {
        let caliper = typed_caliper("distance", spec.distance.caliper())?;
        let config = propensity_config(
            &spec.distance,
            spec.estimand,
            spec.root_seed()?,
            caliper,
            true,
        )?;
        let constraints = Chained {
            first: Chained {
                first: sex,
                second: anti,
            },
            second: calipers,
        };
        let criteria = spec.criteria();
        let fallback = typed_fallback(spec);
        let matched = if plan.rounds {
            estimate_propensity_and_match(
                plan.anchors,
                plan.candidates,
                &criteria,
                Rounds(strategy),
                &fallback,
                &constraints,
                &config,
            )?
        } else {
            estimate_propensity_and_match(
                plan.anchors,
                plan.candidates,
                &criteria,
                strategy,
                &fallback,
                &constraints,
                &config,
            )?
        };
        return Ok((reject_invalid(matched.outcome)?, Some(matched.propensity)));
    }
    let mut job = MatchJob::new_standard(plan.anchors, plan.candidates, spec.root_seed()?)
        .with_ratio(typed_ratio(spec)?)
        .with_birth_window(typed_window(spec)?)
        .with_replacement(spec.replace)
        .with_estimand(spec.estimand)
        .with_exact_matches(spec.exact.iter().cloned())
        .with_unique_by_key(spec.unique_by_key.clone())
        .with_ratio_fallback(typed_fallback(spec))
        .with_common_support(spec.discard.map(Into::into))
        .with_constraints(sex)
        .with_constraints(anti)
        .with_constraints(calipers);
    if let Some(distance) = plan.distance {
        job = job.with_distance_config(distance);
    }
    let job = job.with_strategy(strategy);
    let outcome = if plan.rounds {
        job.with_round_assignment().run()
    } else {
        job.run()
    };
    Ok((outcome?, None))
}

/// Check a spec and run it on the units.
///
/// The spec is checked first, so an option that is out of range or conflicts
/// with another is an error before any unit is read. The typed job is then
/// built from the options and run, and an outcome the matcher built to report
/// an invalid configuration becomes an error too.
///
/// Matching on a covariate distance reads the covariates of the units, so the
/// record type must carry them, as [`BalanceRecord`](crate::BalanceRecord) does.
///
/// # Errors
///
/// A [`Error`] when the spec is invalid, when the units are not those of
/// the spec's mode, when a distance cannot be built from the covariates, or
/// when exact matching cannot run.
pub fn run_spec<R>(spec: &MatchSpec, input: SpecInput<'_, R>) -> Result<SpecOutput, Error>
where
    R: MatchingRecord + CovariateRecord,
{
    spec.validate()?;
    match (spec.mode, input) {
        (
            MatchMode::Standard,
            SpecInput::Standard {
                cases: anchors,
                controls: candidates,
            },
        ) => run_standard(spec, anchors, candidates),
        (MatchMode::RiskSet, SpecInput::RiskSet { cohort }) => Ok(SpecOutput::Pairs {
            outcome: Box::new(run_risk_set(spec, cohort, None)?),
            propensity: None,
        }),
        (MatchMode::ExposureDensity, SpecInput::RiskSet { cohort }) => {
            run_exposure_density(spec, cohort)
        }
        (MatchMode::SequentialTrials, SpecInput::Trials { panel }) => run_trials(spec, panel),
        (mode, _) => Err(Error::InputMode { mode }),
    }
}

/// A score of the caller's own is needed for every unit that is read. A unit
/// without one is an error naming it, as MatchIt refuses missing scores,
/// rather than a unit that silently drops out of the matching.
fn check_scores_cover<'a>(
    distance: &DistanceSpec,
    ids: impl IntoIterator<Item = &'a str>,
) -> Result<(), Error> {
    let DistanceSpec::Scores { scores, .. } = distance else {
        return Ok(());
    };
    match ids.into_iter().find(|id| !scores.contains_key(*id)) {
        Some(id) => Err(Error::option(
            "distance",
            format!("the unit {id} has no score; give every unit a score"),
        )),
        None => Ok(()),
    }
}

fn run_standard<R: MatchingRecord + CovariateRecord>(
    spec: &MatchSpec,
    anchors: &[R],
    candidates: &[R],
) -> Result<SpecOutput, Error> {
    check_scores_cover(
        &spec.distance,
        anchors.iter().chain(candidates).map(MatchingRecord::id),
    )?;
    if spec.discard.is_some()
        && matches!(
            spec.method,
            MatchMethod::Subclass
                | MatchMethod::Quick
                | MatchMethod::Optimal
                | MatchMethod::Full
                | MatchMethod::Genetic
        )
    {
        return run_trimmed(spec, anchors, candidates);
    }
    run_untrimmed(spec, anchors, candidates)
}

/// The methods of a standard match that do not trim before they run, or that
/// trim inside the nearest-family engine.
fn run_untrimmed<R: MatchingRecord + CovariateRecord>(
    spec: &MatchSpec,
    anchors: &[R],
    candidates: &[R],
) -> Result<SpecOutput, Error> {
    match spec.method {
        MatchMethod::Exact => {
            let outcome = exact_match_subclasses(anchors, candidates, &spec.exact)?;
            return Ok(SpecOutput::Exact(Box::new(outcome)));
        }
        MatchMethod::Cem => return run_cem(spec, anchors, candidates),
        MatchMethod::Subclass => {
            let built = build_distance(
                &spec.distance,
                anchors,
                candidates,
                spec.estimand,
                spec.root_seed()?,
                false,
            )?;
            let scores = built
                .config
                .as_ref()
                .and_then(scores_of)
                .ok_or_else(|| Error::option("distance", "no scores were built"))?;
            let config = SubclassificationConfig {
                subclasses: spec.subclass.n,
                reference_group: spec.subclass.reference.map(Into::into),
                min_n: spec.subclass.min_n,
            };
            let outcome = subclassify_by_propensity_score_map(
                anchors,
                candidates,
                &scores.to_map(),
                spec.estimand,
                &config,
            )
            .map_err(|error| Error::option("subclass.n", error.to_string()))?;
            return Ok(SpecOutput::Subclass {
                outcome: Box::new(outcome),
                propensity: built.propensity.map(Box::new),
            });
        }
        MatchMethod::Quick => {
            let built = build_distance(
                &spec.distance,
                anchors,
                candidates,
                spec.estimand,
                spec.root_seed()?,
                false,
            )?;
            let config = built
                .config
                .as_ref()
                .ok_or_else(|| Error::option("distance", "no distance was built"))?;
            let options = QuickMatchOptions {
                min_cases: spec.quick.min_cases,
                min_controls: spec.quick.min_controls,
                size: spec.quick.size,
                caliper: spec.distance.caliper(),
                seed_method: spec.quick.seed_method,
            };
            let outcome = run_quick(config, anchors, candidates, &spec.exact, &options)?;
            return Ok(SpecOutput::Quick(Box::new(outcome)));
        }
        MatchMethod::Genetic => return wiring::run_genetic(spec, anchors, candidates),
        MatchMethod::Cardinality => return wiring::run_cardinality(spec, anchors, candidates),
        MatchMethod::Optimal | MatchMethod::Full => {
            let built = build_distance(
                &spec.distance,
                anchors,
                candidates,
                spec.estimand,
                spec.root_seed()?,
                false,
            )?;
            let calipers = field_calipers(spec, anchors, candidates, None)?;
            let sex: Vec<ExactMatchConstraint> = if spec.match_sex {
                vec![sex_match(&spec.sex_key)]
            } else {
                Vec::new()
            };
            let anti: Vec<AntiExact> = spec
                .antiexact
                .iter()
                .map(|key| AntiExact::on_key(key.clone()))
                .collect();
            let calipers: Vec<_> = calipers
                .iter()
                .map(|(field, width)| caliper_on_field::<R>(field, *width))
                .collect();
            let constraints = Chained {
                first: Chained {
                    first: sex,
                    second: anti,
                },
                second: calipers,
            };
            let criteria = spec.criteria();
            let mut request = OptimalRequest::new(&criteria, &constraints);
            if let Some(config) = built.config.as_ref() {
                request = request.with_distance_config(config);
            }
            if spec.method == MatchMethod::Full {
                let retain = match (spec.full.omit_fraction, spec.full.mean_controls) {
                    (Some(fraction), _) => Retain::OmitFraction(fraction),
                    (None, Some(mean)) => Retain::MeanControls(mean),
                    (None, None) => Retain::All,
                };
                let config = FullMatchConfig {
                    min_controls: spec.full.min_controls,
                    max_controls: spec.full.max_controls.unwrap_or(f64::INFINITY),
                    retain,
                };
                let outcome = match_full(anchors, candidates, &request, &config)?;
                return Ok(SpecOutput::Full(Box::new(outcome)));
            }
            let outcome = match &spec.fine_balance {
                Some(balance) => {
                    match_fine_balance(anchors, candidates, &request, balance)?.outcome
                }
                None => match_optimal(anchors, candidates, &request)?,
            };
            return Ok(SpecOutput::Pairs {
                outcome: Box::new(outcome),
                propensity: built.propensity.map(Box::new),
            });
        }
        MatchMethod::Random
        | MatchMethod::LowestIndex
        | MatchMethod::NearestBirthDate
        | MatchMethod::Nearest => {}
    }

    // With reestimate the typed call fits the model itself, on all units and
    // again on the ones that remain.
    let built = if spec.reestimate {
        BuiltDistance {
            config: None,
            propensity: None,
        }
    } else {
        build_distance(
            &spec.distance,
            anchors,
            candidates,
            spec.estimand,
            spec.root_seed()?,
            false,
        )?
    };
    let calipers = if spec.reestimate && spec.std_caliper.any() {
        // The units that remain are those the first fit leaves, so a field
        // caliper in standard deviations is scaled over them.
        let first = build_distance(
            &spec.distance,
            anchors,
            candidates,
            spec.estimand,
            spec.root_seed()?,
            false,
        )?;
        field_calipers(
            spec,
            anchors,
            candidates,
            first.config.as_ref().and_then(scores_of),
        )?
    } else {
        field_calipers(
            spec,
            anchors,
            candidates,
            built.config.as_ref().and_then(scores_of),
        )?
    };
    let rounds = spec.assignment == Some(AssignmentSpec::Rounds);
    let plan = |rounds| StandardPlan {
        anchors,
        candidates,
        spec,
        distance: built.config.clone(),
        calipers: calipers.clone(),
        rounds,
    };
    let (outcome, refitted) = match spec.method {
        MatchMethod::Random => {
            run_standard_with(plan(rounds), RandomSelection::seeded(spec.root_seed()?))?
        }
        MatchMethod::LowestIndex => run_standard_with(plan(rounds), DeterministicSelection)?,
        MatchMethod::NearestBirthDate => {
            run_standard_with(plan(rounds), NearestBirthDateSelection)?
        }
        MatchMethod::Nearest => {
            let mut strategy = NearestSelection::new(spec.root_seed()?)
                .with_std_caliper(spec.std_caliper.distance());
            if let Some(order) = spec.nearest.m_order {
                strategy = strategy.with_order(order.into());
            }
            if let Some(variable) = spec.nearest.variable_ratio {
                strategy = strategy.with_variable_ratio(
                    VariableRatio::new(variable.average, variable.min, variable.max).map_err(
                        |error| Error::option("nearest.variable_ratio", error.to_string()),
                    )?,
                );
            }
            if let Some(assignment) = spec.assignment {
                strategy = strategy.with_assignment(assignment.into());
            }
            if let Some(ranking) = &spec.nearest.mahvars {
                let ranking = build_distance(
                    ranking,
                    anchors,
                    candidates,
                    spec.estimand,
                    spec.root_seed()?,
                    true,
                )?;
                if let Some(config) = ranking.config {
                    strategy = strategy.with_mahvars(config);
                }
            }
            run_standard_with(plan(false), strategy)?
        }
        MatchMethod::Exact
        | MatchMethod::Subclass
        | MatchMethod::Cem
        | MatchMethod::Quick
        | MatchMethod::Optimal
        | MatchMethod::Full
        | MatchMethod::Genetic
        | MatchMethod::Cardinality => {
            unreachable!("handled above")
        }
    };
    Ok(SpecOutput::Pairs {
        outcome: Box::new(outcome),
        propensity: refitted.or(built.propensity).map(Box::new),
    })
}

/// Subclass, full, optimal, quick and genetic matching after common-support
/// trimming.
///
/// The distance is fitted on all units and the units outside the other
/// group's range of it are set aside, as `discard` of `matchit()` does. The
/// method then runs on the units that remain. Without `reestimate` it uses the
/// scores of the first fit. With it, the propensity model is fitted again on
/// the units that remain.
///
/// Subclass cutpoints follow `matchit()`: without `reestimate` they are the
/// quantiles over every unit of the reference group, trimmed ones included.
/// With it `matchit()` takes them over the first-fit scores of the trimmed
/// units mixed with the second-fit scores of the kept units, which is the
/// distribution of neither fit and a defect, so the cutpoints are taken over
/// the kept units alone.
fn run_trimmed<R: MatchingRecord + CovariateRecord>(
    spec: &MatchSpec,
    anchors: &[R],
    candidates: &[R],
) -> Result<SpecOutput, Error> {
    let policy: CommonSupport = spec
        .discard
        .map(Into::into)
        .ok_or_else(|| Error::option("discard", "no common support was asked for"))?;
    let first = build_distance(
        &spec.distance,
        anchors,
        candidates,
        spec.estimand,
        spec.root_seed()?,
        false,
    )?;
    let scores = first
        .config
        .as_ref()
        .and_then(scores_of)
        .ok_or_else(|| Error::option("distance", "no scores were built"))?;
    let support = crate::matching::standard_common_support(anchors, candidates, policy, scores)
        .map_err(Error::CommonSupport)?;
    let kept = |rows: &'_ [R], out: &std::collections::HashSet<usize>| -> Vec<usize> {
        (0..rows.len()).filter(|idx| !out.contains(idx)).collect()
    };
    let kept_anchors = kept(anchors, &support.anchors);
    let kept_candidates = kept(candidates, &support.candidates);
    let anchor_rows: Vec<&R> = kept_anchors.iter().map(|&idx| &anchors[idx]).collect();
    let candidate_rows: Vec<&R> = kept_candidates
        .iter()
        .map(|&idx| &candidates[idx])
        .collect();
    let mut inner = MatchSpec {
        discard: None,
        reestimate: false,
        ..spec.clone()
    };
    if !spec.reestimate {
        // The units that remain are matched on the scores of the first fit.
        let table: BTreeMap<String, f64> = anchor_rows
            .iter()
            .chain(&candidate_rows)
            .filter_map(|row| {
                scores
                    .get(row.id())
                    .map(|score| (row.id().to_string(), *score))
            })
            .collect();
        inner.distance = DistanceSpec::Scores {
            scores: table,
            caliper: spec.distance.caliper(),
        };
    }
    let trimmed =
        crate::matching::trimmed_ids(anchors, candidates, &support.anchors, &support.candidates);
    if spec.method == MatchMethod::Subclass && !spec.reestimate {
        return run_trimmed_subclass(
            spec,
            policy,
            (anchors, candidates),
            (&anchor_rows, &candidate_rows),
            &support,
            (scores, first.propensity),
        );
    }
    let output = run_untrimmed(&inner, &anchor_rows, &candidate_rows)?;
    // With a refit, the estimate reports the refitted scores for the kept
    // units and the first fit's for the trimmed ones, as `matchit()` does.
    let first_fit = first.propensity.map(Box::new);
    let refit_estimate = first_fit.clone();
    let with_discarded = |refit: Box<PropensityScoreEstimate>| match refit_estimate {
        Some(first) => Box::new(first.with_refit(*refit, &trimmed.cases, &trimmed.controls)),
        None => refit,
    };
    let mut output = match output {
        SpecOutput::Pairs {
            outcome,
            propensity,
        } => SpecOutput::Pairs {
            outcome,
            propensity: if spec.reestimate {
                propensity.map(with_discarded)
            } else {
                first_fit
            },
        },
        SpecOutput::Subclass {
            outcome,
            propensity,
        } => SpecOutput::Subclass {
            outcome,
            propensity: if spec.reestimate {
                propensity.map(with_discarded)
            } else {
                first_fit
            },
        },
        other => other,
    };
    record_trimming(&mut output, policy, &support, anchors, candidates);
    Ok(output)
}

/// Subclassification after trimming, without a refit.
///
/// The cutpoints are the quantiles of the reference group over every unit of
/// the group on the first fit, trimmed units included, as `matchit()` takes
/// them. Only the kept units are subclassified and weighted. With `reestimate`
/// `run_trimmed` takes the cutpoints over the kept units alone instead:
/// `matchit()` then mixes the first-fit scores of the trimmed units with the
/// second-fit scores of the kept ones in one set of quantiles, which has no
/// meaning as a distribution of either fit.
fn run_trimmed_subclass<R: MatchingRecord>(
    spec: &MatchSpec,
    policy: CommonSupport,
    all: (&[R], &[R]),
    kept: (&[&R], &[&R]),
    support: &crate::matching::StandardCommonSupport,
    fit: (&ScoreTable, Option<PropensityScoreEstimate>),
) -> Result<SpecOutput, Error> {
    let (scores, propensity) = fit;
    let config = SubclassificationConfig {
        subclasses: spec.subclass.n,
        reference_group: spec.subclass.reference.map(Into::into),
        min_n: spec.subclass.min_n,
    };
    let set_aside = |rows: &[R], out: &std::collections::HashSet<usize>| -> Vec<f64> {
        rows.iter()
            .enumerate()
            .filter(|(idx, _)| out.contains(idx))
            .filter_map(|(_, row)| scores.get(row.id()).copied())
            .collect()
    };
    let outcome = crate::matching::subclassify_with_set_aside_scores(
        kept.0,
        kept.1,
        &scores.to_map(),
        spec.estimand,
        &config,
        &set_aside(all.0, &support.anchors),
        &set_aside(all.1, &support.candidates),
    )
    .map_err(|error| Error::option("subclass.n", error.to_string()))?;
    let mut output = SpecOutput::Subclass {
        outcome: Box::new(outcome),
        propensity: propensity.map(Box::new),
    };
    record_trimming(&mut output, policy, support, all.0, all.1);
    Ok(output)
}

/// Write the trimming into the outcome that `run_trimmed` got.
fn record_trimming<R: MatchingRecord>(
    output: &mut SpecOutput,
    policy: CommonSupport,
    support: &crate::matching::StandardCommonSupport,
    anchors: &[R],
    candidates: &[R],
) {
    let trimmed =
        crate::matching::trimmed_ids(anchors, candidates, &support.anchors, &support.candidates);
    match output {
        SpecOutput::Pairs { outcome, .. } | SpecOutput::Genetic { outcome, .. } => {
            support.record_after_match(&mut outcome.diagnostics, policy, anchors, candidates);
        }
        SpecOutput::Subclass { outcome, .. } => outcome.trimmed = trimmed,
        SpecOutput::Quick(outcome) | SpecOutput::Full(outcome) => outcome.trimmed = trimmed,
        _ => {}
    }
}

fn run_cem<R: MatchingRecord + CovariateRecord>(
    spec: &MatchSpec,
    anchors: &[R],
    candidates: &[R],
) -> Result<SpecOutput, Error> {
    let variables: Vec<CemVariable> = spec
        .cem
        .variables
        .iter()
        .map(|variable| match variable {
            CemVariableSpec::Numeric { name, binning } => {
                CemVariable::numeric(name.clone(), binning.into())
            }
            CemVariableSpec::Categorical { name, groups } => {
                CemVariable::categorical(name.clone(), groups.clone())
            }
        })
        .collect();
    let Some(k2k) = &spec.cem.k2k else {
        let outcome = cem_subclasses(anchors, candidates, &variables)?;
        return Ok(SpecOutput::Exact(Box::new(outcome)));
    };
    let strata = cem_strata(anchors, candidates, &variables)?;
    let built = k2k
        .distance
        .as_ref()
        .map(|distance| {
            build_distance(
                distance,
                anchors,
                candidates,
                spec.estimand,
                spec.root_seed()?,
                false,
            )
        })
        .transpose()?;
    let order = k2k.m_order.map_or(MatchOrder::Data, Into::into);
    let outcome = cem_k2k_pairs(
        anchors,
        candidates,
        &strata,
        built.as_ref().and_then(|built| built.config.as_ref()),
        order,
        spec.root_seed()?,
        spec.estimand,
    )?;
    Ok(SpecOutput::Pairs {
        outcome: Box::new(outcome),
        propensity: built.and_then(|built| built.propensity.map(Box::new)),
    })
}

fn death_date<R: RoleIndexedRecord>(record: &R) -> Option<NaiveDate> {
    record.death_date()
}

type Rec<R> = RoleTransitionRecord<R>;
type DeathSelector<R> = fn(&Rec<R>) -> Option<NaiveDate>;
type ResidentSelector<R> = fn(&Rec<R>, NaiveDate) -> bool;

fn run_exposure_density<R: MatchingRecord + CovariateRecord>(
    spec: &MatchSpec,
    cohort: &[RoleTransitionRecord<R>],
) -> Result<SpecOutput, Error> {
    let outcome = run_risk_set(spec, cohort, None)?;
    let sets = exposure_density_sets(cohort, outcome)
        .map_err(|error| Error::option("cohort", error.to_string()))?;
    Ok(SpecOutput::ExposureDensity(Box::new(sets)))
}

fn run_trials<R: MatchingRecord + CovariateRecord>(
    spec: &MatchSpec,
    panel: &TrialPanel<'_, R>,
) -> Result<SpecOutput, Error> {
    let outcome = match_trials(panel, &spec.trials, &TrialMatching::default(), |pool| {
        let in_trial = MatchSpec {
            seed: Some(trial_seed(spec.root_seed()?, pool.period())),
            mode: MatchMode::RiskSet,
            ..spec.clone()
        };
        run_risk_set(&in_trial, pool.records(), pool.id_order())
    })
    .map_err(|error| match error {
        Error::Trial { source, .. } => *source,
        other => Error::option("panel", other.to_string()),
    })?;
    Ok(SpecOutput::Trials(Box::new(outcome)))
}

/// The pairs of risk-set matching of `cohort`, which exposure density and
/// each sequential trial also match by.
fn run_risk_set<R: MatchingRecord + CovariateRecord>(
    spec: &MatchSpec,
    cohort: &[RoleTransitionRecord<R>],
    id_order: Option<&IdOrder>,
) -> Result<MatchOutcome, Error> {
    check_scores_cover(&spec.distance, cohort.iter().map(MatchingRecord::id))?;
    let distance = match &spec.distance {
        DistanceSpec::Date { caliper } => {
            typed_caliper("distance", *caliper)?.map(|caliper| DistanceConfig::date(Some(caliper)))
        }
        DistanceSpec::Scores { scores, caliper } => Some(DistanceConfig::propensity_score_map(
            scores
                .iter()
                .map(|(id, score)| (id.clone(), *score))
                .collect(),
            typed_caliper("distance", *caliper)?,
        )),
        DistanceSpec::Propensity { .. }
        | DistanceSpec::Mahalanobis { .. }
        | DistanceSpec::Euclidean { .. }
        | DistanceSpec::Matrix { .. } => {
            return Err(Error::option("distance", "not available in risk-set mode"));
        }
    };

    let sex: Vec<ExactMatchConstraint> = if spec.match_sex {
        vec![sex_match(&spec.sex_key)]
    } else {
        Vec::new()
    };
    let anti: Vec<AntiExact> = spec
        .antiexact
        .iter()
        .map(|key| AntiExact::on_key(key.clone()))
        .collect();
    let calipers: Vec<_> = spec
        .calipers
        .iter()
        .map(|(field, width)| caliper_on_field::<Rec<R>>(field, *width))
        .collect();
    let alive: Vec<MustBeAlive<DeathSelector<R>>> = if spec.require_alive {
        vec![MustBeAlive::at_index_date(
            death_date::<Rec<R>> as DeathSelector<R>,
        )]
    } else {
        Vec::new()
    };
    let resident: Vec<MustBeResident<ResidentSelector<R>>> = if spec.require_resident {
        fn is_resident<R: MatchingRecord>(record: &Rec<R>, index_date: NaiveDate) -> bool {
            crate::ResidentAtIndexRecord::is_resident_at(record, index_date)
        }
        vec![MustBeResident::at_index_date(
            is_resident::<R> as ResidentSelector<R>,
        )]
    } else {
        Vec::new()
    };

    let mut job = MatchJob::new_transition(cohort, spec.root_seed()?)
        .with_id_order(id_order)
        .with_common_support(spec.discard.map(Into::into))
        .with_ratio(typed_ratio(spec)?)
        .with_birth_window(typed_window(spec)?)
        .with_replacement(spec.replace)
        .with_estimand(spec.estimand)
        .with_exact_matches(spec.exact.iter().cloned())
        .with_unique_by_key(spec.unique_by_key.clone())
        .with_ratio_fallback(typed_fallback(spec))
        .with_constraints(sex)
        .with_constraints(anti)
        .with_constraints(calipers)
        .with_constraints(alive)
        .with_constraints(resident);
    if let Some(years) = spec.age_limit_years.and_then(AgeLimitYears::new) {
        job = job.with_age_limit(years);
    }
    if let Some(distance) = distance {
        job = job.with_distance_config(distance);
    }
    match spec.method {
        MatchMethod::Random => job
            .with_strategy(RandomSelection::seeded(spec.root_seed()?))
            .run(),
        MatchMethod::LowestIndex => job.with_strategy(DeterministicSelection).run(),
        MatchMethod::NearestBirthDate => job.with_strategy(NearestBirthDateSelection).run(),
        MatchMethod::Nearest
        | MatchMethod::Exact
        | MatchMethod::Subclass
        | MatchMethod::Cem
        | MatchMethod::Quick
        | MatchMethod::Optimal
        | MatchMethod::Full
        | MatchMethod::Genetic
        | MatchMethod::Cardinality => Err(risk_set_method_error()),
    }
}

/// The error for a method that risk-set matching cannot choose by.
fn risk_set_method_error() -> Error {
    Error::option(
        "method",
        "risk-set matching chooses among the units at risk; \
         use random, lowest_index or nearest_birth_date",
    )
}

/// Equal sex on `key`, counted under the reason of `MatchJob::with_gender_match`.
fn sex_match(key: &str) -> ExactMatchConstraint {
    ExactMatchConstraint::on_key(key).with_reason("gender_mismatch")
}
