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
//! [`run_spec`] turns that into a [`SpecError`] as well, so a caller never
//! mistakes it for a match that found nothing.

use crate::constraints::{
    AntiExact, GenderMatch, MustBeAlive, MustBeResident, caliper_on_field, numeric_field_sd,
};
use crate::matching::{
    Assignment, CemBinning, CemError, CemRule, CemVariable, Chained, CovariateEncodingConfig,
    CovariateRecord, DeterministicSelection, DistanceConfig, ElasticNetCvConfig,
    ElasticNetLogisticConfig, EstimationError, EuclideanPreparationConfig, EuclideanScale,
    ExactMatchError, ExactMatchOutcome, FullMatchConfig, FullMatchOutcome, LambdaRule,
    LogisticRegressionConfig, MahalanobisCovarianceStrategy, MahalanobisPreparationConfig,
    MahalanobisTransform, MatchJob, MatchOrder, MatchingRecord, MissingValuePolicy,
    NearestBirthDateSelection, NearestSelection, OptimalError, OptimalRequest, PairDistanceMatrix,
    PropensityEstimator, PropensityLink, PropensityScoreConfig, PropensityScoreEstimate,
    PropensityScoreOutputScale, QuickMatchError, QuickMatchOptions, QuickMatchOutcome,
    RandomSelection, Retain, RoleIndexedRecord, SeedMethod, SelectionStrategy,
    SubclassReferenceGroup, SubclassificationConfig, SubclassificationOutcome, apply_transposed,
    cem_k2k_pairs, cem_strata, cem_subclasses, cholesky_lower, exact_match_subclasses, match_full,
    match_optimal, prepare_euclidean_distance_config, prepare_mahalanobis_distance_config,
    prepare_propensity_distance_config, quick_match_subclasses,
    subclassify_by_propensity_score_map,
};
use crate::types::{
    AgeLimitYears, BirthDateWindowDays, CriteriaValidationError, DistanceCaliper, Estimand,
    ExclusionReason, MatchOutcome, MatchRatio, MatchingCriteria, RoleTransitionRecord,
};
use chrono::NaiveDate;
use rapidhash::RapidHashMap;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Standard matching of a set of anchors to a set of candidates, or risk-set
/// matching within one cohort.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchMode {
    /// Fixed anchors and candidates.
    #[default]
    Standard,
    /// Each case takes controls from the units still at risk on its index
    /// date. The units are one cohort, and a unit with a transition date is a
    /// case.
    RiskSet,
}

/// How controls are chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchMethod {
    /// A random eligible candidate, from the spec's seed.
    #[default]
    Random,
    /// The eligible candidate that comes first in the data.
    LowestIndex,
    /// The eligible candidate whose birth date is nearest the anchor's.
    NearestBirthDate,
    /// The nearest candidate by the distance, as `matchit(method = "nearest")`.
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
    /// `matchit(method = "optimal")`. Standard mode only. The anchors are the
    /// focal group.
    Optimal,
    /// Subclasses of one anchor with several candidates or the reverse that
    /// minimise the total distance, as `matchit(method = "full")`. Standard
    /// mode only. The anchors are the focal group.
    Full,
}

/// The order in which anchors take their controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssignmentSpec {
    /// Each anchor takes all its controls before the next anchor starts.
    #[default]
    Sequential,
    /// Every anchor takes a first control before any takes a second.
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

/// The order in which anchors are matched, `m.order` of `matchit()`.
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
        /// Scale the columns to unit standard deviation before fitting.
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
        /// The rule that picks the penalty.
        #[serde(default)]
        rule: LambdaSpec,
        /// Scale the columns to unit standard deviation before fitting.
        #[serde(default = "default_true")]
        standardize: bool,
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
        CovariateEncodingConfig::builder()
            .covariate_keys(self.names.clone())
            .include_intercept(intercept)
            .missing_value_policy(self.missing.into())
            .build()
    }
}

/// The distance between an anchor and a candidate.
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
    /// a matrix in `matchit()`. Rows are anchors and columns candidates. An
    /// entry of `None` forbids the pair, as an infinite entry does in R.
    Matrix {
        /// Ids of the rows.
        anchor_ids: Vec<String>,
        /// Ids of the columns.
        candidate_ids: Vec<String>,
        /// The distances, one row per anchor.
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

/// The options of the nearest method.
///
/// Options that only the nearest method has are fields of this struct.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NearestSpec {
    /// The order in which anchors are matched. The default is `matchit()`'s:
    /// largest score first for a score distance, and the order of the data
    /// otherwise.
    pub m_order: Option<OrderSpec>,
    /// Rank the candidates by this distance, usually Mahalanobis, and use the
    /// spec's distance as a caliper and for the order of anchors, as
    /// `mahvars` does.
    pub mahvars: Option<DistanceSpec>,
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
    /// The distance that ranks the candidates of a stratum. Without one the
    /// candidates are taken in an order drawn from the seed. A caliper is not
    /// used.
    pub distance: Option<DistanceSpec>,
    /// The order in which anchors take their candidate. The default is the
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
    /// The anchors.
    Anchors,
    /// The candidates.
    Candidates,
    /// Both groups together.
    Pooled,
}

impl From<ReferenceSpec> for SubclassReferenceGroup {
    fn from(spec: ReferenceSpec) -> Self {
        match spec {
            ReferenceSpec::Anchors => Self::Anchors,
            ReferenceSpec::Candidates => Self::Candidates,
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
    /// Fewest anchors in a subclass.
    pub min_anchors: usize,
    /// Fewest candidates in a subclass.
    pub min_candidates: usize,
    /// Fewest units in a subclass. The default is the two minimums together.
    pub size: Option<usize>,
    /// The order in which units are tried as seeds.
    pub seed_method: SeedMethod,
}

impl Default for QuickSpec {
    fn default() -> Self {
        let options = QuickMatchOptions::default();
        Self {
            min_anchors: options.min_anchors,
            min_candidates: options.min_candidates,
            size: options.size,
            seed_method: options.seed_method,
        }
    }
}

/// Everything a match can be configured with.
///
/// The defaults are those of [`MatchJob`]: random choice among the candidates
/// within 30 days of birth, one control per anchor, no replacement.
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
    /// The distance between an anchor and a candidate.
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
    /// How anchors share the candidates when each takes more than one. The
    /// default is sequential, and rounds for the nearest method as in
    /// `matchit()`. Risk-set matching is always sequential.
    pub assignment: Option<AssignmentSpec>,
    /// Controls per anchor.
    pub ratio: usize,
    /// Descending ratios to fall back to when an anchor finds fewer controls
    /// than the ratio. Not available with the nearest method or rounds.
    pub ratio_fallback: Vec<usize>,
    /// Let a candidate serve several anchors.
    pub replace: bool,
    /// Widest difference in days between the birth dates of a pair.
    pub birth_window_days: i32,
    /// Calipers on numeric fields of the units, by field name, as the named
    /// calipers of `matchit()`.
    pub calipers: BTreeMap<String, f64>,
    /// Read the calipers in standard deviations, as `std.caliper` does: the
    /// distance caliper in standard deviations of the distance, and the
    /// calipers on fields in standard deviations of the field over all units.
    /// Nearest method only. Scaling the calipers on fields is not compared with
    /// MatchIt.
    pub std_caliper: bool,
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
    /// makes a case. The default is 6.
    pub age_limit_years: Option<u8>,
    /// Risk-set matching: controls must be alive on the case's index date.
    pub require_alive: bool,
    /// Risk-set matching: controls must be resident on the case's index date.
    pub require_resident: bool,
    /// The target estimand.
    pub estimand: Estimand,
    /// Seed of every random choice.
    pub seed: u64,
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
            assignment: None,
            ratio: criteria.match_ratio,
            ratio_fallback: Vec::new(),
            replace: criteria.allow_replacement,
            birth_window_days: criteria.birth_date_window_days,
            calipers: BTreeMap::new(),
            std_caliper: false,
            exact: Vec::new(),
            antiexact: Vec::new(),
            match_sex: false,
            sex_key: "sex".to_string(),
            unique_by_key: None,
            age_limit_years: None,
            require_alive: false,
            require_resident: false,
            estimand: criteria.estimand,
            seed: 0,
        }
    }
}

/// Options of the full method, those of `optmatch::fullmatch()`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FullSpec {
    /// Fewest candidates per anchor in a subclass. A number below 1 allows
    /// several anchors per candidate.
    pub min_controls: f64,
    /// Most candidates per anchor in a subclass. Absent for no limit.
    pub max_controls: Option<f64>,
    /// Fraction of the candidates (positive) or anchors (negative) to leave
    /// out. Not with `mean_controls`.
    pub omit_fraction: Option<f64>,
    /// Candidates per anchor to use on average. Not with `omit_fraction`.
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
#[derive(Debug, Clone, Copy)]
pub enum SpecInput<'a, R> {
    /// Anchors and candidates, for [`MatchMode::Standard`].
    Standard {
        /// The anchors.
        anchors: &'a [R],
        /// The candidates.
        candidates: &'a [R],
    },
    /// One cohort, for [`MatchMode::RiskSet`].
    RiskSet {
        /// The cohort. A unit with a transition date is a case on that date.
        cohort: &'a [RoleTransitionRecord<R>],
    },
}

/// What a spec produced.
#[derive(Debug, Clone)]
pub enum SpecOutput {
    /// Pairs of anchors and controls.
    Pairs {
        /// The pairs and their diagnostics.
        outcome: Box<MatchOutcome>,
        /// The propensity model, when the distance was a fitted score.
        propensity: Option<Box<PropensityScoreEstimate>>,
    },
    /// Strata with weights, from the exact method and from coarsened exact
    /// matching without k:k pairing.
    Exact(Box<ExactMatchOutcome>),
    /// Subclasses with weights, from the quick method.
    Quick(Box<QuickMatchOutcome>),
    /// Subclasses with the weights of each estimand, from the full method.
    Full(Box<FullMatchOutcome>),
    /// Subclasses with weights, from the subclass method.
    Subclass {
        /// The subclasses and their weights.
        outcome: Box<SubclassificationOutcome>,
        /// The propensity model, when the scores were fitted.
        propensity: Option<Box<PropensityScoreEstimate>>,
    },
}

/// Why a spec could not be run.
#[derive(Debug, Clone)]
pub enum SpecError {
    /// The ratio or the birth date window is out of range.
    Criteria(CriteriaValidationError),
    /// One option is out of range, or does not fit the others.
    Option {
        /// The name of the option, as in the spec.
        option: &'static str,
        /// What is wrong and how to fix it.
        message: String,
    },
    /// The units given are not those of the spec's mode.
    InputMode {
        /// The mode of the spec.
        mode: MatchMode,
    },
    /// The distance could not be built from the covariates.
    Distance(EstimationError),
    /// Exact matching could not run.
    Exact(ExactMatchError),
    /// Coarsened exact matching could not run.
    Cem(CemError),
    /// Generalised full matching could not run.
    Quick(QuickMatchError),
    /// Optimal matching could not run.
    Optimal(OptimalError),
    /// The matcher refused the configuration.
    Rejected(ExclusionReason),
}

impl SpecError {
    fn option(option: &'static str, message: impl Into<String>) -> Self {
        Self::Option {
            option,
            message: message.into(),
        }
    }
}

impl Display for SpecError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Criteria(error) => write!(f, "{error}"),
            Self::Option { option, message } => write!(f, "option `{option}`: {message}"),
            Self::InputMode { mode } => match mode {
                MatchMode::Standard => f.write_str(
                    "the spec is for standard matching and needs anchors and candidates; \
                     pass them, or set the mode to risk_set",
                ),
                MatchMode::RiskSet => f.write_str(
                    "the spec is for risk-set matching and needs one cohort; \
                     pass it, or set the mode to standard",
                ),
            },
            Self::Distance(error) => write!(f, "the distance could not be built: {error}"),
            Self::Exact(error) => write!(f, "exact matching could not run: {error}"),
            Self::Cem(error) => write!(f, "coarsened exact matching could not run: {error}"),
            Self::Quick(error) => write!(f, "generalised full matching could not run: {error}"),
            Self::Optimal(error) => write!(f, "optimal matching could not run: {error}"),
            Self::Rejected(reason) => write!(
                f,
                "the matcher refused the configuration ({reason}); \
                 check the options of the nearest method against the distance"
            ),
        }
    }
}

impl Error for SpecError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Criteria(error) => Some(error),
            Self::Distance(error) => Some(error),
            Self::Exact(error) => Some(error),
            Self::Cem(error) => Some(error),
            Self::Quick(error) => Some(error),
            Self::Optimal(error) => Some(error),
            Self::Option { .. } | Self::InputMode { .. } | Self::Rejected(_) => None,
        }
    }
}

impl From<CriteriaValidationError> for SpecError {
    fn from(error: CriteriaValidationError) -> Self {
        Self::Criteria(error)
    }
}

impl From<EstimationError> for SpecError {
    fn from(error: EstimationError) -> Self {
        Self::Distance(error)
    }
}

impl From<CemError> for SpecError {
    fn from(error: CemError) -> Self {
        Self::Cem(error)
    }
}

impl From<ExactMatchError> for SpecError {
    fn from(error: ExactMatchError) -> Self {
        Self::Exact(error)
    }
}

impl From<QuickMatchError> for SpecError {
    fn from(error: QuickMatchError) -> Self {
        Self::Quick(error)
    }
}

impl From<OptimalError> for SpecError {
    fn from(error: OptimalError) -> Self {
        Self::Optimal(error)
    }
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

    /// Whether the distance is one number per unit, which anchors can be
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
) -> Result<Option<DistanceCaliper>, SpecError> {
    value
        .map(|width| {
            DistanceCaliper::new(width).ok_or_else(|| {
                SpecError::option(
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
    /// Check the options against each other, before any unit is read.
    ///
    /// [`run_spec`] does this first.
    ///
    /// # Errors
    ///
    /// A [`SpecError`] naming the option that is out of range or conflicts
    /// with another, and how to fix it.
    pub fn validate(&self) -> Result<(), SpecError> {
        self.criteria().validate()?;
        for ratio in &self.ratio_fallback {
            if MatchRatio::new(*ratio).is_none() {
                return Err(SpecError::option(
                    "ratio_fallback",
                    "a fallback ratio is 0; use ratios of 1 or more",
                ));
            }
        }
        for (field, width) in &self.calipers {
            if !(width.is_finite() && *width >= 0.0) {
                return Err(SpecError::option(
                    "calipers",
                    format!("the caliper on {field} is {width}; use a width of 0 or more"),
                ));
            }
        }
        typed_caliper("distance", self.distance.caliper())?;
        if let Some(mahvars) = &self.nearest.mahvars {
            typed_caliper("nearest.mahvars", mahvars.caliper())?;
        }
        if self.subclass.n == 0 {
            return Err(SpecError::option(
                "subclass.n",
                "the number of subclasses is 0; use 1 or more",
            ));
        }

        if self.cem != CemSpec::default() && self.method != MatchMethod::Cem {
            return Err(SpecError::option(
                "cem",
                "the cem options belong to the cem method; set the method to cem or drop them",
            ));
        }
        if self.quick != QuickSpec::default() && self.method != MatchMethod::Quick {
            return Err(SpecError::option(
                "quick",
                "the size, minimums and seed method belong to the quick method; \
                 set the method to quick or drop them",
            ));
        }
        if self.full != FullSpec::default() && self.method != MatchMethod::Full {
            return Err(SpecError::option(
                "full",
                "min_controls, max_controls, omit_fraction and mean_controls belong to the \
                 full method; set the method to full or drop them",
            ));
        }
        let nearest_options = self.nearest != NearestSpec::default() || self.std_caliper;
        if nearest_options && self.method != MatchMethod::Nearest {
            return Err(SpecError::option(
                "nearest",
                "m_order, mahvars and std_caliper belong to the nearest method; \
                 set the method to nearest or drop them",
            ));
        }
        match self.method {
            MatchMethod::Exact => self.validate_exact()?,
            MatchMethod::Subclass => self.validate_subclass()?,
            MatchMethod::Cem => self.validate_cem()?,
            MatchMethod::Quick => self.validate_quick()?,
            MatchMethod::Optimal | MatchMethod::Full => self.validate_optimal()?,
            MatchMethod::Nearest => self.validate_nearest()?,
            MatchMethod::Random | MatchMethod::LowestIndex | MatchMethod::NearestBirthDate => {
                if self.assignment == Some(AssignmentSpec::Rounds)
                    && !self.ratio_fallback.is_empty()
                {
                    return Err(SpecError::option(
                        "assignment",
                        "rounds cannot be combined with ratio_fallback; \
                         use sequential assignment or drop the fallback",
                    ));
                }
            }
        }
        match self.mode {
            MatchMode::Standard => self.validate_standard(),
            MatchMode::RiskSet => self.validate_risk_set(),
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
            common_support: None,
        }
    }

    fn validate_nearest(&self) -> Result<(), SpecError> {
        if self.mode != MatchMode::Standard {
            return Err(SpecError::option(
                "method",
                "the nearest method ranks candidates by a distance and runs in standard mode; \
                 use random, lowest_index or nearest_birth_date for risk-set matching",
            ));
        }
        if !self.ratio_fallback.is_empty() {
            return Err(SpecError::option(
                "ratio_fallback",
                "the nearest method has no fallback ratios; drop them or use another method",
            ));
        }
        let assignment = self.assignment.unwrap_or(AssignmentSpec::Rounds);
        if matches!(
            self.nearest.m_order,
            Some(OrderSpec::Closest | OrderSpec::Farthest)
        ) && assignment == AssignmentSpec::Sequential
        {
            return Err(SpecError::option(
                "nearest.m_order",
                "closest and farthest need rounds; set assignment to rounds",
            ));
        }
        if matches!(
            self.nearest.m_order,
            Some(OrderSpec::Largest | OrderSpec::Smallest)
        ) && !self.distance.is_score()
        {
            return Err(SpecError::option(
                "nearest.m_order",
                "largest and smallest order the anchors by a score, which a Mahalanobis or \
                 Euclidean distance does not have; use data, random, closest or farthest",
            ));
        }
        if self.std_caliper && !self.distance.is_score() && self.distance.caliper().is_some() {
            return Err(SpecError::option(
                "std_caliper",
                "a caliper in standard deviations needs a score distance; \
                 give the caliper in the units of the distance",
            ));
        }
        Ok(())
    }

    fn validate_optimal(&self) -> Result<(), SpecError> {
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
            return Err(SpecError::option(
                option,
                format!("the {name} method does not use it; drop it"),
            ));
        }
        if self.method == MatchMethod::Full && self.ratio != 1 {
            return Err(SpecError::option(
                "ratio",
                "the full method sizes its subclasses by min_controls and max_controls; \
                 drop the ratio",
            ));
        }
        if self.full.omit_fraction.is_some() && self.full.mean_controls.is_some() {
            return Err(SpecError::option(
                "full",
                "omit_fraction and mean_controls cannot both be set",
            ));
        }
        Ok(())
    }

    fn validate_exact(&self) -> Result<(), SpecError> {
        if self.exact.is_empty() {
            return Err(SpecError::option(
                "exact",
                "the exact method needs the strata keys to match on; name at least one",
            ));
        }
        self.reject_pair_options("exact")
    }

    fn validate_cem(&self) -> Result<(), SpecError> {
        if self.mode != MatchMode::Standard {
            return Err(SpecError::option(
                "method",
                "coarsened exact matching needs anchors and candidates in standard mode",
            ));
        }
        if self.cem.variables.is_empty() {
            return Err(SpecError::option(
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
                return Err(SpecError::option(
                    "cem.variables",
                    format!("the binning of {name} has {problem}"),
                ));
            }
        }
        if !self.exact.is_empty() {
            return Err(SpecError::option(
                "exact",
                "coarsened exact matching takes its variables from cem.variables; drop the keys",
            ));
        }
        if let Some(k2k) = &self.cem.k2k {
            if matches!(k2k.m_order, Some(OrderSpec::Largest | OrderSpec::Smallest)) {
                return Err(SpecError::option(
                    "cem.k2k.m_order",
                    "largest and smallest need a score; use data, random, closest or farthest",
                ));
            }
            if let Some(distance) = &k2k.distance {
                if distance.caliper().is_some() {
                    return Err(SpecError::option(
                        "cem.k2k.distance",
                        "k:k matching pairs within strata and takes no caliper; drop it",
                    ));
                }
                if matches!(distance, DistanceSpec::Date { .. }) {
                    return Err(SpecError::option(
                        "cem.k2k.distance",
                        "give a covariate distance, a score or a matrix",
                    ));
                }
            }
        }
        self.reject_pair_options("cem")
    }

    fn validate_subclass(&self) -> Result<(), SpecError> {
        if !matches!(
            self.distance,
            DistanceSpec::Propensity { .. } | DistanceSpec::Scores { .. }
        ) {
            return Err(SpecError::option(
                "distance",
                "the subclass method needs a propensity score or a score of your own; \
                 set the distance to propensity or scores",
            ));
        }
        if !self.exact.is_empty() {
            return Err(SpecError::option(
                "exact",
                "the subclass method does not take exact keys; drop them",
            ));
        }
        self.reject_pair_options("subclass")
    }

    fn validate_quick(&self) -> Result<(), SpecError> {
        if matches!(self.distance, DistanceSpec::Date { .. }) {
            return Err(SpecError::option(
                "distance",
                "the quick method needs a propensity score, a score of your own, or a \
                 Euclidean or Mahalanobis distance; set the distance",
            ));
        }
        if self.std_caliper {
            return Err(SpecError::option(
                "std_caliper",
                "the quick method takes its caliper in the units of the distance; drop it",
            ));
        }
        self.reject_pair_options("quick")
    }

    /// The options that pair formation reads, which a method that forms no
    /// pairs would silently ignore.
    fn reject_pair_options(&self, method: &str) -> Result<(), SpecError> {
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
            return Err(SpecError::option(
                option,
                format!("the {method} method forms no pairs and does not use it; drop it"),
            ));
        }
        if method == "exact" && !self.distance.is_default() {
            return Err(SpecError::option(
                "distance",
                "the exact method does not use a distance; drop it",
            ));
        }
        if method == "cem" && !self.distance.is_default() {
            return Err(SpecError::option(
                "distance",
                "coarsened exact matching takes its distance from cem.k2k.distance; drop it",
            ));
        }
        Ok(())
    }

    fn validate_standard(&self) -> Result<(), SpecError> {
        for (option, set) in [
            ("require_alive", self.require_alive),
            ("require_resident", self.require_resident),
            ("age_limit_years", self.age_limit_years.is_some()),
        ] {
            if set {
                return Err(SpecError::option(
                    option,
                    "this option is read by risk-set matching only; \
                     set the mode to risk_set or drop it",
                ));
            }
        }
        Ok(())
    }

    fn validate_risk_set(&self) -> Result<(), SpecError> {
        if !matches!(
            self.method,
            MatchMethod::Random | MatchMethod::LowestIndex | MatchMethod::NearestBirthDate
        ) {
            return Err(SpecError::option(
                "method",
                "risk-set matching chooses among the units at risk; \
                 use random, lowest_index or nearest_birth_date",
            ));
        }
        if !matches!(
            self.distance,
            DistanceSpec::Date { .. } | DistanceSpec::Scores { .. }
        ) {
            return Err(SpecError::option(
                "distance",
                "risk-set matching takes a birth date or a score of your own as distance; \
                 covariate distances need anchors and candidates in standard mode",
            ));
        }
        if self
            .assignment
            .is_some_and(|a| a != AssignmentSpec::Sequential)
        {
            return Err(SpecError::option(
                "assignment",
                "risk-set matching assigns sequentially; drop the option",
            ));
        }
        if self
            .age_limit_years
            .is_some_and(|years| AgeLimitYears::new(years).is_none())
        {
            return Err(SpecError::option(
                "age_limit_years",
                "the age limit is 0; use 1 year or more",
            ));
        }
        Ok(())
    }
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
    seed: u64,
    mahvars: bool,
) -> Result<BuiltDistance, SpecError> {
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
        DistanceSpec::Propensity {
            covariates,
            link,
            estimator,
            scale,
            sampling_weights,
            ..
        } => {
            let estimator = match *estimator {
                EstimatorSpec::Glm => PropensityEstimator::GlmLogit,
                EstimatorSpec::ElasticNet {
                    alpha,
                    lambda,
                    standardize,
                } => PropensityEstimator::ElasticNetLogit(
                    ElasticNetLogisticConfig::builder()
                        .alpha(alpha)
                        .lambda(lambda)
                        .standardize(standardize)
                        .build(),
                ),
                EstimatorSpec::ElasticNetCv {
                    alpha,
                    nfolds,
                    nlambda,
                    rule,
                    standardize,
                } => PropensityEstimator::ElasticNetCv(
                    ElasticNetCvConfig::builder()
                        .seed(seed)
                        .alpha(alpha)
                        .nfolds(nfolds)
                        .nlambda(nlambda)
                        .rule(match rule {
                            LambdaSpec::Min => LambdaRule::Min,
                            LambdaSpec::OneStandardError => LambdaRule::OneStandardError,
                        })
                        .standardize(standardize)
                        .build(),
                ),
            };
            let logistic = LogisticRegressionConfig::builder()
                .link(match link {
                    LinkSpec::Logit => PropensityLink::Logit,
                    LinkSpec::Probit => PropensityLink::Probit,
                    LinkSpec::Cloglog => PropensityLink::Cloglog,
                })
                .build();
            let config = PropensityScoreConfig::builder()
                .encoding(covariates.encoding(true))
                .estimator(estimator)
                .logistic(logistic)
                .output_scale(match scale {
                    ScaleSpec::Probability => PropensityScoreOutputScale::Probability,
                    ScaleSpec::Linear => PropensityScoreOutputScale::LinearPredictor,
                })
                .maybe_caliper(caliper)
                .maybe_sampling_weights(sampling_weights.as_ref().map(|weights| {
                    weights
                        .iter()
                        .map(|(id, weight)| (id.clone(), *weight))
                        .collect::<RapidHashMap<_, _>>()
                }))
                .build();
            let prepared = prepare_propensity_distance_config(anchors, candidates, &config)?;
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
            let config = MahalanobisPreparationConfig::builder()
                .encoding(covariates.encoding(false))
                .covariance_strategy(match covariance {
                    CovarianceSpec::PooledByDegreesOfFreedom => {
                        MahalanobisCovarianceStrategy::PooledByDegreesOfFreedom
                    }
                    CovarianceSpec::PooledWithinGroups => {
                        MahalanobisCovarianceStrategy::PooledWithinGroups
                    }
                    CovarianceSpec::FullSample => MahalanobisCovarianceStrategy::FullSample,
                })
                .transform(match transform {
                    TransformSpec::Raw => MahalanobisTransform::Raw,
                    TransformSpec::Rank => MahalanobisTransform::Rank,
                    TransformSpec::RankTieAdjusted => MahalanobisTransform::RankTieAdjusted,
                })
                .build();
            BuiltDistance {
                config: Some(
                    prepare_mahalanobis_distance_config(anchors, candidates, caliper, &config)?
                        .distance_config,
                ),
                propensity: None,
            }
        }
        DistanceSpec::Matrix {
            anchor_ids,
            candidate_ids,
            values,
            ..
        } => {
            let flat = values
                .iter()
                .flat_map(|row| row.iter().map(|value| value.unwrap_or(f64::INFINITY)))
                .collect::<Vec<_>>();
            if values.iter().any(|row| row.len() != candidate_ids.len()) {
                return Err(SpecError::option(
                    "distance",
                    "every row of the distance matrix needs one value for each candidate id",
                ));
            }
            let matrix = PairDistanceMatrix::new(anchor_ids.clone(), candidate_ids.clone(), flat)
                .map_err(|error| SpecError::option("distance", error.to_string()))?;
            BuiltDistance {
                config: Some(DistanceConfig::matrix(matrix, caliper)),
                propensity: None,
            }
        }
        DistanceSpec::Euclidean {
            covariates, scaled, ..
        } => {
            let config = EuclideanPreparationConfig::builder()
                .encoding(covariates.encoding(false))
                .scale(if *scaled {
                    EuclideanScale::PooledWithinGroupSd
                } else {
                    EuclideanScale::None
                })
                .build();
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
) -> Result<QuickMatchOutcome, SpecError> {
    let outcome = match config {
        DistanceConfig::PropensityScoreMap { scores, .. } => quick_match_subclasses(
            anchors,
            candidates,
            |row: &R| scores.get(row.id()).map(std::slice::from_ref),
            1,
            exact,
            options,
        ),
        DistanceConfig::EuclideanMap {
            vectors, dimension, ..
        } => quick_match_subclasses(
            anchors,
            candidates,
            |row: &R| vectors.get(row.id()).map(Vec::as_slice),
            *dimension,
            exact,
            options,
        ),
        DistanceConfig::MahalanobisMap {
            vectors,
            inverse_covariance,
            dimension,
            ..
        } => {
            let factor = cholesky_lower(inverse_covariance, *dimension).ok_or_else(|| {
                SpecError::option(
                    "distance",
                    "the inverse covariance matrix is not positive definite; \
                     drop covariates that are collinear or constant",
                )
            })?;
            let embedded: RapidHashMap<String, Vec<f64>> = vectors
                .iter()
                .map(|(id, vector)| (id.clone(), apply_transposed(&factor, vector, *dimension)))
                .collect();
            quick_match_subclasses(
                anchors,
                candidates,
                |row: &R| embedded.get(row.id()).map(Vec::as_slice),
                *dimension,
                exact,
                options,
            )
        }
        DistanceConfig::Date { .. } => {
            return Err(SpecError::option(
                "distance",
                "the quick method needs a score or a covariate distance",
            ));
        }
        DistanceConfig::Matrix { .. } => {
            return Err(SpecError::option(
                "distance",
                "the quick method clusters points and cannot take a distance matrix; \
                 give it a score or the covariates",
            ));
        }
    };
    Ok(outcome?)
}

/// The scores of a distance for subclassification.
fn scores_of(config: &DistanceConfig) -> Option<&RapidHashMap<String, f64>> {
    match config {
        DistanceConfig::PropensityScoreMap { scores, .. } => Some(scores),
        DistanceConfig::Date { .. }
        | DistanceConfig::MahalanobisMap { .. }
        | DistanceConfig::EuclideanMap { .. }
        | DistanceConfig::Matrix { .. } => None,
    }
}

/// Fail on an outcome that the matcher built to say the job was invalid.
fn reject_invalid(outcome: MatchOutcome) -> Result<MatchOutcome, SpecError> {
    let rejected = outcome
        .diagnostics
        .exclusion_counts
        .keys()
        .find(|reason| {
            matches!(
                reason,
                ExclusionReason::InvalidCriteria(_) | ExclusionReason::CommonSupportFailure(_)
            )
        })
        .cloned();
    match rejected {
        Some(reason) => Err(SpecError::Rejected(reason)),
        None => Ok(outcome),
    }
}

/// Calipers on fields with their widths, in standard deviations if asked.
fn field_calipers<R: MatchingRecord>(
    spec: &MatchSpec,
    anchors: &[R],
    candidates: &[R],
) -> Result<Vec<(String, f64)>, SpecError> {
    spec.calipers
        .iter()
        .map(|(field, width)| {
            if !spec.std_caliper {
                return Ok((field.clone(), *width));
            }
            let sd = numeric_field_sd(anchors, candidates, field).ok_or_else(|| {
                SpecError::option(
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

/// What a typed standard job needs beyond the units.
struct StandardPlan<'a, R> {
    anchors: &'a [R],
    candidates: &'a [R],
    spec: &'a MatchSpec,
    distance: Option<DistanceConfig>,
    calipers: Vec<(String, f64)>,
    rounds: bool,
}

fn typed_ratio(spec: &MatchSpec) -> Result<MatchRatio, SpecError> {
    Ok(MatchRatio::new(spec.ratio).ok_or(CriteriaValidationError::ZeroMatchRatio)?)
}

fn typed_window(spec: &MatchSpec) -> Result<BirthDateWindowDays, SpecError> {
    Ok(BirthDateWindowDays::new(spec.birth_window_days)
        .ok_or(CriteriaValidationError::NegativeBirthDateWindow)?)
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
) -> Result<MatchOutcome, SpecError>
where
    R: MatchingRecord,
    S: SelectionStrategy<R> + Clone + Send + Sync,
{
    let spec = plan.spec;
    let sex: Vec<GenderMatch> = if spec.match_sex {
        vec![GenderMatch::on_key(spec.sex_key.clone())]
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
    let mut job = MatchJob::new_standard(plan.anchors, plan.candidates, spec.seed)
        .with_ratio(typed_ratio(spec)?)
        .with_birth_window(typed_window(spec)?)
        .with_replacement(spec.replace)
        .with_estimand(spec.estimand)
        .with_exact_matches(spec.exact.iter().cloned())
        .with_unique_by_key(spec.unique_by_key.clone())
        .with_ratio_fallback(typed_fallback(spec))
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
    reject_invalid(outcome)
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
/// A [`SpecError`] when the spec is invalid, when the units are not those of
/// the spec's mode, when a distance cannot be built from the covariates, or
/// when exact matching cannot run.
pub fn run_spec<R>(spec: &MatchSpec, input: SpecInput<'_, R>) -> Result<SpecOutput, SpecError>
where
    R: MatchingRecord + CovariateRecord,
{
    spec.validate()?;
    match (spec.mode, input) {
        (
            MatchMode::Standard,
            SpecInput::Standard {
                anchors,
                candidates,
            },
        ) => run_standard(spec, anchors, candidates),
        (MatchMode::RiskSet, SpecInput::RiskSet { cohort }) => run_risk_set(spec, cohort),
        (mode, _) => Err(SpecError::InputMode { mode }),
    }
}

fn run_standard<R: MatchingRecord + CovariateRecord>(
    spec: &MatchSpec,
    anchors: &[R],
    candidates: &[R],
) -> Result<SpecOutput, SpecError> {
    match spec.method {
        MatchMethod::Exact => {
            let outcome = exact_match_subclasses(anchors, candidates, &spec.exact)?;
            return Ok(SpecOutput::Exact(Box::new(outcome)));
        }
        MatchMethod::Cem => return run_cem(spec, anchors, candidates),
        MatchMethod::Subclass => {
            let built = build_distance(&spec.distance, anchors, candidates, spec.seed, false)?;
            let scores = built
                .config
                .as_ref()
                .and_then(scores_of)
                .ok_or_else(|| SpecError::option("distance", "no scores were built"))?;
            let config = SubclassificationConfig {
                subclasses: spec.subclass.n,
                reference_group: spec.subclass.reference.map(Into::into),
                min_n: spec.subclass.min_n,
            };
            let outcome = subclassify_by_propensity_score_map(
                anchors,
                candidates,
                scores,
                spec.estimand,
                &config,
            );
            return Ok(SpecOutput::Subclass {
                outcome: Box::new(outcome),
                propensity: built.propensity.map(Box::new),
            });
        }
        MatchMethod::Quick => {
            let built = build_distance(&spec.distance, anchors, candidates, spec.seed, false)?;
            let config = built
                .config
                .as_ref()
                .ok_or_else(|| SpecError::option("distance", "no distance was built"))?;
            let options = QuickMatchOptions {
                min_anchors: spec.quick.min_anchors,
                min_candidates: spec.quick.min_candidates,
                size: spec.quick.size,
                caliper: spec.distance.caliper(),
                seed_method: spec.quick.seed_method,
            };
            let outcome = run_quick(config, anchors, candidates, &spec.exact, &options)?;
            return Ok(SpecOutput::Quick(Box::new(outcome)));
        }
        MatchMethod::Optimal | MatchMethod::Full => {
            let built = build_distance(&spec.distance, anchors, candidates, spec.seed, false)?;
            let calipers = field_calipers(spec, anchors, candidates)?;
            let sex: Vec<GenderMatch> = if spec.match_sex {
                vec![GenderMatch::on_key(spec.sex_key.clone())]
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
            let outcome = match_optimal(anchors, candidates, &request)?;
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

    let built = build_distance(&spec.distance, anchors, candidates, spec.seed, false)?;
    let calipers = field_calipers(spec, anchors, candidates)?;
    let rounds = spec.assignment == Some(AssignmentSpec::Rounds);
    let plan = |rounds| StandardPlan {
        anchors,
        candidates,
        spec,
        distance: built.config.clone(),
        calipers: calipers.clone(),
        rounds,
    };
    let outcome = match spec.method {
        MatchMethod::Random => run_standard_with(plan(rounds), RandomSelection::seeded(spec.seed))?,
        MatchMethod::LowestIndex => run_standard_with(plan(rounds), DeterministicSelection)?,
        MatchMethod::NearestBirthDate => {
            run_standard_with(plan(rounds), NearestBirthDateSelection)?
        }
        MatchMethod::Nearest => {
            let mut strategy = NearestSelection::new(spec.seed).with_std_caliper(spec.std_caliper);
            if let Some(order) = spec.nearest.m_order {
                strategy = strategy.with_order(order.into());
            }
            if let Some(assignment) = spec.assignment {
                strategy = strategy.with_assignment(assignment.into());
            }
            if let Some(ranking) = &spec.nearest.mahvars {
                let ranking = build_distance(ranking, anchors, candidates, spec.seed, true)?;
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
        | MatchMethod::Full => {
            unreachable!("handled above")
        }
    };
    Ok(SpecOutput::Pairs {
        outcome: Box::new(outcome),
        propensity: built.propensity.map(Box::new),
    })
}

fn run_cem<R: MatchingRecord + CovariateRecord>(
    spec: &MatchSpec,
    anchors: &[R],
    candidates: &[R],
) -> Result<SpecOutput, SpecError> {
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
        .map(|distance| build_distance(distance, anchors, candidates, spec.seed, false))
        .transpose()?;
    let order = k2k.m_order.map_or(MatchOrder::Data, Into::into);
    let outcome = cem_k2k_pairs(
        anchors,
        candidates,
        &strata,
        built.as_ref().and_then(|built| built.config.as_ref()),
        order,
        spec.seed,
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

fn run_risk_set<R: MatchingRecord + CovariateRecord>(
    spec: &MatchSpec,
    cohort: &[RoleTransitionRecord<R>],
) -> Result<SpecOutput, SpecError> {
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
            return Err(SpecError::option(
                "distance",
                "not available in risk-set mode",
            ));
        }
    };

    let sex: Vec<GenderMatch> = if spec.match_sex {
        vec![GenderMatch::on_key(spec.sex_key.clone())]
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

    let mut job = MatchJob::new_transition(cohort, spec.seed)
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
    let outcome = match spec.method {
        MatchMethod::Random => job.with_strategy(RandomSelection::seeded(spec.seed)).run(),
        MatchMethod::LowestIndex => job.with_strategy(DeterministicSelection).run(),
        MatchMethod::NearestBirthDate => job.with_strategy(NearestBirthDateSelection).run(),
        MatchMethod::Nearest
        | MatchMethod::Exact
        | MatchMethod::Subclass
        | MatchMethod::Cem
        | MatchMethod::Quick
        | MatchMethod::Optimal
        | MatchMethod::Full => {
            unreachable!("rejected by validation")
        }
    };
    Ok(SpecOutput::Pairs {
        outcome: Box::new(reject_invalid(outcome)?),
        propensity: None,
    })
}
