//! The one error of the crate.
//!
//! Every matcher, the spec, the outputs, the balance, effect, sensitivity and
//! imputation functions and the sequential trials return [`Error`]. Invalid
//! input whose payload the shared type can carry is [`Error::Input`] with
//! [`InputError`], so a column of the wrong length is reported as in every
//! other epistat crate. Input errors that name a unit, a column or a variable
//! by the caller's label have variants of their own. The variants after them
//! are failures of a method on accepted input. [`Classify`] gives a binding the
//! kind and a stable code.

use crate::spec::MatchMode;
use crate::types::{CommonSupportFailureReason, InvalidCriteriaReason};
use epistat_core::{Classify, ErrorKind, InputError};

/// Why a matching, an output or an estimate after matching could not be
/// formed.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    /// The input cannot be used: an argument of the wrong length, a value
    /// that is not finite, an argument outside its range.
    #[error(transparent)]
    Input(#[from] InputError),
    /// An option of a method is out of range or contradicts another, by the
    /// option's name.
    #[error("option `{option}`: {message}")]
    Option {
        /// The option, by the name the spec gives it.
        option: &'static str,
        /// What is wrong with it.
        message: String,
    },
    /// The birth-date window is negative.
    #[error("the birth date window is negative; use a window of 0 days or more")]
    NegativeBirthDateWindow,
    /// The match ratio is 0.
    #[error("the match ratio is 0; use a ratio of 1 or more")]
    ZeroMatchRatio,
    /// The job's criteria are invalid, such as a fallback ratio above the
    /// primary ratio or nearest-neighbour options that contradict the
    /// distance.
    #[error(
        "the matcher refused the configuration ({0}); \
         check the options of the method against the distance"
    )]
    InvalidCriteria(InvalidCriteriaReason),
    /// Common support was asked for and could not be set.
    #[error("common support could not be set ({0})")]
    CommonSupport(CommonSupportFailureReason),
    /// The spec's mode needs input of another shape than was given.
    #[error("{}", input_mode_message(*mode))]
    InputMode {
        /// The mode of the spec.
        mode: MatchMode,
    },
    /// Two units share an id where every unit needs its own.
    #[error("the id `{id}` occurs more than once; give every unit its own id")]
    DuplicateId {
        /// The id.
        id: String,
    },
    /// A case's event date is before its birth date.
    #[error("the event date of `{id}` is before its birth date")]
    EventBeforeBirth {
        /// The id of the case.
        id: String,
    },
    /// A column has `found` rows where the ids have `expected`.
    #[error("column {column} has {found} rows, but there are {expected} ids")]
    ColumnLength {
        /// The column.
        column: String,
        /// Its number of rows.
        found: usize,
        /// The number of ids.
        expected: usize,
    },
    /// Two columns of the same kind have the same name.
    #[error("two columns of the same kind are called {column}")]
    DuplicateColumn {
        /// The name.
        column: String,
    },
    /// A column has more distinct values than a cohort can code.
    #[error("column {column} has more than {} distinct values", u32::MAX - 1)]
    TooManyValues {
        /// The column.
        column: String,
    },
    /// A column is not in the record batch.
    #[error("column {column} is not in the record batch; it has {}", present.join(", "))]
    MissingColumn {
        /// The column asked for.
        column: String,
        /// The columns the batch has.
        present: Vec<String>,
    },
    /// No case, or no control, has a value under a strata key a constraint
    /// requires, so every pair would be refused on it.
    #[error(
        "no {group} has a value under the strata key `{key}`; store the stratum under that key \
         or name the key the records have"
    )]
    MissingStratum {
        /// The strata key.
        key: &'static str,
        /// `"case"` or `"control"`.
        group: &'static str,
    },
    /// A column has another type than the one asked for.
    #[error(
        "column {column} has type {found}, expected {expected}; cast it before building the records"
    )]
    WrongType {
        /// The column.
        column: String,
        /// Its type.
        found: String,
        /// The type needed.
        expected: &'static str,
    },
    /// A column has a null where every record needs a value.
    #[error("column {column} has a null at row {row}; every record needs a value there")]
    NullValue {
        /// The column.
        column: String,
        /// The row.
        row: usize,
    },
    /// A date column has a date out of the range of a calendar date.
    #[error("column {column} has a date at row {row} out of range")]
    DateOutOfRange {
        /// The column.
        column: String,
        /// The row.
        row: usize,
    },
    /// A value of an argument is not finite, or a weight is negative.
    #[error("{what} in row {row} is not finite or is a negative weight")]
    Value {
        /// The argument.
        what: &'static str,
        /// The row.
        row: usize,
    },
    /// A weight is negative or not finite.
    #[error("the weight {weight} of {id} is not a finite number of at least 0")]
    InvalidWeight {
        /// The unit.
        id: String,
        /// The weight.
        weight: f64,
    },
    /// A matched unit is not among the units given.
    #[error("the matched unit {id} is not in the unit data")]
    UnknownUnit {
        /// The unit.
        id: String,
    },
    /// A pair names a case that is not a case among the units given.
    #[error(
        "a pair has the case {id}, which is not a case among the units given; \
         pass the units the match was run on"
    )]
    UnknownCase {
        /// The case.
        id: String,
    },
    /// A pair names a control that is not a control among the units given.
    #[error(
        "a pair has the control {id}, which is not a control among the units given; \
         pass the units the match was run on"
    )]
    UnknownControl {
        /// The control.
        id: String,
    },
    /// An output needs every unit to keep one role, and a unit is a case in
    /// one pair and a control in another.
    #[error(
        "{output} needs each unit to be a case or a control throughout; \
         these units have a role in each pair"
    )]
    NoFixedRoles {
        /// The output.
        output: &'static str,
    },
    /// The selection was not paired.
    #[error("the selection was not paired; give mahvars to pair the selected units")]
    NotPaired,
    /// An exposure-density pair does not match the cohort.
    #[error("the pair `{case_id}`, `{control_id}` does not match the cohort")]
    InconsistentPair {
        /// The case.
        case_id: String,
        /// The control.
        control_id: String,
    },
    /// The periods of a person do not increase.
    #[error(
        "the periods of `{id}` do not increase: period {period} does not follow the one before it"
    )]
    PeriodsNotIncreasing {
        /// The person.
        id: String,
        /// The first period out of order.
        period: i64,
    },
    /// A period has more than one date where every person needs the same.
    #[error(
        "period {period} has more than one date; every person needs the same date for a period"
    )]
    InconsistentDate {
        /// The period.
        period: i64,
    },
    /// A person's dates do not increase with the period.
    #[error("the date of period {period} is not after the date of the period before it")]
    DatesNotIncreasing {
        /// The first period whose date is not after the one before.
        period: i64,
    },
    /// Period strata do not have one value for every row of the panel.
    #[error(
        "the period strata have {found} values for the {expected} rows of the panel; \
         give one per row, in the order of the panel"
    )]
    StrataRows {
        /// The values needed.
        expected: usize,
        /// The values given.
        found: usize,
    },
    /// A member of a matched trial is not enrolled in it.
    #[error(
        "`{id}` is a member of the trial of period {period} but is not enrolled in it; \
         give the panel and options the trials were matched with"
    )]
    UnknownMember {
        /// The member.
        id: String,
        /// The period at which the trial starts.
        period: i64,
    },
    /// The matching of one trial failed.
    #[error("the trial that starts at period {period}: {source}")]
    Trial {
        /// The period at which the trial starts.
        period: i64,
        /// Why its matching failed.
        source: Box<Error>,
    },
    /// No variable to match on exactly or to coarsen was given.
    #[error("no variable to match on was given")]
    NoVariables,
    /// A unit has no value for a variable it is matched on.
    #[error("unit {id} has no value for the variable {variable}")]
    MissingValue {
        /// The unit.
        id: String,
        /// The variable.
        variable: String,
    },
    /// No stratum holds units of both groups.
    #[error("no stratum holds units of both groups")]
    NoSharedStratum,
    /// The binning of a coarsened variable is invalid.
    #[error("the binning of {variable} is invalid: {message}")]
    Binning {
        /// The variable.
        variable: String,
        /// What is wrong with the binning.
        message: String,
    },
    /// The largest and smallest orders of coarsened exact matching need a
    /// score.
    #[error("the largest and smallest orders need a score; use data, random, closest or farthest")]
    CemOrder,
    /// One of the two groups is empty, or too small for the distance.
    #[error(
        "one of the two groups is empty, or too small for the distance; both groups need units"
    )]
    EmptyGroup,
    /// A covariate has missing values and the policy does not impute.
    #[error(
        "the covariate {covariate} has missing values; \
         remove them or choose the imputing missing-value policy"
    )]
    MissingValues {
        /// The covariate.
        covariate: String,
    },
    /// A numeric covariate is not finite for a unit.
    #[error(
        "the covariate {covariate} is not finite for unit {id}; \
         give it a finite value or mark it missing"
    )]
    NonFiniteCovariate {
        /// The unit.
        id: String,
        /// The covariate.
        covariate: String,
    },
    /// A covariate is numeric for some units and text for others.
    #[error(
        "the covariate {covariate} is numeric for some units and text for others; \
         give it one type"
    )]
    MixedCovariateType {
        /// The covariate.
        covariate: String,
    },
    /// A covariate has no observed value.
    #[error("the covariate {covariate} has no observed value")]
    NoObservedValues {
        /// The covariate.
        covariate: String,
    },
    /// There is no covariate that varies to match on.
    #[error("there is no covariate that varies to match on")]
    NoCovariates,
    /// A covariate has no variation.
    #[error("covariate {0} has no variation")]
    ConstantCovariate(usize),
    /// A column does not vary within either group, so it cannot be scaled by
    /// its within-group standard deviation.
    #[error(
        "the column {column} does not vary within either group, \
         so it cannot be scaled by its within-group standard deviation; \
         drop it or do not scale"
    )]
    ZeroWithinGroupVariance {
        /// The column.
        column: String,
    },
    /// The elastic net settings are out of range.
    #[error(
        "the elastic net settings are out of range; \
         alpha must lie in [0, 1] and lambda must not be negative"
    )]
    InvalidElasticNet,
    /// The model's iteration settings are out of range.
    #[error(
        "the model settings are out of range; \
         the iteration limit must be at least 1 and the tolerance positive"
    )]
    InvalidModelControl,
    /// The cross-validation settings are out of range, or a training set
    /// lacks one of the two groups.
    #[error(
        "the cross-validation settings are out of range, \
         or a training set lacks one of the two groups; use at least 3 folds"
    )]
    InvalidCrossValidation,
    /// A unit's sampling weight is missing, negative or not finite.
    #[error(
        "the sampling weight of unit {id} is missing, negative or not finite; \
         every unit needs a finite weight of 0 or more"
    )]
    InvalidSamplingWeight {
        /// The unit.
        id: String,
    },
    /// Refitting the propensity score needs a common-support rule.
    #[error(
        "refitting the propensity score needs a common-support rule; \
         choose which group to discard outside the overlap"
    )]
    ReestimateWithoutDiscard,
    /// The propensity scores of the two groups do not overlap.
    #[error(
        "the propensity scores of the two groups do not overlap, \
         so no common-support rule can be applied"
    )]
    NoCommonSupport,
    /// The covariate balancing propensity score needs an intercept column.
    #[error(
        "the covariate balancing propensity score needs an intercept column; \
         include the intercept in the encoding"
    )]
    CbpsNeedsIntercept,
    /// The covariates are not of full rank, or there are no more units than
    /// columns.
    #[error(
        "the covariates are not of full rank, or there are no more units than columns; \
         drop collinear covariates"
    )]
    NotFullRank,
    /// The covariate balancing settings are out of range.
    #[error(
        "the covariate balancing settings are out of range: \
         the iteration limit must be at least 1, the tolerance positive, \
         and both groups need weight"
    )]
    InvalidCbps,
    /// The covariance matrix of the covariates is not valid.
    #[error(
        "the covariance matrix of the covariates is not valid; \
         check for constant or perfectly collinear covariates"
    )]
    InvalidCovariance,
    /// The distance could not be built from the covariates.
    #[error("the distance could not be built from the covariates")]
    DistanceBuild,
    /// A Mahalanobis distance has no covariates.
    #[error("the distance has no covariates; give at least one")]
    ZeroDimension,
    /// The inverse covariance matrix is not square in the number of
    /// covariates.
    #[error("the inverse covariance matrix is not square in the number of covariates")]
    InverseCovarianceLength,
    /// The vector values are not a whole number of vectors.
    #[error("the vector values are not a whole number of vectors of the given length")]
    VectorLength,
    /// The inverse covariance matrix is not positive semi-definite.
    #[error("the inverse covariance matrix is not positive semi-definite, so it gives no distance")]
    NotPositiveSemiDefinite,
    /// A distance matrix has no number for a pair.
    #[error(
        "the distance matrix has no number at row {case}, column {control}; \
         use an infinite value to forbid a pair"
    )]
    DistanceNotANumber {
        /// The row of the case.
        case: usize,
        /// The column of the control.
        control: usize,
    },
    /// A distance matrix has a negative distance.
    #[error("the distance matrix has a negative distance at row {case}, column {control}")]
    NegativeDistance {
        /// The row of the case.
        case: usize,
        /// The column of the control.
        control: usize,
    },
    /// Coordinates do not make whole units.
    #[error("the coordinates do not make whole units")]
    Coordinates,
    /// A unit has a coordinate that is not finite.
    #[error("unit {unit} has a coordinate that is not finite")]
    NonFiniteCoordinate {
        /// The position of the unit.
        unit: usize,
    },
    /// A unit has no coordinates of the right length.
    #[error("unit {id} has no coordinates of the right length")]
    MissingCoordinates {
        /// The unit.
        id: String,
    },
    /// A subclass minimum is below one.
    #[error(
        "a subclass must hold at least one case and one control; \
         use minimums of 1 or more"
    )]
    SubclassMinimum,
    /// The size constraint is smaller than the minimums together.
    #[error("the size constraint is smaller than the minimum cases and controls together")]
    SizeBelowMinimums,
    /// A caliper is not positive and finite.
    #[error("the caliper must be positive and finite")]
    InvalidCaliper,
    /// Too few units to make one subclass, in a stratum or overall.
    #[error("{}", too_few_units_message(stratum.as_deref()))]
    TooFewUnits {
        /// The stratum, `None` without strata.
        stratum: Option<String>,
    },
    /// No unit fits a subclass within the caliper.
    #[error("no unit fits a subclass within the caliper")]
    NoSubclass,
    /// There are more units than can be indexed.
    #[error("there are more units than can be indexed")]
    TooManyUnits,
    /// The number of subclasses is 0.
    #[error("the number of subclasses must be at least 1")]
    NoSubclasses,
    /// The largest ratio of reconstructed matching is 0.
    #[error("max_ratio must be at least 1")]
    ZeroMaxRatio,
    /// The caliper of reconstructed matching is negative.
    #[error("match_caliper_days is {0}, which is negative")]
    NegativeCaliperDays(i64),
    /// The average ratio of variable-ratio matching is not above 1.
    #[error("the average ratio must be a number above 1")]
    AverageNotAboveOne,
    /// The minimum number of controls is below 1.
    #[error("the minimum number of controls must be at least 1")]
    MinimumBelowOne,
    /// The minimum number of controls is not below the average ratio.
    #[error("the minimum number of controls must be below the average ratio")]
    MinimumNotBelowAverage,
    /// The maximum number of controls is not above the average ratio.
    #[error("the maximum number of controls must be above the average ratio")]
    MaximumNotAboveAverage,
    /// Malformed input, as text.
    #[error("malformed input: {0}")]
    Shape(String),
    /// A value is not finite, as text.
    #[error("{0} is not finite")]
    NotFinite(String),
    /// An argument is invalid, as text.
    #[error("{0}")]
    Invalid(String),
    /// A tolerance is negative.
    #[error("a tolerance is negative")]
    NegativeTolerance,
    /// The ratio of cardinality matching is invalid.
    #[error("invalid ratio: {0}")]
    InvalidRatio(String),
    /// A distance tolerance is not finite and non-negative.
    #[error("the distance tolerance must be finite and non-negative")]
    InvalidTolerance,
    /// The units are all treated or all untreated.
    #[error("the units are all treated or all untreated")]
    OneGroup,
    /// More matches were asked for than a group has.
    #[error("{asked} matches asked for from a group of {available}")]
    TooManyMatches {
        /// The matches asked for.
        asked: usize,
        /// The units of the group.
        available: usize,
    },
    /// More neighbours were asked for the variance than a group has.
    #[error("{asked} neighbours asked for the variance in a group of {group}")]
    TooManyVarianceMatches {
        /// The neighbours asked for.
        asked: usize,
        /// The units of the group.
        group: usize,
    },
    /// The number of matches is 0.
    #[error("the number of matches is zero")]
    ZeroMatches,
    /// Matched sets after matching with replacement need the long form.
    #[error(
        "row {row} has no pair; after matching with replacement build the sample from \
         the long form of the matches"
    )]
    MissingPair {
        /// The row.
        row: usize,
    },
    /// The target has no weighted treated rows or no weighted control rows.
    #[error("the target has no weighted treated rows or no weighted control rows")]
    EmptyTarget,
    /// An outcome is outside [0, 1] for a logistic model.
    #[error("the outcome in row {row} is outside [0, 1] for a logistic model")]
    Outcome {
        /// The row.
        row: usize,
    },
    /// The model has more coefficients than rows.
    #[error("the model has more coefficients than rows")]
    TooFewRows,
    /// The bootstrap needs at least two replicates and two pairs.
    #[error("the bootstrap needs at least two replicates and at least two pairs")]
    Bootstrap,
    /// Fewer than two imputed data sets were given.
    #[error("{found} imputed data sets were given; pooling needs at least 2")]
    TooFewSets {
        /// The number given.
        found: usize,
    },
    /// A unit of an imputed data set is not in the first.
    #[error(
        "unit {id} of data set {set} is not in the first data set; \
         the across approach needs the same units in every set"
    )]
    UnitsDiffer {
        /// The data set.
        set: usize,
        /// The unit.
        id: String,
    },
    /// The group labels do not lie in `0..groups` with at least two groups,
    /// or the focal group is not one of them.
    #[error(
        "the group labels must lie in 0..{groups} with at least two groups, \
         and the focal group must be one of them"
    )]
    Groups {
        /// The number of groups.
        groups: usize,
    },
    /// The number of clusters is 0.
    #[error("the number of clusters must be at least 1")]
    Clusters,
    /// No unit of a group lies inside the common support.
    #[error("no unit of group {group} lies inside the common support")]
    EmptySupport {
        /// The group.
        group: usize,
    },
    /// A matched set does not have one treated unit.
    #[error("set {set} has {treated} treated units, not one")]
    NotOneTreated {
        /// The set.
        set: usize,
        /// Its treated units.
        treated: usize,
    },
    /// A matched set has no control.
    #[error("set {set} has no control")]
    NoControl {
        /// The set.
        set: usize,
    },
    /// A matched set has more than one control where pairs are needed.
    #[error("set {set} has {controls} controls, not one")]
    NotAPair {
        /// The set.
        set: usize,
        /// Its controls.
        controls: usize,
    },
    /// A unit has more than one row in the matched sets.
    #[error("unit {id} has more than one row in the matched sets, so the sets are not independent")]
    ReusedUnit {
        /// The unit.
        id: String,
    },
    /// There are no matched sets.
    #[error("there are no matched sets")]
    NoSets,
    /// An outcome is not 0 or 1.
    #[error("the outcome {0} is not 0 or 1")]
    NotBinary(f64),
    /// The method does not estimate the estimand asked for.
    #[error("{method} does not estimate the {estimand}")]
    UnsupportedEstimand {
        /// The estimand asked for.
        estimand: epistat_core::Estimand,
        /// The method.
        method: &'static str,
    },
    /// Sampling weights need profile matching.
    #[error(
        "sampling weights need profile matching (ratio unset or the ATE): \
         cardinality matching has no fixed target population"
    )]
    SamplingWeightsNeedProfile,
    /// Pairing needs the ATT or the ATE and a whole-number ratio.
    #[error("pairing needs the ATT or the ATE and a whole-number ratio")]
    PairingNeedsWholeRatio,
    /// Reestimating the score is not available with the across approach.
    #[error(
        "reestimate is not available with the across approach; \
         the score is fitted in each set before it is averaged"
    )]
    ReestimateAcross,
    /// No case and control form an admissible pair.
    #[error("no case and control form an admissible pair")]
    NoMatches,
    /// No matching satisfies the limits in a stratum.
    #[error("no matching satisfies the limits in stratum {stratum}: {message}")]
    Infeasible {
        /// The stratum.
        stratum: usize,
        /// Why.
        message: String,
    },
    /// A stratum is too large to solve.
    #[error("stratum {stratum} is too large to solve")]
    TooLarge {
        /// The stratum.
        stratum: usize,
    },
    /// No subset satisfies the balance constraints in a stratum.
    #[error(
        "no subset satisfies the balance constraints in stratum {stratum}; relax the tolerances"
    )]
    InfeasibleBalance {
        /// The stratum, by its key.
        stratum: u64,
    },
    /// The solver found no subset within its limits.
    #[error("the solver found no subset in stratum {stratum} within its limits")]
    NoSolutionInTime {
        /// The stratum, by its key.
        stratum: u64,
    },
    /// The solver failed.
    #[error("the solver failed: {0}")]
    Solver(String),
    /// A group has no variance to standardise by.
    #[error("a group has no variance to standardise by")]
    ZeroVarianceGroup,
    /// A covariance of the covariates is singular.
    #[error("the covariance of the covariates is singular")]
    SingularCovariance,
    /// The regression for the bias adjustment is singular.
    #[error("the regression for the bias adjustment is singular")]
    SingularRegression,
    /// A linear system in a fit could not be solved.
    #[error(
        "a linear system in the fit could not be solved; check the covariates for perfect collinearity"
    )]
    LinearSystem,
    /// A generalised propensity score is 0 or 1 to machine precision.
    #[error(
        "a generalised propensity score is 0 or 1 to machine precision; \
         the groups are separated by the covariates"
    )]
    Separated,
    /// A coefficient of an outcome model is aliased with the others.
    #[error("a coefficient of the model is aliased with the others")]
    Aliased,
    /// An outcome model did not converge.
    #[error("the model did not converge")]
    NotConverged,
    /// A bootstrap replicate could not be fitted.
    #[error("replicate {index} could not be fitted: {reason}")]
    Replicate {
        /// The replicate.
        index: usize,
        /// Why.
        reason: String,
    },
    /// A model of the scores or the outcome could not be fitted.
    #[error(transparent)]
    Glm(#[from] epistat_glm::Error),
    /// An optimiser failed.
    #[error(transparent)]
    Optim(#[from] epistat_optim::Error),
    /// A variance, a pooled estimate or a cluster sum could not be formed.
    #[error(transparent)]
    Inference(#[from] epistat_inference::Error),
    /// A statistic of the shared statistics crate could not be formed.
    #[error(transparent)]
    Stats(#[from] epistat_stats::Error),
    /// A sensitivity bound could not be formed.
    #[error(transparent)]
    Epi(#[from] epistat_epi::Error),
}

fn input_mode_message(mode: MatchMode) -> &'static str {
    match mode {
        MatchMode::Standard => {
            "the spec is for standard matching and needs cases and controls; \
             pass them, or set the mode to risk_set"
        }
        MatchMode::RiskSet | MatchMode::ExposureDensity => {
            "the spec is for risk-set matching and needs one cohort; \
             pass it, or set the mode to standard"
        }
        MatchMode::SequentialTrials => {
            "the spec is for sequential trials and needs a panel of people \
             with their periods; pass it, or set the mode to standard"
        }
    }
}

fn too_few_units_message(stratum: Option<&str>) -> String {
    match stratum {
        None => "too few units to make one subclass with the constraints given".to_string(),
        Some(stratum) => format!(
            "the stratum {stratum} has too few units to make one subclass \
             with the constraints given"
        ),
    }
}

impl Error {
    /// An [`Error::Option`] for `option`.
    pub(crate) fn option(option: &'static str, message: impl Into<String>) -> Self {
        Self::Option {
            option,
            message: message.into(),
        }
    }

    /// An [`InputError::Length`].
    pub(crate) const fn length(what: &'static str, expected: usize, found: usize) -> Self {
        Self::Input(InputError::Length {
            what,
            expected,
            found,
        })
    }

    /// An [`InputError::NotFinite`].
    pub(crate) const fn not_finite(what: &'static str, row: usize) -> Self {
        Self::Input(InputError::NotFinite {
            what,
            row,
            column: None,
        })
    }

    /// An [`InputError::Empty`].
    pub(crate) const fn empty(what: &'static str) -> Self {
        Self::Input(InputError::Empty { what })
    }
}

impl Classify for Error {
    fn kind(&self) -> ErrorKind {
        match self {
            Self::Input(error) => error.kind(),
            Self::Glm(error) => error.kind(),
            Self::Optim(error) => error.kind(),
            Self::Inference(error) => error.kind(),
            Self::Stats(error) => error.kind(),
            Self::Epi(error) => error.kind(),
            Self::Trial { source, .. } => source.kind(),
            Self::InputMode { .. }
            | Self::UnsupportedEstimand { .. }
            | Self::SamplingWeightsNeedProfile
            | Self::PairingNeedsWholeRatio
            | Self::ReestimateAcross
            | Self::CemOrder => ErrorKind::Unsupported,
            Self::NoMatches
            | Self::Infeasible { .. }
            | Self::TooLarge { .. }
            | Self::InfeasibleBalance { .. }
            | Self::NoSolutionInTime { .. }
            | Self::Solver(_)
            | Self::ZeroVarianceGroup
            | Self::SingularCovariance
            | Self::SingularRegression
            | Self::LinearSystem
            | Self::Separated
            | Self::Aliased
            | Self::NotConverged
            | Self::Replicate { .. }
            | Self::NotFullRank
            | Self::InvalidCovariance
            | Self::DistanceBuild
            | Self::NotPositiveSemiDefinite
            | Self::NoCommonSupport
            | Self::CommonSupport(_)
            | Self::NoSubclass
            | Self::TooFewUnits { .. }
            | Self::EmptySupport { .. } => ErrorKind::Numerical,
            _ => ErrorKind::Input,
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Input(error) => error.code(),
            Self::Glm(error) => error.code(),
            Self::Optim(error) => error.code(),
            Self::Inference(error) => error.code(),
            Self::Stats(error) => error.code(),
            Self::Epi(error) => error.code(),
            Self::Trial { source, .. } => source.code(),
            Self::Option { .. } => "matching.option",
            Self::NegativeBirthDateWindow => "matching.negative_birth_date_window",
            Self::ZeroMatchRatio => "matching.zero_match_ratio",
            Self::InvalidCriteria(_) => "matching.invalid_criteria",
            Self::CommonSupport(_) => "matching.common_support",
            Self::InputMode { .. } => "matching.input_mode",
            Self::DuplicateId { .. } => "matching.duplicate_id",
            Self::EventBeforeBirth { .. } => "matching.event_before_birth",
            Self::ColumnLength { .. } => "matching.column_length",
            Self::DuplicateColumn { .. } => "matching.duplicate_column",
            Self::TooManyValues { .. } => "matching.too_many_values",
            Self::MissingColumn { .. } => "matching.missing_column",
            Self::MissingStratum { .. } => "matching.missing_stratum",
            Self::WrongType { .. } => "matching.wrong_type",
            Self::NullValue { .. } => "matching.null_value",
            Self::DateOutOfRange { .. } => "matching.date_out_of_range",
            Self::Value { .. } => "matching.value",
            Self::InvalidWeight { .. } => "matching.invalid_weight",
            Self::UnknownUnit { .. } => "matching.unknown_unit",
            Self::UnknownCase { .. } => "matching.unknown_case",
            Self::UnknownControl { .. } => "matching.unknown_control",
            Self::NoFixedRoles { .. } => "matching.no_fixed_roles",
            Self::NotPaired => "matching.not_paired",
            Self::InconsistentPair { .. } => "matching.inconsistent_pair",
            Self::PeriodsNotIncreasing { .. } => "matching.periods_not_increasing",
            Self::InconsistentDate { .. } => "matching.inconsistent_date",
            Self::DatesNotIncreasing { .. } => "matching.dates_not_increasing",
            Self::StrataRows { .. } => "matching.strata_rows",
            Self::UnknownMember { .. } => "matching.unknown_member",
            Self::NoVariables => "matching.no_variables",
            Self::MissingValue { .. } => "matching.missing_value",
            Self::NoSharedStratum => "matching.no_shared_stratum",
            Self::Binning { .. } => "matching.binning",
            Self::CemOrder => "matching.cem_order",
            Self::EmptyGroup => "matching.empty_group",
            Self::MissingValues { .. } => "matching.missing_values",
            Self::NonFiniteCovariate { .. } => "matching.non_finite_covariate",
            Self::MixedCovariateType { .. } => "matching.mixed_covariate_type",
            Self::NoObservedValues { .. } => "matching.no_observed_values",
            Self::NoCovariates => "matching.no_covariates",
            Self::ConstantCovariate(_) => "matching.constant_covariate",
            Self::ZeroWithinGroupVariance { .. } => "matching.zero_within_group_variance",
            Self::InvalidElasticNet => "matching.invalid_elastic_net",
            Self::InvalidModelControl => "matching.invalid_model_control",
            Self::InvalidCrossValidation => "matching.invalid_cross_validation",
            Self::InvalidSamplingWeight { .. } => "matching.invalid_sampling_weight",
            Self::ReestimateWithoutDiscard => "matching.reestimate_without_discard",
            Self::NoCommonSupport => "matching.no_common_support",
            Self::CbpsNeedsIntercept => "matching.cbps_needs_intercept",
            Self::NotFullRank => "matching.not_full_rank",
            Self::InvalidCbps => "matching.invalid_cbps",
            Self::InvalidCovariance => "matching.invalid_covariance",
            Self::DistanceBuild => "matching.distance_build",
            Self::ZeroDimension => "matching.zero_dimension",
            Self::InverseCovarianceLength => "matching.inverse_covariance_length",
            Self::VectorLength => "matching.vector_length",
            Self::NotPositiveSemiDefinite => "matching.not_positive_semi_definite",
            Self::DistanceNotANumber { .. } => "matching.distance_not_a_number",
            Self::NegativeDistance { .. } => "matching.negative_distance",
            Self::Coordinates => "matching.coordinates",
            Self::NonFiniteCoordinate { .. } => "matching.non_finite_coordinate",
            Self::MissingCoordinates { .. } => "matching.missing_coordinates",
            Self::SubclassMinimum => "matching.subclass_minimum",
            Self::SizeBelowMinimums => "matching.size_below_minimums",
            Self::InvalidCaliper => "matching.invalid_caliper",
            Self::TooFewUnits { .. } => "matching.too_few_units",
            Self::NoSubclass => "matching.no_subclass",
            Self::TooManyUnits => "matching.too_many_units",
            Self::NoSubclasses => "matching.no_subclasses",
            Self::ZeroMaxRatio => "matching.zero_max_ratio",
            Self::NegativeCaliperDays(_) => "matching.negative_caliper_days",
            Self::AverageNotAboveOne => "matching.average_not_above_one",
            Self::MinimumBelowOne => "matching.minimum_below_one",
            Self::MinimumNotBelowAverage => "matching.minimum_not_below_average",
            Self::MaximumNotAboveAverage => "matching.maximum_not_above_average",
            Self::Shape(_) => "matching.shape",
            Self::NotFinite(_) => "matching.not_finite",
            Self::Invalid(_) => "matching.invalid",
            Self::NegativeTolerance => "matching.negative_tolerance",
            Self::InvalidRatio(_) => "matching.invalid_ratio",
            Self::InvalidTolerance => "matching.invalid_tolerance",
            Self::OneGroup => "matching.one_group",
            Self::TooManyMatches { .. } => "matching.too_many_matches",
            Self::TooManyVarianceMatches { .. } => "matching.too_many_variance_matches",
            Self::ZeroMatches => "matching.zero_matches",
            Self::MissingPair { .. } => "matching.missing_pair",
            Self::EmptyTarget => "matching.empty_target",
            Self::Outcome { .. } => "matching.outcome",
            Self::TooFewRows => "matching.too_few_rows",
            Self::Bootstrap => "matching.bootstrap",
            Self::TooFewSets { .. } => "matching.too_few_sets",
            Self::UnitsDiffer { .. } => "matching.units_differ",
            Self::Groups { .. } => "matching.groups",
            Self::Clusters => "matching.clusters",
            Self::EmptySupport { .. } => "matching.empty_support",
            Self::NotOneTreated { .. } => "matching.not_one_treated",
            Self::NoControl { .. } => "matching.no_control",
            Self::NotAPair { .. } => "matching.not_a_pair",
            Self::ReusedUnit { .. } => "matching.reused_unit",
            Self::NoSets => "matching.no_sets",
            Self::NotBinary(_) => "matching.not_binary",
            Self::UnsupportedEstimand { .. } => "matching.unsupported_estimand",
            Self::SamplingWeightsNeedProfile => "matching.sampling_weights_need_profile",
            Self::PairingNeedsWholeRatio => "matching.pairing_needs_whole_ratio",
            Self::ReestimateAcross => "matching.reestimate_across",
            Self::NoMatches => "matching.no_matches",
            Self::Infeasible { .. } => "matching.infeasible",
            Self::TooLarge { .. } => "matching.too_large",
            Self::InfeasibleBalance { .. } => "matching.infeasible_balance",
            Self::NoSolutionInTime { .. } => "matching.no_solution_in_time",
            Self::Solver(_) => "matching.solver",
            Self::ZeroVarianceGroup => "matching.zero_variance_group",
            Self::SingularCovariance => "matching.singular_covariance",
            Self::SingularRegression => "matching.singular_regression",
            Self::LinearSystem => "matching.linear_system",
            Self::Separated => "matching.separated",
            Self::Aliased => "matching.aliased",
            Self::NotConverged => "matching.not_converged",
            Self::Replicate { .. } => "matching.replicate",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One value of every variant whose code is the crate's own. The match
    /// below has no wildcard, so a new variant does not compile until it is
    /// listed here.
    fn every_own_variant() -> Vec<Error> {
        let errors = vec![
            Error::Option {
                option: "x",
                message: String::new(),
            },
            Error::NegativeBirthDateWindow,
            Error::ZeroMatchRatio,
            Error::InvalidCriteria(InvalidCriteriaReason::StudyOptions),
            Error::CommonSupport(CommonSupportFailureReason::NoOverlap),
            Error::InputMode {
                mode: MatchMode::Standard,
            },
            Error::DuplicateId { id: String::new() },
            Error::EventBeforeBirth { id: String::new() },
            Error::ColumnLength {
                column: String::new(),
                found: 0,
                expected: 0,
            },
            Error::DuplicateColumn {
                column: String::new(),
            },
            Error::TooManyValues {
                column: String::new(),
            },
            Error::MissingColumn {
                column: String::new(),
                present: Vec::new(),
            },
            Error::MissingStratum {
                key: "x",
                group: "case",
            },
            Error::WrongType {
                column: String::new(),
                found: String::new(),
                expected: "x",
            },
            Error::NullValue {
                column: String::new(),
                row: 0,
            },
            Error::DateOutOfRange {
                column: String::new(),
                row: 0,
            },
            Error::Value { what: "x", row: 0 },
            Error::InvalidWeight {
                id: String::new(),
                weight: 0.0,
            },
            Error::UnknownUnit { id: String::new() },
            Error::UnknownCase { id: String::new() },
            Error::UnknownControl { id: String::new() },
            Error::NoFixedRoles { output: "x" },
            Error::NotPaired,
            Error::InconsistentPair {
                case_id: String::new(),
                control_id: String::new(),
            },
            Error::PeriodsNotIncreasing {
                id: String::new(),
                period: 0,
            },
            Error::InconsistentDate { period: 0 },
            Error::DatesNotIncreasing { period: 0 },
            Error::StrataRows {
                expected: 0,
                found: 0,
            },
            Error::UnknownMember {
                id: String::new(),
                period: 0,
            },
            Error::NoVariables,
            Error::MissingValue {
                id: String::new(),
                variable: String::new(),
            },
            Error::NoSharedStratum,
            Error::Binning {
                variable: String::new(),
                message: String::new(),
            },
            Error::CemOrder,
            Error::EmptyGroup,
            Error::MissingValues {
                covariate: String::new(),
            },
            Error::NonFiniteCovariate {
                id: String::new(),
                covariate: String::new(),
            },
            Error::MixedCovariateType {
                covariate: String::new(),
            },
            Error::NoObservedValues {
                covariate: String::new(),
            },
            Error::NoCovariates,
            Error::ConstantCovariate(0),
            Error::ZeroWithinGroupVariance {
                column: String::new(),
            },
            Error::InvalidElasticNet,
            Error::InvalidModelControl,
            Error::InvalidCrossValidation,
            Error::InvalidSamplingWeight { id: String::new() },
            Error::ReestimateWithoutDiscard,
            Error::NoCommonSupport,
            Error::CbpsNeedsIntercept,
            Error::NotFullRank,
            Error::InvalidCbps,
            Error::InvalidCovariance,
            Error::DistanceBuild,
            Error::ZeroDimension,
            Error::InverseCovarianceLength,
            Error::VectorLength,
            Error::NotPositiveSemiDefinite,
            Error::DistanceNotANumber {
                case: 0,
                control: 0,
            },
            Error::NegativeDistance {
                case: 0,
                control: 0,
            },
            Error::Coordinates,
            Error::NonFiniteCoordinate { unit: 0 },
            Error::MissingCoordinates { id: String::new() },
            Error::SubclassMinimum,
            Error::SizeBelowMinimums,
            Error::InvalidCaliper,
            Error::TooFewUnits { stratum: None },
            Error::NoSubclass,
            Error::TooManyUnits,
            Error::NoSubclasses,
            Error::ZeroMaxRatio,
            Error::NegativeCaliperDays(0),
            Error::AverageNotAboveOne,
            Error::MinimumBelowOne,
            Error::MinimumNotBelowAverage,
            Error::MaximumNotAboveAverage,
            Error::Shape(String::new()),
            Error::NotFinite(String::new()),
            Error::Invalid(String::new()),
            Error::NegativeTolerance,
            Error::InvalidRatio(String::new()),
            Error::InvalidTolerance,
            Error::OneGroup,
            Error::TooManyMatches {
                asked: 0,
                available: 0,
            },
            Error::TooManyVarianceMatches { asked: 0, group: 0 },
            Error::ZeroMatches,
            Error::MissingPair { row: 0 },
            Error::EmptyTarget,
            Error::Outcome { row: 0 },
            Error::TooFewRows,
            Error::Bootstrap,
            Error::TooFewSets { found: 0 },
            Error::UnitsDiffer {
                set: 0,
                id: String::new(),
            },
            Error::Groups { groups: 0 },
            Error::Clusters,
            Error::EmptySupport { group: 0 },
            Error::NotOneTreated { set: 0, treated: 0 },
            Error::NoControl { set: 0 },
            Error::NotAPair {
                set: 0,
                controls: 0,
            },
            Error::ReusedUnit { id: String::new() },
            Error::NoSets,
            Error::NotBinary(0.0),
            Error::UnsupportedEstimand {
                estimand: epistat_core::Estimand::Ato,
                method: "x",
            },
            Error::SamplingWeightsNeedProfile,
            Error::PairingNeedsWholeRatio,
            Error::ReestimateAcross,
            Error::NoMatches,
            Error::Infeasible {
                stratum: 0,
                message: String::new(),
            },
            Error::TooLarge { stratum: 0 },
            Error::InfeasibleBalance { stratum: 0 },
            Error::NoSolutionInTime { stratum: 0 },
            Error::Solver(String::new()),
            Error::ZeroVarianceGroup,
            Error::SingularCovariance,
            Error::SingularRegression,
            Error::LinearSystem,
            Error::Separated,
            Error::Aliased,
            Error::NotConverged,
            Error::Replicate {
                index: 0,
                reason: String::new(),
            },
        ];
        for error in &errors {
            match error {
                Error::Input(_)
                | Error::Option { .. }
                | Error::NegativeBirthDateWindow
                | Error::ZeroMatchRatio
                | Error::InvalidCriteria(_)
                | Error::CommonSupport(_)
                | Error::InputMode { .. }
                | Error::DuplicateId { .. }
                | Error::EventBeforeBirth { .. }
                | Error::ColumnLength { .. }
                | Error::DuplicateColumn { .. }
                | Error::TooManyValues { .. }
                | Error::MissingColumn { .. }
                | Error::MissingStratum { .. }
                | Error::WrongType { .. }
                | Error::NullValue { .. }
                | Error::DateOutOfRange { .. }
                | Error::Value { .. }
                | Error::InvalidWeight { .. }
                | Error::UnknownUnit { .. }
                | Error::UnknownCase { .. }
                | Error::UnknownControl { .. }
                | Error::NoFixedRoles { .. }
                | Error::NotPaired
                | Error::InconsistentPair { .. }
                | Error::PeriodsNotIncreasing { .. }
                | Error::InconsistentDate { .. }
                | Error::DatesNotIncreasing { .. }
                | Error::StrataRows { .. }
                | Error::UnknownMember { .. }
                | Error::Trial { .. }
                | Error::NoVariables
                | Error::MissingValue { .. }
                | Error::NoSharedStratum
                | Error::Binning { .. }
                | Error::CemOrder
                | Error::EmptyGroup
                | Error::MissingValues { .. }
                | Error::NonFiniteCovariate { .. }
                | Error::MixedCovariateType { .. }
                | Error::NoObservedValues { .. }
                | Error::NoCovariates
                | Error::ConstantCovariate(_)
                | Error::ZeroWithinGroupVariance { .. }
                | Error::InvalidElasticNet
                | Error::InvalidModelControl
                | Error::InvalidCrossValidation
                | Error::InvalidSamplingWeight { .. }
                | Error::ReestimateWithoutDiscard
                | Error::NoCommonSupport
                | Error::CbpsNeedsIntercept
                | Error::NotFullRank
                | Error::InvalidCbps
                | Error::InvalidCovariance
                | Error::DistanceBuild
                | Error::ZeroDimension
                | Error::InverseCovarianceLength
                | Error::VectorLength
                | Error::NotPositiveSemiDefinite
                | Error::DistanceNotANumber { .. }
                | Error::NegativeDistance { .. }
                | Error::Coordinates
                | Error::NonFiniteCoordinate { .. }
                | Error::MissingCoordinates { .. }
                | Error::SubclassMinimum
                | Error::SizeBelowMinimums
                | Error::InvalidCaliper
                | Error::TooFewUnits { .. }
                | Error::NoSubclass
                | Error::TooManyUnits
                | Error::NoSubclasses
                | Error::ZeroMaxRatio
                | Error::NegativeCaliperDays(_)
                | Error::AverageNotAboveOne
                | Error::MinimumBelowOne
                | Error::MinimumNotBelowAverage
                | Error::MaximumNotAboveAverage
                | Error::Shape(_)
                | Error::NotFinite(_)
                | Error::Invalid(_)
                | Error::NegativeTolerance
                | Error::InvalidRatio(_)
                | Error::InvalidTolerance
                | Error::OneGroup
                | Error::TooManyMatches { .. }
                | Error::TooManyVarianceMatches { .. }
                | Error::ZeroMatches
                | Error::MissingPair { .. }
                | Error::EmptyTarget
                | Error::Outcome { .. }
                | Error::TooFewRows
                | Error::Bootstrap
                | Error::TooFewSets { .. }
                | Error::UnitsDiffer { .. }
                | Error::Groups { .. }
                | Error::Clusters
                | Error::EmptySupport { .. }
                | Error::NotOneTreated { .. }
                | Error::NoControl { .. }
                | Error::NotAPair { .. }
                | Error::ReusedUnit { .. }
                | Error::NoSets
                | Error::NotBinary(_)
                | Error::UnsupportedEstimand { .. }
                | Error::SamplingWeightsNeedProfile
                | Error::PairingNeedsWholeRatio
                | Error::ReestimateAcross
                | Error::NoMatches
                | Error::Infeasible { .. }
                | Error::TooLarge { .. }
                | Error::InfeasibleBalance { .. }
                | Error::NoSolutionInTime { .. }
                | Error::Solver(_)
                | Error::ZeroVarianceGroup
                | Error::SingularCovariance
                | Error::SingularRegression
                | Error::LinearSystem
                | Error::Separated
                | Error::Aliased
                | Error::NotConverged
                | Error::Replicate { .. }
                | Error::Glm(_)
                | Error::Optim(_)
                | Error::Inference(_)
                | Error::Stats(_)
                | Error::Epi(_) => {}
            }
        }
        errors
    }

    #[test]
    fn every_code_is_unique_and_names_the_crate() {
        epistat_testkit::assert_codes_unique("matching", &every_own_variant());
    }

    #[test]
    fn a_delegating_variant_has_the_code_of_its_source() {
        let input = Error::from(InputError::Empty { what: "x" });
        assert_eq!(input.code(), "input.empty");
        let trial = Error::Trial {
            period: 1,
            source: Box::new(Error::NoMatches),
        };
        assert_eq!(trial.code(), Error::NoMatches.code());
        assert_eq!(trial.kind(), Error::NoMatches.kind());
    }
}
