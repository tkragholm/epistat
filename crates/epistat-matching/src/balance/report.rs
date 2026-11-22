use super::input::{BalanceInputError, BalanceWeights};
use super::stats::{
    cramers_v_weighted, eqq_distance_weighted_sorted, standard_deviation, standardised_difference,
    variance_ratio, weighted_ecdf_distance_ordered,
};
use crate::matching::{ColumnReader, CovariateRecord, NameScan, effective_sample_size};
use crate::types::{
    BalanceImprovement, BalanceReport, BalanceReportOptions, BalanceSummary,
    BalanceThresholdSummary, BalanceThresholds, CategoricalBalance, CategoricalLevelBalance,
    CovariateRef, CovariateValue, Estimand, GroupSampleSizes, LevelImprovement, MatchOutcome,
    MatchedPair, NumericBalance, NumericBalanceThresholdCheck, NumericBalanceTransform,
    NumericImprovement, PairDistance, SampleSizeTable,
};
use epistat_stats::weighted;
use itertools::Itertools;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Compute pre- and post-match covariate balance summaries.
///
/// The cases are the treated group and the controls the comparison group.
/// "Pre" statistics use every record passed in with weight one. "Post"
/// statistics weight each record by the unit weights the outcome gives for its
/// realised estimand (see [`MatchOutcome::match_weights_for_realized_estimand`]),
/// so a record of weight zero does not enter them. This is
/// [`balance_report_on_weights`] on [`BalanceWeights::from_outcome`].
///
/// Standardised differences follow the documentation of MatchIt's
/// `summary.matchit()`. The difference in weighted means is divided by a
/// standard deviation taken from the records before matching, the same one
/// before and after: the cases' for the ATT, the controls' for the ATC and the
/// square root of the mean of the two variances for the ATE and the ATM. A
/// continuous variable has the sample variance, a variable whose values are all
/// zero or one has `p (1 - p)` and no variance ratio. After matching the
/// variance is the reliability-weight estimator of `cov.wt()`. The eCDF mean and
/// maximum are weighted and evaluated at every distinct value among all
/// records, and for a binary variable both are the absolute difference in
/// proportions.
///
/// The eQQ distances are those of `summary(standardize = FALSE)`: the larger
/// group is read at the smaller group's cumulative weights as a step function
/// (see [`epistat_stats::balance::eqq_distance_weighted`]). Cramér's V uses the
/// weights as counts.
#[must_use]
pub fn balance_report<R: CovariateRecord>(
    cases: &[R],
    controls: &[R],
    outcome: &MatchOutcome,
) -> BalanceReport {
    balance_report_with_options(cases, controls, outcome, &BalanceReportOptions::default())
}

/// Compute pre- and post-match covariate balance summaries with report options.
#[must_use]
pub fn balance_report_with_options<R: CovariateRecord>(
    cases: &[R],
    controls: &[R],
    outcome: &MatchOutcome,
    options: &BalanceReportOptions,
) -> BalanceReport {
    let weights = BalanceWeights::from_outcome(outcome);
    compute_summary(cases, controls, &weights, options, false).report
}

/// Covariate balance before and after matching on weights given from outside.
///
/// `weights` gives a matching weight to each unit, so the weights can come from
/// subclassification, the exact method, full matching, matching with
/// replacement, or from no matching at all ([`BalanceWeights::unmatched`]).
/// With sampling weights, the statistics before matching use them and those
/// after matching use the matching weights multiplied by them, and the
/// standardiser is the standard deviation of the sample before matching under
/// the sampling weights. Otherwise the statistics are those of
/// [`balance_report`].
///
/// # Errors
///
/// [`BalanceInputError::InvalidWeight`] for a weight that is negative or not
/// finite.
pub fn balance_report_on_weights<R: CovariateRecord>(
    cases: &[R],
    controls: &[R],
    weights: &BalanceWeights,
    options: &BalanceReportOptions,
) -> Result<BalanceReport, BalanceInputError> {
    Ok(balance_summary(cases, controls, weights, options)?.report)
}

/// The balance report with the sample size table and the pair distances.
///
/// The pair distances are computed from `weights.pairs`. The sample size table
/// counts a unit as discarded when `weights.discarded` has it.
///
/// # Errors
///
/// [`BalanceInputError::InvalidWeight`] for a weight that is negative or not
/// finite.
pub fn balance_summary<R: CovariateRecord>(
    cases: &[R],
    controls: &[R],
    weights: &BalanceWeights,
    options: &BalanceReportOptions,
) -> Result<BalanceSummary, BalanceInputError> {
    weights.validate()?;
    Ok(compute_summary(cases, controls, weights, options, true))
}

fn compute_summary<R: CovariateRecord>(
    cases: &[R],
    controls: &[R],
    weights: &BalanceWeights,
    options: &BalanceReportOptions,
    with_pair_distances: bool,
) -> BalanceSummary {
    let context = BalanceCovariateContext {
        case_supplemental: &options.supplemental_covariates.cases,
        control_supplemental: &options.supplemental_covariates.controls,
    };
    let (numeric_keys, categorical_keys) = split_covariate_keys_by_kind(
        cases,
        controls,
        context.case_supplemental,
        context.control_supplemental,
    );
    let sample = BalanceSample {
        cases,
        controls,
        case_weights: unit_weights(cases, &weights.matching.anchor, weights.sampling.as_ref()),
        control_weights: unit_weights(
            controls,
            &weights.matching.candidate,
            weights.sampling.as_ref(),
        ),
        sampled: weights.sampling.is_some(),
        estimand: weights.estimand,
        pairs: if with_pair_distances {
            &weights.pairs
        } else {
            &[]
        },
        pair_positions: None,
    };
    let sample = BalanceSample {
        pair_positions: pair_positions(cases, controls, sample.pairs),
        ..sample
    };
    let levels = categorical_keys
        .iter()
        .map(|key| {
            (
                key.clone(),
                Arc::<[String]>::from(collect_levels(
                    cases,
                    controls,
                    context.case_supplemental,
                    context.control_supplemental,
                    key,
                )),
            )
        })
        .collect::<HashMap<_, _>>();
    let (numeric, mut pair_distances) = build_numeric_balance(
        &sample,
        context,
        &numeric_keys,
        &levels,
        options.numeric_transforms,
    );
    let (categorical, categorical_distances) =
        build_categorical_balance(&sample, context, &categorical_keys, &levels);
    pair_distances.extend(categorical_distances);
    if sample.pairs.is_empty() {
        pair_distances.clear();
    }

    BalanceSummary {
        report: BalanceReport {
            numeric,
            categorical,
        },
        sample_sizes: sample_sizes(&sample, weights),
        pair_distances,
    }
}

/// The sample size table: counts and effective sample sizes before and after
/// matching, from the weights of each unit.
fn sample_sizes<R: CovariateRecord>(
    sample: &BalanceSample<'_, R>,
    weights: &BalanceWeights,
) -> SampleSizeTable {
    let group = |records: &[R], unit_weights: &[UnitWeights]| {
        let ess = |values: Vec<f64>| {
            // The weights have been validated, so only a size that is
            // undefined, with no positive weight, is left as `None`.
            effective_sample_size(values.iter().copied()).ok().flatten()
        };
        GroupSampleSizes {
            all: records.len(),
            all_ess: ess(unit_weights.iter().map(|unit| unit.pre).collect()),
            matched: unit_weights.iter().filter(|unit| unit.post > 0.0).count(),
            matched_ess: ess(unit_weights.iter().map(|unit| unit.post).collect()),
            unmatched: records
                .iter()
                .zip(unit_weights)
                .filter(|(record, unit)| {
                    unit.post == 0.0 && !weights.discarded.contains(record.id())
                })
                .count(),
            discarded: records
                .iter()
                .filter(|record| weights.discarded.contains(record.id()))
                .count(),
        }
    };
    SampleSizeTable {
        cases: group(sample.cases, &sample.case_weights),
        controls: group(sample.controls, &sample.control_weights),
    }
}

/// Percent balance improvement of every statistic that has a value before and
/// after matching.
///
/// An improvement is `100 (|before| - |after|) / |before|`. For the variance
/// ratio it is computed on the absolute logarithm of the ratio, since a ratio
/// of 0.5 is as far from balance as a ratio of 2. It is `None` where the value
/// before matching is 0, which MatchIt reports as 0 or negative infinity, and
/// where either value is undefined.
#[must_use]
pub fn balance_improvement(report: &BalanceReport) -> BalanceImprovement {
    let percent = |before: Option<f64>, after: Option<f64>| {
        let (before, after) = (before?.abs(), after?.abs());
        (before.is_finite() && after.is_finite() && before > 0.0)
            .then(|| 100.0 * (before - after) / before)
    };
    let log_ratio = |ratio: Option<f64>| ratio.filter(|value| *value > 0.0).map(f64::ln);
    BalanceImprovement {
        numeric: report
            .numeric
            .iter()
            .map(|row| NumericImprovement {
                name: row.name.clone(),
                smd: percent(row.smd_pre, row.smd_post),
                var_ratio: percent(log_ratio(row.var_ratio_pre), log_ratio(row.var_ratio_post)),
                ecdf_mean: percent(row.ecdf_mean_diff_pre, row.ecdf_mean_diff_post),
                ecdf_max: percent(row.ecdf_max_diff_pre, row.ecdf_max_diff_post),
                eqq_mean: percent(row.eqq_mean_diff_pre, row.eqq_mean_diff_post),
                eqq_max: percent(row.eqq_max_diff_pre, row.eqq_max_diff_post),
            })
            .collect(),
        categorical: report
            .categorical
            .iter()
            .flat_map(|covariate| {
                covariate.levels.iter().map(|level| LevelImprovement {
                    name: covariate.name.clone(),
                    level: level.level.clone(),
                    smd: percent(level.smd_pre, level.smd_post),
                })
            })
            .collect(),
    }
}

/// Evaluate post-match numeric balance diagnostics against configured thresholds.
#[must_use]
pub fn balance_threshold_summary(
    report: &BalanceReport,
    thresholds: &BalanceThresholds,
) -> BalanceThresholdSummary {
    let numeric = report
        .numeric
        .iter()
        .map(|metric| {
            // An undefined statistic cannot show balance, so an enabled check on it fails.
            let smd_post_ok = thresholds.smd_abs_max.map(|threshold| {
                metric
                    .smd_post
                    .is_some_and(|value| value.abs() <= threshold)
            });
            let var_ratio_post_ok = match (thresholds.var_ratio_min, thresholds.var_ratio_max) {
                (None, None) => None,
                (min, max) => Some(metric.var_ratio_post.is_some_and(|value| {
                    min.is_none_or(|min| value >= min) && max.is_none_or(|max| value <= max)
                })),
            };
            let ecdf_max_diff_post_ok = thresholds.ecdf_max_diff_max.map(|threshold| {
                metric
                    .ecdf_max_diff_post
                    .is_some_and(|value| value <= threshold)
            });
            let eqq_max_diff_post_ok = thresholds.eqq_max_diff_max.map(|threshold| {
                metric
                    .eqq_max_diff_post
                    .is_some_and(|value| value <= threshold)
            });

            let enabled_checks = [
                smd_post_ok,
                var_ratio_post_ok,
                ecdf_max_diff_post_ok,
                eqq_max_diff_post_ok,
            ]
            .into_iter()
            .flatten()
            .collect_vec();
            let all_enabled_checks_ok = enabled_checks.iter().all(|check| *check);

            NumericBalanceThresholdCheck {
                name: metric.name.clone(),
                smd_post_ok,
                var_ratio_post_ok,
                ecdf_max_diff_post_ok,
                eqq_max_diff_post_ok,
                all_enabled_checks_ok,
            }
        })
        .collect_vec();

    let all_enabled_checks_ok = numeric.iter().all(|check| check.all_enabled_checks_ok);
    BalanceThresholdSummary {
        numeric,
        all_enabled_checks_ok,
    }
}

#[derive(Clone, Copy)]
enum CovariateKind {
    Numeric,
    Categorical,
}

/// A numeric value makes a numeric covariate; anything else, a missing value
/// included, a categorical one.
const fn kind_of(value: Option<CovariateRef<'_>>) -> CovariateKind {
    match value {
        Some(CovariateRef::Numeric(_)) => CovariateKind::Numeric,
        _ => CovariateKind::Categorical,
    }
}

/// One column of the covariate matrix: a numeric covariate, or the indicator of
/// one level of a categorical covariate. `known` lists every level of the
/// covariate, and a unit whose level is not among them has no value.
#[derive(Debug, Clone, PartialEq)]
enum Column {
    Numeric(String),
    Level {
        key: String,
        level: String,
        known: Arc<[String]>,
    },
}

impl Column {
    /// MatchIt's name: a level is the covariate name followed by the level.
    fn name(&self) -> String {
        match self {
            Self::Numeric(key) => key.clone(),
            Self::Level { key, level, .. } => format!("{key}{level}"),
        }
    }

    fn key(&self) -> &str {
        match self {
            Self::Numeric(key) | Self::Level { key, .. } => key,
        }
    }

    fn value<R: CovariateRecord>(
        &self,
        reader: &mut ColumnReader<'_>,
        record: &R,
        supplemental: &SupplementalCovariateMap,
    ) -> Option<f64> {
        match self {
            Self::Numeric(_) => numeric_covariate(reader, record, supplemental),
            Self::Level { level, known, .. } => {
                let found =
                    map_optional_covariate_to_level(covariate_value(reader, record, supplemental));
                known
                    .contains(&found)
                    .then_some(if found == *level { 1.0 } else { 0.0 })
            }
        }
    }
}

#[derive(Debug, Clone)]
enum NumericFeature {
    Raw {
        key: String,
    },
    /// The product of two columns, the square of a column when they are the
    /// same.
    Product {
        left: Column,
        right: Column,
    },
}

/// The readers of the one or two columns of a [`NumericFeature`].
struct FeatureReaders<'k> {
    left: ColumnReader<'k>,
    right: Option<ColumnReader<'k>>,
}

type SupplementalCovariateMap = HashMap<String, HashMap<String, CovariateValue>>;

#[derive(Clone, Copy)]
struct BalanceCovariateContext<'a> {
    case_supplemental: &'a SupplementalCovariateMap,
    control_supplemental: &'a SupplementalCovariateMap,
}

impl NumericFeature {
    fn name(&self) -> String {
        match self {
            Self::Raw { key } => key.clone(),
            Self::Product { left, right } if left == right => format!("{}^2", left.name()),
            Self::Product { left, right } => format!("{} * {}", left.name(), right.name()),
        }
    }

    /// The readers of the feature's columns, which find each covariate in a
    /// table of columns once and not for every record.
    fn readers(&self) -> FeatureReaders<'_> {
        match self {
            Self::Raw { key } => FeatureReaders {
                left: ColumnReader::new(key),
                right: None,
            },
            Self::Product { left, right } => FeatureReaders {
                left: ColumnReader::new(left.key()),
                right: Some(ColumnReader::new(right.key())),
            },
        }
    }

    fn value<R: CovariateRecord>(
        &self,
        readers: &mut FeatureReaders<'_>,
        record: &R,
        supplemental: &SupplementalCovariateMap,
    ) -> Option<f64> {
        match self {
            Self::Raw { .. } => numeric_covariate(&mut readers.left, record, supplemental),
            Self::Product { left, right } => {
                let right_reader = readers
                    .right
                    .as_mut()
                    .expect("a product has a reader for its second column");
                left.value(&mut readers.left, record, supplemental)
                    .zip(right.value(right_reader, record, supplemental))
                    .map(|(left, right)| left * right)
            }
        }
    }
}

fn collect_covariate_keys<R: CovariateRecord>(
    cases: &[R],
    controls: &[R],
    case_supplemental: &SupplementalCovariateMap,
    control_supplemental: &SupplementalCovariateMap,
) -> Vec<String> {
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut scan = NameScan::default();
    for record in cases.iter().chain(controls) {
        if scan.needs_names(record) {
            seen.extend(record.covariate_names());
        }
    }
    seen.into_iter()
        .map(str::to_owned)
        .chain(
            case_supplemental
                .values()
                .flat_map(|values| values.keys().cloned()),
        )
        .chain(
            control_supplemental
                .values()
                .flat_map(|values| values.keys().cloned()),
        )
        .sorted()
        .dedup()
        .collect_vec()
}

fn split_covariate_keys_by_kind<R: CovariateRecord>(
    cases: &[R],
    controls: &[R],
    case_supplemental: &SupplementalCovariateMap,
    control_supplemental: &SupplementalCovariateMap,
) -> (Vec<String>, Vec<String>) {
    // One sweep of the records. The kind of a key is taken from the first
    // record that carries it: cases before controls, and within each group the
    // record order.
    let mut kinds: HashMap<&str, CovariateKind> = HashMap::new();
    let mut scan = NameScan::default();
    for (records, supplemental) in [(cases, case_supplemental), (controls, control_supplemental)] {
        for record in records {
            // The record's own covariates, then those the supplemental map
            // adds for this id, in the order `covariate_value` reads them.
            // Records of one table carry the same names, and every name is
            // known after the first of them.
            if scan.needs_names(record) {
                for key in record.covariate_names() {
                    if !kinds.contains_key(key) {
                        let value = record.covariate(key);
                        kinds.insert(key, kind_of(value));
                    }
                }
            }
            let supplemental_for_record = if supplemental.is_empty() {
                None
            } else {
                supplemental.get(record.id())
            };
            for (key, value) in supplemental_for_record
                .into_iter()
                .flat_map(|values| values.iter())
            {
                kinds
                    .entry(key.as_str())
                    .or_insert_with(|| kind_of(Some(CovariateRef::from(value))));
            }
        }
    }

    collect_covariate_keys(cases, controls, case_supplemental, control_supplemental)
        .into_iter()
        .fold(
            (Vec::new(), Vec::new()),
            |(mut numeric, mut categorical), key| {
                // Absent from every record: `covariate_kind` fell through both
                // loops to Categorical, so this does too.
                match kinds
                    .get(key.as_str())
                    .copied()
                    .unwrap_or(CovariateKind::Categorical)
                {
                    CovariateKind::Numeric => numeric.push(key),
                    CovariateKind::Categorical => categorical.push(key),
                }
                (numeric, categorical)
            },
        )
}

/// A unit's weights: before matching (the sampling weight, or one) and after
/// (the matching weight times the sampling weight).
#[derive(Clone, Copy)]
struct UnitWeights {
    pre: f64,
    post: f64,
}

fn unit_weights<R: CovariateRecord>(
    records: &[R],
    matching: &rapidhash::RapidHashMap<String, f64>,
    sampling: Option<&HashMap<String, f64>>,
) -> Vec<UnitWeights> {
    records
        .iter()
        .map(|record| {
            let id = record.id();
            let matching = matching.get(id).copied().unwrap_or(0.0);
            match sampling {
                None => UnitWeights {
                    pre: 1.0,
                    post: matching,
                },
                Some(sampling) => {
                    let sampling = sampling.get(id).copied().unwrap_or(1.0);
                    UnitWeights {
                        pre: sampling,
                        post: matching * sampling,
                    }
                }
            }
        })
        .collect()
}

/// The records of one balance report and the weight of each.
struct BalanceSample<'a, R> {
    cases: &'a [R],
    controls: &'a [R],
    case_weights: Vec<UnitWeights>,
    control_weights: Vec<UnitWeights>,
    /// Whether sampling weights were given.
    sampled: bool,
    estimand: Estimand,
    pairs: &'a [MatchedPair],
    /// The pairs as positions in the pooled records, cases first, with the
    /// pairs whose units are not among the records left out. `None` when an
    /// identifier occurs twice in a group, where a pair has to be read by
    /// identifier.
    pair_positions: Option<Vec<(usize, usize)>>,
}

impl<R: CovariateRecord> BalanceSample<'_, R> {
    /// The value of a pair's case and control for each pair in which both have
    /// one, in the order of the pairs. `values` holds one value per record, the
    /// cases first. When an identifier occurs more than once in a group, the
    /// last record of it with a value is the one a pair reads.
    fn pair_values<T: Copy>(&self, values: &[Option<T>]) -> Vec<(T, T)> {
        if let Some(positions) = &self.pair_positions {
            return positions
                .iter()
                .filter_map(|&(case, control)| Some((values[case]?, values[control]?)))
                .collect();
        }
        let split = self.cases.len();
        let by_id = |records: &[R], values: &[Option<T>]| {
            records
                .iter()
                .zip(values)
                .filter_map(|(record, value)| Some((record.id().to_owned(), (*value)?)))
                .collect::<HashMap<_, _>>()
        };
        let case_values = by_id(self.cases, &values[..split]);
        let control_values = by_id(self.controls, &values[split..]);
        self.pairs
            .iter()
            .filter_map(|pair| {
                Some((
                    *case_values.get(pair.case_id.as_str())?,
                    *control_values.get(pair.control_id.as_str())?,
                ))
            })
            .collect()
    }
}

/// The records by identifier, `None` when an identifier occurs twice.
fn index_by_id<R: CovariateRecord>(records: &[R]) -> Option<HashMap<&str, usize>> {
    let map = records
        .iter()
        .enumerate()
        .map(|(position, record)| (record.id(), position))
        .collect::<HashMap<_, _>>();
    (map.len() == records.len()).then_some(map)
}

/// The pairs as positions among the pooled records, `None` when an identifier
/// occurs twice in the cases or in the controls.
fn pair_positions<R: CovariateRecord>(
    cases: &[R],
    controls: &[R],
    pairs: &[MatchedPair],
) -> Option<Vec<(usize, usize)>> {
    if pairs.is_empty() {
        return Some(Vec::new());
    }
    let (case_index, control_index) = (index_by_id(cases)?, index_by_id(controls)?);
    Some(
        pairs
            .iter()
            .filter_map(|pair| {
                Some((
                    *case_index.get(pair.case_id.as_str())?,
                    cases.len() + *control_index.get(pair.control_id.as_str())?,
                ))
            })
            .collect(),
    )
}

/// One record's value of a feature, its group and its weights.
#[derive(Clone, Copy)]
struct Unit<T> {
    value: T,
    is_case: bool,
    weights: UnitWeights,
}

impl<T> Unit<T> {
    const fn weight(&self, matched: bool) -> f64 {
        if matched {
            self.weights.post
        } else {
            self.weights.pre
        }
    }
}

/// Which group's variance standardises a difference.
#[derive(Clone, Copy)]
enum Standardiser {
    Cases,
    Controls,
    Pooled,
}

impl Standardiser {
    const fn for_estimand(estimand: Estimand) -> Self {
        match estimand {
            Estimand::Att => Self::Cases,
            Estimand::Atc => Self::Controls,
            Estimand::Ate | Estimand::Atm => Self::Pooled,
        }
    }

    /// The variance to divide by, `None` when the chosen group's is undefined.
    fn variance(self, case: Option<f64>, control: Option<f64>) -> Option<f64> {
        match self {
            Self::Cases => case,
            Self::Controls => control,
            Self::Pooled => Some(f64::midpoint(case?, control?)),
        }
    }
}

/// Mean and variance of one group under one weighting.
#[derive(Clone, Copy)]
struct GroupMoments {
    mean: Option<f64>,
    variance: Option<f64>,
}

type LevelTable = HashMap<String, Arc<[String]>>;

fn build_numeric_balance<R: CovariateRecord>(
    sample: &BalanceSample<'_, R>,
    context: BalanceCovariateContext<'_>,
    numeric_keys: &[String],
    levels: &LevelTable,
    transforms: NumericBalanceTransform,
) -> (Vec<NumericBalance>, Vec<PairDistance>) {
    // The features are independent, and the report keeps their order.
    numeric_feature_specs(numeric_keys, levels, transforms)
        .par_iter()
        .filter(|feature| !is_redundant_term(sample, context, feature))
        .map(|feature| summarize_numeric_feature(sample, context, feature))
        .unzip()
}

/// A squared or product term that is exactly zero everywhere or exactly equal
/// to its first factor everywhere adds nothing to the covariates it is built from, as for
/// the square of a binary variable. MatchIt drops these from `interactions =
/// TRUE`.
#[expect(
    clippy::float_cmp,
    reason = "redundancy is exact, so a covariate on a small scale is never mistaken for it"
)]
fn is_redundant_term<R: CovariateRecord>(
    sample: &BalanceSample<'_, R>,
    context: BalanceCovariateContext<'_>,
    feature: &NumericFeature,
) -> bool {
    let NumericFeature::Product { left, right } = feature else {
        return false;
    };
    let mut all_zero = true;
    let mut equals_left = true;
    let mut left_reader = ColumnReader::new(left.key());
    let mut right_reader = ColumnReader::new(right.key());
    for (records, supplemental) in [
        (sample.cases, context.case_supplemental),
        (sample.controls, context.control_supplemental),
    ] {
        for record in records {
            let (Some(first), Some(second)) = (
                left.value(&mut left_reader, record, supplemental),
                right.value(&mut right_reader, record, supplemental),
            ) else {
                continue;
            };
            let product = first * second;
            all_zero &= product == 0.0;
            equals_left &= product == first;
        }
    }
    all_zero || equals_left
}

/// The records of both groups with their value of the feature, in the order
/// of the records, cases first, and the value of every record by position, `None`
/// for a record without a finite value, which is left out of the units.
fn feature_units<R: CovariateRecord>(
    sample: &BalanceSample<'_, R>,
    context: BalanceCovariateContext<'_>,
    feature: &NumericFeature,
) -> (Vec<Unit<f64>>, Vec<Option<f64>>) {
    let mut units = Vec::new();
    let mut values = Vec::with_capacity(sample.cases.len() + sample.controls.len());
    let mut readers = feature.readers();
    for (records, weights, is_case, supplemental) in [
        (
            sample.cases,
            &sample.case_weights,
            true,
            context.case_supplemental,
        ),
        (
            sample.controls,
            &sample.control_weights,
            false,
            context.control_supplemental,
        ),
    ] {
        for (record, weights) in records.iter().zip(weights) {
            let value = feature.value(&mut readers, record, supplemental);
            if let Some(value) = value {
                units.push(Unit {
                    value,
                    is_case,
                    weights: *weights,
                });
            }
            values.push(value);
        }
    }
    (units, values)
}

/// The units of a feature in ascending order of value, sorted once for every
/// statistic that needs the order.
///
/// Units that tie stay in the order of the records. The eCDF distance reads the
/// order by `f64::total_cmp`, in which negative zero comes before positive
/// zero, and the eQQ distance reads negative and positive zero as one value.
struct SortedUnits {
    /// Every unit, for the eCDF distance.
    ecdf: Vec<u32>,
    /// The units with a finite value, one list per group, for the eQQ distance.
    cases: Vec<u32>,
    controls: Vec<u32>,
}

impl SortedUnits {
    fn new(units: &[Unit<f64>]) -> Self {
        let mut keyed = units
            .iter()
            .enumerate()
            .map(|(position, unit)| {
                (
                    unit.value,
                    u32::try_from(position).expect("fewer than 2^32 units"),
                )
            })
            .collect_vec();
        let by_value_then_position = |left: &(f64, u32), right: &(f64, u32)| {
            left.0
                .partial_cmp(&right.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(left.1.cmp(&right.1))
        };
        if keyed.len() > PARALLEL_SORT_FROM {
            keyed.par_sort_unstable_by(by_value_then_position);
        } else {
            keyed.sort_unstable_by(by_value_then_position);
        }
        let group = |is_case: bool| {
            keyed
                .iter()
                .filter(|(value, position)| {
                    value.is_finite() && units[*position as usize].is_case == is_case
                })
                .map(|(_, position)| *position)
                .collect_vec()
        };
        let (cases, controls) = (group(true), group(false));

        // Within a run of zeros, negative zero first and each sign in the
        // order of the records.
        let is_zero = |value: f64| value.to_bits() << 1 == 0;
        let mut start = 0;
        while start < keyed.len() {
            if is_zero(keyed[start].0) {
                let end = start + keyed[start..].partition_point(|(value, _)| is_zero(*value));
                keyed[start..end]
                    .sort_by_key(|(value, position)| (value.is_sign_positive(), *position));
                start = end;
            } else {
                start += 1;
            }
        }
        Self {
            ecdf: keyed.into_iter().map(|(_, position)| position).collect(),
            cases,
            controls,
        }
    }

    /// One group's values in order, its weights under the stage and the total
    /// weight of its units with a finite value, summed in the order of the
    /// records.
    fn group_in_order(
        units: &[Unit<f64>],
        order: &[u32],
        is_case: bool,
        matched: bool,
    ) -> (Vec<f64>, Vec<f64>, f64) {
        let total = units
            .iter()
            .filter(|unit| unit.is_case == is_case && unit.value.is_finite())
            .map(|unit| unit.weight(matched))
            .sum::<f64>();
        let (values, weights) = order
            .iter()
            .map(|&position| {
                let unit = &units[position as usize];
                (unit.value, unit.weight(matched))
            })
            .unzip();
        (values, weights, total)
    }
}

/// Features with more units than this sort in parallel. The order does not
/// depend on it.
const PARALLEL_SORT_FROM: usize = 1 << 16;

/// Whether a value is exactly zero or one, as in a binary variable.
#[expect(
    clippy::float_cmp,
    reason = "a binary variable holds exactly zero or one"
)]
fn is_zero_or_one(value: f64) -> bool {
    value == 0.0 || value == 1.0
}

/// Which weights a statistic uses: those after matching, or those before, and
/// whether sampling weights were given.
#[derive(Clone, Copy)]
struct Stage {
    matched: bool,
    sampled: bool,
}

/// The mean and variance of one group. Before matching with no sampling
/// weights every unit counts once. Otherwise a unit counts by its weight and
/// one without weight is left out.
fn group_moments(units: &[Unit<f64>], is_case: bool, stage: Stage, binary: bool) -> GroupMoments {
    let matched = stage.matched;
    let use_weights = matched || stage.sampled;
    let (values, weights): (Vec<f64>, Vec<f64>) = units
        .iter()
        .filter(|unit| unit.is_case == is_case && (!use_weights || unit.weight(matched) > 0.0))
        .map(|unit| (unit.value, unit.weight(matched)))
        .unzip();
    let weights = use_weights.then_some(weights);
    let mean = weighted::mean(&values, weights.as_deref());
    let variance = if binary {
        mean.map(|p| p * (1.0 - p))
    } else {
        weighted::variance(&values, weights.as_deref())
    };
    GroupMoments { mean, variance }
}

fn summarize_numeric_feature<R: CovariateRecord>(
    sample: &BalanceSample<'_, R>,
    context: BalanceCovariateContext<'_>,
    feature: &NumericFeature,
) -> (NumericBalance, PairDistance) {
    let (units, positioned) = feature_units(sample, context, feature);
    let binary = !units.is_empty() && units.iter().all(|unit| is_zero_or_one(unit.value));
    let standardiser = Standardiser::for_estimand(sample.estimand);

    let pre = Stage {
        matched: false,
        sampled: sample.sampled,
    };
    let post = Stage {
        matched: true,
        ..pre
    };
    let case_pre = group_moments(&units, true, pre, binary);
    let control_pre = group_moments(&units, false, pre, binary);
    let case_post = group_moments(&units, true, post, binary);
    let control_post = group_moments(&units, false, post, binary);
    let reference = standardiser.variance(case_pre.variance, control_pre.variance);

    let difference =
        |case: GroupMoments, control: GroupMoments| case.mean.zip(control.mean).map(|(a, b)| a - b);
    let smd = |case: GroupMoments, control: GroupMoments| {
        difference(case, control).and_then(|value| standardised_difference(value, reference))
    };
    let ratio = |case: GroupMoments, control: GroupMoments| {
        if binary {
            None
        } else {
            case.variance
                .zip(control.variance)
                .and_then(|(a, b)| variance_ratio(a, b))
        }
    };

    let sorted = (!binary).then(|| SortedUnits::new(&units));

    // Binary variables: both distribution distances are the absolute
    // difference in proportions.
    let ecdf = |matched: bool, case: GroupMoments, control: GroupMoments| {
        if let Some(sorted) = &sorted {
            let points = units
                .iter()
                .map(|unit| (unit.value, unit.is_case, unit.weight(matched)))
                .collect_vec();
            split_pair(weighted_ecdf_distance_ordered(&points, &sorted.ecdf))
        } else {
            let gap = difference(case, control).map(f64::abs);
            (gap, gap)
        }
    };
    let eqq = |matched: bool, case: GroupMoments, control: GroupMoments| {
        if let Some(sorted) = &sorted {
            let (case_values, case_weights, case_total) =
                SortedUnits::group_in_order(&units, &sorted.cases, true, matched);
            let (control_values, control_weights, control_total) =
                SortedUnits::group_in_order(&units, &sorted.controls, false, matched);
            split_pair(eqq_distance_weighted_sorted(
                (&case_values, &case_weights, case_total),
                (&control_values, &control_weights, control_total),
            ))
        } else {
            let gap = difference(case, control).map(f64::abs);
            (gap, gap)
        }
    };
    let (ecdf_mean_diff_pre, ecdf_max_diff_pre) = ecdf(false, case_pre, control_pre);
    let (ecdf_mean_diff_post, ecdf_max_diff_post) = ecdf(true, case_post, control_post);
    let (eqq_mean_diff_pre, eqq_max_diff_pre) = eqq(false, case_pre, control_pre);
    let (eqq_mean_diff_post, eqq_max_diff_post) = eqq(true, case_post, control_post);

    let name = feature.name();
    let distance = pair_distance(sample, &name, &positioned, standardiser_scale(reference));
    let balance = NumericBalance {
        name,
        mean_case_pre: case_pre.mean,
        mean_control_pre: control_pre.mean,
        smd_pre: smd(case_pre, control_pre),
        var_ratio_pre: ratio(case_pre, control_pre),
        ecdf_mean_diff_pre,
        ecdf_max_diff_pre,
        eqq_mean_diff_pre,
        eqq_max_diff_pre,
        mean_case_post: case_post.mean,
        mean_control_post: control_post.mean,
        smd_post: smd(case_post, control_post),
        var_ratio_post: ratio(case_post, control_post),
        ecdf_mean_diff_post,
        ecdf_max_diff_post,
        eqq_mean_diff_post,
        eqq_max_diff_post,
    };
    (balance, distance)
}

/// The standard deviation that standardises a difference, `None` when the
/// variance is not positive and finite.
fn standardiser_scale(reference: Option<f64>) -> Option<f64> {
    reference.and_then(standard_deviation)
}

/// Mean absolute difference over the matched pairs in which both units have a
/// value, summed in the order of the pairs. `values` holds the value of each
/// record, the cases first.
fn pair_distance<R: CovariateRecord>(
    sample: &BalanceSample<'_, R>,
    name: &str,
    values: &[Option<f64>],
    scale: Option<f64>,
) -> PairDistance {
    let gaps = sample
        .pair_values(values)
        .into_iter()
        .map(|(case, control)| (case - control).abs());
    mean_gap(name, gaps, scale)
}

fn mean_gap(name: &str, gaps: impl Iterator<Item = f64>, scale: Option<f64>) -> PairDistance {
    let (sum, count) = gaps.fold((0.0_f64, 0.0_f64), |(sum, count), gap| {
        (sum + gap, count + 1.0)
    });
    let raw = (count > 0.0).then(|| sum / count);
    PairDistance {
        name: name.to_string(),
        raw,
        standardised: raw.zip(scale).map(|(raw, scale)| raw / scale),
    }
}

fn build_categorical_balance<R: CovariateRecord>(
    sample: &BalanceSample<'_, R>,
    context: BalanceCovariateContext<'_>,
    categorical_keys: &[String],
    levels: &LevelTable,
) -> (Vec<CategoricalBalance>, Vec<PairDistance>) {
    let summaries = categorical_keys
        .par_iter()
        .map(|key| summarize_categorical_feature(sample, context, key, &levels[key.as_str()]))
        .collect::<Vec<_>>();
    let mut balances = Vec::new();
    let mut distances = Vec::new();
    for (balance, level_distances) in summaries {
        balances.push(balance);
        distances.extend(level_distances);
    }
    (balances, distances)
}

/// Weight of each level among the records of one group. `matched` selects the
/// weights after matching, otherwise those before.
fn level_mass(units: &[Unit<usize>], is_case: bool, matched: bool, level_count: usize) -> Vec<f64> {
    let mut mass = vec![0.0_f64; level_count];
    for unit in units.iter().filter(|unit| unit.is_case == is_case) {
        mass[unit.value] += unit.weight(matched);
    }
    mass
}

/// Weighted share of each level among the records of one group, `None` when
/// the group has no weight.
fn level_shares(mass: &[f64]) -> Option<Vec<f64>> {
    let total = mass.iter().sum::<f64>();
    (total > 0.0).then(|| mass.iter().map(|value| value / total).collect())
}

fn summarize_categorical_feature<R: CovariateRecord>(
    sample: &BalanceSample<'_, R>,
    context: BalanceCovariateContext<'_>,
    key: &str,
    levels: &[String],
) -> (CategoricalBalance, Vec<PairDistance>) {
    let level_position = levels
        .iter()
        .enumerate()
        .map(|(idx, level)| (level.as_str(), idx))
        .collect::<HashMap<_, _>>();
    let mut units: Vec<Unit<usize>> = Vec::new();
    let mut positioned: Vec<Option<usize>> =
        Vec::with_capacity(sample.cases.len() + sample.controls.len());
    let mut reader = ColumnReader::new(key);
    for (records, is_case, supplemental, weights) in [
        (
            sample.cases,
            true,
            context.case_supplemental,
            &sample.case_weights,
        ),
        (
            sample.controls,
            false,
            context.control_supplemental,
            &sample.control_weights,
        ),
    ] {
        for (record, weights) in records.iter().zip(weights) {
            let level =
                map_optional_covariate_to_level(covariate_value(&mut reader, record, supplemental));
            let value = level_position.get(level.as_str()).copied();
            if let Some(value) = value {
                units.push(Unit {
                    value,
                    is_case,
                    weights: *weights,
                });
            }
            positioned.push(value);
        }
    }
    let plain = units;
    let mass_case_pre = level_mass(&plain, true, false, levels.len());
    let mass_control_pre = level_mass(&plain, false, false, levels.len());
    let mass_case_post = level_mass(&plain, true, true, levels.len());
    let mass_control_post = level_mass(&plain, false, true, levels.len());
    let share_case_pre = level_shares(&mass_case_pre);
    let share_control_pre = level_shares(&mass_control_pre);
    let share_case_post = level_shares(&mass_case_post);
    let share_control_post = level_shares(&mass_control_post);
    let standardiser = Standardiser::for_estimand(sample.estimand);

    let pair_levels = sample.pair_values(&positioned);

    let mut distances = Vec::new();
    let level_balances = levels
        .iter()
        .enumerate()
        .map(|(idx, level)| {
            let at = |shares: &Option<Vec<f64>>| shares.as_ref().map(|values| values[idx]);
            let p_case_pre = at(&share_case_pre);
            let p_control_pre = at(&share_control_pre);
            let p_case_post = at(&share_case_post);
            let p_control_post = at(&share_control_post);
            // A level is a binary variable: its variance is p (1 - p), taken
            // from the records before matching.
            let reference = standardiser.variance(
                p_case_pre.map(|p| p * (1.0 - p)),
                p_control_pre.map(|p| p * (1.0 - p)),
            );
            let smd = |case: Option<f64>, control: Option<f64>| {
                case.zip(control)
                    .and_then(|(a, b)| standardised_difference(a - b, reference))
            };
            distances.push(mean_gap(
                &format!("{key}{level}"),
                pair_levels.iter().map(|(case, control)| {
                    f64::from(u8::from((*case == idx) != (*control == idx)))
                }),
                standardiser_scale(reference),
            ));

            CategoricalLevelBalance {
                level: level.clone(),
                p_case_pre,
                p_control_pre,
                smd_pre: smd(p_case_pre, p_control_pre),
                p_case_post,
                p_control_post,
                smd_post: smd(p_case_post, p_control_post),
            }
        })
        .collect_vec();

    let balance = CategoricalBalance {
        name: key.to_string(),
        levels: level_balances,
        cramers_v_pre: cramers_v_weighted(&mass_case_pre, &mass_control_pre),
        cramers_v_post: cramers_v_weighted(&mass_case_post, &mass_control_post),
    };
    (balance, distances)
}

/// The numeric rows of the report: the numeric covariates, then, as
/// `interactions = TRUE` of MatchIt orders them, each column's square and its
/// products with every later column. Columns are the numeric covariates and
/// the level indicators of the categorical ones, in the order of the covariate
/// names, with the levels of a covariate in the order of their names.
fn numeric_feature_specs(
    numeric_keys: &[String],
    levels: &LevelTable,
    transforms: NumericBalanceTransform,
) -> Vec<NumericFeature> {
    let mut specs = numeric_keys
        .iter()
        .cloned()
        .map(|key| NumericFeature::Raw { key })
        .collect_vec();
    if transforms == NumericBalanceTransform::None {
        return specs;
    }
    let numeric = numeric_keys
        .iter()
        .collect::<std::collections::HashSet<_>>();
    let columns = numeric_keys
        .iter()
        .chain(levels.keys())
        .sorted()
        .flat_map(|key| {
            if numeric.contains(key) {
                vec![Column::Numeric(key.clone())]
            } else {
                levels[key.as_str()]
                    .iter()
                    .map(|level| Column::Level {
                        key: key.clone(),
                        level: level.clone(),
                        known: Arc::clone(&levels[key.as_str()]),
                    })
                    .collect_vec()
            }
        })
        .collect_vec();
    for (first, left) in columns.iter().enumerate() {
        match transforms {
            NumericBalanceTransform::None => {}
            NumericBalanceTransform::Squares => {
                if matches!(left, Column::Numeric(_)) {
                    specs.push(NumericFeature::Product {
                        left: left.clone(),
                        right: left.clone(),
                    });
                }
            }
            NumericBalanceTransform::SquaresAndPairwiseInteractions => {
                for right in &columns[first..] {
                    specs.push(NumericFeature::Product {
                        left: left.clone(),
                        right: right.clone(),
                    });
                }
            }
        }
    }
    specs
}

fn numeric_covariate<R: CovariateRecord>(
    reader: &mut ColumnReader<'_>,
    record: &R,
    supplemental: &SupplementalCovariateMap,
) -> Option<f64> {
    match covariate_value(reader, record, supplemental) {
        Some(CovariateRef::Numeric(value)) if value.is_finite() => Some(value),
        _ => None,
    }
}

fn split_pair(pair: Option<(f64, f64)>) -> (Option<f64>, Option<f64>) {
    (pair.map(|(first, _)| first), pair.map(|(_, second)| second))
}

fn collect_levels<R: CovariateRecord>(
    cases: &[R],
    controls: &[R],
    case_supplemental: &SupplementalCovariateMap,
    control_supplemental: &SupplementalCovariateMap,
    key: &str,
) -> Vec<String> {
    let mut reader = ColumnReader::new(key);
    let levels = cases
        .iter()
        .map(|record| (record, case_supplemental))
        .chain(controls.iter().map(|record| (record, control_supplemental)))
        .filter_map(|(record, supplemental)| covariate_value(&mut reader, record, supplemental))
        .map(map_covariate_to_level)
        .sorted()
        .dedup()
        .collect_vec();

    if levels.is_empty() {
        vec!["missing".to_string()]
    } else {
        levels
    }
}

fn map_covariate_to_level(value: CovariateRef<'_>) -> String {
    match value {
        CovariateRef::Categorical(value) => value.to_owned(),
        CovariateRef::Numeric(_) => "numeric".to_string(),
        CovariateRef::Missing => "missing".to_string(),
    }
}

fn map_optional_covariate_to_level(value: Option<CovariateRef<'_>>) -> String {
    value.map_or_else(|| "missing".to_string(), map_covariate_to_level)
}

fn covariate_value<'a, R: CovariateRecord>(
    reader: &mut ColumnReader<'_>,
    record: &'a R,
    supplemental: &'a SupplementalCovariateMap,
) -> Option<CovariateRef<'a>> {
    reader.read(record).or_else(|| {
        if supplemental.is_empty() {
            return None;
        }
        supplemental
            .get(record.id())
            .and_then(|values| values.get(reader.key()))
            .map(CovariateRef::from)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::balance::stats::{eqq_distance_weighted, weighted_ecdf_distance};
    use crate::matching::MatchingRecord;
    use crate::types::{
        BalanceRecord, BalanceReportOptions, MatchedPair, NumericBalanceTransform,
        SupplementalBalanceCovariates,
    };
    use crate::{date, test_outcome};
    use std::collections::HashMap;

    fn participant(id: &str) -> BalanceRecord {
        BalanceRecord::new(id, date(2010, 1, 1))
    }

    #[test]
    fn report_includes_numeric_and_categorical_covariates() {
        let mut case_a = participant("case_a");
        case_a
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(10.0));
        case_a.covariates.insert(
            "region".to_string(),
            CovariateValue::Categorical("north".to_string()),
        );

        let mut case_b = participant("case_b");
        case_b
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(14.0));
        case_b.covariates.insert(
            "region".to_string(),
            CovariateValue::Categorical("south".to_string()),
        );

        let mut control_a = participant("control_a");
        control_a
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(8.0));
        control_a.covariates.insert(
            "region".to_string(),
            CovariateValue::Categorical("north".to_string()),
        );

        let mut control_b = participant("control_b");
        control_b
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(16.0));
        control_b.covariates.insert(
            "region".to_string(),
            CovariateValue::Categorical("south".to_string()),
        );

        let report = balance_report(
            &[case_a, case_b],
            &[control_a, control_b],
            &test_outcome(vec![MatchedPair::new("case_a", "control_b")], 1, 0),
        );

        assert_eq!(report.numeric.len(), 1);
        assert_eq!(report.categorical.len(), 1);
        assert_eq!(report.numeric[0].name, "age");
        assert_eq!(report.categorical[0].name, "region");
    }

    #[test]
    fn report_uses_only_matched_pairs_for_post_values() {
        let mut case_a = participant("case_a");
        case_a
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(10.0));
        let mut case_b = participant("case_b");
        case_b
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(30.0));

        let mut control_a = participant("control_a");
        control_a
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(20.0));
        let mut control_b = participant("control_b");
        control_b
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(40.0));

        let report = balance_report(
            &[case_a, case_b],
            &[control_a, control_b],
            &test_outcome(vec![MatchedPair::new("case_a", "control_b")], 1, 0),
        );

        let numeric = &report.numeric[0];
        assert!((numeric.mean_case_post.unwrap() - 10.0).abs() < 1e-12);
        assert!((numeric.mean_control_post.unwrap() - 40.0).abs() < 1e-12);
    }

    #[test]
    fn report_with_options_includes_squared_numeric_terms() {
        let mut case_a = participant("case_a");
        case_a
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(2.0));
        let mut case_b = participant("case_b");
        case_b
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(4.0));

        let mut control_a = participant("control_a");
        control_a
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(1.0));
        let mut control_b = participant("control_b");
        control_b
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(3.0));

        let report = balance_report_with_options(
            &[case_a, case_b],
            &[control_a, control_b],
            &test_outcome(vec![MatchedPair::new("case_a", "control_b")], 1, 0),
            &BalanceReportOptions::builder()
                .numeric_transforms(NumericBalanceTransform::Squares)
                .build(),
        );

        let names = report
            .numeric
            .iter()
            .map(|metric| metric.name.as_str())
            .collect_vec();
        assert_eq!(names, vec!["age", "age^2"]);
    }

    #[test]
    fn report_with_options_includes_pairwise_numeric_interactions() {
        let mut case_a = participant("case_a");
        case_a
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(2.0));
        case_a
            .covariates
            .insert("income".to_string(), CovariateValue::Numeric(10.0));
        let mut case_b = participant("case_b");
        case_b
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(3.0));
        case_b
            .covariates
            .insert("income".to_string(), CovariateValue::Numeric(12.0));

        let mut control_a = participant("control_a");
        control_a
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(1.0));
        control_a
            .covariates
            .insert("income".to_string(), CovariateValue::Numeric(8.0));
        let mut control_b = participant("control_b");
        control_b
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(2.0));
        control_b
            .covariates
            .insert("income".to_string(), CovariateValue::Numeric(9.0));

        let report = balance_report_with_options(
            &[case_a, case_b],
            &[control_a, control_b],
            &test_outcome(vec![MatchedPair::new("case_a", "control_b")], 1, 0),
            &BalanceReportOptions::builder()
                .numeric_transforms(NumericBalanceTransform::SquaresAndPairwiseInteractions)
                .build(),
        );

        let names = report
            .numeric
            .iter()
            .map(|metric| metric.name.clone())
            .collect_vec();
        assert!(names.contains(&"age".to_string()));
        assert!(names.contains(&"income".to_string()));
        assert!(names.contains(&"age^2".to_string()));
        assert!(names.contains(&"income^2".to_string()));
        assert!(names.contains(&"age * income".to_string()));
    }

    #[test]
    fn helper_functions_handle_missing_and_non_finite_values() {
        let mut case_a = participant("case_a");
        case_a
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(f64::NAN));
        case_a
            .covariates
            .insert("region".to_string(), CovariateValue::Missing);
        let mut control_a = participant("control_a");
        control_a
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(12.0));

        let keys = collect_covariate_keys(
            &[case_a.clone()],
            &[control_a.clone()],
            &HashMap::new(),
            &HashMap::new(),
        );
        assert_eq!(keys, vec!["age".to_string(), "region".to_string()]);
        // Through the splitter rather than the per-key classifier it replaced:
        // same question, and it also pins which side each key lands on.
        // `age` is Numeric even though the case's value is NaN, and `region`
        // is Categorical because Missing is not Numeric.
        let (numeric, categorical) = split_covariate_keys_by_kind(
            &[case_a.clone()],
            &[control_a.clone()],
            &HashMap::new(),
            &HashMap::new(),
        );
        assert_eq!(numeric, vec!["age".to_string()]);
        assert_eq!(categorical, vec!["region".to_string()]);
        let raw_age = Column::Numeric("age".to_string());
        assert_eq!(
            raw_age.value(&mut ColumnReader::new("age"), &case_a, &HashMap::new()),
            None
        );

        let levels = collect_levels(
            &[],
            &[control_a],
            &HashMap::new(),
            &HashMap::new(),
            "region",
        );
        assert_eq!(levels, vec!["missing".to_string()]);
    }

    #[test]
    fn report_can_include_supplemental_covariates() {
        let case_a = participant("case_a");
        let control_a = participant("control_a");
        let mut supplemental_cases = HashMap::new();
        supplemental_cases.insert(
            "case_a".to_string(),
            HashMap::from([
                ("age".to_string(), CovariateValue::Numeric(10.0)),
                (
                    "region".to_string(),
                    CovariateValue::Categorical("north".to_string()),
                ),
            ]),
        );
        let mut supplemental_controls = HashMap::new();
        supplemental_controls.insert(
            "control_a".to_string(),
            HashMap::from([
                ("age".to_string(), CovariateValue::Numeric(8.0)),
                (
                    "region".to_string(),
                    CovariateValue::Categorical("south".to_string()),
                ),
            ]),
        );

        let report = balance_report_with_options(
            &[case_a],
            &[control_a],
            &test_outcome(vec![MatchedPair::new("case_a", "control_a")], 1, 0),
            &BalanceReportOptions {
                numeric_transforms: NumericBalanceTransform::None,
                supplemental_covariates: SupplementalBalanceCovariates {
                    cases: supplemental_cases,
                    controls: supplemental_controls,
                },
            },
        );

        assert_eq!(report.numeric.len(), 1);
        assert_eq!(report.numeric[0].name, "age");
        assert_eq!(report.categorical.len(), 1);
        assert_eq!(report.categorical[0].name, "region");
    }

    #[test]
    fn numeric_balance_includes_variance_ratio_and_ecdf_distances() {
        let mut case_a = participant("case_a");
        case_a
            .covariates
            .insert("x".to_string(), CovariateValue::Numeric(1.0));
        let mut case_b = participant("case_b");
        case_b
            .covariates
            .insert("x".to_string(), CovariateValue::Numeric(1.0));

        let mut control_a = participant("control_a");
        control_a
            .covariates
            .insert("x".to_string(), CovariateValue::Numeric(1.0));
        let mut control_b = participant("control_b");
        control_b
            .covariates
            .insert("x".to_string(), CovariateValue::Numeric(2.0));

        let report = balance_report(
            &[case_a, case_b],
            &[control_a, control_b],
            &test_outcome(vec![MatchedPair::new("case_a", "control_b")], 1, 0),
        );
        let numeric = &report.numeric[0];
        assert!((numeric.var_ratio_pre.unwrap() - 0.0).abs() < 1e-12);
        assert!((numeric.ecdf_max_diff_pre.unwrap() - 0.5).abs() < 1e-12);
        assert!(numeric.ecdf_mean_diff_pre.unwrap() > 0.0);
        assert!((numeric.eqq_max_diff_pre.unwrap() - 1.0).abs() < 1e-12);
        assert!(numeric.eqq_mean_diff_pre.unwrap() > 0.0);
    }

    #[test]
    fn constant_covariate_has_undefined_smd_and_variance_ratio() {
        let mut case_a = participant("case_a");
        case_a
            .covariates
            .insert("x".to_string(), CovariateValue::Numeric(3.0));
        let mut control_a = participant("control_a");
        control_a
            .covariates
            .insert("x".to_string(), CovariateValue::Numeric(3.0));

        let report = balance_report(
            &[case_a],
            &[control_a],
            &test_outcome(vec![MatchedPair::new("case_a", "control_a")], 1, 0),
        );
        let numeric = &report.numeric[0];
        assert_eq!(numeric.mean_case_pre, Some(3.0));
        assert_eq!(numeric.smd_pre, None);
        assert_eq!(numeric.var_ratio_pre, None);
        assert_eq!(numeric.smd_post, None);
        assert_eq!(numeric.var_ratio_post, None);

        let summary = balance_threshold_summary(&report, &BalanceThresholds::default());
        assert_eq!(summary.numeric[0].smd_post_ok, Some(false));
        assert_eq!(summary.numeric[0].var_ratio_post_ok, Some(false));
        assert!(!summary.all_enabled_checks_ok);
    }

    #[test]
    fn empty_matched_group_has_undefined_post_statistics() {
        let mut case_a = participant("case_a");
        case_a
            .covariates
            .insert("x".to_string(), CovariateValue::Numeric(1.0));
        let mut control_a = participant("control_a");
        control_a
            .covariates
            .insert("x".to_string(), CovariateValue::Numeric(2.0));

        let report = balance_report(&[case_a], &[control_a], &test_outcome(Vec::new(), 0, 1));
        let numeric = &report.numeric[0];
        assert_eq!(numeric.mean_case_post, None);
        assert_eq!(numeric.mean_control_post, None);
        assert_eq!(numeric.smd_post, None);
        assert_eq!(numeric.ecdf_max_diff_post, None);
        assert_eq!(numeric.eqq_mean_diff_post, None);
    }

    #[test]
    fn threshold_summary_reports_pass_and_fail() {
        let report = BalanceReport {
            numeric: vec![NumericBalance {
                name: "x".to_string(),
                mean_case_pre: Some(0.0),
                mean_control_pre: Some(0.0),
                smd_pre: Some(0.0),
                var_ratio_pre: Some(1.0),
                ecdf_mean_diff_pre: Some(0.0),
                ecdf_max_diff_pre: Some(0.0),
                eqq_mean_diff_pre: Some(0.0),
                eqq_max_diff_pre: Some(0.0),
                mean_case_post: Some(0.0),
                mean_control_post: Some(0.0),
                smd_post: Some(0.05),
                var_ratio_post: Some(1.0),
                ecdf_mean_diff_post: Some(0.02),
                ecdf_max_diff_post: Some(0.06),
                eqq_mean_diff_post: Some(0.03),
                eqq_max_diff_post: Some(0.40),
            }],
            categorical: Vec::new(),
        };

        let passing = balance_threshold_summary(
            &report,
            &BalanceThresholds {
                eqq_max_diff_max: Some(0.5),
                ..BalanceThresholds::default()
            },
        );
        assert!(passing.all_enabled_checks_ok);
        assert_eq!(passing.numeric[0].smd_post_ok, Some(true));
        assert_eq!(passing.numeric[0].eqq_max_diff_post_ok, Some(true));

        let failing = balance_threshold_summary(
            &report,
            &BalanceThresholds {
                ecdf_max_diff_max: Some(0.05),
                ..BalanceThresholds::default()
            },
        );
        assert!(!failing.all_enabled_checks_ok);
        assert_eq!(failing.numeric[0].ecdf_max_diff_post_ok, Some(false));
    }

    #[test]
    fn threshold_summary_supports_disabling_checks() {
        let report = BalanceReport {
            numeric: vec![NumericBalance {
                name: "x".to_string(),
                mean_case_pre: Some(0.0),
                mean_control_pre: Some(0.0),
                smd_pre: Some(0.0),
                var_ratio_pre: Some(1.0),
                ecdf_mean_diff_pre: Some(0.0),
                ecdf_max_diff_pre: Some(0.0),
                eqq_mean_diff_pre: Some(0.0),
                eqq_max_diff_pre: Some(0.0),
                mean_case_post: Some(0.0),
                mean_control_post: Some(0.0),
                smd_post: Some(5.0),
                var_ratio_post: Some(5.0),
                ecdf_mean_diff_post: Some(5.0),
                ecdf_max_diff_post: Some(5.0),
                eqq_mean_diff_post: Some(5.0),
                eqq_max_diff_post: Some(5.0),
            }],
            categorical: Vec::new(),
        };

        let summary = balance_threshold_summary(
            &report,
            &BalanceThresholds {
                smd_abs_max: None,
                var_ratio_min: None,
                var_ratio_max: None,
                ecdf_max_diff_max: None,
                eqq_max_diff_max: None,
            },
        );
        assert!(summary.all_enabled_checks_ok);
        assert_eq!(summary.numeric[0].smd_post_ok, None);
    }

    fn numeric_record(id: &str, age: f64, region: &str) -> BalanceRecord {
        let mut record = participant(id);
        record
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(age));
        record.covariates.insert(
            "region".to_string(),
            CovariateValue::Categorical(region.to_string()),
        );
        record
    }

    fn small_groups() -> (Vec<BalanceRecord>, Vec<BalanceRecord>) {
        (
            vec![
                numeric_record("t1", 10.0, "north"),
                numeric_record("t2", 14.0, "south"),
            ],
            vec![
                numeric_record("c1", 8.0, "north"),
                numeric_record("c2", 12.0, "south"),
                numeric_record("c3", 20.0, "south"),
            ],
        )
    }

    #[test]
    fn the_pair_report_is_the_report_on_the_weights_of_its_pairs() {
        let (cases, controls) = small_groups();
        let outcome = test_outcome(
            vec![
                MatchedPair::new("t1", "c1"),
                MatchedPair::new("t2", "c2"),
                MatchedPair::new("t2", "c3"),
            ],
            2,
            0,
        );
        let options = BalanceReportOptions::builder()
            .numeric_transforms(NumericBalanceTransform::SquaresAndPairwiseInteractions)
            .build();
        let pairs = balance_report_with_options(&cases, &controls, &outcome, &options);
        let weights = BalanceWeights::from_outcome(&outcome);
        let on_weights = balance_report_on_weights(&cases, &controls, &weights, &options).unwrap();
        assert_eq!(format!("{pairs:?}"), format!("{on_weights:?}"));
    }

    #[test]
    fn without_matching_the_statistics_after_equal_those_before() {
        let (cases, controls) = small_groups();
        let weights = BalanceWeights::unmatched(&cases, &controls, Estimand::Att);
        let summary = balance_summary(
            &cases,
            &controls,
            &weights,
            &BalanceReportOptions::default(),
        )
        .unwrap();
        let age = &summary.report.numeric[0];
        let close = |a: Option<f64>, b: Option<f64>| (a.unwrap() - b.unwrap()).abs() < 1e-12;
        assert!(close(age.mean_case_pre, age.mean_case_post));
        assert!(close(age.smd_pre, age.smd_post));
        assert!(close(age.eqq_mean_diff_pre, age.eqq_mean_diff_post));
        assert_eq!(summary.sample_sizes.cases.matched, 2);
        assert_eq!(summary.sample_sizes.controls.unmatched, 0);
        assert_eq!(summary.pair_distances.len(), 0);
    }

    #[test]
    fn invalid_weights_are_an_error() {
        let (cases, controls) = small_groups();
        let mut weights = BalanceWeights::unmatched(&cases, &controls, Estimand::Att);
        weights.matching.anchor.insert("t1".to_string(), -1.0);
        let result = balance_summary(
            &cases,
            &controls,
            &weights,
            &BalanceReportOptions::default(),
        );
        assert_eq!(
            result.unwrap_err(),
            BalanceInputError::InvalidWeight {
                id: "t1".to_string(),
                weight: -1.0
            }
        );
        let sampling = HashMap::from([("c1".to_string(), f64::NAN)]);
        let weights = BalanceWeights::unmatched(&cases, &controls, Estimand::Att)
            .with_sampling_weights(sampling);
        assert!(weights.validate().is_err());
    }

    #[test]
    fn sample_sizes_separate_unmatched_from_discarded() {
        let (cases, controls) = small_groups();
        let given = HashMap::from([("t1".to_string(), 1.0), ("c1".to_string(), 1.0)]);
        let weights = BalanceWeights::from_unit_weights(&cases, &controls, &given, Estimand::Att)
            .with_discarded(["c3".to_string()]);
        let summary = balance_summary(
            &cases,
            &controls,
            &weights,
            &BalanceReportOptions::default(),
        )
        .unwrap();
        let sizes = summary.sample_sizes;
        assert_eq!((sizes.cases.all, sizes.cases.matched), (2, 1));
        assert_eq!((sizes.cases.unmatched, sizes.cases.discarded), (1, 0));
        assert_eq!((sizes.controls.all, sizes.controls.matched), (3, 1));
        assert_eq!((sizes.controls.unmatched, sizes.controls.discarded), (1, 1));
        assert_eq!(sizes.controls.all_ess, Some(3.0));
        assert_eq!(sizes.controls.matched_ess, Some(1.0));
    }

    #[test]
    fn sampling_weights_enter_before_and_after_matching() {
        let (cases, controls) = small_groups();
        let weights =
            BalanceWeights::unmatched(&cases, &controls, Estimand::Att).with_sampling_weights(
                HashMap::from([("c1".to_string(), 3.0), ("c2".to_string(), 1.0)]),
            );
        let summary = balance_summary(
            &cases,
            &controls,
            &weights,
            &BalanceReportOptions::default(),
        )
        .unwrap();
        // Controls weigh 3, 1 and 1 (a missing weight is one): 8, 12 and 20.
        let mean = (3.0 * 8.0 + 12.0 + 20.0) / 5.0;
        let age = &summary.report.numeric[0];
        assert!((age.mean_control_pre.unwrap() - mean).abs() < 1e-12);
        assert!((age.mean_control_post.unwrap() - mean).abs() < 1e-12);
        let ess = summary.sample_sizes.controls.all_ess.unwrap();
        assert!((ess - 25.0 / 11.0).abs() < 1e-12);
    }

    #[test]
    fn improvement_is_none_where_the_statistic_is_zero_before_matching() {
        let report = BalanceReport {
            numeric: vec![
                NumericBalance {
                    name: "a".to_string(),
                    smd_pre: Some(0.4),
                    smd_post: Some(-0.1),
                    var_ratio_pre: Some(2.0),
                    var_ratio_post: Some(1.0),
                    ecdf_mean_diff_pre: Some(0.0),
                    ecdf_mean_diff_post: Some(0.0),
                    ecdf_max_diff_pre: Some(0.2),
                    ecdf_max_diff_post: None,
                    ..NumericBalance::default()
                },
                NumericBalance {
                    name: "b".to_string(),
                    var_ratio_pre: Some(0.5),
                    var_ratio_post: Some(1.25),
                    ..NumericBalance::default()
                },
            ],
            categorical: Vec::new(),
        };
        let improvement = balance_improvement(&report);
        let a = &improvement.numeric[0];
        assert!((a.smd.unwrap() - 75.0).abs() < 1e-12);
        assert!((a.var_ratio.unwrap() - 100.0).abs() < 1e-12);
        assert_eq!(a.ecdf_mean, None);
        assert_eq!(a.ecdf_max, None);
        // A ratio of 0.5 is as far from 1 as a ratio of 2, so 1.25 is closer.
        let expected = 100.0 * (0.5_f64.ln().abs() - 1.25_f64.ln().abs()) / 0.5_f64.ln().abs();
        assert!((improvement.numeric[1].var_ratio.unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn pair_distance_averages_over_pairs_including_repeated_ones() {
        let (cases, controls) = small_groups();
        let outcome = test_outcome(
            vec![
                MatchedPair::new("t1", "c1"),
                MatchedPair::new("t2", "c2"),
                MatchedPair::new("t2", "c3"),
            ],
            2,
            0,
        );
        let weights = BalanceWeights::from_outcome(&outcome);
        let summary = balance_summary(
            &cases,
            &controls,
            &weights,
            &BalanceReportOptions::default(),
        )
        .unwrap();
        // |10 - 8|, |14 - 12| and |14 - 20| average to 10 / 3.
        let age = &summary.pair_distances[0];
        assert_eq!(age.name, "age");
        assert!((age.raw.unwrap() - 10.0 / 3.0).abs() < 1e-12);
        // Standardised by the cases' standard deviation, sqrt(8).
        assert!((age.standardised.unwrap() - 10.0 / 3.0 / 8.0_f64.sqrt()).abs() < 1e-12);
        // Pairs with a different region: t2 and c2 share one, t2 and c3 too, t1 and c1 too.
        let north = summary
            .pair_distances
            .iter()
            .find(|distance| distance.name == "regionnorth")
            .unwrap();
        assert_eq!(north.raw, Some(0.0));
    }

    #[test]
    fn weighted_cramers_v_uses_the_weights_as_counts() {
        let (cases, controls) = small_groups();
        // Cases weigh 1 in north and 1 in south; controls weigh 2 in north (c1
        // doubled) and 1 in south (c2 only): the table is [1, 1] and [2, 1].
        let given = HashMap::from([
            ("t1".to_string(), 1.0),
            ("t2".to_string(), 1.0),
            ("c1".to_string(), 2.0),
            ("c2".to_string(), 1.0),
        ]);
        let weights = BalanceWeights::from_unit_weights(&cases, &controls, &given, Estimand::Att);
        let report = balance_report_on_weights(
            &cases,
            &controls,
            &weights,
            &BalanceReportOptions::default(),
        )
        .unwrap();
        // n = 5, north row 3 and south row 2: expected 1.2, 0.8 for the cases and
        // 1.8, 1.2 for the controls. chi-square = 0.04/1.2 + 0.04/0.8 + 0.04/1.8 + 0.04/1.2.
        let chi2 = 0.04 / 1.2 + 0.04 / 0.8 + 0.04 / 1.8 + 0.04 / 1.2;
        let expected = (chi2 / 5.0_f64).sqrt();
        assert!((report.categorical[0].cramers_v_post.unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn interactions_include_the_dummies_of_a_factor_in_column_order() {
        let (cases, controls) = small_groups();
        let outcome = test_outcome(vec![MatchedPair::new("t1", "c1")], 1, 0);
        let options = BalanceReportOptions::builder()
            .numeric_transforms(NumericBalanceTransform::SquaresAndPairwiseInteractions)
            .build();
        let report = balance_report_with_options(&cases, &controls, &outcome, &options);
        let names = report
            .numeric
            .iter()
            .map(|row| row.name.as_str())
            .collect_vec();
        // Columns are age, regionnorth and regionsouth. A dummy's square and the
        // product of the dummies of one factor are dropped.
        assert_eq!(
            names,
            vec!["age", "age^2", "age * regionnorth", "age * regionsouth"]
        );
    }

    #[test]
    fn square_of_a_small_scale_covariate_is_kept() {
        let make = |prefix: &str, offset: f64| -> Vec<BalanceRecord> {
            (0..6)
                .map(|i| {
                    let mut r = participant(&format!("{prefix}{i}"));
                    r.covariates.insert(
                        "x".to_string(),
                        CovariateValue::Numeric(1e-6 * (f64::from(i) + offset)),
                    );
                    r
                })
                .collect()
        };
        let cases = make("c", 1.0);
        let controls = make("k", 2.0);
        let weights = BalanceWeights::unmatched(&cases, &controls, Estimand::Att);
        let options = BalanceReportOptions {
            numeric_transforms: NumericBalanceTransform::Squares,
            ..BalanceReportOptions::default()
        };
        let report = balance_report_on_weights(&cases, &controls, &weights, &options).unwrap();
        assert!(report.numeric.iter().any(|row| row.name == "x^2"));
    }

    fn feature_units_with_ids<'a, R: CovariateRecord>(
        sample: &BalanceSample<'a, R>,
        context: BalanceCovariateContext<'_>,
        feature: &NumericFeature,
    ) -> Vec<(Unit<f64>, &'a str)> {
        let mut units = Vec::new();
        let mut readers = feature.readers();
        for (records, weights, is_case, supplemental) in [
            (
                sample.cases,
                &sample.case_weights,
                true,
                context.case_supplemental,
            ),
            (
                sample.controls,
                &sample.control_weights,
                false,
                context.control_supplemental,
            ),
        ] {
            for (record, weights) in records.iter().zip(weights) {
                if let Some(value) = feature.value(&mut readers, record, supplemental) {
                    units.push((
                        Unit {
                            value,
                            is_case,
                            weights: *weights,
                        },
                        record.id(),
                    ));
                }
            }
        }
        units
    }

    fn summarize_numeric_feature_reference<R: CovariateRecord>(
        sample: &BalanceSample<'_, R>,
        context: BalanceCovariateContext<'_>,
        feature: &NumericFeature,
    ) -> (NumericBalance, PairDistance) {
        let identified = feature_units_with_ids(sample, context, feature);
        let units = identified.iter().map(|unit| unit.0).collect_vec();
        let binary = !units.is_empty() && units.iter().all(|unit| is_zero_or_one(unit.value));
        let standardiser = Standardiser::for_estimand(sample.estimand);

        let pre = Stage {
            matched: false,
            sampled: sample.sampled,
        };
        let post = Stage {
            matched: true,
            ..pre
        };
        let case_pre = group_moments(&units, true, pre, binary);
        let control_pre = group_moments(&units, false, pre, binary);
        let case_post = group_moments(&units, true, post, binary);
        let control_post = group_moments(&units, false, post, binary);
        let reference = standardiser.variance(case_pre.variance, control_pre.variance);

        let difference = |case: GroupMoments, control: GroupMoments| {
            case.mean.zip(control.mean).map(|(a, b)| a - b)
        };
        let smd = |case: GroupMoments, control: GroupMoments| {
            difference(case, control).and_then(|value| standardised_difference(value, reference))
        };
        let ratio = |case: GroupMoments, control: GroupMoments| {
            if binary {
                None
            } else {
                case.variance
                    .zip(control.variance)
                    .and_then(|(a, b)| variance_ratio(a, b))
            }
        };

        // Binary variables: both distribution distances are the absolute
        // difference in proportions.
        let ecdf = |matched: bool, case: GroupMoments, control: GroupMoments| {
            if binary {
                let gap = difference(case, control).map(f64::abs);
                (gap, gap)
            } else {
                let points = units
                    .iter()
                    .map(|unit| (unit.value, unit.is_case, unit.weight(matched)))
                    .collect_vec();
                split_pair(weighted_ecdf_distance(&points))
            }
        };
        let eqq = |matched: bool, case: GroupMoments, control: GroupMoments| {
            if binary {
                let gap = difference(case, control).map(f64::abs);
                (gap, gap)
            } else {
                let group = |is_case: bool| {
                    units
                        .iter()
                        .filter(|unit| unit.is_case == is_case)
                        .map(|unit| (unit.value, unit.weight(matched)))
                        .unzip::<f64, f64, Vec<f64>, Vec<f64>>()
                };
                let ((case_values, case_weights), (control_values, control_weights)) =
                    (group(true), group(false));
                split_pair(eqq_distance_weighted(
                    &case_values,
                    &case_weights,
                    &control_values,
                    &control_weights,
                ))
            }
        };
        let (ecdf_mean_diff_pre, ecdf_max_diff_pre) = ecdf(false, case_pre, control_pre);
        let (ecdf_mean_diff_post, ecdf_max_diff_post) = ecdf(true, case_post, control_post);
        let (eqq_mean_diff_pre, eqq_max_diff_pre) = eqq(false, case_pre, control_pre);
        let (eqq_mean_diff_post, eqq_max_diff_post) = eqq(true, case_post, control_post);

        let name = feature.name();
        let distance =
            pair_distance_by_id(sample, &name, &identified, standardiser_scale(reference));
        let balance = NumericBalance {
            name,
            mean_case_pre: case_pre.mean,
            mean_control_pre: control_pre.mean,
            smd_pre: smd(case_pre, control_pre),
            var_ratio_pre: ratio(case_pre, control_pre),
            ecdf_mean_diff_pre,
            ecdf_max_diff_pre,
            eqq_mean_diff_pre,
            eqq_max_diff_pre,
            mean_case_post: case_post.mean,
            mean_control_post: control_post.mean,
            smd_post: smd(case_post, control_post),
            var_ratio_post: ratio(case_post, control_post),
            ecdf_mean_diff_post,
            ecdf_max_diff_post,
            eqq_mean_diff_post,
            eqq_max_diff_post,
        };
        (balance, distance)
    }

    fn pair_distance_by_id<R: CovariateRecord>(
        sample: &BalanceSample<'_, R>,
        name: &str,
        units: &[(Unit<f64>, &str)],
        scale: Option<f64>,
    ) -> PairDistance {
        let by_group = |is_case: bool| {
            units
                .iter()
                .filter(|(unit, _)| unit.is_case == is_case)
                .map(|(unit, id)| (*id, unit.value))
                .collect::<HashMap<_, _>>()
        };
        let (case_values, control_values) = (by_group(true), by_group(false));
        let gaps = sample.pairs.iter().filter_map(|pair| {
            let case = case_values.get(pair.case_id.as_str())?;
            let control = control_values.get(pair.control_id.as_str())?;
            Some((case - control).abs())
        });
        mean_gap(name, gaps, scale)
    }

    fn tied_record(id: &str, values: &[(&str, f64)]) -> BalanceRecord {
        let mut record = participant(id);
        for (key, value) in values {
            record
                .covariates
                .insert((*key).to_string(), CovariateValue::Numeric(*value));
        }
        record
    }

    /// A small deterministic generator, so that the cases are fixed.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 33
        }

        fn below(&mut self, bound: usize) -> usize {
            usize::try_from(self.next() % u64::try_from(bound).unwrap()).unwrap()
        }

        /// A value from a short list, so that values tie often, including
        /// negative and positive zero.
        fn tied_value(&mut self) -> f64 {
            [-0.0, 0.0, 1.0, -1.5, 2.0, 2.5, 3.0][self.below(7)]
        }
    }

    /// The numeric rows and pair distances of the report equal those of the
    /// reference that sorts for each statistic and reads pairs by identifier,
    /// on data with many ties and zeros of both signs, with and without
    /// duplicated identifiers, weights of zero and sampling weights.
    #[test]
    fn numeric_rows_equal_the_reference_on_tied_data() {
        for (seed, units, duplicate_ids, sampled) in [
            (1_u64, 7_usize, false, false),
            (2, 40, false, false),
            (3, 40, true, false),
            (4, 300, false, true),
            (5, 300, true, true),
            (6, 3, false, false),
        ] {
            let mut rng = Lcg(seed);
            let make = |prefix: &str, rng: &mut Lcg| {
                (0..units)
                    .map(|idx| {
                        let id = if duplicate_ids && idx % 5 == 4 {
                            format!("{prefix}{}", idx - 1)
                        } else {
                            format!("{prefix}{idx}")
                        };
                        tied_record(
                            &id,
                            &[
                                ("a", rng.tied_value()),
                                ("b", f64::from(u8::try_from(rng.below(2)).unwrap())),
                                ("c", rng.tied_value()),
                            ],
                        )
                    })
                    .collect::<Vec<_>>()
            };
            let cases = make("case", &mut rng);
            let controls = make("control", &mut rng);
            let pairs = (0..units)
                .map(|idx| {
                    MatchedPair::new(
                        format!("case{}", rng.below(units)),
                        if idx % 7 == 6 {
                            "absent".to_string()
                        } else {
                            format!("control{}", rng.below(units))
                        },
                    )
                })
                .collect::<Vec<_>>();
            let mut weights = BalanceWeights::unmatched(&cases, &controls, Estimand::Att);
            for record in cases.iter().chain(&controls) {
                let weight = [0.0, 0.1, 0.3, 0.7, 1.1][rng.below(5)];
                let target = if MatchingRecord::id(record).starts_with("case") {
                    &mut weights.matching.anchor
                } else {
                    &mut weights.matching.candidate
                };
                target.insert(MatchingRecord::id(record).to_string(), weight);
            }
            if sampled {
                weights.sampling = Some(
                    cases
                        .iter()
                        .chain(&controls)
                        .map(|record| {
                            (
                                MatchingRecord::id(record).to_string(),
                                0.37 + 0.1 * f64::from(u32::try_from(rng.next() % 4).unwrap()),
                            )
                        })
                        .collect(),
                );
            }
            weights.pairs = pairs;
            let options = BalanceReportOptions {
                numeric_transforms: NumericBalanceTransform::SquaresAndPairwiseInteractions,
                ..BalanceReportOptions::default()
            };

            let summary = balance_summary(&cases, &controls, &weights, &options).unwrap();

            let context = BalanceCovariateContext {
                case_supplemental: &options.supplemental_covariates.cases,
                control_supplemental: &options.supplemental_covariates.controls,
            };
            let sample = BalanceSample {
                cases: &cases,
                controls: &controls,
                case_weights: unit_weights(
                    &cases,
                    &weights.matching.anchor,
                    weights.sampling.as_ref(),
                ),
                control_weights: unit_weights(
                    &controls,
                    &weights.matching.candidate,
                    weights.sampling.as_ref(),
                ),
                sampled,
                estimand: weights.estimand,
                pairs: &weights.pairs,
                pair_positions: None,
            };
            let keys = ["a", "b", "c"].map(String::from);
            let features = numeric_feature_specs(
                &keys,
                &LevelTable::new(),
                NumericBalanceTransform::SquaresAndPairwiseInteractions,
            );
            let expected = features
                .iter()
                .filter(|feature| !is_redundant_term(&sample, context, feature))
                .map(|feature| summarize_numeric_feature_reference(&sample, context, feature))
                .collect::<Vec<_>>();
            let (rows, distances): (Vec<_>, Vec<_>) = expected.into_iter().unzip();
            assert_eq!(
                format!("{:?}", summary.report.numeric),
                format!("{rows:?}"),
                "numeric rows, seed {seed}"
            );
            assert_eq!(
                format!("{:?}", summary.pair_distances),
                format!("{distances:?}"),
                "pair distances, seed {seed}"
            );
        }
    }

    /// The report, with the sort running in parallel on a large feature, is
    /// the same bit for bit on pools of 1, 2, 4 and 12 threads.
    #[test]
    fn report_is_the_same_on_every_pool_size() {
        let mut rng = Lcg(11);
        let make = |prefix: &str, rng: &mut Lcg| {
            (0..40_000)
                .map(|idx| {
                    tied_record(
                        &format!("{prefix}{idx}"),
                        &[("a", rng.tied_value()), ("c", rng.tied_value())],
                    )
                })
                .collect::<Vec<_>>()
        };
        let cases = make("case", &mut rng);
        let controls = make("control", &mut rng);
        let mut weights = BalanceWeights::unmatched(&cases, &controls, Estimand::Att);
        for (idx, weight) in weights.matching.candidate.values_mut().enumerate() {
            *weight = 0.1 * f64::from(u8::try_from(idx % 7).unwrap());
        }
        weights.pairs = (0..20_000)
            .map(|idx| MatchedPair::new(format!("case{idx}"), format!("control{}", idx * 2)))
            .collect();
        let options = BalanceReportOptions {
            numeric_transforms: NumericBalanceTransform::SquaresAndPairwiseInteractions,
            ..BalanceReportOptions::default()
        };
        epistat_testkit::assert_same_on_pools(&[1, 2, 4, 12], || {
            let summary = balance_summary(&cases, &controls, &weights, &options).unwrap();
            let text = format!("{:?}{:?}", summary.report, summary.pair_distances);
            text.bytes().map(f64::from).collect()
        });
    }
}
