use super::stats::{
    cramers_v, eqq_distance_stats, standardised_difference, variance_ratio, weighted_ecdf_distance,
};
use crate::matching::UnitWeightSet;
use crate::types::{
    BalanceRecord, BalanceReport, BalanceReportOptions, BalanceThresholdSummary, BalanceThresholds,
    CategoricalBalance, CategoricalLevelBalance, CovariateValue, Estimand, MatchOutcome,
    NumericBalance, NumericBalanceThresholdCheck, NumericBalanceTransform,
};
use epistat_stats::weighted;
use itertools::Itertools;
use std::collections::HashMap;

/// Compute pre- and post-match covariate balance summaries.
///
/// The cases are the treated group and the controls the comparison group.
/// "Pre" statistics use every record passed in with weight one. "Post"
/// statistics weight each record by the unit weights the outcome gives for its
/// realised estimand (see [`MatchOutcome::match_weights_for_realized_estimand`]),
/// so a record of weight zero does not enter them.
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
/// The eQQ distances and Cramér's V after matching use the matched pairs as
/// rows, so a unit appears once per pair it is in.
#[must_use]
pub fn balance_report(
    cases: &[BalanceRecord],
    controls: &[BalanceRecord],
    outcome: &MatchOutcome,
) -> BalanceReport {
    balance_report_with_options(cases, controls, outcome, &BalanceReportOptions::default())
}

/// Compute pre- and post-match covariate balance summaries with report options.
#[must_use]
pub fn balance_report_with_options(
    cases: &[BalanceRecord],
    controls: &[BalanceRecord],
    outcome: &MatchOutcome,
    options: &BalanceReportOptions,
) -> BalanceReport {
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
    let (cases_post, controls_post) = matched_samples(cases, controls, outcome);
    let weights = outcome.match_weights_for_realized_estimand();
    let estimand = outcome.diagnostics.realized_estimand;
    let sample = BalanceSample {
        cases,
        controls,
        cases_post: &cases_post,
        controls_post: &controls_post,
        weights: &weights,
        estimand,
    };
    let numeric =
        build_numeric_balance(&sample, context, &numeric_keys, options.numeric_transforms);
    let categorical = build_categorical_balance(&sample, context, &categorical_keys);

    BalanceReport {
        numeric,
        categorical,
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

#[derive(Debug, Clone)]
enum NumericFeature {
    Raw { key: String },
    Square { key: String },
    Interaction { left: String, right: String },
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
            Self::Square { key } => format!("{key}^2"),
            Self::Interaction { left, right } => format!("{left} * {right}"),
        }
    }
}

fn collect_covariate_keys(
    cases: &[BalanceRecord],
    controls: &[BalanceRecord],
    case_supplemental: &SupplementalCovariateMap,
    control_supplemental: &SupplementalCovariateMap,
) -> Vec<String> {
    cases
        .iter()
        .flat_map(|record| record.covariates.keys().cloned())
        .chain(
            controls
                .iter()
                .flat_map(|record| record.covariates.keys().cloned()),
        )
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

fn split_covariate_keys_by_kind(
    cases: &[BalanceRecord],
    controls: &[BalanceRecord],
    case_supplemental: &SupplementalCovariateMap,
    control_supplemental: &SupplementalCovariateMap,
) -> (Vec<String>, Vec<String>) {
    // One sweep of the records rather than one per key. `covariate_kind`
    // returns on the first record carrying the key, so this was cheap for a
    // key the first record has and a full two-list scan for one that only
    // appears late -- or, for a supplemental key no BalanceRecord carries at
    // all, every time.
    //
    // First-wins in the same order preserves the answer exactly: cases before
    // controls, and within each the record order, which is what the per-key
    // search did.
    let mut kinds: HashMap<&str, CovariateKind> = HashMap::new();
    for (records, supplemental) in [(cases, case_supplemental), (controls, control_supplemental)] {
        for record in records {
            // The record's own covariates, then whatever the supplemental map
            // adds for this id -- the two places `covariate_value` looks, in
            // the order it looks in them.
            let supplemental_for_record = supplemental.get(record.core.id.as_str());
            let keys = record.covariates.iter().chain(
                supplemental_for_record
                    .into_iter()
                    .flat_map(|values| values.iter()),
            );
            for (key, value) in keys {
                kinds.entry(key.as_str()).or_insert(match value {
                    CovariateValue::Numeric(_) => CovariateKind::Numeric,
                    _ => CovariateKind::Categorical,
                });
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

/// The records of one balance report: the groups before matching, the matched
/// pairs as rows, and the unit weights.
struct BalanceSample<'a> {
    cases: &'a [BalanceRecord],
    controls: &'a [BalanceRecord],
    cases_post: &'a [BalanceRecord],
    controls_post: &'a [BalanceRecord],
    weights: &'a UnitWeightSet,
    estimand: Estimand,
}

/// One record's value of a feature, its group and its matching weight.
#[derive(Clone, Copy)]
struct Unit<T> {
    value: T,
    is_case: bool,
    weight: f64,
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

fn build_numeric_balance(
    sample: &BalanceSample<'_>,
    context: BalanceCovariateContext<'_>,
    numeric_keys: &[String],
    transforms: NumericBalanceTransform,
) -> Vec<NumericBalance> {
    numeric_feature_specs(numeric_keys, transforms)
        .iter()
        .filter(|feature| !is_redundant_term(sample, context, feature))
        .map(|feature| summarize_numeric_feature(sample, context, feature))
        .collect_vec()
}

/// A squared or product term that is zero everywhere or equal to one of its
/// factors everywhere adds nothing to the covariates it is built from, as for
/// the square of a binary variable. MatchIt drops these from `interactions =
/// TRUE`.
fn is_redundant_term(
    sample: &BalanceSample<'_>,
    context: BalanceCovariateContext<'_>,
    feature: &NumericFeature,
) -> bool {
    let tolerance = f64::EPSILON.sqrt();
    let factors = match feature {
        NumericFeature::Raw { .. } => return false,
        NumericFeature::Square { key } => vec![key.as_str()],
        NumericFeature::Interaction { left, right } => vec![left.as_str(), right.as_str()],
    };
    let rows = feature_units(sample, context, feature)
        .into_iter()
        .map(|unit| unit.0)
        .collect_vec();
    let all_zero = rows.iter().all(|row| row.value.abs() < tolerance);
    let equals_factor = factors.iter().any(|factor| {
        let raw = NumericFeature::Raw {
            key: (*factor).to_string(),
        };
        let raw_rows = feature_units(sample, context, &raw);
        raw_rows.len() == rows.len()
            && raw_rows
                .iter()
                .zip(&rows)
                .all(|(raw, row)| (raw.0.value - row.value).abs() < tolerance)
    });
    all_zero || equals_factor
}

/// The records of both groups with their value of the feature, pre-match order,
/// cases first. A record without a finite value is left out. The second element
/// is the record's identifier.
fn feature_units(
    sample: &BalanceSample<'_>,
    context: BalanceCovariateContext<'_>,
    feature: &NumericFeature,
) -> Vec<(Unit<f64>, String)> {
    let mut units = Vec::new();
    for (records, is_case, supplemental) in [
        (sample.cases, true, context.case_supplemental),
        (sample.controls, false, context.control_supplemental),
    ] {
        let weights = if is_case {
            &sample.weights.anchor
        } else {
            &sample.weights.candidate
        };
        for record in records {
            if let Some(value) = numeric_feature_value(record, feature, supplemental) {
                units.push((
                    Unit {
                        value,
                        is_case,
                        weight: weights.get(record.core.id.as_str()).copied().unwrap_or(0.0),
                    },
                    record.core.id.clone(),
                ));
            }
        }
    }
    units
}

/// Whether a value is exactly zero or one, as in a binary variable.
#[expect(
    clippy::float_cmp,
    reason = "a binary variable holds exactly zero or one"
)]
fn is_zero_or_one(value: f64) -> bool {
    value == 0.0 || value == 1.0
}

fn group_moments(units: &[Unit<f64>], is_case: bool, matched: bool, binary: bool) -> GroupMoments {
    let (values, weights): (Vec<f64>, Vec<f64>) = units
        .iter()
        .filter(|unit| unit.is_case == is_case && (!matched || unit.weight > 0.0))
        .map(|unit| (unit.value, unit.weight))
        .unzip();
    let weights = matched.then_some(weights);
    let mean = weighted::mean(&values, weights.as_deref());
    let variance = if binary {
        mean.map(|p| p * (1.0 - p))
    } else {
        weighted::variance(&values, weights.as_deref())
    };
    GroupMoments { mean, variance }
}

fn summarize_numeric_feature(
    sample: &BalanceSample<'_>,
    context: BalanceCovariateContext<'_>,
    feature: &NumericFeature,
) -> NumericBalance {
    let units = feature_units(sample, context, feature)
        .into_iter()
        .map(|unit| unit.0)
        .collect_vec();
    let binary = !units.is_empty() && units.iter().all(|unit| is_zero_or_one(unit.value));
    let standardiser = Standardiser::for_estimand(sample.estimand);

    let case_pre = group_moments(&units, true, false, binary);
    let control_pre = group_moments(&units, false, false, binary);
    let case_post = group_moments(&units, true, true, binary);
    let control_post = group_moments(&units, false, true, binary);
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

    // Binary variables: both distribution distances are the absolute
    // difference in proportions.
    let ecdf = |matched: bool, case: GroupMoments, control: GroupMoments| {
        if binary {
            let gap = difference(case, control).map(f64::abs);
            (gap, gap)
        } else {
            let points = units
                .iter()
                .map(|unit| {
                    (
                        unit.value,
                        unit.is_case,
                        if matched { unit.weight } else { 1.0 },
                    )
                })
                .collect_vec();
            split_pair(weighted_ecdf_distance(&points))
        }
    };
    let eqq = |cases: &[BalanceRecord],
               controls: &[BalanceRecord],
               case: GroupMoments,
               control: GroupMoments| {
        if binary {
            let gap = difference(case, control).map(f64::abs);
            (gap, gap)
        } else {
            let case_values = numeric_feature_values(cases, feature, context.case_supplemental);
            let control_values =
                numeric_feature_values(controls, feature, context.control_supplemental);
            split_pair(eqq_distance_stats(&case_values, &control_values))
        }
    };
    let (ecdf_mean_diff_pre, ecdf_max_diff_pre) = ecdf(false, case_pre, control_pre);
    let (ecdf_mean_diff_post, ecdf_max_diff_post) = ecdf(true, case_post, control_post);
    let (eqq_mean_diff_pre, eqq_max_diff_pre) =
        eqq(sample.cases, sample.controls, case_pre, control_pre);
    let (eqq_mean_diff_post, eqq_max_diff_post) = eqq(
        sample.cases_post,
        sample.controls_post,
        case_post,
        control_post,
    );

    NumericBalance {
        name: feature.name(),
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
    }
}

fn build_categorical_balance(
    sample: &BalanceSample<'_>,
    context: BalanceCovariateContext<'_>,
    categorical_keys: &[String],
) -> Vec<CategoricalBalance> {
    categorical_keys
        .iter()
        .map(|key| summarize_categorical_feature(sample, context, key))
        .collect_vec()
}

/// Weighted share of each level among the records of one group, `None` when
/// the group has no weight. `matched` selects the matching weights, otherwise
/// every record has weight one.
fn level_shares(
    units: &[Unit<usize>],
    is_case: bool,
    matched: bool,
    level_count: usize,
) -> Option<Vec<f64>> {
    let mut mass = vec![0.0_f64; level_count];
    let mut total = 0.0_f64;
    for unit in units.iter().filter(|unit| unit.is_case == is_case) {
        let weight = if matched { unit.weight } else { 1.0 };
        mass[unit.value] += weight;
        total += weight;
    }
    (total > 0.0).then(|| mass.into_iter().map(|value| value / total).collect())
}

fn summarize_categorical_feature(
    sample: &BalanceSample<'_>,
    context: BalanceCovariateContext<'_>,
    key: &str,
) -> CategoricalBalance {
    let (cases, controls) = (sample.cases, sample.controls);
    let levels = collect_levels(
        cases,
        controls,
        context.case_supplemental,
        context.control_supplemental,
        key,
    );
    let (counts_case_pre, _) = counts_categorical(cases, key, &levels, context.case_supplemental);
    let (counts_control_pre, _) =
        counts_categorical(controls, key, &levels, context.control_supplemental);
    let (counts_case_post, _) =
        counts_categorical(sample.cases_post, key, &levels, context.case_supplemental);
    let (counts_control_post, _) = counts_categorical(
        sample.controls_post,
        key,
        &levels,
        context.control_supplemental,
    );

    let level_position = levels
        .iter()
        .enumerate()
        .map(|(idx, level)| (level.as_str(), idx))
        .collect::<HashMap<_, _>>();
    let mut units = Vec::new();
    for (records, is_case, supplemental, weights) in [
        (
            cases,
            true,
            context.case_supplemental,
            &sample.weights.anchor,
        ),
        (
            controls,
            false,
            context.control_supplemental,
            &sample.weights.candidate,
        ),
    ] {
        for record in records {
            let level = map_optional_covariate_to_level(covariate_value(record, key, supplemental));
            if let Some(&value) = level_position.get(level.as_str()) {
                units.push(Unit {
                    value,
                    is_case,
                    weight: weights.get(record.core.id.as_str()).copied().unwrap_or(0.0),
                });
            }
        }
    }
    let share_case_pre = level_shares(&units, true, false, levels.len());
    let share_control_pre = level_shares(&units, false, false, levels.len());
    let share_case_post = level_shares(&units, true, true, levels.len());
    let share_control_post = level_shares(&units, false, true, levels.len());
    let standardiser = Standardiser::for_estimand(sample.estimand);

    let levels = levels
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

    CategoricalBalance {
        name: key.to_string(),
        levels,
        cramers_v_pre: cramers_v(&counts_case_pre, &counts_control_pre),
        cramers_v_post: cramers_v(&counts_case_post, &counts_control_post),
    }
}

fn numeric_feature_specs(
    keys: &[String],
    transforms: NumericBalanceTransform,
) -> Vec<NumericFeature> {
    let mut specs = keys
        .iter()
        .cloned()
        .map(|key| NumericFeature::Raw { key })
        .collect_vec();
    if matches!(
        transforms,
        NumericBalanceTransform::Squares | NumericBalanceTransform::SquaresAndPairwiseInteractions
    ) {
        specs.extend(
            keys.iter()
                .cloned()
                .map(|key| NumericFeature::Square { key }),
        );
    }
    if matches!(
        transforms,
        NumericBalanceTransform::SquaresAndPairwiseInteractions
    ) {
        for left_idx in 0..keys.len() {
            for right_idx in (left_idx + 1)..keys.len() {
                specs.push(NumericFeature::Interaction {
                    left: keys[left_idx].clone(),
                    right: keys[right_idx].clone(),
                });
            }
        }
    }
    specs
}

fn matched_samples(
    cases: &[BalanceRecord],
    controls: &[BalanceRecord],
    outcome: &MatchOutcome,
) -> (Vec<BalanceRecord>, Vec<BalanceRecord>) {
    let case_map = cases
        .iter()
        .map(|case| (case.core.id.as_str(), case))
        .collect::<HashMap<&str, &BalanceRecord>>();
    let control_map = controls
        .iter()
        .map(|control| (control.core.id.as_str(), control))
        .collect::<HashMap<&str, &BalanceRecord>>();

    let mut cases_out = Vec::new();
    let mut controls_out = Vec::new();
    for pair in &outcome.pairs {
        if let Some(case) = case_map.get(pair.case_id.as_str()) {
            cases_out.push((*case).clone());
        }
        if let Some(control) = control_map.get(pair.control_id.as_str()) {
            controls_out.push((*control).clone());
        }
    }
    (cases_out, controls_out)
}

fn numeric_feature_values(
    records: &[BalanceRecord],
    feature: &NumericFeature,
    supplemental: &SupplementalCovariateMap,
) -> Vec<f64> {
    records
        .iter()
        .filter_map(|record| numeric_feature_value(record, feature, supplemental))
        .collect()
}

fn numeric_feature_value(
    record: &BalanceRecord,
    feature: &NumericFeature,
    supplemental: &SupplementalCovariateMap,
) -> Option<f64> {
    match feature {
        NumericFeature::Raw { key } => numeric_covariate(record, key, supplemental),
        NumericFeature::Square { key } => {
            numeric_covariate(record, key, supplemental).map(|value| value.powi(2))
        }
        NumericFeature::Interaction { left, right } => {
            numeric_covariate(record, left, supplemental)
                .zip(numeric_covariate(record, right, supplemental))
                .map(|(left, right)| left * right)
        }
    }
}

fn numeric_covariate(
    record: &BalanceRecord,
    key: &str,
    supplemental: &SupplementalCovariateMap,
) -> Option<f64> {
    match covariate_value(record, key, supplemental) {
        Some(CovariateValue::Numeric(value)) if value.is_finite() => Some(*value),
        _ => None,
    }
}

fn split_pair(pair: Option<(f64, f64)>) -> (Option<f64>, Option<f64>) {
    (pair.map(|(first, _)| first), pair.map(|(_, second)| second))
}

fn collect_levels(
    cases: &[BalanceRecord],
    controls: &[BalanceRecord],
    case_supplemental: &SupplementalCovariateMap,
    control_supplemental: &SupplementalCovariateMap,
    key: &str,
) -> Vec<String> {
    let levels = cases
        .iter()
        .filter_map(|record| covariate_value(record, key, case_supplemental))
        .chain(
            controls
                .iter()
                .filter_map(|record| covariate_value(record, key, control_supplemental)),
        )
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

fn counts_categorical(
    records: &[BalanceRecord],
    key: &str,
    levels: &[String],
    supplemental: &SupplementalCovariateMap,
) -> (Vec<usize>, usize) {
    let level_pos = levels
        .iter()
        .enumerate()
        .map(|(idx, level)| (level.as_str(), idx))
        .collect::<HashMap<_, _>>();
    let mut counts = vec![0usize; levels.len()];

    for level in records
        .iter()
        .map(|record| map_optional_covariate_to_level(covariate_value(record, key, supplemental)))
    {
        if let Some(pos) = level_pos.get(level.as_str()) {
            counts[*pos] += 1;
        }
    }

    (counts, records.len())
}

fn map_covariate_to_level(value: &CovariateValue) -> String {
    match value {
        CovariateValue::Categorical(value) => value.clone(),
        CovariateValue::Numeric(_) => "numeric".to_string(),
        CovariateValue::Missing => "missing".to_string(),
    }
}

fn map_optional_covariate_to_level(value: Option<&CovariateValue>) -> String {
    value.map_or_else(|| "missing".to_string(), map_covariate_to_level)
}

fn covariate_value<'a>(
    record: &'a BalanceRecord,
    key: &str,
    supplemental: &'a SupplementalCovariateMap,
) -> Option<&'a CovariateValue> {
    record.covariates.get(key).or_else(|| {
        supplemental
            .get(record.core.id.as_str())
            .and_then(|values| values.get(key))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let raw_age = NumericFeature::Raw {
            key: "age".to_string(),
        };
        assert_eq!(
            numeric_feature_values(&[case_a], &raw_age, &HashMap::new()),
            Vec::<f64>::new()
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
    fn counts_categorical_includes_missing_bucket() {
        let mut case_a = participant("case_a");
        case_a.covariates.insert(
            "region".to_string(),
            CovariateValue::Categorical("north".to_string()),
        );
        let case_b = participant("case_b");
        let levels = vec!["missing".to_string(), "north".to_string()];

        let (counts, total) =
            counts_categorical(&[case_a, case_b], "region", &levels, &HashMap::new());
        assert_eq!(total, 2);
        assert_eq!(counts, vec![1, 1]);
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
}
