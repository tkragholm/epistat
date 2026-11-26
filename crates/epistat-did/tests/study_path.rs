//! The difference-in-differences calls of Study I, hashed to the bit.
//!
//! Each case makes the calls of PHD's `kernels::did` with the settings it will
//! pass on epistat-did (decision 117): the doubly robust ATT(g,t) on the panel
//! route with a universal base period, comparators matched to each cohort,
//! the unbalanced panel estimated cell by cell, `refuse_poor_overlap` off,
//! `legacy_repeated_cross_section` on, propensity clip 1e-6, ridge 1e-8, 200
//! iterations to a tolerance of 1e-8 and one bootstrap replicate; the same
//! with the pooled never-treated comparators and with two years of
//! anticipation; the `did::aggte` aggregations by event time, cohort, period
//! and overall on the sample's own unit panel with family clusters, the
//! event-time path restricted by `balance_e` as the kernels restrict it, and
//! the fixed-weight aggregations under equal and total weights; the family
//! cluster sums; the two-period doubly robust estimate and the nuisance fits
//! of the pairwise estimator; and the HonestDiD relative-magnitude surface,
//! joint path and smoothness bound on the event-time curve, scaled as the
//! kernels scale it, with Study I's grids and 20,000 draws.
//!
//! The data are a simulated Study I panel: diagnosed children's parents in
//! cohorts 2004 to 2013 and a few diagnosed in the first period, each family
//! matched to two to four comparator families, parents followed from six
//! years before the index year to eight after with some entering late and
//! leaving early, match weights, and the parent's age, sex and a rare
//! indicator as covariates.
//!
//! A change that moves a hash moves Study I, except as decided (decisions 104,
//! 111, 112 and 117 move named parts at the switch), so a port of these calls
//! to new names must reproduce every hash.
//!
//! The hashes were recorded on aarch64 macOS. Print them with
//! `STUDY_PATH_PRINT=1 cargo test --features honest --test study_path -- --nocapture`.
#![cfg(all(target_os = "macos", target_arch = "aarch64", feature = "honest"))]

use std::collections::BTreeSet;

use epistat_did::{
    AggteConfig, AggteResult, AggteType, AttGtAggregatedEstimate, AttGtAggregationConfig,
    AttGtAggregationWeighting, AttGtConfig, AttGtDrConfig, AttGtDrObservation, AttGtEstimate,
    AttGtInfluenceOutput, BasePeriod, ComparisonGroup, DesignPruning, DrDidConfig,
    DrDidObservation, HonestEventStudyInput, HonestJointPathConfig, HonestJointPathMethod,
    HonestRelativeMagnitudeBound, HonestSensitivity, InferenceConfig, LeastFavorableSimulation,
    PanelFlatInput, RelativeMagnitudeConfidenceSetConfig, UnbalancedPanel, UnitPanel,
    aggregate_att_gt, aggregate_att_gt_by_calendar_time, aggregate_att_gt_by_cohort,
    aggregate_att_gt_event_time_with_influence, aggregate_att_gt_overall,
    assess_honest_event_study_joint_path_region_with_config, did_cluster_sums,
    estimate_att_gt_dr_panel_with_influence, estimate_drdid_panel,
    estimate_smoothness_upper_bound_from_pretrends, fit_drdid_panel_nuisance,
    fit_drdid_panel_propensity_coefficients, independent_design_columns,
    summarize_relative_magnitude_sensitivity_many_with_simulation,
    summarize_smoothness_sensitivity_with_simulation,
};
use epistat_inference::Clusters;
use epistat_testkit::{Fingerprint, SplitMix64};

const CONFIDENCE: f64 = 0.95;
const FIRST_YEAR: i32 = 2000;
const LAST_YEAR: i32 = 2016;
/// Draws behind every least-favorable critical value and the joint path's
/// calibration, Study I's `analysis_honest_least_favorable_draws`.
const DRAWS: usize = 20_000;
const JOINT_SEED: u64 = 20_260_819;
const RM_GRID: [f64; 9] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5];
const JOINT_GRID: [f64; 3] = [0.0, 0.25, 0.5];

/// A fingerprint that hashes NaN and `None` alike, so a value that becomes
/// `Option` without changing leaves the hash alone.
struct Hash(Fingerprint);

impl Hash {
    fn new() -> Self {
        Self(Fingerprint::new())
    }

    fn word(&mut self, value: u64) {
        self.0 = std::mem::take(&mut self.0).u64(value);
    }

    fn int(&mut self, value: i64) {
        self.word(value.cast_unsigned());
    }

    fn text(&mut self, text: &str) {
        self.word(text.len() as u64);
        for byte in text.bytes() {
            self.word(u64::from(byte));
        }
    }

    fn o(&mut self, value: Option<f64>) {
        let value = value.filter(|v| !v.is_nan());
        self.0 = std::mem::take(&mut self.0).option_f64(value);
    }

    fn f(&mut self, value: f64) {
        self.o(Some(value));
    }

    fn fs(&mut self, values: &[f64]) {
        self.word(values.len() as u64);
        for &v in values {
            self.f(v);
        }
    }

    fn estimate(&mut self, e: &AttGtEstimate) {
        self.int(i64::from(e.group));
        self.int(i64::from(e.time));
        self.int(i64::from(e.event_time));
        self.f(e.att);
        self.f(e.se);
        self.f(e.ci_low);
        self.f(e.ci_high);
        self.word(e.treated_n as u64);
        self.word(e.control_n as u64);
        self.f(e.total_weight);
    }

    fn summary(&mut self, s: &AttGtAggregatedEstimate) {
        self.f(s.estimate);
        self.f(s.se);
        self.f(s.ci_low);
        self.f(s.ci_high);
        self.word(s.components as u64);
        self.f(s.total_weight);
    }

    fn aggte(&mut self, result: &AggteResult) {
        self.f(result.overall_att);
        self.f(result.overall_se);
        self.f(result.overall_ci_low);
        self.f(result.overall_ci_high);
        self.fs(&result.overall_influence);
        for row in &result.by_key {
            self.int(i64::from(row.key));
            self.f(row.att);
            self.f(row.se);
            self.f(row.ci_low);
            self.f(row.ci_high);
            self.fs(&row.influence);
        }
        self.f(result.critical_value);
        self.word(u64::from(result.uniform_band));
    }

    fn finish(&self) -> u64 {
        self.0.finish()
    }
}

fn normal(stream: &mut SplitMix64) -> f64 {
    let u1 = stream.unit().max(f64::MIN_POSITIVE);
    let u2 = stream.unit();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// One person-year of the analysis table, in the kernels' columns.
struct Row {
    time: i32,
    /// The cohort, 0 for never treated, as `kernels::did::NEVER_TREATED`.
    group: i32,
    outcome: f64,
    weight: f64,
    unit: u32,
    cluster: u32,
    comparison_cohort: i32,
    covariates: [f64; 3],
}

fn panel() -> Vec<Row> {
    let mut stream = SplitMix64::new(20_261_126);
    let mut rows = Vec::new();
    let mut next_unit = 0_u32;
    let mut next_family = 0_u32;
    let mut family =
        |stream: &mut SplitMix64, rows: &mut Vec<Row>, cohort: i32, treated: bool, weight: f64| {
            let cluster = next_family * 7 + 3;
            next_family += 1;
            let parents = if stream.unit() < 0.8 { 2 } else { 1 };
            let rare = f64::from(stream.unit() < 0.03);
            for parent in 0..parents {
                // Unit codes that do not sort in family order.
                let unit = next_unit.wrapping_mul(40_503).wrapping_add(17) % (1 << 20);
                next_unit += 1;
                let age = 24.0 + 14.0 * stream.unit();
                let mother = f64::from(parent == 0);
                let level = 20.0 * normal(stream);
                let from = (cohort - 6 + i32::from(stream.unit() < 0.15)).max(FIRST_YEAR);
                let to = (cohort + 8 - i32::from(stream.unit() < 0.2) * 3).min(LAST_YEAR);
                for time in from..=to {
                    let event = time - cohort;
                    let effect = if treated && event >= 0 {
                        -15.0 - 2.0 * f64::from(event)
                    } else {
                        0.0
                    };
                    let outcome = 250.0 + 3.0 * age - 12.0 * mother
                        + level
                        + 4.0 * f64::from(time - FIRST_YEAR)
                        + effect
                        + 15.0 * normal(stream);
                    rows.push(Row {
                        time,
                        group: if treated { cohort } else { 0 },
                        outcome,
                        weight,
                        unit,
                        cluster,
                        comparison_cohort: if treated { 0 } else { cohort },
                        covariates: [age, mother, rare],
                    });
                }
            }
        };
    for case in 0..260 {
        let cohort = if case < 6 {
            FIRST_YEAR
        } else {
            2004 + i32::try_from(stream.below(10)).expect("small")
        };
        family(&mut stream, &mut rows, cohort, true, 1.0);
        let comparators = 2 + stream.below(3);
        for _ in 0..comparators {
            family(
                &mut stream,
                &mut rows,
                cohort,
                false,
                1.0 / comparators as f64,
            );
        }
    }
    rows
}

/// `kernels::did::observations`.
fn observations(rows: &[Row]) -> Vec<AttGtDrObservation> {
    rows.iter()
        .map(|row| AttGtDrObservation {
            unit_id: Some(i64::from(row.unit)),
            first_treated_time: (row.group != 0).then_some(row.group),
            time: row.time,
            outcome: row.outcome,
            weight: row.weight,
            covariates: row.covariates.to_vec(),
            comparison_cohort: (row.comparison_cohort != 0).then_some(row.comparison_cohort),
        })
        .collect()
}

fn att_gt_config(comparison_group: ComparisonGroup, anticipation_periods: i32) -> AttGtDrConfig {
    let inference = InferenceConfig {
        confidence_level: CONFIDENCE,
    };
    AttGtDrConfig {
        att_gt: AttGtConfig {
            confidence_level: inference,
            comparison_group,
            base_period: BasePeriod::Universal,
            anticipation_periods,
            skip_incomplete_pairs: true,
            unbalanced_panel: UnbalancedPanel::PerCell,
            refuse_small_never_treated: true,
            refuse_poor_overlap: false,
        },
        drdid: drdid_config(),
    }
}

fn drdid_config() -> DrDidConfig {
    DrDidConfig {
        confidence_level: CONFIDENCE,
        propensity_clip: 1e-6,
        ridge: 1e-8,
        max_iter: 200,
        tol: 1e-8,
        bootstrap_reps: 1,
        legacy_repeated_cross_section: true,
        ..DrDidConfig::default()
    }
}

fn att_gt_digest(fitted: &AttGtInfluenceOutput) -> u64 {
    let mut h = Hash::new();
    for (estimate, influence) in fitted.estimates.iter().zip(&fitted.influence_functions) {
        h.estimate(estimate);
        h.fs(influence);
    }
    for cell in &fitted.skipped {
        h.int(i64::from(cell.group));
        h.int(i64::from(cell.time));
        h.int(i64::from(cell.baseline_time));
        h.text(cell.reason);
    }
    for cell in &fitted.pruned {
        h.int(i64::from(cell.group));
        h.int(i64::from(cell.time));
        h.int(i64::from(cell.baseline_time));
        h.word(cell.columns_dropped as u64);
    }
    let sample = &fitted.sample;
    for &id in &sample.ids {
        h.int(id);
    }
    sample
        .groups
        .iter()
        .for_each(|g| h.int(g.map_or(-1, i64::from)));
    h.fs(&sample.weights);
    for &p in &sample.periods {
        h.int(i64::from(p));
    }
    for &c in &sample.cohorts {
        h.int(i64::from(c));
    }
    for count in [
        sample.recoded_never_treated,
        sample.dropped_late_rows,
        sample.dropped_treated_at_start,
        sample.dropped_unbalanced,
    ] {
        h.word(count as u64);
    }
    h.int(sample.last_cohort_as_comparison.map_or(-1, i64::from));
    h.word(u64::from(sample.unbalanced_as_repeated_cross_section));
    for cell in &fitted.overlap {
        h.int(i64::from(cell.group));
        h.int(i64::from(cell.time));
        h.int(i64::from(cell.baseline_time));
        h.o(cell.max_propensity);
    }
    h.finish()
}

/// The family of each influence position, as the kernels align clusters to the
/// units the estimator kept.
fn unit_clusters(rows: &[Row], ids: &[i64]) -> Vec<u32> {
    ids.iter()
        .map(|&id| {
            rows.iter()
                .find(|row| i64::from(row.unit) == id)
                .expect("a kept unit has rows")
                .cluster
        })
        .collect()
}

/// `kernels::did::event_study`'s `restrict_dynamic_cells` for `balance_e`.
fn balanced(
    estimates: &[AttGtEstimate],
    influence: &[Vec<f64>],
    balance_e: i32,
) -> (Vec<AttGtEstimate>, Vec<Vec<f64>>) {
    let max_time = estimates.iter().map(|e| e.time).max().expect("cells");
    let kept: BTreeSet<i32> = estimates
        .iter()
        .map(|e| e.group)
        .filter(|g| max_time - g >= balance_e)
        .collect();
    estimates
        .iter()
        .zip(influence)
        .filter(|(e, _)| kept.contains(&e.group) && e.event_time <= balance_e)
        .map(|(e, psi)| (*e, psi.clone()))
        .unzip()
}

/// `kernels::did::honest`'s scale: the root of the largest variance.
fn honest_scale(covariance: &[Vec<f64>]) -> f64 {
    let largest = covariance
        .iter()
        .enumerate()
        .map(|(index, row)| row.get(index).copied().unwrap_or(0.0))
        .fold(0.0_f64, f64::max);
    if largest.is_finite() && largest > 0.0 {
        largest.sqrt()
    } else {
        1.0
    }
}

fn scaled(input: &HonestEventStudyInput, scale: f64) -> HonestEventStudyInput {
    HonestEventStudyInput {
        betahat: input.betahat.iter().map(|value| value / scale).collect(),
        covariance: input
            .covariance
            .iter()
            .map(|row| row.iter().map(|value| value / (scale * scale)).collect())
            .collect(),
        pre_periods: input.pre_periods.clone(),
        post_periods: input.post_periods.clone(),
    }
}

/// The event-time curve with its family-clustered covariance, as the kernels
/// form it from the aggregated influence functions, without the reference
/// period -1 and within event times -5 to 8.
fn event_study(dynamic: &AggteResult, clusters: &Clusters) -> HonestEventStudyInput {
    let rows: Vec<_> = dynamic
        .by_key
        .iter()
        .filter(|row| row.key != -1 && (-5..=8).contains(&row.key))
        .collect();
    let sums: Vec<Vec<f64>> = rows
        .iter()
        .map(|row| did_cluster_sums(&row.influence, clusters).expect("sums"))
        .collect();
    let covariance = sums
        .iter()
        .map(|left| {
            sums.iter()
                .map(|right| left.iter().zip(right).map(|(a, b)| a * b).sum())
                .collect()
        })
        .collect();
    HonestEventStudyInput {
        betahat: rows.iter().map(|row| row.att).collect(),
        covariance,
        pre_periods: rows.iter().map(|row| row.key).filter(|&k| k < 0).collect(),
        post_periods: rows.iter().map(|row| row.key).filter(|&k| k >= 0).collect(),
    }
}

fn aggregations(
    out: &mut Vec<(&'static str, u64)>,
    rows: &[Row],
    fitted: &AttGtInfluenceOutput,
) -> (AggteResult, Clusters) {
    let inference = InferenceConfig {
        confidence_level: CONFIDENCE,
    };
    let families = unit_clusters(rows, &fitted.sample.ids);
    let clusters = Clusters::from_labels(&families);
    let panel: UnitPanel = fitted
        .sample
        .unit_panel()
        .clustered_by(families.iter().map(|&f| i64::from(f)).collect());
    let (estimates, influence) = (&fitted.estimates, &fitted.influence_functions);

    let run = |aggregation, estimates: &[AttGtEstimate], influence: &[Vec<f64>]| {
        aggregate_att_gt(
            estimates,
            influence,
            &panel,
            AggteConfig {
                aggregation,
                confidence_level: inference,
                ..AggteConfig::default()
            },
        )
        .expect("aggregation")
    };
    let dynamic = run(AggteType::Dynamic, estimates, influence);
    let mut h = Hash::new();
    h.aggte(&dynamic);
    for kind in [AggteType::Group, AggteType::Calendar, AggteType::Simple] {
        h.aggte(&run(kind, estimates, influence));
    }
    out.push(("aggte/by_treated_count", h.finish()));

    let (kept, kept_influence) = balanced(estimates, influence, 5);
    let mut h = Hash::new();
    h.aggte(&run(AggteType::Dynamic, &kept, &kept_influence));
    out.push(("aggte/balance_e_5", h.finish()));

    for (name, weighting) in [
        (
            "fixed_weights/by_total_weight",
            AttGtAggregationWeighting::ByTotalWeight,
        ),
        ("fixed_weights/equal", AttGtAggregationWeighting::Equal),
    ] {
        let config = AttGtAggregationConfig {
            confidence_level: inference,
            weighting,
        };
        let mut h = Hash::new();
        let event_time = aggregate_att_gt_event_time_with_influence(estimates, influence, config)
            .expect("event time");
        for (row, psi) in event_time
            .estimates
            .iter()
            .zip(&event_time.influence_functions)
        {
            h.int(i64::from(row.event_time));
            h.summary(&row.summary);
            h.fs(psi);
        }
        for row in aggregate_att_gt_by_cohort(estimates, influence, &panel, config).expect("cohort")
        {
            h.int(i64::from(row.group));
            h.summary(&row.summary);
        }
        for row in aggregate_att_gt_by_calendar_time(estimates, influence, &panel, config)
            .expect("calendar")
        {
            h.int(i64::from(row.time));
            h.summary(&row.summary);
        }
        h.summary(
            &aggregate_att_gt_overall(estimates, influence, &panel, config).expect("overall"),
        );
        out.push((name, h.finish()));
    }

    let mut h = Hash::new();
    for psi in influence {
        h.fs(&did_cluster_sums(psi, &clusters).expect("sums"));
    }
    for row in &dynamic.by_key {
        h.fs(&did_cluster_sums(&row.influence, &clusters).expect("sums"));
    }
    out.push(("cluster_sums", h.finish()));
    (dynamic, clusters)
}

/// `kernels::did::honest`'s three entry points on the event-time curve.
fn honest(out: &mut Vec<(&'static str, u64)>, study: &HonestEventStudyInput) {
    let inference = InferenceConfig {
        confidence_level: CONFIDENCE,
    };
    let simulation = LeastFavorableSimulation::new(DRAWS, 0);
    let scale = honest_scale(&study.covariance);
    let input = scaled(study, scale);
    let n_post = input.post_periods.len();
    let mut functionals: Vec<Vec<f64>> = (0..n_post)
        .map(|k| (0..n_post).map(|j| f64::from(j == k)).collect())
        .collect();
    functionals.push(vec![1.0 / n_post as f64; n_post]);
    functionals.push(
        (0..n_post)
            .map(|j| if j < 3 { 1.0 / 3.0 } else { 0.0 })
            .collect(),
    );

    let weights: Vec<&[f64]> = functionals.iter().map(Vec::as_slice).collect();
    let summaries = summarize_relative_magnitude_sensitivity_many_with_simulation(
        &input,
        inference,
        &weights,
        None,
        Some(&RM_GRID),
        Some(HonestRelativeMagnitudeBound::ParallelTrendsDeviation),
        None,
        None,
        simulation,
    )
    .expect("relative magnitude");
    let offset = input.pre_periods.len();
    let mut h = Hash::new();
    for (post_weights, summary) in functionals.iter().zip(&summaries) {
        let estimate: f64 = post_weights
            .iter()
            .enumerate()
            .map(|(index, weight)| weight * input.betahat[offset + index])
            .sum();
        let mut variance = 0.0;
        for (i, wi) in post_weights.iter().enumerate() {
            for (j, wj) in post_weights.iter().enumerate() {
                variance += wi * wj * input.covariance[offset + i][offset + j];
            }
        }
        h.f(estimate * scale);
        h.f(variance.max(0.0).sqrt() * scale);
        for row in &summary.rows {
            h.f(row.sensitivity_value);
            h.f(row.lb * scale);
            h.f(row.ub * scale);
            h.o(row.identified.as_ref().map(|set| set.lb * scale));
            h.o(row.identified.as_ref().map(|set| set.ub * scale));
            h.text(row.delta);
        }
    }
    out.push(("honest/relative_magnitude", h.finish()));

    let mut h = Hash::new();
    for m in JOINT_GRID {
        let region = assess_honest_event_study_joint_path_region_with_config(
            &input,
            HonestSensitivity::RelativeMagnitude(m),
            inference,
            0.0 / scale,
            RelativeMagnitudeConfidenceSetConfig {
                least_favorable_draws: DRAWS,
                least_favorable_seed: 0,
                ..RelativeMagnitudeConfidenceSetConfig::from_inference(inference)
            },
            HonestJointPathConfig {
                method: HonestJointPathMethod::GaussianSimulated,
                simulation_draws: DRAWS,
                simulation_seed: JOINT_SEED,
            },
        )
        .expect("joint path");
        h.f(region.pointwise_confidence_level);
        h.f(region.calibrated_max_t_critical_value);
        for point in &region.points {
            let a = &point.assessment;
            h.int(i64::from(point.post_period));
            for value in [
                a.estimate,
                a.se,
                a.original_ci.0,
                a.original_ci.1,
                a.robust_ci.0,
                a.robust_ci.1,
                a.identified_set.0,
                a.identified_set.1,
            ] {
                h.f(value * scale);
            }
        }
    }
    out.push(("honest/joint_path", h.finish()));

    let m = estimate_smoothness_upper_bound_from_pretrends(study, 0.05).expect("smoothness m");
    let mut h = Hash::new();
    h.f(m);
    let grid = [0.0, 0.5 * m / scale, m / scale];
    let summary = summarize_smoothness_sensitivity_with_simulation(
        &input,
        inference,
        &functionals[n_post],
        None,
        Some(&grid),
        None,
        None,
        simulation,
    )
    .expect("smoothness");
    for row in &summary.rows {
        h.f(row.sensitivity_value * scale);
        h.f(row.lb * scale);
        h.f(row.ub * scale);
        h.text(row.delta);
    }
    out.push(("honest/smoothness", h.finish()));
}

/// The pairwise estimator's two-period panels: parents observed at the
/// baseline event time -1 and at event time `e`, cases against the
/// comparators matched to them.
fn pairwise(out: &mut Vec<(&'static str, u64)>, rows: &[Row]) {
    let mut h_att = Hash::new();
    let mut h_nuisance = Hash::new();
    for event in [0, 3, 7] {
        let mut frame: Vec<(bool, f64, f64, [f64; 3], u32)> = Vec::new();
        let units: BTreeSet<u32> = rows.iter().map(|row| row.unit).collect();
        for unit in units {
            let own: Vec<&Row> = rows.iter().filter(|row| row.unit == unit).collect();
            let cohort = own[0].group.max(own[0].comparison_cohort);
            if cohort == FIRST_YEAR {
                continue;
            }
            let at = |time: i32| own.iter().find(|row| row.time == time);
            if let (Some(base), Some(later)) = (at(cohort - 1), at(cohort + event)) {
                frame.push((
                    base.group != 0,
                    later.outcome - base.outcome,
                    base.weight,
                    base.covariates,
                    base.cluster,
                ));
            }
        }
        let observations: Vec<DrDidObservation> = frame
            .iter()
            .map(
                |&(treated, delta_outcome, weight, covariates, _)| DrDidObservation {
                    treated,
                    delta_outcome,
                    weight,
                    covariates: covariates.to_vec(),
                },
            )
            .collect();
        let fitted = estimate_drdid_panel(
            &observations,
            DrDidConfig {
                bootstrap_seed: 20_260_819,
                ..drdid_config()
            },
        )
        .expect("two-period estimate");
        h_att.f(fitted.att);
        h_att.f(fitted.se);
        h_att.f(fitted.ci_low);
        h_att.f(fitted.ci_high);
        h_att.word(fitted.treated_n as u64);
        h_att.word(fitted.control_n as u64);
        h_att.f(fitted.total_weight);
        h_att.fs(&fitted.influence_function);
        h_att.word(fitted.design_columns_dropped as u64);
        let families: Vec<u32> = frame.iter().map(|row| row.4).collect();
        h_att.fs(&did_cluster_sums(
            &fitted.influence_function,
            &Clusters::from_labels(&families),
        )
        .expect("sums"));

        // `kernels::did::nuisance`: the design with an intercept, row-major.
        let treated: Vec<bool> = frame.iter().map(|row| row.0).collect();
        let delta: Vec<f64> = frame.iter().map(|row| row.1).collect();
        let weight: Vec<f64> = frame.iter().map(|row| row.2).collect();
        let design: Vec<f64> = frame
            .iter()
            .flat_map(|row| [1.0, row.3[0], row.3[1], row.3[2]])
            .collect();
        let nuisance_config = DrDidConfig {
            propensity_clip: 1e-6,
            ridge: 1e-8,
            max_iter: 200,
            tol: 1e-8,
            ..DrDidConfig::default()
        };
        let input = || PanelFlatInput {
            treated: &treated,
            delta_outcome: &delta,
            weight: &weight,
            design_matrix_flat: &design,
            feature_count: 4,
        };
        let fit = fit_drdid_panel_nuisance(input(), nuisance_config).expect("nuisance");
        fit.kept_columns
            .iter()
            .for_each(|&kept| h_nuisance.word(u64::from(kept)));
        h_nuisance.fs(&fit.propensity_coefficients);
        h_nuisance.fs(&fit.propensity_se);
        h_nuisance.fs(&fit.outcome_coefficients);
        h_nuisance.fs(&fit.outcome_se);
        h_nuisance.fs(&fit.propensity_scores);
        h_nuisance.fs(&fit.outcome_predictions);
        h_nuisance.word(fit.iterations as u64);
        h_nuisance.word(u64::from(fit.converged));
        let zeros = vec![0.0; treated.len()];
        let propensity = fit_drdid_panel_propensity_coefficients(
            PanelFlatInput {
                delta_outcome: &zeros,
                ..input()
            },
            nuisance_config,
        )
        .expect("propensity");
        propensity
            .kept_columns
            .iter()
            .for_each(|&kept| h_nuisance.word(u64::from(kept)));
        h_nuisance.fs(&propensity.coefficients);
        h_nuisance.word(propensity.iterations as u64);
        h_nuisance.word(u64::from(propensity.converged));
        let kept = independent_design_columns(&design, 4, &weight, DesignPruning::Pooled, 1e-8)
            .expect("columns");
        for kept in kept {
            h_nuisance.word(u64::from(kept));
        }
    }
    out.push(("pairwise/drdid_panel", h_att.finish()));
    out.push(("pairwise/nuisance", h_nuisance.finish()));
}

fn cases() -> Vec<(&'static str, u64)> {
    let rows = panel();
    let observations = observations(&rows);
    let mut out = Vec::new();

    let fitted = estimate_att_gt_dr_panel_with_influence(
        &observations,
        att_gt_config(ComparisonGroup::MatchedNeverTreated, 0),
    )
    .expect("ATT(g,t)");
    assert!(
        fitted.sample.dropped_treated_at_start > 0,
        "no unit dropped at the start"
    );
    out.push(("att_gt/matched_never_treated", att_gt_digest(&fitted)));
    let pooled = estimate_att_gt_dr_panel_with_influence(
        &observations,
        att_gt_config(ComparisonGroup::NeverTreated, 0),
    )
    .expect("ATT(g,t)");
    out.push(("att_gt/never_treated", att_gt_digest(&pooled)));
    let anticipation = estimate_att_gt_dr_panel_with_influence(
        &observations,
        att_gt_config(ComparisonGroup::MatchedNeverTreated, 2),
    )
    .expect("ATT(g,t)");
    out.push(("att_gt/anticipation_2", att_gt_digest(&anticipation)));

    let (dynamic, clusters) = aggregations(&mut out, &rows, &fitted);
    honest(&mut out, &event_study(&dynamic, &clusters));
    pairwise(&mut out, &rows);
    out
}

const GOLDEN: &[(&str, u64)] = &[
    ("att_gt/matched_never_treated", 0x7ea8_52f0_a5d6_09ec),
    ("att_gt/never_treated", 0x5236_56a5_9ece_2383),
    ("att_gt/anticipation_2", 0xaee5_8409_3b02_978a),
    ("aggte/by_treated_count", 0x4eab_b730_f26e_8fdc),
    ("aggte/balance_e_5", 0x8783_2bcc_bdbf_953b),
    ("fixed_weights/by_total_weight", 0xe22c_d8c7_552b_20a2),
    ("fixed_weights/equal", 0x6255_94e8_f6fc_0a38),
    ("cluster_sums", 0x7586_203b_1610_15b0),
    ("honest/relative_magnitude", 0x9e84_13c7_b9f7_560b),
    ("honest/joint_path", 0x40d8_ce11_2e5b_21b2),
    ("honest/smoothness", 0x6b35_a17a_1a79_e667),
    ("pairwise/drdid_panel", 0xa3a8_c340_00d1_3388),
    ("pairwise/nuisance", 0x11d5_dcb5_3803_4f58),
];

#[test]
fn study_paths_are_bit_identical_to_the_recorded_hashes() {
    let current = cases();
    if std::env::var_os("STUDY_PATH_PRINT").is_some() {
        for (name, hash) in &current {
            println!(
                "    (\"{name}\", 0x{:04x}_{:04x}_{:04x}_{:04x}),",
                hash >> 48,
                (hash >> 32) & 0xffff,
                (hash >> 16) & 0xffff,
                hash & 0xffff
            );
        }
    }
    assert_eq!(current.as_slice(), GOLDEN);
}
