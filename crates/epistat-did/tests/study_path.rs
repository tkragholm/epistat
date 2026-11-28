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
//! of the pairwise estimator. The HonestDiD calls on the event-time curve are
//! in epistat-honestdid's study path.
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
//! The hashes were recorded on aarch64 macOS and are asserted only there.
//! Elsewhere the test computes them and, when `EPISTAT_HASH_OUT` is set,
//! appends them to that file. Print them with
//! `STUDY_PATH_PRINT=1 cargo test --test study_path -- --nocapture`.

use std::collections::BTreeSet;

use epistat_core::{Bootstrap, Level, Seed};
use epistat_did::att_gt::{
    self, Aggte, AggteOptions, AggteType, AggteWeights, AttGt, AttGtOptions, BasePeriod,
    ComparisonGroup, FitFailure, NuisanceOptions, SkipReason, UnbalancedPanel,
};
use epistat_did::two_period::nuisance::{self, Pruning};
use epistat_did::two_period::{self, Estimator, TwoPeriodOptions};
use epistat_did::{StaggeredPanel, TwoPeriodData, did_cluster_sums};
use epistat_inference::Clusters;
use epistat_linalg::MatRef;
use epistat_testkit::{Fingerprint, SplitMix64};

const CONFIDENCE: f64 = 0.95;
const FIRST_YEAR: i32 = 2000;
const LAST_YEAR: i32 = 2016;

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

    /// A cell as `AttGtEstimate` hashed it: cohort, period, event time,
    /// estimate, standard error, interval, counts and total weight.
    fn cell(&mut self, cell: &att_gt::AttGtCell) {
        let estimate = cell.estimate.expect("an estimated cell");
        self.int(cell.group);
        self.int(cell.time);
        self.int(cell.event_time);
        self.f(estimate.estimate);
        self.o(estimate.se);
        self.o(estimate.lower());
        self.o(estimate.upper());
        self.word(cell.n_treated as u64);
        self.word(cell.n_control as u64);
        self.f(cell.total_weight);
    }

    /// A point as `AttGtAggregatedEstimate` hashed it.
    fn summary(&mut self, estimate: &epistat_core::Estimate, n_cells: usize, total_weight: f64) {
        self.f(estimate.estimate);
        self.o(estimate.se);
        self.o(estimate.lower());
        self.o(estimate.upper());
        self.word(n_cells as u64);
        self.f(total_weight);
    }

    /// A summary as `AggteResult` hashed it: the overall effect and its
    /// influence function, then each point with its influence function, the
    /// critical value and whether the band is uniform.
    fn aggte(&mut self, result: &Aggte) {
        let overall = result.overall.expect("an overall effect");
        let influence = result.influence();
        self.f(overall.estimate.estimate);
        self.o(overall.estimate.se);
        self.o(overall.estimate.lower());
        self.o(overall.estimate.upper());
        self.fs(influence.column(result.points.len()));
        for (j, point) in result.points.iter().enumerate() {
            self.int(point.key);
            self.f(point.estimate.estimate);
            self.o(point.estimate.se);
            self.o(point.estimate.lower());
            self.o(point.estimate.upper());
            self.fs(influence.column(j));
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

/// The analysis table as panel columns, sorted by unit and then period, as
/// `kernels::did` hands it to the estimator.
struct Columns {
    unit: Vec<u32>,
    period: Vec<i64>,
    cohort: Vec<Option<i64>>,
    outcome: Vec<f64>,
    weight: Vec<f64>,
    /// Column-major, three columns.
    covariates: Vec<f64>,
    matched: Vec<Option<i64>>,
    cluster: Vec<u32>,
}

fn columns(rows: &[Row]) -> Columns {
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by_key(|&i| (rows[i].unit, rows[i].time));
    let pick = |f: &dyn Fn(&Row) -> f64| order.iter().map(|&i| f(&rows[i])).collect::<Vec<_>>();
    let covariates = (0..3)
        .flat_map(|j| order.iter().map(move |&i| rows[i].covariates[j]))
        .collect();
    Columns {
        unit: order.iter().map(|&i| rows[i].unit).collect(),
        period: order.iter().map(|&i| i64::from(rows[i].time)).collect(),
        cohort: order
            .iter()
            .map(|&i| (rows[i].group != 0).then_some(i64::from(rows[i].group)))
            .collect(),
        outcome: pick(&|row| row.outcome),
        weight: pick(&|row| row.weight),
        covariates,
        matched: order
            .iter()
            .map(|&i| {
                (rows[i].comparison_cohort != 0).then_some(i64::from(rows[i].comparison_cohort))
            })
            .collect(),
        cluster: order.iter().map(|&i| rows[i].cluster).collect(),
    }
}

/// Study I's ATT(g,t): the matched or pooled never-treated comparators, a
/// universal base, the per-cell panel, no overlap refusal, the legacy
/// repeated cross-section computation, and the one-replicate cell bootstrap
/// the kernels set, seeded as the estimator's former default.
fn study_att_gt(
    data: &Columns,
    comparison_group: ComparisonGroup,
    anticipation: i64,
    clustered: bool,
) -> AttGt {
    let covariates = MatRef::from_column_major_slice(&data.covariates, data.unit.len(), 3);
    let builder = StaggeredPanel::builder(&data.unit, &data.period, &data.cohort, &data.outcome)
        .weights(&data.weight)
        .covariates(covariates)
        .comparison_cohort(&data.matched);
    let builder = if clustered {
        builder.clusters(&data.cluster)
    } else {
        builder
    };
    let panel = builder.build().expect("panel");
    let options = AttGtOptions {
        comparison_group,
        base_period: BasePeriod::Universal,
        anticipation,
        unbalanced: UnbalancedPanel::PerCell,
        refuse_poor_overlap: false,
        nuisance: NuisanceOptions {
            legacy_repeated_cross_section: true,
            ..NuisanceOptions::default()
        },
        level: Level::confidence(CONFIDENCE).expect("level"),
        bootstrap: (!clustered).then_some(Bootstrap::new(1, Seed(17_431))),
        ..AttGtOptions::default()
    };
    att_gt::estimate(&panel, &options).expect("ATT(g,t)")
}

/// The label the panel route gave each reason before the reasons were an
/// enum.
fn panel_reason(reason: SkipReason) -> &'static str {
    match reason {
        SkipReason::NoTreated => "treated_panel",
        SkipReason::NoComparison => "control_panel",
        SkipReason::PoorOverlap => "overlap",
        SkipReason::OverlapFitFailed => "overlap_fit",
        SkipReason::NotFinite => "non_finite",
        SkipReason::FitFailed(failure) => match failure {
            FitFailure::Empty => "panel_fit:empty",
            FitFailure::NoTreated => "panel_fit:no_treated",
            FitFailure::NoComparison => "panel_fit:no_control",
            FitFailure::InvalidWeight => "panel_fit:invalid_weight",
            FitFailure::InvalidOutcome => "panel_fit:invalid_outcome",
            FitFailure::InvalidCovariate => "panel_fit:invalid_covariate",
            FitFailure::Singular => "panel_fit:singular",
            _ => "panel_fit:other",
        },
        other => panic!("not a reason of the panel route: {other:?}"),
    }
}

fn att_gt_digest(fitted: &AttGt) -> u64 {
    let mut h = Hash::new();
    let influence = fitted.influence().expect("influence");
    for (j, cell) in fitted
        .cells()
        .iter()
        .filter(|cell| cell.estimate.is_some())
        .enumerate()
    {
        h.cell(cell);
        h.fs(influence.column(j));
    }
    for cell in fitted.cells() {
        if let Some(reason) = cell.skipped {
            h.int(cell.group);
            h.int(cell.time);
            h.int(cell.base);
            h.text(panel_reason(reason));
        }
    }
    for cell in fitted.cells() {
        if cell.estimate.is_some() && cell.columns_dropped > 0 {
            h.int(cell.group);
            h.int(cell.time);
            h.int(cell.base);
            h.word(cell.columns_dropped as u64);
        }
    }
    let sample = fitted.sample();
    for &id in &sample.units {
        h.int(i64::from(id));
    }
    sample.groups.iter().for_each(|g| h.int(g.unwrap_or(-1)));
    h.fs(&sample.weights);
    for &p in &sample.periods {
        h.int(p);
    }
    for &c in &sample.cohorts {
        h.int(c);
    }
    for count in [
        sample.recoded_never_treated,
        sample.dropped_late_rows,
        sample.dropped_treated_at_start,
        sample.dropped_unbalanced,
    ] {
        h.word(count as u64);
    }
    h.int(sample.last_cohort_as_comparison.unwrap_or(-1));
    h.word(u64::from(sample.unbalanced_as_repeated_cross_section));
    for cell in fitted.cells() {
        if let Some(overlap) = cell.overlap {
            h.int(cell.group);
            h.int(cell.time);
            h.int(cell.base);
            h.o(overlap.max_propensity);
        }
    }
    h.finish()
}

/// The aggregations of the kernels: `did::aggte`'s four summaries on the
/// family-clustered fit, the event study restricted by `balance_e`, and the
/// fixed-weight summaries, whose event-time path is unclustered.
fn aggregations(out: &mut Vec<(&'static str, u64)>, fitted: &AttGt, clustered: &AttGt) {
    let level = Level::confidence(CONFIDENCE).expect("level");
    let families = clustered.clusters().expect("clusters").to_vec();
    let clusters = Clusters::from_labels_first_seen(&families).unwrap();
    let run = |fit: &AttGt, options: AggteOptions| fit.aggregate(&options).expect("aggregation");
    let by_share = |aggregation| AggteOptions {
        aggregation,
        level,
        ..AggteOptions::default()
    };

    let dynamic = run(clustered, by_share(AggteType::Dynamic));
    let mut h = Hash::new();
    h.aggte(&dynamic);
    for kind in [AggteType::Group, AggteType::Calendar, AggteType::Simple] {
        h.aggte(&run(clustered, by_share(kind)));
    }
    out.push(("aggte/by_treated_count", h.finish()));

    let mut h = Hash::new();
    h.aggte(&run(
        clustered,
        AggteOptions {
            balance_e: Some(5),
            ..by_share(AggteType::Dynamic)
        },
    ));
    out.push(("aggte/balance_e_5", h.finish()));

    for (name, weights) in [
        ("fixed_weights/by_total_weight", AggteWeights::TotalWeight),
        ("fixed_weights/equal", AggteWeights::Equal),
    ] {
        let fixed = |aggregation| AggteOptions {
            weights,
            ..by_share(aggregation)
        };
        let mut h = Hash::new();
        let event_time = run(fitted, fixed(AggteType::Dynamic));
        for (j, point) in event_time.points.iter().enumerate() {
            h.int(point.key);
            h.summary(&point.estimate, point.n_cells, point.total_weight);
            h.fs(event_time.influence().column(j));
        }
        for kind in [AggteType::Group, AggteType::Calendar] {
            for point in &run(clustered, fixed(kind)).points {
                h.int(point.key);
                h.summary(&point.estimate, point.n_cells, point.total_weight);
            }
        }
        let overall = run(clustered, fixed(AggteType::Simple))
            .overall
            .expect("overall");
        h.summary(&overall.estimate, overall.n_cells, overall.total_weight);
        out.push((name, h.finish()));
    }

    let mut h = Hash::new();
    let influence = fitted.influence().expect("influence");
    for j in 0..influence.n_columns() {
        h.fs(&did_cluster_sums(influence.column(j), &clusters).expect("sums"));
    }
    for j in 0..dynamic.points.len() {
        h.fs(&did_cluster_sums(dynamic.influence().column(j), &clusters).expect("sums"));
    }
    out.push(("cluster_sums", h.finish()));
}

/// Study I's nuisance settings: propensity clip 1e-6, ridge 1e-8, 200
/// iterations to 1e-8, and the legacy repeated cross-section computation,
/// which the panel ignores.
fn nuisance_options() -> NuisanceOptions {
    NuisanceOptions {
        propensity_clip: 1e-6,
        ridge: 1e-8,
        max_iterations: 200,
        tolerance: 1e-8,
        legacy_repeated_cross_section: true,
    }
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
        let treated: Vec<bool> = frame.iter().map(|row| row.0).collect();
        let delta: Vec<f64> = frame.iter().map(|row| row.1).collect();
        let weight: Vec<f64> = frame.iter().map(|row| row.2).collect();
        let n = frame.len();
        // Column-major, three columns.
        let covariates: Vec<f64> = (0..3)
            .flat_map(|j| frame.iter().map(move |row| row.3[j]))
            .collect();
        let covariates = MatRef::from_column_major_slice(&covariates, n, 3);

        // The kernels' two-period call passes the covariates, to which the
        // estimator adds the intercept.
        let data = TwoPeriodData::panel(&treated, &delta)
            .weights(&weight)
            .covariates(covariates)
            .build()
            .expect("two-period data");
        let fitted = two_period::estimate(
            &data,
            &TwoPeriodOptions {
                estimator: Estimator::DoublyRobust,
                nuisance: nuisance_options(),
                level: Level::confidence(CONFIDENCE).expect("level"),
                bootstrap: Some(Bootstrap::new(1, Seed(20_260_819))),
            },
        )
        .expect("two-period estimate");
        let estimate = fitted.estimate;
        h_att.f(estimate.estimate);
        h_att.o(estimate.se);
        h_att.o(estimate.lower());
        h_att.o(estimate.upper());
        h_att.word(fitted.n_treated as u64);
        h_att.word(fitted.n_control as u64);
        h_att.f(fitted.total_weight);
        h_att.fs(fitted.influence.column(0));
        h_att.word(fitted.columns_dropped as u64);
        let families: Vec<u32> = frame.iter().map(|row| row.4).collect();
        h_att.fs(&did_cluster_sums(
            fitted.influence.column(0),
            &Clusters::from_labels_first_seen(&families).unwrap(),
        )
        .expect("sums"));

        // `kernels::did::nuisance`: the covariates, with the intercept the
        // estimators add.
        let nuisance_settings = NuisanceOptions {
            legacy_repeated_cross_section: false,
            ..nuisance_options()
        };
        let data = TwoPeriodData::panel(&treated, &delta)
            .weights(&weight)
            .covariates(covariates)
            .build()
            .expect("two-period data");
        let fit = nuisance::fit(&data, &nuisance_settings).expect("nuisance");
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
        let zeros = vec![0.0; n];
        let zero_change = TwoPeriodData::panel(&treated, &zeros)
            .weights(&weight)
            .covariates(covariates)
            .build()
            .expect("two-period data");
        let propensity =
            nuisance::propensity(&zero_change, &nuisance_settings).expect("propensity");
        propensity
            .kept_columns
            .iter()
            .for_each(|&kept| h_nuisance.word(u64::from(kept)));
        h_nuisance.fs(&propensity.coefficients);
        h_nuisance.word(propensity.iterations as u64);
        h_nuisance.word(u64::from(propensity.converged));
        let kept = nuisance::independent_columns(&data, Pruning::Pooled, 1e-8).expect("columns");
        for kept in kept {
            h_nuisance.word(u64::from(kept));
        }
    }
    out.push(("pairwise/drdid_panel_intercept", h_att.finish()));
    out.push(("pairwise/nuisance", h_nuisance.finish()));
}

fn cases() -> Vec<(&'static str, u64)> {
    let rows = panel();
    let data = columns(&rows);
    let mut out = Vec::new();

    let fitted = study_att_gt(&data, ComparisonGroup::MatchedNeverTreated, 0, false);
    assert!(
        fitted.sample().dropped_treated_at_start > 0,
        "no unit dropped at the start"
    );
    out.push(("att_gt/matched_never_treated", att_gt_digest(&fitted)));
    let pooled = study_att_gt(&data, ComparisonGroup::NeverTreated, 0, false);
    out.push(("att_gt/never_treated", att_gt_digest(&pooled)));
    let anticipation = study_att_gt(&data, ComparisonGroup::MatchedNeverTreated, 2, false);
    out.push(("att_gt/anticipation_2", att_gt_digest(&anticipation)));

    let clustered = study_att_gt(&data, ComparisonGroup::MatchedNeverTreated, 0, true);
    aggregations(&mut out, &fitted, &clustered);
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
    ("pairwise/drdid_panel_intercept", 0xcde5_9623_46a6_5cc1),
    ("pairwise/nuisance", 0x11d5_dcb5_3803_4f58),
];

#[test]
fn study_paths_are_bit_identical_to_the_recorded_hashes() {
    let current = cases();
    for (name, hash) in &current {
        epistat_testkit::hash_out(name, *hash);
    }
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
    if epistat_testkit::RECORDED_PLATFORM {
        assert_eq!(current.as_slice(), GOLDEN);
    }
}
