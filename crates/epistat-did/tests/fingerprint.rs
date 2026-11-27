//! The outputs of the main estimators, hashed to the bit, on the R reference
//! data and on 1, 4 and 12 threads.
//!
//! Each case hashes every number in the `Debug` form of what the estimator
//! returns, in order, so a field that is renamed or wrapped in `Option` leaves
//! the hash alone and a value that moves in its last bit changes it. `Debug`
//! prints an `f64` with the fewest digits that read back to the same bits.
//! Run with `--nocapture` to print the current hashes. They are asserted only
//! on aarch64 macOS, where they were recorded. Elsewhere the test computes them
//! and, when `EPISTAT_HASH_OUT` is set, appends them to that file.

use std::fmt::Debug;

use epistat_core::{Bootstrap, Level};
use epistat_did::att_gt::{
    self, Aggte, AggteOptions, AggteType, AggteWeights, AttGt, AttGtOptions, BasePeriod,
    ComparisonGroup, Design, Estimator, NuisanceOptions, UnbalancedPanel,
};
use epistat_did::continuous::{self, AcrtOptions, BasisSpec, ContinuousData};
use epistat_did::did_cc::{self, Assumption, DidCc, DidCcOptions};
use epistat_did::triple_did::{self, TripleDidData, TripleDidOptions};
use epistat_did::two_period::nuisance::{self, Pruning};
use epistat_did::two_period::{self, Estimator as E, TwoPeriod, TwoPeriodOptions};
use epistat_did::{Error, StaggeredPanel, TwoPeriodData};
use epistat_linalg::MatRef;
use epistat_testkit::{Fingerprint, load_json, on_threads};
use serde::Deserialize;

/// The numbers in `text`, in order: every token that starts with a digit, or
/// with a minus sign before a digit, and is not part of an identifier, read as
/// an `f64`, and `NaN`, `inf`, `true` and `false`.
fn numbers(text: &str) -> Vec<f64> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let previous_is_word =
            i > 0 && (bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
        let starts_number = bytes[i].is_ascii_digit()
            || (bytes[i] == b'-' && bytes.get(i + 1).is_some_and(u8::is_ascii_digit));
        if !previous_is_word && starts_number {
            let start = i;
            i += 1;
            while i < bytes.len()
                && (bytes[i].is_ascii_digit()
                    || matches!(bytes[i], b'.' | b'e' | b'E')
                    || (matches!(bytes[i], b'-' | b'+') && matches!(bytes[i - 1], b'e' | b'E')))
            {
                i += 1;
            }
            out.push(text[start..i].parse::<f64>().expect("a number"));
            continue;
        }
        if !previous_is_word {
            for (word, value) in [
                ("true", 1.0),
                ("false", 0.0),
                ("NaN", f64::NAN),
                ("-inf", f64::NEG_INFINITY),
                ("inf", f64::INFINITY),
            ] {
                let end = i + word.len();
                let next_is_word = bytes
                    .get(end)
                    .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_');
                if text[i..].starts_with(word) && !next_is_word {
                    out.push(value);
                    i = end - 1;
                    break;
                }
            }
        }
        i += 1;
    }
    out
}

fn digest(value: &impl Debug) -> u64 {
    Fingerprint::new()
        .f64s(&numbers(&format!("{value:?}")))
        .finish()
}

/// The hash of numbers in the order the `Debug` form of a result printed
/// them: a NaN as `NaN` prints it, so every NaN hashes alike.
fn hash_numbers(values: &[f64]) -> u64 {
    let values: Vec<f64> = values
        .iter()
        .map(|&v| if v.is_nan() { f64::NAN } else { v })
        .collect();
    Fingerprint::new().f64s(&values).finish()
}

/// A count or a period as the `Debug` form printed it.
#[expect(clippy::cast_precision_loss, reason = "small integers")]
fn int(value: impl Into<i128>) -> f64 {
    value.into() as f64
}

/// An `f64` field that is `Option` here and was NaN when undefined.
fn nan(value: Option<f64>) -> f64 {
    value.unwrap_or(f64::NAN)
}

/// The numbers of a cell as `AttGtEstimate` printed them.
fn cell_numbers(cell: &att_gt::AttGtCell, out: &mut Vec<f64>) {
    let estimate = cell.estimate.expect("an estimated cell");
    out.extend([
        int(cell.group),
        int(cell.time),
        int(cell.event_time),
        estimate.estimate,
        nan(estimate.se),
        nan(estimate.lower()),
        nan(estimate.upper()),
        int(cell.n_treated as u64),
        int(cell.n_control as u64),
        cell.total_weight,
    ]);
}

fn estimated(fit: &AttGt) -> impl Iterator<Item = &att_gt::AttGtCell> {
    fit.cells().iter().filter(|cell| cell.estimate.is_some())
}

/// The digest of an ATT(g,t) route's cells, their influence vectors, the
/// skipped and the pruned cells, without the sample, which
/// `att_gt_dr_panel_sample` pins on its own.
fn attgt_digest(fit: &AttGt) -> u64 {
    let mut out = Vec::new();
    for cell in estimated(fit) {
        cell_numbers(cell, &mut out);
    }
    let influence = fit.influence().expect("influence");
    for j in 0..influence.n_columns() {
        out.extend_from_slice(influence.column(j));
    }
    for cell in fit.cells().iter().filter(|cell| cell.skipped.is_some()) {
        out.extend([int(cell.group), int(cell.time), int(cell.base)]);
    }
    for cell in estimated(fit).filter(|cell| cell.columns_dropped > 0) {
        out.extend([
            int(cell.group),
            int(cell.time),
            int(cell.base),
            int(cell.columns_dropped as u64),
        ]);
    }
    hash_numbers(&out)
}

/// The digest of efficient cells as `EfficientDidEstimate` printed them, with
/// their baseline weights, influence vectors and diagnostics.
fn efficient_digest(fit: &AttGt) -> u64 {
    let mut out = Vec::new();
    let influence = fit.influence().expect("influence");
    for (j, cell) in estimated(fit).enumerate() {
        cell_numbers(cell, &mut out);
        let weights = cell.efficient.as_ref().expect("efficient weights");
        for baseline in &weights.baselines {
            out.extend([
                int(baseline.base),
                baseline.estimate,
                baseline.weight,
                int(baseline.n_treated as u64),
                int(baseline.n_control as u64),
            ]);
        }
        out.extend_from_slice(influence.column(j));
        out.extend(weights.covariance.as_slice().iter().map(|v| nan(*v)));
        out.extend_from_slice(&weights.precision_solution);
        out.extend(weights.ridge);
    }
    hash_numbers(&out)
}

/// The numbers of an `AggteResult`: the overall effect with its influence
/// function, each point with its own, the critical value and the band flag.
fn aggte_digest(result: &Aggte) -> u64 {
    let overall = result.overall.expect("an overall effect");
    let influence = result.influence();
    let mut out = vec![
        overall.estimate.estimate,
        nan(overall.estimate.se),
        nan(overall.estimate.lower()),
        nan(overall.estimate.upper()),
    ];
    out.extend_from_slice(influence.column(result.points.len()));
    for (j, point) in result.points.iter().enumerate() {
        out.extend([
            int(point.key),
            point.estimate.estimate,
            nan(point.estimate.se),
            nan(point.estimate.lower()),
            nan(point.estimate.upper()),
        ]);
        out.extend_from_slice(influence.column(j));
    }
    out.push(result.critical_value);
    out.push(f64::from(u8::from(result.uniform_band)));
    hash_numbers(&out)
}

/// The numbers of a fixed-weight summary as `AttGtAggregatedEstimate`
/// printed them.
fn summary_numbers(
    key: Option<i64>,
    estimate: &epistat_core::Estimate,
    n_cells: usize,
    total_weight: f64,
    out: &mut Vec<f64>,
) {
    out.extend(key.map(int));
    out.extend([
        estimate.estimate,
        nan(estimate.se),
        nan(estimate.lower()),
        nan(estimate.upper()),
        int(n_cells as u64),
        total_weight,
    ]);
}

/// The points of a summary by key, as the per-key aggregations printed them.
fn keyed_digest(result: &Aggte) -> u64 {
    let mut out = Vec::new();
    for point in &result.points {
        summary_numbers(
            Some(point.key),
            &point.estimate,
            point.n_cells,
            point.total_weight,
            &mut out,
        );
    }
    hash_numbers(&out)
}

/// The overall effect as `aggregate_att_gt_overall` printed it.
fn overall_digest(result: &Aggte) -> u64 {
    let overall = result.overall.expect("an overall effect");
    let mut out = Vec::new();
    summary_numbers(
        None,
        &overall.estimate,
        overall.n_cells,
        overall.total_weight,
        &mut out,
    );
    hash_numbers(&out)
}

/// The event-time path with its influence vectors after it, as
/// `AttGtEventTimeInfluenceOutput` printed it; without them for the
/// efficient aggregation, which returned the path alone.
fn event_time_digest(result: &Aggte, with_influence: bool) -> u64 {
    let mut out = Vec::new();
    for point in &result.points {
        summary_numbers(
            Some(point.key),
            &point.estimate,
            point.n_cells,
            point.total_weight,
            &mut out,
        );
    }
    if with_influence {
        for j in 0..result.points.len() {
            out.extend_from_slice(result.influence().column(j));
        }
    }
    hash_numbers(&out)
}

#[derive(Clone, Deserialize)]
struct MpdtaRow {
    year: i32,
    countyreal: i64,
    lemp: f64,
    #[serde(rename = "first.treat")]
    first_treat: i32,
    lpop: f64,
}

#[derive(Deserialize)]
struct Mpdta {
    data_subset: Vec<MpdtaRow>,
}

#[derive(Deserialize)]
struct TripleRow {
    id: i32,
    partition: i32,
    x1: f64,
    x2: f64,
    treat: i32,
    outcome: f64,
}

#[derive(Deserialize)]
struct TripleRef {
    data: Vec<TripleRow>,
}

#[derive(Deserialize)]
struct ContRef {
    delta_y: Vec<f64>,
    dose: Vec<f64>,
}

#[derive(Deserialize)]
struct CcRow {
    treated: bool,
    post_period: bool,
    outcome: f64,
    covariates: Vec<f64>,
}

#[derive(Deserialize)]
struct CcDesign {
    rows: Vec<CcRow>,
}

#[derive(Deserialize)]
struct CcRef {
    composition_shift: CcDesign,
}

/// The triple-difference units: group, partition, outcome change and the
/// two covariates column-major.
struct Triple {
    group: Vec<bool>,
    partition: Vec<bool>,
    change: Vec<f64>,
    covariates: Vec<f64>,
}

/// A repeated cross-section: arm, period, outcome, and the covariates
/// column-major with `k` columns.
struct Repeated {
    treated: Vec<bool>,
    post: Vec<bool>,
    outcome: Vec<f64>,
    weight: Vec<f64>,
    covariates: Vec<f64>,
    k: usize,
}

impl Repeated {
    fn data(&self) -> TwoPeriodData<'_> {
        TwoPeriodData::repeated_cross_section(&self.treated, &self.post, &self.outcome)
            .weights(&self.weight)
            .covariates(MatRef::from_column_major_slice(
                &self.covariates,
                self.treated.len(),
                self.k,
            ))
            .build()
            .expect("repeated cross-section")
    }
}

struct Data {
    mpdta: Vec<MpdtaRow>,
    triple: Triple,
    dose: Vec<f64>,
    dose_change: Vec<f64>,
    cc: Repeated,
}

fn data() -> Data {
    let mpdta: Mpdta = load_json("tests/data/cs_panel_dr_universal_ref.json");
    let triple: TripleRef = load_json("tests/data/triple_did_ref.json");
    let pairs: Vec<&[TripleRow]> = triple.data.chunks(2).collect();
    for pair in &pairs {
        assert_eq!(pair[0].id, pair[1].id);
    }
    let triple = Triple {
        group: pairs.iter().map(|pair| pair[1].treat == 1).collect(),
        partition: pairs.iter().map(|pair| pair[1].partition == 1).collect(),
        change: pairs
            .iter()
            .map(|pair| pair[1].outcome - pair[0].outcome)
            .collect(),
        covariates: pairs
            .iter()
            .map(|pair| pair[1].x1)
            .chain(pairs.iter().map(|pair| pair[1].x2))
            .collect(),
    };
    let continuous: ContRef = load_json("tests/data/contdid_ref.json");
    let cc: CcRef = load_json("tests/data/did_cc_ref_scaffold.json");
    let rows = cc.composition_shift.rows;
    let k = rows[0].covariates.len();
    let cc = Repeated {
        treated: rows.iter().map(|row| row.treated).collect(),
        post: rows.iter().map(|row| row.post_period).collect(),
        outcome: rows.iter().map(|row| row.outcome).collect(),
        weight: vec![1.0; rows.len()],
        covariates: (0..k)
            .flat_map(|j| rows.iter().map(move |row| row.covariates[j]))
            .collect(),
        k,
    };
    Data {
        mpdta: mpdta.data_subset,
        triple,
        dose: continuous.dose,
        dose_change: continuous.delta_y,
        cc,
    }
}

/// A weight that varies with the county, so the weighted paths are exercised.
fn weight(county: i64) -> f64 {
    0.5 + f64::from(u32::try_from(county % 7).expect("small"))
}

/// The `mpdta` rows as panel columns: counties as unit codes on a panel and
/// row numbers on a repeated cross-section, the county weights, log
/// population as the covariate, and county number modulo 37 as the cluster.
struct Columns {
    unit: Vec<u32>,
    period: Vec<i64>,
    cohort: Vec<Option<i64>>,
    outcome: Vec<f64>,
    weight: Vec<f64>,
    lpop: Vec<f64>,
    cluster: Vec<u32>,
}

fn columns(rows: &[MpdtaRow], design: Design) -> Columns {
    let code = |county: i64| u32::try_from(county).expect("a county number");
    Columns {
        unit: match design {
            Design::RepeatedCrossSection => (0..rows.len())
                .map(|i| u32::try_from(i).expect("rows"))
                .collect(),
            _ => rows.iter().map(|row| code(row.countyreal)).collect(),
        },
        period: rows.iter().map(|row| i64::from(row.year)).collect(),
        cohort: rows
            .iter()
            .map(|row| (row.first_treat != 0).then_some(i64::from(row.first_treat)))
            .collect(),
        outcome: rows.iter().map(|row| row.lemp).collect(),
        weight: rows.iter().map(|row| weight(row.countyreal)).collect(),
        lpop: rows.iter().map(|row| row.lpop).collect(),
        cluster: rows.iter().map(|row| code(row.countyreal % 37)).collect(),
    }
}

/// What the panel of a call carries.
#[derive(Clone, Copy, PartialEq)]
enum With {
    /// Weights and the covariate.
    Covariates,
    /// Weights, the covariate and the clusters.
    Clusters,
    /// Neither weights nor covariates.
    Plain,
}

fn fit(data: &Columns, with: With, options: AttGtOptions) -> AttGt {
    let lpop = MatRef::from_column_major_slice(&data.lpop, data.unit.len(), 1);
    let builder = StaggeredPanel::builder(&data.unit, &data.period, &data.cohort, &data.outcome);
    let builder = match with {
        With::Plain => builder,
        With::Covariates => builder.weights(&data.weight).covariates(lpop),
        With::Clusters => builder
            .weights(&data.weight)
            .covariates(lpop)
            .clusters(&data.cluster),
    };
    let panel = builder.build().expect("panel");
    att_gt::estimate(&panel, &options).expect("ATT(g,t)")
}

/// The ATT(g,t) settings of the hashes: `did`'s rules with the overlap check,
/// the legacy repeated cross-section computation and a 999-replicate cell
/// bootstrap, seeded as the estimator's former default.
fn att_gt_options(comparison_group: ComparisonGroup, base_period: BasePeriod) -> AttGtOptions {
    AttGtOptions {
        comparison_group,
        base_period,
        nuisance: NuisanceOptions {
            legacy_repeated_cross_section: true,
            ..NuisanceOptions::default()
        },
        bootstrap: Some(Bootstrap::new(999, 17_431)),
        ..AttGtOptions::default()
    }
}

/// The same settings without the cell bootstrap, which clusters refuse.
fn clustered_options(comparison_group: ComparisonGroup, base_period: BasePeriod) -> AttGtOptions {
    AttGtOptions {
        bootstrap: None,
        ..att_gt_options(comparison_group, base_period)
    }
}

/// The estimators' settings with a 999-replicate bootstrap interval, seeded
/// as the estimators' former default, so the hashes pin the bootstrap path,
/// and the legacy repeated cross-section doubly robust computation, which
/// `drdid_rc` and `att_gt_dr_rc` pin. `drdid_panel_analytic` pins the default
/// analytic interval, and `drdid_rc1` and `att_gt_dr_rc1` the default
/// `drdid_rc1` estimator.
fn two_period_options(estimator: two_period::Estimator) -> TwoPeriodOptions {
    TwoPeriodOptions {
        estimator,
        nuisance: NuisanceOptions {
            legacy_repeated_cross_section: true,
            ..NuisanceOptions::default()
        },
        level: Level::default(),
        bootstrap: Some(Bootstrap::new(999, 17_431)),
    }
}

/// The compositional-change settings with bootstrap intervals, for the same
/// reason as [`two_period_options`].
fn did_cc_options(assumption: Assumption) -> DidCcOptions {
    DidCcOptions {
        assumption,
        nuisance: two_period_options(two_period::Estimator::DoublyRobust).nuisance,
        bootstrap: Some(Bootstrap::new(999, 17_431)),
        ..DidCcOptions::default()
    }
}

/// The numbers of a two-period estimate as `DrDidEstimate` printed them.
fn two_period_digest(fit: &Result<TwoPeriod, Error>) -> u64 {
    let fit = fit.as_ref().expect("a two-period estimate");
    let e = fit.estimate;
    let mut numbers = vec![
        e.estimate,
        nan(e.se),
        nan(e.lower()),
        nan(e.upper()),
        int(fit.n_treated as u64),
        int(fit.n_control as u64),
        fit.total_weight,
    ];
    numbers.extend_from_slice(fit.influence.column(0));
    numbers.push(int(fit.columns_dropped as u64));
    hash_numbers(&numbers)
}

/// The numbers of an estimate under compositional changes as
/// `DidCcEstimate` printed them.
fn did_cc_numbers(fit: &DidCc, numbers: &mut Vec<f64>) {
    let e = fit.estimate;
    numbers.extend([
        e.estimate,
        nan(e.se),
        nan(e.lower()),
        nan(e.upper()),
        int(fit.n_treated_post as u64),
        int(fit.n_treated_pre as u64),
        int(fit.n_control_post as u64),
        int(fit.n_control_pre as u64),
        fit.total_weight,
    ]);
    numbers.extend_from_slice(fit.influence.column(0));
}

/// The 2004 cohort against the never treated, 2003 to 2004: the panel of
/// outcome changes with log population at 2004 as the covariate, and the
/// repeated cross-section of both years.
fn cell(rows: &[MpdtaRow]) -> (Panel, Repeated) {
    let kept = |row: &&MpdtaRow| row.first_treat == 2004 || row.first_treat == 0;
    let mut panel = Panel {
        treated: Vec::new(),
        change: Vec::new(),
        weight: Vec::new(),
        lpop: Vec::new(),
    };
    for post in rows.iter().filter(kept).filter(|row| row.year == 2004) {
        let base = rows
            .iter()
            .find(|row| row.countyreal == post.countyreal && row.year == 2003)
            .expect("base year");
        panel.treated.push(post.first_treat == 2004);
        panel.change.push(post.lemp - base.lemp);
        panel.weight.push(weight(post.countyreal));
        panel.lpop.push(post.lpop);
    }
    let repeated: Vec<&MpdtaRow> = rows
        .iter()
        .filter(kept)
        .filter(|row| row.year == 2003 || row.year == 2004)
        .collect();
    let repeated = Repeated {
        treated: repeated.iter().map(|row| row.first_treat == 2004).collect(),
        post: repeated.iter().map(|row| row.year == 2004).collect(),
        outcome: repeated.iter().map(|row| row.lemp).collect(),
        weight: repeated.iter().map(|row| weight(row.countyreal)).collect(),
        covariates: repeated.iter().map(|row| row.lpop).collect(),
        k: 1,
    };
    (panel, repeated)
}

/// A two-period panel: arm, outcome change, weight and one covariate.
struct Panel {
    treated: Vec<bool>,
    change: Vec<f64>,
    weight: Vec<f64>,
    lpop: Vec<f64>,
}

impl Panel {
    fn data(&self) -> TwoPeriodData<'_> {
        TwoPeriodData::panel(&self.treated, &self.change)
            .weights(&self.weight)
            .covariates(MatRef::from_column_major_slice(
                &self.lpop,
                self.treated.len(),
                1,
            ))
            .build()
            .expect("panel")
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "one list of cases in the order of the recorded hashes"
)]
fn cases(data: &Data) -> Vec<(&'static str, u64)> {
    let mut out = Vec::new();
    let level = Level::default();
    let panel_data = columns(&data.mpdta, Design::Panel);
    let rc_data = columns(&data.mpdta, Design::RepeatedCrossSection);

    // The efficient ATT(g,t) on the panel without covariates.
    let efficient_panel = fit(
        &panel_data,
        With::Plain,
        AttGtOptions {
            estimator: Estimator::Efficient,
            ..AttGtOptions::default()
        },
    );
    out.push(("att_gt_efficient", efficient_digest(&efficient_panel)));

    // The panel route, as Study I runs it, and the aggregations.
    for (name, comparison, base) in [
        (
            "att_gt_dr_panel_universal_never",
            ComparisonGroup::NeverTreated,
            BasePeriod::Universal,
        ),
        (
            "att_gt_dr_panel_varying_notyet",
            ComparisonGroup::NotYetTreated,
            BasePeriod::Varying,
        ),
    ] {
        out.push((
            name,
            attgt_digest(&fit(
                &panel_data,
                With::Covariates,
                att_gt_options(comparison, base),
            )),
        ));
    }
    let fitted = fit(
        &panel_data,
        With::Covariates,
        att_gt_options(ComparisonGroup::NeverTreated, BasePeriod::Universal),
    );
    let aggregate = |fit: &AttGt, options: AggteOptions| fit.aggregate(&options).expect("aggte");
    for (name, aggregation, balance_e) in [
        ("aggte_simple", AggteType::Simple, None),
        ("aggte_dynamic", AggteType::Dynamic, None),
        ("aggte_dynamic_balanced", AggteType::Dynamic, Some(1)),
        ("aggte_group", AggteType::Group, None),
        ("aggte_calendar", AggteType::Calendar, None),
    ] {
        out.push((
            name,
            aggte_digest(&aggregate(
                &fitted,
                AggteOptions {
                    aggregation,
                    balance_e,
                    ..AggteOptions::default()
                },
            )),
        ));
    }
    for (names, weights) in [
        (
            [
                "event_time_equal",
                "cohort_equal",
                "calendar_equal",
                "overall_equal",
            ],
            AggteWeights::Equal,
        ),
        (
            [
                "event_time_count",
                "cohort_count",
                "calendar_count",
                "overall_count",
            ],
            AggteWeights::TreatedCount,
        ),
        (
            [
                "event_time_weight",
                "cohort_weight",
                "calendar_weight",
                "overall_weight",
            ],
            AggteWeights::TotalWeight,
        ),
    ] {
        // The treated-count cohort, calendar and overall summaries were those
        // of `did::aggte`, whose cohort shares are estimated.
        let keyed_weights = if weights == AggteWeights::TreatedCount {
            AggteWeights::Estimated
        } else {
            weights
        };
        let with = |aggregation, weights| AggteOptions {
            aggregation,
            weights,
            ..AggteOptions::default()
        };
        out.push((
            names[0],
            event_time_digest(&aggregate(&fitted, with(AggteType::Dynamic, weights)), true),
        ));
        out.push((
            names[1],
            keyed_digest(&aggregate(&fitted, with(AggteType::Group, keyed_weights))),
        ));
        out.push((
            names[2],
            keyed_digest(&aggregate(
                &fitted,
                with(AggteType::Calendar, keyed_weights),
            )),
        ));
        out.push((
            names[3],
            overall_digest(&aggregate(&fitted, with(AggteType::Simple, keyed_weights))),
        ));
    }
    // Clusters of units: county number modulo 37.
    let clustered = fit(
        &panel_data,
        With::Clusters,
        clustered_options(ComparisonGroup::NeverTreated, BasePeriod::Universal),
    );
    let dynamic = AggteOptions {
        aggregation: AggteType::Dynamic,
        ..AggteOptions::default()
    };
    out.push((
        "aggte_dynamic_clustered",
        aggte_digest(&aggregate(&clustered, dynamic)),
    ));
    out.push((
        "aggte_dynamic_bootstrap",
        aggte_digest(&aggregate(
            &clustered,
            AggteOptions {
                bootstrap: Some(Bootstrap::new(999, 20_261_124)),
                band: att_gt::Band::Uniform,
                ..dynamic
            },
        )),
    ));
    for (name, aggregation) in [
        ("aggte_simple_max_e", AggteType::Simple),
        ("aggte_group_max_e", AggteType::Group),
    ] {
        out.push((
            name,
            aggte_digest(&aggregate(
                &fitted,
                AggteOptions {
                    aggregation,
                    max_e: Some(1),
                    ..AggteOptions::default()
                },
            )),
        ));
    }
    let pretest = fitted.pretest().expect("pre-test");
    let mut numbers = vec![pretest.statistic, int(pretest.df as u64), pretest.p_value];
    for &(group, time) in &pretest.cells {
        numbers.extend([int(group), int(time)]);
    }
    out.push(("pretrend_wald", hash_numbers(&numbers)));
    let bands = Bootstrap::new(999, 20_261_122);
    for (name, fit) in [
        ("mboot_bands", &fitted),
        ("mboot_bands_clustered", &clustered),
    ] {
        let bands = fit.bands(level, bands).expect("bands");
        let mut numbers = Vec::new();
        for band in &bands.cells {
            numbers.extend([
                int(band.group),
                int(band.time),
                int(band.event_time),
                band.estimate,
                nan(band.se),
                nan(band.lower),
                nan(band.upper),
            ]);
        }
        out.push((name, hash_numbers(&numbers)));
    }
    let mut numbers = Vec::new();
    for cell in estimated(&clustered) {
        cell_numbers(cell, &mut numbers);
    }
    out.push(("clustered_standard_errors", hash_numbers(&numbers)));

    // The repeated cross-section route and the other pair estimators.
    let rc_options = |estimator| AttGtOptions {
        estimator,
        design: Design::RepeatedCrossSection,
        ..att_gt_options(ComparisonGroup::NotYetTreated, BasePeriod::Varying)
    };
    for (name, estimator) in [
        ("att_gt_dr_rc", Estimator::TraditionalDoublyRobust),
        ("att_gt_dr_rc_efficient", Estimator::DoublyRobust),
        ("att_gt_ipw_rc", Estimator::InverseProbabilityWeighting),
        ("att_gt_or_rc", Estimator::OutcomeRegression),
    ] {
        out.push((
            name,
            attgt_digest(&fit(&rc_data, With::Covariates, rc_options(estimator))),
        ));
    }
    // The efficient doubly robust estimator reads each row as its own
    // observation.
    let efficient = fit(
        &rc_data,
        With::Covariates,
        AttGtOptions {
            estimator: Estimator::Efficient,
            design: Design::RepeatedCrossSection,
            ..att_gt_options(ComparisonGroup::NeverTreated, BasePeriod::Varying)
        },
    );
    out.push(("att_gt_efficient_dr", efficient_digest(&efficient)));
    out.push((
        "aggregate_efficient_event_time",
        event_time_digest(
            &aggregate(
                &efficient,
                AggteOptions {
                    aggregation: AggteType::Dynamic,
                    weights: AggteWeights::TreatedCount,
                    ..AggteOptions::default()
                },
            ),
            false,
        ),
    ));

    // Two-period estimators on one cell.
    let (panel_cell, rc_cell) = cell(&data.mpdta);
    let (panel, repeated) = (panel_cell.data(), rc_cell.data());
    let two = |data: &TwoPeriodData<'_>, options: TwoPeriodOptions| {
        two_period_digest(&two_period::estimate(data, &options))
    };
    out.push((
        "drdid_panel",
        two(&panel, two_period_options(E::DoublyRobust)),
    ));
    out.push((
        "drdid_panel_analytic",
        two(
            &panel,
            TwoPeriodOptions {
                estimator: E::DoublyRobust,
                ..TwoPeriodOptions::default()
            },
        ),
    ));
    out.push((
        "drdid_improved_panel",
        two(&panel, two_period_options(E::Improved)),
    ));
    out.push((
        "drdid_rc",
        two(&repeated, two_period_options(E::TraditionalDoublyRobust)),
    ));
    out.push((
        "drdid_improved_rc",
        two(&repeated, two_period_options(E::Improved)),
    ));
    out.push((
        "drdid_rc_efficient",
        two(&repeated, two_period_options(E::DoublyRobust)),
    ));
    let plain =
        TwoPeriodData::repeated_cross_section(&rc_cell.treated, &rc_cell.post, &rc_cell.outcome)
            .weights(&rc_cell.weight)
            .build()
            .expect("2x2");
    let two_by_two =
        two_period::two_by_two(&plain, Level::confidence(0.95).expect("level")).expect("2x2");
    out.push((
        "two_by_two",
        hash_numbers(&[
            two_by_two.estimate,
            nan(two_by_two.se),
            nan(two_by_two.lower()),
            nan(two_by_two.upper()),
        ]),
    ));
    let doubled: Vec<f64> = panel_cell
        .lpop
        .iter()
        .copied()
        .chain(panel_cell.lpop.iter().map(|x| x * 2.0))
        .collect();
    let doubled = TwoPeriodData::panel(&panel_cell.treated, &panel_cell.change)
        .weights(&panel_cell.weight)
        .covariates(MatRef::from_column_major_slice(
            &doubled,
            panel_cell.treated.len(),
            2,
        ))
        .build()
        .expect("panel");
    let nuisance_options = two_period_options(E::DoublyRobust).nuisance;
    out.push((
        "panel_nuisance",
        digest(&nuisance::fit(&doubled, &nuisance_options)),
    ));
    out.push((
        "panel_propensity",
        digest(&nuisance::propensity(&doubled, &nuisance_options)),
    ));
    out.push((
        "independent_columns",
        digest(&nuisance::independent_columns(
            &doubled,
            Pruning::PerArm,
            1e-7,
        )),
    ));

    // Triple differences, continuous treatment, compositional changes.
    let triple = &data.triple;
    let n_triple = triple.change.len();
    let triple_data = TripleDidData::builder(&triple.group, &triple.partition, &triple.change)
        .covariates(MatRef::from_column_major_slice(
            &triple.covariates,
            n_triple,
            2,
        ))
        .build()
        .expect("triple");
    let ddd = triple_did::estimate(
        &triple_data,
        &TripleDidOptions {
            nuisance: nuisance_options,
            level: Level::default(),
            bootstrap: Some(Bootstrap::new(999, 17_431)),
        },
    )
    .expect("triple differences");
    let mut numbers = vec![
        ddd.estimate.estimate,
        nan(ddd.estimate.se),
        nan(ddd.estimate.lower()),
        nan(ddd.estimate.upper()),
    ];
    numbers.extend_from_slice(ddd.influence.column(0));
    out.push(("dr_ddd", hash_numbers(&numbers)));
    let doses = ContinuousData::builder(&data.dose, &data.dose_change)
        .build()
        .expect("doses");
    let acrt = continuous::estimate(
        &doses,
        &AcrtOptions {
            basis: BasisSpec::Polynomial {
                degree: 3,
                constant: true,
            },
            ..AcrtOptions::default()
        },
    )
    .expect("ACRT");
    let mut numbers = vec![acrt.estimate.estimate];
    numbers.extend_from_slice(&acrt.coefficients);
    numbers.push(nan(acrt.estimate.se));
    numbers.extend_from_slice(acrt.influence.column(0));
    out.push(("acrt_sieve", hash_numbers(&numbers)));
    let cc = data.cc.data();
    for (name, assumption) in [
        ("did_cc_robust", Assumption::Robust),
        ("did_cc_stationary", Assumption::Stationary),
    ] {
        let fit = did_cc::estimate(&cc, &did_cc_options(assumption)).expect("did_cc");
        let mut numbers = Vec::new();
        did_cc_numbers(&fit, &mut numbers);
        out.push((name, hash_numbers(&numbers)));
    }
    let test = did_cc::test(&cc, &did_cc_options(Assumption::Robust)).expect("did_cc test");
    let mut numbers = Vec::new();
    did_cc_numbers(&test.robust, &mut numbers);
    did_cc_numbers(&test.stationary, &mut numbers);
    numbers.extend([
        test.difference,
        test.se,
        test.statistic,
        test.p_value,
        f64::from(u8::from(test.p_value < 0.05)),
    ]);
    out.push(("did_cc_test", hash_numbers(&numbers)));

    // The panel IPW and outcome-regression routes, the sample of the panel
    // route, and an unbalanced panel under the two rules that keep it.
    out.push((
        "att_gt_ipw_panel",
        attgt_digest(&fit(
            &panel_data,
            With::Covariates,
            AttGtOptions {
                estimator: Estimator::InverseProbabilityWeighting,
                ..att_gt_options(ComparisonGroup::NeverTreated, BasePeriod::Universal)
            },
        )),
    ));
    out.push((
        "att_gt_or_panel",
        attgt_digest(&fit(
            &panel_data,
            With::Covariates,
            AttGtOptions {
                estimator: Estimator::OutcomeRegression,
                ..att_gt_options(ComparisonGroup::NotYetTreated, BasePeriod::Varying)
            },
        )),
    ));
    out.push(("att_gt_dr_panel_sample", digest(fitted.sample())));
    let unbalanced_rows: Vec<MpdtaRow> = data
        .mpdta
        .iter()
        .filter(|row| !(row.countyreal % 7 == 0 && row.year == 2005))
        .cloned()
        .collect();
    let unbalanced_data = columns(&unbalanced_rows, Design::Panel);
    for (name, unbalanced) in [
        (
            "att_gt_dr_panel_unbalanced_rc",
            UnbalancedPanel::RepeatedCrossSection,
        ),
        (
            "att_gt_dr_panel_unbalanced_per_cell",
            UnbalancedPanel::PerCell,
        ),
    ] {
        out.push((
            name,
            attgt_digest(&fit(
                &unbalanced_data,
                With::Covariates,
                AttGtOptions {
                    unbalanced,
                    ..att_gt_options(ComparisonGroup::NeverTreated, BasePeriod::Universal)
                },
            )),
        ));
    }
    let (panel_cell, rc_cell) = cell(&data.mpdta);
    let (panel, repeated) = (panel_cell.data(), rc_cell.data());
    out.push((
        "ipw_did_panel",
        two(&panel, two_period_options(E::InverseProbabilityWeighting)),
    ));
    out.push((
        "or_did_panel",
        two(&panel, two_period_options(E::OutcomeRegression)),
    ));
    out.push((
        "ipw_did_rc",
        two(
            &repeated,
            two_period_options(E::InverseProbabilityWeighting),
        ),
    ));
    out.push((
        "or_did_rc",
        two(&repeated, two_period_options(E::OutcomeRegression)),
    ));
    out.push((
        "drdid_rc1",
        two(
            &repeated,
            TwoPeriodOptions {
                nuisance: NuisanceOptions::default(),
                ..two_period_options(E::TraditionalDoublyRobust)
            },
        ),
    ));
    out.push((
        "att_gt_dr_rc1",
        attgt_digest(&fit(
            &rc_data,
            With::Covariates,
            AttGtOptions {
                nuisance: NuisanceOptions::default(),
                ..rc_options(Estimator::TraditionalDoublyRobust)
            },
        )),
    ));
    let mut numbers = Vec::new();
    for cell in fitted.cells() {
        if let Some(overlap) = cell.overlap {
            numbers.extend([int(cell.group), int(cell.time), int(cell.base)]);
            numbers.extend(overlap.max_propensity);
        }
    }
    out.push(("att_gt_dr_panel_overlap", hash_numbers(&numbers)));
    out
}

const GOLDEN: &[(&str, u64)] = &[
    ("att_gt_efficient", 0x201f_58e9_78ec_8f24),
    ("att_gt_dr_panel_universal_never", 0xcc70_fa0b_a55a_a346),
    ("att_gt_dr_panel_varying_notyet", 0xc9c4_06f6_5f4b_9158),
    ("aggte_simple", 0xda55_7f26_63da_48c7),
    ("aggte_dynamic", 0x50a1_8f2f_2fe8_c5f5),
    ("aggte_dynamic_balanced", 0x4f35_2b2d_2722_852a),
    ("aggte_group", 0xbac2_b5d8_e0d0_ddcc),
    ("aggte_calendar", 0x909c_e3a0_c3af_c20d),
    ("event_time_equal", 0x8879_9cf0_bd43_632c),
    ("cohort_equal", 0x51a3_128f_98c1_c126),
    ("calendar_equal", 0x40fa_2903_c5a4_2bf5),
    ("overall_equal", 0xb2d1_26be_796d_43c7),
    ("event_time_count", 0x4b2c_00d1_96fa_d5e3),
    ("cohort_count", 0x51a3_128f_98c1_c126),
    ("calendar_count", 0x2d87_5901_42bc_02a0),
    ("overall_count", 0x5a63_c7fb_9e3b_6921),
    ("event_time_weight", 0x4f13_5ec5_6c2e_b289),
    ("cohort_weight", 0x51a3_128f_98c1_c126),
    ("calendar_weight", 0xb4c1_658b_4b25_7f67),
    ("overall_weight", 0xf056_78e1_c999_68bb),
    ("aggte_dynamic_clustered", 0x7fde_61d3_537f_a0bf),
    ("aggte_dynamic_bootstrap", 0x5edd_21ef_58a6_3fee),
    ("aggte_simple_max_e", 0x453b_496b_7eab_d0fd),
    ("aggte_group_max_e", 0xbf29_a586_8de7_fe37),
    ("pretrend_wald", 0x8f10_ca3c_4156_c995),
    ("mboot_bands", 0xb8ba_24d8_c5f3_60f0),
    ("mboot_bands_clustered", 0x9c23_7ed5_f361_ca27),
    ("clustered_standard_errors", 0x121c_5c2e_b2b8_f6f3),
    ("att_gt_dr_rc", 0x37ef_b315_808d_1e84),
    ("att_gt_dr_rc_efficient", 0x4944_165b_8d88_f9b4),
    ("att_gt_ipw_rc", 0x56f7_d6ad_d427_e4d3),
    ("att_gt_or_rc", 0x6872_5381_68cf_3c11),
    ("att_gt_efficient_dr", 0x74b6_3e4f_4e86_2f51),
    ("aggregate_efficient_event_time", 0x0b45_6585_4366_2606),
    ("drdid_panel", 0x9a35_5276_7786_14a9),
    ("drdid_panel_analytic", 0x3a8b_258f_748f_a7e0),
    ("drdid_improved_panel", 0x9731_c94f_9fdd_c514),
    ("drdid_rc", 0x5e65_35b6_1358_8f15),
    ("drdid_improved_rc", 0x922c_d1a0_99dc_b53a),
    ("drdid_rc_efficient", 0xfb43_938e_d94e_12eb),
    ("two_by_two", 0x3614_ceb4_ac37_7093),
    ("panel_nuisance", 0xd088_7cbd_b32a_d261),
    ("panel_propensity", 0x0392_1b18_f042_d985),
    ("independent_columns", 0x45f2_030f_bd17_1265),
    ("dr_ddd", 0x8658_3954_e214_b213),
    ("acrt_sieve", 0xdbf7_0235_0d74_2314),
    ("did_cc_robust", 0x4c06_ee7e_0a32_a429),
    ("did_cc_stationary", 0x21ff_0ebd_15c7_0221),
    ("did_cc_test", 0x4c05_eb0a_4dc3_ab4d),
    ("att_gt_ipw_panel", 0x1bdd_03b5_e8bb_b591),
    ("att_gt_or_panel", 0x896f_8af5_dfa0_267f),
    ("att_gt_dr_panel_sample", 0x8d2e_1d4c_fe74_6ac0),
    ("att_gt_dr_panel_unbalanced_rc", 0x4321_bd02_5d7b_9825),
    ("att_gt_dr_panel_unbalanced_per_cell", 0x2677_bd8f_04da_c3f9),
    ("ipw_did_panel", 0xb52f_6a05_a082_a566),
    ("or_did_panel", 0x1a96_d211_3058_aac8),
    ("ipw_did_rc", 0x05db_71a0_55cd_b5c6),
    ("or_did_rc", 0xf733_191b_2d7f_a9fd),
    ("drdid_rc1", 0x0473_5319_db10_7207),
    ("att_gt_dr_rc1", 0x4789_4f8f_bb24_580a),
    ("att_gt_dr_panel_overlap", 0xdb27_39b9_9043_05a9),
];

#[test]
fn estimator_outputs_are_bit_identical_to_the_recorded_hashes() {
    let data = data();
    let runs = on_threads(&[1, 4, 12], || cases(&data));
    for run in &runs[1..] {
        assert_eq!(run, &runs[0], "the hashes depend on the thread count");
    }
    let current = &runs[0];
    for (name, hash) in current {
        epistat_testkit::hash_out(name, *hash);
    }
    // The hashes were recorded on aarch64 macOS; other platforms' maths
    // libraries differ in the last bits.
    if !epistat_testkit::RECORDED_PLATFORM {
        return;
    }
    if current
        .iter()
        .map(|(n, h)| (*n, *h))
        .ne(GOLDEN.iter().copied())
    {
        for (name, hash) in current {
            let hex = format!("{hash:016x}");
            let groups: Vec<&str> = (0..4).map(|i| &hex[4 * i..4 * i + 4]).collect();
            println!("    (\"{name}\", 0x{}),", groups.join("_"));
        }
    }
    assert_eq!(current.as_slice(), GOLDEN);
}
