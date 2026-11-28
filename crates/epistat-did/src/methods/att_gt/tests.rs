use epistat_core::{Bootstrap, Classify, InputError, Seed};
use epistat_linalg::MatRef;

use super::*;
use crate::methods::staggered::StaggeredPanel;

/// Panel columns pushed row by row, in unit and then period order.
#[derive(Default)]
struct Data {
    unit: Vec<u32>,
    period: Vec<i64>,
    cohort: Vec<Option<i64>>,
    outcome: Vec<f64>,
    weight: Vec<f64>,
    /// Column-major, `n_covariates` columns.
    covariates: Vec<Vec<f64>>,
    cluster: Vec<u32>,
    matched: Vec<Option<i64>>,
}

impl Data {
    fn push(&mut self, unit: u32, cohort: Option<i64>, period: i64, outcome: f64, x: &[f64]) {
        self.unit.push(unit);
        self.period.push(period);
        self.cohort.push(cohort);
        self.outcome.push(outcome);
        self.weight.push(1.0);
        self.covariates.resize_with(x.len(), Vec::new);
        for (column, &value) in self.covariates.iter_mut().zip(x) {
            column.push(value);
        }
        self.cluster.push(unit);
        self.matched.push(None);
    }

    /// The fit of the panel, with the covariates, the clusters and the
    /// matched cohorts as asked.
    fn fit(&self, clusters: bool, options: &AttGtOptions) -> Result<AttGt, Error> {
        self.fit_design(Design::Panel, clusters, options)
    }

    /// The fit of the rows as a panel or as repeated cross-sections.
    fn fit_design(
        &self,
        design: Design,
        clusters: bool,
        options: &AttGtOptions,
    ) -> Result<AttGt, Error> {
        let flat: Vec<f64> = self.covariates.concat();
        let n = self.unit.len();
        let builder = match design {
            Design::Panel => {
                StaggeredPanel::builder(&self.unit, &self.period, &self.cohort, &self.outcome)
            }
            _ => StaggeredPanel::cross_sections(&self.period, &self.cohort, &self.outcome),
        };
        let builder =
            builder.weights(epistat_core::CaseWeights::new(&self.weight).expect("weights"));
        let builder = if self.covariates.is_empty() {
            builder
        } else {
            builder.covariates(MatRef::from_column_major_slice(
                &flat,
                n,
                self.covariates.len(),
            ))
        };
        let builder = if clusters {
            builder.clusters(&self.cluster)
        } else {
            builder
        };
        let builder = if self.matched.iter().any(Option::is_some) {
            builder.comparison_cohort(&self.matched)
        } else {
            builder
        };
        estimate(&builder.build()?, options)
    }
}

fn universal() -> AttGtOptions {
    AttGtOptions {
        base_period: BasePeriod::Universal,
        ..AttGtOptions::default()
    }
}

fn att(fit: &AttGt, group: i64, time: i64) -> f64 {
    fit.cells()
        .iter()
        .find(|cell| cell.group == group && cell.time == time)
        .and_then(|cell| cell.estimate)
        .map(|estimate| estimate.estimate)
        .expect("an estimated cell")
}

/// Thirty units over periods 1 to 4: ten first treated in period 2 with an
/// effect of 2, ten in period 3 with an effect of 1, ten never treated. The
/// outcome is linear in the covariate and the period, so the doubly robust
/// estimate is exact up to rounding; `noise` adds a deterministic wobble that
/// gives the influence functions a spread.
fn staggered(noise: f64) -> Data {
    let mut data = Data::default();
    for unit in 0..30_u32 {
        let cohort = match unit / 10 {
            0 => Some(2),
            1 => Some(3),
            _ => None,
        };
        let x = f64::from(unit % 3);
        for period in 1..=4_i64 {
            let effect = match cohort {
                Some(g) if period >= g => (4 - g) as f64,
                _ => 0.0,
            };
            let wobble = noise * ((i64::from(unit) * 7 + period * 3) % 5 - 2) as f64;
            let y = f64::from(unit % 5) + 0.5 * period as f64 + 0.25 * x + effect + wobble;
            data.push(unit, cohort, period, y, &[x]);
        }
    }
    data
}

/// The exact data: sums of about thirty terms of size ten round to within a
/// few times 1e-14, so 1e-10 leaves a wide margin.
const EXACT: f64 = 1e-10;

#[test]
fn a_constant_effect_is_recovered_cell_by_cell() {
    let fit = staggered(0.0).fit(false, &universal()).unwrap();
    // Universal base: cohort 2 against period 1, cohort 3 against period 2.
    let cells: Vec<(i64, i64, i64)> = fit
        .cells()
        .iter()
        .map(|cell| (cell.group, cell.time, cell.base))
        .collect();
    assert_eq!(
        cells,
        [
            (2, 2, 1),
            (2, 3, 1),
            (2, 4, 1),
            (3, 1, 2),
            (3, 3, 2),
            (3, 4, 2)
        ]
    );
    for (group, time, want) in [
        (2, 2, 2.0),
        (2, 3, 2.0),
        (2, 4, 2.0),
        (3, 1, 0.0),
        (3, 3, 1.0),
    ] {
        assert!((att(&fit, group, time) - want).abs() < EXACT);
    }
    let influence = fit.influence().unwrap();
    assert_eq!((influence.n_units(), influence.n_columns()), (30, 6));
    assert_eq!(
        influence.rule(),
        epistat_inference::ClusterRule::NoSmallSample
    );
    assert_eq!(fit.sample().units, (0..30).collect::<Vec<u32>>());
    let cell = &fit.cells()[0];
    assert_eq!(
        (cell.n_treated, cell.n_control, cell.event_time),
        (10, 10, 0)
    );
    assert!(cell.overlap.is_some_and(|o| o.max_propensity.is_some()));
}

#[test]
fn the_four_summaries_weight_the_cells_as_aggte() {
    let fit = staggered(0.0).fit(false, &universal()).unwrap();
    let run = |aggregation, weights| {
        fit.aggregate(&AggteOptions {
            aggregation,
            weights,
            ..AggteOptions::default()
        })
        .unwrap()
    };
    let close = |a: f64, b: f64| (a - b).abs() < EXACT;
    let keyed = |result: &Aggte| -> Vec<(i64, f64)> {
        result
            .points
            .iter()
            .map(|p| (p.key, p.estimate.estimate))
            .collect()
    };
    let overall = |result: &Aggte| result.overall.unwrap().estimate.estimate;

    // Equal cohort shares: event times 0 and 1 average both cohorts.
    let dynamic = run(AggteType::Dynamic, AggteWeights::Estimated);
    let want = [(-2, 0.0), (0, 1.5), (1, 1.5), (2, 2.0)];
    for ((key, value), (want_key, want_value)) in keyed(&dynamic).into_iter().zip(want) {
        assert_eq!(key, want_key);
        assert!(close(value, want_value));
    }
    assert!(close(overall(&dynamic), 5.0 / 3.0));
    assert_eq!(dynamic.influence().n_columns(), 5);

    let group = run(AggteType::Group, AggteWeights::Estimated);
    assert!(close(keyed(&group)[0].1, 2.0) && close(keyed(&group)[1].1, 1.0));
    assert!(close(overall(&group), 1.5));
    let calendar = run(AggteType::Calendar, AggteWeights::Estimated);
    assert!(close(overall(&calendar), 5.0 / 3.0));
    let simple = run(AggteType::Simple, AggteWeights::Estimated);
    assert_eq!(simple.points, []);
    assert!(close(overall(&simple), 1.6));
    assert_eq!(simple.overall.unwrap().n_cells, 5);

    // The fixed weights count every post-treatment cell's total weight, and
    // only the simple summary has an overall effect.
    let cell_weight = fit.cells()[0].total_weight;
    for weights in [
        AggteWeights::Equal,
        AggteWeights::TreatedCount,
        AggteWeights::TotalWeight,
    ] {
        let group = run(AggteType::Group, weights);
        assert!(group.overall.is_none());
        assert_eq!(group.points[0].n_cells, 3);
        assert_eq!(
            group.points[0].total_weight.to_bits(),
            (3.0 * cell_weight).to_bits()
        );
        let simple = run(AggteType::Simple, weights);
        assert_eq!(
            simple.overall.unwrap().total_weight.to_bits(),
            (5.0 * cell_weight).to_bits()
        );
        // Every cell of an event time, the pre-treatment one included.
        assert_eq!(run(AggteType::Dynamic, weights).points.len(), 4);
    }
}

#[test]
fn a_combination_of_points_has_the_standard_error_of_its_influence() {
    let fit = staggered(0.3).fit(true, &universal()).unwrap();
    let dynamic = fit
        .aggregate(&AggteOptions {
            aggregation: AggteType::Dynamic,
            ..AggteOptions::default()
        })
        .unwrap();
    let alone = dynamic.combine(&[(0, 1.0)]).unwrap();
    let point = dynamic.point(0).unwrap().estimate;
    assert_eq!(alone.estimate.to_bits(), point.estimate.to_bits());
    assert_eq!(alone.se, point.se);
    let window = dynamic.combine(&[(1, 0.5), (2, 0.5)]).unwrap();
    let mean = 0.5 * dynamic.point(1).unwrap().estimate.estimate
        + 0.5 * dynamic.point(2).unwrap().estimate.estimate;
    assert_eq!(window.estimate.to_bits(), mean.to_bits());
    assert!(matches!(
        dynamic.combine(&[(7, 1.0)]),
        Err(Error::Input(InputError::Domain { what: "key", .. }))
    ));

    // The event study's covariance has the points' variances on its diagonal.
    let study = dynamic.event_study().unwrap();
    assert_eq!(study.relative_periods(), [-2, 0, 1, 2]);
    assert_eq!(study.reference(), -1);
    for (j, point) in dynamic.points.iter().enumerate() {
        let variance = study.covariance().get(j, j).unwrap();
        assert_eq!(Some(variance.sqrt()), point.estimate.se);
    }
    let varying = staggered(0.3).fit(true, &AttGtOptions::default()).unwrap();
    assert!(matches!(
        varying.event_study(&EventStudyOptions::default()),
        Err(Error::Unsupported { .. })
    ));
}

#[test]
fn singleton_clusters_reproduce_the_cells_own_standard_errors() {
    let data = staggered(0.3);
    let plain = data.fit(false, &universal()).unwrap();
    let clustered = data.fit(true, &universal()).unwrap();
    for (a, b) in plain.cells().iter().zip(clustered.cells()) {
        let (a, b) = (
            a.estimate.unwrap().se.unwrap(),
            b.estimate.unwrap().se.unwrap(),
        );
        // `sqrt(sum psi^2) / n` against the centred sum of the same thirty
        // values, whose mean is zero up to rounding: each sum carries at most
        // 30 roundings of relative size 2^-53, about 4e-15.
        assert!((a - b).abs() <= 1e-12 * a, "{a} against {b}");
    }
    assert_eq!(
        clustered.clusters(),
        Some(&(0..30).collect::<Vec<u32>>()[..])
    );
}

#[test]
fn the_bands_and_the_pretest_read_the_estimated_cells() {
    let fit = staggered(0.3).fit(false, &universal()).unwrap();
    let bands = fit
        .bands(Level::default(), Bootstrap::new(199, Seed(20_261_127)))
        .unwrap();
    assert_eq!(bands.cells.len(), 6);
    assert!(bands.critical_value.is_finite() && bands.critical_value > 0.0);
    for band in &bands.cells {
        let estimate = band.estimate;
        assert!(estimate.lower().unwrap() <= estimate.estimate);
        assert!(estimate.estimate <= estimate.upper().unwrap());
        assert!(estimate.interval.unwrap().coverage.is_simultaneous());
        assert_eq!(estimate.p_value, None);
    }
    let test = fit.pretest().unwrap();
    assert_eq!((test.df, test.cells.as_slice()), (1, &[(3, 1)][..]));
    assert!(test.p_value > 0.0 && test.p_value <= 1.0);
}

#[test]
fn without_the_influence_function_the_aggregations_refuse() {
    let options = AttGtOptions {
        keep_influence: false,
        ..universal()
    };
    let fit = staggered(0.3).fit(true, &options).unwrap();
    assert!(fit.influence().is_none());
    assert!(fit.cells().iter().all(|cell| cell.estimate.is_some()));
    let kept = staggered(0.3).fit(true, &universal()).unwrap();
    assert_eq!(fit.cells(), kept.cells());
    assert_eq!(
        fit.aggregate(&AggteOptions::default()).unwrap_err(),
        Error::NoInfluence
    );
    assert_eq!(fit.pretest().unwrap_err(), Error::NoInfluence);
    assert_eq!(
        fit.bands(Level::default(), Bootstrap::new(9, Seed(1)))
            .unwrap_err(),
        Error::NoInfluence
    );
    assert_eq!(fit.pretest().unwrap_err().code(), "did.no_influence");
}

/// Every unit is eventually treated (cohorts 2, 3 and 4). `did` then drops the
/// periods from the last cohort's start on and lets that cohort serve as the
/// not-yet-treated comparison without cells of its own, so four cells remain:
/// (2, 2), (2, 3), (3, 1) and (3, 3). A not-yet-treated comparator must be
/// untreated at the later of a cell's two periods.
#[test]
fn not_yet_treated_comparators_on_a_repeated_cross_section() {
    let mut data = Data::default();
    let mut unit = 0;
    for _ in 0..40 {
        for period in 1..=4_i64 {
            for x in [0.0, 1.0] {
                let base = 0.4 * x + 10.0 + period as f64;
                for (cohort, effect) in [(2, 2.0), (3, 1.0), (4, 0.0)] {
                    let y = base + if period >= cohort { effect } else { 0.0 };
                    data.push(unit, Some(cohort), period, y, &[x]);
                    unit += 1;
                }
            }
        }
    }
    let fit = data
        .fit_design(
            Design::RepeatedCrossSection,
            false,
            &AttGtOptions {
                comparison_group: ComparisonGroup::NotYetTreated,
                ..universal()
            },
        )
        .unwrap();
    let cells: Vec<(i64, i64)> = fit.cells().iter().map(|c| (c.group, c.time)).collect();
    assert_eq!(cells, [(2, 2), (2, 3), (3, 1), (3, 3)]);
    for (group, time, want) in [(2, 2, 2.0), (2, 3, 2.0), (3, 1, 0.0), (3, 3, 1.0)] {
        assert!((att(&fit, group, time) - want).abs() < EXACT);
    }
    let sample = fit.sample();
    assert_eq!(sample.last_cohort_as_comparison, Some(4));
    assert_eq!(sample.dropped_late_rows, 40 * 2 * 3);
    // On a repeated cross-section the positions are rows.
    assert_eq!(sample.units.len(), 40 * 3 * 2 * 3);
}

/// Eight units treated at period 2 and eight never treated, over periods 1
/// to 3, with the treated units unobserved at period 3. The cell `(2, 3)`
/// has no treated unit: it is reported with its reason, and an error when
/// failed cells are not skipped.
#[test]
fn a_cell_without_treated_units_is_reported_or_refused() {
    let mut data = Data::default();
    for unit in 0..16_u32 {
        let cohort = (unit < 8).then_some(2);
        for period in 1..=3_i64 {
            if cohort.is_some() && period == 3 {
                continue;
            }
            let effect = if cohort.is_some() && period >= 2 {
                5.0
            } else {
                0.0
            };
            let y = 10.0 + period as f64 + f64::from(unit % 4) * 0.5 + effect;
            data.push(unit, cohort, period, y, &[]);
        }
    }
    let options = AttGtOptions {
        unbalanced: UnbalancedPanel::PerCell,
        ..AttGtOptions::default()
    };
    let fit = data.fit(false, &options).unwrap();
    let cells: Vec<_> = fit
        .cells()
        .iter()
        .map(|c| (c.group, c.time, c.base, c.skipped))
        .collect();
    assert_eq!(
        cells,
        [(2, 2, 1, None), (2, 3, 1, Some(SkipReason::NoTreated))]
    );
    assert_eq!(fit.influence().unwrap().n_columns(), 1);
    let strict = AttGtOptions {
        skip_failed_cells: false,
        ..options
    };
    assert_eq!(
        data.fit(false, &strict).unwrap_err(),
        Error::CellFailed {
            group: 2,
            time: 3,
            base: 1,
            reason: SkipReason::NoTreated
        }
    );
}

/// Twenty units treated at period 2 and twenty never treated over periods 1
/// to 4, one continuous covariate, and a rare indicator held by one treated
/// unit and one comparator, the comparator observed at periods 1 and 2 only.
/// In cells (2, 3) and (2, 4) no comparator holds the indicator, so the cell
/// is fitted without it, to the last bit as without the column.
#[test]
fn a_covariate_no_comparator_in_the_cell_holds_is_dropped_for_that_cell() {
    let panel = |with_indicator: bool| {
        let mut data = Data::default();
        for unit in 0..40_u32 {
            let cohort = (unit < 20).then_some(2);
            let holder = unit == 1 || unit == 21;
            let x = f64::from(unit % 7) * 0.3 - 1.0;
            for period in 1..=4_i64 {
                if unit == 21 && period > 2 {
                    continue;
                }
                let effect = if cohort.is_some() && period >= 2 {
                    3.0
                } else {
                    0.0
                };
                let y = 5.0 + period as f64 * 0.5 + 1.2 * x + f64::from(unit % 3) * 0.25 + effect;
                let row = [x, f64::from(u8::from(holder))];
                data.push(
                    unit,
                    cohort,
                    period,
                    y,
                    &row[..=usize::from(with_indicator)],
                );
            }
        }
        data
    };
    // The indicator separates the arms in the cells it is dropped from, which
    // `did`'s overlap check would refuse; the test is of the pruning.
    let options = AttGtOptions {
        unbalanced: UnbalancedPanel::PerCell,
        refuse_poor_overlap: false,
        ..AttGtOptions::default()
    };
    let full = panel(true).fit(false, &options).unwrap();
    let reduced = panel(false).fit(false, &options).unwrap();
    let estimated: Vec<_> = full
        .cells()
        .iter()
        .filter(|c| c.estimate.is_some())
        .map(|c| (c.group, c.time, c.columns_dropped))
        .collect();
    assert_eq!(estimated, [(2, 2, 0), (2, 3, 1), (2, 4, 1)]);
    let both = full.cells().iter().zip(reduced.cells());
    for (with, without) in both.filter(|(c, _)| c.columns_dropped > 0) {
        assert_eq!(with.estimate, without.estimate);
    }
    // The cell that kept the indicator is a different fit.
    assert_ne!(full.cells()[0].estimate, reduced.cells()[0].estimate);
}

/// Three matched sets, indexed at periods 3, 4 and 5, each with treated units
/// and their own never-treated comparators, observed from two periods before
/// the index to three after. Every unit of a set follows the set's own
/// trend, and the treated gain 1 from their index on. Within a set the
/// trends cancel, so ATT(g,t) is 1 after the index and 0 before it; pooling
/// every comparator mixes three trends and reports something else.
#[test]
fn matched_comparators_recover_the_effect_where_the_pool_reports_the_trends() {
    let mut data = Data::default();
    let mut unit = 0_u32;
    for index in 3..=5_i64 {
        let slope = index as f64;
        for member in 0..8 {
            for treated in [true, false] {
                let x = f64::from(member % 3);
                for period in (index - 2)..=(index + 3) {
                    let effect = if treated && period >= index { 1.0 } else { 0.0 };
                    let y = 2.0 * x + slope * period as f64 + effect;
                    data.push(unit, treated.then_some(index), period, y, &[x]);
                    *data.matched.last_mut().unwrap() = (!treated).then_some(index);
                }
                unit += 1;
            }
        }
    }
    let options = |comparison_group| AttGtOptions {
        comparison_group,
        unbalanced: UnbalancedPanel::PerCell,
        ..universal()
    };
    let matched = data
        .fit(false, &options(ComparisonGroup::MatchedNeverTreated))
        .unwrap();
    assert!(
        matched
            .cells()
            .iter()
            .all(|c| c.skipped.is_none_or(|r| r == SkipReason::NoTreated))
    );
    let estimated: Vec<_> = matched
        .cells()
        .iter()
        .filter(|c| c.estimate.is_some())
        .collect();
    assert_eq!(estimated.len(), 15);
    for cell in estimated {
        let want = if cell.time >= cell.group { 1.0 } else { 0.0 };
        assert!((cell.estimate.unwrap().estimate - want).abs() < 1e-9);
        // Eight comparators per cell: the set's own and nobody else's.
        assert_eq!(cell.n_control, 8);
    }
    let pooled = data
        .fit(false, &options(ComparisonGroup::NeverTreated))
        .unwrap();
    assert!(pooled.cells().iter().any(|cell| {
        cell.estimate.is_some_and(|e| {
            let want = if cell.time >= cell.group { 1.0 } else { 0.0 };
            (e.estimate - want).abs() > 0.1
        })
    }));
}

/// Forty units over periods 1 to 4, half first treated in period 3 with an
/// effect of 2. The efficient estimator combines both pre-treatment
/// baselines with weights that sum to one.
#[test]
fn the_efficient_estimator_combines_every_baseline() {
    let mut data = Data::default();
    for unit in 0..40_u32 {
        let cohort = (unit < 20).then_some(3);
        for period in 1..=4_i64 {
            let effect = if unit < 20 && period >= 3 { 2.0 } else { 0.0 };
            data.push(
                unit,
                cohort,
                period,
                f64::from(unit) + period as f64 + effect,
                &[],
            );
        }
    }
    let options = AttGtOptions {
        estimator: Estimator::Efficient,
        ..AttGtOptions::default()
    };
    let fit = data.fit(false, &options).unwrap();
    let cell = fit.cells().iter().find(|c| c.time == 4).unwrap();
    assert!((cell.estimate.unwrap().estimate - 2.0).abs() < EXACT);
    let weights = cell.efficient.as_ref().unwrap();
    let sum: f64 = weights.baselines.iter().map(|b| b.weight).sum();
    assert!((sum - 1.0).abs() < EXACT);
    assert_eq!(weights.baselines.len(), 2);
    assert_eq!(
        (
            weights.baselines[0].n_treated,
            weights.baselines[0].n_control
        ),
        (20, 20)
    );
    assert_eq!(weights.covariance.dim(), 2);
    assert_eq!(weights.precision_solution.len(), 2);
    // Its cells aggregate as any other.
    let dynamic = fit
        .aggregate(&AggteOptions {
            aggregation: AggteType::Dynamic,
            ..AggteOptions::default()
        })
        .unwrap();
    assert!((dynamic.overall.unwrap().estimate.estimate - 2.0).abs() < EXACT);

    let covariates = Data {
        covariates: vec![vec![0.0; data.unit.len()]],
        ..data
    };
    assert!(matches!(
        covariates.fit(false, &options),
        Err(Error::Unsupported { .. })
    ));
}

/// Every treated unit at 10 and every comparator at 20, plus the effect: the
/// baselines' influence functions are all zero, and the weights are equal.
#[test]
fn the_efficient_estimator_weights_degenerate_baselines_equally() {
    let mut data = Data::default();
    for unit in 0..12_u32 {
        let cohort = (unit < 6).then_some(3);
        for period in 1..=4_i64 {
            let level = if unit < 6 { 10.0 } else { 20.0 };
            let effect = if unit < 6 && period >= 3 { 3.0 } else { 0.0 };
            data.push(unit, cohort, period, level + effect, &[]);
        }
    }
    let options = AttGtOptions {
        estimator: Estimator::Efficient,
        ..AttGtOptions::default()
    };
    let fit = data.fit(false, &options).unwrap();
    let cell = fit.cells().iter().find(|c| c.time == 4).unwrap();
    let weights = cell.efficient.as_ref().unwrap();
    assert_eq!(weights.ridge, None);
    assert!(
        weights
            .baselines
            .iter()
            .all(|b| (b.weight - 0.5).abs() < EXACT)
    );
}

#[test]
fn invalid_input_and_options_are_refused() {
    let data = staggered(0.0);
    let mut varying = staggered(0.0);
    varying.weight[1] = 2.0;
    assert!(matches!(
        varying.fit(false, &universal()),
        Err(Error::Input(InputError::NotConstant {
            what: "weights",
            unit: 0
        }))
    ));
    // A repeated cross-section reads each row on its own.
    let mut rows = Data::default();
    for (i, (&cohort, &y)) in data.cohort.iter().zip(&data.outcome).enumerate() {
        rows.push(u32::try_from(i).unwrap(), cohort, data.period[i], y, &[]);
    }
    rows.weight[1] = 2.0;
    assert!(
        rows.fit_design(Design::RepeatedCrossSection, false, &universal())
            .is_ok()
    );
    let mut zero = staggered(0.0);
    zero.weight.iter_mut().take(4).for_each(|w| *w = 0.0);
    assert_eq!(
        zero.fit(false, &universal()).unwrap_err().code(),
        "input.domain"
    );
    for options in [
        AttGtOptions {
            anticipation: -1,
            ..universal()
        },
        AttGtOptions {
            nuisance: NuisanceOptions {
                propensity_clip: 0.5,
                ..NuisanceOptions::default()
            },
            ..universal()
        },
        AttGtOptions {
            bootstrap: Some(Bootstrap::new(0, Seed(1))),
            ..universal()
        },
    ] {
        assert_eq!(
            data.fit(false, &options).unwrap_err().code(),
            "input.domain"
        );
    }
    for (clusters, options) in [
        (
            false,
            AttGtOptions {
                estimator: Estimator::TraditionalDoublyRobust,
                ..universal()
            },
        ),
        (
            true,
            AttGtOptions {
                bootstrap: Some(Bootstrap::new(9, Seed(1))),
                ..universal()
            },
        ),
    ] {
        assert_eq!(
            data.fit(clusters, &options).unwrap_err().code(),
            "did.unsupported"
        );
    }
    let fit = data.fit(false, &universal()).unwrap();
    #[expect(
        deprecated,
        reason = "the deprecated band without its bootstrap is refused"
    )]
    let uniform = AggteOptions {
        band: Band::Uniform,
        ..AggteOptions::default()
    };
    assert_eq!(
        fit.aggregate(&uniform).unwrap_err().code(),
        "did.unsupported"
    );
}

#[test]
fn bad_columns_are_input_errors() {
    use epistat_testkit::{BadInput, assert_rejects_bad_input};
    let data = staggered(0.3);
    let lpop = data.covariates.concat();
    assert_rejects_bad_input("att_gt::estimate", |bad| {
        if bad == BadInput::Unsorted {
            return None;
        }
        let y = bad.column(&data.outcome);
        let x = MatRef::from_column_major_slice(&lpop, data.unit.len(), 1);
        let built = StaggeredPanel::builder(&data.unit, &data.period, &data.cohort, &y)
            .covariates(x)
            .build()
            .and_then(|panel| estimate(&panel, &universal()));
        Some(built.map(|fit| fit.cells().len()))
    });
}

/// Repeated cross-sections built with `StaggeredPanel::cross_sections` give
/// the bits of the deprecated `design` option on invented unit codes, and the
/// rows need no order.
#[test]
fn cross_sections_match_the_deprecated_design_option() {
    let data = staggered(0.3);
    let mut rows = Data::default();
    for (i, (&cohort, &y)) in data.cohort.iter().zip(&data.outcome).enumerate() {
        let x = data.covariates[0][i];
        rows.push(u32::try_from(i).unwrap(), cohort, data.period[i], y, &[x]);
    }
    let options = AttGtOptions {
        comparison_group: ComparisonGroup::NotYetTreated,
        ..universal()
    };
    let new = rows
        .fit_design(Design::RepeatedCrossSection, true, &options)
        .unwrap();
    #[expect(
        deprecated,
        reason = "the deprecated option is compared with its replacement"
    )]
    let legacy = AttGtOptions {
        design: Design::RepeatedCrossSection,
        ..options
    };
    let old = rows.fit(true, &legacy).unwrap();
    assert_eq!(new.cells(), old.cells());
    assert_eq!(new.influence(), old.influence());
    assert_eq!(new.sample(), old.sample());

    // The estimators that follow units refuse repeated cross-sections.
    let y = [1.0, 2.0];
    let view = StaggeredPanel::cross_sections(&[1, 2], &[None, Some(2)], &y)
        .build()
        .unwrap();
    assert_eq!(view.design(), Design::RepeatedCrossSection);
    assert_eq!(view.n_units(), 2);
    assert_eq!(
        crate::twfe::fit(&view, &crate::twfe::TwfeOptions::default())
            .unwrap_err()
            .code(),
        "did.unsupported"
    );
}

/// `Inference` holds the bootstrap and the band, gives the bits of the
/// deprecated pair, and marks a uniform band simultaneous without p-values.
#[test]
fn inference_replaces_the_bootstrap_and_band_fields() {
    let fit = staggered(0.3).fit(false, &universal()).unwrap();
    let draws = Bootstrap::new(299, Seed(20_261_128));
    let dynamic = AggteOptions {
        aggregation: AggteType::Dynamic,
        ..AggteOptions::default()
    };
    let new = fit
        .aggregate(&AggteOptions {
            inference: Inference::UniformBand(draws),
            ..dynamic
        })
        .unwrap();
    #[expect(
        deprecated,
        reason = "the deprecated pair is compared with its replacement"
    )]
    let legacy = AggteOptions {
        bootstrap: Some(draws),
        band: Band::Uniform,
        ..dynamic
    };
    let old = fit.aggregate(&legacy).unwrap();
    assert!(new.uniform_band);
    assert_eq!(new.critical_value.to_bits(), old.critical_value.to_bits());
    for (a, b) in new.points.iter().zip(&old.points) {
        assert_eq!(a.estimate, b.estimate);
        let interval = a.estimate.interval.unwrap();
        assert_eq!(interval.coverage.critical(), Some(new.critical_value));
        assert_eq!(a.estimate.p_value, None);
    }
    let overall = new.overall.unwrap().estimate;
    assert!(!overall.interval.unwrap().coverage.is_simultaneous());
    assert!(overall.p_value.is_some());

    #[expect(deprecated, reason = "both forms at once are refused")]
    let both = AggteOptions {
        inference: Inference::Bootstrap(draws),
        bootstrap: Some(draws),
        ..dynamic
    };
    assert_eq!(fit.aggregate(&both).unwrap_err().code(), "did.unsupported");

    let pointwise = fit
        .aggregate(&AggteOptions {
            inference: Inference::Bootstrap(draws),
            ..dynamic
        })
        .unwrap();
    assert!(!pointwise.uniform_band);
    assert!(
        pointwise
            .points
            .iter()
            .all(|p| p.estimate.p_value.is_some())
    );
}

/// The event study of a fit under the default varying base names the option
/// to set, before any aggregation.
#[test]
fn the_event_study_names_the_base_period_to_set() {
    let varying = staggered(0.3).fit(false, &AttGtOptions::default()).unwrap();
    let error = varying
        .event_study(&EventStudyOptions::default())
        .unwrap_err();
    assert!(
        error.to_string().contains("BasePeriod::Universal"),
        "{error}"
    );
    let universal = staggered(0.3).fit(false, &universal()).unwrap();
    let study = universal
        .event_study(&EventStudyOptions {
            min_e: Some(-1),
            ..EventStudyOptions::default()
        })
        .unwrap();
    assert!(study.relative_periods().iter().all(|&e| e >= -1));
}
