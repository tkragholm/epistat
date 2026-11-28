//! The public entry points against the R references of `HonestDiD` 0.2.8
//! and `pretrends` 0.1.0, and their refusals of input they cannot use.
//!
//! The engine behind them is checked case by case in the crate's own tests
//! (`src/tests`); these check that the entry points reach it with the
//! arguments R was given. Tolerances, derived:
//!
//! - the Wald interval of [`honest::original`] is a closed form of a few
//!   operations on numbers R wrote with 15 significant digits, so it agrees
//!   to `1e-8` on intervals of size `O(0.1)` with a wide margin;
//! - identified sets are the optima of linear programs that Clarabel and the
//!   dense simplex solve to `1e-8`; a hundred times that, relative to the
//!   bound, `1e-6 (1 + |bound|)`, covers the solver tolerance and R's own;
//! - the conditional confidence sets are found on R's grid of 1,000 points
//!   over the identified set widened by the interval's half-width, so each
//!   end is off by at most one grid step, below `5e-3` on these sets;
//! - a pre-trend probability is a randomly shifted lattice estimate with the
//!   standard error `e` it reports, and the tight reference has its own
//!   `e_ref`: the difference of two independent unbiased estimates has
//!   standard error at most `e + e_ref`, and five of those bound it.

use epistat_core::{Bootstrap, Classify, Covariance, ErrorKind, EventStudy, Level, Seed};
use epistat_honestdid::Error;
use epistat_honestdid::honest::{
    self, Calibration, Directions, Grid, HonestOptions, JointMethod, JointPathOptions, Method,
    Restriction, RmBound, Target,
};
use epistat_honestdid::pretrends::{self, Critical, PretrendOptions, Violation};
use epistat_testkit::{
    BadInput, Tolerance, absent, assert_rejects_bad_input_with_codes, load_json,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct HonestRef {
    betahat: Vec<f64>,
    sigma: Vec<Vec<f64>>,
    pre_indices: Vec<usize>,
    post_indices: Vec<usize>,
}

#[derive(Deserialize)]
struct Bounds {
    lb: f64,
    ub: f64,
}

#[derive(Deserialize)]
struct RmRow {
    lb: f64,
    ub: f64,
    id_lb: f64,
    id_ub: f64,
    #[serde(rename = "Delta")]
    delta: String,
    #[serde(rename = "Mbar")]
    mbar: f64,
}

#[derive(Deserialize)]
struct RmFixture {
    l_vec: Vec<f64>,
    original: Bounds,
    relative_magnitude: Vec<RmRow>,
}

/// An event study from coefficients and a covariance given as rows, with
/// `n_pre` coefficients before the reference period -1.
fn study(betahat: Vec<f64>, sigma: &[Vec<f64>], n_pre: usize) -> EventStudy {
    let k = betahat.len();
    let values: Vec<f64> = sigma.iter().flatten().copied().collect();
    let n_pre = i64::try_from(n_pre).expect("small");
    let periods = (0..i64::try_from(k).expect("small"))
        .map(|i| if i < n_pre { i - n_pre - 1 } else { i - n_pre })
        .collect();
    EventStudy::new(
        betahat,
        Covariance::from_values(k, &values).expect("covariance"),
        periods,
        -1,
    )
    .expect("event study")
}

fn bc_study() -> EventStudy {
    let r: HonestRef = load_json("tests/data/honest_did_ref.json");
    assert_eq!(r.betahat.len(), r.pre_indices.len() + r.post_indices.len());
    study(r.betahat, &r.sigma, r.pre_indices.len())
}

#[test]
fn relative_magnitude_matches_the_package_on_the_average_effect() {
    let study = bc_study();
    let r: RmFixture = load_json("tests/data/honest_did_rm_avg_ref.json");
    let target = Target::Weights(r.l_vec.clone());

    let original = honest::original(&study, &target, Level::default()).expect("original");
    assert!(
        Tolerance::exact()
            .abs(1e-8)
            .accepts(absent(original.lower()), r.original.lb)
    );
    assert!(
        Tolerance::exact()
            .abs(1e-8)
            .accepts(absent(original.upper()), r.original.ub)
    );
    let averaged = honest::original(
        &study,
        &Target::Average { first: 0, last: 3 },
        Level::default(),
    )
    .expect("original of the average");
    assert_eq!(averaged, original);

    let restriction = Restriction::RelativeMagnitude {
        mbar: Grid::Values(r.relative_magnitude.iter().map(|row| row.mbar).collect()),
        bound: RmBound::ParallelTrends,
    };
    let summary = honest::summarize(
        &study,
        &target,
        &restriction,
        &HonestOptions {
            least_favorable: Some(Bootstrap::new(1_000, Seed(0))),
            ..HonestOptions::default()
        },
    )
    .expect("summary");
    let identified = honest::summarize(
        &study,
        &target,
        &restriction,
        &HonestOptions {
            method: Some(Method::IdentifiedSet),
            least_favorable: Some(Bootstrap::new(1_000, Seed(0))),
            ..HonestOptions::default()
        },
    )
    .expect("identified sets");
    assert_eq!(summary.rows.len(), r.relative_magnitude.len());
    for ((row, set), reference) in summary
        .rows
        .iter()
        .zip(&identified.rows)
        .zip(&r.relative_magnitude)
    {
        assert_eq!(row.delta.label(), reference.delta);
        assert_eq!(row.method, Method::ConditionalLeastFavorable);
        assert_eq!(row.method.label(), "C-LF");
        assert!(
            Tolerance::exact()
                .abs(5e-3)
                .accepts(absent(row.interval.and_then(|i| i.lower)), reference.lb),
            "Mbar {}",
            reference.mbar
        );
        assert!(
            Tolerance::exact()
                .abs(5e-3)
                .accepts(absent(row.interval.and_then(|i| i.upper)), reference.ub),
            "Mbar {}",
            reference.mbar
        );
        let lp = 1e-6 * (1.0 + reference.id_lb.abs().max(reference.id_ub.abs()));
        let id = row.identified.expect("identified set");
        assert!(
            Tolerance::exact()
                .abs(lp)
                .accepts(absent(id.lower), reference.id_lb)
        );
        assert!(
            Tolerance::exact()
                .abs(lp)
                .accepts(absent(id.upper), reference.id_ub)
        );
        assert_eq!(set.identified, row.identified);
        assert_eq!(set.interval, None);
    }
    // The intervals widen with Mbar, so the breakdown value is the last Mbar
    // whose interval excludes zero.
    let excluding = summary
        .rows
        .iter()
        .take_while(|row| row.interval.and_then(|i| i.lower).is_some_and(|l| l > 0.0))
        .last()
        .map(|row| row.value);
    assert_eq!(summary.breakdown().map(|b| b.value), excluding);
}

#[derive(Deserialize)]
struct SmoothnessRef {
    datasets: Vec<SmoothnessDataset>,
}

#[derive(Deserialize)]
struct SmoothnessDataset {
    name: String,
    betahat: Vec<f64>,
    sigma: Vec<Vec<f64>>,
    num_pre: usize,
    cases: Vec<SmoothnessCase>,
}

#[derive(Deserialize)]
struct SmoothnessCase {
    l_vec: Vec<f64>,
    m: f64,
    identified: Identified,
}

#[derive(Deserialize)]
struct Identified {
    feasible: bool,
    lb: f64,
    ub: f64,
}

#[test]
fn smoothness_identified_sets_match_the_package() {
    let r: SmoothnessRef = load_json("tests/data/honest_did_smoothness_ref.json");
    let mut checked = 0;
    for dataset in &r.datasets {
        let study = study(dataset.betahat.clone(), &dataset.sigma, dataset.num_pre);
        for case in &dataset.cases {
            let summary = honest::summarize(
                &study,
                &Target::Weights(case.l_vec.clone()),
                &Restriction::Smoothness {
                    m: Grid::Values(vec![case.m]),
                },
                &HonestOptions {
                    method: Some(Method::IdentifiedSet),
                    least_favorable: Some(Bootstrap::new(1_000, Seed(0))),
                    ..HonestOptions::default()
                },
            )
            .expect("identified set");
            let set = summary.rows[0].identified.expect("identified set");
            if case.identified.feasible {
                let lp = 1e-6 * (1.0 + case.identified.lb.abs().max(case.identified.ub.abs()));
                assert!(
                    Tolerance::exact()
                        .abs(lp)
                        .accepts(absent(set.lower), case.identified.lb),
                    "{}",
                    dataset.name
                );
                assert!(
                    Tolerance::exact()
                        .abs(lp)
                        .accepts(absent(set.upper), case.identified.ub),
                    "{}",
                    dataset.name
                );
            } else {
                assert_eq!((set.lower, set.upper), (None, None), "{}", dataset.name);
            }
            checked += 1;
        }
    }
    assert!(checked > 0);
}

#[derive(Deserialize)]
struct PretrendRef {
    alpha: f64,
    cases: Vec<PretrendCase>,
}

#[derive(Deserialize)]
struct PretrendCase {
    name: String,
    betahat: Vec<f64>,
    sigma: Vec<Vec<f64>>,
    tvec: Vec<f64>,
    reference: f64,
    delta: Vec<f64>,
    tight: Tight,
}

#[derive(Deserialize)]
struct Tight {
    pass: f64,
    pass_error: f64,
    mean_post: Vec<f64>,
    mean_post_error: Vec<f64>,
}

#[test]
fn pretrend_power_matches_the_tight_reference() {
    let r: PretrendRef = load_json("tests/data/pretrends_ref.json");
    let case = r
        .cases
        .iter()
        .find(|c| c.name == "hewang_linear50")
        .expect("case");
    let k = case.betahat.len();
    let values: Vec<f64> = case.sigma.iter().flatten().copied().collect();
    #[expect(clippy::cast_possible_truncation, reason = "integer relative periods")]
    let periods = case.tvec.iter().map(|&t| t as i64).collect();
    #[expect(clippy::cast_possible_truncation, reason = "an integer reference")]
    let reference = case.reference as i64;
    let study = EventStudy::new(
        case.betahat.clone(),
        Covariance::from_values(k, &values).expect("covariance"),
        periods,
        reference,
    )
    .expect("event study");
    let options = PretrendOptions {
        critical: Critical::Level(Level::alpha(r.alpha).expect("level")),
        tolerance: 1e-6,
        ..PretrendOptions::new(
            Violation::Delta(case.delta.clone()),
            Bootstrap::new(12, Seed(20_260_125)),
        )
    };
    let analysis = pretrends::power(&study, &options).expect("power");
    let tol = 5.0 * (analysis.pass_probability.error + case.tight.pass_error);
    assert!((analysis.pass_probability.value - case.tight.pass).abs() <= tol);
    for (i, m) in analysis.mean_post_after_pretest.iter().enumerate() {
        let tol = 5.0 * (m.error + case.tight.mean_post_error[i]);
        assert!(
            (m.value - case.tight.mean_post[i]).abs() <= tol,
            "post mean {i}"
        );
    }
    assert_eq!(analysis.delta, case.delta);

    // The slope with half the power, then the power at that slope.
    let at_half = pretrends::power(
        &study,
        &PretrendOptions {
            violation: Violation::Power(0.5),
            ..options.clone()
        },
    )
    .expect("slope for power");
    let slope = at_half.slope.expect("a slope").value;
    assert!(slope > 0.0);
    assert!((at_half.power.value - 0.5).abs() <= 5.0 * at_half.power.error + 1e-9);
    let linear = pretrends::power(
        &study,
        &PretrendOptions {
            violation: Violation::LinearTrend { slope },
            ..options
        },
    )
    .expect("linear trend");
    assert_eq!(linear.delta, at_half.delta);
    assert_eq!(linear.power, at_half.power);
}

fn small_study() -> EventStudy {
    let covariance: Vec<f64> = (0..25)
        .map(|i| if i / 5 == i % 5 { 0.04 } else { 0.005 })
        .collect();
    EventStudy::new(
        vec![0.02, -0.01, 0.03, 0.5, 0.7],
        Covariance::from_values(5, &covariance).expect("covariance"),
        vec![-4, -3, -2, 0, 1],
        -1,
    )
    .expect("event study")
}

#[test]
fn joint_regions_cover_every_period_and_calibrate_below_bonferroni() {
    let study = small_study();
    let restriction = Restriction::RelativeMagnitude {
        mbar: Grid::Values(vec![0.5]),
        bound: RmBound::ParallelTrends,
    };
    let bonferroni = honest::joint_path(
        &study,
        &restriction,
        &JointPathOptions {
            least_favorable: Some(Bootstrap::new(1_000, Seed(0))),
            ..JointPathOptions::default()
        },
    )
    .expect("path");
    let gaussian = honest::joint_path(
        &study,
        &restriction,
        &JointPathOptions {
            calibration: Calibration::Gaussian(Bootstrap::new(5_000, Seed(7))),
            least_favorable: Some(Bootstrap::new(1_000, Seed(0))),
            ..JointPathOptions::default()
        },
    )
    .expect("calibrated path");
    assert_eq!(bonferroni.points.len(), 2);
    assert_eq!(bonferroni.points[1].period, Some(1));
    assert_eq!(bonferroni.critical_value, None);
    assert!(gaussian.critical_value.is_some());
    // Two intervals at 1 - 0.05 / 2 each; the simulated level is lower.
    let pointwise = |region: &honest::JointPathRegion| region.pointwise_level.confidence_value();
    assert!((pointwise(&bonferroni) - 0.975).abs() < 1e-12);
    assert!(pointwise(&gaussian) <= pointwise(&bonferroni));

    let flci = honest::joint_path(
        &study,
        &Restriction::Smoothness {
            m: Grid::Values(vec![0.05]),
        },
        &JointPathOptions {
            method: JointMethod::Flci,
            directions: Directions::Weights(vec![vec![1.0, 0.0], vec![0.5, 0.5]]),
            least_favorable: Some(Bootstrap::new(1_000, Seed(0))),
            ..JointPathOptions::default()
        },
    )
    .expect("joint FLCI");
    assert_eq!(flci.points.len(), 2);
    assert_eq!(flci.points[0].period, None);
    for point in &flci.points {
        let (lower, upper) = (
            point.robust.lower.expect("lower"),
            point.robust.upper.expect("upper"),
        );
        assert!(lower < upper);
        assert!(point.original.se.is_some());
    }
}

#[test]
fn unusable_input_is_refused_with_a_code() {
    let study = small_study();
    let rm = |mbar: Vec<f64>| Restriction::RelativeMagnitude {
        mbar: Grid::Values(mbar),
        bound: RmBound::ParallelTrends,
    };
    let options = HonestOptions {
        least_favorable: Some(Bootstrap::new(1_000, Seed(0))),
        ..HonestOptions::default()
    };
    let cases: Vec<(Error, &str, ErrorKind)> = vec![
        (
            honest::summarize(
                &study,
                &Target::Weights(vec![1.0]),
                &rm(vec![1.0]),
                &options,
            )
            .unwrap_err(),
            "input.length",
            ErrorKind::Input,
        ),
        (
            honest::summarize(
                &study,
                &Target::Weights(vec![0.0, 0.0]),
                &rm(vec![1.0]),
                &options,
            )
            .unwrap_err(),
            "input.domain",
            ErrorKind::Input,
        ),
        (
            honest::summarize(&study, &Target::Period(5), &rm(vec![1.0]), &options).unwrap_err(),
            "input.domain",
            ErrorKind::Input,
        ),
        (
            honest::summarize(&study, &Target::Period(0), &rm(vec![-1.0]), &options).unwrap_err(),
            "input.domain",
            ErrorKind::Input,
        ),
        (
            honest::summarize(
                &study,
                &Target::Period(0),
                &rm(vec![1.0]),
                &HonestOptions {
                    method: Some(Method::Flci),
                    ..options
                },
            )
            .unwrap_err(),
            "honestdid.unsupported",
            ErrorKind::Unsupported,
        ),
        (
            honest::summarize(
                &study,
                &Target::Period(0),
                &Restriction::Smoothness {
                    m: Grid::Values(vec![0.1]),
                },
                &HonestOptions {
                    bias: Some(honest::BiasDirection::Positive),
                    monotonicity: Some(honest::Monotonicity::Increasing),
                    ..options
                },
            )
            .unwrap_err(),
            "honestdid.unsupported",
            ErrorKind::Unsupported,
        ),
        (
            honest::joint_path(
                &study,
                &rm(vec![0.5, 1.0]),
                &JointPathOptions {
                    least_favorable: Some(Bootstrap::new(1_000, Seed(0))),
                    ..JointPathOptions::default()
                },
            )
            .unwrap_err(),
            "input.length",
            ErrorKind::Input,
        ),
        (
            honest::joint_path(
                &study,
                &Restriction::RelativeMagnitude {
                    mbar: Grid::Values(vec![1.0]),
                    bound: RmBound::LinearTrend,
                },
                &JointPathOptions {
                    least_favorable: Some(Bootstrap::new(1_000, Seed(0))),
                    ..JointPathOptions::default()
                },
            )
            .unwrap_err(),
            "honestdid.unsupported",
            ErrorKind::Unsupported,
        ),
        (
            pretrends::power(
                &study,
                &PretrendOptions::new(Violation::Power(1.5), Bootstrap::new(12, Seed(1))),
            )
            .unwrap_err(),
            "input.domain",
            ErrorKind::Input,
        ),
        (
            pretrends::power(
                &study,
                &PretrendOptions::new(Violation::Delta(vec![0.0; 5]), Bootstrap::new(1, Seed(1))),
            )
            .unwrap_err(),
            "input.domain",
            ErrorKind::Input,
        ),
    ];
    for (error, code, kind) in &cases {
        assert_eq!(error.code(), *code, "{error}");
        assert_eq!(error.kind(), *kind, "{error}");
    }
    let one_pre = EventStudy::new(
        vec![0.1, 0.2],
        Covariance::from_values(2, &[0.04, 0.0, 0.0, 0.04]).expect("covariance"),
        vec![-2, 0],
        -1,
    )
    .expect("event study");
    let error = honest::smoothness_bound_from_pretrends(
        &one_pre,
        honest::OneSided(Level::alpha(0.05).expect("level")),
    )
    .unwrap_err();
    assert_eq!(error.code(), "input.length");
}

/// The columns the entry points take besides the event study, which
/// `EventStudy::new` checks, refused through the testkit's bad inputs. An
/// empty column is one of the wrong length; an empty list of values asks for
/// the default grid, and the values have no order, so those kinds are not
/// tried there; a value that is not finite is outside `[0, inf)`.
#[test]
fn bad_columns_are_input_errors() {
    let study = small_study();
    let options = HonestOptions {
        method: Some(Method::IdentifiedSet),
        least_favorable: Some(Bootstrap::new(1_000, Seed(0))),
        ..HonestOptions::default()
    };
    let rm = |mbar: Vec<f64>| Restriction::RelativeMagnitude {
        mbar: Grid::Values(mbar),
        bound: RmBound::ParallelTrends,
    };
    let length_if_empty = |bad: BadInput| match bad {
        BadInput::Empty => "input.length",
        other => other.code(),
    };
    let rows = |summary: honest::SensitivitySummary| summary.rows.len();
    assert_rejects_bad_input_with_codes("summarize target weights", length_if_empty, |bad| {
        (bad != BadInput::Unsorted).then(|| {
            let weights = Target::Weights(bad.column(&[0.5, 0.5]));
            honest::summarize(&study, &weights, &rm(vec![1.0]), &options).map(rows)
        })
    });
    assert_rejects_bad_input_with_codes(
        "summarize Mbar",
        |_| "input.domain",
        |bad| {
            matches!(bad, BadInput::Nan | BadInput::Infinite).then(|| {
                let values = rm(bad.column(&[0.0, 0.5, 1.0]));
                honest::summarize(&study, &Target::Period(0), &values, &options).map(rows)
            })
        },
    );
    assert_rejects_bad_input_with_codes("pretrends violation", length_if_empty, |bad| {
        (bad != BadInput::Unsorted).then(|| {
            let violation = Violation::Delta(bad.column(&[0.0, 0.0, 0.0, 0.1, 0.2]));
            pretrends::power(
                &study,
                &PretrendOptions::new(violation, Bootstrap::new(12, Seed(3))),
            )
            .map(|analysis| analysis.power.value)
        })
    });
}

/// HonestDiD's restrictions difference neighbouring periods, so a gap in the
/// relative periods is refused rather than read as two adjacent periods.
#[test]
fn relative_periods_with_a_gap_are_refused() {
    let covariance: Vec<f64> = (0..16)
        .map(|i| if i / 4 == i % 4 { 0.04 } else { 0.005 })
        .collect();
    let gapped = EventStudy::new(
        vec![0.02, -0.01, 0.5, 0.7],
        Covariance::from_values(4, &covariance).expect("covariance"),
        vec![-4, -2, 0, 1],
        -1,
    )
    .expect("event study");
    assert!(!gapped.is_consecutive());
    let restriction = Restriction::Smoothness {
        m: Grid::Values(vec![0.0]),
    };
    let error = honest::summarize(
        &gapped,
        &Target::Period(0),
        &restriction,
        &HonestOptions::default(),
    )
    .unwrap_err();
    assert_eq!(error.code(), "input.domain");
    assert!(error.to_string().contains("consecutive"), "{error}");
    let error = honest::original(&gapped, &Target::Period(0), Level::default()).unwrap_err();
    assert_eq!(error.code(), "input.domain");
    let error =
        honest::smoothness_bound_from_pretrends(&gapped, honest::OneSided::default()).unwrap_err();
    assert_eq!(error.code(), "input.domain");
}

/// The breakdown value reads the rows in increasing order, so a grid that
/// decreases is refused; repeated values are allowed.
#[test]
fn a_grid_must_not_decrease() {
    let study = small_study();
    let options = HonestOptions {
        method: Some(Method::IdentifiedSet),
        ..HonestOptions::default()
    };
    let rm = |values: Vec<f64>| Restriction::RelativeMagnitude {
        mbar: Grid::Values(values),
        bound: RmBound::ParallelTrends,
    };
    let error =
        honest::summarize(&study, &Target::Period(0), &rm(vec![1.0, 0.5]), &options).unwrap_err();
    assert_eq!(error.code(), "input.unsorted");
    let error = honest::summarize(&study, &Target::Period(0), &rm(vec![]), &options).unwrap_err();
    assert_eq!(error.code(), "input.empty");
    let summary = honest::summarize(&study, &Target::Period(0), &rm(vec![0.0, 0.0]), &options)
        .expect("repeated values");
    assert_eq!(summary.rows.len(), 2);
}

/// Only the C-LF hybrid draws, so the other methods need no seed, and the
/// identified set carries no confidence level.
#[test]
fn methods_that_never_draw_need_no_seed() {
    let study = small_study();
    let rm = Restriction::RelativeMagnitude {
        mbar: Grid::Values(vec![0.0, 0.5, 1.0, 4.0]),
        bound: RmBound::ParallelTrends,
    };
    let error =
        honest::summarize(&study, &Target::Period(0), &rm, &HonestOptions::default()).unwrap_err();
    assert_eq!(error.code(), "honestdid.unsupported");
    assert!(error.to_string().contains("least_favorable"), "{error}");

    let identified = honest::summarize(
        &study,
        &Target::Period(0),
        &rm,
        &HonestOptions {
            method: Some(Method::IdentifiedSet),
            ..HonestOptions::default()
        },
    )
    .expect("identified sets without draws");
    assert!(identified.rows.iter().all(|row| row.interval.is_none()));
    assert!(identified.rows.iter().all(|row| row.identified.is_some()));
    let breakdown = identified
        .breakdown()
        .expect("the set at Mbar 0 excludes zero");
    let last_excluding = identified
        .rows
        .iter()
        .take_while(|row| {
            let set = row.identified.expect("set");
            set.lower.is_some_and(|l| l > 0.0) || set.upper.is_some_and(|u| u < 0.0)
        })
        .last()
        .expect("a row");
    assert_eq!(breakdown.value, last_excluding.value);
    assert_eq!(
        breakdown.exact,
        identified.rows.last().map(|row| row.value) != Some(breakdown.value)
    );

    let smoothness = honest::summarize(
        &study,
        &Target::Period(0),
        &Restriction::Smoothness {
            m: Grid::Values(vec![0.0, 0.1]),
        },
        &HonestOptions::default(),
    )
    .expect("the FLCI does not draw");
    assert!(smoothness.rows.iter().all(|row| row.interval.is_some()));
}

/// A critical value given as a number is the threshold `pretrends` uses.
#[test]
fn a_critical_value_replaces_the_level() {
    let study = small_study();
    let run = |critical| {
        pretrends::power(
            &study,
            &PretrendOptions {
                critical,
                ..PretrendOptions::new(
                    Violation::LinearTrend { slope: 0.05 },
                    Bootstrap::new(12, Seed(5)),
                )
            },
        )
        .expect("power")
    };
    let wide = run(Critical::Value(3.0));
    let narrow = run(Critical::Level(Level::confidence(0.5).expect("level")));
    assert!(wide.pass_probability.value > narrow.pass_probability.value);
    let error = pretrends::power(
        &study,
        &PretrendOptions {
            critical: Critical::Value(-1.0),
            ..PretrendOptions::new(
                Violation::LinearTrend { slope: 0.0 },
                Bootstrap::new(12, Seed(5)),
            )
        },
    )
    .unwrap_err();
    assert_eq!(error.code(), "input.domain");
}
