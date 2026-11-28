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

use epistat_core::{Bootstrap, Classify, Covariance, ErrorKind, EventStudy, Level};
use epistat_honestdid::Error;
use epistat_honestdid::honest::{
    self, Calibration, Directions, HonestOptions, JointMethod, JointPathOptions, Method,
    Restriction, RmBound, Target,
};
use epistat_honestdid::pretrends::{self, PretrendOptions, Violation};
use epistat_testkit::{BadInput, assert_rejects_bad_input_with_codes, load_json};
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

fn close(ours: Option<f64>, theirs: f64, tolerance: f64) -> bool {
    ours.is_some_and(|v| (v - theirs).abs() <= tolerance)
}

#[test]
fn relative_magnitude_matches_the_package_on_the_average_effect() {
    let study = bc_study();
    let r: RmFixture = load_json("tests/data/honest_did_rm_avg_ref.json");
    let target = Target::Weights(r.l_vec.clone());

    let original = honest::original(&study, &target, Level::default()).expect("original");
    assert!(close(original.lower(), r.original.lb, 1e-8));
    assert!(close(original.upper(), r.original.ub, 1e-8));
    let averaged = honest::original(
        &study,
        &Target::Average { first: 0, last: 3 },
        Level::default(),
    )
    .expect("original of the average");
    assert_eq!(averaged, original);

    let restriction = Restriction::RelativeMagnitude {
        mbar: r.relative_magnitude.iter().map(|row| row.mbar).collect(),
        bound: RmBound::ParallelTrends,
    };
    let summary = honest::summarize(
        &study,
        &target,
        &restriction,
        &HonestOptions::new(Bootstrap::new(1_000, 0)),
    )
    .expect("summary");
    let identified = honest::summarize(
        &study,
        &target,
        &restriction,
        &HonestOptions {
            method: Some(Method::IdentifiedSet),
            ..HonestOptions::new(Bootstrap::new(1_000, 0))
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
            close(row.interval.lower, reference.lb, 5e-3),
            "Mbar {}",
            reference.mbar
        );
        assert!(
            close(row.interval.upper, reference.ub, 5e-3),
            "Mbar {}",
            reference.mbar
        );
        let lp = 1e-6 * (1.0 + reference.id_lb.abs().max(reference.id_ub.abs()));
        let id = row.identified.expect("identified set");
        assert!(close(id.lower, reference.id_lb, lp));
        assert!(close(id.upper, reference.id_ub, lp));
        assert_eq!(set.identified, row.identified);
        assert_eq!(set.interval.lower, id.lower);
    }
    // The intervals widen with Mbar, so the breakdown value is the last Mbar
    // whose interval excludes zero.
    let excluding = summary
        .rows
        .iter()
        .take_while(|row| row.interval.lower.is_some_and(|l| l > 0.0))
        .last()
        .map(|row| row.value);
    assert_eq!(summary.breakdown(), excluding);
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
                &Restriction::Smoothness { m: vec![case.m] },
                &HonestOptions {
                    method: Some(Method::IdentifiedSet),
                    ..HonestOptions::new(Bootstrap::new(1_000, 0))
                },
            )
            .expect("identified set");
            let set = summary.rows[0].identified.expect("identified set");
            if case.identified.feasible {
                let lp = 1e-6 * (1.0 + case.identified.lb.abs().max(case.identified.ub.abs()));
                assert!(close(set.lower, case.identified.lb, lp), "{}", dataset.name);
                assert!(close(set.upper, case.identified.ub, lp), "{}", dataset.name);
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
        level: Level::alpha(r.alpha).expect("level"),
        tolerance: 1e-6,
        ..PretrendOptions::new(
            Violation::Delta(case.delta.clone()),
            Bootstrap::new(12, 20_260_125),
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
        mbar: vec![0.5],
        bound: RmBound::ParallelTrends,
    };
    let bonferroni = honest::joint_path(
        &study,
        &restriction,
        &JointPathOptions::new(Bootstrap::new(1_000, 0)),
    )
    .expect("path");
    let gaussian = honest::joint_path(
        &study,
        &restriction,
        &JointPathOptions {
            calibration: Calibration::Gaussian(Bootstrap::new(5_000, 7)),
            ..JointPathOptions::new(Bootstrap::new(1_000, 0))
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
        &Restriction::Smoothness { m: vec![0.05] },
        &JointPathOptions {
            method: JointMethod::Flci,
            directions: Directions::Weights(vec![vec![1.0, 0.0], vec![0.5, 0.5]]),
            ..JointPathOptions::new(Bootstrap::new(1_000, 0))
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
        mbar,
        bound: RmBound::ParallelTrends,
    };
    let options = HonestOptions::new(Bootstrap::new(1_000, 0));
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
                &Restriction::Smoothness { m: vec![0.1] },
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
                &JointPathOptions::new(Bootstrap::new(1_000, 0)),
            )
            .unwrap_err(),
            "input.length",
            ErrorKind::Input,
        ),
        (
            honest::joint_path(
                &study,
                &Restriction::RelativeMagnitude {
                    mbar: vec![1.0],
                    bound: RmBound::LinearTrend,
                },
                &JointPathOptions::new(Bootstrap::new(1_000, 0)),
            )
            .unwrap_err(),
            "honestdid.unsupported",
            ErrorKind::Unsupported,
        ),
        (
            pretrends::power(
                &study,
                &PretrendOptions::new(Violation::Power(1.5), Bootstrap::new(12, 1)),
            )
            .unwrap_err(),
            "input.domain",
            ErrorKind::Input,
        ),
        (
            pretrends::power(
                &study,
                &PretrendOptions::new(Violation::Delta(vec![0.0; 5]), Bootstrap::new(1, 1)),
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
    let error =
        honest::smoothness_bound_from_pretrends(&one_pre, Level::alpha(0.05).expect("level"))
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
        ..HonestOptions::new(Bootstrap::new(1_000, 0))
    };
    let rm = |mbar: Vec<f64>| Restriction::RelativeMagnitude {
        mbar,
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
                &PretrendOptions::new(violation, Bootstrap::new(12, 3)),
            )
            .map(|analysis| analysis.power.value)
        })
    });
}
