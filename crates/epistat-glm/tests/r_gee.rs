//! Generalised estimating equations checked against R 4.6's `geepack::geeglm`
//! 1.3.13 and `gee::gee` 4.13-30.
//!
//! The reference is `tests/data/gee.json`, written by `tests/data/gee.R`. It
//! holds `ohio`, `seizure` and `dietox` from geepack and three simulated
//! unbalanced panels with missing waves, prior weights and an offset, with
//! every working correlation `geeglm` runs on, and for each the coefficients,
//! the robust and naive covariances, the scale and the association
//! parameters. `geeglm` is run at `epsilon = 1e-12`, `gee` at `tol = 1e-12`,
//! and the Rust fit at `tolerance = 1e-12` so that no stopping rule limits the
//! comparison.
//!
//! Tolerances. All three iterate to a change below `1e-12` of the parameters,
//! and the iteration contracts by a factor well below one per step, so each
//! stops within about `1e-12` of the fixed point of its own equations. What is
//! left is the rounding of the sums, amplified by the condition number of the
//! information matrix, which is below `1e3` in these data. The worst deviation
//! observed is `1.1e-12` of the largest reference value of a vector. The
//! coefficients, the association parameters and the scale are compared at
//! `1e-10` and the covariances at `1e-9`, a factor of about 100 above the
//! worst case. The independence fit is compared with `glm`, which stops at a
//! relative change of `1e-8` in the deviance and so leaves the covariance
//! about `1e-7` from the fixed point, at `1e-6`.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the test panels have a few thousand rows"
)]

use epistat_core::InputError;
use epistat_glm::{
    Association, Correlation, Error, Family, Gee, Glm, Link, RobustVariance, SandwichType,
    robust_covariance,
};
use epistat_inference::Clusters;
use epistat_linalg::Mat;
use epistat_testkit::{assert_same_on_threads, load_json};
use serde::Deserialize;
use std::collections::BTreeMap;

mod common;
use common::*;

#[derive(Deserialize)]
#[serde(untagged)]
enum Numbers {
    One(f64),
    Many(Vec<f64>),
}

impl Numbers {
    fn to_vec(&self) -> Vec<f64> {
        match self {
            Self::One(v) => vec![*v],
            Self::Many(v) => v.clone(),
        }
    }
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    cases: Vec<Case>,
    independence_glm: Independence,
}

#[derive(Deserialize)]
struct Versions {
    r: String,
    geepack: String,
    gee: String,
}

#[derive(Deserialize)]
struct Independence {
    coef: Vec<f64>,
    robust: Vec<f64>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    family: String,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    cluster: Vec<u32>,
    wave: Vec<u32>,
    weights: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
    fits: BTreeMap<String, Entry>,
}

#[derive(Deserialize)]
struct Entry {
    geeglm: Fit,
    gee: Option<Fit>,
    mancl: Option<Vec<f64>>,
}

#[derive(Deserialize)]
struct Fit {
    coef: Vec<f64>,
    robust: Vec<f64>,
    naive: Vec<f64>,
    scale: f64,
    alpha: Numbers,
}

fn reference() -> Reference {
    load_json(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/gee.json"))
}

fn matrix(rows: &[Vec<f64>]) -> Mat<f64> {
    Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
}

fn model(case: &Case, structure: &str) -> Gee {
    let (family, link) = match case.family.as_str() {
        "binomial" => (Family::Binomial, Link::Logit),
        "poisson" => (Family::Poisson, Link::Log),
        "gaussian" => (Family::Gaussian, Link::Identity),
        other => panic!("family {other}"),
    };
    let correlation = match structure {
        "independence" => Correlation::Independence,
        "exchangeable" => Correlation::Exchangeable,
        "ar1" => Correlation::Ar1,
        "unstructured" => Correlation::Unstructured,
        other => panic!("structure {other}"),
    };
    let mut gee = Gee::new(family, link, correlation);
    gee.tolerance = 1e-12;
    gee.max_iterations = 500;
    gee
}

fn fit(case: &Case, gee: &Gee) -> epistat_glm::GeeFit {
    let x = matrix(&case.x);
    gee.fit6(
        x.as_ref(),
        &case.y,
        &case.cluster,
        Some(&case.wave),
        case.weights.as_deref(),
        case.offset.as_deref(),
    )
    .unwrap()
}

/// The largest absolute deviation over the largest absolute reference value.
fn worst(got: &[f64], want: &[f64], label: &str) -> f64 {
    assert_eq!(got.len(), want.len(), "{label}");
    let scale = want.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs() / scale)
        .fold(0.0, f64::max)
}

fn some(values: &[Option<f64>]) -> Vec<f64> {
    values.iter().map(|v| v.expect("defined")).collect()
}

#[test]
fn reference_was_made_by_the_documented_versions() {
    let r = reference();
    assert!(r.versions.r.starts_with("R version 4.6"));
    assert_eq!(r.versions.geepack, "1.3.13");
    assert_eq!(r.versions.gee, "4.13.30");
}

#[test]
fn yan_fine_fits_match_geeglm() {
    let mut overall = 0.0_f64;
    for case in &reference().cases {
        for (structure, entry) in &case.fits {
            let gee = model(case, structure);
            let got = fit(case, &gee);
            let label = format!("{} {structure}", case.name);
            assert!(got.converged, "{label}");
            let want = &entry.geeglm;
            let a = worst(&some(&got.coefficients), &want.coef, &label);
            let b = worst(&some(got.covariance.as_slice()), &want.robust, &label);
            let c = worst(&some(got.naive_covariance.as_slice()), &want.naive, &label);
            let d = worst(&[got.scale], &[want.scale], &label);
            let alpha = want.alpha.to_vec();
            let e = if alpha.is_empty() {
                assert!(got.alpha.is_empty(), "{label}");
                0.0
            } else {
                worst(&got.alpha, &alpha, &label)
            };
            eprintln!(
                "{label}: coef {a:.1e} robust {b:.1e} naive {c:.1e} scale {d:.1e} alpha {e:.1e} in {} iterations",
                got.iterations
            );
            assert!(a < 1e-10 && d < 1e-10 && e < 1e-10, "{label}");
            assert!(b < 1e-9 && c < 1e-9, "{label}");
            overall = overall.max(a).max(b).max(c).max(d).max(e);
        }
    }
    eprintln!("worst deviation from geeglm {overall:.1e}");
}

#[test]
fn moment_fits_match_gee() {
    let mut overall = 0.0_f64;
    let mut compared = 0;
    for case in &reference().cases {
        for (structure, entry) in &case.fits {
            let Some(want) = &entry.gee else { continue };
            let mut gee = model(case, structure);
            gee.association = Association::Moment;
            let got = fit(case, &gee);
            let label = format!("{} {structure} (gee)", case.name);
            let a = worst(&some(&got.coefficients), &want.coef, &label);
            let b = worst(&some(got.covariance.as_slice()), &want.robust, &label);
            let c = worst(&some(got.naive_covariance.as_slice()), &want.naive, &label);
            let d = worst(&[got.scale], &[want.scale], &label);
            let alpha = want.alpha.to_vec();
            let e = if alpha.is_empty() {
                0.0
            } else {
                worst(&got.alpha, &alpha, &label)
            };
            eprintln!(
                "{label}: coef {a:.1e} robust {b:.1e} naive {c:.1e} scale {d:.1e} alpha {e:.1e}"
            );
            assert!(a < 1e-10 && d < 1e-10 && e < 1e-10, "{label}");
            assert!(b < 1e-9 && c < 1e-9, "{label}");
            overall = overall.max(a).max(b).max(c).max(d).max(e);
            compared += 1;
        }
    }
    assert!(compared >= 10, "{compared}");
    eprintln!("worst deviation from gee {overall:.1e}");
}

#[test]
fn mancl_de_rouen_matches_the_transcription() {
    let mut compared = 0;
    for case in &reference().cases {
        for (structure, entry) in &case.fits {
            let Some(want) = &entry.mancl else { continue };
            let mut gee = model(case, structure);
            gee.variance = RobustVariance::ManclDeRouen;
            let got = fit(case, &gee);
            let label = format!("{} {structure} (Mancl and DeRouen)", case.name);
            let dev = worst(&some(got.covariance.as_slice()), want, &label);
            eprintln!("{label}: {dev:.1e}");
            assert!(dev < 1e-9, "{label}");
            compared += 1;
        }
    }
    assert!(compared >= 6, "{compared}");
}

#[test]
fn independence_is_glm_with_the_clustered_sandwich() {
    let r = reference();
    let case = &r.cases[0];
    assert_eq!(case.name, "ohio");
    let got = fit(case, &model(case, "independence"));
    // glm stops at a relative change of 1e-8 in the deviance, so its
    // coefficients are within about 1e-6 of the maximum.
    let a = worst(&some(&got.coefficients), &r.independence_glm.coef, "coef");
    let b = worst(
        &some(got.covariance.as_slice()),
        &r.independence_glm.robust,
        "robust",
    );
    eprintln!("independence against glm and vcovCL: coef {a:.1e} robust {b:.1e}");
    assert!(a < 1e-6 && b < 1e-6);
    // The same covariance from this crate's own sandwich on the Glm fit.
    let x = matrix(&case.x);
    let model = Glm::new(Family::Binomial, Link::Logit);
    let glm = model.fit5(x.as_ref(), &case.y, None, None, None).unwrap();
    let clusters = Clusters::from_labels(&case.cluster);
    let own = robust_covariance(
        &model,
        &glm,
        x.as_ref(),
        &case.y,
        Some(&clusters),
        SandwichType::Hc0,
    )
    .unwrap();
    let c = worst(
        &some(got.covariance.as_slice()),
        &some(own.as_slice()),
        "own sandwich",
    );
    eprintln!("independence against robust_covariance Hc0: {c:.1e}");
    assert!(c < 1e-6);
}

#[test]
fn an_aliased_column_is_none() {
    let r = reference();
    let case = &r.cases[0];
    let n = case.x.len();
    let x = Mat::from_fn(n, 4, |i, j| {
        if j < 3 {
            case.x[i][j]
        } else {
            2.0 * case.x[i][1]
        }
    });
    let gee = model(case, "exchangeable");
    let got = gee
        .fit6(
            x.as_ref(),
            &case.y,
            &case.cluster,
            Some(&case.wave),
            None,
            None,
        )
        .unwrap();
    assert_eq!(got.rank, 3);
    assert!(got.coefficients[3].is_none());
    assert!(got.covariance.get(3, 3).is_none() && got.covariance.get(0, 3).is_none());
    let plain = fit(case, &gee);
    assert_eq!(
        got.coefficients[..3]
            .iter()
            .map(|v| v.unwrap())
            .collect::<Vec<_>>(),
        some(&plain.coefficients)
    );
}

#[test]
fn bad_input_is_an_error() {
    let r = reference();
    let case = &r.cases[3];
    let x = matrix(&case.x);
    let gee = model(case, "exchangeable");
    let run = |gee: &Gee, cluster: &[u32], wave: Option<&[u32]>, w: Option<&[f64]>| {
        gee.fit6(
            x.as_ref(),
            &case.y,
            cluster,
            wave,
            w,
            case.offset.as_deref(),
        )
    };
    let ones = vec![1.0; case.y.len()];
    // A second row at one wave.
    let mut dup = case.wave.clone();
    dup[1] = dup[0];
    let first = case.cluster[0];
    assert_eq!(case.cluster[1], first);
    assert!(matches!(
        run(&gee, &case.cluster, Some(&dup), None),
        Err(Error::DuplicateWave { .. })
    ));
    // Unstructured without waves, and the moment estimator with weights.
    let un = model(case, "unstructured");
    assert!(matches!(
        run(&un, &case.cluster, None, None),
        Err(Error::Unsupported { .. })
    ));
    let mut moment = gee;
    moment.association = Association::Moment;
    assert!(matches!(
        run(&moment, &case.cluster, Some(&case.wave), Some(&ones)),
        Err(Error::Unsupported { .. })
    ));
    // One row per cluster has no pair to estimate a correlation from.
    let singles: Vec<u32> = (0..case.y.len() as u32).collect();
    assert!(matches!(
        run(&gee, &singles, Some(&case.wave), None),
        Err(Error::Undefined(_))
    ));
    // A wrong length and a negative weight.
    assert!(matches!(
        run(&gee, &case.cluster[1..], None, None),
        Err(Error::Input(InputError::Length { .. }))
    ));
    let mut bad = ones;
    bad[3] = -1.0;
    assert!(matches!(
        run(&gee, &case.cluster, Some(&case.wave), Some(&bad)),
        Err(Error::Input(InputError::Domain {
            what: "weights",
            ..
        }))
    ));
}

#[test]
fn unstructured_waves_that_never_meet_are_an_error() {
    // Clusters hold waves {0, 1} or {2, 3}, so the pair (0, 2) has no data.
    let n = 40;
    let cluster: Vec<u32> = (0..n as u32).map(|i| i / 2).collect();
    let wave: Vec<u32> = (0..n as u32).map(|i| (i % 2) + 2 * ((i / 2) % 2)).collect();
    let y: Vec<f64> = (0..n).map(|i| f64::from((i * 7 % 5) as u8)).collect();
    let x = Mat::from_fn(n, 1, |_, _| 1.0);
    let gee = Gee::new(Family::Gaussian, Link::Identity, Correlation::Unstructured);
    let got = gee.fit6(x.as_ref(), &y, &cluster, Some(&wave), None, None);
    assert!(matches!(got, Err(Error::Undefined(_))), "{got:?}");
}

fn bits(case: &Case, structure: &str, variance: RobustVariance) -> Vec<f64> {
    let mut gee = model(case, structure);
    gee.tolerance = 1e-10;
    gee.variance = variance;
    let got = fit(case, &gee);
    let mut out = some(&got.coefficients);
    out.extend(some(got.covariance.as_slice()));
    out.extend(some(got.naive_covariance.as_slice()));
    out.extend(&got.alpha);
    out.push(got.scale);
    out.extend(&got.fitted);
    out
}

fn all_bits() -> Vec<f64> {
    let r = reference();
    let mut out = Vec::new();
    for (name, structure, variance) in [
        ("ohio", "exchangeable", RobustVariance::Sandwich),
        ("ohio", "ar1", RobustVariance::Sandwich),
        ("ohio", "unstructured", RobustVariance::Sandwich),
        ("sim_gauss", "exchangeable", RobustVariance::ManclDeRouen),
        ("sim_gauss", "ar1", RobustVariance::Sandwich),
        (
            "sim_unstructured",
            "unstructured",
            RobustVariance::ManclDeRouen,
        ),
    ] {
        let case = r.cases.iter().find(|c| c.name == name).unwrap();
        out.extend(bits(case, structure, variance));
    }
    out
}

#[test]
fn results_do_not_depend_on_the_number_of_threads() {
    assert_same_on_threads(all_bits);
}

#[test]
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn the_fits_have_their_recorded_bits() {
    let hash = epistat_testkit::Fingerprint::new()
        .f64s(&all_bits())
        .finish();
    assert_eq!(hash, 3_781_498_169_942_712_397, "recorded hash");
}
