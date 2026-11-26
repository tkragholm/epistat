//! Cumulative link models and the Brant test checked against R 4.6's
//! `ordinal::clm` 2026.7.26, `MASS::polr` 7.3.66, `rms::orm` 8.2.1,
//! `VGAM::vglm` 1.1.14 and a transcription of Brant's formulas.
//!
//! The reference is `tests/data/ordinal.json`, written by
//! `tests/data/ordinal.R`: the housing data of MASS with frequency weights
//! under four links, a simulated five-category outcome with ten covariates,
//! weights, an offset and clusters under three links, a four-category outcome
//! with two category-specific covariates, and the Brant test on the latter.
//!
//! Tolerances. `clm` stops when the largest gradient entry is below `1e-10`.
//! The log-likelihood is concave with curvature of the order of the number of
//! rows, so a parameter is within about `1e-10 / curvature` of the maximum,
//! below `1e-11` here, and the Rust fit stops at a step of `1e-10` relative,
//! quadratically close to the maximum. The two agree to the rounding of the
//! sums, compared at `1e-8` relative to the largest value of the vector. `polr`
//! runs BFGS to `reltol = 1e-14`, which leaves the parameters about
//! `sqrt(1e-14 / curvature)` from the maximum, compared at `1e-6`, and its
//! covariance comes from a numerical Hessian, compared at `1e-5`. `orm` stops
//! at `eps = 1e-12` in the criterion, observed `1e-10`, compared at `1e-8`.
//! VGAM stops at `1e-6` relative in the likelihood, compared at `1e-4`. The cloglog fits agree with `clm`
//! to `1.5e-8`, about a hundred times worse than the logit and probit fits,
//! which is `clm`'s stopping rule on a flatter likelihood, and are compared at
//! `1e-7`.
//!
//! The cluster-robust covariance is compared with the sandwich written out in
//! R from analytic scores. `sandwich::estfun.clm` leaves the offset out of the
//! scores (measured against finite differences), so `vcovCL` on a `clm` with an
//! offset is wrong.

use epistat_core::InputError;
use epistat_glm::{Error, Ordinal, OrdinalLink, brant};
use epistat_linalg::Mat;
use epistat_testkit::{assert_same_on_threads, load_json};
use serde::Deserialize;

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
    housing: Panel,
    sim: Panel,
    ppo: Ppo,
    brant: Vec<BrantCase>,
}

#[derive(Deserialize)]
struct Versions {
    r: String,
    ordinal: String,
    #[serde(rename = "MASS")]
    mass: String,
    rms: String,
    #[serde(rename = "VGAM")]
    vgam: String,
    brant: String,
}

#[derive(Deserialize)]
struct Panel {
    x: Vec<Vec<f64>>,
    y: Vec<u32>,
    weights: Vec<f64>,
    offset: Option<Vec<f64>>,
    cluster: Option<Vec<u32>>,
    fits: Vec<RFit>,
}

#[derive(Deserialize)]
struct RFit {
    link: String,
    theta: Vec<f64>,
    beta: Vec<f64>,
    loglik: f64,
    null_loglik: f64,
    vcov: Vec<f64>,
    probs: Vec<f64>,
    polr: Option<Polr>,
    orm: Option<Orm>,
    cluster: Option<Vec<f64>>,
}

#[derive(Deserialize)]
struct Polr {
    theta: Vec<f64>,
    beta: Vec<f64>,
    deviance: f64,
    vcov: Vec<f64>,
}

#[derive(Deserialize)]
struct Orm {
    theta_clm: Vec<f64>,
    beta_clm: Vec<f64>,
    theta: Vec<f64>,
    beta: Vec<f64>,
}

#[derive(Deserialize)]
struct Ppo {
    x: Vec<Vec<f64>>,
    nominal: Vec<Vec<f64>>,
    y: Vec<u32>,
    theta: Vec<f64>,
    beta: Numbers,
    tau: Vec<f64>,
    loglik: f64,
    vglm_loglik: f64,
    vglm_theta: Vec<f64>,
    vglm_beta: Numbers,
    vglm_tau: Vec<f64>,
}

#[derive(Deserialize)]
struct BrantCase {
    x: Vec<Vec<f64>>,
    y: Vec<u32>,
    weights: Vec<f64>,
    omnibus: f64,
    omnibus_df: usize,
    per_covariate: Vec<f64>,
    per_df: usize,
    coefficients: Vec<f64>,
    pkg_omnibus: f64,
    pkg_omnibus_df: usize,
    pkg_per_covariate: Vec<f64>,
    pkg_per_df: Vec<usize>,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/ordinal.json"
    ))
}

fn matrix(rows: &[Vec<f64>]) -> Mat<f64> {
    Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
}

fn link(name: &str) -> OrdinalLink {
    match name {
        "logit" => OrdinalLink::Logit,
        "probit" => OrdinalLink::Probit,
        "cloglog" => OrdinalLink::Cloglog,
        "loglog" => OrdinalLink::Loglog,
        other => panic!("link {other}"),
    }
}

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

/// The covariance of the parameters in the order of the reference, which has
/// no aliased columns.
fn full_covariance(fit: &epistat_glm::OrdinalFit) -> Vec<f64> {
    some(fit.covariance.as_slice())
}

#[test]
fn reference_was_made_by_the_documented_versions() {
    let v = reference().versions;
    assert!(v.r.starts_with("R version 4.6"));
    assert_eq!(v.ordinal, "2026.7.26");
    assert_eq!(v.mass, "7.3.66");
    assert_eq!(v.rms, "8.2.1");
    assert_eq!(v.vgam, "1.1.14");
    assert_eq!(v.brant, "0.3.0");
}

fn check_panel(panel: &Panel, name: &str, with_offset: bool) {
    let x = matrix(&panel.x);
    for r in &panel.fits {
        let model = Ordinal::new(link(&r.link));
        let offset = if with_offset {
            panel.offset.as_deref()
        } else {
            None
        };
        let fit = model
            .fit5(x.as_ref(), &panel.y, Some(&panel.weights), offset, None)
            .unwrap();
        let label = format!("{name} {}", r.link);
        assert!(fit.converged && !fit.fitted_at_boundary, "{label}");
        let a = worst(&fit.thresholds, &r.theta, &label);
        let b = worst(&some(&fit.coefficients), &r.beta, &label);
        let c = worst(&full_covariance(&fit), &r.vcov, &label);
        let d = (fit.log_likelihood - r.loglik).abs() / r.loglik.abs();
        let e = (fit.null_log_likelihood - r.null_loglik).abs() / r.null_loglik.abs();
        let probs = fit.probabilities(x.as_ref(), None, offset).unwrap();
        let f = worst(&probs, &r.probs, &label);
        eprintln!(
            "{label}: thresholds {a:.1e} coef {b:.1e} vcov {c:.1e} loglik {d:.1e} null {e:.1e} probs {f:.1e} in {} iterations",
            fit.iterations
        );
        let bound = if r.link == "cloglog" { 1e-7 } else { 1e-8 };
        assert!(a < bound && b < bound && c < bound && f < bound, "{label}");
        assert!(d < 1e-11 && e < 1e-11, "{label}");
        if let Some(p) = &r.polr {
            let a = worst(&fit.thresholds, &p.theta, &label);
            let b = worst(&some(&fit.coefficients), &p.beta, &label);
            let c = worst(
                &full_covariance(&fit)[..],
                &polr_layout(&p.vcov, &fit),
                &label,
            );
            let d = (-2.0 * fit.log_likelihood - p.deviance).abs() / p.deviance;
            eprintln!(
                "{label} against polr: thresholds {a:.1e} coef {b:.1e} vcov {c:.1e} deviance {d:.1e}"
            );
            assert!(a < 1e-6 && b < 1e-6 && c < 1e-5 && d < 1e-10, "{label}");
        }
    }
}

/// polr's covariance is of the coefficients and then the thresholds.
fn polr_layout(vcov: &[f64], fit: &epistat_glm::OrdinalFit) -> Vec<f64> {
    let k1 = fit.thresholds.len();
    let p = fit.coefficients.len();
    let t = k1 + p;
    let at = |i: usize| if i < k1 { p + i } else { i - k1 };
    let mut out = vec![0.0; t * t];
    for i in 0..t {
        for j in 0..t {
            out[i * t + j] = vcov[at(i) * t + at(j)];
        }
    }
    out
}

#[test]
fn housing_matches_clm_and_polr() {
    check_panel(&reference().housing, "housing", false);
}

#[test]
fn simulated_five_categories_match_clm() {
    check_panel(&reference().sim, "sim", true);
}

#[test]
fn logit_without_the_offset_matches_orm() {
    let r = reference();
    let x = matrix(&r.sim.x);
    let logit = r.sim.fits.iter().find(|f| f.link == "logit").unwrap();
    let orm = logit.orm.as_ref().unwrap();
    let fit = Ordinal::new(OrdinalLink::Logit)
        .fit5(x.as_ref(), &r.sim.y, Some(&r.sim.weights), None, None)
        .unwrap();
    let clm = worst(&fit.thresholds, &orm.theta_clm, "clm thresholds").max(worst(
        &some(&fit.coefficients),
        &orm.beta_clm,
        "clm beta",
    ));
    let a = worst(&fit.thresholds, &orm.theta, "orm thresholds");
    let b = worst(&some(&fit.coefficients), &orm.beta, "orm beta");
    eprintln!("without offset: clm {clm:.1e}, orm thresholds {a:.1e} coef {b:.1e}");
    assert!(clm < 1e-8 && a < 1e-8 && b < 1e-8);
}

#[test]
fn cluster_robust_covariance_matches_the_scores_in_r() {
    let r = reference();
    let x = matrix(&r.sim.x);
    let logit = r.sim.fits.iter().find(|f| f.link == "logit").unwrap();
    let fit = Ordinal::new(OrdinalLink::Logit)
        .fit5(
            x.as_ref(),
            &r.sim.y,
            Some(&r.sim.weights),
            r.sim.offset.as_deref(),
            r.sim.cluster.as_deref(),
        )
        .unwrap();
    let robust = some(fit.robust_covariance.as_ref().unwrap().as_slice());
    let dev = worst(&robust, logit.cluster.as_ref().unwrap(), "robust");
    eprintln!("cluster robust against the scores written out in R: {dev:.1e}");
    assert!(dev < 1e-7);
}

#[test]
fn partial_proportional_odds_matches_clm_and_vglm() {
    let r = reference();
    let p = &r.ppo;
    let x = matrix(&p.x);
    let w = matrix(&p.nominal);
    let fit = Ordinal::new(OrdinalLink::Logit)
        .fit_partial(x.as_ref(), w.as_ref(), &p.y, None, None, None)
        .unwrap();
    assert!(fit.converged);
    assert_eq!(fit.ordering_violations, 0);
    let a = worst(&fit.thresholds, &p.theta, "thresholds");
    let b = worst(&some(&fit.coefficients), &p.beta.to_vec(), "beta");
    let c = worst(&fit.nominal, &p.tau, "tau");
    let d = (fit.log_likelihood - p.loglik).abs() / p.loglik.abs();
    // VGAM lists the category-specific coefficients covariate by covariate.
    let k1 = fit.thresholds.len();
    let q = 2;
    let vtau: Vec<f64> = (0..k1)
        .flat_map(|k| (0..q).map(move |l| (k, l)))
        .map(|(k, l)| p.vglm_tau[l * k1 + k])
        .collect();
    let e = worst(&fit.thresholds, &p.vglm_theta, "vglm thresholds");
    let f = worst(&fit.nominal, &vtau, "vglm tau");
    let g = worst(&some(&fit.coefficients), &p.vglm_beta.to_vec(), "vglm beta");
    let h = (fit.log_likelihood - p.vglm_loglik).abs() / p.loglik.abs();
    eprintln!(
        "partial: clm thresholds {a:.1e} beta {b:.1e} tau {c:.1e} loglik {d:.1e}; vglm thresholds {e:.1e} beta {g:.1e} tau {f:.1e} loglik {h:.1e}"
    );
    assert!(a < 1e-8 && b < 1e-8 && c < 1e-8 && d < 1e-11);
    assert!(e < 1e-4 && f < 1e-4 && g < 1e-4 && h < 1e-8);
}

#[test]
fn brant_test_matches_the_transcription() {
    for case in &reference().brant {
        let x = matrix(&case.x);
        let got = brant(&ordinal_data(x.as_ref(), None, &case.y, Some(&case.weights), None, None).unwrap()).unwrap();
        let b = worst(&got.coefficients, &case.coefficients, "slopes");
        let o = (got.omnibus.statistic - case.omnibus).abs() / case.omnibus;
        let per: Vec<f64> = got.per_covariate.iter().map(|w| w.statistic).collect();
        let c = worst(&per, &case.per_covariate, "per covariate");
        eprintln!("brant: slopes {b:.1e} omnibus {o:.1e} per covariate {c:.1e}");
        assert_eq!(got.omnibus.df, case.omnibus_df);
        assert!(got.per_covariate.iter().all(|w| w.df == case.per_df));
        assert!(b < 1e-9 && o < 1e-9 && c < 1e-9);
    }
}

// The R package brant 0.3.0 on a MASS::polr fit (frequency weights as repeated
// rows). It differs from Brant's formulas in two ways, both visible against the
// transcription above. The lower-triangle blocks of the covariance of the
// stacked estimates are copies of the upper blocks without the transpose, which
// changes the omnibus statistic by 1.5e-4 relative here. The diagonal blocks are
// glm's vcov, formed from the working weights of the step before the last of
// IRLS, whose parameter error at the default epsilon of 1e-8 is about 1e-4 and
// moves the variance by about 5e-4 relative. The statistics are therefore
// compared to 2e-3 relative, ten times the largest difference seen in the two
// cases (1.8e-4 for the omnibus statistic, 1.2e-4 per covariate), and the
// degrees of freedom exactly.
#[test]
fn brant_test_matches_the_brant_package() {
    for case in &reference().brant {
        let x = matrix(&case.x);
        let got = brant(&ordinal_data(x.as_ref(), None, &case.y, Some(&case.weights), None, None).unwrap()).unwrap();
        let o = (got.omnibus.statistic - case.pkg_omnibus).abs() / case.pkg_omnibus;
        let per: Vec<f64> = got.per_covariate.iter().map(|w| w.statistic).collect();
        let c = per
            .iter()
            .zip(&case.pkg_per_covariate)
            .map(|(a, b)| (a - b).abs() / b)
            .fold(0.0_f64, f64::max);
        eprintln!("brant package: omnibus {o:.1e} per covariate {c:.1e}");
        assert_eq!(got.omnibus.df, case.pkg_omnibus_df);
        let dfs: Vec<usize> = got.per_covariate.iter().map(|w| w.df).collect();
        assert_eq!(dfs, case.pkg_per_df);
        assert!(o < 2e-3 && c < 2e-3);
    }
}

#[test]
fn errors_and_separation() {
    let model = Ordinal::new(OrdinalLink::Logit);
    // Complete separation: the covariate orders the outcome.
    let x = Mat::from_fn(12, 1, |i, _| i as f64);
    let y: Vec<u32> = (0..12).map(|i| i / 4).collect();
    let got = model.fit5(x.as_ref(), &y, None, None, None);
    assert!(
        matches!(got, Err(Error::Separation))
            || got
                .as_ref()
                .is_ok_and(|f| !f.converged && f.fitted_at_boundary),
        "{got:?}"
    );
    let x = Mat::from_fn(9, 1, |i, _| ((i * 5) % 7) as f64);
    // An empty category.
    let y = [0, 2, 0, 2, 0, 2, 0, 2, 0];
    assert!(matches!(
        model.fit5(x.as_ref(), &y, None, None, None),
        Err(Error::EmptyClass { class: 1 })
    ));
    // A constant column.
    let y = [0, 1, 2, 0, 1, 2, 0, 1, 2];
    let c = Mat::from_fn(9, 2, |i, j| if j == 0 { ((i * 5) % 7) as f64 } else { 1.0 });
    assert!(matches!(
        model.fit5(c.as_ref(), &y, None, None, None),
        Err(Error::Intercept { column: 1 })
    ));
    // A negative weight and a wrong length.
    let mut w = vec![1.0; 9];
    w[2] = -1.0;
    assert!(matches!(
        model.fit5(x.as_ref(), &y, Some(&w), None, None),
        Err(Error::Input(InputError::Domain {
            what: "weights",
            ..
        }))
    ));
    assert!(matches!(
        model.fit5(x.as_ref(), &y[1..], None, None, None),
        Err(Error::Input(InputError::Length { .. }))
    ));
}

#[test]
fn an_aliased_column_is_none() {
    let r = reference();
    let n = r.housing.x.len();
    let p = r.housing.x[0].len();
    let x = Mat::from_fn(n, p + 1, |i, j| {
        if j < p {
            r.housing.x[i][j]
        } else {
            r.housing.x[i][0] + r.housing.x[i][1]
        }
    });
    let fit = Ordinal::new(OrdinalLink::Logit)
        .fit5(
            x.as_ref(),
            &r.housing.y,
            Some(&r.housing.weights),
            None,
            None,
        )
        .unwrap();
    assert!(fit.coefficients[p].is_none());
    let t = fit.thresholds.len() + p + 1;
    assert!(fit.covariance.get(t - 1, t - 1).is_none());
    assert!(fit.covariance.get(0, 0).is_some());
}

fn all_bits() -> Vec<f64> {
    let r = reference();
    let mut out = Vec::new();
    let x = matrix(&r.sim.x);
    for l in [
        OrdinalLink::Logit,
        OrdinalLink::Probit,
        OrdinalLink::Cloglog,
    ] {
        let fit = Ordinal::new(l)
            .fit5(
                x.as_ref(),
                &r.sim.y,
                Some(&r.sim.weights),
                r.sim.offset.as_deref(),
                r.sim.cluster.as_deref(),
            )
            .unwrap();
        out.extend(&fit.thresholds);
        out.extend(some(&fit.coefficients));
        out.extend(some(fit.covariance.as_slice()));
        out.extend(some(fit.robust_covariance.as_ref().unwrap().as_slice()));
        out.push(fit.log_likelihood);
    }
    let p = &r.ppo;
    let (x, w) = (matrix(&p.x), matrix(&p.nominal));
    let fit = Ordinal::new(OrdinalLink::Logit)
        .fit_partial(x.as_ref(), w.as_ref(), &p.y, None, None, None)
        .unwrap();
    out.extend(&fit.thresholds);
    out.extend(&fit.nominal);
    out.extend(some(fit.covariance.as_slice()));
    let c = &r.brant[1];
    let got = brant(&ordinal_data(matrix(&c.x).as_ref(), None, &c.y, Some(&c.weights), None, None).unwrap()).unwrap();
    out.extend(&got.coefficients);
    out.push(got.omnibus.statistic);
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
    assert_eq!(hash, 11_258_002_169_801_411_814, "recorded hash");
}
