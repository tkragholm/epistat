//! Synthetic difference-in-differences, synthetic control and
//! difference-in-differences against R's `synthdid` 0.0.9 (commit
//! 70c1ce3eac58e28c30b67435ca377bb48baa9b8a of
//! github.com/synth-inference/synthdid, not on CRAN), from the reference
//! written by `tests/data/generate_synthdid_reference.R`.
//!
//! # Tolerances
//!
//! The weights come from the same Frank-Wolfe steps as in R. Each test checks
//! that every run took exactly R's number of steps and that the same weights
//! are exactly zero. A step only makes a weight positive by choosing its vertex,
//! and sparsification zeroes weights exactly, so a different choice of vertex
//! anywhere on the path would show in the zero pattern or in the step count.
//!
//! On the same path the two differ by rounding alone: R forms its products
//! through OpenBLAS, epistat in index order. A step forms sums of at most
//! `m + n` terms (`m × n` the size of the problem), whose rounding differs by
//! at most `(m + n) u` relative to the sum of the magnitudes (`u = 2^-53`), and
//! moves the weights by a convex combination. Taking each step's rounding to
//! carry undamped into the next, after `k` steps the weights, which lie in
//! `[0, 1]`, differ by at most `k (m + n) u`. Without covariates `k` counts
//! both runs of each set of weights (the first at most 100 steps); with
//! covariates each iteration takes two Frank-Wolfe steps. A step that
//! amplified rounding would make a test fail rather than pass unnoticed.
//!
//! The estimate is linear in the weights given the data: with `n0 + t0`
//! weights, `|d tau| <= 2 (n0 + t0) max|Y| |d w|_inf`, plus the rounding of
//! the final contraction, `(N + T) u max|Y|`. The covariate coefficients take
//! the estimate's tolerance. A replicate is a fit and takes the tolerance of
//! one; a standard error of `B` replicates that each differ by at most `d`
//! differs by at most `2 sqrt(B) d`.

use epistat_did::{
    StaggeredSynthPanel, SynthDidConfig, SynthDidEstimate, SynthDraws, SynthObservation,
    SynthPanel, SynthUndefined, SynthVariance, SynthWeights, estimate_synthdid,
    estimate_synthdid_staggered, synthdid_staggered_variance, synthdid_variance,
};
use epistat_testkit::{assert_same_on_threads, load_json};
use serde::Deserialize;

const U: f64 = f64::EPSILON / 2.0;

#[derive(Deserialize, Clone)]
struct Fit {
    estimate: f64,
    omega: Vec<f64>,
    lambda: Vec<f64>,
    beta: Vec<f64>,
    curve: Vec<f64>,
    zeta_omega: f64,
    zeta_lambda: f64,
    min_decrease: f64,
    lambda_steps: usize,
    omega_steps: usize,
    joint_steps: usize,
}

#[derive(Deserialize, Clone)]
struct Resampled {
    #[serde(default)]
    draws: Vec<Vec<usize>>,
    replicates: Vec<f64>,
    se: f64,
    curve_se: Vec<f64>,
}

#[derive(Deserialize)]
struct WithVariance {
    #[serde(flatten)]
    fit: Fit,
    jackknife: Option<Resampled>,
    placebo: Option<Resampled>,
    bootstrap: Option<Resampled>,
}

#[derive(Deserialize)]
struct Long {
    unit: Vec<i64>,
    time: Vec<i32>,
    outcome: Vec<f64>,
    treated: Vec<u8>,
}

#[derive(Deserialize)]
struct California {
    #[serde(rename = "Y")]
    y: Vec<Vec<f64>>,
    #[serde(rename = "N0")]
    n0: usize,
    #[serde(rename = "T0")]
    t0: usize,
    noise_level: f64,
    long: Long,
    row_units: Vec<i64>,
    synthdid: WithVariance,
    sc: WithVariance,
    did: WithVariance,
}

#[derive(Deserialize)]
struct Options {
    no_sparsify: Fit,
    no_intercepts: Fit,
    eta_omega: Fit,
    noise_level: Fit,
    max_iter: Fit,
    lambda_fixed: Fit,
    lambda_start: Fit,
    lambda_given: Vec<f64>,
}

#[derive(Deserialize)]
struct Sim {
    #[serde(rename = "Y")]
    y: Vec<Vec<f64>>,
    #[serde(rename = "N0")]
    n0: usize,
    #[serde(rename = "T0")]
    t0: usize,
    noise_level: f64,
    synthdid: WithVariance,
    sc: WithVariance,
    did: WithVariance,
    options: Options,
}

#[derive(Deserialize)]
struct Covariates {
    #[serde(rename = "Y")]
    y: Vec<Vec<f64>>,
    #[serde(rename = "X")]
    x: Vec<Vec<Vec<f64>>>,
    #[serde(rename = "N0")]
    n0: usize,
    #[serde(rename = "T0")]
    t0: usize,
    synthdid: WithVariance,
    sc: WithVariance,
    did: WithVariance,
}

#[derive(Deserialize)]
struct Cohort {
    adoption: usize,
    #[serde(flatten)]
    fit: Fit,
}

#[derive(Deserialize)]
struct Staggered {
    estimate: f64,
    event_time: Vec<f64>,
    cohorts: Vec<Cohort>,
    jackknife: Resampled,
    placebo: Resampled,
    bootstrap: Resampled,
}

#[derive(Deserialize)]
struct StaggeredData {
    #[serde(rename = "Y")]
    y: Vec<Vec<f64>>,
    adoption: Vec<Option<usize>>,
    synthdid: Staggered,
    did: Staggered,
}

#[derive(Deserialize)]
struct Reference {
    california: California,
    sim: Sim,
    covariates: Covariates,
    staggered: StaggeredData,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/synthdid_ref.json"
    ))
}

fn flat(rows: &[Vec<f64>]) -> Vec<f64> {
    rows.iter().flatten().copied().collect()
}

fn panel(y: &[Vec<f64>], n0: usize, t0: usize) -> SynthPanel {
    SynthPanel::new(flat(y), y.len(), y[0].len(), n0, t0).expect("panel")
}

fn max_abs(rows: &[Vec<f64>]) -> f64 {
    rows.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs()))
}

fn worst(got: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(got.len(), expected.len(), "lengths differ");
    got.iter()
        .zip(expected)
        .map(|(g, e)| (g - e).abs())
        .fold(0.0, f64::max)
}

/// The tolerances of the module documentation for a fit with `n0` controls,
/// `t0` pre-periods and data of magnitude `scale`, on a panel of `units ×
/// periods`.
struct Tol {
    weights: f64,
    estimate: f64,
}

fn tolerances(fit: &Fit, n0: usize, t0: usize, scale: f64, units: usize, periods: usize) -> Tol {
    let steps = (fit.lambda_steps + fit.omega_steps + 2 * fit.joint_steps + 200) as f64;
    let size = (n0 + t0) as f64;
    let weights = steps * size * U;
    let estimate = (2.0 * size * weights + (units + periods) as f64 * U) * scale;
    Tol { weights, estimate }
}

/// The tolerance of a replicate refit, whose step counts R does not keep: the
/// most the fit's runs can take, `max.iter + max.iter.pre.sparsify` = 10100
/// for each set of weights that is estimated, 10000 iterations with
/// covariates.
fn replicate_tolerance(
    fit: &Fit,
    n0: usize,
    t0: usize,
    scale: f64,
    units: usize,
    periods: usize,
) -> f64 {
    let most = |steps: usize, limit: usize| if steps > 0 { limit } else { 0 };
    let worst = Fit {
        lambda_steps: most(fit.lambda_steps, 10_100),
        omega_steps: most(fit.omega_steps, 10_100),
        joint_steps: most(fit.joint_steps, 10_000),
        ..fit.clone()
    };
    tolerances(&worst, n0, t0, scale, units, periods).estimate
}

/// The tolerance of a standard error over `b` replicates.
fn se_tolerance(b: usize, replicate: f64) -> f64 {
    2.0 * (b as f64).sqrt() * replicate
}

fn check_fit(name: &str, got: &SynthDidEstimate, want: &Fit, tol: &Tol) {
    assert_eq!(
        got.trace.lambda.len(),
        want.lambda_steps,
        "{name}: lambda steps"
    );
    assert_eq!(
        got.trace.omega.len(),
        want.omega_steps,
        "{name}: omega steps"
    );
    assert_eq!(
        got.trace.joint.len(),
        want.joint_steps,
        "{name}: joint steps"
    );
    let dz = (got.zeta_omega - want.zeta_omega).abs() / want.zeta_omega.abs().max(1.0);
    assert!(
        dz < 1e-13,
        "{name}: zeta_omega {} vs {}",
        got.zeta_omega,
        want.zeta_omega
    );
    let dz = (got.zeta_lambda - want.zeta_lambda).abs() / want.zeta_lambda.abs().max(1e-300);
    assert!(
        dz < 1e-13,
        "{name}: zeta_lambda {} vs {}",
        got.zeta_lambda,
        want.zeta_lambda
    );
    let dm = (got.min_decrease - want.min_decrease).abs() / want.min_decrease;
    assert!(dm < 1e-13, "{name}: min_decrease");
    let zeros = |v: &[f64]| v.iter().map(|&x| x == 0.0).collect::<Vec<_>>();
    assert_eq!(
        zeros(&got.omega),
        zeros(&want.omega),
        "{name}: zero unit weights"
    );
    assert_eq!(
        zeros(&got.lambda),
        zeros(&want.lambda),
        "{name}: zero time weights"
    );
    let dw = worst(&got.omega, &want.omega).max(worst(&got.lambda, &want.lambda));
    let db = worst(&got.beta, &want.beta);
    let de = (got.estimate - want.estimate)
        .abs()
        .max(worst(&got.effect_curve, &want.curve));
    eprintln!(
        "{name}: weights {dw:.2e} (tol {:.2e}), beta {db:.2e}, estimate {de:.2e} (tol {:.2e})",
        tol.weights, tol.estimate
    );
    assert!(dw <= tol.weights, "{name}: weights differ by {dw:e}");
    assert!(db <= tol.estimate, "{name}: beta differs by {db:e}");
    assert!(de <= tol.estimate, "{name}: estimate differs by {de:e}");
}

fn check_replicates(name: &str, got: &epistat_did::SynthDidVariance, want: &Resampled, tol: f64) {
    let du = worst(&got.replicates, &want.replicates);
    let se = got.se.expect("defined");
    let ds = (se - want.se).abs();
    let curve: Vec<f64> = got
        .effect_curve_se
        .iter()
        .map(|v| v.expect("defined"))
        .collect();
    let dc = worst(&curve, &want.curve_se);
    let tol_se = se_tolerance(want.replicates.len(), tol);
    eprintln!(
        "{name}: replicates {du:.2e} (tol {tol:.2e}), se {ds:.2e}, curve se {dc:.2e} (tol {tol_se:.2e})"
    );
    assert!(du <= tol, "{name}: replicates differ by {du:e}");
    assert!(ds <= tol_se, "{name}: se {se} vs {}", want.se);
    assert!(dc <= tol_se, "{name}: curve se differs by {dc:e}");
}

fn config(name: &str) -> SynthDidConfig {
    match name {
        "synthdid" => SynthDidConfig::synthdid(),
        "sc" => SynthDidConfig::synthetic_control(),
        "did" => SynthDidConfig::did(),
        _ => unreachable!(),
    }
}

#[test]
fn california_matches_synthdid() {
    let r = reference().california;
    let p = panel(&r.y, r.n0, r.t0);
    let scale = max_abs(&r.y);
    for (name, want) in [("synthdid", &r.synthdid), ("sc", &r.sc), ("did", &r.did)] {
        let est = estimate_synthdid(&p, &config(name)).expect("estimate");
        assert!((est.noise_level - r.noise_level).abs() <= 1e-13 * r.noise_level);
        let tol = tolerances(&want.fit, r.n0, r.t0, scale, r.y.len(), r.y[0].len());
        check_fit(&format!("california {name}"), &est, &want.fit, &tol);
        // One treated unit: synthdid's jackknife and bootstrap are NA.
        let jk = synthdid_variance(&p, &est, SynthVariance::Jackknife { clusters: None }).unwrap();
        assert_eq!(jk.undefined, Some(SynthUndefined::OneTreatedUnit));
        assert!(jk.se.is_none());
        let draws = [vec![0; 39], vec![0; 39]];
        let bs = synthdid_variance(
            &p,
            &est,
            SynthVariance::Bootstrap(SynthDraws::Given(&draws)),
        )
        .unwrap();
        assert_eq!(bs.undefined, Some(SynthUndefined::OneTreatedUnit));
        let placebo = want.placebo.as_ref().expect("placebo");
        let got = synthdid_variance(
            &p,
            &est,
            SynthVariance::Placebo(SynthDraws::Given(&placebo.draws)),
        )
        .expect("placebo");
        let tol = replicate_tolerance(&want.fit, r.n0, r.t0, scale, r.y.len(), r.y[0].len());
        check_replicates(&format!("california {name} placebo"), &got, placebo, tol);
    }
}

#[test]
fn california_long_data_gives_the_same_panel() {
    let r = reference().california;
    let rows: Vec<SynthObservation> = (0..r.long.unit.len())
        .map(|i| SynthObservation {
            unit: r.long.unit[i],
            time: r.long.time[i],
            outcome: r.long.outcome[i],
            treated: r.long.treated[i] == 1,
            covariates: Vec::new(),
        })
        .collect();
    let p = SynthPanel::from_observations(&rows).expect("panel");
    assert_eq!(p.unit_ids(), &r.row_units[..]);
    assert_eq!((p.controls(), p.pre_periods()), (r.n0, r.t0));
    assert_eq!(p.outcome(), &flat(&r.y)[..]);
}

#[test]
fn simulated_panel_matches_synthdid_with_every_variance() {
    let r = reference().sim;
    let p = panel(&r.y, r.n0, r.t0);
    let scale = max_abs(&r.y);
    for (name, want) in [("synthdid", &r.synthdid), ("sc", &r.sc), ("did", &r.did)] {
        let est = estimate_synthdid(&p, &config(name)).expect("estimate");
        assert!((est.noise_level - r.noise_level).abs() <= 1e-13 * r.noise_level);
        let tol = tolerances(&want.fit, r.n0, r.t0, scale, r.y.len(), r.y[0].len());
        check_fit(&format!("sim {name}"), &est, &want.fit, &tol);
        let jk = synthdid_variance(&p, &est, SynthVariance::Jackknife { clusters: None }).unwrap();
        check_replicates(
            &format!("sim {name} jackknife"),
            &jk,
            want.jackknife.as_ref().unwrap(),
            tol.estimate,
        );
        // The influence values reproduce the jackknife through the crate's
        // cluster variance with one cluster per unit.
        let influence = jk.influence.as_ref().unwrap();
        let se = epistat_did::did_standard_error(
            influence,
            &epistat_inference::Clusters::singletons(influence.len()),
        )
        .unwrap();
        assert!(
            (se - jk.se.unwrap()).abs() <= 1e-12 * se,
            "{se} vs {:?}",
            jk.se
        );
        let refit_tol = replicate_tolerance(&want.fit, r.n0, r.t0, scale, r.y.len(), r.y[0].len());
        for (method, want) in [("placebo", &want.placebo), ("bootstrap", &want.bootstrap)] {
            let want = want.as_ref().unwrap();
            let draws = SynthDraws::Given(&want.draws);
            let m = if method == "placebo" {
                SynthVariance::Placebo(draws)
            } else {
                SynthVariance::Bootstrap(draws)
            };
            let got = synthdid_variance(&p, &est, m).expect("variance");
            check_replicates(&format!("sim {name} {method}"), &got, want, refit_tol);
        }
    }
}

#[test]
fn options_of_synthdid_estimate_match() {
    let r = reference().sim;
    let p = panel(&r.y, r.n0, r.t0);
    let scale = max_abs(&r.y);
    let o = &r.options;
    let base = SynthDidConfig::synthdid;
    let cases: [(&str, SynthDidConfig, &Fit); 7] = [
        (
            "no sparsify",
            SynthDidConfig {
                sparsify: false,
                ..base()
            },
            &o.no_sparsify,
        ),
        (
            "no intercepts",
            SynthDidConfig {
                omega_intercept: false,
                lambda_intercept: false,
                ..base()
            },
            &o.no_intercepts,
        ),
        (
            "eta omega",
            SynthDidConfig {
                eta_omega: Some(0.3),
                ..base()
            },
            &o.eta_omega,
        ),
        (
            "noise level",
            SynthDidConfig {
                noise_level: Some(2.0),
                ..base()
            },
            &o.noise_level,
        ),
        (
            "max iter",
            SynthDidConfig {
                max_iter: 50,
                max_iter_pre_sparsify: 10,
                ..base()
            },
            &o.max_iter,
        ),
        (
            "lambda fixed",
            SynthDidConfig {
                lambda: SynthWeights::Fixed(o.lambda_given.clone()),
                ..base()
            },
            &o.lambda_fixed,
        ),
        (
            "lambda start",
            SynthDidConfig {
                lambda: SynthWeights::Start(o.lambda_given.clone()),
                ..base()
            },
            &o.lambda_start,
        ),
    ];
    for (name, config, want) in cases {
        let est = estimate_synthdid(&p, &config).expect("estimate");
        check_fit(
            name,
            &est,
            want,
            &tolerances(want, r.n0, r.t0, scale, r.y.len(), r.y[0].len()),
        );
    }
}

fn covariate_panel(r: &Covariates) -> SynthPanel {
    let x: Vec<f64> = r.x.iter().flat_map(|block| flat(block)).collect();
    panel(&r.y, r.n0, r.t0)
        .with_covariates(x, r.x.len())
        .expect("covariates")
}

#[test]
fn covariates_match_synthdid() {
    let r = reference().covariates;
    let p = covariate_panel(&r);
    let scale = max_abs(&r.y).max(r.x.iter().map(|b| max_abs(b)).fold(0.0, f64::max));
    for (name, want) in [("synthdid", &r.synthdid), ("sc", &r.sc), ("did", &r.did)] {
        let est = estimate_synthdid(&p, &config(name)).expect("estimate");
        let tol = tolerances(&want.fit, r.n0, r.t0, scale, r.y.len(), r.y[0].len());
        check_fit(&format!("covariates {name}"), &est, &want.fit, &tol);
        if name != "synthdid" {
            continue;
        }
        let jk = synthdid_variance(&p, &est, SynthVariance::Jackknife { clusters: None }).unwrap();
        let refit_tol = replicate_tolerance(&want.fit, r.n0, r.t0, scale, r.y.len(), r.y[0].len());
        check_replicates(
            "covariates jackknife",
            &jk,
            want.jackknife.as_ref().unwrap(),
            refit_tol,
        );
        let pl = want.placebo.as_ref().unwrap();
        let got = synthdid_variance(
            &p,
            &est,
            SynthVariance::Placebo(SynthDraws::Given(&pl.draws)),
        )
        .unwrap();
        check_replicates("covariates placebo", &got, pl, refit_tol);
        let bs = want.bootstrap.as_ref().unwrap();
        let got = synthdid_variance(
            &p,
            &est,
            SynthVariance::Bootstrap(SynthDraws::Given(&bs.draws)),
        )
        .unwrap();
        check_replicates("covariates bootstrap", &got, bs, refit_tol);
    }
}

#[test]
fn staggered_adoption_matches_the_cohort_by_cohort_reference() {
    let r = reference().staggered;
    let n = r.y.len();
    let periods = r.y[0].len();
    let p = StaggeredSynthPanel::new(flat(&r.y), n, periods, r.adoption.clone()).expect("panel");
    let scale = max_abs(&r.y);
    for (name, want) in [("synthdid", &r.synthdid), ("did", &r.did)] {
        let est = estimate_synthdid_staggered(&p, &config(name)).expect("estimate");
        assert_eq!(est.cohorts.len(), want.cohorts.len());
        let mut tol = 0.0_f64;
        let mut refit_tol = 0.0_f64;
        for (got, cohort) in est.cohorts.iter().zip(&want.cohorts) {
            assert_eq!(got.adoption, cohort.adoption);
            let t = tolerances(
                &cohort.fit,
                est.controls.len(),
                cohort.adoption,
                scale,
                n,
                periods,
            );
            check_fit(
                &format!("staggered {name} cohort {}", cohort.adoption),
                &got.fit,
                &cohort.fit,
                &t,
            );
            tol = tol.max(t.estimate);
            let controls = est.controls.len();
            refit_tol = refit_tol.max(replicate_tolerance(
                &cohort.fit,
                controls,
                cohort.adoption,
                scale,
                n,
                periods,
            ));
        }
        assert!(
            (est.estimate - want.estimate).abs() <= tol,
            "{name}: aggregate"
        );
        assert!(
            worst(&est.event_time, &want.event_time) <= tol,
            "{name}: event time"
        );
        let jk = synthdid_staggered_variance(&p, &est, SynthVariance::Jackknife { clusters: None })
            .unwrap();
        check_replicates(
            &format!("staggered {name} jackknife"),
            &jk,
            &want.jackknife,
            tol,
        );
        let pl = synthdid_staggered_variance(
            &p,
            &est,
            SynthVariance::Placebo(SynthDraws::Given(&want.placebo.draws)),
        )
        .unwrap();
        check_replicates(
            &format!("staggered {name} placebo"),
            &pl,
            &want.placebo,
            refit_tol,
        );
        let bs = synthdid_staggered_variance(
            &p,
            &est,
            SynthVariance::Bootstrap(SynthDraws::Given(&want.bootstrap.draws)),
        )
        .unwrap();
        check_replicates(
            &format!("staggered {name} bootstrap"),
            &bs,
            &want.bootstrap,
            refit_tol,
        );
    }
}

/// One cohort through the staggered route is the single-date estimate, bit
/// for bit, and so are its standard errors.
#[test]
fn one_cohort_is_the_single_date_estimate() {
    let r = reference().sim;
    let p = panel(&r.y, r.n0, r.t0);
    let n = r.y.len();
    let adoption: Vec<Option<usize>> = (0..n).map(|i| (i >= r.n0).then_some(r.t0)).collect();
    let s = StaggeredSynthPanel::new(flat(&r.y), n, r.y[0].len(), adoption).unwrap();
    let single = estimate_synthdid(&p, &SynthDidConfig::synthdid()).unwrap();
    let staggered = estimate_synthdid_staggered(&s, &SynthDidConfig::synthdid()).unwrap();
    assert_eq!(single.estimate.to_bits(), staggered.estimate.to_bits());
    assert_eq!(single.effect_curve, staggered.event_time);
    let method = SynthVariance::Placebo(SynthDraws::Seeded {
        replications: 20,
        seed: 7,
    });
    let a = synthdid_variance(&p, &single, method).unwrap();
    let b = synthdid_staggered_variance(&s, &staggered, method).unwrap();
    assert_eq!(a, b);
}

/// The seeded placebo and bootstrap against R's own, which used R's
/// generator: the two are independent estimates of the same standard error.
/// With `B` replicates of kurtosis `k`, the sample standard deviation has
/// relative standard error `sqrt((k - 1) / (4 B))` to first order, so the
/// difference of two independent ones has standard deviation `sqrt(2)` times
/// that, relative to the standard error. The test allows four of those, with
/// `k` estimated from R's replicates.
#[test]
fn seeded_draws_agree_with_r_within_monte_carlo_error() {
    let r = reference().sim;
    let p = panel(&r.y, r.n0, r.t0);
    let est = estimate_synthdid(&p, &SynthDidConfig::synthdid()).unwrap();
    for method in ["placebo", "bootstrap"] {
        let want = if method == "placebo" {
            r.synthdid.placebo.as_ref().unwrap()
        } else {
            r.synthdid.bootstrap.as_ref().unwrap()
        };
        let b = want.replicates.len();
        let draws = SynthDraws::Seeded {
            replications: b,
            seed: 20_261_125,
        };
        let m = if method == "placebo" {
            SynthVariance::Placebo(draws)
        } else {
            SynthVariance::Bootstrap(draws)
        };
        let got = synthdid_variance(&p, &est, m).unwrap();
        let u = &want.replicates;
        let mean = u.iter().sum::<f64>() / b as f64;
        let m2 = u.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / b as f64;
        let m4 = u.iter().map(|v| (v - mean).powi(4)).sum::<f64>() / b as f64;
        let kurtosis = m4 / (m2 * m2);
        let relative = ((kurtosis - 1.0) / (4.0 * b as f64)).sqrt() * 2.0_f64.sqrt();
        let se = got.se.unwrap();
        eprintln!(
            "{method}: epistat {se:.4}, R {:.4}, allowed {:.4}",
            want.se,
            4.0 * relative * want.se
        );
        assert!((se - want.se).abs() <= 4.0 * relative * want.se, "{method}");
    }
}

#[test]
fn seeded_variance_does_not_depend_on_threads() {
    let r = reference().sim;
    let p = panel(&r.y, r.n0, r.t0);
    let est = estimate_synthdid(&p, &SynthDidConfig::synthdid()).unwrap();
    assert_same_on_threads(|| {
        let draws = SynthDraws::Seeded {
            replications: 16,
            seed: 11,
        };
        let a = synthdid_variance(&p, &est, SynthVariance::Placebo(draws)).unwrap();
        let b = synthdid_variance(&p, &est, SynthVariance::Bootstrap(draws)).unwrap();
        let mut out = a.replicates;
        out.extend(b.replicates);
        out.push(
            estimate_synthdid(&p, &SynthDidConfig::synthdid())
                .unwrap()
                .estimate,
        );
        out
    });
}

/// The cluster jackknife with one unit per cluster is the unit jackknife.
#[test]
fn singleton_clusters_are_the_unit_jackknife() {
    let r = reference().sim;
    let p = panel(&r.y, r.n0, r.t0);
    let est = estimate_synthdid(&p, &SynthDidConfig::synthdid()).unwrap();
    let clusters = epistat_inference::Clusters::singletons(r.y.len());
    let a = synthdid_variance(&p, &est, SynthVariance::Jackknife { clusters: None }).unwrap();
    let b = synthdid_variance(
        &p,
        &est,
        SynthVariance::Jackknife {
            clusters: Some(&clusters),
        },
    )
    .unwrap();
    assert_eq!(a.se, b.se);
    assert_eq!(a.replicates, b.replicates);
    assert!(b.influence.is_none());
}

#[test]
fn refuses_what_it_cannot_estimate() {
    use epistat_did::SynthDidError;
    let r = reference().sim;
    let y = flat(&r.y);
    let (n, t) = (r.y.len(), r.y[0].len());
    assert_eq!(
        SynthPanel::new(y.clone(), n, t, n, r.t0).unwrap_err(),
        SynthDidError::Design
    );
    assert_eq!(
        SynthPanel::new(y.clone(), n, t, r.n0, 0).unwrap_err(),
        SynthDidError::Design
    );
    // One pre-period leaves no first difference for the noise level.
    let one = SynthPanel::new(y.clone(), n, t, r.n0, 1).unwrap();
    assert_eq!(
        estimate_synthdid(&one, &SynthDidConfig::synthdid()).unwrap_err(),
        SynthDidError::NoiseLevel
    );
    assert!(
        estimate_synthdid(
            &one,
            &SynthDidConfig {
                noise_level: Some(1.0),
                ..SynthDidConfig::synthdid()
            }
        )
        .is_ok()
    );
    let p = panel(&r.y, r.n0, r.t0);
    let est = estimate_synthdid(&p, &SynthDidConfig::synthdid()).unwrap();
    let bad = [vec![0; r.n0]];
    assert_eq!(
        synthdid_variance(&p, &est, SynthVariance::Placebo(SynthDraws::Given(&bad))).unwrap_err(),
        SynthDidError::Replications
    );
    let bad = [(0..r.n0).collect::<Vec<_>>(), vec![0; r.n0]];
    assert_eq!(
        synthdid_variance(&p, &est, SynthVariance::Placebo(SynthDraws::Given(&bad))).unwrap_err(),
        SynthDidError::Draw(1)
    );
    // A bootstrap draw of controls only is refused rather than redrawn.
    let bad = [vec![0; n], vec![0; n]];
    assert_eq!(
        synthdid_variance(&p, &est, SynthVariance::Bootstrap(SynthDraws::Given(&bad))).unwrap_err(),
        SynthDidError::Draw(0)
    );
    // Ten treated units and twelve controls: too few controls for the placebo.
    let small = SynthPanel::new(y[(r.n0 - 12) * t..].to_vec(), n - r.n0 + 12, t, 12, r.t0).unwrap();
    let est = estimate_synthdid(&small, &SynthDidConfig::did()).unwrap();
    let ok = synthdid_variance(
        &small,
        &est,
        SynthVariance::Placebo(SynthDraws::Seeded {
            replications: 2,
            seed: 1,
        }),
    );
    assert!(ok.is_ok());
    let tiny = SynthPanel::new(y[(r.n0 - 10) * t..].to_vec(), n - r.n0 + 10, t, 10, r.t0).unwrap();
    let est = estimate_synthdid(&tiny, &SynthDidConfig::did()).unwrap();
    assert_eq!(
        synthdid_variance(
            &tiny,
            &est,
            SynthVariance::Placebo(SynthDraws::Seeded {
                replications: 2,
                seed: 1
            })
        )
        .unwrap_err(),
        SynthDidError::PlaceboControls
    );
    // An estimate of another panel.
    assert_eq!(
        synthdid_variance(&p, &est, SynthVariance::Jackknife { clusters: None }).unwrap_err(),
        SynthDidError::Mismatch
    );
    // Weight vectors do not carry across cohorts.
    let s = reference().staggered;
    let sp = StaggeredSynthPanel::new(flat(&s.y), s.y.len(), s.y[0].len(), s.adoption).unwrap();
    let config = SynthDidConfig {
        lambda: SynthWeights::Start(vec![1.0]),
        ..SynthDidConfig::synthdid()
    };
    assert!(matches!(
        estimate_synthdid_staggered(&sp, &config).unwrap_err(),
        SynthDidError::Setting(_)
    ));
}

#[test]
fn staggered_long_data_and_clusters() {
    let s = reference().staggered;
    let (n, t) = (s.y.len(), s.y[0].len());
    let rows: Vec<SynthObservation> = (0..n)
        .flat_map(|i| {
            let (y, a) = (&s.y[i], s.adoption[i]);
            (0..t).map(move |j| SynthObservation {
                unit: 100 + i64::try_from(i).unwrap(),
                time: 2000 + i32::try_from(j).unwrap(),
                outcome: y[j],
                treated: a.is_some_and(|a| j >= a),
                covariates: Vec::new(),
            })
        })
        .collect();
    let p = StaggeredSynthPanel::from_observations(&rows).unwrap();
    assert_eq!(p.adoption(), &s.adoption[..]);
    assert_eq!(p.times()[0], 2000);
    let est = estimate_synthdid_staggered(&p, &SynthDidConfig::synthdid()).unwrap();
    assert!((est.estimate - s.synthdid.estimate).abs() < 1e-12);
    // Pairs of units as clusters. The units are independent, so the cluster
    // jackknife estimates the same standard error as the unit jackknife; half
    // of it is far outside the error of either estimate.
    let labels: Vec<usize> = (0..n).map(|i| i / 2).collect();
    let clusters = epistat_inference::Clusters::from_labels(&labels);
    let jk = synthdid_staggered_variance(
        &p,
        &est,
        SynthVariance::Jackknife {
            clusters: Some(&clusters),
        },
    )
    .unwrap();
    assert_eq!(jk.replicates.len(), clusters.count());
    let se = jk.se.expect("defined");
    assert!(se.is_finite() && se > 0.5 * s.synthdid.jackknife.se, "{se}");
    // A treated unit treated in its first period is refused.
    let mut first = rows.clone();
    let victim = s.adoption.iter().position(Option::is_some).unwrap();
    first[victim * t].treated = true;
    assert!(StaggeredSynthPanel::from_observations(&first).is_err());
}
