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

use epistat_core::{Bootstrap, Codes, Estimate, Panel};
use epistat_did::{
    Error, StaggeredPanel, SynthDid, SynthDidOptions, SynthDraws, SynthFit, SynthReplicates,
    SynthUndefined, SynthVariance, SynthWeights, synthdid,
};
use epistat_linalg::Mat;
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

/// A balanced panel in long form: unit `i` is row `i` of `y`, observed in
/// periods `0..T`, first treated in period `adoption[i]`.
struct Columns {
    unit: Vec<u32>,
    period: Vec<i64>,
    first_treated: Vec<Option<i64>>,
    outcome: Vec<f64>,
    covariates: Option<Mat<f64>>,
    clusters: Option<Vec<u32>>,
}

impl Columns {
    fn new(y: &[Vec<f64>], adoption: &[Option<usize>]) -> Self {
        let t = y[0].len();
        let rows = y.len() * t;
        let unit = (0..rows)
            .map(|r| u32::try_from(r / t).expect("unit"))
            .collect();
        let period = (0..rows)
            .map(|r| i64::try_from(r % t).expect("period"))
            .collect();
        let first_treated = (0..rows)
            .map(|r| adoption[r / t].map(|a| i64::try_from(a).expect("period")))
            .collect();
        Self {
            unit,
            period,
            first_treated,
            outcome: y.concat(),
            covariates: None,
            clusters: None,
        }
    }

    /// The single-adoption panel of `synthdid`: the first `n0` rows of `y`
    /// are controls, the others treated from period `t0` on.
    fn single(y: &[Vec<f64>], n0: usize, t0: usize) -> Self {
        let adoption: Vec<Option<usize>> = (0..y.len()).map(|i| (i >= n0).then_some(t0)).collect();
        Self::new(y, &adoption)
    }

    /// The covariates `x[k][unit][period]`.
    fn with_covariates(mut self, x: &[Vec<Vec<f64>>]) -> Self {
        let t = x[0][0].len();
        self.covariates = Some(Mat::from_fn(self.outcome.len(), x.len(), |r, k| {
            x[k][r / t][r % t]
        }));
        self
    }

    /// One cluster code per unit, `cluster(unit)`.
    fn with_clusters(mut self, cluster: impl Fn(u32) -> u32) -> Self {
        self.clusters = Some(self.unit.iter().map(|&u| cluster(u)).collect());
        self
    }

    fn panel(&self) -> StaggeredPanel<'_> {
        let mut b =
            StaggeredPanel::builder(&self.unit, &self.period, &self.first_treated, &self.outcome);
        if let Some(x) = &self.covariates {
            b = b.covariates(x.as_ref());
        }
        if let Some(c) = &self.clusters {
            b = b.clusters(c);
        }
        b.build().expect("panel")
    }
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

/// The parts of a fit that the reference holds.
struct Got<'a> {
    estimate: f64,
    curve: Vec<f64>,
    fit: &'a SynthFit,
}

impl<'a> Got<'a> {
    fn single(e: &'a SynthDid) -> Self {
        Self {
            estimate: e.estimate.estimate,
            curve: e.effect_curve.iter().map(|e| e.estimate).collect(),
            fit: &e.fit,
        }
    }
}

fn check_fit(name: &str, got: &Got<'_>, want: &Fit, tol: &Tol) {
    let fit = got.fit;
    assert_eq!(
        fit.trace.lambda.len(),
        want.lambda_steps,
        "{name}: lambda steps"
    );
    assert_eq!(
        fit.trace.omega.len(),
        want.omega_steps,
        "{name}: omega steps"
    );
    assert_eq!(
        fit.trace.joint.len(),
        want.joint_steps,
        "{name}: joint steps"
    );
    let dz = (fit.zeta_omega - want.zeta_omega).abs() / want.zeta_omega.abs().max(1.0);
    assert!(
        dz < 1e-13,
        "{name}: zeta_omega {} vs {}",
        fit.zeta_omega,
        want.zeta_omega
    );
    let dz = (fit.zeta_lambda - want.zeta_lambda).abs() / want.zeta_lambda.abs().max(1e-300);
    assert!(
        dz < 1e-13,
        "{name}: zeta_lambda {} vs {}",
        fit.zeta_lambda,
        want.zeta_lambda
    );
    let dm = (fit.min_decrease - want.min_decrease).abs() / want.min_decrease;
    assert!(dm < 1e-13, "{name}: min_decrease");
    let zeros = |v: &[f64]| v.iter().map(|&x| x == 0.0).collect::<Vec<_>>();
    assert_eq!(
        zeros(&fit.omega),
        zeros(&want.omega),
        "{name}: zero unit weights"
    );
    assert_eq!(
        zeros(&fit.lambda),
        zeros(&want.lambda),
        "{name}: zero time weights"
    );
    let dw = worst(&fit.omega, &want.omega).max(worst(&fit.lambda, &want.lambda));
    let db = worst(&fit.beta, &want.beta);
    let de = (got.estimate - want.estimate)
        .abs()
        .max(worst(&got.curve, &want.curve));
    eprintln!(
        "{name}: weights {dw:.2e} (tol {:.2e}), beta {db:.2e}, estimate {de:.2e} (tol {:.2e})",
        tol.weights, tol.estimate
    );
    assert!(dw <= tol.weights, "{name}: weights differ by {dw:e}");
    assert!(db <= tol.estimate, "{name}: beta differs by {db:e}");
    assert!(de <= tol.estimate, "{name}: estimate differs by {de:e}");
}

/// A standard error with its curve and replicates.
struct Se<'a> {
    value: Option<f64>,
    curve: Vec<Option<f64>>,
    replicates: &'a SynthReplicates,
}

impl<'a> Se<'a> {
    fn of(
        estimate: &Estimate,
        curve: &[Estimate],
        replicates: Option<&'a SynthReplicates>,
    ) -> Self {
        Self {
            value: estimate.se,
            curve: curve.iter().map(|e| e.se).collect(),
            replicates: replicates.expect("a variance"),
        }
    }

    fn single(e: &'a SynthDid) -> Self {
        Self::of(&e.estimate, &e.effect_curve, e.variance.as_ref())
    }

    fn staggered(e: &'a epistat_did::StaggeredSynthDid) -> Self {
        Self::of(&e.estimate, &e.event_time, e.variance.as_ref())
    }
}

fn check_replicates(name: &str, got: &Se<'_>, want: &Resampled, tol: f64) {
    let du = worst(&got.replicates.replicates, &want.replicates);
    let se = got.value.expect("defined");
    let ds = (se - want.se).abs();
    let curve: Vec<f64> = got.curve.iter().map(|v| v.expect("defined")).collect();
    let dc = worst(&curve, &want.curve_se);
    let tol_se = se_tolerance(want.replicates.len(), tol);
    eprintln!(
        "{name}: replicates {du:.2e} (tol {tol:.2e}), se {ds:.2e}, curve se {dc:.2e} (tol {tol_se:.2e})"
    );
    assert!(du <= tol, "{name}: replicates differ by {du:e}");
    assert!(ds <= tol_se, "{name}: se {se} vs {}", want.se);
    assert!(dc <= tol_se, "{name}: curve se differs by {dc:e}");
}

fn config(name: &str) -> SynthDidOptions {
    match name {
        "synthdid" => SynthDidOptions::default(),
        "sc" => SynthDidOptions::synthetic_control(),
        "did" => SynthDidOptions::did(),
        _ => unreachable!(),
    }
}

/// `options` with the standard error `variance`.
fn with(options: &SynthDidOptions, variance: SynthVariance) -> SynthDidOptions {
    SynthDidOptions {
        variance,
        ..options.clone()
    }
}

fn given(method: &str, draws: &[Vec<usize>]) -> SynthVariance {
    let draws = SynthDraws::Given(draws.to_vec());
    if method == "placebo" {
        SynthVariance::Placebo(draws)
    } else {
        SynthVariance::Bootstrap(draws)
    }
}

fn seeded(method: &str, replicates: usize, seed: u64) -> SynthVariance {
    let draws = SynthDraws::Seeded(Bootstrap::new(replicates, seed));
    if method == "placebo" {
        SynthVariance::Placebo(draws)
    } else {
        SynthVariance::Bootstrap(draws)
    }
}

#[test]
fn california_matches_synthdid() {
    let r = reference().california;
    let columns = Columns::single(&r.y, r.n0, r.t0);
    let p = columns.panel();
    let scale = max_abs(&r.y);
    for (name, want) in [("synthdid", &r.synthdid), ("sc", &r.sc), ("did", &r.did)] {
        let options = config(name);
        let est = synthdid::estimate(&p, &options).expect("estimate");
        assert!((est.fit.noise_level - r.noise_level).abs() <= 1e-13 * r.noise_level);
        let tol = tolerances(&want.fit, r.n0, r.t0, scale, r.y.len(), r.y[0].len());
        check_fit(
            &format!("california {name}"),
            &Got::single(&est),
            &want.fit,
            &tol,
        );
        // One treated unit: synthdid's jackknife and bootstrap are NA.
        let jk = synthdid::estimate(&p, &with(&options, SynthVariance::Jackknife)).unwrap();
        let replicates = jk.variance.as_ref().expect("variance");
        assert_eq!(replicates.undefined, Some(SynthUndefined::OneTreatedUnit));
        assert!(jk.estimate.se.is_none() && jk.estimate.interval.is_none());
        let draws = [vec![0; 39], vec![0; 39]];
        let bs = synthdid::estimate(&p, &with(&options, given("bootstrap", &draws))).unwrap();
        assert_eq!(
            bs.variance.as_ref().and_then(|v| v.undefined),
            Some(SynthUndefined::OneTreatedUnit)
        );
        let placebo = want.placebo.as_ref().expect("placebo");
        let got = synthdid::estimate(&p, &with(&options, given("placebo", &placebo.draws)))
            .expect("placebo");
        let tol = replicate_tolerance(&want.fit, r.n0, r.t0, scale, r.y.len(), r.y[0].len());
        check_replicates(
            &format!("california {name} placebo"),
            &Se::single(&got),
            placebo,
            tol,
        );
    }
}

/// The long data of `panel.matrices`, with the states' names as labels,
/// give the estimate of the matrix: the same units in the same roles, and
/// the same bits, since the fits read the controls and the treated units
/// each in increasing code.
#[test]
fn california_long_data_gives_the_same_estimate() {
    let r = reference().california;
    let names = Codes::from_labels(&r.long.unit).expect("labels");
    let order: Vec<usize> = Panel::sort_order(
        names.codes(),
        &r.long
            .time
            .iter()
            .map(|&t| i64::from(t))
            .collect::<Vec<_>>(),
    )
    .expect("order")
    .into_iter()
    .map(|i| usize::try_from(i).expect("row"))
    .collect();
    let unit: Vec<u32> = order.iter().map(|&i| names.codes()[i]).collect();
    let period: Vec<i64> = order.iter().map(|&i| i64::from(r.long.time[i])).collect();
    let outcome: Vec<f64> = order.iter().map(|&i| r.long.outcome[i]).collect();
    let first_year = r
        .long
        .time
        .iter()
        .zip(&r.long.treated)
        .filter(|&(_, &d)| d == 1)
        .map(|(&t, _)| i64::from(t))
        .min()
        .expect("a treated row");
    let treated_unit: Vec<bool> = order
        .iter()
        .map(|&i| {
            let u = r.long.unit[i];
            r.long
                .unit
                .iter()
                .zip(&r.long.treated)
                .any(|(&v, &d)| v == u && d == 1)
        })
        .collect();
    let first_treated: Vec<Option<i64>> = treated_unit
        .iter()
        .map(|&d| d.then_some(first_year))
        .collect();
    let long = StaggeredPanel::builder(&unit, &period, &first_treated, &outcome)
        .build()
        .expect("panel");
    let est = synthdid::estimate(&long, &SynthDidOptions::default()).expect("estimate");
    let labels = |codes: &[u32]| -> Vec<i64> {
        codes
            .iter()
            .map(|&c| *names.label(c).expect("code"))
            .collect()
    };
    assert_eq!(labels(&est.controls), r.row_units[..r.n0].to_vec());
    assert_eq!(labels(&est.treated), r.row_units[r.n0..].to_vec());
    let columns = Columns::single(&r.y, r.n0, r.t0);
    let matrix = synthdid::estimate(&columns.panel(), &SynthDidOptions::default()).unwrap();
    assert_eq!(est.estimate, matrix.estimate);
    assert_eq!(est.fit, matrix.fit);
}

#[test]
fn simulated_panel_matches_synthdid_with_every_variance() {
    let r = reference().sim;
    let columns = Columns::single(&r.y, r.n0, r.t0);
    let p = columns.panel();
    let scale = max_abs(&r.y);
    for (name, want) in [("synthdid", &r.synthdid), ("sc", &r.sc), ("did", &r.did)] {
        let options = config(name);
        let est =
            synthdid::estimate(&p, &with(&options, SynthVariance::Jackknife)).expect("estimate");
        assert!((est.fit.noise_level - r.noise_level).abs() <= 1e-13 * r.noise_level);
        let tol = tolerances(&want.fit, r.n0, r.t0, scale, r.y.len(), r.y[0].len());
        check_fit(&format!("sim {name}"), &Got::single(&est), &want.fit, &tol);
        check_replicates(
            &format!("sim {name} jackknife"),
            &Se::single(&est),
            want.jackknife.as_ref().unwrap(),
            tol.estimate,
        );
        // The influence values reproduce the jackknife through the shared
        // variance with one cluster per unit.
        let influence = est
            .variance
            .as_ref()
            .and_then(|v| v.influence.as_ref())
            .expect("influence");
        let se = influence
            .variance(None)
            .expect("variance")
            .get(0, 0)
            .expect("defined")
            .sqrt();
        let jk = est.estimate.se.expect("se");
        assert!((se - jk).abs() <= 1e-12 * se, "{se} vs {jk}");
        let refit_tol = replicate_tolerance(&want.fit, r.n0, r.t0, scale, r.y.len(), r.y[0].len());
        for (method, want) in [("placebo", &want.placebo), ("bootstrap", &want.bootstrap)] {
            let want = want.as_ref().unwrap();
            let got = synthdid::estimate(&p, &with(&options, given(method, &want.draws)))
                .expect("variance");
            check_replicates(
                &format!("sim {name} {method}"),
                &Se::single(&got),
                want,
                refit_tol,
            );
        }
    }
}

#[test]
fn options_of_synthdid_estimate_match() {
    let r = reference().sim;
    let columns = Columns::single(&r.y, r.n0, r.t0);
    let p = columns.panel();
    let scale = max_abs(&r.y);
    let o = &r.options;
    let base = SynthDidOptions::default;
    let cases: [(&str, SynthDidOptions, &Fit); 7] = [
        (
            "no sparsify",
            SynthDidOptions {
                sparsify: false,
                ..base()
            },
            &o.no_sparsify,
        ),
        (
            "no intercepts",
            SynthDidOptions {
                omega_intercept: false,
                lambda_intercept: false,
                ..base()
            },
            &o.no_intercepts,
        ),
        (
            "eta omega",
            SynthDidOptions {
                eta_omega: Some(0.3),
                ..base()
            },
            &o.eta_omega,
        ),
        (
            "noise level",
            SynthDidOptions {
                noise_level: Some(2.0),
                ..base()
            },
            &o.noise_level,
        ),
        (
            "max iter",
            SynthDidOptions {
                max_iterations: 50,
                max_iterations_pre_sparsify: 10,
                ..base()
            },
            &o.max_iter,
        ),
        (
            "lambda fixed",
            SynthDidOptions {
                lambda: SynthWeights::Fixed(o.lambda_given.clone()),
                ..base()
            },
            &o.lambda_fixed,
        ),
        (
            "lambda start",
            SynthDidOptions {
                lambda: SynthWeights::Start(o.lambda_given.clone()),
                ..base()
            },
            &o.lambda_start,
        ),
    ];
    for (name, options, want) in cases {
        let est = synthdid::estimate(&p, &options).expect("estimate");
        check_fit(
            name,
            &Got::single(&est),
            want,
            &tolerances(want, r.n0, r.t0, scale, r.y.len(), r.y[0].len()),
        );
    }
}

#[test]
fn covariates_match_synthdid() {
    let r = reference().covariates;
    let columns = Columns::single(&r.y, r.n0, r.t0).with_covariates(&r.x);
    let p = columns.panel();
    let scale = max_abs(&r.y).max(r.x.iter().map(|b| max_abs(b)).fold(0.0, f64::max));
    for (name, want) in [("synthdid", &r.synthdid), ("sc", &r.sc), ("did", &r.did)] {
        let options = config(name);
        let est =
            synthdid::estimate(&p, &with(&options, SynthVariance::Jackknife)).expect("estimate");
        let tol = tolerances(&want.fit, r.n0, r.t0, scale, r.y.len(), r.y[0].len());
        check_fit(
            &format!("covariates {name}"),
            &Got::single(&est),
            &want.fit,
            &tol,
        );
        if name != "synthdid" {
            continue;
        }
        let refit_tol = replicate_tolerance(&want.fit, r.n0, r.t0, scale, r.y.len(), r.y[0].len());
        check_replicates(
            "covariates jackknife",
            &Se::single(&est),
            want.jackknife.as_ref().unwrap(),
            refit_tol,
        );
        for method in ["placebo", "bootstrap"] {
            let want = if method == "placebo" {
                want.placebo.as_ref().unwrap()
            } else {
                want.bootstrap.as_ref().unwrap()
            };
            let got = synthdid::estimate(&p, &with(&options, given(method, &want.draws))).unwrap();
            check_replicates(
                &format!("covariates {method}"),
                &Se::single(&got),
                want,
                refit_tol,
            );
        }
    }
}

#[test]
fn staggered_adoption_matches_the_cohort_by_cohort_reference() {
    let r = reference().staggered;
    let n = r.y.len();
    let periods = r.y[0].len();
    let columns = Columns::new(&r.y, &r.adoption);
    let p = columns.panel();
    let scale = max_abs(&r.y);
    for (name, want) in [("synthdid", &r.synthdid), ("did", &r.did)] {
        let options = config(name);
        let est = synthdid::estimate_staggered(&p, &with(&options, SynthVariance::Jackknife))
            .expect("estimate");
        assert_eq!(est.cohorts.len(), want.cohorts.len());
        let mut tol = 0.0_f64;
        let mut refit_tol = 0.0_f64;
        for (got, cohort) in est.cohorts.iter().zip(&want.cohorts) {
            assert_eq!(got.adoption, i64::try_from(cohort.adoption).unwrap());
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
                &Got {
                    estimate: got.estimate,
                    curve: got.effect_curve.clone(),
                    fit: &got.fit,
                },
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
            (est.estimate.estimate - want.estimate).abs() <= tol,
            "{name}: aggregate"
        );
        let event_time: Vec<f64> = est.event_time.iter().map(|e| e.estimate).collect();
        assert!(
            worst(&event_time, &want.event_time) <= tol,
            "{name}: event time"
        );
        check_replicates(
            &format!("staggered {name} jackknife"),
            &Se::staggered(&est),
            &want.jackknife,
            tol,
        );
        for (method, want) in [("placebo", &want.placebo), ("bootstrap", &want.bootstrap)] {
            let got = synthdid::estimate_staggered(&p, &with(&options, given(method, &want.draws)))
                .unwrap();
            check_replicates(
                &format!("staggered {name} {method}"),
                &Se::staggered(&got),
                want,
                refit_tol,
            );
        }
    }
}

/// One cohort through the staggered route is the single-date estimate, bit
/// for bit, and so are its standard errors.
#[test]
fn one_cohort_is_the_single_date_estimate() {
    let r = reference().sim;
    let columns = Columns::single(&r.y, r.n0, r.t0);
    let p = columns.panel();
    let options = with(&SynthDidOptions::default(), seeded("placebo", 20, 7));
    let single = synthdid::estimate(&p, &options).unwrap();
    let staggered = synthdid::estimate_staggered(&p, &options).unwrap();
    assert_eq!(single.estimate, staggered.estimate);
    assert_eq!(single.effect_curve, staggered.event_time);
    assert_eq!(single.variance, staggered.variance);
    assert_eq!(single.fit, staggered.cohorts[0].fit);
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
    let columns = Columns::single(&r.y, r.n0, r.t0);
    let p = columns.panel();
    for method in ["placebo", "bootstrap"] {
        let want = if method == "placebo" {
            r.synthdid.placebo.as_ref().unwrap()
        } else {
            r.synthdid.bootstrap.as_ref().unwrap()
        };
        let b = want.replicates.len();
        let options = with(&SynthDidOptions::default(), seeded(method, b, 20_261_125));
        let got = synthdid::estimate(&p, &options).unwrap();
        let u = &want.replicates;
        let mean = u.iter().sum::<f64>() / b as f64;
        let m2 = u.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / b as f64;
        let m4 = u.iter().map(|v| (v - mean).powi(4)).sum::<f64>() / b as f64;
        let kurtosis = m4 / (m2 * m2);
        let relative = ((kurtosis - 1.0) / (4.0 * b as f64)).sqrt() * 2.0_f64.sqrt();
        let se = got.estimate.se.unwrap();
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
    let columns = Columns::single(&r.y, r.n0, r.t0);
    let p = columns.panel();
    assert_same_on_threads(|| {
        let mut out = Vec::new();
        for method in ["placebo", "bootstrap"] {
            let options = with(&SynthDidOptions::default(), seeded(method, 16, 11));
            let e = synthdid::estimate(&p, &options).unwrap();
            out.extend(e.variance.unwrap().replicates);
            out.push(e.estimate.estimate);
        }
        out
    });
}

/// The cluster jackknife with one unit per cluster is the unit jackknife.
#[test]
fn singleton_clusters_are_the_unit_jackknife() {
    let r = reference().sim;
    let plain = Columns::single(&r.y, r.n0, r.t0);
    let clustered = Columns::single(&r.y, r.n0, r.t0).with_clusters(|u| u);
    let options = with(&SynthDidOptions::default(), SynthVariance::Jackknife);
    let a = synthdid::estimate(&plain.panel(), &options).unwrap();
    let b = synthdid::estimate(&clustered.panel(), &options).unwrap();
    assert_eq!(a.estimate, b.estimate);
    let (a, b) = (a.variance.unwrap(), b.variance.unwrap());
    assert_eq!(a.replicates, b.replicates);
    assert!(a.influence.is_some() && b.influence.is_none());
}

#[test]
fn refuses_what_it_cannot_estimate() {
    use epistat_core::Classify;
    let r = reference().sim;
    let n = r.y.len();
    let default = SynthDidOptions::default();
    // No treated unit, every unit treated, treatment from the first period.
    let none = Columns::new(&r.y, &vec![None; n]);
    assert_eq!(
        synthdid::estimate(&none.panel(), &default).unwrap_err(),
        Error::NoTreatment
    );
    let all = Columns::new(&r.y, &vec![Some(r.t0); n]);
    assert_eq!(
        synthdid::estimate(&all.panel(), &default).unwrap_err(),
        Error::NoNeverTreated
    );
    let first = Columns::single(&r.y, r.n0, 0);
    assert!(matches!(
        synthdid::estimate(&first.panel(), &default).unwrap_err(),
        Error::Adoption { .. }
    ));
    // Two adoption periods are the staggered estimator's.
    let two: Vec<Option<usize>> = (0..n)
        .map(|i| (i >= r.n0).then_some(r.t0 - usize::from(i % 2 == 0)))
        .collect();
    let two = Columns::new(&r.y, &two);
    assert!(matches!(
        synthdid::estimate(&two.panel(), &default).unwrap_err(),
        Error::Unsupported { .. }
    ));
    assert!(synthdid::estimate_staggered(&two.panel(), &default).is_ok());
    // One pre-period leaves no first difference for the noise level.
    let one = Columns::single(&r.y, r.n0, 1);
    assert_eq!(
        synthdid::estimate(&one.panel(), &default).unwrap_err(),
        Error::NoiseLevel
    );
    let given_noise = SynthDidOptions {
        noise_level: Some(1.0),
        ..SynthDidOptions::default()
    };
    assert!(synthdid::estimate(&one.panel(), &given_noise).is_ok());
    let columns = Columns::single(&r.y, r.n0, r.t0);
    let p = columns.panel();
    let bad = [vec![0; r.n0]];
    let error = synthdid::estimate(&p, &with(&default, given("placebo", &bad))).unwrap_err();
    assert_eq!(error.code(), "input.domain");
    let bad = [(0..r.n0).collect::<Vec<_>>(), vec![0; r.n0]];
    assert_eq!(
        synthdid::estimate(&p, &with(&default, given("placebo", &bad))).unwrap_err(),
        Error::Draw { replicate: 1 }
    );
    // A bootstrap draw of controls only is refused rather than redrawn.
    let bad = [vec![0; n], vec![0; n]];
    assert_eq!(
        synthdid::estimate(&p, &with(&default, given("bootstrap", &bad))).unwrap_err(),
        Error::Draw { replicate: 0 }
    );
    let error = synthdid::estimate(&p, &with(&default, seeded("bootstrap", 1, 3))).unwrap_err();
    assert_eq!(error.code(), "input.domain");
    // Ten treated units and twelve controls: too few controls for the placebo.
    let small = Columns::single(&r.y[r.n0 - 12..], 12, r.t0);
    let did = with(&SynthDidOptions::did(), seeded("placebo", 2, 1));
    assert!(synthdid::estimate(&small.panel(), &did).is_ok());
    let tiny = Columns::single(&r.y[r.n0 - 10..], 10, r.t0);
    assert_eq!(
        synthdid::estimate(&tiny.panel(), &did).unwrap_err(),
        Error::PlaceboControls
    );
    // Weight vectors do not carry across cohorts.
    let s = reference().staggered;
    let sp = Columns::new(&s.y, &s.adoption);
    let options = SynthDidOptions {
        lambda: SynthWeights::Start(vec![1.0]),
        ..SynthDidOptions::default()
    };
    assert!(matches!(
        synthdid::estimate_staggered(&sp.panel(), &options).unwrap_err(),
        Error::Unsupported { .. }
    ));
    // Weights of the wrong length.
    let options = SynthDidOptions {
        lambda: SynthWeights::Fixed(vec![1.0]),
        ..SynthDidOptions::default()
    };
    assert_eq!(
        synthdid::estimate(&p, &options).unwrap_err().code(),
        "input.length"
    );
    // An unbalanced panel and prior weights.
    let unit = [0, 0, 1];
    let period = [0, 1, 0];
    let first = [None, None, Some(1)];
    let y = [1.0, 2.0, 3.0];
    let unbalanced = StaggeredPanel::builder(&unit, &period, &first, &y)
        .build()
        .unwrap();
    assert_eq!(
        synthdid::estimate(&unbalanced, &default).unwrap_err(),
        Error::Unbalanced
    );
    let weights = vec![1.0; columns.outcome.len()];
    let weighted = StaggeredPanel::builder(
        &columns.unit,
        &columns.period,
        &columns.first_treated,
        &columns.outcome,
    )
    .weights(&weights)
    .build()
    .unwrap();
    assert!(matches!(
        synthdid::estimate(&weighted, &default).unwrap_err(),
        Error::Unsupported { .. }
    ));
}

/// Labels for units and years work as for the matrix, and pairs of units
/// as clusters leave out pairs.
#[test]
fn staggered_labels_and_clusters() {
    let s = reference().staggered;
    let (n, t) = (s.y.len(), s.y[0].len());
    let rows = n * t;
    let unit: Vec<u32> = (0..rows)
        .map(|r| 100 + u32::try_from(r / t).unwrap())
        .collect();
    let period: Vec<i64> = (0..rows)
        .map(|r| 2000 + i64::try_from(r % t).unwrap())
        .collect();
    let first_treated: Vec<Option<i64>> = (0..rows)
        .map(|r| s.adoption[r / t].map(|a| 2000 + i64::try_from(a).unwrap()))
        .collect();
    let outcome = s.y.concat();
    let pairs: Vec<u32> = unit.iter().map(|u| (u - 100) / 2).collect();
    let p = StaggeredPanel::builder(&unit, &period, &first_treated, &outcome)
        .clusters(&pairs)
        .build()
        .unwrap();
    let options = with(&SynthDidOptions::default(), SynthVariance::Jackknife);
    let est = synthdid::estimate_staggered(&p, &options).unwrap();
    assert!((est.estimate.estimate - s.synthdid.estimate).abs() < 1e-12);
    assert!(est.cohorts.iter().all(|c| c.adoption >= 2001));
    assert_eq!(
        est.controls[0],
        100 + u32::try_from(s.adoption.iter().position(Option::is_none).unwrap()).unwrap()
    );
    // The units are independent, so the cluster jackknife estimates the same
    // standard error as the unit jackknife; half of it is far outside the
    // error of either estimate.
    let replicates = est.variance.as_ref().unwrap();
    assert_eq!(replicates.replicates.len(), n.div_ceil(2));
    let se = est.estimate.se.expect("defined");
    assert!(se.is_finite() && se > 0.5 * s.synthdid.jackknife.se, "{se}");
    // A unit treated in the first period is refused.
    let mut first = first_treated.clone();
    let victim = s.adoption.iter().position(Option::is_some).unwrap();
    for f in &mut first[victim * t..(victim + 1) * t] {
        *f = Some(2000);
    }
    let p = StaggeredPanel::builder(&unit, &period, &first, &outcome)
        .build()
        .unwrap();
    assert_eq!(
        synthdid::estimate_staggered(&p, &SynthDidOptions::default()).unwrap_err(),
        Error::Adoption {
            unit: 100 + u32::try_from(victim).unwrap()
        }
    );
}
