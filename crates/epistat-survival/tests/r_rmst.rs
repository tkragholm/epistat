//! `rmst::rmst2` checked against `survRM2::rmst2` 1.0.4 (and `survival` 3.8-12):
//! the restricted mean survival time and time lost of two arms with their
//! standard errors and limits, the difference and the two ratios with limits and
//! p-values, the default horizon in every way the arms can end, and the horizons
//! `rmst2` refuses. The data are `survival::lung` with sex as the arm and
//! simulated sets with tied times. See `data/rmst.R`.
//!
//! # Tolerances
//!
//! The risk sets are whole numbers, so the counts are exact, and every difference
//! is rounding. With `n` the rows of an arm, which is at least the number of
//! its curve times `m`:
//!
//! - A Kaplan-Meier value is a product of at most `m` factors `1 - d / Y`, each
//!   formed with three roundings by either program, so its relative error is at
//!   most `6 (m + 2) eps`.
//! - The restricted mean is a sum of at most `m + 1` positive rectangles, so
//!   its relative error is at most `(6 + 1) (m + 2) eps` plus the rounding of the
//!   widths, and `8 (n + 2) eps` bounds it. The time lost is `tau` less the mean,
//!   so its absolute error is that of the mean plus `eps tau`.
//! - The variance is a sum of positive terms, each the square of an area (twice
//!   its relative error) times a Greenwood increment (six roundings), so its
//!   relative error is at most `20 (n + 2) eps` and that of the standard error
//!   half of it.
//! - A difference has the sum of the absolute errors of its two means, and
//!   the standard error of the difference the larger of the relative errors of
//!   the two variances. The logarithm of a ratio has the sum of the two
//!   relative errors of the ratio's terms, and its variance `v1 / r1^2 + v0 / r0^2`
//!   the relative error of the variances plus twice that of the means, which
//!   `40 (n + 2) eps` bounds.
//! - A limit adds `z` times the error of the standard error and the standard
//!   error times the error of `z`, which `qnorm` and `quantile` agree on to
//!   two units in the last place, `4 eps` relative, plus three roundings of the
//!   limit. A ratio's limit is the exponential, so its relative error is the
//!   absolute error of the logarithm's.
//! - A p-value is `2 Phi(-|s|)`, whose derivative in the statistic `s` is
//!   `-2 phi(s)`, and `phi <= 0.4`, so the error is `0.8` times the error of
//!   `s`, which is `|s|` times the sum of the relative errors of its numerator and
//!   denominator, plus `8 eps` for the evaluation of `Phi`.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::manual_midpoint,
    reason = "arm codes and whole counts are compared exactly, and half the relative error of a variance is not a midpoint"
)]

use std::collections::HashMap;

use epistat_survival::SurvData;
use epistat_survival::rmst::{Contrast, GroupRmst, RmstOptions, Tau, rmst2};
use epistat_testkit::load_json;
use serde::Deserialize;

const EPS: f64 = f64::EPSILON;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    datasets: HashMap<String, Dataset>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Versions {
    #[serde(rename = "R")]
    r: String,
    survival: String,
    #[serde(rename = "survRM2")]
    survrm2: String,
}

#[derive(Deserialize)]
struct Dataset {
    time: Vec<f64>,
    status: Vec<f64>,
    arm: Vec<f64>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    fits: Vec<Fit>,
}

#[derive(Deserialize)]
struct Fit {
    error: bool,
    tau_given: Option<f64>,
    alpha: f64,
    tau: Option<f64>,
    arm0: Option<Arm>,
    arm1: Option<Arm>,
    difference: Option<Row>,
    ratio: Option<Row>,
    loss_ratio: Option<Row>,
}

#[derive(Deserialize)]
struct Arm {
    rmst: f64,
    se: f64,
    rmst_lower: f64,
    rmst_upper: f64,
    rmtl: f64,
    rmtl_lower: f64,
    rmtl_upper: f64,
}

#[derive(Deserialize)]
struct Row {
    est: f64,
    lower: f64,
    upper: f64,
    p: f64,
}

#[derive(Default)]
struct Worst(HashMap<&'static str, (f64, String)>);

impl Worst {
    #[track_caller]
    fn see(&mut self, output: &'static str, label: &str, got: f64, want: f64, tolerance: f64) {
        let deviation = (got - want).abs();
        let ratio = deviation / tolerance;
        assert!(
            ratio <= 1.0,
            "{output} at {label}: got {got:e}, R {want:e}, deviation {deviation:e} against tolerance {tolerance:e}"
        );
        let entry = self.0.entry(output).or_insert((0.0, String::new()));
        if ratio >= entry.0 {
            *entry = (ratio, label.to_string());
        }
    }
}

fn check_arm(w: &mut Worst, label: &str, got: &GroupRmst, r: &Arm, tau: f64, n: f64, z: f64) {
    let rel_mean = 8.0 * (n + 2.0) * EPS;
    let mean_tol = rel_mean * r.rmst.abs() + 2.0 * EPS * tau;
    let se_tol = 10.0 * (n + 2.0) * EPS * r.se + EPS;
    let limit_tol = mean_tol + z * se_tol + z * r.se * 4.0 * EPS + 3.0 * EPS * tau;
    w.see("rmst", label, got.rmst, r.rmst, mean_tol);
    w.see("rmst se", label, got.se, r.se, se_tol);
    w.see("rmst lower", label, got.rmst_lower, r.rmst_lower, limit_tol);
    w.see("rmst upper", label, got.rmst_upper, r.rmst_upper, limit_tol);
    w.see("rmtl", label, got.rmtl, r.rmtl, mean_tol);
    w.see("rmtl lower", label, got.rmtl_lower, r.rmtl_lower, limit_tol);
    w.see("rmtl upper", label, got.rmtl_upper, r.rmtl_upper, limit_tol);
}

fn p_tolerance(stat: f64, rel_numerator: f64, rel_denominator: f64) -> f64 {
    0.8 * stat.abs() * (rel_numerator + rel_denominator) + 8.0 * EPS
}

#[test]
fn rmst2_agrees_with_survrm2() {
    let reference: Reference =
        load_json(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/rmst.json"));
    eprintln!(
        "{}, survival {}, survRM2 {}",
        reference.versions.r, reference.versions.survival, reference.versions.survrm2
    );
    let mut worst = Worst::default();
    let mut compared = 0;
    let mut refused = 0;
    for case in &reference.cases {
        let d = &reference.datasets[&case.name];
        let stop = &d.time;
        let status: Vec<i32> = d.status.iter().map(|&s| s as i32).collect();
        let arm: Vec<u32> = d.arm.iter().map(|&a| a as u32).collect();
        let n_arm = |a: u32| arm.iter().filter(|&&x| x == a).count() as f64;
        let data = SurvData::builder(stop, &status)
            .stratum(&arm)
            .build()
            .unwrap();
        for (i, fit) in case.fits.iter().enumerate() {
            let label = format!("{} fit {i} (tau {:?})", case.name, fit.tau_given);
            let tau = fit.tau_given.map_or(Tau::Largest, Tau::Fixed);
            let mut opts = RmstOptions::new(tau);
            opts.conf_level = 1.0 - fit.alpha;
            let got = rmst2(&data, &opts);
            if fit.error {
                assert!(got.is_err(), "{label}: R refuses the horizon");
                refused += 1;
                continue;
            }
            let got = got.unwrap_or_else(|e| panic!("{label}: {e}"));
            compared += 1;
            let r_tau = fit.tau.unwrap();
            assert_eq!(got.tau.to_bits(), r_tau.to_bits(), "{label}: tau");
            let (r0, r1) = (fit.arm0.as_ref().unwrap(), fit.arm1.as_ref().unwrap());
            let z = epistat_stats::dist::normal::quantile(1.0 - fit.alpha / 2.0);
            let (n0, n1) = (n_arm(0), n_arm(1));
            check_arm(
                &mut worst,
                &format!("{label} arm 0"),
                &got.group0,
                r0,
                r_tau,
                n0,
                z,
            );
            check_arm(
                &mut worst,
                &format!("{label} arm 1"),
                &got.group1,
                r1,
                r_tau,
                n1,
                z,
            );
            assert_eq!(got.group0.group, 0);
            assert_eq!(got.group1.group, 1);

            let n = n0.max(n1);
            let rel_mean = 8.0 * (n + 2.0) * EPS;
            let se_rel = 10.0 * (n + 2.0) * EPS;
            let mean_tol = |r: &Arm| rel_mean * r.rmst.abs() + 2.0 * EPS * r_tau;
            // Difference.
            let rd = fit.difference.as_ref().unwrap();
            let diff_tol = mean_tol(r0) + mean_tol(r1) + EPS * rd.est.abs();
            let diff_se = (r0.se * r0.se + r1.se * r1.se).sqrt();
            let diff_se_tol = se_rel * diff_se + EPS;
            let limit_tol =
                diff_tol + z * diff_se_tol + z * diff_se * 4.0 * EPS + 3.0 * EPS * rd.est.abs();
            check_contrast(
                &mut worst,
                &format!("{label} difference"),
                &got.difference,
                rd,
                diff_se,
                [diff_tol, diff_se_tol, limit_tol, limit_tol],
                p_tolerance(rd.est / diff_se, diff_tol / rd.est.abs(), se_rel),
            );
            // Ratios, on the log scale.
            for (name, got_row, want, a, b) in [
                (
                    "ratio",
                    &got.ratio,
                    fit.ratio.as_ref().unwrap(),
                    (r1.rmst, mean_tol(r1)),
                    (r0.rmst, mean_tol(r0)),
                ),
                (
                    "loss ratio",
                    &got.loss_ratio,
                    fit.loss_ratio.as_ref().unwrap(),
                    (r1.rmtl, mean_tol(r1)),
                    (r0.rmtl, mean_tol(r0)),
                ),
            ] {
                let rel_a = a.1 / a.0.abs();
                let rel_b = b.1 / b.0.abs();
                let log_tol = rel_a + rel_b + 3.0 * EPS;
                let v = r1.se * r1.se / (a.0 * a.0) + r0.se * r0.se / (b.0 * b.0);
                let se = v.sqrt();
                let se_tol = 0.5 * (40.0 * (n + 2.0) * EPS + 2.0 * (rel_a + rel_b)) * se + EPS;
                let log_limit_tol =
                    log_tol + z * se_tol + z * se * 4.0 * EPS + 3.0 * EPS * (want.est.ln()).abs();
                let rel = |abs: f64, value: f64| abs * value.abs() + 3.0 * EPS * value.abs();
                check_contrast(
                    &mut worst,
                    &format!("{label} {name}"),
                    got_row,
                    want,
                    se,
                    [
                        rel(log_tol, want.est),
                        se_tol,
                        rel(log_limit_tol, want.lower),
                        rel(log_limit_tol, want.upper),
                    ],
                    p_tolerance(
                        want.est.ln() / se,
                        log_tol / want.est.ln().abs().max(EPS),
                        se_tol / se,
                    ),
                );
            }
        }
    }
    eprintln!("compared {compared} fits, {refused} refused horizons");
    let mut rows: Vec<_> = worst.0.into_iter().collect();
    rows.sort_by(|a, b| a.0.cmp(b.0));
    for (output, (ratio, label)) in rows {
        eprintln!("worst {output}: {ratio:.3} of tolerance at {label}");
    }
    assert_eq!((compared, refused), (50, 22));
}

fn check_contrast(
    w: &mut Worst,
    label: &str,
    got: &Contrast,
    want: &Row,
    want_se: f64,
    tolerances: [f64; 4],
    p_tol: f64,
) {
    w.see(
        "estimate",
        label,
        got.estimate.unwrap(),
        want.est,
        tolerances[0],
    );
    w.see(
        "contrast se",
        label,
        got.se.unwrap(),
        want_se,
        tolerances[1],
    );
    w.see(
        "lower",
        label,
        got.lower.unwrap(),
        want.lower,
        tolerances[2],
    );
    w.see(
        "upper",
        label,
        got.upper.unwrap(),
        want.upper,
        tolerances[3],
    );
    w.see("p", label, got.p_value.unwrap(), want.p, p_tol);
}
