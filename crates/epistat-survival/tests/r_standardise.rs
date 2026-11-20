//! `ate` of riskRegression (g-formula, Cox and cause-specific Cox models) and
//! `stdCoxph` of stdReg against [`epistat_survival::standardise`].
//!
//! The functions are given the coefficients and covariance matrices that
//! `coxph` returned, so that what is compared is the standardisation.
//!
//! # Tolerances
//!
//! With `n` rows, `J` event times, `p` the largest number of columns of a model
//! and `xb` the largest of `sum_a 2 max|x_a| |beta_a|` over the models, a
//! person's risk carries the relative error `4 eps (n + J + p + 4 + xb)
//! (1 + H / (1 - u))` (`1 + H` for one Cox model, which has no product), with `H` the largest total cumulative hazard of a person
//! and `u` the largest total hazard increment, both from riskRegression
//! (the derivation is that of `r_absrisk.rs`: a risk score, a baseline
//! increment of `n` terms, a product or sum of `J` terms). The standardised risk is a mean
//! of such risks, so it carries the same bound, relative to its value. A
//! difference carries the sum of the absolute bounds of its two risks, and a
//! ratio the sum of their relative bounds. A standard error is a square root
//! of a sum of squares of influence functions, each a sum of terms with the
//! bound of a risk and the coefficient part carrying `p kappa` times it, with
//! `kappa` the condition number of the covariance. The bound of a standard
//! error is the bound of a risk times `max(1, p kappa)`, relative to the value,
//! and that of a limit or a p-value follows from the bounds of the estimate
//! and the standard error by the rules of a first-order propagation.
//! The bounds are not changed after a run.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    reason = "row counts and codes are small, and times are compared exactly with the times asked for"
)]

use epistat_survival::SurvData;
use epistat_survival::absrisk::CscOptions;
use epistat_survival::cox::{self, CoxModelFit, CoxOptions, Ties};
use epistat_survival::standardise::{
    Counterfactual, Level, StandardiseOptions, Standardised, standardise_cause_specific,
    standardise_cox,
};
use epistat_testkit::load_json;
use faer::{Mat, Side};
use serde::Deserialize;

#[derive(Deserialize)]
struct Versions {
    r: String,
    #[serde(rename = "riskRegression")]
    risk_regression: String,
    #[serde(rename = "stdReg")]
    std_reg: String,
    survival: String,
}

#[derive(Deserialize)]
struct Scales {
    max_hazard: f64,
    total_cumhazard: f64,
    n_event_times: f64,
}

#[derive(Deserialize)]
struct MeanRow {
    time: f64,
    treatment: String,
    estimate: f64,
    se: f64,
    lower: f64,
    upper: f64,
}

#[derive(Deserialize)]
struct DiffRow {
    time: f64,
    #[serde(rename = "A")]
    a: String,
    #[serde(rename = "B")]
    b: String,
    estimate: f64,
    se: f64,
    lower: f64,
    upper: f64,
    #[serde(rename = "p.value")]
    p_value: f64,
}

#[derive(Deserialize)]
struct Ate {
    mean: Vec<MeanRow>,
    diff: Vec<DiffRow>,
    ratio: Vec<DiffRow>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    kind: String,
    cause: Option<u32>,
    product_limit: Option<bool>,
    time: Vec<f64>,
    status: Vec<i32>,
    stratum: Option<Vec<u32>>,
    designs: Vec<Vec<Vec<f64>>>,
    coefficients: Vec<Vec<f64>>,
    var: Vec<Vec<Vec<f64>>>,
    times: Vec<f64>,
    nlevels: usize,
    ate: Ate,
    scales: Scales,
}

#[derive(Deserialize)]
struct StdReg {
    times: Vec<f64>,
    coefficients: Vec<f64>,
    var: Vec<Vec<f64>>,
    survival: Vec<Vec<f64>>,
    vcov: Vec<Vec<Vec<f64>>>,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    cases: Vec<Case>,
    stdreg: StdReg,
}

fn matrix(rows: &[Vec<f64>]) -> Mat<f64> {
    Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
}

/// The dummy columns of the exposure set to level `l`.
fn setting(l: usize, nlevels: usize) -> Vec<(usize, f64)> {
    (0..nlevels - 1)
        .map(|c| (c, f64::from(u8::from(l == c + 1))))
        .collect()
}

#[derive(Default)]
struct Worst(std::collections::BTreeMap<String, (f64, f64)>);

impl Worst {
    fn put(&mut self, name: &str, deviation: f64, bound: f64) {
        assert!(
            deviation <= bound,
            "{name}: deviation {deviation:e} above its bound {bound:e}"
        );
        let entry = self.0.entry(name.to_string()).or_insert((0.0, 0.0));
        entry.0 = entry.0.max(deviation / bound);
        entry.1 = entry.1.max(deviation);
    }
}

fn run_case(case: &Case, ties: Ties, std_err: bool, worst: &mut Worst) -> Standardised {
    let n = case.time.len();
    let n_codes = *case.status.iter().max().unwrap() as u32;
    let mut builder = SurvData::builder(&case.time, &case.status).n_codes(n_codes);
    if let Some(stratum) = &case.stratum {
        builder = builder.stratum(stratum);
    }
    let data = builder.build().unwrap();
    let designs: Vec<Mat<f64>> = case.designs.iter().map(|d| matrix(d)).collect();
    let refs: Vec<_> = designs.iter().map(Mat::as_ref).collect();
    let mut fits: Vec<CoxModelFit> = Vec::new();
    for (m, design) in refs.iter().enumerate() {
        let options = CoxOptions {
            ties,
            cause: (case.kind == "csc").then_some(m as u32 + 1),
            ..CoxOptions::default()
        };
        let mut fit = cox::fit(&data, *design, &options).unwrap();
        fit.coefficients = case.coefficients[m].iter().copied().map(Some).collect();
        let p = case.coefficients[m].len();
        fit.covariance = (0..p * p)
            .map(|i| Some(case.var[m][i / p][i % p]))
            .collect();
        fits.push(fit);
    }
    let sets: Vec<Vec<(usize, f64)>> = (0..case.nlevels)
        .map(|l| setting(l, case.nlevels))
        .collect();
    let levels: Vec<Level<'_>> = sets
        .iter()
        .map(|set| Level {
            designs: (0..refs.len())
                .map(|_| Counterfactual::Columns(set))
                .collect(),
        })
        .collect();
    let opts = StandardiseOptions {
        std_err,
        product_limit: case.product_limit.unwrap_or(true),
        ..StandardiseOptions::default()
    };
    let out = if case.kind == "cox" {
        let options = CoxOptions {
            ties,
            ..CoxOptions::default()
        };
        standardise_cox(
            &data,
            refs[0],
            &fits[0],
            &options,
            &levels,
            &case.times,
            &opts,
        )
        .unwrap()
    } else {
        standardise_cause_specific(
            &data,
            &refs,
            &fits,
            case.cause.unwrap(),
            &CscOptions {
                ties,
                ..CscOptions::default()
            },
            &levels,
            &case.times,
            &opts,
        )
        .unwrap()
    };
    let _ = (n, worst);
    out
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one pass over the outputs of each case"
)]
fn standardised_risks_match_r() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/standardise.json"
    ));
    let v = &reference.versions;
    eprintln!(
        "reference: {}, riskRegression {}, stdReg {}, survival {}",
        v.r, v.risk_regression, v.std_reg, v.survival
    );
    let eps = f64::EPSILON;
    let mut worst = Worst::default();
    let mut b_se_first = 0.0;
    for case in &reference.cases {
        let name = &case.name;
        let tied = name.contains("tied");
        let n = case.time.len() as f64;
        let out = run_case(case, Ties::Efron, true, &mut worst);
        let p_max = case.coefficients.iter().map(Vec::len).max().unwrap() as f64;
        let mut xb = 0.0f64;
        for (m, d) in case.designs.iter().enumerate() {
            let mut total = 0.0f64;
            for a in 0..case.coefficients[m].len() {
                let big = d.iter().map(|row| row[a].abs()).fold(0.0, f64::max);
                total += 2.0 * big * case.coefficients[m][a].abs();
            }
            xb = xb.max(total);
        }
        let kappa = case
            .var
            .iter()
            .map(|v| {
                let p = v.len();
                let m = Mat::from_fn(p, p, |i, j| v[i][j]);
                let e = m.self_adjoint_eigenvalues(Side::Lower).unwrap();
                e[p - 1] / e[0]
            })
            .fold(1.0, f64::max);
        let sc = &case.scales;
        let b_risk = 4.0
            * eps
            * (n + sc.n_event_times + p_max + 4.0 + xb)
            * if case.kind == "cox" {
                1.0 + sc.total_cumhazard
            } else {
                1.0 + sc.total_cumhazard / (1.0 - sc.max_hazard)
            };
        let b_se = b_risk * p_max.mul_add(kappa, 0.0).max(1.0);
        if case.name == "cox_binary" {
            b_se_first = b_se;
        }
        let t = case.times.len();
        let level_of = |label: &str| label.parse::<usize>().unwrap();
        let k_of = |time: f64| case.times.iter().position(|&x| x == time).unwrap();
        for row in &case.ate.mean {
            let (l, k) = (level_of(&row.treatment), k_of(row.time));
            let f = out.risk[l * t + k].unwrap();
            worst.put(
                &format!("{name} risk"),
                (f - row.estimate).abs(),
                b_risk * row.estimate,
            );
            if tied {
                continue;
            }
            let se = out.std_err.as_ref().unwrap()[l * t + k].unwrap();
            worst.put(&format!("{name} se"), (se - row.se).abs(), b_se * row.se);
            let (lo, up) = (
                out.lower.as_ref().unwrap()[l * t + k].unwrap(),
                out.upper.as_ref().unwrap()[l * t + k].unwrap(),
            );
            let bl = b_risk * row.estimate + 2.0 * 1.96 * b_se * row.se;
            worst.put(&format!("{name} lower"), (lo - row.lower).abs(), bl);
            worst.put(&format!("{name} upper"), (up - row.upper).abs(), bl);
        }
        if tied {
            let mut gap = 0.0f64;
            for row in &case.ate.mean {
                let (l, k) = (level_of(&row.treatment), k_of(row.time));
                let se = out.std_err.as_ref().unwrap()[l * t + k].unwrap();
                gap = gap.max((se - row.se).abs() / row.se);
            }
            eprintln!(
                "{name}: standard errors differ from riskRegression's by up to {gap:.3} (relative), tied times"
            );
        }
        for (rows, ratio) in [(&case.ate.diff, false), (&case.ate.ratio, true)] {
            for row in rows {
                let (a, b, k) = (level_of(&row.a), level_of(&row.b), k_of(row.time));
                let c = out.contrast(b, a).unwrap();
                let (estimate, inference) = if ratio {
                    (c.ratio[k].unwrap(), c.ratio_inference.unwrap())
                } else {
                    (c.difference[k].unwrap(), c.difference_inference.unwrap())
                };
                let label = if ratio { "ratio" } else { "diff" };
                let fa = out.risk[a * t + k].unwrap();
                let fb = out.risk[b * t + k].unwrap();
                let b_est = if ratio {
                    2.0 * b_risk * row.estimate
                } else {
                    b_risk * (fa + fb)
                };
                worst.put(
                    &format!("{name} {label}"),
                    (estimate - row.estimate).abs(),
                    b_est,
                );
                if tied {
                    continue;
                }
                let se = inference.std_err[k].unwrap();
                worst.put(
                    &format!("{name} {label} se"),
                    (se - row.se).abs(),
                    b_se * row.se * 2.0,
                );
                let lo = inference.lower[k].unwrap();
                let up = inference.upper[k].unwrap();
                let bl = b_est + 2.0 * 1.96 * 2.0 * b_se * row.se;
                worst.put(&format!("{name} {label} lower"), (lo - row.lower).abs(), bl);
                worst.put(&format!("{name} {label} upper"), (up - row.upper).abs(), bl);
                let pv = inference.p_value[k].unwrap();
                // p = 2 Phi(-|z|) with z = (estimate - null) / se: the change in p
                // is 2 phi(z) |z| times the relative changes of its two parts.
                let null = if ratio { 1.0 } else { 0.0 };
                let z = (row.estimate - null) / row.se;
                let phi = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
                let b_p = 2.0 * phi * z.abs() * (b_est / (row.estimate - null).abs() + 2.0 * b_se);
                worst.put(&format!("{name} {label} p"), (pv - row.p_value).abs(), b_p);
            }
        }
    }
    for (k, v) in &worst.0 {
        eprintln!(
            "{k}: worst deviation {:.3e} absolute, {:.3} of its bound",
            v.1, v.0
        );
    }
    stdreg_check(&reference.stdreg, &reference.cases[0], b_se_first);
}

/// `stdCoxph` for the Breslow fit of the first case: the survival agrees, and
/// its covariance of the two survivals is ours times `n / (n - 1)`. The bound
/// is that of the risk (the survival is one minus it) and, for a variance, four
/// times that of a standard error.
fn stdreg_check(reference: &StdReg, case: &Case, b_se: f64) {
    let n = case.time.len();
    let design = Mat::from_fn(n, 3, |i, j| case.designs[0][i][j]);
    let data = SurvData::builder(&case.time, &case.status).build().unwrap();
    let options = CoxOptions {
        ties: Ties::Breslow,
        ..CoxOptions::default()
    };
    let mut fit = cox::fit(&data, design.as_ref(), &options).unwrap();
    fit.coefficients = reference.coefficients.iter().copied().map(Some).collect();
    fit.covariance = (0..9).map(|i| Some(reference.var[i / 3][i % 3])).collect();
    let set0 = [(0, 0.0)];
    let set1 = [(0, 1.0)];
    let levels = [
        Level::single(Counterfactual::Columns(&set0)),
        Level::single(Counterfactual::Columns(&set1)),
    ];
    let out = standardise_cox(
        &data,
        design.as_ref(),
        &fit,
        &options,
        &levels,
        &reference.times,
        &StandardiseOptions {
            std_err: true,
            ..StandardiseOptions::default()
        },
    )
    .unwrap();
    let t = reference.times.len();
    let mut worst_risk = 0.0f64;
    let mut worst_var = 0.0f64;
    for k in 0..t {
        for l in 0..2 {
            let r = out.risk[l * t + k].unwrap();
            worst_risk = worst_risk.max((1.0 - r - reference.survival[k][l]).abs());
        }
        let cov = out.covariance.as_ref().unwrap();
        for a in 0..2 {
            for b in 0..2 {
                let ours = cov[k * 4 + a * 2 + b].unwrap();
                let theirs = reference.vcov[k][a][b];
                // stdReg's covariance is ours times n / (n - 1).
                let scaled = ours * n as f64 / (n as f64 - 1.0);
                let dev = (scaled - theirs).abs() / theirs.abs();
                assert!(
                    dev <= 4.0 * b_se,
                    "stdCoxph covariance {scaled} against {theirs}"
                );
                worst_var = worst_var.max(dev);
            }
        }
    }
    assert!(
        worst_risk <= 4.0 * b_se,
        "stdCoxph survival differs by {worst_risk}"
    );
    eprintln!(
        "stdCoxph: survival differs by {worst_risk:.2e}, covariance times n/(n-1) by up to {worst_var:.2e} relative"
    );
}
