//! `ate` of riskRegression (g-formula) for data with entry times, for models
//! with an offset (each person predicted with their own) and for
//! cause-specific Cox models with `surv.type = "survival"`, against
//! [`epistat_survival::standardise`]. The reference is `tests/data/influence_ext.R`.
//!
//! The functions are given the coefficients and covariance matrices that
//! `coxph` returned, so that what is compared is the standardisation. The
//! event times are not tied, so the standard errors are comparable with
//! riskRegression's, whose first-order form of the influence function of the
//! product limit is `InfluenceForm::Exponential`.
//!
//! # Tolerances
//!
//! Those of `r_standardise.rs`: with `n` rows, `J` event times, `p` the
//! largest number of columns of a model and `xb` the largest of
//! `sum_a 2 max|x_a| |beta_a|`, a person's risk carries the relative error
//! `4 eps (n + J + p + 4 + xb) (1 + H / (1 - u))` (`1 + H` for one Cox model),
//! with `H` the largest total cumulative hazard of a person and `u` the
//! largest total hazard increment, both from riskRegression. Risk sets with
//! entry times are sums with rows taken out again, in compensated sums, which
//! adds no term. A standard error carries that bound times `max(1, p kappa)`,
//! `kappa` the condition number of the covariance. Limits, differences, ratios
//! and p-values follow by first-order propagation. The bounds are not changed
//! after a run.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "row counts and codes are small"
)]

use epistat_linalg::{Mat, Side};
use epistat_survival::SurvData;
use epistat_survival::absrisk::{CscOptions, SurvType};
use epistat_survival::cox::{self, CoxModelFit, CoxOptions, Ties};
use epistat_survival::standardise::{
    Counterfactual, InfluenceForm, Level, StandardiseOptions, Standardised,
    standardise_cause_specific, standardise_cox,
};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Versions {
    r: String,
    #[serde(rename = "riskRegression")]
    risk_regression: String,
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
struct Numeric {
    se: Vec<Vec<f64>>,
    tol: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct Case {
    numeric: Option<Numeric>,
    name: String,
    kind: String,
    cause: Option<u32>,
    start: Vec<f64>,
    offset: Option<Vec<f64>>,
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
struct Reference {
    versions: Versions,
    cases: Vec<Case>,
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

fn run_case(case: &Case, std_err: bool, influence: InfluenceForm) -> Standardised {
    let n_codes = *case.status.iter().max().unwrap() as u32;
    let mut builder = SurvData::builder(&case.time, &case.status)
        .n_codes(n_codes)
        .start(&case.start);
    if let Some(stratum) = &case.stratum {
        builder = builder.stratum(stratum);
    }
    if let Some(offset) = &case.offset {
        builder = builder.offset(offset);
    }
    let data = builder.build().unwrap();
    let any: Vec<i32> = case.status.iter().map(|&s| i32::from(s != 0)).collect();
    let mut any_builder = SurvData::builder(&case.time, &any).start(&case.start);
    if let Some(offset) = &case.offset {
        any_builder = any_builder.offset(offset);
    }
    let any_data = any_builder.build().unwrap();
    let designs: Vec<Mat<f64>> = case.designs.iter().map(|d| matrix(d)).collect();
    let refs: Vec<_> = designs.iter().map(Mat::as_ref).collect();
    let survival_type = case.kind == "csc_survival";
    let mut fits: Vec<CoxModelFit> = Vec::new();
    for (m, design) in refs.iter().enumerate() {
        let options = CoxOptions {
            ties: Ties::Efron,
            cause: match (case.kind.as_str(), m) {
                ("csc", m) => Some(m as u32 + 1),
                ("cox", _) | ("csc_survival", 0) => Some(1),
                _ => None,
            },
            ..CoxOptions::default()
        };
        let mut fit = if survival_type && m == 1 {
            cox::fit(&any_data, *design, &options).unwrap()
        } else {
            cox::fit(&data, *design, &options).unwrap()
        };
        fit.coefficients = case.coefficients[m].iter().copied().map(Some).collect();
        let p = case.coefficients[m].len();
        fit.covariance = epistat_survival::Covariance::new(
            p,
            (0..p * p)
                .map(|i| Some(case.var[m][i / p][i % p]))
                .collect(),
        )
        .unwrap();
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
        se: std_err,
        product_limit: case.product_limit.unwrap_or(true),
        influence,
        ..StandardiseOptions::default()
    };
    if case.kind == "cox" {
        let options = CoxOptions {
            cause: Some(1),
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
                surv_type: if survival_type {
                    SurvType::Survival {
                        cause: case.cause.unwrap(),
                    }
                } else {
                    SurvType::Hazard
                },
                ..CscOptions::default()
            },
            &levels,
            &case.times,
            &opts,
        )
        .unwrap()
    }
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one pass over the outputs of each case"
)]
fn standardised_risks_match_r() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/influence_ext.json"
    ));
    let v = &reference.versions;
    eprintln!(
        "reference: {}, riskRegression {}, survival {}",
        v.r, v.risk_regression, v.survival
    );
    let eps = f64::EPSILON;
    let mut worst = Worst::default();
    for case in &reference.cases {
        let name = &case.name;
        // riskRegression's standard errors are not the derivative of the
        // estimate when the data have entry times, see `influence_ext.R`.
        let tied = case.start.iter().any(|&v| v > 0.0);
        let n = case.time.len() as f64;
        // riskRegression's first-order form, which the reference has.
        let out = run_case(case, true, InfluenceForm::Exponential);
        {
            let exact = run_case(case, true, InfluenceForm::Exact);
            let (mut gap, mut gap_cov) = (0.0f64, 0.0f64);
            for (a, b) in out
                .se
                .as_ref()
                .unwrap()
                .iter()
                .zip(exact.se.as_ref().unwrap())
            {
                let (a, b) = (a.unwrap(), b.unwrap());
                if a > 0.0 {
                    gap = gap.max((a - b).abs() / a);
                }
            }
            for (a, b) in out
                .covariance
                .as_ref()
                .unwrap()
                .iter()
                .flat_map(epistat_survival::Covariance::as_slice)
                .zip(
                    exact
                        .covariance
                        .as_ref()
                        .unwrap()
                        .iter()
                        .flat_map(epistat_survival::Covariance::as_slice),
                )
            {
                let (a, b) = (a.unwrap(), b.unwrap());
                if a != 0.0 {
                    gap_cov = gap_cov.max((a - b).abs() / a.abs());
                }
            }
            assert_eq!(
                out.risk, exact.risk,
                "{name}: the form does not move the risk"
            );
            eprintln!(
                "{name}: exact against first-order standard error, up to {gap:.3e} relative, covariance {gap_cov:.3e}"
            );
        }
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
        let t = case.times.len();
        let level_of = |label: &str| label.parse::<usize>().unwrap();
        let k_of = |time: f64| {
            case.times
                .iter()
                .position(|&x| x.to_bits() == time.to_bits())
                .unwrap()
        };
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
            let se = out.se.as_ref().unwrap()[l * t + k].unwrap();
            worst.put(&format!("{name} se"), (se - row.se).abs(), b_se * row.se);
            let (lo, up) = (
                out.lower.as_ref().unwrap()[l * t + k].unwrap(),
                out.upper.as_ref().unwrap()[l * t + k].unwrap(),
            );
            let bl = b_risk * row.estimate + 2.0 * 1.96 * b_se * row.se;
            worst.put(&format!("{name} lower"), (lo - row.lower).abs(), bl);
            worst.put(&format!("{name} upper"), (up - row.upper).abs(), bl);
        }
        if let (true, Some(numeric)) = (tied, case.numeric.as_ref()) {
            let mut gap_rr = 0.0f64;
            for row in &case.ate.mean {
                let (l, k) = (level_of(&row.treatment), k_of(row.time));
                let se = out.se.as_ref().unwrap()[l * t + k].unwrap();
                // The derivative of the estimate, from refitted weights in R.
                let reference = numeric.se[l][k];
                worst.put(
                    &format!("{name} se against the derivative"),
                    (se - reference).abs(),
                    numeric.tol[l][k] + b_se * reference,
                );
                gap_rr = gap_rr.max((row.se - reference).abs() / reference);
            }
            assert!(
                gap_rr > 0.05,
                "{name}: riskRegression's standard error is expected to differ from the derivative"
            );
            eprintln!(
                "{name}: riskRegression's standard errors differ from the derivative by up to {gap_rr:.3} (relative)"
            );
        }
        if tied {
            let mut gap = 0.0f64;
            for row in &case.ate.mean {
                let (l, k) = (level_of(&row.treatment), k_of(row.time));
                let se = out.se.as_ref().unwrap()[l * t + k].unwrap();
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
                let se = inference.se[k].unwrap();
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
}
