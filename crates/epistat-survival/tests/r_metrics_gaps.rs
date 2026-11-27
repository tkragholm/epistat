//! The Cox censoring model of `metrics::score` and the contrasts of
//! `metrics::contrast`, checked against `riskRegression::Score` (2026.3.11) with
//! `cens.model = "cox"`, `conservative = FALSE` and `contrasts = TRUE`, on
//! data with continuous times that the R script writes out.
//!
//! `Score` fits the censoring model with `rms::cph` at `eps = 1e-6`, and the
//! same stopping rule is given to `cox::fit`, so both programs stop at the same
//! iterate. `Score`'s influence function counts a tied time once, so the data
//! has no ties; the influence function with ties is checked against numerical
//! derivatives in `metrics_censoring_ties.rs`.
//!
//! # Tolerances
//!
//! `n` rows, `m` censored rows (distinct censoring times), `w` the largest
//! weight `1 / G` as `getCensoringWeights` returns it, `U` the largest entry of
//! the influence functions computed here, `kappa` the 1-norm condition number
//! of the information matrix of the censoring model (1 without covariates).
//!
//! - A Brier score is a mean of `n` terms in `[0, w]`, each a product of at most
//!   `m + 3` rounded factors: `b_B = 4 (n + m + 3) eps w`. The coefficients of
//!   the censoring model agree to `kappa eps`, which moves a weight by
//!   `kappa eps` relative, so `b_B` is multiplied by `1 + kappa`.
//! - The area is a ratio of weighted sums: `b_A = 8 (n + m + 3) eps (1 +
//!   kappa)`.
//! - A standard error is the root mean square of an influence function whose
//!   entries are sums of at most `n + m` terms of the size `U`, so `b_se = 8 (n
//!   + m + 3) eps U (1 + kappa)`.
//! - A difference has the sum of the two bounds. Its limits add `z b_se` for
//!   the normal quantile `z`. Its p-value `2 Phi(-|z_d|)` with `z_d = delta /
//!   se` has, to first order, the error `2 phi(z_d) (b_delta + |z_d| b_se) /
//!   se`, at most `(0.8 b_delta + 0.5 b_se) / se` since `phi <= 0.399` and `phi(z)
//!   |z| <= 0.242`. The normal distribution function is the same algorithm as
//!   R's and adds nothing.
//!
//! The bounds are not changed after a run. The worst deviation in units of its
//! bound is printed for each output.

#![expect(
    clippy::float_cmp,
    reason = "evaluation times are written and read back exactly"
)]

use std::collections::BTreeMap;

use epistat_inference::Influence;
use epistat_linalg::{Attempt, Mat, inverse};
use epistat_survival::SurvData;
use epistat_survival::cox::{CoxOptions, fit};
use epistat_survival::metrics::contrast::{ContrastOptions, auc_difference, brier_difference};
use epistat_survival::metrics::score::{
    Censoring, CoxCensoring, ScoreOptions, auc, brier, ipa, null_brier,
};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Data {
    time: Vec<f64>,
    cr: Vec<i32>,
    any: Vec<i32>,
    x1: Vec<f64>,
    x2: Vec<f64>,
    times: Vec<f64>,
    risk_a: Vec<Vec<f64>>,
    risk_b: Vec<Vec<f64>>,
    srisk_a: Vec<Vec<f64>>,
    srisk_b: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct Model {
    brier: Vec<f64>,
    brier_se: Vec<f64>,
    auc: Option<Vec<f64>>,
    auc_se: Option<Vec<f64>>,
    ipa: Option<Vec<f64>>,
    ipa_se: Option<Vec<f64>>,
}

#[derive(Deserialize)]
struct Row {
    times: f64,
    model: String,
    reference: String,
    #[serde(rename = "delta.Brier", default)]
    delta_brier: Option<f64>,
    #[serde(rename = "delta.AUC", default)]
    delta_auc: Option<f64>,
    se: f64,
    lower: f64,
    upper: f64,
    p: f64,
}

#[derive(Deserialize)]
struct Run {
    conservative: bool,
    cens: String,
    models: BTreeMap<String, Model>,
    brier_contrasts: Vec<Row>,
    auc_contrasts: Vec<Row>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    wmax: f64,
    kind: String,
    cens_cols: Option<Vec<String>>,
    runs: Vec<Run>,
}

#[derive(Deserialize)]
struct Reference {
    versions: BTreeMap<String, String>,
    data: Data,
    cases: Vec<Case>,
}

#[derive(Default)]
struct Worst(BTreeMap<String, f64>);

impl Worst {
    fn check(&mut self, tag: &str, what: &str, got: f64, want: f64, bound: f64) {
        let dev = (got - want).abs() / bound;
        assert!(
            dev <= 1.0,
            "{tag}: {what} is {got}, R has {want}, bound {bound}"
        );
        let entry = self.0.entry(what.to_string()).or_insert(0.0);
        *entry = entry.max(dev);
    }
}

fn condition(x: &Mat<f64>, data: &SurvData<'_>) -> f64 {
    let p = x.ncols();
    if p == 0 {
        return 1.0;
    }
    let model = fit(
        data,
        x.as_ref(),
        &CoxOptions {
            tolerance: 1e-6,
            ..CoxOptions::default()
        },
    )
    .unwrap();
    let v = Mat::from_fn(p, p, |i, j| model.covariance.get(i, j).unwrap());
    let inv = inverse(v.as_ref(), &Attempt::SYMMETRIC).unwrap().x;
    let norm = |m: &Mat<f64>| {
        (0..p)
            .map(|j| (0..p).map(|i| m[(i, j)].abs()).sum::<f64>())
            .fold(0.0f64, f64::max)
    };
    norm(&v) * norm(&inv)
}

#[test]
fn cox_censoring_and_contrasts_against_riskregression() {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/metrics_gaps.json"
    ));
    eprintln!("reference: {:?}", r.versions);
    let d = &r.data;
    let n = d.time.len();
    let t = d.times.len();
    let eps = f64::EPSILON;
    let z = epistat_stats::dist::normal::quantile(0.975);
    let mut worst = Worst::default();
    for case in &r.cases {
        let (status, n_codes, ra, rb) = if case.kind == "survival" {
            (&d.any, 1, &d.srisk_a, &d.srisk_b)
        } else {
            (&d.cr, 2, &d.risk_a, &d.risk_b)
        };
        let data = SurvData::builder(&d.time, status)
            .n_codes(n_codes)
            .build()
            .unwrap();
        let censoring_status: Vec<i32> = d.any.iter().map(|&s| 1 - s).collect();
        let cdata = SurvData::builder(&d.time, &censoring_status)
            .build()
            .unwrap();
        let ma = Mat::from_fn(n, t, |i, j| ra[i][j]);
        let mb = Mat::from_fn(n, t, |i, j| rb[i][j]);
        let cols: Vec<&Vec<f64>> = case
            .cens_cols
            .as_ref()
            .map(|c| {
                c.iter()
                    .map(|s| if s == "x1" { &d.x1 } else { &d.x2 })
                    .collect()
            })
            .unwrap_or_default();
        let x = Mat::from_fn(n, cols.len(), |i, j| cols[j][i]);
        let kappa = condition(&x, &cdata);
        let m = d.any.iter().filter(|&&s| s == 0).count() as f64;
        let nf = n as f64;
        let w = case.wmax;
        for run in &case.runs {
            let tag = format!("{} / conservative {}", case.name, run.conservative);
            let mut opts = ScoreOptions {
                keep_influence: true,
                conservative: run.conservative,
                ..ScoreOptions::default()
            };
            if run.cens == "cox" {
                opts.censoring = Censoring::Cox(CoxCensoring {
                    x: x.as_ref(),
                    options: CoxOptions {
                        tolerance: 1e-6,
                        ..CoxOptions::default()
                    },
                });
            }
            let ka = if run.cens == "cox" { 1.0 + kappa } else { 1.0 };
            let b_b = 4.0 * (nf + m + 3.0) * eps * w * ka;
            let b_a = 8.0 * (nf + m + 3.0) * eps * ka;
            let sa = brier(&data, ma.as_ref(), &d.times, &opts).unwrap();
            let sb = brier(&data, mb.as_ref(), &d.times, &opts).unwrap();
            let s0 = null_brier(&data, &d.times, &opts).unwrap();
            let aa = auc(&data, ma.as_ref(), &d.times, &opts).unwrap();
            let ab = auc(&data, mb.as_ref(), &d.times, &opts).unwrap();
            let big = |m: &Option<Influence>| {
                let m = m.as_ref().unwrap();
                (0..t)
                    .flat_map(|j| m.column(j).iter().map(|v| v.abs()))
                    .fold(0.0f64, f64::max)
            };
            let u = big(&sa.influence)
                .max(big(&sb.influence))
                .max(big(&s0.influence))
                .max(big(&aa.influence))
                .max(big(&ab.influence));
            let b_se = 8.0 * (nf + m + 3.0) * eps * u * ka;
            let ipa_a = ipa(&sa, &s0).unwrap();
            for (name, score, area) in [("a", &sa, &aa), ("b", &sb, &ab)] {
                let rm = &run.models[name];
                for j in 0..t {
                    worst.check(&tag, "brier", score.brier[j], rm.brier[j], b_b);
                    worst.check(&tag, "brier se", score.se[j].unwrap(), rm.brier_se[j], b_se);
                    worst.check(
                        &tag,
                        "auc",
                        area.auc[j].unwrap(),
                        rm.auc.as_ref().unwrap()[j],
                        b_a,
                    );
                    worst.check(
                        &tag,
                        "auc se",
                        area.se[j].unwrap(),
                        rm.auc_se.as_ref().unwrap()[j],
                        b_se,
                    );
                }
            }
            let rz = &run.models["Null model"];
            let ra_model = &run.models["a"];
            for j in 0..t {
                let (bj, zj) = (sa.brier[j], s0.brier[j]);
                worst.check(
                    &tag,
                    "null brier se",
                    s0.se[j].unwrap(),
                    rz.brier_se[j],
                    b_se,
                );
                // IPA moves by at most (b_B + (B / B0) b_B0) / B0, b_B0 = b_B.
                let b_ipa = b_b * (1.0 + bj / zj) / zj;
                worst.check(
                    &tag,
                    "ipa",
                    ipa_a.ipa[j].unwrap(),
                    ra_model.ipa.as_ref().unwrap()[j],
                    b_ipa,
                );
                worst.check(
                    &tag,
                    "ipa se",
                    ipa_a.se[j].unwrap(),
                    ra_model.ipa_se.as_ref().unwrap()[j],
                    b_se * 2.0 * (1.0 / zj + bj / (zj * zj)),
                );
            }
            let level = ContrastOptions::default();
            let pick = |name: &str| match name {
                "a" => &sa,
                "b" => &sb,
                _ => &s0,
            };
            for row in &run.brier_contrasts {
                let c = brier_difference(pick(&row.model), pick(&row.reference), &level).unwrap();
                let j = d.times.iter().position(|&x| x == row.times).unwrap();
                check_contrast(
                    &mut worst,
                    &tag,
                    "brier",
                    &c,
                    j,
                    row,
                    row.delta_brier.unwrap(),
                    2.0 * b_b,
                    b_se,
                    z,
                );
            }
            for row in &run.auc_contrasts {
                let pick_a = |name: &str| if name == "a" { &aa } else { &ab };
                let c = auc_difference(pick_a(&row.model), pick_a(&row.reference), &level).unwrap();
                let j = d.times.iter().position(|&x| x == row.times).unwrap();
                check_contrast(
                    &mut worst,
                    &tag,
                    "auc",
                    &c,
                    j,
                    row,
                    row.delta_auc.unwrap(),
                    2.0 * b_a,
                    b_se,
                    z,
                );
            }
        }
    }
    for (what, dev) in &worst.0 {
        eprintln!("worst {what}: {dev:.3} of its bound");
    }
}

#[expect(clippy::too_many_arguments, reason = "the bounds of one comparison")]
fn check_contrast(
    worst: &mut Worst,
    tag: &str,
    what: &str,
    c: &epistat_survival::metrics::contrast::Contrast,
    j: usize,
    row: &Row,
    delta: f64,
    b_delta: f64,
    b_se: f64,
    z: f64,
) {
    let name = |s: &str| format!("{what} {s}");
    worst.check(
        tag,
        &name("difference"),
        c.difference[j].unwrap(),
        delta,
        b_delta,
    );
    worst.check(tag, &name("difference se"), c.se[j].unwrap(), row.se, b_se);
    worst.check(
        tag,
        &name("lower"),
        c.lower[j].unwrap(),
        row.lower,
        b_delta + z * b_se + 4.0 * f64::EPSILON * row.lower.abs(),
    );
    worst.check(
        tag,
        &name("upper"),
        c.upper[j].unwrap(),
        row.upper,
        b_delta + z * b_se + 4.0 * f64::EPSILON * row.upper.abs(),
    );
    worst.check(
        tag,
        &name("p"),
        c.p_value[j].unwrap(),
        row.p,
        (0.8 * b_delta + 0.5 * b_se) / row.se,
    );
}
