//! Net survival and excess hazard models against R (`tests/data/relsurv.R`):
//! `relsurv::rs.surv` for the Pohar Perme and Ederer II estimators at every
//! exit time, the continuous estimators and the interval estimators written in
//! R from their definitions, `popEpi::relpois` and a `glm` on aggregated cells
//! for the piecewise excess hazard model, and `rstpm2::stpm2` with `bhazard` for
//! the flexible parametric excess hazard model. Each kind of quantity is
//! compared to a stated bound, and the worst ratio of deviation to bound is
//! printed with `--nocapture`.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    reason = "statuses, sexes, groups and levels are small whole numbers written by R as doubles"
)]

use std::collections::BTreeMap;

use epistat_linalg::Mat;
use epistat_survival::Level;
use epistat_survival::SurvData;
use epistat_survival::flexpar::{
    self, FlexParOptions, FlexparPredictOptions, FlexparScale, Predict,
};
use epistat_survival::ratetable::{RateDim, RateTable};
use epistat_survival::relsurv::{
    ExcessOptions, ExpectedHazard, Grid, NetMethod, NetOptions, NetTransform, excess_hazard,
    net_survival,
};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
#[expect(
    dead_code,
    reason = "the versions are read so that a missing one fails"
)]
struct Versions {
    r: String,
    survival: String,
    relsurv: String,
    #[serde(rename = "popEpi")]
    pop_epi: String,
    #[serde(rename = "Epi")]
    epi: String,
    rstpm2: String,
    jsonlite: String,
}

#[derive(Deserialize, Clone)]
struct Cohort {
    age0: Vec<f64>,
    date0: Vec<f64>,
    sex: Vec<f64>,
    x1: Vec<f64>,
    x2: Vec<f64>,
    x3: Vec<f64>,
    grp: Vec<f64>,
    time: Vec<f64>,
    status: Vec<f64>,
    start: Vec<f64>,
}

#[derive(Deserialize)]
struct Curve {
    time: Vec<f64>,
    n_risk: Vec<f64>,
    events: Vec<f64>,
    surv: Vec<f64>,
    std_err: Vec<f64>,
    lower: Vec<Option<f64>>,
    upper: Vec<Option<f64>>,
}

#[derive(Deserialize)]
struct RsRun {
    method: String,
    conf: String,
    strata: Vec<Curve>,
}

#[derive(Deserialize)]
struct IntervalRes {
    time: Vec<f64>,
    lambda: Vec<f64>,
    variance: Vec<f64>,
    n: Vec<f64>,
    events: Vec<f64>,
}

#[derive(Deserialize)]
struct IntervalRun {
    method: String,
    width: f64,
    res: IntervalRes,
}

#[derive(Deserialize)]
struct Glm {
    coef: Vec<f64>,
    vcov: Vec<Vec<f64>>,
    deviance: f64,
    n_records: Option<usize>,
    n_cells: Option<usize>,
}

#[derive(Deserialize)]
struct Band {
    #[serde(rename = "Estimate")]
    estimate: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

#[derive(Deserialize)]
struct Preds {
    surv: Vec<Band>,
    hazard: Vec<Band>,
}

#[derive(Deserialize)]
struct Stpm {
    df: usize,
    entry: bool,
    cohort: Cohort,
    bh: Vec<f64>,
    loglik: f64,
    coef: Vec<f64>,
    se: Vec<f64>,
    preds: Preds,
}

#[derive(Deserialize)]
struct Exact {
    pp: Vec<IntervalRes>,
    e2: Vec<IntervalRes>,
}

#[derive(Deserialize)]
#[expect(
    dead_code,
    reason = "the versions are read so that a missing one fails"
)]
struct Reference {
    versions: Versions,
    age_cuts: Vec<f64>,
    year_cuts: Vec<f64>,
    rates: Vec<f64>,
    cohort_a: Cohort,
    relsurv: Vec<RsRun>,
    relsurv_rule_deviation: f64,
    cohort_e: Cohort,
    exact: Exact,
    identity_deviation: f64,
    interval: Vec<IntervalRun>,
    cohort_c: Cohort,
    cuts: Vec<f64>,
    relpois: Glm,
    aggregated: Glm,
    stpm: Vec<Stpm>,
}

/// The two-sided 95% normal quantile.
const Z95: f64 = 1.959_963_984_540_054;

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/relsurv.json"
    ))
}

/// The table of the script, with the last cell of each time dimension
/// reaching far out, as the last cell of an R rate table does.
fn table(r: &Reference) -> RateTable {
    let mut age = r.age_cuts.clone();
    age.push(1e9);
    let mut year = r.year_cuts.clone();
    year.push(1e9);
    RateTable::new(
        vec![
            RateDim::Time { cuts: age },
            RateDim::Factor { levels: 2 },
            RateDim::Time { cuts: year },
        ],
        r.rates.clone(),
        1.0,
    )
    .unwrap()
}

struct Columns {
    status: Vec<i32>,
    sex: Vec<u32>,
    grp: Vec<u32>,
}

fn columns(c: &Cohort) -> Columns {
    Columns {
        status: c.status.iter().map(|&s| s as i32).collect(),
        sex: c.sex.iter().map(|&s| s as u32 - 1).collect(),
        grp: c.grp.iter().map(|&g| g as u32 - 1).collect(),
    }
}

#[derive(Default)]
struct Worst(BTreeMap<String, f64>);

impl Worst {
    fn note(&mut self, kind: &str, deviation: f64, bound: f64) {
        assert!(deviation.is_finite(), "{kind}: {deviation}");
        let e = self.0.entry(kind.to_owned()).or_insert(0.0);
        *e = e.max(deviation / bound);
    }
    /// `|a - b| / max(|b|, floor)` against `bound`.
    fn rel(&mut self, kind: &str, a: f64, b: f64, floor: f64, bound: f64) {
        self.note(kind, (a - b).abs() / b.abs().max(floor), bound);
    }
    fn finish(&self) {
        for (kind, w) in &self.0 {
            println!("{kind}: worst {w:.3} of its bound");
        }
        for (kind, w) in &self.0 {
            assert!(*w <= 1.0, "{kind}: {w} of its bound");
        }
    }
}

/// The bound on the error of `relsurv`'s Pohar Perme integral between two
/// consecutive exit times `a < b` of a stratum. Its integrand on each step of
/// half a day is a mean of values that `f = sum w lambda / sum w` takes on
/// `[a, b]`, and `f` is increasing within a step, whose hazards are constant,
/// since its derivative is the weighted variance of the hazards. So both its
/// integral and the exact one are within `(b - a) (max f - min f)` of each
/// other, with the extremes over the ends of the steps. The bound at each exit
/// time is the sum over the intervals up to it, with `1e-11` for rounding.
fn relsurv_bounds(c: &Cohort, rows: &[usize], expected: &ExpectedHazard) -> Vec<f64> {
    let mut times: Vec<f64> = rows.iter().map(|&i| c.time[i]).collect();
    times.sort_by(f64::total_cmp);
    times.dedup();
    let f_at = |active: &[usize], u: f64, lambda: &[f64]| {
        let (mut num, mut den) = (0.0, 0.0);
        for (k, &i) in active.iter().enumerate() {
            let w = expected.cumulative(i, u).unwrap().exp();
            num += w * lambda[k];
            den += w;
        }
        num / den
    };
    let (mut total, mut previous) = (1e-11, 0.0);
    let mut out = Vec::with_capacity(times.len());
    for &b in &times {
        let active: Vec<usize> = rows.iter().copied().filter(|&i| c.time[i] >= b).collect();
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        let mut u = previous;
        while u < b {
            let step = (b - u).min(0.5);
            let lambda: Vec<f64> = active
                .iter()
                .map(|&i| expected.hazard(i, u + step / 2.0).unwrap())
                .collect();
            for v in [f_at(&active, u, &lambda), f_at(&active, u + step, &lambda)] {
                lo = lo.min(v);
                hi = hi.max(v);
            }
            u += step;
        }
        total += (b - previous) * (hi - lo);
        out.push(total);
        previous = b;
    }
    out
}

#[test]
fn relsurv_rules_hold() {
    let r = reference();
    // relsurv's Pohar Perme estimate is its running-mean rule, transcribed in
    // the script, and the identity for the integral holds against quadrature.
    assert!(
        r.relsurv_rule_deviation < 1e-12,
        "{}",
        r.relsurv_rule_deviation
    );
    assert!(r.identity_deviation < 1e-10, "{}", r.identity_deviation);
}

#[test]
fn net_survival_matches_relsurv() {
    let r = reference();
    let t = table(&r);
    let c = &r.cohort_a;
    let cols = columns(c);
    let data = SurvData::builder(&c.time, &cols.status)
        .stratum(&cols.grp)
        .build()
        .unwrap();
    let expected =
        ExpectedHazard::from_table(&t, &data, &[&c.age0, &c.date0], &[&cols.sex]).unwrap();
    let bounds: Vec<Vec<f64>> = (0..3)
        .map(|g| {
            let rows: Vec<usize> = (0..c.time.len()).filter(|&i| cols.grp[i] == g).collect();
            relsurv_bounds(c, &rows, &expected)
        })
        .collect();
    println!(
        "bound on the Pohar Perme cumulative excess hazard at the last time: {:?}",
        bounds
            .iter()
            .map(|b| b.last().copied().unwrap())
            .collect::<Vec<_>>()
    );
    let mut worst = Worst::default();
    for run in &r.relsurv {
        let method = match run.method.as_str() {
            "pohar-perme" => NetMethod::PoharPerme,
            _ => NetMethod::EdererII,
        };
        let transform = match run.conf.as_str() {
            "log-log" => NetTransform::LogLog,
            "log" => NetTransform::Log,
            _ => NetTransform::Plain,
        };
        let opts = NetOptions {
            method,
            grid: Grid::EventTimes,
            level: Level::default(),
            transform,
            potential: None,
        };
        let curves = net_survival(&data, &expected, &opts).unwrap();
        assert_eq!(curves.len(), run.strata.len());
        for (g, (got, want)) in curves.iter().zip(&run.strata).enumerate() {
            assert_eq!(got.times, want.time, "{} times", run.method);
            assert_eq!(got.n_at_risk, want.n_risk, "{} at risk", run.method);
            assert_eq!(got.events, want.events, "{} events", run.method);
            let kind = |what: &str| format!("rs.surv {} {what}", run.method);
            assert_eq!(bounds[g].len(), got.times.len());
            for (k, &rule_bound) in bounds[g].iter().enumerate() {
                // Ederer II has relsurv's sums of the hazards of about 5000
                // steps, so rounding of about 1e-12 relative; Pohar Perme the
                // bound of its rule. An error in Lambda is a relative error in
                // exp(-Lambda).
                let bound = if method == NetMethod::PoharPerme {
                    rule_bound
                } else {
                    1e-11
                };
                worst.rel(&kind("survival"), got.survival[k], want.surv[k], 1.0, bound);
                // The variance has no integral, only rounding.
                worst.rel(
                    &kind("std.err"),
                    got.variance[k].sqrt(),
                    want.std_err[k],
                    1e-300,
                    1e-11,
                );
                assert_eq!(got.lower[k].is_some(), want.lower[k].is_some());
                // The limits are the estimate times a factor of the standard
                // error, divided by Lambda on the log log scale, which moves
                // by up to the bound relative to Lambda.
                let se = got.variance[k].sqrt();
                let lambda = got.cumulative_excess_hazard[k].abs();
                let limit_bound = bound * (1.0 + 2.0 * Z95 * se / lambda.max(1e-3)) + 1e-11;
                for (a, b) in [(got.lower[k], want.lower[k]), (got.upper[k], want.upper[k])] {
                    if let (Some(a), Some(b)) = (a, b) {
                        worst.rel(
                            &kind(&format!("{} limits", run.conf)),
                            a,
                            b,
                            1.0,
                            limit_bound,
                        );
                    }
                }
            }
        }
    }
    worst.finish();
}

#[test]
fn exact_estimators_match_their_definition() {
    let r = reference();
    let t = table(&r);
    let c = &r.cohort_e;
    let cols = columns(c);
    let data = SurvData::builder(&c.time, &cols.status)
        .start(&c.start)
        .stratum(&cols.grp)
        .build()
        .unwrap();
    let expected =
        ExpectedHazard::from_table(&t, &data, &[&c.age0, &c.date0], &[&cols.sex]).unwrap();
    let mut worst = Worst::default();
    for (method, refs, name) in [
        (NetMethod::PoharPerme, &r.exact.pp, "pp"),
        (NetMethod::EdererII, &r.exact.e2, "e2"),
    ] {
        let opts = NetOptions {
            method,
            ..NetOptions::default()
        };
        let curves = net_survival(&data, &expected, &opts).unwrap();
        for (got, want) in curves.iter().zip(refs) {
            assert_eq!(got.times, want.time);
            assert_eq!(got.n_at_risk, want.n);
            assert_eq!(got.events, want.events);
            // Each of up to 600 increments carries an error of a few eps, the
            // script's log of a ratio of sums near 1 the most, so 1e-11
            // absolute; the variance is a sum of positive terms.
            for k in 0..got.times.len() {
                worst.rel(
                    &format!("exact {name} Lambda"),
                    got.cumulative_excess_hazard[k],
                    want.lambda[k],
                    1.0,
                    1e-11,
                );
                worst.rel(
                    &format!("exact {name} variance"),
                    got.variance[k],
                    want.variance[k],
                    1e-300,
                    1e-12,
                );
            }
        }
    }
    worst.finish();
}

#[test]
fn interval_estimators_match_their_definition() {
    let r = reference();
    let t = table(&r);
    let all = &r.cohort_e;
    let keep: Vec<usize> = (0..all.grp.len()).filter(|&i| all.grp[i] == 1.0).collect();
    let pick = |v: &Vec<f64>| keep.iter().map(|&i| v[i]).collect::<Vec<f64>>();
    let c = Cohort {
        age0: pick(&all.age0),
        date0: pick(&all.date0),
        sex: pick(&all.sex),
        x1: pick(&all.x1),
        x2: pick(&all.x2),
        x3: pick(&all.x3),
        grp: pick(&all.grp),
        time: pick(&all.time),
        status: pick(&all.status),
        start: pick(&all.start),
    };
    let cols = columns(&c);
    let data = SurvData::builder(&c.time, &cols.status)
        .start(&c.start)
        .build()
        .unwrap();
    let expected =
        ExpectedHazard::from_table(&t, &data, &[&c.age0, &c.date0], &[&cols.sex]).unwrap();
    let mut worst = Worst::default();
    for run in &r.interval {
        let method = if run.method == "pp" {
            NetMethod::PoharPerme
        } else {
            NetMethod::EdererII
        };
        let opts = NetOptions {
            method,
            grid: Grid::Width(run.width),
            ..NetOptions::default()
        };
        let got = net_survival(&data, &expected, &opts).unwrap().remove(0);
        assert_eq!(got.times, run.res.time);
        assert_eq!(got.events, run.res.events);
        // The same sums in another order: each increment is a difference of
        // sums of about 200 terms of up to 3 times the increment, so about
        // 1e3 eps of the increment, and Lambda adds up to 100 of them.
        let kind = |what: &str| format!("interval {} {what}", run.method);
        for k in 0..got.times.len() {
            worst.rel(
                &kind("Lambda"),
                got.cumulative_excess_hazard[k],
                run.res.lambda[k],
                1e-2,
                1e-11,
            );
            worst.rel(
                &kind("variance"),
                got.variance[k],
                run.res.variance[k],
                1e-300,
                1e-12,
            );
            worst.rel(&kind("at risk"), got.n_at_risk[k], run.res.n[k], 1.0, 1e-12);
        }
    }
    worst.finish();
}

fn cohort_c(r: &Reference) -> (Columns, Vec<f64>, Vec<f64>) {
    let c = &r.cohort_c;
    let x3_1 =
        c.x3.iter()
            .map(|&v| f64::from(u8::from(v == 1.0)))
            .collect();
    let x3_2 =
        c.x3.iter()
            .map(|&v| f64::from(u8::from(v == 2.0)))
            .collect();
    (columns(c), x3_1, x3_2)
}

fn compare_glm(worst: &mut Worst, kind: &str, fit: &epistat_survival::relsurv::ExcessFit, g: &Glm) {
    let d = g.coef.len();
    assert_eq!(fit.coefficients.len(), d);
    assert!(fit.converged, "{kind}");
    let cov = fit.covariance.as_ref().unwrap();
    // glm stops at a relative change of 1e-14 in the deviance, after which a
    // quadratically convergent iteration leaves an error far below the square
    // root of that change in the coefficients; both are compared at 1e-9
    // relative, the deviance at 1e-12.
    for j in 0..d {
        worst.rel(
            &format!("{kind} coefficients"),
            fit.coefficients[j].unwrap(),
            g.coef[j],
            1e-2,
            1e-9,
        );
        for k in 0..d {
            worst.rel(
                &format!("{kind} covariance"),
                cov.get(j, k).unwrap(),
                g.vcov[j][k],
                (g.vcov[j][j] * g.vcov[k][k]).sqrt(),
                1e-9,
            );
        }
    }
    worst.rel(
        &format!("{kind} deviance"),
        fit.deviance,
        g.deviance,
        1.0,
        1e-12,
    );
}

#[test]
fn excess_hazard_matches_relpois_and_glm() {
    let r = reference();
    let t = table(&r);
    let c = &r.cohort_c;
    let (cols, x3_1, x3_2) = cohort_c(&r);
    let data = SurvData::builder(&c.time, &cols.status)
        .start(&c.start)
        .build()
        .unwrap();
    let expected =
        ExpectedHazard::from_table(&t, &data, &[&c.age0, &c.date0], &[&cols.sex]).unwrap();
    let mut worst = Worst::default();

    let fit = excess_hazard(
        &data,
        &[&c.x1, &c.x2],
        &expected,
        &ExcessOptions::new(&r.cuts),
    )
    .unwrap();
    assert_eq!(Some(fit.n_cells), r.relpois.n_records);
    assert!(fit.at_bound.iter().all(|&b| !b));
    compare_glm(&mut worst, "relpois", &fit, &r.relpois);

    let opts = ExcessOptions {
        aggregate: true,
        ..ExcessOptions::new(&r.cuts)
    };
    let fit = excess_hazard(&data, &[&c.x2, &x3_1, &x3_2], &expected, &opts).unwrap();
    assert_eq!(Some(fit.n_cells), r.aggregated.n_cells);
    compare_glm(&mut worst, "aggregated glm", &fit, &r.aggregated);
    worst.finish();
}

#[test]
fn flexible_excess_model_matches_stpm2() {
    let r = reference();
    let t = table(&r);
    let mut worst = Worst::default();
    let year = 365.25;
    for s in &r.stpm {
        let c = &s.cohort;
        let n = c.time.len();
        let cols = columns(c);
        let days = SurvData::builder(&c.time, &cols.status).build().unwrap();
        let expected =
            ExpectedHazard::from_table(&t, &days, &[&c.age0, &c.date0], &[&cols.sex]).unwrap();
        let bh: Vec<f64> = (0..n)
            .map(|i| expected.hazard(i, c.time[i]).unwrap() * year)
            .collect();
        for (a, b) in bh.iter().zip(&s.bh) {
            worst.rel("bhazard", *a, *b, 1e-300, 1e-15);
        }
        let time: Vec<f64> = c.time.iter().map(|v| v / year).collect();
        let start: Vec<f64> = c.start.iter().map(|v| v / year).collect();
        let mut b = SurvData::builder(&time, &cols.status);
        if s.entry {
            b = b.start(&start);
        }
        let data = b.build().unwrap();
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { c.x1[i] } else { c.x2[i] });
        let opts = FlexParOptions {
            bhazard: Some(&bh),
            ..FlexParOptions::new(FlexparScale::Hazard, s.df)
        };
        let f = flexpar::fit(&data, x.as_ref(), &opts).unwrap();
        assert!(f.converged);
        let label = format!("stpm2 df {}{}", s.df, if s.entry { " entry" } else { "" });
        // rstpm2 stops at a relative change of 1e-14 in the log likelihood,
        // its covariance is from a numerical Hessian, and its limits are formed
        // from that, as in r_flexpar.
        worst.rel(
            &format!("{label} loglik"),
            f.log_likelihood,
            s.loglik,
            1.0,
            1e-10,
        );
        let cov = f.covariance.as_ref().unwrap();
        for j in 0..2 {
            let bj = s.coef[1 + j];
            let sj = s.se[1 + j];
            worst.note(
                &format!("{label} beta in se"),
                (f.beta[j].unwrap() - bj).abs() / sj,
                1e-5,
            );
            let k = f.beta_index(j).unwrap().unwrap();
            worst.rel(
                &format!("{label} beta se"),
                cov.get(k, k).unwrap().sqrt(),
                sj,
                1e-300,
                1e-4,
            );
        }
        let times = [0.1, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0];
        let profiles = Mat::from_fn(2, 2, |i, j| [[0.5, 1.0], [-0.7, 0.0]][i][j]);
        for (what, bands) in [
            (Predict::Survival, &s.preds.surv),
            (Predict::Hazard, &s.preds.hazard),
        ] {
            let p = flexpar::predict(
                &f,
                profiles.as_ref(),
                &times,
                what,
                &FlexparPredictOptions::default(),
            )
            .unwrap();
            for (row, band) in p.iter().zip(bands.iter()) {
                for (k, pr) in row.iter().enumerate() {
                    worst.rel(
                        &format!("{label} {what:?}"),
                        pr.estimate,
                        band.estimate[k],
                        1e-300,
                        1e-6,
                    );
                    // With bhazard rstpm2 forms the limits of the hazard on its
                    // own scale and those of the survival on the log scale, so
                    // the standard errors they imply are compared.
                    let se = if what == Predict::Hazard {
                        (band.upper[k] - band.lower[k]) / (2.0 * Z95)
                    } else {
                        band.estimate[k] * (band.upper[k] / band.lower[k]).ln() / (2.0 * Z95)
                    };
                    worst.rel(&format!("{label} {what:?} se"), pr.se, se, 1e-300, 1e-5);
                }
            }
        }
    }
    worst.finish();
}
