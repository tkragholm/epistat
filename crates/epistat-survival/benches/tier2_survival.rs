//! Wall time of the risk by a time with censoring by augmented inverse
//! probability weighting, the per-protocol weights and marginal structural model
//! of sequential trials, and clone-censor-weight, set against
//! `riskRegression::ate`, `TrialEmulation` and a written-out R estimator for
//! clone-censor-weight (`bench/tier2/survival.R`). The
//! driver environment is described in `bench/tier1/common.rs`: one process times
//! one method on one size.
//!
//! Run by hand, with `SCRATCH` outside the repository:
//!
//! ```text
//! N=100000 METHOD=gen_surv SCRATCH=/x cargo bench -p epistat-survival --bench tier2_survival
//! N=100000 METHOD=aipw_cox SCRATCH=/x REPS=3 cargo bench -p epistat-survival --bench tier2_survival
//! ```
//!
//! Methods:
//!
//! * `aipw_km`, `aipw_cox` (data `surv`, `N` persons): the augmented risks at
//!   three times with a Cox outcome model, a logistic treatment model and a
//!   Kaplan-Meier or Cox model of the censoring.
//! * `trial_msm` (data `trial`, `N` persons with eight periods): the switch
//!   weights, the weights of the rows of the stacked trials and the weighted
//!   outcome model with its covariance. The trials are expanded before the
//!   timing, which `TrialEmulation` includes in its time.
//! * `ccw` (data `ccw`, `N` persons with ten periods): the clones, the weights of
//!   the artificial censoring and the weighted outcome model.
//! * `trial_msm_rows`, `ccw_rows`: the same with the outcome model fitted by
//!   `fit_msm_trials` and `fit_msm_clones`, which form the rows of the trials
//!   or clones block by block instead of holding them, and for `ccw_rows` the
//!   deviation weights by `deviation_weights_rows` from designs formed row by
//!   row. The data are read into
//!   columns of their final size and the table is dropped before the timing,
//!   so that the peak memory is that of the method. The numbers are those of
//!   `trial_msm` and `ccw` bit for bit.

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::aipw::{AipwSurvOptions, AipwSurvSpec, Censoring, Estimator, aipw_cox};
use epistat_survival::cox::Ties;
use epistat_survival::standardise::Counterfactual;
use epistat_survival::weighting::clone::{
    Adherence, CloneInput, DeviationDesign, DeviationModel, DeviationRows, Strategy, clone_censor,
    deviation_weights, deviation_weights_rows, fit_msm_clones,
};
use epistat_survival::weighting::trials::WeightRows;
use epistat_survival::weighting::trials::{
    MsmInput, PersonPeriod, TrialRows, TrialSpans, WeightModel, fit_msm, fit_msm_trials,
    switch_weights, trial_weights,
};

const SEED: u64 = 20_261_207;
const PERIODS: usize = 8;

fn expit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn generate_surv(n: usize) {
    let mut rng = Rng(SEED);
    let mut out = Writer::create(&common::data_path("surv", n), "time,status,a,x1,x2");
    for _ in 0..n {
        let x1 = rng.normal();
        let x2 = f64::from(u8::from(rng.bernoulli(0.4)));
        let a = f64::from(u8::from(rng.bernoulli(expit(0.5 * x1 - 0.4 * x2))));
        let te = -rng.uniform().ln() / (0.12 * (0.5 * x1 - 0.4 * x2 - 0.6 * a).exp());
        let tc = (-rng.uniform().ln() / (0.06 * (0.3 * x1 + 0.3 * a).exp())).min(12.0);
        out.row(&[te.min(tc), f64::from(u8::from(te <= tc)), a, x1, x2]);
    }
}

/// Person-period rows with a covariate that depends on the earlier treatment,
/// a persistent treatment and an outcome.
fn generate_periods(kind: &str, n: usize, periods: usize) {
    let mut rng = Rng(SEED + 1);
    let mut out = Writer::create(
        &common::data_path(kind, n),
        "id,period,a,am1,l,l0,x1,outcome",
    );
    for id in 0..n {
        let x1 = f64::from(u8::from(rng.bernoulli(0.5)));
        let l0 = rng.normal();
        let (mut l, mut prev) = (l0, 0.0);
        for p in 0..periods {
            l = 0.5 * l + 0.4 * prev + 0.7 * rng.normal();
            let pa = if prev > 0.5 {
                expit(2.4 - 0.5 * l)
            } else {
                expit(-2.2 + 0.7 * l + 0.1 * p as f64 + 0.3 * x1)
            };
            let a = f64::from(u8::from(rng.bernoulli(pa)));
            let y = rng.bernoulli(expit(-4.0 - 0.6 * a + 0.4 * l + 0.05 * p as f64));
            out.row(&[
                id as f64,
                p as f64,
                a,
                prev,
                l,
                l0,
                x1,
                f64::from(u8::from(y)),
            ]);
            prev = a;
            if y {
                break;
            }
        }
    }
}

fn main() {
    let (method, n) = (common::method(), common::size());
    match method.as_str() {
        "gen_surv" => return generate_surv(n),
        "gen_trial" => return generate_periods("trial", n, PERIODS),
        "gen_ccw" => return generate_periods("ccw", n, 10),
        _ => {}
    }
    let mut lines = Lines::default();
    let times = match method.as_str() {
        "aipw_km" | "aipw_cox" => {
            let t = Table::read(&common::data_path("surv", n));
            let status: Vec<i32> = t.ints("status");
            let treated = t.flags("a");
            let (a, x1, x2) = (t.col("a"), t.col("x1"), t.col("x2"));
            let outcome_x = Mat::from_fn(n, 3, |i, j| [a[i], x1[i], x2[i]][j]);
            let propensity_x = Mat::from_fn(n, 3, |i, j| [1.0, x1[i], x2[i]][j]);
            let group: Vec<u32> = treated.iter().map(|&b| u32::from(b)).collect();
            let data = SurvData::builder(t.col("time"), &status)
                .build()
                .expect("data");
            let (set0, set1) = ([(0_usize, 0.0)], [(0_usize, 1.0)]);
            let censoring = if method == "aipw_km" {
                Censoring::KaplanMeier {
                    groups: Some(&group),
                }
            } else {
                Censoring::Cox {
                    x: outcome_x.as_ref(),
                }
            };
            let spec = AipwSurvSpec {
                treatment: &treated,
                outcome_x: outcome_x.as_ref(),
                levels: [
                    Counterfactual::Columns(&set0),
                    Counterfactual::Columns(&set1),
                ],
                propensity_x: propensity_x.as_ref(),
                censoring,
            };
            let opts = AipwSurvOptions {
                estimator: Estimator::Aiptw,
                ties: Ties::Breslow,
                ..AipwSurvOptions::default()
            };
            let at = [2.0, 5.0, 8.0];
            let (times, r) = common::timed(common::reps(), || {
                aipw_cox(&data, &spec, &at, &opts).expect("aipw")
            });
            let c = r.risks.contrast(1, 0).expect("contrast");
            for (k, &tau) in at.iter().enumerate() {
                lines.push_opt(&format!("risk0_{tau}"), r.risks.risk[k]);
                lines.push_opt(&format!("risk1_{tau}"), r.risks.risk[at.len() + k]);
                lines.push_opt(&format!("diff_{tau}"), c.difference[k]);
                lines.push_opt(
                    &format!("diff_se_{tau}"),
                    c.difference_inference.as_ref().and_then(|i| i.se[k]),
                );
            }
            times
        }
        "trial_msm" => {
            let t = Table::read(&common::data_path("trial", n));
            let rows = t.len();
            let id: Vec<u64> = t.ints("id");
            let period: Vec<i64> = t.ints("period");
            let a = t.flags("a");
            let am1 = t.flags("am1");
            let (l, x1) = (t.col("l"), t.col("x1"));
            let outcome = t.flags("outcome");
            let den = Mat::from_fn(rows, 3, |i, j| [1.0, l[i], x1[i]][j]);
            let num = Mat::from_fn(rows, 2, |i, j| [1.0, x1[i]][j]);
            // Every period starts a trial that lasts until the person's last row.
            let (mut tid, mut tstart, mut tfollow, mut src) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let mut first = 0;
            while first < rows {
                let mut last = first;
                while last + 1 < rows && id[last + 1] == id[first] {
                    last += 1;
                }
                for s in first..=last {
                    for r in s..=last {
                        tid.push(id[first]);
                        tstart.push(period[s]);
                        tfollow.push(period[r] - period[s]);
                        src.push((s, r));
                    }
                }
                first = last + 1;
            }
            let m = tid.len();
            let design = Mat::from_fn(m, 5, |i, j| {
                let (s, _) = src[i];
                let f = tfollow[i] as f64;
                [1.0, f64::from(u8::from(a[s])), f, f * f, x1[s]][j]
            });
            let y: Vec<bool> = src.iter().map(|&(_, r)| outcome[r]).collect();
            let data = PersonPeriod {
                id: &id,
                period: &period,
                treatment: &a,
                previous: &am1,
            };
            let model = WeightModel {
                numerator: num.as_ref(),
                denominator: den.as_ref(),
            };
            let (times, msm) = common::timed(common::reps(), || {
                let w =
                    switch_weights(&data, &model, &WeightRows::default()).expect("switch weights");
                let tw = trial_weights(
                    &data,
                    &w.weight,
                    &TrialRows {
                        id: &tid,
                        trial_period: &tstart,
                        follow_up: &tfollow,
                    },
                )
                .expect("trial weights");
                fit_msm(&MsmInput {
                    design: design.as_ref(),
                    outcome: &y,
                    weight: &tw,
                    person: &tid,
                })
                .expect("msm")
            });
            lines.push("rows", m as f64);
            lines.push_all("coef", &msm.coefficients);
            for (i, se) in msm.se().into_iter().enumerate() {
                lines.push_opt(&format!("se{i}"), se);
            }
            times
        }
        "ccw" => {
            let t = Table::read(&common::data_path("ccw", n));
            let rows = t.len();
            let id: Vec<u64> = t.ints("id");
            let period: Vec<i64> = t.ints("period");
            let a = t.flags("a");
            let outcome = t.flags("outcome");
            let (l, l0) = (t.col("l"), t.col("l0"));
            let strategies = [
                Strategy::InitiateWithin {
                    grace: 3,
                    adherence: Adherence::Continuous,
                },
                Strategy::NeverInitiate { until: None },
            ];
            let (times, msm) = common::timed(common::reps(), || {
                let c = clone_censor(
                    &CloneInput {
                        id: &id,
                        period: &period,
                        treatment: &a,
                        outcome: &outcome,
                        covariates: &[],
                    },
                    &strategies,
                )
                .expect("clones");
                let m = c.len();
                let den = Mat::from_fn(m, 5, |i, j| {
                    let s = c.source[i];
                    let p = period[s] as f64;
                    [1.0, l[s], l0[s], p, p * p][j]
                });
                let num = Mat::from_fn(m, 2, |i, j| [1.0, period[c.source[i]] as f64][j]);
                let model = DeviationModel {
                    denominator: den.as_ref(),
                    numerator: Some(num.as_ref()),
                };
                let w = deviation_weights(&c, &[model, model], None).expect("weights");
                let keep: Vec<usize> = (0..m).filter(|&i| !c.deviation[i]).collect();
                let design = Mat::from_fn(keep.len(), 5, |k, j| {
                    let i = keep[k];
                    let p = period[c.source[i]] as f64;
                    let arm = c.arm[i] as f64;
                    [1.0, arm, p, p * p, arm * p][j]
                });
                let y: Vec<bool> = keep.iter().map(|&i| c.outcome[i]).collect();
                let wt: Vec<f64> = keep.iter().map(|&i| w.weight[i]).collect();
                let person: Vec<u64> = keep.iter().map(|&i| u64::from(c.person[i])).collect();
                fit_msm(&MsmInput {
                    design: design.as_ref(),
                    outcome: &y,
                    weight: &wt,
                    person: &person,
                })
                .expect("msm")
            });
            lines.push("rows", rows as f64);
            lines.push_all("coef", &msm.coefficients);
            for (i, se) in msm.se().into_iter().enumerate() {
                lines.push_opt(&format!("se{i}"), se);
            }
            times
        }
        "trial_msm_rows" => {
            let (id, period, a, am1, l, x1, outcome) = {
                let t = Table::read_sized(&common::data_path("trial", n));
                (
                    t.ints::<u64>("id"),
                    t.ints::<i64>("period"),
                    t.flags("a"),
                    t.flags("am1"),
                    t.col("l").to_vec(),
                    t.col("x1").to_vec(),
                    t.flags("outcome"),
                )
            };
            let rows = id.len();
            let den = Mat::from_fn(rows, 3, |i, j| [1.0, l[i], x1[i]][j]);
            let num = Mat::from_fn(rows, 2, |i, j| [1.0, x1[i]][j]);
            drop(l);
            // Every period starts a trial that lasts until the person's last row.
            let mut end = vec![0; rows];
            let mut first = 0;
            while first < rows {
                let mut last = first;
                while last + 1 < rows && id[last + 1] == id[first] {
                    last += 1;
                }
                end[first..=last].fill(last + 1);
                first = last + 1;
            }
            let start: Vec<usize> = (0..rows).collect();
            let m: usize = end.iter().zip(&start).map(|(e, s)| e - s).sum();
            let data = PersonPeriod {
                id: &id,
                period: &period,
                treatment: &a,
                previous: &am1,
            };
            let model = WeightModel {
                numerator: num.as_ref(),
                denominator: den.as_ref(),
            };
            let (times, msm) = common::timed(common::reps(), || {
                let w =
                    switch_weights(&data, &model, &WeightRows::default()).expect("switch weights");
                fit_msm_trials(
                    &data,
                    &w.weight,
                    &TrialSpans {
                        start: &start,
                        end: &end,
                    },
                    &outcome,
                    5,
                    &|s, r, out| {
                        let f = (period[r] - period[s]) as f64;
                        out.copy_from_slice(&[1.0, f64::from(u8::from(a[s])), f, f * f, x1[s]]);
                    },
                )
                .expect("msm")
            });
            lines.push("rows", m as f64);
            lines.push_all("coef", &msm.coefficients);
            for (i, se) in msm.se().into_iter().enumerate() {
                lines.push_opt(&format!("se{i}"), se);
            }
            times
        }
        "ccw_rows" => {
            let (id, period, a, outcome, l, l0) = {
                let t = Table::read_sized_only(
                    &common::data_path("ccw", n),
                    &["id", "period", "a", "outcome", "l", "l0"],
                );
                (
                    t.ints::<u64>("id"),
                    t.ints::<i64>("period"),
                    t.flags("a"),
                    t.flags("outcome"),
                    t.col("l").to_vec(),
                    t.col("l0").to_vec(),
                )
            };
            let rows = id.len();
            let strategies = [
                Strategy::InitiateWithin {
                    grace: 3,
                    adherence: Adherence::Continuous,
                },
                Strategy::NeverInitiate { until: None },
            ];
            let (times, msm) = common::timed(common::reps(), || {
                let c = clone_censor(
                    &CloneInput {
                        id: &id,
                        period: &period,
                        treatment: &a,
                        outcome: &outcome,
                        covariates: &[],
                    },
                    &strategies,
                )
                .expect("clones");
                let den = |i: usize, out: &mut [f64]| {
                    let s = c.source[i];
                    let p = period[s] as f64;
                    out.copy_from_slice(&[1.0, l[s], l0[s], p, p * p]);
                };
                let num = |i: usize, out: &mut [f64]| {
                    out.copy_from_slice(&[1.0, period[c.source[i]] as f64]);
                };
                let model = DeviationRows {
                    denominator: DeviationDesign {
                        columns: 5,
                        fill: &den,
                    },
                    numerator: Some(DeviationDesign {
                        columns: 2,
                        fill: &num,
                    }),
                };
                let w = deviation_weights_rows(&c, &[model, model], None).expect("weights");
                fit_msm_clones(&c, &w.weight, 5, &|i, out| {
                    let p = period[c.source[i]] as f64;
                    let arm = c.arm[i] as f64;
                    out.copy_from_slice(&[1.0, arm, p, p * p, arm * p]);
                })
                .expect("msm")
            });
            lines.push("rows", rows as f64);
            lines.push_all("coef", &msm.coefficients);
            for (i, se) in msm.se().into_iter().enumerate() {
                lines.push_opt(&format!("se{i}"), se);
            }
            times
        }
        other => panic!("unknown method {other}"),
    };
    common::finish(&method, n, &times, &lines);
}
