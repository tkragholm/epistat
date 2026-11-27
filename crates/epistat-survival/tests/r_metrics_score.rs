//! `metrics::score` and `metrics::calibration` checked against
//! `riskRegression::Score` and `plotCalibration` (2026.3.11) and against
//! `timeROC::timeROC` (0.4.1), on risks that the R script writes out.
//!
//! The area's variance is matched to `Score`'s, not to `timeROC`'s. `Score`
//! reports `sd(IF) / sqrt(N)` for an analytic influence function, which the
//! crate reproduces to rounding error, ties between deaths and censorings
//! included. `timeROC`'s `iid = TRUE` standard error is another finite-sample
//! estimator of the same variance. It agrees with `Score`'s to within rounding
//! error at some times and differs by up to 6 parts in 10^4 (the area with
//! competing events as controls) at others, and the differences are printed
//! here, not asserted.
//! The point estimate of `timeROC` is asserted, with the crate's
//! `HorizonRule::Exclusive` and both control sets.
//!
//! # Tolerances
//!
//! `n` rows, `m` distinct censoring times, `w` the largest weight `1 / G`, with
//! `G` the censoring distribution computed in the test from its definition.
//!
//! - The Brier score is a mean of `n` terms in `[0, w]`, each with the relative
//!   error of a product of at most `m` factors: `b_B = 4 (n + m + 3) eps w`
//!   between two implementations. The null model's risk has the absolute
//!   error `4 (n + 2) eps` between them, which moves the score by at most twice
//!   that times `w`, so `b_B0 = b_B + 8 (n + 2) eps w`. The IPA moves by at
//!   most `(b_B + (B / B_0) b_B0) / B_0`.
//! - The area is a ratio of weighted sums of `n` terms and a double sum
//!   evaluated by cumulative sums in both programs: `b_A = 8 (n + m + 3) eps`.
//! - A standard error is the root mean square of an influence function. Its
//!   entries are sums of at most `n + m` terms of the size `U` of the pieces of
//!   the function, so `b_se = 8 (n + m + 3) eps U`. With `kappa = max 1 / Y(T)
//!   + sum_u dC(u) / Y(u)^2`, `Y` the number with a time at or after `u` and
//!   `dC` the number censored, the pieces of the Brier score's function are at
//!   most `U_B = w (1 + n kappa)`. For the area they are at most
//!   `U_A = h (2 + n kappa)` with `h = 2 w n / min(cases, controls)`, since a
//!   weight `n a_i` is at most `w` and the sums of weights are at least
//!   `cases / n` and `controls / n`. The IPA's function has pieces of at most
//!   `U_B (1 / B_0 + B / B_0^2)`.
//! - The calibration's mean risk has the absolute error `2 (n + 2) eps` and its
//!   observed risk, a product-limit estimate, the same.
//!
//! The bounds are not changed after a run. The worst deviation in units of its
//! bound is printed for each output.

#![expect(
    clippy::float_cmp,
    reason = "times tie when they are equal, which is the definition of a tie"
)]

use std::collections::BTreeMap;

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::metrics::calibration::{self, CalibrationOptions, Groups};
use epistat_survival::metrics::score::{
    Censoring, ControlSet, HorizonRule, ScoreOptions, SuppliedCensoring, auc, brier, ipa,
    null_brier,
};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Versions {
    r: String,
    survival: String,
    #[serde(rename = "riskRegression")]
    risk_regression: String,
    #[serde(rename = "timeROC")]
    time_roc: String,
    prodlim: String,
}

#[derive(Deserialize)]
struct Weights {
    at_times: Vec<Vec<f64>>,
    before_own: Vec<f64>,
}

#[derive(Deserialize)]
struct Run {
    label: String,
    brier: Vec<f64>,
    brier_se: Vec<f64>,
    null_brier: Vec<f64>,
    ipa: Vec<f64>,
    ipa_se: Vec<f64>,
    auc: Vec<f64>,
    auc_se: Vec<f64>,
    weights: Option<Weights>,
    conservative: bool,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    dataset: String,
    kind: String,
    times: Vec<f64>,
    score: Vec<Run>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct Dataset {
    time: Vec<f64>,
    status: Vec<i32>,
    cr: Vec<i32>,
    x: Vec<f64>,
    risk: Vec<Vec<f64>>,
    risk_obs: Vec<Vec<f64>>,
    risk_s: Vec<Vec<f64>>,
    risk_cr: Vec<Vec<f64>>,
    risk_cr_obs: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct Troc {
    times: Vec<f64>,
    #[serde(deserialize_with = "epistat_testkit::na_f64_vec")]
    auc1: Vec<f64>,
    auc2: Option<Vec<f64>>,
    #[serde(deserialize_with = "epistat_testkit::na_f64_vec")]
    se1: Vec<f64>,
    #[serde(deserialize_with = "epistat_testkit::na_f64_vec")]
    se2: Vec<f64>,
}

#[derive(Deserialize)]
struct CalFrame {
    time: f64,
    q: usize,
    pred: Vec<f64>,
    obs: Vec<Option<f64>>,
    groups: Vec<f64>,
}

#[derive(Deserialize)]
struct CalFew {
    time: f64,
    risk: Vec<f64>,
    r_pred: Vec<f64>,
    values: Vec<f64>,
    obs_aligned: Vec<f64>,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    datasets: BTreeMap<String, Dataset>,
    cases: Vec<Case>,
    timeroc: BTreeMap<String, Troc>,
    calibration: BTreeMap<String, Vec<CalFrame>>,
    calibration_few: CalFew,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/metrics_score.json"
    ))
}

#[derive(Default)]
struct Worst(BTreeMap<String, f64>);

impl Worst {
    fn put(&mut self, name: &str, value: f64) {
        let entry = self.0.entry(name.to_string()).or_insert(0.0);
        *entry = entry.max(value);
    }
    fn report(&self) {
        for (what, value) in &self.0 {
            eprintln!("worst {what}: {value:.3} of its bound");
        }
    }
}

/// `(m, w, kappa)` for the data and the largest time, from the definitions.
fn facts(time: &[f64], status: &[i32], t_max: f64) -> (f64, f64, f64) {
    let n = time.len();
    let mut cens: Vec<f64> = (0..n)
        .filter(|&i| status[i] == 0)
        .map(|i| time[i])
        .collect();
    cens.sort_by(f64::total_cmp);
    cens.dedup();
    let mut g = 1.0;
    let mut g_at = 1.0;
    let mut kappa = 0.0f64;
    let mut sum = 0.0;
    for &u in &cens {
        let dc = (0..n).filter(|&i| status[i] == 0 && time[i] == u).count() as f64;
        // Deaths at u have left the risk set of the estimate.
        let risk = (0..n)
            .filter(|&i| time[i] > u || (time[i] == u && status[i] == 0))
            .count() as f64;
        g *= 1.0 - dc / risk;
        if u <= t_max {
            g_at = g;
        }
        let plain = (0..n).filter(|&i| time[i] >= u).count() as f64;
        sum += dc / (plain * plain);
        kappa = kappa.max(1.0 / plain);
    }
    (cens.len() as f64, 1.0 / g_at, kappa + sum)
}

#[test]
fn score_against_riskregression() {
    let r = reference();
    eprintln!(
        "reference: {}, survival {}, riskRegression {}, timeROC {}, prodlim {}",
        r.versions.r,
        r.versions.survival,
        r.versions.risk_regression,
        r.versions.time_roc,
        r.versions.prodlim
    );
    let eps = f64::EPSILON;
    let mut worst = Worst::default();
    for case in &r.cases {
        let d = &r.datasets[&case.dataset];
        let (status, n_codes, risk) = match (
            case.dataset.as_str(),
            case.kind.as_str(),
            case.name.as_str(),
        ) {
            ("sim", _, "sim") => (&d.status, 1, &d.risk),
            ("sim", _, _) => (&d.status, 1, &d.risk_obs),
            (_, "survival", _) => (&d.status, 1, &d.risk_s),
            (_, _, name) if name.contains("observed") => (&d.cr, 2, &d.risk_cr_obs),
            _ => (&d.cr, 2, &d.risk_cr),
        };
        let n = d.time.len();
        let nf = n as f64;
        let t = case.times.len();
        let data = SurvData::builder(&d.time, status)
            .n_codes(n_codes)
            .build()
            .unwrap();
        let m = Mat::from_fn(n, t, |i, j| risk[i][j]);
        let t_max = case.times.last().copied().unwrap();
        let (m_cens, mut w, kappa) = facts(&d.time, status, t_max);
        for run in &case.score {
            let weights = run.weights.as_ref().map(|w| {
                (
                    w.before_own.clone(),
                    Mat::from_fn(n, t, |i, j| w.at_times[i][j]),
                )
            });
            if let Some((own, at)) = &weights {
                let smallest = own
                    .iter()
                    .copied()
                    .chain(
                        (0..n)
                            .flat_map(|i| (0..t).map(move |j| (i, j)))
                            .map(|(i, j)| at[(i, j)]),
                    )
                    .fold(1.0f64, f64::min);
                w = 1.0 / smallest;
            }
            let mut opts = ScoreOptions {
                keep_influence: true,
                conservative: run.conservative,
                ..ScoreOptions::default()
            };
            if let Some((own, at)) = &weights {
                opts.censoring = Censoring::Supplied(SuppliedCensoring {
                    before_own_time: own,
                    at_times: at.as_ref(),
                });
            }
            let tag = format!("{} / {}", case.name, run.label);
            let b = brier(&data, m.as_ref(), &case.times, &opts).unwrap();
            let z = null_brier(&data, &case.times, &opts).unwrap();
            let p = ipa(&b, &z).unwrap();
            let a = auc(&data, m.as_ref(), &case.times, &opts).unwrap();
            let base = 8.0 * (nf + m_cens + 3.0) * eps;
            let b_b = 4.0 * (nf + m_cens + 3.0) * eps * w;
            let u_b = w * (1.0 + nf * kappa);
            let mut check = |what: &str, j: usize, got: f64, want: f64, bound: f64| {
                let dev = (got - want).abs() / bound;
                assert!(
                    dev <= 1.0,
                    "{tag}: {what} at time {} is {got}, R has {want}, bound {bound}",
                    case.times[j]
                );
                worst.put(what, dev);
            };
            for j in 0..t {
                let (bj, zj) = (b.brier[j], z.brier[j]);
                let b_b0 = b_b + 8.0 * (nf + 2.0) * eps * w;
                check("brier", j, bj, run.brier[j], b_b);
                check("null brier", j, zj, run.null_brier[j], b_b0);
                check(
                    "ipa",
                    j,
                    p.ipa[j].unwrap(),
                    run.ipa[j],
                    (b_b + (bj / zj) * b_b0) / zj,
                );
                check("brier se", j, b.se[j].unwrap(), run.brier_se[j], base * u_b);
                let mut inf: f64 = 0.0;
                for i in 0..n {
                    inf = inf.max(b.influence.as_ref().unwrap().column(j)[i].abs());
                }
                assert!(
                    inf <= u_b * (1.0 + 1e-9),
                    "{tag}: influence {inf} exceeds its piece bound"
                );
                check(
                    "ipa se",
                    j,
                    p.se[j].unwrap(),
                    run.ipa_se[j],
                    base * u_b * 2.0 * (1.0 / zj + bj / (zj * zj)),
                );
                // The area.
                let (cases, controls) = (
                    a.n_cases[j] as f64,
                    (a.n_survivors[j] + a.n_competing[j]) as f64,
                );
                let h = 2.0 * w * nf / cases.min(controls);
                check("auc", j, a.auc[j].unwrap(), run.auc[j], base);
                check(
                    "auc se",
                    j,
                    a.se[j].unwrap(),
                    run.auc_se[j],
                    base * h * (2.0 + nf * kappa),
                );
            }
        }
    }
    worst.report();
}

#[test]
fn auc_against_timeroc() {
    let r = reference();
    let eps = f64::EPSILON;
    let mut worst = Worst::default();
    for (key, ds, competing) in [
        ("sim", "sim", false),
        ("sim_obs", "sim", false),
        ("melanoma", "melanoma", false),
        ("melanoma_cr", "melanoma", true),
        ("melanoma_cr_obs", "melanoma", true),
    ] {
        let tr = &r.timeroc[key];
        let d = &r.datasets[ds];
        let status = if competing { &d.cr } else { &d.status };
        let n = d.time.len();
        let nf = n as f64;
        let n_codes = if competing { 2 } else { 1 };
        let data = SurvData::builder(&d.time, status)
            .n_codes(n_codes)
            .build()
            .unwrap();
        let marker: Vec<f64> = if ds == "sim" {
            d.x.clone()
        } else {
            d.risk_cr.iter().map(|r| r[1]).collect()
        };
        let t = tr.times.len();
        let m = Mat::from_fn(n, t, |i, _| marker[i]);
        let (m_cens, _, _) = facts(&d.time, status, tr.times[t - 1]);
        let bound = 8.0 * (nf + m_cens + 3.0) * eps;
        for (controls, want) in [
            (ControlSet::EventFree, &tr.auc1),
            (
                ControlSet::EventFreeAndCompeting,
                tr.auc2.as_ref().unwrap_or(&tr.auc1),
            ),
        ] {
            let opts = ScoreOptions {
                controls,
                horizon: HorizonRule::Exclusive,
                ..ScoreOptions::default()
            };
            let a = auc(&data, m.as_ref(), &tr.times, &opts).unwrap();
            let se_want = if controls == ControlSet::EventFree {
                &tr.se1
            } else {
                &tr.se2
            };
            for j in 0..t {
                let dev = (a.auc[j].unwrap() - want[j]).abs() / bound;
                assert!(
                    dev <= 1.0,
                    "{key} {controls:?} t={}: {} against {}",
                    tr.times[j],
                    a.auc[j].unwrap(),
                    want[j]
                );
                worst.put("auc against timeROC", dev);
                if se_want[j].is_finite() {
                    eprintln!(
                        "{key} {controls:?} t={}: se {:.6e}, timeROC {:.6e}, relative difference {:.2e}",
                        tr.times[j],
                        a.se[j].unwrap(),
                        se_want[j],
                        (a.se[j].unwrap() - se_want[j]).abs() / se_want[j]
                    );
                }
            }
        }
        // Where Score and timeROC differ: a death exactly at the time.
        let inclusive = auc(&data, m.as_ref(), &tr.times, &ScoreOptions::default()).unwrap();
        let exclusive = auc(
            &data,
            m.as_ref(),
            &tr.times,
            &ScoreOptions {
                horizon: HorizonRule::Exclusive,
                ..ScoreOptions::default()
            },
        )
        .unwrap();
        let hits = tr.times.iter().filter(|t| d.time.contains(t)).count();
        let differs = (0..t)
            .filter(|&j| inclusive.auc[j] != exclusive.auc[j])
            .count();
        eprintln!("{key}: {hits} times are observed times, the two rules differ at {differs}");
        if hits == 0 {
            assert_eq!(differs, 0, "{key}: no time is observed, so the rules agree");
        }
    }
    worst.report();
}

#[test]
fn calibration_against_plotcalibration() {
    let r = reference();
    let eps = f64::EPSILON;
    let mut worst = Worst::default();
    for (key, frames) in &r.calibration {
        let d = &r.datasets[if key.starts_with("sim") {
            "sim"
        } else {
            "melanoma"
        }];
        let competing = key == "melanoma_cr";
        let status = if competing { &d.cr } else { &d.status };
        let n_codes = if competing { 2 } else { 1 };
        let n = d.time.len();
        let nf = n as f64;
        let data = SurvData::builder(&d.time, status)
            .n_codes(n_codes)
            .build()
            .unwrap();
        let (times, risk): (Vec<f64>, &Vec<Vec<f64>>) = match key.as_str() {
            "melanoma" => (vec![1000.0, 1800.0, 2500.0, 3000.0], &d.risk_s),
            "melanoma_cr" => (vec![1000.0, 1800.0, 2500.0, 3000.0], &d.risk_cr),
            _ => (vec![0.5, 1.0, 1.5, 2.0], &d.risk),
        };
        for f in frames {
            let j = times.iter().position(|&t| t == f.time).unwrap();
            let column: Vec<f64> = (0..n).map(|i| risk[i][j]).collect();
            let got = calibration::estimate(
                &data,
                &column,
                f.time,
                &CalibrationOptions::new(Groups::Quantiles(f.q)),
            )
            .unwrap();
            let tag = format!("{key} t={} q={}", f.time, f.q);
            assert_eq!(got.groups.len(), f.pred.len(), "{tag}: groups");
            assert_eq!(got.breaks.len(), f.groups.len(), "{tag}: breaks");
            for (a, b) in got.breaks.iter().zip(&f.groups) {
                assert!((a - b).abs() <= 8.0 * eps, "{tag}: break {a} against {b}");
            }
            let bound = 2.0 * (nf + 2.0) * eps;
            for (g, (&pred, obs)) in got.groups.iter().zip(f.pred.iter().zip(&f.obs)) {
                let dev = (g.predicted - pred).abs() / bound;
                assert!(
                    dev <= 1.0,
                    "{tag}: predicted {} against {pred}",
                    g.predicted
                );
                worst.put("predicted", dev);
                match (g.observed, obs) {
                    (Some(a), Some(b)) => {
                        let dev = (a - b).abs() / bound;
                        assert!(dev <= 1.0, "{tag}: observed {a} against {b}");
                        worst.put("observed", dev);
                    }
                    (None, None) => worst.put("missing where prodlim has none", 0.0),
                    (a, b) => panic!("{tag}: observed {a:?} against {b:?}"),
                }
            }
        }
    }
    // Few distinct predictions: the groups are the values, with the observed
    // risk of the rows that have them. plotCalibration's predicted risks are not
    // the values, since it orders them apart from the rows.
    let few = &r.calibration_few;
    let d = &r.datasets["sim"];
    let data = SurvData::builder(&d.time, &d.status).build().unwrap();
    let got = calibration::estimate(
        &data,
        &few.risk,
        few.time,
        &CalibrationOptions::new(Groups::Quantiles(5)),
    )
    .unwrap();
    assert_eq!(got.breaks.len(), 0);
    let n = d.time.len() as f64;
    for (g, (&value, &obs)) in got
        .groups
        .iter()
        .zip(few.values.iter().zip(&few.obs_aligned))
    {
        assert!((g.predicted - value).abs() <= 2.0 * (n + 2.0) * eps);
        assert!((g.observed.unwrap() - obs).abs() <= 2.0 * (n + 2.0) * eps);
    }
    let r_off = few
        .r_pred
        .iter()
        .zip(&few.values)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(
        r_off > 0.05,
        "plotCalibration's predicted risks should not be the values: {r_off}"
    );
    eprintln!("plotCalibration's predicted risks differ from the group values by {r_off:.3}");
    worst.report();
}
