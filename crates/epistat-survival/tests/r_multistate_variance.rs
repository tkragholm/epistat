//! `multistate::variance` checked against R 4.6.2's `survival::survfit(Surv(start,
//! stop, state) ~ 1, id =, istate =)`: `std.err`, `std.chaz`, `std.auc`, `se0`,
//! the influence of each cluster, and the `se(rmean)` of `summary(fit, rmean =)`,
//! with estimated and given initial distributions, weights, clusters and delayed
//! entry. The bound each deviation is held to is derived in `influence_bound`.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the times in the data are exact integers and the sizes are far below the limits of the casts"
)]

mod influence_bound;

use epistat_survival::SurvData;
use epistat_survival::multistate::{
    MultistateOptions, OccupationVarianceOptions, estimate, variance,
};
use epistat_testkit::load_json;
use influence_bound::{Row, influence_bound, kappa};
use serde::Deserialize;

const STATES: usize = 3;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Versions {
    #[serde(rename = "R")]
    r: String,
    survival: String,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    weights: String,
    estimated_p0: bool,
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    from: Vec<u32>,
    id: Vec<u32>,
    cluster: Vec<u32>,
    weight: Vec<f64>,
    time: Vec<f64>,
    p0: Vec<f64>,
    t0: f64,
    se0: Vec<f64>,
    cname: Vec<String>,
    std_err: Vec<Vec<f64>>,
    cumhaz: Vec<Vec<f64>>,
    std_chaz: Vec<Vec<f64>>,
    std_auc: Vec<Vec<f64>>,
    tau: Vec<f64>,
    rmean: Vec<RMean>,
    uclust: Option<Vec<u32>>,
    influence: Option<Vec<Vec<Vec<f64>>>>,
}

#[derive(Deserialize)]
struct RMean {
    se: Vec<f64>,
}

#[derive(Default)]
struct Worst(f64, String);

impl Worst {
    #[track_caller]
    fn see(&mut self, what: &str, deviation: f64, tolerance: f64) {
        let ratio = if tolerance > 0.0 {
            deviation / tolerance
        } else if deviation == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
        assert!(
            ratio <= 1.0,
            "{what}: deviation {deviation:e} against tolerance {tolerance:e}"
        );
        if ratio >= self.0 {
            self.0 = ratio;
            self.1 = what.to_string();
        }
    }
}

#[test]
fn matches_r_survfit() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/multistate_variance.json"
    ));
    eprintln!(
        "{}, survival {}",
        reference.versions.r, reference.versions.survival
    );
    let eps = f64::EPSILON;
    let mut worst = [
        "std.err",
        "std.chaz",
        "std.auc",
        "se(rmean)",
        "influence",
        "se0",
        "cumhaz",
    ]
    .map(|name| (name, Worst::default()));
    for case in &reference.cases {
        let n = case.stop.len();
        let istate: Vec<u32> = case.from.clone();
        let mut builder = SurvData::builder(&case.stop, &case.status)
            .start(&case.start)
            .n_codes(STATES as u32)
            .id(&case.id)
            .istate(&istate);
        if !case.cluster.is_empty() {
            builder = builder.cluster(&case.cluster);
        }
        if case.weights != "none" {
            builder = builder.weight(&case.weight);
        }
        let data = builder.build().unwrap();
        let opts = MultistateOptions {
            initial: (!case.estimated_p0).then(|| case.p0.clone()),
            ..MultistateOptions::default()
        };
        let all_taus = case.tau.clone();
        let influence_times = case.time.clone();
        let mut v = OccupationVarianceOptions::new(case.t0);
        v.horizons = all_taus.clone();
        v.influence_times = influence_times.clone();
        let got = variance(&data, &case.time, &opts, &v).unwrap();
        assert_eq!(got.n_rows, n);

        // The same recursion as the estimate, up to rounding.
        let est = estimate(&data, &case.time, &opts).unwrap();
        for (a, b) in got.probabilities.iter().zip(&est.probabilities) {
            assert!((a - b).abs() <= 2.0 * (4 * case.time.len() + 2 * n) as f64 * eps);
        }

        // Clusters and rows, as the data have them.
        let labels = &got.clusters;
        let label_of = |i: usize| {
            if case.cluster.is_empty() {
                case.id[i]
            } else {
                case.cluster[i]
            }
        };
        let rows: Vec<Row> = (0..n)
            .map(|i| Row {
                start: case.start[i],
                stop: case.stop[i],
                from: case.from[i] as usize - 1,
                to: (case.status[i] > 0).then(|| case.status[i] as usize - 1),
                weight: case.weight[i],
                cluster: labels.binary_search(&label_of(i)).unwrap(),
            })
            .collect();
        // The opening rows add 2 w / W per cluster when p0 is estimated.
        let mut opening = vec![0.0f64; labels.len()];
        if case.estimated_p0 {
            let mut seen = std::collections::HashSet::new();
            let mut first: Vec<usize> = (0..n).collect();
            first.sort_by(|&a, &b| case.stop[a].total_cmp(&case.stop[b]));
            let total: f64 = first
                .iter()
                .filter(|&&i| seen.insert(case.id[i]))
                .map(|&i| case.weight[i])
                .sum();
            seen.clear();
            for &i in &first {
                if seen.insert(case.id[i]) {
                    opening[rows[i].cluster] += 2.0 * case.weight[i] / total;
                }
            }
        }
        let a = influence_bound(&rows, &case.time, STATES, labels.len(), &opening);
        let norm = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let kap = kappa(case.time.len(), STATES, n);
        let tol_se = 2.0 * kap * eps * norm;
        let tol_auc = |tau: f64| {
            (tau - case.t0).max(0.0) * 2.0 * (kap + case.time.len() as f64 + 2.0) * eps * norm
        };

        let h = got.transitions.len();
        assert_eq!(h, case.cname.len(), "{}", case.name);
        let w = &mut worst;
        for t in 0..case.time.len() {
            let at = |s: &str| format!("{} t={} {s}", case.name, case.time[t]);
            for s in 0..STATES {
                w[0].1.see(
                    &at(&format!("std.err {s}")),
                    (got.se[t * STATES + s] - case.std_err[s][t]).abs(),
                    tol_se,
                );
                w[2].1.see(
                    &at(&format!("std.auc {s}")),
                    (got.std_time[t * STATES + s] - case.std_auc[s][t]).abs(),
                    tol_auc(case.time[t]),
                );
            }
            for (jk, &(from, to)) in got.transitions.iter().enumerate() {
                let name = format!("{}:{}", from + 1, to + 1);
                let col = case.cname.iter().position(|x| *x == name).unwrap();
                w[1].1.see(
                    &at(&format!("std.chaz {name}")),
                    (got.std_cumhaz[t * h + jk] - case.std_chaz[col][t]).abs(),
                    tol_se,
                );
                let value = case.cumhaz[col][t];
                w[6].1.see(
                    &at(&format!("cumhaz {name}")),
                    (got.cumhaz[t * h + jk] - value).abs(),
                    2.0 * (2 * n + case.time.len() + 4) as f64 * eps * value.abs().max(1.0),
                );
            }
        }
        for (ti, &tau) in case.tau.iter().enumerate() {
            for s in 0..STATES {
                w[3].1.see(
                    &format!("{} tau={tau} state {s}", case.name),
                    (got.std_rmean[ti * STATES + s] - case.rmean[ti].se[s]).abs(),
                    tol_auc(tau),
                );
            }
        }
        if case.estimated_p0 && !case.se0.is_empty() {
            for s in 0..STATES {
                w[5].1.see(
                    &format!("{} se0 state {s}", case.name),
                    (got.std_initial[s] - case.se0[s]).abs(),
                    2.0 * kap * eps * opening.iter().map(|x| x * x).sum::<f64>().sqrt(),
                );
            }
        } else {
            assert!(got.std_initial.iter().all(|&x| x == 0.0), "{}", case.name);
        }
        if let (Some(uclust), Some(influence)) = (&case.uclust, &case.influence) {
            for (s, per_state) in influence.iter().enumerate() {
                for (cr, values) in per_state.iter().enumerate() {
                    let g = labels.binary_search(&uclust[cr]).unwrap();
                    for (t, &value) in values.iter().enumerate() {
                        // R's influence is on the dfbeta scale, `1 / n` of ours.
                        let column = got.influence.as_ref().unwrap().column(t * STATES + s);
                        let got_value = column[g] / column.len() as f64;
                        w[4].1.see(
                            &format!(
                                "{} influence cluster {} t={} s{s}",
                                case.name, uclust[cr], case.time[t]
                            ),
                            (got_value - value).abs(),
                            2.0 * kap * eps * a[g],
                        );
                    }
                }
            }
        }
    }
    for (name, w) in &worst {
        eprintln!("{name}: worst {:.3e} of its tolerance ({})", w.0, w.1);
    }
}
