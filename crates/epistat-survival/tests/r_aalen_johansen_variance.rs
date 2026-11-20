//! `aalen_johansen::variance` checked against R 4.6.2's `survival::survfit` with
//! a factor status: `std.err`, `std.chaz` and `std.auc`, the influence of each
//! cluster, and the `se(rmean)` of `summary(fit, rmean =)`, for plain data,
//! groups, case weights, clusters, ids with several rows per subject and
//! delayed entry. The bound each deviation is held to is derived in
//! `influence_bound`.

#![expect(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the times in the data are exact integers and the sizes are far below the limits of the casts"
)]

mod influence_bound;

use epistat_survival::SurvData;
use epistat_survival::aalen_johansen::{AalenJohansenOptions, VarianceOptions, variance};
use epistat_testkit::load_json;
use influence_bound::{Row, influence_bound, kappa};
use serde::Deserialize;

const STATES: usize = 4;

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
    time: Vec<f64>,
    status: Vec<i32>,
    entry: Vec<f64>,
    group: Vec<u32>,
    id: Vec<u32>,
    cluster: Vec<u32>,
    weight: Vec<f64>,
    use_entry: bool,
    use_group: bool,
    uses: String,
    cname: Vec<String>,
    t0: f64,
    groups: Vec<RGroup>,
    tau: Vec<f64>,
    rmean: Vec<Vec<RMean>>,
}

#[derive(Deserialize)]
struct RGroup {
    group: u32,
    time: Vec<f64>,
    std_err: Vec<Vec<f64>>,
    cumhaz: Vec<Vec<f64>>,
    std_chaz: Vec<Vec<f64>>,
    std_auc: Vec<Vec<f64>>,
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
    let reference: Reference = load_json("tests/data/aalen_johansen_variance.json");
    println!(
        "{} / survival {}",
        reference.versions.r, reference.versions.survival
    );
    let eps = f64::EPSILON;
    let mut worst = [
        "std.err",
        "std.chaz",
        "std.auc",
        "se(rmean)",
        "influence",
        "cumhaz",
    ]
    .map(|name| (name, Worst::default()));
    for case in &reference.cases {
        let n = case.time.len();
        let mut builder = SurvData::builder(&case.time, &case.status).n_codes(3);
        if case.use_entry {
            builder = builder.start(&case.entry);
        }
        if case.use_group {
            builder = builder.stratum(&case.group);
        }
        match case.uses.as_str() {
            "cluster" => builder = builder.cluster(&case.cluster),
            "id" => builder = builder.id(&case.id),
            _ => {}
        }
        if case.weight.iter().any(|&w| w != 1.0) {
            builder = builder.weight(&case.weight);
        }
        let data = builder.build().unwrap();

        let mut all_times: Vec<f64> = case
            .groups
            .iter()
            .flat_map(|g| g.time.iter().copied())
            .collect();
        all_times.sort_by(f64::total_cmp);
        all_times.dedup();
        let opts = VarianceOptions {
            horizons: case.tau.clone(),
            influence_times: all_times.clone(),
        };
        let got = variance(&data, &AalenJohansenOptions::default(), &opts).unwrap();
        assert_eq!(got.origin, case.t0, "{}", case.name);
        assert_eq!(got.n_rows, n);

        for (gi, rg) in case.groups.iter().enumerate() {
            let rows_of_group: Vec<usize> = (0..n)
                .filter(|&i| !case.use_group || case.group[i] == rg.group)
                .collect();
            // clusters of the group: as the data do
            let labels = &got.groups[gi].clusters;
            assert_eq!(got.groups[gi].group, rg.group);
            let label_of = |i: usize| match case.uses.as_str() {
                "cluster" => case.cluster[i],
                "id" => case.id[i],
                _ => i as u32,
            };
            let rows: Vec<Row> = rows_of_group
                .iter()
                .map(|&i| Row {
                    start: if case.use_entry {
                        case.entry[i]
                    } else {
                        f64::NEG_INFINITY
                    },
                    stop: case.time[i],
                    from: 0,
                    to: (case.status[i] > 0).then(|| case.status[i] as usize),
                    weight: case.weight[i],
                    cluster: labels.binary_search(&label_of(i)).unwrap(),
                })
                .collect();
            let mut times: Vec<f64> = rows.iter().map(|r| r.stop).collect();
            times.sort_by(f64::total_cmp);
            times.dedup();
            let a = influence_bound(&rows, &times, STATES, labels.len(), &[]);
            let norm = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            let kap = kappa(times.len(), STATES, rows.len());
            let tol_se = 2.0 * kap * eps * norm;
            let width = |tau: f64| (tau - case.t0).max(0.0);
            let tol_auc =
                |tau: f64| width(tau) * 2.0 * (kap + times.len() as f64 + 2.0) * eps * norm;

            // The rows of the result in this group.
            let ours: Vec<usize> = (0..got.time.len())
                .filter(|&r| got.group[r] == rg.group)
                .collect();
            for (rt, &t) in rg.time.iter().enumerate() {
                let r = *ours
                    .iter()
                    .find(|&&r| got.time[r] == t)
                    .unwrap_or_else(|| panic!("{}: time {t} missing", case.name));
                let what = |s: &str| format!("{} group {} t={t} {s}", case.name, rg.group);
                let w = &mut worst;
                // state 0 is the event-free state
                w[0].1.see(
                    &what("std.err s0"),
                    (got.std_event_free[r] - rg.std_err[0][rt]).abs(),
                    tol_se,
                );
                w[2].1.see(
                    &what("std.auc s0"),
                    (got.std_time_event_free[r] - rg.std_auc[0][rt]).abs(),
                    tol_auc(t),
                );
                for c in 0..3 {
                    w[0].1.see(
                        &what(&format!("std.err {c}")),
                        (got.std_cif[c][r] - rg.std_err[c + 1][rt]).abs(),
                        tol_se,
                    );
                    w[2].1.see(
                        &what(&format!("std.auc {c}")),
                        (got.std_time[c][r] - rg.std_auc[c + 1][rt]).abs(),
                        tol_auc(t),
                    );
                    let col = case
                        .cname
                        .iter()
                        .position(|x| *x == format!("1:{}", c + 2))
                        .unwrap();
                    w[1].1.see(
                        &what(&format!("std.chaz {c}")),
                        (got.std_cumhaz[c][r] - rg.std_chaz[col][rt]).abs(),
                        tol_se,
                    );
                    let value = rg.cumhaz[col][rt];
                    w[5].1.see(
                        &what(&format!("cumhaz {c}")),
                        (got.cumhaz[c][r] - value).abs(),
                        2.0 * (2 * rows.len() + times.len() + 4) as f64
                            * eps
                            * value.abs().max(1.0),
                    );
                }
            }

            // se(rmean) at each tau: R uses the influence for it.
            for (ti, &tau) in case.tau.iter().enumerate() {
                for s in 0..STATES {
                    let ours = got.groups[gi].std_rmean[ti * STATES + s];
                    worst[3].1.see(
                        &format!("{} group {} tau={tau} state {s}", case.name, rg.group),
                        (ours - case.rmean[ti][gi].se[s]).abs(),
                        tol_auc(tau),
                    );
                }
            }

            // The influence itself, where R gave it.
            if let (Some(uclust), Some(influence)) = (&rg.uclust, &rg.influence) {
                for (s, per_state) in influence.iter().enumerate() {
                    for (cr, values) in per_state.iter().enumerate() {
                        let g = labels.binary_search(&uclust[cr]).unwrap();
                        for (rt, &t) in rg.time.iter().enumerate() {
                            let k = all_times.iter().position(|&x| x == t).unwrap();
                            let at = (g * all_times.len() + k) * STATES + s;
                            worst[4].1.see(
                                &format!(
                                    "{} group {} influence cluster {} t={t} s{s}",
                                    case.name, rg.group, uclust[cr]
                                ),
                                (got.groups[gi].influence[at] - values[rt]).abs(),
                                2.0 * kap * eps * a[g],
                            );
                        }
                    }
                }
            }
        }
    }
    for (name, w) in &worst {
        println!(
            "{name}: worst deviation {:.3e} of its tolerance ({})",
            w.0, w.1
        );
    }
}
