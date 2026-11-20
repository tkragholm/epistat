//! `aalen_johansen::estimate` checked against R 4.6.2's `survival::survfit` with
//! a factor status, and against `etm::etm`, `cmprsk::cuminc` and
//! `prodlim::prodlim` where they cover the case. The bootstraps have no R counterpart and are not checked
//! here.

use epistat_survival::SurvData;
use epistat_survival::aalen_johansen::{AalenJohansenOptions, estimate};
use epistat_testkit::load_json;
use serde::Deserialize;

const CAUSES: f64 = 3.0;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    time: Vec<f64>,
    status: Vec<i32>,
    entry: Vec<f64>,
    group: Vec<u32>,
    weight_exact: Vec<f64>,
    weight_decimal: Vec<f64>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Versions {
    #[serde(rename = "R")]
    r: String,
    survival: String,
    etm: String,
    cmprsk: String,
    prodlim: String,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    use_entry: bool,
    weights: String,
    use_group: bool,
    groups: Vec<RGroup>,
}

#[derive(Deserialize)]
struct RGroup {
    group: u32,
    time: Vec<f64>,
    n_risk: Vec<f64>,
    n_at_risk: Vec<f64>,
    n_event: Vec<f64>,
    event_free: Vec<f64>,
    cif: Vec<Vec<f64>>,
    etm_time: Option<Vec<f64>>,
    etm_event_free: Option<Vec<f64>>,
    etm_cif: Option<Vec<Vec<f64>>>,
    cmprsk_time: Option<Vec<Vec<f64>>>,
    cmprsk_cif: Option<Vec<Vec<f64>>>,
    prodlim_cif: Option<Vec<Vec<f64>>>,
}

/// The largest deviation seen, in units of its tolerance.
#[derive(Default)]
struct Worst {
    value: f64,
    case: String,
}

impl Worst {
    #[track_caller]
    fn see(&mut self, case: &str, deviation: f64, tolerance: f64) {
        let ratio = if tolerance > 0.0 {
            deviation / tolerance
        } else if deviation == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
        assert!(
            ratio <= 1.0,
            "{case} (line {}): deviation {deviation:e} against tolerance {tolerance:e}",
            std::panic::Location::caller().line()
        );
        if ratio >= self.value {
            self.value = ratio;
            self.case = case.to_string();
        }
    }
}

/// Per step of one group the tolerances on the event-free probability and on
/// the cumulative incidence of each cause, in absolute terms.
///
/// Both estimates are products and sums of the same steps, each step formed in
/// double precision by one of two implementations, and the tolerance is a bound
/// on how far two such implementations can differ.
///
/// - The weights. A weighted count is a sum of at most `n` positive weights,
///   or, with delayed entry, a running balance of entries and exits. In either
///   case each addition rounds by at most `eps / 2` times a running value no
///   larger than the total weight `W` of the group, so one implementation is
///   off by at most `n eps W / 2` and two differ by at most `n eps W`. Weights
///   that are multiples of 1/8 sum exactly, and then the bound is zero and the
///   counts must agree to the last bit.
/// - A step. With `q = d / Y`, the factor `f = 1 - sum_k d_k / Y` differs by
///   `(dW + q dW) / Y` from the weights, `eps q` from the two divisions, and
///   `(K + 1) eps` from the `K` divisions' sum and the subtraction, `dW` being
///   the bound on a count.
/// - The event-free probability. A product of `m` factors, so its relative
///   difference is at most the sum over the steps of the factor's difference
///   divided by the factor, plus `m eps` for the products. The factor in the
///   denominator is what makes a step that nearly empties the risk set costly.
/// - The incidence. A sum of `m` positive terms `S(t-) d_k / Y`. The term
///   differs by the weights' share, `S (dW + q_k dW) / Y`, and by its own
///   relative error, the event-free probability's plus `3 eps` for the product
///   and the quotient. The sum adds `m eps` times its value. The scale of the
///   incidence is 1, so the bound is absolute.
fn tolerances(r: &RGroup, weight_bound: f64) -> (Vec<f64>, Vec<Vec<f64>>) {
    let eps = f64::EPSILON;
    let m = r.time.len();
    let mut s_tol = Vec::with_capacity(m);
    let mut f_tol: Vec<Vec<f64>> = (0..3).map(|_| Vec::with_capacity(m)).collect();
    let mut relative_s = 0.0;
    let mut absolute_f = [0.0; 3];
    let mut s_before = 1.0;
    for step in 0..m {
        let y = r.n_risk[step];
        let d = r.n_event[step];
        let q = d / y;
        let f = r.event_free[step] / s_before;
        let dq = weight_bound * (1.0 + q) / y + eps * q;
        let df = dq + (CAUSES + 1.0) * eps;
        let s_rel_before = relative_s;
        relative_s += df / f + eps;
        s_tol.push(relative_s * r.event_free[step]);
        for k in 0..3 {
            let previous = if step == 0 { 0.0 } else { r.cif[k][step - 1] };
            let term = r.cif[k][step] - previous;
            let q_k = term / s_before;
            absolute_f[k] += s_before * weight_bound * (1.0 + q_k.min(1.0)) / y
                + term * (s_rel_before + 3.0 * eps);
            f_tol[k].push(absolute_f[k] + (step as f64 + 1.0) * eps * r.cif[k][step]);
        }
        s_before = r.event_free[step];
    }
    (s_tol, f_tol)
}

/// The value of a right-continuous step function at `at`, with the index of
/// its step, or `initial` and no index before the first step.
fn step_value(times: &[f64], values: &[f64], at: f64, initial: f64) -> (Option<usize>, f64) {
    match times.partition_point(|&t| t <= at) {
        0 => (None, initial),
        index => (Some(index - 1), values[index - 1]),
    }
}

/// The tolerance at a step, zero before the first, where both estimates are
/// the exact starting values.
fn at(tolerance: &[f64], index: Option<usize>) -> f64 {
    index.map_or(0.0, |i| tolerance[i])
}

#[test]
fn curves_match_r() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/aalen_johansen.json"
    ));
    eprintln!(
        "{}, survival {}, etm {}, cmprsk {}, prodlim {}",
        reference.versions.r,
        reference.versions.survival,
        reference.versions.etm,
        reference.versions.cmprsk,
        reference.versions.prodlim
    );
    let eps = f64::EPSILON;
    let n = reference.time.len();
    let mut worst: Vec<(&str, Worst)> = [
        "time",
        "n_risk",
        "n_at_risk",
        "n_event",
        "event_free",
        "cif",
        "etm event_free",
        "etm cif",
        "cmprsk cif",
        "prodlim cif",
    ]
    .into_iter()
    .map(|name| (name, Worst::default()))
    .collect();

    for case in &reference.cases {
        let weight: Vec<f64> = match case.weights.as_str() {
            "none" => vec![1.0; n],
            "exact" => reference.weight_exact.clone(),
            "decimal" => reference.weight_decimal.clone(),
            other => panic!("unknown weights {other}"),
        };
        let group: Vec<u32> = if case.use_group {
            reference.group.clone()
        } else {
            vec![0; n]
        };
        let mut builder = SurvData::builder(&reference.time, &reference.status).n_codes(3);
        if case.weights != "none" {
            builder = builder.weight(&weight);
        }
        if case.use_group {
            builder = builder.stratum(&group);
        }
        if case.use_entry {
            builder = builder.start(&reference.entry);
        }
        let data = builder.build().unwrap();
        let got = estimate(&data, &AalenJohansenOptions::default()).unwrap();
        assert_eq!(got.n_rows, n, "{}", case.name);
        assert_eq!(got.n_used, n, "{}", case.name);
        assert_eq!(got.causes, vec![1, 2, 3], "{}", case.name);

        for r in &case.groups {
            let name = format!("{} group {}", case.name, r.group);
            let rows: Vec<usize> = (0..got.group.len())
                .filter(|&i| got.group[i] == r.group)
                .collect();
            let pick = |column: &[f64]| -> Vec<f64> { rows.iter().map(|&i| column[i]).collect() };
            let times = pick(&got.time);
            assert_eq!(times, r.time, "{name}: exit times");
            let (rows_in_group, total): (f64, f64) = (0..n)
                .filter(|&i| group[i] == r.group)
                .fold((0.0, 0.0), |(c, w), i| (c + 1.0, w + weight[i]));
            let weight_bound = if case.weights == "decimal" {
                rows_in_group * eps * total
            } else {
                0.0
            };

            let risk = pick(&got.n_risk);
            let events = pick(&got.n_event);
            for (ours, theirs, index) in [(&risk, &r.n_risk, 1), (&events, &r.n_event, 3)] {
                let mut dev = 0.0f64;
                for (a, b) in ours.iter().zip(theirs) {
                    dev = dev.max((a - b).abs());
                }
                worst[index].1.see(&name, dev, weight_bound);
            }
            let people = pick(&got.n_at_risk);
            assert_eq!(people, r.n_at_risk, "{name}: individuals at risk");
            worst[2].1.see(&name, 0.0, 0.0);

            let (s_tol, f_tol) = tolerances(r, weight_bound);
            let event_free = pick(&got.event_free);
            let cif: Vec<Vec<f64>> = got.cif.iter().map(|c| pick(c)).collect();
            for (i, (a, b)) in event_free.iter().zip(&r.event_free).enumerate() {
                worst[4].1.see(&name, (a - b).abs(), s_tol[i]);
            }
            for k in 0..3 {
                for i in 0..times.len() {
                    worst[5]
                        .1
                        .see(&name, (cif[k][i] - r.cif[k][i]).abs(), f_tol[k][i]);
                }
            }

            // The other packages take no weights. Their steps are integer
            // counts, so only the arithmetic of a step enters the bound.
            if let (Some(t), Some(s), Some(c)) = (&r.etm_time, &r.etm_event_free, &r.etm_cif) {
                for (j, &time) in t.iter().enumerate() {
                    let (i, ours) = step_value(&times, &event_free, time, 1.0);
                    worst[6].1.see(&name, (ours - s[j]).abs(), at(&s_tol, i));
                    for k in 0..3 {
                        let (i, ours) = step_value(&times, &cif[k], time, 0.0);
                        worst[7]
                            .1
                            .see(&name, (ours - c[k][j]).abs(), at(&f_tol[k], i));
                    }
                }
            }
            if let (Some(t), Some(c)) = (&r.cmprsk_time, &r.cmprsk_cif) {
                for k in 0..3 {
                    for (j, &time) in t[k].iter().enumerate() {
                        let (i, ours) = step_value(&times, &cif[k], time, 0.0);
                        worst[8]
                            .1
                            .see(&name, (ours - c[k][j]).abs(), at(&f_tol[k], i));
                    }
                }
            }
            if let Some(c) = &r.prodlim_cif {
                for k in 0..3 {
                    for i in 0..times.len() {
                        worst[9]
                            .1
                            .see(&name, (cif[k][i] - c[k][i]).abs(), f_tol[k][i]);
                    }
                }
            }
        }
    }
    for (name, w) in &worst {
        eprintln!(
            "{name:16} worst {:.3e} of the tolerance ({})",
            w.value, w.case
        );
    }
}
