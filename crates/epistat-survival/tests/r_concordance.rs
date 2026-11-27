//! `concordance::estimate` checked against R's `survival::concordance`.
//!
//! The tolerances follow from the arithmetic of the sweep. Each count is a sum
//! over deaths of `w v` times a weight read from a Fenwick tree. A tree node
//! holds a partial sum of at most `W`, the weight of the data, and takes at
//! most `n` additions and `n` removals, so it carries an error of at most
//! `n eps W / 2 * 2`. A prefix reads `ceil(log2 n) + 1` nodes, so a read is
//! off by at most `K W` with `K = n (ceil(log2 n) + 1) eps`. A death adds `w v`
//! times that, so a count is off by at most `K W D`, with `D` the sum of `w v`
//! over the deaths. R's tree has the same bound, so two implementations differ
//! by at most twice that. `D` and `v` are computed in the test from the
//! definitions, not taken from the code under test. When no weight is given and
//! `v` is 1 every sum is of integers below `2^53` and the counts must be equal.
//!
//! A row's partner sums add `v w` terms read from the tree of risk sets, bound
//! by `K W v_max`, and from the tree of deaths, bound by `K D`, so a partner sum
//! is off by at most `2 K (W v_max + D)` including R's. The concordance, with
//! `s = (C - D) / P`, moves by at most `(|dC| + |dD| + |dT|) / P`, and the
//! influence of row `k` by at most
//! `w_k (|1 - s| + |1 + s| + |s|) tol_partner / (2 P)`. The variance is a sum
//! of squares of sums of influences, so it moves by at most
//! `sum_g (2 |S_g| e_g + e_g^2)` with `e_g` the sum of the influence tolerances
//! in the cluster. The variance of the Cox score statistic is a sum over
//! event times of `w v z2 / T`, with `z2` a running sum that is updated by
//! entering and leaving rows. Each update rounds by `eps` times the largest
//! value `z2` took, at most `W^3`, so its error is at most `2 n eps W^3` per
//! event time, divided by the risk-set weight `T`, and doubled for R.

use epistat_survival::concordance::{ConcordanceOptions, TimeWeight, estimate};
use epistat_survival::{SurvData, timefix::TimeFix};
use epistat_testkit::load_json;
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct Versions {
    r: String,
    survival: String,
    jsonlite: String,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct Dataset {
    stop: Vec<f64>,
    status: Vec<i32>,
    status12: Vec<i32>,
    lp: Vec<f64>,
    ecog: Vec<f64>,
    age: Vec<f64>,
    x: Vec<f64>,
    weight: Vec<f64>,
    stratum: Vec<u32>,
    stratum12: Vec<u32>,
    cluster: Vec<u32>,
    start: Vec<f64>,
}

#[derive(Deserialize)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "one flag per option of the R call"
)]
struct Case {
    name: String,
    data: String,
    x: String,
    timewt: String,
    reverse: bool,
    weights: bool,
    strata: Option<String>,
    cluster: bool,
    delayed: bool,
    ymin: Option<f64>,
    ymax: Option<f64>,
    timefix: bool,
    status: String,
    count: Vec<f64>,
    concordance: f64,
    var: f64,
    cvar: f64,
    dfbeta: Vec<f64>,
    n: usize,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    datasets: BTreeMap<String, Dataset>,
    cases: Vec<Case>,
}

fn kind_of(name: &str) -> TimeWeight {
    match name {
        "n" => TimeWeight::N,
        "S" => TimeWeight::S,
        "S/G" => TimeWeight::SOverG,
        "n/G2" => TimeWeight::NOverG2,
        "I" => TimeWeight::I,
        other => panic!("time weight {other}"),
    }
}

/// What the tolerances need, computed from the definitions of the weights.
struct Scale {
    /// Sum over deaths of `w v`.
    dv: f64,
    vmax: f64,
    /// Sum over event times of `d v / T`.
    cvar: f64,
}

#[expect(
    clippy::float_cmp,
    reason = "event and censoring times tie when they are equal"
)]
fn scale(
    stop: &[f64],
    start: Option<&[f64]>,
    status: &[i32],
    w: &[f64],
    stratum: &[u32],
    kind: TimeWeight,
    ymax: Option<f64>,
) -> Scale {
    let mut out = Scale {
        dv: 0.0,
        vmax: 0.0,
        cvar: 0.0,
    };
    let mut levels: Vec<u32> = stratum.to_vec();
    levels.sort_unstable();
    levels.dedup();
    for level in levels {
        let rows: Vec<usize> = (0..stop.len()).filter(|&i| stratum[i] == level).collect();
        let total: f64 = rows.iter().map(|&i| w[i]).sum();
        let n_deaths = rows.iter().filter(|&&i| status[i] == 1).count();
        let kind = if n_deaths < 2 { TimeWeight::N } else { kind };
        let mut times: Vec<f64> = rows.iter().map(|&i| stop[i]).collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times.dedup();
        let (mut surv, mut cens) = (1.0_f64, 1.0_f64);
        for &t in &times {
            let at_risk: f64 = rows
                .iter()
                .filter(|&&i| stop[i] >= t && start.is_none_or(|s| s[i] < t))
                .map(|&i| w[i])
                .sum();
            let d: f64 = rows
                .iter()
                .filter(|&&i| stop[i] == t && status[i] == 1)
                .map(|&i| w[i])
                .sum();
            let c: f64 = rows
                .iter()
                .filter(|&&i| stop[i] == t && status[i] == 0)
                .map(|&i| w[i])
                .sum();
            if d > 0.0 {
                let mut v = match kind {
                    TimeWeight::N => 1.0,
                    TimeWeight::S => total * surv / at_risk,
                    TimeWeight::SOverG => total * surv / (cens * at_risk),
                    TimeWeight::NOverG2 => 1.0 / (cens * cens),
                    TimeWeight::I => 1.0 / at_risk,
                    _ => unreachable!("a kind this test does not cover"),
                };
                if ymax.is_some_and(|m| t > m) {
                    v = 0.0;
                }
                out.dv += d * v;
                out.vmax = out.vmax.max(v);
                out.cvar += d * v / at_risk;
                surv *= (at_risk - d) / at_risk;
            }
            if c > 0.0 {
                let later: f64 = rows.iter().filter(|&&i| stop[i] > t).map(|&i| w[i]).sum();
                cens *= later / (later + c);
            }
        }
    }
    out
}

#[derive(Default)]
struct Worst(BTreeMap<&'static str, f64>);

impl Worst {
    #[track_caller]
    #[expect(clippy::float_cmp, reason = "a zero tolerance asks for equality")]
    fn check(&mut self, what: &'static str, case: &str, got: f64, want: f64, tol: f64) {
        let ratio = if tol == 0.0 {
            f64::from(u8::from(got != want))
        } else {
            (got - want).abs() / tol
        };
        let entry = self.0.entry(what).or_insert(0.0);
        *entry = entry.max(ratio);
        assert!(
            ratio <= 1.0,
            "{case}: {what} {got:e} against {want:e}, {ratio:.3} times the tolerance {tol:e}"
        );
    }
}

#[test]
#[expect(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "one pass over the cases that checks every output"
)]
fn every_case_matches_survival_concordance() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/concordance.json"
    ));
    assert!(reference.versions.r.starts_with("R version 4.6"));
    assert!(reference.versions.survival.starts_with("3.8"));
    assert_ne!(reference.versions.jsonlite, "");
    let mut worst = Worst::default();
    for case in &reference.cases {
        let d = &reference.datasets[&case.data];
        let x = match case.x.as_str() {
            "lp" => &d.lp,
            "ecog" => &d.ecog,
            "age" => &d.age,
            "x" => &d.x,
            other => panic!("score {other}"),
        };
        let status = if case.status == "status12" {
            &d.status12
        } else {
            &d.status
        };
        let stratum_col = case.strata.as_deref().map(|s| match s {
            "stratum" => &d.stratum,
            "stratum12" => &d.stratum12,
            other => panic!("stratum {other}"),
        });
        let mut builder = SurvData::builder(&d.stop, status);
        if case.delayed {
            builder = builder.start(&d.start);
        }
        if case.weights {
            builder = builder.weight(&d.weight);
        }
        if let Some(s) = stratum_col {
            builder = builder.stratum(s);
        }
        if case.cluster {
            builder = builder.cluster(&d.cluster);
        }
        let data = builder.build().unwrap();
        let kind = kind_of(&case.timewt);
        let opts = ConcordanceOptions {
            timefix: if case.timefix {
                TimeFix::default()
            } else {
                TimeFix::Exact
            },
            time_weight: kind,
            reverse: case.reverse,
            ymin: case.ymin,
            ymax: case.ymax,
            variance: true,
            influence: true,
        };
        let got = estimate(&data, x, &opts).unwrap();
        let name = case.name.as_str();
        assert_eq!(got.n_rows, case.n, "{name}");

        let n = d.stop.len();
        let ones = vec![1.0; n];
        let zeros = vec![0_u32; n];
        let w = if case.weights { &d.weight } else { &ones };
        // The stops after ymin, as the estimator sees them.
        let stops: Vec<f64> = d
            .stop
            .iter()
            .map(|&t| case.ymin.map_or(t, |m| t.max(m)))
            .collect();
        let sc = scale(
            &stops,
            case.delayed.then_some(&d.start[..]),
            status,
            w,
            stratum_col.map_or(&zeros[..], |s| &s[..]),
            kind,
            case.ymax,
        );
        let total_w: f64 = w.iter().sum();
        let k = n as f64 * (f64::from((n as f64).log2().ceil() as u32) + 1.0) * f64::EPSILON;
        let exact = !case.weights && kind == TimeWeight::N;
        let count_tol = if exact {
            0.0
        } else {
            2.0 * k * total_w * sc.dv
        };

        let c = &got.counts;
        // R swaps the counts itself when the orientation is reversed.
        let (rc, rd) = (case.count[0], case.count[1]);
        worst.check("concordant", name, c.concordant, rc, count_tol);
        worst.check("discordant", name, c.discordant, rd, count_tol);
        worst.check("tied x", name, c.tied_x, case.count[2], count_tol);
        let pairs = rc + rd + case.count[2];
        let tied_scale = 4.0 * n as f64 * f64::EPSILON * (case.count[3] + case.count[4]);
        let tied_tol = if exact { 0.0 } else { tied_scale };
        worst.check("tied y", name, c.tied_y, case.count[3], tied_tol);
        worst.check("tied xy", name, c.tied_xy, case.count[4], tied_tol);

        let conc_tol = if exact {
            // Exact counts differ from R's only by the division and the final
            // half: three roundings.
            4.0 * f64::EPSILON
        } else {
            3.0 * 2.0 * count_tol / pairs + 4.0 * f64::EPSILON
        };
        worst.check(
            "concordance",
            name,
            got.concordance.unwrap(),
            case.concordance,
            conc_tol,
        );

        // Influence and variances.
        let somer = (rc - rd) / pairs;
        let partner_tol = 2.0 * k * (total_w * sc.vmax + sc.dv);
        let coef = ((1.0 - somer).abs() + (1.0 + somer).abs() + somer.abs()) / (2.0 * pairs);
        // The influence function is `n` times the `dfbeta` that R returns.
        let units = got.influence.as_ref().unwrap();
        let n_units = units.n_units() as f64;
        let influence: Vec<f64> = units.column(0).iter().map(|v| v / n_units).collect();
        let tols: Vec<f64> = (0..n)
            .map(|i| w[i] * coef * partner_tol + 4.0 * f64::EPSILON * influence[i].abs())
            .collect();
        // With a cluster R returns the influences added up by cluster, in
        // the order of the cluster codes.
        let groups: Vec<u32> = if case.cluster {
            d.cluster.clone()
        } else {
            (0..n as u32).collect()
        };
        let mut mine: BTreeMap<u32, (f64, f64)> = BTreeMap::new();
        for i in 0..n {
            let e = mine.entry(groups[i]).or_default();
            e.0 += influence[i];
            e.1 += tols[i];
        }
        assert_eq!(mine.len(), case.dfbeta.len(), "{name}");
        for ((s, e), want) in mine.values().zip(&case.dfbeta) {
            worst.check("dfbeta", name, *s, *want, *e);
        }
        let sums = mine;
        let var_tol: f64 = sums
            .values()
            .map(|&(s, e)| 2.0 * s.abs() * e + e * e)
            .sum::<f64>()
            + 8.0 * f64::EPSILON * case.var;
        worst.check("variance", name, got.variance.unwrap(), case.var, var_tol);
        let cvar_tol = 2.0 * 2.0 * n as f64 * f64::EPSILON * total_w.powi(3) * sc.cvar
            / (4.0 * pairs * pairs)
            + 8.0 * n as f64 * f64::EPSILON * case.cvar;
        worst.check(
            "cox variance",
            name,
            got.cox_variance.unwrap(),
            case.cvar,
            cvar_tol,
        );
    }
    for (what, ratio) in &worst.0 {
        eprintln!("worst {what}: {ratio:.4} of the tolerance");
    }
}
