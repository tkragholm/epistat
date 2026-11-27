//! The expected survival of a cohort, Ederer I and Hakulinen relative
//! survival, age-standardised net survival, the crude probabilities of death
//! and the test of equal net survival, against R
//! (`tests/data/relsurv_extras.R`): `survival::survexp` 3.8-12 and
//! `relsurv::rs.surv`, `cmp.rel` and `rs.diff` 2.3-3. The worst ratio of
//! deviation to bound is printed with `--nocapture`.
//!
//! Tolerance. The cumulative reference hazards are sums of a few dozen pieces
//! in both implementations, a relative error of about `1e-15`, and every
//! quantity here is a sum over at most a few thousand time points of terms of
//! one sign, or a ratio of such sums, so its rounding error is below `1e-12`
//! relative; R's step sums over days (`cmp.rel` and `rs.diff` on a grid of
//! whole days, about 3000 steps) add about `3000 eps`, so those are checked to
//! `1e-11`. The covariance of the test cancels in its off-diagonal entries,
//! which are checked to `1e-11` of the diagonal. The Hakulinen estimate of
//! `rs.surv` differs from the exact one by its rule, whose bound the script
//! writes out.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    reason = "statuses, sexes and groups are small whole numbers written by R as doubles, and times are compared with the values they were copied from"
)]

use epistat_survival::Level;
use epistat_survival::SurvData;
use epistat_survival::ratetable::{LongDim, RateDim, RateTable};
use epistat_survival::relsurv::{
    ExpectedHazard, ExpectedMethod, ICSS, Method, NetOptions, Transform, crude_mortality,
    expected_survival, net_survival, net_survival_test, standardise_net,
};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    age_cuts: Vec<f64>,
    year_cuts: Vec<f64>,
    rates: Vec<f64>,
    cohort_a: Cohort,
    cohort_b: Cohort,
    survexp: Survexp,
    ederer1: Vec<RsCurve>,
    hakulinen: Vec<RsCurve>,
    hakulinen_exact: Vec<HakExact>,
    standardised: Standardised,
    crude: Vec<Crude>,
    crude_relsurv: Vec<Crude>,
    tests: Tests,
}

#[derive(Deserialize)]
struct Cohort {
    age0: Vec<f64>,
    date0: Vec<f64>,
    sex: Vec<f64>,
    grp: Vec<f64>,
    agegr: Vec<f64>,
    time: Vec<f64>,
    status: Vec<f64>,
    start: Vec<f64>,
    potential: Option<Vec<f64>>,
}

#[derive(Deserialize)]
struct Survexp {
    asked: Vec<f64>,
    ederer: Vec<Vec<f64>>,
    hakulinen: Vec<AtAsked>,
    conditional: Vec<AtAsked>,
}

#[derive(Deserialize)]
struct AtAsked {
    surv: Vec<f64>,
    n: Vec<f64>,
}

#[derive(Deserialize)]
struct RsCurve {
    time: Vec<f64>,
    surv: Vec<f64>,
    std_err: Vec<f64>,
}

#[derive(Deserialize)]
struct HakExact {
    time: Vec<f64>,
    surv: Vec<f64>,
    bound: f64,
}

#[derive(Deserialize)]
struct Standardised {
    times: Vec<f64>,
    weights: Vec<f64>,
    values: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct Crude {
    time: Vec<f64>,
    disease: Vec<f64>,
    disease_var: Vec<f64>,
    other: Vec<f64>,
    other_var: Vec<f64>,
}

#[derive(Deserialize)]
struct Tests {
    plain: Test,
    strata: Test,
    entry: Test,
    entry_relsurv: Test,
    plain_written: Test,
}

#[derive(Deserialize)]
struct Test {
    statistic: f64,
    df: usize,
    z: Vec<f64>,
    covariance: Vec<f64>,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/relsurv_extras.json"
    ))
}

/// The table of the script, whose last cells reach without limit as in R.
fn table(r: &Reference) -> RateTable {
    let close = |cuts: &[f64]| {
        let mut c = cuts.to_vec();
        c.push(cuts[cuts.len() - 1] + 365.0);
        c
    };
    RateTable::new(
        vec![
            RateDim::Time {
                cuts: close(&r.age_cuts),
            },
            RateDim::Factor { levels: 2 },
            RateDim::Time {
                cuts: close(&r.year_cuts),
            },
        ],
        r.rates.clone(),
        1.0,
    )
    .unwrap()
    .extend_ends()
}

struct Columns {
    status: Vec<i32>,
    sex: Vec<u32>,
    grp: Vec<u32>,
    agegr: Vec<u32>,
}

fn columns(c: &Cohort) -> Columns {
    Columns {
        status: c.status.iter().map(|&s| s as i32).collect(),
        sex: c.sex.iter().map(|&s| s as u32 - 1).collect(),
        grp: c.grp.iter().map(|&g| g as u32 - 1).collect(),
        agegr: c.agegr.iter().map(|&g| g as u32 - 1).collect(),
    }
}

#[derive(Default)]
struct Worst(f64, String);

impl Worst {
    fn rel(&mut self, what: &str, got: f64, want: f64, bound: f64) {
        let r = (got - want).abs() / (bound * want.abs().max(1e-300));
        if r > self.0 {
            *self = Worst(r, what.to_string());
        }
    }

    fn abs(&mut self, what: &str, got: f64, want: f64, bound: f64) {
        let r = (got - want).abs() / bound;
        if r > self.0 {
            *self = Worst(r, what.to_string());
        }
    }

    fn report(&self, name: &str) {
        eprintln!("{name}: worst {:.3} of its bound, in {}", self.0, self.1);
        assert!(self.0 <= 1.0, "{name}: {} in {}", self.0, self.1);
    }
}

#[test]
fn a_long_table_is_the_table_of_the_script() {
    let r = reference();
    let (na, ny) = (r.age_cuts.len(), r.year_cuts.len());
    let mut age = Vec::new();
    let mut sex = Vec::new();
    let mut year = Vec::new();
    for y in 0..ny {
        for s in 0..2 {
            for a in 0..na {
                age.push(r.age_cuts[a]);
                sex.push(s);
                year.push(r.year_cuts[y]);
            }
        }
    }
    let long = RateTable::from_long(
        &[
            LongDim::Time {
                start: &age,
                end: r.age_cuts[na - 1] + 365.0,
            },
            LongDim::Factor { level: &sex },
            LongDim::Time {
                start: &year,
                end: r.year_cuts[ny - 1] + 365.0,
            },
        ],
        &r.rates,
        1.0,
    )
    .unwrap()
    .extend_ends();
    assert_eq!(long, table(&r));
}

#[test]
fn expected_survival_against_survexp() {
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
    let potential = c.potential.clone().unwrap();
    let asked = &r.survexp.asked;
    let mut worst = Worst::default();
    let ederer = expected_survival(&data, &expected, asked, ExpectedMethod::EdererI).unwrap();
    for (got, want) in ederer.iter().zip(&r.survexp.ederer) {
        for k in 0..asked.len() {
            worst.rel("Ederer I", got.survival[k].unwrap(), want[k], 1e-12);
        }
    }
    for (method, refs, name) in [
        (
            ExpectedMethod::Hakulinen {
                potential: &potential,
            },
            &r.survexp.hakulinen,
            "Hakulinen",
        ),
        (
            ExpectedMethod::Conditional,
            &r.survexp.conditional,
            "conditional",
        ),
    ] {
        let curves = expected_survival(&data, &expected, asked, method).unwrap();
        for (got, want) in curves.iter().zip(refs) {
            for k in 0..asked.len() {
                assert_eq!(got.n[k] as f64, want.n[k], "{name} n at {}", asked[k]);
                match got.survival[k] {
                    Some(s) => worst.rel(name, s, want.surv[k], 1e-12),
                    None => assert_eq!(want.n[k], 0.0, "{name} at {}", asked[k]),
                }
            }
        }
    }
    worst.report("expected survival");
}

#[test]
fn relative_survival_of_ederer_one_and_hakulinen() {
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
    let potential = c.potential.clone().unwrap();
    let mut worst = Worst::default();
    let e1 = net_survival(
        &data,
        &expected,
        &NetOptions {
            method: Method::EdererI,
            ..NetOptions::default()
        },
    )
    .unwrap();
    for (got, want) in e1.iter().zip(&r.ederer1) {
        assert_eq!(got.times, want.time);
        for k in 0..got.times.len() {
            worst.rel("Ederer I", got.survival[k], want.surv[k], 1e-12);
            worst.rel(
                "Ederer I std.err",
                got.variance[k].sqrt(),
                want.std_err[k],
                1e-12,
            );
        }
    }
    let hak = net_survival(
        &data,
        &expected,
        &NetOptions {
            method: Method::Hakulinen,
            potential: Some(&potential),
            ..NetOptions::default()
        },
    )
    .unwrap();
    let mut rule = Worst::default();
    for ((got, exact), rs) in hak.iter().zip(&r.hakulinen_exact).zip(&r.hakulinen) {
        assert_eq!(got.times, exact.time);
        for k in 0..got.times.len() {
            worst.rel("Hakulinen", got.survival[k], exact.surv[k], 1e-12);
            // relsurv reports at its own steps too; its value at an exit time
            // differs from the exact one by at most the bound in Lambda, so
            // by that share of the survival.
            let j = rs.time.iter().position(|&x| x == got.times[k]).unwrap();
            rule.rel(
                "rs.surv Hakulinen",
                got.survival[k],
                rs.surv[j],
                exact.bound,
            );
        }
    }
    worst.report("relative survival");
    rule.report("relative survival against relsurv's Hakulinen rule");
}

#[test]
fn age_standardised_net_survival() {
    let r = reference();
    let t = table(&r);
    let c = &r.cohort_a;
    let cols = columns(c);
    let data = SurvData::builder(&c.time, &cols.status)
        .stratum(&cols.agegr)
        .build()
        .unwrap();
    let expected =
        ExpectedHazard::from_table(&t, &data, &[&c.age0, &c.date0], &[&cols.sex]).unwrap();
    let curves = net_survival(
        &data,
        &expected,
        &NetOptions {
            method: Method::EdererII,
            ..NetOptions::default()
        },
    )
    .unwrap();
    assert_eq!(ICSS[0].to_vec(), r.standardised.weights);
    let s = standardise_net(
        &curves,
        &ICSS[0],
        &r.standardised.times,
        Level::default(),
        Transform::LogLog,
    )
    .unwrap();
    let mut worst = Worst::default();
    for (k, want) in r.standardised.values.iter().enumerate() {
        // Ederer II agrees with rs.surv at precision 0.5 to about 1e-11
        // (r_relsurv.rs), and the weighted sum keeps that.
        worst.rel("survival", s.survival[k].unwrap(), want[0], 1e-11);
        worst.rel("se", s.se[k].unwrap(), want[1], 1e-10);
        worst.rel("lower", s.lower[k].unwrap(), want[2], 1e-10);
        worst.rel("upper", s.upper[k].unwrap(), want[3], 1e-10);
    }
    worst.report("standardised net survival");
    // Beyond the follow-up of a group the curve is undefined.
    let late = standardise_net(
        &curves,
        &ICSS[0],
        &[1e7],
        Level::default(),
        Transform::LogLog,
    )
    .unwrap();
    assert_eq!(late.survival, vec![None]);
}

#[test]
fn crude_probabilities_against_cmp_rel() {
    let r = reference();
    let t = table(&r);
    let c = &r.cohort_b;
    let cols = columns(c);
    let zero = vec![0.0; c.time.len()];
    let mut worst = Worst::default();
    // With the entry times, against the definition written out; without them,
    // against cmp.rel, which leaves them out.
    for (entry, refs) in [(&c.start, &r.crude), (&zero, &r.crude_relsurv)] {
        let data = SurvData::builder(&c.time, &cols.status)
            .start(entry)
            .stratum(&cols.grp)
            .build()
            .unwrap();
        let expected =
            ExpectedHazard::from_table(&t, &data, &[&c.age0, &c.date0], &[&cols.sex]).unwrap();
        let crude = crude_mortality(&data, &expected, Level::default()).unwrap();
        for (got, want) in crude.iter().zip(refs) {
            assert_eq!(got.times, want.time);
            for k in 0..got.times.len() {
                // The probabilities start at 0 and grow, so their error is
                // relative to the total probability of death, at most 1.
                let scale = got.disease[k] + got.other[k];
                worst.abs(
                    "disease",
                    got.disease[k],
                    want.disease[k],
                    1e-11 * scale.max(1e-300),
                );
                worst.abs(
                    "other",
                    got.other[k],
                    want.other[k],
                    1e-11 * scale.max(1e-300),
                );
                worst.rel(
                    "disease variance",
                    got.disease_variance[k],
                    want.disease_var[k],
                    1e-10,
                );
                if want.other_var[k] > 0.0 {
                    worst.rel(
                        "other variance",
                        got.other_variance[k],
                        want.other_var[k],
                        1e-10,
                    );
                }
                let sum = got.disease[k] + got.other[k] + got.survival[k];
                worst.abs("1 - S", sum, 1.0, 1e-13);
            }
        }
    }
    worst.report("crude probabilities");
}

#[test]
fn test_of_equal_net_survival_against_rs_diff() {
    let r = reference();
    let t = table(&r);
    let mut worst = Worst::default();
    let zero_b = vec![0.0; r.cohort_b.time.len()];
    // rs.diff leaves out the entry times, so with them the reference is the
    // test written out in R, which equals rs.diff without delayed entry.
    for (name, c, want, stratified, entry) in [
        (
            "plain",
            &r.cohort_a,
            &r.tests.plain,
            false,
            &r.cohort_a.start,
        ),
        (
            "plain written",
            &r.cohort_a,
            &r.tests.plain_written,
            false,
            &r.cohort_a.start,
        ),
        (
            "strata",
            &r.cohort_a,
            &r.tests.strata,
            true,
            &r.cohort_a.start,
        ),
        (
            "entry",
            &r.cohort_b,
            &r.tests.entry,
            false,
            &r.cohort_b.start,
        ),
        (
            "entry left out",
            &r.cohort_b,
            &r.tests.entry_relsurv,
            false,
            &zero_b,
        ),
    ] {
        let cols = columns(c);
        let mut builder = SurvData::builder(&c.time, &cols.status).start(entry);
        if stratified {
            builder = builder.stratum(&cols.sex);
        }
        let data = builder.build().unwrap();
        let expected =
            ExpectedHazard::from_table(&t, &data, &[&c.age0, &c.date0], &[&cols.sex]).unwrap();
        let got = net_survival_test(&data, &cols.grp, &expected, Some(1.0)).unwrap();
        assert_eq!(got.df, want.df, "{name}");
        worst.rel(
            &format!("{name} statistic"),
            got.statistic,
            want.statistic,
            1e-10,
        );
        let r = got.df;
        for a in 0..r {
            worst.rel(&format!("{name} z"), got.z[a], want.z[a], 1e-10);
            for b in 0..r {
                let diag =
                    (got.covariance.get(a, a).unwrap() * got.covariance.get(b, b).unwrap()).sqrt();
                worst.abs(
                    &format!("{name} covariance"),
                    got.covariance.get(a, b).unwrap(),
                    want.covariance[a * r + b],
                    1e-11 * diag,
                );
            }
        }
        let total: f64 = got.z.iter().sum();
        worst.abs(
            &format!("{name} z sums to 0"),
            total,
            0.0,
            1e-10 * got.z[0].abs(),
        );
    }
    worst.report("net survival test");
}
