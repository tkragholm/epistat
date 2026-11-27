//! Mantel-Haenszel estimators and tests checked against R 4.6.2:
//! `metafor::rma.mh` 5.2.1 for the five estimators, the Mantel-Haenszel,
//! Breslow-Day and Tarone statistics, and `stats::mantelhaen.test` for the
//! odds ratio, its interval and the corrected and uncorrected statistic.
//!
//! The reference is `tests/data/mh.json`, written by `tests/data/mh.R` with 17
//! significant digits. `rma.mh` is run with `add = 0`, so no zero-cell
//! constant enters. The continuity correction is compared with
//! `mantelhaen.test`, which corrects by 1/2 only when `|delta| >= 1/2`, and not with
//! `rma.mh`, which subtracts 1/2 without the minimum.
//!
//! Tolerance. Each estimate and variance is a ratio of sums of at most
//! `k = 500` terms of one sign or of a few signs, so the error is bounded by a
//! few `k eps kappa` with `kappa` below 10 for these data, which is about
//! `1e-12`. The Breslow-Day statistic adds a square root and a division per
//! stratum on the root `N11` and sums squares, with `kappa` up to 100 when the
//! pooled odds ratio is near 1, so it and Tarone's are held to `1e-9`. The
//! p-values of a chi-square statistic in the far tail are held to `1e-8`
//! relative, the sensitivity of the tail to its argument. The checks print the
//! largest deviation in units of its bound.

use epistat_core::Level;
use epistat_epi::mantel_haenszel::{
    breslow_day, cmh_test, odds_ratio, rate_difference, rate_ratio, rate_ratio_test,
    risk_difference, risk_ratio,
};
use epistat_epi::{Counts2x2, PersonTime2};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    tables: Vec<TableCase>,
    rates: Vec<RateCase>,
}

#[derive(Deserialize)]
struct Pooled {
    estimate: Option<f64>,
    se: Option<f64>,
    lower: Option<f64>,
    upper: Option<f64>,
    p: Option<f64>,
}

#[derive(Deserialize)]
struct Outcome {
    stat: Option<f64>,
    p: Option<f64>,
    k_pos: Option<usize>,
}

#[derive(Deserialize)]
struct Hh {
    stat: Option<f64>,
    p: Option<f64>,
    or: Option<f64>,
    lower: Option<f64>,
    upper: Option<f64>,
    stat_corrected: Option<f64>,
    p_corrected: Option<f64>,
}

#[derive(Deserialize)]
struct TableCase {
    name: String,
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    d: Vec<f64>,
    #[serde(rename = "OR")]
    or: Pooled,
    #[serde(rename = "RR")]
    rr: Pooled,
    #[serde(rename = "RD")]
    rd: Pooled,
    test_uncorrected: Outcome,
    bd: Outcome,
    ta: Outcome,
    mantelhaen: Option<Hh>,
}

#[derive(Deserialize)]
struct RateCase {
    name: String,
    x1: Vec<f64>,
    t1: Vec<f64>,
    x0: Vec<f64>,
    t0: Vec<f64>,
    #[serde(rename = "IRR")]
    irr: Pooled,
    #[serde(rename = "IRD")]
    ird: Pooled,
    test_uncorrected: Outcome,
    test_corrected: Outcome,
}

fn reference() -> Reference {
    load_json(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/mh.json"))
}

struct Worst(f64);

impl Worst {
    fn check(&mut self, what: &str, got: Option<f64>, want: Option<f64>, bound: f64) {
        match (got, want) {
            (None, None) => {}
            (Some(g), Some(w)) => {
                let r = (g - w).abs() / (bound * w.abs().max(1e-300));
                self.0 = self.0.max(r);
                assert!(
                    r <= 1.0 || (g == 0.0 && w == 0.0),
                    "{what}: {g:e} against {w:e}"
                );
            }
            _ => panic!("{what}: {got:?} against {want:?}"),
        }
    }

    fn pooled(&mut self, what: &str, got: &epistat_epi::mantel_haenszel::Pooled, want: &Pooled) {
        let got = got.estimate;
        self.check(
            &format!("{what} estimate"),
            got.map(|e| e.estimate),
            want.estimate,
            1e-12,
        );
        let se = got.and_then(|e| e.se);
        self.check(&format!("{what} se"), se, want.se, 1e-12);
        let lower = got.and_then(|e| e.lower());
        self.check(&format!("{what} lower"), lower, want.lower, 1e-12);
        let upper = got.and_then(|e| e.upper());
        self.check(&format!("{what} upper"), upper, want.upper, 1e-12);
        let p = got.and_then(|e| e.p_value);
        self.check(&format!("{what} p"), p, want.p, 1e-8);
    }
}

fn tables_of(c: &TableCase) -> Vec<Counts2x2> {
    (0..c.a.len())
        .map(|i| Counts2x2 {
            exposed_cases: c.a[i],
            exposed_noncases: c.b[i],
            unexposed_cases: c.c[i],
            unexposed_noncases: c.d[i],
        })
        .collect()
}

#[test]
fn table_estimators_and_tests() {
    let mut worst = Worst(0.0);
    for case in reference().tables {
        let tables = tables_of(&case);
        let name = &case.name;
        worst.pooled(
            &format!("{name} OR"),
            &odds_ratio(&tables, Level::default()).unwrap(),
            &case.or,
        );
        worst.pooled(
            &format!("{name} RR"),
            &risk_ratio(&tables, Level::default()).unwrap(),
            &case.rr,
        );
        worst.pooled(
            &format!("{name} RD"),
            &risk_difference(&tables, Level::default()).unwrap(),
            &case.rd,
        );

        let cmh = cmh_test(&tables, false).unwrap();
        worst.check(
            &format!("{name} CMH"),
            cmh.map(|t| t.statistic),
            case.test_uncorrected.stat,
            1e-12,
        );
        worst.check(
            &format!("{name} CMH p"),
            cmh.map(|t| t.p_value),
            case.test_uncorrected.p,
            1e-8,
        );
        if let Some(hh) = &case.mantelhaen {
            let or = odds_ratio(&tables, Level::default()).unwrap();
            if let Some(or) = or.estimate {
                worst.check(&format!("{name} mh or"), Some(or.estimate), hh.or, 1e-12);
                worst.check(&format!("{name} mh lower"), or.lower(), hh.lower, 1e-12);
                worst.check(&format!("{name} mh upper"), or.upper(), hh.upper, 1e-12);
            } else {
                // R reports an odds ratio of 0 without an interval where the
                // pooled ratio is undefined here.
                assert_eq!((hh.or, hh.lower), (Some(0.0), None), "{name}");
            }
            worst.check(
                &format!("{name} mh stat"),
                cmh.map(|t| t.statistic),
                hh.stat,
                1e-12,
            );
            worst.check(&format!("{name} mh p"), cmh.map(|t| t.p_value), hh.p, 1e-8);
            let corrected = cmh_test(&tables, true).unwrap();
            worst.check(
                &format!("{name} cc stat"),
                corrected.map(|t| t.statistic),
                hh.stat_corrected,
                1e-12,
            );
            worst.check(
                &format!("{name} cc p"),
                corrected.map(|t| t.p_value),
                hh.p_corrected,
                1e-8,
            );
        }

        let bd = breslow_day(&tables, false).unwrap();
        if case.bd.k_pos.is_some_and(|k| k < 2) {
            // R reports rounding noise on no degrees of freedom.
            assert_eq!(bd, None, "{name}");
            continue;
        }
        worst.check(
            &format!("{name} BD"),
            bd.map(|t| t.statistic),
            case.bd.stat,
            1e-9,
        );
        worst.check(
            &format!("{name} BD p"),
            bd.map(|t| t.p_value),
            case.bd.p,
            1e-7,
        );
        if let (Some(t), Some(k)) = (bd, case.bd.k_pos) {
            assert_eq!(t.df + 1, k, "{name} df");
        }
        let ta = breslow_day(&tables, true).unwrap();
        worst.check(
            &format!("{name} TA"),
            ta.map(|t| t.statistic),
            case.ta.stat,
            1e-9,
        );
        worst.check(
            &format!("{name} TA p"),
            ta.map(|t| t.p_value),
            case.ta.p,
            1e-7,
        );
    }
    eprintln!("tables: worst deviation {:.3} of its bound", worst.0);
}

#[test]
fn rate_estimators_and_tests() {
    let mut worst = Worst(0.0);
    for case in reference().rates {
        let strata: Vec<PersonTime2> = (0..case.x1.len())
            .map(|i| PersonTime2 {
                exposed_events: case.x1[i],
                exposed_time: case.t1[i],
                unexposed_events: case.x0[i],
                unexposed_time: case.t0[i],
            })
            .collect();
        let name = &case.name;
        worst.pooled(
            &format!("{name} IRR"),
            &rate_ratio(&strata, Level::default()).unwrap(),
            &case.irr,
        );
        worst.pooled(
            &format!("{name} IRD"),
            &rate_difference(&strata, Level::default()).unwrap(),
            &case.ird,
        );
        let plain = rate_ratio_test(&strata, false).unwrap();
        worst.check(
            &format!("{name} MH"),
            plain.map(|t| t.statistic),
            case.test_uncorrected.stat,
            1e-12,
        );
        worst.check(
            &format!("{name} MH p"),
            plain.map(|t| t.p_value),
            case.test_uncorrected.p,
            1e-8,
        );
        // rma.mh subtracts 1/2 without the minimum, so compare where the
        // numerator exceeds 1/2, which every case here does.
        let corrected = rate_ratio_test(&strata, true).unwrap();
        worst.check(
            &format!("{name} MH cc"),
            corrected.map(|t| t.statistic),
            case.test_corrected.stat,
            1e-12,
        );
    }
    eprintln!("rates: worst deviation {:.3} of its bound", worst.0);
}

/// The values go through `ln`, `exp` and `powf`, whose last bits depend on
/// the maths library and, on Windows, on the processor the library dispatches
/// for, so the bits are checked only on aarch64 macOS, where they were recorded.
/// Elsewhere the test computes the hash and, when `EPISTAT_HASH_OUT` is set,
/// appends it to that file.
#[test]
fn fingerprint_of_the_bcg_data() {
    let case = reference().tables.remove(0);
    let tables = tables_of(&case);
    let or = odds_ratio(&tables, Level::default()).unwrap();
    let rr = risk_ratio(&tables, Level::default()).unwrap();
    let rd = risk_difference(&tables, Level::default()).unwrap();
    let bd = breslow_day(&tables, true).unwrap().unwrap();
    let cmh = cmh_test(&tables, true).unwrap().unwrap();
    let (or, rr, rd) = (
        or.estimate.unwrap(),
        rr.estimate.unwrap(),
        rd.estimate.unwrap(),
    );
    let hash = epistat_testkit::Fingerprint::new()
        .option_f64(Some(or.estimate))
        .option_f64(or.se)
        .option_f64(Some(rr.estimate))
        .option_f64(rr.se)
        .option_f64(Some(rd.estimate))
        .option_f64(rd.se)
        .f64(bd.statistic)
        .f64(cmh.statistic)
        .finish();
    eprintln!("mh fingerprint {hash:#018x}");
    epistat_testkit::hash_out("fingerprint_of_the_bcg_data", hash);
    if epistat_testkit::RECORDED_PLATFORM {
        assert_eq!(hash, 0x31d5_6360_1ff5_15a2);
    }
}
