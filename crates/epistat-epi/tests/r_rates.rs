//! Rates, rate ratios and standardised rates checked against R 4.6.2.
//!
//! The reference is `tests/data/rates.json`, written by `tests/data/rates.R`:
//! `poisson.test` for the exact interval of a crude rate, the exact
//! conditional test of two rates and its estimate, the Byar and Wald formulas
//! and the Fay-Feuer gamma interval written out in R with `qchisq`, and
//! `popEpi::rate` for the standardised point estimate.
//!
//! Tolerance. The reference is written with 17 significant digits, so it is
//! exact. The rate, the variance and the Wald limits are a few dozen
//! additions and products, so they agree to `1e-13`. The gamma limits
//! add the accuracy of `gamma_quantile` (about `1e-14`) divided by the
//! sensitivity of the quantile to its probability, which is below 10 for
//! these shapes, so the bound is `1e-12`. The exact p-values are sums of
//! point probabilities that are accurate to `1e-14`, over at most a few
//! thousand terms, with a tail that is a regularised beta function, so the
//! bound is `1e-11`. The checks print the largest deviation in units of its
//! bound.

use epistat_core::Level;
use epistat_epi::rates::{
    crude_rate, direct_standardised, rate_difference, rate_ratio, rate_ratio_exact_test,
};
use epistat_stats::dist::poisson::{Method, cipoisson};
use epistat_testkit::{Tolerance, load_json};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    crude: Vec<Crude>,
    standardised: Vec<Standardised>,
    pairs: Vec<Pair>,
    exact: Vec<Exact>,
}

#[derive(Deserialize)]
struct Crude {
    events: f64,
    time: f64,
    level: f64,
    scale: f64,
    rate: f64,
    lower: f64,
    upper: f64,
    byar_lower: Option<f64>,
    byar_upper: f64,
}

#[derive(Deserialize)]
struct Standardised {
    name: String,
    events: Vec<f64>,
    time: Vec<f64>,
    weights: Vec<f64>,
    level: f64,
    rate: f64,
    variance: f64,
    lower: f64,
    upper: f64,
    popepi_rate: f64,
}

#[derive(Deserialize)]
struct Pair {
    e1: f64,
    t1: f64,
    e0: f64,
    t0: f64,
    exact_p: f64,
    estimate: f64,
    #[serde(rename = "rr_0.95")]
    rr95: Option<Vec<f64>>,
    #[serde(rename = "rr_0.99")]
    rr99: Option<Vec<f64>>,
    #[serde(rename = "rd_0.95")]
    rd95: Vec<f64>,
    #[serde(rename = "rd_0.99")]
    rd99: Vec<f64>,
}

#[derive(Deserialize)]
struct Exact {
    e1: f64,
    t1: f64,
    e0: f64,
    t0: f64,
    p: f64,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/rates.json"
    ))
}

/// Records the deviation of `got` from `expected` in units of `bound` and
/// keeps the largest.
fn ratio(worst: &mut f64, got: f64, expected: f64, bound: f64) {
    let ratio = (got - expected).abs() / (bound * expected.abs().max(1e-300));
    *worst = worst.max(ratio);
}

#[test]
fn crude_rates_and_byar() {
    let mut worst = 0.0;
    for c in reference().crude {
        let level = Level::confidence(c.level).unwrap();
        let r = crude_rate(c.events, c.time, level, c.scale)
            .unwrap()
            .unwrap();
        ratio(&mut worst, r.estimate, c.rate, 1e-13);
        ratio(&mut worst, r.upper().unwrap(), c.upper, 1e-12);
        if c.lower > 0.0 {
            ratio(&mut worst, r.lower().unwrap(), c.lower, 1e-12);
        } else {
            assert_eq!(r.lower(), Some(0.0));
        }
        let b = cipoisson(c.events, c.time, level, Method::Byar).unwrap();
        ratio(&mut worst, b.upper.unwrap(), c.byar_upper, 1e-13);
        match c.byar_lower {
            Some(x) if x > 0.0 => ratio(&mut worst, b.lower.unwrap(), x, 1e-13),
            Some(_) => assert_eq!(b.lower, Some(0.0)),
            None => assert_eq!(b.lower, None),
        }
    }
    eprintln!("crude: worst deviation {worst:.3} of its bound");
    assert!(worst <= 1.0, "{worst}");
}

#[test]
fn standardised_rates() {
    let mut worst = 0.0;
    for s in reference().standardised {
        let level = Level::confidence(s.level).unwrap();
        let got = direct_standardised(&s.events, &s.time, &s.weights, level)
            .unwrap()
            .unwrap();
        let name = &s.name;
        let rate = got.rate;
        if s.rate > 0.0 {
            ratio(&mut worst, rate.estimate, s.rate, 1e-13);
            ratio(&mut worst, rate.estimate, s.popepi_rate, 1e-13);
            ratio(&mut worst, got.variance, s.variance, 1e-13);
            ratio(&mut worst, rate.lower().unwrap(), s.lower, 1e-12);
        } else {
            assert_eq!((rate.estimate, rate.lower()), (0.0, Some(0.0)), "{name}");
        }
        ratio(&mut worst, rate.upper().unwrap(), s.upper, 1e-12);
    }
    eprintln!("standardised: worst deviation {worst:.3} of its bound");
    assert!(worst <= 1.0, "{worst}");
}

#[test]
fn ratios_and_differences() {
    let mut worst = 0.0;
    for p in reference().pairs {
        for (level, rr, rd) in [(0.95, &p.rr95, &p.rd95), (0.99, &p.rr99, &p.rd99)] {
            let level = Level::confidence(level).unwrap();
            let d = rate_difference(p.e1, p.t1, p.e0, p.t0, level)
                .unwrap()
                .unwrap();
            for (got, want) in [Some(d.estimate), d.lower(), d.upper(), d.p_value]
                .into_iter()
                .zip(rd)
            {
                ratio(&mut worst, got.unwrap(), *want, 1e-12);
            }
            let r = rate_ratio(p.e1, p.t1, p.e0, p.t0, level).unwrap();
            match rr {
                Some(rr) => {
                    let r = r.unwrap();
                    for (got, want) in [Some(r.estimate), r.lower(), r.upper(), r.p_value]
                        .into_iter()
                        .zip(rr)
                    {
                        ratio(&mut worst, got.unwrap(), *want, 1e-12);
                    }
                }
                None => assert_eq!(r.and_then(|r| r.lower()), None),
            }
        }
        if p.e0 > 0.0 {
            let r = rate_ratio(p.e1, p.t1, p.e0, p.t0, Level::default())
                .unwrap()
                .unwrap();
            assert!(
                Tolerance::exact()
                    .rel(1e-13)
                    .accepts(r.estimate, p.estimate)
            );
        }
        let got = rate_ratio_exact_test(p.e1, p.t1, p.e0, p.t0)
            .unwrap()
            .unwrap();
        ratio(&mut worst, got, p.exact_p, 1e-11);
    }
    eprintln!("ratios: worst deviation {worst:.3} of its bound");
    assert!(worst <= 1.0, "{worst}");
}

#[test]
fn exact_test_grid() {
    let mut worst = 0.0;
    for e in reference().exact {
        let got = rate_ratio_exact_test(e.e1, e.t1, e.e0, e.t0)
            .unwrap()
            .unwrap();
        ratio(&mut worst, got, e.p, 1e-11);
    }
    eprintln!("exact grid: worst deviation {worst:.3} of its bound");
    assert!(worst <= 1.0, "{worst}");
}

/// The values go through `ln`, `exp` and `powf`, whose last bits depend on
/// the maths library and, on Windows, on the processor the library dispatches
/// for, so the bits are checked only on aarch64 macOS, where they were recorded.
/// Elsewhere the test computes the hash and, when `EPISTAT_HASH_OUT` is set,
/// appends it to that file.
#[test]
fn fingerprint_of_a_fixed_table() {
    let s = &reference().standardised[0];
    let level = Level::default();
    let g = direct_standardised(&s.events, &s.time, &s.weights, level)
        .unwrap()
        .unwrap();
    let c = crude_rate(7.0, 12.5, level, 1000.0).unwrap().unwrap();
    let r = rate_ratio(5.0, 100.0, 9.0, 120.0, level).unwrap().unwrap();
    let p = rate_ratio_exact_test(5.0, 100.0, 9.0, 120.0).unwrap();
    let hash = epistat_testkit::Fingerprint::new()
        .option_f64(Some(g.rate.estimate))
        .option_f64(Some(g.variance))
        .option_f64(g.rate.lower())
        .option_f64(g.rate.upper())
        .option_f64(c.lower())
        .option_f64(c.upper())
        .option_f64(r.lower())
        .option_f64(r.upper())
        .option_f64(p)
        .finish();
    eprintln!("rates fingerprint {hash:#018x}");
    epistat_testkit::hash_out("fingerprint_of_a_fixed_table", hash);
    if epistat_testkit::RECORDED_PLATFORM {
        assert_eq!(hash, 0xe37e_3a2a_92f8_6a17);
    }
}
