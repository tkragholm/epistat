//! The ratio and difference of two directly standardised rates and the tests
//! of a common rate ratio over strata, checked against R 4.6.2.
//!
//! The reference is `tests/data/rates_compare.json`, written by
//! `tests/data/rates_compare.R`: the F interval of Fay (1999) and the
//! modified F interval of Tiwari, Clegg and Zou (2006) written out with `qf`,
//! their shift checked in R against `asht::wspoissonTest(wmtype = "tcz")`,
//! and the tests of homogeneity from `glm`.
//!
//! Tolerance. The reference has 17 significant digits. The rates, variances
//! and their ratio and difference are sums of a few dozen terms, so they
//! agree to `1e-13`, as in `r_rates.rs`. An F limit is a rate ratio times an F
//! quantile. The quantile is `(1 / y - 1) df2 / df1` with `y` a beta
//! quantile accurate to about `1e-14` relative in both implementations, which
//! the form multiplies by `1 / (1 - y)`, below 10 for these degrees of
//! freedom, and the degrees of freedom `2 R^2 / V` carry `1e-14`, which the
//! quantile passes on with an elasticity below 1; the bound is `1e-12`. The
//! Wald limits are an exponential of a few sums, `1e-13`. `glm` stops when
//! the relative change of the deviance falls below `1e-14`; its Newton steps
//! converge quadratically, so its coefficient is within about `1e-12` of the
//! maximum and the Pearson statistic and the ratio, which move linearly with
//! it, are checked to `1e-10`, while the deviance, which is stationary
//! there, is checked to `1e-12`. The Breslow-Day statistics are sums written
//! out in the same order, `1e-12`.

use epistat_core::Level;
use epistat_epi::PersonTime2;
use epistat_epi::mantel_haenszel::{rate_ratio_breslow_day, rate_ratio_homogeneity};
use epistat_epi::rates::{RatioInterval, Strata, standardised_difference, standardised_ratio};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    ratios: Vec<RatioCase>,
    homogeneity: Vec<Homogeneity>,
}

#[derive(Deserialize)]
struct RatioCase {
    name: String,
    events1: Vec<f64>,
    time1: Vec<f64>,
    events0: Vec<f64>,
    time0: Vec<f64>,
    weights: Vec<f64>,
    level: f64,
    ratio: f64,
    fay: Vec<f64>,
    tcz: Vec<f64>,
    difference: f64,
    difference_limits: Vec<f64>,
    log_limits: Option<Vec<f64>>,
}

#[derive(Deserialize)]
struct Homogeneity {
    name: String,
    x1: Vec<f64>,
    t1: Vec<f64>,
    x0: Vec<f64>,
    t0: Vec<f64>,
    df: usize,
    ratio: f64,
    deviance: f64,
    pearson: f64,
    poisson_lr: f64,
    breslow_day: f64,
    tarone: f64,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/rates_compare.json"
    ))
}

/// Records the deviation of `got` from `expected` in units of `bound`,
/// relative to `|expected|` or to 1 near 0, and keeps the largest.
fn check(worst: &mut f64, got: f64, expected: f64, bound: f64) {
    let ratio = (got - expected).abs() / (bound * expected.abs().max(1e-300));
    *worst = worst.max(ratio);
}

#[test]
fn ratio_and_difference_of_standardised_rates() {
    let mut worst = 0.0;
    for case in reference().ratios {
        let group = Strata {
            events: &case.events1,
            time: &case.time1,
        };
        let reference = Strata {
            events: &case.events0,
            time: &case.time0,
        };
        let name = &case.name;
        let level = Level::confidence(case.level).unwrap();
        for (interval, limits) in [
            (RatioInterval::F, &case.fay),
            (RatioInterval::ModifiedF, &case.tcz),
        ] {
            let r = standardised_ratio(group, reference, &case.weights, level, interval)
                .unwrap()
                .unwrap();
            check(&mut worst, r.estimate, case.ratio, 1e-13);
            if limits[0] == 0.0 {
                assert_eq!(r.lower(), Some(0.0), "{name}");
            } else {
                check(&mut worst, r.lower().unwrap(), limits[0], 1e-12);
            }
            check(&mut worst, r.upper().unwrap(), limits[1], 1e-12);
            assert_eq!(r.p_value, None);
        }
        let log = standardised_ratio(group, reference, &case.weights, level, RatioInterval::Log)
            .unwrap()
            .unwrap();
        match &case.log_limits {
            Some(l) => {
                for (got, want) in [log.lower(), log.upper(), log.p_value].into_iter().zip(l) {
                    check(&mut worst, got.unwrap(), *want, 1e-13);
                }
            }
            None => assert_eq!((log.estimate, log.lower()), (0.0, None), "{name}"),
        }
        let d = standardised_difference(group, reference, &case.weights, level, 1.0)
            .unwrap()
            .unwrap();
        // The difference is a difference of two sums: its error is relative to
        // the rates, not to the difference.
        let scale = case.difference.abs().max(case.difference_limits[1].abs());
        let mut absolute = |got: f64, want: f64, bound: f64| {
            worst = f64::max(worst, (got - want).abs() / (bound * scale));
        };
        absolute(d.estimate, case.difference, 1e-13);
        absolute(d.lower().unwrap(), case.difference_limits[0], 1e-13);
        absolute(d.upper().unwrap(), case.difference_limits[1], 1e-13);
        check(
            &mut worst,
            d.p_value.unwrap(),
            case.difference_limits[2],
            1e-12,
        );
    }
    eprintln!("standardised ratios: worst deviation {worst:.3} of its bound");
    assert!(worst <= 1.0, "{worst}");
}

#[test]
fn tests_of_a_common_rate_ratio() {
    let mut worst = 0.0;
    for case in reference().homogeneity {
        let strata: Vec<PersonTime2> = (0..case.x1.len())
            .map(|k| PersonTime2 {
                exposed_events: case.x1[k],
                exposed_time: case.t1[k],
                unexposed_events: case.x0[k],
                unexposed_time: case.t0[k],
            })
            .collect();
        let h = rate_ratio_homogeneity(&strata).unwrap().unwrap();
        let name = &case.name;
        assert_eq!(h.likelihood_ratio.df, case.df, "{name}");
        assert_eq!(h.pearson.df, case.df, "{name}");
        check(&mut worst, h.ratio, case.ratio, 1e-10);
        check(
            &mut worst,
            h.likelihood_ratio.statistic,
            case.deviance,
            1e-12,
        );
        check(
            &mut worst,
            h.likelihood_ratio.statistic,
            case.poisson_lr,
            1e-12,
        );
        check(&mut worst, h.pearson.statistic, case.pearson, 1e-10);
        let bd = rate_ratio_breslow_day(&strata, false).unwrap().unwrap();
        let ta = rate_ratio_breslow_day(&strata, true).unwrap().unwrap();
        assert_eq!((bd.df, ta.df), (case.df, case.df), "{name}");
        check(&mut worst, bd.statistic, case.breslow_day, 1e-12);
        check(&mut worst, ta.statistic, case.tarone, 1e-12);
    }
    eprintln!("homogeneity: worst deviation {worst:.3} of its bound");
    assert!(worst <= 1.0, "{worst}");
}

#[test]
fn undefined_comparisons() {
    let weights = [1.0, 1.0];
    let some = Strata {
        events: &[2.0, 3.0],
        time: &[10.0, 20.0],
    };
    let none = Strata {
        events: &[0.0, 0.0],
        time: &[10.0, 20.0],
    };
    let missing = Strata {
        events: &[2.0, 0.0],
        time: &[10.0, 0.0],
    };
    for interval in [
        RatioInterval::F,
        RatioInterval::ModifiedF,
        RatioInterval::Log,
    ] {
        let level = Level::default();
        let r = standardised_ratio(some, none, &weights, level, interval).unwrap();
        assert_eq!(r, None);
        let r = standardised_ratio(some, missing, &weights, level, interval).unwrap();
        assert_eq!(r, None);
    }
    let d = standardised_difference(none, none, &weights, Level::default(), 1.0)
        .unwrap()
        .unwrap();
    assert_eq!((d.estimate, d.interval), (0.0, None));
    // A common ratio with every event in one arm has no finite estimate.
    let strata = [
        PersonTime2 {
            exposed_events: 3.0,
            exposed_time: 10.0,
            unexposed_events: 0.0,
            unexposed_time: 10.0,
        },
        PersonTime2 {
            exposed_events: 2.0,
            exposed_time: 10.0,
            unexposed_events: 0.0,
            unexposed_time: 12.0,
        },
    ];
    assert_eq!(rate_ratio_homogeneity(&strata).unwrap(), None);
    assert_eq!(rate_ratio_breslow_day(&strata, true).unwrap(), None);
}
