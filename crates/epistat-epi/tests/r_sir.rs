//! SIRs checked against R 4.6.2: `stats::poisson.test` for the exact limits and
//! p-value, `uniroot` on the deviance equation for the profile limits, the
//! Wald formula, `popEpi::sir` 0.5.0 for the three interval types and the test
//! of homogeneity, and a `glm` deviance test for the trend.
//!
//! The reference is `tests/data/sir.json`, written by `tests/data/sir.R` with 17
//! significant digits.
//!
//! Tolerance. The exact limits are gamma quantiles, accurate to about `1e-13`
//! relative, so they are held to `1e-12`. The exact p-value is a sum of point
//! probabilities, held to `1e-11` as for the rate ratio. The profile limits
//! are roots found to a tolerance of `1e-15` in R and to four units in the
//! last place here, and the derivative of the deviance at the root is of order
//! `sqrt(O)`, so the roots agree to `1e-11` relative. The Wald limits are a
//! few operations, `1e-13`. The tests of homogeneity and trend are sums of at
//! most eight terms with a logarithm each, and the trend adds a Newton fit
//! that stops at a step of `1e-10`, so they are held to `1e-9`. `popEpi`
//! rounds its SIRs and limits to four decimals and its p-value to five, so they
//! are compared to `5.1e-5`. Its profile limits interpolate a profile on a grid,
//! which differs from the root by up to `2e-3` for these counts.

use epistat_epi::sir::{SirInterval, excess_absolute_risk, sir, sir_homogeneity, sir_trend};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    grid: Vec<Grid>,
    strata: Strata,
    big: Big,
}

#[derive(Deserialize)]
struct Grid {
    o: f64,
    e: f64,
    level: f64,
    exact_p: f64,
    exact_lower: f64,
    exact_upper: f64,
    profile_lower: f64,
    profile_upper: f64,
    wald_lower: Option<f64>,
    wald_upper: Option<f64>,
}

#[derive(Deserialize)]
struct Interval {
    sir: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

#[derive(Deserialize)]
struct PopEpi {
    univariate: Interval,
    wald: Interval,
    profile: Interval,
}

#[derive(Deserialize)]
struct Test {
    statistic: f64,
    p: f64,
}

#[derive(Deserialize)]
struct Strata {
    observed: Vec<f64>,
    expected: Vec<f64>,
    popepi: PopEpi,
    homogeneity: Test,
    trend: Test,
    trend_two: Test,
    score: Vec<f64>,
}

#[derive(Deserialize)]
struct Big {
    observed: Vec<f64>,
    expected: Vec<f64>,
    score: Vec<f64>,
    trend: Test,
    homogeneity: Test,
}

fn reference() -> Reference {
    load_json(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/sir.json"))
}

fn ratio(worst: &mut f64, what: &str, got: f64, want: f64, bound: f64) {
    let r = (got - want).abs() / (bound * want.abs().max(1e-300));
    *worst = worst.max(r);
    assert!(r <= 1.0, "{what}: {got:e} against {want:e}");
}

#[test]
fn grid_of_counts() {
    let mut worst = 0.0;
    for g in reference().grid {
        let what = format!("o {} e {} level {}", g.o, g.e, g.level);
        let exact = sir(g.o, g.e, g.level, SirInterval::Exact).unwrap();
        ratio(&mut worst, &what, exact.p_value.unwrap(), g.exact_p, 1e-11);
        ratio(
            &mut worst,
            &what,
            exact.upper.unwrap(),
            g.exact_upper,
            1e-12,
        );
        if g.exact_lower > 0.0 {
            ratio(
                &mut worst,
                &what,
                exact.lower.unwrap(),
                g.exact_lower,
                1e-12,
            );
        } else {
            assert_eq!(exact.lower, Some(0.0));
        }
        let profile = sir(g.o, g.e, g.level, SirInterval::Profile).unwrap();
        ratio(
            &mut worst,
            &what,
            profile.upper.unwrap(),
            g.profile_upper,
            1e-11,
        );
        if g.profile_lower > 0.0 {
            ratio(
                &mut worst,
                &what,
                profile.lower.unwrap(),
                g.profile_lower,
                1e-11,
            );
        } else {
            assert_eq!(profile.lower, Some(0.0));
        }
        let wald = sir(g.o, g.e, g.level, SirInterval::Wald).unwrap();
        match (g.wald_lower, g.wald_upper) {
            (Some(lo), Some(hi)) => {
                ratio(&mut worst, &what, wald.lower.unwrap(), lo, 1e-13);
                ratio(&mut worst, &what, wald.upper.unwrap(), hi, 1e-13);
            }
            _ => assert_eq!((wald.lower, wald.upper), (None, None)),
        }
    }
    eprintln!("grid: worst deviation {worst:.3} of its bound");
}

#[test]
fn popepi_strata() {
    let r = reference().strata;
    let near = 5.1e-5;
    let mut worst = 0.0_f64;
    for (kind, interval, tolerance) in [
        (SirInterval::Exact, &r.popepi.univariate, near),
        (SirInterval::Wald, &r.popepi.wald, near),
        (SirInterval::Profile, &r.popepi.profile, 2e-3),
    ] {
        for g in 0..r.observed.len() {
            let s = sir(r.observed[g], r.expected[g], 0.95, kind).unwrap();
            for (got, want) in [
                (s.sir.unwrap(), interval.sir[g]),
                (s.lower.unwrap(), interval.lower[g]),
                (s.upper.unwrap(), interval.upper[g]),
            ] {
                let d = (got - want).abs() / tolerance;
                worst = worst.max(d);
                assert!(d <= 1.0, "{kind:?} group {g}: {got} against {want}");
            }
        }
    }
    eprintln!("popEpi: worst deviation {worst:.3} of its bound");
}

#[test]
fn homogeneity_and_trend() {
    let r = reference();
    let mut worst = 0.0;
    let h = sir_homogeneity(&r.strata.observed, &r.strata.expected)
        .unwrap()
        .unwrap();
    ratio(
        &mut worst,
        "homogeneity",
        h.statistic,
        r.strata.homogeneity.statistic,
        1e-9,
    );
    ratio(
        &mut worst,
        "homogeneity p",
        h.p_value,
        r.strata.homogeneity.p,
        1e-7,
    );
    assert_eq!(h.df, 2);
    let t = sir_trend(&r.strata.observed, &r.strata.expected, &r.strata.score)
        .unwrap()
        .unwrap();
    ratio(
        &mut worst,
        "trend",
        t.statistic,
        r.strata.trend.statistic,
        1e-9,
    );
    ratio(&mut worst, "trend p", t.p_value, r.strata.trend.p, 1e-7);
    let two = sir_trend(
        &[r.strata.observed[0], r.strata.observed[2]],
        &[r.strata.expected[0], r.strata.expected[2]],
        &[1.0, 4.0],
    )
    .unwrap()
    .unwrap();
    ratio(
        &mut worst,
        "trend, two strata",
        two.statistic,
        r.strata.trend_two.statistic,
        1e-9,
    );
    let b = &r.big;
    let h = sir_homogeneity(&b.observed, &b.expected).unwrap().unwrap();
    ratio(
        &mut worst,
        "big homogeneity",
        h.statistic,
        b.homogeneity.statistic,
        1e-9,
    );
    ratio(
        &mut worst,
        "big homogeneity p",
        h.p_value,
        b.homogeneity.p,
        1e-7,
    );
    let t = sir_trend(&b.observed, &b.expected, &b.score)
        .unwrap()
        .unwrap();
    ratio(
        &mut worst,
        "big trend",
        t.statistic,
        b.trend.statistic,
        1e-9,
    );
    ratio(&mut worst, "big trend p", t.p_value, b.trend.p, 1e-7);
    eprintln!("tests: worst deviation {worst:.3} of its bound");
}

#[test]
fn excess_absolute_risk_by_hand() {
    assert_eq!(
        excess_absolute_risk(30.0, 20.0, 5000.0, 1000.0).unwrap(),
        Some(2.0)
    );
    assert_eq!(excess_absolute_risk(3.0, 2.0, 0.0, 1000.0).unwrap(), None);
}

/// The values go through `ln`, `exp` and `powf`, whose last bits depend on
/// the maths library and, on Windows, on the processor the library dispatches
/// for, so the bits are checked only on aarch64 macOS, where they were recorded.
#[test]
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn fingerprint_of_the_strata() {
    let r = reference();
    let mut hash = epistat_testkit::Fingerprint::new();
    for kind in [
        SirInterval::Exact,
        SirInterval::Byar,
        SirInterval::Wald,
        SirInterval::Profile,
    ] {
        for g in 0..r.strata.observed.len() {
            let s = sir(r.strata.observed[g], r.strata.expected[g], 0.95, kind).unwrap();
            hash = hash
                .option_f64(s.sir)
                .option_f64(s.lower)
                .option_f64(s.upper)
                .option_f64(s.p_value);
        }
    }
    let h = sir_homogeneity(&r.big.observed, &r.big.expected)
        .unwrap()
        .unwrap();
    let t = sir_trend(&r.big.observed, &r.big.expected, &r.big.score)
        .unwrap()
        .unwrap();
    let hash = hash.f64(h.statistic).f64(t.statistic).finish();
    eprintln!("sir fingerprint {hash:#018x}");
    assert_eq!(hash, 0xfd35_816f_3da9_0113);
}
