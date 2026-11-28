//! Sensitivity values, amplification, binary and censored outcomes in
//! matched sets and the interval of an m-statistic, checked against
//! `sensitivitymv` 1.4.4, `sensitivitymw` 2.1 and `sensitivity2x2xk` 1.1.
//!
//! The reference is `tests/data/rosenbaum_methods.json`, written by
//! `tests/data/rosenbaum_methods.R`.
//!
//! * Sensitivity values: R's are `uniroot` with a tolerance of 1e-12 on the
//!   bound minus 0.05. The bounds agree with R's to 1e-13 absolute (R takes
//!   `1 - pnorm` and convolves by FFT), and the slope of each bound at the
//!   crossing is checked to exceed 0.01 a unit of `gamma`, so the two roots
//!   differ by at most `1e-12 + 1e-13 / 0.01 = 1.1e-11`; 1e-10 is allowed.
//! * `amplify` is the same expression in the same order, and must agree to
//!   the bit.
//! * `mh`: R convolves by FFT, whose error is absolute, about 1e-16 times
//!   the logarithm of the support, and ours sums the tail of a direct
//!   convolution to about 1e-14 relative: 1e-14 absolute plus 1e-12
//!   relative. `mhLS`'s moments to 1e-12 relative (sums of 200 terms) and its
//!   `1 - pnorm` p-value to 1e-14 absolute.
//! * Gehan's scores are whole numbers, so the statistic is exact; the
//!   expectation and variance of `separable1v` to 1e-12 relative and the
//!   p-value to 1e-14 absolute, as for `senmv` in `r_sensitivity`.
//! * `senmwCI` finds each end with `uniroot` to a tolerance of 1e-9 here,
//!   and ours is the crossing to adjacent doubles, so they agree to within
//!   twice that tolerance.

#[path = "common/r_reference.rs"]
mod r_reference;

use epistat_core::Level;
use epistat_epi::rosenbaum::{self, BinaryStratum};
use epistat_matching::sensitivity::{
    BinaryDistribution, Censored, CensoredSet, EventDirection, MScore, MatchedSet, Sides,
    WilcoxonDistribution, amplify, binary_bounds, binary_sensitivity_value, censored_bounds,
    censored_sensitivity_value, m_statistic_bounds, m_statistic_interval,
    m_statistic_sensitivity_value, wilcoxon_bounds, wilcoxon_sensitivity_value,
};
use epistat_testkit::{Tolerance, assert_close_to};
use r_reference::{Versions, assert_r_version, reference as load_reference};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    alpha: f64,
    gammas: Vec<f64>,
    pair_diffs: Vec<f64>,
    signed_value: f64,
    sets_mv: Vec<Vec<f64>>,
    mv_value: f64,
    binom_value: f64,
    binom_tail: Tail,
    binary_treated: Vec<f64>,
    binary_controls: Vec<Vec<f64>>,
    binary_13: Vec<Binary>,
    mh_value: f64,
    strata: Vec<[u64; 4]>,
    binary_strata: Vec<Binary>,
    censored: Vec<CensoredRows>,
    censored_bounds: Vec<Separable>,
    censored_value: f64,
    amplify: Vec<Amplify>,
    sets_mw: Vec<Vec<f64>>,
    senmwci: Vec<Interval>,
}

#[derive(Deserialize)]
struct Tail {
    upper: Vec<f64>,
    lower: Vec<f64>,
}

#[derive(Deserialize)]
struct Binary {
    gamma: f64,
    statistic: u64,
    p_upper: f64,
    p_lower: f64,
    ls_p_upper: f64,
    ls_expectation: f64,
    ls_variance: f64,
}

#[derive(Deserialize)]
struct CensoredRows {
    time: Vec<f64>,
    event: Vec<bool>,
}

#[derive(Deserialize)]
struct Separable {
    gamma: f64,
    pval: f64,
    statistic: f64,
    expectation: f64,
    variance: f64,
}

#[derive(Deserialize)]
struct Amplify {
    gamma: f64,
    lambda: Vec<f64>,
    delta: Vec<f64>,
}

#[derive(Deserialize)]
struct Interval {
    gamma: f64,
    one_sided: bool,
    estimate: [f64; 2],
    #[serde(rename = "interval")]
    limits: [Option<f64>; 2],
    tol: f64,
}

fn reference() -> Reference {
    let r: Reference = load_reference("rosenbaum_methods.json");
    assert_r_version(&r.versions["r"]);
    assert_eq!(r.versions["sensitivitymv"], "1.4.4");
    assert_eq!(r.versions["sensitivitymw"], "2.1");
    assert_eq!(r.versions["sensitivity2x2xk"], "1.1");
    r
}

fn sets(rows: &[Vec<f64>]) -> Vec<MatchedSet> {
    rows.iter()
        .map(|r| MatchedSet {
            treated: r[0],
            controls: r[1..].to_vec(),
        })
        .collect()
}

fn pairs(diffs: &[f64]) -> Vec<MatchedSet> {
    diffs
        .iter()
        .map(|d| MatchedSet {
            treated: *d,
            controls: vec![0.0],
        })
        .collect()
}

fn binary_pairs() -> Vec<MatchedSet> {
    let mut out = Vec::new();
    for (t, c, n) in [
        (1.0, 0.0, 30),
        (0.0, 1.0, 15),
        (1.0, 1.0, 10),
        (0.0, 0.0, 345),
    ] {
        for _ in 0..n {
            out.push(MatchedSet {
                treated: t,
                controls: vec![c],
            });
        }
    }
    out
}

fn binary_sets(r: &Reference) -> Vec<MatchedSet> {
    r.binary_treated
        .iter()
        .zip(&r.binary_controls)
        .map(|(t, c)| MatchedSet {
            treated: *t,
            controls: c.clone(),
        })
        .collect()
}

fn censored_sets(r: &Reference) -> Vec<CensoredSet> {
    r.censored
        .iter()
        .map(|s| {
            let units: Vec<Censored> = s
                .time
                .iter()
                .zip(&s.event)
                .map(|(t, e)| Censored {
                    time: *t,
                    event: *e,
                })
                .collect();
            CensoredSet {
                treated: units[0],
                controls: units[1..].to_vec(),
            }
        })
        .collect()
}

/// The value against R's root, after checking that the bound rises by at
/// least 0.01 a unit of gamma at the crossing.
fn check_value(got: f64, expected: f64, bound: impl Fn(f64) -> f64, what: &str) {
    let h = 1e-4;
    let slope = (bound(got + h) - bound(got - h)) / (2.0 * h);
    assert!(slope > 0.01, "{what}: slope {slope}");
    assert_close_to(what, got, expected, Tolerance::exact().abs(1e-10));
}

#[test]
fn sensitivity_values_agree_with_uniroot() {
    let r = reference();
    let paired = pairs(&r.pair_diffs);
    let v = wilcoxon_sensitivity_value(&paired, r.alpha, WilcoxonDistribution::Normal).unwrap();
    check_value(
        v,
        r.signed_value,
        |g| wilcoxon_bounds(&paired, &[g], WilcoxonDistribution::Normal).unwrap()[0].p_upper,
        "signed rank",
    );

    let mv = sets(&r.sets_mv);
    let spec = MScore::default();
    let v = m_statistic_sensitivity_value(&mv, r.alpha, &spec).unwrap();
    check_value(
        v,
        r.mv_value,
        |g| m_statistic_bounds(&mv, &[g], &spec).unwrap()[0].p_upper,
        "senmv",
    );

    let bp = binary_pairs();
    let v = binary_sensitivity_value(&bp, r.alpha, BinaryDistribution::Exact).unwrap();
    check_value(
        v,
        r.binom_value,
        |g| binary_bounds(&bp, &[g], BinaryDistribution::Exact).unwrap()[0].p_upper,
        "binomial",
    );
    // The exact signed rank bound of 0-1 pairs is the same binomial tail.
    let w = wilcoxon_sensitivity_value(&bp, r.alpha, WilcoxonDistribution::Exact).unwrap();
    assert_close_to(
        "signed rank of binary pairs",
        w,
        r.binom_value,
        Tolerance::exact().abs(1e-10),
    );

    let b13 = binary_sets(&r);
    let v = binary_sensitivity_value(&b13, r.alpha, BinaryDistribution::Exact).unwrap();
    check_value(
        v,
        r.mh_value,
        |g| binary_bounds(&b13, &[g], BinaryDistribution::Exact).unwrap()[0].p_upper,
        "mh",
    );

    let cs = censored_sets(&r);
    let v = censored_sensitivity_value(&cs, r.alpha, EventDirection::Earlier).unwrap();
    check_value(
        v,
        r.censored_value,
        |g| censored_bounds(&cs, &[g], EventDirection::Earlier).unwrap()[0].p_upper,
        "gehan",
    );
}

#[test]
fn binary_pairs_give_the_binomial_tail() {
    let r = reference();
    let bp = binary_pairs();
    let got = binary_bounds(&bp, &r.gammas, BinaryDistribution::Exact).unwrap();
    let exact = wilcoxon_bounds(&bp, &r.gammas, WilcoxonDistribution::Exact).unwrap();
    for (i, b) in got.iter().enumerate() {
        assert_eq!(b.informative, 45);
        // pbinom is accurate to a few units in the last place, and ours sums
        // 46 terms: 1e-13 relative.
        assert_close_to(
            "upper",
            b.p_upper,
            r.binom_tail.upper[i],
            Tolerance::exact().rel(1e-13),
        );
        assert_close_to(
            "lower",
            b.p_lower,
            r.binom_tail.lower[i],
            Tolerance::exact().rel(1e-13),
        );
        assert_close_to(
            "signed rank",
            exact[i].p_upper,
            r.binom_tail.upper[i],
            Tolerance::exact().rel(1e-13),
        );
    }
}

fn check_binary(
    got: &[epistat_matching::sensitivity::BinaryBounds],
    normal: &[epistat_matching::sensitivity::BinaryBounds],
    expected: &[Binary],
    what: &str,
) {
    for ((b, n), e) in got.iter().zip(normal).zip(expected) {
        assert_eq!(b.gamma.to_bits(), e.gamma.to_bits());
        assert_eq!(b.statistic, e.statistic, "{what}");
        assert_close_to(
            &format!("{what} upper {}", e.gamma),
            b.p_upper,
            e.p_upper,
            Tolerance::exact().abs(1e-12 * (e.p_upper).abs() + 1e-14),
        );
        assert_close_to(
            &format!("{what} lower {}", e.gamma),
            b.p_lower,
            e.p_lower,
            Tolerance::exact().abs(1e-12 * (e.p_lower).abs() + 1e-14),
        );
        assert_close_to(
            &format!("{what} mean"),
            n.expectation,
            e.ls_expectation,
            Tolerance::exact().rel(1e-12),
        );
        assert_close_to(
            &format!("{what} variance"),
            n.variance,
            e.ls_variance,
            Tolerance::exact().rel(1e-12),
        );
        assert_close_to(
            &format!("{what} normal"),
            n.p_upper,
            e.ls_p_upper,
            Tolerance::exact().abs(1e-14),
        );
    }
}

#[test]
fn binary_sets_agree_with_mh_and_mh_ls() {
    let r = reference();
    let b13 = binary_sets(&r);
    let exact = binary_bounds(&b13, &r.gammas, BinaryDistribution::Exact).unwrap();
    let normal = binary_bounds(&b13, &r.gammas, BinaryDistribution::Normal).unwrap();
    check_binary(&exact, &normal, &r.binary_13, "1:3 sets");

    let strata: Vec<BinaryStratum> = r
        .strata
        .iter()
        .map(|s| BinaryStratum {
            treated_cases: s[0],
            treated_noncases: s[1],
            control_cases: s[2],
            control_noncases: s[3],
        })
        .collect();
    let exact: Vec<_> = r
        .gammas
        .iter()
        .map(|g| rosenbaum::binary_bounds(&strata, *g, BinaryDistribution::Exact).unwrap())
        .collect();
    let normal: Vec<_> = r
        .gammas
        .iter()
        .map(|g| rosenbaum::binary_bounds(&strata, *g, BinaryDistribution::Normal).unwrap())
        .collect();
    check_binary(&exact, &normal, &r.binary_strata, "strata");
}

#[test]
fn censored_bounds_agree_with_separable1v_on_gehan_scores() {
    let r = reference();
    let cs = censored_sets(&r);
    let got = censored_bounds(&cs, &r.gammas, EventDirection::Earlier).unwrap();
    for (b, e) in got.iter().zip(&r.censored_bounds) {
        assert_eq!(b.gamma.to_bits(), e.gamma.to_bits());
        assert_eq!(b.statistic.to_bits(), e.statistic.to_bits());
        assert_close_to(
            "expectation",
            b.expectation,
            e.expectation,
            Tolerance::exact().abs(1e-12 * (e.expectation).abs() + 1e-13),
        );
        assert_close_to(
            "variance",
            b.variance,
            e.variance,
            Tolerance::exact().rel(1e-12),
        );
        assert_close_to("p", b.p_upper, e.pval, Tolerance::exact().abs(1e-14));
    }
    // The other direction negates the scores and so the statistic.
    let later = censored_bounds(&cs, &[1.0], EventDirection::Later).unwrap();
    assert_eq!(later[0].statistic, -got[0].statistic);
}

#[test]
fn amplification_agrees_with_amplify_to_the_bit() {
    let r = reference();
    for a in &r.amplify {
        for (lambda, delta) in a.lambda.iter().zip(&a.delta) {
            let got = amplify(a.gamma, *lambda).unwrap().unwrap();
            assert_eq!(got.to_bits(), delta.to_bits(), "{} {lambda}", a.gamma);
            // The relation is symmetric.
            let back = amplify(a.gamma, got).unwrap().unwrap();
            assert!((back - lambda).abs() <= 1e-12 * lambda);
        }
        assert_eq!(amplify(a.gamma, a.gamma).unwrap(), None);
    }
}

#[test]
fn the_interval_agrees_with_senmw_ci() {
    let r = reference();
    let mw = sets(&r.sets_mw);
    let spec = MScore::default();
    for case in &r.senmwci {
        let sides = if case.one_sided {
            Sides::One
        } else {
            Sides::Two
        };
        let level = Level::alpha(r.alpha).unwrap();
        let got = m_statistic_interval(&mw, case.gamma, &spec, level, sides).unwrap();
        let slack = 2.0 * case.tol;
        assert_close_to(
            "estimate lower",
            got.estimate_lower,
            case.estimate[0],
            Tolerance::exact().abs(slack),
        );
        assert_close_to(
            "estimate upper",
            got.estimate_upper,
            case.estimate[1],
            Tolerance::exact().abs(slack),
        );
        let lower = got.interval.lower.unwrap();
        assert_close_to(
            "lower",
            lower,
            case.limits[0].unwrap(),
            Tolerance::exact().abs(slack),
        );
        match case.limits[1] {
            Some(upper) => assert_close_to(
                "upper",
                got.interval.upper.unwrap(),
                upper,
                Tolerance::exact().abs(slack),
            ),
            None => assert_eq!(got.interval.upper, None),
        }
    }
}
