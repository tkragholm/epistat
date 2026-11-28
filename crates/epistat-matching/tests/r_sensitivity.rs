//! Rosenbaum bounds and E-values, checked against `rbounds` 2.2,
//! `sensitivitymv` 1.4.4, `sensitivitymw` 2.1 and `EValue` 4.1.4.
//!
//! The reference is `tests/data/sensitivity.json`, written by
//! `tests/data/sensitivity.R`.
//!
//! * `psens`: 60 simulated pairs with ties and zero differences, gamma 1 to 4.
//!   `psens` rounds its p-values to four digits. The test compares them to
//!   that rounding (a deviation of at most 5e-5 plus 1e-12) and compares the
//!   unrounded values the script computes by the same formula to 1e-12
//!   relative, the error of a normal tail from a sum of 60 squared ranks.
//! * The exact signed rank tail: against `psignrank` at gamma 1 for 25 pairs
//!   without ties, to 1e-12 relative (the recursion adds up to 25 terms per
//!   cell), and against a sum over all 2^12 sign patterns for 12 pairs with
//!   ties at gamma 1 to 3, to 1e-13 absolute.
//! * `hlsens`: the bounds of the Hodges-Lehmann estimate, which `hlsens`
//!   finds by steps of `pr` upwards from the estimate. Our bound is the exact
//!   crossing point, so the upper bound of `hlsens` lies in `[ours, ours +
//!   pr]` and the lower in `[ours - pr, ours]`. `hlsens` also rounds to five
//!   significant digits, 5e-5 at most for these values, which is added to the
//!   interval. At gamma 1 `hlsens` returns the estimate of `wilcox.test`,
//!   which uniroot finds to about 1e-4, so there it must lie within 1e-3 of
//!   our two bounds.
//! * `senmv` and `senmw`: statistic, expectation, variance and deviate at
//!   1e-10 relative (sums over 40 sets in a fixed order against R's), with an
//!   absolute allowance of 1e-13 for the statistic and the expectation, which
//!   are sums of up to 160 scores of size at most 1 and can be near zero, and
//!   of 1e-12 for the deviate. The p-value is compared at 1e-14 absolute,
//!   since R's `1 - pnorm` cannot be more accurate than that.
//! * `evalues.*`: 1e-12 relative, the error of a square root and a power.

#[path = "common/r_reference.rs"]
mod r_reference;

use epistat_epi::rosenbaum;
use epistat_matching::sensitivity::{
    EValues, MScore, MatchedSet, Outcome, WilcoxonDistribution, evalues_hr, evalues_md, evalues_or,
    evalues_rr, hodges_lehmann_bounds, m_statistic_bounds, wilcoxon_bounds,
};

use epistat_testkit::{Tolerance, assert_close_to};
use r_reference::reference as load_reference;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    pairs: Pairs,
    exact1: Exact1,
    exact2: Exact2,
    sets_mv: Vec<Vec<f64>>,
    sets_mw: Vec<Vec<f64>>,
    pair_diffs: Vec<f64>,
    senmv: Vec<Mv>,
    senmw: Vec<Mw>,
    evalues: Vec<Ev>,
}

#[derive(Deserialize)]
struct Pairs {
    trt: Vec<f64>,
    ctrl: Vec<f64>,
    gammas: Vec<f64>,
    psens_lower: Vec<f64>,
    psens_upper: Vec<f64>,
    psens_exact_lower: Vec<f64>,
    psens_exact_upper: Vec<f64>,
    statistic: f64,
    pr: f64,
    hl_lower: Vec<f64>,
    hl_upper: Vec<f64>,
}

#[derive(Deserialize)]
struct Exact1 {
    diffs: Vec<f64>,
    p: f64,
}

#[derive(Deserialize)]
struct Exact2 {
    diffs: Vec<f64>,
    gammas: Vec<f64>,
    p_lower: Vec<f64>,
    p_upper: Vec<f64>,
}

#[derive(Deserialize)]
struct Mv {
    case: String,
    input: String,
    gamma: f64,
    pval: f64,
    deviate: f64,
    statistic: f64,
    expectation: f64,
    variance: f64,
}

#[derive(Deserialize)]
struct Mw {
    gamma: f64,
    pval: f64,
    deviate: f64,
    statistic: f64,
    expectation: f64,
    variance: f64,
}

#[derive(Deserialize)]
struct Ev {
    kind: String,
    est: f64,
    lo: Option<f64>,
    hi: Option<f64>,
    se: Option<f64>,
    rare: Option<bool>,
    #[serde(rename = "true")]
    true_value: f64,
    point: Option<f64>,
    lower: Option<f64>,
    upper: Option<f64>,
}

fn reference() -> Reference {
    load_reference("sensitivity.json")
}

fn pair_sets(trt: &[f64], ctrl: &[f64]) -> Vec<MatchedSet> {
    trt.iter()
        .zip(ctrl)
        .map(|(t, c)| MatchedSet {
            treated: *t,
            controls: vec![*c],
        })
        .collect()
}

#[test]
fn signed_rank_bounds_agree_with_psens() {
    let r = reference();
    let p = &r.pairs;
    let sets = pair_sets(&p.trt, &p.ctrl);
    let got = wilcoxon_bounds(&sets, &p.gammas, WilcoxonDistribution::Normal).unwrap();
    assert!((got[0].statistic - p.statistic).abs() < 1e-12);
    for (i, b) in got.iter().enumerate() {
        assert_close_to(
            "lower",
            b.p_lower,
            p.psens_exact_lower[i],
            Tolerance::exact().rel(1e-12),
        );
        assert_close_to(
            "upper",
            b.p_upper,
            p.psens_exact_upper[i],
            Tolerance::exact().rel(1e-12),
        );
        assert!((b.p_lower - p.psens_lower[i]).abs() <= 5e-5 + 1e-12);
        assert!((b.p_upper - p.psens_upper[i]).abs() <= 5e-5 + 1e-12);
    }
}

#[test]
fn exact_signed_rank_tail_agrees_with_r() {
    let r = reference();
    let b = rosenbaum::wilcoxon_bounds(&r.exact1.diffs, 1.0, WilcoxonDistribution::Exact).unwrap();
    assert_close_to(
        "psignrank",
        b.p_upper,
        r.exact1.p,
        Tolerance::exact().rel(1e-12),
    );
    let e = &r.exact2;
    for (i, g) in e.gammas.iter().enumerate() {
        let b = rosenbaum::wilcoxon_bounds(&e.diffs, *g, WilcoxonDistribution::Exact).unwrap();
        assert!((b.p_lower - e.p_lower[i]).abs() < 1e-13, "lower {g}");
        assert!((b.p_upper - e.p_upper[i]).abs() < 1e-13, "upper {g}");
    }
}

#[test]
fn hodges_lehmann_bounds_agree_with_hlsens() {
    let r = reference();
    let p = &r.pairs;
    let sets = pair_sets(&p.trt, &p.ctrl);
    let got = hodges_lehmann_bounds(&sets, &p.gammas).unwrap();
    let slack = p.pr + 5e-5;
    for (i, b) in got.iter().enumerate() {
        if (b.gamma - 1.0).abs() < 1e-12 {
            assert!((b.lower - p.hl_lower[i]).abs() < 1e-3);
            assert!((b.upper - p.hl_upper[i]).abs() < 1e-3);
        } else {
            assert!(
                b.upper - 1e-9 <= p.hl_upper[i] && p.hl_upper[i] <= b.upper + slack,
                "upper {}: {} against {}",
                b.gamma,
                b.upper,
                p.hl_upper[i]
            );
            assert!(
                b.lower - slack <= p.hl_lower[i] && p.hl_lower[i] <= b.lower + 1e-9,
                "lower {}: {} against {}",
                b.gamma,
                b.lower,
                p.hl_lower[i]
            );
        }
    }
}

fn check_set_bound(
    got: &epistat_matching::sensitivity::SetBounds,
    pval: f64,
    deviate: f64,
    statistic: f64,
    expectation: f64,
    variance: f64,
    label: &str,
) {
    assert_close_to(
        label,
        got.statistic,
        statistic,
        Tolerance::exact().abs(1e-10 * (statistic).abs() + 1e-13),
    );
    assert_close_to(
        label,
        got.expectation,
        expectation,
        Tolerance::exact().abs(1e-10 * (expectation).abs() + 1e-13),
    );
    assert_close_to(label, got.variance, variance, Tolerance::exact().rel(1e-10));
    assert_close_to(
        label,
        got.deviate,
        deviate,
        Tolerance::exact().abs(1e-10 * (deviate).abs() + 1e-12),
    );
    assert!(
        (got.p_upper - pval).abs() < 1e-14,
        "{label} p: {} {pval}",
        got.p_upper
    );
}

#[test]
fn matched_set_bounds_agree_with_senmv() {
    let r = reference();
    let sets: Vec<MatchedSet> = r
        .sets_mv
        .iter()
        .map(|s| MatchedSet {
            treated: s[0],
            controls: s[1..].to_vec(),
        })
        .collect();
    let diff_sets: Vec<MatchedSet> = r
        .pair_diffs
        .iter()
        .map(|d| MatchedSet {
            treated: d / 2.0,
            controls: vec![-d / 2.0],
        })
        .collect();
    for case in &r.senmv {
        let spec = match case.case.as_str() {
            "default" | "tau" => MScore::default(),
            "inner" => MScore {
                inner: 0.5,
                ..MScore::default()
            },
            "lambda" => MScore {
                inner: 0.25,
                trim: 2.0,
                lambda: 0.75,
                ..MScore::default()
            },
            "trimmean" => MScore {
                trim: f64::INFINITY,
                treated_on_treated: true,
                ..MScore::default()
            },
            other => panic!("{other}"),
        };
        let used: Vec<MatchedSet> = if case.input == "diff" {
            diff_sets.clone()
        } else if case.case == "tau" {
            sets.iter()
                .map(|s| MatchedSet {
                    treated: s.treated - 0.2,
                    controls: s.controls.clone(),
                })
                .collect()
        } else {
            sets.clone()
        };
        let got = m_statistic_bounds(&used, &[case.gamma], &spec).unwrap();
        check_set_bound(
            &got[0],
            case.pval,
            case.deviate,
            case.statistic,
            case.expectation,
            case.variance,
            &format!("senmv {} {} {}", case.case, case.input, case.gamma),
        );
    }
}

#[test]
fn matched_set_bounds_agree_with_senmw() {
    let r = reference();
    let sets: Vec<MatchedSet> = r
        .sets_mw
        .iter()
        .map(|s| MatchedSet {
            treated: s[0],
            controls: s[1..].to_vec(),
        })
        .collect();
    // senmw's default outer trim is 3.
    let spec = MScore {
        trim: 3.0,
        ..MScore::default()
    };
    for case in &r.senmw {
        let got = m_statistic_bounds(&sets, &[case.gamma], &spec).unwrap();
        check_set_bound(
            &got[0],
            case.pval,
            case.deviate,
            case.statistic,
            case.expectation,
            case.variance,
            &format!("senmw {}", case.gamma),
        );
    }
}

fn same(got: Option<f64>, expected: Option<f64>, label: &str) {
    match (got, expected) {
        (None, None) => {}
        (Some(g), Some(e)) => assert_close_to(label, g, e, Tolerance::exact().rel(1e-12)),
        other => panic!("{label}: {other:?}"),
    }
}

#[test]
fn evalues_agree_with_the_evalue_package() {
    let r = reference();
    for (i, c) in r.evalues.iter().enumerate() {
        let outcome = match c.rare {
            Some(true) => Outcome::Rare,
            _ => Outcome::Common,
        };
        let got: EValues = match c.kind.as_str() {
            "RR" => evalues_rr(c.est, c.lo, c.hi, c.true_value),
            "OR" => evalues_or(c.est, c.lo, c.hi, outcome, c.true_value),
            "HR" => evalues_hr(c.est, c.lo, c.hi, outcome, c.true_value),
            "MD" => evalues_md(c.est, c.se, c.true_value),
            other => panic!("{other}"),
        }
        .unwrap();
        let label = format!("case {i} {} est {}", c.kind, c.est);
        same(Some(got.estimate), c.point, &format!("{label} point"));
        same(got.lower, c.lower, &format!("{label} lower"));
        same(got.upper, c.upper, &format!("{label} upper"));
    }
}
