//! `survdiff::test` and `survdiff::one_sample` checked against `survdiff` of
//! survival 3.8.12 (R 4.6.2). `tests/data/survdiff.R` makes the data and the
//! reference and records the versions.
//!
//! How the call maps. The groups are the integer codes of the `group` column,
//! which R reads as a factor with ascending levels, and the strata the codes
//! of the `stratum` column, so rows and columns of the matrices line up. R's
//! `obs` and `exp` are group by stratum matrices flattened by column, which is
//! the stratum-major layout of the Rust result. `rho` is the same. The offset
//! of the one-sample test is the expected survival per row. R reports a chi
//! square of 0 for a fit with fewer than two groups left, and the Rust test
//! reports no statistic. No case here has that, and the Rust behaviour is
//! tested in `survdiff_edges`.
//!
//! Tolerances, with `n` rows, `k` groups, `u` the machine epsilon, `rho` the
//! exponent and `E` the largest expected count. Each side of the comparison
//! has its own rounding error and the bound is twice the one-sided bound.
//!
//! - `O` and `E` are sums of non-negative terms over at most `n` event
//!   times, with weights that are products of up to `n` factors raised to the
//!   power `rho`: relative error `2 n (1 + |rho|) u`.
//! - `V` entries: absolute error `8 n (1 + |rho|) u E`, since each term is
//!   bounded by the expected count of its time.
//! - The statistic is `d' V^- d`. Its error from `V` is `cond` times the
//!   relative error of `V`, with `cond` the ratio of the largest to the
//!   smallest retained eigenvalue of R's matrix, and from `d` the term
//!   `2 sqrt(stat) |delta d| / sqrt(lambda_min)`, with `|delta d|` the 2-norm
//!   of the error in `O - E`. Tolerance `8 (n + k) u cond stat` plus the
//!   second term with twice the bound of `O` and `E`.
//! - The p-value: the statistic's tolerance times the chi-square density at
//!   the statistic, plus the error of the tail function, which is measured
//!   against R's own `pchisq` at R's statistic in `tail_function_agrees`.

use epistat_stats::dist::chi_square::sf as chi_square_sf;
use epistat_survival::SurvData;
use epistat_survival::survdiff::{Options, one_sample, test};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    cases: Vec<Case>,
    one_sample: Vec<One>,
}

#[derive(Deserialize)]
struct Versions {
    #[serde(rename = "R")]
    r: String,
    survival: String,
}

#[derive(Deserialize)]
struct Case {
    label: String,
    rho: f64,
    use_strata: bool,
    time: Vec<f64>,
    status: Vec<i32>,
    group: Vec<u32>,
    stratum: Option<Vec<u32>>,
    observed: Vec<f64>,
    expected: Vec<f64>,
    variance: Vec<f64>,
    chisq: f64,
    pvalue: f64,
    eigenvalues: Vec<f64>,
    n_event_times: usize,
    n_groups: usize,
}

#[derive(Deserialize)]
struct One {
    rho: f64,
    time: Vec<f64>,
    status: Vec<i32>,
    offset: Vec<f64>,
    observed: f64,
    expected: f64,
    variance: f64,
    chisq: f64,
    pvalue: f64,
}

const U: f64 = f64::EPSILON;

/// Density of the chi-square distribution with `df` degrees of freedom.
fn chi_square_pdf(x: f64, df: usize) -> f64 {
    // Gamma(df / 2) for df >= 1 by the recurrence from Gamma(1) and Gamma(1/2).
    let mut g = if df.is_multiple_of(2) {
        1.0
    } else {
        std::f64::consts::PI.sqrt()
    };
    let mut a = if df.is_multiple_of(2) { 1.0 } else { 0.5 };
    while a < df as f64 / 2.0 {
        g *= a;
        a += 1.0;
    }
    let h = df as f64 / 2.0;
    x.powf(h - 1.0) * (-x / 2.0).exp() / (2.0_f64.powf(h) * g)
}

/// Deviation as a multiple of its tolerance, with the worst per output kept.
#[derive(Default)]
struct Worst {
    observed: f64,
    expected: f64,
    variance: f64,
    statistic: f64,
    p_value: f64,
}

fn ratio(got: f64, want: f64, tol: f64) -> f64 {
    (got - want).abs() / tol
}

#[test]
fn survdiff_matches_survival() {
    let reference: Reference = load_json("tests/data/survdiff.json");
    assert_eq!(reference.versions.survival, "3.8.12");
    assert!(reference.versions.r.starts_with("R version 4.6"));
    let mut worst = Worst::default();
    for case in &reference.cases {
        let n = case.time.len() as f64;
        let k = case.n_groups;
        let rho = case.rho;
        let mut builder = SurvData::builder(&case.time, &case.status);
        if let Some(stratum) = &case.stratum {
            builder = builder.stratum(stratum);
        }
        let data = builder.build().unwrap();
        let got = test(
            &data,
            &case.group,
            &Options {
                rho,
                ..Options::default()
            },
        )
        .unwrap();
        assert_eq!(got.groups.len(), k, "{}", case.label);
        assert_eq!(
            got.n_strata,
            if case.use_strata {
                got.observed_by_stratum.len() / k
            } else {
                1
            }
        );
        assert_eq!(
            got.observed_by_stratum.len(),
            case.observed.len(),
            "{}",
            case.label
        );
        assert_eq!(got.n_event_times, case.n_event_times, "{}", case.label);

        let rel = 2.0 * n * (1.0 + rho.abs()) * U;
        for (g, w) in got.observed_by_stratum.iter().zip(&case.observed) {
            worst.observed =
                worst
                    .observed
                    .max(ratio(*g, *w, rel * w.abs().max(f64::MIN_POSITIVE)));
        }
        for (g, w) in got.expected_by_stratum.iter().zip(&case.expected) {
            worst.expected =
                worst
                    .expected
                    .max(ratio(*g, *w, rel * w.abs().max(f64::MIN_POSITIVE)));
        }
        // Totals over strata of the reference, summed here in stratum order.
        let max_e = case.expected.iter().fold(0.0_f64, |acc, &e| acc.max(e));
        let n_strata = case.expected.len() / k;
        let mut total_e_err_sq = 0.0;
        for g in 0..k {
            let o: f64 = (0..n_strata).map(|s| case.observed[s * k + g]).sum();
            let e: f64 = (0..n_strata).map(|s| case.expected[s * k + g]).sum();
            worst.observed = worst.observed.max(ratio(got.observed[g], o, rel * o));
            worst.expected = worst.expected.max(ratio(got.expected[g], e, rel * e));
            let d_err = rel * (o + e);
            total_e_err_sq += d_err * d_err;
        }
        let v_tol = 8.0 * n * (1.0 + rho.abs()) * U * max_e;
        for (g, w) in got.variance.iter().zip(&case.variance) {
            worst.variance = worst.variance.max(ratio(*g, *w, v_tol));
        }

        let stat = got.statistic.expect("statistic");
        assert_eq!(got.df, Some(k - 1), "{}", case.label);
        let lambda_max = case.eigenvalues[0];
        let lambda_min = case.eigenvalues[k - 2];
        let cond = lambda_max / lambda_min;
        let stat_tol = 8.0 * (n + k as f64) * U * cond * case.chisq
            + 2.0 * case.chisq.sqrt() * total_e_err_sq.sqrt() / lambda_min.sqrt();
        worst.statistic = worst.statistic.max(ratio(stat, case.chisq, stat_tol));
        let p = got.p_value.unwrap();
        let p_tol = chi_square_pdf(case.chisq, k - 1) * stat_tol + 1e-13 * case.pvalue;
        worst.p_value = worst.p_value.max(ratio(p, case.pvalue, p_tol));
    }
    println!(
        "worst deviation in units of tolerance: observed {:.3}, expected {:.3}, variance {:.3}, statistic {:.3}, p {:.3}",
        worst.observed, worst.expected, worst.variance, worst.statistic, worst.p_value
    );
    for w in [
        worst.observed,
        worst.expected,
        worst.variance,
        worst.statistic,
        worst.p_value,
    ] {
        assert!(w <= 1.0, "a deviation exceeds its tolerance: {w}");
    }
}

/// The tail function of `epistat-stats` against `pchisq` at R's statistics.
#[test]
fn tail_function_agrees() {
    let reference: Reference = load_json("tests/data/survdiff.json");
    let mut worst = 0.0_f64;
    for case in &reference.cases {
        let df = (case.n_groups - 1) as f64;
        let p = chi_square_sf(case.chisq, df);
        worst = worst.max(((p - case.pvalue) / case.pvalue).abs());
    }
    println!("worst relative error of the tail function: {worst:e}");
    assert!(worst <= 1e-13, "{worst:e}");
}

#[test]
fn one_sample_matches_survival() {
    let reference: Reference = load_json("tests/data/survdiff.json");
    let mut worst = [0.0_f64; 4];
    for case in &reference.one_sample {
        let n = case.time.len() as f64;
        let data = SurvData::builder(&case.time, &case.status).build().unwrap();
        let got = one_sample(
            &data,
            &case.offset,
            &Options {
                rho: case.rho,
                ..Options::default()
            },
        )
        .unwrap();
        // ln of each expected survival is one rounding, summed over n terms.
        let rel = 2.0 * n * (1.0 + case.rho.abs()) * U;
        worst[0] = worst[0].max(ratio(got.observed, case.observed, 0.5));
        worst[1] = worst[1].max(ratio(got.expected, case.expected, rel * case.expected));
        worst[2] = worst[2].max(ratio(got.variance, case.variance, rel * case.variance));
        // num = observed - expected cancels: its error is rel times the sum of
        // the two, and the statistic num^2 / var has twice the relative error
        // of num plus that of var.
        let num = case.chisq.sqrt() * case.variance.sqrt();
        let num_err = rel * (case.observed + case.expected + case.variance);
        let stat_tol = 2.0 * 2.0 * num_err / num * case.chisq + rel * case.chisq;
        let stat = got.statistic.unwrap();
        worst[3] = worst[3].max(ratio(stat, case.chisq, stat_tol));
        let p_tol = chi_square_pdf(case.chisq, 1) * stat_tol + 1e-13 * case.pvalue;
        assert!(ratio(got.p_value.unwrap(), case.pvalue, p_tol) <= 1.0, "p");
    }
    println!("one sample worst in tolerances: {worst:?}");
    assert!(worst.iter().all(|&w| w <= 1.0), "{worst:?}");
}
