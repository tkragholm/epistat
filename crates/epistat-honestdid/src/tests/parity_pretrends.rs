//! The pre-trends power calculations against the R package `pretrends` 0.1.0.
//!
//! The reference is written by `tests/data/generate_pretrends_reference.R`.
//! It holds the package's numbers averaged over 20 seeds, since the package
//! draws its probabilities with `mvtnorm::pmvnorm` at its default tolerance,
//! and a tight reference (pass probability with `abseps` 1e-8, the
//! conditional mean by Tallis's formula) with the errors it reports.
//!
//! Tolerances, derived:
//! - against the tight reference, `5 (e_rust + e_ref)`: both errors are
//!   standard errors of unbiased randomly shifted lattice or Genz-Bretz
//!   estimates, and the difference of two independent ones has standard error
//!   at most the sum;
//! - against the package, `5 sd / sqrt(20) + 5 e_rust`: the reference is the
//!   mean of 20 independent runs of the package, whose standard deviation `sd`
//!   is recorded;
//! - against the numbers printed in the package README, `5 sd` of a single run
//!   plus the rounding of the printed digits, since the seed is not recorded.

use crate::pretrends::engine::{
    PretrendConfig, PretrendInput, analyse_pretest, pretest_pass_probability, slope_for_power,
};
use epistat_testkit::load_json;
use rstest::rstest;
use serde::Deserialize;

/// jsonlite writes a vector of length one as a scalar.
fn one_or_many<'de, D, T>(d: D) -> Result<Vec<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum V<T> {
        One(T),
        Many(Vec<T>),
    }
    Ok(match V::<T>::deserialize(d)? {
        V::One(x) => vec![x],
        V::Many(v) => v,
    })
}

#[derive(Deserialize)]
struct Reference {
    package_version: String,
    package_commit: String,
    alpha: f64,
    n_seeds: f64,
    readme: Readme,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Readme {
    slope50: f64,
    power: f64,
    bayes_factor: f64,
    likelihood_ratio: f64,
    #[serde(deserialize_with = "one_or_many")]
    mean_after: Vec<f64>,
    quad_power: f64,
    quad_bayes_factor: f64,
    quad_likelihood_ratio: f64,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    #[serde(deserialize_with = "one_or_many")]
    betahat: Vec<f64>,
    sigma: Vec<Vec<f64>>,
    #[serde(deserialize_with = "one_or_many")]
    tvec: Vec<f64>,
    reference: f64,
    #[serde(deserialize_with = "one_or_many")]
    delta: Vec<f64>,
    #[serde(deserialize_with = "one_or_many")]
    pre: Vec<usize>,
    #[serde(deserialize_with = "one_or_many")]
    post: Vec<usize>,
    pkg: Pkg,
    tight: Tight,
    slopes: Vec<Slope>,
}

#[derive(Deserialize)]
struct Pkg {
    power: f64,
    power_sd: f64,
    bayes_factor: f64,
    bayes_factor_sd: f64,
    likelihood_ratio: f64,
    likelihood_ratio_sd: f64,
    #[serde(deserialize_with = "one_or_many")]
    mean_after: Vec<f64>,
    #[serde(deserialize_with = "one_or_many")]
    mean_after_sd: Vec<f64>,
}

#[derive(Deserialize)]
struct Tight {
    pass: f64,
    pass_error: f64,
    pass_null: f64,
    pass_null_error: f64,
    #[serde(deserialize_with = "one_or_many")]
    mean_pre: Vec<f64>,
    #[serde(deserialize_with = "one_or_many")]
    mean_pre_error: Vec<f64>,
    #[serde(deserialize_with = "one_or_many")]
    mean_post: Vec<f64>,
    #[serde(deserialize_with = "one_or_many")]
    mean_post_error: Vec<f64>,
}

#[derive(Deserialize)]
struct Slope {
    target: f64,
    pkg_mean: f64,
    pkg_sd: f64,
}

/// Critical value of the tight reference: the normal quantile at `alpha`.
fn config_for(r: &Reference) -> PretrendConfig {
    PretrendConfig {
        alpha: r.alpha,
        abs_tol: 1e-6,
        ..PretrendConfig::default()
    }
}

/// The package tests each coefficient against the t-statistic 1.96.
fn package_config(r: &Reference) -> PretrendConfig {
    PretrendConfig {
        threshold: Some(1.96),
        ..config_for(r)
    }
}

impl Case {
    fn sigma_pre(&self) -> Vec<Vec<f64>> {
        let p = self.pre.len();
        self.sigma[..p].iter().map(|r| r[..p].to_vec()).collect()
    }
}

fn input(c: &Case) -> PretrendInput {
    let pick = |idx: &[usize]| idx.iter().map(|&i| c.tvec[i]).collect::<Vec<_>>();
    PretrendInput {
        betahat: c.betahat.clone(),
        covariance: c.sigma.clone(),
        pre_periods: pick(&c.pre),
        post_periods: pick(&c.post),
    }
}

#[rstest]
#[case("hewang_linear50")]
#[case("hewang_quadratic")]
#[case("hewang_linear_neg")]
#[case("syn_pre1")]
#[case("syn_pre2")]
#[case("syn_pre5_pos")]
#[case("syn_pre5_neg")]
#[case("syn_pre8")]
fn matches_the_package_and_the_tight_reference(#[case] case: &str) {
    let r: Reference = load_json("tests/data/pretrends_ref.json");
    assert_eq!(r.package_version, "0.1.0");
    assert_eq!(r.package_commit, "122731d082a5990e274f57fd9af0968e44977e7a");
    let cfg = config_for(&r);
    let pkg_cfg = package_config(&r);
    let root_n = r.n_seeds.sqrt();
    for c in r.cases.iter().filter(|c| c.name == case) {
        let inp = input(c);
        let a = analyse_pretest(&inp, &c.delta, &cfg).unwrap();
        let name = &c.name;

        let t = &c.tight;
        let tol = 5.0 * (a.pass_probability.error + t.pass_error) + 1e-12;
        assert!(
            (a.pass_probability.value - t.pass).abs() <= tol,
            "{name} pass"
        );
        let tol0 = 5.0 * (a.pass_probability_null.error + t.pass_null_error) + 1e-12;
        assert!(
            (a.pass_probability_null.value - t.pass_null).abs() <= tol0,
            "{name} pass0"
        );
        for (i, m) in a.mean_pre_after_pretest.iter().enumerate() {
            let tol = 5.0 * (m.error + t.mean_pre_error[i]) + 1e-12;
            assert!(
                (m.value - t.mean_pre[i]).abs() <= tol,
                "{name} pre mean {i}"
            );
        }
        for (i, m) in a.mean_post_after_pretest.iter().enumerate() {
            let tol = 5.0 * (m.error + t.mean_post_error[i]) + 1e-12;
            assert!(
                (m.value - t.mean_post[i]).abs() <= tol,
                "{name} post mean {i}"
            );
        }

        let p = analyse_pretest(&inp, &c.delta, &pkg_cfg).unwrap();
        let tol = 5.0 * c.pkg.power_sd / root_n + 5.0 * p.power.error + 1e-12;
        assert!((p.power.value - c.pkg.power).abs() <= tol, "{name} power");
        let tol = 5.0 * c.pkg.bayes_factor_sd / root_n + 5.0 * p.bayes_factor.error + 1e-12;
        assert!(
            (p.bayes_factor.value - c.pkg.bayes_factor).abs() <= tol,
            "{name} bf"
        );
        // The likelihood ratio is closed form in both: only rounding differs.
        assert!(
            (p.likelihood_ratio / c.pkg.likelihood_ratio - 1.0).abs() < 1e-11,
            "{name} lr {} {}",
            p.likelihood_ratio,
            c.pkg.likelihood_ratio
        );
        // The package lists the pre and post means in event-time order, the
        // reference period excluded.
        let ours: Vec<_> = p
            .mean_pre_after_pretest
            .iter()
            .chain(&p.mean_post_after_pretest)
            .collect();
        for (i, m) in ours.iter().enumerate() {
            let tol = 5.0 * c.pkg.mean_after_sd[i] / root_n + 5.0 * m.error + 1e-12;
            assert!(
                (m.value - c.pkg.mean_after[i]).abs() <= tol,
                "{name} pkg mean {i}"
            );
        }

        for s in &c.slopes {
            let got = slope_for_power(&inp, c.reference, s.target, &pkg_cfg).unwrap();
            // The slope moves by the error of the power divided by the slope of
            // the power curve; a slope error of the package sd covers it.
            // The package's root finder is `uniroot` at its default tolerance,
            // `.Machine$double.eps^0.25`, 1.22e-4, which bounds its slope.
            let tol = 5.0 * s.pkg_sd / root_n + 1.22e-4 + 5.0 * got.error + 1e-12;
            assert!(
                (got.value - s.pkg_mean).abs() <= tol,
                "{name} slope {}",
                s.target
            );
            // The slope has the requested power under our own evaluation.
            let own = pretest_pass_probability(
                &inp.linear_trend(got.value, c.reference)[..inp.n_pre()],
                &c.sigma_pre(),
                &pkg_cfg,
            )
            .unwrap();
            assert!(
                (1.0 - own.value - s.target).abs() <= 5.0 * got.error + 1e-11,
                "{name} {}: power {}",
                s.target,
                1.0 - own.value
            );
        }
    }
}

#[test]
fn reproduces_the_numbers_printed_in_the_package_readme() {
    let r: Reference = load_json("tests/data/pretrends_ref.json");
    let pkg_cfg = package_config(&r);
    let by = |n: &str| r.cases.iter().find(|c| c.name == n).unwrap();
    let c = by("hewang_linear50");
    let inp = input(c);
    let slope = slope_for_power(&inp, c.reference, 0.5, &pkg_cfg)
        .unwrap()
        .value;
    // Printed with 7 digits; the package's own root carries its power noise
    // (sd of the slope over seeds).
    let sd = c.slopes[0].pkg_sd;
    assert!(
        (slope - r.readme.slope50).abs() <= 5.0 * sd + 1e-7,
        "slope {slope}"
    );
    // The README hypothesises the printed slope.
    let delta = inp.linear_trend(r.readme.slope50, c.reference);
    let a = analyse_pretest(&inp, &delta, &pkg_cfg).unwrap();
    let sdp = c.pkg.power_sd;
    assert!((a.power.value - r.readme.power).abs() <= 5.0 * sdp + 1e-7);
    assert!((a.power.value - 0.5).abs() <= 5.0 * (a.power.error + sdp) + 1e-7);
    let sdb = c.pkg.bayes_factor_sd;
    assert!((a.bayes_factor.value - r.readme.bayes_factor).abs() <= 5.0 * sdb + 1e-5);
    assert!(
        (a.likelihood_ratio - r.readme.likelihood_ratio).abs()
            <= 5.0 * c.pkg.likelihood_ratio_sd + 1e-7
    );
    let ours: Vec<f64> = a
        .mean_pre_after_pretest
        .iter()
        .chain(&a.mean_post_after_pretest)
        .map(|m| m.value)
        .collect();
    for (i, want) in r.readme.mean_after.iter().enumerate() {
        let tol = 5.0 * c.pkg.mean_after_sd[i] + 1e-7;
        assert!(
            (ours[i] - want).abs() <= tol,
            "mean {i}: {} vs {want}",
            ours[i]
        );
    }

    let q = by("hewang_quadratic");
    let a = analyse_pretest(&input(q), &q.delta, &pkg_cfg).unwrap();
    assert!((a.power.value - r.readme.quad_power).abs() <= 5.0 * q.pkg.power_sd + 1e-7);
    assert!(
        (a.bayes_factor.value - r.readme.quad_bayes_factor).abs()
            <= 5.0 * q.pkg.bayes_factor_sd + 1e-7
    );
    assert!(
        (a.likelihood_ratio - r.readme.quad_likelihood_ratio).abs()
            <= 5.0 * q.pkg.likelihood_ratio_sd + 1e-7
    );
}

#[test]
fn results_are_deterministic() {
    let r: Reference = load_json("tests/data/pretrends_ref.json");
    let cfg = config_for(&r);
    let c = r.cases.iter().find(|c| c.name == "syn_pre5_pos").unwrap();
    let a = analyse_pretest(&input(c), &c.delta, &cfg).unwrap();
    let b = analyse_pretest(&input(c), &c.delta, &cfg).unwrap();
    assert_eq!(a, b);
}
