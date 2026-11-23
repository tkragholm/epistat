//! The refusals of the estimators, and the double robustness of the stacked
//! estimator on simulated data: with the outcome model wrong and the propensity
//! model right, and the reverse, the mean estimate is unbiased and the stacked
//! confidence interval covers the true effect.
//!
//! The tolerances are multiples of the Monte Carlo standard error computed in the
//! test: the bias of the mean of `R` estimates is within 4 standard errors of
//! their mean, and the coverage within 4 binomial standard errors of 95 percent.

use epistat_causal::aipw::{AipwInput, AipwOptions, GlmInput, Target, aipw, aipw_glm};
use epistat_glm::Family;
use epistat_linalg::Mat;
use epistat_rng::Stream;

fn input<'a>(
    y: &'a [f64],
    a: &'a [bool],
    m1: &'a [f64],
    m0: &'a [f64],
    e: &'a [f64],
) -> AipwInput<'a> {
    AipwInput {
        outcome: y,
        treatment: a,
        mu1: m1,
        mu0: m0,
        propensity: e,
        weight: None,
        cluster: None,
    }
}

#[test]
fn refusals() {
    let (y, a) = ([1.0, 0.0, 1.0, 0.0], [true, false, true, false]);
    let (m1, m0, e) = ([0.5; 4], [0.4; 4], [0.5; 4]);
    let ok = input(&y, &a, &m1, &m0, &e);
    assert!(aipw(&ok, Target::Ate, &AipwOptions::default()).is_ok());
    let short = [0.5; 3];
    assert!(
        aipw(
            &input(&y, &a, &short, &m0, &e),
            Target::Ate,
            &AipwOptions::default()
        )
        .is_err()
    );
    for bad in [0.0, 1.0, -0.1, 1.2, f64::NAN] {
        let p = [0.5, bad, 0.5, 0.5];
        assert!(
            aipw(
                &input(&y, &a, &m1, &m0, &p),
                Target::Ate,
                &AipwOptions::default()
            )
            .is_err(),
            "propensity {bad}"
        );
    }
    let all_treated = [true; 4];
    assert!(
        aipw(
            &input(&y, &all_treated, &m1, &m0, &e),
            Target::Ate,
            &AipwOptions::default()
        )
        .is_err()
    );
    let nan_y = [1.0, f64::NAN, 1.0, 0.0];
    assert!(
        aipw(
            &input(&nan_y, &a, &m1, &m0, &e),
            Target::Ate,
            &AipwOptions::default()
        )
        .is_err()
    );
    for truncation in [(0.0, 0.9), (0.1, 1.0), (0.6, 0.4)] {
        let opts = AipwOptions {
            truncation: Some(truncation),
            ..AipwOptions::default()
        };
        assert!(aipw(&ok, Target::Ate, &opts).is_err(), "{truncation:?}");
    }
    for alpha in [0.0, 1.0] {
        let opts = AipwOptions {
            alpha,
            ..AipwOptions::default()
        };
        assert!(aipw(&ok, Target::Ate, &opts).is_err());
    }
    // One cluster has no variance.
    let one = [1_u32; 4];
    let clustered = AipwInput {
        cluster: Some(&one),
        ..ok
    };
    assert!(aipw(&clustered, Target::Ate, &AipwOptions::default()).is_err());
    // The ratio is undefined when the mean of the controls is 0.
    let zero = [0.0; 4];
    let y0 = [1.0, 0.0, 1.0, 0.0];
    let r = aipw(
        &input(&y0, &a, &m1, &zero, &e),
        Target::Att,
        &AipwOptions::default(),
    )
    .unwrap();
    assert!(r.ratio.is_none() || r.mean0.estimate != 0.0);
}

#[test]
fn separated_propensity_model_is_an_error() {
    let n = 40;
    let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { i as f64 });
    // Treated if and only if the covariate is above 19.5.
    let a: Vec<bool> = (0..n).map(|i| i >= 20).collect();
    let y: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
    let input = GlmInput {
        outcome: &y,
        treatment: &a,
        outcome_design: x.as_ref(),
        propensity_design: x.as_ref(),
        weight: None,
        cluster: None,
    };
    let r = aipw_glm(
        &input,
        Family::Gaussian,
        Target::Ate,
        &AipwOptions::default(),
    );
    assert!(r.is_err(), "{r:?}");
    // An aliased design column and an unsupported family are errors.
    let dup = Mat::from_fn(n, 3, |i, j| if j == 0 { 1.0 } else { (i % 7) as f64 });
    let a2: Vec<bool> = (0..n).map(|i| i % 2 == 0 || i % 5 == 0).collect();
    let input = GlmInput {
        outcome: &y,
        treatment: &a2,
        outcome_design: dup.as_ref(),
        propensity_design: x.as_ref(),
        weight: None,
        cluster: None,
    };
    assert!(
        aipw_glm(
            &input,
            Family::Gaussian,
            Target::Ate,
            &AipwOptions::default()
        )
        .is_err()
    );
    let input = GlmInput {
        outcome_design: x.as_ref(),
        ..input
    };
    assert!(aipw_glm(&input, Family::Gamma, Target::Ate, &AipwOptions::default()).is_err());
}

/// One simulated data set: `x` standard normal truncated at 2.5 in absolute value (the weights of the tails of an untruncated normal make the sampling distribution of the estimate heavy-tailed, which no variance estimate describes), a propensity and an outcome that depend on
/// `x` and `x^2 - 1`, the true average effect 1.
fn simulate(rng: &mut Stream, n: usize) -> (Vec<f64>, Vec<bool>, Vec<f64>) {
    let mut y = Vec::with_capacity(n);
    let mut a = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = rng.normal().clamp(-2.5, 2.5);
        let q = xi * xi - 1.0;
        let e = 1.0 / (1.0 + (-(0.2 + 0.4 * xi + 0.3 * q)).exp());
        let t = rng.uniform() < e;
        let mu = 1.0 + f64::from(u8::from(t)) + 0.8 * xi + 1.2 * q;
        y.push(mu + rng.normal());
        a.push(t);
        x.push(xi);
    }
    (y, a, x)
}

fn design(x: &[f64], quadratic: bool) -> Mat<f64> {
    Mat::from_fn(x.len(), if quadratic { 3 } else { 2 }, |i, j| match j {
        0 => 1.0,
        1 => x[i],
        _ => x[i] * x[i] - 1.0,
    })
}

#[test]
fn estimator_is_unbiased_when_one_model_is_wrong() {
    let reps: usize = 300;
    let n: usize = 1000;
    for (name, outcome_quadratic, propensity_quadratic) in [
        ("both models right", true, true),
        ("outcome model wrong", false, true),
        ("propensity model wrong", true, false),
    ] {
        let mut rng = Stream::seeded(20_261_124);
        let (mut estimates, mut ses, mut covered) = (Vec::new(), Vec::new(), 0usize);
        for _ in 0..reps {
            let (y, a, x) = simulate(&mut rng, n);
            let (xo, xp) = (
                design(&x, outcome_quadratic),
                design(&x, propensity_quadratic),
            );
            let r = aipw_glm(
                &GlmInput {
                    outcome: &y,
                    treatment: &a,
                    outcome_design: xo.as_ref(),
                    propensity_design: xp.as_ref(),
                    weight: None,
                    cluster: None,
                },
                Family::Gaussian,
                Target::Ate,
                &AipwOptions::default(),
            )
            .unwrap();
            estimates.push(r.difference.estimate);
            ses.push(r.difference.se);
            covered += usize::from(r.difference.lower <= 1.0 && 1.0 <= r.difference.upper);
        }
        let mean = estimates.iter().sum::<f64>() / reps as f64;
        let var = estimates.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (reps as f64 - 1.0);
        let mc_se = (var / reps as f64).sqrt();
        println!(
            "{name}: mean {mean:.4}, sd {:.4}, mean se {:.4}, coverage {:.3}",
            var.sqrt(),
            ses.iter().sum::<f64>() / reps as f64,
            covered as f64 / reps as f64
        );
        assert!(
            (mean - 1.0).abs() <= 4.0 * mc_se,
            "{name}: mean {mean}, Monte Carlo standard error {mc_se}"
        );
        let coverage = covered as f64 / reps as f64;
        let coverage_se = (0.95 * 0.05 / reps as f64).sqrt();
        assert!(
            (coverage - 0.95).abs() <= 4.0 * coverage_se,
            "{name}: coverage {coverage} (standard error {coverage_se})"
        );
        println!(
            "{name}: bias {:.4} (Monte Carlo se {mc_se:.4}), sd {:.4}, mean se {:.4}, coverage {coverage:.3}",
            mean - 1.0,
            var.sqrt(),
            ses.iter().sum::<f64>() / reps as f64
        );
    }
}
