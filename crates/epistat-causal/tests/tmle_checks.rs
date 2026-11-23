//! Properties of the targeted estimators on simulated data: the score equations
//! they solve, the range of their estimates, their approach to the augmented
//! estimator as the sample grows, and their refusals.

use epistat_causal::aipw::{AipwInput, AipwOptions, GlmInput, Target, aipw, aipw_glm};
use epistat_causal::tmle::{
    OutcomeModel, OutcomeRange, PropensityBound, Targeting, TmleOptions, tmle, tmle_glm,
};
use epistat_glm::Family;
use epistat_linalg::Mat;
use epistat_rng::Stream;

struct Data {
    y: Vec<f64>,
    a: Vec<bool>,
    x: Mat<f64>,
}

fn expit(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn simulate(seed: u64, n: usize) -> Data {
    let mut rng = Stream::seeded(seed);
    let (mut y, mut a, mut u, mut v) = (vec![], vec![], vec![], vec![]);
    for _ in 0..n {
        let (x1, x2) = (rng.normal(), f64::from(u8::from(rng.uniform() < 0.4)));
        let t = rng.uniform() < expit(0.6 * x1 - 0.5 * x2 - 0.1);
        let p = expit(-0.5 + 0.8 * x1 + 0.6 * x2 + 0.7 * f64::from(u8::from(t)));
        y.push(f64::from(u8::from(rng.uniform() < p)));
        a.push(t);
        u.push(x1);
        v.push(x2);
    }
    let x = Mat::from_fn(n, 3, |i, j| [1.0, u[i], v[i]][j]);
    Data { y, a, x }
}

impl Data {
    fn input(&self) -> GlmInput<'_> {
        GlmInput {
            outcome: &self.y,
            treatment: &self.a,
            outcome_design: self.x.as_ref(),
            propensity_design: self.x.as_ref(),
            weight: None,
            cluster: None,
        }
    }
}

fn binomial_deviance(y: f64, q: f64) -> f64 {
    let term = |a: f64, b: f64| if a > 0.0 { a * (a / b).ln() } else { 0.0 };
    2.0 * (term(y, q) + term(1.0 - y, 1.0 - q))
}

/// The fluctuation stops when the deviance changes by less than `1e-8 (D + 0.1)`.
/// A Newton step lowers the deviance by about `U' I^-1 U` for the score `U` and
/// information `I` at its start, so at the start of the last step `|U_a| <=
/// sqrt(1e-8 (D + 0.1) I_aa)` for each of the two clever covariates, whose
/// information matrix is diagonal since no person has both. The step only shrinks
/// the score. The score of the result is `(u - l) (U_1 - U_0) / n`.
#[test]
fn fluctuation_solves_its_score_equations() {
    let d = simulate(31, 2000);
    let n = d.y.len() as f64;
    let opts = TmleOptions {
        propensity_bound: PropensityBound::Tmle,
        ..TmleOptions::default()
    };
    let r = tmle_glm(
        &d.input(),
        Family::Binomial,
        OutcomeModel::Pooled,
        Target::Ate,
        &opts,
    )
    .unwrap();
    let Targeting::Fluctuation { epsilon, .. } = r.targeting else {
        panic!("targeting");
    };
    let b = r.propensity_bound;
    let g = propensity(&d);
    let (mut u, mut info, mut deviance) = ([0.0f64; 2], [0.0f64; 2], 0.0);
    for i in 0..d.y.len() {
        let (q, cov) = if d.a[i] {
            (r.targeted1[i].unwrap(), 1.0 / g[i].max(b))
        } else {
            (r.targeted0[i].unwrap(), 1.0 / (1.0 - g[i]).max(b))
        };
        let k = usize::from(d.a[i]);
        u[k] += cov * (d.y[i] - q);
        info[k] += cov * cov * q * (1.0 - q);
        deviance += binomial_deviance(d.y[i], q);
    }
    let limit = |k: usize| (1e-8 * (deviance + 0.1) * info[k]).sqrt();
    for (k, score) in u.iter().enumerate() {
        assert!(score.abs() <= limit(k), "score {k}: {score} > {}", limit(k));
    }
    assert!(r.score.abs() <= (limit(0) + limit(1)) / n);
    assert!(epsilon.iter().all(|e| e.is_finite()));
    println!(
        "scores {u:?}, limits {:?}, score {:e}",
        [limit(0), limit(1)],
        r.score
    );
}

/// The propensities of the propensity model, bounded as the estimator bounds them.
fn propensity(d: &Data) -> Vec<f64> {
    let z: Vec<f64> = d.a.iter().map(|&t| f64::from(u8::from(t))).collect();
    let fit = epistat_glm::Glm::new(Family::Binomial, epistat_glm::Link::Logit)
        .fit(d.x.as_ref(), &z, None, None, None)
        .unwrap();
    fit.fitted
}

/// The one-step path stops within one step `h` of the minimum of the loss along
/// it. Along the path the mean of the influence function changes by about `q
/// P_n IC^2 <= m se^2 / (u - l)` per unit of `epsilon`, so the score at step `h`
/// is within `m se^2 h / (u - l)` of its limit as `h` goes to 0, and the test
/// allows twice that, with the score at the smallest step standing in for the
/// limit. The limit is not 0: it is `psi (q - P_n g) / q` (see the module
/// documentation), which the test does not need.
#[test]
fn one_step_score_converges_with_the_step() {
    let d = simulate(32, 2000);
    let steps = [1e-3, 1e-4, 1e-5];
    for target in [Target::Att, Target::Atc] {
        let runs: Vec<_> = steps
            .iter()
            .map(|&step| {
                let opts = TmleOptions {
                    step,
                    ..TmleOptions::default()
                };
                tmle_glm(
                    &d.input(),
                    Family::Binomial,
                    OutcomeModel::Pooled,
                    target,
                    &opts,
                )
                .unwrap()
            })
            .collect();
        let last = runs.last().unwrap();
        let curvature = 2.0 * last.n_used as f64 * last.difference.se.powi(2);
        for (r, &h) in runs.iter().zip(&steps) {
            let Targeting::OneStep { steps: k, .. } = r.targeting else {
                panic!("targeting");
            };
            println!("{target:?} step {h:e}: {k} steps, score {:e}", r.score);
            let bound = curvature * (h + steps[2]);
            assert!(
                (r.score - last.score).abs() <= bound,
                "{target:?} {h}: {} against {} (bound {bound:e})",
                r.score,
                last.score
            );
        }
    }
}

/// With a few treated persons of very small propensity whose outcome the outcome
/// model misses, the augmented mean of the treated leaves `[0, 1]`, and the
/// targeted mean, a mean of predictions in `(0, 1)`, does not.
#[test]
fn estimates_stay_in_the_range_of_the_outcome() {
    let n = 200;
    let a: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    let y: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i % 4 != 2))).collect();
    let e: Vec<f64> = (0..n)
        .map(|i| if i % 2 == 0 && i < 20 { 0.01 } else { 0.5 })
        .collect();
    let m1 = vec![0.9; n];
    let m0 = vec![0.5; n];
    let input = AipwInput {
        outcome: &y,
        treatment: &a,
        mu1: &m1,
        mu0: &m0,
        propensity: &e,
        weight: None,
        cluster: None,
    };
    let augmented = aipw(&input, Target::Ate, &AipwOptions::default()).unwrap();
    assert!(
        !(0.0..=1.0).contains(&augmented.mean1.estimate),
        "{}",
        augmented.mean1.estimate
    );
    let targeted = tmle(&input, Target::Ate, &TmleOptions::default()).unwrap();
    for m in [targeted.mean1.estimate, targeted.mean0.estimate] {
        assert!(m > 0.0 && m < 1.0, "{m}");
    }
}

/// The targeted and the augmented estimators differ by a term of order `1 / n`
/// on the same models, so their difference shrinks relative to the standard
/// error, by about `sqrt(8)` for each eightfold sample. The test asks for a
/// factor of 2 each time and a difference below a tenth of the standard error
/// at the largest size.
#[test]
fn targeted_estimate_approaches_the_augmented_one() {
    let mut ratios = Vec::new();
    for (seed, n) in [(41, 1000), (42, 8000), (43, 64000)] {
        let d = simulate(seed, n);
        let opts = TmleOptions::default();
        let t = tmle_glm(
            &d.input(),
            Family::Binomial,
            OutcomeModel::ByArm,
            Target::Ate,
            &opts,
        )
        .unwrap();
        let s = aipw_glm(
            &d.input(),
            Family::Binomial,
            Target::Ate,
            &AipwOptions::default(),
        )
        .unwrap();
        let ratio = (t.difference.estimate - s.difference.estimate).abs() / t.difference.se;
        println!(
            "n {n}: tmle {} aipw {} se {} ratio {ratio:e}",
            t.difference.estimate, s.difference.estimate, t.difference.se
        );
        ratios.push(ratio);
    }
    assert!(
        ratios[1] < ratios[0] / 2.0 && ratios[2] < ratios[1] / 2.0,
        "{ratios:?}"
    );
    assert!(ratios[2] < 0.1, "{ratios:?}");
}

#[test]
fn options_that_change_nothing_change_no_bits() {
    let d = simulate(33, 500);
    let base = TmleOptions::default();
    let lo = d.y.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = d.y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let known = TmleOptions {
        range: OutcomeRange::Known {
            lower: lo,
            upper: hi,
        },
        ..base
    };
    for target in [Target::Ate, Target::Att, Target::Atc] {
        let r1 = tmle_glm(
            &d.input(),
            Family::Binomial,
            OutcomeModel::Pooled,
            target,
            &base,
        )
        .unwrap();
        let r2 = tmle_glm(
            &d.input(),
            Family::Binomial,
            OutcomeModel::Pooled,
            target,
            &known,
        )
        .unwrap();
        assert_eq!(r1.influence, r2.influence);
        assert_eq!(
            r1.difference.estimate.to_bits(),
            r2.difference.estimate.to_bits()
        );
    }
    let open = TmleOptions {
        restrict_support: false,
        ..base
    };
    let r = tmle_glm(
        &d.input(),
        Family::Binomial,
        OutcomeModel::Pooled,
        Target::Att,
        &open,
    )
    .unwrap();
    assert_eq!(r.n_used, d.y.len());
    assert!(r.targeted1.iter().all(Option::is_some));
}

#[test]
fn refusals() {
    let d = simulate(34, 300);
    let m = vec![0.5; 300];
    let e = vec![0.5; 300];
    let input = |e: &[f64]| -> Result<(), String> {
        let input = AipwInput {
            outcome: &d.y,
            treatment: &d.a,
            mu1: &m,
            mu0: &m,
            propensity: e,
            weight: None,
            cluster: None,
        };
        tmle(&input, Target::Ate, &TmleOptions::default())
            .map(|_| ())
            .map_err(|e| e.to_string())
    };
    assert!(input(&e).is_ok());
    let mut bad = e.clone();
    bad[3] = 0.0;
    assert!(input(&bad).is_err());
    bad[3] = 1.0;
    assert!(input(&bad).is_err());
    assert!(input(&e[1..]).is_err());
    let run = |opts: TmleOptions| {
        tmle_glm(
            &d.input(),
            Family::Binomial,
            OutcomeModel::Pooled,
            Target::Ate,
            &opts,
        )
    };
    let base = TmleOptions::default();
    for opts in [
        TmleOptions {
            range: OutcomeRange::Known {
                lower: 0.0,
                upper: 0.5,
            },
            ..base
        },
        TmleOptions {
            range: OutcomeRange::Known {
                lower: 1.0,
                upper: 0.0,
            },
            ..base
        },
        TmleOptions {
            propensity_bound: PropensityBound::At(0.5),
            ..base
        },
        TmleOptions {
            outcome_bound: 0.0,
            ..base
        },
        TmleOptions { step: 0.0, ..base },
        TmleOptions { alpha: 1.0, ..base },
    ] {
        assert!(run(opts).is_err(), "{opts:?}");
    }
    assert!(
        tmle_glm(
            &d.input(),
            Family::Poisson,
            OutcomeModel::Pooled,
            Target::Ate,
            &base
        )
        .is_err()
    );
    let constant = vec![1.0; 300];
    let flat = GlmInput {
        outcome: &constant,
        ..d.input()
    };
    assert!(
        tmle_glm(
            &flat,
            Family::Binomial,
            OutcomeModel::Pooled,
            Target::Ate,
            &base
        )
        .is_err()
    );
}
