//! Clone-censor-weight, `epistat_survival::weighting::clone`, checked against a
//! base-R implementation of Maringe et al. (2020) in `data/clone_censor_weight.R`,
//! with properties of the cloning, and the bootstrap over persons.
//!
//! # The tolerances
//!
//! The cloning is exact: the rows, the deviations and the outcomes are compared
//! for equality. The logistic models of the deviation are fitted by IRLS from the
//! same start and with the same stopping rule as `glm`, so a coefficient differs
//! by `8 kappa eps` relative to `max(|c|, 1)` with `kappa` at most `10^3` for the
//! covariates here, a linear predictor by less than `1e-11`, a probability by a
//! quarter of that and a factor `(1 - p_n) / (1 - p_d)` of the weight, with
//! `p_d` at most 0.9, by 10 times that relative. The product over at most ten
//! rows is within `1e-9`, the outcome coefficients and covariance within `1e-8`
//! and the cumulative incidence, with a derivative of at most 1 in a coefficient,
//! within `1e-8`. Observed agreement is `3e-13`.

mod ccw_run;

use epistat_linalg::Mat;
use epistat_survival::weighting::clone::{
    Adherence, CloneInput, DeviationModel, Strategy, bootstrap_persons, clone_censor,
    deviation_weights,
};

fn worst(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn the_clones_the_weights_and_the_outcome_model_agree_with_the_reference() {
    let r = ccw_run::load();
    let d = ccw_run::data(&r);
    let c = ccw_run::cloned(&r, &d);
    let k = &r.clones;
    assert_eq!(c.source, k.row);
    assert_eq!(c.arm, k.arm);
    let as_u8 = |v: &[bool]| v.iter().map(|&b| u8::from(b)).collect::<Vec<_>>();
    assert_eq!(as_u8(&c.deviation), k.dev);
    assert_eq!(as_u8(&c.at_risk), k.risk);
    assert_eq!(as_u8(&c.outcome), k.outcome);
    let o = ccw_run::analyse(&r, &c, None);
    assert!(worst(&o.probability, &k.pd) < 1e-10);
    assert!(worst(&o.weight, &k.weight) < 1e-9);
    assert!(worst(&o.msm.coefficients, &r.msm_coef) < 1e-8);
    assert!(worst(&o.msm.covariance, &r.msm_vcov) < 1e-8);
    assert!(worst(&o.risk[0], &r.cum_inc_0) < 1e-8);
    assert!(worst(&o.risk[1], &r.cum_inc_1) < 1e-8);
}

#[test]
fn the_rules_of_the_strategies_on_a_small_data_set() {
    // Person 0: never treated, outcome in period 3. Person 1: treated from
    // period 1. Person 2: treated in period 0 and stops in period 2. Person 3:
    // treated in period 0 only, with no grace period.
    let id = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3];
    let period = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1];
    let treatment = [
        false, false, false, false, false, true, true, true, true, true, false, false, true, false,
    ]
    .map(|b: bool| b);
    let outcome = [
        false, false, false, true, false, false, false, false, false, false, false, false, false,
        false,
    ];
    let input = CloneInput {
        id: &id,
        period: &period,
        treatment: &treatment,
        outcome: &outcome,
        covariates: &[],
    };
    let strategies = [
        Strategy::InitiateWithin {
            grace: 2,
            adherence: Adherence::Continuous,
        },
        Strategy::NeverInitiate { until: None },
    ];
    let c = clone_censor(&input, &strategies).unwrap();
    // The rows of each clone: (arm, source rows, deviation of the last row).
    let mut by: Vec<(usize, usize, Vec<usize>, bool)> = Vec::new();
    for i in 0..c.len() {
        if by
            .last()
            .is_none_or(|b| b.0 != c.person[i] as usize || b.1 != c.arm[i])
        {
            by.push((c.person[i] as usize, c.arm[i], Vec::new(), false));
        }
        let last = by.last_mut().unwrap();
        last.2.push(c.source[i]);
        last.3 = c.deviation[i];
    }
    let expect = [
        // person 0: not started at the end of the grace period (period 2): the
        // clone is censored in row 2; the never clone follows to the outcome.
        (0, 0, vec![0, 1, 2], true),
        (0, 1, vec![0, 1, 2, 3], false),
        // person 1: starts in period 1, stays on it; the never clone deviates in
        // period 1.
        (1, 0, vec![4, 5, 6, 7], false),
        (1, 1, vec![4, 5], true),
        // person 2: starts in period 0 and stops in period 2: deviation in row 2
        // for the initiate clone; the never clone deviates in period 0.
        (2, 0, vec![8, 9, 10], true),
        (2, 1, vec![8], true),
        // person 3: starts in period 0 and stops in period 1.
        (3, 0, vec![12, 13], true),
        (3, 1, vec![12], true),
    ];
    assert_eq!(by, expect);
    assert_eq!(c.n_persons, 4);
    assert_eq!(c.n_without_follow_up, 0);
    // The deviation row carries no outcome and the never clone of person 0 keeps
    // its outcome.
    assert_eq!(c.outcome.iter().filter(|&&o| o).count(), 1);
    // A grace period beyond the follow-up is an error.
    let long = [Strategy::InitiateWithin {
        grace: 9,
        adherence: Adherence::None,
    }];
    assert!(clone_censor(&input, &long).is_err());
    // Rows that are not in order are an error.
    let bad_period = [0, 2, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1];
    assert!(
        clone_censor(
            &CloneInput {
                period: &bad_period,
                ..input
            },
            &strategies
        )
        .is_err()
    );
}

#[test]
fn a_numerator_equal_to_the_denominator_gives_weights_of_one() {
    let r = ccw_run::load();
    let d = ccw_run::data(&r);
    let c = ccw_run::cloned(&r, &d);
    let m = c.len();
    let den = Mat::from_fn(m, 3, |i, j| {
        let s = c.source[i];
        [1.0, r.input.l[s], r.input.period[s] as f64][j]
    });
    let model = DeviationModel {
        denominator: den.as_ref(),
        numerator: Some(den.as_ref()),
    };
    let w = deviation_weights(&c, &[model, model], None).unwrap();
    assert!(w.weight.iter().all(|&v| (v - 1.0).abs() < 1e-12));
}

#[test]
fn a_bootstrap_of_counts_of_one_is_the_data_and_depends_on_the_seed_only() {
    let r = ccw_run::load();
    let d = ccw_run::data(&r);
    let c = ccw_run::cloned(&r, &d);
    let ones = vec![1_u32; c.n_persons];
    let plain = ccw_run::analyse(&r, &c, None);
    let counted = ccw_run::analyse(&r, &c, Some(&ones));
    assert!(worst(&plain.weight, &counted.weight) < 1e-12);
    assert!(worst(&plain.msm.coefficients, &counted.msm.coefficients) < 1e-10);
    let seeds = [11_u64, 12, 13];
    let run = |t: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .unwrap();
        pool.install(|| {
            bootstrap_persons(c.n_persons, &seeds, |counts| {
                Ok(ccw_run::analyse(&r, &c, Some(counts)).flatten())
            })
            .into_iter()
            .map(Result::unwrap)
            .collect::<Vec<_>>()
        })
    };
    let (one, three) = (run(1), run(3));
    for (a, b) in one.iter().zip(&three) {
        assert!(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()));
    }
    // The replicates differ from each other.
    assert_ne!(one[0], one[1]);
}
