//! `clone_censor` with a window of starting, a minimum duration and dynamic
//! strategies, and `deviation_weights` on their clones, against the cloning
//! written out in base R from the definitions, `data/clone_strategies.R`.
//!
//! # The tolerances
//!
//! The clone rows are compared exactly. The deviation models are fitted by
//! IRLS from the same start with the same stopping rule as `glm`, so a linear
//! predictor differs by `eta = 8 kappa eps sum_j |z_j| max(|c_j|, 1)`, with the
//! condition number `kappa` below `10^3` and a design of four columns of size
//! at most 10 here, so `eta < 2e-10`. A probability differs by `eta / 4`, a
//! factor `1 / (1 - p)` by `eta` relative (`p (1 - p) / (1 - p) <= 1`), and the
//! weight, a product of at most ten factors, by `10 eta = 2e-9` relative,
//! checked to `1e-8`.

use epistat_linalg::Mat;
use epistat_survival::weighting::clone::{
    Adherence, CloneInput, DeviationModel, Strategy, Trigger, clone_censor, deviation_weights,
};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    id: Vec<u64>,
    period: Vec<i64>,
    treatment: Vec<u8>,
    outcome: Vec<u8>,
    f: Vec<f64>,
    l: Vec<f64>,
}

#[derive(Deserialize)]
struct Clones {
    row: Vec<usize>,
    arm: Vec<usize>,
    dev: Vec<u8>,
    risk: Vec<u8>,
    outcome: Vec<u8>,
    pd: Vec<f64>,
    weight: Vec<f64>,
}

#[derive(Deserialize)]
struct Reference {
    input: Input,
    clones: Clones,
}

const STRATEGIES: [Strategy; 5] = [
    Strategy::InitiateBetween {
        first: 2,
        last: 4,
        adherence: Adherence::Continuous,
    },
    Strategy::InitiateWithin {
        grace: 3,
        adherence: Adherence::For(3),
    },
    Strategy::InitiateWhen {
        trigger: Trigger::AtLeast {
            covariate: 0,
            threshold: 1.0,
        },
        grace: 1,
        adherence: Adherence::Continuous,
    },
    Strategy::InitiateWhen {
        trigger: Trigger::Below {
            covariate: 1,
            threshold: -0.3,
        },
        grace: 2,
        adherence: Adherence::None,
    },
    Strategy::NeverInitiate { until: None },
];

#[test]
fn the_clones_and_their_weights_agree_with_r() {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/clone_strategies.json"
    ));
    let d = &r.input;
    let treatment: Vec<bool> = d.treatment.iter().map(|&v| v == 1).collect();
    let outcome: Vec<bool> = d.outcome.iter().map(|&v| v == 1).collect();
    let covariates: [&[f64]; 2] = [&d.f, &d.l];
    let c = clone_censor(
        &CloneInput {
            id: &d.id,
            period: &d.period,
            treatment: &treatment,
            outcome: &outcome,
            covariates: &covariates,
        },
        &STRATEGIES,
    )
    .expect("clones");
    let w = &r.clones;
    assert_eq!(c.source, w.row);
    assert_eq!(c.arm, w.arm);
    let flags = |v: &[bool]| v.iter().map(|&b| u8::from(b)).collect::<Vec<_>>();
    assert_eq!(flags(&c.deviation), w.dev);
    assert_eq!(flags(&c.at_risk), w.risk);
    assert_eq!(flags(&c.outcome), w.outcome);

    let m = c.len();
    #[expect(clippy::cast_precision_loss, reason = "periods are small integers")]
    let den = Mat::from_fn(m, 4, |i, j| {
        let s = c.source[i];
        [1.0, d.l[s], d.f[s], d.period[s] as f64][j]
    });
    let model = DeviationModel {
        denominator: den.as_ref().into(),
        numerator: None,
    };
    let fit = deviation_weights(&c, &[model; 5], None).expect("weights");
    for i in 0..m {
        assert!(
            (fit.probability[i] - w.pd[i]).abs() < 1e-10,
            "probability of clone row {i}"
        );
        assert!(
            (fit.weight[i] - w.weight[i]).abs() / w.weight[i] < 1e-8,
            "weight of clone row {i}: {} against {}",
            fit.weight[i],
            w.weight[i]
        );
    }
}

#[test]
fn strategies_that_cannot_be_followed_are_errors() {
    let id = [0, 0, 0];
    let period = [0, 1, 2];
    let flags = [false, false, false];
    let x = [0.0, 1.0, f64::NAN];
    let input = CloneInput {
        id: &id,
        period: &period,
        treatment: &flags,
        outcome: &flags,
        covariates: &[],
    };
    let window = Strategy::InitiateBetween {
        first: 2,
        last: 1,
        adherence: Adherence::None,
    };
    assert!(clone_censor(&input, &[window]).is_err());
    let none = Strategy::InitiateWithin {
        grace: 1,
        adherence: Adherence::For(0),
    };
    assert!(clone_censor(&input, &[none]).is_err());
    let when = Strategy::InitiateWhen {
        trigger: Trigger::AtLeast {
            covariate: 0,
            threshold: 1.0,
        },
        grace: 0,
        adherence: Adherence::None,
    };
    // No covariate, and a covariate that is not finite.
    assert!(clone_censor(&input, &[when]).is_err());
    let covariates: [&[f64]; 1] = [&x];
    assert!(
        clone_censor(
            &CloneInput {
                covariates: &covariates,
                ..input
            },
            &[when]
        )
        .is_err()
    );
}

/// `InitiateBetween` from period 0 is `InitiateWithin`, and `For(1)` is no
/// constraint after the start.
#[test]
fn the_general_strategies_contain_the_simple_ones() {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/clone_strategies.json"
    ));
    let d = &r.input;
    let treatment: Vec<bool> = d.treatment.iter().map(|&v| v == 1).collect();
    let outcome: Vec<bool> = d.outcome.iter().map(|&v| v == 1).collect();
    let input = CloneInput {
        id: &d.id,
        period: &d.period,
        treatment: &treatment,
        outcome: &outcome,
        covariates: &[],
    };
    for adherence in [Adherence::Continuous, Adherence::None] {
        let within = clone_censor(
            &input,
            &[Strategy::InitiateWithin {
                grace: 3,
                adherence,
            }],
        )
        .expect("clones");
        let between = clone_censor(
            &input,
            &[Strategy::InitiateBetween {
                first: 0,
                last: 3,
                adherence,
            }],
        )
        .expect("clones");
        assert_eq!(within, between);
    }
    let none = clone_censor(
        &input,
        &[Strategy::InitiateWithin {
            grace: 3,
            adherence: Adherence::None,
        }],
    )
    .expect("clones");
    let one = clone_censor(
        &input,
        &[Strategy::InitiateWithin {
            grace: 3,
            adherence: Adherence::For(1),
        }],
    )
    .expect("clones");
    assert_eq!(none, one);
}
