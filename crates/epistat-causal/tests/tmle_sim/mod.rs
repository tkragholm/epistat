//! The cases of the thread and golden tests of the targeted estimators, on the
//! simulated data of `sim`.

use epistat_causal::Estimand;
use epistat_causal::Influence;
use epistat_causal::aipw::AipwInput;
use epistat_causal::tmle::{
    OutcomeModel, PropensityBound, Targeting, Tmle, TmleGlmOptions, TmleOptions, tmle, tmle_glm,
};
use epistat_causal::{CaseWeights, ProbabilityBounds};
use epistat_glm::Family;

use crate::sim::{Simulated, push_estimate};

/// `(name, target, fitted, model)`.
pub type Case = (&'static str, Estimand, bool, OutcomeModel);

pub const CASES: [Case; 5] = [
    ("glm ate pooled", Estimand::Ate, true, OutcomeModel::Pooled),
    ("glm att pooled", Estimand::Att, true, OutcomeModel::Pooled),
    ("glm atc by arm", Estimand::Atc, true, OutcomeModel::ByArm),
    ("core ate", Estimand::Ate, false, OutcomeModel::Pooled),
    ("core att", Estimand::Att, false, OutcomeModel::Pooled),
];

pub fn run(s: &Simulated, case: Case) -> Vec<f64> {
    let (_, target, fitted, model) = case;
    let opts = TmleOptions {
        propensity_bound: PropensityBound::Tmle,
        ..TmleOptions::default()
    };
    let r = if fitted {
        tmle_glm(
            &s.input(),
            Family::Binomial,
            &TmleGlmOptions {
                tmle: TmleOptions {
                    estimand: target,

                    ..opts
                },
                outcome_model: model,
            },
        )
        .unwrap()
    } else {
        let m = s.models();
        tmle(
            &AipwInput {
                outcome: &s.y,
                treatment: &s.a,
                mu1: &m.0,
                mu0: &m.1,
                propensity: &m.2,
                weight: Some(CaseWeights::new(&s.weight).expect("weights")),
                cluster: Some(&s.cluster),
            },
            &TmleOptions {
                estimand: target,
                ..opts
            },
        )
        .unwrap()
    };
    flatten(&r)
}

pub fn flatten(r: &Tmle) -> Vec<f64> {
    let mut out = vec![
        r.n_trimmed as f64,
        r.n_used as f64,
        r.propensity_bound.map_or(0.0, |b| b.lower()),
        r.range.0,
        r.range.1,
        r.score,
    ];
    match r.targeting {
        Targeting::Fluctuation {
            epsilon,
            iterations,
        } => out.extend([epsilon[0], epsilon[1], iterations as f64]),
        Targeting::OneStep {
            steps,
            recalibrated,
        } => out.extend([steps as f64, f64::from(u8::from(recalibrated))]),
        _ => unreachable!("the targetings of this version"),
    }
    let c = &r.contrast;
    for e in [c.mean1, c.mean0, c.difference].into_iter().chain(c.ratio) {
        push_estimate(&mut out, &e);
    }
    out.extend(by_person(&r.influence, r.targeted1.len()));
    for t in r.targeted1.iter().chain(&r.targeted0) {
        out.push(t.unwrap_or(f64::NAN));
    }
    out.extend([r.initial.mean1, r.initial.mean0, r.initial.difference]);
    out.extend(r.influence.column(0));
    out.extend(r.influence.column(1));
    out
}

/// The influence function of the difference of `influence` for each of `n`
/// persons, 0 for persons it does not cover.
pub fn by_person(influence: &Influence, n: usize) -> Vec<f64> {
    let difference = influence.combine(&[1.0, -1.0]).unwrap();
    let mut out = vec![0.0; n];
    for (&unit, &v) in influence.keys().iter().zip(difference.column(0)) {
        out[unit as usize] = v;
    }
    out
}

/// The survival estimator on simulated follow-up of `s`'s persons in five
/// intervals, with hazards and propensity from simple formulas.
pub fn run_survival(s: &Simulated, horizon: usize) -> Vec<f64> {
    use epistat_causal::tmle::survival::{SurvivalInput, SurvivalOptions, tmle_survival};
    use epistat_linalg::Mat;
    use epistat_rng::Stream;
    let n = s.y.len();
    let k = 5;
    let expit = |z: f64| 1.0 / (1.0 + (-z).exp());
    let hazard = |arm: f64| {
        Mat::from_fn(n, k, |i, j| {
            expit(-2.0 + 0.1 * j as f64 + 0.6 * s.x[(i, 1)] + 0.4 * s.x[(i, 2)] - 0.5 * arm)
        })
    };
    let censoring =
        |arm: f64| Mat::from_fn(n, k, |i, _| expit(-2.5 + 0.3 * s.x[(i, 1)] + 0.3 * arm));
    let (h1, h0, c1, c0) = (hazard(1.0), hazard(0.0), censoring(1.0), censoring(0.0));
    let g: Vec<f64> = (0..n)
        .map(|i| expit(0.4 * s.x[(i, 1)] - 0.5 * s.x[(i, 2)] - 0.1))
        .collect();
    let mut rng = Stream::seeded(25);
    let (mut time, mut event) = (vec![k; n], vec![false; n]);
    for i in 0..n {
        let (h, c) = if s.a[i] { (&h1, &c1) } else { (&h0, &c0) };
        for j in 0..k {
            if rng.uniform() < h[(i, j)] {
                (time[i], event[i]) = (j + 1, true);
                break;
            }
            if rng.uniform() < c[(i, j)] {
                time[i] = j + 1;
                break;
            }
        }
    }
    let r = tmle_survival(
        &SurvivalInput {
            time: &time,
            event: &event,
            treatment: &s.a,
            hazard1: h1.as_ref(),
            hazard0: h0.as_ref(),
            censoring1: c1.as_ref(),
            censoring0: c0.as_ref(),
            propensity: &g,
            weight: Some(CaseWeights::new(&s.weight).expect("weights")),
            cluster: Some(&s.cluster),
        },
        &SurvivalOptions {
            propensity_bound: PropensityBound::At(
                ProbabilityBounds::new(0.05, 1.0 - 0.05).unwrap(),
            ),
            ..SurvivalOptions::new(horizon)
        },
    )
    .unwrap();
    let mut out = vec![
        r.iterations[0] as f64,
        r.iterations[1] as f64,
        r.score[0],
        r.score[1],
        r.propensity_bound.unwrap_or(0.0),
        r.n_trimmed as f64,
    ];
    out.extend(r.epsilon.iter().flatten());
    let c = &r.contrast;
    for e in [c.mean1, c.mean0, c.difference].into_iter().chain(c.ratio) {
        push_estimate(&mut out, &e);
    }
    out.extend(by_person(&r.influence, n));
    out.extend([r.initial.mean1, r.initial.mean0, r.initial.difference]);
    out.extend(r.influence.column(0));
    out.extend(r.influence.column(1));
    out
}
