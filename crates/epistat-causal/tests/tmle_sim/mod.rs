//! The cases of the thread and golden tests of the targeted estimators, on the
//! simulated data of `sim`.

use epistat_causal::aipw::{AipwInput, Target};
use epistat_causal::tmle::{
    OutcomeModel, PropensityBound, Targeting, Tmle, TmleOptions, tmle, tmle_glm,
};
use epistat_glm::Family;

use crate::sim::Simulated;

/// `(name, target, fitted, model)`.
pub type Case = (&'static str, Target, bool, OutcomeModel);

pub const CASES: [Case; 5] = [
    ("glm ate pooled", Target::Ate, true, OutcomeModel::Pooled),
    ("glm att pooled", Target::Att, true, OutcomeModel::Pooled),
    ("glm atc by arm", Target::Atc, true, OutcomeModel::ByArm),
    ("core ate", Target::Ate, false, OutcomeModel::Pooled),
    ("core att", Target::Att, false, OutcomeModel::Pooled),
];

pub fn run(s: &Simulated, case: Case) -> Vec<f64> {
    let (_, target, fitted, model) = case;
    let opts = TmleOptions {
        propensity_bound: PropensityBound::Tmle,
        ..TmleOptions::default()
    };
    let r = if fitted {
        tmle_glm(&s.input(), Family::Binomial, model, target, &opts).unwrap()
    } else {
        let m = s.models();
        tmle(
            &AipwInput {
                outcome: &s.y,
                treatment: &s.a,
                mu1: &m.0,
                mu0: &m.1,
                propensity: &m.2,
                weight: Some(&s.weight),
                cluster: Some(&s.cluster),
            },
            target,
            &opts,
        )
        .unwrap()
    };
    flatten(&r)
}

pub fn flatten(r: &Tmle) -> Vec<f64> {
    let mut out = vec![
        r.n_trimmed as f64,
        r.n_used as f64,
        r.propensity_bound,
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
    }
    for e in [r.mean1, r.mean0, r.difference].into_iter().chain(r.ratio) {
        out.extend([
            e.estimate,
            e.se,
            e.lower,
            e.upper,
            e.p_value.unwrap_or(f64::NAN),
        ]);
    }
    out.extend(&r.influence);
    for t in r.targeted1.iter().chain(&r.targeted0) {
        out.push(t.unwrap_or(f64::NAN));
    }
    out
}
