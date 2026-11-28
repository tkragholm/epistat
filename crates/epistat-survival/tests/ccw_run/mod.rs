//! The data and reference of `data/clone_censor_weight.R`, and the analysis of
//! Rust run on it.

#![allow(dead_code)]
#![expect(clippy::cast_precision_loss, reason = "periods are small integers")]

use epistat_core::Panel;
use epistat_linalg::Mat;
use epistat_survival::weighting::clone::{
    Adherence, CloneInput, Cloned, DeviationModel, Strategy, clone_censor, deviation_weights,
};
use epistat_survival::weighting::trials::{
    Msm, MsmInput, Population, cumulative_incidence, fit_msm,
};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct Input {
    pub id: Vec<u32>,
    pub period: Vec<i64>,
    pub treatment: Vec<f64>,
    pub outcome: Vec<f64>,
    pub l: Vec<f64>,
    pub l0: Vec<f64>,
}

#[derive(Deserialize)]
pub struct Clones {
    pub row: Vec<usize>,
    pub arm: Vec<usize>,
    pub dev: Vec<u8>,
    pub risk: Vec<u8>,
    pub outcome: Vec<u8>,
    pub weight: Vec<f64>,
    pub pd: Vec<f64>,
}

#[derive(Deserialize)]
pub struct Reference {
    pub grace: u32,
    pub input: Input,
    pub clones: Clones,
    pub msm_coef: Vec<f64>,
    pub msm_vcov: Vec<f64>,
    pub times: Vec<u32>,
    pub cum_inc_0: Vec<f64>,
    pub cum_inc_1: Vec<f64>,
}

pub fn load() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/clone_censor_weight.json"
    ))
}

pub struct Data {
    pub treatment: Vec<bool>,
    pub outcome: Vec<bool>,
}

pub fn data(r: &Reference) -> Data {
    Data {
        treatment: r.input.treatment.iter().map(|&v| v > 0.5).collect(),
        outcome: r.input.outcome.iter().map(|&v| v > 0.5).collect(),
    }
}

pub fn strategies(grace: u32) -> [Strategy; 2] {
    [
        Strategy::InitiateWithin {
            grace,
            adherence: Adherence::Continuous,
        },
        Strategy::NeverInitiate { until: None },
    ]
}

pub fn cloned(r: &Reference, d: &Data) -> Cloned {
    clone_censor(
        &CloneInput {
            panel: Panel::new(&r.input.id, &r.input.period).unwrap(),
            treatment: &d.treatment,
            outcome: &d.outcome,
            covariates: &[],
        },
        &strategies(r.grace),
    )
    .unwrap()
}

pub struct Outputs {
    pub weight: Vec<f64>,
    pub probability: Vec<f64>,
    pub msm: Msm,
    pub risk: [Vec<f64>; 2],
}

/// The weights and the outcome model, with `case` the case weight of each clone
/// row, from the draws of its person (all 1 for the data itself, given as
/// `None`).
pub fn analyse(r: &Reference, c: &Cloned, case: Option<&[f64]>) -> Outputs {
    let m = c.len();
    let den = Mat::from_fn(m, 5, |i, j| {
        let s = c.row[i] as usize;
        let p = r.input.period[s] as f64;
        [1.0, r.input.l[s], r.input.l0[s], p, p * p][j]
    });
    let num = Mat::from_fn(m, 2, |i, j| {
        [1.0, r.input.period[c.row[i] as usize] as f64][j]
    });
    let model = DeviationModel {
        denominator: den.as_ref().into(),
        numerator: Some(num.as_ref().into()),
    };
    let w = deviation_weights(c, &[model, model], case).unwrap();
    let keep: Vec<usize> = (0..m).filter(|&i| !c.deviation[i]).collect();
    let design = Mat::from_fn(keep.len(), 5, |k, j| {
        let i = keep[k];
        let p = r.input.period[c.row[i] as usize] as f64;
        let arm = c.arm[i] as f64;
        [1.0, arm, p, p * p, arm * p][j]
    });
    let outcome: Vec<bool> = keep.iter().map(|&i| c.outcome[i]).collect();
    let weight: Vec<f64> = keep
        .iter()
        .map(|&i| w.weight[i] * case.map_or(1.0, |k| k[i]))
        .collect();
    let person: Vec<u32> = keep.iter().map(|&i| c.person[i]).collect();
    let msm = fit_msm(MsmInput {
        design: design.as_ref(),
        outcome: &outcome,
        weight: &weight,
        unit: &person,
    })
    .unwrap();
    let risk_of = |arm: f64| {
        let designs: Vec<Mat<f64>> = r
            .times
            .iter()
            .map(|&t| {
                let p = f64::from(t);
                Mat::from_fn(1, 5, |_, j| [1.0, arm, p, p * p, arm * p][j])
            })
            .collect();
        let refs: Vec<_> = designs.iter().map(Mat::as_ref).collect();
        cumulative_incidence(&msm, &Population::Designs(&refs))
            .unwrap()
            .risk
    };
    Outputs {
        weight: w.weight,
        probability: w.probability,
        risk: [risk_of(0.0), risk_of(1.0)],
        msm,
    }
}

impl Outputs {
    pub fn flatten(&self) -> Vec<f64> {
        let mut v = self.weight.clone();
        v.extend(&self.probability);
        v.extend(&self.msm.coefficients);
        v.extend(self.msm.covariance.as_slice().iter().map(|x| x.unwrap()));
        v.extend(&self.risk[0]);
        v.extend(&self.risk[1]);
        v
    }
}
