//! The person-period data and the references of `data/trial_msm.R`, and the
//! analysis of Rust run on them.

#![allow(dead_code)]
#![expect(
    clippy::cast_precision_loss,
    reason = "periods and times are small integers"
)]

use epistat_linalg::Mat;
use epistat_survival::weighting::trials::WeightRows;
use epistat_survival::weighting::trials::{
    CumulativeIncidence, Fit, MsmInput, PersonPeriod, TrialRows, WeightModel, censoring_weights,
    cumulative_incidence, fit_msm, switch_weights, trial_weights,
};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct Rows {
    pub id: Vec<u64>,
    pub period: Vec<i64>,
    pub treatment: Vec<f64>,
    pub am_1: Vec<f64>,
    #[serde(default)]
    pub censored: Vec<f64>,
    pub x1: Vec<f64>,
    pub x2: Vec<f64>,
    pub age_s: Vec<f64>,
    pub p0_d: Vec<Option<f64>>,
    pub p0_n: Vec<Option<f64>>,
    pub p1_d: Vec<Option<f64>>,
    pub p1_n: Vec<Option<f64>>,
    #[serde(default, rename = "pC_d")]
    pub pc_d: Vec<f64>,
    #[serde(default, rename = "pC_n")]
    pub pc_n: Vec<f64>,
    pub wt: Vec<f64>,
}

#[derive(Deserialize)]
pub struct Trials {
    pub id: Vec<u64>,
    pub trial_period: Vec<i64>,
    pub followup_time: Vec<i64>,
    pub outcome: Vec<f64>,
    pub weight: Vec<f64>,
    pub assigned: Vec<f64>,
    pub x1: Vec<f64>,
    pub x2: Vec<f64>,
    pub age_s: Vec<f64>,
}

#[derive(Deserialize)]
pub struct Analysis {
    pub use_censor: bool,
    pub rows: Rows,
    pub trials: Trials,
    pub msm_coef: Vec<f64>,
    pub msm_vcov: Vec<f64>,
    pub predict_times: Vec<i64>,
    pub cum_inc_0: Vec<f64>,
    pub cum_inc_1: Vec<f64>,
}

#[derive(Deserialize)]
struct Reference {
    analyses: Vec<Analysis>,
}

pub fn load() -> Vec<Analysis> {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/trial_msm.json"
    ));
    r.analyses
}

/// What Rust gives for one analysis.
pub struct Outputs {
    pub switch_denominator: Vec<f64>,
    pub switch_numerator: Vec<f64>,
    pub censor_denominator: Vec<f64>,
    pub censor_numerator: Vec<f64>,
    pub row_weight: Vec<f64>,
    pub trial_weight: Vec<f64>,
    pub coefficients: Vec<f64>,
    pub covariance: Vec<f64>,
    pub risk: [Vec<f64>; 2],
    pub std_err: [Vec<f64>; 2],
    pub msm: epistat_survival::weighting::trials::Msm,
    pub designs: [Vec<Mat<f64>>; 2],
}

pub fn run(a: &Analysis) -> Outputs {
    let w = &a.rows;
    let n = w.id.len();
    let treatment: Vec<bool> = w.treatment.iter().map(|&v| v > 0.5).collect();
    let previous: Vec<bool> = w.am_1.iter().map(|&v| v > 0.5).collect();
    let data = PersonPeriod {
        id: &w.id,
        period: &w.period,
        treatment: &treatment,
        previous: &previous,
    };
    let den = Mat::from_fn(n, 4, |i, j| [1.0, w.x1[i], w.x2[i], w.age_s[i]][j]);
    let num = Mat::from_fn(n, 2, |i, j| [1.0, w.x1[i]][j]);
    let model = WeightModel {
        numerator: num.as_ref(),
        denominator: den.as_ref(),
    };
    let sw = switch_weights(&data, &model, &WeightRows::default()).unwrap();
    let mut row_weight = sw.weight.clone();
    let (mut censor_denominator, mut censor_numerator) = (Vec::new(), Vec::new());
    if a.use_censor {
        let censored: Vec<bool> = w.censored.iter().map(|&v| v > 0.5).collect();
        let cw = censoring_weights(
            &data,
            &censored,
            &model,
            Fit::ByPreviousTreatment,
            &WeightRows::default(),
        )
        .unwrap();
        for (v, c) in row_weight.iter_mut().zip(&cw.weight) {
            *v *= c;
        }
        censor_denominator = cw.denominator_probability;
        censor_numerator = cw.numerator_probability;
    }
    let t = &a.trials;
    let m = t.id.len();
    let trial_weight = trial_weights(
        &data,
        &w.wt,
        &TrialRows {
            id: &t.id,
            trial_period: &t.trial_period,
            follow_up: &t.followup_time,
        },
    )
    .unwrap();
    let design = Mat::from_fn(m, 8, |i, j| {
        let f = t.followup_time[i] as f64;
        [
            1.0,
            t.assigned[i],
            t.trial_period[i] as f64,
            f,
            f * f,
            t.x1[i],
            t.x2[i],
            t.age_s[i],
        ][j]
    });
    let outcome: Vec<bool> = t.outcome.iter().map(|&v| v > 0.5).collect();
    let msm = fit_msm(&MsmInput {
        design: design.as_ref(),
        outcome: &outcome,
        weight: &t.weight,
        person: &t.id,
    })
    .unwrap();
    let base: Vec<usize> = (0..m)
        .filter(|&i| t.trial_period[i] == 0 && t.followup_time[i] == 0)
        .collect();
    let designs_for = |assigned: f64| -> Vec<Mat<f64>> {
        a.predict_times
            .iter()
            .map(|&f| {
                Mat::from_fn(base.len(), 8, |k, j| {
                    let i = base[k];
                    let f = f as f64;
                    [1.0, assigned, 0.0, f, f * f, t.x1[i], t.x2[i], t.age_s[i]][j]
                })
            })
            .collect()
    };
    let designs = [designs_for(0.0), designs_for(1.0)];
    let ci: Vec<CumulativeIncidence> = designs
        .iter()
        .map(|d| {
            let refs: Vec<_> = d.iter().map(Mat::as_ref).collect();
            cumulative_incidence(&msm, &refs).unwrap()
        })
        .collect();
    Outputs {
        switch_denominator: sw.denominator_probability,
        switch_numerator: sw.numerator_probability,
        censor_denominator,
        censor_numerator,
        row_weight,
        trial_weight,
        coefficients: msm.coefficients.clone(),
        covariance: msm
            .covariance
            .as_slice()
            .iter()
            .map(|x| x.unwrap())
            .collect(),
        risk: [ci[0].risk.clone(), ci[1].risk.clone()],
        std_err: [ci[0].se.clone(), ci[1].se.clone()],
        msm,
        designs,
    }
}

impl Outputs {
    pub fn flatten(&self) -> Vec<f64> {
        let mut v = Vec::new();
        for x in [
            &self.switch_denominator,
            &self.switch_numerator,
            &self.censor_denominator,
            &self.censor_numerator,
            &self.row_weight,
            &self.trial_weight,
            &self.coefficients,
            &self.covariance,
            &self.risk[0],
            &self.risk[1],
            &self.std_err[0],
            &self.std_err[1],
        ] {
            v.extend(x);
        }
        v
    }
}
