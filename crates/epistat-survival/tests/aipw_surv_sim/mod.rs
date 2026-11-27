//! A simulated cohort with confounded treatment and covariate-dependent
//! censoring, for the bit checks of the augmented risk.

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::aipw::{
    AipwSurv, AipwSurvOptions, AipwSurvSpec, Censoring, Estimator, aipw_cox,
};
use epistat_survival::cox::Ties;
use epistat_survival::standardise::Counterfactual;

pub struct Simulated {
    pub time: Vec<f64>,
    pub status: Vec<i32>,
    pub treated: Vec<bool>,
    pub outcome_x: Mat<f64>,
    pub propensity_x: Mat<f64>,
    pub group: Vec<u32>,
}

pub fn simulate(n: usize, seed: u64) -> Simulated {
    let mut rng = Stream::seeded(seed);
    let (mut time, mut status, mut treated) = (Vec::new(), Vec::new(), Vec::new());
    let mut cols: Vec<[f64; 2]> = Vec::new();
    for _ in 0..n {
        let x1 = rng.normal();
        let x2 = f64::from(u8::from(rng.uniform() < 0.4));
        let p = 1.0 / (1.0 + (-(0.5 * x1 - 0.4 * x2)).exp());
        let a = rng.uniform() < p;
        let a_f = f64::from(u8::from(a));
        let te = -rng.uniform().ln() / (0.12 * (0.5 * x1 - 0.4 * x2 - 0.6 * a_f).exp());
        let tc = (-rng.uniform().ln() / (0.06 * (0.3 * x1 + 0.3 * a_f).exp())).min(12.0);
        // Rounded to a quarter, so that there are ties.
        let t = (te.min(tc) * 4.0).ceil() / 4.0;
        time.push(t);
        status.push(i32::from(te <= tc));
        treated.push(a);
        cols.push([x1, x2]);
    }
    let outcome_x = Mat::from_fn(n, 3, |i, j| {
        [f64::from(u8::from(treated[i])), cols[i][0], cols[i][1]][j]
    });
    let propensity_x = Mat::from_fn(n, 3, |i, j| [1.0, cols[i][0], cols[i][1]][j]);
    let group = treated.iter().map(|&a| u32::from(a)).collect();
    Simulated {
        time,
        status,
        treated,
        outcome_x,
        propensity_x,
        group,
    }
}

pub fn run(s: &Simulated, estimator: Estimator, km: bool, times: &[f64]) -> AipwSurv {
    let data = SurvData::builder(&s.time, &s.status).build().unwrap();
    let set0 = [(0usize, 0.0)];
    let set1 = [(0usize, 1.0)];
    let censoring = if km {
        Censoring::KaplanMeier {
            groups: Some(&s.group),
        }
    } else {
        Censoring::Cox {
            x: s.outcome_x.as_ref(),
        }
    };
    let spec = AipwSurvSpec {
        treatment: &s.treated,
        outcome_x: s.outcome_x.as_ref(),
        levels: [
            Counterfactual::Columns(&set0),
            Counterfactual::Columns(&set1),
        ],
        propensity_x: s.propensity_x.as_ref(),
        censoring,
    };
    aipw_cox(
        &data,
        &spec,
        times,
        &AipwSurvOptions {
            estimator,
            ties: Ties::Breslow,
            ..AipwSurvOptions::default()
        },
    )
    .unwrap()
}

pub fn flatten(r: &AipwSurv) -> Vec<f64> {
    let mut v = Vec::new();
    for x in [&r.risks.risk]
        .into_iter()
        .chain(r.risks.se.as_ref())
        .chain(r.risks.covariance.as_ref())
    {
        v.extend(x.iter().map(|o| o.unwrap_or(f64::NAN)));
    }
    v.extend(&r.influence[0]);
    v.extend(&r.influence[1]);
    v
}
