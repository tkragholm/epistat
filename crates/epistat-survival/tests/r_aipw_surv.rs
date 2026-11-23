//! The risk by a time with censoring, `epistat_survival::aipw`, checked against
//! `riskRegression::ate` with `estimator = "AIPTW"` and `"IPTW"`, with
//! `known.nuisance = TRUE`. See `data/aipw_surv.R`.
//!
//! # The tolerances
//!
//! Both implementations fit the Cox models by Newton-Raphson from zero with the
//! same stopping rule and the logistic model by IRLS from the same start, so the
//! coefficients differ by `8 kappa eps` relative to `max(|c|, 1)` with `kappa`
//! the condition number of the information matrix, at most `10^2` for the
//! standardised covariates here. A term of the estimate depends on the linear
//! predictors through the risk `1 - exp(-theta H)`, the weight `1 / pi` and the
//! integral, with a derivative with respect to a linear predictor of at most
//! `L = 1 + w_max` times the largest cumulative hazard, which is below 10 at
//! the times asked for. The mean of `n` such terms then differs by `8 kappa eps
//! L = 2e-12`, which is the tolerance of the estimates. A standard error is a
//! square root of sums of squares of the same terms and has the relative
//! tolerance `1e-11`.

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::aipw::{AipwSurvOptions, AipwSurvSpec, Censoring, Estimator, aipw_cox};
use epistat_survival::cox::Ties;
use epistat_survival::standardise::Counterfactual;
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Block {
    mean1: Vec<f64>,
    se1: Vec<f64>,
    mean0: Vec<f64>,
    se0: Vec<f64>,
    diff: Vec<f64>,
    diff_se: Vec<f64>,
    ratio: Vec<f64>,
}

#[derive(Deserialize)]
struct Scenario {
    name: String,
    times: Vec<f64>,
    censor: String,
    time: Vec<f64>,
    status: Vec<i32>,
    a: Vec<f64>,
    x1: Vec<f64>,
    x2: Vec<f64>,
    aiptw: Block,
    iptw: Block,
}

#[derive(Deserialize)]
struct Reference {
    scenarios: Vec<Scenario>,
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}

#[test]
fn the_risks_agree_with_ate() {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/aipw_surv.json"
    ));
    for s in &r.scenarios {
        let n = s.time.len();
        let treated: Vec<bool> = s.a.iter().map(|&a| a == 1.0).collect();
        let x = Mat::from_fn(n, 3, |i, j| [s.a[i], s.x1[i], s.x2[i]][j]);
        let z = Mat::from_fn(n, 3, |i, j| [1.0, s.x1[i], s.x2[i]][j]);
        let cx = Mat::from_fn(n, 3, |i, j| [s.a[i], s.x1[i], s.x2[i]][j]);
        let group: Vec<u32> = s.a.iter().map(|&a| a as u32).collect();
        let data = SurvData::builder(&s.time, &s.status).build().unwrap();
        let cols = [(0usize, 0.0)];
        let cols1 = [(0usize, 1.0)];
        let censoring = if s.censor == "km" {
            Censoring::KaplanMeier {
                groups: Some(&group),
            }
        } else {
            Censoring::Cox { x: cx.as_ref() }
        };
        for (est, block) in [(Estimator::Aiptw, &s.aiptw), (Estimator::Iptw, &s.iptw)] {
            let spec = AipwSurvSpec {
                treatment: &treated,
                outcome_x: x.as_ref(),
                levels: [
                    Counterfactual::Columns(&cols),
                    Counterfactual::Columns(&cols1),
                ],
                propensity_x: z.as_ref(),
                censoring,
            };
            let out = aipw_cox(
                &data,
                &spec,
                &s.times,
                &AipwSurvOptions {
                    estimator: est,
                    ties: Ties::Breslow,
                    ..AipwSurvOptions::default()
                },
            )
            .unwrap();
            let k = s.times.len();
            let c = out.risks.contrast(1, 0).unwrap();
            let mut worst = [0.0f64; 5];
            for t in 0..k {
                worst[0] = worst[0].max((out.risks.risk[k + t].unwrap() - block.mean1[t]).abs());
                worst[0] = worst[0].max((out.risks.risk[t].unwrap() - block.mean0[t]).abs());
                let se = out.risks.std_err.as_ref().unwrap();
                worst[1] = worst[1].max(rel(se[k + t].unwrap(), block.se1[t]));
                worst[1] = worst[1].max(rel(se[t].unwrap(), block.se0[t]));
                worst[2] = worst[2].max((c.difference[t].unwrap() - block.diff[t]).abs());
                worst[3] = worst[3].max(rel(
                    c.difference_inference.as_ref().unwrap().std_err[t].unwrap(),
                    block.diff_se[t],
                ));
                worst[4] = worst[4].max((c.ratio[t].unwrap() - block.ratio[t]).abs());
            }
            let label = format!("{} {:?}", s.name, est);
            assert!(worst[0] < 2e-12, "{label}: risks {}", worst[0]);
            assert!(worst[1] < 1e-11, "{label}: standard errors {}", worst[1]);
            assert!(worst[2] < 2e-12, "{label}: differences {}", worst[2]);
            assert!(worst[3] < 1e-11, "{label}: difference se {}", worst[3]);
            assert!(worst[4] < 2e-12, "{label}: ratios {}", worst[4]);
            assert_eq!(out.n_trimmed, 0);
        }
    }
}
