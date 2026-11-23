//! Samuelsen's weights, the weighted Cox fit and the absolute risk of nested
//! case-control data, checked against R 4.6.2: the inclusion probabilities
//! against the formula by an explicit loop and `multipleNCC::KMprob`, the fit
//! against `multipleNCC::wpl` and `survival::coxph`, and the baseline hazard
//! against `survival::basehaz`.

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::cox::CoxOptions;
use epistat_survival::ncc::{
    NccDesign, absolute_risk, baseline, fit_weighted, inclusion_probabilities, weights,
};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Versions {
    r: String,
    survival: String,
    #[serde(rename = "multipleNCC")]
    multiple_ncc: String,
    jsonlite: String,
}

#[derive(Deserialize)]
struct Set {
    name: String,
    m: usize,
    entry_rule: bool,
    strata: bool,
    entry: Vec<f64>,
    exit: Vec<f64>,
    case: Vec<u8>,
    grp: Vec<u32>,
    x1: Vec<f64>,
    x2: Vec<f64>,
    state: Vec<u8>,
    n_risk: Vec<usize>,
    p: Vec<f64>,
    coefficients: Vec<f64>,
    var_model: Vec<Vec<f64>>,
    var_robust: Vec<Vec<f64>>,
    loglik: f64,
    loglik_null: f64,
    iter: u32,
    kappa: f64,
    base_time: Vec<f64>,
    base_hazard: Vec<f64>,
    profiles: Vec<Vec<f64>>,
    times: Vec<f64>,
    risk: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    sets: Vec<Set>,
}

fn matrix_deviation(name: &str, got: &[Option<f64>], expected: &[Vec<f64>], bound: f64) -> f64 {
    let p = expected.len();
    let mut worst = 0.0f64;
    for a in 0..p {
        for b in 0..p {
            let g = got[a * p + b].unwrap_or_else(|| panic!("{name}: ({a}, {b}) is None"));
            let scale = (expected[a][a].abs() * expected[b][b].abs()).sqrt();
            worst = worst.max((g - expected[a][b]).abs() / (bound * scale));
        }
    }
    worst
}

#[test]
fn weights_fit_and_absolute_risk_match_r() {
    let reference: Reference =
        load_json(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/ncc.json"));
    let v = &reference.versions;
    eprintln!(
        "reference: {}, survival {}, multipleNCC {}, jsonlite {}",
        v.r, v.survival, v.multiple_ncc, v.jsonlite
    );
    let eps = f64::EPSILON;
    for s in &reference.sets {
        let n = s.exit.len();
        let case: Vec<bool> = s.case.iter().map(|&c| c == 1).collect();
        let design = NccDesign {
            entry: s.entry_rule.then_some(s.entry.as_slice()),
            exit: &s.exit,
            case: &case,
            controls_per_case: s.m,
            stratum: s.strata.then_some(s.grp.as_slice()),
        };
        let inc = inclusion_probabilities(&design).unwrap();
        // The sizes are counts.
        assert_eq!(inc.risk_set_size, s.n_risk, "{}", s.name);
        // The probability is a sum of up to K logarithms, a difference of
        // prefix sums held to double-double, so each is rounded about
        // `K eps` in R's plain product and `eps` in the prefix sums. The
        // bound is relative, `K eps` with K the number of cases, which
        // R's product carries.
        let k = s.n_risk.len() as f64;
        let mut worst_p = 0.0f64;
        for (g, e) in inc.probability.iter().zip(&s.p) {
            worst_p = worst_p.max((g - e).abs() / (k * eps * e.abs().max(1e-300)));
        }
        eprintln!(
            "{}: inclusion probability deviation over tolerance {worst_p:.3}",
            s.name
        );
        assert!(worst_p <= 1.0, "{}: p deviates {worst_p}", s.name);

        // The fit of the cases and the sampled controls.
        let sampled: Vec<bool> = s.state.iter().map(|&t| t > 0).collect();
        let w_all = weights(&inc, &sampled, &case).unwrap();
        let rows: Vec<usize> = (0..n).filter(|&i| sampled[i]).collect();
        let stop: Vec<f64> = rows.iter().map(|&i| s.exit[i]).collect();
        let start: Vec<f64> = rows.iter().map(|&i| s.entry[i]).collect();
        let status: Vec<i32> = rows.iter().map(|&i| i32::from(case[i])).collect();
        let w: Vec<f64> = rows.iter().map(|&i| w_all[i]).collect();
        let x = Mat::from_fn(rows.len(), 2, |i, j| {
            if j == 0 { s.x1[rows[i]] } else { s.x2[rows[i]] }
        });
        let data = SurvData::builder(&stop, &status)
            .start(&start)
            .build()
            .unwrap();
        let opts = CoxOptions::default();
        let model = fit_weighted(&data, x.as_ref(), &w, &opts).unwrap();
        assert_eq!(model.iterations, s.iter, "{}", s.name);
        let bound = 8.0 * s.kappa * eps + 4.0 * (rows.len() as f64).sqrt() * eps;
        let mut coefficient = 0.0f64;
        for (g, e) in model.coefficients.iter().zip(&s.coefficients) {
            coefficient = coefficient.max((g.unwrap() - e).abs() / (bound * e.abs().max(1.0)));
        }
        let cov_bound = 3.0 * 2.0 * bound;
        let model_cov = matrix_deviation(&s.name, &model.covariance, &s.var_model, cov_bound);
        let robust = matrix_deviation(
            &s.name,
            model.robust_covariance.as_ref().unwrap(),
            &s.var_robust,
            cov_bound,
        );
        let loglik =
            (model.log_partial_likelihood.unwrap() - s.loglik).abs() / (bound * s.loglik.abs());
        let null = (model.null_log_partial_likelihood.unwrap() - s.loglik_null).abs()
            / (bound * s.loglik_null.abs());

        // The baseline hazard and the absolute risk.
        let base = baseline(&model, &data, x.as_ref(), &w, &opts).unwrap();
        // `basehaz` also lists the censoring times, where the hazard is the
        // one of the last event time. Each event time of the fit is one of
        // its times.
        let mut hazard = 0.0f64;
        for (t, g) in base.times.iter().zip(&base.cumulative_hazard) {
            let k = s
                .base_time
                .iter()
                .position(|r| r.to_bits() == t.to_bits())
                .unwrap_or_else(|| panic!("{}: event time {t} is not in basehaz", s.name));
            let e = s.base_hazard[k];
            hazard = hazard.max((g - e).abs() / (bound * 4.0 * base.times.len() as f64 * e.abs()));
        }
        let profiles = Mat::from_fn(s.profiles.len(), 2, |i, j| s.profiles[i][j]);
        let risk = absolute_risk(&base, &model.coefficients, profiles.as_ref(), &s.times).unwrap();
        let mut absolute = 0.0f64;
        for (g_row, e_row) in risk.iter().zip(&s.risk) {
            for (g, e) in g_row.iter().zip(e_row) {
                absolute =
                    absolute.max((g.unwrap() - e).abs() / (bound * 4.0 * base.times.len() as f64));
            }
        }
        eprintln!(
            "{}: coefficient {coefficient:.3} model {model_cov:.3} robust {robust:.3} loglik {loglik:.3} null {null:.3} hazard {hazard:.3} risk {absolute:.3}",
            s.name
        );
        for (what, dev) in [
            ("coefficient", coefficient),
            ("model covariance", model_cov),
            ("robust covariance", robust),
            ("log likelihood", loglik),
            ("null log likelihood", null),
            ("baseline hazard", hazard),
            ("absolute risk", absolute),
        ] {
            assert!(
                dev <= 1.0,
                "{}: {what} deviates {dev} of its tolerance",
                s.name
            );
        }
    }
}
