//! Per-protocol weights and the marginal structural model of sequential trials,
//! `epistat_survival::weighting::trials`, checked against TrialEmulation 0.0.5
//! `initiators(estimand_type = "PP")` and against differences of the model.
//! See `data/trial_msm.R`.
//!
//! # The tolerances
//!
//! The logistic models are fitted by IRLS from the same start with the same
//! stopping rule as `glm`, so a coefficient differs by `8 kappa eps` relative to
//! `max(|c|, 1)` with `kappa` the condition number of the information matrix, at
//! most `10^3` for the covariates here, and the linear predictor of a row by
//! `eta = 8 kappa eps sum_j |z_j| max(|c_j|, 1)`, below `1e-11`. A probability
//! then differs by `eta / 4`, a weight (a ratio of two probabilities, or of one
//! minus them) by at most `2 eta` relative, and the product of up to seven
//! weights by `14 eta`, so the weights agree to `1e-9` relative to a weight
//! of at most 4. The weighted outcome model has a weight error `delta = 1e-9`
//! and a condition number of `10^3` too, so its coefficients and covariance
//! agree to `1e-8`, and the cumulative incidence, whose derivative with respect
//! to a coefficient is at most 1, to `1e-8`. Observed agreement is `1e-13`.

mod trial_msm_run;

use epistat_linalg::Mat;
use epistat_survival::weighting::trials::cumulative_incidence;

fn worst(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn the_weights_and_the_outcome_model_agree_with_trial_emulation() {
    for a in trial_msm_run::load() {
        let o = trial_msm_run::run(&a);
        let w = &a.rows;
        let n = w.id.len();
        let (mut pd, mut pn) = (Vec::new(), Vec::new());
        for i in 0..n {
            if w.am_1[i] > 0.5 {
                pd.push(w.p1_d[i].unwrap());
                pn.push(w.p1_n[i].unwrap());
            } else {
                pd.push(w.p0_d[i].unwrap());
                pn.push(w.p0_n[i].unwrap());
            }
        }
        assert!(worst(&o.switch_denominator, &pd) < 1e-10);
        assert!(worst(&o.switch_numerator, &pn) < 1e-10);
        if a.use_censor {
            assert!(worst(&o.censor_denominator, &w.pc_d) < 1e-10);
            assert!(worst(&o.censor_numerator, &w.pc_n) < 1e-10);
        }
        assert!(worst(&o.row_weight, &w.wt) < 1e-9);
        assert!(worst(&o.trial_weight, &a.trials.weight) < 1e-9);
        assert!(worst(&o.coefficients, &a.msm_coef) < 1e-8);
        assert!(worst(&o.covariance, &a.msm_vcov) < 1e-8);
        assert!(worst(&o.risk[0], &a.cum_inc_0) < 1e-8);
        assert!(worst(&o.risk[1], &a.cum_inc_1) < 1e-8);
    }
}

#[test]
fn the_standard_error_of_the_incidence_is_the_delta_method() {
    // The derivative of the risk with respect to each coefficient by central
    // differences, with step h = 1e-5: the error of the difference is
    // O(h^2 f''') < 1e-9 for a risk with derivatives of order 1, and the
    // rounding error 2 eps / h = 5e-11. The variance is a quadratic form of
    // derivatives with a covariance of order 1, so it agrees to 1e-7 relative.
    let a = trial_msm_run::load().pop().unwrap();
    let o = trial_msm_run::run(&a);
    let p = o.msm.coefficients.len();
    let h = 1e-5;
    let risk_at = |coef: &[f64]| -> Vec<f64> {
        let mut msm = o.msm.clone();
        msm.coefficients = coef.to_vec();
        let refs: Vec<_> = o.designs[1].iter().map(Mat::as_ref).collect();
        cumulative_incidence(&msm, &refs).unwrap().risk
    };
    let k = o.risk[1].len();
    let mut grad = vec![vec![0.0; p]; k];
    for j in 0..p {
        let (mut up, mut down) = (o.msm.coefficients.clone(), o.msm.coefficients.clone());
        up[j] += h;
        down[j] -= h;
        let (ru, rd) = (risk_at(&up), risk_at(&down));
        for (g, (u, d)) in grad.iter_mut().zip(ru.iter().zip(&rd)) {
            g[j] = (u - d) / (2.0 * h);
        }
    }
    for (t, g) in grad.iter().enumerate() {
        let mut var = 0.0;
        for i in 0..p {
            for j in 0..p {
                var += g[i] * o.msm.covariance[i * p + j] * g[j];
            }
        }
        let se = var.sqrt();
        assert!(
            (se - o.std_err[1][t]).abs() <= 1e-7 * se.max(1e-3),
            "time {t}: {se} against {}",
            o.std_err[1][t]
        );
    }
}
