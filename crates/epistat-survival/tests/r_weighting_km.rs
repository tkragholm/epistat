//! Weighted Kaplan-Meier curves checked against R 4.6.2: `survfit` with case
//! weights for the curve and `survfit(robust = TRUE)` for the standard error with
//! known weights, and finite differences of `survfit` refits for the covariance
//! that accounts for the estimated propensity weights, with and without delayed
//! entry. The weights come from the logistic propensity model of the lalonde
//! data of cobalt, with simulated survival times.
//!
//! # The tolerances
//!
//! With the weights of R given to `estimate_known`, the curve is a product of
//! `m` factors `1 - d/Y` of sums of weights, so it differs by at most
//! `4 (n + m) eps` relative. The standard error of `survfit` is its
//! infinitesimal jackknife, a sum over the `n` units of squares, with the same
//! bound. With the weights of this crate, which differ from those of R by a
//! relative error of at most `omega` (see `r_weighting.rs`), the curve differs by
//! at most `S (4 omega (1 + |ln S|) + 8 m eps)`: the logarithm of the curve is a sum
//! of `ln(Y - d) - ln Y`, whose weight derivatives sum to at most
//! `2 |ln S|` in relative terms. The covariance entries are compared
//! relative to `sqrt(V_gg V_hh)` with `64 omega + 64 (n + m) eps` plus twice the
//! difference between the finite differences with and without Richardson
//! extrapolation, which bounds the error of the reference.

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::weighting::Weights;
use epistat_survival::weighting::iptw::{self, Estimand, IptwOptions};
use epistat_survival::weighting::km::{KmOptions, estimate};
use epistat_testkit::load_json;
use serde::Deserialize;

const EPS: f64 = f64::EPSILON;

#[derive(Deserialize)]
struct RIptw {
    name: String,
    coefficients: Vec<f64>,
    kappa: f64,
}

#[derive(Deserialize)]
struct RKm {
    name: String,
    estimand: String,
    stabilise: bool,
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    times: Vec<f64>,
    iptw_weights: Vec<f64>,
    surv: Vec<Vec<f64>>,
    se_robust_survfit: Vec<f64>,
    covariance: Vec<Vec<Vec<f64>>>,
    known: Vec<Vec<Vec<f64>>>,
    covariance_plain: Vec<Vec<Vec<f64>>>,
    known_plain: Vec<Vec<Vec<f64>>>,
}

#[derive(Deserialize)]
struct Reference {
    treat: Vec<u8>,
    design: Vec<Vec<f64>>,
    iptw: Vec<RIptw>,
    km: Vec<RKm>,
}

#[test]
fn weighted_curves_match_survfit_and_finite_differences() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/weighting.json"
    ));
    let n = reference.treat.len();
    let p = reference.design[0].len();
    let z = Mat::from_fn(n, p, |i, j| reference.design[i][j]);
    let treat: Vec<bool> = reference.treat.iter().map(|&a| a == 1).collect();
    let stratum: Vec<u32> = reference.treat.iter().map(|&a| u32::from(a)).collect();
    for r in &reference.km {
        let m_times = r.times.len();
        let entry = r.start.iter().any(|&s| s > 0.0);
        let mut builder = SurvData::builder(&r.stop, &r.status).stratum(&stratum);
        if entry {
            builder = builder.start(&r.start);
        }
        // Known weights: R's own weights.
        let known_data = {
            let mut b = SurvData::builder(&r.stop, &r.status)
                .stratum(&stratum)
                .weight(&r.iptw_weights);
            if entry {
                b = b.start(&r.start);
            }
            b.build().unwrap()
        };
        let k = estimate(
            &known_data,
            &r.times,
            &KmOptions::default(),
            &Weights::known(),
        )
        .unwrap();
        assert_eq!(k.groups, vec![0, 1]);
        let mut worst = [0.0f64; 4];
        for g in 0..2 {
            for t in 0..m_times {
                let tol = r.surv[t][g] * 4.0 * (n + m_times) as f64 * EPS;
                let dev = (k.surv[g * m_times + t] - r.surv[t][g]).abs() / tol;
                assert!(dev <= 1.0, "{} curve ({t},{g})", r.name);
                worst[0] = worst[0].max(dev);
                let se = k.se_known(g, t).unwrap();
                let tol = r.se_robust_survfit[g * m_times + t] * 4.0 * (n + m_times) as f64 * EPS;
                let dev = (se - r.se_robust_survfit[g * m_times + t]).abs() / tol.max(1e-300);
                // survfit prints its standard error to the rounding of its own sums.
                assert!(
                    dev <= 1.0 || tol == 0.0,
                    "{} survfit se ({t},{g}): {se} against {}",
                    r.name,
                    r.se_robust_survfit[g * m_times + t]
                );
                worst[1] = worst[1].max(dev);
            }
        }
        // The estimated weights of this crate.
        let opts = IptwOptions {
            estimand: match r.estimand.as_str() {
                "ATE" => Estimand::Ate,
                "ATT" => Estimand::Att,
                _ => Estimand::Atc,
            },
            stabilise: r.stabilise,
            alpha: None,
        };
        let w = iptw::fit(&treat, z.as_ref(), &opts).unwrap();
        let data = builder.build().unwrap();
        let est = estimate(
            &data,
            &r.times,
            &KmOptions::default(),
            &Weights::iptw(&w, None),
        )
        .unwrap();
        let ri = reference
            .iptw
            .iter()
            .find(|x| x.name == if r.stabilise { "ate_stab" } else { "ate" })
            .unwrap_or(&reference.iptw[0]);
        let bound = 8.0 * ri.kappa * EPS;
        let omega = (0..n)
            .map(|i| {
                2.0 * bound
                    * (0..p)
                        .map(|j| z[(i, j)].abs() * ri.coefficients[j].abs().max(1.0))
                        .sum::<f64>()
                    + 4.0 * EPS
            })
            .fold(0.0, f64::max);
        for g in 0..2 {
            for t in 0..m_times {
                let s = r.surv[t][g];
                let tol = s * (4.0 * omega * (1.0 + s.ln().abs()) + 8.0 * m_times as f64 * EPS);
                assert!(
                    (est.surv[g * m_times + t] - s).abs() <= tol,
                    "{} estimated-weight curve ({t},{g})",
                    r.name
                );
            }
        }
        let rel = 64.0 * omega + 64.0 * (n + m_times) as f64 * EPS;
        for t in 0..m_times {
            for g in 0..2 {
                for h in 0..2 {
                    for (mine, theirs, plain, slot) in [
                        (
                            est.covariance[t].get(g, h),
                            r.covariance[t][g][h],
                            r.covariance_plain[t][g][h],
                            1,
                        ),
                        (
                            est.covariance_known[t].get(g, h),
                            r.known[t][g][h],
                            r.known_plain[t][g][h],
                            2,
                        ),
                    ] {
                        let scale = (r.covariance[t][g][g] * r.covariance[t][h][h]).sqrt();
                        let tol = rel * scale + 2.0 * (plain - theirs).abs();
                        let dev = (mine.unwrap() - theirs).abs() / tol;
                        assert!(
                            dev <= 1.0,
                            "{} covariance ({t},{g},{h}) slot {slot}: {} against {theirs} (tolerance {tol:e})",
                            r.name,
                            mine.unwrap()
                        );
                        worst[slot + 1] = worst[slot + 1].max(dev);
                    }
                }
            }
        }
        // The risk difference of the treated (group 1) and the controls: its
        // variance is V_11 + V_00 - 2 V_10, so its error is at most the sum of
        // the tolerances of the three entries, and that of its square root
        // that sum over the square root of the variance.
        for t in 0..m_times {
            let d = est.risk_difference(1, 0, t);
            let tol: f64 = (0..2)
                .map(|g| {
                    let s = r.surv[t][g];
                    s * (4.0 * omega * (1.0 + s.ln().abs()) + 8.0 * m_times as f64 * EPS)
                })
                .sum();
            assert!(
                (d.estimate - (r.surv[t][0] - r.surv[t][1])).abs() <= tol,
                "{} difference ({t})",
                r.name
            );
            for (mine, cov, plain) in [
                (d.se, &r.covariance, &r.covariance_plain),
                (d.se_known, &r.known, &r.known_plain),
            ] {
                let entry = |g: usize, h: usize| {
                    let scale = (r.covariance[t][g][g] * r.covariance[t][h][h]).sqrt();
                    rel * scale + 2.0 * (plain[t][g][h] - cov[t][g][h]).abs()
                };
                let var = cov[t][1][1] + cov[t][0][0] - 2.0 * cov[t][1][0];
                if var <= 0.0 {
                    continue;
                }
                let tol = (entry(1, 1) + entry(0, 0) + 2.0 * entry(1, 0)) / var.sqrt();
                let mine = mine.unwrap();
                assert!(
                    (mine - var.sqrt()).abs() <= tol,
                    "{} difference se ({t}): {mine} against {}",
                    r.name,
                    var.sqrt()
                );
            }
        }
        println!(
            "{}: worst deviation in tolerances: curve {:.3e}, survfit se {:.3e}, adjusted covariance {:.3e}, known covariance {:.3e}",
            r.name, worst[0], worst[1], worst[2], worst[3]
        );
    }
}
