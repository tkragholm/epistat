//! Inverse probability of censoring weights checked against weights built in R
//! 4.6.2 from the definition with `survfit` (the Kaplan-Meier censoring
//! survival) and `coxph` with `basehaz` (the Cox model of censoring), on
//! counting-process data with a covariate that changes during follow-up and
//! delayed entry, with and without strata and stabilisation, and with both tie
//! rules.
//!
//! # The tolerances
//!
//! A weight is `exp(L)`, with `L` a sum over the censoring event times of terms
//! `exp(x'b) d/S0` for Cox, or of `log(1 - d/Y)` for Kaplan-Meier. The terms carry
//! the relative error of the coefficients, `bound X` with `bound = 8 kappa eps`
//! as in `r_cox.rs` and `X = sum_j max_i |x_ij| max(|b_j|, 1)`, and of a sum of
//! `n` terms, `n eps`. So `L` has a relative error of at most
//! `4 (bound X + n eps)`, and the weight a relative error of at most
//! `|L|` times that, plus `4 eps`. For Kaplan-Meier there is no coefficient and
//! the bound is `4 n eps |L|`. The `L` is read from the weight of R.

use epistat_survival::SurvData;
use epistat_survival::cox::Ties;
use epistat_survival::weighting::ipcw::{CensoringModel, IpcwOptions, fit};
use epistat_testkit::load_json;
use faer::Mat;
use serde::Deserialize;

const EPS: f64 = f64::EPSILON;

#[derive(Deserialize)]
struct RIpcw {
    name: String,
    model: String,
    by_group: bool,
    stabilise: bool,
    ties: String,
    id: Vec<u32>,
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    x: Vec<Vec<f64>>,
    group: Vec<u32>,
    #[serde(default)]
    kappa: Option<f64>,
    weight_stop: Vec<f64>,
    weight_start: Vec<f64>,
}

#[derive(Deserialize)]
struct Reference {
    ipcw: Vec<RIpcw>,
}

#[test]
fn censoring_weights_match_r() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/weighting.json"
    ));
    for r in &reference.ipcw {
        let n = r.stop.len();
        let x = Mat::from_fn(n, 2, |i, j| r.x[i][j]);
        let mut builder = SurvData::builder(&r.stop, &r.status)
            .start(&r.start)
            .id(&r.id);
        if r.by_group {
            builder = builder.stratum(&r.group);
        }
        let data = builder.build().unwrap();
        let model = if r.model == "km" {
            CensoringModel::KaplanMeier
        } else {
            CensoringModel::Cox { x: x.as_ref() }
        };
        let opts = IpcwOptions {
            stabilise: r.stabilise,
            ties: if r.ties == "efron" {
                Ties::Efron
            } else {
                Ties::Breslow
            },
            ..IpcwOptions::default()
        };
        let w = fit(&data, model, &opts).unwrap();
        let rel = match r.kappa {
            Some(kappa) => {
                let bound = 8.0 * kappa * EPS;
                let big_x: f64 = (0..2)
                    .map(|j| {
                        (0..n).map(|i| x[(i, j)].abs()).fold(0.0, f64::max)
                            * w.coefficients.as_ref().unwrap()[j].unwrap().abs().max(1.0)
                    })
                    .sum();
                4.0 * (bound * big_x + n as f64 * EPS)
            }
            None => 4.0 * n as f64 * EPS,
        };
        let mut worst = [0.0f64; 2];
        for i in 0..n {
            for (k, (mine, theirs)) in [
                (w.weight_at_stop[i], r.weight_stop[i]),
                (w.weight_at_start[i], r.weight_start[i]),
            ]
            .into_iter()
            .enumerate()
            {
                let l = theirs.ln().abs();
                let tol = theirs * (l * rel + 4.0 * EPS);
                let dev = (mine - theirs).abs() / tol;
                assert!(
                    dev <= 1.0,
                    "{} row {i} weight {k}: {mine} against {theirs} (tolerance {tol:e})",
                    r.name
                );
                worst[k] = worst[k].max(dev);
            }
        }
        println!(
            "{}: worst deviation in tolerances: at stop {:.3e}, at start {:.3e}",
            r.name, worst[0], worst[1]
        );
        // The censoring events are the last rows with status 0.
        assert_eq!(w.censored.iter().filter(|&&c| c).count(), {
            let mut last = std::collections::HashMap::new();
            for i in 0..n {
                let e = last.entry(r.id[i]).or_insert(i);
                if r.stop[i] > r.stop[*e] {
                    *e = i;
                }
            }
            last.values().filter(|&&i| r.status[i] == 0).count()
        });
    }
}
