//! Pooled logistic censoring weights checked against R 4.6.2: `ipw::ipwtm(type =
//! "cens", family = "binomial")` and weights built from `glm` and `cumprod`, on
//! person-period data, unstabilised and stabilised, and with rows after the
//! first censoring row of a subject.
//!
//! # The tolerances
//!
//! Both sides run IRLS from the same start with the same stopping rule, so a
//! coefficient differs by `bound = 8 kappa eps` relative to `max(|c|, 1)` with
//! `kappa` the condition number of the information on its scaled columns (see
//! `r_weighting.rs`). The linear predictor of a row then has an error of at most
//! `eta_j = bound sum_k |z_jk| max(|c_k|, 1)`, and the log of a factor
//! `(1 - n_j) / (1 - d_j)` at most `eta^den_j + eta^num_j`, since the derivative of
//! `log(1 - p)` with respect to the linear predictor is `-p`, at most 1 in size.
//! The weight of a row is the product over the rows of its subject up to it, so
//! its relative error is at most `2 L + 4 m eps`, with `L` the sum of those
//! bounds over the `m` rows. A fitted probability differs by at most `eta / 4`.

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::weighting::pooled::{PooledEvent, PooledOptions, fit};
use epistat_testkit::load_json;
use serde::Deserialize;

const EPS: f64 = f64::EPSILON;

#[derive(Deserialize)]
struct Versions {
    r: String,
    ipw: String,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    stabilise: bool,
    id: Vec<u32>,
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    censored: Vec<u8>,
    design: Vec<Vec<f64>>,
    numerator_design: Option<Vec<Vec<f64>>>,
    ipw_weights: Vec<f64>,
    weights: Vec<f64>,
    weights_before: Vec<f64>,
    probability: Vec<f64>,
    coefficients: Vec<f64>,
    numerator_coefficients: Option<Vec<f64>>,
    active: Vec<u8>,
    kappa_den: f64,
    kappa_num: Option<f64>,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    cases: Vec<Case>,
}

fn matrix(rows: &[Vec<f64>]) -> Mat<f64> {
    Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
}

fn eta_bounds(z: &Mat<f64>, coefficients: &[f64], kappa: f64) -> Vec<f64> {
    let bound = 8.0 * kappa * EPS;
    (0..z.nrows())
        .map(|i| {
            bound
                * (0..z.ncols())
                    .map(|j| z[(i, j)].abs() * coefficients[j].abs().max(1.0))
                    .sum::<f64>()
        })
        .collect()
}

#[test]
fn pooled_weights_match_ipwtm_and_glm() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/weighting_pooled.json"
    ));
    assert!(reference.versions.r.starts_with("R version 4.6"));
    assert_eq!(reference.versions.ipw, "1.3.2");
    for c in &reference.cases {
        let n = c.stop.len();
        let z = matrix(&c.design);
        let zn = c.numerator_design.as_ref().map(|m| matrix(m));
        let data = SurvData::builder(&c.stop, &c.status)
            .start(&c.start)
            .id(&c.id)
            .build()
            .unwrap();
        let censored: Vec<bool> = c.censored.iter().map(|&x| x == 1).collect();
        let w = fit(
            &data,
            PooledEvent::Censoring {
                censored: Some(&censored),
            },
            z.as_ref(),
            zn.as_ref().map(Mat::as_ref),
            &PooledOptions::default(),
        )
        .unwrap();
        assert_eq!(c.stabilise, zn.is_some());
        assert_eq!(w.censored, censored);
        assert_eq!(
            w.active.iter().map(|&a| u8::from(a)).collect::<Vec<_>>(),
            c.active
        );
        let eta_d = eta_bounds(&z, &c.coefficients, c.kappa_den);
        let eta_n = zn.as_ref().map_or_else(
            || vec![0.0; n],
            |m| {
                eta_bounds(
                    m,
                    c.numerator_coefficients.as_ref().unwrap(),
                    c.kappa_num.unwrap(),
                )
            },
        );
        let bound_d = 8.0 * c.kappa_den * EPS;
        let mut worst = [0.0f64; 4];
        let (mut l, mut m_rows) = (0.0f64, 0usize);
        for i in 0..n {
            if i == 0 || c.id[i] != c.id[i - 1] {
                l = 0.0;
                m_rows = 0;
            }
            let before_l = l;
            let before_m = m_rows;
            if c.active[i] == 1 {
                l += eta_d[i] + eta_n[i];
                m_rows += 1;
            }
            for (mine, theirs, big_l, rows, slot) in [
                (w.weight[i], c.weights[i], l, m_rows, 0),
                (w.weight[i], c.ipw_weights[i], l, m_rows, 1),
                (
                    w.weight_before[i],
                    c.weights_before[i],
                    before_l,
                    before_m,
                    2,
                ),
            ] {
                let tol = theirs * (2.0 * big_l + 4.0 * (rows as f64 + 1.0) * EPS);
                let dev = (mine - theirs).abs() / tol;
                assert!(
                    dev <= 1.0,
                    "{} row {i} slot {slot}: {mine} against {theirs} (tolerance {tol:e})",
                    c.name
                );
                worst[slot] = worst[slot].max(dev);
            }
            let tol = eta_d[i] / 4.0 + 4.0 * EPS;
            let dev = (w.probability[i] - c.probability[i]).abs() / tol;
            assert!(dev <= 1.0, "{} probability {i}", c.name);
            worst[3] = worst[3].max(dev);
        }
        for (j, (a, b)) in w.coefficients.iter().zip(&c.coefficients).enumerate() {
            let d = (a - b).abs() / (bound_d * b.abs().max(1.0));
            assert!(d <= 1.0, "{} coefficient {j}", c.name);
        }
        println!(
            "{}: worst deviation in tolerances: weight {:.3e}, against ipwtm {:.3e}, before {:.3e}, probability {:.3e}",
            c.name, worst[0], worst[1], worst[2], worst[3]
        );
        // The default rule for the censoring events gives the same weights on data
        // that end at the censoring row.
        if c.name != "pooled_stab_extra_rows" {
            let d = fit(
                &data,
                PooledEvent::Censoring { censored: None },
                z.as_ref(),
                zn.as_ref().map(Mat::as_ref),
                &PooledOptions::default(),
            )
            .unwrap();
            assert_eq!(d.censored, censored);
            assert_eq!(
                d.weight.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                w.weight.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
        }
    }
}
