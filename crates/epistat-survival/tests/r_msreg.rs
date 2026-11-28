//! `msreg::fit`, `MsModel::msfit` and `msreg::probtrans` checked against R 4.6.2:
//! `survival::coxph` with a multi-state outcome for the fits, and `mstate`'s
//! `msfit` and `probtrans` on a stacked `coxph` for the predictions, on 700 of
//! the patients of `mstate::ebmt3`.
//!
//! The predictions are made from the coefficients `coxph` returned: the model is
//! fitted with those as starting values and no iteration, so that what is
//! compared is the prediction and not the fit.
//!
//! # Tolerances
//!
//! - A fit: two implementations of one Newton iteration that take the same
//!   steps differ by the error of the solve, the condition number `kappa` of the
//!   information matrix times the machine epsilon. The bound is `8 kappa eps`,
//!   relative to the value or to 1 for a value below 1. A covariance matrix
//!   carries that error in each of its `p` entries per row, so its bound is `p`
//!   times it, relative to the geometric mean of the two variances. The robust
//!   matrix is `V M V`: twice that, plus the rounding of the `n` rows of the
//!   meat, `4 p 8 kappa eps + 2 n eps`.
//! - A cumulative hazard is a compensated sum of at most `T` positive
//!   increments, each with a relative error of `(p + 4 + xb) eps`, where `xb`
//!   bounds `|(x - c)' beta|`. The bound is `8 eps (T + p + 4 + xb)` relative to
//!   the value, as in `cox::predict`.
//! - A variance or covariance of cumulative hazards is a sum over the times
//!   of products of such sums with the matrix `V`, whose cancellation is at
//!   most `p` times its condition number `kappa_v`. The bound is the
//!   cumulative one times `p kappa_v`, relative to the geometric mean of the
//!   two variances at the time.
//! - A transition probability is a product of at most `T` factors
//!   `I + dA`, each multiplication adding `S eps` to an entry, and its
//!   inputs the cumulative hazards, whose relative bound enters the sum
//!   of the increments, so its bound is `T S eps` plus the cumulative-hazard
//!   bound times the largest cumulative hazard, absolute.
//!
//! The worst deviation in units of its bound is printed for each output.

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::cox::Ties;
use epistat_survival::msreg::{
    MsBaseline, MsOptions, MsTerm, MsTerms, ProbTransOptions, fit, probtrans,
};
use epistat_testkit::{NanRule, load_json, worst_deviation};
use serde::Deserialize;

#[derive(Deserialize)]
struct Versions {
    r: String,
    survival: String,
    mstate: String,
    jsonlite: String,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    data: RData,
    fits: Vec<RFit>,
    msfit: Vec<RMsfit>,
}

#[derive(Deserialize)]
struct RData {
    id: Vec<u32>,
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    istate: Vec<u32>,
    x: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct RTerm {
    column: i64,
    transitions: Vec<usize>,
}

#[derive(Deserialize)]
struct RFit {
    name: String,
    ties: String,
    baseline: String,
    terms: Vec<RTerm>,
    coefficients: Vec<f64>,
    vcov_robust: Vec<Vec<f64>>,
    vcov_model: Vec<Vec<f64>>,
    loglik: f64,
    loglik_null: f64,
    iterations: usize,
    information_condition: f64,
    n_events: usize,
}

#[derive(Deserialize)]
struct RVar {
    trans1: usize,
    trans2: usize,
    value: Vec<f64>,
}

#[derive(Deserialize)]
struct RProb {
    time: Vec<f64>,
    p: Vec<Vec<f64>>,
    se: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct RMsfit {
    name: String,
    ties: String,
    profile: Vec<f64>,
    coefficients: Vec<f64>,
    vcov: Vec<Vec<f64>>,
    time: Vec<f64>,
    haz: Vec<Vec<f64>>,
    varhaz: Vec<RVar>,
    probtrans0: Vec<RProb>,
    probtrans200: Vec<RProb>,
}

fn ties_of(name: &str) -> Ties {
    if name == "efron" {
        Ties::Efron
    } else {
        Ties::Breslow
    }
}

fn load() -> Reference {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/msreg.json"
    ));
    eprintln!(
        "reference: {}, survival {}, mstate {}, jsonlite {}",
        reference.versions.r,
        reference.versions.survival,
        reference.versions.mstate,
        reference.versions.jsonlite
    );
    reference
}

fn design(d: &RData) -> Mat<f64> {
    Mat::from_fn(d.x.len(), 3, |i, j| d.x[i][j])
}

/// The largest deviation of a covariance matrix in units of
/// `bound * sqrt(e_aa e_bb)`.
fn covariance_deviation(got: &[Option<f64>], expected: &[Vec<f64>], bound: f64) -> f64 {
    let p = expected.len();
    let mut worst = 0.0f64;
    for a in 0..p {
        for b in 0..p {
            let scale = (expected[a][a] * expected[b][b]).sqrt();
            worst = worst.max((got[a * p + b].unwrap() - expected[a][b]).abs() / (bound * scale));
        }
    }
    worst
}

#[test]
fn fits_match_coxph() {
    let reference = load();
    let d = &reference.data;
    let x = design(d);
    let data = SurvData::builder(&d.stop, &d.status)
        .start(&d.start)
        .id(&d.id)
        .istate(&d.istate)
        .n_codes(3)
        .build()
        .unwrap();
    for r in &reference.fits {
        let terms: Vec<MsTerm> = r
            .terms
            .iter()
            .map(|t| {
                if t.column < 0 {
                    MsTerm::indicator(&t.transitions)
                } else {
                    MsTerm::covariate(usize::try_from(t.column).unwrap(), &t.transitions)
                }
            })
            .collect();
        let opts = MsOptions {
            ties: ties_of(&r.ties),
            terms: MsTerms::Custom(terms.clone()),
            baseline: if r.baseline == "shared" {
                MsBaseline::Shared
            } else {
                MsBaseline::Separate
            },
            ..MsOptions::default()
        };
        let model = fit(&data, x.as_ref(), &opts).unwrap();
        assert!(model.fit.converged, "{}", r.name);
        assert_eq!(model.fit.iterations, r.iterations, "{}", r.name);
        assert_eq!(model.fit.n_rows, d.stop.len());
        assert_eq!(model.n_ids, 700);
        assert_eq!(
            model.transitions.iter().map(|t| t.n_events).sum::<usize>(),
            r.n_events,
            "{}",
            r.name
        );
        let p = r.coefficients.len() as f64;
        let bound = 8.0 * r.information_condition * f64::EPSILON;
        let coefficients: Vec<f64> = model.fit.coefficients.iter().map(|c| c.unwrap()).collect();
        let coefficient = worst_deviation(
            &coefficients,
            &r.coefficients,
            bound,
            1.0,
            NanRule::Coincide,
        );
        // `coxph` adds and removes rows from a running risk-set sum without
        // compensation, which leaves a relative error of up to `m eps` in the
        // sum of `m` operations, and each of the `D` events takes the logarithm
        // of one, so the log likelihood carries `D m eps` more.
        let sums = r.n_events as f64 * model.n_stacked as f64 * f64::EPSILON;
        let loglik = (model.fit.log_partial_likelihood.unwrap() - r.loglik).abs()
            / (bound * r.loglik.abs().max(1.0) + sums);
        let null = (model.fit.null_log_partial_likelihood.unwrap() - r.loglik_null).abs()
            / (bound * r.loglik_null.abs().max(1.0) + sums);
        let covariance =
            covariance_deviation(model.fit.covariance.as_slice(), &r.vcov_model, p * bound);
        let robust = covariance_deviation(
            model.fit.robust_covariance.as_ref().unwrap().as_slice(),
            &r.vcov_robust,
            4.0 * p * bound + 2.0 * d.stop.len() as f64 * f64::EPSILON,
        );
        eprintln!(
            "{:18} kappa {:9.3e}  in units of the bound: coefficient {coefficient:.2e}  loglik {loglik:.2e}  \
             null {null:.2e}  covariance {covariance:.2e}  robust {robust:.2e}",
            r.name, r.information_condition
        );
        for (what, value) in [
            ("coefficient", coefficient),
            ("loglik", loglik),
            ("null loglik", null),
            ("covariance", covariance),
            ("robust covariance", robust),
        ] {
            assert!(value <= 1.0, "{} {what}: {value} of the bound", r.name);
        }
    }
}

/// Reorder mstate's coefficients, by covariate and then transition, into the
/// order of `MsTerms::Separate`, by transition and then covariate.
fn separate_order(values: &[f64]) -> Vec<f64> {
    (0..9).map(|i| values[(i % 3) * 3 + i / 3]).collect()
}

#[test]
fn predictions_match_mstate() {
    let reference = load();
    let d = &reference.data;
    let x = design(d);
    let data = SurvData::builder(&d.stop, &d.status)
        .start(&d.start)
        .id(&d.id)
        .istate(&d.istate)
        .n_codes(3)
        .build()
        .unwrap();
    for r in &reference.msfit {
        let common = r.name.starts_with("common");
        let shared = r.name == "common_shared";
        let init = if common {
            r.coefficients.clone()
        } else {
            separate_order(&r.coefficients)
        };
        let opts = MsOptions {
            ties: ties_of(&r.ties),
            terms: if common {
                MsTerms::Common
            } else {
                MsTerms::Separate
            },
            baseline: if shared {
                MsBaseline::Shared
            } else {
                MsBaseline::Separate
            },
            start: Some(init.clone()),
            max_iterations: 0,
            ..MsOptions::default()
        };
        let model = fit(&data, x.as_ref(), &opts).unwrap();
        let h = model
            .msfit(
                &r.profile,
                &epistat_survival::msreg::MsFitOptions::default(),
            )
            .unwrap();
        assert_eq!(h.time, r.time, "{}: the times", r.name);
        let n_t = h.time.len() as f64;
        let p = init.len() as f64;
        let xb = 2.0 * init.iter().map(|b| b.abs()).sum::<f64>();
        let hazard_bound = 8.0 * f64::EPSILON * (n_t + p + 4.0 + xb);
        // mstate's coefficient order is by covariate and then transition.
        let at = |i: usize| if common { i } else { (i % 3) * 3 + i / 3 };
        // The condition number of the correlation matrix of the coefficients,
        // the scale of the quadratic forms with it.
        let kappa_v = {
            let v = Mat::from_fn(init.len(), init.len(), |i, j| {
                r.vcov[at(i)][at(j)] / (r.vcov[at(i)][at(i)] * r.vcov[at(j)][at(j)]).sqrt()
            });
            let eig = v.self_adjoint_eigen(epistat_linalg::Side::Lower).unwrap();
            let values: Vec<f64> = eig.S().column_vector().iter().copied().collect();
            values.iter().copied().fold(f64::MIN, f64::max)
                / values.iter().copied().fold(f64::MAX, f64::min)
        };
        let mut worst = [0.0f64; 5];
        for k in 0..3 {
            for (g, e) in h.cumulative_hazard[k].iter().zip(&r.haz[k]) {
                worst[0] =
                    worst[0].max((g - e).abs() / (hazard_bound * e.abs().max(f64::MIN_POSITIVE)));
            }
        }
        let cov_bound = hazard_bound * p * kappa_v;
        let mut cov_max = 0.0f64;
        for v in &r.varhaz {
            let (a, b) = (v.trans1 - 1, v.trans2 - 1);
            let got = h.covariance_of(a, b).unwrap();
            let va = h.covariance_of(a, a).unwrap();
            let vb = h.covariance_of(b, b).unwrap();
            for (i, (g, e)) in got.iter().zip(&v.value).enumerate() {
                cov_max = cov_max.max(e.abs());
                let scale = (va[i] * vb[i]).sqrt();
                if scale > 0.0 {
                    worst[1] = worst[1].max((g - e).abs() / (cov_bound * scale));
                } else {
                    assert!(*g == 0.0 && *e == 0.0, "{} zero variance", r.name);
                }
            }
        }
        let hmax = h
            .cumulative_hazard
            .iter()
            .flat_map(|c| c.iter().copied())
            .fold(0.0, f64::max);
        let p_bound = n_t * 3.0 * f64::EPSILON + hazard_bound * 3.0 * hmax;
        // The variance of a probability is built from increments of the
        // covariances of the cumulative hazards, each a difference of two
        // values with the relative error `cov_bound` of the largest one, summed
        // over the `T` times with weights below the size of the matrix.
        let var_bound = 2.0 * n_t * 9.0 * cov_bound * cov_max;
        for (predt, probs) in [(0.0, &r.probtrans0), (200.0, &r.probtrans200)] {
            let pt = probtrans(&h, predt, &ProbTransOptions::default()).unwrap();
            for (s, prob) in probs.iter().enumerate() {
                assert_eq!(pt.time, prob.time, "{} {predt}: times", r.name);
                for (ti, _) in pt.time.iter().enumerate() {
                    for to in 0..3 {
                        let e = prob.p[ti][to];
                        worst[2] = worst[2]
                            .max((pt.probability_of(ti, s, to).unwrap() - e).abs() / p_bound);
                        let es = prob.se[ti][to];
                        // Standard errors are compared as variances.
                        let gs = pt.se_of(ti, s, to).unwrap();
                        worst[3] = worst[3].max((gs * gs - es * es).abs() / var_bound);
                    }
                }
            }
        }
        eprintln!(
            "{:16} in units of the bound: hazard {:.2e}  covariance {:.2e}  probability {:.2e}  \
             variance of probability {:.2e}",
            r.name, worst[0], worst[1], worst[2], worst[3]
        );
        for (what, value) in [
            ("hazard", worst[0]),
            ("covariance", worst[1]),
            ("probability", worst[2]),
            ("variance of probability", worst[3]),
        ] {
            assert!(value <= 1.0, "{} {what}: {value} of the bound", r.name);
        }
    }
}
