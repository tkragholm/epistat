//! The Greenwood variances, probabilities at a fixed horizon, covariances of the
//! probabilities, several profiles, profiles that differ between transitions,
//! time-varying covariates and weights of `msreg`, checked against R 4.6.2,
//! `survival` 3.8-12 and `mstate` 0.3.3 on 300 of the patients of
//! `mstate::ebmt3`. `tests/data/msreg_more.R` makes the reference.
//!
//! The predictions are made from the coefficients `coxph` returned, with no
//! iteration, so that what is compared is the prediction and not the fit. The
//! models with time-varying covariates are also fitted and compared with
//! `coxph`.
//!
//! # Tolerances
//!
//! They are those of `r_msreg.rs`, with these additions.
//!
//! - A model without coefficients has no coefficient term in a quadratic form,
//!   so the covariance bound is the cumulative one times `p = 1` and
//!   `kappa_v = 1`.
//! - `mstate` takes the cumulative hazard of a model without coefficients as
//!   `-log(exp(-H))` of the survival curve, which leaves an absolute error of one
//!   rounding of the survival, `eps`, on top of the relative one. The bound adds
//!   `2 eps` to the hazard, and its variance is formed from the standard
//!   error of the curve and the survival, so the variance bound carries the
//!   same relative bound.
//! - A covariance of probabilities has the bound of a variance of one.
//! - The probabilities and the variances of a probability depend on the
//!   cumulative hazards and their covariances only through sums of at most `T`
//!   terms, as in `r_msreg.rs`, in both directions.

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::cox::Ties;
use epistat_survival::msreg::{
    CumulativeHazards, MsBaseline, MsFitOptions, MsModel, MsOptions, MsProfile, MsTerm, MsTerms,
    MsVarType, ProbTransDirection, ProbTransMethod, ProbTransOptions, fit, probtrans,
    probtrans_many,
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
    column: usize,
    transitions: Vec<usize>,
}

#[derive(Deserialize)]
struct RVar {
    trans1: usize,
    trans2: usize,
    value: Vec<f64>,
}

#[derive(Deserialize)]
struct RPt {
    predt: f64,
    direction: String,
    method: String,
    covariance: bool,
    time: Vec<f64>,
    /// By state left, time, and state entered.
    p: Vec<Vec<Vec<f64>>>,
    se: Vec<Vec<Vec<Option<f64>>>>,
    varmatrix: Option<Vec<Vec<Vec<f64>>>>,
}

#[derive(Deserialize)]
struct RProfile {
    name: String,
    values: Vec<Vec<f64>>,
    time: Vec<f64>,
    haz: Vec<Vec<f64>>,
    varhaz: Vec<RVar>,
    probtrans: Vec<RPt>,
}

#[derive(Deserialize)]
struct RCase {
    name: String,
    kind: String,
    ties: String,
    vartype: Option<String>,
    terms: Option<Vec<RTerm>>,
    coefficients: Option<Vec<f64>>,
    vcov: Option<Vec<Vec<f64>>>,
    loglik: Option<f64>,
    iterations: Option<usize>,
    information_condition: Option<f64>,
    n_events: Option<usize>,
    profiles: Vec<RProfile>,
}

#[derive(Deserialize)]
struct RCurve {
    time: Vec<f64>,
    cumhaz: Vec<f64>,
    std_err: Vec<f64>,
}

#[derive(Deserialize)]
struct RWeights {
    ties: String,
    coefficients: Vec<f64>,
    vcov: Vec<Vec<f64>>,
    profile: Vec<f64>,
    hazards: Vec<RCurve>,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    data: RData,
    tvdata: RData,
    cases: Vec<RCase>,
    refusal: String,
    weights: Vec<RWeights>,
}

fn load() -> Reference {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/msreg_more.json"
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

fn ties_of(name: &str) -> Ties {
    if name == "efron" {
        Ties::Efron
    } else {
        Ties::Breslow
    }
}

fn design(d: &RData, columns: usize) -> Mat<f64> {
    Mat::from_fn(d.x.len(), columns, |i, j| d.x[i][j])
}

/// mstate's coefficients go by transition within covariate of the formula, the
/// order of `MsTerms::Separate` is covariate and then transition.
fn separate_order(values: &[f64]) -> Vec<f64> {
    (0..9).map(|i| values[(i % 3) * 3 + i / 3]).collect()
}

fn kappa_of(vcov: &[Vec<f64>]) -> f64 {
    let p = vcov.len();
    let v = Mat::from_fn(p, p, |i, j| vcov[i][j] / (vcov[i][i] * vcov[j][j]).sqrt());
    let eig = v.self_adjoint_eigen(epistat_linalg::Side::Lower).unwrap();
    let values: Vec<f64> = eig.S().column_vector().iter().copied().collect();
    values.iter().copied().fold(f64::MIN, f64::max)
        / values.iter().copied().fold(f64::MAX, f64::min)
}

fn same_profile_bits(a: &CumulativeHazards, b: &CumulativeHazards) {
    assert_eq!(a, b);
}

/// The design, the terms, the coefficients and their covariance of a case.
type Pieces = (Mat<f64>, Option<Vec<MsTerm>>, Vec<f64>, Vec<Vec<f64>>);

/// The model of a case at the coefficients of R, with the data it was made from.
struct Built {
    model: MsModel,
    /// The coefficients in the order of the model's terms.
    coefficients: Vec<f64>,
    /// The covariance matrix of the coefficients in the order of the model's
    /// terms, from R.
    vcov: Vec<Vec<f64>>,
}

fn build(reference: &Reference, case: &RCase) -> Built {
    let tv = case.kind == "tv";
    let d = if tv {
        &reference.tvdata
    } else {
        &reference.data
    };
    let data = SurvData::builder(&d.stop, &d.status)
        .start(&d.start)
        .id(&d.id)
        .istate(&d.istate)
        .n_codes(3)
        .build()
        .unwrap();
    let ties = ties_of(&case.ties);
    let (x, terms, init, vcov): Pieces = match case.kind.as_str() {
        "null" => (Mat::zeros(d.x.len(), 0), None, vec![], vec![]),
        "separate" => {
            let c = case.coefficients.as_ref().unwrap();
            let v = case.vcov.as_ref().unwrap();
            let at = |i: usize| (i % 3) * 3 + i / 3;
            (
                design(d, 3),
                None,
                separate_order(c),
                (0..9)
                    .map(|i| (0..9).map(|j| v[at(i)][at(j)]).collect())
                    .collect(),
            )
        }
        "common" => (
            design(d, 3),
            None,
            case.coefficients.clone().unwrap(),
            case.vcov.clone().unwrap(),
        ),
        "tv" => {
            let terms: Vec<MsTerm> = case
                .terms
                .as_ref()
                .unwrap()
                .iter()
                .map(|t| MsTerm::covariate(t.column, &t.transitions))
                .collect();
            (
                design(d, 4),
                Some(terms),
                case.coefficients.clone().unwrap(),
                case.vcov.clone().unwrap(),
            )
        }
        other => panic!("unknown kind {other}"),
    };
    let opts = MsOptions {
        ties,
        terms: match (case.kind.as_str(), &terms) {
            (_, Some(t)) => MsTerms::Custom(t),
            ("common", _) => MsTerms::Common,
            _ => MsTerms::Separate,
        },
        baseline: MsBaseline::Separate,
        start: (!init.is_empty()).then(|| init.clone()),
        max_iterations: 0,
        ..MsOptions::default()
    };
    let model = fit(&data, x.as_ref(), &opts).unwrap();
    Built {
        model,
        coefficients: init,
        vcov,
    }
}

/// The worst deviation, in units of its bound, of each output.
struct Worst {
    hazard: f64,
    covariance: f64,
    probability: f64,
    variance: f64,
    prob_covariance: f64,
}

fn check_profile(
    name: &str,
    built: &Built,
    case: &RCase,
    profile: &RProfile,
    h: &CumulativeHazards,
    worst: &mut Worst,
) {
    let null = case.kind == "null";
    assert_eq!(h.time, profile.time, "{name}: the times");
    let n_t = h.time.len() as f64;
    let p = built.coefficients.len().max(1) as f64;
    let xb = 2.0 * built.coefficients.iter().map(|b| b.abs()).sum::<f64>();
    let hazard_bound = 8.0 * f64::EPSILON * (n_t + p + 4.0 + xb);
    let kappa_v = if null { 1.0 } else { kappa_of(&built.vcov) };
    let cov_bound = hazard_bound * p * kappa_v;
    let absolute = if null { 2.0 * f64::EPSILON } else { 0.0 };
    for k in 0..3 {
        for (g, e) in h.cumulative_hazard[k].iter().zip(&profile.haz[k]) {
            worst.hazard = worst
                .hazard
                .max((g - e).abs() / (hazard_bound * e.abs() + absolute).max(f64::MIN_POSITIVE));
        }
    }
    let mut cov_max = 0.0f64;
    for v in &profile.varhaz {
        let (a, b) = (v.trans1 - 1, v.trans2 - 1);
        let got = h.covariance_of(a, b).unwrap();
        let va = h.covariance_of(a, a).unwrap();
        let vb = h.covariance_of(b, b).unwrap();
        for (i, (g, e)) in got.iter().zip(&v.value).enumerate() {
            cov_max = cov_max.max(e.abs());
            let scale = (va[i] * vb[i]).sqrt();
            if scale > 0.0 {
                worst.covariance = worst.covariance.max((g - e).abs() / (cov_bound * scale));
            } else {
                assert!(*g == 0.0 && *e == 0.0, "{name}: zero variance");
            }
        }
    }
    let hmax = h
        .cumulative_hazard
        .iter()
        .flat_map(|c| c.iter().copied())
        .fold(0.0, f64::max);
    let p_bound = n_t * 3.0 * f64::EPSILON + hazard_bound * 3.0 * hmax;
    let var_bound = 2.0 * n_t * 9.0 * cov_bound * cov_max;
    for pt in &profile.probtrans {
        let direction = if pt.direction == "forward" {
            ProbTransDirection::Forward
        } else {
            ProbTransDirection::FixedHorizon
        };
        let method = if pt.method == "aalen" {
            ProbTransMethod::Aalen
        } else {
            ProbTransMethod::Greenwood
        };
        let label = format!(
            "{name} {} {} {} cov={}",
            pt.direction, pt.method, pt.predt, pt.covariance
        );
        let got = probtrans(
            h,
            pt.predt,
            &ProbTransOptions {
                se: true,
                method,
                direction,
                covariance: pt.covariance,
            },
        )
        .unwrap();
        assert_eq!(got.time, pt.time, "{label}: times");
        let rows = got.time.len();
        for s in 0..3 {
            for ti in 0..rows {
                for to in 0..3 {
                    let e = pt.p[s][ti][to];
                    worst.probability = worst
                        .probability
                        .max((got.probability_of(ti, s, to).unwrap() - e).abs() / p_bound);
                    let es = pt.se[s][ti][to];
                    let gs = got.se_of(ti, s, to);
                    // Standard errors are compared as variances, and an
                    // undefined one, a variance below zero, as zero.
                    let (gv, ev) = (gs.map_or(0.0, |v| v * v), es.map_or(0.0, |v| v * v));
                    worst.variance = worst.variance.max((gv - ev).abs() / var_bound);
                }
            }
        }
        if let Some(slices) = &pt.varmatrix {
            let n = 9;
            let fixed = direction == ProbTransDirection::FixedHorizon;
            if fixed {
                // R's slice k + 1 holds the covariance whose standard errors are
                // those of row k, and the first slice repeats the second.
                for (k, s) in slices.iter().enumerate().skip(1) {
                    for from in 0..3 {
                        for to in 0..3 {
                            let diag = s[from + 3 * to][from + 3 * to];
                            let se = pt.se[from][k - 1][to].map_or(0.0, |v| v * v);
                            assert!(
                                (diag - se).abs() <= var_bound,
                                "{label}: slice {k} is not the covariance of row {}",
                                k - 1
                            );
                        }
                    }
                }
                assert_eq!(slices[0], slices[1], "{label}: first slice");
            }
            let covariance = got.covariance.as_ref().unwrap();
            for (ti, matrix) in covariance.iter().enumerate().take(rows) {
                let expected: Option<&Vec<Vec<f64>>> = if fixed {
                    slices.get(ti + 1)
                } else {
                    slices.get(ti)
                };
                for r in 0..n {
                    for c in 0..n {
                        let e = expected.map_or(0.0, |m| m[r][c]);
                        let g = matrix.get(r, c).unwrap();
                        worst.prob_covariance =
                            worst.prob_covariance.max((g - e).abs() / var_bound);
                    }
                }
            }
        } else {
            assert!(got.covariance.is_none());
        }
    }
}

fn units(worst: &Worst, label: &str) {
    eprintln!(
        "{label:28} in units of the bound: hazard {:.2e}  covariance {:.2e}  probability {:.2e}  \
         variance of probability {:.2e}  covariance of probability {:.2e}",
        worst.hazard, worst.covariance, worst.probability, worst.variance, worst.prob_covariance
    );
    for (what, value) in [
        ("hazard", worst.hazard),
        ("covariance", worst.covariance),
        ("probability", worst.probability),
        ("variance of probability", worst.variance),
        ("covariance of probability", worst.prob_covariance),
    ] {
        assert!(value <= 1.0, "{label} {what}: {value} of the bound");
    }
}

#[test]
fn predictions_match_mstate() {
    let reference = load();
    for case in &reference.cases {
        let built = build(&reference, case);
        let opts = MsFitOptions {
            vartype: if case.vartype.as_deref() == Some("greenwood") {
                MsVarType::Greenwood
            } else {
                MsVarType::Aalen
            },
            ..MsFitOptions::default()
        };
        let n_covariates = built.model.n_covariates();
        let rows: Vec<Vec<Vec<f64>>> = case
            .profiles
            .iter()
            .map(|p| {
                if p.values.is_empty() {
                    vec![vec![0.0; n_covariates]; 3]
                } else {
                    p.values.clone()
                }
            })
            .collect();
        let profiles: Vec<MsProfile<'_>> =
            rows.iter().map(|r| MsProfile::ByTransition(r)).collect();
        let many = built.model.msfit_many(&profiles, &opts).unwrap();
        let mut worst = Worst {
            hazard: 0.0,
            covariance: 0.0,
            probability: 0.0,
            variance: 0.0,
            prob_covariance: 0.0,
        };
        for ((profile, h), values) in case.profiles.iter().zip(&many).zip(&rows) {
            // Several profiles in one call are the profiles one by one, and a
            // profile that is the same for every transition is the plain one.
            let one = built
                .model
                .msfit_profile(&MsProfile::ByTransition(values), &opts)
                .unwrap();
            same_profile_bits(h, &one);
            if values.iter().all(|v| *v == values[0]) {
                let plain = built.model.msfit(&values[0], &opts).unwrap();
                same_profile_bits(h, &plain);
            }
            check_profile(
                &format!("{} {}", case.name, profile.name),
                &built,
                case,
                profile,
                h,
                &mut worst,
            );
        }
        // Several sets of hazards in one call.
        let popts = ProbTransOptions {
            covariance: true,
            direction: ProbTransDirection::FixedHorizon,
            ..ProbTransOptions::default()
        };
        let predt = case.profiles[0].probtrans[0].predt.max(500.0);
        let together = probtrans_many(&many, predt, &popts).unwrap();
        for (h, t) in many.iter().zip(&together) {
            assert_eq!(&probtrans(h, predt, &popts).unwrap(), t);
        }
        units(&worst, &case.name);
    }
}

#[test]
fn time_varying_fits_match_coxph() {
    let reference = load();
    let d = &reference.tvdata;
    let x = design(d, 4);
    let data = SurvData::builder(&d.stop, &d.status)
        .start(&d.start)
        .id(&d.id)
        .istate(&d.istate)
        .n_codes(3)
        .build()
        .unwrap();
    for case in reference.cases.iter().filter(|c| c.kind == "tv") {
        let terms: Vec<MsTerm> = case
            .terms
            .as_ref()
            .unwrap()
            .iter()
            .map(|t| MsTerm::covariate(t.column, &t.transitions))
            .collect();
        let model = fit(
            &data,
            x.as_ref(),
            &MsOptions {
                ties: ties_of(&case.ties),
                terms: MsTerms::Custom(&terms),
                ..MsOptions::default()
            },
        )
        .unwrap();
        assert!(model.fit.converged, "{}", case.name);
        assert_eq!(
            model.fit.iterations,
            case.iterations.unwrap(),
            "{}",
            case.name
        );
        assert_eq!(
            model.transitions.iter().map(|t| t.n_events).sum::<usize>(),
            case.n_events.unwrap()
        );
        let coefficients = case.coefficients.as_ref().unwrap();
        let vcov = case.vcov.as_ref().unwrap();
        let kappa = case.information_condition.unwrap();
        let bound = 8.0 * kappa * f64::EPSILON;
        let got: Vec<f64> = model.fit.coefficients.iter().map(|c| c.unwrap()).collect();
        let coefficient = worst_deviation(&got, coefficients, bound, 1.0, NanRule::Coincide);
        let p = coefficients.len();
        let mut covariance = 0.0f64;
        for a in 0..p {
            for b in 0..p {
                let scale = (vcov[a][a] * vcov[b][b]).sqrt();
                let g = model.fit.covariance.get(a, b).unwrap();
                covariance = covariance.max((g - vcov[a][b]).abs() / (p as f64 * bound * scale));
            }
        }
        let sums = case.n_events.unwrap() as f64 * model.n_stacked as f64 * f64::EPSILON;
        let loglik_got = model.fit.log_partial_likelihood.unwrap();
        let loglik_want = case.loglik.unwrap();
        let loglik = (loglik_got - loglik_want).abs() / (bound * loglik_want.abs().max(1.0) + sums);
        eprintln!(
            "{:12} kappa {kappa:9.3e}  in units of the bound: coefficient {coefficient:.2e}  \
             covariance {covariance:.2e}  loglik {loglik:.2e}",
            case.name
        );
        assert!(
            coefficient <= 1.0 && covariance <= 1.0 && loglik <= 1.0,
            "{}",
            case.name
        );
    }
}

#[test]
fn weighted_hazards_match_survfit() {
    let reference = load();
    assert!(
        reference.refusal.contains("weighted"),
        "mstate refuses a weighted model: {}",
        reference.refusal
    );
    let d = &reference.data;
    let x = design(d, 3);
    let weight: Vec<f64> = d.id.iter().map(|&i| 0.5 + f64::from(i % 4) / 2.0).collect();
    let data = SurvData::builder(&d.stop, &d.status)
        .start(&d.start)
        .id(&d.id)
        .istate(&d.istate)
        .n_codes(3)
        .weight(&weight)
        .build()
        .unwrap();
    for w in &reference.weights {
        let init = separate_order(&w.coefficients);
        let model = fit(
            &data,
            x.as_ref(),
            &MsOptions {
                ties: ties_of(&w.ties),
                start: Some(init.clone()),
                max_iterations: 0,
                robust: false,
                ..MsOptions::default()
            },
        )
        .unwrap();
        let h = model.msfit(&w.profile, &MsFitOptions::default()).unwrap();
        let at = |i: usize| (i % 3) * 3 + i / 3;
        let vcov: Vec<Vec<f64>> = (0..9)
            .map(|i| (0..9).map(|j| w.vcov[at(i)][at(j)]).collect())
            .collect();
        let n_t = h.time.len() as f64;
        let xb = 2.0 * init.iter().map(|b| b.abs()).sum::<f64>();
        let hazard_bound = 8.0 * f64::EPSILON * (n_t + 9.0 + 4.0 + xb);
        let cov_bound = hazard_bound * 9.0 * kappa_of(&vcov);
        let mut worst = [0.0f64; 2];
        for (k, curve) in w.hazards.iter().enumerate() {
            for (i, &t) in curve.time.iter().enumerate() {
                // The curve of `survfit` has a row at every time of the data,
                // the hazards here at event times and the last time, carried
                // forward.
                let ti = h.time.partition_point(|&g| g <= t);
                if ti == 0 {
                    assert_eq!(curve.cumhaz[i].to_bits(), 0.0f64.to_bits());
                    continue;
                }
                let ti = ti - 1;
                let e = curve.cumhaz[i];
                worst[0] = worst[0].max(
                    (h.cumulative_hazard[k][ti] - e).abs()
                        / (hazard_bound * e).max(f64::MIN_POSITIVE),
                );
                let v = curve.std_err[i] * curve.std_err[i];
                let g = h.covariance_of(k, k).unwrap()[ti];
                worst[1] = worst[1].max((g - v).abs() / (cov_bound * v).max(f64::MIN_POSITIVE));
            }
        }
        eprintln!(
            "weighted {}: in units of the bound: hazard {:.2e}  variance {:.2e}",
            w.ties, worst[0], worst[1]
        );
        assert!(worst[0] <= 1.0 && worst[1] <= 1.0, "weighted {}", w.ties);
    }
}
