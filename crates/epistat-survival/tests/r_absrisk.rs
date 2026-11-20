//! `CSC` and `predict.CauseSpecificCox` of riskRegression against
//! [`epistat_survival::absrisk`], on the Melanoma data and on simulated data
//! with strata, tied times, three causes, both `surv.type` values, with and
//! without the product limit, Breslow and Efron ties.
//!
//! The functions are given the coefficients and the covariance matrices that
//! `coxph` returned, so that what is compared is the prediction.
//!
//! # Tolerances
//!
//! With `n` the number of rows, `J` the number of event times, `p` the largest
//! number of columns of a model and `xb` the largest of `sum_a 2 |x_a beta_a|`
//! over the rows and new rows of a model:
//!
//! - a risk score has the relative error `(p + 2 + xb) eps` in each of the two
//!   implementations, a baseline increment `(n + 2) eps` (a risk-set sum of `n`
//!   positive terms, summed in a different order), and a product or a running
//!   sum of `J` terms `J eps`. The absolute risk is a sum of positive terms
//!   `S theta dL`, so its relative error is that of its terms. Those are
//!   `S`, a product of `J` factors `1 - h` with `h` the total hazard increment,
//!   whose relative error is `sum h / (1 - h)` times the relative error of `h`,
//!   plus `J eps`, and a risk score times an increment. With `H` the total
//!   cumulative hazard of the row and `u` the largest total increment, so that
//!   `sum h / (1 - h) <= H / (1 - u)` (both are in the reference, from
//!   `predictCox`), the bound for the risk is
//!   `4 eps (n + J + p + 4 + xb) (1 + H / (1 - u))`, relative to the value;
//! - the standard error is the square root of a sum of squares of influence
//!   functions. An influence function is a sum of terms each of which carries
//!   the relative error of the bound above, and the influence function of the
//!   coefficients, `V u`, carries `p kappa` times it with `kappa` the condition
//!   number of the covariance matrix. What a rounding error does to the sum is
//!   scaled by `c = sqrt(sum_i (sum of |terms of i|)^2) / se`, which the direct
//!   evaluation in `tests/absrisk_naive` returns. The bound for the standard
//!   error is the bound of the risk times `max(1, p kappa) c`, relative to the
//!   value;
//! - a confidence limit is a smooth function of the risk and of its standard
//!   error, so its bound is the largest change of the limit when the risk and
//!   the standard error move by their bounds, plus `8 eps` relative for the
//!   evaluation of the powers.
//!
//! The bounds are not changed after a run. The worst deviation in units of its
//! bound is printed for each output.

#![allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::float_cmp,
    reason = "status codes are small positive integers, and event times are compared exactly"
)]

#[path = "absrisk_naive/mod.rs"]
mod absrisk_naive;

use absrisk_naive::{NCase, NModel, model_pieces, naive_se};
use epistat_linalg::{Mat, Side};
use epistat_stats::normal;
use epistat_survival::absrisk::{
    CauseSpecificCox, CscOptions, InfluenceForm, NewRows, RiskLimits, RiskOptions, SurvType,
};
use epistat_survival::cox::{CoxOptions, Ties};
use epistat_survival::{SurvBuf, SurvData};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Versions {
    r: String,
    #[serde(rename = "riskRegression")]
    risk_regression: String,
    prodlim: String,
    survival: String,
    jsonlite: String,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    cases: Vec<Case>,
    ties_note: TiesNote,
}

#[derive(Deserialize)]
struct TiesNote {
    tied: IfDiff,
    untied: IfDiff,
}

#[derive(Deserialize)]
struct IfDiff {
    max_diff: f64,
    max_abs: f64,
}

type Table = Vec<Vec<Option<f64>>>;

#[derive(Deserialize)]
struct Case {
    name: String,
    surv_type: String,
    ties: String,
    cause: u32,
    causes_order: Vec<u32>,
    time: Vec<f64>,
    status: Vec<i32>,
    stratum: Option<Vec<u32>>,
    designs: Vec<Vec<Vec<f64>>>,
    coefficients: Vec<Vec<f64>>,
    var: Vec<Vec<Vec<f64>>>,
    new_designs: Vec<Vec<Vec<f64>>>,
    new_stratum: Option<Vec<u32>>,
    times: Vec<f64>,
    max_hazard: Vec<f64>,
    total_cumhazard: Vec<f64>,
    results: Vec<Result_>,
}

#[derive(Deserialize)]
struct Result_ {
    product_limit: bool,
    absrisk: Table,
    se: Table,
    lower: Table,
    upper: Table,
    lower_none: Table,
    upper_none: Table,
    lower_log: Table,
    upper_log: Table,
    lower_cloglog: Table,
    upper_cloglog: Table,
    absrisk_nose: Table,
}

/// A difference in units of its bound, zero when there is none.
fn ratio(difference: f64, bound: f64) -> f64 {
    if difference == 0.0 {
        0.0
    } else {
        difference / bound
    }
}

fn matrix(rows: &[Vec<f64>]) -> Mat<f64> {
    Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
}

/// The limits as `transformCI` has them, restated for the bounds.
fn limit(kind: RiskLimits, f: f64, se: f64, z: f64) -> (f64, f64) {
    let q = |d: f64| if se == 0.0 { 0.0 } else { se / d };
    match kind {
        RiskLimits::None => unreachable!(),
        RiskLimits::Plain => ((f - z * q(1.0)).max(0.0), (f + z * q(1.0)).min(1.0)),
        RiskLimits::Log => (f * (-z * q(f)).exp(), (f * (z * q(f)).exp()).min(1.0)),
        RiskLimits::LogLog => {
            let t = q(-f * f.ln());
            (f.powf((z * t).exp()), f.powf((-z * t).exp()))
        }
        RiskLimits::CLogLog => {
            let t = q(-(1.0 - f) * (1.0 - f).ln());
            (
                1.0 - (1.0 - f).powf((-z * t).exp()),
                1.0 - (1.0 - f).powf((z * t).exp()),
            )
        }
    }
}

fn limit_bound(kind: RiskLimits, f: f64, se: f64, z: f64, b_f: f64, b_se: f64) -> (f64, f64) {
    let nominal = limit(kind, f, se, z);
    let mut worst = (0.0f64, 0.0f64);
    for df in [-b_f, b_f] {
        for ds in [-b_se, b_se] {
            let f2 = (f * (1.0 + df)).clamp(f64::MIN_POSITIVE, 1.0 - 1e-300);
            let moved = limit(kind, f2, se * (1.0 + ds), z);
            worst.0 = worst.0.max((moved.0 - nominal.0).abs());
            worst.1 = worst.1.max((moved.1 - nominal.1).abs());
        }
    }
    let r = 8.0 * f64::EPSILON;
    (worst.0 + r * nominal.0.abs(), worst.1 + r * nominal.1.abs())
}

#[derive(Default)]
struct Worst(std::collections::BTreeMap<String, f64>);

impl Worst {
    fn put(&mut self, name: &str, value: f64) {
        let entry = self.0.entry(name.to_string()).or_insert(0.0);
        *entry = entry.max(value);
    }
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one pass over the outputs of each case"
)]
#[expect(clippy::cast_precision_loss, reason = "row counts are far below 2^52")]
fn absolute_risk_matches_r() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/absrisk.json"
    ));
    let v = &reference.versions;
    eprintln!(
        "reference: {}, riskRegression {}, prodlim {}, survival {}, jsonlite {}",
        v.r, v.risk_regression, v.prodlim, v.survival, v.jsonlite
    );
    let eps = f64::EPSILON;
    let mut worst = Worst::default();
    let mut compared = 0usize;
    for case in &reference.cases {
        let name = &case.name;
        let n = case.time.len();
        let hazard_type = case.surv_type == "hazard";
        let n_codes = *case.status.iter().max().unwrap();
        let mut builder = SurvData::builder(&case.time, &case.status).n_codes(n_codes as u32);
        if let Some(stratum) = &case.stratum {
            builder = builder.stratum(stratum);
        }
        let data = builder.build().unwrap();
        let ties = if case.ties == "efron" {
            Ties::Efron
        } else {
            Ties::Breslow
        };
        // R's models are listed with the cause first. For the hazard type
        // ours are in the order of the codes.
        let order: Vec<usize> = if hazard_type {
            (1..=n_codes as u32)
                .map(|code| case.causes_order.iter().position(|&c| c == code).unwrap())
                .collect()
        } else {
            vec![0, 1]
        };
        let designs: Vec<Mat<f64>> = order.iter().map(|&m| matrix(&case.designs[m])).collect();
        let new_designs: Vec<Mat<f64>> = order
            .iter()
            .map(|&m| matrix(&case.new_designs[m]))
            .collect();
        let design_refs: Vec<_> = designs.iter().map(Mat::as_ref).collect();
        let new_refs: Vec<_> = new_designs.iter().map(Mat::as_ref).collect();
        let opts = CscOptions {
            ties,
            surv_type: if hazard_type {
                SurvType::Hazard
            } else {
                SurvType::Survival { cause: case.cause }
            },
            ..CscOptions::default()
        };
        let fitted = CauseSpecificCox::fit(&data, &design_refs, &opts).unwrap();
        // The coefficients and the covariances are R's.
        let mut fits = fitted.fits().to_vec();
        for (fit, &m) in fits.iter_mut().zip(&order) {
            fit.coefficients = case.coefficients[m].iter().copied().map(Some).collect();
            let p = case.coefficients[m].len();
            fit.covariance = (0..p * p)
                .map(|i| Some(case.var[m][i / p][i % p]))
                .collect();
            // The fitted values agree with R's to the convergence of the fit.
        }
        let model = CauseSpecificCox::from_fits(&data, &design_refs, fits.clone(), &opts).unwrap();
        let new = NewRows {
            designs: &new_refs,
            stratum: case.new_stratum.as_deref(),
        };

        // The bounds.
        let t_events = {
            let mut t: Vec<f64> = case
                .time
                .iter()
                .zip(&case.status)
                .filter(|&(_, &s)| s != 0)
                .map(|(t, _)| *t)
                .collect();
            t.sort_by(f64::total_cmp);
            t.dedup();
            t.len() as f64
        };
        let p_max = designs.iter().map(Mat::ncols).max().unwrap() as f64;
        let mut xb = 0.0f64;
        for (m, &rm) in order.iter().enumerate() {
            let mut total = 0.0f64;
            for a in 0..designs[m].ncols() {
                let big = (0..n)
                    .map(|i| designs[m][(i, a)].abs())
                    .chain((0..new_designs[m].nrows()).map(|i| new_designs[m][(i, a)].abs()))
                    .fold(0.0, f64::max);
                total += 2.0 * big * case.coefficients[rm][a].abs();
            }
            xb = xb.max(total);
        }
        let kappa = order
            .iter()
            .map(|&m| {
                let p = case.coefficients[m].len();
                let v = Mat::from_fn(p, p, |i, j| case.var[m][i][j]);
                let e = v.self_adjoint_eigenvalues(Side::Lower).unwrap();
                e[p - 1] / e[0]
            })
            .fold(1.0, f64::max);
        let b_base = 4.0 * eps * (n as f64 + t_events + p_max + 4.0 + xb);
        let b_row: Vec<f64> = (0..case.max_hazard.len())
            .map(|r| {
                let u = case.max_hazard[r];
                assert!(u < 1.0, "{name}: the largest hazard increment is {u}");
                b_base * (1.0 + case.total_cumhazard[r] / (1.0 - u))
            })
            .collect();

        // The data of the direct evaluation.
        let stratum_dense: Vec<usize> = case.stratum.as_ref().map_or_else(
            || vec![0; n],
            |s| {
                let mut levels = s.clone();
                levels.sort_unstable();
                levels.dedup();
                s.iter().map(|c| levels.binary_search(c).unwrap()).collect()
            },
        );
        let n_strata = stratum_dense.iter().max().unwrap() + 1;
        let mut grid: Vec<f64> = case
            .time
            .iter()
            .zip(&case.status)
            .filter(|&(_, &s)| s != 0)
            .map(|(t, _)| *t)
            .collect();
        grid.sort_by(f64::total_cmp);
        grid.dedup();
        let overall_buf = (!hazard_type).then(|| {
            let any: Vec<i32> = case.status.iter().map(|&s| i32::from(s != 0)).collect();
            let mut b = SurvBuf::builder(case.time.clone(), any);
            if let Some(s) = &case.stratum {
                b = b.stratum(s.clone());
            }
            b.build().unwrap()
        });
        let overall_data = overall_buf.as_ref().map(SurvBuf::as_data);
        let mut n_models = Vec::new();
        let model_options = opts_for(&opts, n_codes as u32, hazard_type, case.cause);
        for (m, fit) in fits.iter().enumerate() {
            let model_data = match (&overall_data, m) {
                (Some(d), 1) => d.clone(),
                _ => data.clone(),
            };
            let (lam, if_beta, v_matrix) = model_pieces(
                &model_data,
                &designs[m],
                fit,
                &model_options[m],
                &grid,
                n_strata,
            );
            let code = model_options[m].cause;
            n_models.push(NModel {
                p: designs[m].ncols(),
                x: (0..n)
                    .map(|i| {
                        (0..designs[m].ncols())
                            .map(|a| designs[m][(i, a)])
                            .collect()
                    })
                    .collect(),
                event: case
                    .status
                    .iter()
                    .map(|&s| code.map_or(s != 0, |c| s == c as i32))
                    .collect(),
                beta: fit.coefficients.iter().map(|c| c.unwrap()).collect(),
                v: v_matrix,
                lam,
                if_beta,
            });
        }
        let (cause_model, survival_models) = if hazard_type {
            ((case.cause - 1) as usize, (0..fits.len()).collect())
        } else {
            (0, vec![1])
        };
        // riskRegression's influence function of the coefficients is wrong for
        // a model with tied event times, see TiesNote, and so is its standard
        // error. Those cases are compared with the direct evaluation only.
        let any_tied = n_models.iter().any(|m| {
            let mut t: Vec<f64> = (0..n)
                .filter(|&i| m.event[i])
                .map(|i| case.time[i])
                .collect();
            t.sort_by(f64::total_cmp);
            t.windows(2).any(|w| w[0] == w[1])
        });
        let mut naive_case = NCase {
            stop: case.time.clone(),
            stratum: stratum_dense.clone(),
            grid: grid.clone(),
            models: n_models,
            cause_model,
            survival_models,
            product_limit: true,
            exact: false,
        };

        for res in &case.results {
            let ropts = RiskOptions {
                product_limit: res.product_limit,
                std_err: true,
                // riskRegression's first-order form, which the reference has.
                influence: InfluenceForm::Exponential,
                limits: RiskLimits::LogLog,
                conf_level: 0.95,
            };
            let got = model
                .predict(&new, &case.times, case.cause, &ropts)
                .unwrap();
            let exact = model
                .predict(
                    &new,
                    &case.times,
                    case.cause,
                    &RiskOptions {
                        influence: InfluenceForm::Exact,
                        ..ropts
                    },
                )
                .unwrap();
            assert_eq!(
                exact.risk, got.risk,
                "{name}: the form does not move the risk"
            );
            let plain = model
                .predict(
                    &new,
                    &case.times,
                    case.cause,
                    &RiskOptions {
                        std_err: false,
                        ..ropts
                    },
                )
                .unwrap();
            assert_eq!(plain.risk, got.risk, "{name}: se does not move the risk");
            assert!(plain.std_err.is_none() && plain.lower.is_none());
            let n_times = case.times.len();
            let z = normal::quantile(0.975);
            naive_case.product_limit = res.product_limit;
            let last: Vec<f64> = (0..n_strata)
                .map(|s| {
                    (0..n)
                        .filter(|&i| stratum_dense[i] == s)
                        .map(|i| case.time[i])
                        .fold(f64::NEG_INFINITY, f64::max)
                })
                .collect();
            for r in 0..got.n_rows {
                let s = case.new_stratum.as_ref().map_or(0, |ns| {
                    let mut levels = case.stratum.clone().unwrap();
                    levels.sort_unstable();
                    levels.dedup();
                    levels.binary_search(&ns[r]).unwrap()
                });
                // The direct evaluation, at the counts of the times.
                let theta: Vec<f64> = (0..naive_case.models.len())
                    .map(|m| {
                        let xr: Vec<f64> = (0..new_designs[m].ncols())
                            .map(|a| new_designs[m][(r, a)])
                            .collect();
                        let mut eta = 0.0;
                        for (a, b) in naive_case.models[m].beta.iter().enumerate() {
                            eta += xr[a] * b;
                        }
                        eta.exp()
                    })
                    .collect();
                let xr: Vec<Vec<f64>> = (0..naive_case.models.len())
                    .map(|m| {
                        (0..new_designs[m].ncols())
                            .map(|a| new_designs[m][(r, a)])
                            .collect()
                    })
                    .collect();
                let counts: Vec<usize> = case
                    .times
                    .iter()
                    .map(|&t| grid.partition_point(|&g| g <= t))
                    .collect();
                let want_counts: Vec<usize> = counts.iter().copied().filter(|&c| c > 0).collect();
                naive_case.exact = false;
                let naive = naive_se(&naive_case, s, &theta, &xr, &want_counts);
                naive_case.exact = true;
                let naive_exact = naive_se(&naive_case, s, &theta, &xr, &want_counts);
                let mut naive_iter = naive.iter();
                let mut naive_exact_iter = naive_exact.iter();
                for (t_idx, &t) in case.times.iter().enumerate() {
                    let at = r * n_times + t_idx;
                    let expect_risk = res.absrisk[r][t_idx];
                    let expect_se = res.se[r][t_idx];
                    let beyond = t > last[s];
                    // The values at the time.
                    match (got.risk[at], expect_risk) {
                        (Some(g), Some(e)) => {
                            let bound = b_row[r] * e.abs().max(f64::MIN_POSITIVE);
                            let dev = ratio((g - e).abs(), bound);
                            assert!(dev <= 1.0, "{name} risk [{r},{t_idx}]: {g} against {e}");
                            worst.put("risk", dev);
                        }
                        (None, None) => assert!(beyond, "{name}: undefined without cause"),
                        other => panic!("{name} risk [{r},{t_idx}]: {other:?}"),
                    }
                    let sum = got.std_err.as_ref().unwrap()[at];
                    let mut cond = 1.0;
                    if counts[t_idx] > 0 && !beyond {
                        let (se_naive, se_abs) = *naive_iter.next().unwrap();
                        let g = sum.unwrap();
                        // The fast evaluation against the direct one.
                        let fast_bound = b_row[r]
                            * (p_max * kappa).max(1.0)
                            * (se_abs / se_naive).max(1.0)
                            * se_naive.max(f64::MIN_POSITIVE);
                        let dev = ratio((g - se_naive).abs(), fast_bound);
                        assert!(
                            dev <= 1.0,
                            "{name} se vs direct [{r},{t_idx}]: {g} {se_naive}"
                        );
                        worst.put("se against the direct evaluation", dev);
                        cond = (se_abs / se_naive).max(1.0);
                        // The exact form against its own direct evaluation.
                        let (se_x, se_x_abs) = *naive_exact_iter.next().unwrap();
                        let gx = exact.std_err.as_ref().unwrap()[at].unwrap();
                        let fast_bound_x = b_row[r]
                            * (p_max * kappa).max(1.0)
                            * (se_x_abs / se_x).max(1.0)
                            * se_x.max(f64::MIN_POSITIVE);
                        let dev = ratio((gx - se_x).abs(), fast_bound_x);
                        assert!(
                            dev <= 1.0,
                            "{name} exact se vs direct [{r},{t_idx}]: {gx} {se_x}"
                        );
                        worst.put("exact se against the direct evaluation", dev);
                        if res.product_limit {
                            worst.put(
                                "exact se against the first-order se, relative",
                                (gx - g).abs() / g,
                            );
                        } else {
                            worst.put(
                                "exact se without the product limit against the first-order se, relative",
                                (gx - g).abs() / g,
                            );
                        }
                    }
                    match (sum, expect_se) {
                        (Some(g), Some(e)) => {
                            if any_tied {
                                worst.put("se of R with tied events, relative", (g - e).abs() / e);
                                continue;
                            }
                            let b_se = b_row[r] * (p_max * kappa).max(1.0) * cond;
                            let bound = b_se * e.abs().max(f64::MIN_POSITIVE);
                            let dev = ratio((g - e).abs(), bound);
                            assert!(dev <= 1.0, "{name} se [{r},{t_idx}]: {g} against {e}");
                            worst.put("se", dev);
                            compared += 1;
                            // the limits
                            let f = expect_risk.unwrap();
                            let kinds = [
                                (RiskLimits::LogLog, &res.lower, &res.upper, "loglog"),
                                (RiskLimits::Plain, &res.lower_none, &res.upper_none, "none"),
                                (RiskLimits::Log, &res.lower_log, &res.upper_log, "log"),
                                (
                                    RiskLimits::CLogLog,
                                    &res.lower_cloglog,
                                    &res.upper_cloglog,
                                    "cloglog",
                                ),
                            ];
                            for (kind, lo, up, label) in kinds {
                                let lim = model
                                    .predict(
                                        &new,
                                        &case.times,
                                        case.cause,
                                        &RiskOptions {
                                            limits: kind,
                                            ..ropts
                                        },
                                    )
                                    .unwrap();
                                let (bl, bu) = limit_bound(kind, f, e, z, b_row[r], b_se);
                                for (got_l, want_l, b, side) in [
                                    (lim.lower.as_ref().unwrap()[at], lo[r][t_idx], bl, "lower"),
                                    (lim.upper.as_ref().unwrap()[at], up[r][t_idx], bu, "upper"),
                                ] {
                                    match (got_l, want_l) {
                                        (Some(g), Some(w)) => {
                                            let bound = b.max(8.0 * eps * w.abs());
                                            let dev = ratio((g - w).abs(), bound);
                                            assert!(
                                                dev <= 1.0,
                                                "{name} {label} {side} [{r},{t_idx}]: {g} against {w}"
                                            );
                                            worst.put(&format!("limit {label}"), dev);
                                        }
                                        (None, None) => {}
                                        // R gives a number for a limit on the
                                        // boundary of the unit interval where
                                        // the scale is undefined.
                                        (None, Some(w)) => assert!(
                                            w.is_nan() || f <= 0.0 || f >= 1.0,
                                            "{name} {label} {side} [{r},{t_idx}]: None against {w}"
                                        ),
                                        (Some(g), None) => panic!(
                                            "{name} {label} {side} [{r},{t_idx}]: {g} against NA"
                                        ),
                                    }
                                }
                            }
                        }
                        (None, None) => {}
                        other => panic!("{name} se [{r},{t_idx}]: {other:?}"),
                    }
                }
            }
            // Without a standard error the risk is the same.
            let nose = &res.absrisk_nose;
            for (r, row) in nose.iter().enumerate() {
                for (t_idx, &expected) in row.iter().enumerate() {
                    match (plain.risk[r * n_times + t_idx], expected) {
                        (Some(g), Some(e)) => {
                            let dev =
                                ratio((g - e).abs(), b_row[r] * e.abs().max(f64::MIN_POSITIVE));
                            assert!(dev <= 1.0, "{name} risk without se");
                        }
                        (None, None) => {}
                        other => panic!("{name} risk without se: {other:?}"),
                    }
                }
            }
        }
    }
    eprintln!("{compared} standard errors compared with R");
    assert!(compared > 100);
    let note = &reference.ties_note;
    assert!(note.untied.max_diff <= 8.0 * eps * note.untied.max_abs.max(1.0));
    assert!(
        note.tied.max_diff > note.tied.max_abs * 0.1,
        "the influence function of riskRegression differs from dfbeta with tied events"
    );
    for (what, value) in &worst.0 {
        eprintln!("worst {what}: {value:.3e} of its bound");
    }
}

/// The options of the models, as the library builds them, for the direct
/// evaluation. The event code is `None` for the model of any event.
fn opts_for(
    opts: &CscOptions,
    n_codes: u32,
    hazard_type: bool,
    cause: u32,
) -> Vec<CoxOptions<'static>> {
    let base = CoxOptions {
        ties: opts.ties,
        timefix: opts.timefix,
        ..CoxOptions::default()
    };
    if hazard_type {
        (1..=n_codes)
            .map(|code| CoxOptions {
                cause: Some(code),
                ..base
            })
            .collect()
    } else {
        vec![
            CoxOptions {
                cause: Some(cause),
                ..base
            },
            base,
        ]
    }
}
