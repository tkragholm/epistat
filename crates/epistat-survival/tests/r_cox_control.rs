//! Cox fits with offsets, starting values and iteration control, and the Wald,
//! score and likelihood-ratio tests, checked against R 4.6.2's
//! `survival::coxph` and `summary.coxph`.

use epistat_linalg::Mat;
use epistat_stats::gamma::chi_square_sf;
use epistat_survival::SurvData;
use epistat_survival::cox::{ChiSquareTest, CoxModelFit, CoxOptions, Ties, fit};
use epistat_testkit::{NanRule, load_json, na_f64_vec, none_as_nan, worst_deviation};
use serde::Deserialize;

#[derive(Deserialize)]
struct Versions {
    r: String,
    survival: String,
    jsonlite: String,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    columns: Vec<Vec<f64>>,
    column_names: Vec<String>,
    start: Vec<f64>,
    stop: Vec<f64>,
    event: Vec<u8>,
    stratum: Vec<u32>,
    weight: Vec<f64>,
    id: Vec<u32>,
    fits: Vec<RFit>,
}

#[derive(Deserialize)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "one flag per option of the R call"
)]
struct RFit {
    name: String,
    columns: Vec<String>,
    ties: String,
    use_start: bool,
    use_strata: bool,
    use_weights: bool,
    use_cluster: bool,
    offset: Vec<f64>,
    init: Vec<f64>,
    iter_max: u32,
    eps: f64,
    #[serde(deserialize_with = "na_f64_vec")]
    coefficients: Vec<f64>,
    #[serde(deserialize_with = "na_f64_vec")]
    se: Vec<f64>,
    #[serde(deserialize_with = "na_f64_vec")]
    robust_se: Vec<f64>,
    vcov: Vec<Vec<Option<f64>>>,
    loglik_init: f64,
    loglik: f64,
    iterations: u32,
    rank: usize,
    ran_out: bool,
    monotone_columns: Vec<usize>,
    score: f64,
    wald: Option<f64>,
    wald_model: Option<f64>,
    /// The robust score test from the score residuals at the starting values
    /// with the offset in.
    robust_score_at_init: Option<f64>,
    kappa_fit: f64,
    kappa_init: f64,
    kappa_path: Vec<f64>,
    kappa_meat: Option<f64>,
    kappa_robust: Option<f64>,
}

/// Deviation of a covariance matrix in units of `bound * sqrt(e_aa e_bb)`.
fn covariance_deviation(got: &[Option<f64>], expected: &[Vec<Option<f64>>], bound: f64) -> f64 {
    let p = expected.len();
    assert_eq!(got.len(), p * p);
    let mut worst = 0.0f64;
    for a in 0..p {
        for b in 0..p {
            let (g, e) = (got[a * p + b], expected[a][b]);
            assert_eq!(g.is_none(), e.is_none(), "entry ({a}, {b}): {g:?} {e:?}");
            if let (Some(g), Some(e)) = (g, e) {
                let scale = (expected[a][a].unwrap() * expected[b][b].unwrap()).sqrt();
                worst = worst.max((g - e).abs() / (bound * scale));
            }
        }
    }
    worst
}

/// A test against R's statistic, in units of `bound * max(statistic, 1)`. The
/// degrees of freedom are the rank, and the p-value is the upper tail of the
/// chi-square distribution at R's statistic, to compare like with like.
fn test_deviation(
    name: &str,
    got: Option<ChiSquareTest>,
    expected: f64,
    rank: usize,
    bound: f64,
) -> f64 {
    let got = got.unwrap_or_else(|| panic!("{name}: the test is missing"));
    assert_eq!(got.df, rank, "{name}");
    let value = (got.statistic - expected).abs() / (bound * expected.abs().max(1.0));
    // The p-value follows from the statistic by the chi-square function that
    // `pchisq` computes, so its deviation is that of the statistic times the
    // density, which is below 0.5 for one degree of freedom or more, relative
    // to the value of the statistic.
    let from_r = chi_square_sf(expected, rank as f64);
    let p_value = (got.p_value - from_r).abs() / (bound * expected.abs().max(1.0));
    value.max(p_value)
}

#[test]
fn fits_match_r() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/cox_control.json"
    ));
    eprintln!(
        "reference: {}, survival {}, jsonlite {}",
        reference.versions.r, reference.versions.survival, reference.versions.jsonlite
    );
    let rows = reference.stop.len();
    let eps = f64::EPSILON;
    let mut lines = Vec::new();
    for r in &reference.fits {
        let indices: Vec<usize> = r
            .columns
            .iter()
            .map(|c| reference.column_names.iter().position(|n| n == c).unwrap())
            .collect();
        let x = Mat::from_fn(rows, indices.len(), |i, j| reference.columns[i][indices[j]]);
        let status: Vec<i32> = reference.event.iter().map(|&e| i32::from(e)).collect();
        let mut builder = SurvData::builder(&reference.stop, &status);
        if r.use_start {
            builder = builder.start(&reference.start);
        }
        if r.use_strata {
            builder = builder.stratum(&reference.stratum);
        }
        if r.use_weights {
            builder = builder.weight(&reference.weight);
        }
        if r.use_cluster {
            builder = builder.cluster(&reference.id);
        }
        let data = builder.build().unwrap();
        let opts = CoxOptions {
            ties: if r.ties == "efron" {
                Ties::Efron
            } else {
                Ties::Breslow
            },
            offset: (!r.offset.is_empty()).then_some(r.offset.as_slice()),
            init: (!r.init.is_empty()).then_some(r.init.as_slice()),
            iter_max: r.iter_max,
            eps: r.eps,
            ..CoxOptions::default()
        };
        let model: CoxModelFit = fit(&data, x.as_ref(), &opts).unwrap();
        let name = &r.name;
        let p = r.columns.len() as f64;
        let n = rows as f64;

        // The score test is at the starting values, which no iteration
        // changes, so it is defined also when a coefficient diverges. Its
        // error is that of one solve with the information there, `kappa_init`
        // times the machine epsilon, and of the sums over the rows in the score,
        // `n` times it.
        let score_bound = 8.0 * r.kappa_init * eps + n * eps;
        let rank = r.rank;
        let score = test_deviation(name, model.tests.score, r.score, rank, score_bound);
        let robust_score =
            if let (Some(expected), Some(kappa)) = (r.robust_score_at_init, r.kappa_meat) {
                let bound = 8.0 * kappa * eps + n * eps;
                test_deviation(
                    name,
                    model.tests.robust_score,
                    expected,
                    rank,
                    bound.max(score_bound),
                )
            } else {
                assert!(model.tests.robust_score.is_none(), "{name}");
                0.0
            };
        let mut worst = vec![("score", score), ("robust score", robust_score)];

        if !r.monotone_columns.is_empty() && r.iter_max > 1 {
            // R reports a finite coefficient with a warning; the fit reports
            // none, names the same columns and keeps only the score tests.
            let columns: Vec<usize> = r.monotone_columns.iter().map(|c| c - 1).collect();
            assert_eq!(model.monotone_columns, columns, "{name}");
            assert!(model.coefficients.iter().all(Option::is_none), "{name}");
            assert!(model.tests.wald.is_none(), "{name}");
            assert!(model.tests.likelihood_ratio.is_none(), "{name}");
        } else {
            // Two implementations of one Newton iteration that take the same
            // steps differ by the error of each solve, about the condition
            // number of the information matrix times the machine epsilon, 8
            // times that as in `r_cox.rs`. The condition number is the larger of
            // the one at the fit and at the starting values. A fit that has
            // converged has contracted the errors of its early steps. One that
            // stopped at the iteration limit has not, so its bound is 8
            // times the sum of the condition numbers at the points where a
            // solve was made, which R's script records.
            let bound = if model.converged || r.iter_max == 0 {
                8.0 * r.kappa_fit.max(r.kappa_init) * eps
            } else {
                8.0 * r.kappa_path.iter().sum::<f64>() * eps
            };
            assert_eq!(model.rank, r.rank, "{name}");
            if r.iter_max >= 2 {
                assert_eq!(model.converged, !r.ran_out, "{name}");
            } else if r.iter_max == 0 {
                assert!(!model.converged, "{name}");
            }
            // coxph.fit, which has no entry times, counts one trial point more
            // than the limit when it runs out of iterations.
            let expected_iterations = if !r.use_start && r.iter_max > 0 && !model.converged {
                r.iterations - 1
            } else {
                r.iterations
            };
            assert_eq!(model.iterations, expected_iterations, "{name}");
            // The error of a Newton solve is the condition number times the
            // machine epsilon times the size of the step, and the first steps
            // are as long as the distance from the starting values to the
            // estimate. So the tolerance of a coefficient is relative to the
            // largest of its value, its starting value and 1.
            let coefficient = {
                let got = none_as_nan(&model.coefficients);
                let mut worst = 0.0f64;
                for (a, (g, e)) in got.iter().zip(&r.coefficients).enumerate() {
                    assert_eq!(g.is_nan(), e.is_nan(), "{name}");
                    if !e.is_nan() {
                        let start = r.init.get(a).map_or(0.0, |v| v.abs());
                        worst = worst.max((g - e).abs() / (bound * e.abs().max(start).max(1.0)));
                    }
                }
                worst
            };
            let se = worst_deviation(
                &none_as_nan(&model.standard_errors),
                &r.se,
                bound,
                1.0,
                NanRule::Coincide,
            );
            let covariance = covariance_deviation(&model.covariance, &r.vcov, p * bound);
            let robust = if r.use_cluster {
                worst_deviation(
                    &none_as_nan(model.robust_standard_errors.as_ref().unwrap()),
                    &r.robust_se,
                    bound,
                    1.0,
                    NanRule::Coincide,
                )
            } else {
                0.0
            };
            let loglik = worst_deviation(
                &[model.log_partial_likelihood.unwrap()],
                &[r.loglik],
                bound,
                1.0,
                NanRule::Coincide,
            );
            let null = if r.init.is_empty() {
                worst_deviation(
                    &[model.null_log_partial_likelihood.unwrap()],
                    &[r.loglik_init],
                    bound,
                    1.0,
                    NanRule::Coincide,
                )
            } else {
                0.0
            };
            // The likelihood-ratio statistic is twice the difference of two
            // log partial likelihoods. The one at the fit is off by `bound`
            // relative to its size and the one at the starting values by the
            // rounding error of a sum over `n` rows.
            let lr_tolerance = 2.0 * bound * r.loglik.abs().max(1.0)
                + 2.0 * n * eps * r.loglik_init.abs().max(1.0);
            let lr = model.tests.likelihood_ratio.expect("likelihood ratio");
            assert_eq!(lr.df, r.rank, "{name}");
            let lr_expected = -2.0 * (r.loglik_init - r.loglik);
            let lr_value = (lr.statistic - lr_expected).abs() / lr_tolerance;
            // The Wald statistic is a quadratic form in the distance from the
            // starting values, off by `2 p bound` relative to its size (the
            // distance by `bound` and the information by `p bound`).
            let wald_expected = r.wald_model.unwrap();
            let wald = test_deviation(
                name,
                model.tests.wald,
                wald_expected,
                r.rank,
                2.0 * p * bound,
            );
            let robust_wald = if r.use_cluster {
                // The statistic is `d' V^-1 d` with `V` the robust covariance.
                // To first order a change `dV` of `V` changes it by
                // `z' dV z` with `z = V^-1 d`, which relative to the statistic
                // is at most `kappa(V)` times the relative size of `dV`. That
                // size is the tolerance of the robust covariance,
                // `4 p bound + 2 n eps`, as in `r_cox.rs`. The solve adds
                // `8 kappa eps`. For a well-conditioned `V` this is a few
                // times the covariance tolerance. For `near_alias`, with a
                // condition number of 5e8, it is larger than the statistic
                // and so no check at all; the deviation is reported.
                let kappa = r.kappa_robust.unwrap();
                let bound_robust = kappa * (4.0 * p * bound + 2.0 * n * eps) + 8.0 * kappa * eps;
                test_deviation(
                    name,
                    model.tests.robust_wald,
                    r.wald.unwrap(),
                    r.rank,
                    bound_robust,
                )
            } else {
                assert!(model.tests.robust_wald.is_none(), "{name}");
                0.0
            };
            worst.extend([
                ("coefficient", coefficient),
                ("se", se),
                ("covariance", covariance),
                ("robust se", robust),
                ("loglik", loglik),
                ("null loglik", null),
                ("likelihood ratio", lr_value),
                ("wald", wald),
                ("robust wald", robust_wald),
            ]);
        }
        for (what, value) in &worst {
            assert!(
                *value <= 1.0,
                "{name} {what}: {value} of the bound\n{model:?}\n{:?} {:?}",
                r.coefficients,
                r.se
            );
        }
        lines.push(format!(
            "{name:24} {}",
            worst
                .iter()
                .map(|(w, v)| format!("{w} {v:.2e}"))
                .collect::<Vec<_>>()
                .join("  ")
        ));
    }
    for line in lines {
        eprintln!("{line}");
    }
}
