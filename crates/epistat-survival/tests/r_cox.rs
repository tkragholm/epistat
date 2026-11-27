//! Cox fits on a design matrix checked against R 4.6.2's `survival::coxph`.

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::cox::{CoxModelFit, CoxOptions, Ties, fit, score_residuals};
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
    #[serde(deserialize_with = "na_f64_vec")]
    coefficients: Vec<f64>,
    #[serde(deserialize_with = "na_f64_vec")]
    se: Vec<f64>,
    #[serde(deserialize_with = "na_f64_vec")]
    robust_se: Vec<f64>,
    loglik: f64,
    loglik_null: f64,
    iterations: u32,
    rank: usize,
    monotone_columns: Vec<usize>,
    information_condition: f64,
    /// Model-based covariance matrix, row by row, null in an aliased row or
    /// column.
    vcov: Vec<Vec<Option<f64>>>,
    /// Cluster-robust covariance matrix, empty without a cluster.
    vcov_robust: Vec<Vec<Option<f64>>>,
    /// `residuals(fit, "score")`, which leaves the case weight out. One row per
    /// observation, null in an aliased column.
    score_residuals: Vec<Vec<Option<f64>>>,
    /// `residuals(fit, "score", weighted = TRUE)`, which sum to the score.
    score_residuals_weighted: Vec<Vec<Option<f64>>>,
}

/// The values, with NaN where the fit has `None`.
/// The fit and the score residuals at its coefficients, one vector per column.
fn fit_of(reference: &Reference, r: &RFit) -> (CoxModelFit, Vec<Option<Vec<f64>>>) {
    let indices: Vec<usize> = r
        .columns
        .iter()
        .map(|c| reference.column_names.iter().position(|n| n == c).unwrap())
        .collect();
    let rows = reference.stop.len();
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
        ..CoxOptions::default()
    };
    let model = fit(&data, x.as_ref(), &opts).unwrap();
    assert_eq!(model.n_rows, rows, "{}", r.name);
    let residuals = score_residuals(&data, x.as_ref(), &model.coefficients, &opts).unwrap();
    (model, residuals)
}

/// The largest deviation of a covariance matrix from R's, in units of
/// `bound * sqrt(e_aa * e_bb)` for the entry `(a, b)`, and a check that `None`
/// and NA coincide, entry by entry. `got` is row-major.
///
/// The scale of an entry is the geometric mean of the two variances, so that
/// the tolerance of an off-diagonal entry follows the standard errors of its
/// two columns.
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

/// The largest deviation of the score residuals from R's in units of
/// `bound * max_i |e_ia|` for column `a`, with a check that `None` and NA
/// coincide. `expected` has one row per observation, and the residuals are
/// divided by `divisor` row by row before they are compared.
///
/// The residual of a row is a smooth function of the coefficients, with a
/// derivative of the size of the residuals themselves, and the coefficients of
/// the two fits differ by `bound` relative to their scale. A residual then
/// differs by `bound` times the largest residual of its column, once for each
/// of the `p` coefficients. The residual is also a difference of running sums
/// over the event times, as in R, each of at most `rows` terms, and a running
/// sum of `m` terms is off by at most `m` times the machine epsilon times its
/// size, which is the size of the residuals. The `bound` passed in is the sum
/// of the two.
fn residual_deviation(
    got: &[Option<Vec<f64>>],
    expected: &[Vec<Option<f64>>],
    bound: f64,
    divisor: &[f64],
) -> f64 {
    let mut worst = 0.0f64;
    for (a, column) in got.iter().enumerate() {
        match column {
            None => assert!(expected.iter().all(|row| row[a].is_none())),
            Some(values) => {
                assert_eq!(values.len(), expected.len());
                let scale = expected
                    .iter()
                    .map(|row| row[a].unwrap().abs())
                    .fold(0.0f64, f64::max);
                for ((g, row), d) in values.iter().zip(expected).zip(divisor) {
                    worst = worst.max((g / d - row[a].unwrap()).abs() / (bound * scale));
                }
            }
        }
    }
    worst
}

/// Two implementations of one Newton iteration that take the same steps differ
/// by the error of the solve, about the condition number of the information
/// matrix on the scaled covariates times the machine epsilon. The bound is 8
/// times that, relative to the value or to 1 for a value below 1.
///
/// A covariance matrix is the inverse of the information matrix, and each of
/// the `p` entries of a row of the inverse carries that error, so its
/// tolerance is `p` times the bound. The robust matrix is `V M V`. The error of
/// `V` enters twice, `2 p` times the bound, and the meat `M` is a sum of
/// products of score residuals, each off by the residual bound described at
/// `residual_deviation`, `p` times the bound plus `n` units of rounding for the
/// running sums over the risk sets, entering twice. So the robust tolerance is
/// `2 p bound + 2 (p bound + n eps)`, with `n` the rows used. The `n eps` part
/// is negligible when the information is ill-conditioned and decides the
/// tolerance when its condition number is near 1.
#[test]
fn fits_match_r() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/cox_matrix.json"
    ));
    eprintln!(
        "reference: {}, survival {}, jsonlite {}",
        reference.versions.r, reference.versions.survival, reference.versions.jsonlite
    );
    let mut report = Vec::new();
    let mut worst = Vec::new();
    for r in &reference.fits {
        let (fit, residuals) = fit_of(&reference, r);
        let bound = 8.0 * r.information_condition * f64::EPSILON;
        if !r.monotone_columns.is_empty() {
            // R reports a finite coefficient with a warning; the fit reports
            // none, and names the same columns.
            let columns: Vec<usize> = r.monotone_columns.iter().map(|c| c - 1).collect();
            assert_eq!(fit.monotone_columns, columns, "{}", r.name);
            assert!(fit.coefficients.iter().all(Option::is_none), "{}", r.name);
            assert!(fit.se.iter().all(Option::is_none), "{}", r.name);
            assert!(
                fit.log_partial_likelihood.is_none() && !fit.converged,
                "{}",
                r.name
            );
            assert_eq!(fit.iterations, r.iterations, "{}", r.name);
            continue;
        }
        assert!(
            fit.converged && fit.monotone_columns.is_empty(),
            "{}",
            r.name
        );
        assert_eq!(fit.rank, r.rank, "{}", r.name);
        assert_eq!(fit.iterations, r.iterations, "{}", r.name);
        let coefficient = worst_deviation(
            &none_as_nan(&fit.coefficients),
            &r.coefficients,
            bound,
            1.0,
            NanRule::Coincide,
        );
        let se = worst_deviation(&none_as_nan(&fit.se), &r.se, bound, 1.0, NanRule::Coincide);
        let robust = if r.use_cluster {
            worst_deviation(
                &none_as_nan(fit.robust_se.as_ref().unwrap()),
                &r.robust_se,
                bound,
                1.0,
                NanRule::Coincide,
            )
        } else {
            assert!(fit.robust_se.is_none());
            0.0
        };
        let loglik = worst_deviation(
            &[fit.log_partial_likelihood.unwrap()],
            &[r.loglik],
            bound,
            1.0,
            NanRule::Coincide,
        );
        let null = worst_deviation(
            &[fit.null_log_partial_likelihood.unwrap()],
            &[r.loglik_null],
            bound,
            1.0,
            NanRule::Coincide,
        );
        // The deviations in absolute terms, relative to the value or to 1.
        let p = r.columns.len() as f64;
        let covariance = covariance_deviation(&fit.covariance, &r.vcov, p * bound);
        let rows_used = if r.use_weights {
            reference.weight.iter().filter(|&&w| w > 0.0).count()
        } else {
            reference.stop.len()
        } as f64;
        let robust_covariance = if r.use_cluster {
            covariance_deviation(
                fit.robust_covariance.as_ref().unwrap(),
                &r.vcov_robust,
                4.0 * p * bound + 2.0 * rows_used * f64::EPSILON,
            )
        } else {
            assert!(fit.robust_covariance.is_none() && r.vcov_robust.is_empty());
            0.0
        };
        let ones = vec![1.0; reference.stop.len()];
        let residual_bound = p * bound + reference.stop.len() as f64 * f64::EPSILON;
        let weights = if r.use_weights {
            &reference.weight
        } else {
            &ones
        };
        // Weighted, and divided by the case weight as `residuals()` does by
        // default.
        let score_weighted = residual_deviation(
            &residuals,
            &r.score_residuals_weighted,
            residual_bound,
            &ones,
        );
        let score_default =
            residual_deviation(&residuals, &r.score_residuals, residual_bound, weights);
        let score = score_weighted.max(score_default);
        let raw = [coefficient, se, robust, loglik, null].map(|value| value * bound);
        report.push((r.name.clone(), r.information_condition, raw));
        worst.push((r.name.clone(), [covariance, robust_covariance, score]));
        for (what, value) in [
            ("coefficient", coefficient),
            ("se", se),
            ("robust se", robust),
            ("loglik", loglik),
            ("null loglik", null),
            ("covariance", covariance),
            ("robust covariance", robust_covariance),
            ("score residuals", score),
        ] {
            assert!(
                value <= 1.0,
                "{} {what}: {value} of the bound {fit:?} {:?} {:?}",
                r.name,
                r.coefficients,
                r.se
            );
        }
    }
    for (name, kappa, [c, s, rb, l, n]) in &report {
        eprintln!(
            "{name:24} kappa {kappa:9.3e}  coef {c:.2e}  se {s:.2e}  robust {rb:.2e}  loglik {l:.2e}  null {n:.2e}"
        );
    }
    for (name, [covariance, robust, score]) in &worst {
        eprintln!(
            "{name:24} in units of the bound: covariance {covariance:.2e}  robust covariance {robust:.2e}  score residuals {score:.2e}"
        );
    }
}
