//! Cox fits on a design matrix checked against R 4.6.2's `survival::coxph`.

use epistat_survival::cox::{CoxDesign, CoxModelFit, Ties, cox_score_residuals, fit_cox};
use epistat_testkit::{load_json, na_f64_vec};
use faer::Mat;
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
fn flat(values: &[Option<f64>]) -> Vec<f64> {
    values.iter().map(|v| v.unwrap_or(f64::NAN)).collect()
}

/// The largest deviation of `got` from `expected` in units of the tolerance
/// `bound * max(|expected|, 1)`, and a check that `None` and NA coincide.
fn deviation(got: &[f64], expected: &[f64], bound: f64) -> f64 {
    assert_eq!(got.len(), expected.len());
    let mut worst = 0.0f64;
    for (g, e) in got.iter().zip(expected) {
        assert_eq!(
            g.is_nan(),
            e.is_nan(),
            "aliased columns differ: {got:?} {expected:?}"
        );
        if !e.is_nan() {
            worst = worst.max((g - e).abs() / (bound * e.abs().max(1.0)));
        }
    }
    worst
}

/// The fit and the score residuals at its coefficients, one vector per column.
fn fit_of(reference: &Reference, r: &RFit) -> (CoxModelFit, Vec<Option<Vec<f64>>>) {
    let indices: Vec<usize> = r
        .columns
        .iter()
        .map(|c| reference.column_names.iter().position(|n| n == c).unwrap())
        .collect();
    let rows = reference.stop.len();
    let x = Mat::from_fn(rows, indices.len(), |i, j| reference.columns[i][indices[j]]);
    let is_event: Vec<bool> = reference.event.iter().map(|&e| e == 1).collect();
    let start = if r.use_start {
        reference.start.clone()
    } else {
        vec![f64::NEG_INFINITY; rows]
    };
    let stratum = if r.use_strata {
        reference.stratum.clone()
    } else {
        vec![0; rows]
    };
    let weight = if r.use_weights {
        reference.weight.clone()
    } else {
        vec![1.0; rows]
    };
    let design = CoxDesign {
        x: x.as_ref(),
        is_event: &is_event,
        start: &start,
        stop: &reference.stop,
        weight: &weight,
        stratum: &stratum,
        ties: if r.ties == "efron" {
            Ties::Efron
        } else {
            Ties::Breslow
        },
        cluster: r.use_cluster.then_some(reference.id.as_slice()),
    };
    let fit = fit_cox(&design).unwrap();
    let residuals = cox_score_residuals(&design, &fit.coefficients).unwrap();
    (fit, residuals)
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
/// tolerance is `p` times the bound. The robust matrix is `V M V` with the
/// error of `V` entering twice, so its tolerance is twice that. The score
/// residuals are described at `residual_deviation`.
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
            assert!(
                fit.standard_errors.iter().all(Option::is_none),
                "{}",
                r.name
            );
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
        let coefficient = deviation(&flat(&fit.coefficients), &r.coefficients, bound);
        let se = deviation(&flat(&fit.standard_errors), &r.se, bound);
        let robust = if r.use_cluster {
            deviation(
                &flat(fit.robust_standard_errors.as_ref().unwrap()),
                &r.robust_se,
                bound,
            )
        } else {
            assert!(fit.robust_standard_errors.is_none());
            0.0
        };
        let loglik = deviation(&[fit.log_partial_likelihood.unwrap()], &[r.loglik], bound);
        let null = deviation(
            &[fit.null_log_partial_likelihood.unwrap()],
            &[r.loglik_null],
            bound,
        );
        // The deviations in absolute terms, relative to the value or to 1.
        let p = r.columns.len() as f64;
        let covariance = covariance_deviation(&fit.covariance, &r.vcov, p * bound);
        let robust_covariance = if r.use_cluster {
            covariance_deviation(
                fit.robust_covariance.as_ref().unwrap(),
                &r.vcov_robust,
                2.0 * p * bound,
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
