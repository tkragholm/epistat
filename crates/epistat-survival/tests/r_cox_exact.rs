//! Cox fits with exact ties and conditional logistic regression, checked
//! against R 4.6.2's `survival::coxph(ties = "exact")` and `survival::clogit`.
//!
//! Two implementations of one Newton iteration that take the same steps differ
//! by the error of each solve, about the condition number of the information
//! matrix on the scaled covariates times the machine epsilon, and here also by
//! the error of the sums over the subsets of a risk set. For data without entry
//! times R forms these by the recursion of Gail, Lubin and Rubinstein, with
//! another association of the sums than ours. A sum of positive terms carries a
//! relative error of at most the number of additions times the machine epsilon,
//! and the recursion adds at most `n_risk_max * d_max` terms into one value, so
//! the tolerance adds twice that product times the machine epsilon, relative, to
//! the 8 times the condition number of `r_cox_control.rs`. For data with entry
//! times R enumerates the subsets and adds them one at a time, `N` of them at
//! the largest risk set. The worst case error of that sum is `N` times the
//! machine epsilon, which is far above what occurs, and the rounding errors of
//! a sum of `N` terms are independent, so the standard deviation of the error is
//! `sqrt(N)` times the machine epsilon. The tolerance for these fits adds that,
//! in place of the recursion's term. R does not scale the covariates of these
//! fits, so its solves have the condition number of the unscaled information
//! matrix, which the reference does not hold. The terms above are the errors of
//! the sums that enter that matrix and its inverse, and the tolerance of a
//! fit counts them once.

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::clogit::{ClogitMethod, ClogitOptions, clogit};
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
struct Dataset {
    columns: Vec<Vec<f64>>,
    column_names: Vec<String>,
    start: Vec<f64>,
    stop: Vec<f64>,
    event: Vec<u8>,
    stratum: Vec<u32>,
    offset: Vec<f64>,
}

#[derive(Deserialize)]
struct Datasets {
    tied: Dataset,
    small: Dataset,
    infert: Dataset,
    matched: Dataset,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    datasets: Datasets,
    fits: Vec<RFit>,
}

#[derive(Deserialize)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "one flag per option of the R call"
)]
struct RFit {
    name: String,
    data: String,
    kind: String,
    columns: Vec<String>,
    method: String,
    use_start: bool,
    use_strata: bool,
    use_offset: bool,
    init: Vec<f64>,
    iter_max: u32,
    eps: f64,
    #[serde(deserialize_with = "na_f64_vec")]
    coefficients: Vec<f64>,
    #[serde(deserialize_with = "na_f64_vec")]
    se: Vec<f64>,
    vcov: Vec<Vec<Option<f64>>>,
    loglik_init: f64,
    loglik: f64,
    iterations: u32,
    rank: usize,
    ran_out: bool,
    monotone_columns: Vec<usize>,
    score: f64,
    wald: Option<f64>,
    n: usize,
    nevent: f64,
    d_max: f64,
    n_risk_max: f64,
    subsets_max: f64,
    kappa_fit: Option<f64>,
    kappa_init: Option<f64>,
    kappa_path: Vec<f64>,
}

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

fn test_value(
    name: &str,
    got: Option<ChiSquareTest>,
    expected: f64,
    rank: usize,
    bound: f64,
) -> f64 {
    let got = got.unwrap_or_else(|| panic!("{name}: the test is missing"));
    assert_eq!(got.df, rank, "{name}");
    (got.statistic - expected).abs() / (bound * expected.abs().max(1.0))
}

fn run(reference: &Reference, r: &RFit) -> CoxModelFit {
    let dataset = match r.data.as_str() {
        "tied" => &reference.datasets.tied,
        "small" => &reference.datasets.small,
        "infert" => &reference.datasets.infert,
        "matched" => &reference.datasets.matched,
        other => panic!("unknown data set {other}"),
    };
    let rows = dataset.stop.len();
    let indices: Vec<usize> = r
        .columns
        .iter()
        .map(|c| dataset.column_names.iter().position(|n| n == c).unwrap())
        .collect();
    let x = Mat::from_fn(rows, indices.len(), |i, j| dataset.columns[i][indices[j]]);
    let offset = r.use_offset.then_some(dataset.offset.as_slice());
    let init = (!r.init.is_empty()).then_some(r.init.as_slice());
    if r.kind == "clogit" {
        let case: Vec<bool> = dataset.event.iter().map(|&e| e == 1).collect();
        let opts = ClogitOptions {
            method: match r.method.as_str() {
                "exact" => ClogitMethod::Exact,
                "approximate" => ClogitMethod::Approximate,
                "efron" => ClogitMethod::Efron,
                "breslow" => ClogitMethod::Breslow,
                other => panic!("unknown method {other}"),
            },
            offset,
            init,
            iter_max: r.iter_max,
            eps: r.eps,
            ..ClogitOptions::default()
        };
        return clogit(&case, &dataset.stratum, x.as_ref(), &opts).unwrap();
    }
    let status: Vec<i32> = dataset.event.iter().map(|&e| i32::from(e)).collect();
    let mut builder = SurvData::builder(&dataset.stop, &status);
    if r.use_start {
        builder = builder.start(&dataset.start);
    }
    if r.use_strata {
        builder = builder.stratum(&dataset.stratum);
    }
    let data = builder.build().unwrap();
    let opts = CoxOptions {
        ties: match r.method.as_str() {
            "exact" => Ties::Exact,
            "efron" => Ties::Efron,
            "breslow" => Ties::Breslow,
            other => panic!("unknown method {other}"),
        },
        offset,
        init,
        iter_max: r.iter_max,
        eps: r.eps,
        ..CoxOptions::default()
    };
    fit(&data, x.as_ref(), &opts).unwrap()
}

#[test]
fn fits_match_r() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/cox_exact.json"
    ));
    eprintln!(
        "reference: {}, survival {}, jsonlite {}",
        reference.versions.r, reference.versions.survival, reference.versions.jsonlite
    );
    let eps = f64::EPSILON;
    let mut lines = Vec::new();
    for r in &reference.fits {
        let model = run(&reference, r);
        let name = &r.name;
        let p = r.columns.len() as f64;
        let n = r.n as f64;
        // The error of the sums over the subsets of a risk set, relative.
        let recursion = if r.use_start && r.method == "exact" {
            r.subsets_max.sqrt() * eps
        } else {
            2.0 * r.n_risk_max * r.d_max * eps
        };
        let kappa_init = r.kappa_init.unwrap_or(1.0);
        let score_bound = 8.0 * kappa_init * eps + n * eps + recursion;
        let rank = r.rank;
        let mut worst = vec![(
            "score",
            test_value(name, model.tests.score, r.score, rank, score_bound),
        )];

        assert_eq!(model.n_rows, r.n, "{name}");
        assert_eq!(model.n_events.to_bits(), r.nevent.to_bits(), "{name}");
        if !r.monotone_columns.is_empty() && r.iter_max > 1 {
            // R reports a finite coefficient with a warning; the fit reports
            // none, names the same columns and keeps only the score tests.
            let columns: Vec<usize> = r.monotone_columns.iter().map(|c| c - 1).collect();
            assert_eq!(model.monotone_columns, columns, "{name}");
            assert!(model.coefficients.iter().all(Option::is_none), "{name}");
            assert!(model.tests.wald.is_none(), "{name}");
        } else {
            let kappa_fit = r.kappa_fit.unwrap_or(1.0);
            let bound = if model.converged || r.iter_max == 0 {
                8.0 * kappa_fit.max(kappa_init) * eps + recursion
            } else {
                8.0 * r.kappa_path.iter().sum::<f64>() * eps + recursion
            };
            assert_eq!(model.rank, r.rank, "{name}");
            assert_eq!(model.n_used, r.n, "{name}");
            if r.iter_max >= 2 {
                assert_eq!(model.converged, !r.ran_out, "{name}");
            } else if r.iter_max == 0 {
                assert!(!model.converged, "{name}");
            }
            assert_eq!(model.iterations, r.iterations, "{name}");
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
            let lr_tolerance = 2.0 * bound * r.loglik.abs().max(1.0)
                + 2.0 * (n * eps + recursion) * r.loglik_init.abs().max(1.0);
            let lr = model.tests.likelihood_ratio.expect("likelihood ratio");
            assert_eq!(lr.df, r.rank, "{name}");
            let lr_value = (lr.statistic - 2.0 * (r.loglik - r.loglik_init)).abs() / lr_tolerance;
            let wald = test_value(
                name,
                model.tests.wald,
                r.wald.unwrap(),
                r.rank,
                2.0 * p * bound,
            );
            worst.extend([
                ("coefficient", coefficient),
                ("se", se),
                ("covariance", covariance),
                ("loglik", loglik),
                ("null loglik", null),
                ("likelihood ratio", lr_value),
                ("wald", wald),
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
            "{name:28} {}",
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
