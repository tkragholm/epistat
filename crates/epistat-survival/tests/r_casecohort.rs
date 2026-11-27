//! Case-cohort fits checked against R 4.6.2's `survival::cch` on the `nwtco`
//! data of its help page: the five methods, a stratified subcohort, delayed
//! entry, failure times without ties, `robust = TRUE` and Breslow ties (which
//! `cch` has no option for, so the reference runs its fitters with a `coxph`
//! that takes `ties`).

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::casecohort::{CaseCohortFit, CaseCohortMethod, CaseCohortOptions, fit};
use epistat_survival::cox::Ties;
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Versions {
    r: String,
    survival: String,
    jsonlite: String,
}

#[derive(Deserialize)]
struct Dataset {
    name: String,
    stop: Vec<f64>,
    start: Vec<f64>,
    counting: bool,
    stratum: Vec<u32>,
    cohort_sizes: Vec<f64>,
}

#[derive(Deserialize)]
struct RFit {
    name: String,
    dataset: String,
    method: String,
    ties: String,
    robust: bool,
    coefficients: Vec<f64>,
    var: Vec<Vec<f64>>,
    phase2var: Vec<Vec<f64>>,
    model_var: Vec<Vec<f64>>,
    iter: usize,
    opt: Option<Vec<Vec<f64>>>,
    kappa: f64,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    n_cohort: f64,
    x: Vec<Vec<f64>>,
    status: Vec<u8>,
    subcohort: Vec<u8>,
    datasets: Vec<Dataset>,
    fits: Vec<RFit>,
}

/// The largest deviation of a matrix from R's in units of
/// `bound * sqrt(e_aa e_bb)`, where `e` is the matrix that R gave.
fn matrix_deviation(name: &str, got: &[Option<f64>], expected: &[Vec<f64>], bound: f64) -> f64 {
    let p = expected.len();
    assert_eq!(got.len(), p * p, "{name}");
    let mut worst = 0.0f64;
    for a in 0..p {
        for b in 0..p {
            let g = got[a * p + b].unwrap_or_else(|| panic!("{name}: ({a}, {b}) is None"));
            let scale = (expected[a][a].abs() * expected[b][b].abs()).sqrt();
            worst = worst.max((g - expected[a][b]).abs() / (bound * scale));
        }
    }
    worst
}

fn method_of(name: &str) -> CaseCohortMethod {
    match name {
        "Prentice" => CaseCohortMethod::Prentice,
        "SelfPrentice" => CaseCohortMethod::SelfPrentice,
        "LinYing" => CaseCohortMethod::LinYing,
        "I.Borgan" => CaseCohortMethod::BorganI,
        "II.Borgan" => CaseCohortMethod::BorganII,
        other => panic!("unknown method {other}"),
    }
}

#[test]
fn fits_match_cch() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/casecohort.json"
    ));
    eprintln!(
        "reference: {}, survival {}, jsonlite {}",
        reference.versions.r, reference.versions.survival, reference.versions.jsonlite
    );
    let rows = reference.status.len();
    let p = reference.x[0].len();
    let x = Mat::from_fn(rows, p, |i, j| reference.x[i][j]);
    let status: Vec<i32> = reference.status.iter().map(|&s| i32::from(s)).collect();
    let subcohort: Vec<bool> = reference.subcohort.iter().map(|&s| s == 1).collect();
    let eps = f64::EPSILON;
    let mut worst_all = 0.0f64;
    for r in &reference.fits {
        let ds = reference
            .datasets
            .iter()
            .find(|d| d.name == r.dataset)
            .unwrap();
        let mut builder = SurvData::builder(&ds.stop, &status);
        if ds.counting {
            builder = builder.start(&ds.start);
        }
        let data = builder.build().unwrap();
        let method = method_of(&r.method);
        let sizes: Vec<f64> = if matches!(
            method,
            CaseCohortMethod::BorganI | CaseCohortMethod::BorganII
        ) {
            ds.cohort_sizes.clone()
        } else {
            vec![reference.n_cohort]
        };
        let mut opts = CaseCohortOptions::new(method, &sizes);
        opts.robust = r.robust;
        opts.ties = if r.ties == "breslow" {
            Ties::Breslow
        } else {
            Ties::Efron
        };
        let stratum = (matches!(
            method,
            CaseCohortMethod::BorganI | CaseCohortMethod::BorganII
        ))
        .then_some(ds.stratum.as_slice());
        let got: CaseCohortFit = fit(&data, x.as_ref(), &subcohort, stratum, &opts).unwrap();
        assert!(got.converged, "{}", r.name);
        assert_eq!(got.iterations, r.iter, "{}", r.name);

        // Two implementations of the same Newton iterations differ by the
        // solve, the condition number of the information times the machine
        // epsilon, 8 times that as in `r_cox.rs`, plus the accumulation of
        // rounding in the sums over the rows of the information and the
        // score, which grows as the square root of the number of terms, 4
        // times `sqrt(n) eps`. The weighted fits and the stacked rows with
        // an offset of -100 sum terms of different sizes, which the solve
        // term does not cover. The coefficient is relative to the larger of
        // its size and 1.
        let bound = 8.0 * r.kappa * eps + 4.0 * (rows as f64).sqrt() * eps;
        let mut coefficient = 0.0f64;
        for (g, e) in got.coefficients.iter().zip(&r.coefficients) {
            coefficient = coefficient.max((g.unwrap() - e).abs() / (bound * e.abs().max(1.0)));
        }
        // The covariance matrices are products of the inverse information
        // (error `p * bound`) with sums of `dfbeta` or score cross-products,
        // each `dfbeta` itself a product with the inverse information, so the
        // bound is `p` times three factors of `bound`.
        let cov_bound = 3.0 * (p as f64) * bound;
        let model = matrix_deviation(&r.name, &got.naive_covariance, &r.model_var, cov_bound);
        let phase2 = matrix_deviation(&r.name, &got.phase_two_covariance, &r.phase2var, cov_bound);
        let var = matrix_deviation(&r.name, &got.covariance, &r.var, cov_bound);
        let opt = match (&got.optimal_allocation, &r.opt) {
            (Some(g), Some(e)) => {
                let mut worst = 0.0f64;
                for (j, row) in e.iter().enumerate() {
                    for (a, &v) in row.iter().enumerate() {
                        worst = worst.max((g[j * p + a].unwrap() - v).abs() / cov_bound);
                    }
                }
                worst
            }
            (None, None) => 0.0,
            _ => panic!("{}: allocation present on one side only", r.name),
        };
        let worst = coefficient.max(model).max(phase2).max(var).max(opt);
        eprintln!(
            "{:<26} coefficient {coefficient:.3} model {model:.3} phase2 {phase2:.3} var {var:.3} opt {opt:.3}",
            r.name
        );
        assert!(
            worst <= 1.0,
            "{}: deviation {worst} of its tolerance",
            r.name
        );
        worst_all = worst_all.max(worst);
    }
    eprintln!("worst deviation over tolerance: {worst_all:.3}");
}
