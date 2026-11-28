//! The Abadie-Imbens matching estimator, checked against `Matching::Match`
//! 4.10.15.
//!
//! The reference is `tests/data/abadie_imbens.json`, written by
//! `tests/data/abadie_imbens.R`. For `lalonde` (age, educ, married,
//! nodegree and re74 as covariates, re78 as the outcome, with many tied
//! distances) and for a simulated cohort of 300 units without ties, the script
//! runs `Match()` with replacement and ties for the ATT, ATE and ATC, with one
//! and three matches, with and without bias adjustment, with the
//! homoskedastic variance and with two neighbours for the conditional
//! variances, and with the standardised Euclidean and the Mahalanobis
//! distance. It records the estimate, the unadjusted estimate and the standard
//! errors `se`, `se.cond`, `se.standard` and, from a second call with
//! `sample = TRUE`, the standard error of the sample effect. Two more runs on
//! the simulated cohort adjust for regressors that are not the matching
//! covariates.
//!
//! Tolerances. The matches are found from the same distances, so the sets of
//! matches agree and the estimate is a mean of outcome differences that
//! differ from R's only by the order of sums, 1e-12 relative. The standard
//! errors add sums of squares of the same kind, 1e-10 relative. The bias
//! adjustment solves a weighted least squares problem by Cholesky on the normal
//! equations where R inverts them with `solve()`. On `lalonde` the normal
//! equations have a condition number of about 1e9 (re74 is in the thousands
//! next to a constant), which makes the coefficients uncertain at about 1e-7
//! relative, and the adjusted results are compared at 1e-6. The standard
//! error `se.cond` is the square root of a difference of variances that are
//! equal to about 1e-14, so it is compared at 1e-7 of `se`. The test prints the
//! worst ratio of deviation to tolerance in each group.

#[path = "common/r_reference.rs"]
mod r_reference;

use epistat_core::Estimand;
use epistat_matching::abadie_imbens::{
    AbadieImbensData, AbadieImbensSpec, DistanceWeight, abadie_imbens,
};

use r_reference::{Versions, reference as load_reference};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    lalonde: Lalonde,
    sim: Sim,
    results: Vec<Result>,
}

#[derive(Deserialize)]
struct Lalonde {
    treat: Vec<f64>,
    re78: Vec<f64>,
    age: Vec<f64>,
    educ: Vec<f64>,
    married: Vec<f64>,
    nodegree: Vec<f64>,
    re74: Vec<f64>,
}

#[derive(Deserialize)]
struct Sim {
    treat: Vec<f64>,
    y: Vec<f64>,
    x1: Vec<f64>,
    x2: Vec<f64>,
    x3: Vec<f64>,
    x4: Vec<f64>,
    extra: Vec<f64>,
}

#[derive(Deserialize)]
struct Result {
    data: String,
    estimand: String,
    #[serde(rename = "M")]
    m: usize,
    bias: bool,
    var_calc: usize,
    weight: u8,
    z: bool,
    est: f64,
    est_noadj: f64,
    se: f64,
    se_sample: f64,
    se_cond: f64,
    se_standard: f64,
}

fn reference() -> Reference {
    load_reference("abadie_imbens.json")
}

fn data(r: &Reference, name: &str, with_z: bool) -> AbadieImbensData {
    if name == "lalonde" {
        let l = &r.lalonde;
        return AbadieImbensData::new(
            l.treat.iter().map(|t| *t > 0.5).collect(),
            l.re78.clone(),
            vec![
                l.age.clone(),
                l.educ.clone(),
                l.married.clone(),
                l.nodegree.clone(),
                l.re74.clone(),
            ],
        )
        .unwrap();
    }
    let s = &r.sim;
    let columns = vec![s.x1.clone(), s.x2.clone(), s.x3.clone(), s.x4.clone()];
    let mut z = columns.clone();
    z.push(s.extra.clone());
    let d = AbadieImbensData::new(
        s.treat.iter().map(|t| *t > 0.5).collect(),
        s.y.clone(),
        columns,
    )
    .unwrap();
    if with_z {
        d.with_bias_covariates(z).unwrap()
    } else {
        d
    }
}

#[test]
fn estimate_and_standard_errors_agree_with_match() {
    let r = reference();
    assert!(
        r.versions["matching"].starts_with("4.10"),
        "{}",
        r.versions["r"]
    );
    let mut worst = [0.0_f64; 4];
    let mut count = 0;
    for case in &r.results {
        let target = match case.estimand.as_str() {
            "ATT" => Estimand::Att,
            "ATE" => Estimand::Ate,
            _ => Estimand::Atc,
        };
        let spec = AbadieImbensSpec {
            target,
            matches: case.m,
            bias_adjust: case.bias,
            var_calc: case.var_calc,
            weight: if case.weight == 1 {
                DistanceWeight::Diagonal
            } else {
                DistanceWeight::Mahalanobis
            },
            ..AbadieImbensSpec::new()
        };
        let got = abadie_imbens(&data(&r, &case.data, case.z), &spec).unwrap();
        let label = format!(
            "{} {} M={} bias={} var={} weight={} z={}",
            case.data, case.estimand, case.m, case.bias, case.var_calc, case.weight, case.z
        );
        let (est_tol, se_tol): (f64, f64) = if case.bias && case.data == "lalonde" {
            (1e-6, 1e-6)
        } else {
            (1e-12, 1e-10)
        };
        let pairs = [
            ("est", got.estimate, case.est, est_tol, 0),
            (
                "est_noadj",
                got.estimate_unadjusted,
                case.est_noadj,
                1e-12,
                0,
            ),
            ("se", got.se, case.se, se_tol, 1),
            ("se_sample", got.se_sample, case.se_sample, se_tol, 2),
            (
                "se_cond",
                got.se_conditional,
                case.se_cond,
                se_tol.max(1e-7),
                3,
            ),
            ("se_standard", got.se_standard, case.se_standard, 1e-10, 1),
        ];
        for (name, ours, theirs, tol, slot) in pairs {
            let dev = (ours - theirs).abs() / (tol * theirs.abs().max(1e-300));
            // se.cond is the square root of a difference of two variances
            // that are equal up to rounding, about 1e-14 of their size, so
            // its error is about 1e-7 of se, and it is judged against se.
            let dev = if name == "se_cond" {
                (ours - theirs).abs() / (tol * case.se.abs())
            } else {
                dev
            };
            worst[slot] = worst[slot].max(dev);
            assert!(dev <= 1.0, "{label} {name}: {ours} against {theirs}");
        }
        count += 1;
    }
    println!("{count} runs; worst ratios of deviation to tolerance: {worst:?}");
}
