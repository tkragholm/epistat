//! Effect estimates after matching, checked against `marginaleffects` 1.0.0
//! and `sandwich` 3.1.3.
//!
//! The reference is `tests/data/effects.json`, written by
//! `tests/data/effects.R`. The script matches `lalonde` with MatchIt in six
//! ways (nearest, full matching for the ATT and ATE, subclassification for the
//! ATT and ATC, and 2:1 nearest with replacement), fits the outcome model to
//! the matched rows with the matching weights, and records the coefficients,
//! the cluster-robust covariance from `vcovCL()` and the estimates from
//! `avg_comparisons()` with the weights as `wts`. The test builds the matched
//! sample from the rows MatchIt gave, through `match_data` or `get_matches`
//! shapes, so it checks the effect code and not the matching.
//!
//! It covers a linear model of `re78` and a logistic model of `re78 > 0`,
//! each with the treatment alone, with age and educ, and with their
//! interactions with the treatment. The variance is clustered by the subclass
//! without replacement, and by the subclass and the unit, and by the unit
//! alone, with replacement. For the logistic model it compares the difference,
//! the ratio and the log ratio, and the log odds ratio of the averaged
//! predictions with their delta-method standard errors.
//!
//! Tolerances. Our linear fit solves the normal equations where R uses a QR
//! decomposition. The weighted information matrix of the largest model has a
//! condition number of about 2e5, which bounds the loss at 2e5 times 1e-16
//! times a small factor, so the linear bound is 1e-9 relative. The observed
//! worst deviation is 4e-12. For the logistic model the reference itself is
//! uncertain at about 1e-7: refitting the full-matching model with age, educ
//! and their interactions in R from a start 1e-9 away from its solution, at
//! the same stopping rule of 1e-14, moves its covariance by 7e-8 relative.
//! The logistic bound is 1e-6, ten times that, and the observed worst
//! deviation is 2e-7. The script runs R's logistic fits to a relative deviance
//! change of 1e-14, where its default of 1e-8 would add an error of the same
//! order as the bound.

#[path = "common/r_reference.rs"]
mod r_reference;

use epistat_core::Estimand;
use epistat_matching::UnitRole;
use epistat_matching::effects::{
    Clustering, EffectSpec, MatchedSample, OutcomeModel, UnitData, estimate_effects,
};
use epistat_matching::output::{LongMatches, MatchData};

use r_reference::{Versions, assert_r_version, reference as load_reference};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    lalonde: Lalonde,
    cases: Vec<Case>,
    results: Vec<Result>,
}

#[derive(Deserialize)]
struct Lalonde {
    treat: Vec<f64>,
    re78: Vec<f64>,
    age: Vec<f64>,
    educ: Vec<f64>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    estimand: String,
    rows: Vec<usize>,
    weight: Vec<f64>,
    subclass: Vec<Option<usize>>,
}

#[derive(Deserialize)]
struct Pair {
    estimate: f64,
    se: f64,
}

#[derive(Deserialize)]
struct Result {
    case: String,
    clustering: String,
    outcome: String,
    covariates: Vec<String>,
    interactions: bool,
    coefficients: Vec<f64>,
    vcov: Vec<Vec<f64>>,
    difference: Pair,
    ratio: Option<Pair>,
    lnratio: Option<Pair>,
    lnor: Option<Pair>,
}

fn reference() -> Reference {
    load_reference("effects.json")
}

fn units(r: &Reference, binary: bool, covariates: &[String]) -> UnitData {
    let n = r.lalonde.treat.len();
    let ids = (1..=n).map(|i| format!("u{i}")).collect();
    let treated = r.lalonde.treat.iter().map(|t| *t > 0.5).collect();
    let outcome = r
        .lalonde
        .re78
        .iter()
        .map(|y| {
            if binary {
                f64::from(u8::from(*y > 0.0))
            } else {
                *y
            }
        })
        .collect();
    let mut data = UnitData::new(ids, treated, outcome).unwrap();
    for name in covariates {
        let column = match name.as_str() {
            "age" => r.lalonde.age.clone(),
            "educ" => r.lalonde.educ.clone(),
            other => panic!("covariate {other}"),
        };
        data = data.with_covariate(name.clone(), column).unwrap();
    }
    data
}

fn sample(case: &Case, units: &UnitData) -> MatchedSample {
    let id: Vec<String> = case.rows.iter().map(|r| format!("u{r}")).collect();
    let role = vec![UnitRole::Case; id.len()];
    if case.name.starts_with("replace") {
        let long = LongMatches {
            id,
            role,
            pair_id: case.subclass.clone(),
            weight: case.weight.clone(),
        };
        MatchedSample::from_long_matches(&long, units).unwrap()
    } else {
        let data = MatchData {
            id,
            role,
            weight: case.weight.clone(),
            subclass: case.subclass.clone(),
        };
        MatchedSample::from_match_data(&data, units).unwrap()
    }
}

fn target(estimand: &str) -> Estimand {
    match estimand {
        "ATT" => Estimand::Att,
        "ATC" => Estimand::Atc,
        "ATE" => Estimand::Ate,
        other => panic!("estimand {other}"),
    }
}

fn clustering(name: &str) -> Clustering {
    match name {
        "pair" => Clustering::Pair,
        "unit" => Clustering::Unit,
        "pair_unit" => Clustering::PairAndUnit,
        other => panic!("clustering {other}"),
    }
}

/// The largest `|got - expected| / (rel * |expected|)` seen, which must stay
/// at most 1.
struct Worst(f64);

impl Worst {
    /// `floor` is the scale used in place of an expected value that is smaller
    /// than it, which is zero in exact arithmetic for an effect that the
    /// weights balance away.
    fn check(&mut self, label: &str, got: f64, expected: f64, rel: f64, floor: f64) {
        let scale = rel * expected.abs().max(floor);
        let ratio = (got - expected).abs() / scale;
        self.0 = self.0.max(ratio);
        assert!(
            ratio <= 1.0,
            "{label}: got {got}, R {expected}, ratio {ratio}"
        );
    }
}

#[test]
fn effects_agree_with_marginaleffects() {
    let r = reference();
    assert_r_version(&r.versions["r"]);
    assert_eq!(r.versions["marginaleffects"], "1.0.0");
    assert_eq!(r.versions["sandwich"], "3.1.3");
    let mut worst = Worst(0.0);
    for res in &r.results {
        let case = r.cases.iter().find(|c| c.name == res.case).unwrap();
        let binary = res.outcome == "binary";
        let sample = sample(case, &units(&r, binary, &res.covariates));
        let mut spec = EffectSpec::new(if binary {
            OutcomeModel::Logistic
        } else {
            OutcomeModel::Linear
        });
        spec.target = target(&case.estimand);
        spec.interactions = res.interactions;
        spec.clustering = clustering(&res.clustering);
        let out = estimate_effects(&sample, &spec).unwrap();
        let label = format!(
            "{} {} {} cov={} inter={}",
            res.case,
            res.clustering,
            res.outcome,
            res.covariates.len(),
            res.interactions
        );
        let rel = if binary { 1e-6 } else { 1e-9 };
        let p = res.coefficients.len();
        // A coefficient or effect that is zero in exact arithmetic, as the
        // treatment coefficient is where the weights balance the groups, is
        // compared on a thousandth of the largest coefficient.
        let largest = res.coefficients.iter().fold(0.0_f64, |m, c| m.max(c.abs()));
        let floor = 1e-3 * largest;
        for (k, expected) in res.coefficients.iter().enumerate() {
            worst.check(
                &format!("{label} coef {k}"),
                out.coefficients[k],
                *expected,
                rel,
                floor,
            );
        }
        let v = out.covariance.as_ref().unwrap();
        for a in 0..p {
            for b in 0..p {
                // The covariance entries are small beside their diagonals, so
                // each is compared on the scale of the geometric mean of the two.
                let scale = (res.vcov[a][a] * res.vcov[b][b]).sqrt();
                let ratio = (v[a * p + b] - res.vcov[a][b]).abs() / (rel * scale);
                worst.0 = worst.0.max(ratio);
                assert!(ratio <= 1.0, "{label} vcov {a} {b}: ratio {ratio}");
            }
        }
        let check = |name: &str,
                     got: Option<epistat_matching::effects::Effect>,
                     want: &Option<Pair>,
                     scale: f64,
                     w: &mut Worst| {
            let want = want
                .as_ref()
                .unwrap_or_else(|| panic!("{label} {name} in R"));
            let got = got.unwrap_or_else(|| panic!("{label} {name} missing"));
            w.check(
                &format!("{label} {name}"),
                got.estimate,
                scale * want.estimate,
                rel,
                floor,
            );
            w.check(
                &format!("{label} {name} se"),
                got.se.unwrap(),
                want.se,
                rel,
                floor,
            );
        };
        worst.check(
            &format!("{label} difference"),
            out.difference.estimate,
            res.difference.estimate,
            rel,
            floor,
        );
        worst.check(
            &format!("{label} difference se"),
            out.difference.se.unwrap(),
            res.difference.se,
            rel,
            floor,
        );
        if binary {
            check("ratio", out.ratio, &res.ratio, 1.0, &mut worst);
            check("lnratio", out.log_ratio, &res.lnratio, 1.0, &mut worst);
            check("lnor", out.log_odds_ratio, &res.lnor, 1.0, &mut worst);
        }
    }
    println!("worst deviation as a share of its tolerance: {}", worst.0);
}
