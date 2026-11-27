//! Matching after multiple imputation and the pooled effect, checked against
//! R 4.6's `MatchThem` 1.2.1 and `mice` 3.19.0.
//!
//! The reference is `tests/data/matchthem.json`, written by
//! `tests/data/matchthem.R`. It holds five data sets that `mice()` imputed
//! from `lalonde` with values removed at random from age, educ, re74, re75 and
//! re78, and for the approaches `within` and `across` of `matchthem()` the
//! control matched to each treated row in every data set, the distance the
//! matching used, the coefficient of the treatment in `lm(re78 ~ treat)` on the
//! matched rows with its cluster-robust variance by subclass, and
//! `mice::pool.scalar()` of the five.
//!
//! The crate matches the same five data sets from the covariates, as
//! `match_imputed` does, and must give the same pairs in every data set and
//! approach. The propensity scores agree with R's to about 1e-13 (see
//! `r_propensity.rs`) and the averaged scores to the same, so a pair would
//! differ only where two controls lie that close to each other, which the
//! continuous scores do not. The scores are compared at 1e-11 absolute, a
//! hundred times that, and the effects at 1e-9 relative as in `r_effects.rs`,
//! where the linear fit solves the normal equations and R a QR
//! decomposition. The pooled quantities add a few operations on those and
//! are compared at 1e-9.
//!
//! The pooled estimate is also compared with the coefficient that
//! `MatchThem`'s own `with()` and `pool()` return. Its model-based standard
//! error is not compared: the crate pools the cluster-robust one.

#[path = "common/r_reference.rs"]
mod r_reference;

use epistat_matching::Error;
use epistat_matching::effects::{Clustering, EffectEstimates, EffectSpec, OutcomeModel, UnitData};
use epistat_matching::imputation::{
    CompleteDataDf, ImputationApproach, ImputedSet, estimate_pooled, match_imputed, pool_effects,
};
use epistat_matching::prelude::*;

use r_reference::{
    Versions, assert_r_version, covariate_unit, reference as load_reference, split_by_group,
};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    data: Vec<BTreeMap<String, Vec<f64>>>,
    approaches: Vec<Approach>,
}

#[derive(Deserialize)]
struct Approach {
    #[serde(rename = "approach")]
    name: String,
    sets: Vec<RSet>,
    pooled: RPooled,
    matchthem_estimate: f64,
}

#[derive(Deserialize)]
struct RSet {
    treated: Vec<usize>,
    control: Vec<usize>,
    distance: Vec<f64>,
    estimate: f64,
    variance: f64,
}

#[derive(Deserialize)]
struct RPooled {
    qbar: f64,
    ubar: f64,
    b: f64,
    t: f64,
    df: f64,
    fmi: f64,
    r: f64,
}

fn reference() -> Reference {
    load_reference("matchthem.json")
}

const COVARIATES: [&str; 6] = ["age", "educ", "married", "nodegree", "re74", "re75"];

/// The anchors and candidates of one imputed data set, ids being the 1-based
/// rows.
fn records(data: &BTreeMap<String, Vec<f64>>) -> (Vec<BalanceRecord>, Vec<BalanceRecord>) {
    split_by_group(data["treat"].iter().enumerate().map(|(row, treat)| {
        let covariates = COVARIATES.iter().map(|name| {
            (
                (*name).to_string(),
                CovariateValue::Numeric(data[*name][row]),
            )
        });
        (
            *treat > 0.5,
            covariate_unit((row + 1).to_string(), covariates),
        )
    }))
}

fn close(got: f64, want: f64, rel: f64) -> bool {
    (got - want).abs() <= rel * want.abs().max(1e-300)
}

#[test]
fn reference_was_made_by_the_documented_versions() {
    let r = reference();
    assert_r_version(&r.versions["r"]);
    assert_eq!(r.versions["MatchThem"], "1.2.1");
    assert_eq!(r.versions["mice"], "3.19.0");
}

#[test]
fn matches_and_pooled_effects_agree_with_matchthem() {
    let r = reference();
    let prepared: Vec<_> = r.data.iter().map(records).collect();
    let sets: Vec<ImputedSet<'_, BalanceRecord>> = prepared
        .iter()
        .map(|(a, c)| ImputedSet {
            anchors: a,
            candidates: c,
        })
        .collect();
    let units: Vec<UnitData> = r
        .data
        .iter()
        .map(|d| {
            let ids = (1..=d["treat"].len()).map(|i| i.to_string()).collect();
            let treated = d["treat"].iter().map(|t| *t > 0.5).collect();
            UnitData::new(ids, treated, d["re78"].clone()).unwrap()
        })
        .collect();
    let criteria = MatchingCriteria::builder().match_ratio(1).build();
    let config = PropensityScoreConfig::default();
    let mut spec = EffectSpec::new(OutcomeModel::Linear);
    spec.clustering = Clustering::Pair;

    for reference in &r.approaches {
        let approach = match reference.name.as_str() {
            "within" => ImputationApproach::Within,
            _ => ImputationApproach::Across,
        };
        let matches = match_imputed(
            &sets,
            approach,
            &criteria,
            &NearestSelection::new(0),
            &[],
            &(),
            &config,
        )
        .unwrap();
        let mut worst_score = 0.0_f64;
        for (i, (ours, rs)) in matches.sets.iter().zip(&reference.sets).enumerate() {
            let label = format!("{} set {i}", reference.name);
            let by_anchor: HashMap<usize, usize> = ours
                .outcome
                .pairs
                .iter()
                .map(|p| {
                    (
                        p.anchor_id().parse().unwrap(),
                        p.comparator_id().parse().unwrap(),
                    )
                })
                .collect();
            assert_eq!(by_anchor.len(), rs.treated.len(), "{label}: pairs");
            for (t, c) in rs.treated.iter().zip(&rs.control) {
                assert_eq!(by_anchor[t], *c, "{label}: control of {t}");
            }
            for (row, want) in rs.distance.iter().enumerate() {
                let got = ours
                    .propensity
                    .distance_score(&(row + 1).to_string())
                    .unwrap();
                worst_score = worst_score.max((got - want).abs());
            }
        }
        assert!(
            worst_score < 1e-11,
            "{}: scores {worst_score:e}",
            reference.name
        );

        let effects = estimate_pooled(
            &sets,
            &matches,
            &units,
            MatchDataOptions::default(),
            &spec,
            CompleteDataDf::Residual,
        )
        .unwrap();
        for (i, (ours, rs)) in effects.per_set.iter().zip(&reference.sets).enumerate() {
            let se = ours.difference.se.unwrap();
            assert!(
                close(ours.difference.estimate, rs.estimate, 1e-9),
                "set {i} estimate {} vs {}",
                ours.difference.estimate,
                rs.estimate
            );
            assert!(
                close(se * se, rs.variance, 1e-9),
                "set {i} variance {} vs {}",
                se * se,
                rs.variance
            );
        }
        let p = effects.pooled.difference.expect("a pooled difference");
        let want = &reference.pooled;
        eprintln!(
            "{}: worst score deviation {worst_score:e}, pooled estimate {:e}, total variance {:e}, df {:e}",
            reference.name,
            (p.estimate - want.qbar).abs() / want.qbar.abs(),
            (p.total - want.t).abs() / want.t,
            (p.df.unwrap() - want.df).abs() / want.df
        );
        assert!(close(p.estimate, want.qbar, 1e-9), "estimate");
        assert!(
            close(p.estimate, reference.matchthem_estimate, 1e-9),
            "MatchThem"
        );
        assert!(close(p.within, want.ubar, 1e-9), "within");
        assert!(close(p.between, want.b, 1e-9), "between");
        assert!(close(p.total, want.t, 1e-9), "total");
        assert!(
            close(p.df.unwrap(), want.df, 1e-9),
            "df {:?} vs {}",
            p.df,
            want.df
        );
        assert!(close(p.fmi.unwrap(), want.fmi, 1e-9), "fmi");
        assert!(close(p.riv.unwrap(), want.r, 1e-9), "riv");
        assert!(p.between > 0.0, "the sets differ in their outcomes");
        let rows_rule =
            pool_effects(&effects.per_set, CompleteDataDf::RowsLessCoefficients).unwrap();
        assert_eq!(rows_rule, effects.pooled, "the rules agree for many rows");
        if reference.name == "within" {
            check_given_degrees_of_freedom(&effects.per_set);
        }
    }
}

#[derive(Deserialize)]
struct DfReference {
    versions: Versions,
    cases: Vec<DfCase>,
}

#[derive(Deserialize)]
struct DfCase {
    dfcom: serde_json::Value,
    df: f64,
    t: f64,
    fmi: f64,
}

/// The pooled difference under a given complete-data degrees of freedom against
/// `mice::pool.scalar()` with `n` set to `dfcom + 2`, finite and infinite.
fn check_given_degrees_of_freedom(per_set: &[EffectEstimates]) {
    let r: DfReference = load_reference("matchthem_df.json");
    assert_eq!(r.versions["mice"], "3.19.0");
    for case in &r.cases {
        let dfcom = match &case.dfcom {
            serde_json::Value::String(text) => {
                assert_eq!(text, "Inf");
                None
            }
            number => number.as_f64(),
        };
        let pooled = pool_effects(per_set, CompleteDataDf::Given(dfcom)).unwrap();
        assert_eq!(pooled.complete_data_df, dfcom);
        let p = pooled.difference.unwrap();
        assert!(close(p.total, case.t, 1e-9), "total {dfcom:?}");
        assert!(close(p.df.unwrap(), case.df, 1e-9), "df {dfcom:?}");
        assert!(close(p.fmi.unwrap(), case.fmi, 1e-9), "fmi {dfcom:?}");
    }
    // Sets with no more rows than coefficients: the clamped rule pools with one
    // degree of freedom, the unclamped rule has none to give.
    let mut tiny = per_set.to_vec();
    for e in &mut tiny {
        e.rows = e.coefficients.len();
    }
    let pooled = pool_effects(&tiny, CompleteDataDf::Residual).unwrap();
    assert_eq!(pooled.complete_data_df, Some(1.0));
    assert!(matches!(
        pool_effects(&tiny, CompleteDataDf::RowsLessCoefficients),
        Err(Error::Inference(_))
    ));
}

#[test]
fn input_errors_are_reported() {
    let r = reference();
    let prepared: Vec<_> = r.data.iter().take(2).map(records).collect();
    let sets: Vec<ImputedSet<'_, BalanceRecord>> = prepared
        .iter()
        .map(|(a, c)| ImputedSet {
            anchors: a,
            candidates: c,
        })
        .collect();
    let criteria = MatchingCriteria::builder().match_ratio(1).build();
    let run = |sets: &[ImputedSet<'_, BalanceRecord>], approach, config: &PropensityScoreConfig| {
        match_imputed(
            sets,
            approach,
            &criteria,
            &NearestSelection::new(0),
            &[],
            &(),
            config,
        )
    };
    let config = PropensityScoreConfig::default();
    assert!(matches!(
        run(&sets[..1], ImputationApproach::Within, &config),
        Err(Error::TooFewSets { found: 1 })
    ));
    let reestimate = PropensityScoreConfig::builder().reestimate(true).build();
    assert!(matches!(
        run(&sets, ImputationApproach::Across, &reestimate),
        Err(Error::ReestimateAcross)
    ));
    // A unit missing from the second set.
    let short = [
        sets[0],
        ImputedSet {
            anchors: &prepared[1].0[1..],
            candidates: &prepared[1].1,
        },
    ];
    assert!(matches!(
        run(&short, ImputationApproach::Across, &config),
        Err(Error::UnitsDiffer { set: 1, .. })
    ));
}
