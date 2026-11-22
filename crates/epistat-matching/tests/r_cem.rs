//! Coarsened exact matching checked against MatchIt 4.8.2's
//! `matchit(method = "cem")`.
//!
//! The reference is `tests/data/matchit_cem.json`, written by
//! `tests/data/matchit_cem.R`. It holds a sample of 400 rows of `lalonde` in
//! random order, the same sample with its covariates made unique, and two sets
//! whose values lie on the equal-width cut points. The 34 runs cover the
//! default binning (Sturges), the Freedman-Diaconis and Scott rules, a number
//! of bins, explicit cut points, raw values, quantile groups, `grouping`, the
//! estimands ATT, ATC and ATE, values on a cut point, and `k2k = TRUE` with
//! `k2k.method` set to `mahalanobis`, `euclidean` and `scaled_euclidean` and
//! the orders `data`, `closest` and `farthest`.
//!
//! What the tests check:
//!
//! * Without `k2k`, the units kept, the stratum of each and the weights equal
//!   MatchIt's. MatchIt numbers strata by the group of the first row of the
//!   data, ours by the anchors, so MatchIt's numbers are renumbered by the
//!   anchors, as in `tests/r_exact.rs`. MatchIt rescales the weights of each
//!   group to sum to the number of its kept units, and ours are rescaled the
//!   same way before the comparison.
//! * With `k2k`, the control of every anchor equals MatchIt's. The covariates
//!   of the second set are unique, so no two candidates of a stratum are at the
//!   same distance and the rule for ties does not enter. With ties the
//!   choice among candidates at the same distance is not compared.
//! * A value on a cut point goes to the bin above, which the `boundary` runs
//!   decide. With 10 bins on the multiples of 0.1, the cut point `3 * 0.1` is
//!   above 0.3 in floating point and 0.3 falls below it, so the crate must
//!   compute the cut points as `min + i * ((max - min) / bins)`. The test
//!   `a_value_on_a_cut_point_follows_the_floating_point_cut` asserts that the
//!   cut point and the value differ in the last bit.
//! * "q2" gives one quantile, and `matchit()` reads a single number as a number
//!   of bins, so its strata are those of the median as a number of bins. That
//!   is a defect in MatchIt 4.8.2 (`q2_defect` in the reference). The test
//!   asserts that R's strata are those of the median read as bins and that
//!   ours are those of the median as the only cut point.
//!
//! Tolerances. Strata and pairs are exact comparisons. The weights are
//! `p / (1 - p)` for the anchor share `p` of a stratum in ours and `n1 / n0`
//! in the script's, after the rescaling to a group size of at most 400, so
//! the relative error is below 400 roundings, 1e-13. The tolerance is 1e-12.

#[path = "common/r_reference.rs"]
mod r_reference;

use chrono::NaiveDate;
use epistat_matching::prelude::*;
use epistat_matching::{
    CemBinning, CemRule, CemVariable, EuclideanPreparationConfig, EuclideanScale,
    MahalanobisPreparationConfig, MatchOrder, cem_k2k_pairs, cem_strata, cem_subclasses,
};

use r_reference::{Versions, assert_r_version, reference as load_reference, unit_id as id};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    datasets: BTreeMap<String, BTreeMap<String, Column>>,
    runs: Vec<Run>,
    q2_defect: Q2Defect,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Column {
    Numbers(Vec<f64>),
    Text(Vec<String>),
}

#[derive(Deserialize)]
struct Q2Defect {
    median_age: f64,
    subclass_of_q2: Vec<Option<usize>>,
    #[expect(dead_code, reason = "recorded in the reference, compared in R")]
    subclass_of_median_as_bins: Vec<Option<usize>>,
    agree: bool,
}

#[derive(Deserialize, Clone)]
struct Binning {
    kind: String,
    rule: Option<String>,
    n: Option<usize>,
    values: Option<Vec<f64>>,
}

#[derive(Deserialize)]
struct Var {
    name: String,
    #[serde(rename = "type")]
    kind: String,
    cut: Option<Binning>,
    groups: Option<Vec<Vec<String>>>,
}

#[derive(Deserialize)]
struct K2k {
    method: String,
    order: Option<String>,
}

#[derive(Deserialize)]
struct Run {
    name: String,
    dataset: String,
    estimand: String,
    vars: Vec<Var>,
    cut_all: Option<Binning>,
    k2k: Option<K2k>,
    error: Option<String>,
    subclass: Option<Vec<Option<usize>>>,
    weights: Option<Vec<f64>>,
    match_rows: Option<Vec<Option<usize>>>,
    focal_rows: Option<Vec<usize>>,
}

fn reference() -> Reference {
    load_reference("matchit_cem.json")
}

fn binning(spec: &Binning) -> CemBinning {
    match spec.kind.as_str() {
        "rule" => CemBinning::Rule(match spec.rule.as_deref().expect("rule") {
            "sturges" => CemRule::Sturges,
            "fd" => CemRule::FreedmanDiaconis,
            "scott" => CemRule::Scott,
            other => panic!("rule {other}"),
        }),
        "bins" => CemBinning::Bins(spec.n.expect("bins")),
        "cuts" => CemBinning::Cutpoints(spec.values.clone().expect("values")),
        "quantiles" => CemBinning::Quantiles(spec.n.expect("quantiles")),
        "exact" => CemBinning::Exact,
        other => panic!("binning {other}"),
    }
}

fn variables(run: &Run) -> Vec<CemVariable> {
    run.vars
        .iter()
        .map(|var| {
            if var.kind == "numeric" {
                let spec = run.cut_all.as_ref().or(var.cut.as_ref());
                CemVariable::numeric(&var.name, spec.map_or_else(CemBinning::default, binning))
            } else {
                CemVariable::categorical(&var.name, var.groups.clone().unwrap_or_default())
            }
        })
        .collect()
}

/// Every unit as a record with all numeric columns as numeric fields and
/// strata, and as covariates. The anchors are the units whose group is
/// `anchor_group`. Rows are 1-based in the ids.
fn records(
    reference: &Reference,
    run: &Run,
    anchor_group: f64,
) -> (Vec<BalanceRecord>, Vec<BalanceRecord>, Vec<f64>) {
    let columns = &reference.datasets[&run.dataset];
    let Column::Numbers(treat) = &columns["treat"] else {
        panic!("treat is numeric")
    };
    let mut anchors = Vec::new();
    let mut candidates = Vec::new();
    for (row, group) in treat.iter().enumerate() {
        let mut record = BalanceRecord::new(
            id(row + 1),
            NaiveDate::from_ymd_opt(2000, 1, 1).expect("valid date"),
        );
        for (name, column) in columns {
            match column {
                Column::Numbers(values) => {
                    record.core.numerics.insert(name.clone(), values[row]);
                    record
                        .covariates
                        .insert(name.clone(), CovariateValue::Numeric(values[row]));
                }
                Column::Text(values) => {
                    record.core.strata.insert(name.clone(), values[row].clone());
                    record.covariates.insert(
                        name.clone(),
                        CovariateValue::Categorical(values[row].clone()),
                    );
                }
            }
        }
        if (*group - anchor_group).abs() < 0.5 {
            anchors.push(record);
        } else {
            candidates.push(record);
        }
    }
    (anchors, candidates, treat.clone())
}

fn anchor_group(run: &Run) -> f64 {
    if run.estimand == "ATC" { 0.0 } else { 1.0 }
}

fn estimand_of(name: &str) -> Estimand {
    match name {
        "ATT" => Estimand::Att,
        "ATC" => Estimand::Atc,
        "ATE" => Estimand::Ate,
        other => panic!("estimand {other}"),
    }
}

/// MatchIt's numbers, renumbered by first appearance among the units of
/// `group`.
fn renumbered(treat: &[f64], subclass: &[Option<usize>], group: f64) -> Vec<Option<usize>> {
    let mut order: Vec<usize> = Vec::new();
    for (idx, label) in treat.iter().enumerate() {
        if (*label - group).abs() < 0.5
            && let Some(s) = subclass[idx]
            && !order.contains(&s)
        {
            order.push(s);
        }
    }
    subclass
        .iter()
        .map(|s| s.map(|s| order.iter().position(|o| *o == s).expect("seen") + 1))
        .collect()
}

/// Rescale the weights of each group to sum to the number of its kept units.
fn rescaled(treat: &[f64], weights: &[Option<f64>]) -> Vec<f64> {
    let group = |idx: usize| usize::from(treat[idx] < 0.5);
    let mut sum = [0.0_f64; 2];
    let mut count = [0.0_f64; 2];
    for (idx, w) in weights.iter().enumerate() {
        if let Some(w) = w {
            sum[group(idx)] += w;
            count[group(idx)] += 1.0;
        }
    }
    weights
        .iter()
        .enumerate()
        .map(|(idx, w)| w.map_or(0.0, |w| w * count[group(idx)] / sum[group(idx)]))
        .collect()
}

const WEIGHT_TOLERANCE: f64 = 1e-12;

#[test]
fn reference_was_made_by_the_documented_versions() {
    let reference = reference();
    assert_r_version(&reference.versions["r"]);
    assert_eq!(reference.versions["matchit"], "4.8.2");
    assert_eq!(reference.runs.len(), 34);
}

/// Kept units, strata and weights equal MatchIt's in every run without k2k.
#[test]
fn strata_and_weights_match_matchit() {
    let reference = reference();
    let mut compared = 0;
    for run in reference
        .runs
        .iter()
        .filter(|run| run.k2k.is_none() && run.name != "q2")
    {
        // As in the exact method's test, the treated units are the anchors and
        // the estimand picks the weights.
        let group = 1.0;
        let (anchors, candidates, treat) = records(&reference, run, group);
        let got = cem_subclasses(&anchors, &candidates, &variables(run))
            .unwrap_or_else(|error| panic!("{}: {error}", run.name));
        assert!(run.error.is_none(), "{}", run.name);
        let r_subclass = run.subclass.as_ref().expect("subclass");
        let r_weights = run.weights.as_ref().expect("weights");
        let by_anchors = renumbered(&treat, r_subclass, group);
        let weights: HashMap<String, f64> = {
            let (ids, values) = got
                .weights_sorted(estimand_of(&run.estimand))
                .expect("weights for the estimand");
            ids.into_iter().map(str::to_owned).zip(values).collect()
        };
        let assigned: HashMap<String, usize> = got
            .assignments()
            .map(|(id, subclass)| (id.to_owned(), subclass))
            .collect();
        let ours: Vec<Option<f64>> = (0..treat.len())
            .map(|row| weights.get(&id(row + 1)).copied())
            .collect();
        let scaled = rescaled(&treat, &ours);
        for row in 0..treat.len() {
            let subclass = assigned.get(&id(row + 1)).map(|s| s + 1);
            assert_eq!(
                subclass,
                by_anchors[row],
                "{}: stratum of row {}",
                run.name,
                row + 1
            );
            let want = r_weights[row];
            assert!(
                (scaled[row] - want).abs() <= WEIGHT_TOLERANCE * want.abs().max(1.0),
                "{}: weight of row {}: {} against {}",
                run.name,
                row + 1,
                scaled[row],
                want
            );
        }
        compared += 1;
    }
    assert_eq!(compared, 25);
}

/// The strata of every run are the same whichever group is the anchors: the
/// coarsening does not read the groups.
#[test]
fn strata_do_not_depend_on_which_group_is_the_anchors() {
    let reference = reference();
    let run = &reference.runs[0];
    let variables = variables(run);
    let (anchors, candidates, treat) = records(&reference, run, 1.0);
    let (control_anchors, treated_candidates, _) = records(&reference, run, 0.0);
    let one = cem_strata(&anchors, &candidates, &variables).expect("strata");
    let other = cem_strata(&control_anchors, &treated_candidates, &variables).expect("strata");
    assert_eq!(one.count, other.count);
    // Both kept sets are the same units.
    let kept = |strata: &epistat_matching::CemStrata, a: &[BalanceRecord], c: &[BalanceRecord]| {
        let mut ids: Vec<String> = a
            .iter()
            .zip(&strata.anchors)
            .chain(c.iter().zip(&strata.candidates))
            .filter(|(_, stratum)| stratum.is_some())
            .map(|(record, _)| record.core.id.clone())
            .collect();
        ids.sort();
        ids
    };
    assert_eq!(
        kept(&one, &anchors, &candidates),
        kept(&other, &control_anchors, &treated_candidates)
    );
    assert_eq!(treat.len(), 400);
}

/// The units of each stratum are paired one to one as `k2k = TRUE` does.
#[test]
fn k2k_pairs_match_matchit() {
    let reference = reference();
    let mut compared = 0;
    for run in reference.runs.iter().filter(|run| run.k2k.is_some()) {
        let k2k = run.k2k.as_ref().expect("k2k");
        let group = anchor_group(run);
        let (anchors, candidates, _) = records(&reference, run, group);
        let strata = cem_strata(&anchors, &candidates, &variables(run)).expect("strata");
        let names: Vec<String> = run.vars.iter().map(|var| var.name.clone()).collect();
        let encoding = CovariateEncodingConfig::builder()
            .covariate_keys(names)
            .include_intercept(false)
            .build();
        let config = match k2k.method.as_str() {
            "mahalanobis" => {
                prepare_mahalanobis_distance_config(
                    &anchors,
                    &candidates,
                    None,
                    &MahalanobisPreparationConfig::builder()
                        .encoding(encoding)
                        .build(),
                )
                .expect("mahalanobis")
                .distance_config
            }
            "euclidean" | "scaled_euclidean" => {
                prepare_euclidean_distance_config(
                    &anchors,
                    &candidates,
                    None,
                    &EuclideanPreparationConfig::builder()
                        .encoding(encoding)
                        .scale(if k2k.method == "euclidean" {
                            EuclideanScale::None
                        } else {
                            EuclideanScale::PooledWithinGroupSd
                        })
                        .build(),
                )
                .expect("euclidean")
                .distance_config
            }
            other => panic!("method {other}"),
        };
        let order = match k2k.order.as_deref() {
            None => MatchOrder::Data,
            Some("closest") => MatchOrder::Closest,
            Some("farthest") => MatchOrder::Farthest,
            Some(other) => panic!("order {other}"),
        };
        let outcome = cem_k2k_pairs(
            &anchors,
            &candidates,
            &strata,
            Some(&config),
            order,
            0,
            estimand_of(&run.estimand),
        )
        .expect("pairs");
        let partner: BTreeMap<String, String> = outcome
            .pairs
            .iter()
            .map(|pair| {
                (
                    pair.anchor_id().to_string(),
                    pair.comparator_id().to_string(),
                )
            })
            .collect();
        let rows = run.match_rows.as_ref().expect("match rows");
        let focal = run.focal_rows.as_ref().expect("focal rows");
        assert_eq!(anchors.len(), focal.len(), "{}", run.name);
        let mut matched = 0;
        for (row, expected) in focal.iter().zip(rows) {
            let ours = partner.get(&id(*row));
            assert_eq!(
                ours.cloned(),
                expected.map(id),
                "{}: control of row {row}",
                run.name
            );
            matched += usize::from(ours.is_some());
        }
        assert!(matched > 20, "{}: only {matched} pairs", run.name);
        compared += 1;
    }
    assert_eq!(compared, 8);
}

/// Without a distance the candidates of a stratum are taken in an order drawn
/// from the seed: each stratum pairs min(anchors, candidates) units, no
/// candidate twice, and the same seed gives the same pairs.
#[test]
fn k2k_without_a_distance_pairs_within_strata() {
    let reference = reference();
    let run = reference
        .runs
        .iter()
        .find(|run| run.name == "k2k_euclidean")
        .expect("run");
    let (anchors, candidates, _) = records(&reference, run, 1.0);
    let strata = cem_strata(&anchors, &candidates, &variables(run)).expect("strata");
    let draw = |seed| {
        cem_k2k_pairs(
            &anchors,
            &candidates,
            &strata,
            None,
            MatchOrder::Data,
            seed,
            Estimand::Att,
        )
        .expect("pairs")
    };
    let first = draw(3);
    assert_eq!(first.pairs, draw(3).pairs);
    assert_ne!(first.pairs, draw(4).pairs);
    let mut used = std::collections::HashSet::new();
    for pair in &first.pairs {
        assert!(used.insert(pair.comparator_id().to_string()));
        let a = anchors
            .iter()
            .position(|r| r.core.id == pair.anchor_id())
            .expect("anchor");
        let c = candidates
            .iter()
            .position(|r| r.core.id == pair.comparator_id())
            .expect("candidate");
        assert_eq!(strata.anchors[a], strata.candidates[c]);
    }
    let mut expected = 0;
    for stratum in 0..strata.count {
        let a = strata
            .anchors
            .iter()
            .filter(|s| **s == Some(stratum))
            .count();
        let c = strata
            .candidates
            .iter()
            .filter(|s| **s == Some(stratum))
            .count();
        expected += a.min(c);
    }
    assert_eq!(first.pairs.len(), expected);
}

/// "q2" is a defect in MatchIt 4.8.2: the one quantile is read as a number of
/// bins. Its strata are those of the median as a number of bins, and ours
/// are those of the median as the only cut point.
#[test]
fn quantile_groups_of_two_cut_at_the_median() {
    let reference = reference();
    assert!(reference.q2_defect.agree, "R's q2 is the median as bins");
    let run = reference
        .runs
        .iter()
        .find(|run| run.name == "q2")
        .expect("run");
    let (anchors, candidates, treat) = records(&reference, run, 1.0);
    let columns = &reference.datasets["lalonde"];
    let Column::Numbers(age) = &columns["age"] else {
        panic!("age")
    };
    let mut sorted = age.clone();
    sorted.sort_by(f64::total_cmp);
    let median = f64::midpoint(sorted[199], sorted[200]);
    assert!((median - reference.q2_defect.median_age).abs() < 1e-12);

    let strata_of = |binning: CemBinning| {
        let variables = vec![
            CemVariable::numeric("age", binning.clone()),
            CemVariable::numeric("educ", binning),
            CemVariable::categorical("race", vec![]),
            CemVariable::categorical("married", vec![]),
        ];
        cem_strata(&anchors, &candidates, &variables).expect("strata")
    };
    let ours = strata_of(CemBinning::Quantiles(2));
    let by_median = {
        let variables = vec![
            CemVariable::numeric("age", CemBinning::Cutpoints(vec![median])),
            CemVariable::numeric(
                "educ",
                CemBinning::Cutpoints(vec![{
                    let Column::Numbers(educ) = &columns["educ"] else {
                        panic!("educ")
                    };
                    let mut sorted = educ.clone();
                    sorted.sort_by(f64::total_cmp);
                    f64::midpoint(sorted[199], sorted[200])
                }]),
            ),
            CemVariable::categorical("race", vec![]),
            CemVariable::categorical("married", vec![]),
        ];
        cem_strata(&anchors, &candidates, &variables).expect("strata")
    };
    assert_eq!(ours, by_median);
    // MatchIt's strata, with the numbering by the anchors, differ from ours.
    let r_strata = renumbered(&treat, &reference.q2_defect.subclass_of_q2, 1.0);
    let r_kept = r_strata.iter().filter(|s| s.is_some()).count();
    let our_kept = ours
        .anchors
        .iter()
        .chain(&ours.candidates)
        .flatten()
        .count();
    assert_ne!(r_kept, our_kept, "R keeps {r_kept}, ours {our_kept}");
}

/// With 10 bins on 0 to 1 the third cut point is `3 * 0.1`, which is a bit
/// above 0.3, so 0.3 shares a bin with 0.2 and 0.6 with 0.5. This is the case
/// the `tenths` runs of the reference decide against MatchIt.
#[test]
fn a_value_on_a_cut_point_follows_the_floating_point_cut() {
    let tenth = std::hint::black_box(0.1_f64);
    assert!(3.0 * tenth > 0.3);
    let unit = |id: &str, x: f64| {
        BaseRecord::new(id, NaiveDate::from_ymd_opt(2000, 1, 1).expect("valid date"))
            .with_numeric("x", x)
    };
    let anchors = [
        unit("a0", 0.0),
        unit("a2", 0.2),
        unit("a4", 0.4),
        unit("a1", 1.0),
    ];
    let candidates = [unit("c3", 0.3), unit("c6", 0.6), unit("c5", 0.5)];
    let strata = cem_strata(
        &anchors,
        &candidates,
        &[CemVariable::numeric("x", CemBinning::Bins(10))],
    )
    .expect("strata");
    assert_eq!(strata.anchors, vec![None, Some(0), None, None]);
    assert_eq!(strata.candidates, vec![Some(0), None, None]);
}
