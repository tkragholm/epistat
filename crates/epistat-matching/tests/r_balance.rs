//! Balance statistics checked against R 4.6.2 with MatchIt's `summary.matchit()`.
//!
//! `tests/data/balance.R` fits `matchit()` on `lalonde` (a factor `race`,
//! binary `married` and `nodegree`, and `re74`, which has many zeros) and on a
//! simulated data set (a skewed variable, a variable with 60 percent zeros, a
//! count, a binary variable and a four-level factor). It writes the pairs and
//! the unit weights R chose and the tables of `summary()`. Here the balance
//! report is computed on exactly those pairs, so the comparison does not depend
//! on how pairs are selected.
//!
//! The pairs are given to `balance_report` as a `MatchOutcome`
//! whose realised estimand is the one R used. The report derives the unit
//! weights from it, and the weights are checked against R's. Fits with pairs
//! are nearest-neighbour with replacement for the ATT and ATC, 1:2 without
//! replacement (so that some treated units have one match), and 1:1 without
//! replacement (equal group sizes). For the ATE R used subclassification, whose
//! weights are not pair counts, so only the statistics before matching are
//! compared there.
//!
//! Compared: weighted means, standardised mean difference, variance ratio, eCDF
//! mean and maximum, and the sample sizes with the effective sample size, for
//! every covariate and each level of a factor, before and after matching, with
//! `interactions = TRUE` (the terms of numeric covariates) and `addlvariables`.
//! The eQQ distances are compared for binary variables, and for continuous ones
//! when the groups have equal size. For groups of unequal size the larger
//! group is read at the smaller group's cumulative weights as a step function,
//! and where a cumulative weight of one group equals one of the other the
//! result depends on the last bit of the weights. The weights derived here from
//! pair counts are not R's bit for bit, so those values are compared in
//! `r_balance_weights.rs` on R's own weights. That file also covers
//! subclassification, the exact method, no matching, sampling weights, discarded
//! units, the pair distance, the improvement and interactions with factor
//! dummies. Cramér's V is in neither MatchIt nor cobalt.

#[path = "common/r_reference.rs"]
mod r_reference;

use chrono::NaiveDate;
use epistat_matching::MatchWeightMethod;
use epistat_matching::prelude::*;
use epistat_matching::{
    BalanceRecord, BalanceReport, BalanceReportOptions, CovariateValue, Estimand, MatchDiagnostics,
    NumericBalance, NumericBalanceTransform, SupplementalBalanceCovariates, balance_report,
};
use epistat_testkit::{Tolerance, load_json, na_f64, na_f64_vec};
use r_reference::{Versions, assert_r_version};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;

#[derive(Deserialize, Clone, Copy)]
struct Cell(#[serde(deserialize_with = "na_f64")] f64);

type Row = HashMap<String, Cell>;
type Table = HashMap<String, Row>;
/// A covariate's row before and after matching.
type RowPair<'a> = (&'a Row, &'a Row);

#[derive(Deserialize)]
struct Summ {
    all: Table,
    matched: Table,
    nn: Table,
}

#[derive(Deserialize)]
struct Fit {
    name: String,
    estimand: String,
    pairs: Vec<(String, String)>,
    #[serde(deserialize_with = "na_f64_vec")]
    weights: Vec<f64>,
    standardised: Summ,
    unstandardised: Summ,
    interactions_std: Summ,
    addl_std: Summ,
}

#[derive(Deserialize)]
struct Set {
    data: HashMap<String, Vec<Value>>,
    fits: Vec<Fit>,
}

#[derive(Deserialize)]
struct Degenerate {
    data: HashMap<String, Vec<Value>>,
    standardised: Summ,
    pairs: Vec<(String, String)>,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    lalonde: Set,
    sim: Set,
    degenerate: Degenerate,
}

/// The sums are over at most 614 terms, each rounded to a relative 1.1e-16, and
/// the variances subtract means of the same size as the spread at worst by a
/// factor of ten here, so 1e-10 relative is a wide margin over the roughly 1e-13
/// the arithmetic allows. The absolute bound is for statistics that are
/// themselves near 1e-3 or smaller, such as a standardised difference of 0.0015.
fn tolerance() -> Tolerance {
    Tolerance::exact().rel(1e-10).abs(1e-12)
}

fn load() -> Reference {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/balance.json"
    ));
    assert_r_version(&reference.versions["r"]);
    assert_eq!(reference.versions["matchit"], "4.8.2");
    assert_eq!(reference.versions["cobalt"], "5.0.0");
    reference
}

fn day() -> NaiveDate {
    NaiveDate::from_ymd_opt(2000, 1, 1).unwrap()
}

struct Groups {
    cases: Vec<BalanceRecord>,
    controls: Vec<BalanceRecord>,
    supplemental: SupplementalBalanceCovariates,
    treated: HashMap<String, bool>,
}

fn groups(data: &HashMap<String, Vec<Value>>, matched_on: &[&str], addl: &[&str]) -> Groups {
    let mut out = Groups {
        cases: Vec::new(),
        controls: Vec::new(),
        supplemental: SupplementalBalanceCovariates::default(),
        treated: HashMap::new(),
    };
    for (i, id) in data["id"].iter().enumerate() {
        let id = id.as_str().unwrap().to_string();
        let treated = data["treat"][i].as_i64() == Some(1);
        let mut record = BalanceRecord::new(id.clone(), day());
        for name in matched_on {
            let value = match &data[*name][i] {
                Value::String(text) => CovariateValue::Categorical(text.clone()),
                other => CovariateValue::Numeric(other.as_f64().unwrap()),
            };
            record.covariates.insert((*name).to_string(), value);
        }
        let extra = addl
            .iter()
            .map(|name| {
                (
                    (*name).to_string(),
                    CovariateValue::Numeric(data[*name][i].as_f64().unwrap()),
                )
            })
            .collect::<HashMap<_, _>>();
        out.treated.insert(id.clone(), treated);
        if treated {
            out.supplemental.cases.insert(id, extra);
            out.cases.push(record);
        } else {
            out.supplemental.controls.insert(id, extra);
            out.controls.push(record);
        }
    }
    out
}

fn estimand(name: &str) -> Estimand {
    match name {
        "ATT" => Estimand::Att,
        "ATC" => Estimand::Atc,
        "ATE" => Estimand::Ate,
        other => panic!("estimand {other}"),
    }
}

fn outcome(pairs: &[(String, String)], estimand: Estimand) -> MatchOutcome {
    MatchOutcome {
        pairs: pairs
            .iter()
            .map(|(treated, control)| MatchedPair::new(treated.clone(), control.clone()))
            .collect(),
        unmatched_cases: 0,
        used_controls: 0,
        matched_cases: 0,
        avg_controls_per_case: None,
        diagnostics: {
            let mut diagnostics = MatchDiagnostics::default();
            diagnostics.requested_estimand = estimand;
            diagnostics.realized_estimand = estimand;
            diagnostics
        },
    }
}

#[track_caller]
fn same(label: &str, got: Option<f64>, expected: Cell) {
    match got {
        None => assert!(
            expected.0.is_nan(),
            "{label}: undefined here, R has {}",
            expected.0
        ),
        Some(value) => assert!(
            tolerance().accepts(value, expected.0),
            "{label}: {value} against R {}",
            expected.0
        ),
    }
}

/// Compare one numeric row. `binary` rows have no variance ratio in MatchIt.
/// `eqq` says whether the eQQ distances are comparable for this fit.
#[track_caller]
fn compare_numeric(
    label: &str,
    row: &NumericBalance,
    std: RowPair<'_>,
    raw: Option<RowPair<'_>>,
    post_eqq_comparable: bool,
) {
    let (pre, post) = std;
    let binary = pre["Var. Ratio"].0.is_nan();
    same(
        &format!("{label} mean case pre"),
        row.mean_case_pre,
        pre["Means Treated"],
    );
    same(
        &format!("{label} mean control pre"),
        row.mean_control_pre,
        pre["Means Control"],
    );
    same(
        &format!("{label} smd pre"),
        row.smd_pre,
        pre["Std. Mean Diff."],
    );
    same(
        &format!("{label} vr pre"),
        row.var_ratio_pre,
        pre["Var. Ratio"],
    );
    same(
        &format!("{label} ecdf mean pre"),
        row.ecdf_mean_diff_pre,
        pre["eCDF Mean"],
    );
    same(
        &format!("{label} ecdf max pre"),
        row.ecdf_max_diff_pre,
        pre["eCDF Max"],
    );
    same(
        &format!("{label} mean case post"),
        row.mean_case_post,
        post["Means Treated"],
    );
    same(
        &format!("{label} mean control post"),
        row.mean_control_post,
        post["Means Control"],
    );
    same(
        &format!("{label} smd post"),
        row.smd_post,
        post["Std. Mean Diff."],
    );
    same(
        &format!("{label} vr post"),
        row.var_ratio_post,
        post["Var. Ratio"],
    );
    same(
        &format!("{label} ecdf mean post"),
        row.ecdf_mean_diff_post,
        post["eCDF Mean"],
    );
    same(
        &format!("{label} ecdf max post"),
        row.ecdf_max_diff_post,
        post["eCDF Max"],
    );
    if let Some((raw_pre, raw_post)) = raw {
        if binary {
            same(
                &format!("{label} eqq mean pre"),
                row.eqq_mean_diff_pre,
                raw_pre["eQQ Mean"],
            );
            same(
                &format!("{label} eqq max pre"),
                row.eqq_max_diff_pre,
                raw_pre["eQQ Max"],
            );
        }
        if binary || post_eqq_comparable {
            same(
                &format!("{label} eqq mean post"),
                row.eqq_mean_diff_post,
                raw_post["eQQ Mean"],
            );
            same(
                &format!("{label} eqq max post"),
                row.eqq_max_diff_post,
                raw_post["eQQ Max"],
            );
        }
    }
}

fn compare_fit(
    label: &str,
    report: &BalanceReport,
    fit: &Fit,
    names: &dyn Fn(&str) -> String,
    equal_sizes: bool,
) {
    for row in &report.numeric {
        let name = names(&row.name);
        compare_numeric(
            &format!("{label} {name}"),
            row,
            (
                &fit.standardised.all[&name],
                &fit.standardised.matched[&name],
            ),
            Some((
                &fit.unstandardised.all[&name],
                &fit.unstandardised.matched[&name],
            )),
            equal_sizes,
        );
    }
    for factor in &report.categorical {
        for level in &factor.levels {
            let name = format!("{}{}", factor.name, level.level);
            let (pre, post) = (
                &fit.standardised.all[&name],
                &fit.standardised.matched[&name],
            );
            let label = format!("{label} {name}");
            same(
                &format!("{label} p case pre"),
                level.p_case_pre,
                pre["Means Treated"],
            );
            same(
                &format!("{label} p control pre"),
                level.p_control_pre,
                pre["Means Control"],
            );
            same(
                &format!("{label} smd pre"),
                level.smd_pre,
                pre["Std. Mean Diff."],
            );
            same(
                &format!("{label} p case post"),
                level.p_case_post,
                post["Means Treated"],
            );
            same(
                &format!("{label} p control post"),
                level.p_control_post,
                post["Means Control"],
            );
            same(
                &format!("{label} smd post"),
                level.smd_post,
                post["Std. Mean Diff."],
            );
        }
    }
}

const LALONDE: [&str; 6] = ["age", "educ", "race", "married", "nodegree", "re74"];
const SIM: [&str; 6] = ["x1", "x2", "z", "b", "cnt", "f"];

fn sets(
    reference: &Reference,
) -> [(
    &'static str,
    &Set,
    &'static [&'static str],
    &'static [&'static str],
); 2] {
    [
        ("lalonde", &reference.lalonde, &LALONDE, &["re75"]),
        ("sim", &reference.sim, &SIM, &["w"]),
    ]
}

/// Statistics on R's matched units and weights, for the ATT and the ATC with
/// replacement, 1:2 and 1:1 without replacement.
#[test]
fn balance_on_matchit_pairs_agrees_with_summary_matchit() {
    let reference = load();
    let mut compared = 0;
    for (set_name, set, on, _) in sets(&reference) {
        let sample = groups(&set.data, on, &[]);
        for fit in set.fits.iter().filter(|fit| !fit.pairs.is_empty()) {
            let outcome = outcome(&fit.pairs, estimand(&fit.estimand));
            let report = balance_report(
                &sample.cases,
                &sample.controls,
                &outcome,
                &epistat_matching::BalanceReportOptions::default(),
            );
            let label = format!("{set_name} {}", fit.name);
            compare_fit(
                &label,
                &report,
                fit,
                &|name| name.to_string(),
                fit.name == "att_noreplace",
            );
            compared += 1;
        }
    }
    assert_eq!(compared, 8);
}

/// Before matching under the ATE, where the standardiser is the square root of
/// the mean of the two variances. MatchIt's ATE fits here are subclassified, so
/// their matched statistics rest on weights that are not pair counts and are
/// not compared.
#[test]
fn standardised_differences_before_matching_under_the_ate() {
    let reference = load();
    for (set_name, set, on, _) in sets(&reference) {
        let sample = groups(&set.data, on, &[]);
        let fit = set.fits.iter().find(|fit| fit.estimand == "ATE").unwrap();
        let report = balance_report(
            &sample.cases,
            &sample.controls,
            &outcome(&[], Estimand::Ate),
            &epistat_matching::BalanceReportOptions::default(),
        );
        for row in &report.numeric {
            let r = &fit.standardised.all[&row.name];
            let label = format!("{set_name} ate {}", row.name);
            same(&format!("{label} smd"), row.smd_pre, r["Std. Mean Diff."]);
            same(&format!("{label} vr"), row.var_ratio_pre, r["Var. Ratio"]);
            same(
                &format!("{label} ecdf mean"),
                row.ecdf_mean_diff_pre,
                r["eCDF Mean"],
            );
            same(
                &format!("{label} ecdf max"),
                row.ecdf_max_diff_pre,
                r["eCDF Max"],
            );
            assert_eq!(row.smd_post, None, "no matched units");
        }
        for factor in &report.categorical {
            for level in &factor.levels {
                let name = format!("{}{}", factor.name, level.level);
                let r = &fit.standardised.all[&name];
                same(
                    &format!("{set_name} ate {name}"),
                    level.smd_pre,
                    r["Std. Mean Diff."],
                );
            }
        }
    }
}

/// The unit weights the crate derives from the pairs are MatchIt's, and the
/// sample sizes table (all, matched, unmatched and the effective sample size)
/// has MatchIt's values. For the subclassified ATE fits the weights are R's
/// and only the effective sample size is computed.
#[test]
fn unit_weights_and_sample_sizes_agree_with_matchit() {
    let reference = load();
    for (set_name, set, _, _) in sets(&reference) {
        let ids = set.data["id"]
            .iter()
            .map(|value| value.as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        let treat = set.data["treat"]
            .iter()
            .map(|value| value.as_i64() == Some(1))
            .collect::<Vec<_>>();
        for fit in &set.fits {
            let label = format!("{set_name} {}", fit.name);
            let r_weight = |id: &str| fit.weights[ids.iter().position(|x| x == id).unwrap()];
            let (anchor, candidate): (HashMap<String, f64>, HashMap<String, f64>) =
                if fit.pairs.is_empty() {
                    let split = |want: bool| {
                        ids.iter()
                            .zip(&treat)
                            .filter(|(_, is_treated)| **is_treated == want)
                            .map(|(id, _)| (id.clone(), r_weight(id)))
                            .filter(|(_, weight)| *weight > 0.0)
                            .collect::<HashMap<_, _>>()
                    };
                    (split(true), split(false))
                } else {
                    let matched = outcome(&fit.pairs, estimand(&fit.estimand));
                    let weights = matched.match_weights(MatchWeightMethod::for_estimand(
                        matched.diagnostics.realized_estimand,
                    ));
                    let groups: [HashMap<String, f64>; 2] = [
                        weights.cases().map(|(id, w)| (id.to_string(), w)).collect(),
                        weights
                            .controls()
                            .map(|(id, w)| (id.to_string(), w))
                            .collect(),
                    ];
                    // MatchIt scales the weights of the matched controls (ATT) or
                    // treated units (ATC) so that they sum to the size of the other
                    // group. The scale is constant within a group and drops out of
                    // every statistic, so the weights are compared up to it.
                    for group in &groups {
                        let ours: f64 = group.values().sum();
                        let theirs: f64 = group.keys().map(|id| r_weight(id)).sum();
                        for (id, weight) in group {
                            assert!(
                                tolerance().accepts(*weight * theirs / ours, r_weight(id)),
                                "{label}: weight of {id}: {weight} against R {}",
                                r_weight(id)
                            );
                        }
                    }
                    let positive = fit.weights.iter().filter(|w| **w > 0.0).count();
                    assert_eq!(groups[0].len() + groups[1].len(), positive, "{label}");
                    let [cases, controls] = groups;
                    (cases, controls)
                };
            let n_treated = treat.iter().filter(|t| **t).count();
            let n_control = treat.len() - n_treated;
            for (group, weights, n_all) in [
                ("Treated", &anchor, n_treated),
                ("Control", &candidate, n_control),
            ] {
                let expect = |row: &str| fit.standardised.nn.get(row).map(|cells| cells[group].0);
                let ess = epistat_stats::weighted::effective_sample_size(
                    &weights.values().copied().collect::<Vec<_>>(),
                )
                .unwrap()
                .unwrap();
                let matched = weights.len();
                assert!(
                    tolerance().accepts(matched as f64, expect("Matched").unwrap()),
                    "{label} {group} matched"
                );
                assert!(
                    tolerance().accepts((n_all - matched) as f64, expect("Unmatched").unwrap()),
                    "{label} {group} unmatched"
                );
                // MatchIt prints the ESS row only for non-uniform weights.
                if let Some(r_ess) = expect("Matched (ESS)") {
                    assert!(
                        tolerance().accepts(ess, r_ess),
                        "{label} {group} ess {ess} against {r_ess}"
                    );
                } else {
                    assert!(
                        tolerance().accepts(ess, matched as f64),
                        "{label} {group} ess of uniform weights"
                    );
                }
            }
        }
    }
}

/// `interactions = TRUE`: the squares and pairwise products of the numeric
/// covariates. Only the terms that involve numeric covariates alone are compared
/// here, and every term this crate reports must be one MatchIt reports. Terms
/// with the dummy columns of factors are compared in `r_balance_weights.rs`.
/// MatchIt drops a term that is zero everywhere or equal to its first factor,
/// such as the square of a binary variable, and so does this crate.
#[test]
fn squares_and_interactions_agree_with_matchit() {
    let reference = load();
    for (set_name, set, on, _) in sets(&reference) {
        let sample = groups(&set.data, on, &[]);
        let numeric = report_numeric_names(&sample, on);
        let fit = set
            .fits
            .iter()
            .find(|fit| fit.name == "att_replace")
            .unwrap();
        let options = BalanceReportOptions {
            numeric_transforms: NumericBalanceTransform::SquaresAndPairwiseInteractions,
            ..BalanceReportOptions::default()
        };
        let report = balance_report(
            &sample.cases,
            &sample.controls,
            &outcome(&fit.pairs, estimand(&fit.estimand)),
            &options,
        );
        // MatchIt's names: "a²" for a square, "a * b" for a product.
        let r_key = |name: &str| unordered(&name.replace('²', "^2"));
        let r_rows = fit
            .interactions_std
            .all
            .keys()
            .map(|name| (r_key(name), name.clone()))
            .collect::<HashMap<_, _>>();
        let mut compared = 0;
        for row in &report.numeric {
            let r_name = r_rows
                .get(&unordered(&row.name))
                .unwrap_or_else(|| panic!("{set_name}: {} is not a MatchIt row", row.name));
            compare_numeric(
                &format!("{set_name} {}", row.name),
                row,
                (
                    &fit.interactions_std.all[r_name],
                    &fit.interactions_std.matched[r_name],
                ),
                None,
                false,
            );
            compared += 1;
        }
        // Every MatchIt term that involves only numeric covariates is reported.
        for (key, r_name) in &r_rows {
            let only_numeric = key.split(" * ").all(|part| {
                numeric
                    .iter()
                    .any(|name| name == part.trim_end_matches("^2"))
            });
            if only_numeric {
                assert!(
                    report
                        .numeric
                        .iter()
                        .any(|row| &unordered(&row.name) == key),
                    "{set_name}: MatchIt row {r_name} is missing"
                );
            }
        }
        assert!(compared > 10, "{set_name}: {compared} terms compared");
    }
}

fn report_numeric_names(sample: &Groups, on: &[&str]) -> Vec<String> {
    on.iter()
        .filter(|name| {
            matches!(
                sample.cases[0].covariates[**name],
                CovariateValue::Numeric(_)
            )
        })
        .map(|name| (*name).to_string())
        .collect()
}

/// A product named the same whichever factor comes first.
fn unordered(name: &str) -> String {
    let mut parts = name.split(" * ").map(str::to_string).collect::<Vec<_>>();
    parts.sort();
    parts.join(" * ")
}

/// `addlvariables`: a covariate that was not matched on, given as a
/// supplemental covariate, has MatchIt's statistics in the same report, and its
/// presence does not change the others.
#[test]
fn supplemental_covariates_agree_with_addlvariables() {
    let reference = load();
    for (set_name, set, on, addl) in sets(&reference) {
        let sample = groups(&set.data, on, addl);
        for fit in set.fits.iter().filter(|fit| !fit.pairs.is_empty()) {
            let options = BalanceReportOptions {
                supplemental_covariates: sample.supplemental.clone(),
                ..BalanceReportOptions::default()
            };
            let report = balance_report(
                &sample.cases,
                &sample.controls,
                &outcome(&fit.pairs, estimand(&fit.estimand)),
                &options,
            );
            let name = addl[0];
            let row = report.numeric.iter().find(|row| row.name == name).unwrap();
            compare_numeric(
                &format!("{set_name} {} {name}", fit.name),
                row,
                (&fit.addl_std.all[name], &fit.addl_std.matched[name]),
                None,
                false,
            );
            assert_eq!(
                report.numeric.len(),
                report_numeric_names(&sample, on).len() + addl.len()
            );
        }
    }
}

/// What MatchIt prints for a covariate without variance, and what this crate
/// reports. A variable that is constant within each group (`a`: 3 among the
/// treated, 0 among the controls) has an infinite standardised difference and
/// variance ratio in MatchIt. A constant `k` has a standardised difference of 0
/// there, which is the difference in means returned unstandardised. A variable
/// that is constant among the treated (`tz`) is standardised by the pooled
/// standard deviation instead of the treated one, because MatchIt falls back
/// to it when the estimand's standard deviation is below the square root of the
/// machine epsilon. This crate reports no number when the standardiser is zero
/// or undefined.
#[test]
fn covariates_without_variance_have_no_standardised_difference() {
    let reference = load();
    let degenerate = &reference.degenerate;
    let sample = groups(&degenerate.data, &["a", "k", "tz"], &[]);
    let report = balance_report(
        &sample.cases,
        &sample.controls,
        &outcome(&degenerate.pairs, Estimand::Att),
        &epistat_matching::BalanceReportOptions::default(),
    );
    let row = |name: &str| report.numeric.iter().find(|row| row.name == name).unwrap();
    let r = |name: &str| &degenerate.standardised.all[name];

    assert!(r("a")["Std. Mean Diff."].0.is_infinite());
    assert!(r("a")["Var. Ratio"].0.is_infinite());
    for value in [row("a").smd_pre, row("a").smd_post, row("a").var_ratio_pre] {
        assert_eq!(value, None);
    }

    assert!(tolerance().accepts(r("k")["Std. Mean Diff."].0, 0.0));
    assert_eq!(row("k").smd_pre, None);
    assert_eq!(row("k").smd_post, None);
    assert_eq!(row("k").var_ratio_pre, None);
    assert_eq!(row("k").ecdf_max_diff_pre, Some(0.0));

    assert!(r("tz")["Std. Mean Diff."].0.is_finite());
    assert_eq!(row("tz").smd_pre, None);
    assert_eq!(row("tz").smd_post, None);
    assert_eq!(row("tz").var_ratio_pre, Some(0.0));
}
