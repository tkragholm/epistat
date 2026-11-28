//! Balance on weights given from outside, checked against R 4.6.2 with
//! MatchIt 4.8.2's `summary.matchit()`.
//!
//! `tests/data/balance_weights.R` fits `matchit()` on `lalonde` and a
//! simulated data set, both with a sampling weight, and writes the weights, the
//! discarded flags and the pairs MatchIt produced together with the tables of
//! `summary()`. Here the weights go to `balance_summary` as they are. The fits
//! are subclassification (ATT, ATC and ATE), the exact method, no matching
//! (`method = NULL`), nearest neighbour with and without sampling weights,
//! and fits with `discard`. For nearest neighbour the weights are the ones this
//! crate derives from the pairs.
//!
//! Compared against the standardised table: weighted means, standardised mean
//! difference, variance ratio, eCDF mean and maximum, for each covariate and
//! each level of a factor, before and after matching. Against the
//! unstandardised table: the eQQ mean and maximum, and for pair methods the pair
//! distance. Against the sample size table: every row, for both groups. Against
//! the improvement table: every statistic. With no matching, MatchIt prints no
//! table after matching, and the statistics after matching are compared with
//! those before.
//!
//! Two departures are asserted in both values.
//!
//! - After subclassification and the exact method MatchIt computes the
//!   improvement of the variance ratio as `100 (before - after) / before`, and
//!   after the other methods on the absolute logarithm of the ratio, as its
//!   documentation says. This crate uses the logarithm throughout. The test
//!   checks that R's subclass value is the formula without the logarithm and
//!   ours the one with it.
//! - MatchIt reports an improvement of 0 where the statistic is 0 before and
//!   after, and negative infinity where it is 0 before only. This crate reports
//!   `None`.
//!
//! Where the difference in means is below the square root of the machine
//! epsilon MatchIt does not standardise the pair distance, so the standardised
//! pair distance is compared only above it.

#![expect(
    clippy::float_cmp,
    reason = "statistics that are exactly zero or one in the reference are compared for equality"
)]

#[path = "common/r_reference.rs"]
mod r_reference;

use chrono::NaiveDate;
use epistat_core::Estimand;
use epistat_matching::prelude::*;
use epistat_matching::{
    BalanceRecord, BalanceReport, BalanceReportOptions, BalanceSummary, BalanceWeights,
    CategoricalBalance, CovariateValue, NumericBalance, NumericBalanceTransform, SampleSizeTable,
    balance_improvement, balance_summary,
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

#[derive(Deserialize)]
struct Summ {
    all: Table,
    matched: Table,
    nn: Table,
    reduction: Table,
}

#[derive(Deserialize)]
struct Fit {
    name: String,
    estimand: String,
    #[serde(deserialize_with = "na_f64_vec")]
    weights: Vec<f64>,
    discarded: Vec<bool>,
    pairs: Vec<(String, String)>,
    std: Summ,
    raw: Summ,
    int_std: Option<Summ>,
    int_raw: Option<Summ>,
    int_order: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct Set {
    data: HashMap<String, Vec<Value>>,
    fits: Vec<Fit>,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    lalonde: Set,
    sim: Set,
}

/// The sums run over at most 614 terms, each rounded to a relative 1.1e-16, so
/// 1e-10 relative is a wide margin over the roughly 1e-13 the arithmetic
/// allows. The absolute bound is for statistics near zero, such as the
/// difference after exact matching, which is rounding noise in both tools.
fn tolerance() -> Tolerance {
    Tolerance::exact().rel(1e-10).abs(1e-12)
}

/// An improvement is a percentage of a difference of two statistics that each
/// carry a relative error of 1e-13, so its absolute error is below 1e-11 of the
/// larger statistic over the smaller. 1e-9 relative covers a ratio of 100.
fn improvement_tolerance() -> Tolerance {
    Tolerance::exact().rel(1e-9).abs(1e-9)
}

fn load() -> Reference {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/balance_weights.json"
    ));
    assert_r_version(&reference.versions["r"]);
    assert_eq!(reference.versions["matchit"], "4.8.2");
    reference
}

fn day() -> NaiveDate {
    NaiveDate::from_ymd_opt(2000, 1, 1).unwrap()
}

#[track_caller]
fn same(label: &str, got: Option<f64>, expected: Cell, tolerance: Tolerance) {
    match got {
        None => assert!(
            expected.0.is_nan(),
            "{label}: undefined here, R has {}",
            expected.0
        ),
        Some(value) => assert!(
            tolerance.accepts(value, expected.0),
            "{label}: {value} against R {}",
            expected.0
        ),
    }
}

/// An improvement where MatchIt reports 0 or negative infinity because the
/// statistic is 0 before matching is `None` here.
#[track_caller]
fn same_improvement(label: &str, got: Option<f64>, expected: Cell, before_is_zero: bool) {
    if before_is_zero {
        assert_eq!(got, None, "{label}: before matching the statistic is 0");
    } else {
        same(label, got, expected, improvement_tolerance());
    }
}

struct Sample {
    cases: Vec<BalanceRecord>,
    controls: Vec<BalanceRecord>,
    sampling: HashMap<String, f64>,
    treated: HashMap<String, bool>,
}

fn sample(data: &HashMap<String, Vec<Value>>, covariates: &[&str]) -> Sample {
    let mut out = Sample {
        cases: Vec::new(),
        controls: Vec::new(),
        sampling: HashMap::new(),
        treated: HashMap::new(),
    };
    for (i, id) in data["id"].iter().enumerate() {
        let id = id.as_str().unwrap().to_string();
        let treated = data["treat"][i].as_i64() == Some(1);
        let mut record = BalanceRecord::new(id.clone(), day());
        for name in covariates {
            let value = match &data[*name][i] {
                Value::String(text) => CovariateValue::Categorical(text.clone()),
                other => CovariateValue::Numeric(other.as_f64().unwrap()),
            };
            record.covariates.insert((*name).to_string(), value);
        }
        out.sampling
            .insert(id.clone(), data["sw"][i].as_f64().unwrap());
        out.treated.insert(id, treated);
        if treated {
            out.cases.push(record);
        } else {
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
    MatchOutcome::from_pairs(
        pairs
            .iter()
            .map(|(treated, control)| MatchedPair::new(treated.clone(), control.clone()))
            .collect(),
        estimand,
    )
}

const LALONDE: [&str; 6] = ["age", "educ", "race", "married", "nodegree", "re74"];
const LALONDE_EXACT: [&str; 3] = ["married", "nodegree", "race"];
const SIM: [&str; 6] = ["x1", "x2", "z", "b", "cnt", "f"];
const SIM_EXACT: [&str; 3] = ["b", "cnt", "f"];

/// The covariates of the fit's formula.
fn covariates(set: &str, fit: &Fit) -> &'static [&'static str] {
    match (set, fit.name.starts_with("exact")) {
        ("lalonde", false) => &LALONDE,
        ("lalonde", true) => &LALONDE_EXACT,
        (_, false) => &SIM,
        (_, true) => &SIM_EXACT,
    }
}

/// Where the weights come from.
#[derive(Clone, Copy, PartialEq)]
enum Source {
    /// Derived from the pairs by this crate.
    Pairs,
    /// The weights MatchIt returned.
    MatchIt,
}

fn weights_for(fit: &Fit, ids: &[String], sample: &Sample, source: Source) -> BalanceWeights {
    let sampling = fit.name.contains("sampling");
    let mut weights = if source == Source::Pairs {
        BalanceWeights::from_outcome(&outcome(&fit.pairs, estimand(&fit.estimand)))
            .expect("a matched estimand")
    } else if fit.name == "none" || fit.name == "none_sampling" {
        BalanceWeights::unmatched(&sample.cases, &sample.controls, estimand(&fit.estimand))
    } else {
        // Subclassification, the exact method and no matching with discarded
        // units: MatchIt's own weights.
        let given = ids
            .iter()
            .zip(&fit.weights)
            .filter(|(_, weight)| **weight > 0.0)
            .map(|(id, weight)| (id.clone(), *weight))
            .collect::<HashMap<_, _>>();
        BalanceWeights::from_unit_weights(
            &sample.cases,
            &sample.controls,
            &given,
            estimand(&fit.estimand),
        )
        .with_pairs(
            fit.pairs
                .iter()
                .map(|(case, control)| MatchedPair::new(case.clone(), control.clone()))
                .collect(),
        )
    };
    if sampling {
        weights = weights.with_sampling_weights(sample.sampling.clone());
    }
    weights.with_discarded(
        ids.iter()
            .zip(&fit.discarded)
            .filter(|(_, flag)| **flag)
            .map(|(id, _)| id.clone()),
    )
}

/// The tables after matching. MatchIt has none with no matching; without
/// discarded units the statistics after matching are those before, and with
/// them there is nothing to compare.
fn after<'a>(fit: &Fit, summary: &'a Summ) -> Option<&'a Table> {
    if summary.matched.is_empty() {
        assert!(fit.name.starts_with("none"));
        (fit.name != "none_discard").then_some(&summary.all)
    } else {
        Some(&summary.matched)
    }
}

/// What of a row is compared.
#[derive(Clone, Copy)]
struct Scope {
    /// The statistics after matching.
    post: bool,
    /// The eQQ distances of a variable that is not binary. With weights this
    /// crate derives from pairs they are left out, see `weights_for`.
    eqq_continuous: bool,
}

#[track_caller]
fn compare_numeric(
    label: &str,
    row: &NumericBalance,
    std: (&Row, Option<&Row>),
    raw: (&Row, Option<&Row>),
    scope: Scope,
) {
    let (pre, post) = std;
    let t = tolerance();
    let binary = pre["Var. Ratio"].0.is_nan();
    let mut cells = vec![
        ("mean case pre", row.mean_case_pre, pre["Means Treated"]),
        (
            "mean control pre",
            row.mean_control_pre,
            pre["Means Control"],
        ),
        ("smd pre", row.smd_pre, pre["Std. Mean Diff."]),
        ("vr pre", row.var_ratio_pre, pre["Var. Ratio"]),
        ("ecdf mean pre", row.ecdf_mean_diff_pre, pre["eCDF Mean"]),
        ("ecdf max pre", row.ecdf_max_diff_pre, pre["eCDF Max"]),
    ];
    if scope.eqq_continuous || binary {
        cells.push(("eqq mean pre", row.eqq_mean_diff_pre, raw.0["eQQ Mean"]));
        cells.push(("eqq max pre", row.eqq_max_diff_pre, raw.0["eQQ Max"]));
    }
    if let (true, Some(post), Some(raw_post)) = (scope.post, post, raw.1) {
        cells.extend([
            ("mean case post", row.mean_case_post, post["Means Treated"]),
            (
                "mean control post",
                row.mean_control_post,
                post["Means Control"],
            ),
            ("smd post", row.smd_post, post["Std. Mean Diff."]),
            ("vr post", row.var_ratio_post, post["Var. Ratio"]),
            ("ecdf mean post", row.ecdf_mean_diff_post, post["eCDF Mean"]),
            ("ecdf max post", row.ecdf_max_diff_post, post["eCDF Max"]),
        ]);
        if scope.eqq_continuous || binary {
            cells.push((
                "eqq mean post",
                row.eqq_mean_diff_post,
                raw_post["eQQ Mean"],
            ));
            cells.push(("eqq max post", row.eqq_max_diff_post, raw_post["eQQ Max"]));
        }
    }
    for (what, got, cell) in cells {
        same(&format!("{label} {what}"), got, cell, t);
    }
}

#[track_caller]
fn compare_levels(
    label: &str,
    factor: &CategoricalBalance,
    pre_table: &Table,
    post_table: Option<&Table>,
    scope: Scope,
) {
    for level in &factor.levels {
        let name = format!("{}{}", factor.name, level.level);
        let pre = &pre_table[&name];
        let label = format!("{label} {name}");
        let t = tolerance();
        same(
            &format!("{label} p case pre"),
            level.p_case_pre,
            pre["Means Treated"],
            t,
        );
        same(
            &format!("{label} p control pre"),
            level.p_control_pre,
            pre["Means Control"],
            t,
        );
        same(
            &format!("{label} smd pre"),
            level.smd_pre,
            pre["Std. Mean Diff."],
            t,
        );
        if let (true, Some(post_table)) = (scope.post, post_table) {
            let post = &post_table[&name];
            same(
                &format!("{label} p case post"),
                level.p_case_post,
                post["Means Treated"],
                t,
            );
            same(
                &format!("{label} p control post"),
                level.p_control_post,
                post["Means Control"],
                t,
            );
            same(
                &format!("{label} smd post"),
                level.smd_post,
                post["Std. Mean Diff."],
                t,
            );
        }
    }
}

#[track_caller]
fn compare_sample_sizes(label: &str, table: &SampleSizeTable, fit: &Fit) {
    let t = tolerance();
    for (group, sizes) in [("Treated", &table.cases), ("Control", &table.controls)] {
        let nn = &fit.std.nn;
        let get = |row: &str| nn[row][group].0;
        #[expect(clippy::cast_precision_loss, reason = "counts below 2^53")]
        let count = |value: usize| value as f64;
        for (row, got) in [
            ("All", count(sizes.all)),
            ("Matched", count(sizes.matched)),
            ("Unmatched", count(sizes.unmatched)),
            ("Discarded", count(sizes.discarded)),
        ] {
            assert!(
                t.accepts(got, get(row)),
                "{label} {group} {row}: {got} against {}",
                get(row)
            );
        }
        for (row, got) in [
            ("All (ESS)", sizes.all_ess),
            ("Matched (ESS)", sizes.matched_ess),
        ] {
            assert!(
                t.accepts(got.unwrap(), get(row)),
                "{label} {group} {row}: {got:?} against {}",
                get(row)
            );
        }
    }
}

#[track_caller]
fn compare_pair_distances(label: &str, summary: &BalanceSummary, fit: &Fit) {
    let (std_all, std_matched) = (&fit.std.all, &fit.std.matched);
    let raw_matched = &fit.raw.matched;
    let mut compared = 0;
    for distance in &summary.pair_distances {
        let Some(row) = std_matched.get(&distance.name) else {
            continue;
        };
        let raw = &raw_matched[&distance.name];
        same(
            &format!("{label} {} pair dist", distance.name),
            distance.raw,
            raw["Pair Dist."],
            tolerance(),
        );
        let difference = (std_matched[&distance.name]["Means Treated"].0
            - std_matched[&distance.name]["Means Control"].0)
            .abs();
        if difference > f64::EPSILON.sqrt() {
            same(
                &format!("{label} {} std pair dist", distance.name),
                distance.standardised,
                row["Std. Pair Dist."],
                tolerance(),
            );
        }
        let _ = std_all;
        compared += 1;
    }
    assert!(compared >= 6, "{label}: {compared} pair distances");
}

/// Improvements against `summary(improvement = TRUE)`.
#[track_caller]
fn compare_improvement(label: &str, report: &BalanceReport, fit: &Fit) {
    let improvement = balance_improvement(report);
    let subclass_formula = fit.name.starts_with("subclass");
    let (std, raw) = (&fit.std, &fit.raw);
    let mut compared = 0;
    for row in &improvement.numeric {
        let Some(reduction) = std.reduction.get(&row.name) else {
            continue;
        };
        let (pre, post) = (&std.all[&row.name], &std.matched[&row.name]);
        let zero = |cell: Cell| cell.0 == 0.0;
        let label = format!("{label} {}", row.name);
        same_improvement(
            &format!("{label} smd"),
            row.smd,
            reduction["Std. Mean Diff."],
            zero(pre["Std. Mean Diff."]),
        );
        same_improvement(
            &format!("{label} ecdf mean"),
            row.ecdf_mean,
            reduction["eCDF Mean"],
            zero(pre["eCDF Mean"]),
        );
        same_improvement(
            &format!("{label} ecdf max"),
            row.ecdf_max,
            reduction["eCDF Max"],
            zero(pre["eCDF Max"]),
        );
        let (vr_before, vr_after) = (pre["Var. Ratio"].0, post["Var. Ratio"].0);
        if vr_before.is_finite() && vr_after.is_finite() && vr_before > 0.0 && vr_after > 0.0 {
            let logged =
                100.0 * (vr_before.ln().abs() - vr_after.ln().abs()) / vr_before.ln().abs();
            let plain = 100.0 * (vr_before - vr_after) / vr_before;
            if subclass_formula {
                // MatchIt's subclass summary has no logarithm; this crate does.
                assert!(
                    tolerance().accepts(reduction["Var. Ratio"].0, plain)
                        || vr_before == 1.0
                        || (vr_before == vr_after),
                    "{label}: R's variance ratio improvement {} is not {plain}",
                    reduction["Var. Ratio"].0
                );
                if vr_before.ln().abs() > 0.0 {
                    assert!(
                        improvement_tolerance().accepts(row.var_ratio.unwrap(), logged),
                        "{label}: {:?} against {logged}",
                        row.var_ratio
                    );
                }
            } else {
                same_improvement(
                    &format!("{label} vr"),
                    row.var_ratio,
                    reduction["Var. Ratio"],
                    vr_before == 1.0,
                );
            }
        } else {
            assert_eq!(row.var_ratio, None, "{label} vr");
        }
        let (raw_pre, raw_post) = (&raw.all[&row.name], &raw.matched[&row.name]);
        let raw_reduction = &raw.reduction[&row.name];
        if raw_pre["eQQ Mean"].0.is_finite() && raw_post["eQQ Mean"].0.is_finite() {
            let expected = |name: &str| {
                let before = raw_pre[name].0.abs();
                (before > 0.0).then(|| 100.0 * (before - raw_post[name].0.abs()) / before)
            };
            for (what, got, name) in [
                ("eqq mean", row.eqq_mean, "eQQ Mean"),
                ("eqq max", row.eqq_max, "eQQ Max"),
            ] {
                match expected(name) {
                    Some(value) => {
                        assert!(
                            improvement_tolerance().accepts(got.unwrap(), value),
                            "{label} {what}: {got:?} against {value}"
                        );
                        same(
                            &format!("{label} {what} R"),
                            Some(value),
                            raw_reduction[name],
                            improvement_tolerance(),
                        );
                    }
                    None => assert_eq!(got, None, "{label} {what}"),
                }
            }
        }
        compared += 1;
    }
    assert!(compared > 0, "{label}: no improvement compared");
}

#[track_caller]
fn compare_interactions(label: &str, report: &BalanceReport, fit: &Fit, eqq_continuous: bool) {
    let (std, raw) = (fit.int_std.as_ref().unwrap(), fit.int_raw.as_ref().unwrap());
    // MatchIt's rows of squares and products, in its order.
    let r_names = fit
        .int_order
        .as_ref()
        .unwrap()
        .iter()
        .filter(|name| name.contains('²') || name.contains(" * "))
        .cloned()
        .collect::<Vec<_>>();
    let ours = report
        .numeric
        .iter()
        .filter(|row| row.name.contains('^') || row.name.contains(" * "))
        .collect::<Vec<_>>();
    let our_names = ours
        .iter()
        .map(|row| row.name.replace("^2", "²"))
        .collect::<Vec<_>>();
    assert_eq!(
        our_names, r_names,
        "{label}: the interaction rows and their order"
    );
    for row in ours {
        let name = row.name.replace("^2", "²");
        compare_numeric(
            &format!("{label} {name}"),
            row,
            (&std.all[&name], Some(&std.matched[&name])),
            (&raw.all[&name], Some(&raw.matched[&name])),
            Scope {
                post: true,
                eqq_continuous,
            },
        );
    }
}

fn run_fit(set_name: &str, set: &Set, fit: &Fit, source: Source) {
    let covs = covariates(set_name, fit);
    let sample = sample(&set.data, covs);
    let ids = set.data["id"]
        .iter()
        .map(|id| id.as_str().unwrap().to_string())
        .collect::<Vec<_>>();
    let weights = weights_for(fit, &ids, &sample, source);
    let interactions = fit.int_std.is_some();
    let options = BalanceReportOptions {
        numeric_transforms: if interactions {
            NumericBalanceTransform::SquaresAndPairwiseInteractions
        } else {
            NumericBalanceTransform::None
        },
        ..BalanceReportOptions::default()
    };
    let summary = balance_summary(&sample.cases, &sample.controls, &weights, &options).unwrap();
    let label = format!(
        "{set_name} {} ({})",
        fit.name,
        if source == Source::Pairs {
            "pairs"
        } else {
            "R weights"
        }
    );
    let std_post = after(fit, &fit.std);
    let raw_post = after(fit, &fit.raw);
    let scope = Scope {
        post: std_post.is_some(),
        eqq_continuous: source == Source::MatchIt,
    };
    for row in summary
        .report
        .numeric
        .iter()
        .filter(|row| !row.name.contains('^') && !row.name.contains(" * "))
    {
        compare_numeric(
            &format!("{label} {}", row.name),
            row,
            (
                &fit.std.all[&row.name],
                std_post.map(|table| &table[&row.name]),
            ),
            (
                &fit.raw.all[&row.name],
                raw_post.map(|table| &table[&row.name]),
            ),
            scope,
        );
    }
    for factor in &summary.report.categorical {
        compare_levels(&label, factor, &fit.std.all, std_post, scope);
    }
    compare_sample_sizes(&label, &summary.sample_sizes, fit);
    if !fit.pairs.is_empty() {
        compare_pair_distances(&label, &summary, fit);
    }
    if !fit.std.matched.is_empty() && source == Source::MatchIt {
        compare_improvement(&label, &summary.report, fit);
    }
    if interactions {
        compare_interactions(&label, &summary.report, fit, scope.eqq_continuous);
    }
}

fn fits_named<'a>(
    reference: &'a Reference,
    prefix: &'a str,
) -> Vec<(&'static str, &'a Set, &'a Fit)> {
    [("lalonde", &reference.lalonde), ("sim", &reference.sim)]
        .into_iter()
        .flat_map(|(name, set)| {
            set.fits
                .iter()
                .filter(move |fit| fit.name.starts_with(prefix))
                .map(move |fit| (name, set, fit))
        })
        .collect()
}

fn check_all(prefix: &str, expected: usize) {
    let reference = load();
    let fits = fits_named(&reference, prefix);
    assert_eq!(fits.len(), expected, "{prefix}");
    for (set_name, set, fit) in fits {
        run_fit(set_name, set, fit, Source::MatchIt);
        if !fit.pairs.is_empty() {
            run_fit(set_name, set, fit, Source::Pairs);
        }
    }
}

/// Subclassification for the ATT, ATC and ATE, and its fits with sampling weights,
/// discarded units and interactions.
#[test]
fn subclassification_agrees_with_summary_matchit() {
    check_all("subclass", 12);
}

/// The exact method, whose weights are subclass weights.
#[test]
fn exact_method_agrees_with_summary_matchit() {
    check_all("exact", 6);
}

/// No matching: the statistics before matching on all units, with sampling
/// weights and with units discarded.
#[test]
fn balance_without_matching_agrees_with_summary_matchit() {
    check_all("none", 6);
}

/// Nearest neighbour, with and without sampling weights and discarded units.
#[test]
fn pair_methods_agree_with_summary_matchit() {
    check_all("att_", 14);
    check_all("atc_", 2);
}
