//! Wall time and peak memory of `balance_summary` on weights given from
//! outside, for `N` units held in a [`Cohort`]. The program makes its data in
//! memory, so the peak memory of the process is the cohort, the weights and
//! the report. The driver and the environment are described in
//! `bench/tier1/common.rs`; there is no R counterpart.
//!
//! Method `balance_weights`: every tenth unit is a case, with matching weight
//! one. The unit after each case is its matched control, with weight one and
//! a pair, the unit after that has weight one half, and the other controls
//! are unmatched. Every 97th unit is discarded. The covariates are four
//! numeric ones, one of them binary, and two categorical ones of two and five
//! levels, with some values missing. The numbers written are the statistics
//! of the report and the sample sizes, so two builds can be compared for
//! equality.

#[path = "../../../bench/tier1/common.rs"]
mod common;

use chrono::NaiveDate;
use common::{Lines, Rng};
use epistat_matching::{
    BalanceReportOptions, BalanceWeights, Cohort, CohortRow, Estimand, MatchedPair, RapidHashMap,
    UnitWeightSet, balance_summary,
};

const SEED: u64 = 20_261_126;
const REGIONS: [&str; 5] = ["north", "central", "south", "zealand", "capital"];

fn cohort(n: usize) -> Cohort {
    let mut rng = Rng(SEED);
    let ids: Vec<String> = (0..n).map(|i| format!("u{i:08}")).collect();
    let birth = vec![NaiveDate::from_ymd_opt(2000, 1, 1).expect("a date"); n];
    let mut age = Vec::with_capacity(n);
    let mut income = Vec::with_capacity(n);
    let mut score = Vec::with_capacity(n);
    let mut smoker = Vec::with_capacity(n);
    let mut sex = Vec::with_capacity(n);
    let mut region = Vec::with_capacity(n);
    for i in 0..n {
        let case = i.is_multiple_of(10);
        let shift = if case { 0.3 } else { 0.0 };
        age.push(Some((40.0 + 10.0 * (rng.normal() + shift)).round()));
        income.push((i % 31 != 7).then(|| (5.0 + 0.5 * rng.normal() + shift).exp()));
        score.push(Some(rng.normal() + shift));
        smoker.push(Some(f64::from(u8::from(rng.bernoulli(0.25 + shift / 3.0)))));
        sex.push(Some(if rng.bernoulli(0.5) { "F" } else { "M" }));
        region.push(
            (i % 53 != 11).then(|| REGIONS[usize::try_from(rng.below(5)).expect("a small index")]),
        );
    }
    Cohort::builder(ids, birth)
        .numeric_covariate("age", age)
        .numeric_covariate("income", income)
        .numeric_covariate("score", score)
        .numeric_covariate("smoker", smoker)
        .categorical_covariate("sex", sex)
        .categorical_covariate("region", region)
        .build()
        .expect("a valid cohort")
}

fn main() {
    let (method, n) = (common::method(), common::size());
    assert_eq!(method, "balance_weights", "unknown method {method}");
    let cohort = cohort(n);
    let rows: Vec<CohortRow<'_>> = cohort.rows().collect();
    let id = |i: usize| format!("u{i:08}");
    let (mut cases, mut controls) = (Vec::new(), Vec::new());
    let mut anchor = RapidHashMap::default();
    let mut candidate = RapidHashMap::default();
    let mut pairs = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        match i % 10 {
            0 => {
                cases.push(*row);
                anchor.insert(id(i), 1.0);
                if i + 1 < n {
                    pairs.push(MatchedPair::new(id(i), id(i + 1)));
                }
            }
            1 => {
                controls.push(*row);
                candidate.insert(id(i), 1.0);
            }
            2 => {
                controls.push(*row);
                candidate.insert(id(i), 0.5);
            }
            _ => controls.push(*row),
        }
    }
    let weights = BalanceWeights::new(Estimand::Att, UnitWeightSet { anchor, candidate })
        .with_pairs(pairs)
        .with_discarded((0..n).filter(|i| i % 97 == 5).map(id));
    let options = BalanceReportOptions::default();
    let (times, summary) = common::timed(common::reps(), || {
        balance_summary(&cases, &controls, &weights, &options).expect("valid weights")
    });
    let mut lines = Lines::default();
    for row in &summary.report.numeric {
        let name = &row.name;
        for (stat, value) in [
            ("mean_case_pre", row.mean_case_pre),
            ("mean_control_pre", row.mean_control_pre),
            ("smd_pre", row.smd_pre),
            ("var_ratio_pre", row.var_ratio_pre),
            ("ecdf_mean_pre", row.ecdf_mean_diff_pre),
            ("ecdf_max_pre", row.ecdf_max_diff_pre),
            ("eqq_mean_pre", row.eqq_mean_diff_pre),
            ("eqq_max_pre", row.eqq_max_diff_pre),
            ("mean_case_post", row.mean_case_post),
            ("mean_control_post", row.mean_control_post),
            ("smd_post", row.smd_post),
            ("var_ratio_post", row.var_ratio_post),
            ("ecdf_mean_post", row.ecdf_mean_diff_post),
            ("ecdf_max_post", row.ecdf_max_diff_post),
            ("eqq_mean_post", row.eqq_mean_diff_post),
            ("eqq_max_post", row.eqq_max_diff_post),
        ] {
            lines.push_opt(&format!("{name}_{stat}"), value);
        }
    }
    for row in &summary.report.categorical {
        lines.push_opt(&format!("{}_cramers_v_pre", row.name), row.cramers_v_pre);
        lines.push_opt(&format!("{}_cramers_v_post", row.name), row.cramers_v_post);
        for level in &row.levels {
            let name = format!("{}{}", row.name, level.level);
            lines.push_opt(&format!("{name}_p_case_pre"), level.p_case_pre);
            lines.push_opt(&format!("{name}_p_control_pre"), level.p_control_pre);
            lines.push_opt(&format!("{name}_smd_pre"), level.smd_pre);
            lines.push_opt(&format!("{name}_p_case_post"), level.p_case_post);
            lines.push_opt(&format!("{name}_p_control_post"), level.p_control_post);
            lines.push_opt(&format!("{name}_smd_post"), level.smd_post);
        }
    }
    for distance in &summary.pair_distances {
        lines.push_opt(&format!("{}_pair_raw", distance.name), distance.raw);
        lines.push_opt(
            &format!("{}_pair_std", distance.name),
            distance.standardised,
        );
    }
    for (group, sizes) in [
        ("cases", &summary.sample_sizes.cases),
        ("controls", &summary.sample_sizes.controls),
    ] {
        lines.push(&format!("{group}_all"), sizes.all as f64);
        lines.push_opt(&format!("{group}_all_ess"), sizes.all_ess);
        lines.push(&format!("{group}_matched"), sizes.matched as f64);
        lines.push_opt(&format!("{group}_matched_ess"), sizes.matched_ess);
        lines.push(&format!("{group}_unmatched"), sizes.unmatched as f64);
        lines.push(&format!("{group}_discarded"), sizes.discarded as f64);
    }
    common::finish(&method, n, &times, &lines);
}
