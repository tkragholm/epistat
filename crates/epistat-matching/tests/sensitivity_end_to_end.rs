//! Sensitivity analysis from the engine's own output: a match by the engine,
//! its `get_matches`, the matched sets and the bounds. The R comparisons in
//! `r_sensitivity.rs` start from numbers; this test checks that the crate's
//! output reaches the bounds with the right treated and control outcomes.

use chrono::NaiveDate;
use epistat_matching::effects::UnitData;
use epistat_matching::output::{Units, get_matches};
use epistat_matching::prelude::*;
use epistat_matching::sensitivity::{
    WilcoxonDistribution, hodges_lehmann_bounds, matched_sets, pair_differences, wilcoxon_bounds,
};

#[test]
fn pairs_from_the_engine_give_the_bounds_of_their_differences() {
    let birth = NaiveDate::from_ymd_opt(1990, 6, 1).expect("valid date");
    let (mut anchors, mut candidates) = (Vec::new(), Vec::new());
    let (mut ids, mut treated, mut outcome) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..120_u32 {
        let id = format!("u{i:03}");
        let is_anchor = i < 40;
        let record = BaseRecord::new(id.clone(), birth);
        if is_anchor {
            anchors.push(record);
        } else {
            candidates.push(record);
        }
        ids.push(id);
        treated.push(is_anchor);
        // Treated units gain about 1 over the controls, with a spread.
        outcome.push(f64::from((i * 37) % 11) / 5.0 + f64::from(u8::from(is_anchor)));
    }
    let matched = MatchJob::new_standard(&anchors, &candidates, 5).run();
    assert_eq!(matched.matched_cases, 40);
    let units = Units::new(ids.clone(), treated.clone()).expect("unique ids");
    let long = get_matches(&matched, &units, false).expect("fixed roles");
    let data = UnitData::new(ids.clone(), treated, outcome.clone()).expect("valid units");
    let sets = matched_sets(&long, &data).expect("pairs");
    assert_eq!(sets.len(), 40);

    let row = |id: &str| ids.iter().position(|u| u == id).expect("known id");
    let mut by_hand: Vec<f64> = matched
        .pairs
        .iter()
        .map(|p| outcome[row(&p.case_id)] - outcome[row(&p.control_id)])
        .collect();
    let mut got = pair_differences(&sets).expect("1:1 pairs");
    by_hand.sort_by(f64::total_cmp);
    got.sort_by(f64::total_cmp);
    assert_eq!(got, by_hand);

    let bounds =
        wilcoxon_bounds(&sets, &[1.0, 2.0, 4.0], WilcoxonDistribution::Normal).expect("bounds");
    assert!(bounds[0].p_upper < 0.05, "a clear effect at gamma 1");
    assert!(bounds[2].p_upper > bounds[0].p_upper);
    let hl = hodges_lehmann_bounds(&sets, &[1.0, 2.0]).expect("bounds");
    assert!(hl[1].lower <= hl[0].lower && hl[0].upper <= hl[1].upper);
}
