//! Time of optimal matching on one stratum of register size, run with
//! `cargo test --release --test optimal_scale -- --ignored --nocapture`.
//!
//! 2,000 anchors and 8,000 candidates with scores on a line and a caliper of
//! 0.02, which leaves a few hundred candidates per anchor.

use chrono::NaiveDate;
use epistat_matching::prelude::*;
use epistat_matching::{FullMatchConfig, OptimalRequest, RapidHashMap, match_full, match_optimal};
use std::time::Instant;

#[test]
#[ignore = "timing, not a check"]
fn one_stratum_of_2000_anchors_and_8000_candidates() {
    let mut state = 12_345_u64;
    let mut draw = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1_u64 << 53) as f64
    };
    let birth = NaiveDate::from_ymd_opt(2000, 1, 1).expect("valid date");
    let mut anchors = Vec::new();
    let mut candidates = Vec::new();
    let mut scores: RapidHashMap<String, f64> = RapidHashMap::default();
    for (rows, prefix, count, shift) in [
        (&mut anchors, "a", 2_000, 0.0),
        (&mut candidates, "c", 8_000, 0.0),
    ] {
        for index in 0..count {
            let id = format!("{prefix}{index}");
            scores.insert(id.clone(), draw() * 0.9 + shift);
            rows.push(BaseRecord::new(id, birth));
        }
    }
    let width = DistanceCaliper::new(0.02).expect("caliper");
    let distance = DistanceConfig::propensity_score_map(scores, Some(width));
    for ratio in [1, 2] {
        let criteria = MatchingCriteria::builder().match_ratio(ratio).build();
        let request = OptimalRequest::new(&criteria, &()).with_distance_config(&distance);
        let start = Instant::now();
        let outcome = match_optimal(&anchors, &candidates, &request).expect("feasible");
        println!(
            "optimal ratio {ratio}: {} pairs in {:?}",
            outcome.pairs.len(),
            start.elapsed()
        );
    }
    let criteria = MatchingCriteria::builder().build();
    let request = OptimalRequest::new(&criteria, &()).with_distance_config(&distance);
    let start = Instant::now();
    let outcome =
        match_full(&anchors, &candidates, &request, &FullMatchConfig::default()).expect("feasible");
    println!(
        "full: {} subclasses in {:?}",
        outcome.subclasses.len(),
        start.elapsed()
    );
}
