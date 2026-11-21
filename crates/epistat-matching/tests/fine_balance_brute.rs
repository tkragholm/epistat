//! Near-fine and refined balance against exhaustive search.
//!
//! Each of 60 small random problems (4 anchors, 7 or 9 candidates, two nested
//! balance levels, one or two candidates per anchor) has every
//! assignment of candidates to anchors enumerated. The best is the one with the
//! least deviation at the first level, then the second, then the least total
//! distance. The crate's match must have the same deviations, and a total
//! distance equal to the best to the arithmetic of its integer costs.

use chrono::NaiveDate;
use epistat_matching::prelude::*;
use epistat_matching::{FineBalance, RapidHashMap, match_fine_balance};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1_u64 << 53) as f64
    }
    fn below(&mut self, n: usize) -> usize {
        let _ = self.next();
        usize::try_from(self.0 >> 33).expect("31 bits") % n
    }
}

fn deviations(counts: &[usize], targets: &[usize]) -> usize {
    counts
        .iter()
        .zip(targets)
        .map(|(a, b)| a.abs_diff(*b))
        .sum()
}

/// Every way to give each anchor `k` distinct candidates, used once overall.
fn assign(
    anchor: usize,
    k: usize,
    used: &mut Vec<bool>,
    chosen: &mut Vec<usize>,
    start: usize,
    out: &mut Vec<Vec<usize>>,
    anchors: usize,
) {
    if anchor == anchors {
        out.push(chosen.clone());
        return;
    }
    let taken = chosen.len() - anchor * k;
    if taken == k {
        assign(anchor + 1, k, used, chosen, 0, out, anchors);
        return;
    }
    for c in start..used.len() {
        if !used[c] {
            used[c] = true;
            chosen.push(c);
            assign(anchor, k, used, chosen, c + 1, out, anchors);
            chosen.pop();
            used[c] = false;
        }
    }
}

#[test]
fn matches_exhaustive_search() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let birth = NaiveDate::from_ymd_opt(2000, 1, 1).expect("date");
    for case in 0..60 {
        let k = 1 + case % 2;
        let (na, nc) = (4, 5 + 2 * k);
        let level_a = |r: &mut Rng| r.below(2);
        let level_b = |r: &mut Rng| r.below(2);
        let units: Vec<(usize, usize)> = (0..na + nc)
            .map(|_| (level_a(&mut rng), level_b(&mut rng)))
            .collect();
        let x: Vec<f64> = (0..na + nc).map(|_| rng.next()).collect();
        let mut records = Vec::new();
        let mut scores: RapidHashMap<String, f64> = RapidHashMap::default();
        for (i, (a, b)) in units.iter().enumerate() {
            let id = i.to_string();
            let mut record = BaseRecord::new(id.clone(), birth);
            record.strata.insert("a".into(), a.to_string());
            record.strata.insert("b".into(), format!("{a}{b}"));
            scores.insert(id, x[i]);
            records.push(record);
        }
        let (anchors, candidates) = records.split_at(na);
        let distance = DistanceConfig::propensity_score_map(scores, None);
        let criteria = MatchingCriteria::builder().match_ratio(k).build();
        let request = OptimalRequest::new(&criteria, &()).with_distance_config(&distance);
        let balance = FineBalance {
            levels: vec![vec!["a".into()], vec!["b".into()]],
            ..FineBalance::default()
        };
        let got = match_fine_balance(anchors, candidates, &request, &balance).expect("feasible");

        let target_a: Vec<usize> = (0..2)
            .map(|v| k * units[..na].iter().filter(|u| u.0 == v).count())
            .collect();
        let target_b: Vec<usize> = (0..4)
            .map(|v| k * units[..na].iter().filter(|u| u.0 * 2 + u.1 == v).count())
            .collect();
        let mut all = Vec::new();
        assign(0, k, &mut vec![false; nc], &mut Vec::new(), 0, &mut all, na);
        let mut best: Option<(usize, usize, f64)> = None;
        for chosen in &all {
            let (mut ca, mut cb) = (vec![0; 2], vec![0; 4]);
            let mut total = 0.0;
            for (slot, c) in chosen.iter().enumerate() {
                let anchor = slot / k;
                let u = units[na + c];
                ca[u.0] += 1;
                cb[u.0 * 2 + u.1] += 1;
                total += (x[anchor] - x[na + c]).abs();
            }
            let key = (
                deviations(&ca, &target_a),
                deviations(&cb, &target_b),
                total,
            );
            let better = best.is_none_or(|b| {
                (key.0, key.1) < (b.0, b.1) || ((key.0, key.1) == (b.0, b.1) && key.2 < b.2)
            });
            if better {
                best = Some(key);
            }
        }
        let best = best.expect("an assignment");
        let levels = &got.levels;
        assert_eq!(
            (levels[0].total_deviation(), levels[1].total_deviation()),
            (best.0, best.1),
            "case {case}"
        );
        let total: f64 = got
            .outcome
            .pairs
            .iter()
            .map(|p| {
                let a: usize = p.case_id.parse().expect("id");
                let c: usize = p.control_id.parse().expect("id");
                (x[a] - x[c]).abs()
            })
            .sum();
        // Costs are scaled to 2^40 of the largest distance, at most 1, with
        // 1.5 / 2^40 per pair of rounding.
        assert!(
            (total - best.2).abs() <= 1e-9,
            "case {case}: total {total}, best {}",
            best.2
        );
    }
}
