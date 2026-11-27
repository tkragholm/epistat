//! Scores held by unit position give the same matches as scores looked up by
//! id, and nearest-neighbour matching with replacement, which searches the
//! anchors at the same time, gives the same matches on every pool and the
//! matches of a search of every candidate.
//!
//! The units are enough to pass the sizes at which the candidate pool is sorted
//! on the rayon pool and the anchors are searched together.

use epistat_matching::prelude::*;
use epistat_matching::{BaseRecord, ScoreTable};
use epistat_testkit::{POOLS, assert_same_on_pools};
use rapidhash::RapidHashMap;

const ANCHORS: usize = 5_000;
const CANDIDATES: usize = 70_000;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
}

struct Units {
    anchors: Vec<BaseRecord>,
    candidates: Vec<BaseRecord>,
    /// Ids of the anchors and then the candidates, and their scores. Scores
    /// are rounded to three digits so that many are equal, and one in 50 of
    /// the candidates has none.
    ids: Vec<String>,
    scores: Vec<f64>,
}

fn units(seed: u64) -> Units {
    let mut rng = Lcg(seed);
    let birth = chrono::NaiveDate::from_ymd_opt(2010, 1, 1).unwrap();
    let mut ids = Vec::new();
    let mut scores = Vec::new();
    let mut anchors = Vec::new();
    let mut candidates = Vec::new();
    for i in 0..ANCHORS + CANDIDATES {
        let anchor = i < ANCHORS;
        let id = format!("{}{i}", if anchor { "a" } else { "c" });
        let base = (rng.next() % 1000) as f64 / 1000.0;
        let score = if anchor { base * 0.6 + 0.4 } else { base };
        let score = if !anchor && rng.next().is_multiple_of(50) {
            f64::NAN
        } else {
            score
        };
        let record = BaseRecord::new(id.clone(), birth);
        if anchor {
            anchors.push(record);
        } else {
            candidates.push(record);
        }
        ids.push(id);
        scores.push(score);
    }
    Units {
        anchors,
        candidates,
        ids,
        scores,
    }
}

fn run(units: &Units, config: DistanceConfig, replacement: bool, ratio: usize) -> Vec<MatchedPair> {
    MatchJob::new_standard(&units.anchors, &units.candidates, 7)
        .with_nearest()
        .with_m_order(MatchOrder::Data)
        .with_ratio(MatchRatio::new(ratio).unwrap())
        .with_replacement(replacement)
        .with_distance_config(config)
        .run()
        .expect("a valid job")
        .pairs
}

fn by_anchor(pairs: &[MatchedPair]) -> RapidHashMap<&str, Vec<&str>> {
    let mut map: RapidHashMap<&str, Vec<&str>> = RapidHashMap::default();
    for pair in pairs {
        map.entry(pair.case_id.as_str())
            .or_default()
            .push(pair.control_id.as_str());
    }
    map
}

#[test]
fn a_table_in_the_order_of_the_units_matches_as_one_looked_up_by_id() {
    let units = units(11);
    let aligned = ScoreTable::new(units.ids.clone().into(), units.scores.clone().into());
    let by_id: RapidHashMap<String, f64> = units
        .ids
        .iter()
        .cloned()
        .zip(units.scores.iter().copied())
        .collect();
    // The same table with its units in another order is read by id.
    let shuffled = ScoreTable::from_map(&by_id);
    for (replacement, ratio) in [(false, 1), (false, 3), (true, 1), (true, 2)] {
        let want = run(
            &units,
            DistanceConfig::propensity_score_table(aligned.clone(), None),
            replacement,
            ratio,
        );
        assert_ne!(want.len(), 0, "the run matched nobody");
        let got = run(
            &units,
            DistanceConfig::propensity_score_table(shuffled.clone(), None),
            replacement,
            ratio,
        );
        assert_eq!(want, got, "replacement {replacement}, ratio {ratio}");
        let from_map = run(
            &units,
            DistanceConfig::propensity_score_map(by_id.clone(), None),
            replacement,
            ratio,
        );
        assert_eq!(want, from_map, "replacement {replacement}, ratio {ratio}");
    }
}

#[test]
fn matching_with_replacement_takes_the_nearest_candidates_by_slice_position() {
    let units = units(23);
    let table = ScoreTable::new(units.ids.clone().into(), units.scores.clone().into());
    for ratio in [1, 3] {
        let pairs = run(
            &units,
            DistanceConfig::propensity_score_table(table.clone(), None),
            true,
            ratio,
        );
        let got = by_anchor(&pairs);
        let candidate_scores = &units.scores[ANCHORS..];
        for (i, anchor) in units.anchors.iter().enumerate().step_by(25) {
            let x = units.scores[i];
            let mut ranked: Vec<(f64, usize)> = candidate_scores
                .iter()
                .enumerate()
                .filter(|(_, s)| s.is_finite())
                .map(|(j, s)| ((x - s).abs(), j))
                .collect();
            ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let want: Vec<&str> = ranked
                .iter()
                .take(ratio)
                .map(|(_, j)| units.candidates[*j].id())
                .collect();
            assert_eq!(
                got.get(anchor.id()).map(Vec::as_slice),
                Some(want.as_slice()),
                "anchor {i}, ratio {ratio}"
            );
        }
    }
}

#[test]
fn matching_with_replacement_is_the_same_on_every_pool() {
    let units = units(37);
    let table = ScoreTable::new(units.ids.clone().into(), units.scores.clone().into());
    assert_same_on_pools(&POOLS, || {
        let mut digest = Vec::new();
        for (replacement, ratio) in [(true, 1), (true, 3), (false, 2)] {
            let pairs = run(
                &units,
                DistanceConfig::propensity_score_table(table.clone(), None),
                replacement,
                ratio,
            );
            digest.push(pairs.len() as f64);
            for pair in &pairs {
                for text in [&pair.case_id, &pair.control_id] {
                    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
                    for byte in text.bytes() {
                        hash ^= u64::from(byte);
                        hash = hash.wrapping_mul(0x0100_0000_01b3);
                    }
                    digest.push(f64::from_bits(hash));
                }
            }
        }
        digest
    });
}
