//! Generalised full matching checked against quickmatch 0.2.3 and MatchIt
//! 4.8.2's `matchit(method = "quick")`.
//!
//! The reference is `tests/data/matchit_quick.json`, written by
//! `tests/data/matchit_quick.R`. It holds simulated points on a line, in the
//! plane and in space, the linear predictor of the propensity score of
//! lalonde and two covariates for a Mahalanobis distance. No two units are at
//! the same distance from a third, so the order in which quickmatch breaks
//! ties does not enter.
//!
//! quickmatch's subclasses depend on how seeds are found. The checks are:
//!
//! 1. For each seed method, the partition into subclasses equals quickmatch's
//!    unit for unit, under several treatment constraints, size constraints and
//!    calipers.
//! 2. Whatever the partition, every subclass meets the constraints and the
//!    caliper, and the largest within-subclass distance is at most four times
//!    the smallest possible on small sets solved by exhaustive search. The
//!    bound is the one of the paper.
//! 3. The weights computed from the subclasses equal MatchIt's, by estimand,
//!    on a line, for lalonde's score, and with a caliper. MatchIt rescales the
//!    weights of each group to sum to the number of its kept units, and the
//!    test applies the same rescaling to ours. The script checks the unscaled
//!    weights against MatchIt's the same way.
//! 4. `run_spec` with a Mahalanobis distance reproduces MatchIt's
//!    `distance = "mahalanobis"` partition and weights.

#[path = "common/r_reference.rs"]
mod r_reference;

use chrono::NaiveDate;
use epistat_matching::prelude::*;
use epistat_matching::{
    BalanceRecord, CovariateValue, DistanceSpec, MatchMethod, MatchSpec, QuickMatchError,
    QuickMatchOptions, QuickMatchOutcome, SeedMethod, SpecInput, SpecOutput, quick_clusters,
    quick_match_subclasses, run_spec,
};

use r_reference::{Versions, assert_r_version, reference as load_reference};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    datasets: Vec<Dataset>,
}

#[derive(Deserialize)]
struct Dataset {
    name: String,
    dim: usize,
    treat: Vec<u8>,
    points: Vec<f64>,
    runs: Vec<QmRun>,
    matchit: Vec<MatchItRun>,
    #[serde(default)]
    z1: Vec<f64>,
    #[serde(default)]
    z2: Vec<f64>,
}

#[derive(Deserialize)]
struct QmRun {
    min_anchors: usize,
    min_candidates: usize,
    size: Option<usize>,
    caliper: Option<f64>,
    seed_method: String,
    error: bool,
    cluster: Option<Vec<Option<usize>>>,
}

#[derive(Deserialize)]
struct MatchItRun {
    estimand: String,
    seed_method: String,
    caliper: Option<f64>,
    subclass: Vec<Option<usize>>,
    weights: Vec<f64>,
    unscaled_weights: Vec<f64>,
}

fn reference() -> Reference {
    load_reference("matchit_quick.json")
}

fn seed_method(name: &str) -> SeedMethod {
    match name {
        "lexical" => SeedMethod::Lexical,
        "inwards_order" => SeedMethod::InwardsOrder,
        "inwards_updating" => SeedMethod::InwardsUpdating,
        other => panic!("seed method {other}"),
    }
}

fn estimand_of(name: &str) -> Estimand {
    match name {
        "ATT" => Estimand::Att,
        "ATC" => Estimand::Atc,
        "ATE" => Estimand::Ate,
        other => panic!("estimand {other}"),
    }
}

/// Whether two labelings are the same partition, with the same units left out.
fn same_partition(a: &[Option<usize>], b: &[Option<usize>]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut forward: HashMap<usize, usize> = HashMap::new();
    let mut backward: HashMap<usize, usize> = HashMap::new();
    for (x, y) in a.iter().zip(b) {
        match (x, y) {
            (None, None) => {}
            (Some(x), Some(y)) => {
                if *forward.entry(*x).or_insert(*y) != *y || *backward.entry(*y).or_insert(*x) != *x
                {
                    return false;
                }
            }
            _ => return false,
        }
    }
    true
}

fn options_of(run: &QmRun) -> QuickMatchOptions {
    QuickMatchOptions {
        min_anchors: run.min_anchors,
        min_candidates: run.min_candidates,
        size: run.size,
        caliper: run.caliper,
        seed_method: seed_method(&run.seed_method),
    }
}

fn anchors_of(data: &Dataset) -> Vec<bool> {
    data.treat.iter().map(|t| *t == 1).collect()
}

fn distance(points: &[f64], dim: usize, a: usize, b: usize) -> f64 {
    (0..dim)
        .map(|k| {
            let delta = points[a * dim + k] - points[b * dim + k];
            delta * delta
        })
        .sum::<f64>()
        .sqrt()
}

fn max_within(points: &[f64], dim: usize, labels: &[Option<usize>]) -> f64 {
    let mut widest = 0.0_f64;
    for a in 0..labels.len() {
        for b in (a + 1)..labels.len() {
            if labels[a].is_some() && labels[a] == labels[b] {
                widest = widest.max(distance(points, dim, a, b));
            }
        }
    }
    widest
}

fn assert_constraints(labels: &[Option<usize>], anchor: &[bool], options: &QuickMatchOptions) {
    let count = labels.iter().flatten().max().map_or(0, |m| m + 1);
    let mut anchors = vec![0; count];
    let mut candidates = vec![0; count];
    for (label, is_anchor) in labels.iter().zip(anchor) {
        if let Some(label) = label {
            if *is_anchor {
                anchors[*label] += 1;
            } else {
                candidates[*label] += 1;
            }
        }
    }
    let size = options
        .size
        .unwrap_or(options.min_anchors + options.min_candidates);
    for subclass in 0..count {
        assert!(anchors[subclass] >= options.min_anchors, "anchors");
        assert!(candidates[subclass] >= options.min_candidates, "candidates");
        assert!(anchors[subclass] + candidates[subclass] >= size, "size");
    }
}

#[test]
fn reference_was_made_by_the_documented_versions() {
    let reference = reference();
    assert_r_version(&reference.versions["r"]);
    assert_eq!(reference.versions["matchit"], "4.8.2");
    assert_eq!(reference.versions["quickmatch"], "0.2.3");
}

/// The partition equals quickmatch's for every seed method, constraint set and
/// caliper in the reference.
#[test]
fn partitions_equal_quickmatchs() {
    let reference = reference();
    let mut compared = [0_usize; 3];
    for data in &reference.datasets {
        let anchor = anchors_of(data);
        for run in &data.runs {
            assert!(!run.error, "{}: quickmatch refused", data.name);
            let expected = run.cluster.as_ref().expect("labels");
            let options = options_of(run);
            let got = quick_clusters(&data.points, data.dim, &anchor, &options).expect("runs");
            assert!(
                same_partition(&got, expected),
                "{} {} constraints {}/{}/{:?} caliper {:?}",
                data.name,
                run.seed_method,
                run.min_anchors,
                run.min_candidates,
                run.size,
                run.caliper
            );
            let slot = match options.seed_method {
                SeedMethod::Lexical => 0,
                SeedMethod::InwardsOrder => 1,
                SeedMethod::InwardsUpdating => 2,
            };
            compared[slot] += 1;
        }
    }
    println!("compared (lexical, order, updating): {compared:?}");
    assert_eq!(compared, [35, 35, 35]);
}

/// Every subclass meets the constraints and the caliper, in every run and
/// seed method of the reference.
#[test]
fn subclasses_meet_constraints_and_caliper() {
    let reference = reference();
    let mut calipered = 0;
    for data in &reference.datasets {
        let anchor = anchors_of(data);
        for run in &data.runs {
            let options = options_of(run);
            let got = quick_clusters(&data.points, data.dim, &anchor, &options).expect("runs");
            assert_constraints(&got, &anchor, &options);
            if let Some(caliper) = run.caliper {
                // The distances are Euclidean sums of squares and square roots
                // taken in the same order for every pair, so a pair at the
                // caliper itself is the only one a rounding could move.
                assert!(
                    max_within(&data.points, data.dim, &got) <= caliper,
                    "{} caliper {caliper}",
                    data.name
                );
                calipered += 1;
            } else {
                assert!(got.iter().all(Option::is_some), "{}", data.name);
            }
        }
    }
    assert!(calipered >= 8);
}

/// Whether units from `unit` on can join the blocks so that every block has
/// both groups and no two units of a block are further apart than `limit`.
fn feasible(
    unit: usize,
    limit: f64,
    blocks: &mut Vec<Vec<usize>>,
    points: &[f64],
    dim: usize,
    anchor: &[bool],
) -> bool {
    if unit == anchor.len() {
        return blocks
            .iter()
            .all(|block| block.iter().any(|u| anchor[*u]) && block.iter().any(|u| !anchor[*u]));
    }
    for b in 0..blocks.len() {
        if blocks[b]
            .iter()
            .all(|other| distance(points, dim, unit, *other) <= limit)
        {
            blocks[b].push(unit);
            if feasible(unit + 1, limit, blocks, points, dim, anchor) {
                return true;
            }
            blocks[b].pop();
        }
    }
    blocks.push(vec![unit]);
    if feasible(unit + 1, limit, blocks, points, dim, anchor) {
        return true;
    }
    blocks.pop();
    false
}

/// The smallest largest within-group distance of any partition into groups
/// that each hold an anchor and a candidate, by exhaustive search over
/// partitions at each candidate distance.
fn optimal_bottleneck(points: &[f64], dim: usize, anchor: &[bool]) -> f64 {
    let n = anchor.len();
    let mut distances: Vec<f64> = Vec::new();
    for a in 0..n {
        for b in (a + 1)..n {
            distances.push(distance(points, dim, a, b));
        }
    }
    distances.sort_by(f64::total_cmp);
    distances.dedup();

    for limit in distances {
        if feasible(0, limit, &mut Vec::new(), points, dim, anchor) {
            return limit;
        }
    }
    unreachable!("one block holding everyone is feasible at the largest distance");
}

/// On sets of 9 to 11 units the largest within-subclass distance is within four
/// times the optimum, for every seed method. The factor is the bound proved in
/// the paper. The bound is on real distances, so a ratio is compared as it is,
/// with no tolerance: the ratios seen are far below 4.
#[test]
fn largest_distance_is_within_four_times_the_optimum() {
    let reference = reference();
    let mut worst = 0.0_f64;
    let mut sets = 0;
    for data in reference
        .datasets
        .iter()
        .filter(|d| d.name.starts_with("small_"))
    {
        let anchor = anchors_of(data);
        let best = optimal_bottleneck(&data.points, data.dim, &anchor);
        for run in &data.runs {
            let options = options_of(run);
            let got = quick_clusters(&data.points, data.dim, &anchor, &options).expect("runs");
            let widest = max_within(&data.points, data.dim, &got);
            assert!(
                widest <= 4.0 * best,
                "{} {}: {widest} against optimum {best}",
                data.name,
                run.seed_method
            );
            worst = worst.max(widest / best);
        }
        sets += 1;
    }
    println!("worst ratio to the optimum: {worst}");
    assert_eq!(sets, 8);
}

fn units(data: &Dataset, point_of: &[Vec<f64>]) -> (Vec<BaseRecord>, Vec<BaseRecord>) {
    let birth = NaiveDate::from_ymd_opt(2000, 1, 1).expect("valid date");
    let mut anchors = Vec::new();
    let mut candidates = Vec::new();
    for (row, treat) in data.treat.iter().enumerate() {
        let record = BaseRecord::new(format!("u{row}"), birth);
        assert_eq!(point_of[row].len(), data.dim);
        if *treat == 1 {
            anchors.push(record);
        } else {
            candidates.push(record);
        }
    }
    (anchors, candidates)
}

fn row_of(record: &BaseRecord) -> usize {
    record.id()[1..].parse().expect("row")
}

/// Rescale the weights of each group to sum to the number of its kept units.
fn rescaled(treat: &[u8], weights: &[Option<f64>]) -> Vec<f64> {
    let group = |idx: usize| usize::from(treat[idx] == 0);
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

/// Relative tolerance for weights. A weight is `p / (1 - p)` for the anchor
/// share `p = n1 / n` of a subclass, and the script's is `n1 / n0`. `1 - p` is
/// at least `1 / n`, so the relative error is at most `n` roundings of
/// 1.1e-16, below 7e-14 for the 614 units of lalonde. The rescaling adds a sum
/// of at most 614 terms, the same again. 1e-12 leaves a margin of about seven.
const WEIGHT_TOLERANCE: f64 = 1e-12;

fn close(got: f64, expected: f64) -> bool {
    (got - expected).abs() <= WEIGHT_TOLERANCE * expected.abs().max(1.0)
}

fn check_against_matchit(
    data: &Dataset,
    run: &MatchItRun,
    mut quick: impl FnMut(&Dataset, &MatchItRun) -> QuickMatchOutcome,
) {
    let label = format!(
        "{} {} {} caliper {:?}",
        data.name, run.seed_method, run.estimand, run.caliper
    );
    let got = quick(data, run);
    let n = data.treat.len();
    let assigned: HashMap<String, usize> = got
        .assignments()
        .map(|(id, subclass)| (id.to_owned(), subclass))
        .collect();
    let dropped: Vec<String> = got.dropped_ids().into_iter().map(str::to_owned).collect();
    let ours: Vec<Option<usize>> = (0..n)
        .map(|row| assigned.get(&format!("u{row}")).copied())
        .collect();
    assert!(same_partition(&ours, &run.subclass), "{label}: partition");
    let weights: HashMap<String, f64> = {
        let (ids, values) = got
            .weights_sorted(estimand_of(&run.estimand))
            .expect("weights");
        ids.into_iter().map(str::to_owned).zip(values).collect()
    };
    let unscaled: Vec<Option<f64>> = (0..n)
        .map(|row| weights.get(&format!("u{row}")).copied())
        .collect();
    let scaled = rescaled(&data.treat, &unscaled);
    for row in 0..n {
        assert!(
            close(scaled[row], run.weights[row]),
            "{label}: weight of {row}: {} against {}",
            scaled[row],
            run.weights[row]
        );
        assert!(
            close(unscaled[row].unwrap_or(0.0), run.unscaled_weights[row]),
            "{label}: unscaled weight of {row}"
        );
        assert_eq!(
            dropped.contains(&format!("u{row}")),
            run.subclass[row].is_none(),
            "{label}: dropped {row}"
        );
    }
}

/// Subclasses and weights equal MatchIt's, by estimand and seed method, on a
/// line, on lalonde's score, and with a caliper.
#[test]
fn weights_equal_matchits() {
    let reference = reference();
    let mut compared = 0;
    for data in reference
        .datasets
        .iter()
        .filter(|d| d.dim == 1 && !d.matchit.is_empty())
    {
        let point_of: Vec<Vec<f64>> = data.points.iter().map(|p| vec![*p]).collect();
        let (anchors, candidates) = units(data, &point_of);
        for run in &data.matchit {
            check_against_matchit(data, run, |data, run| {
                let options = QuickMatchOptions {
                    caliper: run.caliper,
                    seed_method: seed_method(&run.seed_method),
                    ..QuickMatchOptions::default()
                };
                quick_match_subclasses::<_, &str, _>(
                    &anchors,
                    &candidates,
                    |record: &BaseRecord| {
                        let row = row_of(record);
                        Some(&data.points[row..=row])
                    },
                    1,
                    &[],
                    &options,
                )
                .expect("matches")
            });
            compared += 1;
        }
    }
    assert_eq!(compared, 33, "runs compared");
}

/// `run_spec` with MatchIt's Mahalanobis distance and the quick method gives
/// MatchIt's partition and weights.
#[test]
fn mahalanobis_spec_equals_matchit() {
    let reference = reference();
    let data = reference
        .datasets
        .iter()
        .find(|d| d.name == "mahal_120")
        .expect("data set");
    let birth = NaiveDate::from_ymd_opt(2000, 1, 1).expect("valid date");
    let mut anchors = Vec::new();
    let mut candidates = Vec::new();
    for (row, treat) in data.treat.iter().enumerate() {
        let mut record = BalanceRecord::new(format!("u{row}"), birth);
        record
            .covariates
            .insert("z1".into(), CovariateValue::Numeric(data.z1[row]));
        record
            .covariates
            .insert("z2".into(), CovariateValue::Numeric(data.z2[row]));
        if *treat == 1 {
            anchors.push(record);
        } else {
            candidates.push(record);
        }
    }
    for run in &data.matchit {
        let mut spec = MatchSpec {
            method: MatchMethod::Quick,
            distance: DistanceSpec::Mahalanobis {
                covariates: epistat_matching::CovariatesSpec::default(),
                covariance: epistat_matching::CovarianceSpec::default(),
                transform: epistat_matching::TransformSpec::default(),
                caliper: None,
            },
            ..MatchSpec::default()
        };
        spec.quick.seed_method = seed_method(&run.seed_method);
        check_against_matchit(data, run, |_, _| {
            match run_spec(
                &spec,
                SpecInput::Standard {
                    anchors: &anchors,
                    candidates: &candidates,
                },
            )
            .expect("runs")
            {
                SpecOutput::Quick(outcome) => *outcome,
                _ => panic!("a quick outcome"),
            }
        });
    }
    assert_eq!(data.matchit.len(), 9);
}

/// Input that cannot be matched is refused with the reason.
#[test]
fn refuses_what_cannot_be_matched() {
    let points = [0.0, 1.0, 2.0, 3.0];
    let options = QuickMatchOptions::default();
    assert_eq!(
        quick_clusters(&points, 1, &[true, true, true, true], &options).unwrap_err(),
        QuickMatchError::TooFewUnits { stratum: None }
    );
    let bad = QuickMatchOptions {
        size: Some(1),
        ..options
    };
    assert_eq!(
        quick_clusters(&points, 1, &[true, false, true, false], &bad).unwrap_err(),
        QuickMatchError::SizeBelowMinimums
    );
}

/// Time on a line and in the plane at 10^5 and 10^6 units, with the default
/// constraints, with minimums of 2 anchors and 3 candidates and a size of 7,
/// and with the two seed methods that sweep in order. Run with
/// `cargo test --release -p epistat-matching --test r_quick -- --ignored --nocapture`
/// and, for one thread, with `RAYON_NUM_THREADS=1` in front.
#[test]
#[ignore = "benchmark"]
fn time_on_a_line_and_in_the_plane() {
    for dim in [1_usize, 2] {
        for n in [100_000_usize, 1_000_000] {
            let mut state = 88_172_645_463_325_252_u64;
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 11) as f64 / (1_u64 << 53) as f64
            };
            let points: Vec<f64> = (0..n * dim).map(|_| next()).collect();
            let anchor: Vec<bool> = (0..n)
                .map(|row| next() < 0.2 + 0.3 * points[row * dim])
                .collect();
            for (min_anchors, min_candidates, size) in [(1, 1, None), (2, 3, Some(7))] {
                for method in [SeedMethod::InwardsUpdating, SeedMethod::Lexical] {
                    let options = QuickMatchOptions {
                        min_anchors,
                        min_candidates,
                        size,
                        caliper: None,
                        seed_method: method,
                    };
                    let start = std::time::Instant::now();
                    let labels = quick_clusters(&points, dim, &anchor, &options).expect("runs");
                    let subclasses = labels.iter().flatten().max().map_or(0, |m| m + 1);
                    println!(
                        "dim {dim} n = {n} constraints {min_anchors}/{min_candidates}/{size:?} \
                         {method:?}: {:.2} s, {subclasses} subclasses",
                        start.elapsed().as_secs_f64()
                    );
                }
            }
        }
    }
}
