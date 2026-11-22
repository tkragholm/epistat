//! Common-support trimming before subclass, full, optimal, quick and genetic
//! matching, checked against R 4.6's `MatchIt` 4.8.2,
//! `matchit(method = ..., distance = "glm", discard = policy, reestimate = ...)`.
//!
//! The reference is `tests/data/matchit_discard_methods.json`, written by
//! `tests/data/matchit_discard_methods.R`. It holds, for `lalonde` with its
//! continuous covariates jittered so that no two units tie, each policy
//! (`treated`, `control`, `both`) with and without `reestimate`, and each of
//! the five methods: the units MatchIt discarded, the distance it matched on
//! and its result. The crate is run through `run_spec` with the same options.
//!
//! What is compared, on every run:
//!
//! * The discarded units equal MatchIt's, as sets, and the crate reports them
//!   by id, anchors and candidates apart.
//! * The distance of every unit, kept or discarded, equals MatchIt's to 1e-9.
//!   The crate's propensity fit agrees with R's to about 1e-13 (see
//!   `r_propensity.rs`), so 1e-9 leaves room for the refit on fewer units and
//!   no more.
//! * Subclass: the cutpoints (to 1e-9, for the same reason), the subclass of
//!   every kept unit and the rescaled weights.
//! * Quick: the partition and the rescaled weights. Scores are distinct, so
//!   quickmatch's tie rule does not enter.
//! * Full and optimal: the number of matched units, or of controls per treated
//!   unit, and the total distance of the treated-control pairs
//!   against R's, within the bounds of `r_full.rs` and `r_optimal.rs`. Where
//!   four perturbations of the distance leave R's match unchanged (`unique`),
//!   the partition or the pairs, and for full matching the weights, are
//!   compared too. On a line the optimum is rarely a single one.
//! * Genetic: the pairs, with R's weights pinned to 1 and the crate's search
//!   evaluating its starting weights, which are 1, only.
//!
//! MatchIt rescales the weights of each group to sum to the number of its
//! kept units. The test applies the same rescaling to the crate's weights.

#[path = "common/r_reference.rs"]
mod r_reference;

use chrono::NaiveDate;
use epistat_matching::prelude::*;
use epistat_matching::{
    BalanceRecord, CovariateValue, CovariatesSpec, DiscardSpec, DistanceSpec, FullMatchOutcome,
    GeneticSpec, LinkSpec, MatchMethod, MatchSpec, ScaleSpec, SpecInput, SpecOutput, SubclassSpec,
    TrimmedUnits, run_spec,
};

use r_reference::{Versions, assert_r_version, reference as load_reference};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    data: Data,
    runs: Vec<Run>,
}

#[derive(Deserialize)]
struct Data {
    treat: Vec<u8>,
    names: Vec<String>,
    covariates: Vec<Vec<f64>>,
}

/// R writes a vector of one element as a bare number.
#[derive(Deserialize)]
#[serde(untagged)]
enum Rows {
    Many(Vec<usize>),
    One(usize),
}

impl Rows {
    fn to_vec(&self) -> Vec<usize> {
        match self {
            Self::Many(values) => values.clone(),
            Self::One(value) => vec![*value],
        }
    }
}

#[derive(Deserialize)]
struct Pairs {
    treated: Vec<usize>,
    controls: Vec<Rows>,
}

#[derive(Deserialize)]
struct Run {
    method: String,
    policy: String,
    reestimate: bool,
    estimand: String,
    link: String,
    #[serde(default)]
    extra: serde_json::Value,
    discarded: Vec<usize>,
    distance: Vec<f64>,
    subclass: Option<Vec<Option<usize>>>,
    weights: Option<Vec<f64>>,
    q_cut: Option<Vec<f64>>,
    q_cut_with_discard: Option<Vec<f64>>,
    pairs: Option<Pairs>,
    unique: Option<bool>,
}

fn reference() -> Reference {
    load_reference("matchit_discard_methods.json")
}

fn id(row: usize) -> String {
    row.to_string()
}

fn row_of(id: &str) -> usize {
    id.parse().expect("row number")
}

/// The units of the data, treated first as anchors, with rows numbered from 1.
fn units(data: &Data) -> (Vec<BalanceRecord>, Vec<BalanceRecord>) {
    let birth = NaiveDate::from_ymd_opt(2010, 1, 1).expect("valid date");
    let mut anchors = Vec::new();
    let mut candidates = Vec::new();
    for (row, label) in data.treat.iter().enumerate() {
        let mut record = BalanceRecord::new(id(row + 1), birth);
        for (name, column) in data.names.iter().zip(&data.covariates) {
            record
                .covariates
                .insert(name.clone(), CovariateValue::Numeric(column[row]));
        }
        if *label == 1 {
            anchors.push(record);
        } else {
            candidates.push(record);
        }
    }
    (anchors, candidates)
}

fn spec_of(run: &Run) -> MatchSpec {
    let method = match run.method.as_str() {
        "subclass" => MatchMethod::Subclass,
        "full" => MatchMethod::Full,
        "optimal" => MatchMethod::Optimal,
        "quick" => MatchMethod::Quick,
        "genetic" => MatchMethod::Genetic,
        other => panic!("method {other}"),
    };
    let scale = match run.link.as_str() {
        "logit" => ScaleSpec::Probability,
        "linear.logit" => ScaleSpec::Linear,
        other => panic!("link {other}"),
    };
    let mut spec = MatchSpec {
        method,
        distance: DistanceSpec::Propensity {
            covariates: CovariatesSpec::default(),
            link: LinkSpec::Logit,
            estimator: epistat_matching::EstimatorSpec::default(),
            scale,
            sampling_weights: None,
            caliper: None,
        },
        estimand: match run.estimand.as_str() {
            "ATT" => Estimand::Att,
            "ATE" => Estimand::Ate,
            other => panic!("estimand {other}"),
        },
        discard: Some(match run.policy.as_str() {
            "treated" => DiscardSpec::Treated,
            "control" => DiscardSpec::Control,
            "both" => DiscardSpec::Both,
            other => panic!("policy {other}"),
        }),
        reestimate: run.reestimate,
        ..MatchSpec::default()
    };
    if let Some(count) = run
        .extra
        .get("subclass")
        .and_then(serde_json::Value::as_u64)
    {
        spec.subclass = SubclassSpec {
            n: usize::try_from(count).expect("count"),
            ..SubclassSpec::default()
        };
    }
    if let Some(ratio) = run.extra.get("ratio").and_then(serde_json::Value::as_u64) {
        spec.ratio = usize::try_from(ratio).expect("ratio");
    }
    if method == MatchMethod::Genetic {
        spec.genetic = GeneticSpec {
            pop_size: 0,
            ..GeneticSpec::default()
        };
    }
    spec
}

fn sorted_rows(ids: &[String]) -> Vec<usize> {
    let mut rows: Vec<usize> = ids.iter().map(|id| row_of(id)).collect();
    rows.sort_unstable();
    rows
}

/// The partition as the set of its subclasses, each a set of rows.
fn partition(assignment: impl IntoIterator<Item = (usize, usize)>) -> BTreeSet<BTreeSet<usize>> {
    let mut groups: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for (row, subclass) in assignment {
        groups.entry(subclass).or_default().insert(row);
    }
    groups.into_values().collect()
}

fn r_partition(subclass: &[Option<usize>]) -> BTreeSet<BTreeSet<usize>> {
    partition(
        subclass
            .iter()
            .enumerate()
            .filter_map(|(idx, s)| s.map(|s| (idx + 1, s))),
    )
}

/// Rescale the weights of each group to sum to the number of its kept units,
/// as MatchIt does, and return them by row with 0 for a unit without one.
fn rescaled(data: &Data, weights: &HashMap<String, f64>) -> Vec<f64> {
    let group = |idx: usize| usize::from(data.treat[idx] != 1);
    let mut sum = [0.0_f64; 2];
    let mut count = [0.0_f64; 2];
    for idx in 0..data.treat.len() {
        if let Some(w) = weights.get(&id(idx + 1)) {
            sum[group(idx)] += w;
            count[group(idx)] += 1.0;
        }
    }
    (0..data.treat.len())
        .map(|idx| {
            weights
                .get(&id(idx + 1))
                .map_or(0.0, |w| w * count[group(idx)] / sum[group(idx)])
        })
        .collect()
}

/// A weight is a ratio of counts, rescaled by a ratio of two sums of at most a
/// few hundred such ratios, so 1e-12 relative bounds the rounding.
const WEIGHT_TOLERANCE: f64 = 1e-12;

/// The crate's propensity fit agrees with R's to about 1e-13; the discards come
/// from the first fit and the matching from a fit on fewer units.
const DISTANCE_TOLERANCE: f64 = 1e-9;

fn check_trimmed(label: &str, run: &Run, data: &Data, trimmed: &TrimmedUnits) {
    let mut ours = sorted_rows(&trimmed.anchors);
    ours.extend(sorted_rows(&trimmed.candidates));
    ours.sort_unstable();
    assert_eq!(ours, run.discarded, "{label}: discarded units");
    for anchor in &trimmed.anchors {
        assert_eq!(
            data.treat[row_of(anchor) - 1],
            1,
            "{label}: anchor {anchor}"
        );
    }
    for candidate in &trimmed.candidates {
        assert_eq!(
            data.treat[row_of(candidate) - 1],
            0,
            "{label}: candidate {candidate}"
        );
    }
}

fn check_distance(label: &str, run: &Run, estimate: &PropensityScoreEstimate, kept: &[usize]) {
    for row in kept {
        let got = estimate
            .distance_score(&id(*row))
            .unwrap_or_else(|| panic!("{label}: no score for {row}"));
        let want = run.distance[row - 1];
        assert!(
            (got - want).abs() <= DISTANCE_TOLERANCE * want.abs().max(1.0),
            "{label}: distance of {row} is {got}, R has {want}"
        );
    }
}

fn distance_between(run: &Run, a: usize, b: usize) -> f64 {
    (run.distance[a - 1] - run.distance[b - 1]).abs()
}

/// The largest distance between a treated and a control unit among `kept`.
fn largest_distance(data: &Data, run: &Run, kept: &[usize]) -> f64 {
    let mut largest = 0.0_f64;
    for a in kept {
        for b in kept {
            if data.treat[a - 1] == 1 && data.treat[b - 1] == 0 {
                largest = largest.max(distance_between(run, *a, *b));
            }
        }
    }
    largest
}

/// The ends of the interval around the optimum in which both totals must lie.
///
/// The crate solves with costs of about 40 bits, so its total is at most
/// `3 n d_max / 2^40` above the optimum, as in `r_optimal.rs`. optmatch rounds
/// the distances to integers with resolution `d_max / (i32::MAX / 64 - 2)`, so
/// its total is at most `1.5` resolutions per pair above the optimum. The
/// totals are taken with R's distances, while the crate optimised its own, which
/// differ by at most 1e-12 each, so a pair's distance moves by at most 2e-12.
fn total_bounds(pairs: usize, d_max: f64) -> (f64, f64) {
    let n = pairs as f64;
    let ours = 3.0 * n * d_max / 2_f64.powi(40) + 2e-12 * n;
    let epsilon = d_max / (f64::from(i32::MAX) / 64.0 - 2.0);
    (ours, 1.5 * n * epsilon)
}

fn pair_total(run: &Run, pairs: &[(usize, usize)]) -> f64 {
    pairs
        .iter()
        .map(|(a, b)| distance_between(run, *a, *b))
        .sum()
}

fn partition_total(data: &Data, run: &Run, groups: &BTreeSet<BTreeSet<usize>>) -> (f64, usize) {
    let mut total = 0.0;
    let mut pairs = 0;
    for group in groups {
        for a in group {
            for b in group {
                if data.treat[a - 1] == 1 && data.treat[b - 1] == 0 {
                    total += distance_between(run, *a, *b);
                    pairs += 1;
                }
            }
        }
    }
    (total, pairs)
}

fn check_totals(label: &str, ours: f64, r: f64, pairs: usize, d_max: f64) {
    let (our_bound, r_bound) = total_bounds(pairs, d_max);
    assert!(
        ours <= r + our_bound,
        "{label}: crate total {ours} above R's {r}"
    );
    assert!(
        r <= ours + r_bound + our_bound,
        "{label}: R total {r} above the crate's {ours}"
    );
}

/// The quantiles at `0, 1/k, ..., 1` of `values` by linear interpolation
/// between order statistics, R's `quantile(type = 7)`.
fn quantiles(mut values: Vec<f64>, k: usize) -> Vec<f64> {
    values.sort_by(f64::total_cmp);
    let span = values.len() - 1;
    (0..=k)
        .map(|bucket| {
            let low = span * bucket / k;
            let high = low + usize::from(!(span * bucket).is_multiple_of(k));
            let share = (span * bucket % k) as f64 / k as f64;
            values[low] + share * (values[high] - values[low])
        })
        .collect()
}

/// `matchit()` takes the cutpoints over all units of the reference group,
/// discarded ones included. Without a refit the crate does the same and its
/// cutpoints are R's. With a refit the distances of the discarded units are
/// those of the first fit and the others those of the second, so R's cutpoints
/// are the quantiles of that mixture, which the crate does not reproduce: it
/// takes them over the kept units. R's value is asserted here, the crate's in
/// [`check_subclass`] against `matchit()` on the kept units.
fn check_matchit_cutpoints_include_discarded(label: &str, run: &Run, data: &Data) {
    let theirs = run.q_cut_with_discard.as_ref().expect("q.cut with discard");
    let reference: Vec<f64> = (0..data.treat.len())
        .filter(|idx| run.estimand == "ATE" || data.treat[*idx] == 1)
        .map(|idx| run.distance[idx])
        .collect();
    let all = quantiles(reference, theirs.len() - 1);
    for (a, b) in all.iter().zip(theirs) {
        assert!(
            (a - b).abs() <= 1e-12 * b.abs().max(1.0),
            "{label}: matchit cutpoint {b} is not the quantile {a} over all units"
        );
    }
}

fn check_subclass(label: &str, run: &Run, data: &Data, outcome: &SubclassificationOutcome) {
    let q_cut = run.q_cut.as_ref().expect("q.cut");
    assert_eq!(outcome.quantile_cuts.len(), q_cut.len(), "{label}: cuts");
    for (a, b) in outcome.quantile_cuts.iter().zip(q_cut) {
        assert!(
            (a - b).abs() <= DISTANCE_TOLERANCE * b.abs().max(1.0),
            "{label}: cutpoint {a} against {b}"
        );
    }
    let expected = run.subclass.as_ref().expect("subclass");
    let assigned: HashMap<usize, usize> = outcome
        .assignments()
        .map(|(unit, subclass)| (row_of(unit), subclass))
        .collect();
    let expected_kept: BTreeSet<usize> = expected
        .iter()
        .enumerate()
        .filter_map(|(idx, s)| s.map(|_| idx + 1))
        .collect();
    assert_eq!(
        assigned.keys().copied().collect::<BTreeSet<_>>(),
        expected_kept,
        "{label}: kept units"
    );
    // MatchIt numbers the subclasses it keeps from 1.
    let present: BTreeSet<usize> = assigned.values().copied().collect();
    let rank: HashMap<usize, usize> = present
        .iter()
        .enumerate()
        .map(|(r, s)| (*s, r + 1))
        .collect();
    for (row, subclass) in &assigned {
        assert_eq!(
            Some(rank[subclass]),
            expected[row - 1],
            "{label}: subclass of {row}"
        );
    }
    let (ids, values) = outcome.weights_sorted();
    let weights: HashMap<String, f64> = ids.into_iter().map(str::to_owned).zip(values).collect();
    check_weights(label, run, &rescaled(data, &weights));
}

fn check_weights(label: &str, run: &Run, got: &[f64]) {
    for (idx, (a, b)) in got
        .iter()
        .zip(run.weights.as_ref().expect("weights"))
        .enumerate()
    {
        assert!(
            (a - b).abs() <= WEIGHT_TOLERANCE * b.abs().max(1.0),
            "{label}: weight of row {} is {a}, R has {b}",
            idx + 1
        );
    }
}

fn check_full(label: &str, run: &Run, data: &Data, outcome: &FullMatchOutcome, kept: &[usize]) {
    let expected = run.subclass.as_ref().expect("subclass");
    let ours: BTreeMap<usize, usize> = outcome
        .assignments()
        .map(|(unit, subclass)| (row_of(unit), subclass))
        .collect();
    assert_eq!(
        ours.len(),
        expected.iter().flatten().count(),
        "{label}: matched units"
    );
    let our_partition = partition(ours.iter().map(|(row, s)| (*row, *s)));
    let their_partition = r_partition(expected);
    let d_max = largest_distance(data, run, kept);
    let (our_total, pairs) = partition_total(data, run, &our_partition);
    let (r_total, r_pairs) = partition_total(data, run, &their_partition);
    check_totals(label, our_total, r_total, pairs.max(r_pairs), d_max);
    if run.unique == Some(true) {
        assert_eq!(our_partition, their_partition, "{label}: partition");
        let att: HashMap<String, f64> = (0..outcome.len())
            .filter_map(|at| {
                outcome
                    .weight(Estimand::Att, at)
                    .map(|weight| (outcome.id(at).to_string(), weight))
            })
            .collect();
        let weights = rescaled(data, &att);
        check_weights(label, run, &weights);
    }
}

fn check_optimal(label: &str, run: &Run, data: &Data, outcome: &MatchOutcome, kept: &[usize]) {
    let reference = run.pairs.as_ref().expect("pairs");
    let r_pairs: Vec<(usize, usize)> = reference
        .treated
        .iter()
        .zip(&reference.controls)
        .flat_map(|(treated, controls)| {
            controls
                .to_vec()
                .into_iter()
                .map(move |control| (*treated, control))
        })
        .collect();
    let our_pairs: Vec<(usize, usize)> = outcome
        .pairs
        .iter()
        .map(|pair| (row_of(pair.anchor_id()), row_of(pair.comparator_id())))
        .collect();
    let mut per_anchor: HashMap<usize, usize> = HashMap::new();
    for (anchor, _) in &our_pairs {
        *per_anchor.entry(*anchor).or_default() += 1;
    }
    for (treated, controls) in reference.treated.iter().zip(&reference.controls) {
        assert_eq!(
            per_anchor.get(treated).copied().unwrap_or(0),
            controls.to_vec().len(),
            "{label}: controls of {treated}"
        );
    }
    let d_max = largest_distance(data, run, kept);
    check_totals(
        label,
        pair_total(run, &our_pairs),
        pair_total(run, &r_pairs),
        our_pairs.len(),
        d_max,
    );
    if run.unique == Some(true) {
        let set = |pairs: &[(usize, usize)]| pairs.iter().copied().collect::<BTreeSet<_>>();
        assert_eq!(set(&our_pairs), set(&r_pairs), "{label}: pairs");
    }
}

fn check_genetic(label: &str, run: &Run, outcome: &MatchOutcome) {
    let reference = run.pairs.as_ref().expect("pairs");
    let mut ours: HashMap<usize, BTreeSet<usize>> = HashMap::new();
    for pair in &outcome.pairs {
        ours.entry(row_of(pair.anchor_id()))
            .or_default()
            .insert(row_of(pair.comparator_id()));
    }
    for (treated, controls) in reference.treated.iter().zip(&reference.controls) {
        let expected: BTreeSet<usize> = controls.to_vec().into_iter().collect();
        assert_eq!(
            ours.get(treated).cloned().unwrap_or_default(),
            expected,
            "{label}: controls of {treated}"
        );
    }
}

#[test]
fn reference_was_made_by_the_documented_versions() {
    let reference = reference();
    assert_r_version(&reference.versions["r"]);
    assert_eq!(reference.versions["matchit"], "4.8.2");
}

#[test]
fn trimmed_units_and_results_equal_matchit_for_each_method() {
    let reference = reference();
    let (anchors, candidates) = units(&reference.data);
    let mut compared: BTreeMap<String, usize> = BTreeMap::new();
    for run in &reference.runs {
        let label = format!(
            "{} {} reestimate {} {}",
            run.method, run.policy, run.reestimate, run.estimand
        );
        let output = run_spec(
            &spec_of(run),
            SpecInput::Standard {
                anchors: &anchors,
                candidates: &candidates,
            },
        )
        .unwrap_or_else(|error| panic!("{label}: {error}"));
        let discarded: BTreeSet<usize> = run.discarded.iter().copied().collect();
        let kept: Vec<usize> = (1..=reference.data.treat.len())
            .filter(|row| !discarded.contains(row))
            .collect();
        // Every unit has a distance: a refit gives the kept units new scores
        // and the discarded ones keep those of the first fit, as in R.
        let every: Vec<usize> = (1..=reference.data.treat.len()).collect();
        match &output {
            SpecOutput::Subclass {
                outcome,
                propensity,
            } => {
                check_trimmed(&label, run, &reference.data, &outcome.trimmed);
                check_distance(&label, run, propensity.as_ref().expect("fit"), &every);
                check_subclass(&label, run, &reference.data, outcome);
                check_matchit_cutpoints_include_discarded(&label, run, &reference.data);
                if run.reestimate
                    && outcome
                        .quantile_cuts
                        .iter()
                        .zip(run.q_cut_with_discard.as_ref().expect("cuts"))
                        .any(|(a, b)| (a - b).abs() > 1e-6)
                {
                    *compared.entry("mixed cutpoints differ".into()).or_default() += 1;
                }
            }
            SpecOutput::Full(outcome) => {
                check_trimmed(&label, run, &reference.data, &outcome.trimmed);
                check_full(&label, run, &reference.data, outcome, &kept);
            }
            SpecOutput::Quick(outcome) => {
                check_trimmed(&label, run, &reference.data, &outcome.trimmed);
                let expected = run.subclass.as_ref().expect("subclass");
                let ours: Vec<(usize, usize)> = outcome
                    .assignments()
                    .map(|(unit, subclass)| (row_of(unit), subclass))
                    .collect();
                assert_eq!(
                    partition(ours.iter().copied()),
                    r_partition(expected),
                    "{label}: partition"
                );
                let (ids, values) = outcome
                    .weights_sorted(Estimand::Att)
                    .expect("weights for att");
                let weights: HashMap<String, f64> =
                    ids.into_iter().map(str::to_owned).zip(values).collect();
                check_weights(&label, run, &rescaled(&reference.data, &weights));
            }
            SpecOutput::Pairs {
                outcome,
                propensity,
            } => {
                check_trimmed(
                    &label,
                    run,
                    &reference.data,
                    &outcome.diagnostics.common_support_trimmed_ids,
                );
                check_distance(&label, run, propensity.as_ref().expect("fit"), &every);
                check_optimal(&label, run, &reference.data, outcome, &kept);
            }
            SpecOutput::Genetic { outcome, .. } => {
                check_trimmed(
                    &label,
                    run,
                    &reference.data,
                    &outcome.diagnostics.common_support_trimmed_ids,
                );
                check_genetic(&label, run, outcome);
            }
            _ => panic!("{label}: unexpected output"),
        }
        let used: Vec<usize> = match &output {
            SpecOutput::Subclass { outcome, .. } => outcome
                .assignments()
                .map(|(unit, _)| row_of(unit))
                .collect(),
            SpecOutput::Quick(outcome) => outcome
                .assignments()
                .map(|(unit, _)| row_of(unit))
                .collect(),
            SpecOutput::Full(outcome) => outcome
                .assignments()
                .map(|(unit, _)| row_of(unit))
                .collect(),
            SpecOutput::Pairs { outcome, .. } | SpecOutput::Genetic { outcome, .. } => outcome
                .pairs
                .iter()
                .flat_map(|pair| [row_of(pair.anchor_id()), row_of(pair.comparator_id())])
                .collect(),
            _ => Vec::new(),
        };
        assert!(!used.is_empty(), "{label}: nothing matched");
        assert!(
            used.iter().all(|row| !discarded.contains(row)),
            "{label}: a discarded unit was matched"
        );
        *compared.entry(run.method.clone()).or_default() += 1;
    }
    assert!(
        compared.get("mixed cutpoints differ").copied().unwrap_or(0) > 0,
        "the mixture of fits moves no cutpoint"
    );
    for method in ["subclass", "full", "optimal", "quick", "genetic"] {
        assert!(compared.get(method).copied().unwrap_or(0) >= 6, "{method}");
    }
}

#[test]
fn trimming_counts_match_the_listed_units() {
    let reference = reference();
    let (anchors, candidates) = units(&reference.data);
    for run in reference.runs.iter().filter(|run| run.method == "optimal") {
        let SpecOutput::Pairs { outcome, .. } = run_spec(
            &spec_of(run),
            SpecInput::Standard {
                anchors: &anchors,
                candidates: &candidates,
            },
        )
        .expect("valid spec") else {
            panic!("expected pairs");
        };
        let diagnostics = &outcome.diagnostics;
        assert_eq!(
            diagnostics.common_support_trimmed_anchors,
            diagnostics.common_support_trimmed_ids.anchors.len()
        );
        assert_eq!(
            diagnostics.common_support_trimmed_candidates,
            diagnostics.common_support_trimmed_ids.candidates.len()
        );
    }
}
