//! Exposure density sampling checked against the R package `heaven` and
//! `Epi::ccwc`.
//!
//! The reference is `tests/data/exposure_density.json`, written by
//! `tests/data/exposure_density.R` on a simulated cohort of 300 people with
//! two matching variables, staggered entry, deaths, emigrations, exposures on
//! the same day, exposures after the age limit and people who leave or enter
//! on the exposure day of a case. The file records the versions of R, heaven
//! and Epi.
//!
//! `heaven::exposureMatch` samples at random, so individual sets cannot agree.
//! What is compared is what both define:
//!
//! * the risk set of every exposed person, from `exposureMatch` with more
//!   controls than anyone has at risk, against the controls the engine returns
//!   with a ratio larger than the cohort and replacement allowed. The same sets
//!   come from `Epi::ccwc` with the exposure as the failure, which the script
//!   checks equal to heaven's, so the three agree;
//! * how often each member of a risk set is drawn when one control is drawn per
//!   exposed person, over fixed seeds, against a chi-square threshold derived
//!   from the level and the number of tests.
//!
//! What the engine adds is checked here and not against R: the index date of
//! each set and the date on which each control is later exposed.

#[path = "common/r_reference.rs"]
mod r_reference;

use chrono::{Days, NaiveDate};
use epistat_matching::DeterministicSelection;
use epistat_matching::prelude::*;
use epistat_stats::gamma::chi_square_sf;

use r_reference::{assert_r_version, reference as load_reference};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    age_limit_years: u8,
    reps: usize,
    people: People,
    cases: Vec<String>,
    risk_set_heaven: BTreeMap<String, Vec<String>>,
    risk_set_ccwc: BTreeMap<String, Vec<String>>,
    draws: Vec<Draw>,
    r_chi_square: Vec<f64>,
    heaven_k3_controls_in_several_sets: usize,
    heaven_k3_future_cases_as_controls: usize,
}

#[derive(Deserialize)]
struct Versions {
    #[serde(rename = "R")]
    r: String,
    heaven: String,
    #[serde(rename = "Epi")]
    epi: String,
}

#[derive(Deserialize)]
struct People {
    id: Vec<String>,
    sex: Vec<String>,
    area: Vec<String>,
    birth: Vec<i64>,
    entry: Vec<i64>,
    exposure: Vec<Option<i64>>,
    death: Vec<Option<i64>>,
    emigration: Vec<Option<i64>>,
    case: Vec<i64>,
}

#[derive(Deserialize)]
struct Draw {
    case: String,
    members: Vec<String>,
    counts: Vec<usize>,
}

type Rec = RoleTransitionRecord<BalanceRecord>;

fn reference() -> Reference {
    load_reference("exposure_density.json")
}

fn day(days: i64) -> NaiveDate {
    NaiveDate::from_ymd_opt(1970, 1, 1)
        .expect("valid date")
        .checked_add_days(Days::new(u64::try_from(days).expect("after 1970")))
        .expect("in range")
}

/// The cohort, with the exposure date as the transition date.
fn records(people: &People) -> Vec<Rec> {
    (0..people.id.len())
        .map(|i| {
            let mut base = BaseRecord::new(people.id[i].clone(), day(people.birth[i]));
            base.strata.insert("sex".into(), people.sex[i].clone());
            base.strata.insert("area".into(), people.area[i].clone());
            base = base.with_entry_date(day(people.entry[i]));
            if let Some(d) = people.death[i] {
                base = base.with_death_date(day(d));
            }
            if let Some(d) = people.emigration[i] {
                base = base.with_emigration_date(day(d));
            }
            let record = BalanceRecord::new(base.id.clone(), base.birth_date);
            RoleTransitionRecord::from_record(
                BalanceRecord {
                    core: base,
                    ..record
                },
                people.exposure[i].map(day),
            )
        })
        .collect()
}

fn ratio(k: usize) -> MatchRatio {
    MatchRatio::new(k).expect("non-zero ratio")
}

/// About 82 years: no birth-date restriction on a cohort born in a decade.
fn window() -> BirthDateWindowDays {
    BirthDateWindowDays::new(30_000).expect("non-negative")
}

fn run_job(
    reference: &Reference,
    recs: &[Rec],
    k: usize,
    replacement: bool,
    seed: u64,
) -> MatchOutcome {
    MatchJob::new_transition(recs, seed)
        .with_ratio(ratio(k))
        .with_birth_window(window())
        .with_age_limit(AgeLimitYears::new(reference.age_limit_years).expect("positive"))
        .with_exact_matches(["sex", "area"])
        .with_replacement(replacement)
        .with_alive_check()
        .with_resident_check()
        .run()
}

fn run(
    reference: &Reference,
    recs: &[Rec],
    k: usize,
    replacement: bool,
    seed: u64,
) -> ExposureDensityOutcome {
    let outcome = run_job(reference, recs, k, replacement, seed);
    exposure_density_sets(recs, outcome).expect("unique ids")
}

fn by_exposed(sets: &ExposureDensityOutcome) -> BTreeMap<String, Vec<String>> {
    let mut map: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for control in &sets.controls {
        map.entry(control.exposed_id.clone())
            .or_default()
            .push(control.control_id.clone());
    }
    map
}

#[test]
fn versions_are_recorded() {
    let reference = reference();
    assert_r_version(&reference.versions.r);
    assert_eq!(reference.versions.heaven, "2026.10.30");
    assert_eq!(reference.versions.epi, "2.66");
}

/// With a ratio above every risk set and replacement, the engine returns each
/// exposed person's whole risk set. Those sets equal the ones heaven returns
/// and the ones `Epi::ccwc` returns for the same cohort with the exposure as
/// the failure. The data has people who enter on an exposure day, who are in
/// the set, and people who die, emigrate or are exposed on it, who are not.
#[test]
fn risk_sets_equal_heaven_and_ccwc() {
    let reference = reference();
    let recs = records(&reference.people);
    let sets = run(&reference, &recs, 10_000, true, 0);
    let got = by_exposed(&sets);
    assert_eq!(reference.risk_set_heaven, reference.risk_set_ccwc);
    assert_eq!(
        got.keys().collect::<Vec<_>>(),
        reference.cases.iter().collect::<Vec<_>>(),
        "the exposed people with a non-empty risk set"
    );
    for (case, expected) in &reference.risk_set_heaven {
        let mut controls = got[case].clone();
        controls.sort();
        assert_eq!(&controls, expected, "risk set of {case}");
    }
    assert_eq!(
        sets.outcome.diagnostics.total_anchors_evaluated,
        reference.cases.len(),
        "the exposed people the age limit leaves"
    );
}

/// The output carries what an analysis needs: every control has the index date
/// of its set, which is the exposure date of the exposed person, and the date
/// it is itself exposed, which is after the index date and equal to the data.
/// The data has controls who are exposed later and controls who never are.
#[test]
fn sets_carry_index_dates_and_later_exposure() {
    let reference = reference();
    let people = &reference.people;
    let recs = records(people);
    let position: HashMap<&str, usize> = people
        .id
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_str(), i))
        .collect();
    let sets = run(&reference, &recs, 3, true, 5);
    assert_ne!(sets.controls.len(), 0);
    let (mut later, mut never) = (0, 0);
    for control in &sets.controls {
        let exposed = position[control.exposed_id.as_str()];
        assert_eq!(
            control.index_date,
            day(people.exposure[exposed].expect("exposed"))
        );
        let own = position[control.control_id.as_str()];
        assert_eq!(control.exposure_date, people.exposure[own].map(day));
        match control.days_to_exposure() {
            Some(days) => {
                assert!(days > 0, "a control is unexposed on the index date");
                later += 1;
            }
            None => never += 1,
        }
    }
    assert!(later > 0 && never > 0, "later {later}, never {never}");

    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for control in &sets.controls {
        *counts.entry(control.exposed_id.as_str()).or_default() += 1;
    }
    assert_eq!(sets.exposed.len(), counts.len());
    for window in sets.exposed.windows(2) {
        assert!((window[0].index_date, &window[0].id) < (window[1].index_date, &window[1].id));
    }
    for person in &sets.exposed {
        assert_eq!(person.n_controls, counts[person.id.as_str()]);
        assert_eq!(
            person.index_date,
            day(people.exposure[position[person.id.as_str()]].expect("exposed"))
        );
    }
}

/// A person drawn as a control is exposed later and is then an exposed person
/// with controls of their own, in heaven (three controls per case, seed 11)
/// and in the engine. A person is a control in several sets with replacement
/// and in one set without.
#[test]
fn controls_are_exposed_later_and_reused_with_replacement() {
    let reference = reference();
    let recs = records(&reference.people);
    assert!(reference.heaven_k3_controls_in_several_sets > 0);
    assert!(reference.heaven_k3_future_cases_as_controls > 0);
    let cases: BTreeSet<&String> = reference.cases.iter().collect();

    let with = run(&reference, &recs, 3, true, 11);
    let exposed_ids: BTreeSet<&String> = with.exposed.iter().map(|p| &p.id).collect();
    let mut uses: HashMap<&str, usize> = HashMap::new();
    let mut future_cases = 0;
    for control in &with.controls {
        *uses.entry(control.control_id.as_str()).or_default() += 1;
        if cases.contains(&control.control_id) {
            future_cases += 1;
            assert!(exposed_ids.contains(&control.control_id));
        }
    }
    assert!(uses.values().any(|n| *n > 1));
    assert!(future_cases > 0);

    let without = run(&reference, &recs, 3, false, 11);
    let mut uses: HashMap<&str, usize> = HashMap::new();
    for control in &without.controls {
        *uses.entry(control.control_id.as_str()).or_default() += 1;
    }
    assert!(uses.values().all(|n| *n == 1));
    assert!(
        without
            .controls
            .iter()
            .any(|c| cases.contains(&c.control_id)),
        "an exposed person is a control without replacement too"
    );
}

/// The mode of the spec gives the same sets as the matching job.
#[test]
fn spec_mode_matches_the_job() {
    let reference = reference();
    let recs = records(&reference.people);
    let spec = MatchSpec {
        mode: MatchMode::ExposureDensity,
        method: MatchMethod::Random,
        seed: 23,
        ratio: 2,
        replace: true,
        birth_window_days: 30_000,
        age_limit_years: Some(reference.age_limit_years),
        exact: vec!["sex".into(), "area".into()],
        require_alive: true,
        require_resident: true,
        ..MatchSpec::default()
    };
    let SpecOutput::ExposureDensity(from_spec) =
        run_spec(&spec, SpecInput::RiskSet { cohort: &recs }).expect("valid spec")
    else {
        panic!("expected exposure density sets");
    };
    let by_hand = run(&reference, &recs, 2, true, 23);
    assert_eq!(from_spec.outcome.pairs, by_hand.outcome.pairs);
    assert_eq!(from_spec.controls, by_hand.controls);
    assert_eq!(from_spec.exposed, by_hand.exposed);

    // The function on a request gives the sets of the job's pairs.
    let criteria = MatchingCriteria::builder()
        .match_ratio(2)
        .birth_date_window_days(30_000)
        .required_strata(vec!["sex".to_string(), "area".to_string()])
        .allow_replacement(true)
        .build();
    let options = RoleTransitionOptions {
        transition_age_limit_years: AgeLimitYears::new(reference.age_limit_years)
            .expect("positive"),
        ratio_fallback: Vec::new(),
    };
    let request = TransitionMatchRequest::new(&criteria, &options, DeterministicSelection, &());
    let direct = match_exposure_density(&recs, request).expect("unique ids");
    let job = MatchJob::new_transition(&recs, 0)
        .with_ratio(ratio(2))
        .with_birth_window(window())
        .with_age_limit(AgeLimitYears::new(reference.age_limit_years).expect("positive"))
        .with_exact_matches(["sex", "area"])
        .with_replacement(true)
        .with_strategy(DeterministicSelection)
        .run();
    assert_eq!(direct.outcome.pairs, job.pairs);
    assert_ne!(direct.controls.len(), 0);
}

/// A cohort with two units of one id has pairs that name no single unit.
#[test]
fn duplicate_ids_are_an_error() {
    let reference = reference();
    let mut recs = records(&reference.people);
    let first = recs[0].clone();
    recs.push(first);
    let outcome = run_job(&reference, &recs, 1, true, 1);
    assert!(matches!(
        exposure_density_sets(&recs, outcome),
        Err(ExposureDensityError::DuplicateId { .. })
    ));
}

/// Chi-square critical value for `df` at upper-tail probability `alpha`,
/// found by bisection on the survival function to a relative width of 1e-12.
fn chi_square_critical(df: f64, alpha: f64) -> f64 {
    let (mut lo, mut hi) = (0.0_f64, df + 100.0 * df.sqrt() + 100.0);
    while hi - lo > 1e-12 * hi {
        let mid = f64::midpoint(lo, hi);
        if chi_square_sf(mid, df) > alpha {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi
}

fn chi_square(counts: &[usize]) -> f64 {
    let total: usize = counts.iter().sum();
    let expected = total as f64 / counts.len() as f64;
    counts
        .iter()
        .map(|c| {
            let d = *c as f64 - expected;
            d * d / expected
        })
        .sum()
}

/// Level of the whole family of tests: the chance that a correct sampler
/// fails the per-case or the pooled test on the fixed seeds is below this.
const FAMILY_ALPHA: f64 = 1e-3;

fn assert_uniform(counts_by_case: &[(String, Vec<usize>)], label: &str) {
    // Each case is tested at FAMILY_ALPHA / (2 * cases) and the pooled
    // statistic, which sums independent chi-squares, at FAMILY_ALPHA / 2, so
    // the Bonferroni bound on the family is FAMILY_ALPHA.
    let n = counts_by_case.len() as f64;
    let mut pooled = 0.0;
    let mut pooled_df = 0.0;
    for (case, counts) in counts_by_case {
        let df = (counts.len() - 1) as f64;
        let stat = chi_square(counts);
        let limit = chi_square_critical(df, FAMILY_ALPHA / (2.0 * n));
        assert!(
            stat < limit,
            "{label}: case {case}, chi-square {stat} with {df} df, limit {limit}"
        );
        pooled += stat;
        pooled_df += df;
    }
    let limit = chi_square_critical(pooled_df, FAMILY_ALPHA / 2.0);
    assert!(
        pooled < limit,
        "{label}: pooled chi-square {pooled} with {pooled_df} df, limit {limit}"
    );
}

/// Every member of a risk set is drawn equally often, in heaven (the counts in
/// the reference, 1500 runs with seeds 1 to 1500) and in the engine (1500 runs
/// with seeds 1 to 1500), for each exposed person with at least two people at
/// risk. A run draws one control per exposed person, with replacement across
/// them. The sums of the counts are checked to be the number of runs.
#[test]
fn draws_are_uniform_over_the_risk_set_in_both() {
    let reference = reference();
    let recs = records(&reference.people);
    assert_eq!(reference.draws.len(), reference.r_chi_square.len());

    let r_counts: Vec<(String, Vec<usize>)> = reference
        .draws
        .iter()
        .map(|d| {
            assert_eq!(d.counts.iter().sum::<usize>(), reference.reps);
            (d.case.clone(), d.counts.clone())
        })
        .collect();
    for (draw, stat) in reference.draws.iter().zip(&reference.r_chi_square) {
        let ours = chi_square(&draw.counts);
        assert!((ours - stat).abs() <= 1e-9 * stat.max(1.0), "{ours} {stat}");
    }
    assert_uniform(&r_counts, "heaven");

    let mut ours: Vec<HashMap<String, usize>> = vec![HashMap::new(); reference.draws.len()];
    for seed in 1..=reference.reps as u64 {
        let got = by_exposed(&run(&reference, &recs, 1, true, seed));
        for (q, draw) in reference.draws.iter().enumerate() {
            let chosen = &got[&draw.case];
            assert_eq!(chosen.len(), 1);
            *ours[q].entry(chosen[0].clone()).or_default() += 1;
        }
    }
    let our_counts: Vec<(String, Vec<usize>)> = reference
        .draws
        .iter()
        .zip(&ours)
        .map(|(draw, seen)| {
            for key in seen.keys() {
                assert!(draw.members.contains(key), "drawn outside the risk set");
            }
            let counts: Vec<usize> = draw
                .members
                .iter()
                .map(|m| seen.get(m).copied().unwrap_or(0))
                .collect();
            (draw.case.clone(), counts)
        })
        .collect();
    assert_uniform(&our_counts, "engine");
}

/// The cases are the people exposed before the age limit, the same people
/// heaven gets as cases.
#[test]
fn age_limit_selects_the_cases_heaven_gets() {
    let reference = reference();
    let p = &reference.people;
    let expected: Vec<&String> = (0..p.id.len())
        .filter(|i| p.case[*i] == 1)
        .map(|i| &p.id[i])
        .collect();
    assert_eq!(expected, reference.cases.iter().collect::<Vec<_>>());
    assert!(
        p.exposure
            .iter()
            .zip(&p.case)
            .any(|(e, c)| e.is_some() && *c == 0),
        "the data has an exposure after the age limit"
    );
}

/// The mode needs one cohort, and takes the options of risk-set matching.
#[test]
fn spec_mode_needs_a_cohort_and_risk_set_options() {
    let reference = reference();
    let recs = records(&reference.people);
    let spec = MatchSpec {
        mode: MatchMode::ExposureDensity,
        ..MatchSpec::default()
    };
    assert!(matches!(
        run_spec::<BalanceRecord>(
            &spec,
            SpecInput::Standard {
                anchors: &[],
                candidates: &[]
            }
        ),
        Err(SpecError::InputMode {
            mode: MatchMode::ExposureDensity
        })
    ));
    let nearest = MatchSpec {
        method: MatchMethod::Nearest,
        ..spec
    };
    assert!(matches!(
        run_spec(&nearest, SpecInput::RiskSet { cohort: &recs }),
        Err(SpecError::Option {
            option: "method",
            ..
        })
    ));
}
