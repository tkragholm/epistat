//! Rows of a [`Cohort`] give the same results as owned records.
//!
//! Each test builds one simulated cohort twice, as [`BalanceRecord`]s and as
//! a [`Cohort`], from the same values, runs an entry point on both and asserts
//! that the outputs print the same with `{:?}`. Rust prints an `f64` as the
//! shortest text that reads back to the same bits, so equal text is equal
//! output to the bit: every pair, weight, subclass, count and diagnostic. The
//! cohorts have exact ties (covariates on a grid of a few values, few strata
//! and birth dates within a few days), missing values in every kind of column,
//! and are run on several seeds. After every entry point the cohort must not
//! have built the per-row maps, which would cost what owned records cost.

#[path = "common/trial_panel.rs"]
mod trial_panel;
use epistat_matching::output::{MatchDataOptions, Units, match_data};
use epistat_matching::spec::{
    BinningSpec, CemSpec, CemVariableSpec, CovarianceSpec, CovariatesSpec, DistanceSpec,
    EstimatorSpec, LinkSpec, MatchMethod, MatchMode, MissingSpec, ScaleSpec, StdCaliperSpec,
    SubclassSpec, TransformSpec,
};
use epistat_matching::{
    BalanceRecord, BalanceReportOptions, BalanceWeights, CovariateValue, NumericBalanceTransform,
    SpecOutput, balance_diagnostics, balance_report, balance_summary,
};
use std::collections::BTreeMap;
use trial_panel::{PanelPeriod, PanelPerson, PanelRows};

use chrono::{Days, NaiveDate};
use epistat_matching::effects::{
    Clustering, EffectSpec, MatchedSample, OutcomeModel, UnitData, estimate_effects,
};
use epistat_matching::prelude::*;
use epistat_matching::{Cohort, CohortRow};

const SEEDS: [u64; 3] = [1, 2, 3];
const COVARIATES: [&str; 3] = ["x1", "x2", "x3"];

/// The `{:?}` text of a value with the entries of every map and set sorted,
/// since a `HashMap` prints in an order of its own.
fn canonical(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut at = 0;
    group(&chars, &mut at, None)
}

/// The text up to `close`, which is consumed, with maps and sets sorted.
fn group(chars: &[char], at: &mut usize, close: Option<char>) -> String {
    let mut out = String::new();
    while *at < chars.len() {
        let c = chars[*at];
        *at += 1;
        if Some(c) == close {
            return out;
        }
        match c {
            '"' => {
                out.push(c);
                while *at < chars.len() {
                    let d = chars[*at];
                    *at += 1;
                    out.push(d);
                    if d == '\\' && *at < chars.len() {
                        out.push(chars[*at]);
                        *at += 1;
                    } else if d == '"' {
                        break;
                    }
                }
            }
            '(' | '[' => {
                out.push(c);
                out += &group(chars, at, Some(if c == '(' { ')' } else { ']' }));
                out.push(if c == '(' { ')' } else { ']' });
            }
            '{' => {
                // A struct prints as `Name { ... }`; anything else is a map or a set.
                let is_struct = out.ends_with(' ')
                    && out
                        .trim_end()
                        .chars()
                        .last()
                        .is_some_and(|p| p.is_alphanumeric() || p == '_' || p == '>');
                let inner = group(chars, at, Some('}'));
                out.push('{');
                if is_struct {
                    out += &inner;
                } else {
                    let mut entries = top_level_entries(&inner);
                    entries.sort();
                    out += &entries.join(", ");
                }
                out.push('}');
            }
            _ => out.push(c),
        }
    }
    out
}

/// `text` split at the commas outside brackets and strings.
fn top_level_entries(text: &str) -> Vec<String> {
    let mut entries = Vec::new();
    let mut current = String::new();
    let mut depth = 0_i32;
    let mut in_string = false;
    let mut escaped = false;
    for c in text.chars() {
        if in_string {
            current.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ',' if depth == 0 => {
                entries.push(current.trim().to_string());
                current.clear();
                continue;
            }
            _ => {}
        }
        current.push(c);
    }
    if !current.trim().is_empty() {
        entries.push(current.trim().to_string());
    }
    entries
}

/// Asserts that two outputs print the same up to the order of map entries,
/// showing where they first differ.
fn assert_same(owned: &str, rows: &str, label: &str) {
    let (owned, rows) = (canonical(owned), canonical(rows));
    if owned == rows {
        return;
    }
    let at = owned
        .bytes()
        .zip(rows.bytes())
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| owned.len().min(rows.len()));
    let from = at.saturating_sub(200);
    let window = |text: &str| {
        text.get(from..(at + 200).min(text.len()))
            .unwrap_or("")
            .to_string()
    };
    panic!(
        "{label}: outputs differ at byte {at}\nowned: {}\nrows:  {}",
        window(&owned),
        window(&rows)
    );
}

#[test]
fn canonical_text_sorts_maps_but_not_structs_or_lists() {
    assert_eq!(
        canonical(r#"A { b: {"y": [2, 1], "x": B { c: 1, a: 2 }}, s: "}, {" }"#),
        r#"A { b: {"x": B { c: 1, a: 2 }, "y": [2, 1]}, s: "}, {" }"#
    );
}

use epistat_testkit::SplitMix64 as Rng;

trait Percent {
    fn chance(&mut self, percent: u64) -> bool;
}

impl Percent for Rng {
    fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
}

/// The values of one unit, from which both kinds of record are made.
struct Unit {
    id: String,
    birth: NaiveDate,
    treated: bool,
    strata: Vec<(&'static str, Option<String>)>,
    x: [Option<f64>; 3],
    level: Option<String>,
    family: Option<String>,
    death: Option<NaiveDate>,
    emigration: Option<NaiveDate>,
    entry: Option<NaiveDate>,
    event: Option<NaiveDate>,
    outcome: f64,
}

fn origin() -> NaiveDate {
    NaiveDate::from_ymd_opt(2000, 1, 1).expect("valid date")
}

fn day(offset: u64) -> NaiveDate {
    origin() + Days::new(offset)
}

/// `n` units, about a quarter treated. Covariates take a few values so pairs
/// tie, and every column has some rows without a value.
fn units(seed: u64, n: usize) -> Vec<Unit> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|i| {
            let half_steps = rng.below(5);
            let x1 = half_steps as f64 / 2.0;
            let x2 = rng.below(3) as f64;
            let x3 = if rng.chance(3) {
                None
            } else {
                Some(rng.below(4) as f64 - 1.5)
            };
            let treated = rng.below(100) < 20 + 10 * (half_steps / 2);
            let event = rng.chance(8).then(|| day(400 + rng.below(1500)));
            Unit {
                id: format!("u{i}"),
                birth: day(rng.below(6)),
                treated,
                strata: vec![
                    ("s", Some(format!("s{}", rng.below(3)))),
                    ("sex", (!rng.chance(2)).then(|| format!("{}", rng.below(2)))),
                    ("fam", Some(format!("f{}", rng.below((n / 3) as u64 + 1)))),
                    ("x3c", Some(format!("{}", rng.below(4)))),
                ],
                x: [Some(x1), Some(x2), x3],
                level: (!rng.chance(5)).then(|| {
                    ["a", "b", "c"][usize::try_from(rng.below(3)).expect("small")].to_string()
                }),
                family: rng
                    .chance(90)
                    .then(|| format!("k{}", rng.below((n / 2) as u64 + 1))),
                death: rng.chance(15).then(|| day(300 + rng.below(1800))),
                emigration: rng.chance(5).then(|| day(300 + rng.below(1800))),
                entry: rng.chance(10).then(|| day(rng.below(600))),
                event,
                outcome: x1 + 0.5 * f64::from(u8::from(treated)) + rng.below(7) as f64 / 3.0,
            }
        })
        .collect()
}

fn owned(unit: &Unit) -> BalanceRecord {
    let mut record = BalanceRecord::new(unit.id.clone(), unit.birth);
    for (key, value) in &unit.strata {
        if let Some(value) = value {
            record.core.strata.insert((*key).to_string(), value.clone());
        }
    }
    for (name, value) in COVARIATES.iter().zip(unit.x) {
        if let Some(value) = value {
            record.core.numerics.insert((*name).to_string(), value);
        }
        record.covariates.insert(
            (*name).to_string(),
            value
                .filter(|v| v.is_finite())
                .map_or(CovariateValue::Missing, CovariateValue::Numeric),
        );
    }
    record.covariates.insert(
        "level".into(),
        unit.level
            .clone()
            .map_or(CovariateValue::Missing, CovariateValue::Categorical),
    );
    record.core.unique_key.clone_from(&unit.family);
    record.core.death_date = unit.death;
    record.core.emigration_date = unit.emigration;
    record.core.entry_date = unit.entry;
    record
}

fn cohort(units: &[Unit]) -> Cohort {
    let mut builder = Cohort::builder(
        units.iter().map(|u| u.id.as_str()),
        units.iter().map(|u| u.birth).collect(),
    );
    for (k, (key, _)) in units[0].strata.iter().enumerate() {
        builder = builder.stratum(*key, units.iter().map(|u| u.strata[k].1.as_deref()));
    }
    for (k, name) in COVARIATES.iter().enumerate() {
        builder = builder.numeric_field_and_covariate(*name, units.iter().map(|u| u.x[k]));
    }
    builder
        .categorical_covariate("level", units.iter().map(|u| u.level.as_deref()))
        .unique_key(units.iter().map(|u| u.family.as_deref()))
        .death_date(units.iter().map(|u| u.death).collect())
        .emigration_date(units.iter().map(|u| u.emigration).collect())
        .entry_date(units.iter().map(|u| u.entry).collect())
        .event_date(units.iter().map(|u| u.event).collect())
        .build()
        .expect("valid cohort")
}

/// Both groups of a standard match, owned and as rows.
struct Groups<'a> {
    anchors: Vec<BalanceRecord>,
    candidates: Vec<BalanceRecord>,
    anchor_rows: Vec<CohortRow<'a>>,
    candidate_rows: Vec<CohortRow<'a>>,
}

fn groups<'a>(units: &[Unit], cohort: &'a Cohort) -> Groups<'a> {
    let mut groups = Groups {
        anchors: Vec::new(),
        candidates: Vec::new(),
        anchor_rows: Vec::new(),
        candidate_rows: Vec::new(),
    };
    for (unit, row) in units.iter().zip(cohort.rows()) {
        if unit.treated {
            groups.anchors.push(owned(unit));
            groups.anchor_rows.push(row);
        } else {
            groups.candidates.push(owned(unit));
            groups.candidate_rows.push(row);
        }
    }
    groups
}

fn covariates() -> CovariatesSpec {
    CovariatesSpec {
        names: COVARIATES.iter().map(|s| (*s).to_string()).collect(),
        missing: MissingSpec::Impute,
    }
}

fn propensity(caliper: Option<f64>) -> DistanceSpec {
    DistanceSpec::Propensity {
        covariates: CovariatesSpec {
            names: vec!["x1".into(), "x2".into(), "x3".into(), "level".into()],
            missing: MissingSpec::Impute,
        },
        link: LinkSpec::Logit,
        estimator: EstimatorSpec::Glm,
        scale: ScaleSpec::Probability,
        sampling_weights: None,
        caliper,
    }
}

/// The standard-mode specs checked: nearest neighbour with and without
/// replacement, strata, calipers, a unique key and vector distances, and the
/// subclass, exact, coarsened exact, quick, optimal and full methods.
fn standard_specs() -> Vec<(&'static str, MatchSpec)> {
    let base = MatchSpec {
        method: MatchMethod::Nearest,
        distance: propensity(None),
        birth_window_days: 36_500,
        ..MatchSpec::default()
    };
    let unwindowed = MatchSpec::default().birth_window_days;
    let mut field_caliper = BTreeMap::new();
    field_caliper.insert("x2".to_string(), 1.0);
    vec![
        ("nearest", base.clone()),
        (
            "nearest_replace_ratio",
            MatchSpec {
                replace: true,
                ratio: 2,
                ..base.clone()
            },
        ),
        (
            "nearest_strata_caliper",
            MatchSpec {
                exact: vec!["s".into(), "sex".into()],
                distance: propensity(Some(0.2)),
                std_caliper: StdCaliperSpec::All(true),
                ..base.clone()
            },
        ),
        (
            "nearest_field_caliper_antiexact",
            MatchSpec {
                calipers: field_caliper,
                antiexact: vec!["x3c".into()],
                ..base.clone()
            },
        ),
        (
            "nearest_unique",
            MatchSpec {
                unique_by_key: Some("fam".into()),
                ..base.clone()
            },
        ),
        (
            "nearest_random",
            MatchSpec {
                method: MatchMethod::Random,
                distance: DistanceSpec::default(),
                exact: vec!["s".into()],
                birth_window_days: 3,
                seed: 9,
                ..base.clone()
            },
        ),
        (
            "mahalanobis",
            MatchSpec {
                distance: DistanceSpec::Mahalanobis {
                    covariates: covariates(),
                    covariance: CovarianceSpec::default(),
                    transform: TransformSpec::default(),
                    caliper: None,
                },
                exact: vec!["s".into()],
                ..base.clone()
            },
        ),
        (
            "euclidean",
            MatchSpec {
                distance: DistanceSpec::Euclidean {
                    covariates: covariates(),
                    scaled: true,
                    caliper: None,
                },
                ..base.clone()
            },
        ),
        (
            "subclass",
            MatchSpec {
                method: MatchMethod::Subclass,
                subclass: SubclassSpec {
                    n: 4,
                    min_n: 1,
                    reference: None,
                },
                birth_window_days: unwindowed,
                ..base.clone()
            },
        ),
        (
            "exact",
            MatchSpec {
                method: MatchMethod::Exact,
                distance: DistanceSpec::default(),
                exact: vec!["s".into(), "x3c".into()],
                birth_window_days: unwindowed,
                ..base.clone()
            },
        ),
        (
            "cem",
            MatchSpec {
                method: MatchMethod::Cem,
                distance: DistanceSpec::default(),
                cem: CemSpec {
                    variables: vec![
                        CemVariableSpec::Numeric {
                            name: "x1".into(),
                            binning: BinningSpec::Quantiles { n: 3 },
                        },
                        CemVariableSpec::Categorical {
                            name: "s".into(),
                            groups: vec![],
                        },
                    ],
                    k2k: None,
                },
                birth_window_days: unwindowed,
                ..base.clone()
            },
        ),
        (
            "quick",
            MatchSpec {
                method: MatchMethod::Quick,
                birth_window_days: unwindowed,
                ..base.clone()
            },
        ),
        (
            "optimal",
            MatchSpec {
                method: MatchMethod::Optimal,
                exact: vec!["s".into()],
                ..base.clone()
            },
        ),
        (
            "full",
            MatchSpec {
                method: MatchMethod::Full,
                exact: vec!["s".into()],
                ..base
            },
        ),
    ]
}

#[test]
fn standard_matching_is_the_same_on_rows() {
    for seed in SEEDS {
        let units = units(seed, 900);
        let cohort = cohort(&units);
        let g = groups(&units, &cohort);
        for (name, spec) in standard_specs() {
            let on_owned = run_spec(
                &spec,
                SpecInput::Standard {
                    cases: &g.anchors,
                    controls: &g.candidates,
                },
            );
            let on_rows = run_spec(
                &spec,
                SpecInput::Standard {
                    cases: &g.anchor_rows,
                    controls: &g.candidate_rows,
                },
            );
            assert!(on_owned.is_ok(), "{name}, seed {seed}: {on_owned:?}");
            assert_same(
                &format!("{on_owned:?}"),
                &format!("{on_rows:?}"),
                &format!("{name}, seed {seed}"),
            );
        }
        assert!(!cohort.has_built_record_maps());
    }
}

#[test]
fn match_job_nearest_is_the_same_on_rows() {
    for seed in SEEDS {
        let units = units(seed, 900);
        let cohort = cohort(&units);
        let g = groups(&units, &cohort);
        let ratio = MatchRatio::new(2).expect("non-zero ratio");
        let window = BirthDateWindowDays::new(2).expect("non-negative window");
        let on_owned = MatchJob::new_standard(&g.anchors, &g.candidates, seed)
            .with_ratio(ratio)
            .with_birth_window(window)
            .with_exact_matches(["s"])
            .with_unique_by_key(Some("fam".to_string()))
            .with_gender_match()
            .run()
            .expect("a valid job");
        let on_rows = MatchJob::new_standard(&g.anchor_rows, &g.candidate_rows, seed)
            .with_ratio(ratio)
            .with_birth_window(window)
            .with_exact_matches(["s"])
            .with_unique_by_key(Some("fam".to_string()))
            .with_gender_match()
            .run()
            .expect("a valid job");
        assert!(on_owned.matched_cases > 0);
        assert_same(
            &format!("{on_owned:?}"),
            &format!("{on_rows:?}"),
            &format!("seed {seed}"),
        );
        assert!(!cohort.has_built_record_maps());
    }
}

fn risk_set_owned(units: &[Unit]) -> Vec<RoleTransitionRecord<BalanceRecord>> {
    units
        .iter()
        .map(|unit| RoleTransitionRecord::from_record(owned(unit), unit.event))
        .collect()
}

fn risk_set_specs() -> Vec<(&'static str, MatchSpec)> {
    let base = MatchSpec {
        mode: MatchMode::RiskSet,
        method: MatchMethod::Random,
        exact: vec!["s".into()],
        birth_window_days: 4,
        require_alive: true,
        age_limit_years: Some(6),
        seed: 7,
        ..MatchSpec::default()
    };
    let mut field_caliper = BTreeMap::new();
    field_caliper.insert("x1".to_string(), 0.5);
    vec![
        ("risk_set", base.clone()),
        (
            "risk_set_replace",
            MatchSpec {
                replace: true,
                ratio: 3,
                ..base.clone()
            },
        ),
        (
            "risk_set_resident_caliper_unique",
            MatchSpec {
                require_resident: true,
                calipers: field_caliper,
                unique_by_key: Some("fam".into()),
                match_sex: true,
                sex_key: "sex".into(),
                ..base.clone()
            },
        ),
        (
            "risk_set_nearest_birth_date",
            MatchSpec {
                method: MatchMethod::NearestBirthDate,
                ratio: 2,
                ..base.clone()
            },
        ),
        (
            "exposure_density",
            MatchSpec {
                mode: MatchMode::ExposureDensity,
                ..base
            },
        ),
    ]
}

#[test]
fn risk_set_and_exposure_density_are_the_same_on_rows() {
    for seed in SEEDS {
        let units = units(seed, 3000);
        let cohort = cohort(&units);
        let owned = risk_set_owned(&units);
        let rows = cohort.risk_set_rows();
        for (name, spec) in risk_set_specs() {
            let on_owned = run_spec(&spec, SpecInput::RiskSet { cohort: &owned });
            let on_rows = run_spec(&spec, SpecInput::RiskSet { cohort: &rows });
            assert!(on_owned.is_ok(), "{name}, seed {seed}: {on_owned:?}");
            assert_same(
                &format!("{on_owned:?}"),
                &format!("{on_rows:?}"),
                &format!("{name}, seed {seed}"),
            );
        }
        assert!(!cohort.has_built_record_maps());
    }
}

#[test]
fn match_job_transition_is_the_same_on_rows() {
    for seed in SEEDS {
        let units = units(seed, 3000);
        let cohort = cohort(&units);
        let owned = risk_set_owned(&units);
        let rows = cohort.risk_set_rows();
        let ratio = MatchRatio::new(2).expect("non-zero ratio");
        let fallback = vec![MatchRatio::new(1).expect("non-zero ratio")];
        let window = BirthDateWindowDays::new(3).expect("non-negative window");
        let age = AgeLimitYears::new(6).expect("positive age");
        let on_owned = MatchJob::new_transition(&owned, seed)
            .with_ratio(ratio)
            .with_ratio_fallback(fallback.clone())
            .with_birth_window(window)
            .with_age_limit(age)
            .with_exact_matches(["s"])
            .with_alive_check()
            .with_resident_check()
            .with_constraints(vec![epistat_matching::constraints::caliper_on_field(
                "x2", 1.0,
            )])
            .run()
            .expect("a valid job");
        let on_rows = MatchJob::new_transition(&rows, seed)
            .with_ratio(ratio)
            .with_ratio_fallback(fallback)
            .with_birth_window(window)
            .with_age_limit(age)
            .with_exact_matches(["s"])
            .with_alive_check()
            .with_resident_check()
            .with_constraints(vec![epistat_matching::constraints::caliper_on_field(
                "x2", 1.0,
            )])
            .run()
            .expect("a valid job");
        assert!(on_owned.matched_cases > 0);
        assert_same(
            &format!("{on_owned:?}"),
            &format!("{on_rows:?}"),
            &format!("seed {seed}"),
        );
        assert!(!cohort.has_built_record_maps());
    }
}

fn panel<R>(units: &[Unit], record: impl Fn(usize) -> R) -> Vec<PanelPerson<R>> {
    units
        .iter()
        .enumerate()
        .map(|(i, unit)| {
            let start = i % 4;
            let event_period = unit.event.map(|d| {
                usize::try_from((d - origin()).num_days() / 500).expect("after the origin")
            });
            PanelPerson {
                record: record(i),
                periods: (0..4)
                    .map(|k| PanelPeriod {
                        period: i64::try_from(k).expect("small"),
                        date: day(500 * (k as u64 + 1)),
                        eligible: !unit.treated || k <= start,
                        treatment: Some(unit.treated && k >= start),
                        event: event_period == Some(k),
                    })
                    .collect(),
            }
        })
        .collect()
}

#[test]
fn sequential_trials_are_the_same_on_rows() {
    for seed in SEEDS {
        let units = units(seed, 1500);
        let cohort = cohort(&units);
        let owned_panel = panel(&units, |i| owned(&units[i]));
        let row_panel = panel(&units, |i| cohort.row(i).expect("row in cohort"));
        for replace in [false, true] {
            let spec = MatchSpec {
                mode: MatchMode::SequentialTrials,
                method: MatchMethod::Random,
                exact: vec!["s".into()],
                birth_window_days: 4,
                require_alive: true,
                age_limit_years: Some(8),
                replace,
                seed,
                ..MatchSpec::default()
            };
            let on_owned = run_spec(
                &spec,
                SpecInput::Trials {
                    panel: &PanelRows::of(&owned_panel).panel(),
                },
            );
            let on_rows = run_spec(
                &spec,
                SpecInput::Trials {
                    panel: &PanelRows::of(&row_panel).panel(),
                },
            );
            assert!(on_owned.is_ok(), "seed {seed}: {on_owned:?}");
            assert_same(
                &format!("{on_owned:?}"),
                &format!("{on_rows:?}"),
                &format!("seed {seed}, replace {replace}"),
            );
        }
        assert!(!cohort.has_built_record_maps());
    }
}

#[test]
fn balance_and_effects_are_the_same_on_rows() {
    for seed in SEEDS {
        let units = units(seed, 900);
        let cohort = cohort(&units);
        let g = groups(&units, &cohort);
        let spec = standard_specs().remove(0).1;
        let Ok(SpecOutput::Pairs { outcome, .. }) = run_spec(
            &spec,
            SpecInput::Standard {
                cases: &g.anchors,
                controls: &g.candidates,
            },
        ) else {
            panic!("pairs")
        };
        let options = BalanceReportOptions {
            numeric_transforms: NumericBalanceTransform::SquaresAndPairwiseInteractions,
            ..BalanceReportOptions::default()
        };
        let on_owned = balance_report(&g.anchors, &g.candidates, &outcome, &options);
        let on_rows = balance_report(&g.anchor_rows, &g.candidate_rows, &outcome, &options);
        assert!(!on_owned.numeric.is_empty() && !on_owned.categorical.is_empty());
        assert_same(
            &format!("{on_owned:?}"),
            &format!("{on_rows:?}"),
            &format!("seed {seed}"),
        );

        let weights = BalanceWeights::from_outcome(&outcome);
        let summary_owned = balance_summary(&g.anchors, &g.candidates, &weights, &options);
        let summary_rows = balance_summary(&g.anchor_rows, &g.candidate_rows, &weights, &options);
        assert_same(
            &format!("{summary_owned:?}"),
            &format!("{summary_rows:?}"),
            "summary_owned",
        );
        let diagnostics_owned =
            balance_diagnostics(&g.anchors, &g.candidates, &outcome, &["s".to_string()]);
        let diagnostics_rows = balance_diagnostics(
            &g.anchor_rows,
            &g.candidate_rows,
            &outcome,
            &["s".to_string()],
        );
        assert_same(
            &format!("{diagnostics_owned:?}"),
            &format!("{diagnostics_rows:?}"),
            "diagnostics_owned",
        );

        let effect = |ids: Vec<String>| {
            let treated = (0..ids.len()).map(|i| i < g.anchors.len()).collect();
            let y = ids
                .iter()
                .map(|id| units[id[1..].parse::<usize>().expect("numeric id")].outcome)
                .collect();
            let x1 = ids
                .iter()
                .map(|id| units[id[1..].parse::<usize>().expect("numeric id")].x[0].expect("x1"))
                .collect();
            let data = UnitData::new(ids.clone(), treated, y)
                .expect("unit data")
                .with_covariate("x1", x1)
                .expect("covariate");
            let anchor = (0..ids.len()).map(|i| i < g.anchors.len()).collect();
            let units = Units::new(ids, anchor).expect("distinct ids");
            let md = match_data(&outcome, &units, MatchDataOptions::default()).expect("match data");
            let sample = MatchedSample::from_match_data(&md, &data).expect("sample");
            let mut spec = EffectSpec::new(OutcomeModel::Linear);
            spec.clustering = Clustering::Pair;
            estimate_effects(&sample, &spec).expect("effects")
        };
        let ids_owned = g
            .anchors
            .iter()
            .chain(&g.candidates)
            .map(|r| r.id().to_string())
            .collect();
        let ids_rows = g
            .anchor_rows
            .iter()
            .chain(&g.candidate_rows)
            .map(|r| r.id().to_string())
            .collect();
        assert_eq!(
            format!("{:?}", effect(ids_owned)),
            format!("{:?}", effect(ids_rows))
        );
        assert!(!cohort.has_built_record_maps());
    }
}
