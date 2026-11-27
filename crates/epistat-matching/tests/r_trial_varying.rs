//! `match_trials` on a time scale of each person's own (per-person dates)
//! and with a stratum that changes between trials (period strata), checked
//! against `Epi::ccwc`. The reference is `tests/data/trial_varying.json`,
//! written by `tests/data/trial_varying.R`.
//!
//! With a ratio above every risk set and replacement, the engine returns the
//! whole risk set of every initiator in every trial, which is deterministic and
//! compared exactly: the non-initiators of the trial of the same sex and the
//! same flare status in the trial's period, entered, alive and resident on the
//! initiator's own date.

#[path = "common/r_reference.rs"]
mod r_reference;

#[path = "common/trial_panel.rs"]
mod trial_panel;
use chrono::{Days, NaiveDate};
use epistat_matching::Error;
use epistat_matching::match_trials;
use epistat_matching::prelude::*;
use epistat_matching::{PeriodDates, PeriodStrata, TrialMatching, TrialOptions, trial_seed};
use r_reference::{Versions, assert_r_version, reference as load_reference};
use serde::Deserialize;
use std::collections::BTreeMap;
use trial_panel::{PanelPeriod, PanelPerson, PanelRows};

#[derive(Deserialize)]
struct People {
    id: Vec<String>,
    sex: Vec<String>,
    birth: Vec<i64>,
    entry: Vec<i64>,
    death: Vec<Option<i64>>,
    emigration: Vec<Option<i64>>,
}

#[derive(Deserialize)]
struct PanelCols {
    id: Vec<String>,
    period: Vec<i64>,
    date: Vec<i64>,
    eligible: Vec<u8>,
    treatment: Vec<u8>,
    outcome: Vec<u8>,
    flare: Vec<String>,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    age_limit_years: u8,
    people: People,
    panel: PanelCols,
    risk_sets: BTreeMap<String, Vec<String>>,
}

fn day(days: i64) -> NaiveDate {
    NaiveDate::from_ymd_opt(1970, 1, 1)
        .expect("valid date")
        .checked_add_days(Days::new(u64::try_from(days).expect("after 1970")))
        .expect("in range")
}

/// The panel in the order of the reference's rows, with each person's own
/// dates, and the flare of every row as period strata.
fn panel(r: &Reference) -> (Vec<PanelPerson<BaseRecord>>, PeriodStrata) {
    let p = &r.people;
    let c = &r.panel;
    let mut panel: Vec<PanelPerson<BaseRecord>> = Vec::new();
    for i in 0..c.id.len() {
        if panel.last().is_none_or(|last| last.record.id != c.id[i]) {
            let k = p.id.iter().position(|id| *id == c.id[i]).expect("a person");
            let mut record =
                BaseRecord::new(p.id[k].clone(), day(p.birth[k])).with_entry_date(day(p.entry[k]));
            record.strata.insert("sex".into(), p.sex[k].clone());
            if let Some(d) = p.death[k] {
                record = record.with_death_date(day(d));
            }
            if let Some(d) = p.emigration[k] {
                record = record.with_emigration_date(day(d));
            }
            panel.push(PanelPerson {
                record,
                periods: Vec::new(),
            });
        }
        panel
            .last_mut()
            .expect("a person")
            .periods
            .push(PanelPeriod {
                period: c.period[i],
                date: day(c.date[i]),
                eligible: c.eligible[i] == 1,
                treatment: Some(c.treatment[i] == 1),
                event: c.outcome[i] == 1,
            });
    }
    let strata = PeriodStrata::new(vec!["flare".into()], vec![c.flare.clone()]);
    (panel, strata)
}

fn sets(
    r: &Reference,
    panel: &[PanelPerson<BaseRecord>],
    matching: &TrialMatching<'_>,
) -> Result<BTreeMap<String, Vec<String>>, Error> {
    let rows = PanelRows::of(panel);
    let outcome = match_trials(&rows.panel(), &TrialOptions::default(), matching, |pool| {
        pool.job(trial_seed(1, pool.period()))
            .with_ratio(MatchRatio::new(10_000).expect("a ratio"))
            .with_replacement(true)
            .with_birth_window(BirthDateWindowDays::new(30_000).expect("non-negative"))
            .with_age_limit(AgeLimitYears::new(r.age_limit_years).expect("positive"))
            .with_exact_matches(["sex", "flare"])
            .with_alive_check()
            .with_resident_check()
            .run()
    })?;
    let mut map: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for trial in &outcome.trials {
        for pair in &trial.outcome.pairs {
            map.entry(format!("{}:{}", trial.period, pair.case_id.as_str()))
                .or_default()
                .push(pair.control_id.as_str().to_string());
        }
        // Each member starts follow-up on the date of its own period.
        for m in &trial.members {
            let person = panel
                .iter()
                .find(|p| p.record.id == m.id)
                .expect("in the panel");
            let row = person
                .periods
                .iter()
                .find(|row| row.period == trial.period)
                .expect("a row in the trial's period");
            assert_eq!(m.follow_up_start, row.date);
        }
    }
    for controls in map.values_mut() {
        controls.sort();
    }
    Ok(map)
}

#[test]
fn risk_sets_with_own_dates_and_period_strata_equal_ccwc() {
    let r: Reference = load_reference("trial_varying.json");
    assert_r_version(&r.versions["R"]);
    let (panel, strata) = panel(&r);
    let got = sets(
        &r,
        &panel,
        &TrialMatching {
            dates: PeriodDates::PerPerson,
            strata: Some(&strata),
        },
    )
    .expect("valid panel");
    let want: BTreeMap<String, Vec<String>> = r
        .risk_sets
        .iter()
        .filter(|(_, set)| !set.is_empty())
        .map(|(k, set)| (k.clone(), set.clone()))
        .collect();
    assert!(want.len() > 50, "the reference has many sets");
    assert_eq!(got.len(), want.len(), "initiators with a set");
    for (initiator, controls) in &want {
        assert_eq!(&got[initiator], controls, "risk set of {initiator}");
    }
}

/// The same panel cannot be matched with shared dates, and strata without a
/// value for every row are refused.
#[test]
fn shared_dates_and_short_strata_are_errors() {
    let r: Reference = load_reference("trial_varying.json");
    let (panel, _) = panel(&r);
    assert!(matches!(
        sets(&r, &panel, &TrialMatching::default()),
        Err(Error::InconsistentDate { .. })
    ));
    let short = PeriodStrata::new(vec!["flare".into()], vec![r.panel.flare[1..].to_vec()]);
    assert!(matches!(
        sets(
            &r,
            &panel,
            &TrialMatching {
                dates: PeriodDates::PerPerson,
                strata: Some(&short),
            }
        ),
        Err(Error::StrataRows { .. })
    ));
}
