//! Risk-set matching of the shape of a register study's job: a policy of the
//! caller's with each control's at-risk span, residence as periods, an alive
//! check, a caliper on a numeric field, exact matching on sex and
//! municipality, ratio 4 with fallback ratios 3 and 2.
//!
//! Run with `PEOPLE=1000000 WINDOW=365 MUNICIPALITIES=98 cargo bench --bench risk_set_caliper`.
//! Prints the wall time of matching, over `RUNS` repetitions, and a hash of
//! the outcome, which is the same for any two builds that give the same pairs
//! and diagnostics.

#[path = "../tests/common/mod.rs"]
mod common;

use chrono::{Days, NaiveDate};
use common::{Lcg, Person, Rec};
use epistat_matching::RiskSetPolicy;
use epistat_matching::constraints::{CaliperMissing, ResidenceSpell, caliper_on_field};
use epistat_matching::prelude::*;
use epistat_matching::{AtRiskSpan, DefaultRiskSetPolicy};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::Instant;

struct Own<'a> {
    until: &'a HashMap<String, NaiveDate>,
}

impl RiskSetPolicy<Rec> for Own<'_> {
    fn is_control_at_risk(&self, case: &Rec, control: &Rec, date: NaiveDate) -> bool {
        self.until
            .get(control.id())
            .is_none_or(|until| *until > date)
            && DefaultRiskSetPolicy.is_control_at_risk(case, control, date)
    }

    fn control_at_risk_span(&self, control: &Rec) -> Option<AtRiskSpan> {
        Some(AtRiskSpan::of_default_rule(control).ending_by(self.until.get(control.id()).copied()))
    }
}

fn env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let n = env("PEOPLE", 100_000);
    let window = i32::try_from(env("WINDOW", 365)).expect("window");
    let runs = env("RUNS", 3);
    let municipalities = env("MUNICIPALITIES", 98) as u64;
    let people = common::cohort(n, 50, municipalities, 2026);
    let cohort = common::records(&people, common::Level::Municipality, false, true);
    let until: HashMap<String, NaiveDate> = people
        .iter()
        .filter_map(|p| p.eligible_until.map(|d| (p.id.clone(), d)))
        .collect();
    let mut rng = Lcg::new(11);
    let spells: HashMap<String, Vec<ResidenceSpell>> = people
        .iter()
        .filter_map(|p: &Person| {
            if rng.below(10) >= 3 {
                return None;
            }
            let out = p.birth + Days::new(400 + rng.below(3000));
            let back = out + Days::new(30 + rng.below(600));
            Some((
                p.id.clone(),
                vec![
                    ResidenceSpell {
                        from: None,
                        to: Some(out),
                    },
                    ResidenceSpell {
                        from: Some(back),
                        to: None,
                    },
                ],
            ))
        })
        .collect();
    let cases = cohort.iter().filter(|r| r.event_date().is_some()).count();
    println!("people {n}, municipalities {municipalities}, with an event {cases}, window {window}");
    let mut times = Vec::new();
    for run in 0..runs {
        let start = Instant::now();
        let outcome = MatchJob::new_transition(&cohort, 7)
            .with_ratio(MatchRatio::new(4).expect("ratio"))
            .with_ratio_fallback(vec![
                MatchRatio::new(3).expect("ratio"),
                MatchRatio::new(2).expect("ratio"),
            ])
            .with_birth_window(BirthDateWindowDays::new(window).expect("window"))
            .with_age_limit(AgeLimitYears::new(6).expect("age limit"))
            .with_exact_matches(["sex", "municipality"])
            .with_alive_check()
            .with_resident_spells(|r: &Rec| spells.get(r.id()).map(Vec::as_slice))
            .with_constraints(vec![
                caliper_on_field::<Rec>("parent_year", 5.0).on_missing(CaliperMissing::Refuse),
            ])
            .with_risk_set_policy(Own { until: &until })
            .run()
            .expect("a valid job");
        let elapsed = start.elapsed().as_secs_f64();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        format!("{outcome:?}").hash(&mut hasher);
        println!(
            "run {run}: wall {elapsed:.3} s, pairs {}, matched {}, hash {:016x}",
            outcome.pairs.len(),
            outcome.matched_cases,
            hasher.finish()
        );
        times.push(elapsed);
    }
    times.sort_by(f64::total_cmp);
    println!("median wall {:.3} s", times[times.len() / 2]);
}
