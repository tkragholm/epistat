//! The risk-set matching of Studies I and II, hashed to the bit.
//!
//! [`run_matching`] is PHD's `models::matching::run_matching` with its record
//! type and risk-set policy: two transition jobs, sex and municipality, then
//! sex and region for the cases the first pass left unmatched with the
//! comparators it used removed, residence spells, comparators censored at
//! their own eligibility end, the fractional anchor-unit weights and their
//! effective sample sizes. Study I matches 1:4 with fallbacks 3, 2 and 1 and
//! calipers of 1,826 days on both parents' birth days with unknown matching
//! unknown; Study II matches 1:4 with fallbacks 3 and 2 and no caliper, its
//! residence read from emigration dates. Both use a 30-day birth window and an
//! age limit of 6 years. A change that moves a hash moves the matched sets of
//! both studies, so a port of these calls to new names must reproduce every
//! hash.
//!
//! The hashes were recorded on aarch64 macOS. Print them with
//! `STUDY_PATH_PRINT=1 cargo test --test study_path -- --nocapture`.
#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::collections::{HashMap, HashSet};

use chrono::{Days, NaiveDate};
use epistat_matching::constraints::{MissingPolicy, ResidenceSpell, caliper_on_field};
use epistat_matching::prelude::*;
use epistat_matching::{PairWeightTable, UnitRole, UnitWeightTable};
use epistat_stats::weighted::effective_sample_size;
use epistat_testkit::{Fingerprint, SplitMix64};

struct CohortRow {
    person_id: String,
    birth_date: NaiveDate,
    sex: String,
    municipality: String,
    region: String,
    first_scd_date: Option<NaiveDate>,
    death_date: Option<NaiveDate>,
    emigration_date: Option<NaiveDate>,
    exposed: bool,
    comparator_eligible_until: Option<NaiveDate>,
    numerics: Vec<(String, f64)>,
}

struct MatchConfig {
    seed: u64,
    ratio: usize,
    fallback_ratios: Vec<usize>,
    birth_window_days: i32,
    age_limit_years: u8,
    calipers: Vec<(String, f64)>,
    caliper_missing: MissingPolicy,
}

struct MatchedPairRow {
    exposed_id: String,
    control_id: String,
    match_type: &'static str,
    match_ratio: i32,
}

struct MatchOutput {
    pairs: Vec<MatchedPairRow>,
    passes: [(usize, usize); 2],
    unit_weights: UnitWeightTable,
    pair_weights: PairWeightTable,
    effective_sample_sizes: [Option<f64>; 3],
    unit_weight_total: f64,
}

struct Person {
    base: BaseRecord,
    unexposed: bool,
    eligible_until: Option<NaiveDate>,
}

type Member = RoleTransitionRecord<Person>;

impl MatchingRecord for Person {
    fn id(&self) -> &str {
        self.base.id()
    }

    fn birth_date(&self) -> NaiveDate {
        self.base.birth_date()
    }

    fn strata(&self) -> &HashMap<String, String> {
        self.base.strata()
    }

    fn numeric(&self, name: &str) -> Option<f64> {
        self.base.numeric(name)
    }

    fn unique_key(&self) -> Option<&str> {
        self.base.unique_key()
    }

    fn death_date(&self) -> Option<NaiveDate> {
        self.base.death_date()
    }

    fn emigration_date(&self) -> Option<NaiveDate> {
        self.base.emigration_date()
    }
}

struct PassPolicy<'a> {
    matched_case_ids: &'a HashSet<&'a str>,
    used_control_ids: &'a HashSet<&'a str>,
}

impl RiskSetPolicy<Member> for PassPolicy<'_> {
    fn is_case_eligible(&self, row: &Member, age_limit_years: u8) -> bool {
        !row.unexposed
            && !self.matched_case_ids.contains(row.id())
            && DefaultRiskSetPolicy.is_case_eligible(row, age_limit_years)
    }

    fn is_control_at_risk(
        &self,
        case: &Member,
        control: &Member,
        case_event_date: NaiveDate,
    ) -> bool {
        !self.used_control_ids.contains(control.id())
            && control
                .eligible_until
                .is_none_or(|until| until > case_event_date)
            && DefaultRiskSetPolicy.is_control_at_risk(case, control, case_event_date)
    }

    fn control_at_risk_span(&self, control: &Member) -> Option<AtRiskSpan> {
        Some(if self.used_control_ids.contains(control.id()) {
            AtRiskSpan::Never
        } else {
            AtRiskSpan::of_default_rule(control).ending_by(control.eligible_until)
        })
    }
}

fn make_member(row: &CohortRow) -> Member {
    let mut record = BaseRecord::new(&row.person_id, row.birth_date);
    record.strata = HashMap::with_capacity(3);
    record.strata.insert("sex".to_string(), row.sex.clone());
    record
        .strata
        .insert("municipality".to_string(), row.municipality.clone());
    record
        .strata
        .insert("region".to_string(), row.region.clone());
    for (name, value) in &row.numerics {
        record = record.with_numeric(name, *value);
    }
    if let Some(d) = row.death_date {
        record = record.with_death_date(d);
    }
    if let Some(d) = row.emigration_date {
        record = record.with_emigration_date(d);
    }
    let person = Person {
        unexposed: !row.exposed,
        eligible_until: row.comparator_eligible_until,
        base: record,
    };
    RoleTransitionRecord::from_record(person, row.first_scd_date)
}

fn matched_rows(outcome: &MatchOutcome, match_type: &'static str) -> Vec<MatchedPairRow> {
    let mut ratio_by_case: HashMap<&str, i32> = HashMap::new();
    for pair in &outcome.pairs {
        *ratio_by_case.entry(pair.case_id.as_str()).or_insert(0) += 1;
    }
    outcome
        .pairs
        .iter()
        .map(|pair| MatchedPairRow {
            exposed_id: pair.case_id.clone(),
            control_id: pair.control_id.clone(),
            match_type,
            match_ratio: ratio_by_case[pair.case_id.as_str()],
        })
        .collect()
}

fn run_matching(
    rows: &[CohortRow],
    spells: Option<&HashMap<String, Vec<ResidenceSpell>>>,
    cfg: &MatchConfig,
) -> MatchOutput {
    let calipers = || {
        cfg.calipers
            .iter()
            .map(|(field, window)| {
                caliper_on_field::<Member>(field, *window).on_missing(cfg.caliper_missing)
            })
            .collect::<Vec<_>>()
    };
    let ratio = MatchRatio::new(cfg.ratio).expect("ratio");
    let fallback: Vec<MatchRatio> = cfg
        .fallback_ratios
        .iter()
        .filter_map(|&r| MatchRatio::new(r))
        .collect();
    let birth_window = BirthDateWindowDays::new(cfg.birth_window_days).expect("window");
    let age_limit = AgeLimitYears::new(cfg.age_limit_years).expect("age limit");
    let no_ids = HashSet::new();
    let cohort: Vec<Member> = rows.iter().map(make_member).collect();
    let periods: HashMap<String, Vec<ResidenceSpell>> = match spells {
        Some(spells) => spells.clone(),
        None => cohort
            .iter()
            .filter_map(|m| {
                m.emigration_date().map(|e| {
                    (
                        m.id().to_string(),
                        vec![ResidenceSpell {
                            from: None,
                            to: Some(e),
                        }],
                    )
                })
            })
            .collect(),
    };
    let resident = |record: &Member| periods.get(record.id()).map(Vec::as_slice);

    let pass1 = MatchJob::new_transition(&cohort, cfg.seed)
        .with_ratio(ratio)
        .with_ratio_fallback(fallback.clone())
        .with_birth_window(birth_window)
        .with_age_limit(age_limit)
        .with_exact_matches(["sex", "municipality"])
        .with_alive_check()
        .with_resident_spells(resident)
        .with_constraints(calipers())
        .with_risk_set_policy(PassPolicy {
            matched_case_ids: &no_ids,
            used_control_ids: &no_ids,
        })
        .run()
        .expect("a valid job");
    let matched: HashSet<&str> = pass1.pairs.iter().map(|p| p.case_id.as_str()).collect();
    let used: HashSet<&str> = pass1.pairs.iter().map(|p| p.control_id.as_str()).collect();
    let mut pairs = matched_rows(&pass1, "municipality");
    let mut passes = [(pass1.matched_cases, pass1.unmatched_cases), (0, 0)];
    if pass1.unmatched_cases > 0 && !cohort.is_empty() {
        let pass2 = MatchJob::new_transition(&cohort, cfg.seed.wrapping_add(1))
            .with_ratio(ratio)
            .with_ratio_fallback(fallback)
            .with_birth_window(birth_window)
            .with_age_limit(age_limit)
            .with_exact_matches(["sex", "region"])
            .with_alive_check()
            .with_resident_spells(resident)
            .with_constraints(calipers())
            .with_risk_set_policy(PassPolicy {
                matched_case_ids: &matched,
                used_control_ids: &used,
            })
            .run()
            .expect("a valid job");
        passes[1] = (pass2.matched_cases, pass2.unmatched_cases);
        pairs.extend(matched_rows(&pass2, "region"));
    }

    let matched_pairs: Vec<MatchedPair> = pairs
        .iter()
        .map(|row| MatchedPair::new(row.exposed_id.clone(), row.control_id.clone()))
        .collect();
    let method = MatchWeightMethod::AnchorUnitCandidateFractional;
    let unit_weights = match_weights_from_pairs(&matched_pairs, method).to_table();
    let pair_weights = pair_weights_from_pairs(&matched_pairs, method).to_table();
    let role = |role: UnitRole| {
        unit_weights
            .rows
            .iter()
            .filter(move |row| row.role == role)
            .map(|row| row.match_weight)
    };
    let effective_sample_sizes = [
        effective_sample_size(&role(UnitRole::Anchor).collect::<Vec<_>>()).expect("weights"),
        effective_sample_size(&role(UnitRole::Candidate).collect::<Vec<_>>()).expect("weights"),
        effective_sample_size(
            &unit_weights
                .rows
                .iter()
                .map(|row| row.analysis_weight)
                .collect::<Vec<_>>(),
        )
        .expect("weights"),
    ];
    let unit_weight_total = unit_weights.rows.iter().map(|row| row.match_weight).sum();
    MatchOutput {
        pairs,
        passes,
        unit_weights,
        pair_weights,
        effective_sample_sizes,
        unit_weight_total,
    }
}

/// A birth cohort over 16 years with about 9% cases diagnosed in their first
/// six years, a few other diagnoses, deaths, emigrations and comparator
/// eligibility ends, parents' birth days with some unknown, and residence
/// spells with a gap for some.
fn cohort(n: usize, seed: u64) -> (Vec<CohortRow>, HashMap<String, Vec<ResidenceSpell>>) {
    let mut stream = SplitMix64::new(seed);
    let base = NaiveDate::from_ymd_opt(2000, 1, 1).expect("valid");
    let after =
        |date: NaiveDate, days: u64| date.checked_add_days(Days::new(days)).expect("in range");
    let mut spells = HashMap::new();
    let rows = (0..n)
        .map(|i| {
            let person_id = format!(
                "{:010}",
                (i as u64).wrapping_mul(2_654_435_761) % 9_999_999_967
            );
            let birth_date = after(base, stream.below(5_800));
            let municipality = stream.below(14);
            let exposed = stream.below(1_000) < 90;
            let other = stream.below(1_000) < 40;
            let first_scd_date =
                (exposed || other).then(|| after(birth_date, 200 + stream.below(1_900)));
            let death_date =
                (stream.below(1_000) < 15).then(|| after(birth_date, 300 + stream.below(4_000)));
            let emigration_date =
                (stream.below(1_000) < 25).then(|| after(birth_date, 300 + stream.below(4_000)));
            let comparator_eligible_until =
                (stream.below(1_000) < 30).then(|| after(birth_date, 300 + stream.below(4_000)));
            let mother = 9_000.0 + (stream.below(6_000) as f64);
            let father = mother - 800.0 + (stream.below(2_400) as f64);
            let mut numerics = Vec::new();
            if stream.below(100) >= 2 {
                numerics.push(("mother_birth_day".to_string(), mother));
            }
            if stream.below(100) >= 8 {
                numerics.push(("father_birth_day".to_string(), father));
            }
            if stream.below(100) < 6 {
                let leave = after(birth_date, 100 + stream.below(2_000));
                let back = after(leave, 30 + stream.below(700));
                spells.insert(
                    person_id.clone(),
                    vec![
                        ResidenceSpell {
                            from: None,
                            to: Some(leave),
                        },
                        ResidenceSpell {
                            from: Some(back),
                            to: None,
                        },
                    ],
                );
            }
            CohortRow {
                person_id,
                birth_date,
                sex: if stream.below(2) == 0 { "1" } else { "2" }.to_string(),
                municipality: format!("{:03}", 101 + municipality),
                region: format!("R{}", municipality % 5),
                first_scd_date,
                death_date,
                emigration_date,
                exposed,
                comparator_eligible_until,
                numerics,
            }
        })
        .collect();
    (rows, spells)
}

struct Hash(Fingerprint);

impl Hash {
    fn new() -> Self {
        Self(Fingerprint::new())
    }

    fn word(&mut self, value: u64) {
        self.0 = std::mem::take(&mut self.0).u64(value);
    }

    fn text(&mut self, text: &str) {
        self.word(text.len() as u64);
        for byte in text.bytes() {
            self.word(u64::from(byte));
        }
    }

    fn o(&mut self, value: Option<f64>) {
        let value = value.filter(|v| !v.is_nan());
        self.0 = std::mem::take(&mut self.0).option_f64(value);
    }

    fn f(&mut self, value: f64) {
        self.o(Some(value));
    }

    fn finish(&self) -> u64 {
        self.0.finish()
    }
}

fn digest(output: &MatchOutput) -> [u64; 3] {
    let mut pairs = Hash::new();
    for (matched, unmatched) in output.passes {
        pairs.word(matched as u64);
        pairs.word(unmatched as u64);
    }
    for row in &output.pairs {
        pairs.text(&row.exposed_id);
        pairs.text(&row.control_id);
        pairs.text(row.match_type);
        pairs.word(u64::from(row.match_ratio.unsigned_abs()));
    }
    let mut units = Hash::new();
    for row in &output.unit_weights.rows {
        units.text(&row.unit_id);
        units.word(u64::from(row.role == UnitRole::Anchor));
        units.f(row.match_weight);
        units.o(row.sampling_weight);
        units.f(row.analysis_weight);
    }
    for row in &output.pair_weights.rows {
        units.text(&row.anchor_id);
        units.text(&row.candidate_id);
        units.f(row.pair_weight);
    }
    let mut diagnostics = Hash::new();
    output
        .effective_sample_sizes
        .iter()
        .for_each(|&ess| diagnostics.o(ess));
    diagnostics.f(output.unit_weight_total);
    diagnostics.word(output.pairs.len() as u64);
    [pairs.finish(), units.finish(), diagnostics.finish()]
}

fn cases() -> Vec<(&'static str, u64)> {
    let (rows, spells) = cohort(8_000, 20_261_126);
    assert!(
        rows.iter().any(|row| !row
            .numerics
            .iter()
            .any(|(name, _)| name == "father_birth_day")),
        "no unknown parent"
    );
    let study_one = MatchConfig {
        seed: 20_260_819,
        ratio: 4,
        fallback_ratios: vec![3, 2, 1],
        birth_window_days: 30,
        age_limit_years: 6,
        calipers: vec![
            ("mother_birth_day".to_string(), 1_826.0),
            ("father_birth_day".to_string(), 1_826.0),
        ],
        caliper_missing: MissingPolicy::MatchMissing,
    };
    let study_two = MatchConfig {
        seed: 42,
        ratio: 4,
        fallback_ratios: vec![3, 2],
        birth_window_days: 30,
        age_limit_years: 6,
        calipers: Vec::new(),
        caliper_missing: MissingPolicy::Refuse,
    };
    let mut out = Vec::new();
    let one = run_matching(&rows, Some(&spells), &study_one);
    let two = run_matching(&rows, None, &study_two);
    for (output, names) in [
        (
            &one,
            [
                "study_one/pairs",
                "study_one/weights",
                "study_one/diagnostics",
            ],
        ),
        (
            &two,
            [
                "study_two/pairs",
                "study_two/weights",
                "study_two/diagnostics",
            ],
        ),
    ] {
        assert!(output.passes[1].0 > 0, "the region pass matched no case");
        for (name, hash) in names.into_iter().zip(digest(output)) {
            out.push((name, hash));
        }
    }
    out
}

const GOLDEN: &[(&str, u64)] = &[
    ("study_one/pairs", 0x1459_5733_1c01_5fe0),
    ("study_one/weights", 0x82de_0844_97ab_1e01),
    ("study_one/diagnostics", 0xc32e_b716_d188_d621),
    ("study_two/pairs", 0x9eb0_036c_8379_99bb),
    ("study_two/weights", 0x1697_4375_9811_e9d8),
    ("study_two/diagnostics", 0x49ca_23ac_d671_a092),
];

#[test]
fn study_paths_are_bit_identical_to_the_recorded_hashes() {
    let current = cases();
    if std::env::var_os("STUDY_PATH_PRINT").is_some() {
        for (name, hash) in &current {
            println!(
                "    (\"{name}\", 0x{:04x}_{:04x}_{:04x}_{:04x}),",
                hash >> 48,
                (hash >> 32) & 0xffff,
                (hash >> 16) & 0xffff,
                hash & 0xffff
            );
        }
    }
    assert_eq!(current.as_slice(), GOLDEN);
}
