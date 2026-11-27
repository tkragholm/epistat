//! Wall time of the event definitions, the comorbidity score and the exposure
//! periods, set against base R with `data.table`, the R package `comorbidity`
//! and `AdhereR::compute.treatment.episodes` (`bench/tier1/bench.R`). The
//! driver and the environment are described in `bench/tier1/common.rs`.
//!
//! `N` is the number of rows (events, diagnoses or dispensings), about six per
//! person. Methods: `events` (two definitions: a first event with a
//! look-back of 365 days, and a case confirmed by a second event within 30
//! days), `comorbidity` (the Charlson index of the diagnoses of each person,
//! Quan's ICD-10 codes, the hierarchy and the weights of 2011) and `exposure`
//! (periods from dispensings with a grace of 30 days).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "a benchmark program: sizes, codes and days are small whole numbers"
)]

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_register::codes::Dictionary;
use epistat_register::comorbidity::{
    ComorbidityMap, IndexTable, charlson_quan2011, flags_at_index,
};
use epistat_register::events::{
    CaseDate, ConfirmBy, EventRows, Lookback, PersonWindows, PriorStatus, Rule, evaluate,
};
use epistat_register::exposure::{
    Assumption, Dispensings, DoseRule, Stock, Supply, exposure_periods,
};

const SEED: u64 = 20_261_205;
const ROWS_PER_PERSON: u64 = 11;
/// Diagnosis codes: some in the Charlson lists, some in none.
const CODES: [&str; 40] = [
    "I21", "I219", "I50", "I500", "I70", "I700", "G45", "G459", "I639", "F00", "F03", "G30", "J44",
    "J449", "M05", "M059", "K25", "K259", "B18", "K74", "E109", "E119", "E100", "G81", "G819",
    "N18", "N189", "C50", "C509", "C61", "C34", "C80", "B20", "A09", "R10", "Z00", "M54", "J06",
    "N39", "H25",
];
/// A day number beyond every row.
const INDEX_DAY: i64 = 1_000_000;

/// The rows of one person, as the number of rows and a sorted list of days in
/// `lo..hi`.
fn person_days(rng: &mut Rng, lo: i64, span: u64, left: usize) -> Vec<i64> {
    let k = (1 + rng.below(ROWS_PER_PERSON) as usize).min(left);
    let mut days: Vec<i64> = (0..k).map(|_| lo + rng.below(span) as i64).collect();
    days.sort_unstable();
    days
}

fn generate_events(n: usize) {
    let mut rng = Rng(SEED);
    let mut rows = Writer::create(&common::data_path("evt", n), "id,day,mask");
    let mut persons = Writer::create(&common::data_path("evtp", n), "entry,exit,hist");
    let (mut id, mut written) = (0.0, 0);
    while written < n {
        let entry = 10_000 + rng.below(3000) as i64;
        let exit = entry + 365 + rng.below(3300) as i64;
        persons.row(&[entry as f64, exit as f64, 0.0]);
        let days = person_days(
            &mut rng,
            entry - 1000,
            (exit - entry + 1100) as u64,
            n - written,
        );
        for &d in &days {
            let mask = u8::from(rng.bernoulli(0.5)) | (u8::from(rng.bernoulli(0.6)) << 1);
            rows.row(&[id, d as f64, f64::from(mask)]);
        }
        written += days.len();
        id += 1.0;
    }
}

fn generate_comorbidity(n: usize) {
    let mut rng = Rng(SEED + 1);
    let mut rows = Writer::create(&common::data_path("cmb", n), "id,day,code");
    let (mut id, mut written) = (0, 0);
    while written < n {
        let days = person_days(&mut rng, 10_000, 4000, n - written);
        for &d in &days {
            let code = CODES[rng.below(CODES.len() as u64) as usize];
            rows.raw(&format!("{id},{d},{code}"));
        }
        written += days.len();
        id += 1;
    }
}

fn generate_exposure(n: usize) {
    let mut rng = Rng(SEED + 2);
    let mut rows = Writer::create(&common::data_path("exp", n), "id,day,quantity");
    let (mut id, mut written) = (0.0, 0);
    while written < n {
        let k = (1 + rng.below(ROWS_PER_PERSON) as usize).min(n - written);
        let mut day = 10_000 + rng.below(2000) as i64;
        for _ in 0..k {
            let duration = [30, 60, 90][rng.below(3) as usize];
            rows.row(&[id, day as f64, 2.0 * f64::from(duration)]);
            let gap = [-20, 0, 10, 29, 31, 60, 120][rng.below(7) as usize];
            day += (i64::from(duration) + gap).max(1);
        }
        written += k;
        id += 1.0;
    }
}

/// The offsets of the rows of each person, from the person of each row.
fn offsets(person: &[u32]) -> Vec<usize> {
    let persons = person.last().map_or(0, |&p| p as usize + 1);
    let mut out = vec![0_usize; persons + 1];
    for &p in person {
        out[p as usize + 1] += 1;
    }
    for p in 0..persons {
        out[p + 1] += out[p];
    }
    out
}

fn main() {
    let (method, n) = (common::method(), common::size());
    match method.as_str() {
        "gen_evt" => return generate_events(n),
        "gen_cmb" => return generate_comorbidity(n),
        "gen_exp" => return generate_exposure(n),
        _ => {}
    }
    let mut lines = Lines::default();
    let times = match method.as_str() {
        "events" => {
            let rows = Table::read(&common::data_path("evt", n));
            let persons = Table::read(&common::data_path("evtp", n));
            let offsets = offsets(&rows.ints::<u32>("id"));
            let day: Vec<i64> = rows.ints("day");
            let mask: Vec<u64> = rows.ints("mask");
            let (entry, exit, hist): (Vec<i64>, Vec<i64>, Vec<i64>) = (
                persons.ints("entry"),
                persons.ints("exit"),
                persons.ints("hist"),
            );
            let event_rows = EventRows {
                person_offsets: &offsets,
                day: &day,
                mask: &mask,
                contact: None,
            };
            let windows = PersonWindows {
                entry: &entry,
                exit: &exit,
                history_start: &hist,
            };
            let rules = [
                Rule::single(Lookback::Days(365)),
                Rule {
                    lookback: Lookback::None,
                    require: 2,
                    confirm_within: Some(30),
                    confirm_by: ConfirmBy::Dates,
                    date_of_case: CaseDate::FirstEvent,
                    entry_inclusive: false,
                },
            ];
            let (times, out) = common::timed(common::reps(), || {
                evaluate(&event_rows, &windows, &rules).expect("events")
            });
            for (k, outcomes) in out.iter().enumerate() {
                let sum = |f: &dyn Fn(&epistat_register::events::Outcome) -> Option<i64>| {
                    let v: Vec<i64> = outcomes.iter().filter_map(f).collect();
                    (v.len() as f64, v.iter().sum::<i64>() as f64)
                };
                let (nf, sf) = sum(&|o| o.first_in_window);
                let (nc, sc) = sum(&|o| o.case_day);
                lines.push(&format!("rule{k}_nfirst"), nf);
                lines.push(&format!("rule{k}_sumfirst"), sf);
                lines.push(&format!("rule{k}_ncase"), nc);
                lines.push(&format!("rule{k}_sumcase"), sc);
                lines.push(
                    &format!("rule{k}_nevents"),
                    outcomes
                        .iter()
                        .map(|o| f64::from(o.n_events_in_window))
                        .sum(),
                );
                if k == 0 {
                    lines.push(
                        "rule0_nprevalent",
                        outcomes
                            .iter()
                            .filter(|o| o.prior == PriorStatus::Prevalent)
                            .count() as f64,
                    );
                }
            }
            times
        }
        "comorbidity" => {
            let text = common::read_text_rows(&common::data_path("cmb", n));
            let person: Vec<u32> = text.iter().map(|r| r[0].parse().expect("id")).collect();
            let day: Vec<i64> = text.iter().map(|r| r[1].parse().expect("day")).collect();
            let codes: Vec<&str> = text.iter().map(|r| r[2].as_str()).collect();
            let offsets = offsets(&person);
            let persons = offsets.len() - 1;
            let index_person: Vec<u32> = (0..persons as u32).collect();
            let index_day = vec![INDEX_DAY; persons];
            let history = vec![0_i64; persons];
            let map = ComorbidityMap::charlson_quan_icd10();
            let weights = charlson_quan2011();
            let (times, scores) = common::timed(common::reps(), || {
                let dict = Dictionary::new(&codes).expect("dictionary");
                let per_code = map.classify(dict.distinct()).masks;
                let mask = dict.gather(&per_code);
                let rows = EventRows {
                    person_offsets: &offsets,
                    day: &day,
                    mask: &mask,
                    contact: None,
                };
                let index = IndexTable {
                    person: &index_person,
                    day: &index_day,
                    history_start: &history,
                };
                let flags =
                    flags_at_index(&rows, &index, Lookback::Unlimited, true).expect("flags");
                flags
                    .iter()
                    .map(|f| {
                        f.score(&map, &weights, false)
                            .expect("score")
                            .expect("score")
                    })
                    .collect::<Vec<f64>>()
            });
            lines.push("persons", persons as f64);
            lines.push("score_sum", scores.iter().sum());
            lines.push("score_square_sum", scores.iter().map(|s| s * s).sum());
            lines.push(
                "n_nonzero",
                scores.iter().filter(|&&s| s != 0.0).count() as f64,
            );
            for (i, s) in scores.iter().take(8).enumerate() {
                lines.push(&format!("score{i}"), *s);
            }
            times
        }
        "exposure" => {
            let t = Table::read(&common::data_path("exp", n));
            let offsets = offsets(&t.ints::<u32>("id"));
            let day: Vec<i64> = t.ints("day");
            let group = vec![Some(0_u32); day.len()];
            let dispensings = Dispensings {
                person_offsets: &offsets,
                group: &group,
                day: &day,
                quantity: t.col("quantity"),
            };
            let assumptions = [Assumption {
                dose_default: 2.0,
                dose_min: 1.0,
                dose_max: 4.0,
                max_stock_days: f64::INFINITY,
                grace_days: 30.0,
            }];
            let supply = Supply {
                assumptions: &assumptions,
                rule: DoseRule::Default,
                stock: Stock::CarryForward,
                pauses: None,
            };
            let (times, periods) = common::timed(common::reps(), || {
                exposure_periods(&dispensings, &supply).expect("exposure")
            });
            lines.push("episodes", periods.len() as f64);
            lines.push("sum_start", periods.start().iter().map(|&s| s as f64).sum());
            lines.push("sum_end", periods.end().iter().sum());
            times
        }
        other => panic!("unknown method {other}"),
    };
    common::finish(&method, n, &times, &lines);
}
