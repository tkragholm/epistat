//! The calls Studies I and II make into this crate, hashed to the bit.
//!
//! Each case makes the calls of PHD's `crates/kernels` (`cpi.rs`, `spans.rs`,
//! `relations.rs`, `person_time.rs`) and `crates/native` with the arguments
//! they pass: Danish CPI adjustment factors to a price year, the merge of
//! sorted spans with a grace period, connected components over relation keys,
//! and the civil-date arithmetic and calendar-year overlaps of person time. A
//! change that moves a hash moves a protected study, so a port of these calls
//! to new paths must reproduce every hash.
//!
//! The hashes were recorded on aarch64 macOS and are asserted only there.
//! Elsewhere the test computes them and, when `EPISTAT_HASH_OUT` is set,
//! appends them to that file. Print them with
//! `STUDY_PATH_PRINT=1 cargo test --test study_path --features denmark -- --nocapture`.
#![cfg(feature = "denmark")]

use epistat_register::person_time::{
    calendar_year_overlaps, civil_from_days, days_from_civil, year_from_days,
};
use epistat_register::{CpiTable, DENMARK_ANNUAL_CPI, connected_components, merge_sorted_spans};
use epistat_testkit::{Fingerprint, SplitMix64};

struct Hash(Fingerprint);

impl Hash {
    fn new() -> Self {
        Self(Fingerprint::new())
    }

    fn word(&mut self, value: u64) {
        self.0 = std::mem::take(&mut self.0).u64(value);
    }

    fn int(&mut self, value: i64) {
        self.word(value.cast_unsigned());
    }

    fn f(&mut self, value: f64) {
        let value = (!value.is_nan()).then_some(value);
        self.0 = std::mem::take(&mut self.0).option_f64(value);
    }

    fn finish(&self) -> u64 {
        self.0.finish()
    }
}

fn cases() -> Vec<(&'static str, u64)> {
    let mut out = Vec::new();

    // native::cpi::{annual_cpi_index, cpi_adjustment_factors}.
    let table = CpiTable::new(DENMARK_ANNUAL_CPI.iter().copied());
    let mut h = Hash::new();
    let rows: Vec<(i32, f64)> = table.rows().collect();
    for &(year, index) in &rows {
        h.int(i64::from(year));
        h.f(index);
    }
    for &(reference, _) in &rows {
        for &(year, _) in &rows {
            h.f(table.adjustment_factor(year, reference).expect("covered"));
        }
    }
    out.push(("cpi", h.finish()));

    // kernels::spans::merge_sorted_spans: drug and residence spans per person,
    // sorted by person, start and end, with grace periods of 0 and 30 days.
    let mut stream = SplitMix64::new(11);
    let (mut new_key, mut start, mut end) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..800 {
        let rows = 1 + stream.below(6);
        let mut day = 12_000 + i64::try_from(stream.below(3_000)).expect("small");
        for row in 0..rows {
            day += i64::try_from(stream.below(90)).expect("small");
            let length = i64::try_from(stream.below(120)).expect("small");
            new_key.push(row == 0);
            start.push(day);
            end.push(day + length);
        }
    }
    let mut h = Hash::new();
    for grace in [0, 30] {
        let merged = merge_sorted_spans(&new_key, &start, &end, grace).expect("sorted");
        h.word(merged.first_row.len() as u64);
        merged
            .first_row
            .iter()
            .for_each(|&row| h.word(u64::from(row)));
        merged
            .last_row
            .iter()
            .for_each(|&row| h.word(u64::from(row)));
    }
    out.push(("merge_sorted_spans", h.finish()));

    // kernels::relations::connected_components: families from a child's
    // mother and father, either unknown.
    let mut stream = SplitMix64::new(12);
    let n = 3_000;
    let mut parent = |share_missing: u64| -> Vec<Option<u32>> {
        (0..n)
            .map(|_| {
                (stream.below(100) >= share_missing)
                    .then(|| u32::try_from(stream.below(1_400)).expect("small"))
            })
            .collect()
    };
    let (mother, father) = (parent(3), parent(12));
    let components = connected_components(&[&mother, &father]).expect("components");
    let mut h = Hash::new();
    for &c in &components {
        h.word(u64::from(c));
    }
    out.push(("connected_components", h.finish()));

    // kernels::person_time and native::person_time::calendar_year_overlaps.
    let mut h = Hash::new();
    for day in (-800_000_i64..800_000).step_by(37) {
        let (y, m, d) = civil_from_days(day);
        h.int(y);
        h.int(m);
        h.int(d);
        h.int(year_from_days(day));
        h.int(days_from_civil(y, m, d));
    }
    let mut stream = SplitMix64::new(13);
    let mut overlaps = Vec::new();
    for _ in 0..2_000 {
        let start = 10_000 + i64::try_from(stream.below(9_000)).expect("small");
        let end = start + i64::try_from(stream.below(2_500)).expect("small");
        let floor = (stream.below(3) == 0).then(|| start + 200);
        overlaps.clear();
        calendar_year_overlaps(start, end, floor, stream.below(2) == 0, &mut overlaps);
        for overlap in &overlaps {
            h.int(overlap.calendar_year);
            h.int(overlap.overlap_days);
            h.int(overlap.calendar_days);
        }
    }
    out.push(("person_time", h.finish()));
    out
}

const GOLDEN: &[(&str, u64)] = &[
    ("cpi", 0xc694_e9d6_569d_1181),
    ("merge_sorted_spans", 0xa063_d5e5_f0d4_6a68),
    ("connected_components", 0x94b7_33af_24ef_e577),
    ("person_time", 0x50c8_243d_7b72_a56d),
];

#[test]
fn study_paths_are_bit_identical_to_the_recorded_hashes() {
    let current = cases();
    for (name, hash) in &current {
        epistat_testkit::hash_out(name, *hash);
    }
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
    if epistat_testkit::RECORDED_PLATFORM {
        assert_eq!(current.as_slice(), GOLDEN);
    }
}
