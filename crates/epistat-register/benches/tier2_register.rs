//! Wall time of the Lexis splitting, interval algebra, residence spells and
//! pedigree methods, set against R (`bench/tier2/bench.R`). The driver and the
//! environment are those of `bench/tier1/common.rs`; `bench/run.sh` is
//! the driver.
//!
//! `N` is the number of persons. Methods: `lexis` (follow-up cut at calendar
//! years and at ages 5, 18, 40 and 65 on the 365.25-day rule, against
//! `popEpi::splitMulti`), `intervals` (intersection, difference and clipping
//! of two interval sets, against `data.table`), `residence` (spells from about
//! four events a person, and residence on a day, against `data.table`) and
//! `pedigree` (sibling and cousin pairs and family clusters, against
//! `data.table` counts).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "a benchmark program: sizes, codes and days are small whole numbers"
)]

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_register::AgeRule;
use epistat_register::intervals::{SpanSet, clip, intersect, subtract};
use epistat_register::lexis::{Cuts, split_at};
use epistat_register::pedigree::{CousinKinds, Pedigree, SiblingKind, SiblingKinds};
use epistat_register::residence::{
    Direction, Events, SameDay, first_departure_on_or_after, residence_spells, resident_on,
};

const SEED: u64 = 20_261_223;
const AGES: [u32; 4] = [5, 18, 40, 65];

fn generate_lexis(n: usize) {
    let mut rng = Rng(SEED);
    let mut out = Writer::create(&common::data_path("lx", n), "birth,start,end");
    for _ in 0..n {
        let birth = -10_000 + rng.below(25_000) as i64;
        let start = birth + rng.below(20_000) as i64;
        let end = start + 1 + rng.below(9_000) as i64;
        out.row(&[birth as f64, start as f64, end as f64]);
    }
}

/// Rows `key,set,start,end`: each key has up to three disjoint intervals in
/// each of two sets, in day order.
fn generate_intervals(n: usize) {
    let mut rng = Rng(SEED + 1);
    let mut out = Writer::create(&common::data_path("iv", n), "key,set,start,end");
    for key in 0..n {
        for set in 0..2 {
            let mut day = rng.below(100) as i64;
            for _ in 0..rng.below(4) {
                let len = 1 + rng.below(60) as i64;
                out.row(&[key as f64, f64::from(set), day as f64, (day + len) as f64]);
                day += len + rng.below(40) as i64;
            }
        }
    }
}

fn generate_residence(n: usize) {
    let mut rng = Rng(SEED + 2);
    let mut out = Writer::create(&common::data_path("rs", n), "person,day,out");
    for p in 0..n {
        let mut days: Vec<i64> = (0..rng.below(8)).map(|_| rng.below(3_000) as i64).collect();
        days.sort_unstable();
        for d in days {
            out.row(&[p as f64, d as f64, f64::from(u8::from(rng.bernoulli(0.5)))]);
        }
    }
}

/// Persons in generations of about equal size. Each person draws a mother and a
/// father from a small pool of couples of the previous generation, or has an
/// unknown parent with probability 0.1.
fn generate_pedigree(n: usize) {
    let mut rng = Rng(SEED + 3);
    let mut out = Writer::create(&common::data_path("pd", n), "id,mother,father");
    let generations = 5;
    let per = (n / generations).max(1);
    let mut start_of_previous = 0_usize;
    for g in 0..generations {
        let first = g * per;
        let last = if g + 1 == generations { n } else { first + per };
        let previous = start_of_previous..first;
        // Couples of the previous generation: roughly one for every 3 children.
        let couples: Vec<(usize, usize)> = (0..((last - first) / 3).max(1))
            .filter(|_| !previous.is_empty())
            .map(|_| {
                (
                    previous.start + rng.below(previous.len() as u64) as usize,
                    previous.start + rng.below(previous.len() as u64) as usize,
                )
            })
            .filter(|(m, f)| m != f)
            .collect();
        for id in first..last {
            let (m, f) = if couples.is_empty() {
                (f64::NAN, f64::NAN)
            } else {
                let (m, f) = couples[rng.below(couples.len() as u64) as usize];
                (
                    if rng.bernoulli(0.1) {
                        f64::NAN
                    } else {
                        m as f64
                    },
                    if rng.bernoulli(0.1) {
                        f64::NAN
                    } else {
                        f as f64
                    },
                )
            };
            out.row(&[id as f64, m, f]);
        }
        start_of_previous = first;
    }
}

fn main() {
    let (method, n) = (common::method(), common::size());
    match method.as_str() {
        "gen_lx" => return generate_lexis(n),
        "gen_iv" => return generate_intervals(n),
        "gen_rs" => return generate_residence(n),
        "gen_pd" => return generate_pedigree(n),
        _ => {}
    }
    let mut lines = Lines::default();
    let times = match method.as_str() {
        "lexis" => {
            let t = Table::read(&common::data_path("lx", n));
            let (birth, start, end): (Vec<i64>, Vec<i64>, Vec<i64>) =
                (t.ints("birth"), t.ints("start"), t.ints("end"));
            let (times, split) = common::timed(common::reps(), || {
                split_at(
                    &start,
                    &end,
                    &[
                        Cuts::CalendarYears,
                        Cuts::AgeYears {
                            birth: &birth,
                            ages: &AGES,
                            rule: AgeRule::DaysOver365_25,
                        },
                    ],
                )
                .expect("split")
            });
            lines.push("pieces", split.start.len() as f64);
            lines.push(
                "days",
                split
                    .start
                    .iter()
                    .zip(&split.end)
                    .map(|(s, e)| (e - s) as f64)
                    .sum(),
            );
            times
        }
        "intervals" => {
            let t = Table::read(&common::data_path("iv", n));
            let (key, set): (Vec<usize>, Vec<u8>) = (t.ints("key"), t.ints("set"));
            let (start, end): (Vec<i64>, Vec<i64>) = (t.ints("start"), t.ints("end"));
            let build = |which: u8| {
                let mut s = vec![];
                let mut e = vec![];
                let mut offsets = vec![0_usize; n + 1];
                for r in 0..key.len() {
                    if set[r] == which {
                        s.push(start[r]);
                        e.push(end[r]);
                        offsets[key[r] + 1] += 1;
                    }
                }
                for k in 0..n {
                    offsets[k + 1] += offsets[k];
                }
                SpanSet::new(s, e, offsets).expect("set")
            };
            let (a, b) = (build(0), build(1));
            let (lo, hi) = (vec![50_i64; n], vec![250_i64; n]);
            let (times, (i, s, c)) = common::timed(common::reps(), || {
                (
                    intersect(&a, &b).expect("intersect"),
                    subtract(&a, &b).expect("subtract"),
                    clip(&a, &lo, &hi).expect("clip"),
                )
            });
            lines.push("a_days", a.total_days() as f64);
            lines.push("intersect_days", i.total_days() as f64);
            lines.push("subtract_days", s.total_days() as f64);
            lines.push("clip_days", c.total_days() as f64);
            lines.push("intersect_pieces", i.len() as f64);
            lines.push("subtract_pieces", s.len() as f64);
            times
        }
        "residence" => {
            let t = Table::read(&common::data_path("rs", n));
            let person: Vec<usize> = t.ints("person");
            let day: Vec<i64> = t.ints("day");
            let direction: Vec<Direction> = t
                .flags("out")
                .iter()
                .map(|&o| if o { Direction::Out } else { Direction::In })
                .collect();
            let persons = person.last().map_or(0, |&p| p + 1);
            let mut offsets = vec![0_usize; persons + 1];
            for &p in &person {
                offsets[p + 1] += 1;
            }
            for p in 0..persons {
                offsets[p + 1] += offsets[p];
            }
            let query_person: Vec<u32> = (0..persons as u32).collect();
            let query_day = vec![1_500_i64; persons];
            let (times, (spells, resident, departure)) = common::timed(common::reps(), || {
                let e = Events {
                    person_offsets: &offsets,
                    day: &day,
                    direction: &direction,
                };
                let spells = residence_spells(&e, SameDay::Cancel).expect("spells");
                let resident = resident_on(&spells, &query_person, &query_day).expect("resident");
                let departure =
                    first_departure_on_or_after(&e, SameDay::Cancel, &query_person, &query_day)
                        .expect("departure");
                (spells, resident, departure)
            });
            lines.push("spells", spells.from.len() as f64);
            lines.push(
                "from_sum",
                spells.from.iter().flatten().map(|&d| d as f64).sum(),
            );
            lines.push(
                "to_sum",
                spells.to.iter().flatten().map(|&d| d as f64).sum(),
            );
            lines.push("resident", resident.iter().filter(|&&r| r).count() as f64);
            lines.push(
                "departures",
                departure.iter().filter(|d| d.is_some()).count() as f64,
            );
            times
        }
        "pedigree" => {
            let t = Table::read(&common::data_path("pd", n));
            let id: Vec<u64> = t.ints("id");
            let opt = |name: &str| -> Vec<Option<u64>> {
                t.col(name)
                    .iter()
                    .map(|&v| (!v.is_nan()).then_some(v as u64))
                    .collect()
            };
            let (mother, father) = (opt("mother"), opt("father"));
            let (times, (siblings, cousins, clusters, depth)) =
                common::timed(common::reps(), || {
                    let ped = Pedigree::new(&id, &mother, &father).expect("pedigree");
                    let siblings = ped
                        .sibling_pairs(SiblingKinds::ALL, 1_000)
                        .expect("siblings");
                    let cousins = ped
                        .cousin_pairs(1, CousinKinds::ALL, 10_000)
                        .expect("cousins");
                    let clusters = ped.family_clusters(FamilyLinks::SharedParents).expect("clusters");
                    (siblings, cousins, clusters, ped.depth())
                });
            let by = |k: SiblingKind| siblings.kind.iter().filter(|&&x| x == k).count() as f64;
            lines.push("full", by(SiblingKind::Full));
            lines.push(
                "same_mother",
                by(SiblingKind::Full) + by(SiblingKind::MaternalHalf) + by(SiblingKind::MotherOnly),
            );
            lines.push(
                "same_father",
                by(SiblingKind::Full) + by(SiblingKind::PaternalHalf) + by(SiblingKind::FatherOnly),
            );
            lines.push("info_cousin_pairs", cousins.a.len() as f64);
            lines.push(
                "clusters",
                f64::from(clusters.iter().copied().max().map_or(0, |m| m + 1)),
            );
            lines.push(
                "max_depth",
                f64::from(depth.iter().copied().max().unwrap_or(0)),
            );
            times
        }
        other => panic!("unknown method {other}"),
    };
    common::finish(&method, n, &times, &lines);
}
