//! Residence spells, residence on a day and the first departure against an
//! independent implementation of the stated rules in `data.table`
//! (`tests/data/residence_reference.R`). No package offers these, so the
//! reference is a second implementation. The results are integers, so
//! agreement is exact.

use epistat_register::residence::{
    Direction, Events, SameDay, first_departure_on_or_after, residence_spells, resident_on,
};
use epistat_testkit::{Fingerprint, assert_same_on_threads, load_json};
use serde::Deserialize;

#[derive(Deserialize)]
struct EventsRef {
    person: Vec<usize>,
    day: Vec<i64>,
    out: Vec<bool>,
}

#[derive(Deserialize)]
struct SpellsRef {
    person: Vec<usize>,
    from: Vec<Option<i64>>,
    to: Vec<Option<i64>>,
}

#[derive(Deserialize)]
struct Queries {
    person: Vec<u32>,
    day: Vec<i64>,
    resident: Vec<bool>,
    departure: Vec<Option<i64>>,
}

#[derive(Deserialize)]
struct Reference {
    persons: usize,
    events: EventsRef,
    spells: SpellsRef,
    queries: Queries,
}

#[test]
fn spells_residence_and_departures_agree_with_r() {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/residence_reference.json"
    ));
    let mut offsets = vec![0_usize; r.persons + 1];
    for &p in &r.events.person {
        offsets[p + 1] += 1;
    }
    for p in 0..r.persons {
        offsets[p + 1] += offsets[p];
    }
    let direction: Vec<Direction> = r
        .events
        .out
        .iter()
        .map(|&o| if o { Direction::Out } else { Direction::In })
        .collect();
    let e = Events {
        person_offsets: &offsets,
        day: &r.events.day,
        direction: &direction,
    };
    let s = residence_spells(&e, SameDay::Cancel).unwrap();

    let mut person = Vec::new();
    for p in 0..r.persons {
        person.extend(std::iter::repeat_n(
            p,
            s.person_offsets[p + 1] - s.person_offsets[p],
        ));
    }
    assert_eq!(person, r.spells.person);
    assert_eq!(s.from, r.spells.from);
    assert_eq!(s.to, r.spells.to);

    let q = &r.queries;
    let resident = resident_on(&s, &q.person, &q.day).unwrap();
    assert_eq!(resident, q.resident);
    let departure = first_departure_on_or_after(&e, SameDay::Cancel, &q.person, &q.day).unwrap();
    assert_eq!(departure, q.departure);
    // Both outcomes occur, or agreement says little.
    assert!(resident.iter().filter(|&&x| x).count() > 500);
    assert!(resident.iter().filter(|&&x| !x).count() > 500);
    assert!(s.from.iter().any(Option::is_none) && s.to.iter().any(Option::is_none));
    assert_same_on_threads(|| {
        residence_spells(&e, SameDay::Cancel)
            .unwrap()
            .to
            .iter()
            .map(|t| t.map_or(f64::NAN, |d| d as f64))
            .collect()
    });
    let mut fp = Fingerprint::new();
    for i in 0..s.from.len() {
        fp = fp
            .u64(s.from[i].map_or(u64::MAX, i64::cast_unsigned))
            .u64(s.to[i].map_or(u64::MAX, i64::cast_unsigned));
    }
    assert_eq!(fp.finish(), 4_660_581_780_602_145_558, "golden hash");
}
