//! The bits of every output of the Markov test on varied inputs.
//!
//! The recorded values are an FNV-1a digest of the `to_bits` of each output
//! field, with its length. They were taken from the function as it was before
//! the shared input type, and they must not change. To record again, run with
//! `GOLDEN_RECORD=1 -- --nocapture` and paste the lines.
//!
//! The inputs come from a seeded stream. They cover several qualifying states
//! and landmarks, tied exit times, delayed entry, sparse person codes, stays
//! in states that do not qualify, stays of one person that overlap, slots of
//! zero variance, no stays at risk, no replicates and the refusals.
//!
//! The bits depend on the platform's `ln` and `exp`, which come from its maths
//! library, so the test runs only on the target it was recorded on, aarch64
//! macOS. CI's macOS runner is that target. Thread-count independence, which
//! does not depend on the platform, is checked everywhere by the thread tests.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]
#![expect(
    deprecated,
    reason = "the deprecated functions are what this test pins"
)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "the inputs are a few hundred rows, far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::markov::{MarkovTest, Stays, markov_test};
use epistat_testkit::{Fingerprint, absent_all};

/// `(name, length, digest)` of each recorded field.
const EXPECTED: &[(&str, usize, u64)] = &[
    ("refuse/no states", 61, 0x0122_be26_44a9_c212),
    ("refuse/no landmark", 61, 0x0122_be26_44a9_c212),
    ("refuse/lengths", 66, 0xbd0e_59ea_1130_7534),
    ("refuse/nan start", 66, 0xa3f0_9656_4267_6707),
    ("refuse/infinite stop", 65, 0x0cc2_515b_02df_f979),
    ("dense q1/trace", 7, 0x5535_2787_a143_1629),
    ("dense q1/variance", 7, 0xd5c8_2aa0_83b2_46b3),
    ("dense q1/n_occupied", 7, 0x1c9d_1507_6382_07d3),
    ("dense q1/n_events_after", 7, 0x3b3e_c9ef_529b_d6d6),
    ("dense q1/statistic", 1, 0xc59c_c993_19d2_a7d7),
    ("dense q1/p_value", 1, 0xb6cf_8c3f_4ea2_8624),
    ("dense q2/trace", 14, 0xf4e1_6d0f_0836_b0ba),
    ("dense q2/variance", 14, 0xb71b_7d52_fc62_db7a),
    ("dense q2/n_occupied", 14, 0xf1cc_1f9f_b46e_5268),
    ("dense q2/n_events_after", 7, 0x3b3e_c9ef_529b_d6d6),
    ("dense q2/statistic", 2, 0x6267_2e62_f1c0_f111),
    ("dense q2/p_value", 2, 0x7014_0d3e_703c_613d),
    ("dense q3/trace", 21, 0xc6cd_a1c8_78fb_a89a),
    ("dense q3/variance", 21, 0x131e_1348_bf72_971a),
    ("dense q3/n_occupied", 21, 0x169c_f5c9_5bbb_8f28),
    ("dense q3/n_events_after", 7, 0x3b3e_c9ef_529b_d6d6),
    ("dense q3/statistic", 3, 0x924d_81c7_a86c_fd31),
    ("dense q3/p_value", 3, 0x08bb_9cce_154c_94e0),
    ("dense no seeds/trace", 14, 0xf4e1_6d0f_0836_b0ba),
    ("dense no seeds/variance", 14, 0xb71b_7d52_fc62_db7a),
    ("dense no seeds/n_occupied", 14, 0xf1cc_1f9f_b46e_5268),
    ("dense no seeds/n_events_after", 7, 0x3b3e_c9ef_529b_d6d6),
    ("dense no seeds/statistic", 2, 0x6267_2e62_f1c0_f111),
    ("dense no seeds/p_value", 2, 0x00a0_b04e_fa3c_16c9),
    ("dense one seed/trace", 14, 0xf4e1_6d0f_0836_b0ba),
    ("dense one seed/variance", 14, 0xb71b_7d52_fc62_db7a),
    ("dense one seed/n_occupied", 14, 0xf1cc_1f9f_b46e_5268),
    ("dense one seed/n_events_after", 7, 0x3b3e_c9ef_529b_d6d6),
    ("dense one seed/statistic", 2, 0x6267_2e62_f1c0_f111),
    ("dense one seed/p_value", 2, 0x271b_e7d4_e51a_6a45),
    ("dense odd grid/trace", 21, 0x1f5b_b8bf_5bc3_81e2),
    ("dense odd grid/variance", 21, 0xa37d_39bc_9362_f4dc),
    ("dense odd grid/n_occupied", 21, 0x6571_25eb_5b97_6eeb),
    ("dense odd grid/n_events_after", 7, 0xd573_a563_53b6_fead),
    ("dense odd grid/statistic", 3, 0x2ca8_cbfa_8b77_01d3),
    ("dense odd grid/p_value", 3, 0x3474_c7ea_8d7b_c558),
    ("dense one landmark/trace", 2, 0xcb28_45bd_8b81_d922),
    ("dense one landmark/variance", 2, 0xa83d_e6d1_6b8c_71be),
    ("dense one landmark/n_occupied", 2, 0x170c_689b_f08f_aeb5),
    (
        "dense one landmark/n_events_after",
        1,
        0xa956_f432_2893_eb3f,
    ),
    ("dense one landmark/statistic", 2, 0x6437_92f3_9680_4d1e),
    ("dense one landmark/p_value", 2, 0x7014_0d3e_703c_613d),
    ("sparse q2/trace", 14, 0x950f_7821_1e25_a2c1),
    ("sparse q2/variance", 14, 0x5216_32ce_c3e6_6229),
    ("sparse q2/n_occupied", 14, 0x13be_0805_4090_e97b),
    ("sparse q2/n_events_after", 7, 0x0b80_3f57_e639_2f20),
    ("sparse q2/statistic", 2, 0xd01f_1a03_3610_6ee1),
    ("sparse q2/p_value", 2, 0x7a6c_690c_88e0_3fa4),
    ("sparse q3/trace", 21, 0xfdb8_e27d_f5d9_41c1),
    ("sparse q3/variance", 21, 0x56cc_6d40_d9a6_8509),
    ("sparse q3/n_occupied", 21, 0x7403_133d_4ddd_0edb),
    ("sparse q3/n_events_after", 7, 0x0b80_3f57_e639_2f20),
    ("sparse q3/statistic", 3, 0x8d22_32a0_063a_5501),
    ("sparse q3/p_value", 3, 0x2db0_15ca_493b_7579),
    ("far states/trace", 14, 0xda96_746b_4ae2_3309),
    ("far states/variance", 14, 0x8697_b62a_d726_0117),
    ("far states/n_occupied", 14, 0x8577_d07e_5966_2e4b),
    ("far states/n_events_after", 7, 0x15e3_59b9_5fca_81ba),
    ("far states/statistic", 2, 0xfc9e_aa91_ce55_ca6a),
    ("far states/p_value", 2, 0x27cc_67d4_e5b0_2f71),
    ("overlapping stays/trace", 21, 0x00db_91d1_1b29_4002),
    ("overlapping stays/variance", 21, 0x1005_265f_5702_adab),
    ("overlapping stays/n_occupied", 21, 0x286d_01ec_6027_1f0d),
    ("overlapping stays/n_events_after", 7, 0x15e3_59b9_5fca_81ba),
    ("overlapping stays/statistic", 3, 0xe82b_6dcd_3aa7_17e5),
    ("overlapping stays/p_value", 3, 0xdba7_394e_cd8f_6aec),
    ("no events/trace", 21, 0x35da_7620_6393_6645),
    ("no events/variance", 21, 0x35da_7620_6393_6645),
    ("no events/n_occupied", 21, 0x9f47_da2f_167a_c217),
    ("no events/n_events_after", 7, 0x8ac1_23d6_f7dc_e585),
    ("no events/statistic", 3, 0x81d2_3fd7_003c_2305),
    ("no events/p_value", 3, 0x439b_b40f_bb1a_9658),
    ("every stay an event/trace", 21, 0x1f6a_6baf_500d_ad6f),
    ("every stay an event/variance", 21, 0x7e84_eb2e_3774_f288),
    ("every stay an event/n_occupied", 21, 0x9f47_da2f_167a_c217),
    (
        "every stay an event/n_events_after",
        7,
        0x0987_01ce_bb19_d2dd,
    ),
    ("every stay an event/statistic", 3, 0xd97e_5dca_18c5_e160),
    ("every stay an event/p_value", 3, 0x0776_cbd8_61f3_3958),
    ("empty qualifying state/trace", 35, 0x7b4e_793d_3eef_74b1),
    ("empty qualifying state/variance", 35, 0x649c_3eb3_011f_cccf),
    (
        "empty qualifying state/n_occupied",
        35,
        0x8f52_d7b5_7066_3c57,
    ),
    (
        "empty qualifying state/n_events_after",
        7,
        0x15e3_59b9_5fca_81ba,
    ),
    ("empty qualifying state/statistic", 5, 0x9bf4_f638_7892_f8b2),
    ("empty qualifying state/p_value", 5, 0xb6fa_10e7_cbe7_eff8),
    ("nobody at risk/trace", 4, 0x0c82_1078_4d8a_f5a5),
    ("nobody at risk/variance", 4, 0x0c82_1078_4d8a_f5a5),
    ("nobody at risk/n_occupied", 4, 0xc5d5_7a0b_8a7e_f5a5),
    ("nobody at risk/n_events_after", 2, 0xd2f5_f0ef_1385_3465),
    ("nobody at risk/statistic", 2, 0x8820_1fb9_60ff_6465),
    ("nobody at risk/p_value", 2, 0x2be2_cbea_19a8_27c5),
    ("no stays/trace", 1, 0xa8c7_f832_281a_39c5),
    ("no stays/variance", 1, 0xa8c7_f832_281a_39c5),
    ("no stays/n_occupied", 1, 0xa8c7_7832_2819_6045),
    ("no stays/n_events_after", 1, 0xa8c7_7832_2819_6045),
    ("no stays/statistic", 1, 0xa8c7_f832_281a_39c5),
    ("no stays/p_value", 1, 0xaab1_6932_29ba_1db8),
    ("one stay/trace", 1, 0xa8c7_f832_281a_39c5),
    ("one stay/variance", 1, 0xa8c7_f832_281a_39c5),
    ("one stay/n_occupied", 1, 0xaab1_6932_29ba_1db8),
    ("one stay/n_events_after", 1, 0xaab1_6932_29ba_1db8),
    ("one stay/statistic", 1, 0xa8c7_f832_281a_39c5),
    ("one stay/p_value", 1, 0xaab1_6932_29ba_1db8),
];

fn digest(values: &[f64]) -> u64 {
    Fingerprint::new().f64s(values).finish()
}

#[track_caller]
fn check(name: &str, values: &[f64]) {
    let got = (values.len(), digest(values));
    if std::env::var_os("GOLDEN_RECORD").is_some() {
        let hex = format!("{:016x}", got.1);
        let hex: Vec<&str> = (0..4).map(|i| &hex[4 * i..4 * i + 4]).collect();
        println!("    (\"{name}\", {}, 0x{}),", got.0, hex.join("_"));
        return;
    }
    let want = EXPECTED
        .iter()
        .find(|entry| entry.0 == name)
        .unwrap_or_else(|| panic!("{name} has no recorded value"));
    assert_eq!(
        got,
        (want.1, want.2),
        "{name}: the bits of the output changed"
    );
}

#[track_caller]
fn check_text(name: &str, text: &str) {
    if std::env::var_os("GOLDEN_RECORD").is_some() {
        println!("    // {name}: {text}");
    }
    let bytes: Vec<f64> = text.bytes().map(f64::from).collect();
    check(name, &bytes);
}

fn check_test(name: &str, t: &MarkovTest) {
    check(&format!("{name}/trace"), &t.trace);
    check(&format!("{name}/variance"), &t.variance);
    check(&format!("{name}/n_occupied"), &t.n_occupied);
    check(&format!("{name}/n_events_after"), &t.n_events_after);
    check(&format!("{name}/statistic"), &t.statistic);
    check(&format!("{name}/p_value"), &absent_all(&t.p_value));
}

struct Columns {
    person: Vec<u32>,
    start: Vec<f64>,
    stop: Vec<f64>,
    state: Vec<u32>,
    event: Vec<f64>,
}

impl Columns {
    fn stays(&self) -> Stays<'_> {
        Stays::new(
            &self.person,
            &self.start,
            &self.stop,
            &self.state,
            &self.event,
        )
        .unwrap()
    }
}

/// An illness-death process with recovery on a grid of tenths, so exits tie.
/// State 2 absorbs. The stays are in the state they start in, and the event is
/// the move from state 0 to state 1. With `delayed` some people enter after 0,
/// and `code` maps the person number to the code given.
fn process(people: usize, seed: u64, delayed: bool, code: impl Fn(usize) -> u32) -> Columns {
    let mut rng = Stream::seeded(seed);
    let mut c = Columns {
        person: Vec::new(),
        start: Vec::new(),
        stop: Vec::new(),
        state: Vec::new(),
        event: Vec::new(),
    };
    for id in 0..people {
        let mut state = rng.below(2) as u32;
        let mut clock = if delayed && rng.below(3) == 0 {
            1.0 + rng.below(30) as f64 / 10.0
        } else {
            0.0
        };
        loop {
            let stop = clock + 0.1 + (rng.exp1() * 30.0).floor() / 10.0;
            let next = match (state, rng.below(10)) {
                (_, 0) => 2,
                (0, _) => 1,
                _ => 0,
            };
            let last = next == 2 || stop > 12.0 || rng.below(6) == 0;
            c.person.push(code(id));
            c.start.push(clock);
            c.stop.push(stop);
            c.state.push(state);
            c.event
                .push(f64::from(u8::from(state == 0 && next == 1 && !last)));
            if last {
                break;
            }
            state = next;
            clock = stop;
        }
    }
    c
}

const GRID: [f64; 7] = [0.3, 0.9, 1.6, 2.4, 3.5, 4.8, 6.1];

#[test]
fn tests_on_a_process() {
    let seeds: Vec<u64> = (0..25).map(|r| 77 + r * 7919).collect();
    let dense = process(120, 41, false, |id| id as u32);
    for qualifying in 1..=3 {
        let stays = dense.stays();
        let t = markov_test(&stays, &GRID, qualifying, &seeds).unwrap();
        check_test(&format!("dense q{qualifying}"), &t);
    }
    let stays = dense.stays();
    check_test(
        "dense no seeds",
        &markov_test(&stays, &GRID, 2, &[]).unwrap(),
    );
    check_test(
        "dense one seed",
        &markov_test(&stays, &GRID, 2, &[5]).unwrap(),
    );
    // Landmarks out of order, repeated, before the data and past it.
    let odd_grid = [4.8, 0.3, 0.3, 0.0, 99.0, 2.4, -1.0];
    check_test(
        "dense odd grid",
        &markov_test(&stays, &odd_grid, 3, &seeds).unwrap(),
    );
    check_test(
        "dense one landmark",
        &markov_test(&stays, &[1.6], 2, &seeds).unwrap(),
    );

    // Delayed entry, sparse person codes.
    let sparse = process(90, 42, true, |id| 5 * id as u32 + 3);
    let stays = sparse.stays();
    check_test("sparse q2", &markov_test(&stays, &GRID, 2, &seeds).unwrap());
    check_test("sparse q3", &markov_test(&stays, &GRID, 3, &seeds).unwrap());
}

#[test]
fn tests_on_awkward_stays() {
    let seeds: Vec<u64> = (0..15).collect();
    let base = process(80, 43, false, |id| id as u32);

    // Stays in states that do not qualify: relabel state 2 and some of state 1
    // far above the qualifying count.
    let mut far = process(80, 43, false, |id| id as u32);
    for (index, state) in far.state.iter_mut().enumerate() {
        if *state == 2 || (*state == 1 && index % 4 == 0) {
            *state = 40 + index as u32 % 3;
        }
    }
    check_test(
        "far states",
        &markov_test(&far.stays(), &GRID, 2, &seeds).unwrap(),
    );

    // Stays of one person that overlap: a second copy of every fifth stay,
    // later in the data and in another state, covers the same landmarks.
    let mut overlap = process(80, 43, false, |id| id as u32);
    for row in (0..base.person.len()).step_by(5) {
        overlap.person.push(base.person[row]);
        overlap.start.push(base.start[row] + 0.05);
        overlap.stop.push(base.stop[row] + 0.4);
        overlap.state.push((base.state[row] + 1) % 3);
        overlap.event.push(0.0);
    }
    check_test(
        "overlapping stays",
        &markov_test(&overlap.stays(), &GRID, 3, &seeds).unwrap(),
    );

    // No event at all, and every at-risk stay an event.
    let mut quiet = process(80, 43, false, |id| id as u32);
    quiet.event.fill(0.0);
    check_test(
        "no events",
        &markov_test(&quiet.stays(), &GRID, 3, &seeds).unwrap(),
    );
    let mut all = process(80, 43, false, |id| id as u32);
    all.event.fill(1.0);
    check_test(
        "every stay an event",
        &markov_test(&all.stays(), &GRID, 3, &seeds).unwrap(),
    );

    // A qualifying state that nobody occupies has zero variance at every
    // landmark.
    check_test(
        "empty qualifying state",
        &markov_test(&base.stays(), &GRID, 5, &seeds).unwrap(),
    );

    // No stay in the origin state.
    let none = Columns {
        person: vec![0, 1, 2],
        start: vec![0.0, 0.0, 1.0],
        stop: vec![2.0, 3.0, 4.0],
        state: vec![1, 1, 2],
        event: vec![0.0, 0.0, 0.0],
    };
    check_test(
        "nobody at risk",
        &markov_test(&none.stays(), &[1.0, 2.0], 2, &seeds).unwrap(),
    );
    let nothing = Columns {
        person: vec![],
        start: vec![],
        stop: vec![],
        state: vec![],
        event: vec![],
    };
    check_test(
        "no stays",
        &markov_test(&nothing.stays(), &[1.0], 1, &seeds).unwrap(),
    );

    // One stay.
    let one = Columns {
        person: vec![0],
        start: vec![0.0],
        stop: vec![2.0],
        state: vec![0],
        event: vec![1.0],
    };
    check_test(
        "one stay",
        &markov_test(&one.stays(), &[1.0], 1, &seeds).unwrap(),
    );
}

fn message<T>(result: Result<T, epistat_survival::Error>) -> String {
    match result {
        Ok(_) => "ok".to_owned(),
        Err(error) => error.to_string(),
    }
}

#[test]
fn refusals() {
    let c = process(10, 44, false, |id| id as u32);
    let stays = c.stays();
    check_text(
        "refuse/no states",
        &message(markov_test(&stays, &GRID, 0, &[])),
    );
    check_text(
        "refuse/no landmark",
        &message(markov_test(&stays, &[], 1, &[])),
    );
    check_text(
        "refuse/lengths",
        &message(Stays::new(
            &c.person,
            &c.start[1..],
            &c.stop,
            &c.state,
            &c.event,
        )),
    );
    let mut bad = c.start.clone();
    bad[2] = f64::NAN;
    check_text(
        "refuse/nan start",
        &message(Stays::new(&c.person, &bad, &c.stop, &c.state, &c.event)),
    );
    let mut bad = c.stop.clone();
    bad[1] = f64::INFINITY;
    check_text(
        "refuse/infinite stop",
        &message(Stays::new(&c.person, &c.start, &bad, &c.state, &c.event)),
    );
}
