//! The bits of every output of the Markov test on varied inputs.
//!
//! The recorded values are an FNV-1a digest of the `to_bits` of each output
//! field, with its length, and they must not change. To record again, run with
//! `GOLDEN_RECORD=1 -- --nocapture` and paste the lines.
//!
//! The inputs come from a seeded stream. They cover several qualifying states
//! and landmarks, tied exit times, delayed entry, sparse person codes, stays
//! in states that do not qualify, slots of zero variance, no stays at risk, no
//! replicates and the refusals.
//!
//! The bits depend on the platform's `ln` and `exp`, which come from its maths
//! library, so the test asserts the recorded values only on the target it was
//! recorded on, aarch64 macOS. CI's macOS runner is that target. Thread-count
//! independence, which does not depend on the platform, is checked everywhere
//! by the thread tests. Elsewhere it computes the hashes and, when
//! `EPISTAT_HASH_OUT` is set, appends them to that file.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "the inputs are a few hundred rows, far below the limits of the casts"
)]

use epistat_core::Bootstrap;
use epistat_rng::Stream;
use epistat_survival::markov::{MarkovOptions, MarkovTest, test};
use epistat_survival::timefix::TimeFix;
use epistat_survival::{SurvBuf, SurvData};
use epistat_testkit::{Fingerprint, absent_all};

/// `(name, length, digest)` of each recorded field.
const EXPECTED: &[(&str, usize, u64)] = &[
    ("refuse/no states", 61, 0x0122_be26_44a9_c212),
    ("refuse/no landmark", 61, 0x0122_be26_44a9_c212),
    ("refuse/lengths", 32, 0xd323_0afd_6ede_6465),
    ("refuse/nan start", 84, 0x20ee_653f_0e0c_5105),
    ("refuse/infinite stop", 83, 0x1593_d5d1_9847_47d2),
    ("dense q1/trace", 7, 0x5535_2787_a143_1629),
    ("dense q1/variance", 7, 0xd5c8_2aa0_83b2_46b3),
    ("dense q1/n_occupied", 7, 0x1c9d_1507_6382_07d3),
    ("dense q1/n_events_after", 7, 0x3b3e_c9ef_529b_d6d6),
    ("dense q1/statistic", 1, 0xc59c_c993_19d2_a7d7),
    ("dense q1/p_value", 1, 0x0032_b592_eb35_5554),
    ("dense q2/trace", 14, 0xf4e1_6d0f_0836_b0ba),
    ("dense q2/variance", 14, 0xb71b_7d52_fc62_db7a),
    ("dense q2/n_occupied", 14, 0xf1cc_1f9f_b46e_5268),
    ("dense q2/n_events_after", 7, 0x3b3e_c9ef_529b_d6d6),
    ("dense q2/statistic", 2, 0x6267_2e62_f1c0_f111),
    ("dense q2/p_value", 2, 0x1a28_ec3b_8cf5_dc09),
    ("dense q3/trace", 21, 0xc6cd_a1c8_78fb_a89a),
    ("dense q3/variance", 21, 0x131e_1348_bf72_971a),
    ("dense q3/n_occupied", 21, 0x169c_f5c9_5bbb_8f28),
    ("dense q3/n_events_after", 7, 0x3b3e_c9ef_529b_d6d6),
    ("dense q3/statistic", 3, 0xc2d4_2a2d_1e77_1c7e),
    ("dense q3/p_value", 3, 0x1fdb_9f7e_d840_b616),
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
    ("dense one seed/p_value", 2, 0x2be2_cbea_19a8_27c5),
    ("dense odd grid/trace", 21, 0x1f5b_b8bf_5bc3_81e2),
    ("dense odd grid/variance", 21, 0xa37d_39bc_9362_f4dc),
    ("dense odd grid/n_occupied", 21, 0x6571_25eb_5b97_6eeb),
    ("dense odd grid/n_events_after", 7, 0xd573_a563_53b6_fead),
    ("dense odd grid/statistic", 3, 0x6d62_9621_f994_ad40),
    ("dense odd grid/p_value", 3, 0xf7d3_00dc_dc09_a166),
    ("dense one landmark/trace", 2, 0xcb28_45bd_8b81_d922),
    ("dense one landmark/variance", 2, 0xa83d_e6d1_6b8c_71be),
    ("dense one landmark/n_occupied", 2, 0x170c_689b_f08f_aeb5),
    (
        "dense one landmark/n_events_after",
        1,
        0xa956_f432_2893_eb3f,
    ),
    ("dense one landmark/statistic", 2, 0x6437_92f3_9680_4d1e),
    ("dense one landmark/p_value", 2, 0xf012_55f3_4fd0_08f5),
    ("sparse q2/trace", 14, 0x950f_7821_1e25_a2c1),
    ("sparse q2/variance", 14, 0x5216_32ce_c3e6_6229),
    ("sparse q2/n_occupied", 14, 0x13be_0805_4090_e97b),
    ("sparse q2/n_events_after", 7, 0x0b80_3f57_e639_2f20),
    ("sparse q2/statistic", 2, 0xd01f_1a03_3610_6ee1),
    ("sparse q2/p_value", 2, 0x463e_f5bf_8037_3c14),
    ("sparse q3/trace", 21, 0xfdb8_e27d_f5d9_41c1),
    ("sparse q3/variance", 21, 0x56cc_6d40_d9a6_8509),
    ("sparse q3/n_occupied", 21, 0x7403_133d_4ddd_0edb),
    ("sparse q3/n_events_after", 7, 0x0b80_3f57_e639_2f20),
    ("sparse q3/statistic", 3, 0xe5d0_4ecb_c2e0_9a0e),
    ("sparse q3/p_value", 3, 0x585c_5838_787f_996b),
    ("far states/trace", 14, 0x0fcf_6fa0_210f_3b5c),
    ("far states/variance", 14, 0xf6da_b316_9794_67ea),
    ("far states/n_occupied", 14, 0x8577_d07e_5966_2e4b),
    ("far states/n_events_after", 7, 0x72c2_e05d_889f_b5fa),
    ("far states/statistic", 2, 0xbd6b_46fd_f35f_9c7c),
    ("far states/p_value", 2, 0x3cc2_3a4f_e839_9b0d),
    ("refuse/overlapping stays", 126, 0x1136_cd91_f73a_db7a),
    ("no events/trace", 21, 0x35da_7620_6393_6645),
    ("no events/variance", 21, 0x35da_7620_6393_6645),
    ("no events/n_occupied", 21, 0x9f47_da2f_167a_c217),
    ("no events/n_events_after", 7, 0x8ac1_23d6_f7dc_e585),
    ("no events/statistic", 3, 0x6d33_1ee0_d0fb_e4d6),
    ("no events/p_value", 3, 0x6d33_1ee0_d0fb_e4d6),
    ("every stay an event/trace", 21, 0x4fa0_1df4_b432_b356),
    ("every stay an event/variance", 21, 0x3cbe_9c97_b18b_9c0e),
    ("every stay an event/n_occupied", 21, 0xb5d7_f5f1_5ae2_ac39),
    (
        "every stay an event/n_events_after",
        7,
        0x55ce_87de_b65c_a307,
    ),
    ("every stay an event/statistic", 3, 0x1b13_c964_0741_af36),
    ("every stay an event/p_value", 3, 0x61f0_4c5e_7aca_e0d8),
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
    ("empty qualifying state/statistic", 5, 0xe6bf_3876_7846_a025),
    ("empty qualifying state/p_value", 5, 0xb2c2_ee52_5e3e_55a6),
    ("nobody at risk/trace", 4, 0x0c82_1078_4d8a_f5a5),
    ("nobody at risk/variance", 4, 0x0c82_1078_4d8a_f5a5),
    ("nobody at risk/n_occupied", 4, 0xc5d5_7a0b_8a7e_f5a5),
    ("nobody at risk/n_events_after", 2, 0xd2f5_f0ef_1385_3465),
    ("nobody at risk/statistic", 2, 0x00a0_b04e_fa3c_16c9),
    ("nobody at risk/p_value", 2, 0x00a0_b04e_fa3c_16c9),
    ("refuse/no stays", 48, 0x8d31_09a0_5c1b_c999),
    ("one stay/trace", 1, 0xa8c7_f832_281a_39c5),
    ("one stay/variance", 1, 0xa8c7_f832_281a_39c5),
    ("one stay/n_occupied", 1, 0xaab1_6932_29ba_1db8),
    ("one stay/n_events_after", 1, 0xaab1_6932_29ba_1db8),
    ("one stay/statistic", 1, 0x726c_0978_9ece_f67a),
    ("one stay/p_value", 1, 0x726c_0978_9ece_f67a),
];

fn digest(values: &[f64]) -> u64 {
    Fingerprint::new().f64s(values).finish()
}

#[track_caller]
fn check(name: &str, values: &[f64]) {
    let got = (values.len(), digest(values));
    epistat_testkit::hash_out(name, got.1);
    if std::env::var_os("GOLDEN_RECORD").is_some() {
        let hex = format!("{:016x}", got.1);
        let hex: Vec<&str> = (0..4).map(|i| &hex[4 * i..4 * i + 4]).collect();
        println!("    (\"{name}\", {}, 0x{}),", got.0, hex.join("_"));
        return;
    }
    if !epistat_testkit::RECORDED_PLATFORM {
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
    check(&format!("{name}/statistic"), &absent_all(&t.statistic));
    check(&format!("{name}/p_value"), &absent_all(&t.p_value));
}

/// The stays of a process as the columns of a `SurvData`: the state occupied
/// during the stay is its `istate` code, and the status is the code of the
/// state the stay ends in, 0 when it ends in the end of follow-up.
struct Columns {
    person: Vec<u32>,
    start: Vec<f64>,
    stop: Vec<f64>,
    istate: Vec<u32>,
    status: Vec<i32>,
    n_codes: u32,
}

impl Columns {
    fn buf(&self) -> Result<SurvBuf, epistat_survival::Error> {
        SurvBuf::builder(self.stop.clone(), self.status.clone())
            .start(self.start.clone())
            .id(self.person.clone())
            .istate(self.istate.clone())
            .n_codes(self.n_codes)
            .build()
    }
}

/// The options of every test here: times are compared as they are.
const OPTIONS: MarkovOptions = MarkovOptions {
    timefix: TimeFix::Exact,
};

/// The test of the transition from the first qualifying code to code 2.
fn run(
    c: &Columns,
    grid: &[f64],
    qualifying: &[u32],
    seeds: Bootstrap,
) -> Result<MarkovTest, epistat_survival::Error> {
    let buf = c.buf()?;
    test(&buf.as_data(), qualifying, 2, grid, seeds, &OPTIONS)
}

/// An illness-death process with recovery on a grid of tenths, so exits tie.
/// State 2 absorbs. Code `s + 1` is state `s`. With `delayed` some people
/// enter after 0, and `code` maps the person number to the code given.
fn process(people: usize, seed: u64, delayed: bool, code: impl Fn(usize) -> u32) -> Columns {
    let mut rng = Stream::seeded(seed);
    let mut c = Columns {
        person: Vec::new(),
        start: Vec::new(),
        stop: Vec::new(),
        istate: Vec::new(),
        status: Vec::new(),
        n_codes: 3,
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
            c.istate.push(state + 1);
            c.status.push(if last && next != 2 {
                0
            } else {
                next as i32 + 1
            });
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
    let seeds = Bootstrap::new(25, 77);
    let dense = process(120, 41, false, |id| id as u32);
    for qualifying in 1..=3 {
        let codes: Vec<u32> = (1..=qualifying).collect();
        let t = run(&dense, &GRID, &codes, seeds).unwrap();
        check_test(&format!("dense q{qualifying}"), &t);
    }
    check_test(
        "dense no seeds",
        &run(&dense, &GRID, &[1, 2], Bootstrap::new(0, 0)).unwrap(),
    );
    check_test(
        "dense one seed",
        &run(&dense, &GRID, &[1, 2], Bootstrap::new(1, 5)).unwrap(),
    );
    // Landmarks out of order, repeated, before the data and past it.
    let odd_grid = [4.8, 0.3, 0.3, 0.0, 99.0, 2.4, -1.0];
    check_test(
        "dense odd grid",
        &run(&dense, &odd_grid, &[1, 2, 3], seeds).unwrap(),
    );
    check_test(
        "dense one landmark",
        &run(&dense, &[1.6], &[1, 2], seeds).unwrap(),
    );

    // Delayed entry, sparse person codes.
    let sparse = process(90, 42, true, |id| 5 * id as u32 + 3);
    check_test("sparse q2", &run(&sparse, &GRID, &[1, 2], seeds).unwrap());
    check_test(
        "sparse q3",
        &run(&sparse, &GRID, &[1, 2, 3], seeds).unwrap(),
    );
}

#[test]
fn tests_on_awkward_stays() {
    let seeds = Bootstrap::new(15, 0);
    let base = process(80, 43, false, |id| id as u32);

    // Stays in states that do not qualify: state 2 and some stays in state 1
    // are recoded to codes above the qualifying ones, and the status of the
    // stay before carries the code the next stay starts in.
    let mut far = process(80, 43, false, |id| id as u32);
    far.n_codes = 8;
    for index in 0..far.istate.len() {
        if far.istate[index] == 3 || (far.istate[index] == 2 && index % 4 == 0) {
            far.istate[index] = 6 + index as u32 % 3;
        }
    }
    for index in 0..far.istate.len() {
        let same_person = far.person.get(index + 1) == Some(&far.person[index]);
        far.status[index] = if same_person {
            far.istate[index + 1] as i32
        } else if far.status[index] == 3 {
            6
        } else {
            far.status[index]
        };
    }
    check_test("far states", &run(&far, &GRID, &[1, 2], seeds).unwrap());

    // Stays of one person that overlap are refused: a second copy of every
    // fifth stay, later in the data and in another state, covers the same
    // landmarks.
    let mut overlap = process(80, 43, false, |id| id as u32);
    for row in (0..base.person.len()).step_by(5) {
        overlap.person.push(base.person[row]);
        overlap.start.push(base.start[row] + 0.05);
        overlap.stop.push(base.stop[row] + 0.4);
        overlap.istate.push(base.istate[row] % 3 + 1);
        overlap.status.push(0);
    }
    check_text(
        "refuse/overlapping stays",
        &message(run(&overlap, &GRID, &[1, 2, 3], seeds)),
    );

    // No event at all: nobody moves to code 2.
    let mut quiet = process(80, 43, false, |id| id as u32);
    quiet.n_codes = 4;
    let t = run_target(&quiet, &GRID, &[1, 2, 3], 4, seeds).unwrap();
    check_test("no events", &t);

    // Every stay at risk ends in the event: the person goes from state 0 to
    // state 1 and is then censored.
    let all = Columns {
        person: (0..40).collect(),
        start: (0..40).map(|i| f64::from(i % 5) * 0.2).collect(),
        stop: (0..40).map(|i| 1.0 + f64::from(i % 7) * 0.3).collect(),
        istate: vec![1; 40],
        status: vec![2; 40],
        n_codes: 3,
    };
    check_test(
        "every stay an event",
        &run(&all, &GRID, &[1, 2, 3], seeds).unwrap(),
    );

    // A qualifying state that nobody occupies has zero variance at every
    // landmark.
    let mut wide = process(80, 43, false, |id| id as u32);
    wide.n_codes = 5;
    check_test(
        "empty qualifying state",
        &run(&wide, &GRID, &[1, 2, 3, 4, 5], seeds).unwrap(),
    );

    // No stay in the origin state.
    let none = Columns {
        person: vec![0, 1, 2],
        start: vec![0.0, 0.0, 1.0],
        stop: vec![2.0, 3.0, 4.0],
        istate: vec![2, 2, 3],
        status: vec![0, 0, 0],
        n_codes: 3,
    };
    check_test(
        "nobody at risk",
        &run(&none, &[1.0, 2.0], &[1, 2], seeds).unwrap(),
    );
    let nothing = Columns {
        person: vec![],
        start: vec![],
        stop: vec![],
        istate: vec![],
        status: vec![],
        n_codes: 2,
    };
    check_text(
        "refuse/no stays",
        &message(run(&nothing, &[1.0], &[1], seeds)),
    );

    // One stay.
    let one = Columns {
        person: vec![0],
        start: vec![0.0],
        stop: vec![2.0],
        istate: vec![1],
        status: vec![2],
        n_codes: 2,
    };
    check_test("one stay", &run(&one, &[1.0], &[1], seeds).unwrap());
}

/// The test of the transition from the first qualifying code to `target`.
fn run_target(
    c: &Columns,
    grid: &[f64],
    qualifying: &[u32],
    target: u32,
    seeds: Bootstrap,
) -> Result<MarkovTest, epistat_survival::Error> {
    let buf = c.buf()?;
    test(&buf.as_data(), qualifying, target, grid, seeds, &OPTIONS)
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
    check_text(
        "refuse/no states",
        &message(run(&c, &GRID, &[], Bootstrap::new(0, 0))),
    );
    check_text(
        "refuse/no landmark",
        &message(run(&c, &[], &[1], Bootstrap::new(0, 0))),
    );
    let short = SurvBuf::builder(c.stop.clone(), c.status.clone())
        .start(c.start[1..].to_vec())
        .id(c.person.clone())
        .istate(c.istate.clone())
        .n_codes(3)
        .build();
    check_text("refuse/lengths", &message(short));
    let mut bad = c.start.clone();
    bad[2] = f64::NAN;
    let nan = SurvData::builder(&c.stop, &c.status)
        .start(&bad)
        .id(&c.person)
        .istate(&c.istate)
        .n_codes(3)
        .build();
    check_text("refuse/nan start", &message(nan));
    let mut bad = c.stop.clone();
    bad[1] = f64::INFINITY;
    let infinite = SurvData::builder(&bad, &c.status)
        .start(&c.start)
        .id(&c.person)
        .istate(&c.istate)
        .n_codes(3)
        .build();
    check_text("refuse/infinite stop", &message(infinite));
}
