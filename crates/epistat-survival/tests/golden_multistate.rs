//! The bits of every output of the multi-state functions on varied inputs.
//!
//! The recorded values are an FNV-1a digest of the `to_bits` of each output
//! field, with its length. They were taken from the functions as they were
//! before the shared input type, and they must not change: a study that has
//! used these functions gets the same numbers after any later change. To
//! record again, run with `GOLDEN_RECORD=1 -- --nocapture` and paste the lines.
//!
//! The inputs come from a seeded stream. They cover ascending times, which use
//! the precomputed plan, and descending, unsorted and tied times, which sum
//! over the times directly. They cover weights with zeros, delayed entry,
//! states outside the space on both ends of a move, a missing start time, an
//! empty state space and the refusals.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "the inputs are a few hundred rows, far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::multistate::{
    Intervals, bootstrap_state_occupation, integrate, state_occupation_curve,
};

/// `(name, length, digest)` of each recorded field.
const EXPECTED: &[(&str, usize, u64)] = &[
    ("refuse/no states", 64, 0x0a36_fcf6_333f_2fcd),
    ("refuse/base length", 65, 0x8569_aa8e_4c9b_7ee8),
    ("refuse/cluster code", 38, 0x071f_3d3b_e7d4_6e37),
    ("refuse/position", 40, 0x5542_0fed_53d6_fa19),
    ("refuse/position lengths", 82, 0xb4a1_aa93_69d6_19ef),
    ("refuse/opening state", 42, 0xeb0c_a8e9_a250_9e94),
    ("refuse/opening without weight", 93, 0xf775_e26a_fc64_e76d),
    ("refuse/opening outside", 93, 0xf775_e26a_fc64_e76d),
    ("plain ascending/curves", 2484, 0xdf16_23ba_b590_a02e),
    ("plain ascending/restricted", 54, 0xc4dc_7e01_0e0a_c5f6),
    ("plain ascending/single seed", 3, 0x9db9_e0c7_6414_7e49),
    ("plain ascending/no seeds", 0, 0xcbf2_9ce4_8422_2325),
    ("plain descending/curves", 2484, 0xc9e1_4415_f5e9_cc76),
    ("plain descending/restricted", 54, 0xfeec_f043_dee0_3656),
    ("plain descending/single seed", 3, 0x7e02_a796_e43f_520d),
    ("plain descending/no seeds", 0, 0xcbf2_9ce4_8422_2325),
    ("rich ascending/curves", 2718, 0x0605_ae59_b4a0_5a83),
    ("rich ascending/restricted", 54, 0x10bf_c4ac_92f0_d594),
    ("rich ascending/single seed", 3, 0x0932_4ebd_9158_4c67),
    ("rich ascending/no seeds", 0, 0xcbf2_9ce4_8422_2325),
    ("rich padded/curves", 4626, 0x3b6c_2b50_12ca_c701),
    ("rich padded/restricted", 54, 0xd98b_e865_d2ce_de20),
    ("rich padded/single seed", 3, 0xa577_e502_c017_3a56),
    ("rich padded/no seeds", 0, 0xcbf2_9ce4_8422_2325),
    ("rich descending/curves", 2718, 0x5f5c_0dc8_4389_984f),
    ("rich descending/restricted", 54, 0x5fbd_a106_8b2f_eb5b),
    ("rich descending/single seed", 3, 0xf781_b9b9_2b8e_89fc),
    ("rich descending/no seeds", 0, 0xcbf2_9ce4_8422_2325),
    ("rich shuffled/curves", 2718, 0x86db_f6e2_83bd_5f73),
    ("rich shuffled/restricted", 54, 0x0a7e_0919_58f7_5cd4),
    ("rich shuffled/single seed", 3, 0xf781_b9b9_2b8e_89fc),
    ("rich shuffled/no seeds", 0, 0xcbf2_9ce4_8422_2325),
    ("two states/curves", 1812, 0x9816_8b00_f92c_3570),
    ("two states/restricted", 36, 0xf547_1d5e_47f9_0033),
    ("two states/single seed", 2, 0x6182_27da_7f75_267f),
    ("two states/no seeds", 0, 0xcbf2_9ce4_8422_2325),
    ("plain ascending/probabilities", 360, 0x883a_1b01_5ad4_3c79),
    ("plain ascending/at_risk", 120, 0xe166_05a9_b7fa_d256),
    ("plain ascending/integrate", 18, 0x7e74_66ac_b5e2_a687),
    ("plain descending/probabilities", 360, 0x3e5a_79ae_69e3_2ffc),
    ("plain descending/at_risk", 120, 0xe330_0b39_9558_8c6f),
    ("plain descending/integrate", 18, 0xb1ec_2a30_6856_3041),
    ("zeros ascending/probabilities", 414, 0x807a_fd78_0324_19dc),
    ("zeros ascending/at_risk", 138, 0x2f56_d660_220d_ff8b),
    ("zeros ascending/integrate", 18, 0xb93b_2f3a_5721_6e1b),
    ("zeros padded/probabilities", 702, 0xac83_8db8_4431_a329),
    ("zeros padded/at_risk", 234, 0xf445_91dc_799f_23f9),
    ("zeros padded/integrate", 18, 0x44af_5f2b_4b80_6abe),
    ("zeros descending/probabilities", 414, 0x3888_0dec_3913_0c48),
    ("zeros descending/at_risk", 138, 0xf525_4d1d_a01e_8777),
    ("zeros descending/integrate", 18, 0x3229_dcce_ddd0_c628),
    ("zeros shuffled/probabilities", 414, 0x9da9_1a19_7a16_4fdd),
    ("zeros shuffled/at_risk", 138, 0x364d_caeb_bae0_2c57),
    ("zeros shuffled/integrate", 18, 0x95de_7c5d_bf0f_8428),
    (
        "outside ascending/probabilities",
        393,
        0x76a3_91ab_f808_8bd9,
    ),
    ("outside ascending/at_risk", 131, 0x6502_6a54_6b42_0b1f),
    ("outside ascending/integrate", 18, 0x4469_6e0d_5833_b7ac),
    (
        "outside descending/probabilities",
        393,
        0xd035_bee6_0a1e_2d66,
    ),
    ("outside descending/at_risk", 131, 0x3336_376a_1c91_4643),
    ("outside descending/integrate", 18, 0x7cce_81a2_e861_7e05),
    ("two states/probabilities", 262, 0x7422_242b_c4d2_d815),
    ("two states/at_risk", 131, 0x6502_6a54_6b42_0b1f),
    ("two states/integrate", 12, 0x0393_4f08_2042_4c16),
    ("one state/probabilities", 131, 0x569c_56c2_aff9_1e58),
    ("one state/at_risk", 131, 0x51d9_10c3_0e89_e1dd),
    ("one state/integrate", 6, 0x11a9_855c_1bb0_8335),
    (
        "nan start ascending/probabilities",
        207,
        0xfed1_4f3c_0cc6_0d19,
    ),
    ("nan start ascending/at_risk", 69, 0xee4a_07d6_f724_5aef),
    ("nan start ascending/integrate", 18, 0x968a_8d43_5519_553c),
    (
        "nan start descending/probabilities",
        207,
        0x0d7f_45dd_b9ae_d70f,
    ),
    ("nan start descending/at_risk", 69, 0xa5ee_c2d7_3e68_5613),
    ("nan start descending/integrate", 18, 0x1526_1a11_27e5_ef25),
    ("no times/probabilities", 0, 0xcbf2_9ce4_8422_2325),
    ("no times/at_risk", 0, 0xcbf2_9ce4_8422_2325),
    ("no times/integrate", 18, 0xc72f_2d6c_dfc3_9b75),
    ("no rows/probabilities", 6, 0xbe19_a168_b74b_7f05),
    ("no rows/at_risk", 2, 0x8820_1fb9_60ff_6465),
    ("no rows/integrate", 18, 0xc72f_2d6c_dfc3_9b75),
    ("zero length/probabilities", 4, 0xbcb9_9a05_911b_b965),
    ("zero length/at_risk", 2, 0x2be2_cbea_19a8_27c5),
    ("refuse/initial length", 54, 0x4d5c_9720_950f_7cd3),
    ("refuse/weights length", 68, 0x7c47_0879_e6b8_5046),
    ("refuse/rate tolerance", 100, 0xaf15_6ea7_0f89_fb46),
    ("refuse/interval lengths", 64, 0xe96f_356f_c6d3_b986),
];

fn digest(values: &[f64]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    }
    hash
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

struct Process {
    starts: Vec<f64>,
    stops: Vec<f64>,
    from: Vec<u32>,
    to: Vec<u32>,
    person: Vec<u32>,
    weight: Vec<f64>,
    first: Vec<u32>,
}

/// An illness-death process with recovery, on a grid of tenths so that stops
/// tie. State 2 absorbs, 9 is a censoring and stands outside the space. With
/// `delayed` some people enter after time 0, with `zeros` some rows have weight
/// 0, with `outside` some rows start in state 7 and some moves go to state 5,
/// both outside the space of three states.
fn process(people: usize, seed: u64, delayed: bool, zeros: bool, outside: bool) -> Process {
    let mut rng = Stream::seeded(seed);
    let mut p = Process {
        starts: Vec::new(),
        stops: Vec::new(),
        from: Vec::new(),
        to: Vec::new(),
        person: Vec::new(),
        weight: Vec::new(),
        first: Vec::new(),
    };
    for id in 0..people {
        let mut state = rng.below(2) as u32;
        let mut clock = if delayed && rng.below(3) == 0 {
            1.0 + rng.below(30) as f64 / 10.0
        } else {
            0.0
        };
        p.first.push(p.starts.len() as u32);
        loop {
            let stop = clock + 0.1 + (rng.exp1() * 30.0).floor() / 10.0;
            let next = match (state, rng.below(10)) {
                (_, 0) => 2,
                (0, _) => 1,
                _ => 0,
            };
            let last = next == 2 || stop > 12.0 || rng.below(6) == 0;
            let mut to = if last && next != 2 { 9 } else { next };
            let mut from = state;
            if outside && rng.below(9) == 0 {
                to = 5;
            }
            if outside && rng.below(11) == 0 {
                from = 7;
            }
            p.starts.push(clock);
            p.stops.push(stop);
            p.from.push(from);
            p.to.push(to);
            p.person.push(id as u32);
            let weight = if zeros && rng.below(6) == 0 {
                0.0
            } else {
                0.5 + rng.uniform()
            };
            p.weight.push(weight);
            if last {
                break;
            }
            state = next;
            clock = stop;
        }
    }
    p
}

fn event_times(p: &Process) -> Vec<f64> {
    let mut times = p.stops.clone();
    times.sort_by(f64::total_cmp);
    times.dedup();
    times
}

fn reversed(times: &[f64]) -> Vec<f64> {
    let mut out = times.to_vec();
    out.reverse();
    out
}

/// Ascending with repeats and with times that are no one's stop.
fn padded(times: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0, 0.0, 0.05];
    for (index, &time) in times.iter().enumerate() {
        out.push(time);
        if index % 3 == 0 {
            out.push(time);
            out.push(time + 0.05);
        }
    }
    out.push(40.0);
    out.sort_by(f64::total_cmp);
    out
}

fn shuffled(times: &[f64], seed: u64) -> Vec<f64> {
    let mut out = times.to_vec();
    Stream::seeded(seed).shuffle(&mut out);
    out
}

const HORIZONS: [(f64, f64); 6] = [
    (5.0, 0.0),
    (20.0, 1.3),
    (0.5, 0.0),
    (3.0, 3.0),
    (2.0, 4.0),
    (100.0, 0.0),
];

fn curve_case(
    name: &str,
    p: &Process,
    weights: &[f64],
    times: &[f64],
    initial: &[f64],
    states: usize,
) {
    let intervals = Intervals::new(&p.starts, &p.stops, &p.from, &p.to).unwrap();
    let occupation =
        state_occupation_curve(intervals, weights, times, initial, states, 1e-9).unwrap();
    check(&format!("{name}/probabilities"), &occupation.probabilities);
    check(&format!("{name}/at_risk"), &occupation.at_risk);
    let mut totals = Vec::new();
    for (horizon, origin) in HORIZONS {
        totals.extend(integrate(
            &occupation.probabilities,
            times,
            initial,
            states,
            horizon,
            origin,
        ));
    }
    check(&format!("{name}/integrate"), &totals);
}

#[test]
fn occupation_curves() {
    let plain = process(60, 1, false, false, false);
    let ones = vec![1.0; plain.starts.len()];
    let ascending = event_times(&plain);
    curve_case(
        "plain ascending",
        &plain,
        &ones,
        &ascending,
        &[0.6, 0.4, 0.0],
        3,
    );
    curve_case(
        "plain descending",
        &plain,
        &plain.weight,
        &reversed(&ascending),
        &[0.6, 0.4, 0.0],
        3,
    );

    let rich = process(70, 2, true, true, false);
    let times = event_times(&rich);
    let initial = [0.5, 0.3, 0.2];
    curve_case("zeros ascending", &rich, &rich.weight, &times, &initial, 3);
    curve_case(
        "zeros padded",
        &rich,
        &rich.weight,
        &padded(&times),
        &initial,
        3,
    );
    curve_case(
        "zeros descending",
        &rich,
        &rich.weight,
        &reversed(&times),
        &initial,
        3,
    );
    curve_case(
        "zeros shuffled",
        &rich,
        &rich.weight,
        &shuffled(&times, 9),
        &initial,
        3,
    );

    let outside = process(70, 3, true, true, true);
    let times = event_times(&outside);
    for (name, times) in [
        ("outside ascending", times.clone()),
        ("outside descending", reversed(&times)),
    ] {
        curve_case(name, &outside, &outside.weight, &times, &initial, 3);
    }
    // Two states: state 2 and everything above it is outside the space.
    curve_case(
        "two states",
        &outside,
        &outside.weight,
        &times,
        &[0.5, 0.5],
        2,
    );
    curve_case("one state", &outside, &ones_of(&outside), &times, &[1.0], 1);

    // A start time that is not a number is at risk at no time.
    let mut broken = process(30, 4, false, false, false);
    broken.starts[3] = f64::NAN;
    broken.starts[11] = f64::NAN;
    let times = event_times(&broken);
    let weights = ones_of(&broken);
    curve_case(
        "nan start ascending",
        &broken,
        &weights,
        &times,
        &[0.5, 0.5, 0.0],
        3,
    );
    curve_case(
        "nan start descending",
        &broken,
        &weights,
        &reversed(&times),
        &[0.5, 0.5, 0.0],
        3,
    );

    // No times, and no intervals.
    curve_case("no times", &plain, &ones, &[], &[1.0, 0.0, 0.0], 3);
    let empty = Process {
        starts: vec![],
        stops: vec![],
        from: vec![],
        to: vec![],
        person: vec![],
        weight: vec![],
        first: vec![],
    };
    curve_case("no rows", &empty, &[], &[1.0, 2.0], &[1.0, 0.0, 0.0], 3);

    // A zero-length interval is not a move.
    let intervals = Intervals::new(&[0.0, 1.0], &[3.0, 1.0], &[0, 0], &[0, 1]).unwrap();
    let out =
        state_occupation_curve(intervals, &[1.0; 2], &[1.0, 3.0], &[1.0, 0.0], 2, 1e-9).unwrap();
    check("zero length/probabilities", &out.probabilities);
    check("zero length/at_risk", &out.at_risk);
}

fn message<T>(result: Result<T, epistat_survival::Error>) -> String {
    match result {
        Ok(_) => "ok".to_owned(),
        Err(error) => error.to_string(),
    }
}

fn ones_of(p: &Process) -> Vec<f64> {
    vec![1.0; p.starts.len()]
}

#[test]
fn occupation_refusals() {
    let p = process(20, 5, false, false, false);
    let ones = ones_of(&p);
    let times = event_times(&p);
    let intervals = Intervals::new(&p.starts, &p.stops, &p.from, &p.to).unwrap();
    check_text(
        "refuse/initial length",
        &message(state_occupation_curve(
            intervals,
            &ones,
            &times,
            &[1.0],
            3,
            1e-9,
        )),
    );
    check_text(
        "refuse/weights length",
        &message(state_occupation_curve(
            intervals,
            &ones[1..],
            &times,
            &[1.0, 0.0, 0.0],
            3,
            1e-9,
        )),
    );
    check_text(
        "refuse/rate tolerance",
        &message(state_occupation_curve(
            intervals,
            &ones,
            &times,
            &[1.0, 0.0, 0.0],
            3,
            -1.5,
        )),
    );
    check_text(
        "refuse/interval lengths",
        &message(Intervals::new(&[0.0], &[1.0, 2.0], &[0], &[0])),
    );
}

fn bootstrap_case(
    name: &str,
    p: &Process,
    base: &[f64],
    clusters: &[u32],
    n_clusters: usize,
    times: &[f64],
    states: usize,
) {
    let intervals = Intervals::new(&p.starts, &p.stops, &p.from, &p.to).unwrap();
    let first_states: Vec<i32> = p
        .first
        .iter()
        .map(|&i| {
            let from = p.from[i as usize];
            if (from as usize) < states {
                from as i32
            } else {
                -1
            }
        })
        .collect();
    let seeds: Vec<u64> = (0..6).map(|r| 500 + r * 7919).collect();
    for (label, horizons) in [("curves", vec![]), ("restricted", vec![2.0, 7.5, 30.0])] {
        let out = bootstrap_state_occupation(
            intervals,
            base,
            clusters,
            &p.first,
            &first_states,
            times,
            &horizons,
            0.5,
            &seeds,
            n_clusters,
            states,
            1e-9,
        )
        .unwrap();
        check(&format!("{name}/{label}"), &out);
    }
    // One seed, and no seeds.
    let single = bootstrap_state_occupation(
        intervals,
        base,
        clusters,
        &p.first,
        &first_states,
        times,
        &[4.0],
        0.0,
        &[77],
        n_clusters,
        states,
        1e-9,
    )
    .unwrap();
    check(&format!("{name}/single seed"), &single);
    let none = bootstrap_state_occupation(
        intervals,
        base,
        clusters,
        &p.first,
        &first_states,
        times,
        &[],
        0.0,
        &[],
        n_clusters,
        states,
        1e-9,
    )
    .unwrap();
    check(&format!("{name}/no seeds"), &none);
}

#[test]
fn bootstraps() {
    let plain = process(60, 11, false, false, false);
    let ascending = event_times(&plain);
    let weights = ones_of(&plain);
    bootstrap_case(
        "plain ascending",
        &plain,
        &weights,
        &plain.person,
        60,
        &ascending,
        3,
    );
    bootstrap_case(
        "plain descending",
        &plain,
        &plain.weight,
        &plain.person,
        60,
        &reversed(&ascending),
        3,
    );

    let rich = process(80, 12, true, true, true);
    let times = event_times(&rich);
    // Clusters of three people, and clusters that no row belongs to.
    let clusters: Vec<u32> = rich.person.iter().map(|&p| p / 3).collect();
    for (name, times) in [
        ("rich ascending", times.clone()),
        ("rich padded", padded(&times)),
        ("rich descending", reversed(&times)),
        ("rich shuffled", shuffled(&times, 4)),
    ] {
        bootstrap_case(name, &rich, &rich.weight, &clusters, 40, &times, 3);
    }
    bootstrap_case("two states", &rich, &rich.weight, &clusters, 27, &times, 2);
}

#[test]
fn bootstrap_refusals() {
    let p = process(20, 13, false, false, false);
    let ones = ones_of(&p);
    let times = event_times(&p);
    let intervals = Intervals::new(&p.starts, &p.stops, &p.from, &p.to).unwrap();
    let first_states = vec![0; p.first.len()];
    let run = |base: &[f64],
               codes: &[u32],
               positions: &[u32],
               opening: &[i32],
               n_clusters: usize,
               states: usize| {
        message(bootstrap_state_occupation(
            intervals,
            base,
            codes,
            positions,
            opening,
            &times,
            &[],
            0.0,
            &[1, 2],
            n_clusters,
            states,
            1e-9,
        ))
    };
    check_text(
        "refuse/no states",
        &run(&ones, &p.person, &p.first, &first_states, 20, 0),
    );
    check_text(
        "refuse/base length",
        &run(&ones[1..], &p.person, &p.first, &first_states, 20, 3),
    );
    check_text(
        "refuse/cluster code",
        &run(&ones, &p.person, &p.first, &first_states, 19, 3),
    );
    check_text(
        "refuse/position",
        &run(&ones, &p.person, &[10_000], &[0], 20, 3),
    );
    check_text(
        "refuse/position lengths",
        &run(&ones, &p.person, &p.first, &first_states[1..], 20, 3),
    );
    check_text(
        "refuse/opening state",
        &run(&ones, &p.person, &[0], &[3], 20, 3),
    );
    check_text(
        "refuse/opening without weight",
        &run(
            &vec![0.0; ones.len()],
            &p.person,
            &p.first,
            &first_states,
            20,
            3,
        ),
    );
    check_text(
        "refuse/opening outside",
        &run(&ones, &p.person, &p.first, &vec![-1; p.first.len()], 20, 3),
    );
}
