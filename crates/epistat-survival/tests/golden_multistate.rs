//! The bits of every output of the multi-state functions on varied inputs.
//!
//! The recorded values are an FNV-1a digest of the `to_bits` of each output
//! field, with its length, and they must not change: a study that has used
//! these functions gets the same numbers after any later change. To record
//! again, run with `GOLDEN_RECORD=1 -- --nocapture` and paste the lines.
//!
//! The inputs come from a seeded stream. They cover ascending times, which use
//! the precomputed plan, and descending, unsorted and tied times, which sum
//! over the times directly. They cover weights with zeros, delayed entry,
//! moves to a state left out of the analysis, no times, no rows and the
//! refusals.
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
use epistat_survival::lowlevel::integrate;
use epistat_survival::multistate::{MultistateOptions, bootstrap, estimate};
use epistat_survival::timefix::TimeFix;
use epistat_survival::{SurvBuf, SurvData};
use epistat_testkit::Fingerprint;

/// `(name, length, digest)` of each recorded field.
const EXPECTED: &[(&str, usize, u64)] = &[
    ("refuse/initial given", 84, 0x1ead_8649_64b0_384a),
    ("refuse/no cluster", 67, 0x21d1_005e_95fc_6f57),
    ("refuse/no weight", 59, 0x13fa_029d_9cad_5cf3),
    ("plain ascending/curves", 2484, 0xacad_e9a3_189b_cf93),
    ("plain ascending/restricted", 54, 0x6f58_d1d5_5f3d_a0a6),
    ("plain ascending/single seed", 3, 0xe50e_f2b0_ef4a_3053),
    ("plain ascending/no seeds", 0, 0xcbf2_9ce4_8422_2325),
    ("plain descending/curves", 2484, 0xc8fd_2f4b_85f3_bba0),
    ("plain descending/restricted", 54, 0xef31_8b7b_6f45_1f3e),
    ("plain descending/single seed", 3, 0x50eb_0151_a4a2_45c0),
    ("plain descending/no seeds", 0, 0xcbf2_9ce4_8422_2325),
    ("rich ascending/curves", 2736, 0xcd86_ed30_b833_0e65),
    ("rich ascending/restricted", 54, 0x8eed_da5d_0a60_66e4),
    ("rich ascending/single seed", 3, 0x06a0_e356_8294_1234),
    ("rich ascending/no seeds", 0, 0xcbf2_9ce4_8422_2325),
    ("rich padded/curves", 4644, 0x75c1_f7b3_0e76_6996),
    ("rich padded/restricted", 54, 0x5d57_3e63_aa4a_5a65),
    ("rich padded/single seed", 3, 0x0069_3379_6087_3599),
    ("rich padded/no seeds", 0, 0xcbf2_9ce4_8422_2325),
    ("rich descending/curves", 2736, 0x2d5c_f3e8_bf55_0c47),
    ("rich descending/restricted", 54, 0xcce6_fcdf_bbfe_8429),
    ("rich descending/single seed", 3, 0xadf8_74c2_2859_d8af),
    ("rich descending/no seeds", 0, 0xcbf2_9ce4_8422_2325),
    ("rich shuffled/curves", 2736, 0x25ae_9ee5_f132_3cfb),
    ("rich shuffled/restricted", 54, 0x0029_44db_8d21_a9e2),
    ("rich shuffled/single seed", 3, 0xadf8_74c2_2859_d8af),
    ("rich shuffled/no seeds", 0, 0xcbf2_9ce4_8422_2325),
    ("two states/curves", 1824, 0xca22_e7a5_6da4_fa7b),
    ("two states/restricted", 36, 0x73c7_0e45_9fd1_7a7a),
    ("two states/single seed", 2, 0xc745_c5a4_1518_0ebf),
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
        324,
        0x7090_3468_90c8_e333,
    ),
    ("outside ascending/at_risk", 108, 0xb377_b0fc_762f_02ca),
    ("outside ascending/integrate", 18, 0x74b9_109c_00cb_eb25),
    (
        "outside descending/probabilities",
        324,
        0x8389_5b7e_e162_a273,
    ),
    ("outside descending/at_risk", 108, 0xf4c3_6a5a_1197_70d2),
    ("outside descending/integrate", 18, 0xbbe9_b6cd_b499_9004),
    ("two states/probabilities", 216, 0xbe3d_100b_fb67_0bf5),
    ("two states/at_risk", 108, 0xb377_b0fc_762f_02ca),
    ("two states/integrate", 12, 0xfb39_b5bb_f369_f82e),
    ("no times/probabilities", 0, 0xcbf2_9ce4_8422_2325),
    ("no times/at_risk", 0, 0xcbf2_9ce4_8422_2325),
    ("no times/integrate", 18, 0xc72f_2d6c_dfc3_9b75),
    ("refuse/initial length", 54, 0x4d5c_9720_950f_7cd3),
    ("refuse/weights length", 33, 0xea58_6307_f4f2_7092),
    ("refuse/rate tolerance", 56, 0xe069_4bd1_a89c_cb35),
    ("refuse/interval lengths", 30, 0xc0b9_a2d3_f76c_7efc),
    ("refuse/nan start", 84, 0x75ab_e541_820b_a685),
    ("refuse/zero length", 56, 0x131c_7799_4edf_3aa3),
    ("refuse/no rows", 59, 0x13fa_029d_9cad_5cf3),
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

/// The rows of a process in the form `SurvData` takes: the state a row starts
/// in is the code `istate`, and the state it ends in the code `status`, 0 when
/// it ends outside the state space or in the end of follow-up.
struct Process {
    starts: Vec<f64>,
    stops: Vec<f64>,
    istate: Vec<u32>,
    status: Vec<i32>,
    person: Vec<u32>,
    weight: Vec<f64>,
}

/// An illness-death process with recovery, on a grid of tenths so that stops
/// tie. Code `s + 1` is state `s`, and state 2 absorbs. With `delayed` some
/// people enter after time 0, with `zeros` some rows have weight 0, with
/// `outside` some rows have weight 0 as if they started outside the space and
/// some moves are to a state that is left out of the analysis, which ends the
/// follow-up with status 0.
fn process(people: usize, seed: u64, delayed: bool, zeros: bool, outside: bool) -> Process {
    let mut rng = Stream::seeded(seed);
    let mut p = Process {
        starts: Vec::new(),
        stops: Vec::new(),
        istate: Vec::new(),
        status: Vec::new(),
        person: Vec::new(),
        weight: Vec::new(),
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
            let mut last = next == 2 || stop > 12.0 || rng.below(6) == 0;
            let mut status = if last && next != 2 {
                0
            } else {
                next as i32 + 1
            };
            if outside && rng.below(9) == 0 {
                status = 0;
                last = true;
            }
            let unused = outside && rng.below(11) == 0;
            p.starts.push(clock);
            p.stops.push(stop);
            p.istate.push(state + 1);
            p.status.push(status);
            p.person.push(id as u32);
            let weight = if (zeros && rng.below(6) == 0) || unused {
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

impl Process {
    /// The process with every state above the first `states` left out: a move
    /// to one of them is a censoring.
    fn restricted(&self, states: i32) -> Process {
        Process {
            starts: self.starts.clone(),
            stops: self.stops.clone(),
            istate: self.istate.clone(),
            status: self
                .status
                .iter()
                .map(|&s| if s > states { 0 } else { s })
                .collect(),
            person: self.person.clone(),
            weight: self.weight.clone(),
        }
    }

    fn buf(
        &self,
        weights: &[f64],
        clusters: Option<Vec<u32>>,
        n_codes: u32,
    ) -> Result<SurvBuf, epistat_survival::Error> {
        let mut builder = SurvBuf::builder(self.stops.clone(), self.status.clone())
            .start(self.starts.clone())
            .id(self.person.clone())
            .istate(self.istate.clone())
            .weight(weights.to_vec())
            .n_codes(n_codes);
        if let Some(clusters) = clusters {
            builder = builder.cluster(clusters);
        }
        builder.build()
    }
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

/// Times are compared as they are, and the leaving rate may exceed 1 by 1e-9.
fn options(initial: Option<&[f64]>) -> MultistateOptions {
    MultistateOptions {
        timefix: TimeFix::Exact,
        rate_tolerance: Some(1e-9),
        initial: initial.map(<[f64]>::to_vec),
    }
}

fn curve_case(
    name: &str,
    p: &Process,
    weights: &[f64],
    times: &[f64],
    initial: &[f64],
    states: u32,
) {
    let buf = p.buf(weights, None, states).unwrap();
    let occupation = estimate(&buf.as_data(), times, &options(Some(initial))).unwrap();
    check(&format!("{name}/probabilities"), &occupation.probabilities);
    check(&format!("{name}/at_risk"), &occupation.at_risk);
    let mut totals = Vec::new();
    for (horizon, origin) in HORIZONS {
        let integral = integrate(
            &occupation.probabilities,
            times,
            initial,
            states as usize,
            horizon,
            origin,
        );
        // A horizon before the origin is refused; the recorded totals hold 0
        // for it, as for a horizon at the origin.
        if horizon < origin {
            assert!(integral.is_err());
            totals.extend(vec![0.0; states as usize]);
        } else {
            totals.extend(integral.unwrap());
        }
    }
    check(&format!("{name}/integrate"), &totals);
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
    // Two states: state 2 is left out of the analysis.
    curve_case(
        "two states",
        &outside.restricted(2),
        &outside.weight,
        &times,
        &[0.5, 0.5],
        2,
    );

    // No times.
    curve_case("no times", &plain, &ones, &[], &[1.0, 0.0, 0.0], 3);
}

#[test]
fn occupation_refusals() {
    let p = process(20, 5, false, false, false);
    let ones = ones_of(&p);
    let times = event_times(&p);
    let buf = p.buf(&ones, None, 3).unwrap();
    let data = buf.as_data();
    check_text(
        "refuse/initial length",
        &message(estimate(&data, &times, &options(Some(&[1.0])))),
    );
    let short = p.buf(&ones[1..], None, 3);
    check_text("refuse/weights length", &message(short));
    let tolerance = MultistateOptions {
        rate_tolerance: Some(-1.5),
        ..options(Some(&[1.0, 0.0, 0.0]))
    };
    check_text(
        "refuse/rate tolerance",
        &message(estimate(&data, &times, &tolerance)),
    );
    let lengths = SurvData::builder(&[1.0, 2.0], &[0, 0])
        .start(&[0.0])
        .build();
    check_text("refuse/interval lengths", &message(lengths));
    let mut broken = p.starts.clone();
    broken[3] = f64::NAN;
    let nan = SurvData::builder(&p.stops, &p.status)
        .start(&broken)
        .id(&p.person)
        .istate(&p.istate)
        .n_codes(3)
        .build();
    check_text("refuse/nan start", &message(nan));
    // A zero-length row is not an interval.
    let zero = SurvData::builder(&[3.0, 1.0], &[0, 1])
        .start(&[0.0, 1.0])
        .id(&[0, 1])
        .istate(&[1, 1])
        .n_codes(2)
        .build();
    check_text("refuse/zero length", &message(zero));
    let empty = SurvData::builder(&[], &[])
        .start(&[])
        .id(&[])
        .istate(&[])
        .n_codes(3)
        .build()
        .unwrap();
    check_text(
        "refuse/no rows",
        &message(estimate(
            &empty,
            &[1.0, 2.0],
            &options(Some(&[1.0, 0.0, 0.0])),
        )),
    );
}

fn bootstrap_case(
    name: &str,
    p: &Process,
    weights: &[f64],
    clusters: &[u32],
    times: &[f64],
    states: u32,
) {
    let buf = p.buf(weights, Some(clusters.to_vec()), states).unwrap();
    let data = buf.as_data();
    let seeds = Bootstrap::new(6, 500);
    for (label, horizons) in [("curves", vec![]), ("restricted", vec![2.0, 7.5, 30.0])] {
        let out = bootstrap(&data, times, &horizons, 0.5, seeds, &options(None)).unwrap();
        check(&format!("{name}/{label}"), &out.values);
    }
    // One seed, and no seeds.
    let single = bootstrap(
        &data,
        times,
        &[4.0],
        0.0,
        Bootstrap::new(1, 77),
        &options(None),
    )
    .unwrap();
    check(&format!("{name}/single seed"), &single.values);
    let none = bootstrap(&data, times, &[], 0.0, Bootstrap::new(0, 0), &options(None)).unwrap();
    check(&format!("{name}/no seeds"), &none.values);
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
        &ascending,
        3,
    );
    bootstrap_case(
        "plain descending",
        &plain,
        &plain.weight,
        &plain.person,
        &reversed(&ascending),
        3,
    );

    let rich = process(80, 12, true, true, true);
    let times = event_times(&rich);
    // Clusters of three people.
    let clusters: Vec<u32> = rich.person.iter().map(|&p| p / 3).collect();
    for (name, times) in [
        ("rich ascending", times.clone()),
        ("rich padded", padded(&times)),
        ("rich descending", reversed(&times)),
        ("rich shuffled", shuffled(&times, 4)),
    ] {
        bootstrap_case(name, &rich, &rich.weight, &clusters, &times, 3);
    }
    bootstrap_case(
        "two states",
        &rich.restricted(2),
        &rich.weight,
        &clusters,
        &times,
        2,
    );
}

#[test]
fn bootstrap_refusals() {
    let p = process(20, 13, false, false, false);
    let ones = ones_of(&p);
    let times = event_times(&p);
    let seeds = Bootstrap::new(2, 1);
    let clustered = p.buf(&ones, Some(p.person.clone()), 3).unwrap();
    check_text(
        "refuse/initial given",
        &message(bootstrap(
            &clustered.as_data(),
            &times,
            &[],
            0.0,
            seeds,
            &options(Some(&[1.0, 0.0, 0.0])),
        )),
    );
    let plain = p.buf(&ones, None, 3).unwrap();
    check_text(
        "refuse/no cluster",
        &message(bootstrap(
            &plain.as_data(),
            &times,
            &[],
            0.0,
            seeds,
            &options(None),
        )),
    );
    let zeros = vec![0.0; ones.len()];
    let unweighted = p.buf(&zeros, Some(p.person.clone()), 3).unwrap();
    check_text(
        "refuse/no weight",
        &message(bootstrap(
            &unweighted.as_data(),
            &times,
            &[],
            0.0,
            seeds,
            &options(None),
        )),
    );
}
