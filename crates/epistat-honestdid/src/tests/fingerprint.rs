//! The engine's outputs on the HonestDiD paths, hashed to the bit, on the R
//! reference event study and on 1, 4 and 12 threads.
//!
//! Each case hashes every number in the `Debug` form of what the engine
//! returns, in order, so a field that is renamed or wrapped in `Option` leaves
//! the hash alone and a value that moves in its last bit changes it. `Debug`
//! prints an `f64` with the fewest digits that read back to the same bits.
//! Run with `--nocapture` to print the current hashes. They are asserted only
//! on aarch64 macOS, where they were recorded. Elsewhere the test computes them
//! and, when `EPISTAT_HASH_OUT` is set, appends them to that file.

use std::fmt::Debug;

use epistat_testkit::{Fingerprint, load_json, on_threads};
use serde::Deserialize;

use crate::sensitivity::{
    HonestEventStudyInput, HonestJointPathConfig, HonestJointPathMethod,
    HonestMonotonicityDirection, HonestPostFunctional, HonestRelativeMagnitudeBound,
    HonestSensitivity, LeastFavorableSimulation, RelativeMagnitudeConfidenceSetConfig,
    SensitivitySummaryMethod, assess_honest_event_study_directional_region_with_config,
    assess_honest_event_study_joint_path_region_with_config, assess_honest_event_study_workflow,
    compute_original_confidence_set, compute_relative_magnitude_confidence_set,
    compute_relative_magnitude_identified_set, estimate_smoothness_upper_bound_from_pretrends,
    summarize_relative_magnitude_sensitivity_many_with_simulation,
    summarize_smoothness_sensitivity_with_simulation,
};
use crate::util::InferenceConfig;

/// The numbers in `text`, in order: every token that starts with a digit, or
/// with a minus sign before a digit, and is not part of an identifier, read as
/// an `f64`, and `NaN`, `inf`, `true` and `false`.
fn numbers(text: &str) -> Vec<f64> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let previous_is_word =
            i > 0 && (bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
        let starts_number = bytes[i].is_ascii_digit()
            || (bytes[i] == b'-' && bytes.get(i + 1).is_some_and(u8::is_ascii_digit));
        if !previous_is_word && starts_number {
            let start = i;
            i += 1;
            while i < bytes.len()
                && (bytes[i].is_ascii_digit()
                    || matches!(bytes[i], b'.' | b'e' | b'E')
                    || (matches!(bytes[i], b'-' | b'+') && matches!(bytes[i - 1], b'e' | b'E')))
            {
                i += 1;
            }
            out.push(text[start..i].parse::<f64>().expect("a number"));
            continue;
        }
        if !previous_is_word {
            for (word, value) in [
                ("true", 1.0),
                ("false", 0.0),
                ("NaN", f64::NAN),
                ("-inf", f64::NEG_INFINITY),
                ("inf", f64::INFINITY),
            ] {
                let end = i + word.len();
                let next_is_word = bytes
                    .get(end)
                    .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_');
                if text[i..].starts_with(word) && !next_is_word {
                    out.push(value);
                    i = end - 1;
                    break;
                }
            }
        }
        i += 1;
    }
    out
}

fn digest(value: &impl Debug) -> u64 {
    Fingerprint::new()
        .f64s(&numbers(&format!("{value:?}")))
        .finish()
}

#[derive(Deserialize)]
struct HonestRef {
    betahat: Vec<f64>,
    sigma: Vec<Vec<f64>>,
    pre_indices: Vec<usize>,
    post_indices: Vec<usize>,
}

fn data() -> HonestEventStudyInput {
    let honest: HonestRef = load_json("tests/data/honest_did_ref.json");
    let pre_periods = honest
        .pre_indices
        .iter()
        .map(|i| -i32::try_from(*i).expect("fits"))
        .collect();
    let post_periods = (0..honest.post_indices.len())
        .map(|i| i32::try_from(i).expect("fits"))
        .collect();
    HonestEventStudyInput {
        betahat: honest.betahat,
        covariance: honest.sigma,
        pre_periods,
        post_periods,
    }
}

fn cases(honest: &HonestEventStudyInput) -> Vec<(&'static str, u64)> {
    let mut out = Vec::new();
    let inference = InferenceConfig::default();

    // HonestDiD, on the paths Study I calls.
    let first = [1.0, 0.0, 0.0, 0.0];
    let average = [0.25, 0.25, 0.25, 0.25];
    let simulation = LeastFavorableSimulation {
        draws: 200,
        seed: 3,
    };
    out.push((
        "honest_original",
        digest(&compute_original_confidence_set(honest, &first, inference)),
    ));
    out.push((
        "honest_rm_identified",
        digest(&compute_relative_magnitude_identified_set(
            honest, &first, 1.0,
        )),
    ));
    out.push((
        "honest_rm_conditional",
        digest(&compute_relative_magnitude_confidence_set(
            honest, &average, 0.5, inference,
        )),
    ));
    out.push((
        "honest_smoothness_m",
        digest(&estimate_smoothness_upper_bound_from_pretrends(
            honest, 0.05,
        )),
    ));
    out.push((
        "honest_smoothness_summary",
        digest(&summarize_smoothness_sensitivity_with_simulation(
            honest,
            inference,
            &average,
            None,
            Some(&[0.0, 0.2, 0.5]),
            None,
            None,
            simulation,
        )),
    ));
    // The fixed-length interval of a contrast, whose weights have both signs,
    // and the conditional-FLCI hybrid under a monotone restriction at an M the
    // pre-period coefficients contradict and one they do not.
    out.push((
        "honest_smoothness_flci",
        digest(&summarize_smoothness_sensitivity_with_simulation(
            honest,
            inference,
            &[1.0, -1.0, 0.0, 0.0],
            Some(SensitivitySummaryMethod::Flci),
            Some(&[0.0, 0.05]),
            None,
            None,
            simulation,
        )),
    ));
    out.push((
        "honest_smoothness_conditional",
        digest(&summarize_smoothness_sensitivity_with_simulation(
            honest,
            inference,
            &average,
            None,
            Some(&[0.05, 0.2]),
            None,
            Some(HonestMonotonicityDirection::Decreasing),
            simulation,
        )),
    ));
    out.push((
        "honest_rm_summary_many",
        digest(
            &summarize_relative_magnitude_sensitivity_many_with_simulation(
                honest,
                inference,
                &[&first, &average],
                None,
                Some(&[0.5, 1.0]),
                Some(HonestRelativeMagnitudeBound::ParallelTrendsDeviation),
                None,
                None,
                simulation,
            ),
        ),
    ));
    for (name, method) in [
        ("honest_joint_bonferroni", HonestJointPathMethod::Bonferroni),
        (
            "honest_joint_gaussian",
            HonestJointPathMethod::GaussianSimulated,
        ),
    ] {
        out.push((
            name,
            digest(&assess_honest_event_study_joint_path_region_with_config(
                honest,
                HonestSensitivity::RelativeMagnitude(1.0),
                inference,
                0.0,
                RelativeMagnitudeConfidenceSetConfig {
                    least_favorable_draws: 200,
                    least_favorable_seed: 0,
                    ..RelativeMagnitudeConfidenceSetConfig::from_inference(
                        inference,
                        crate::sensitivity::LeastFavorableSimulation::new(1_000, 0),
                    )
                },
                HonestJointPathConfig {
                    method,
                    simulation_draws: 500,
                    simulation_seed: 11,
                },
            )),
        ));
    }
    // The workflow's directional region adds random directions drawn by
    // Box-Muller, whose sines and cosines differ in the last bit between debug
    // and release builds on some targets, so it is checked on fixed directions.
    let workflow = assess_honest_event_study_workflow(
        honest,
        &[
            HonestPostFunctional::Period(0),
            HonestPostFunctional::AverageWindow {
                start_period: 0,
                end_period: 3,
            },
        ],
        HonestSensitivity::Smoothness(0.2),
        inference,
        0.0,
    )
    .map(|result| (result.functional_assessments, result.joint_path_region));
    out.push(("honest_workflow", digest(&workflow)));
    let mut directions = (0..4)
        .map(|i| {
            (0..4)
                .map(|j| f64::from(u8::from(i == j)))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    directions.push(vec![0.25; 4]);
    directions.push(vec![1.0, -1.0, 0.0, 0.0]);
    directions.push(vec![0.0, 0.5, 0.5, -1.0]);
    out.push((
        "honest_directional_gaussian",
        digest(&assess_honest_event_study_directional_region_with_config(
            honest,
            &directions,
            HonestSensitivity::RelativeMagnitude(1.0),
            inference,
            0.0,
            RelativeMagnitudeConfidenceSetConfig {
                least_favorable_draws: 200,
                least_favorable_seed: 0,
                ..RelativeMagnitudeConfidenceSetConfig::from_inference(
                    inference,
                    crate::sensitivity::LeastFavorableSimulation::new(1_000, 0),
                )
            },
            HonestJointPathConfig {
                method: HonestJointPathMethod::GaussianSimulated,
                simulation_draws: 500,
                simulation_seed: 13,
            },
        )),
    ));

    out
}

const GOLDEN: &[(&str, u64)] = &[
    ("honest_original", 0xa7ba_d0f9_4eb4_a945),
    ("honest_rm_identified", 0xfb67_64de_8a7f_f5f5),
    ("honest_rm_conditional", 0x045a_d121_20ee_75ac),
    ("honest_smoothness_m", 0xefb8_e9c8_0987_2f59),
    ("honest_smoothness_summary", 0x74cf_3a3e_4526_d589),
    ("honest_smoothness_flci", 0x0a0d_3085_b508_3e3d),
    ("honest_smoothness_conditional", 0x6807_6f48_f895_fb2b),
    ("honest_rm_summary_many", 0x9b83_003d_f87a_07e8),
    ("honest_joint_bonferroni", 0x3dac_f805_663e_d12e),
    ("honest_joint_gaussian", 0xc186_649e_b5d3_2cfd),
    ("honest_workflow", 0x5f2a_2dea_d42c_dd95),
    ("honest_directional_gaussian", 0x5ac6_f9d4_2bfd_fd45),
];

#[test]
fn engine_outputs_are_bit_identical_to_the_recorded_hashes() {
    let data = data();
    let runs = on_threads(&[1, 4, 12], || cases(&data));
    for run in &runs[1..] {
        assert_eq!(run, &runs[0], "the hashes depend on the thread count");
    }
    let current = &runs[0];
    for (name, hash) in current {
        epistat_testkit::hash_out(name, *hash);
    }
    // The hashes were recorded on aarch64 macOS; other platforms' maths
    // libraries differ in the last bits.
    if !epistat_testkit::RECORDED_PLATFORM {
        return;
    }
    if current
        .iter()
        .map(|(n, h)| (*n, *h))
        .ne(GOLDEN.iter().copied())
    {
        for (name, hash) in current {
            let hex = format!("{hash:016x}");
            let groups: Vec<&str> = (0..4).map(|i| &hex[4 * i..4 * i + 4]).collect();
            println!("    (\"{name}\", 0x{}),", groups.join("_"));
        }
    }
    assert_eq!(current.as_slice(), GOLDEN);
}
