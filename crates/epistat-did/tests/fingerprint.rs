//! The outputs of the main estimators, hashed to the bit, on the R reference
//! data and on 1, 4 and 12 threads.
//!
//! Each case hashes every number in the `Debug` form of what the estimator
//! returns, in order, so a field that is renamed or wrapped in `Option` leaves
//! the hash alone and a value that moves in its last bit changes it. `Debug`
//! prints an `f64` with the fewest digits that read back to the same bits.
//! Run with `--nocapture` to print the current hashes.

use std::fmt::Debug;

use epistat_did::{
    AggteConfig, AggteType, AttGtAggregationConfig, AttGtAggregationWeighting, AttGtBandConfig,
    AttGtConfig, AttGtDrConfig, AttGtDrObservation, AttGtObservation, BasePeriod, ClusterIndex,
    ComparisonGroup, ContinuousObservation, DesignPruning, DidCcConfig, DidConfig, DrDidConfig,
    DrDidObservation, DrDidRepeatedObservation, EfficientDidConfig, HonestEventStudyInput,
    HonestJointPathConfig, HonestJointPathMethod, HonestPostFunctional,
    HonestRelativeMagnitudeBound, HonestSensitivity, InferenceConfig, LeastFavorableSimulation,
    PanelFlatInput, PanelObservation, PolynomialBasis, RelativeMagnitudeConfidenceSetConfig,
    TripleDidObservation, aggregate_att_gt, aggregate_att_gt_by_calendar_time,
    aggregate_att_gt_by_cohort, aggregate_att_gt_event_time_with_influence,
    aggregate_att_gt_overall, aggregate_efficient_att_gt_event_time,
    assess_honest_event_study_directional_region_with_config,
    assess_honest_event_study_joint_path_region_with_config, assess_honest_event_study_workflow,
    att_gt_clustered_standard_errors, att_gt_mboot_bands, att_gt_simultaneous_bands,
    clustered_standard_error_from_index, compute_original_confidence_set,
    compute_relative_magnitude_confidence_set, compute_relative_magnitude_identified_set,
    estimate_acrt_sieve, estimate_att_gt, estimate_att_gt_dr_efficient_with_influence,
    estimate_att_gt_dr_panel_with_influence, estimate_att_gt_dr_with_influence,
    estimate_att_gt_efficient, estimate_att_gt_efficient_dr, estimate_att_gt_ipw_with_influence,
    estimate_att_gt_or_with_influence, estimate_att_two_by_two, estimate_did_cc_robust,
    estimate_did_cc_stationary, estimate_dr_ddd, estimate_drdid_improved_panel,
    estimate_drdid_improved_repeated_cross_section, estimate_drdid_panel,
    estimate_drdid_repeated_cross_section, estimate_drdid_repeated_efficient,
    estimate_smoothness_upper_bound_from_pretrends, fit_drdid_panel_nuisance,
    fit_drdid_panel_propensity_coefficients, independent_design_columns,
    summarize_relative_magnitude_sensitivity_many_with_simulation,
    summarize_smoothness_sensitivity_with_simulation, test_did_cc_stationarity, unit_panel,
};
use epistat_testkit::{Fingerprint, load_json, on_threads};
use serde::Deserialize;

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
struct MpdtaRow {
    year: i32,
    countyreal: i64,
    lemp: f64,
    #[serde(rename = "first.treat")]
    first_treat: i32,
    lpop: f64,
}

#[derive(Deserialize)]
struct Mpdta {
    data_subset: Vec<MpdtaRow>,
}

#[derive(Deserialize)]
struct HonestRef {
    betahat: Vec<f64>,
    sigma: Vec<Vec<f64>>,
    pre_indices: Vec<usize>,
    post_indices: Vec<usize>,
}

#[derive(Deserialize)]
struct TripleRow {
    id: i32,
    partition: i32,
    x1: f64,
    x2: f64,
    treat: i32,
    outcome: f64,
}

#[derive(Deserialize)]
struct TripleRef {
    data: Vec<TripleRow>,
}

#[derive(Deserialize)]
struct ContRef {
    delta_y: Vec<f64>,
    dose: Vec<f64>,
}

#[derive(Deserialize)]
struct CcRow {
    treated: bool,
    post_period: bool,
    outcome: f64,
    covariates: Vec<f64>,
}

#[derive(Deserialize)]
struct CcDesign {
    rows: Vec<CcRow>,
}

#[derive(Deserialize)]
struct CcRef {
    composition_shift: CcDesign,
}

struct Data {
    mpdta: Vec<MpdtaRow>,
    honest: HonestEventStudyInput,
    triple: Vec<TripleDidObservation>,
    continuous: Vec<ContinuousObservation>,
    cc: Vec<DrDidRepeatedObservation>,
}

fn data() -> Data {
    let mpdta: Mpdta = load_json("tests/cs_panel_dr_universal_ref.json");
    let honest: HonestRef = load_json("tests/honest_did_ref.json");
    let pre_periods = honest
        .pre_indices
        .iter()
        .map(|i| -i32::try_from(*i).expect("fits"))
        .collect();
    let post_periods = (0..honest.post_indices.len())
        .map(|i| i32::try_from(i).expect("fits"))
        .collect();
    let triple: TripleRef = load_json("tests/triple_did_ref.json");
    let triple = triple
        .data
        .chunks(2)
        .map(|pair| {
            assert_eq!(pair[0].id, pair[1].id);
            TripleDidObservation {
                treated: pair[1].treat == 1,
                group_s: pair[1].treat == 1,
                partition_q: pair[1].partition == 1,
                delta_outcome: pair[1].outcome - pair[0].outcome,
                weight: 1.0,
                covariates: vec![1.0, pair[1].x1, pair[1].x2],
            }
        })
        .collect();
    let continuous: ContRef = load_json("tests/contdid_ref.json");
    let continuous = continuous
        .dose
        .iter()
        .zip(&continuous.delta_y)
        .map(|(&dose, &delta_outcome)| ContinuousObservation {
            dose,
            delta_outcome,
            weight: 1.0,
        })
        .collect();
    let cc: CcRef = load_json("tests/did_cc_ref_scaffold.json");
    let cc = cc
        .composition_shift
        .rows
        .into_iter()
        .map(|row| DrDidRepeatedObservation {
            treated: row.treated,
            post_period: row.post_period,
            outcome: row.outcome,
            weight: 1.0,
            covariates: row.covariates,
        })
        .collect();
    Data {
        mpdta: mpdta.data_subset,
        honest: HonestEventStudyInput {
            betahat: honest.betahat,
            covariance: honest.sigma,
            pre_periods,
            post_periods,
        },
        triple,
        continuous,
        cc,
    }
}

/// A weight that varies with the county, so the weighted paths are exercised.
fn weight(county: i64) -> f64 {
    0.5 + f64::from(u32::try_from(county % 7).expect("small"))
}

fn dr_rows(rows: &[MpdtaRow], panel: bool, weighted: bool) -> Vec<AttGtDrObservation> {
    rows.iter()
        .map(|row| AttGtDrObservation {
            unit_id: panel.then_some(row.countyreal),
            first_treated_time: (row.first_treat != 0).then_some(row.first_treat),
            time: row.year,
            outcome: row.lemp,
            weight: if weighted {
                weight(row.countyreal)
            } else {
                1.0
            },
            covariates: vec![row.lpop],
            comparison_cohort: None,
        })
        .collect()
}

fn dr_config(comparison_group: ComparisonGroup, base_period: BasePeriod) -> AttGtDrConfig {
    AttGtDrConfig {
        att_gt: AttGtConfig {
            confidence_level: InferenceConfig::default(),
            comparison_group,
            base_period,
            anticipation_periods: 0,
            skip_incomplete_pairs: true,
        },
        drdid: DrDidConfig::builder().build(),
    }
}

/// The 2004 cohort against the never treated, 2003 to 2004.
fn cell(rows: &[MpdtaRow]) -> (Vec<DrDidObservation>, Vec<DrDidRepeatedObservation>) {
    let kept = |row: &&MpdtaRow| row.first_treat == 2004 || row.first_treat == 0;
    let mut panel = Vec::new();
    for post in rows.iter().filter(kept).filter(|row| row.year == 2004) {
        let base = rows
            .iter()
            .find(|row| row.countyreal == post.countyreal && row.year == 2003)
            .expect("base year");
        panel.push(DrDidObservation {
            treated: post.first_treat == 2004,
            delta_outcome: post.lemp - base.lemp,
            weight: weight(post.countyreal),
            covariates: vec![1.0, post.lpop],
        });
    }
    let repeated = rows
        .iter()
        .filter(kept)
        .filter(|row| row.year == 2003 || row.year == 2004)
        .map(|row| DrDidRepeatedObservation {
            treated: row.first_treat == 2004,
            post_period: row.year == 2004,
            outcome: row.lemp,
            weight: weight(row.countyreal),
            // The repeated estimators prepend the intercept themselves.
            covariates: vec![row.lpop],
        })
        .collect();
    (panel, repeated)
}

fn cases(data: &Data) -> Vec<(&'static str, u64)> {
    let mut out = Vec::new();
    let inference = InferenceConfig::default();

    // ATT(g,t) without covariates, the AttGtObservation route.
    let plain = data
        .mpdta
        .iter()
        .map(|row| {
            AttGtObservation::with_unit_id(
                row.countyreal,
                (row.first_treat != 0).then_some(row.first_treat),
                row.year,
                row.lemp,
            )
        })
        .collect::<Vec<_>>();
    out.push((
        "att_gt",
        digest(&estimate_att_gt(&plain, AttGtConfig::default())),
    ));
    out.push((
        "att_gt_efficient",
        digest(&estimate_att_gt_efficient(
            &plain,
            EfficientDidConfig {
                confidence_level: inference,
            },
        )),
    ));

    // The panel route, as Study I runs it, and the aggregations.
    let panel_rows = dr_rows(&data.mpdta, true, true);
    for (name, comparison, base) in [
        (
            "att_gt_dr_panel_universal_never",
            ComparisonGroup::NeverTreated,
            BasePeriod::Universal,
        ),
        (
            "att_gt_dr_panel_varying_notyet",
            ComparisonGroup::NotYetTreated,
            BasePeriod::Varying,
        ),
    ] {
        out.push((
            name,
            digest(&estimate_att_gt_dr_panel_with_influence(
                &panel_rows,
                dr_config(comparison, base),
            )),
        ));
    }
    let config = dr_config(ComparisonGroup::NeverTreated, BasePeriod::Universal);
    let fitted = estimate_att_gt_dr_panel_with_influence(&panel_rows, config).expect("panel fit");
    let panel = unit_panel(&panel_rows).expect("unit panel");
    for (name, aggregation, balance_e) in [
        ("aggte_simple", AggteType::Simple, None),
        ("aggte_dynamic", AggteType::Dynamic, None),
        ("aggte_dynamic_balanced", AggteType::Dynamic, Some(1)),
        ("aggte_group", AggteType::Group, None),
        ("aggte_calendar", AggteType::Calendar, None),
    ] {
        out.push((
            name,
            digest(&aggregate_att_gt(
                &fitted.estimates,
                &fitted.influence_functions,
                &panel,
                AggteConfig {
                    aggregation,
                    balance_e,
                    min_e: None,
                    max_e: None,
                    confidence_level: inference,
                },
            )),
        ));
    }
    for (names, weighting) in [
        (
            [
                "event_time_equal",
                "cohort_equal",
                "calendar_equal",
                "overall_equal",
            ],
            AttGtAggregationWeighting::Equal,
        ),
        (
            [
                "event_time_count",
                "cohort_count",
                "calendar_count",
                "overall_count",
            ],
            AttGtAggregationWeighting::ByTreatedCount,
        ),
        (
            [
                "event_time_weight",
                "cohort_weight",
                "calendar_weight",
                "overall_weight",
            ],
            AttGtAggregationWeighting::ByTotalWeight,
        ),
    ] {
        let aggregation = AttGtAggregationConfig {
            confidence_level: inference,
            weighting,
        };
        out.push((
            names[0],
            digest(&aggregate_att_gt_event_time_with_influence(
                &fitted.estimates,
                &fitted.influence_functions,
                aggregation,
            )),
        ));
        out.push((
            names[1],
            digest(&aggregate_att_gt_by_cohort(&fitted.estimates, aggregation)),
        ));
        out.push((
            names[2],
            digest(&aggregate_att_gt_by_calendar_time(
                &fitted.estimates,
                aggregation,
            )),
        ));
        out.push((
            names[3],
            digest(&aggregate_att_gt_overall(&fitted.estimates, aggregation)),
        ));
    }
    // Clusters of units, indexed like the influence vectors: by sorted unit id.
    let mut ids = panel_rows
        .iter()
        .map(|row| row.unit_id.expect("panel"))
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids.dedup();
    let units = ids.iter().map(|id| id % 37).collect::<Vec<_>>();
    let mut clustered = panel.clone();
    clustered.clusters = Some(units.clone());
    out.push((
        "aggte_dynamic_clustered",
        digest(&aggregate_att_gt(
            &fitted.estimates,
            &fitted.influence_functions,
            &clustered,
            AggteConfig {
                aggregation: AggteType::Dynamic,
                balance_e: None,
                min_e: None,
                max_e: None,
                confidence_level: inference,
            },
        )),
    ));
    let bands = AttGtBandConfig {
        confidence_level: inference,
        reps: 999,
        seed: 20_261_122,
    };
    out.push((
        "mboot_bands",
        digest(&att_gt_mboot_bands(
            &fitted.estimates,
            &fitted.influence_functions,
            None,
            bands,
        )),
    ));
    out.push((
        "mboot_bands_clustered",
        digest(&att_gt_mboot_bands(
            &fitted.estimates,
            &fitted.influence_functions,
            Some(&units),
            bands,
        )),
    ));
    out.push((
        "simultaneous_bands",
        digest(&att_gt_simultaneous_bands(&fitted.estimates, bands)),
    ));
    out.push((
        "clustered_standard_errors",
        digest(&att_gt_clustered_standard_errors(
            &fitted.estimates,
            &fitted.influence_functions,
            &clustered,
            inference,
        )),
    ));
    let mut seen = Vec::new();
    let assignments = units
        .iter()
        .map(|code| {
            seen.iter().position(|c| c == code).unwrap_or_else(|| {
                seen.push(*code);
                seen.len() - 1
            })
        })
        .collect::<Vec<_>>();
    let index = ClusterIndex {
        n_groups: seen.len(),
        group_assignments: assignments,
    };
    out.push((
        "clustered_standard_error_from_index",
        digest(
            &fitted
                .influence_functions
                .iter()
                .map(|psi| clustered_standard_error_from_index(psi, &index))
                .collect::<Vec<_>>(),
        ),
    ));

    // The repeated cross-section route and the other pair estimators.
    let rc_rows = dr_rows(&data.mpdta, false, true);
    let rc_config = dr_config(ComparisonGroup::NotYetTreated, BasePeriod::Varying);
    out.push((
        "att_gt_dr_rc",
        digest(&estimate_att_gt_dr_with_influence(&rc_rows, rc_config)),
    ));
    out.push((
        "att_gt_dr_rc_efficient",
        digest(&estimate_att_gt_dr_efficient_with_influence(
            &rc_rows, rc_config,
        )),
    ));
    out.push((
        "att_gt_ipw_rc",
        digest(&estimate_att_gt_ipw_with_influence(&rc_rows, rc_config)),
    ));
    out.push((
        "att_gt_or_rc",
        digest(&estimate_att_gt_or_with_influence(&rc_rows, rc_config)),
    ));
    let efficient = estimate_att_gt_efficient_dr(
        &panel_rows,
        dr_config(ComparisonGroup::NeverTreated, BasePeriod::Varying),
    );
    out.push(("att_gt_efficient_dr", digest(&efficient)));
    if let Ok(efficient) = &efficient {
        out.push((
            "aggregate_efficient_event_time",
            digest(&aggregate_efficient_att_gt_event_time(
                efficient,
                AttGtAggregationConfig {
                    confidence_level: inference,
                    weighting: AttGtAggregationWeighting::ByTreatedCount,
                },
            )),
        ));
    }

    // Two-period estimators on one cell.
    let (panel_cell, rc_cell) = cell(&data.mpdta);
    let drdid = DrDidConfig::builder().build();
    out.push((
        "drdid_panel",
        digest(&estimate_drdid_panel(&panel_cell, drdid)),
    ));
    out.push((
        "drdid_improved_panel",
        digest(&estimate_drdid_improved_panel(&panel_cell, drdid)),
    ));
    out.push((
        "drdid_rc",
        digest(&estimate_drdid_repeated_cross_section(&rc_cell, drdid)),
    ));
    out.push((
        "drdid_improved_rc",
        digest(&estimate_drdid_improved_repeated_cross_section(
            &rc_cell, drdid,
        )),
    ));
    out.push((
        "drdid_rc_efficient",
        digest(&estimate_drdid_repeated_efficient(&rc_cell, drdid)),
    ));
    let two_by_two = rc_cell
        .iter()
        .map(|row| PanelObservation {
            treated: row.treated,
            post_period: row.post_period,
            outcome: row.outcome,
            weight: row.weight,
        })
        .collect::<Vec<_>>();
    out.push((
        "two_by_two",
        digest(&estimate_att_two_by_two(
            &two_by_two,
            DidConfig {
                confidence_level: 0.95,
            },
        )),
    ));
    let treated = panel_cell.iter().map(|r| r.treated).collect::<Vec<_>>();
    let delta = panel_cell
        .iter()
        .map(|r| r.delta_outcome)
        .collect::<Vec<_>>();
    let weights = panel_cell.iter().map(|r| r.weight).collect::<Vec<_>>();
    let design = panel_cell
        .iter()
        .flat_map(|r| [r.covariates[0], r.covariates[1], r.covariates[1] * 2.0])
        .collect::<Vec<_>>();
    let input = PanelFlatInput {
        treated: &treated,
        delta_outcome: &delta,
        weight: &weights,
        design_matrix_flat: &design,
        feature_count: 3,
    };
    out.push((
        "panel_nuisance",
        digest(&fit_drdid_panel_nuisance(input, drdid)),
    ));
    out.push((
        "panel_propensity",
        digest(&fit_drdid_panel_propensity_coefficients(input, drdid)),
    ));
    out.push((
        "independent_columns",
        digest(&independent_design_columns(
            &design,
            3,
            &weights,
            DesignPruning::PerArm { treated: &treated },
            1e-7,
        )),
    ));

    // Triple differences, continuous treatment, compositional changes.
    out.push(("dr_ddd", digest(&estimate_dr_ddd(&data.triple, drdid))));
    out.push((
        "acrt_sieve",
        digest(&estimate_acrt_sieve(
            &data.continuous,
            &PolynomialBasis::new(3),
        )),
    ));
    out.push((
        "did_cc_robust",
        digest(&estimate_did_cc_robust(&data.cc, DidCcConfig::default())),
    ));
    out.push((
        "did_cc_stationary",
        digest(&estimate_did_cc_stationary(
            &data.cc,
            DidCcConfig::default(),
        )),
    ));
    out.push((
        "did_cc_test",
        digest(&test_did_cc_stationarity(&data.cc, DidCcConfig::default())),
    ));

    // HonestDiD, on the paths Study I calls.
    let honest = &data.honest;
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
                    ..RelativeMagnitudeConfidenceSetConfig::from_inference(inference)
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
                ..RelativeMagnitudeConfidenceSetConfig::from_inference(inference)
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
    ("att_gt", 0x6772d3675b0dfc97),
    ("att_gt_efficient", 0x9d5279a72a4d9451),
    ("att_gt_dr_panel_universal_never", 0xcc70fa0ba55aa346),
    ("att_gt_dr_panel_varying_notyet", 0xa43a8b6e21a5fcaf),
    ("aggte_simple", 0xe09259f2e1999f54),
    ("aggte_dynamic", 0x5084f4b98e6935c7),
    ("aggte_dynamic_balanced", 0xa4683905d2502c2e),
    ("aggte_group", 0x3982cc8cd68fd88f),
    ("aggte_calendar", 0x48a620d1d6c8be68),
    ("event_time_equal", 0x88799cf0bd43632c),
    ("cohort_equal", 0xb6564651642a8b3e),
    ("calendar_equal", 0x5550c4c945db3dd8),
    ("overall_equal", 0xe5df63e7e955aa72),
    ("event_time_count", 0x4b2c00d196fad5e3),
    ("cohort_count", 0xd89203a004d02d76),
    ("calendar_count", 0x0103ca655d5581d5),
    ("overall_count", 0xf4fe93cf293f1452),
    ("event_time_weight", 0x4f135ec56c2eb289),
    ("cohort_weight", 0xdb165bf22a8d8d03),
    ("calendar_weight", 0x2290abf831fbe7c7),
    ("overall_weight", 0x0b51ceaead5c5853),
    ("aggte_dynamic_clustered", 0x1f6b80309cfe7783),
    ("mboot_bands", 0xb8ba24d8c5f360f0),
    ("mboot_bands_clustered", 0x9c237ed5f361ca27),
    ("simultaneous_bands", 0xc259eb6c1fd4c655),
    ("clustered_standard_errors", 0xe4983a1c4c9fce0a),
    ("clustered_standard_error_from_index", 0xc62a1ff2fb3a07c3),
    ("att_gt_dr_rc", 0x8bc496e79ac189c3),
    ("att_gt_dr_rc_efficient", 0x9b0150e0964a97e5),
    ("att_gt_ipw_rc", 0xf21add79195a712f),
    ("att_gt_or_rc", 0x2ee9cccd9fa411dd),
    ("att_gt_efficient_dr", 0x8edc15dd7bc5bd6d),
    ("aggregate_efficient_event_time", 0x89b44752f854e12c),
    ("drdid_panel", 0x9a355276778614a9),
    ("drdid_improved_panel", 0x9731c94f9fddc514),
    ("drdid_rc", 0x5e6535b613588f15),
    ("drdid_improved_rc", 0x922cd1a099dcb53a),
    ("drdid_rc_efficient", 0xad58a807f94f69fd),
    ("two_by_two", 0x3614ceb4ac377093),
    ("panel_nuisance", 0xd0887cbdb32ad261),
    ("panel_propensity", 0x03921b18f042d985),
    ("independent_columns", 0x45f2030fbd171265),
    ("dr_ddd", 0x86583954e214b213),
    ("acrt_sieve", 0x304369db872c3d81),
    ("did_cc_robust", 0x4c06ee7e0a32a429),
    ("did_cc_stationary", 0x21ff0ebd15c70221),
    ("did_cc_test", 0x4c05eb0a4dc3ab4d),
    ("honest_original", 0xa7bad0f94eb4a945),
    ("honest_rm_identified", 0xfb6764de8a7ff5f5),
    ("honest_rm_conditional", 0x045ad12120ee75ac),
    ("honest_smoothness_m", 0xefb8e9c809872f59),
    ("honest_smoothness_summary", 0xdea5be26ca16ec1c),
    ("honest_rm_summary_many", 0x9b83003df87a07e8),
    ("honest_joint_bonferroni", 0x3dacf805663ed12e),
    ("honest_joint_gaussian", 0xc186649eb5d32cfd),
    ("honest_workflow", 0x92cc2cd5dfaddc54),
    ("honest_directional_gaussian", 0x5ac6f9d42bfdfd45),
];

#[test]
fn estimator_outputs_are_bit_identical_to_the_recorded_hashes() {
    let data = data();
    let runs = on_threads(&[1, 4, 12], || cases(&data));
    for run in &runs[1..] {
        assert_eq!(run, &runs[0], "the hashes depend on the thread count");
    }
    let current = &runs[0];
    if current
        .iter()
        .map(|(n, h)| (*n, *h))
        .ne(GOLDEN.iter().copied())
    {
        for (name, hash) in current {
            println!("    (\"{name}\", {hash:#018x}),");
        }
    }
    assert_eq!(current.as_slice(), GOLDEN);
}
