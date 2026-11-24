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
    HonestJointPathConfig, HonestJointPathMethod, HonestMonotonicityDirection,
    HonestPostFunctional, HonestRelativeMagnitudeBound, HonestSensitivity, InferenceConfig,
    LeastFavorableSimulation, PanelFlatInput, PanelObservation, PolynomialBasis,
    RelativeMagnitudeConfidenceSetConfig, SensitivitySummaryMethod, TripleDidObservation,
    aggregate_att_gt, aggregate_att_gt_by_calendar_time, aggregate_att_gt_by_cohort,
    aggregate_att_gt_event_time_with_influence, aggregate_att_gt_overall,
    aggregate_efficient_att_gt_event_time,
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
    let mpdta: Mpdta = load_json("tests/data/cs_panel_dr_universal_ref.json");
    let honest: HonestRef = load_json("tests/data/honest_did_ref.json");
    let pre_periods = honest
        .pre_indices
        .iter()
        .map(|i| -i32::try_from(*i).expect("fits"))
        .collect();
    let post_periods = (0..honest.post_indices.len())
        .map(|i| i32::try_from(i).expect("fits"))
        .collect();
    let triple: TripleRef = load_json("tests/data/triple_did_ref.json");
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
    let continuous: ContRef = load_json("tests/data/contdid_ref.json");
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
    let cc: CcRef = load_json("tests/data/did_cc_ref_scaffold.json");
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
        drdid: drdid_config(),
    }
}

/// The estimators' settings with a 999-replicate bootstrap interval, so the
/// hashes pin the bootstrap path. `drdid_panel_analytic` pins the default
/// analytic interval.
fn drdid_config() -> DrDidConfig {
    DrDidConfig::builder().bootstrap_reps(999).build()
}

/// The compositional-change settings with bootstrap intervals, for the same
/// reason as [`drdid_config`].
fn did_cc_config() -> DidCcConfig {
    DidCcConfig {
        drdid: drdid_config(),
        bootstrap_confidence_intervals: true,
        ..DidCcConfig::default()
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
            digest(&aggregate_att_gt_by_cohort(
                &fitted.estimates,
                &fitted.influence_functions,
                &panel,
                aggregation,
            )),
        ));
        out.push((
            names[2],
            digest(&aggregate_att_gt_by_calendar_time(
                &fitted.estimates,
                &fitted.influence_functions,
                &panel,
                aggregation,
            )),
        ));
        out.push((
            names[3],
            digest(&aggregate_att_gt_overall(
                &fitted.estimates,
                &fitted.influence_functions,
                &panel,
                aggregation,
            )),
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
    let drdid = drdid_config();
    out.push((
        "drdid_panel",
        digest(&estimate_drdid_panel(&panel_cell, drdid)),
    ));
    out.push((
        "drdid_panel_analytic",
        digest(&estimate_drdid_panel(&panel_cell, DrDidConfig::default())),
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
        digest(&estimate_did_cc_robust(&data.cc, did_cc_config())),
    ));
    out.push((
        "did_cc_stationary",
        digest(&estimate_did_cc_stationary(
            &data.cc,
            did_cc_config(),
        )),
    ));
    out.push((
        "did_cc_test",
        digest(&test_did_cc_stationarity(&data.cc, did_cc_config())),
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
    ("att_gt", 0x6772_d367_5b0d_fc97),
    ("att_gt_efficient", 0x201f_58e9_78ec_8f24),
    ("att_gt_dr_panel_universal_never", 0xcc70_fa0b_a55a_a346),
    ("att_gt_dr_panel_varying_notyet", 0xa43a_8b6e_21a5_fcaf),
    ("aggte_simple", 0xe092_59f2_e199_9f54),
    ("aggte_dynamic", 0x5084_f4b9_8e69_35c7),
    ("aggte_dynamic_balanced", 0xa468_3905_d250_2c2e),
    ("aggte_group", 0x3982_cc8c_d68f_d88f),
    ("aggte_calendar", 0x48a6_20d1_d6c8_be68),
    ("event_time_equal", 0x8879_9cf0_bd43_632c),
    ("cohort_equal", 0x87d6_17a2_6507_6c7f),
    ("calendar_equal", 0xb6c5_7c1f_f5da_692e),
    ("overall_equal", 0x5f8c_8955_89bd_2aad),
    ("event_time_count", 0x4b2c_00d1_96fa_d5e3),
    ("cohort_count", 0x87d6_17a2_6507_6c7f),
    ("calendar_count", 0x23b1_6e04_5eda_c4ee),
    ("overall_count", 0x72de_8cd2_d121_f9f9),
    ("event_time_weight", 0x4f13_5ec5_6c2e_b289),
    ("cohort_weight", 0x87d6_17a2_6507_6c7f),
    ("calendar_weight", 0x1834_994b_8dea_7b1e),
    ("overall_weight", 0xd06d_1803_84a1_71bc),
    ("aggte_dynamic_clustered", 0x1f6b_8030_9cfe_7783),
    ("mboot_bands", 0xb8ba_24d8_c5f3_60f0),
    ("mboot_bands_clustered", 0x9c23_7ed5_f361_ca27),
    ("simultaneous_bands", 0xc259_eb6c_1fd4_c655),
    ("clustered_standard_errors", 0xe498_3a1c_4c9f_ce0a),
    ("clustered_standard_error_from_index", 0xc62a_1ff2_fb3a_07c3),
    ("att_gt_dr_rc", 0x8bc4_96e7_9ac1_89c3),
    ("att_gt_dr_rc_efficient", 0x9b01_50e0_964a_97e5),
    ("att_gt_ipw_rc", 0xf21a_dd79_195a_712f),
    ("att_gt_or_rc", 0x2ee9_cccd_9fa4_11dd),
    ("att_gt_efficient_dr", 0x74b6_3e4f_4e86_2f51),
    ("aggregate_efficient_event_time", 0x7693_9d38_4755_f4ab),
    ("drdid_panel", 0x9a35_5276_7786_14a9),
    ("drdid_panel_analytic", 0x3a8b_258f_748f_a7e0),
    ("drdid_improved_panel", 0x9731_c94f_9fdd_c514),
    ("drdid_rc", 0x5e65_35b6_1358_8f15),
    ("drdid_improved_rc", 0x922c_d1a0_99dc_b53a),
    ("drdid_rc_efficient", 0xad58_a807_f94f_69fd),
    ("two_by_two", 0x3614_ceb4_ac37_7093),
    ("panel_nuisance", 0xd088_7cbd_b32a_d261),
    ("panel_propensity", 0x0392_1b18_f042_d985),
    ("independent_columns", 0x45f2_030f_bd17_1265),
    ("dr_ddd", 0x8658_3954_e214_b213),
    ("acrt_sieve", 0xdbf7_0235_0d74_2314),
    ("did_cc_robust", 0x4c06_ee7e_0a32_a429),
    ("did_cc_stationary", 0x21ff_0ebd_15c7_0221),
    ("did_cc_test", 0x4c05_eb0a_4dc3_ab4d),
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
#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    ignore = "the hashes were recorded on aarch64 macOS; other platforms' maths libraries differ in the last bits"
)]
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
            let hex = format!("{hash:016x}");
            let groups: Vec<&str> = (0..4).map(|i| &hex[4 * i..4 * i + 4]).collect();
            println!("    (\"{name}\", 0x{}),", groups.join("_"));
        }
    }
    assert_eq!(current.as_slice(), GOLDEN);
}
