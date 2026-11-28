//! The entry points that take weights meet bad input with an error of kind
//! input and the code of the shared check, never a panic.
//!
//! An empty weight column is a column of the wrong length here, since the
//! panel fixes how many weights there are, so it has the code of a length
//! error.

use epistat_core::Bootstrap;
use epistat_sequences::positionwise::{self, silhouette};
use epistat_sequences::{
    Distance, FitSettings, NullModel, Panel, Restarts, fit_typology, indel_for, null_sequences,
    null_silhouettes, partition_agreement, refit_agreements, substitution_costs, transition_costs,
};
use epistat_testkit::{BadInput, assert_rejects_bad_input_with_codes};

const ENCODED: [u8; 20] = [0, 1, 2, 0, 1, 1, 2, 0, 0, 2, 2, 1, 0, 0, 1, 0, 2, 1, 1, 2];
const WEIGHTS: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
const LABELS: [u32; 5] = [0, 1, 0, 1, 1];

fn panel() -> Panel<'static> {
    Panel::new(&ENCODED, 4, 3, None).unwrap()
}

/// The code of each kind of bad weight column.
fn code(bad: BadInput) -> &'static str {
    match bad {
        BadInput::Empty => "input.length",
        other => other.code(),
    }
}

/// The kinds a weight column can carry: order does not matter to weights.
fn weights(bad: BadInput) -> Option<Vec<f64>> {
    (bad != BadInput::Unsorted).then(|| bad.column(&WEIGHTS))
}

/// The kinds that only the length of a column can carry.
fn length_only(bad: BadInput) -> Option<Vec<f64>> {
    matches!(bad, BadInput::Empty | BadInput::Mismatched).then(|| bad.column(&WEIGHTS))
}

fn settings(distance: Distance) -> FitSettings {
    FitSettings {
        distance,
        k: 2,
        restarts: 2,
        max_iterations: 10,
    }
}

#[test]
fn weights_of_a_typology_are_checked() {
    let panel = panel();
    for distance in [
        Distance::Hamming,
        Distance::DynamicHamming,
        Distance::OptimalMatching,
    ] {
        assert_rejects_bad_input_with_codes("fit_typology", code, |bad| {
            weights(bad).map(|w| fit_typology(&panel, &w, settings(distance), 1))
        });
        assert_rejects_bad_input_with_codes("null_silhouettes", code, |bad| {
            weights(bad).map(|w| {
                null_silhouettes(
                    &panel,
                    &w,
                    NullModel::StateIndependence,
                    settings(distance),
                    Bootstrap::new(2, 1),
                )
            })
        });
    }
}

#[test]
fn weights_of_the_replicate_loops_are_checked() {
    let panel = panel();
    assert_rejects_bad_input_with_codes("refit_agreements", code, |bad| {
        weights(bad).map(|w| {
            refit_agreements(
                &panel,
                &w,
                &WEIGHTS,
                &LABELS,
                settings(Distance::Hamming),
                1,
            )
        })
    });
    assert_rejects_bad_input_with_codes("null_sequences", code, |bad| {
        weights(bad).map(|w| null_sequences(&panel, &w, NullModel::FirstOrderMarkov, 1))
    });
    assert_rejects_bad_input_with_codes("partition_agreement", code, |bad| {
        weights(bad).map(|w| partition_agreement(&LABELS, &LABELS, &w))
    });
}

#[test]
fn weights_of_a_position_wise_fit_are_checked() {
    let panel = panel();
    let restarts = Restarts::from_seed(1, 2).unwrap();
    assert_rejects_bad_input_with_codes("positionwise::kmedoids", code, |bad| {
        weights(bad).map(|w| positionwise::kmedoids(&panel, &w, 2, &restarts, 10))
    });
    assert_rejects_bad_input_with_codes("positionwise::silhouette", code, |bad| {
        weights(bad).map(|w| silhouette(&panel, &w, &LABELS))
    });
}

#[test]
fn weights_of_the_cost_helpers_are_checked() {
    let panel = panel();
    assert_rejects_bad_input_with_codes("transition_costs", code, |bad| {
        weights(bad).map(|w| transition_costs(&panel, &w))
    });
    assert_rejects_bad_input_with_codes("substitution_costs", code, |bad| {
        weights(bad).map(|w| substitution_costs(ENCODED.chunks(4), &w, 3))
    });
    assert_rejects_bad_input_with_codes("indel_for", code, |bad| {
        length_only(bad).map(|_| {
            let table = bad.column(&[0.0, 1.0, 1.0, 0.0]);
            indel_for(&table, 2)
        })
    });
}

#[test]
fn a_panel_refuses_a_ragged_or_empty_shape() {
    use epistat_core::Classify;
    let code = |result: Result<Panel<'_>, epistat_sequences::Error>| result.err().map(|e| e.code());
    assert_eq!(
        code(Panel::new(&ENCODED[..19], 4, 3, None)),
        Some("sequences.ragged")
    );
    assert_eq!(code(Panel::new(&ENCODED, 0, 3, None)), Some("input.domain"));
    assert_eq!(
        Distance::named("lcs").unwrap_err().code(),
        "sequences.unknown_distance"
    );
}
