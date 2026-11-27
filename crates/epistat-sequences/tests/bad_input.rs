//! The entry points that take weights meet bad input with an error of kind
//! input, never a panic.

use epistat_sequences::{
    Distance, FitSettings, Panel, fit_typology, indel_for, substitution_costs, transition_costs,
};
use epistat_testkit::{BadInput, assert_rejects_bad_input};

const ENCODED: [u8; 20] = [0, 1, 2, 0, 1, 1, 2, 0, 0, 2, 2, 1, 0, 0, 1, 0, 2, 1, 1, 2];
const WEIGHTS: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];

fn panel() -> Panel<'static> {
    Panel::new(&ENCODED, 4, 3, None).unwrap()
}

#[test]
fn weights_of_a_typology_are_checked() {
    let panel = panel();
    for distance in [
        Distance::Hamming,
        Distance::DynamicHamming,
        Distance::OptimalMatching,
    ] {
        let settings = FitSettings {
            distance,
            k: 2,
            restarts: 2,
            max_iterations: 10,
        };
        assert_rejects_bad_input("fit_typology", |bad| match bad {
            BadInput::Unsorted => None,
            _ => Some(fit_typology(&panel, &bad.column(&WEIGHTS), settings, 1)),
        });
    }
}

#[test]
fn weights_of_the_cost_helpers_are_checked() {
    let panel = panel();
    assert_rejects_bad_input("transition_costs", |bad| match bad {
        BadInput::Empty | BadInput::Mismatched => {
            Some(transition_costs(&panel, &bad.column(&WEIGHTS)))
        }
        _ => None,
    });
    assert_rejects_bad_input("substitution_costs", |bad| match bad {
        BadInput::Empty | BadInput::Mismatched => Some(substitution_costs(
            ENCODED.chunks(4),
            &bad.column(&WEIGHTS),
            3,
        )),
        _ => None,
    });
    assert_rejects_bad_input("indel_for", |bad| match bad {
        BadInput::Empty | BadInput::Mismatched => {
            Some(indel_for(&bad.column(&[0.0, 1.0, 1.0, 0.0]), 2))
        }
        _ => None,
    });
}
