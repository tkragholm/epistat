//! Edge cases of the pseudo-values of a multi-state process: what is refused,
//! that the bound on influence held at once changes no value, and that the
//! cumulative hazard with no transition in a stratum is 0 there.

use epistat_survival::SurvData;
use epistat_survival::pseudo::{StateOptions, StateTarget, pseudo_states};

// Three subjects in state 1, one moving to 2 at 2, one to 3 at 3, one
// censored at 4, and a second stratum of two subjects with one move to 2.
const START: [f64; 9] = [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
const STOP: [f64; 9] = [2.0, 5.0, 3.0, 4.0, 3.0, 1.0, 4.0, 6.0, 2.0];
const STATUS: [i32; 9] = [2, 3, 3, 0, 0, 2, 0, 0, 2];
const ID: [u32; 9] = [1, 1, 2, 3, 4, 5, 5, 6, 7];
const ISTATE: [u32; 9] = [1, 2, 1, 1, 1, 1, 2, 1, 1];
const STRATUM: [u32; 9] = [1, 1, 1, 1, 1, 2, 2, 2, 2];

fn data() -> SurvData<'static> {
    SurvData::builder(&STOP, &STATUS)
        .n_codes(3)
        .start(&START)
        .id(&ID)
        .istate(&ISTATE)
        .stratum(&STRATUM)
        .build()
        .unwrap()
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn what_is_refused() {
    let d = data();
    let o = StateOptions::new(StateTarget::Occupation, 0.0);
    assert!(pseudo_states(&d, &[], &o).is_err());
    assert!(pseudo_states(&d, &[2.0, 1.0], &o).is_err());
    assert!(
        pseudo_states(
            &d,
            &[1.0],
            &StateOptions::new(StateTarget::Occupation, f64::NAN)
        )
        .is_err()
    );
    // A time in state before the origin.
    assert!(
        pseudo_states(
            &d,
            &[0.5],
            &StateOptions::new(StateTarget::TimeInState, 1.0)
        )
        .is_err()
    );
    // An origin after the first exit time.
    assert!(pseudo_states(&d, &[2.0], &StateOptions::new(StateTarget::Occupation, 1.5)).is_err());
    // A subject with rows in two strata.
    let crossing = [1u32, 1, 1, 1, 1, 2, 1, 2, 2];
    let c = SurvData::builder(&STOP, &STATUS)
        .n_codes(3)
        .start(&START)
        .id(&ID)
        .istate(&ISTATE)
        .stratum(&crossing)
        .build()
        .unwrap();
    assert!(pseudo_states(&c, &[2.0], &o).is_err());
    // The influence at one time is 4 clusters times 3 states in the first
    // stratum, so a bound below 12 is refused.
    let mut small = StateOptions::new(StateTarget::TimeInState, 0.0);
    small.max_influence_values = Some(11);
    assert!(pseudo_states(&d, &[3.0], &small).is_err());
    small.max_influence_values = Some(12);
    assert!(pseudo_states(&d, &[3.0], &small).is_ok());
}

#[test]
fn the_bound_changes_no_value() {
    let d = data();
    let times = [1.0, 2.5, 4.0, 7.0];
    for target in [StateTarget::Occupation, StateTarget::TimeInState] {
        let whole = pseudo_states(&d, &times, &StateOptions::new(target, 0.0)).unwrap();
        for limit in [12, 24, 100] {
            let mut o = StateOptions::new(target, 0.0);
            o.max_influence_values = Some(limit);
            let chunked = pseudo_states(&d, &times, &o).unwrap();
            assert_eq!(
                bits(&whole.values),
                bits(&chunked.values),
                "{target:?} {limit}"
            );
        }
    }
}

#[test]
fn a_transition_absent_from_a_stratum_has_the_value_zero_there() {
    let d = data();
    let p = pseudo_states(
        &d,
        &[1.0, 6.0],
        &StateOptions::new(StateTarget::CumulativeHazard, 0.0),
    )
    .unwrap();
    // Stratum 1 has 1->2, 1->3 and 2->3, stratum 2 only 1->2.
    assert_eq!(p.transitions, [(0, 1), (0, 2), (1, 2)]);
    let first_of_second: usize = p.groups[0].n_subjects;
    for c in 0..p.groups[1].n_subjects {
        for slot in 0..2 {
            for t in [1, 2] {
                let v = p.values[((first_of_second + c) * 2 + slot) * p.width + t];
                assert_eq!(v.to_bits(), 0.0f64.to_bits());
            }
        }
    }
    assert!(p.groups[1].estimate[p.width] > 0.0);
}
