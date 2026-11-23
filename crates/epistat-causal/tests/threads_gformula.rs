//! The g-formula gives identical bits in rayon pools of 1, 2, 3, 4 and 8
//! threads: with every person simulated once and with a number of simulated
//! persons that is not a multiple of the block size, with a bootstrap, and for
//! an outcome at the end of follow-up.

mod gformula_sim;

use epistat_causal::gformula::{GFormulaOptions, gformula};
use epistat_testkit::{POOLS, assert_same_on_pools};
use gformula_sim::{complete, covariates, flatten, interventions, simulate};

#[test]
fn the_g_formula_is_the_same_on_every_pool() {
    let sim = simulate(700, 4, 11);
    let baseline = sim.baseline();
    let tv = sim.time_varying();
    let data = sim.data(&baseline, &tv);
    let outcome = sim.survival();
    for simulated in [None, Some(1_300)] {
        let options = GFormulaOptions {
            simulated,
            bootstrap: 6,
            ..GFormulaOptions::default()
        };
        assert_same_on_pools(&POOLS, || {
            let g = gformula(&data, &covariates(), &outcome, &interventions(), &options).unwrap();
            flatten(&g)
        });
    }
    let end = complete(&sim, 4);
    let baseline = end.baseline();
    let tv = end.time_varying();
    let data = end.data(&baseline, &tv);
    let outcome = end.binary_end();
    let options = GFormulaOptions {
        bootstrap: 4,
        ..GFormulaOptions::default()
    };
    assert_same_on_pools(&POOLS, || {
        let g = gformula(&data, &covariates(), &outcome, &interventions(), &options).unwrap();
        flatten(&g)
    });
}
