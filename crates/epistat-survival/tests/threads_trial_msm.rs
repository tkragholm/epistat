//! The weights and the outcome model of the sequential trials give identical bits
//! in rayon pools of 1, 2, 3, 4 and 8 threads.

mod trial_msm_run;

use epistat_testkit::{POOLS, assert_same_on_pools};

#[test]
fn the_analysis_is_the_same_on_every_pool() {
    let analyses = trial_msm_run::load();
    for a in &analyses {
        assert_same_on_pools(&POOLS, || trial_msm_run::run(a).flatten());
    }
}
