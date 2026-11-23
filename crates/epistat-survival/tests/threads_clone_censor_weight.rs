//! The clone-censor-weight analysis and its bootstrap give identical bits in
//! rayon pools of 1, 2, 3, 4 and 8 threads.

mod ccw_run;

use epistat_survival::weighting::clone::bootstrap_persons;
use epistat_testkit::{POOLS, assert_same_on_pools};

#[test]
fn the_analysis_is_the_same_on_every_pool() {
    let r = ccw_run::load();
    let d = ccw_run::data(&r);
    let c = ccw_run::cloned(&r, &d);
    assert_same_on_pools(&POOLS, || ccw_run::analyse(&r, &c, None).flatten());
    assert_same_on_pools(&POOLS, || {
        bootstrap_persons(c.n_persons, &[1, 2, 3, 4, 5], |counts| {
            Ok(ccw_run::analyse(&r, &c, Some(counts)).flatten())
        })
        .into_iter()
        .flat_map(|v| v.unwrap())
        .collect()
    });
}
