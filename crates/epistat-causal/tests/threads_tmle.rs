//! The targeted estimators give identical bits in rayon pools of 1, 2, 3, 4 and 8
//! threads.

#[expect(
    dead_code,
    reason = "the augmented estimators' flatten is not used here"
)]
mod sim;
mod tmle_sim;

use epistat_testkit::{POOLS, assert_same_on_pools};

#[test]
fn estimators_are_the_same_on_every_pool() {
    let s = sim::simulate();
    for case in tmle_sim::CASES {
        assert_same_on_pools(&POOLS, || tmle_sim::run(&s, case));
    }
}
