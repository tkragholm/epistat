//! The augmented risk gives identical bits in rayon pools of 1, 2, 3, 4 and 8
//! threads.

mod aipw_surv_sim;

use epistat_survival::aipw::Estimator;
use epistat_testkit::{POOLS, assert_same_on_pools};

#[test]
fn the_risks_are_the_same_on_every_pool() {
    let s = aipw_surv_sim::simulate(1200, 7);
    for estimator in [Estimator::Aiptw, Estimator::Iptw] {
        for km in [true, false] {
            assert_same_on_pools(&POOLS, || {
                aipw_surv_sim::flatten(&aipw_surv_sim::run(&s, estimator, km, &[2.0, 5.0, 8.0]))
            });
        }
    }
}
