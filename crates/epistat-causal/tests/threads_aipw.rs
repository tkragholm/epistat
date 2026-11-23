//! The augmented estimators give identical bits in rayon pools of 1, 2, 3, 4 and 8
//! threads.

mod sim;

use epistat_causal::aipw::{AipwInput, AipwOptions, Target, aipw, aipw_glm};
use epistat_glm::Family;
use epistat_testkit::{POOLS, assert_same_on_pools};

#[test]
fn estimators_are_the_same_on_every_pool() {
    let s = sim::simulate();
    for target in [Target::Ate, Target::Att, Target::Atc] {
        assert_same_on_pools(&POOLS, || {
            let r = aipw_glm(
                &s.input(),
                Family::Binomial,
                target,
                &AipwOptions::default(),
            )
            .unwrap();
            sim::flatten(&r)
        });
    }
    assert_same_on_pools(&POOLS, || {
        let m = s.models();
        let r = aipw(
            &AipwInput {
                outcome: &s.y,
                treatment: &s.a,
                mu1: &m.0,
                mu0: &m.1,
                propensity: &m.2,
                weight: Some(&s.weight),
                cluster: Some(&s.cluster),
            },
            Target::Ate,
            &AipwOptions {
                truncation: Some((0.05, 0.95)),
                ..AipwOptions::default()
            },
        )
        .unwrap();
        sim::flatten(&r)
    });
}
