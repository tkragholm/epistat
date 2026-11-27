//! The Aalen-Johansen step gives identical bits in rayon pools of 1, 2, 3, 4
//! and 8 threads.

use epistat_rng::Stream;
use epistat_survival::lowlevel::advance;
use epistat_testkit::{POOLS, assert_same_on_pools};

/// `advance` is a single sequential step, and every estimator is built from
/// it. A long chain of steps shows it carries no state between calls.
#[test]
fn aalen_johansen_advance() {
    let mut rng = Stream::seeded(14);
    let steps: Vec<(f64, f64, [f64; 2])> = (0..5000)
        .map(|_| {
            let at_risk = 1.0 + rng.uniform() * 100.0;
            let d1 = rng.uniform() * 0.4;
            let d2 = rng.uniform() * 0.4;
            (at_risk, d1 + d2, [d1, d2])
        })
        .collect();
    assert_same_on_pools(&POOLS, || {
        let mut survival = 1.0;
        let mut incidence = [0.0; 2];
        for (at_risk, d_any, d_causes) in &steps {
            advance(&mut survival, *at_risk, *d_any, d_causes, &mut incidence).unwrap();
        }
        vec![survival, incidence[0], incidence[1]]
    });
}
