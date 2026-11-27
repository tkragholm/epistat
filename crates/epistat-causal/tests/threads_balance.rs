//! Entropy balancing and CBPS give identical bits in rayon pools of 1, 2, 3, 4 and 8
//! threads. The simulated data have more rows than one block of rows.

mod balance_sim;

use epistat_causal::Estimand;
use epistat_causal::balance::{
    CbpsInput, CbpsOptions, EntropyInput, EntropyOptions, Identification, balance_columns, cbps,
    entropy_balance,
};
use epistat_testkit::{POOLS, assert_same_on_pools};

#[test]
fn entropy_balancing_is_the_same_on_every_pool() {
    let s = balance_sim::simulate(4500);
    let cols = balance_columns(s.c.as_ref(), 2, true).unwrap();
    for target in [Estimand::Att, Estimand::Ate, Estimand::Atc] {
        assert_same_on_pools(&POOLS, || {
            let r = entropy_balance(
                &EntropyInput {
                    treatment: &s.a,
                    covariates: cols.as_ref(),
                    sample_weight: Some(&s.s),
                    base_weight: Some(&s.q),
                },
                &EntropyOptions {
                    estimand: target,
                    ..EntropyOptions::default()
                },
            )
            .unwrap();
            let mut v = balance_sim::flatten(&r.weights, &s);
            v.push(r.max_imbalance);
            v
        });
    }
}

#[test]
fn cbps_is_the_same_on_every_pool() {
    let s = balance_sim::simulate(4500);
    for (target, identification, two_step) in [
        (Estimand::Ate, Identification::Exact, true),
        (Estimand::Att, Identification::Over, true),
        (Estimand::Atc, Identification::Over, false),
    ] {
        assert_same_on_pools(&POOLS, || {
            let r = cbps(
                &CbpsInput {
                    treatment: &s.a,
                    design: s.x.as_ref(),
                    sample_weight: Some(&s.s),
                },
                &CbpsOptions {
                    estimand: target,
                    identification,
                    two_step,
                    ..CbpsOptions::default()
                },
            )
            .unwrap();
            let mut v = balance_sim::flatten(&r.weights, &s);
            v.extend(&r.coefficients);
            v.extend(&r.propensity);
            v.extend([r.objective.unwrap_or(f64::NAN), r.max_imbalance]);
            v
        });
    }
}
