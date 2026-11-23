//! The imputation does not depend on the number of threads, and one fixed run
//! has a pinned fingerprint.
//!
//! The data have a continuous column imputed by `norm`, one by predictive
//! mean matching, a binary one by logistic regression, a nominal one by
//! multinomial regression, a sampled column and a passive column. Every
//! imputed value and every chain statistic goes into the comparison, so a
//! draw that depended on how rows or chains were divided would show.

use epistat_impute::{
    Column, ImputeOptions, ImputeSpec, Imputed, Incomplete, Method, PassiveFn, impute,
};
use epistat_testkit::{Fingerprint, SplitMix64, assert_same_on_pools, assert_same_on_threads};

fn normal(rng: &mut SplitMix64) -> f64 {
    let (u, v) = (1.0 - rng.unit(), rng.unit());
    (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
}

/// The data and spec of the run.
fn problem(n: usize) -> (Incomplete, ImputeSpec) {
    let mut rng = SplitMix64::new(20_261_123);
    let (mut a, mut b, mut c, mut d) = (vec![0.0; n], vec![0u8; n], vec![0.0; n], vec![0.0; n]);
    let (mut e, mut f) = (vec![0u32; n], vec![0u8; n]);
    let mut g = vec![0.0; n];
    for i in 0..n {
        a[i] = normal(&mut rng);
        b[i] = u8::from(rng.unit() < 0.5);
        c[i] = (0.3 * a[i] + 0.5 * normal(&mut rng)).exp();
        d[i] = 1.0 + 0.5 * a[i] - 0.3 * f64::from(b[i]) + normal(&mut rng);
        e[i] = u32::try_from(rng.below(3)).unwrap();
        f[i] = u8::from(rng.unit() < 1.0 / (1.0 + (-(0.4 * a[i] - 0.2)).exp()));
        g[i] = a[i] * 2.0;
    }
    let miss = |rng: &mut SplitMix64, p: f64| (0..n).map(|_| rng.unit() < p).collect::<Vec<bool>>();
    let none = vec![false; n];
    let masks = vec![
        none.clone(),
        none.clone(),
        miss(&mut rng, 0.3),
        miss(&mut rng, 0.25),
        miss(&mut rng, 0.2),
        miss(&mut rng, 0.2),
        miss(&mut rng, 0.15),
    ];
    let data = Incomplete::new(
        vec![
            Column::Continuous(a),
            Column::Binary(b),
            Column::Continuous(c),
            Column::Continuous(d),
            Column::Nominal {
                codes: e,
                levels: 3,
            },
            Column::Binary(f),
            Column::Continuous(g),
        ],
        &masks,
    )
    .unwrap();
    let mut spec = ImputeSpec::default_for(&data);
    spec.methods[2] = Some(Method::pmm());
    spec.methods[3] = Some(Method::Norm);
    spec.methods[6] = Some(Method::Passive(PassiveFn::new(|cols| {
        cols[0].iter().zip(&cols[3]).map(|(x, y)| x + y).collect()
    })));
    // The visit order puts the passive column after the columns it uses.
    spec.visit_order = vec![2, 3, 4, 5, 6];
    spec.predictors[6] = vec![];
    (data, spec)
}

fn flatten(imputed: &Imputed) -> Vec<f64> {
    let mut out = Vec::new();
    for j in 0..imputed.values.len() {
        for set in &imputed.values[j] {
            out.extend_from_slice(set);
        }
        out.extend_from_slice(&imputed.chain_mean[j]);
        out.extend(
            imputed.chain_variance[j]
                .iter()
                .map(|v| v.unwrap_or(f64::NAN)),
        );
    }
    out.push(imputed.n_aliased_dropped as f64);
    out.push(imputed.n_augmented as f64);
    out
}

fn run(seed: u64) -> Imputed {
    let (data, spec) = problem(1_500);
    impute(
        &data,
        &spec,
        &ImputeOptions {
            m: 6,
            max_iterations: 3,
            seed,
        },
    )
    .unwrap()
}

#[test]
fn imputed_values_do_not_depend_on_the_thread_count() {
    assert_same_on_threads(|| flatten(&run(11)));
    assert_same_on_pools(&[1, 3, 5], || flatten(&run(12)));
}

#[test]
fn the_seed_changes_the_values_and_a_repeat_does_not() {
    let (a, b, c) = (flatten(&run(1)), flatten(&run(1)), flatten(&run(2)));
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn golden_fingerprint() {
    let values = flatten(&run(20_261_123));
    let hash = Fingerprint::new().f64s(&values).finish();
    assert_eq!(hash, GOLDEN, "{hash:#018x}");
}

const GOLDEN: u64 = 0x302e_bd21_86b8_c47c;
