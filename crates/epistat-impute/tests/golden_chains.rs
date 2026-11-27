//! The bits of an imputation with several chains, recorded before the chains
//! shared the complete columns of the data instead of copying them.
//!
//! The data have a complete binary and a complete nominal predictor, a
//! continuous column imputed by predictive mean matching on drawn
//! coefficients, a binary one by logistic regression, a nominal one by
//! multinomial regression without drawn coefficients and a sampled one.
//! The fingerprint depends on the platform's `ln`, `exp` and `cos`, so it is
//! checked only on aarch64 macOS, where it was recorded. Elsewhere the test
//! computes it and, when `EPISTAT_HASH_OUT` is set, appends it to that file.

use epistat_impute::{
    Column, ImputeOptions, ImputeSpec, Imputed, Incomplete, MatchType, Method, impute,
};
use epistat_testkit::SplitMix64;

fn normal(rng: &mut SplitMix64) -> f64 {
    let (u, v) = (1.0 - rng.unit(), rng.unit());
    (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
}

fn problem(n: usize) -> (Incomplete, ImputeSpec) {
    let mut rng = SplitMix64::new(20_261_124);
    let mut x = vec![0.0; n];
    let mut b = vec![0u8; n];
    let mut k = vec![0u32; n];
    let mut inc = vec![0.0; n];
    let mut flag = vec![0u8; n];
    let mut cat = vec![0u32; n];
    let mut s = vec![0.0; n];
    for i in 0..n {
        x[i] = normal(&mut rng);
        b[i] = u8::from(rng.unit() < 0.4);
        k[i] = u32::try_from(rng.below(4)).unwrap();
        inc[i] = (2.0 + 0.4 * x[i] + 0.1 * f64::from(k[i]) + 0.5 * normal(&mut rng)).exp();
        flag[i] = u8::from(rng.unit() < 1.0 / (1.0 + (-(0.5 * x[i] - 0.3)).exp()));
        cat[i] = u32::try_from(rng.below(3)).unwrap();
        s[i] = normal(&mut rng);
    }
    let miss = |rng: &mut SplitMix64, p: f64| (0..n).map(|_| rng.unit() < p).collect::<Vec<bool>>();
    let none = vec![false; n];
    let masks = vec![
        none.clone(),
        none.clone(),
        none,
        miss(&mut rng, 0.3),
        miss(&mut rng, 0.2),
        miss(&mut rng, 0.25),
        miss(&mut rng, 0.1),
    ];
    let data = Incomplete::new(
        vec![
            Column::Continuous(x),
            Column::Binary(b),
            Column::Nominal {
                codes: k,
                levels: 4,
            },
            Column::Continuous(inc),
            Column::Binary(flag),
            Column::Nominal {
                codes: cat,
                levels: 3,
            },
            Column::Continuous(s),
        ],
        &masks,
    )
    .unwrap();
    let mut spec = ImputeSpec::default_for(&data);
    spec.methods[3] = Some(Method::Pmm {
        donors: 3,
        match_type: MatchType::Drawn,
    });
    spec.methods[4] = Some(Method::Logreg);
    spec.methods[5] = Some(Method::Polyreg { proper: false });
    spec.methods[6] = Some(Method::Sample);
    (data, spec)
}

fn flatten(imputed: &Imputed) -> Vec<f64> {
    let mut out = Vec::new();
    for j in 0..imputed.values.len() {
        out.extend(imputed.missing_rows[j].iter().map(|&r| r as f64));
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

#[test]
fn golden_fingerprint() {
    let (data, spec) = problem(4000);
    let imputed = impute(
        &data,
        &spec,
        &ImputeOptions {
            m: 5,
            max_iterations: 4,
            seed: 20_261_124,
        },
    )
    .unwrap();
    let values = flatten(&imputed);
    let hash = epistat_testkit::Fingerprint::new()
        .u64(values.len() as u64)
        .f64s(&values)
        .finish();
    epistat_testkit::hash_out("golden_fingerprint", hash);
    if epistat_testkit::RECORDED_PLATFORM {
        assert_eq!(hash, GOLDEN, "{hash:#018x}");
    }
}

const GOLDEN: u64 = 0xe70e_91fe_78d7_f186;
