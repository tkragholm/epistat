//! The fixed-effects solver on poorly connected designs: workers who move
//! between neighbouring firms on a chain, with a third factor of years or of
//! occupations, a path of firms, and a sparse random worker-firm graph. Both
//! solvers are checked against the exact projections on the dummies (the QR
//! of `lm.fit`) and set against `fixest::feols` 0.14.2 at the same
//! tolerance. See `tests/data/absorb_hard.R`.

use epistat_glm::{Absorb, AbsorbMethod, FixedEffects, Singletons};
use epistat_testkit::{assert_same_on_threads, load_json};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    fixest: String,
    tolerance: f64,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    factors: Vec<Vec<u32>>,
    weights: Option<Vec<f64>>,
    y: Vec<f64>,
    kept: Vec<usize>,
    exact_residuals: Vec<f64>,
    lambda_plus: f64,
    fixest_iterations: usize,
    fixest_error: f64,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/absorb_hard.json"
    ))
}

const EPS: f64 = f64::EPSILON;

const METHODS: [AbsorbMethod; 2] = [AbsorbMethod::Projections, AbsorbMethod::ConjugateGradient];

fn fixed_effects(case: &Case) -> FixedEffects {
    let factors: Vec<&[u32]> = case.factors.iter().map(Vec::as_slice).collect();
    FixedEffects::new(&factors, case.weights.as_deref(), Singletons::Remove).unwrap()
}

fn control(method: AbsorbMethod, tolerance: f64) -> Absorb {
    Absorb {
        tolerance,
        max_iterations: 100_000,
        method,
    }
}

#[test]
fn both_solvers_meet_the_criterion_and_its_bound_on_hard_designs() {
    // At the returned point every weighted level mean of the residual is at
    // most tol * s; recomputed here, each mean carries the rounding of a sum
    // of at most `largest group` terms, so the check allows that many ulps
    // of the largest residual. The error e = r - r* then satisfies
    // ||e||_W <= ||gamma|| / sqrt(lambda_+), gamma the level sums of w r
    // (module documentation of `absorb`), plus the error of the Householder
    // QR of the n x m dummy matrix, of order n m eps ||v|| (Higham, Accuracy
    // and Stability, thm 20.3).
    let reference = reference();
    assert_eq!(reference.fixest, "0.14.2");
    let tol = reference.tolerance;
    for case in &reference.cases {
        let fe = fixed_effects(case);
        assert_eq!(fe.rows(), case.kept.as_slice(), "{}", case.name);
        let w: Vec<f64> = fe
            .rows()
            .iter()
            .map(|&i| case.weights.as_ref().map_or(1.0, |w| w[i]))
            .collect();
        let n = w.len();
        let m: usize = fe.levels().iter().sum();
        let y: Vec<f64> = fe.rows().iter().map(|&i| case.y[i]).collect();
        let total: f64 = w.iter().sum();
        let mean = (0..n).map(|k| w[k] * y[k]).sum::<f64>() / total;
        let norm_v = (0..n)
            .map(|k| w[k] * (y[k] - mean).powi(2))
            .sum::<f64>()
            .sqrt();
        for method in METHODS {
            let out = fe.demean(&[&case.y], &control(method, tol)).unwrap();
            let r = out.columns.col_as_slice(0);
            let s = out.scale[0];
            let largest_r = r.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
            let mut gamma2 = 0.0;
            for q in 0..fe.n_factors() {
                let g = fe.levels()[q];
                let (mut sum, mut weight, mut count) =
                    (vec![0.0; g], vec![0.0; g], vec![0usize; g]);
                for (k, &c) in fe.codes(q).iter().enumerate() {
                    sum[c as usize] += w[k] * r[k];
                    weight[c as usize] += w[k];
                    count[c as usize] += 1;
                }
                let largest_group = count.iter().copied().max().unwrap() as f64;
                for c in 0..g {
                    let level_mean = sum[c] / weight[c];
                    assert!(
                        level_mean.abs() <= tol * s + largest_group * EPS * largest_r,
                        "{} {method:?} factor {q}: {level_mean:e} against {:e}",
                        case.name,
                        tol * s
                    );
                    gamma2 += sum[c] * sum[c];
                }
            }
            let error = (0..n)
                .map(|k| w[k] * (r[k] - case.exact_residuals[k]).powi(2))
                .sum::<f64>()
                .sqrt();
            let bound = gamma2.sqrt() / case.lambda_plus.sqrt() + (n * m) as f64 * EPS * norm_v;
            assert!(
                error <= bound,
                "{} {method:?}: error {error:e} above bound {bound:e}",
                case.name
            );
            let largest_error = (0..n)
                .map(|k| (r[k] - case.exact_residuals[k]).abs())
                .fold(0.0_f64, f64::max)
                / s;
            println!(
                "{:24} {method:?}: {} iterations, largest error {largest_error:.1e} of s; fixest {} iterations, {:.1e}",
                case.name, out.iterations[0], case.fixest_iterations, case.fixest_error
            );
        }
    }
}

#[test]
fn conjugate_gradients_need_fewer_sweeps_on_chains() {
    // An iteration of the projections takes at least two sweeps, one of the
    // conjugate gradients one sweep's work. On the chain and the path the
    // projections converge at the rate 1 - O(lambda_+) a sweep and the
    // conjugate gradients at 1 - O(sqrt(lambda_+)), so the gradients take
    // fewer sweeps.
    let reference = reference();
    for case in reference
        .cases
        .iter()
        .filter(|c| c.name.starts_with("chain") || c.name == "path")
    {
        let fe = fixed_effects(case);
        let tol = reference.tolerance;
        let projections = fe
            .demean(&[&case.y], &control(AbsorbMethod::Projections, tol))
            .unwrap();
        let gradients = fe
            .demean(&[&case.y], &control(AbsorbMethod::ConjugateGradient, tol))
            .unwrap();
        assert!(
            gradients.iterations[0] < 2 * projections.iterations[0],
            "{}: {} against {}",
            case.name,
            gradients.iterations[0],
            projections.iterations[0]
        );
    }
}

#[test]
fn the_solvers_agree_with_each_other_within_their_bounds() {
    // Each level mean of a residual is at most tol s, plus the rounding of
    // its sum of at most `largest group` terms, so ||gamma|| is at most
    // sqrt(sum_q sum_g W_q[g]^2) times that, and each residual lies within
    // ||gamma|| / sqrt(lambda_+) of the exact one in the W norm: within
    // twice that of each other.
    let reference = reference();
    let tol = 1e-12;
    for case in &reference.cases {
        let fe = fixed_effects(case);
        let a = fe
            .demean(&[&case.y], &control(AbsorbMethod::Projections, tol))
            .unwrap();
        let b = fe
            .demean(&[&case.y], &control(AbsorbMethod::ConjugateGradient, tol))
            .unwrap();
        let w: Vec<f64> = fe
            .rows()
            .iter()
            .map(|&i| case.weights.as_ref().map_or(1.0, |w| w[i]))
            .collect();
        let (mut level_weights2, mut largest_group) = (0.0, 0usize);
        for q in 0..fe.n_factors() {
            let (mut weight, mut count) = (vec![0.0; fe.levels()[q]], vec![0usize; fe.levels()[q]]);
            for (k, &c) in fe.codes(q).iter().enumerate() {
                weight[c as usize] += w[k];
                count[c as usize] += 1;
            }
            level_weights2 += weight.iter().map(|v| v * v).sum::<f64>();
            largest_group = largest_group.max(count.iter().copied().max().unwrap());
        }
        let s = a.scale[0];
        let ra = a.columns.col_as_slice(0);
        let rb = b.columns.col_as_slice(0);
        let largest_r = ra.iter().chain(rb).fold(0.0_f64, |m, &v| m.max(v.abs()));
        let distance = (0..w.len())
            .map(|k| w[k] * (ra[k] - rb[k]).powi(2))
            .sum::<f64>()
            .sqrt();
        let mean_bound = tol * s + largest_group as f64 * EPS * largest_r;
        let bound = 2.0 * mean_bound * level_weights2.sqrt() / case.lambda_plus.sqrt();
        assert!(
            distance <= bound,
            "{}: {distance:e} above {bound:e}",
            case.name
        );
    }
}

#[test]
fn conjugate_gradients_solve_from_level_sums() {
    // With b = D' W v the fitted D alpha is v less its exact residual, and
    // the criterion with the scale of the level means of b bounds the error
    // as in the first test.
    let reference = reference();
    let tol = 1e-10;
    for case in &reference.cases {
        let fe = fixed_effects(case);
        let w: Vec<f64> = fe
            .rows()
            .iter()
            .map(|&i| case.weights.as_ref().map_or(1.0, |w| w[i]))
            .collect();
        let y: Vec<f64> = fe.rows().iter().map(|&i| case.y[i]).collect();
        let n = w.len();
        let m: usize = fe.levels().iter().sum();
        let sums: Vec<Vec<f64>> = (0..fe.n_factors())
            .map(|q| {
                let mut s = vec![0.0; fe.levels()[q]];
                for (k, &c) in fe.codes(q).iter().enumerate() {
                    s[c as usize] += w[k] * y[k];
                }
                s
            })
            .collect();
        let refs: Vec<&[f64]> = sums.iter().map(Vec::as_slice).collect();
        let alpha = fe
            .solve_sums(&refs, &control(AbsorbMethod::ConjugateGradient, tol))
            .unwrap();
        let fitted = fe.fitted(&alpha);
        let r: Vec<f64> = y.iter().zip(&fitted).map(|(v, f)| v - f).collect();
        let mut gamma2 = 0.0;
        for q in 0..fe.n_factors() {
            let mut sum = vec![0.0; fe.levels()[q]];
            for (k, &c) in fe.codes(q).iter().enumerate() {
                sum[c as usize] += w[k] * r[k];
            }
            gamma2 += sum.iter().map(|v| v * v).sum::<f64>();
        }
        let norm_v = (0..n).map(|k| w[k] * y[k] * y[k]).sum::<f64>().sqrt();
        let error = (0..n)
            .map(|k| w[k] * (r[k] - case.exact_residuals[k]).powi(2))
            .sum::<f64>()
            .sqrt();
        let bound = gamma2.sqrt() / case.lambda_plus.sqrt() + (n * m) as f64 * EPS * norm_v;
        assert!(error <= bound, "{}: {error:e} above {bound:e}", case.name);
    }
}

fn hard_bits() -> Vec<f64> {
    let reference = reference();
    let mut out = Vec::new();
    for case in &reference.cases {
        let fe = fixed_effects(case);
        for method in METHODS {
            let d = fe
                .demean(&[&case.y], &control(method, reference.tolerance))
                .unwrap();
            out.extend(d.columns.col_as_slice(0));
            out.push(d.iterations[0] as f64);
        }
    }
    out
}

#[test]
fn hard_designs_do_not_depend_on_the_number_of_threads() {
    assert_same_on_threads(hard_bits);
}

#[test]
fn hard_designs_have_their_recorded_bits() {
    let hash = epistat_testkit::Fingerprint::new()
        .f64s(&hard_bits())
        .finish();
    epistat_testkit::hash_out("hard_designs_have_their_recorded_bits", hash);
    if epistat_testkit::RECORDED_PLATFORM {
        assert_eq!(hash, 4_005_707_165_456_125_297, "recorded hash");
    }
}
