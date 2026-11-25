//! The multi-way fixed-effects solver and the linear fit on it, checked against
//! `fixest::feols` 0.14.2 and against the exact projections on the dummies
//! (the QR of `lm.fit`), on unbalanced weighted panels with singletons, with
//! two and three factors, two connected components and few clusters. See
//! `tests/data/feols.R`.

use std::collections::HashMap;

use epistat_glm::{Absorb, Feols, FixedEffects, GDf, KFixef, Singletons, Ssc, Vcov};
use epistat_linalg::Mat;
use epistat_testkit::{assert_same_on_threads, load_json};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    tolerance: f64,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct SscSpec {
    k_adj: bool,
    k_fixef: String,
    k_exact: bool,
    g_adj: bool,
    g_df: String,
}

#[derive(Deserialize)]
struct VcovSpec {
    matrix: Vec<Option<f64>>,
    fixed: Vec<Option<f64>>,
    df_k: usize,
    #[serde(rename = "type")]
    kind: String,
    vars: Vec<String>,
    ssc: SscSpec,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    factors: Vec<Vec<u32>>,
    weights: Option<Vec<f64>>,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    variables: HashMap<String, Vec<u32>>,
    kept: Vec<usize>,
    coef: Vec<Option<f64>>,
    exact_coef: Vec<Option<f64>>,
    nparams: usize,
    r2_within: f64,
    vcov: HashMap<String, VcovSpec>,
    fixef: Vec<Vec<Option<f64>>>,
    references: Option<Vec<usize>>,
    exact_residuals: Vec<Vec<f64>>,
    lambda_plus: f64,
    fixest_error: f64,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/feols.json"
    ))
}

const EPS: f64 = f64::EPSILON;

fn design(case: &Case) -> Mat<f64> {
    Mat::from_fn(case.x.len(), case.x[0].len(), |i, j| case.x[i][j])
}

fn fixed_effects(case: &Case) -> FixedEffects {
    let factors: Vec<&[u32]> = case.factors.iter().map(Vec::as_slice).collect();
    FixedEffects::new(&factors, case.weights.as_deref(), Singletons::Remove).unwrap()
}

fn ssc(spec: &SscSpec) -> Ssc {
    Ssc {
        k_adj: spec.k_adj,
        k_fixef: match spec.k_fixef.as_str() {
            "none" => KFixef::None,
            "full" => KFixef::Full,
            _ => KFixef::Nonnested,
        },
        k_exact: spec.k_exact,
        g_adj: spec.g_adj,
        g_df: if spec.g_df == "min" {
            GDf::Min
        } else {
            GDf::Conventional
        },
    }
}

/// The kept rows' weights, 1 without weights.
fn kept_weights(case: &Case) -> Vec<f64> {
    case.kept
        .iter()
        .map(|&i| case.weights.as_ref().map_or(1.0, |w| w[i]))
        .collect()
}

/// The columns of the case on the kept rows: covariates, then the outcome.
fn kept_columns(case: &Case) -> Vec<Vec<f64>> {
    let p = case.x[0].len();
    let mut out: Vec<Vec<f64>> = (0..p)
        .map(|j| case.kept.iter().map(|&i| case.x[i][j]).collect())
        .collect();
    out.push(case.kept.iter().map(|&i| case.y[i]).collect());
    out
}

fn condition(gram: &[f64], k: usize) -> f64 {
    // The condition number of the Gram matrix of the columns scaled to unit
    // norm, which governs the rounding of the least-squares solution.
    let d: Vec<f64> = (0..k).map(|a| gram[a * k + a].sqrt()).collect();
    let scaled: Vec<f64> = (0..k * k)
        .map(|ab| gram[ab] / (d[ab / k] * d[ab % k]))
        .collect();
    let (values, _) = epistat_linalg::eigen::symmetric_eigen_jacobi(&scaled, k);
    let max = values.iter().copied().fold(f64::MIN, f64::max);
    let min = values.iter().copied().fold(f64::MAX, f64::min);
    max / min
}

#[test]
fn rows_levels_and_parameters_match_fixest() {
    for case in &reference().cases {
        let fe = fixed_effects(case);
        assert_eq!(fe.rows(), case.kept.as_slice(), "{}", case.name);
        let x = design(case);
        let fit = Feols::default().fit(&case.y, x.as_ref(), &fe).unwrap();
        assert_eq!(fit.n_parameters(), case.nparams, "{}", case.name);
        let fx = fit.fixed_effects();
        if let Some(references) = &case.references {
            assert_eq!(&fx.references, references, "{}", case.name);
        }
        assert!(fx.normalised, "{}", case.name);
        let aliased: Vec<usize> = (0..case.coef.len())
            .filter(|&j| case.coef[j].is_none())
            .collect();
        assert_eq!(fit.dropped_columns, aliased, "{}", case.name);
    }
}

#[test]
fn the_criterion_bounds_the_distance_to_the_exact_projection() {
    // A loose tolerance, so that the error is well above rounding and the
    // bound is tested. At the returned point every weighted level mean of the
    // residual is at most tol * s; recomputed here, each mean carries the
    // rounding of a sum of at most `largest group` terms, so the check allows
    // that many ulps of the largest residual. The error e = r - r* satisfies
    // ||e||_W <= ||gamma|| / sqrt(lambda_+), gamma the level sums of w r
    // (module documentation of `absorb`). The exact residuals come from a
    // Householder QR of the n x m dummy matrix, whose residual carries an
    // error of order n m eps ||v|| (Higham, Accuracy and Stability, thm
    // 20.3), which is added.
    let tol = 1e-8;
    for case in &reference().cases {
        let fe = fixed_effects(case);
        let columns = kept_columns(case);
        let full: Vec<Vec<f64>> = (0..columns.len())
            .map(|j| {
                let mut v = vec![0.0; case.y.len()];
                for (k, &i) in case.kept.iter().enumerate() {
                    v[i] = columns[j][k];
                }
                v
            })
            .collect();
        let refs: Vec<&[f64]> = full.iter().map(Vec::as_slice).collect();
        let out = fe
            .demean(
                &refs,
                &Absorb {
                    tolerance: tol,
                    max_iterations: 10_000,
                },
            )
            .unwrap();
        let w = kept_weights(case);
        let n = w.len();
        let m: usize = fe.levels().iter().sum();
        for (j, column) in columns.iter().enumerate() {
            let s = out.scale[j];
            let r = out.columns.col_as_slice(j);
            let largest_r = r.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
            let mut gamma2 = 0.0;
            for q in 0..fe.n_factors() {
                let codes = fe.codes(q);
                let g = fe.levels()[q];
                let (mut sum, mut weight, mut count) =
                    (vec![0.0; g], vec![0.0; g], vec![0usize; g]);
                for k in 0..n {
                    let c = codes[k] as usize;
                    sum[c] += w[k] * r[k];
                    weight[c] += w[k];
                    count[c] += 1;
                }
                let largest_group = count.iter().copied().max().unwrap() as f64;
                for c in 0..g {
                    let mean = sum[c] / weight[c];
                    assert!(
                        mean.abs() <= tol * s + largest_group * EPS * largest_r,
                        "{} column {j} factor {q}: {mean:e} against {:e}",
                        case.name,
                        tol * s
                    );
                    gamma2 += sum[c] * sum[c];
                }
            }
            let mean_v: f64 = (0..n).map(|k| w[k] * column[k]).sum::<f64>() / w.iter().sum::<f64>();
            let norm_v: f64 = (0..n)
                .map(|k| w[k] * (column[k] - mean_v).powi(2))
                .sum::<f64>()
                .sqrt();
            let error: f64 = (0..n)
                .map(|k| w[k] * (r[k] - case.exact_residuals[k][j]).powi(2))
                .sum::<f64>()
                .sqrt();
            let bound = gamma2.sqrt() / case.lambda_plus.sqrt() + (n * m) as f64 * EPS * norm_v;
            assert!(
                error <= bound,
                "{} column {j}: error {error:e} above bound {bound:e}",
                case.name
            );
        }
    }
}

/// The largest deviation of the demeaned columns from the exact projection,
/// over the columns' root mean squares, measured as `feols.R` measures
/// fixest's.
fn demeaning_error(case: &Case, fe: &FixedEffects, control: &Absorb) -> f64 {
    let columns = kept_columns(case);
    let full: Vec<Vec<f64>> = (0..columns.len())
        .map(|j| {
            let mut v = vec![0.0; case.y.len()];
            for (k, &i) in case.kept.iter().enumerate() {
                v[i] = columns[j][k];
            }
            v
        })
        .collect();
    let refs: Vec<&[f64]> = full.iter().map(Vec::as_slice).collect();
    let out = fe.demean(&refs, control).unwrap();
    let mut worst = 0.0_f64;
    for j in 0..columns.len() {
        for k in 0..columns[j].len() {
            let d = (out.columns[(k, j)] - case.exact_residuals[k][j]).abs() / out.scale[j];
            worst = worst.max(d);
        }
    }
    worst
}

const TIGHT: Absorb = Absorb {
    tolerance: 1e-13,
    max_iterations: 100_000,
};

#[test]
fn coefficients_covariances_and_fixed_effects_match_fixest() {
    let reference = reference();
    assert!(reference.tolerance > 0.0);
    for case in &reference.cases {
        let fe = fixed_effects(case);
        let x = design(case);
        let fit = Feols { absorb: TIGHT }
            .fit(&case.y, x.as_ref(), &fe)
            .unwrap();
        // Both sides' distance from the exact projection, relative to each
        // column's root mean square.
        let mine = demeaning_error(case, &fe, &TIGHT);
        let e = mine + case.fixest_error;
        let p = case.coef.len();
        let k = fit.rank;
        let w = kept_weights(case);
        let n = w.len();
        let design = fit.design();
        let gram: Vec<f64> = (0..k * k)
            .map(|ab| {
                (0..n)
                    .map(|i| w[i] * design[(i, ab / k)] * design[(i, ab % k)])
                    .sum()
            })
            .collect();
        let kappa = condition(&gram, k);

        // Coefficients. The demeaning errors lie in the span of the dummies
        // and cancel at first order (module documentation of `absorb`), so
        // they enter as e^2 kappa; the rest is the rounding of two
        // least-squares solutions, a few kappa eps each. 100 kappa eps covers
        // both with room for the sums over n rows.
        let scale = case
            .coef
            .iter()
            .flatten()
            .fold(0.0_f64, |a, &b| a.max(b.abs()));
        let tol_beta = (100.0 * EPS + e * e) * kappa * scale;
        for j in 0..p {
            match (fit.coefficients[j], case.coef[j], case.exact_coef[j]) {
                (Some(a), Some(b), Some(c)) => {
                    assert!((a - b).abs() <= tol_beta, "{} beta {j}: {a} {b}", case.name);
                    assert!(
                        (a - c).abs() <= tol_beta,
                        "{} exact beta {j}: {a} {c}",
                        case.name
                    );
                }
                (None, None, None) => {}
                other => panic!("{} beta {j}: {other:?}", case.name),
            }
        }
        assert!(
            (fit.r2_within.unwrap() - case.r2_within).abs() <= 100.0 * kappa * EPS + e * e,
            "{} r2",
            case.name
        );

        // Scale of each demeaned column, for the first-order bounds below.
        let columns = kept_columns(case);
        let rms: Vec<f64> = columns
            .iter()
            .map(|c| {
                let tw: f64 = w.iter().sum();
                let m: f64 = c.iter().zip(&w).map(|(v, w)| v * w).sum::<f64>() / tw;
                (c.iter()
                    .zip(&w)
                    .map(|(v, w)| w * (v - m).powi(2))
                    .sum::<f64>()
                    / tw)
                    .sqrt()
            })
            .collect();
        let kept = fit.kept_columns().to_vec();
        let beta: Vec<f64> = kept.iter().map(|&j| fit.coefficients[j].unwrap()).collect();
        // Each demeaned covariate is off by at most e rms_a per row, and each
        // residual by at most e (rms_y + sum |beta_b| rms_b).
        let delta_e = e
            * (rms[p]
                + kept
                    .iter()
                    .zip(&beta)
                    .map(|(&j, b)| b.abs() * rms[j])
                    .sum::<f64>());
        let bread = fit.bread();

        for (label, spec) in &case.vcov {
            let ways: Vec<&[u32]> = spec
                .vars
                .iter()
                .map(|v| case.variables[v].as_slice())
                .collect();
            let kind = match spec.kind.as_str() {
                "iid" => Vcov::Iid,
                "hetero" => Vcov::Hetero,
                _ => Vcov::Cluster(&ways),
            };
            let cov = fit.vcov(kind, ssc(&spec.ssc)).unwrap();
            assert_eq!(cov.df_k, spec.df_k, "{} {label} K", case.name);
            let largest = spec
                .matrix
                .iter()
                .flatten()
                .fold(0.0_f64, |a, &b| a.max(b.abs()));
            // The bound on the error of each entry. The iid covariance is
            // rss B times a constant, and both rss and B change at second
            // order only, so its bound is rounding. For the sandwiches, B
            // changes at second order and the meat at first: with s_g the
            // score sums and U_g bounds on their errors,
            // |dM| <= sum_g |s_g| |U_g|' + |U_g| |s_g|', and |dV| <= |B| |dM| |B|
            // times the product of the small-sample factors, which is below
            // 2 here. Rounding adds 100 kappa eps of the largest entry.
            let mut bound = vec![100.0 * kappa * EPS * largest; k * k];
            if spec.kind != "iid" {
                let rows = fe.rows();
                let mut terms: Vec<Vec<usize>> = Vec::new();
                if spec.kind == "hetero" {
                    terms.push((0..n).collect());
                } else {
                    for subset in 1..(1usize << ways.len()) {
                        let members: Vec<usize> =
                            (0..ways.len()).filter(|&j| subset >> j & 1 == 1).collect();
                        let mut number: HashMap<Vec<u32>, usize> = HashMap::new();
                        terms.push(
                            rows.iter()
                                .map(|&i| {
                                    let key: Vec<u32> =
                                        members.iter().map(|&m| ways[m][i]).collect();
                                    let next = number.len();
                                    *number.entry(key).or_insert(next)
                                })
                                .collect(),
                        );
                    }
                }
                let mut dm = vec![0.0; k * k];
                for of in &terms {
                    let g = of.iter().copied().max().unwrap() + 1;
                    let mut s = vec![0.0; g * k];
                    let mut u = vec![0.0; g * k];
                    for i in 0..n {
                        for a in 0..k {
                            s[of[i] * k + a] += w[i] * fit.residuals[i] * design[(i, a)];
                            u[of[i] * k + a] += w[i]
                                * (fit.residuals[i].abs() * e * rms[kept[a]]
                                    + design[(i, a)].abs() * delta_e);
                        }
                    }
                    for c in 0..g {
                        for a in 0..k {
                            for b in 0..k {
                                dm[a * k + b] += s[c * k + a].abs() * u[c * k + b]
                                    + u[c * k + a] * s[c * k + b].abs();
                            }
                        }
                    }
                }
                for a in 0..k {
                    for b in 0..k {
                        let mut total = 0.0;
                        for c in 0..k {
                            for d in 0..k {
                                total += bread[(a, c)].abs() * dm[c * k + d] * bread[(d, b)].abs();
                            }
                        }
                        bound[a * k + b] += 2.0 * total;
                    }
                }
            }
            for (a, &i) in kept.iter().enumerate() {
                for (b, &j) in kept.iter().enumerate() {
                    let got = cov.matrix[i * p + j].unwrap();
                    let want = spec.matrix[i * p + j].unwrap();
                    assert!(
                        (got - want).abs() <= bound[a * k + b],
                        "{} {label} ({i},{j}): {got:e} {want:e} bound {:e}",
                        case.name,
                        bound[a * k + b]
                    );
                }
            }
            // fixest's repair, where it applies, moves the matrix by more
            // than the bound, so the repaired one is compared on its own.
            if spec.fixed == spec.matrix {
                assert!(cov.min_eigenvalue.unwrap() > 0.0, "{} {label}", case.name);
            } else {
                assert!(cov.min_eigenvalue.unwrap() < 0.0, "{} {label}", case.name);
                // The repair goes through an eigendecomposition on each side
                // (Jacobi here, LAPACK in R), both backward stable, so the
                // rebuilt entries differ by a few ulps of the largest
                // eigenvalue per term of k^2 = 9; 1e3 eps of the largest
                // entry covers that.
                let repaired = cov.positive_definite();
                for (a, &i) in kept.iter().enumerate() {
                    for (b, &j) in kept.iter().enumerate() {
                        let got = repaired[i * p + j].unwrap();
                        let want = spec.fixed[i * p + j].unwrap();
                        assert!(
                            (got - want).abs() <= bound[a * k + b] + 1e3 * EPS * largest,
                            "{} {label} repaired ({i},{j}): {got:e} {want:e}",
                            case.name
                        );
                    }
                }
            }
        }

        // Fixed effects. The fitted sum of the fixed effects of a row is
        // y - x beta - e, so its error is at most delta_e plus |x| |d beta|,
        // and a level's value is reached from a reference through at most
        // as many rows as there are levels.
        let fx = fit.fixed_effects();
        let levels: usize = fe.levels().iter().sum();
        let largest_x: f64 = kept
            .iter()
            .map(|&j| case.x.iter().map(|r| r[j].abs()).fold(0.0, f64::max))
            .sum();
        let per_row = delta_e + largest_x * tol_beta + 100.0 * EPS * rms[p];
        let tol_fe = levels as f64 * per_row;
        for (q, want) in case.fixef.iter().enumerate() {
            let got = &fx.values[q];
            assert_eq!(got.len(), want.len(), "{} factor {q}", case.name);
            for (g, (a, b)) in got.iter().zip(want).enumerate() {
                match (a, b) {
                    (Some(a), Some(b)) => {
                        if *b == 0.0 && case.fixef.len() > 1 {
                            // A reference of fixest is exactly zero here too.
                            assert_eq!(*a, 0.0, "{} reference {q} {g}", case.name);
                        }
                        assert!(
                            (a - b).abs() <= tol_fe,
                            "{} fixef {q} {g}: {a} {b} ({tol_fe:e})",
                            case.name
                        );
                    }
                    (None, None) => {}
                    other => panic!("{} fixef {q} {g}: {other:?}", case.name),
                }
            }
        }
        eprintln!(
            "{}: demeaning error {mine:e} (fixest {:e}), kappa {kappa:.1}",
            case.name, case.fixest_error
        );
    }
}

#[test]
fn fixed_effects_predict_the_fitted_sums_and_refuse_unidentified_rows() {
    let reference = reference();
    let case = reference
        .cases
        .iter()
        .find(|c| c.name == "disconnected")
        .unwrap();
    let fe = fixed_effects(case);
    let fit = Feols { absorb: TIGHT }
        .fit(&case.y, design(case).as_ref(), &fe)
        .unwrap();
    let fx = fit.fixed_effects();
    let factors: Vec<&[u32]> = case.factors.iter().map(Vec::as_slice).collect();
    let predicted = fx.predict(&factors).unwrap();
    let kept = fit.kept_columns().to_vec();
    for (k, &i) in fe.rows().iter().enumerate() {
        let xb: f64 = kept
            .iter()
            .map(|&j| case.x[i][j] * fit.coefficients[j].unwrap())
            .sum();
        let fitted = case.y[i] - xb - fit.residuals[k];
        assert!((predicted[i].unwrap() - fitted).abs() < 1e-9, "row {i}");
    }
    // Unit 0 is in the first component and period 8 in the second, and code
    // 500 has no rows.
    assert_eq!(fx.sum(&[0, 8]), None);
    assert!(fx.sum(&[0, 0]).is_some());
    assert_eq!(fx.sum(&[500, 0]), None);
}

#[test]
fn zero_weights_select_the_same_fit_as_dropping_the_rows() {
    let reference = reference();
    let case = &reference.cases[0];
    let w = case.weights.as_ref().unwrap();
    let keep: Vec<usize> = (0..w.len()).filter(|&i| w[i] > 0.0).collect();
    let factors: Vec<Vec<u32>> = case
        .factors
        .iter()
        .map(|f| keep.iter().map(|&i| f[i]).collect())
        .collect();
    let refs: Vec<&[u32]> = factors.iter().map(Vec::as_slice).collect();
    let weights: Vec<f64> = keep.iter().map(|&i| w[i]).collect();
    let fe = FixedEffects::new(&refs, Some(&weights), Singletons::Remove).unwrap();
    let x = Mat::from_fn(keep.len(), case.x[0].len(), |i, j| case.x[keep[i]][j]);
    let y: Vec<f64> = keep.iter().map(|&i| case.y[i]).collect();
    let a = Feols::default().fit(&y, x.as_ref(), &fe).unwrap();
    let fe_full = fixed_effects(case);
    let b = Feols::default()
        .fit(&case.y, design(case).as_ref(), &fe_full)
        .unwrap();
    for j in 0..case.coef.len() {
        assert_eq!(
            a.coefficients[j].map(f64::to_bits),
            b.coefficients[j].map(f64::to_bits)
        );
    }
}

fn fit_bits() -> Vec<f64> {
    let reference = reference();
    let mut out = Vec::new();
    for case in &reference.cases {
        let fe = fixed_effects(case);
        let fit = Feols::default()
            .fit(&case.y, design(case).as_ref(), &fe)
            .unwrap();
        out.extend(fit.coefficients.iter().flatten());
        out.extend(fit.residuals.iter().take(50));
        let mut names: Vec<&String> = case.variables.keys().collect();
        names.sort();
        let ways: Vec<&[u32]> = vec![case.variables[names[0]].as_slice()];
        let cov = fit.vcov(Vcov::Cluster(&ways), Ssc::new()).unwrap();
        out.extend(cov.matrix.iter().flatten());
        let fx = fit.fixed_effects();
        for v in &fx.values {
            out.extend(v.iter().flatten());
        }
    }
    out
}

#[test]
fn the_fit_does_not_depend_on_the_number_of_threads() {
    assert_same_on_threads(fit_bits);
}

#[test]
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn the_fit_has_its_recorded_bits() {
    let hash = epistat_testkit::Fingerprint::new()
        .f64s(&fit_bits())
        .finish();
    assert_eq!(hash, 239_393_430_769_115_130, "recorded hash");
}

#[test]
fn influence_sums_are_the_sums_of_the_row_influence() {
    let reference = reference();
    let case = &reference.cases[0];
    let fe = fixed_effects(case);
    let fit = Feols::default()
        .fit(&case.y, design(case).as_ref(), &fe)
        .unwrap();
    let rows = fit.influence();
    let units = fe.codes(0);
    let count = fe.levels()[0];
    let sums = fit.influence_sums(units, count).unwrap();
    for a in 0..fit.rank {
        let mut want = vec![0.0; count];
        let mut size = vec![0.0_f64; count];
        for (k, &u) in units.iter().enumerate() {
            want[u as usize] += rows[(k, a)];
            size[u as usize] += rows[(k, a)].abs();
        }
        for g in 0..count {
            // The two orders of summation differ by the rounding of sums of
            // at most 8 rows and a product with B of rank 2: 16 eps of the
            // sum of magnitudes.
            assert!((sums[(g, a)] - want[g]).abs() <= 16.0 * EPS * size[g].max(f64::MIN_POSITIVE));
        }
    }
}

#[test]
fn solving_from_level_sums_gives_the_projection_of_the_column() {
    // With b = D' W v, the fitted D alpha is v minus its exact residual. The
    // residual v - D alpha meets the criterion with the scale of the level
    // means of b, so the bound of
    // `the_criterion_bounds_the_distance_to_the_exact_projection` applies
    // with that scale, and the rounding allowance is the same.
    let tol = 1e-8;
    let control = Absorb {
        tolerance: tol,
        max_iterations: 10_000,
    };
    for case in &reference().cases {
        let fe = fixed_effects(case);
        let w = kept_weights(case);
        let n = w.len();
        let m: usize = fe.levels().iter().sum();
        for (j, column) in kept_columns(case).iter().enumerate() {
            let sums: Vec<Vec<f64>> = (0..fe.n_factors())
                .map(|q| {
                    let mut s = vec![0.0; fe.levels()[q]];
                    for (k, &c) in fe.codes(q).iter().enumerate() {
                        s[c as usize] += w[k] * column[k];
                    }
                    s
                })
                .collect();
            let refs: Vec<&[f64]> = sums.iter().map(Vec::as_slice).collect();
            let alpha = fe.solve_sums(&refs, &control).unwrap();
            let fitted = fe.fitted(&alpha);
            let r: Vec<f64> = column.iter().zip(&fitted).map(|(v, f)| v - f).collect();
            let mut gamma2 = 0.0;
            let mut scale = 0.0_f64;
            for (q, level_sums) in sums.iter().enumerate() {
                let g = fe.levels()[q];
                let (mut sum, mut weight) = (vec![0.0; g], vec![0.0; g]);
                for (k, &c) in fe.codes(q).iter().enumerate() {
                    sum[c as usize] += w[k] * r[k];
                    weight[c as usize] += w[k];
                }
                for c in 0..g {
                    gamma2 += sum[c] * sum[c];
                    scale = scale.max((level_sums[c] / weight[c]).abs());
                }
            }
            let norm_v: f64 = (0..n)
                .map(|k| w[k] * column[k] * column[k])
                .sum::<f64>()
                .sqrt();
            let error: f64 = (0..n)
                .map(|k| w[k] * (r[k] - case.exact_residuals[k][j]).powi(2))
                .sum::<f64>()
                .sqrt();
            let bound = gamma2.sqrt() / case.lambda_plus.sqrt() + (n * m) as f64 * EPS * norm_v;
            assert!(scale > 0.0);
            assert!(
                error <= bound,
                "{} column {j}: {error:e} above {bound:e}",
                case.name
            );
        }
    }
}
