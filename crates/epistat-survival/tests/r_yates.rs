//! `yates` of survival against [`epistat_survival::standardise::yates`]: a
//! linear model, Poisson and logistic regression and a Cox model, with the
//! populations "data", "factorial" and "sas", the linear predictor, the link,
//! the response and the Cox risk, and the global and pairwise tests.
//!
//! The populations are the design matrices that `yates` itself builds, so that
//! what is compared is the averaging, the variance and the tests.
//!
//! # What is compared and the tolerances
//!
//! With the identity link `yates` has an exact variance, and its estimates,
//! covariance and tests are compared with ours. With another link it
//! simulates, so its variance carries a Monte Carlo error and a curvature term,
//! see the module documentation of `yates`. Ours is compared with a delta
//! method computed in R from a central difference of the estimates in the
//! coefficients (`delta_mvar`, `delta_tests`), and the simulated standard errors
//! are only reported.
//!
//! With `n` rows in a population, `p` coefficients, `X` the largest absolute design value and
//! `xb = sum_a 2 max|x_a| |beta_a|` over the columns (the centre included):
//!
//! - an estimate is a mean of `n` values of the link at a linear predictor with
//!   the absolute error `(p + 2 + xb) eps`; for the identity link its bound is
//!   `4 (n + p + 4) eps sum_a mean|x_a| |beta_a|`, and for the exponential and
//!   logistic links, whose derivative is below the value or 1/4, the relative
//!   bound `4 (n + p + 4 + xb) eps`;
//! - a covariance entry is `sum_ij G_ai V_ij G_bj` with `G` the gradient, a mean of
//!   `n` terms, so its bound is `4 (n + 2 p + 4 + xb) eps sum_ij |G_ai| |V_ij| |G_bj|`,
//!   with the mean absolute gradient. The R delta method adds, for a gradient
//!   entry, the rounding error of a difference quotient with step `h`,
//!   `eps_m / h` with `eps_m` the absolute bound of an estimate, and the
//!   truncation `h^2 X^3 max(1, m) / 6` (the third derivative of the mean of an
//!   exponential or logistic link is below `X^3 max(1/4, m)` times a constant
//!   below 1), which enters as `delta_j sum_i |V_ij| (|G_ai| + |G_bi|)`;
//! - a test statistic `chi^2 = c' M^+ c` carries `2 |dc| / |c| + kappa |dM| / |M|`
//!   relative, with `c` the contrasts (error from the estimates), `M` their
//!   covariance (error from the covariance) and `kappa` the condition number
//!   of `M`.
//!
//! The bounds are not changed after a run.

#![expect(clippy::cast_precision_loss, reason = "counts are small")]

use epistat_linalg::Mat;
use epistat_survival::standardise::Counterfactual;
use epistat_survival::standardise::yates::{InverseLink, Test, YatesLevel, YatesOptions, yates};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Versions {
    r: String,
    survival: String,
}

#[derive(Deserialize)]
struct TestRow {
    chisq: f64,
    df: f64,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    link: String,
    test: String,
    coefficients: Vec<f64>,
    var: Vec<Vec<f64>>,
    centre: Option<Vec<f64>>,
    designs: Vec<Vec<Vec<f64>>>,
    estimate: Vec<f64>,
    std: Vec<f64>,
    mvar: Vec<Vec<f64>>,
    tests: Vec<TestRow>,
    nsim: Option<f64>,
    delta_mvar: Vec<Vec<f64>>,
    gradient: Vec<Vec<f64>>,
    fd_step: f64,
    delta_tests: Vec<TestRow>,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    cases: Vec<Case>,
}

#[derive(Default)]
struct Worst(std::collections::BTreeMap<String, (f64, f64)>);

impl Worst {
    fn put(&mut self, name: &str, deviation: f64, bound: f64) {
        assert!(
            deviation <= bound,
            "{name}: deviation {deviation:e} above its bound {bound:e}"
        );
        let e = self.0.entry(name.to_string()).or_insert((0.0, 0.0));
        e.0 = e.0.max(deviation / bound);
        e.1 = e.1.max(deviation);
    }
}

fn condition(m: &[Vec<f64>]) -> f64 {
    let k = m.len();
    if k == 1 {
        return 1.0;
    }
    let mat = Mat::from_fn(k, k, |i, j| m[i][j]);
    let e = mat
        .self_adjoint_eigenvalues(epistat_linalg::Side::Lower)
        .unwrap();
    e[k - 1] / e[0]
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one pass over the outputs of each case"
)]
fn population_means_match_r() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/yates.json"
    ));
    eprintln!(
        "reference: {}, survival {}",
        reference.versions.r, reference.versions.survival
    );
    let eps = f64::EPSILON;
    let mut worst = Worst::default();
    for case in &reference.cases {
        let name = &case.name;
        let link = match case.link.as_str() {
            "identity" => InverseLink::Identity,
            "exp" => InverseLink::Exp,
            "logit" => InverseLink::Logistic,
            other => panic!("link {other}"),
        };
        let identity = link == InverseLink::Identity;
        let p = case.coefficients.len();
        let n_levels = case.designs.len();
        let mats: Vec<Mat<f64>> = case
            .designs
            .iter()
            .map(|d| Mat::from_fn(d.len(), d[0].len(), |i, j| d[i][j]))
            .collect();
        let levels: Vec<YatesLevel<'_>> = mats
            .iter()
            .map(|m| YatesLevel {
                population: Counterfactual::Design(m.as_ref()),
                base: None,
                weights: None,
            })
            .collect();
        let var: Vec<f64> = case.var.iter().flatten().copied().collect();
        let test = if case.test == "pairwise" {
            Test::Pairwise
        } else {
            Test::Global
        };
        let out = yates(
            &case.coefficients,
            &var,
            &levels,
            &YatesOptions {
                link,
                centre: case.centre.as_deref(),
                test,
            },
        )
        .unwrap();

        // Bound ingredients per level.
        let zero = vec![0.0; p];
        let centre = case.centre.as_ref().unwrap_or(&zero);
        let mut b_est = vec![0.0; n_levels];
        let mut mean_abs: Vec<Vec<f64>> = Vec::new();
        let mut xb = 0.0f64;
        let mut big_x = 0.0f64;
        let mut rows = 0.0f64;
        for (l, d) in case.designs.iter().enumerate() {
            let n = d.len() as f64;
            rows = rows.max(n);
            let mut ma = vec![0.0; p];
            let mut s = 0.0;
            let mut level_xb = 0.0;
            for a in 0..p {
                let col_max = d
                    .iter()
                    .map(|r| (r[a] - centre[a]).abs())
                    .fold(0.0, f64::max);
                big_x = big_x.max(col_max);
                level_xb += 2.0 * col_max * case.coefficients[a].abs();
                ma[a] = d.iter().map(|r| r[a].abs()).sum::<f64>() / n;
                s += (ma[a] + centre[a].abs()) * case.coefficients[a].abs();
            }
            xb = xb.max(level_xb);
            b_est[l] = if identity {
                4.0 * (n + p as f64 + 4.0) * eps * s
            } else {
                4.0 * (n + p as f64 + 4.0 + level_xb) * eps * case.estimate[l].abs()
            };
            mean_abs.push(ma);
        }
        for ((got, want), bound) in out.estimate.iter().zip(&case.estimate).zip(&b_est) {
            worst.put(&format!("{name} estimate"), (got - want).abs(), *bound);
        }

        // The covariance of the estimates: yates' for the identity link, the R
        // delta method otherwise.
        let (want_var, want_tests) = if identity {
            (&case.mvar, &case.tests)
        } else {
            (&case.delta_mvar, &case.delta_tests)
        };
        let mut b_cov = vec![vec![0.0; n_levels]; n_levels];
        let m_max = case.estimate.iter().fold(0.0f64, |a, b| a.max(b.abs()));
        for a in 0..n_levels {
            for b in 0..n_levels {
                // |G| is the mean absolute gradient. For the identity link it is
                // mean|x|, otherwise it is bounded by the R gradient itself.
                let ga: Vec<f64> = if identity {
                    mean_abs[a].clone()
                } else {
                    case.gradient[a].iter().map(|g| g.abs()).collect()
                };
                let gb: Vec<f64> = if identity {
                    mean_abs[b].clone()
                } else {
                    case.gradient[b].iter().map(|g| g.abs()).collect()
                };
                let mut sum = 0.0;
                let mut fd = 0.0;
                for (i, &ga_i) in ga.iter().enumerate().take(p) {
                    for (j, &gb_j) in gb.iter().enumerate().take(p) {
                        let v = case.var[i][j].abs();
                        sum += ga_i * v * gb_j;
                        if !identity {
                            let delta = |k: usize| {
                                let h = case.fd_step * case.coefficients[k].abs().max(1.0);
                                let eps_m =
                                    4.0 * (rows + p as f64 + 4.0 + xb) * eps * m_max.max(0.25);
                                eps_m / h + h * h * big_x.powi(3) * m_max.max(1.0) / 6.0
                            };
                            fd += v * (delta(i) * gb_j + ga_i * delta(j));
                        }
                    }
                }
                b_cov[a][b] = 4.0 * (rows + 2.0 * p as f64 + 4.0 + xb) * eps * sum + fd;
            }
        }
        for a in 0..n_levels {
            for b in 0..n_levels {
                worst.put(
                    &format!("{name} covariance"),
                    (out.covariance[a * n_levels + b] - want_var[a][b]).abs(),
                    b_cov[a][b],
                );
            }
            // A standard error is the root of a diagonal entry, so its
            // deviation is bounded by the bound of the variance over twice the
            // standard error, and for the identity link the std of yates.
            let want = want_var[a][a].sqrt();
            worst.put(
                &format!("{name} std"),
                (out.se[a] - want).abs(),
                b_cov[a][a] / want,
            );
            if identity {
                worst.put(
                    &format!("{name} std (yates)"),
                    (out.se[a] - case.std[a]).abs(),
                    b_cov[a][a] / want,
                );
            }
        }

        // The tests.
        assert_eq!(out.tests.len(), want_tests.len(), "{name}: number of tests");
        let kappa = condition(want_var);
        let b_c: f64 = b_est.iter().sum::<f64>() * 2.0;
        let norm_s: f64 = want_var
            .iter()
            .flatten()
            .map(|v| v.abs())
            .fold(0.0, f64::max);
        let b_s: f64 = b_cov.iter().flatten().copied().fold(0.0, f64::max);
        for (got, want) in out.tests.iter().zip(want_tests) {
            assert_eq!(
                f64::to_bits(got.df as f64),
                f64::to_bits(want.df),
                "{name}: degrees of freedom"
            );
            let c_norm = want.chisq.sqrt().max(f64::MIN_POSITIVE);
            let rel = 2.0 * b_c / c_norm + kappa * b_s / norm_s;
            let bound = want.chisq * rel * 4.0 + 8.0 * eps * kappa * want.chisq;
            worst.put(
                &format!("{name} chi-square"),
                (got.chi_square - want.chisq).abs(),
                bound,
            );
        }
        if !identity {
            let mut gap = 0.0f64;
            for a in 0..n_levels {
                gap = gap.max((out.se[a] - case.std[a]).abs() / case.std[a]);
            }
            let mc = 1.0 / (2.0 * (case.nsim.unwrap() - 1.0)).sqrt();
            eprintln!(
                "{name}: delta standard errors differ from yates's simulated ones by up to {gap:.4} (relative; Monte Carlo sd {mc:.4})"
            );
        }
    }
    for (k, v) in &worst.0 {
        eprintln!(
            "{k}: worst deviation {:.3e} absolute, {:.3} of its bound",
            v.1, v.0
        );
    }
}
