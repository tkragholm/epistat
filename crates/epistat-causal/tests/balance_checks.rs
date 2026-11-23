//! Properties of the balancing weights that need no reference: exact balance,
//! refusals, equivalence of case weights and repeated rows, and the stacked
//! variance against the infinitesimal jackknife in a small case.

mod balance_sim;

use epistat_causal::Error;
use epistat_causal::aipw::Target;
use epistat_causal::balance::{
    CbpsInput, CbpsOptions, EntropyInput, EntropyOptions, Identification, WeightVariance,
    balance_columns, cbps, entropy_balance,
};
use epistat_linalg::Mat;

fn weighted_mean(col: impl Fn(usize) -> f64, rows: &[usize], s: &[f64], w: &[f64]) -> f64 {
    let d: f64 = rows.iter().map(|&i| s[i] * w[i]).sum();
    rows.iter().map(|&i| s[i] * w[i] * col(i)).sum::<f64>() / d
}

#[test]
fn entropy_balancing_matches_the_target_moments_exactly() {
    let sim = balance_sim::simulate(1500);
    let cols = balance_columns(sim.c.as_ref(), 2, true).unwrap();
    assert_eq!(cols.ncols(), 3 + 2 + 3);
    for target in [Target::Att, Target::Atc, Target::Ate] {
        let r = entropy_balance(
            &EntropyInput {
                treatment: &sim.a,
                covariates: cols.as_ref(),
                sample_weight: Some(&sim.s),
                base_weight: Some(&sim.q),
            },
            &EntropyOptions {
                target,
                ..EntropyOptions::default()
            },
        )
        .unwrap();
        assert!(r.max_imbalance <= 1e-10, "{target:?} {}", r.max_imbalance);
        let w = &r.weights.weights;
        let all: Vec<usize> = (0..sim.a.len()).collect();
        let tr: Vec<usize> = all.iter().copied().filter(|&i| sim.a[i]).collect();
        let co: Vec<usize> = all.iter().copied().filter(|&i| !sim.a[i]).collect();
        let tgt = match target {
            Target::Att => &tr,
            Target::Atc => &co,
            Target::Ate => &all,
        };
        let ones = vec![1.0; w.len()];
        for j in 0..cols.ncols() {
            let col = |i: usize| cols[(i, j)];
            let m = weighted_mean(col, tgt, &sim.s, &ones);
            let sd = (tgt
                .iter()
                .map(|&i| sim.s[i] * (col(i) - m).powi(2))
                .sum::<f64>()
                / tgt.iter().map(|&i| sim.s[i]).sum::<f64>())
            .sqrt();
            for (group, flag) in [(&tr, true), (&co, false)] {
                let weighted = (flag && target != Target::Att) || (!flag && target != Target::Atc);
                if weighted {
                    let got = weighted_mean(col, group, &sim.s, w);
                    assert!((got - m).abs() <= 1e-9 * sd, "{target:?} col {j}");
                }
            }
        }
        let size: f64 = tgt.iter().map(|&i| sim.s[i]).sum();
        for (g, summary) in [(false, r.weights.controls), (true, r.weights.treated)] {
            if summary.weighted {
                assert!((summary.sum - size).abs() <= 1e-9 * size, "{g}");
            }
        }
    }
}

fn base(sim: &balance_sim::Simulated) -> EntropyInput<'_> {
    EntropyInput {
        treatment: &sim.a,
        covariates: sim.c.as_ref(),
        sample_weight: None,
        base_weight: None,
    }
}

#[test]
fn entropy_balancing_refuses_an_unreachable_target() {
    let sim = balance_sim::simulate(400);
    // The controls' third covariate never reaches the treated mean.
    let (mut hi, mut lo) = (0.0_f64, 0.0_f64);
    let mut c = sim.c.clone();
    for i in 0..400 {
        if sim.a[i] {
            hi += c[(i, 2)];
            lo += 1.0;
        }
    }
    let cut = hi / lo;
    for i in 0..400 {
        if !sim.a[i] {
            c[(i, 2)] = c[(i, 2)].min(cut * 0.9);
        }
    }
    let input = EntropyInput {
        covariates: c.as_ref(),
        ..base(&sim)
    };
    let err = entropy_balance(
        &input,
        &EntropyOptions {
            target: Target::Att,
            ..EntropyOptions::default()
        },
    )
    .unwrap_err();
    assert!(
        matches!(err, Error::Infeasible(ref m) if m.contains("column 2")),
        "{err}"
    );
}

#[test]
fn entropy_balancing_refuses_a_hull_problem_without_a_marginal_cause() {
    // The target lies inside each range but outside the hull: two points of the
    // group, (-1, 1) and (1, -1) and a third (0.4, 0.4) cannot give (1.5, 1.5)/... .
    let n = 12;
    let a: Vec<bool> = (0..n).map(|i| i < 6).collect();
    let pts = [(-1.0, 1.0), (1.0, -1.0), (0.2, 0.2)];
    let c = Mat::from_fn(n, 2, |i, j| {
        if a[i] {
            // treated around (0.9, 0.9), spread so each column varies
            let d = f64::from(u8::try_from(i).unwrap()) * 0.02;
            if j == 0 { 0.9 + d } else { 0.9 - d }
        } else {
            let p = pts[(i - 6) % 3];
            if j == 0 { p.0 } else { p.1 }
        }
    });
    let err = entropy_balance(
        &EntropyInput {
            treatment: &a,
            covariates: c.as_ref(),
            sample_weight: None,
            base_weight: None,
        },
        &EntropyOptions {
            target: Target::Att,
            ..EntropyOptions::default()
        },
    )
    .unwrap_err();
    assert!(matches!(err, Error::Infeasible(_)), "{err}");
}

#[test]
fn entropy_balancing_refuses_constant_and_collinear_columns() {
    let sim = balance_sim::simulate(300);
    let constant = Mat::from_fn(300, 2, |i, j| if j == 0 { sim.c[(i, 0)] } else { 3.0 });
    let r = entropy_balance(
        &EntropyInput {
            covariates: constant.as_ref(),
            ..base(&sim)
        },
        &EntropyOptions::default(),
    );
    assert!(
        matches!(r, Err(Error::Input(ref m)) if m.contains("column 1")),
        "{r:?}"
    );
    let collinear = Mat::from_fn(300, 2, |i, j| {
        sim.c[(i, 0)] * (1.0 + f64::from(u8::try_from(j).unwrap()))
    });
    let r = entropy_balance(
        &EntropyInput {
            covariates: collinear.as_ref(),
            ..base(&sim)
        },
        &EntropyOptions::default(),
    );
    assert!(
        matches!(r, Err(Error::Input(ref m)) if m.contains("collinear")),
        "{r:?}"
    );
}

#[test]
fn entropy_balancing_refuses_bad_inputs() {
    let sim = balance_sim::simulate(100);
    let short = vec![1.0; 99];
    let r = entropy_balance(
        &EntropyInput {
            sample_weight: Some(&short),
            ..base(&sim)
        },
        &EntropyOptions::default(),
    );
    assert!(matches!(r, Err(Error::Input(_))));
    let neg: Vec<f64> = (0..100).map(|i| if i == 3 { -1.0 } else { 1.0 }).collect();
    let r = entropy_balance(
        &EntropyInput {
            base_weight: Some(&neg),
            ..base(&sim)
        },
        &EntropyOptions::default(),
    );
    assert!(matches!(r, Err(Error::Input(_))));
    let all_treated = vec![true; 100];
    let r = entropy_balance(
        &EntropyInput {
            treatment: &all_treated,
            ..base(&sim)
        },
        &EntropyOptions::default(),
    );
    assert!(r.is_err());
}

#[test]
fn a_case_weight_of_two_is_a_repeated_row() {
    let sim = balance_sim::simulate(200);
    let rep = |i: usize| i.is_multiple_of(7);
    let s: Vec<f64> = (0..200).map(|i| if rep(i) { 2.0 } else { 1.0 }).collect();
    let rows: Vec<usize> = (0..200)
        .flat_map(|i| if rep(i) { vec![i, i] } else { vec![i] })
        .collect();
    let m = rows.len();
    let c2 = Mat::from_fn(m, 3, |i, j| sim.c[(rows[i], j)]);
    let a2: Vec<bool> = rows.iter().map(|&i| sim.a[i]).collect();
    let y2: Vec<f64> = rows.iter().map(|&i| sim.y[i]).collect();
    for target in [Target::Att, Target::Ate] {
        let opts = EntropyOptions {
            target,
            ..EntropyOptions::default()
        };
        let one = entropy_balance(
            &EntropyInput {
                sample_weight: Some(&s),
                ..base(&sim)
            },
            &opts,
        )
        .unwrap();
        let two = entropy_balance(
            &EntropyInput {
                treatment: &a2,
                covariates: c2.as_ref(),
                sample_weight: None,
                base_weight: None,
            },
            &opts,
        )
        .unwrap();
        for (k, &i) in rows.iter().enumerate() {
            assert!(
                (one.weights.weights[i] - two.weights.weights[k]).abs() <= 1e-9,
                "{target:?}"
            );
        }
        let d1 = one
            .weights
            .mean_difference(&sim.y, None, WeightVariance::Fixed, 0.05)
            .unwrap();
        let d2 = two
            .weights
            .mean_difference(&y2, None, WeightVariance::Fixed, 0.05)
            .unwrap();
        assert!((d1.difference.estimate - d2.difference.estimate).abs() <= 1e-9);
    }
    // CBPS, over identified.
    let x2 = Mat::from_fn(
        m,
        4,
        |i, j| if j == 0 { 1.0 } else { sim.c[(rows[i], j - 1)] },
    );
    let opts = CbpsOptions::default();
    let one = cbps(
        &CbpsInput {
            treatment: &sim.a,
            design: sim.x.as_ref(),
            sample_weight: Some(&s),
        },
        &opts,
    )
    .unwrap();
    let two = cbps(
        &CbpsInput {
            treatment: &a2,
            design: x2.as_ref(),
            sample_weight: None,
        },
        &opts,
    )
    .unwrap();
    for j in 0..4 {
        assert!(
            (one.coefficients[j] - two.coefficients[j]).abs() <= 1e-8,
            "coefficient {j}"
        );
    }
}

#[test]
fn cbps_balances_the_weighted_groups_and_flips_for_the_controls() {
    let sim = balance_sim::simulate(1200);
    for target in [Target::Ate, Target::Att, Target::Atc] {
        let r = cbps(
            &CbpsInput {
                treatment: &sim.a,
                design: sim.x.as_ref(),
                sample_weight: Some(&sim.s),
            },
            &CbpsOptions {
                target,
                identification: Identification::Exact,
                ..CbpsOptions::default()
            },
        )
        .unwrap();
        assert!(r.max_imbalance <= 1e-8, "{target:?} {}", r.max_imbalance);
        assert!(r.objective.is_none() && r.start.is_none());
        for g in [r.weights.controls, r.weights.treated] {
            assert!((g.sum - 1.0).abs() <= 1e-12);
        }
    }
    let flipped: Vec<bool> = sim.a.iter().map(|&a| !a).collect();
    let opts = |target| CbpsOptions {
        target,
        identification: Identification::Exact,
        ..CbpsOptions::default()
    };
    let atc = cbps(
        &CbpsInput {
            treatment: &sim.a,
            design: sim.x.as_ref(),
            sample_weight: None,
        },
        &opts(Target::Atc),
    )
    .unwrap();
    let att_flipped = cbps(
        &CbpsInput {
            treatment: &flipped,
            design: sim.x.as_ref(),
            sample_weight: None,
        },
        &opts(Target::Att),
    )
    .unwrap();
    for j in 0..4 {
        assert!((atc.coefficients[j] + att_flipped.coefficients[j]).abs() <= 1e-9);
    }
    for i in 0..1200 {
        assert!((atc.weights.weights[i] - att_flipped.weights.weights[i]).abs() <= 1e-12);
    }
}

#[test]
fn over_identified_cbps_does_not_do_worse_than_its_starts() {
    let sim = balance_sim::simulate(800);
    for two_step in [true, false] {
        let r = cbps(
            &CbpsInput {
                treatment: &sim.a,
                design: sim.x.as_ref(),
                sample_weight: None,
            },
            &CbpsOptions {
                target: Target::Att,
                two_step,
                ..CbpsOptions::default()
            },
        )
        .unwrap();
        let (j, jl) = (r.objective.unwrap(), r.objective_logistic.unwrap());
        assert!(j <= jl * (1.0 + 1e-12), "{two_step}: {j} > {jl}");
        assert!(r.start.is_some());
    }
}

#[test]
fn cbps_refuses_what_it_cannot_fit() {
    let sim = balance_sim::simulate(200);
    let dup = Mat::from_fn(
        200,
        5,
        |i, j| if j == 4 { sim.x[(i, 1)] } else { sim.x[(i, j)] },
    );
    let r = cbps(
        &CbpsInput {
            treatment: &sim.a,
            design: dup.as_ref(),
            sample_weight: None,
        },
        &CbpsOptions::default(),
    );
    assert!(
        matches!(r, Err(Error::Input(ref m)) if m.contains("aliased")),
        "{r:?}"
    );
    let all = vec![true; 200];
    let r = cbps(
        &CbpsInput {
            treatment: &all,
            design: sim.x.as_ref(),
            sample_weight: None,
        },
        &CbpsOptions::default(),
    );
    assert!(matches!(r, Err(Error::Input(_))));
    // Perfect separation: no solution of the balance conditions.
    let sep: Vec<bool> = (0..200).map(|i| sim.x[(i, 1)] > 0.0).collect();
    let r = cbps(
        &CbpsInput {
            treatment: &sep,
            design: sim.x.as_ref(),
            sample_weight: None,
        },
        &CbpsOptions {
            identification: Identification::Exact,
            ..CbpsOptions::default()
        },
    );
    assert!(r.is_err(), "separation is refused");
}

#[test]
fn the_outcome_stage_agrees_with_group_means_and_refuses_aliased_designs() {
    let sim = balance_sim::simulate(500);
    let r = entropy_balance(&base(&sim), &EntropyOptions::default())
        .unwrap()
        .weights;
    let md = r
        .mean_difference(&sim.y, None, WeightVariance::Estimated, 0.05)
        .unwrap();
    let mean = |arm: bool| {
        let rows: Vec<usize> = (0..500).filter(|&i| sim.a[i] == arm).collect();
        weighted_mean(|i| sim.y[i], &rows, &vec![1.0; 500], &r.weights)
    };
    assert!((md.mean1.estimate - mean(true)).abs() <= 1e-12);
    assert!((md.mean0.estimate - mean(false)).abs() <= 1e-12);
    assert!(md.difference.se > 0.0 && md.difference.lower < md.difference.upper);
    let fixed = r
        .mean_difference(&sim.y, None, WeightVariance::Fixed, 0.05)
        .unwrap();
    assert!((fixed.difference.se - md.difference.se).abs() > 1e-6);
    let bad = Mat::from_fn(500, 3, |i, j| {
        if j == 2 {
            2.0 * sim.x[(i, 1)]
        } else {
            sim.x[(i, j)]
        }
    });
    assert!(
        r.weighted_regression(&sim.y, bad.as_ref(), None, WeightVariance::Estimated)
            .is_err()
    );
    let one_cluster = vec![0_u32; 500];
    assert!(
        r.mean_difference(&sim.y, Some(&one_cluster), WeightVariance::Estimated, 0.05)
            .is_err()
    );
}

#[test]
fn a_repeated_call_gives_the_same_bits() {
    let sim = balance_sim::simulate(700);
    let cols = balance_columns(sim.c.as_ref(), 2, false).unwrap();
    let run = || {
        let r = entropy_balance(
            &EntropyInput {
                treatment: &sim.a,
                covariates: cols.as_ref(),
                sample_weight: Some(&sim.s),
                base_weight: None,
            },
            &EntropyOptions::default(),
        )
        .unwrap();
        balance_sim::flatten(&r.weights, &sim)
    };
    let (a, b) = (run(), run());
    assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
}
