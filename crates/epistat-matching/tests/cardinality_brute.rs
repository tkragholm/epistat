//! Cardinality and profile matching against exhaustive enumeration, and the
//! input checks of the pairing step.
//!
//! Random programs of 10 to 14 units with raw tolerances have every subset
//! enumerated. The size of the largest subset that meets the program of the
//! module documentation of `cardinality` must be the size the solver returns,
//! and the solver must report it proven.

use epistat_matching::{
    CardinalityError, CardinalityLimits, CardinalityRequest, Estimand, match_cardinality,
    pair_units,
};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1_u64 << 53) as f64
    }
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    Card(f64),
    ProfileAtt,
    ProfileAte(Option<f64>),
}

/// Largest subset that meets the program of the module documentation.
fn brute_size(treated: &[bool], x: &[Vec<f64>], tols: &[f64], mode: Mode) -> Option<usize> {
    let n = treated.len();
    let p = x.len();
    let mean = |members: &dyn Fn(usize) -> bool, c: usize| {
        let idx: Vec<usize> = (0..n).filter(|&i| members(i)).collect();
        idx.iter().map(|&i| x[c][i]).sum::<f64>() / idx.len() as f64
    };
    let mu_att: Vec<f64> = (0..p).map(|c| mean(&|i| treated[i], c)).collect();
    let mu_all: Vec<f64> = (0..p).map(|c| mean(&|_| true, c)).collect();
    let mut best: Option<usize> = None;
    for mask in 0..(1_u32 << n) {
        let sel = |i: usize| mask >> i & 1 == 1;
        let nt = (0..n).filter(|&i| sel(i) && treated[i]).count();
        let nc = (0..n).filter(|&i| sel(i) && !treated[i]).count();
        let sum = |g: bool, c: usize| -> f64 {
            (0..n).filter(|&i| sel(i) && treated[i] == g).map(|i| x[c][i]).sum()
        };
        let eps = 1e-9;
        let ok = match mode {
            Mode::Card(r) => {
                (nc as f64 - r * nt as f64).abs() < 1e-12
                    && (0..p).all(|c| (sum(true, c) - sum(false, c) / r).abs() <= tols[c] * nt as f64 + eps)
            }
            Mode::ProfileAtt => {
                (0..n).all(|i| !treated[i] || sel(i))
                    && (0..p).all(|c| (sum(false, c) - nc as f64 * mu_att[c]).abs() <= tols[c] * nc as f64 + eps)
            }
            Mode::ProfileAte(ratio) => {
                nt >= 1
                    && nc >= 1
                    && ratio.is_none_or(|r| (nc as f64 - r * nt as f64).abs() < 1e-12)
                    && (0..p).all(|c| {
                        (sum(true, c) - nt as f64 * mu_all[c]).abs() <= tols[c] / 2.0 * nt as f64 + eps
                            && (sum(false, c) - nc as f64 * mu_all[c]).abs()
                                <= tols[c] / 2.0 * nc as f64 + eps
                    })
            }
        };
        if ok {
            let size = match mode {
                Mode::ProfileAtt => nc,
                _ => nt + nc,
            };
            if best.is_none_or(|b| size > b) {
                best = Some(size);
            }
        }
    }
    // A program whose largest subset is empty has no solution.
    best.filter(|&size| size > 0)
}

#[test]
fn cardinality_and_profile_sizes_agree_with_enumeration() {
    let mut rng = Rng(7);
    let mut failures = Vec::new();
    for trial in 0..240 {
        let nt = 4 + trial % 3;
        let nc = 6 + trial % 5;
        let n = nt + nc;
        let treated: Vec<bool> = (0..n).map(|i| i < nt).collect();
        let p = 1 + trial % 3;
        let x: Vec<Vec<f64>> = (0..p)
            .map(|c| {
                (0..n)
                    .map(|i| {
                        if c == 2 {
                            f64::from(u8::from(rng.next() < 0.4))
                        } else {
                            rng.next() * 2.0 + if treated[i] { 0.6 } else { 0.0 }
                        }
                    })
                    .collect()
            })
            .collect();
        let tol = 0.05 + 0.3 * rng.next();
        let tols = vec![tol; p];
        let (mode, estimand, ratio) = match trial % 5 {
            0 => (Mode::Card(1.0), Estimand::Att, Some(1.0)),
            1 => (Mode::Card(2.0), Estimand::Att, Some(2.0)),
            2 => (Mode::ProfileAtt, Estimand::Att, None),
            3 => (Mode::ProfileAte(None), Estimand::Ate, None),
            _ => (Mode::ProfileAte(Some(1.0)), Estimand::Ate, Some(1.0)),
        };
        let mut request = CardinalityRequest::new(&treated, &x);
        request.estimand = estimand;
        request.ratio = ratio;
        request.tols = tols.clone();
        request.std_tols = vec![false];
        request.limits = CardinalityLimits { time: None, nodes: None };
        let reference = brute_size(&treated, &x, &tols, mode);
        match (reference, match_cardinality(&request)) {
            (Some(r), Ok(out)) => {
                let s = &out.strata[0];
                let size = match mode {
                    Mode::ProfileAtt => s.selected_control,
                    _ => s.selected_treated + s.selected_control,
                };
                if size != r || !s.proven_optimal {
                    failures.push(format!(
                        "trial {trial} {mode:?}: {size} selected (proven {}, bound {}), optimum {r}",
                        s.proven_optimal, s.bound
                    ));
                }
            }
            (None, Err(_)) => {}
            (r, g) => failures.push(format!("trial {trial} {mode:?}: reference {r:?}, got {:?}", g.map(|o| o.strata))),
        }
    }
    assert!(failures.is_empty(), "{} failures: {:#?}", failures.len(), &failures[..failures.len().min(6)]);
}

/// Profile matching for the ATT with sampling weights and tolerances so wide
/// that every control fits: every control should be selected.
#[test]
fn profile_att_with_sampling_weights_keeps_every_control_when_all_fit() {
    let mut rng = Rng(11);
    let mut short = Vec::new();
    for trial in 0..300 {
        let (nt, nc) = (5, 7 + trial % 6);
        let n = nt + nc;
        let treated: Vec<bool> = (0..n).map(|i| i < nt).collect();
        let x = vec![(0..n).map(|_| rng.next()).collect::<Vec<f64>>()];
        let w: Vec<f64> = (0..n).map(|_| 0.05 + rng.next()).collect();
        let mut request = CardinalityRequest::new(&treated, &x);
        request.ratio = None;
        request.tols = vec![1e6];
        request.std_tols = vec![false];
        request.sampling_weights = Some(&w);
        let out = match_cardinality(&request).expect("feasible");
        let kept = out.strata[0].selected_control;
        if kept != nc {
            short.push((trial, kept, nc));
        }
    }
    assert!(short.is_empty(), "{} of 300 drop a control: {:?}", short.len(), &short[..short.len().min(5)]);
}

/// `pair_units` is public and takes slices of the lengths of the units. A
/// shorter selection, or a stratum position beyond the units, is an error.
#[test]
fn pair_units_with_inputs_of_the_wrong_length_is_an_error_not_a_panic() {
    let treated = vec![true, false, false];
    let pair_on = vec![vec![0.0, 1.0, 2.0]];
    let strata: Vec<&[usize]> = vec![&[0, 1, 2]];
    let short = vec![true, true];
    let result = pair_units(&treated, &short, &pair_on, 1, &strata, None, None);
    assert!(matches!(result, Err(CardinalityError::Shape(_))), "{result:?}");
    let full = vec![true, true, true];
    let result = pair_units(&treated, &full, &pair_on, 1, &strata, Some(&short), None);
    assert!(matches!(result, Err(CardinalityError::Shape(_))), "{result:?}");
    let result = pair_units(&treated, &full, &pair_on, 1, &strata, None, Some(&[1.0]));
    assert!(matches!(result, Err(CardinalityError::Shape(_))), "{result:?}");
    let outside: Vec<&[usize]> = vec![&[0, 1, 7]];
    let result = pair_units(&treated, &full, &pair_on, 1, &outside, None, None);
    assert!(matches!(result, Err(CardinalityError::Shape(_))), "{result:?}");
}

/// A pairing column is dropped only when it has no variance, whatever its
/// scale. A column of values near 1e-9 that differ is kept, and a constant
/// column is dropped, so a distance is defined and the pairs are the nearest.
#[test]
fn pair_units_keeps_a_pairing_column_on_a_small_scale() {
    let treated = vec![true, true, false, false];
    let selected = vec![true; 4];
    let strata: Vec<&[usize]> = vec![&[0, 1, 2, 3]];
    let small = vec![0.0, 1.0e-9, 1.1e-9, 0.1e-9];
    let constant = vec![5.0; 4];
    let pairs = pair_units(
        &treated,
        &selected,
        &[small, constant],
        1,
        &strata,
        None,
        None,
    )
    .expect("pairs");
    // Treated 0 (0) with control 3 (0.1e-9), treated 1 (1e-9) with control 2.
    assert_eq!(pairs[0], pairs[3]);
    assert_eq!(pairs[1], pairs[2]);
    assert_ne!(pairs[0], pairs[1]);
}

/// Cardinality matching whose only feasible subset is the empty one is
/// infeasible. MatchIt returns NaN control weights here.
#[test]
fn cardinality_matching_that_selects_nobody_is_not_a_success() {
    let treated = [true, true, false, false];
    let covariates = vec![vec![0.0, 0.0, 10.0, 10.0]];
    let mut r = CardinalityRequest::new(&treated, &covariates);
    r.tols = vec![0.1];
    r.std_tols = vec![false];
    match match_cardinality(&r) {
        Ok(out) => panic!(
            "Ok with weights {:?} and {} + {} selected",
            out.weights, out.strata[0].selected_treated, out.strata[0].selected_control
        ),
        Err(e) => assert_eq!(e, CardinalityError::Infeasible { key: 0 }),
    }
}
