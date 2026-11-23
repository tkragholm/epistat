//! Refusals, a hand-computed example, the prefix-sum computation against a
//! direct loop, and a Monte Carlo check of the weighted estimates against the
//! full-cohort Cox fit and of the conditional logistic estimate against the
//! weighted one.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the inputs are a few thousand rows, far below the limits of the casts"
)]

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::clogit::{ClogitOptions, clogit};
use epistat_survival::cox::{CoxOptions, fit};
use epistat_survival::ncc::{
    NccDesign, absolute_risk, baseline, fit_weighted, inclusion_probabilities, weights,
};

fn design<'a>(
    exit: &'a [f64],
    case: &'a [bool],
    m: usize,
    entry: Option<&'a [f64]>,
) -> NccDesign<'a> {
    NccDesign {
        entry,
        exit,
        case,
        controls_per_case: m,
        stratum: None,
    }
}

#[test]
fn a_small_cohort_by_hand() {
    // Cases at 2 and 4 with one control each. At 2 the risk set has 4 people,
    // so a given other member is sampled with probability 1/3; at 4 it has 2
    // people and the one other member is sampled for certain.
    let exit = [1.0, 2.0, 3.0, 4.0, 5.0];
    let case = [false, true, false, true, false];
    let inc = inclusion_probabilities(&design(&exit, &case, 1, None)).unwrap();
    assert_eq!(inc.case_rows, [1, 3]);
    assert_eq!(inc.risk_set_size, [4, 2]);
    assert!((inc.sampling_probability[0] - 1.0 / 3.0).abs() < 1e-16);
    assert!((inc.sampling_probability[1] - 1.0).abs() < 1e-16);
    let expected = [0.0, 1.0, 1.0 / 3.0, 1.0, 1.0];
    for (g, e) in inc.probability.iter().zip(expected) {
        assert!((g - e).abs() < 1e-15, "{g} {e}");
    }
    // Person 0 is at risk at no case time, so a sampled person 0 is refused.
    let sampled = [true, false, false, false, false];
    assert!(weights(&inc, &sampled, &case).is_err());
    let sampled = [false, false, true, false, true];
    let w = weights(&inc, &sampled, &case).unwrap();
    assert_eq!(w, [0.0, 1.0, 3.0, 1.0, 1.0]);
}

#[test]
fn refuses_what_has_no_meaning() {
    let exit = [1.0, 2.0, 3.0, 4.0];
    let case = [false, true, false, true];
    assert!(inclusion_probabilities(&design(&exit, &case, 0, None)).is_err());
    // Three controls from a risk set of two others.
    let text = inclusion_probabilities(&design(&exit, &case, 3, None))
        .unwrap_err()
        .to_string();
    assert!(text.contains("fewer than"), "{text}");
    assert!(inclusion_probabilities(&design(&exit, &[false; 4], 1, None)).is_err());
    assert!(inclusion_probabilities(&design(&exit, &case[..3], 1, None)).is_err());
    let entry = [0.0, 2.0, 0.0, 0.0];
    let text = inclusion_probabilities(&design(&exit, &case, 1, Some(&entry)))
        .unwrap_err()
        .to_string();
    assert!(text.contains("enters at or after"), "{text}");
    let bad = [1.0, f64::NAN, 3.0, 4.0];
    assert!(inclusion_probabilities(&design(&bad, &case, 1, None)).is_err());
}

/// Cohort of `n` with exits, entries and cases from a seed.
fn cohort(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<bool>, Vec<u32>) {
    let mut rng = Stream::seeded(seed);
    let mut exit = Vec::new();
    let mut entry = Vec::new();
    let mut case = Vec::new();
    let mut stratum = Vec::new();
    for _ in 0..n {
        let t = rng.exp1() / 0.1;
        let c = rng.exp1() / 0.04;
        exit.push(t.min(c).min(6.0));
        entry.push(if rng.below(3) == 0 {
            rng.uniform() * 0.5 * exit[exit.len() - 1]
        } else {
            0.0
        });
        case.push(t <= c.min(6.0));
        stratum.push(rng.below(3) as u32);
    }
    (entry, exit, case, stratum)
}

#[test]
fn prefix_sums_match_a_direct_loop() {
    // Short follow-up among many case times, where a difference of two long
    // prefix sums would lose digits in plain f64.
    let n = 3000;
    let (entry, exit, case, stratum) = cohort(n, 12);
    let m = 2;
    let design = NccDesign {
        entry: Some(&entry),
        exit: &exit,
        case: &case,
        controls_per_case: m,
        stratum: Some(&stratum),
    };
    let inc = inclusion_probabilities(&design).unwrap();
    let cases: Vec<usize> = (0..n).filter(|&i| case[i]).collect();
    // The direct loop: per case, in its stratum.
    let size: Vec<usize> = (0..n)
        .map(|j| {
            (0..n)
                .filter(|&k| stratum[k] == stratum[j] && entry[k] < exit[j] && exit[j] <= exit[k])
                .count()
        })
        .collect();
    let mut worst = 0.0f64;
    for i in 0..n {
        if case[i] {
            assert!((inc.probability[i] - 1.0).abs() < f64::EPSILON);
            continue;
        }
        let mut sum = 0.0f64;
        let mut comp = 0.0f64;
        for &j in &cases {
            if stratum[j] != stratum[i] || !(entry[i] < exit[j] && exit[j] <= exit[i]) {
                continue;
            }
            let term = (-(m as f64) / (size[j] - 1) as f64).ln_1p();
            // Neumaier summation.
            let t = sum + term;
            comp += if sum.abs() >= term.abs() {
                (sum - t) + term
            } else {
                (term - t) + sum
            };
            sum = t;
        }
        let p = -(sum + comp).exp_m1();
        if p > 0.0 {
            worst = worst.max((inc.probability[i] - p).abs() / p);
        } else {
            assert_eq!(inc.probability[i].to_bits(), 0.0f64.to_bits());
        }
    }
    // The direct sum is compensated, so the difference is that of the prefix
    // sums in double-double and of `exp_m1`: a few epsilon.
    assert!(worst < 1e-14, "relative deviation {worst}");
}

/// Sample `m` controls at each case time from the risk set, without
/// replacement, by a partial shuffle.
type Sets = Vec<(usize, Vec<usize>)>;

fn sample(
    entry: &[f64],
    exit: &[f64],
    case: &[bool],
    m: usize,
    rng: &mut Stream,
) -> (Vec<bool>, Sets) {
    let n = exit.len();
    let mut sampled = vec![false; n];
    let mut sets = Vec::new();
    for j in (0..n).filter(|&j| case[j]) {
        let mut risk: Vec<usize> = (0..n)
            .filter(|&i| i != j && entry[i] < exit[j] && exit[j] <= exit[i])
            .collect();
        let mut chosen = Vec::new();
        for k in 0..m {
            let pick = k + rng.below(risk.len() - k);
            risk.swap(k, pick);
            sampled[risk[k]] = true;
            chosen.push(risk[k]);
        }
        sets.push((j, chosen));
    }
    for i in 0..n {
        sampled[i] |= case[i];
    }
    (sampled, sets)
}

#[test]
fn weighted_estimates_are_unbiased_over_simulated_studies() {
    let studies = 200;
    let n = 600;
    let m = 3;
    let mut rng = Stream::seeded(2026);
    let mut d_beta = [Vec::new(), Vec::new()];
    let mut d_risk = Vec::new();
    let mut d_clogit = [Vec::new(), Vec::new()];
    let profile = Mat::from_fn(1, 2, |_, j| [0.5, 1.0][j]);
    let at = [3.0];
    for _ in 0..studies {
        let (mut x1, mut x2) = (Vec::new(), Vec::new());
        let (mut entry, mut exit, mut case) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..n {
            let a = rng.normal();
            let b = f64::from(u8::from(rng.below(10) < 4));
            let t = rng.exp1() / (0.1 * (0.5 * a - 0.4 * b).exp());
            let c = rng.exp1() / 0.04;
            exit.push(t.min(c).min(6.0));
            entry.push(if rng.below(3) == 0 {
                rng.uniform() * 0.5 * exit[exit.len() - 1]
            } else {
                0.0
            });
            case.push(t <= c.min(6.0));
            x1.push(a);
            x2.push(b);
        }
        let all_x = Mat::from_fn(n, 2, |i, j| if j == 0 { x1[i] } else { x2[i] });
        let status: Vec<i32> = case.iter().map(|&c| i32::from(c)).collect();
        let full_data = SurvData::builder(&exit, &status)
            .start(&entry)
            .build()
            .unwrap();
        let opts = CoxOptions::default();
        let full = fit(&full_data, all_x.as_ref(), &opts).unwrap();
        let ones = vec![1.0; n];
        let full_base = baseline(&full, &full_data, all_x.as_ref(), &ones, &opts).unwrap();
        let full_risk = absolute_risk(&full_base, &full.coefficients, profile.as_ref(), &at)
            .unwrap()[0][0]
            .unwrap();

        let (sampled, sets) = sample(&entry, &exit, &case, m, &mut rng);
        let inc = inclusion_probabilities(&design(&exit, &case, m, Some(&entry))).unwrap();
        let w_all = weights(&inc, &sampled, &case).unwrap();
        let rows: Vec<usize> = (0..n).filter(|&i| sampled[i]).collect();
        let stop: Vec<f64> = rows.iter().map(|&i| exit[i]).collect();
        let start: Vec<f64> = rows.iter().map(|&i| entry[i]).collect();
        let st: Vec<i32> = rows.iter().map(|&i| status[i]).collect();
        let w: Vec<f64> = rows.iter().map(|&i| w_all[i]).collect();
        let x = Mat::from_fn(rows.len(), 2, |i, j| all_x[(rows[i], j)]);
        let data = SurvData::builder(&stop, &st).start(&start).build().unwrap();
        let model = fit_weighted(&data, x.as_ref(), &w, &opts).unwrap();
        let base = baseline(&model, &data, x.as_ref(), &w, &opts).unwrap();
        let risk = absolute_risk(&base, &model.coefficients, profile.as_ref(), &at).unwrap()[0][0]
            .unwrap();
        for j in 0..2 {
            d_beta[j].push(model.coefficients[j].unwrap() - full.coefficients[j].unwrap());
        }
        d_risk.push(risk - full_risk);

        // Conditional logistic regression on the same sampled sets.
        let (mut set_case, mut set_id, mut set_rows) = (Vec::new(), Vec::new(), Vec::new());
        for (k, (j, controls)) in sets.iter().enumerate() {
            set_case.push(true);
            set_id.push(k as u32);
            set_rows.push(*j);
            for &c in controls {
                set_case.push(false);
                set_id.push(k as u32);
                set_rows.push(c);
            }
        }
        let cx = Mat::from_fn(set_rows.len(), 2, |i, j| all_x[(set_rows[i], j)]);
        let cl = clogit(&set_case, &set_id, cx.as_ref(), &ClogitOptions::default()).unwrap();
        for j in 0..2 {
            d_clogit[j].push(cl.coefficients[j].unwrap() - model.coefficients[j].unwrap());
        }
    }
    let check = |name: &str, d: &[f64]| {
        let mean = d.iter().sum::<f64>() / d.len() as f64;
        let var = d.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (d.len() - 1) as f64;
        let se = (var / d.len() as f64).sqrt();
        eprintln!(
            "{name}: mean difference {mean:.5}, Monte Carlo SE {se:.5}, ratio {:.2}",
            mean / se
        );
        assert!(mean.abs() < 3.0 * se, "{name}: bias {mean} against {se}");
    };
    check("weighted beta 1 less full cohort", &d_beta[0]);
    check("weighted beta 2 less full cohort", &d_beta[1]);
    check("weighted risk at 3 less full cohort", &d_risk);
    check("clogit beta 1 less weighted", &d_clogit[0]);
    check("clogit beta 2 less weighted", &d_clogit[1]);
}
