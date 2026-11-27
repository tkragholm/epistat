//! Properties of the estimator for a pre-treatment outcome missing at random:
//! the complete-data case against the panel DR-DiD estimator, clusters,
//! cross-fitting, the simulation design of the paper, and the errors.

use epistat_did::{
    DrDidConfig, DrDidObservation, MissingAssumption, MissingConfig, MissingError, MissingFolds,
    MissingModels, MissingPreData, MissingPredictions, PanelFlatInput, att_missing_pre,
    att_missing_pre_from_predictions, estimate_drdid_panel, fit_drdid_panel_nuisance,
    missing_cross_fit_folds,
};
use epistat_inference::Clusters;
use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_testkit::assert_same_on_threads;

const U: f64 = f64::EPSILON / 2.0;

fn expit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn neumaier(values: impl Iterator<Item = f64>) -> f64 {
    let (mut sum, mut carry) = (0.0_f64, 0.0_f64);
    for v in values {
        let t = sum + v;
        carry += if sum.abs() >= v.abs() {
            (sum - t) + v
        } else {
            (v - t) + sum
        };
        sum = t;
    }
    sum + carry
}

/// The simulation design of the paper's Section 4 (their `dgp.py`), with
/// the covariates `Z` only: `pi`, the outcomes and `gamma` are linear in `Z`
/// on their scales, so every parametric model is correctly specified.
struct Sim {
    z: Mat<f64>,
    treated: Vec<bool>,
    y0: Vec<Option<f64>>,
    y1: Vec<f64>,
}

fn simulate(n: usize, seed: u64, assumption: MissingAssumption) -> Sim {
    let mut s = Stream::seeded(seed);
    let mut z = Mat::zeros(n, 4);
    let (mut treated, mut y0, mut y1) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        for j in 0..4 {
            z[(i, j)] = s.normal();
        }
        let (z1, z2, z3, z4) = (z[(i, 0)], z[(i, 1)], z[(i, 2)], z[(i, 3)]);
        let a = s.uniform() < expit(-z1 + 0.5 * z2 - 0.25 * z3 - 0.1 * z4);
        let pre = 210.0 + 27.4 * z1 + 13.7 * (z2 + z3 + z4) + s.normal();
        let post = pre + s.normal() + if a { 5.0 } else { 0.0 };
        let mut index = -0.25 * z1 - 0.1 * z2 - 0.5 * z3 + 0.3 * z4 - if a { 0.2 } else { 0.0 };
        if assumption == MissingAssumption::MarXY1 {
            index += 0.01 * post;
        }
        let observed = s.uniform() < expit(index);
        treated.push(a);
        y0.push(observed.then_some(pre));
        y1.push(post);
    }
    Sim { z, treated, y0, y1 }
}

impl Sim {
    fn data(&self) -> MissingPreData<'_> {
        MissingPreData {
            treated: &self.treated,
            outcome_pre: &self.y0,
            outcome_post: &self.y1,
            clusters: None,
        }
    }
}

const ALL: [usize; 4] = [0, 1, 2, 3];
const MODELS: MissingModels<'static> = MissingModels {
    propensity: &ALL,
    missingness: &ALL,
    outcome: &ALL,
    nested: &ALL,
};

/// Three strata of a discrete covariate, every pre-treatment outcome
/// observed. A saturated propensity model then fits the share treated in
/// each stratum and the outcome regressions the stratum means of the
/// controls, so both estimators have a closed form.
struct Strata {
    stratum: Vec<usize>,
    dummies: Mat<f64>,
    treated: Vec<bool>,
    y0: Vec<Option<f64>>,
    y1: Vec<f64>,
}

fn strata(n: usize, seed: u64) -> Strata {
    let mut s = Stream::seeded(seed);
    let mut out = Strata {
        stratum: Vec::new(),
        dummies: Mat::zeros(n, 2),
        treated: Vec::new(),
        y0: Vec::new(),
        y1: Vec::new(),
    };
    for i in 0..n {
        let c = s.below(3);
        let a = s.uniform() < [0.3, 0.5, 0.7][c];
        let pre = c as f64 + s.normal();
        let post = pre + [0.5, 1.0, -0.5][c] + if a { 2.0 } else { 0.0 } + s.normal();
        if c > 0 {
            out.dummies[(i, c - 1)] = 1.0;
        }
        out.stratum.push(c);
        out.treated.push(a);
        out.y0.push(Some(pre));
        out.y1.push(post);
    }
    out
}

/// With every `Y0` observed and a saturated model, both estimators equal the
/// closed form: the treated's mean change less the controls' mean change in
/// their stratum, and influence functions without estimation terms. Each is
/// checked against it within the errors of its own nuisance fits, measured
/// against the closed form, propagated as the comments say.
#[test]
fn complete_data_with_a_saturated_covariate_is_the_panel_dr_estimate() {
    let d = strata(3000, 11);
    let n = d.treated.len();
    let nf = n as f64;
    let delta: Vec<f64> = (0..n).map(|i| d.y1[i] - d.y0[i].unwrap()).collect();

    // The closed form: the share treated and the mean change of the
    // controls in each stratum.
    let mut n_arm = [[0.0_f64; 2]; 3];
    for i in 0..n {
        n_arm[d.stratum[i]][usize::from(d.treated[i])] += 1.0;
    }
    let mean0: Vec<f64> = (0..3)
        .map(|c| {
            neumaier(
                (0..n)
                    .filter(|&i| d.stratum[i] == c && !d.treated[i])
                    .map(|i| delta[i]),
            ) / n_arm[c][0]
        })
        .collect();
    let odds: Vec<f64> = (0..3).map(|c| n_arm[c][1] / n_arm[c][0]).collect();
    let n1 = d.treated.iter().filter(|&&a| a).count() as f64;
    let theta = neumaier(
        (0..n)
            .filter(|&i| d.treated[i])
            .map(|i| delta[i] - mean0[d.stratum[i]]),
    ) / n1;
    let phi: Vec<f64> = (0..n)
        .map(|i| {
            let r = delta[i] - mean0[d.stratum[i]];
            if d.treated[i] {
                nf / n1 * (r - theta)
            } else {
                -nf / n1 * odds[d.stratum[i]] * r
            }
        })
        .collect();
    // Rounding: a few operations per unit on values of the size of the
    // changes and the odds, and sums of n terms, each erring by at most
    // `n u` times the sum of the absolute terms (Higham, Accuracy and
    // Stability of Numerical Algorithms, 2nd ed., section 4.2).
    let size = delta.iter().map(|v| v.abs()).fold(0.0, f64::max) + 1.0;
    let w_max = odds.iter().copied().fold(0.0, f64::max);
    let rounding = 64.0 * nf * U * size * (1.0 + w_max) * nf / n1;

    // This crate's estimator, with the dummies as every model's covariates.
    let mine = att_missing_pre(
        &MissingPreData {
            treated: &d.treated,
            outcome_pre: &d.y0,
            outcome_post: &d.y1,
            clusters: None,
        },
        d.dummies.as_ref(),
        MissingAssumption::MarX,
        &MissingModels {
            propensity: &[0, 1],
            missingness: &[0, 1],
            outcome: &[0, 1],
            nested: &[],
        },
        MissingFolds::None,
        &MissingConfig::default(),
    )
    .unwrap();
    assert_eq!(mine.nuisance.gamma_treated, vec![1.0; n]);
    assert_eq!(mine.nuisance.gamma_control, vec![1.0; n]);
    // Its nuisance errors: the control weights and the regression of the
    // change, `mu1(X, 0) - mu0(X, 0)`, against the closed form. The control
    // term sums a weight constant within a stratum times residuals that sum
    // to the regression error, so the estimate errs by at most twice the
    // largest regression error; a control's influence errs by its weight
    // error times its residual plus its weight times its regression error.
    let nu = &mine.nuisance;
    let dm: Vec<f64> = (0..n)
        .map(|i| nu.post_control[i] - nu.pre_control[i] - mean0[d.stratum[i]])
        .collect();
    let dm_max = dm.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let theta_bound = 2.0 * dm_max + rounding;
    assert!(
        (mine.att - theta).abs() <= theta_bound,
        "{} vs {theta}",
        mine.att
    );
    for i in 0..n {
        let r = delta[i] - mean0[d.stratum[i]];
        let bound = if d.treated[i] {
            nf / n1 * (dm[i].abs() + theta_bound)
        } else {
            let p = nu.propensity[i];
            let w = p / (1.0 - p);
            nf / n1 * ((w - odds[d.stratum[i]]).abs() * r.abs() + w * dm[i].abs())
        } + rounding;
        assert!((mine.influence[i] - phi[i]).abs() <= bound, "unit {i}");
    }

    // The panel DR-DiD estimator of the same data, without its ridge.
    let mut design = Vec::with_capacity(3 * n);
    for i in 0..n {
        design.extend_from_slice(&[1.0, d.dummies[(i, 0)], d.dummies[(i, 1)]]);
    }
    let config = DrDidConfig {
        ridge: 0.0,
        ..DrDidConfig::default()
    };
    let rows: Vec<DrDidObservation> = (0..n)
        .map(|i| {
            DrDidObservation::builder()
                .treated(d.treated[i])
                .delta_outcome(delta[i])
                .covariates(design[3 * i..3 * i + 3].to_vec())
                .build()
        })
        .collect();
    let dr = estimate_drdid_panel(&rows, config).unwrap();
    let ones = vec![1.0; n];
    let fit = fit_drdid_panel_nuisance(
        PanelFlatInput {
            treated: &d.treated,
            delta_outcome: &delta,
            weight: &ones,
            design_matrix_flat: &design,
            feature_count: 3,
        },
        config,
    )
    .unwrap();
    // Its errors, from its own nuisance fit. With weights constant within a
    // stratum and saturated models, the estimation terms of its influence
    // function reduce to multiples of `C`, the weighted mean of the control
    // residuals, and of the regression errors: in a control's influence the
    // normalisation by `sum w` and the outcome-regression term combine to
    // `n w*_c r / n1 - n w_c C / sum w`, so the weight errors cancel. The
    // propensity term of unit `i` in stratum `c` is
    // `n (A_i - p_c) w_c (S_c - n0_c C) / (n_c p_c (1 - p_c) sum w)`, with
    // `S_c` the sum of the stratum's control residuals.
    let w_dr: Vec<f64> = fit
        .propensity_scores
        .iter()
        .map(|p| p / (1.0 - p))
        .collect();
    let resid: Vec<f64> = (0..n)
        .map(|i| delta[i] - fit.outcome_predictions[i])
        .collect();
    let w_sum = neumaier((0..n).filter(|&i| !d.treated[i]).map(|i| w_dr[i]));
    let c_mean = neumaier(
        (0..n)
            .filter(|&i| !d.treated[i])
            .map(|i| w_dr[i] * resid[i]),
    ) / w_sum;
    let dm_dr: Vec<f64> = (0..n)
        .map(|i| fit.outcome_predictions[i] - mean0[d.stratum[i]])
        .collect();
    let dm_dr_max = dm_dr.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let s_c: Vec<f64> = (0..3)
        .map(|c| {
            neumaier(
                (0..n)
                    .filter(|&i| d.stratum[i] == c && !d.treated[i])
                    .map(|i| resid[i]),
            )
        })
        .collect();
    let theta_dr_bound = dm_dr_max + 2.0 * c_mean.abs() + rounding;
    assert!(
        (dr.att - theta).abs() <= theta_dr_bound,
        "{} vs {theta}",
        dr.att
    );
    for i in 0..n {
        let c = d.stratum[i];
        let p = fit.propensity_scores[i];
        let a = if d.treated[i] { 1.0 } else { 0.0 };
        let ps = nf * (a - p).abs() * w_dr[i] * (s_c[c].abs() + n_arm[c][0] * c_mean.abs())
            / ((n_arm[c][0] + n_arm[c][1]) * p * (1.0 - p) * w_sum);
        let bound = if d.treated[i] {
            nf / n1 * (dm_dr[i].abs() + theta_dr_bound + c_mean.abs())
        } else {
            nf / n1 * odds[c] * dm_dr[i].abs() + nf * w_dr[i] * c_mean.abs() / w_sum
        } + ps
            + rounding;
        assert!(
            (dr.influence_function[i] - phi[i]).abs() <= bound,
            "unit {i}: {} vs {}, bound {bound:e}",
            dr.influence_function[i],
            phi[i]
        );
    }
}

/// With every `Y0` observed, `gamma = 1` and the panel DR-DiD estimator's own
/// nuisance fits as predictions, this estimator is that estimator with the
/// control weights divided by the number treated instead of by their sum.
#[test]
fn complete_data_is_the_panel_dr_estimate_with_unnormalised_control_weights() {
    // A change that is quadratic in z1, so that the linear outcome
    // regression is misspecified and the control residuals do not average
    // to zero under either weighting.
    let sim = simulate(4000, 5, MissingAssumption::MarX);
    let n = sim.treated.len();
    let mut noise = Stream::seeded(55);
    let complete: Vec<Option<f64>> = (0..n)
        .map(|i| {
            let z1 = sim.z[(i, 0)];
            let a = if sim.treated[i] { 2.0 } else { 0.0 };
            Some(sim.y1[i] - (1.0 + z1 * z1 + 0.5 * sim.z[(i, 1)] + a + noise.normal()))
        })
        .collect();
    let delta: Vec<f64> = (0..n).map(|i| sim.y1[i] - complete[i].unwrap()).collect();
    let mut design = Vec::with_capacity(5 * n);
    for i in 0..n {
        design.push(1.0);
        design.extend((0..4).map(|j| sim.z[(i, j)]));
    }
    let ones = vec![1.0; n];
    let input = PanelFlatInput {
        treated: &sim.treated,
        delta_outcome: &delta,
        weight: &ones,
        design_matrix_flat: &design,
        feature_count: 5,
    };
    let config = DrDidConfig::default();
    let fit = fit_drdid_panel_nuisance(input, config).unwrap();
    let rows: Vec<DrDidObservation> = (0..n)
        .map(|i| DrDidObservation {
            treated: sim.treated[i],
            delta_outcome: delta[i],
            weight: 1.0,
            covariates: design[5 * i..5 * (i + 1)].to_vec(),
        })
        .collect();
    let dr = estimate_drdid_panel(&rows, config).unwrap();
    assert!(
        fit.propensity_scores.iter().all(|&p| p < 0.995),
        "no trimming"
    );
    // Its nuisance as predictions: gamma 1, the change regression as
    // `mu1(X, 0) - mu0(X, 0)` with `mu0` zero.
    let zeros = vec![0.0; n];
    let mine = att_missing_pre_from_predictions(
        &MissingPreData {
            treated: &sim.treated,
            outcome_pre: &complete,
            outcome_post: &sim.y1,
            clusters: None,
        },
        MissingAssumption::MarX,
        &MissingPredictions {
            propensity: &fit.propensity_scores,
            gamma_treated: &ones,
            gamma_control: &ones,
            pre_treated: &zeros,
            post_control: &fit.outcome_predictions,
            pre_control: &zeros,
            nested_control: None,
        },
        &MissingConfig::default(),
    )
    .unwrap();
    // DRDID gives T - C, with C the control residuals weighted by w / sum w;
    // this estimator T - C sum w / n1. Each side sums n terms of size at
    // most `size` (Higham, section 4.2) and forms a term in a few operations.
    let n1 = sim.treated.iter().filter(|&&a| a).count() as f64;
    let w: Vec<f64> = fit
        .propensity_scores
        .iter()
        .map(|p| p / (1.0 - p))
        .collect();
    let w_sum = neumaier((0..n).filter(|&i| !sim.treated[i]).map(|i| w[i]));
    let resid: Vec<f64> = (0..n)
        .map(|i| delta[i] - fit.outcome_predictions[i])
        .collect();
    let c = neumaier((0..n).filter(|&i| !sim.treated[i]).map(|i| w[i] * resid[i])) / w_sum;
    let size = (0..n)
        .map(|i| resid[i].abs() * (1.0 + w[i]) + delta[i].abs() + sim.y1[i].abs())
        .fold(0.0, f64::max);
    let bound = 16.0 * n as f64 * U * size * n as f64 / n1;
    let expected = dr.att + c * (1.0 - w_sum / n1);
    assert!(
        (mine.att - dr.att).abs() > 1e3 * bound,
        "the weightings differ here"
    );
    assert!(
        (mine.att - expected).abs() <= bound,
        "{} vs {expected}",
        mine.att
    );
}

#[test]
fn singleton_clusters_give_the_unclustered_standard_error() {
    let sim = simulate(2000, 3, MissingAssumption::MarX);
    let plain = att_missing_pre(
        &sim.data(),
        sim.z.as_ref(),
        MissingAssumption::MarX,
        &MODELS,
        MissingFolds::None,
        &MissingConfig::default(),
    )
    .unwrap();
    let clusters = Clusters::singletons(sim.treated.len());
    let clustered = att_missing_pre(
        &MissingPreData {
            clusters: Some(&clusters),
            ..sim.data()
        },
        sim.z.as_ref(),
        MissingAssumption::MarX,
        &MODELS,
        MissingFolds::None,
        &MissingConfig::default(),
    )
    .unwrap();
    assert_eq!(plain.att.to_bits(), clustered.att.to_bits());
    assert_eq!(plain.influence, clustered.influence);
    // The same centred sum of squares formed in two ways, each erring by at
    // most `2 n u` relative.
    let bound = 4.0 * 2000.0 * U * plain.se;
    assert!((plain.se - clustered.se).abs() <= bound);
}

/// Each unit appears twice, the copies forming a cluster. Cross-fitting
/// deals clusters to folds, so both copies are predicted by the same models
/// and get the same predictions to the bit; units dealt separately would
/// have been predicted by models fitted on different folds.
#[test]
fn cross_fitting_keeps_a_cluster_in_one_fold() {
    let base = simulate(1500, 8, MissingAssumption::MarXY1);
    let n = 3000;
    let z = Mat::from_fn(n, 4, |i, j| base.z[(i / 2, j)]);
    let treated: Vec<bool> = (0..n).map(|i| base.treated[i / 2]).collect();
    let y0: Vec<Option<f64>> = (0..n).map(|i| base.y0[i / 2]).collect();
    let y1: Vec<f64> = (0..n).map(|i| base.y1[i / 2]).collect();
    let labels: Vec<usize> = (0..n).map(|i| i / 2).collect();
    let clusters = Clusters::from_labels(&labels);
    let est = att_missing_pre(
        &MissingPreData {
            treated: &treated,
            outcome_pre: &y0,
            outcome_post: &y1,
            clusters: Some(&clusters),
        },
        z.as_ref(),
        MissingAssumption::MarXY1,
        &MODELS,
        MissingFolds::CrossFit { folds: 4, seed: 4 },
        &MissingConfig::default(),
    )
    .unwrap();
    assert_eq!(est.folds, 4);
    let nu = &est.nuisance;
    let nested = nu.nested_control.as_ref().unwrap();
    for i in (0..n).step_by(2) {
        for v in [
            &nu.propensity,
            &nu.gamma_treated,
            &nu.gamma_control,
            &nu.pre_treated,
            &nu.post_control,
            &nu.pre_control,
            nested,
        ] {
            assert_eq!(v[i].to_bits(), v[i + 1].to_bits(), "unit {i}");
        }
    }
    // And the folds differ: the copies' predictions are not those of a fit
    // on all units.
    let all = att_missing_pre(
        &MissingPreData {
            treated: &treated,
            outcome_pre: &y0,
            outcome_post: &y1,
            clusters: Some(&clusters),
        },
        z.as_ref(),
        MissingAssumption::MarXY1,
        &MODELS,
        MissingFolds::None,
        &MissingConfig::default(),
    )
    .unwrap();
    assert_ne!(all.nuisance.propensity, nu.propensity);
}

/// The folds are dealt in turn from a random order, so their sizes differ
/// by at most one unit (or cluster), as do the two halves within a fold; a
/// cluster lies in one fold and one half.
#[test]
fn the_folds_are_balanced_and_keep_clusters_whole() {
    let plain = missing_cross_fit_folds(1003, None, 5, 7).unwrap();
    let mut sizes = [0_usize; 5];
    let mut halves = [[0_usize; 2]; 5];
    for (f, h) in plain.fold.iter().zip(&plain.nested_half) {
        sizes[*f] += 1;
        halves[*f][usize::from(*h)] += 1;
    }
    assert!(sizes.iter().max().unwrap() - sizes.iter().min().unwrap() <= 1);
    for h in halves {
        assert!(h[0].abs_diff(h[1]) <= 1);
    }
    assert_ne!(
        plain.fold,
        missing_cross_fit_folds(1003, None, 5, 8).unwrap().fold
    );

    let labels: Vec<usize> = (0..1000).map(|i| (i * 7) % 100).collect();
    let clusters = Clusters::from_labels(&labels);
    let grouped = missing_cross_fit_folds(1000, Some(&clusters), 4, 7).unwrap();
    for i in 0..1000 {
        for j in 0..1000 {
            if labels[i] == labels[j] {
                assert_eq!(grouped.fold[i], grouped.fold[j]);
                assert_eq!(grouped.nested_half[i], grouped.nested_half[j]);
            }
        }
    }
    assert!(missing_cross_fit_folds(1000, Some(&clusters), 101, 7).is_err());
    assert!(missing_cross_fit_folds(999, Some(&clusters), 4, 7).is_err());
}

#[test]
fn cross_fitting_does_not_depend_on_the_number_of_threads() {
    let sim = simulate(3000, 21, MissingAssumption::MarXY1);
    assert_same_on_threads(|| {
        let est = att_missing_pre(
            &sim.data(),
            sim.z.as_ref(),
            MissingAssumption::MarXY1,
            &MODELS,
            MissingFolds::CrossFit { folds: 5, seed: 9 },
            &MissingConfig::default(),
        )
        .unwrap();
        let mut out = vec![est.att, est.se];
        out.extend(est.influence);
        out.extend(est.nuisance.nested_control.unwrap());
        out
    });
}

/// On the paper's design with correctly specified models the estimate is
/// within four standard errors of the true ATT of 5, with and without
/// cross-fitting, under both assumptions. The data are fixed by the seed,
/// so this is a deterministic check that fails only if the estimator is
/// biased by more than a few standard errors. The influence function sums
/// to zero at the estimate, to the rounding of a sum of `n` terms.
#[test]
fn the_estimate_recovers_the_att_of_the_paper_design() {
    for assumption in [MissingAssumption::MarX, MissingAssumption::MarXY1] {
        let sim = simulate(20_000, 2026, assumption);
        for folds in [
            MissingFolds::None,
            MissingFolds::CrossFit { folds: 5, seed: 1 },
        ] {
            let est = att_missing_pre(
                &sim.data(),
                sim.z.as_ref(),
                assumption,
                &MODELS,
                folds,
                &MissingConfig::default(),
            )
            .unwrap();
            assert!(
                (est.att - 5.0).abs() < 4.0 * est.se,
                "{assumption:?} {folds:?}: {} se {}",
                est.att,
                est.se
            );
            assert!(est.ci_low < est.att && est.att < est.ci_high);
            let total = neumaier(est.influence.iter().copied());
            let size: f64 = est.influence.iter().map(|v| v.abs()).sum();
            assert!(total.abs() <= 4.0 * 20_000.0 * U * size);
        }
    }
}

#[test]
fn invalid_input_is_an_error() {
    let sim = simulate(500, 1, MissingAssumption::MarX);
    let n = sim.treated.len();
    let half = vec![0.5; n];
    let p = MissingPredictions {
        propensity: &half,
        gamma_treated: &half,
        gamma_control: &half,
        pre_treated: &sim.y1,
        post_control: &sim.y1,
        pre_control: &sim.y1,
        nested_control: None,
    };
    let config = MissingConfig::default();
    let run = |p: &MissingPredictions<'_>, a, config: &MissingConfig| {
        att_missing_pre_from_predictions(&sim.data(), a, p, config)
    };
    assert!(run(&p, MissingAssumption::MarX, &config).is_ok());
    assert_eq!(
        run(&p, MissingAssumption::MarXY1, &config).unwrap_err(),
        MissingError::Nested
    );
    let with_nested = MissingPredictions {
        nested_control: Some(&sim.y1),
        ..p
    };
    assert_eq!(
        run(&with_nested, MissingAssumption::MarX, &config).unwrap_err(),
        MissingError::Nested
    );
    // A gamma at the caller's floor, for a unit whose Y0 is observed.
    let floor = MissingConfig {
        gamma_floor: 0.5,
        ..config
    };
    assert!(matches!(
        run(&p, MissingAssumption::MarX, &floor),
        Err(MissingError::Gamma { .. })
    ));
    // A propensity at the ceiling, for a control.
    let ceiling = MissingConfig {
        propensity_ceiling: 0.5,
        ..config
    };
    assert!(matches!(
        run(&p, MissingAssumption::MarX, &ceiling),
        Err(MissingError::Propensity { .. })
    ));
    // A value that is not finite.
    let mut bad = sim.y1.clone();
    bad[3] = f64::NAN;
    let nan = MissingPredictions {
        post_control: &bad,
        ..p
    };
    assert!(matches!(
        run(&nan, MissingAssumption::MarX, &config),
        Err(MissingError::NotFinite { unit: 3, .. })
    ));

    let fit = |data: &MissingPreData<'_>, x: &Mat<f64>, models: &MissingModels<'_>, folds| {
        att_missing_pre(
            data,
            x.as_ref(),
            MissingAssumption::MarX,
            models,
            folds,
            &config,
        )
    };
    // No control with Y0 observed.
    let y0: Vec<Option<f64>> = (0..n)
        .map(|i| if sim.treated[i] { sim.y0[i] } else { None })
        .collect();
    let data = MissingPreData {
        outcome_pre: &y0,
        ..sim.data()
    };
    assert_eq!(
        fit(&data, &sim.z, &MODELS, MissingFolds::None).unwrap_err(),
        MissingError::NoObservedPre("control")
    );
    // A constant covariate, two that are proportional, and one that does
    // not exist.
    let mut x = Mat::from_fn(n, 5, |i, j| if j < 4 { sim.z[(i, j)] } else { 1.0 });
    assert!(matches!(
        fit(
            &sim.data(),
            &x,
            &MissingModels {
                propensity: &[0, 4],
                ..MODELS
            },
            MissingFolds::None
        ),
        Err(MissingError::Constant { .. })
    ));
    for i in 0..n {
        x[(i, 4)] = 2.0 * sim.z[(i, 0)];
    }
    assert!(matches!(
        fit(
            &sim.data(),
            &x,
            &MissingModels {
                outcome: &[0, 4],
                ..MODELS
            },
            MissingFolds::None
        ),
        Err(MissingError::Fit { .. })
    ));
    assert!(matches!(
        fit(
            &sim.data(),
            &x,
            &MissingModels {
                outcome: &[9],
                ..MODELS
            },
            MissingFolds::None
        ),
        Err(MissingError::Column {
            column: 9,
            columns: 5
        })
    ));
    assert!(matches!(
        fit(
            &sim.data(),
            &sim.z,
            &MODELS,
            MissingFolds::CrossFit { folds: 1, seed: 0 }
        ),
        Err(MissingError::Config(_))
    ));
}
