//! The targeted estimators checked against R 4.6.2: `tmle::tmle` 2.1.1 with its
//! learners replaced by `glm` (`Q.SL.library = "SL.glm"`, `g.SL.library =
//! "SL.glm"`, `cvQinit = FALSE`), and given the predictions; and the estimator
//! written out by hand in R, which reproduces both `tmle` runs before it is
//! written and gives the variance of this crate. See `data/tmle.R`.
//!
//! # The tolerances
//!
//! The glm fits run IRLS from the same start with the same stopping rule as R's,
//! so a coefficient differs by `8 kappa eps` relative to `max(|beta|, 1)`, with
//! `kappa` the largest condition number of the information matrices of the
//! nuisance models and of the fluctuation on their scaled columns (as in
//! `r_aipw.rs`). Every reported number `v` is a smooth function of the
//! coefficients, with the steps of the one-step path held at their number, and R
//! records `L = sum_j |dv / dbeta_j| max(|beta_j|, 1)` for the nuisance
//! coefficients (the whole procedure, refits included) and for the coefficients of
//! the fluctuation and of the recalibration with the predictions fixed. The
//! estimator on given predictions gets R's predictions and only the second term.
//! Sums over `n` persons add `16 n eps max(|v|, 1)`, and each of the `K` steps of
//! the one-step path moves every prediction by a few rounding errors of `ln` and
//! `exp`, which adds `64 K eps max(|v|, 1)`. A variance `V = se^2` has the
//! tolerance `2 se` times that of `se`.

use epistat_causal::aipw::{AipwInput, GlmInput, Target};
use epistat_causal::tmle::{
    OutcomeModel, OutcomeRange, PropensityBound, Targeting, Tmle, TmleOptions, tmle, tmle_glm,
};
use epistat_glm::Family;
use epistat_linalg::Mat;
use epistat_testkit::load_json;
use serde::Deserialize;

const EPS: f64 = f64::EPSILON;

#[derive(Deserialize)]
struct Versions {
    r: String,
    tmle: String,
}

#[derive(Deserialize)]
struct TmleRun {
    ate: f64,
    var_ate: f64,
    ey1: f64,
    ey0: f64,
    rr: Option<f64>,
    att: f64,
    var_att: f64,
    atc: f64,
    var_atc: f64,
    epsilon: Vec<f64>,
}

#[derive(Deserialize)]
struct Scenario {
    name: String,
    data: String,
    outcome: String,
    weighted: bool,
    gbound: Option<f64>,
    model: String,
    range: Vec<f64>,
    range_kind: String,
    restrict: bool,
    y: Vec<f64>,
    weights: Vec<f64>,
    cluster: Option<Vec<u32>>,
    mu1: Vec<f64>,
    mu0: Vec<f64>,
    propensity: Vec<f64>,
    bound: f64,
    numbers: Vec<f64>,
    steps: Vec<usize>,
    eps: Vec<f64>,
    branches: Vec<String>,
    n_used: Vec<usize>,
    n_trimmed: usize,
    lip_nuis: Vec<f64>,
    lip_eps: Vec<f64>,
    kappa: f64,
    tmle_fit: Option<TmleRun>,
    tmle_core: Option<TmleRun>,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    x: Vec<Vec<f64>>,
    x_overlap: Vec<Vec<f64>>,
    treatment: Vec<u8>,
    treatment_overlap: Vec<u8>,
    scenarios: Vec<Scenario>,
}

fn load() -> Reference {
    let r: Reference = load_json(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/tmle.json"));
    assert!(r.versions.r.starts_with("R version 4.6"));
    assert_eq!(r.versions.tmle, "2.1.1");
    r
}

const TARGETS: [Target; 3] = [Target::Ate, Target::Att, Target::Atc];

fn options(s: &Scenario) -> TmleOptions {
    TmleOptions {
        propensity_bound: match s.gbound {
            None => PropensityBound::Tmle,
            Some(b) if b == 0.0 => PropensityBound::None,
            Some(b) => PropensityBound::At(b),
        },
        range: if s.range_kind == "known" {
            OutcomeRange::Known {
                lower: s.range[0],
                upper: s.range[1],
            }
        } else {
            OutcomeRange::Observed
        },
        restrict_support: s.restrict,
        ..TmleOptions::default()
    }
}

struct Data {
    treatment: Vec<bool>,
    x: Mat<f64>,
}

fn data(r: &Reference, s: &Scenario) -> Data {
    let (a, x) = if s.data == "main" {
        (&r.treatment, &r.x)
    } else {
        (&r.treatment_overlap, &r.x_overlap)
    };
    Data {
        treatment: a.iter().map(|&t| t == 1).collect(),
        x: Mat::from_fn(x.len(), x[0].len(), |i, j| x[i][j]),
    }
}

/// The eight numbers of a result in the order of the reference.
fn numbers(r: &Tmle) -> [f64; 8] {
    let ratio = r.ratio.unwrap();
    [
        r.mean1.estimate,
        r.mean0.estimate,
        r.difference.estimate,
        ratio.estimate,
        r.mean1.se,
        r.mean0.se,
        r.difference.se,
        ratio.se,
    ]
}

/// `tmle`'s variance of the difference from the influences: `var(m IF) / m` over
/// the `m` persons used, and with clusters the variance of the cluster means of
/// `m IF` over the number of clusters.
fn tmle_variance(r: &Tmle, used: &[bool], cluster: Option<&[u32]>) -> f64 {
    let rows: Vec<usize> = (0..used.len()).filter(|&i| used[i]).collect();
    let m = rows.len() as f64;
    let values: Vec<f64> = match cluster {
        None => rows.iter().map(|&i| m * r.influence[i]).collect(),
        Some(c) => {
            let mut groups = std::collections::BTreeMap::<u32, (f64, f64)>::new();
            for &i in &rows {
                let e = groups.entry(c[i]).or_default();
                e.0 += m * r.influence[i];
                e.1 += 1.0;
            }
            groups.values().map(|(s, k)| s / k).collect()
        }
    };
    let k = values.len() as f64;
    let mean = values.iter().sum::<f64>() / k;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (k - 1.0) / k
}

struct Tolerances<'a> {
    s: &'a Scenario,
    n: usize,
    /// Whether the nuisance models were fitted by this crate.
    fitted: bool,
}

impl Tolerances<'_> {
    /// The tolerance of number `k` (0 to 23) of the reference.
    fn of(&self, k: usize) -> f64 {
        let s = self.s;
        let bound = 8.0 * s.kappa * EPS;
        let steps = match k / 8 {
            0 => 0,
            t => s.steps[t - 1],
        };
        let v = s.numbers[k].abs().max(1.0);
        let nuisance = if self.fitted {
            bound * s.lip_nuis[k]
        } else {
            0.0
        };
        nuisance
            + bound * s.lip_eps[k]
            + 16.0 * self.n as f64 * EPS * v
            + 64.0 * steps as f64 * EPS * v
    }
}

fn check(name: &str, what: &str, got: f64, want: f64, tol: f64, worst: &mut f64) {
    let d = (got - want).abs() / tol;
    assert!(
        d <= 1.0,
        "{name} {what}: {got} against {want} (tolerance {tol:e})"
    );
    *worst = worst.max(d);
}

fn compare(s: &Scenario, got: &Tmle, t: usize, tol: &Tolerances<'_>, worst: &mut f64) {
    let name = format!("{} {:?}", s.name, TARGETS[t]);
    let labels = [
        "mean1",
        "mean0",
        "difference",
        "ratio",
        "se mean1",
        "se mean0",
        "se difference",
        "se ratio",
    ];
    for (j, (label, g)) in labels.iter().zip(numbers(got)).enumerate() {
        let k = 8 * t + j;
        check(&name, label, g, s.numbers[k], tol.of(k), worst);
    }
    match (TARGETS[t], got.targeting) {
        (Target::Ate, Targeting::Fluctuation { .. }) => {
            assert_eq!(got.n_trimmed, s.n_trimmed, "{name}");
            assert_eq!(got.n_used, tol.n);
        }
        (
            _,
            Targeting::OneStep {
                steps,
                recalibrated,
            },
        ) => {
            assert_eq!(steps, s.steps[t - 1], "{name}: steps");
            assert_eq!(got.n_used, s.n_used[t - 1], "{name}: persons used");
            assert_eq!(recalibrated, s.branches[t - 1] != "none", "{name}");
        }
        (target, targeting) => panic!("{name}: {target:?} with {targeting:?}"),
    }
    assert_eq!(got.propensity_bound.to_bits(), s.bound.to_bits(), "{name}");
}

/// Compares with a `tmle` run: the estimates, the relative risk and the variances
/// of the difference.
fn compare_tmle(
    s: &Scenario,
    got: &Tmle,
    t: usize,
    run: &TmleRun,
    tol: &Tolerances<'_>,
    worst: &mut f64,
) {
    let name = format!("{} {:?} against tmle", s.name, TARGETS[t]);
    let used: Vec<bool> = got.targeted1.iter().map(Option::is_some).collect();
    let variance = tmle_variance(got, &used, s.cluster.as_deref());
    let se_tol = tol.of(8 * t + 6);
    let var_tol = 2.0 * got.difference.se * se_tol;
    match TARGETS[t] {
        Target::Ate => {
            check(
                &name,
                "ATE",
                got.difference.estimate,
                run.ate,
                tol.of(2),
                worst,
            );
            check(&name, "EY1", got.mean1.estimate, run.ey1, tol.of(0), worst);
            check(&name, "EY0", got.mean0.estimate, run.ey0, tol.of(1), worst);
            if let Some(rr) = run.rr {
                check(
                    &name,
                    "RR",
                    got.ratio.unwrap().estimate,
                    rr,
                    tol.of(3),
                    worst,
                );
            }
            check(&name, "var ATE", variance, run.var_ate, var_tol, worst);
            // On the same predictions the fluctuation coefficients differ by the
            // error of the logistic fit only.
            if !tol.fitted {
                let Targeting::Fluctuation { epsilon, .. } = got.targeting else {
                    panic!("{name}: targeting");
                };
                for ((e, r), h) in epsilon.iter().zip(&run.epsilon).zip(&s.eps) {
                    let t = 8.0 * s.kappa * EPS * r.abs().max(1.0);
                    check(&name, "epsilon", *e, *r, t, worst);
                    check(&name, "epsilon (by hand)", *e, *h, t, worst);
                }
            }
        }
        Target::Att => {
            check(
                &name,
                "ATT",
                got.difference.estimate,
                run.att,
                tol.of(10),
                worst,
            );
            check(&name, "var ATT", variance, run.var_att, var_tol, worst);
        }
        Target::Atc => {
            check(
                &name,
                "ATC",
                got.difference.estimate,
                run.atc,
                tol.of(18),
                worst,
            );
            check(&name, "var ATC", variance, run.var_atc, var_tol, worst);
        }
    }
}

#[test]
fn glm_wrapper_matches_tmle_and_the_hand_written_estimator() {
    let reference = load();
    for s in &reference.scenarios {
        let d = data(&reference, s);
        let input = GlmInput {
            outcome: &s.y,
            treatment: &d.treatment,
            outcome_design: d.x.as_ref(),
            propensity_design: d.x.as_ref(),
            weight: s.weighted.then_some(s.weights.as_slice()),
            cluster: s.cluster.as_deref(),
        };
        let family = if s.outcome == "binary" {
            Family::Binomial
        } else {
            Family::Gaussian
        };
        let model = if s.model == "pooled" {
            OutcomeModel::Pooled
        } else {
            OutcomeModel::ByArm
        };
        let tol = Tolerances {
            s,
            n: s.y.len(),
            fitted: true,
        };
        let mut worst = 0.0f64;
        for (t, &target) in TARGETS.iter().enumerate() {
            let got = tmle_glm(&input, family, model, target, &options(s)).unwrap();
            compare(s, &got, t, &tol, &mut worst);
            if let Some(run) = &s.tmle_fit {
                compare_tmle(s, &got, t, run, &tol, &mut worst);
            }
        }
        println!(
            "{}: wrapper, worst deviation in tolerances {worst:.3e}",
            s.name
        );
    }
}

#[test]
fn core_matches_tmle_given_the_predictions() {
    let reference = load();
    for s in &reference.scenarios {
        let d = data(&reference, s);
        let input = AipwInput {
            outcome: &s.y,
            treatment: &d.treatment,
            mu1: &s.mu1,
            mu0: &s.mu0,
            propensity: &s.propensity,
            weight: s.weighted.then_some(s.weights.as_slice()),
            cluster: s.cluster.as_deref(),
        };
        let tol = Tolerances {
            s,
            n: s.y.len(),
            fitted: false,
        };
        let mut worst = 0.0f64;
        for (t, &target) in TARGETS.iter().enumerate() {
            // The core recalibrates by an intercept where tmle refits on the
            // covariates.
            if t > 0 && s.branches[t - 1] == "refit" {
                continue;
            }
            let got = tmle(&input, target, &options(s)).unwrap();
            compare(s, &got, t, &tol, &mut worst);
            if let Some(run) = &s.tmle_core {
                compare_tmle(s, &got, t, run, &tol, &mut worst);
            }
        }
        println!(
            "{}: core, worst deviation in tolerances {worst:.3e}",
            s.name
        );
    }
}
