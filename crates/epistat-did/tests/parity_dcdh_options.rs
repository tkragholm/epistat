//! The options of `estimate_dcdh` and the functions around it checked
//! against `DIDmultiplegtDYN::did_multiplegt_dyn` 2.2.0 on simulated panels:
//! `trends_lin` (absorbing and on-off treatments, weighted and clustered,
//! with controls, `trends_nonparam`, normalised), `same_switchers` and
//! `same_switchers_pl` (balanced and unbalanced), `less_conservative_se`
//! (also clustered and with three treatment levels), `continuous` (orders 1
//! and 2), `normalized_weights`, `predict_het`, `by`, `by_path`, `design`,
//! `date_first_switch`, `avg_time_periods` and the bootstrap. See
//! `data/generate_dcdh_options_reference.R`, which also says where it runs
//! 2.4.0's rule instead of 2.2.0's.
//!
//! # The tolerances
//!
//! As in `parity_dcdh.rs`, every estimate, standard error and variance term
//! is a ratio of sums over at most `n` cells, so two orders of summation
//! differ by about `n eps` relative to the sum of the absolute terms, which
//! for these panels is within a factor 10 of the largest term:
//! `TOL = 16 n eps`. With `trends_lin`, effect `l` sums `l` such terms, so
//! the tolerance grows by the number of effects. With controls (and with
//! `continuous`, which adds them), the solutions of the controls' systems
//! differ by a relative `kappa eps`, with `kappa` the condition number of
//! their Gram matrix that the script records.
//!
//! The regressions of `predict_het` solve the normal equations here and
//! use a QR decomposition in R, whose solutions differ by a relative
//! `kappa_X^2 eps` for `kappa_X` the condition number of the weighted
//! design (recorded by the script), so their tolerance is `TOL kappa_X^2`.
//! The F test's statistic `b' V^-1 b / k` then moves by a relative
//! `(2 + kappa_V) TOL kappa_X^2`, with `kappa_V` the condition number of the
//! HC2 covariance (recorded too), and its p-value by at most half of that,
//! since `x f(x) <= 1/2` for the F density `f` with one or two numerator
//! degrees of freedom.

#![expect(
    clippy::float_cmp,
    reason = "numbers of switchers are whole numbers, compared exactly"
)]

use epistat_did::{
    DcdhConfig, DcdhEstimate, DcdhObservation, DcdhResult, DcdhSwitchers, dcdh_design,
    dcdh_first_switch_dates, estimate_dcdh, estimate_dcdh_bootstrap, estimate_dcdh_by,
    estimate_dcdh_by_path, estimate_dcdh_heterogeneity,
};
use epistat_linalg::{Attempt, Mat, Method, solve};
use epistat_testkit::load_json;
use serde::Deserialize;

const EPS: f64 = f64::EPSILON;

#[derive(Deserialize)]
struct Versions {
    #[serde(rename = "DIDmultiplegtDYN")]
    package: String,
    #[serde(rename = "DIDmultiplegtDYN_cpp")]
    package_cpp: String,
}

#[expect(
    clippy::struct_excessive_bools,
    reason = "the R function's logical arguments"
)]
#[derive(Deserialize, Default)]
#[serde(default)]
struct Args {
    effects: usize,
    placebo: usize,
    switchers: Option<String>,
    normalized: bool,
    weight: Option<String>,
    cluster: Option<String>,
    controls: Option<Vec<String>>,
    trends_nonparam: Option<String>,
    trends_lin: bool,
    same_switchers: bool,
    same_switchers_pl: bool,
    less_conservative_se: bool,
    continuous: Option<usize>,
    predict_het: Option<(serde_json::Value, serde_json::Value)>,
    by: Option<String>,
    by_path: Option<i64>,
}

#[derive(Deserialize)]
struct Data {
    g: Vec<i64>,
    t: Vec<i64>,
    #[serde(rename = "D")]
    d: Vec<Option<f64>>,
    #[serde(rename = "Y")]
    y: Vec<Option<f64>>,
    w: Option<Vec<f64>>,
    x1: Option<Vec<f64>>,
    x2: Option<Vec<f64>>,
    h1: Option<Vec<f64>>,
    h2: Option<Vec<f64>>,
    sub: Option<Vec<i64>>,
    stratum: Option<Vec<i64>>,
    cluster: Option<Vec<i64>>,
}

#[derive(Deserialize)]
struct VarTerms {
    ids: Vec<i64>,
    effects: Vec<Vec<Option<f64>>>,
    placebos: Vec<Vec<Option<f64>>>,
    average: Vec<Option<f64>>,
}

#[derive(Deserialize)]
struct Results {
    /// `None` for the whole panel of a case with `by` or `by_path`.
    effects: Option<Vec<Vec<Option<f64>>>>,
    ate: Option<Vec<Option<f64>>>,
    placebos: Option<Vec<Vec<Option<f64>>>>,
    p_jointeffects: Option<f64>,
    p_jointplacebo: Option<f64>,
    #[serde(rename = "delta_D_avg_total")]
    delta_d_avg_total: Option<f64>,
}

#[derive(Deserialize)]
struct Design {
    paths: Vec<Vec<f64>>,
    #[serde(rename = "const")]
    constants: Vec<f64>,
}

#[derive(Deserialize)]
struct SwitchTable {
    baseline: Option<f64>,
    times: Vec<f64>,
    n: Vec<f64>,
    share: Vec<f64>,
}

#[derive(Deserialize)]
struct AvgTime {
    average: Option<f64>,
    num: f64,
    den: f64,
    nswitch: Vec<f64>,
}

#[derive(Deserialize)]
struct Het {
    effect: i64,
    covariate: String,
    estimate: f64,
    se: f64,
    t: f64,
    lb: f64,
    ub: f64,
    n: f64,
    p_f: f64,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    args: Args,
    data: Data,
    #[serde(flatten)]
    results: Results,
    var_terms: VarTerms,
    kappa_x: Option<f64>,
    normalized_weights: Option<Vec<Vec<Option<f64>>>>,
    design_100: Option<Design>,
    design_50: Option<Design>,
    first_switch: Option<Vec<SwitchTable>>,
    first_switch_by_baseline: Option<Vec<SwitchTable>>,
    avg_time_periods: Option<AvgTime>,
    predict_het: Option<Vec<Het>>,
    het_kappa: Option<Vec<(f64, f64)>>,
    by_levels: Option<Vec<String>>,
    by_results: Option<Vec<Results>>,
}

/// R writes a vector of one value as that value.
#[derive(Deserialize)]
#[serde(untagged)]
enum OneOrMany {
    One(Option<f64>),
    Many(Vec<Option<f64>>),
}

impl OneOrMany {
    fn values(&self) -> Vec<Option<f64>> {
        match self {
            Self::One(v) => vec![*v],
            Self::Many(v) => v.clone(),
        }
    }
}

#[derive(Deserialize)]
struct Replicate {
    effects: OneOrMany,
    placebos: Option<OneOrMany>,
    ate: Option<f64>,
}

#[derive(Deserialize)]
struct Boot {
    name: String,
    args: Args,
    data: Data,
    reps: usize,
    units: Vec<i64>,
    draws: Vec<Vec<u32>>,
    replicates: Vec<Replicate>,
    bootstrap_effects: Option<Vec<Vec<Option<f64>>>>,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    cases: Vec<Case>,
    bootstrap: Vec<Boot>,
}

fn load() -> Reference {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/dcdh_options_ref.json"
    ));
    assert_eq!(r.versions.package, "2.2.0");
    assert_eq!(r.versions.package_cpp, "2.4.0");
    r
}

fn observations(args: &Args, d: &Data) -> Vec<DcdhObservation> {
    (0..d.g.len())
        .map(|i| DcdhObservation {
            group: d.g[i],
            time: d.t[i],
            outcome: d.y[i],
            treatment: d.d[i],
            weight: if args.weight.is_some() {
                d.w.as_ref().expect("weights")[i]
            } else {
                1.0
            },
            controls: if args.controls.is_some() {
                vec![d.x1.as_ref().expect("x1")[i], d.x2.as_ref().expect("x2")[i]]
            } else {
                Vec::new()
            },
            stratum: if args.trends_nonparam.is_some() {
                d.stratum.as_ref().expect("stratum")[i]
            } else {
                0
            },
            cluster: args
                .cluster
                .as_ref()
                .map(|_| d.cluster.as_ref().expect("cluster")[i]),
        })
        .collect()
}

fn config(args: &Args) -> DcdhConfig {
    DcdhConfig {
        effects: args.effects,
        placebo: args.placebo,
        normalized: args.normalized,
        switchers: match args.switchers.as_deref() {
            Some("in") => DcdhSwitchers::In,
            Some("out") => DcdhSwitchers::Out,
            _ => DcdhSwitchers::Both,
        },
        trends_lin: args.trends_lin,
        same_switchers: args.same_switchers,
        same_switchers_pl: args.same_switchers_pl,
        less_conservative_se: args.less_conservative_se,
        continuous: args.continuous,
        avg_time_periods: true,
        ..DcdhConfig::default()
    }
}

thread_local! {
    static WORST: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
}

fn close(what: &str, got: f64, want: f64, scale: f64, tol: f64) {
    let bound = tol * want.abs().max(scale);
    WORST.with(|w| w.set(w.get().max((got - want).abs() / bound)));
    assert!(
        (got - want).abs() <= bound,
        "{what}: {got} against R's {want} (difference {:e}, bound {bound:e})",
        (got - want).abs()
    );
}

/// A row of R's results: Estimate, SE, LB CI, UB CI, N, Switchers, N.w,
/// Switchers.w.
/// With `trends_lin`, R's unweighted number of switchers is that of the
/// last pass that touches the estimate, not of the pass that gives it (see
/// the module documentation of `estimate_dcdh`), so it is not compared.
fn check_row(what: &str, e: &DcdhEstimate, row: &[Option<f64>], tol: f64, dw: bool) {
    let scale = row[0].map_or(1.0, f64::abs).max(row[1].unwrap_or(0.0));
    match row[0] {
        None => assert!(e.estimate.is_none(), "{what}: R has no estimate"),
        Some(want) => close(
            &format!("{what} estimate"),
            e.estimate.expect("estimate"),
            want,
            scale,
            tol,
        ),
    }
    if let Some(want) = row[1] {
        close(&format!("{what} se"), e.se.expect("se"), want, scale, tol);
        let (lo, hi) = e.ci.expect("ci");
        close(
            &format!("{what} lower"),
            lo,
            row[2].expect("lb"),
            scale,
            tol,
        );
        close(
            &format!("{what} upper"),
            hi,
            row[3].expect("ub"),
            scale,
            tol,
        );
    }
    let count = |v: Option<f64>| v.expect("count");
    assert_eq!(e.n_cells as f64, count(row[4]), "{what} N");
    if dw {
        assert_eq!(e.n_switchers, count(row[5]), "{what} Switchers");
    }
    close(
        &format!("{what} N.w"),
        e.n_cells_weighted,
        count(row[6]),
        1.0,
        tol,
    );
    close(
        &format!("{what} Switchers.w"),
        e.n_switchers_weighted,
        count(row[7]),
        1.0,
        tol,
    );
}

fn check_results(name: &str, r: &DcdhResult, want: &Results, trends_lin: bool, tol: f64) {
    let effects = want.effects.as_ref().expect("R's effects");
    assert_eq!(r.effects.len(), effects.len(), "{name}: effects");
    for (i, (e, row)) in r.effects.iter().zip(effects).enumerate() {
        check_row(
            &format!("{name} effect {}", i + 1),
            e,
            row,
            tol,
            !trends_lin,
        );
    }
    if trends_lin {
        assert!(r.average_total_effect.estimate.is_none(), "{name}: average");
        assert!(r.delta_d_avg_total.is_none(), "{name}: delta_D_avg_total");
    } else if let Some(ate) = &want.ate {
        check_row(
            &format!("{name} average"),
            &r.average_total_effect,
            ate,
            tol,
            true,
        );
    }
    let placebos = want.placebos.clone().unwrap_or_default();
    assert_eq!(r.placebos.len(), placebos.len(), "{name}: placebos");
    for (i, (e, row)) in r.placebos.iter().zip(&placebos).enumerate() {
        check_row(
            &format!("{name} placebo {}", i + 1),
            e,
            row,
            tol,
            !trends_lin,
        );
    }
    if !trends_lin && let Some(want) = want.delta_d_avg_total {
        close(
            &format!("{name} delta_D_avg_total"),
            r.delta_d_avg_total.expect("delta_D_avg_total"),
            want,
            1.0,
            tol,
        );
    }
    let k = r.effects.len() + r.placebos.len();
    let cov = |rows: std::ops::Range<usize>| -> Vec<Vec<f64>> {
        rows.clone()
            .map(|a| rows.clone().map(|b| r.covariance[a * k + b]).collect())
            .collect()
    };
    let b_eff: Vec<f64> = r
        .effects
        .iter()
        .map(|e| e.estimate.unwrap_or(0.0))
        .collect();
    let b_pl: Vec<f64> = r
        .placebos
        .iter()
        .map(|e| e.estimate.unwrap_or(0.0))
        .collect();
    for (got, want, b, v, what) in [
        (
            r.p_joint_effects,
            want.p_jointeffects,
            &b_eff,
            cov(0..r.effects.len()),
            "joint effects",
        ),
        (
            r.p_joint_placebos,
            want.p_jointplacebo,
            &b_pl,
            cov(r.effects.len()..k),
            "joint placebos",
        ),
    ] {
        if let Some(w) = want {
            let g = got.unwrap_or_else(|| panic!("{name} {what}: no p-value"));
            let bound = p_tolerance(b, &v, tol);
            assert!(
                (g - w).abs() <= bound + 4.0 * EPS,
                "{name} {what}: {g} against R's {w}"
            );
        }
    }
}

/// The bound on a p-value of the Wald test of `b` with covariance `v`, when
/// each estimate carries the absolute error `tol m_i` with
/// `m_i = max(|b_i|, sqrt(v_ii))` and the covariance the relative error
/// `tol` in norm. With `sigma` the smallest singular value of `v`, `kappa`
/// its condition number and `q = |m| / sqrt(sigma)`, the statistic
/// `s = b' v^-1 b` moves by at most `tol (2 sqrt(s) q + q^2 + kappa s)`
/// to first order, and the p-value by at most half of that for two or more
/// degrees of freedom, where the chi-square density is below 1/2.
///
/// R's side adds the error of `matlib::Ginv`, which 2.2.0 uses for the
/// inverse: its Gauss-Jordan elimination skips row operations whose factor
/// is below `sqrt(eps)` in absolute value, so each entry of the inverse may
/// be off by `k sqrt(eps)` times the largest entry, and the statistic by that
/// times `(sum |b_i|)^2`. (2.4.0 uses `MASS::ginv`.)
fn p_tolerance(b: &[f64], v: &[Vec<f64>], tol: f64) -> f64 {
    let k = b.len();
    let m = Mat::from_fn(k, k, |i, j| v[i][j]);
    let rhs = Mat::from_fn(k, 1, |i, _| b[i]);
    let x = solve(
        m.as_ref(),
        rhs.as_ref(),
        &[Attempt::plain(Method::FullPivLu)],
    )
    .expect("covariance solves")
    .x;
    let s: f64 = (0..k).map(|i| b[i] * x[(i, 0)]).sum();
    let sv = m.singular_values().expect("singular values");
    let (big, small) = sv
        .iter()
        .fold((0.0_f64, f64::INFINITY), |(a, z), &v| (a.max(v), z.min(v)));
    let norm_m = (0..k)
        .map(|i| b[i].abs().max(v[i][i].sqrt()).powi(2))
        .sum::<f64>()
        .sqrt();
    let q = norm_m / small.sqrt();
    let inv = solve(
        m.as_ref(),
        Mat::<f64>::identity(k, k).as_ref(),
        &[Attempt::plain(Method::FullPivLu)],
    )
    .expect("covariance inverts")
    .x;
    let largest = (0..k)
        .flat_map(|i| (0..k).map(move |j| (i, j)))
        .fold(0.0_f64, |a, (i, j)| a.max(inv[(i, j)].abs()));
    let l1: f64 = b.iter().map(|v| v.abs()).sum();
    let ginv = k as f64 * EPS.sqrt() * largest * l1 * l1;
    let ours = tol * (2.0 * s.sqrt() * q + q * q + big / small * s);
    f64::midpoint(ours, ginv)
}

/// R's variance terms against the influence function, by group or by
/// cluster.
#[expect(clippy::too_many_arguments, reason = "the pieces of one comparison")]
fn check_terms(
    what: &str,
    e: &DcdhEstimate,
    want: &[Option<f64>],
    ids: &[i64],
    groups: &[i64],
    clusters: Option<&Vec<i64>>,
    delta: f64,
    tol: f64,
) {
    let Some(psi) = &e.influence else {
        assert!(want.iter().all(|v| v.is_none_or(|v| v == 0.0)), "{what}");
        return;
    };
    let got: Vec<f64> = ids
        .iter()
        .map(|id| {
            psi.iter()
                .zip(groups)
                .enumerate()
                .filter(|(j, (_, g))| match clusters {
                    Some(c) => c[*j] == *id,
                    None => **g == *id,
                })
                .map(|(_, (v, _))| v * delta)
                .sum()
        })
        .collect();
    let scale = want
        .iter()
        .flatten()
        .fold(0.0_f64, |m, v| m.max(v.abs()))
        .max(f64::MIN_POSITIVE);
    for ((id, g), w) in ids.iter().zip(&got).zip(want) {
        close(
            &format!("{what} term of {id}"),
            *g,
            w.unwrap_or(0.0),
            scale,
            tol,
        );
    }
}

fn case_tol(case: &Case, n: usize) -> f64 {
    let mut tol = 16.0 * n as f64 * EPS;
    if case.args.trends_lin {
        tol *= case.args.effects as f64;
    }
    if let Some(k) = case.kappa_x {
        tol *= k.max(1.0);
    }
    tol
}

#[test]
fn options_match_did_multiplegt_dyn() {
    let reference = load();
    for case in &reference.cases {
        if case.args.predict_het.is_some() || case.args.by.is_some() || case.args.by_path.is_some()
        {
            continue;
        }
        let obs = observations(&case.args, &case.data);
        let tol = case_tol(case, obs.len());
        let cfg = config(&case.args);
        let name = &case.name;
        let r = estimate_dcdh(&obs, &cfg).unwrap_or_else(|e| panic!("{name}: {e}"));
        check_results(name, &r, &case.results, case.args.trends_lin, tol);

        let vt = &case.var_terms;
        let cl = r.clusters.as_ref();
        for (i, e) in r.effects.iter().enumerate() {
            let delta = r.normalization.get(i).copied().flatten().unwrap_or(1.0);
            check_terms(
                &format!("{name} effect {}", i + 1),
                e,
                &vt.effects[i],
                &vt.ids,
                &r.groups,
                cl,
                delta,
                tol,
            );
        }
        for (i, e) in r.placebos.iter().enumerate() {
            let delta = r
                .placebo_normalization
                .get(i)
                .copied()
                .flatten()
                .unwrap_or(1.0);
            check_terms(
                &format!("{name} placebo {}", i + 1),
                e,
                &vt.placebos[i],
                &vt.ids,
                &r.groups,
                cl,
                delta,
                tol,
            );
        }
        if !case.args.trends_lin {
            check_terms(
                &format!("{name} average"),
                &r.average_total_effect,
                &vt.average,
                &vt.ids,
                &r.groups,
                cl,
                1.0,
                tol,
            );
        }

        // With switchers in or out only, R's weights also count the
        // switchers of the other direction (see the module documentation).
        let one_direction = case
            .args
            .switchers
            .as_deref()
            .is_some_and(|s| !s.is_empty());
        if let Some(want) = case.normalized_weights.as_ref().filter(|_| !one_direction) {
            // R's matrix: row k (the lag), column l (the effect).
            for (l, w) in r.normalized_weights.iter().enumerate() {
                let w = w.as_ref().expect("weights");
                for (k, v) in w.iter().enumerate() {
                    let r_value = want[k][l].expect("weight");
                    close(
                        &format!("{name} weight of lag {k} in effect {}", l + 1),
                        *v,
                        r_value,
                        1.0,
                        tol,
                    );
                }
                for row in want.iter().skip(l + 1) {
                    assert!(row[l].is_none(), "{name}: R has a weight past the lags");
                }
            }
        }
        if let Some(want) = &case.avg_time_periods {
            let got = r.avg_time_periods.as_ref().expect("avg_time_periods");
            match want.average {
                Some(a) => close(
                    &format!("{name} avg_cumul"),
                    got.average.expect("avg"),
                    a,
                    1.0,
                    tol,
                ),
                None => assert!(got.average.is_none(), "{name}: avg_cumul"),
            }
            close(
                &format!("{name} avg_cumul num"),
                got.numerator,
                want.num,
                1.0,
                tol,
            );
            close(
                &format!("{name} avg_cumul den"),
                got.denominator,
                want.den,
                1.0,
                tol,
            );
            assert_eq!(got.switchers.len(), want.nswitch.len(), "{name}: nswitch");
            for (g, w) in got.switchers.iter().zip(&want.nswitch) {
                close(&format!("{name} nswitch"), *g, *w, 1.0, tol);
            }
        }
        if let Some(want) = &case.design_100 {
            check_design(name, &obs, &cfg, 1.0, want, tol);
        }
        if let Some(want) = &case.design_50 {
            check_design(name, &obs, &cfg, 0.5, want, tol);
        }
        if let Some(want) = &case.first_switch {
            check_dates(name, &obs, &cfg, false, want);
        }
        if let Some(want) = &case.first_switch_by_baseline {
            check_dates(name, &obs, &cfg, true, want);
        }
    }
    eprintln!(
        "largest difference from R: {:.3} of the bound",
        WORST.with(std::cell::Cell::get)
    );
}

/// R's design rows are N, Share (percent) and the treatments. Paths with the
/// same number of switchers are ordered by first appearance in 2.2.0 and
/// lexicographically here, so both sides are compared in the second order.
fn check_design(
    name: &str,
    obs: &[DcdhObservation],
    cfg: &DcdhConfig,
    p: f64,
    want: &Design,
    tol: f64,
) {
    let got = dcdh_design(obs, cfg, p).expect("design");
    assert_eq!(
        got.effects as f64, want.constants[0],
        "{name}: design effects"
    );
    assert_eq!(
        got.n_switchers as f64, want.constants[2],
        "{name}: design switchers"
    );
    close(
        &format!("{name} design coverage"),
        100.0 * got.coverage,
        want.constants[3],
        100.0,
        tol,
    );
    assert_eq!(
        got.paths.len(),
        want.paths.len(),
        "{name}: design paths at {p}"
    );
    let mut r_paths: Vec<&Vec<f64>> = want.paths.iter().collect();
    r_paths.sort_by(|a, b| {
        b[0].partial_cmp(&a[0])
            .expect("counts")
            .then_with(|| a[2..].partial_cmp(&b[2..]).expect("treatments"))
    });
    // At a coverage below 1 the last path listed is the first past it, and
    // which of several tied paths that is depends on the order of ties.
    let last_tied = p < 1.0
        && got.paths.len() >= 2
        && r_paths
            .iter()
            .filter(|r| r[0] == r_paths[r_paths.len() - 1][0])
            .count()
            > 1;
    for (i, (g, w)) in got.paths.iter().zip(&r_paths).enumerate() {
        assert_eq!(g.n_groups as f64, w[0], "{name}: path {i} N");
        close(
            &format!("{name} path {i} share"),
            100.0 * g.share,
            w[1],
            100.0,
            tol,
        );
        if !(last_tied && w[0] == r_paths[r_paths.len() - 1][0]) {
            assert_eq!(g.treatments, w[2..].to_vec(), "{name}: path {i}");
        }
    }
}

fn check_dates(
    name: &str,
    obs: &[DcdhObservation],
    cfg: &DcdhConfig,
    by_baseline: bool,
    want: &[SwitchTable],
) {
    let got = dcdh_first_switch_dates(obs, cfg, by_baseline).expect("dates");
    assert_eq!(got.len(), want.len(), "{name}: first-switch tables");
    for (g, w) in got.iter().zip(want) {
        assert_eq!(g.baseline, w.baseline, "{name}: baseline");
        assert_eq!(g.dates.len(), w.times.len(), "{name}: dates");
        for (i, d) in g.dates.iter().enumerate() {
            assert_eq!(d.time as f64, w.times[i], "{name}: date");
            assert_eq!(d.n_groups as f64, w.n[i], "{name}: switchers at a date");
            close(
                &format!("{name} date share"),
                100.0 * d.share,
                w.share[i],
                100.0,
                16.0 * EPS,
            );
        }
    }
}

#[test]
fn heterogeneity_matches_predict_het() {
    let reference = load();
    for case in reference
        .cases
        .iter()
        .filter(|c| c.args.predict_het.is_some())
    {
        let name = &case.name;
        let obs = observations(&case.args, &case.data);
        let (vars, effects) = case.args.predict_het.as_ref().expect("predict_het");
        let vars: Vec<String> = match vars {
            serde_json::Value::String(s) => vec![s.clone()],
            v => serde_json::from_value(v.clone()).expect("names"),
        };
        let effects: Vec<i64> = match effects {
            serde_json::Value::Number(n) => vec![n.as_i64().expect("integer")],
            v => serde_json::from_value(v.clone()).expect("effects"),
        };
        let column = |v: &str| match v {
            "h1" => case.data.h1.as_ref().expect("h1"),
            _ => case.data.h2.as_ref().expect("h2"),
        };
        let x: Vec<Vec<f64>> = (0..obs.len())
            .map(|i| vars.iter().map(|v| column(v)[i]).collect())
            .collect();
        let selected: Option<Vec<usize>> = (effects != [-1]).then(|| {
            effects
                .iter()
                .map(|&e| usize::try_from(e).expect("effect"))
                .collect()
        });
        let cfg = config(&case.args);
        let got = estimate_dcdh_heterogeneity(&obs, &cfg, &x, selected.as_deref())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let want = case.predict_het.as_ref().expect("R's table");
        assert_eq!(got.len(), want.len(), "{name}: rows");
        let kappas = case.het_kappa.as_ref().expect("condition numbers");
        // The regressions in R's order: the effects, then the placebos.
        let mut order: Vec<i64> = got.iter().map(|g| g.relative_period).collect();
        order.sort_by_key(|&p| (p < 0, p.abs()));
        order.dedup();
        let base = 16.0 * obs.len() as f64 * EPS;
        for w in want {
            let j = vars
                .iter()
                .position(|v| *v == w.covariate)
                .expect("covariate");
            let g = got
                .iter()
                .find(|g| g.relative_period == w.effect && g.covariate == j)
                .unwrap_or_else(|| panic!("{name}: no row for {} {}", w.effect, w.covariate));
            let (kx, kv) = kappas[order.iter().position(|&p| p == w.effect).expect("order")];
            let tol = base * kx * kx;
            let what = format!("{name} {} {}", w.effect, w.covariate);
            let scale = w.estimate.abs().max(w.se);
            close(
                &format!("{what} estimate"),
                g.estimate,
                w.estimate,
                scale,
                tol,
            );
            close(&format!("{what} se"), g.se, w.se, scale, tol);
            close(
                &format!("{what} t"),
                g.t,
                w.t,
                w.t.abs().max(1.0),
                2.0 * tol,
            );
            close(&format!("{what} lb"), g.ci.0, w.lb, scale, tol);
            close(&format!("{what} ub"), g.ci.1, w.ub, scale, tol);
            assert_eq!(g.n as f64, w.n, "{what}: N");
            let p_tol = f64::midpoint(2.0, kv) * tol;
            assert!(
                (g.p_joint - w.p_f).abs() <= p_tol + 4.0 * EPS,
                "{what}: p {} against R's {} (bound {p_tol:e})",
                g.p_joint,
                w.p_f
            );
        }
    }
}

#[test]
fn subgroups_match_by_and_by_path() {
    let reference = load();
    for case in reference
        .cases
        .iter()
        .filter(|c| c.args.by.is_some() || c.args.by_path.is_some())
    {
        let name = &case.name;
        let obs = observations(&case.args, &case.data);
        let tol = case_tol(case, obs.len());
        let cfg = config(&case.args);
        let levels = case.by_levels.as_ref().expect("levels");
        let results = case.by_results.as_ref().expect("results");
        if case.args.by.is_some() {
            let by = case.data.sub.as_ref().expect("sub");
            let got = estimate_dcdh_by(&obs, &cfg, by).expect("by");
            assert_eq!(got.len(), levels.len(), "{name}: levels");
            for ((g, level), want) in got.iter().zip(levels).zip(results) {
                assert_eq!(g.level.to_string(), *level, "{name}: level");
                let r = g.result.as_ref().expect("estimates");
                check_results(&format!("{name} {level}"), r, want, false, tol);
            }
        } else {
            let k = case.args.by_path.expect("by_path");
            let got = estimate_dcdh_by_path(&obs, &cfg, usize::try_from(k).ok()).expect("by_path");
            assert_eq!(got.len(), levels.len(), "{name}: paths");
            for (level, want) in levels.iter().zip(results) {
                let path: Vec<f64> = level
                    .split(',')
                    .map(|v| v.parse().expect("number"))
                    .collect();
                let g = got
                    .iter()
                    .find(|g| g.path == path)
                    .unwrap_or_else(|| panic!("{name}: no path {level}"));
                let r = g.result.as_ref().expect("estimates");
                check_results(&format!("{name} {level}"), r, want, false, tol);
            }
        }
    }
}

#[test]
fn bootstrap_matches_did_multiplegt_dyn() {
    let reference = load();
    for case in &reference.bootstrap {
        let name = &case.name;
        let obs = observations(&case.args, &case.data);
        let cfg = config(&case.args);
        let tol = 16.0 * obs.len() as f64 * EPS;
        let unit = |o: &DcdhObservation| o.cluster.unwrap_or(o.group);
        let l = case.args.effects;
        // R's draws, replicate by replicate.
        let mut columns: Vec<Vec<f64>> = vec![Vec::new(); l];
        for (j, (draw, want)) in case.draws.iter().zip(&case.replicates).enumerate() {
            let rows: Vec<DcdhObservation> = obs
                .iter()
                .filter_map(|o| {
                    let u = case.units.binary_search(&unit(o)).expect("unit");
                    (draw[u] > 0).then(|| DcdhObservation {
                        weight: o.weight * f64::from(draw[u]),
                        ..o.clone()
                    })
                })
                .collect();
            let r = estimate_dcdh(&rows, &cfg).expect("replicate");
            for (i, w) in want.effects.values().iter().enumerate() {
                let got = r.effects[i].estimate.expect("effect");
                let w = w.expect("R's effect");
                close(
                    &format!("{name} replicate {j} effect {}", i + 1),
                    got,
                    w,
                    w.abs().max(1.0),
                    tol,
                );
                columns[i].push(got);
            }
            for (i, w) in want.placebos.iter().flat_map(OneOrMany::values).enumerate() {
                let w = w.expect("R's placebo");
                let got = r.placebos[i].estimate.expect("placebo");
                close(
                    &format!("{name} replicate {j} placebo {}", i + 1),
                    got,
                    w,
                    w.abs().max(1.0),
                    tol,
                );
            }
            if let Some(w) = want.ate {
                let got = r.average_total_effect.estimate.expect("average");
                close(
                    &format!("{name} replicate {j} average"),
                    got,
                    w,
                    w.abs().max(1.0),
                    tol,
                );
            }
        }
        // With those draws, R's bootstrap standard errors.
        if let Some(want) = &case.bootstrap_effects {
            for (i, row) in want.iter().enumerate() {
                let sd = sd(&columns[i]);
                close(
                    &format!("{name} R's bootstrap se {}", i + 1),
                    sd,
                    row[1].expect("se"),
                    1.0,
                    tol,
                );
            }
        }
        // The crate's own draws. Both standard errors estimate the same
        // bootstrap standard deviation `s` from `B` replicates each, with a
        // sampling standard deviation of `s sqrt((kurtosis - 1) / (4 B))`
        // each, so they differ by more than four times
        // `s sqrt((kurtosis - 1) / (2 B))` with probability below 1e-4.
        let own = estimate_dcdh_bootstrap(&obs, &cfg, case.reps, 20_251_125).expect("bootstrap");
        assert_eq!(own.draws.len(), case.reps);
        for (i, col) in columns.iter().enumerate() {
            let s = sd(col);
            let mean = col.iter().sum::<f64>() / col.len() as f64;
            let m4 = col.iter().map(|v| (v - mean).powi(4)).sum::<f64>() / col.len() as f64;
            let kurtosis = m4 / s.powi(4);
            let bound = 4.0 * s * ((kurtosis - 1.0) / (2.0 * case.reps as f64)).sqrt();
            let got = own.effects[i].se.expect("se");
            eprintln!(
                "{name} effect {}: se {got} against {s} (bound {bound})",
                i + 1
            );
            assert!(
                (got - s).abs() <= bound,
                "{name} effect {}: {got} against {s} (bound {bound})",
                i + 1
            );
        }
        // The draws do not depend on the number of threads.
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("pool")
            .install(|| estimate_dcdh_bootstrap(&obs, &cfg, 20, 7).expect("bootstrap"));
        let many = estimate_dcdh_bootstrap(&obs, &cfg, 20, 7).expect("bootstrap");
        assert_eq!(one, many, "{name}: threads");
    }
}

fn sd(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0)).sqrt()
}
