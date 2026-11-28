//! `dcdh::estimate` checked against `DIDmultiplegtDYN::did_multiplegt_dyn` on
//! simulated panels: an absorbing binary treatment, a binary treatment that
//! goes on and off (both directions, each alone, never-switchers as the only
//! controls, normalised with the equality test), a treatment with three
//! levels (with and without the rule that drops groups going both above and
//! below their baseline), an unbalanced panel with missing outcomes and
//! treatments (with both conventions for missing treatments), weights,
//! clusters, controls and `trends_nonparam`. See
//! `data/generate_dcdh_reference.R`.
//!
//! The reference runs `DIDmultiplegtDYN` 2.2.0, the last release that runs
//! without the R package polars. Where 2.4.0, which this crate follows,
//! computes something else on these data, the case says so; the
//! differences are listed in the module documentation of `dcdh`.
//!
//! # The tolerances
//!
//! Every estimate, standard error and variance term is a ratio of sums over
//! at most `n` cells (the rows of the case), each with a few products. Two
//! orders of summation then differ by at most about `n eps` relative to the
//! sum of the absolute terms, which for these panels is within a factor 10
//! of the largest term, so the tolerance is `TOL = 16 n eps`, relative to
//! the larger of the value and the largest value of its kind in the case.
//! With controls, both sides solve the small `theta_d` systems, by Cholesky
//! here and by a generalised inverse or `invsym` in R, whose solutions differ
//! by a relative `kappa eps` with `kappa` the condition number of the
//! controls' Gram matrix that R records; the tolerance grows by that
//! factor. A p-value `P(chi2 > s)` moves by at most `|ds| / 2` for two or
//! more degrees of freedom, plus the `eps` of R's `1 - pchisq`.

#[path = "common/switching.rs"]
mod switching;

use epistat_did::{DcdhEffectsEqual, DcdhOptions, DcdhSwitchers, DcdhTerm, dcdh};
use epistat_linalg::{Attempt, Mat, Method, solve};
use epistat_testkit::{Tolerance, load_json};
use serde::Deserialize;
use switching::{Columns, Row};

const EPS: f64 = f64::EPSILON;

#[derive(Deserialize)]
struct Versions {
    #[serde(rename = "DIDmultiplegtDYN")]
    package: String,
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
    only_never_switchers: bool,
    normalized: bool,
    effects_equal: bool,
    dont_drop_larger_lower: bool,
    drop_if_d_miss_before_first_switch: bool,
    weight: Option<String>,
    cluster: Option<String>,
    controls: Option<Vec<String>>,
    trends_nonparam: Option<String>,
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
struct Case {
    name: String,
    args: Args,
    data: Data,
    effects: Vec<Vec<Option<f64>>>,
    ate: Vec<Option<f64>>,
    placebos: Option<Vec<Vec<Option<f64>>>>,
    p_jointeffects: Option<f64>,
    p_jointplacebo: Option<f64>,
    p_equality_effects: Option<f64>,
    #[serde(rename = "delta_D_avg_total")]
    delta_d_avg_total: Option<f64>,
    kappa_x: Option<f64>,
    var_terms: VarTerms,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    cases: Vec<Case>,
}

fn load() -> Reference {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/dcdh_ref.json"
    ));
    assert_eq!(r.versions.package, "2.2.0");
    r
}

fn observations(case: &Case) -> Vec<Row> {
    let d = &case.data;
    let with_controls = case.args.controls.is_some();
    (0..d.g.len())
        .map(|i| Row {
            group: d.g[i],
            time: d.t[i],
            outcome: d.y[i],
            treatment: d.d[i],
            weight: if case.args.weight.is_some() {
                d.w.as_ref().expect("weights")[i]
            } else {
                1.0
            },
            controls: if with_controls {
                vec![d.x1.as_ref().expect("x1")[i], d.x2.as_ref().expect("x2")[i]]
            } else {
                Vec::new()
            },
            stratum: if case.args.trends_nonparam.is_some() {
                d.stratum.as_ref().expect("stratum")[i]
            } else {
                0
            },
            cluster: case
                .args
                .cluster
                .as_ref()
                .map(|_| d.cluster.as_ref().expect("cluster")[i]),
        })
        .collect()
}

fn config(args: &Args) -> DcdhOptions {
    DcdhOptions {
        effects: args.effects,
        placebo: args.placebo,
        normalized: args.normalized,
        switchers: match args.switchers.as_deref() {
            Some("in") => DcdhSwitchers::In,
            Some("out") => DcdhSwitchers::Out,
            _ => DcdhSwitchers::Both,
        },
        only_never_switchers: args.only_never_switchers,
        dont_drop_larger_lower: args.dont_drop_larger_lower,
        drop_if_d_miss_before_first_switch: args.drop_if_d_miss_before_first_switch,
        effects_equal: if args.effects_equal {
            DcdhEffectsEqual::All
        } else {
            DcdhEffectsEqual::No
        },
        ..DcdhOptions::default()
    }
}

thread_local! {
    /// The largest difference seen, in units of the case's `TOL`.
    static WORST: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
}

fn close(what: &str, got: f64, want: f64, scale: f64, tol: f64) {
    let bound = tol * want.abs().max(scale);
    WORST.with(|w| w.set(w.get().max((got - want).abs() / bound)));
    assert!(
        Tolerance::exact()
            .abs(tol * scale)
            .rel(tol)
            .accepts(got, want),
        "{what}: {got} against R's {want} (difference {:e}, bound {bound:e})",
        (got - want).abs()
    );
}

/// The row of R's results matrix: Estimate, SE, LB CI, UB CI, N,
/// Switchers, N.w, Switchers.w.
fn check_row(what: &str, e: &DcdhTerm, row: &[Option<f64>], tol: f64, se_too: bool) {
    let scale = row[0].map_or(1.0, f64::abs).max(row[1].unwrap_or(0.0));
    match row[0] {
        None => assert!(e.estimate.is_none(), "{what}: R has no estimate"),
        Some(want) => close(
            &format!("{what} estimate"),
            e.estimate.expect("estimate").estimate,
            want,
            scale,
            tol,
        ),
    }
    if se_too && let Some(want) = row[1] {
        let estimate = e.estimate.expect("estimate");
        close(
            &format!("{what} se"),
            estimate.se.expect("se"),
            want,
            scale,
            tol,
        );
        let (lo, hi) = (
            estimate.lower().expect("lower"),
            estimate.upper().expect("upper"),
        );
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
    assert_eq!(e.n_switchers, count(row[5]), "{what} Switchers");
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

/// R's variance terms against the influence function: by group, or by
/// cluster when clustered. R's terms are not normalised.
fn check_terms(
    what: &str,
    psi: Option<&[f64]>,
    want: &[Option<f64>],
    ids: &[i64],
    groups: &[i64],
    clusters: Option<&Vec<i64>>,
    delta: f64,
    tol: f64,
) {
    let Some(psi) = psi else {
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
    // Groups that R no longer has carry no term.
    let covered: f64 = got.iter().map(|v| v.abs()).sum();
    let all: f64 = psi.iter().map(|v| (v * delta).abs()).sum();
    if clusters.is_none() {
        close(
            &format!("{what} groups outside R's"),
            covered,
            all,
            scale,
            tol,
        );
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

/// `D b` and `D v D'` for `D` the first `k - 1` rows of the identity less
/// `1/k`, the contrasts of the equality test.
#[expect(
    clippy::needless_range_loop,
    reason = "the contrast matrix is indexed as in its formula"
)]
fn demeaned(b: &[f64], v: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let k = b.len();
    let d = |i: usize, j: usize| f64::from(u8::from(i == j)) - 1.0 / k as f64;
    let db = (0..k - 1)
        .map(|i| (0..k).map(|j| d(i, j) * b[j]).sum())
        .collect();
    let dvd = (0..k - 1)
        .map(|i| {
            (0..k - 1)
                .map(|j| {
                    let mut acc = 0.0;
                    for p in 0..k {
                        for q in 0..k {
                            acc += d(i, p) * v[p][q] * d(j, q);
                        }
                    }
                    acc
                })
                .collect()
        })
        .collect();
    (db, dvd)
}

fn check_p(what: &str, got: Option<f64>, want: Option<f64>, tol: f64) {
    match want {
        None => {}
        Some(w) => {
            let g = got.unwrap_or_else(|| panic!("{what}: no p-value"));
            assert!(
                (g - w).abs() <= tol + 4.0 * EPS,
                "{what}: {g} against R's {w}"
            );
        }
    }
}

#[test]
fn dcdh_matches_did_multiplegt_dyn() {
    let reference = load();
    for case in &reference.cases {
        let obs = observations(case);
        let columns = Columns::new(&obs);
        let n = obs.len() as f64;
        let mut tol = 16.0 * n * EPS;
        if let Some(k) = case.kappa_x {
            tol *= k.max(1.0);
        }
        // The standard errors with two baseline treatments and controls
        // differ between 2.2.0 and 2.4.0 (`rsum`); see the R script.
        let se_too = case.name != "controls_two_baselines";
        let cfg = config(&case.args);
        let r =
            dcdh::estimate(&columns.panel(), &cfg).unwrap_or_else(|e| panic!("{}: {e}", case.name));
        let name = &case.name;
        assert_eq!(r.effects.len(), case.effects.len(), "{name}: effects");
        for (i, (e, row)) in r.effects.iter().zip(&case.effects).enumerate() {
            check_row(&format!("{name} effect {}", i + 1), e, row, tol, se_too);
        }
        check_row(
            &format!("{name} average"),
            &r.average_total_effect,
            &case.ate,
            tol,
            se_too,
        );
        let placebos = case.placebos.clone().unwrap_or_default();
        assert_eq!(r.placebos.len(), placebos.len(), "{name}: placebos");
        for (i, (e, row)) in r.placebos.iter().zip(&placebos).enumerate() {
            check_row(&format!("{name} placebo {}", i + 1), e, row, tol, se_too);
        }
        if let Some(want) = case.delta_d_avg_total {
            close(
                &format!("{name} delta_D_avg_total"),
                r.delta_d_avg_total.expect("delta_D_avg_total"),
                want,
                1.0,
                tol,
            );
        }
        if !se_too {
            continue;
        }
        let k = r.effects.len() + r.placebos.len();
        let cov = |rows: std::ops::Range<usize>| -> Vec<Vec<f64>> {
            rows.clone()
                .map(|a| {
                    rows.clone()
                        .map(|b| r.covariance.get(a, b).expect("covariance"))
                        .collect()
                })
                .collect()
        };
        let b_eff: Vec<f64> = r
            .effects
            .iter()
            .map(|e| e.estimate.map_or(0.0, |e| e.estimate))
            .collect();
        let b_pl: Vec<f64> = r
            .placebos
            .iter()
            .map(|e| e.estimate.map_or(0.0, |e| e.estimate))
            .collect();
        let v_eff = cov(0..r.effects.len());
        let v_pl = cov(r.effects.len()..k);
        if case.p_jointeffects.is_some() {
            check_p(
                &format!("{name} joint effects"),
                r.p_joint_effects,
                case.p_jointeffects,
                p_tolerance(&b_eff, &v_eff, tol),
            );
        }
        if case.p_jointplacebo.is_some() {
            check_p(
                &format!("{name} joint placebos"),
                r.p_joint_placebos,
                case.p_jointplacebo,
                p_tolerance(&b_pl, &v_pl, tol),
            );
        }
        if case.p_equality_effects.is_some() {
            let (db, dvd) = demeaned(&b_eff, &v_eff);
            check_p(
                &format!("{name} equal effects"),
                r.p_equal_effects,
                case.p_equality_effects,
                p_tolerance(&db, &dvd, tol),
            );
        }

        let vt = &case.var_terms;
        let groups = columns.labels(&r.units);
        let clusters = r.clusters.as_ref().map(|c| columns.cluster_labels(c));
        let cl = clusters.as_ref();
        let l = r.effects.len();
        for i in 0..l {
            let delta = r.normalization.get(i).copied().flatten().unwrap_or(1.0);
            check_terms(
                &format!("{name} effect {}", i + 1),
                r.influence_of(i),
                &vt.effects[i],
                &vt.ids,
                &groups,
                cl,
                delta,
                tol,
            );
        }
        for i in 0..r.placebos.len() {
            let delta = r
                .placebo_normalization
                .get(i)
                .copied()
                .flatten()
                .unwrap_or(1.0);
            check_terms(
                &format!("{name} placebo {}", i + 1),
                r.influence_of(l + i),
                &vt.placebos[i],
                &vt.ids,
                &groups,
                cl,
                delta,
                tol,
            );
        }
        check_terms(
            &format!("{name} average"),
            r.influence_of(l + r.placebos.len()),
            &vt.average,
            &vt.ids,
            &groups,
            cl,
            1.0,
            tol,
        );
    }
    eprintln!(
        "largest difference from R: {:.3} of the bound",
        WORST.with(std::cell::Cell::get)
    );
}
