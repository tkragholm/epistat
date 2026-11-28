//! Residuals and the test of proportional hazards of Cox fits checked against
//! R 4.6.2's `survival` 3.8.12: `residuals.coxph` of every type and `cox.zph`
//! with each transform.
//!
//! # The tolerances
//!
//! Both implementations iterate to the same coefficients up to the error of the
//! solve, which is about the condition number `kappa` of the information matrix
//! on the scaled covariates times the machine epsilon. As in `r_cox.rs`, a
//! coefficient differs by `bound = 8 kappa eps` relative to `max(|beta|, 1)`.
//! The largest error this puts in a linear predictor is `bound * X` with
//! `X = sum_j max_i |x_ij| max(|beta_j|, 1)`. A risk score then has a relative
//! error of at most `bound * X`, a sum of risk scores over a risk set the same,
//! and a ratio of two such sums at most twice that. A residual is a row term
//! less a sum over event times of such ratios, so it is off by at most
//! `2 bound X` times its scale, and a running sum of `n` terms adds `n eps`
//! times the largest value that the sum takes, which is not the scale of the
//! residual when the residual is a difference of two running sums. The largest
//! value of the running sum of hazards is bounded by `theta_i H_s`, with `H_s`
//! the sum of the hazard increments over all event times of the stratum of row
//! `i`, which the test computes from the definition of the partial likelihood
//! and R's coefficients (`cs_i = theta_i H_s`). Write `bx = 2 bound X`,
//! `ne = n eps` and `rel = bx + ne`.
//!
//! The scale of an entry is what the sums are made of, not the size of the
//! residual, which can be much smaller after cancellation:
//!
//! - martingale: `bx (1 + |m|) + ne cs_i`, since `theta H` is `1 - m` for an
//!   event row and `-m` for the others;
//! - Schoenfeld: `rel max_i |x_ij|`;
//! - score: `max_i |x_ij| (bx (2 + |m|) + ne (2 + cs_i))`, the two terms of the
//!   residual, times the case weight when weighted;
//! - `dfbeta`: `sum_k scale_ik |V_kj|` from the score scales, plus `p bound`
//!   times `sum_k |S_ik V_kj|` for the error of `V`, which is that of the
//!   covariance matrices of `r_cox.rs`; `dfbetas` is the same over the standard
//!   error;
//! - scaled Schoenfeld: `d sum_k |V_kj| tol_ek` for the residuals,
//!   `p bound d sum_k |V_kj r_ek|` for `V`, and `bound max(|beta_j|, 1)` for the
//!   coefficient;
//! - deviance: that of the martingale residual times the derivative of the
//!   deviance with respect to it;
//! - collapsed: the sum of the tolerances of the rows of the group.
//!
//! The test of proportional hazards inverts the `2p` by `2p` information
//! matrix of the extended model, whose condition number on its scaled form
//! `kappa_z` R's file records, from a score and an information that carry the
//! errors above. Its statistic and scaled residuals are then off by at most
//! `4 kappa_z rel` relative to their scale: for the statistic the larger of 1
//! and its value, for a scaled residual `sum_k xmax_k |var_jk|`, the size of the
//! residual less its coefficient and `max(|beta_j|, 1)`, and for the variance the
//! geometric mean of the two variances. The factor 4 is the quadratic form
//! counting the score twice and the information once, and a margin of one for
//! the sums that make up the score cancelling. A p-value is compared over the
//! interval that the tolerance of the statistic leaves, plus `64 eps` for the
//! two implementations of the chi-square distribution.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the data have a few hundred rows, far below the limits of the casts"
)]

use std::collections::HashMap;

use epistat_linalg::Mat;
use epistat_stats::dist::chi_square::sf as chi_square_sf;
use epistat_survival::SurvData;
use epistat_survival::cox::{
    Collapse, CoxModelFit, CoxOptions, CoxResidualKind, CoxResiduals, ResidualOptions, Ties,
    ZphOptions, ZphTransform, fit, residuals, zph,
};
use epistat_testkit::{load_json, na_f64_vec};
use serde::Deserialize;

type Matrix = Vec<Vec<Option<f64>>>;

#[derive(Deserialize)]
struct Versions {
    r: String,
    survival: String,
    jsonlite: String,
}

#[derive(Deserialize)]
struct Dataset {
    #[serde(default)]
    start: Vec<f64>,
    stop: Vec<f64>,
    event: Vec<u8>,
    stratum: Vec<u32>,
    weight: Vec<f64>,
    id: Vec<u32>,
    grp: Vec<u32>,
    columns: HashMap<String, Vec<f64>>,
}

#[derive(Deserialize)]
struct Residuals {
    martingale: Vec<f64>,
    martingale_weighted: Vec<f64>,
    deviance: Vec<f64>,
    score: Matrix,
    score_weighted: Matrix,
    dfbeta: Matrix,
    dfbeta_unweighted: Matrix,
    dfbetas: Matrix,
    schoenfeld: Matrix,
    schoenfeld_weighted: Matrix,
    schoenfeld_time: Vec<f64>,
    schoenfeld_row: Vec<u32>,
    schoenfeld_stratum: Vec<u32>,
    #[serde(default)]
    scaledsch: Option<Matrix>,
    martingale_collapse_codes: Matrix,
    deviance_collapse_codes: Matrix,
    score_collapse_codes: Matrix,
    dfbeta_collapse_codes: Matrix,
    dfbetas_collapse_codes: Matrix,
    martingale_weighted_collapse_codes: Matrix,
    score_weighted_collapse_codes: Matrix,
    status_collapse_codes: Vec<f64>,
    #[serde(default)]
    martingale_collapse_cluster: Option<Matrix>,
    #[serde(default)]
    score_collapse_cluster: Option<Matrix>,
    #[serde(default)]
    dfbeta_collapse_cluster: Option<Matrix>,
    #[serde(default)]
    dfbetas_collapse_cluster: Option<Matrix>,
    #[serde(default)]
    status_collapse_cluster: Option<Vec<f64>>,
}

#[derive(Deserialize)]
struct RZph {
    transform: String,
    singledf: bool,
    table: Vec<Vec<f64>>,
    x: Vec<f64>,
    time: Vec<f64>,
    stratum: Vec<u32>,
    row: Vec<u32>,
    y: Matrix,
    var: Vec<Vec<f64>>,
    kappa: f64,
}

#[derive(Deserialize)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "one flag per option of the R call"
)]
struct RFit {
    name: String,
    data: String,
    columns: Vec<String>,
    ties: String,
    use_start: bool,
    use_strata: bool,
    use_weights: bool,
    use_cluster: bool,
    #[serde(default)]
    terms: Option<Vec<Vec<usize>>>,
    #[serde(deserialize_with = "na_f64_vec")]
    coefficients: Vec<f64>,
    vcov: Matrix,
    information_condition: f64,
    residuals: Residuals,
    zph: Vec<RZph>,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    datasets: HashMap<String, Dataset>,
    fits: Vec<RFit>,
}

const EPS: f64 = f64::EPSILON;

struct Setup<'a> {
    r: &'a RFit,
    ds: &'a Dataset,
    x: Mat<f64>,
    status: Vec<i32>,
    fit: CoxModelFit,
    ties: Ties,
    /// `bx = 2 bound X`, `ne = n eps`, `rel = bx + ne`, and `bound`.
    bx: f64,
    ne: f64,
    rel: f64,
    /// `theta_i H_s` of each row.
    cs: Vec<f64>,
    bound: f64,
    /// `max_i |x_ij|` of each column.
    xmax: Vec<f64>,
    n: usize,
    p: usize,
}

/// `theta_i H_s` for every row: the sum over the event times of the stratum of
/// the Breslow or Efron hazard increments, from the definition, times the risk
/// score of the row.
fn cumulative_scale(ds: &Dataset, r: &RFit, x: &Mat<f64>) -> Vec<f64> {
    let n = ds.stop.len();
    let eta: Vec<f64> = (0..n)
        .map(|i| {
            (0..r.columns.len())
                .filter(|&j| !r.coefficients[j].is_nan())
                .map(|j| x[(i, j)] * r.coefficients[j])
                .sum()
        })
        .collect();
    let stratum = |i: usize| if r.use_strata { ds.stratum[i] } else { 0 };
    let weight = |i: usize| if r.use_weights { ds.weight[i] } else { 1.0 };
    let start = |i: usize| {
        if r.use_start {
            ds.start[i]
        } else {
            f64::NEG_INFINITY
        }
    };
    let mut total: HashMap<u32, f64> = HashMap::new();
    let mut seen: Vec<(u32, u64)> = Vec::new();
    for i in 0..n {
        let key = (stratum(i), ds.stop[i].to_bits());
        if ds.event[i] == 0 || seen.contains(&key) {
            continue;
        }
        seen.push(key);
        let t = ds.stop[i];
        let (mut s0, mut e0, mut wsum, mut d) = (0.0f64, 0.0f64, 0.0f64, 0usize);
        for (j, &eta_j) in eta.iter().enumerate() {
            if stratum(j) == stratum(i) && start(j) < t && t <= ds.stop[j] {
                let risk = weight(j) * eta_j.exp();
                s0 += risk;
                if ds.event[j] == 1 && ds.stop[j].to_bits() == t.to_bits() {
                    e0 += risk;
                    wsum += weight(j);
                    d += 1;
                }
            }
        }
        let increment = if r.ties == "efron" {
            (0..d)
                .map(|k| wsum / d as f64 / (s0 - k as f64 / d as f64 * e0))
                .sum::<f64>()
        } else {
            wsum / s0
        };
        *total.entry(stratum(i)).or_insert(0.0) += increment;
    }
    (0..n)
        .map(|i| eta[i].exp() * total.get(&stratum(i)).copied().unwrap_or(0.0))
        .collect()
}

fn build<'a>(ds: &'a Dataset, status: &'a [i32], r: &RFit, cluster: bool) -> SurvData<'a> {
    let mut builder = SurvData::builder(&ds.stop, status);
    if r.use_start {
        builder = builder.start(&ds.start);
    }
    if r.use_strata {
        builder = builder.stratum(&ds.stratum);
    }
    if r.use_weights {
        builder = builder.weight(&ds.weight);
    }
    if cluster && r.use_cluster {
        builder = builder.cluster(&ds.id);
    }
    builder.build().unwrap()
}

fn setup<'a>(reference: &'a Reference, r: &'a RFit) -> Setup<'a> {
    let ds = &reference.datasets[&r.data];
    let n = ds.stop.len();
    let p = r.columns.len();
    let x = Mat::from_fn(n, p, |i, j| ds.columns[&r.columns[j]][i]);
    let status: Vec<i32> = ds.event.iter().map(|&e| i32::from(e)).collect();
    let ties = if r.ties == "efron" {
        Ties::Efron
    } else {
        Ties::Breslow
    };
    let opts = CoxOptions {
        ties,
        ..CoxOptions::default()
    };
    let data = build(ds, &status, r, true);
    let model = fit(&data, x.as_ref(), &opts).unwrap();
    let bound = 8.0 * r.information_condition * EPS;
    let xmax: Vec<f64> = (0..p)
        .map(|j| (0..n).map(|i| x[(i, j)].abs()).fold(0.0, f64::max))
        .collect();
    let big_x: f64 = (0..p)
        .filter(|&j| !r.coefficients[j].is_nan())
        .map(|j| xmax[j] * r.coefficients[j].abs().max(1.0))
        .sum();
    let cs = cumulative_scale(ds, r, &x);
    Setup {
        r,
        ds,
        x,
        status,
        fit: model,
        ties,
        bx: 2.0 * bound * big_x,
        ne: n as f64 * EPS,
        rel: 2.0 * bound * big_x + n as f64 * EPS,
        cs,
        bound,
        xmax,
        n,
        p,
    }
}

impl Setup<'_> {
    fn opts(&self) -> CoxOptions {
        CoxOptions {
            ties: self.ties,
            ..CoxOptions::default()
        }
    }

    fn data(&self) -> SurvData<'_> {
        build(self.ds, &self.status, self.r, true)
    }

    fn residuals(
        &self,
        kind: CoxResidualKind,
        weighted: Option<bool>,
        collapse: Collapse<'_>,
    ) -> CoxResiduals {
        let opts = ResidualOptions {
            kind,
            weighted,
            collapse,
            cox: self.opts(),
        };
        let got = residuals(&self.data(), self.x.as_ref(), &self.fit, &opts).unwrap();
        if got.columns.len() == self.p {
            for (j, column) in got.columns.iter().enumerate() {
                assert_eq!(
                    column.is_none(),
                    self.aliased(j),
                    "{} column {j}",
                    self.r.name
                );
            }
        }
        got
    }

    fn weight(&self, i: usize) -> f64 {
        if self.r.use_weights {
            self.ds.weight[i]
        } else {
            1.0
        }
    }

    fn v(&self, k: usize, j: usize) -> f64 {
        self.r.vcov[k][j].unwrap_or(0.0)
    }

    fn aliased(&self, j: usize) -> bool {
        self.r.coefficients[j].is_nan()
    }
}

/// The matrix of R as rows of numbers with NaN for NA.
fn entry(m: &Matrix, i: usize, j: usize) -> Option<f64> {
    m[i][j]
}

/// The worst `|got - expected| / tol` over the entries of one column, with a
/// check that `None` and NA coincide.
fn worst_column(
    got: Option<&Vec<f64>>,
    expected: &dyn Fn(usize) -> Option<f64>,
    length: usize,
    tol: &dyn Fn(usize) -> f64,
    what: &str,
) -> f64 {
    let mut worst = 0.0f64;
    match got {
        // A column with no coefficient is `None`; R computes one anyway.
        None => {}
        Some(values) => {
            assert_eq!(values.len(), length, "{what}: length");
            for (i, g) in values.iter().enumerate() {
                let e = expected(i).unwrap_or_else(|| panic!("{what}: R has NA at {i}, got {g}"));
                let ratio = (g - e).abs() / tol(i);
                assert!(
                    ratio <= 1.0,
                    "{what} entry {i}: got {g}, R {e}, tolerance {}",
                    tol(i)
                );
                worst = worst.max(ratio);
            }
        }
    }
    worst
}

/// Group sums of `values` by `codes` in ascending order of the code.
fn group_sum(codes: &[u32], values: &[f64]) -> Vec<f64> {
    let mut labels: Vec<u32> = codes.to_vec();
    labels.sort_unstable();
    labels.dedup();
    let mut out = vec![0.0; labels.len()];
    for (c, v) in codes.iter().zip(values) {
        out[labels.binary_search(c).unwrap()] += v;
    }
    out
}

/// The deviance for a martingale residual `m` and status `s`, and the absolute
/// derivative of it with respect to `m`.
fn deviance(m: f64, s: f64) -> (f64, f64) {
    let (inner, slope) = if s == 0.0 {
        (m, -1.0)
    } else {
        (m + s * (-m / s).ln_1p(), 1.0 - 1.0 / (1.0 - m / s))
    };
    let f = -2.0 * inner;
    let d = f.max(0.0).sqrt() * m.signum();
    let derivative = if f > 0.0 {
        (2.0 * slope).abs() / (2.0 * f.sqrt())
    } else {
        1.0
    };
    (if m == 0.0 { 0.0 } else { d }, derivative)
}

type ClusterCase<'a> = (
    &'a str,
    CoxResidualKind,
    &'a Option<Matrix>,
    &'a dyn Fn(usize, usize) -> f64,
);

struct Worst(Vec<(String, f64)>);

impl Worst {
    fn add(&mut self, what: &str, value: f64) {
        assert!(value <= 1.0, "{what}: {value} of the tolerance");
        self.0.push((what.to_string(), value));
    }
}

/// Every kind of residual of one fit against R.
#[expect(clippy::too_many_lines, reason = "one block per kind of residual")]
fn check_residuals(s: &Setup<'_>, worst: &mut Worst) {
    let r = &s.r.residuals;
    let p = s.p;
    let n = s.n;
    let name = &s.r.name;
    let rows: Vec<usize> = (0..n).collect();
    let mart = &r.martingale;

    // Martingale, unweighted and weighted.
    for (weighted, expected) in [(false, &r.martingale), (true, &r.martingale_weighted)] {
        let got = s.residuals(CoxResidualKind::Martingale, Some(weighted), Collapse::None);
        assert_eq!(got.rows, rows.iter().map(|&i| i as u32).collect::<Vec<_>>());
        let w = |i: usize| if weighted { s.weight(i) } else { 1.0 };
        let value = worst_column(
            got.columns[0].as_ref(),
            &|i| Some(expected[i]),
            n,
            &|i| s.bx * (1.0 + mart[i].abs()) + s.ne * s.cs[i] * w(i),
            &format!("{name} martingale weighted {weighted}"),
        );
        worst.add(&format!("{name} martingale weighted={weighted}"), value);
    }
    let by_default = s.residuals(CoxResidualKind::Martingale, None, Collapse::None);
    assert_eq!(by_default.columns[0].as_ref().unwrap().len(), n);

    // Deviance. The derivative is evaluated at R's martingale residual.
    let status: Vec<f64> = s.status.iter().map(|&e| f64::from(e)).collect();
    let got = s.residuals(CoxResidualKind::Deviance, None, Collapse::None);
    let value = worst_column(
        got.columns[0].as_ref(),
        &|i| Some(r.deviance[i]),
        n,
        &|i| s.bx * (1.0 + mart[i].abs()) + s.ne * s.cs[i] * deviance(mart[i], status[i]).1,
        &format!("{name} deviance"),
    );
    worst.add(&format!("{name} deviance"), value);

    // Score residuals.
    let xs = |j: usize, i: usize| {
        s.xmax[j] * (s.bx * (2.0 + mart[i].abs()) + s.ne * (2.0 + s.cs[i])) * s.weight(i)
    };
    for (weighted, expected) in [(false, &r.score), (true, &r.score_weighted)] {
        let got = s.residuals(CoxResidualKind::Score, Some(weighted), Collapse::None);
        let mut value = 0.0f64;
        for (j, column) in got.columns.iter().enumerate().take(p) {
            value = value.max(worst_column(
                column.as_ref(),
                &|i| entry(expected, i, j),
                n,
                &|i| xs(j, i) / if weighted { 1.0 } else { s.weight(i) },
                &format!("{name} score {weighted} column {j}"),
            ));
        }
        worst.add(&format!("{name} score weighted={weighted}"), value);
    }

    // dfbeta, dfbeta unweighted, dfbetas.
    let tol_dfbeta = |i: usize, j: usize, weighted: bool| -> f64 {
        let scale: f64 = (0..p)
            .filter(|&k| !s.aliased(k))
            .map(|k| {
                let sk = xs(k, i) / if weighted { 1.0 } else { s.weight(i) };
                let sij = if weighted {
                    r.score_weighted[i][k]
                } else {
                    r.score[i][k]
                };
                sk * s.v(k, j).abs() + p as f64 * s.bound * (sij.unwrap_or(0.0) * s.v(k, j)).abs()
            })
            .sum();
        scale
    };
    for (kind, weighted, expected) in [
        (CoxResidualKind::Dfbeta, None, &r.dfbeta),
        (CoxResidualKind::Dfbeta, Some(false), &r.dfbeta_unweighted),
        (CoxResidualKind::Dfbetas, None, &r.dfbetas),
    ] {
        let got = s.residuals(kind, weighted, Collapse::None);
        let w = weighted.unwrap_or(true);
        let mut value = 0.0f64;
        for (j, column) in got.columns.iter().enumerate().take(p) {
            let se = if kind == CoxResidualKind::Dfbetas {
                s.v(j, j).sqrt()
            } else {
                1.0
            };
            value = value.max(worst_column(
                column.as_ref(),
                &|i| entry(expected, i, j),
                n,
                &|i| tol_dfbeta(i, j, w) / se,
                &format!("{name} {kind:?} {weighted:?} column {j}"),
            ));
        }
        worst.add(&format!("{name} {kind:?} weighted={weighted:?}"), value);
    }

    // Schoenfeld, weighted or not, and scaled.
    let events = r.schoenfeld_time.len();
    for (weighted, expected) in [(false, &r.schoenfeld), (true, &r.schoenfeld_weighted)] {
        let got = s.residuals(CoxResidualKind::Schoenfeld, Some(weighted), Collapse::None);
        assert_eq!(got.time, r.schoenfeld_time, "{name}: event times");
        assert_eq!(got.rows, r.schoenfeld_row, "{name}: rows of the events");
        assert_eq!(
            got.stratum, r.schoenfeld_stratum,
            "{name}: strata of the events"
        );
        let mut value = 0.0f64;
        for (j, column) in got.columns.iter().enumerate().take(p) {
            value = value.max(worst_column(
                column.as_ref(),
                &|e| entry(expected, e, j),
                events,
                &|e| {
                    s.rel
                        * s.xmax[j]
                        * if weighted {
                            s.weight(r.schoenfeld_row[e] as usize)
                        } else {
                            1.0
                        }
                },
                &format!("{name} schoenfeld weighted {weighted} column {j}"),
            ));
        }
        worst.add(&format!("{name} schoenfeld weighted={weighted}"), value);
    }
    if let Some(expected) = &r.scaledsch {
        let got = s.residuals(CoxResidualKind::ScaledSchoenfeld, None, Collapse::None);
        let d = events as f64;
        let mut value = 0.0f64;
        for (j, column) in got.columns.iter().enumerate().take(p) {
            value = value.max(worst_column(
                column.as_ref(),
                &|e| entry(expected, e, j),
                events,
                &|e| {
                    let mut tol = s.bound * s.r.coefficients[j].abs().max(1.0);
                    for k in 0..p {
                        tol += d * s.v(k, j).abs() * s.rel * s.xmax[k]
                            + s.p as f64
                                * s.bound
                                * d
                                * (s.v(k, j) * r.schoenfeld[e][k].unwrap_or(0.0)).abs();
                    }
                    tol
                },
                &format!("{name} scaledsch column {j}"),
            ));
        }
        worst.add(&format!("{name} scaledsch"), value);
    }

    // Collapse by codes: every kind that has one row per row of the data.
    let codes = &s.ds.grp;
    let groups = {
        let mut g = codes.clone();
        g.sort_unstable();
        g.dedup();
        g
    };
    let tol_sum = |f: &dyn Fn(usize) -> f64| -> Vec<f64> {
        group_sum(codes, &(0..n).map(f).collect::<Vec<_>>())
    };
    let collapse_check = |kind: CoxResidualKind,
                          weighted: Option<bool>,
                          collapse: Collapse<'_>,
                          expected: &Matrix,
                          tol_rows: &dyn Fn(usize, usize) -> f64,
                          expect_groups: &[u32],
                          label: &str|
     -> f64 {
        let got = s.residuals(kind, weighted, collapse);
        assert!(got.rows.is_empty(), "{label}");
        assert_eq!(got.groups, expect_groups, "{label}: groups");
        let columns = if matches!(
            kind,
            CoxResidualKind::Martingale | CoxResidualKind::Deviance
        ) {
            1
        } else {
            p
        };
        let mut value = 0.0f64;
        for (j, column) in got.columns.iter().enumerate().take(columns) {
            let tols = if matches!(collapse, Collapse::Codes(_)) {
                tol_sum(&|i| tol_rows(i, j))
            } else {
                group_sum(
                    &s.ds.id,
                    &(0..n).map(|i| tol_rows(i, j)).collect::<Vec<_>>(),
                )
            };
            value = value.max(worst_column(
                column.as_ref(),
                &|g| entry(expected, g, j),
                expect_groups.len(),
                &|g| tols[g],
                label,
            ));
        }
        value
    };
    let mart_tol = |i: usize, _j: usize| s.bx * (1.0 + mart[i].abs()) + s.ne * s.cs[i];
    let mart_tol_weighted = |i: usize, _j: usize| mart_tol(i, 0) * s.weight(i);
    let score_tol = |i: usize, j: usize| xs(j, i) / s.weight(i);
    let score_tol_weighted = |i: usize, j: usize| xs(j, i);
    let plain_tol = |i: usize, j: usize| tol_dfbeta(i, j, true);
    let scaled_tol = |i: usize, j: usize| tol_dfbeta(i, j, true) / s.v(j, j).sqrt();
    let mut cases: Vec<(&str, f64)> = Vec::new();
    let c = |kind, weighted, expected: &Matrix, tol: &dyn Fn(usize, usize) -> f64, label: &str| {
        collapse_check(
            kind,
            weighted,
            Collapse::Codes(codes),
            expected,
            tol,
            &groups,
            label,
        )
    };
    cases.push((
        "martingale",
        c(
            CoxResidualKind::Martingale,
            None,
            &r.martingale_collapse_codes,
            &mart_tol,
            "martingale",
        ),
    ));
    cases.push((
        "martingale weighted",
        c(
            CoxResidualKind::Martingale,
            Some(true),
            &r.martingale_weighted_collapse_codes,
            &mart_tol_weighted,
            "martingale weighted",
        ),
    ));
    cases.push((
        "score",
        c(
            CoxResidualKind::Score,
            None,
            &r.score_collapse_codes,
            &score_tol,
            "score",
        ),
    ));
    cases.push((
        "score weighted",
        c(
            CoxResidualKind::Score,
            Some(true),
            &r.score_weighted_collapse_codes,
            &score_tol_weighted,
            "score weighted",
        ),
    ));
    cases.push((
        "dfbeta",
        c(
            CoxResidualKind::Dfbeta,
            None,
            &r.dfbeta_collapse_codes,
            &plain_tol,
            "dfbeta",
        ),
    ));
    cases.push((
        "dfbetas",
        c(
            CoxResidualKind::Dfbetas,
            None,
            &r.dfbetas_collapse_codes,
            &scaled_tol,
            "dfbetas",
        ),
    ));
    for (what, value) in cases {
        worst.add(&format!("{name} collapse codes {what}"), value);
    }
    // Deviance of groups: the formula of the deviance at R's collapsed
    // martingale residual and status. R's own value is NaN where the status
    // of the group is 2 or more, and equal to ours where it is 0 or 1.
    let m_groups: Vec<f64> = r
        .martingale_collapse_codes
        .iter()
        .map(|row| row[0].unwrap())
        .collect();
    let got = s.residuals(CoxResidualKind::Deviance, None, Collapse::Codes(codes));
    let tols = tol_sum(&|i| s.bx * (1.0 + mart[i].abs()) + s.ne * s.cs[i]);
    let ours = got.columns[0].as_ref().unwrap();
    let mut value = 0.0f64;
    for g in 0..groups.len() {
        let (expected, slope) = deviance(m_groups[g], r.status_collapse_codes[g]);
        let tol = tols[g] * slope;
        let ratio = (ours[g] - expected).abs() / tol;
        assert!(
            ratio <= 1.0,
            "{name} deviance group {g}: {} vs {expected}",
            ours[g]
        );
        value = value.max(ratio);
        if let Some(rv) = r.deviance_collapse_codes[g][0] {
            assert!(
                r.status_collapse_codes[g] <= 1.0,
                "R gives a value for status 2 or more"
            );
            assert!(
                (rv - expected).abs() <= tol,
                "{name}: our formula against R's at status <= 1"
            );
        } else {
            assert!(
                r.status_collapse_codes[g] >= 2.0,
                "{name}: R has NaN only for status of 2 or more"
            );
        }
    }
    worst.add(&format!("{name} collapse codes deviance"), value);

    // Collapse by the cluster of the data, and by its id, which is the cluster.
    if s.r.use_cluster {
        let mut ids: Vec<u32> = s.ds.id.clone();
        ids.sort_unstable();
        ids.dedup();
        let cluster_cases: [ClusterCase; 4] = [
            (
                "martingale",
                CoxResidualKind::Martingale,
                &r.martingale_collapse_cluster,
                &mart_tol,
            ),
            (
                "score",
                CoxResidualKind::Score,
                &r.score_collapse_cluster,
                &score_tol,
            ),
            (
                "dfbeta",
                CoxResidualKind::Dfbeta,
                &r.dfbeta_collapse_cluster,
                &plain_tol,
            ),
            (
                "dfbetas",
                CoxResidualKind::Dfbetas,
                &r.dfbetas_collapse_cluster,
                &scaled_tol,
            ),
        ];
        for (what, kind, expected, tol) in cluster_cases {
            let expected = expected.as_ref().unwrap();
            let by_cluster =
                collapse_check(kind, None, Collapse::Cluster, expected, tol, &ids, what);
            // The data have clusters but no ids; id is refused.
            worst.add(&format!("{name} collapse cluster {what}"), by_cluster);
        }
        let got = s.residuals(CoxResidualKind::Deviance, None, Collapse::Cluster);
        let m_clusters = r.martingale_collapse_cluster.as_ref().unwrap();
        let status = r.status_collapse_cluster.as_ref().unwrap();
        let tols = group_sum(
            &s.ds.id,
            &(0..n)
                .map(|i| s.bx * (1.0 + mart[i].abs()) + s.ne * s.cs[i])
                .collect::<Vec<_>>(),
        );
        let mut value = 0.0f64;
        for (g, m) in m_clusters.iter().enumerate() {
            let (expected, slope) = deviance(m[0].unwrap(), status[g]);
            value = value
                .max((got.columns[0].as_ref().unwrap()[g] - expected).abs() / (tols[g] * slope));
        }
        worst.add(&format!("{name} collapse cluster deviance"), value);
    }
}

fn zph_transform(name: &str) -> ZphTransform {
    match name {
        "km" => ZphTransform::Km,
        "rank" => ZphTransform::Rank,
        "identity" => ZphTransform::Identity,
        "log" => ZphTransform::Log,
        other => panic!("{other}"),
    }
}

fn check_zph(s: &Setup<'_>, worst: &mut Worst) {
    let name = &s.r.name;
    let p = s.p;
    let kept: Vec<usize> = (0..p).filter(|&j| !s.aliased(j)).collect();
    for z in &s.r.zph {
        let groups = s.r.terms.clone();
        let opts = ZphOptions {
            cox: s.opts(),
            transform: zph_transform(&z.transform),
            groups: groups.clone(),
            singledf: z.singledf,
        };
        let got = zph(&s.data(), s.x.as_ref(), &s.fit, &opts).unwrap();
        let label = format!("{name} zph {} singledf {}", z.transform, z.singledf);
        let nterm = got.terms.len();
        assert_eq!(z.table.len(), nterm + 1, "{label}: rows of the table");
        // Terms: columns of the design, aliased ones removed.
        let expected_terms: Vec<Vec<usize>> = match &groups {
            Some(g) => g.clone(),
            None => kept.iter().map(|&j| vec![j]).collect(),
        };
        assert_eq!(got.terms, expected_terms, "{label}: terms");
        let kz = 4.0 * z.kappa * s.rel;
        // Table: statistic, df, p, for each term and then the model.
        let mut value = 0.0f64;
        for i in 0..=nterm {
            let test = if i < nterm { got.tests[i] } else { got.global };
            let test = test.unwrap_or_else(|| panic!("{label}: test {i} is None"));
            let (stat, df, p_value) = (z.table[i][0], z.table[i][1], z.table[i][2]);
            assert!((test.df as f64 - df).abs() < 0.5, "{label}: df {i}");
            let tol = kz * stat.max(1.0);
            let ratio = (test.statistic - stat).abs() / tol;
            assert!(
                ratio <= 1.0,
                "{label}: statistic {i}: {} vs {stat}, tolerance {tol}",
                test.statistic
            );
            value = value.max(ratio);
            let band = chi_square_sf((stat - tol).max(0.0), df) - chi_square_sf(stat + tol, df)
                + 64.0 * EPS;
            let ratio = (test.p_value - p_value).abs() / band;
            assert!(
                ratio <= 1.0,
                "{label}: p {i}: {} vs {p_value}",
                test.p_value
            );
            value = value.max(ratio);
        }
        worst.add(&format!("{label} table"), value);

        // Times, transformed times, strata and the rows of the events. R lists
        // tied events of data with delayed entry by descending row; they are
        // matched by row.
        let events = z.time.len();
        assert_eq!(got.time.len(), events, "{label}: events");
        let mut by_row: HashMap<u32, usize> = HashMap::new();
        for (e, &row) in z.row.iter().enumerate() {
            by_row.insert(row, e);
        }
        let mut codes: Vec<u32> = s.ds.stratum.clone();
        codes.sort_unstable();
        codes.dedup();
        let mut value = 0.0f64;
        for (e, &row) in got.rows.iter().enumerate() {
            let r = *by_row
                .get(&row)
                .unwrap_or_else(|| panic!("{label}: row {row} is not an event of R"));
            assert_eq!(got.time[e].to_bits(), z.time[r].to_bits(), "{label}: time");
            if s.r.use_strata {
                assert_eq!(
                    got.stratum[e],
                    codes[z.stratum[r] as usize - 1],
                    "{label}: stratum"
                );
            }
            // The transformed time: the Kaplan-Meier value is a product of up
            // to n factors, the rank and the identity are exact.
            let tol = (s.n as f64 + 4.0) * EPS * z.x[r].abs().max(1.0);
            let ratio = (got.x[e] - z.x[r]).abs() / tol;
            assert!(ratio <= 1.0, "{label}: x {} vs {}", got.x[e], z.x[r]);
            value = value.max(ratio);
        }
        worst.add(&format!("{label} x"), value);
        // The scaled residuals, term by term.
        let var_scale = |i: usize, k: usize| (z.var[i][i] * z.var[k][k]).sqrt();
        let mut value = 0.0f64;
        for i in 0..nterm {
            let beta = if got.terms[i].len() == 1 {
                s.r.coefficients[got.terms[i][0]]
            } else {
                1.0
            };
            for (e, &row) in got.rows.iter().enumerate() {
                let r = by_row[&row];
                let expected = z.y[r][i];
                match (got.y[i][e], expected) {
                    (None, None) => {}
                    (Some(g), Some(x)) => {
                        let xterm: f64 = got.terms[i]
                            .iter()
                            .map(|&j| {
                                s.xmax[j]
                                    * if got.terms[i].len() == 1 {
                                        1.0
                                    } else {
                                        s.r.coefficients[j].abs()
                                    }
                            })
                            .sum();
                        let spread: f64 = (0..nterm).map(|k| xterm * z.var[i][k].abs()).sum();
                        let tol = kz * (spread + (x - beta).abs() + beta.abs().max(1.0));
                        let ratio = (g - x).abs() / tol;
                        assert!(ratio <= 1.0, "{label}: y term {i} event {e}: {g} vs {x}");
                        value = value.max(ratio);
                    }
                    (g, x) => panic!("{label}: y term {i} event {e}: {g:?} vs {x:?}"),
                }
            }
        }
        worst.add(&format!("{label} y"), value);
        let mut value = 0.0f64;
        for (i, row) in z.var.iter().enumerate().take(nterm) {
            for (k, &want) in row.iter().enumerate().take(nterm) {
                let g = got.variance.get(i, k).unwrap();
                value = value.max((g - want).abs() / (kz * var_scale(i, k)));
            }
        }
        worst.add(&format!("{label} variance"), value);
    }
}

#[test]
fn residuals_and_zph_match_r() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/cox_residuals.json"
    ));
    eprintln!(
        "reference: {}, survival {}, jsonlite {}",
        reference.versions.r, reference.versions.survival, reference.versions.jsonlite
    );
    let mut worst = Worst(Vec::new());
    for r in &reference.fits {
        let s = setup(&reference, r);
        check_residuals(&s, &mut worst);
        check_zph(&s, &mut worst);
    }
    for (what, value) in &worst.0 {
        eprintln!("{what:60} {value:.2e} of the tolerance");
    }
}
