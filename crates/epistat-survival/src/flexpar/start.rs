//! Knots, aliased columns and starting values.

use epistat_linalg::MatRef;
use epistat_stats::normal;
use epistat_stats::quantile::type7;

use super::Scale;
use super::design::Design;
use crate::error::{Error, Result};
use crate::ldl;
use crate::par;

/// `df + 1` knots on the log scale: the smallest and largest of `events` and the
/// quantiles (type 7) at `j / df` between them. `events` is sorted.
pub(super) fn quantile_knots(events: &[f64], df: usize) -> Result<Vec<f64>> {
    let mut knots = Vec::with_capacity(df + 1);
    for j in 0..=df {
        #[expect(clippy::cast_precision_loss, reason = "a few knots")]
        let p = j as f64 / df as f64;
        let k = type7(events, p).ok_or_else(|| Error::input("no event times to place knots"))?;
        knots.push(k);
    }
    Ok(knots)
}

/// The columns of `x` that cannot be estimated beside the intercept: constant
/// columns, and columns that are linear combinations of earlier ones, found by
/// the `L D L'` factorisation of the weighted correlation matrix of the
/// columns that vary.
pub(super) fn aliased_columns(x: MatRef<'_, f64>, weight: Option<&[f64]>) -> Vec<bool> {
    let (n, p) = (x.nrows(), x.ncols());
    let w = |i: usize| weight.map_or(1.0, |w| w[i]);
    let block = 4096;
    let sums = par::ordered_reduce(
        n,
        block,
        64,
        |r| {
            let mut s = vec![0.0; p + 1];
            for i in r {
                let c = w(i);
                s[p] += c;
                for j in 0..p {
                    s[j] += c * x[(i, j)];
                }
            }
            s
        },
        |t, q| t.iter_mut().zip(&q).for_each(|(a, b)| *a += b),
    );
    let total = sums[p];
    let mean: Vec<f64> = sums[..p].iter().map(|s| s / total).collect();
    let moments = par::ordered_reduce(
        n,
        block,
        64,
        |r| {
            let mut m = vec![0.0; p * p];
            let mut d = vec![0.0; p];
            for i in r {
                let c = w(i);
                for j in 0..p {
                    d[j] = x[(i, j)] - mean[j];
                }
                for j in 0..p {
                    for k in 0..=j {
                        m[j * p + k] += c * d[j] * d[k];
                    }
                }
            }
            m
        },
        |t, q| t.iter_mut().zip(&q).for_each(|(a, b)| *a += b),
    );
    let sd: Vec<f64> = (0..p)
        .map(|j| (moments[j * p + j] / total).sqrt())
        .collect();
    let mut aliased: Vec<bool> = (0..p)
        .map(|j| !(sd[j] > 1e-12 * mean[j].abs() && sd[j] > 0.0))
        .collect();
    let varying: Vec<usize> = (0..p).filter(|&j| !aliased[j]).collect();
    let m = varying.len();
    let mut corr = vec![0.0; m * m];
    for (a, &j) in varying.iter().enumerate() {
        for (b, &k) in varying.iter().enumerate() {
            corr[a * m + b] = moments[j.max(k) * p + j.min(k)] / (total * sd[j] * sd[k]);
        }
    }
    let mut factor = corr;
    ldl::factorise(&mut factor, m, 1e-10);
    for (a, &j) in varying.iter().enumerate() {
        if factor[a * m + a] == 0.0 {
            aliased[j] = true;
        }
    }
    aliased
}

/// The Nelson-Aalen cumulative hazard at each distinct event time, from the
/// weights and the entry times, as pairs of the time and the cumulative hazard
/// with the weighted number of events at it.
fn nelson_aalen(
    stop: &[f64],
    start: Option<&[f64]>,
    event: &[i32],
    weight: Option<&[f64]>,
) -> Vec<(f64, f64, f64)> {
    let n = stop.len();
    let c = |i: usize| weight.map_or(1.0, |w| w[i]);
    let mut exits: Vec<(f64, f64, f64)> = (0..n)
        .filter(|&i| c(i) > 0.0)
        .map(|i| (stop[i], c(i), if event[i] != 0 { c(i) } else { 0.0 }))
        .collect();
    exits.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut entries: Vec<(f64, f64)> = match start {
        Some(s) => (0..n)
            .filter(|&i| c(i) > 0.0 && s[i] > 0.0)
            .map(|i| (s[i], c(i)))
            .collect(),
        None => Vec::new(),
    };
    entries.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total_entries: f64 = entries.iter().map(|x| x.1).sum();
    let mut at_risk: f64 = exits.iter().map(|e| e.1).sum();
    let (mut out, mut hazard) = (Vec::new(), 0.0);
    let (mut i, mut e) = (0, 0);
    let mut entered_weight = 0.0;
    while i < exits.len() {
        let t = exits[i].0;
        // Rows that enter at or after `t` are not at risk at `t`.
        while e < entries.len() && entries[e].0 < t {
            entered_weight += entries[e].1;
            e += 1;
        }
        let (mut deaths, mut leaving) = (0.0, 0.0);
        while i < exits.len() && exits[i].0 == t {
            deaths += exits[i].2;
            leaving += exits[i].1;
            i += 1;
        }
        // Weight at risk at t: those who exit at t or later, less those that
        // enter at t or later.
        let late = total_entries - entered_weight;
        let risk = at_risk - late;
        if deaths > 0.0 && risk > 0.0 {
            hazard += deaths / risk;
            out.push((t, hazard, deaths));
        }
        at_risk -= leaving;
    }
    out
}

/// Starting values: the baseline spline fitted by weighted least squares to the
/// transformed Nelson-Aalen estimate at the event times, falling back to a
/// straight line in `ln t` where that has a derivative that is not positive
/// at every event. The covariate effects start at zero.
pub(super) fn starting_values(
    design: &Design,
    scale: Scale,
    stop: &[f64],
    start: Option<&[f64]>,
    event: &[i32],
    weight: Option<&[f64]>,
) -> Vec<f64> {
    let q = design.dim();
    let df = design.baseline.len();
    let points = nelson_aalen(stop, start, event, weight);
    let g = |h: f64| match scale {
        Scale::Hazard => h.ln(),
        Scale::Odds => h.exp_m1().ln(),
        Scale::Probit => -normal::quantile((-h).exp()),
    };
    let usable: Vec<(f64, f64, f64)> = points
        .iter()
        .map(|&(t, h, d)| (t.ln(), g(h), d))
        .filter(|p| p.1.is_finite())
        .collect();
    let mut theta = vec![0.0; q];
    let fit = |columns: usize| -> Option<Vec<f64>> {
        let m = columns + 1;
        let mut a = vec![0.0; m * m];
        let mut b = vec![0.0; m];
        let mut row = vec![0.0; 1 + df];
        for &(u, y, d) in &usable {
            row[0] = 1.0;
            design.baseline.values_into(u, &mut row[1..]);
            for j in 0..m {
                b[j] += d * row[j] * y;
                for k in 0..=j {
                    a[j * m + k] += d * row[j] * row[k];
                }
            }
        }
        let pivots = ldl::factorise(&mut a, m, 1e-12);
        if !pivots.clean || pivots.rank < m {
            return None;
        }
        ldl::solve(&a, m, &mut b);
        Some(b)
    };
    let monotone = |gamma: &[f64]| {
        let mut w = vec![0.0; df];
        (0..stop.len())
            .filter(|&i| event[i] != 0 && weight.is_none_or(|c| c[i] > 0.0))
            .all(|i| {
                design.baseline.derivatives_into(stop[i].ln(), &mut w);
                w.iter().zip(&gamma[1..]).map(|(a, b)| a * b).sum::<f64>() > 0.0
            })
    };
    if usable.len() > df {
        if let Some(gamma) = fit(df) {
            if gamma.iter().all(|v| v.is_finite()) && monotone(&gamma) {
                theta[..=df].copy_from_slice(&gamma);
                return theta;
            }
        }
    }
    // A straight line in ln t.
    let (mut sw, mut su, mut sy, mut suu, mut suy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &(u, y, d) in &usable {
        sw += d;
        su += d * u;
        sy += d * y;
        suu += d * u * u;
        suy += d * u * y;
    }
    let denominator = sw * suu - su * su;
    let slope = if usable.len() >= 2 && denominator > 0.0 {
        (sw * suy - su * sy) / denominator
    } else {
        1.0
    };
    let slope = if slope.is_finite() && slope > 0.0 {
        slope
    } else {
        1.0
    };
    let intercept = if sw > 0.0 {
        (sy - slope * su) / sw
    } else {
        0.0
    };
    theta[0] = if intercept.is_finite() {
        intercept
    } else {
        0.0
    };
    theta[1] = slope;
    theta
}
