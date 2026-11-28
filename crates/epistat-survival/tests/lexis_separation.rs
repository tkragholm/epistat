//! The rate model of `lexis::rates` with groups that have no events: the
//! coefficient of such a group has no finite estimate and is `None`, the
//! other coefficients are those `glm` approaches, and data in which every
//! piece has events fit as before.

use epistat_core::Level;
use epistat_glm::{Family, Fit, Glm, GlmData, Link};
use epistat_linalg::Mat;
use epistat_survival::lexis::rates::{RateOptions, Rates, Unestimable, fit_rates};

/// Pieces of `levels` groups, four of each, with the person-time and events
/// of piece `i` in group `i % levels`. Groups listed in `empty` have no
/// events.
fn pieces(levels: usize, empty: &[usize]) -> (Vec<usize>, Vec<f64>, Vec<f64>) {
    let n = 4 * levels;
    let group: Vec<usize> = (0..n).map(|i| i % levels).collect();
    let length: Vec<f64> = (0..n).map(|i| 1.5 + (i * 7 % 5) as f64).collect();
    let events: Vec<f64> = (0..n)
        .map(|i| {
            if empty.contains(&group[i]) {
                0.0
            } else {
                (1 + (i * 3 + group[i]) % 4) as f64
            }
        })
        .collect();
    (group, length, events)
}

/// An intercept and an indicator of each group but the first.
fn treatment_columns(group: &[usize], levels: usize) -> Vec<Vec<f64>> {
    let mut columns = vec![vec![1.0; group.len()]];
    for level in 1..levels {
        columns.push(group.iter().map(|&g| f64::from(g == level)).collect());
    }
    columns
}

/// `glm(events ~ x, family = poisson, offset = log(length))` on the pieces
/// `rows`, with the columns `keep` of the design.
fn glm(
    columns: &[Vec<f64>],
    keep: &[usize],
    rows: &[usize],
    length: &[f64],
    events: &[f64],
) -> Fit {
    let x = Mat::from_fn(rows.len(), keep.len(), |r, j| columns[keep[j]][rows[r]]);
    let y: Vec<f64> = rows.iter().map(|&i| events[i]).collect();
    let offset: Vec<f64> = rows.iter().map(|&i| length[i].ln()).collect();
    let data = GlmData::builder(x.as_ref(), &y)
        .offset(&offset)
        .build()
        .unwrap();
    Glm::new(Family::Poisson, Link::Log).fit(&data).unwrap()
}

fn fit(columns: &[Vec<f64>], length: &[f64], events: &[f64]) -> Rates {
    let refs: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();
    fit_rates(length, events, &refs, &RateOptions::default()).unwrap()
}

/// Checks a fit with the groups `empty` separated against `glm`.
///
/// The fit of `fit_rates` is the fit of the pieces of the other groups, with
/// the separated columns all 0 there; it agrees to the bit with `glm` on those
/// pieces and columns alone.
///
/// `glm` on all pieces stops with the separated coefficients at a large
/// negative value, where the separated pieces keep fitted events summing to
/// a small `m_k` in each column `k`. To first order those events shift the
/// other coefficients by `V s`, with `V` the covariance of the restricted fit
/// and `s_k = sum x_ik mu_i` over the separated pieces; the second-order term
/// is of order `m^2`, below `1e-16` here, and the factor 2 covers it. Each fit
/// also stops at a relative change of the deviance below `1e-8`; Fisher scoring
/// is quadratic near the maximum here (the log link is canonical), so the last
/// step was below `sqrt(1e-8 (|D| + 0.1))` in the metric of the information
/// and the error after it is below that times the standard error.
fn check(levels: usize, empty: &[usize]) {
    let (group, length, events) = pieces(levels, empty);
    let columns = treatment_columns(&group, levels);
    let rates = fit(&columns, &length, &events);
    assert_eq!(rates.separated, empty);
    let rows: Vec<usize> = (0..group.len())
        .filter(|&i| !empty.contains(&group[i]))
        .collect();
    assert_eq!(rates.n_used, rows.len());
    assert_eq!(rates.n_separated, group.len() - rows.len());
    let kept: Vec<usize> = (0..levels).filter(|k| !empty.contains(k)).collect();
    let restricted = glm(&columns, &kept, &rows, &length, &events);
    let full: Vec<usize> = (0..group.len()).collect();
    let all: Vec<usize> = (0..levels).collect();
    let unrestricted = glm(&columns, &all, &full, &length, &events);
    assert!(unrestricted.converged);

    let v = &restricted.covariance;
    let mut s = vec![0.0; kept.len()];
    for i in (0..group.len()).filter(|i| !rows.contains(i)) {
        for (a, &k) in kept.iter().enumerate() {
            s[a] += columns[k][i] * unrestricted.fitted[i];
        }
    }
    for (a, &k) in kept.iter().enumerate() {
        let got = rates.fit.coefficients[k].unwrap();
        let alone = restricted.coefficients[a].unwrap();
        assert_eq!(got.to_bits(), alone.to_bits(), "column {k}");
        let shift: f64 = (0..kept.len()).map(|b| v.get(a, b).unwrap() * s[b]).sum();
        let se = v.get(a, a).unwrap().sqrt();
        let deviance = restricted.deviance.abs().max(unrestricted.deviance.abs());
        let bound = 2.0 * shift.abs() + (1e-8 * (deviance + 0.1)).sqrt() * se;
        let r = unrestricted.coefficients[k].unwrap();
        assert!((got - r).abs() <= bound, "column {k}: {got} {r} {bound}");
    }
    for &k in empty {
        assert!(rates.fit.coefficients[k].is_none());
        assert!(rates.fit.covariance.get(k, k).is_none());
        assert_eq!(
            rates.unestimable(k),
            Some(Unestimable::Separated { column: k })
        );
        // R stops far out, with a rate ratio near 0 and limits 0 and a large
        // number.
        assert!(unrestricted.coefficients[k].unwrap() < -15.0);
    }

    // The rate ratio of each group against the first: those without events
    // have none, the others are those of the restricted fit.
    let contrasts: Vec<Vec<f64>> = (1..levels)
        .map(|k| (0..levels).map(|j| f64::from(j == k)).collect())
        .collect();
    let refs: Vec<&[f64]> = contrasts.iter().map(Vec::as_slice).collect();
    let ratios = rates.ci_exp(&refs, Level::default()).unwrap();
    for (r, k) in (1..levels).enumerate() {
        if empty.contains(&k) {
            assert!(ratios[r].estimate.is_none() && ratios[r].lower.is_none());
            assert_eq!(
                ratios[r].unestimable,
                Some(Unestimable::Separated { column: k })
            );
        } else {
            let a = kept.iter().position(|&j| j == k).unwrap();
            assert_eq!(
                ratios[r].estimate.unwrap().to_bits(),
                restricted.coefficients[a].unwrap().exp().to_bits()
            );
            assert!(ratios[r].unestimable.is_none());
        }
    }
}

#[test]
fn one_group_without_events_is_separated() {
    check(3, &[2]);
}

#[test]
fn two_groups_without_events_are_separated() {
    check(5, &[1, 3]);
}

/// With events in every group nothing is separated and the fit is `glm`'s on
/// every piece, to the bit.
#[test]
fn events_in_every_group_fit_as_glm() {
    let levels = 4;
    let (group, length, events) = pieces(levels, &[]);
    let columns = treatment_columns(&group, levels);
    let rates = fit(&columns, &length, &events);
    assert_eq!(rates.separated, Vec::<usize>::new());
    assert_eq!(rates.n_separated, 0);
    let all: Vec<usize> = (0..levels).collect();
    let rows: Vec<usize> = (0..group.len()).collect();
    let direct = glm(&columns, &all, &rows, &length, &events);
    for k in 0..levels {
        assert_eq!(
            rates.fit.coefficients[k].unwrap().to_bits(),
            direct.coefficients[k].unwrap().to_bits()
        );
        assert!(rates.unestimable(k).is_none());
    }
    assert_eq!(rates.fit.deviance.to_bits(), direct.deviance.to_bits());
}

/// A column that changes sign on the pieces without events is not separated
/// until the pieces of another separated column are left out: the search runs
/// again on the pieces that remain.
#[test]
fn a_column_one_sided_on_the_remaining_pieces_is_separated_next() {
    let (group, length, events) = pieces(3, &[1, 2]);
    // Column 1 is 0 on the first group, -1 on the second and 1 on the third,
    // two-sided until column 2, the indicator of the second group, leaves the
    // second group's pieces out. The third group has no column of its own.
    let columns = vec![
        vec![1.0; group.len()],
        group
            .iter()
            .map(|&g| match g {
                1 => -1.0,
                2 => 1.0,
                _ => 0.0,
            })
            .collect(),
        group.iter().map(|&g| f64::from(g == 1)).collect(),
    ];
    let rates = fit(&columns, &length, &events);
    assert_eq!(rates.separated, [1, 2]);
    assert_eq!(rates.n_used, 4);
    assert_eq!(rates.n_separated, 8);
    let rows: Vec<usize> = (0..group.len()).filter(|&i| group[i] == 0).collect();
    let alone = glm(&columns, &[0], &rows, &length, &events);
    assert_eq!(
        rates.fit.coefficients[0].unwrap().to_bits(),
        alone.coefficients[0].unwrap().to_bits()
    );
}

/// Data without events leave nothing to estimate.
#[test]
fn no_events_at_all_is_an_error() {
    let (group, length, events) = pieces(2, &[0, 1]);
    let columns = treatment_columns(&group, 2);
    let refs: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();
    let error = fit_rates(&length, &events, &refs, &RateOptions::default()).unwrap_err();
    assert_eq!(
        epistat_core::Classify::code(&error),
        "survival.no_information"
    );
}
