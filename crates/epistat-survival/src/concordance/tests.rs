//! The sweep checked against enumeration of every pair.

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    reason = "small integer-valued inputs, and counts that must be exact"
)]

use super::{ConcordanceOptions, TimeWeight, estimate};
use crate::SurvData;
use epistat_rng::Stream;

/// What enumerating every pair gives for time weight `N`, as the five counts
/// and the partner sums of each row.
struct Brute {
    counts: [f64; 5],
    partners: Vec<[f64; 3]>,
}

fn brute(
    start: Option<&[f64]>,
    stop: &[f64],
    death: &[bool],
    w: &[f64],
    x: &[f64],
    stratum: &[u32],
) -> Brute {
    let n = stop.len();
    let mut counts = [0.0; 5];
    let mut partners = vec![[0.0; 3]; n];
    for i in 0..n {
        if !death[i] {
            continue;
        }
        for j in 0..n {
            if i == j || stratum[i] != stratum[j] {
                continue;
            }
            let at_risk = stop[j] >= stop[i] && start.is_none_or(|s| s[j] < stop[i]);
            if !at_risk {
                continue;
            }
            let pair = w[i] * w[j];
            if stop[j] == stop[i] && death[j] {
                if j > i {
                    if x[i] == x[j] {
                        counts[4] += pair;
                    } else {
                        counts[3] += pair;
                    }
                }
                continue;
            }
            let kind = if x[j] > x[i] {
                0
            } else if x[j] < x[i] {
                1
            } else {
                2
            };
            counts[kind] += pair;
            // The death gains j's weight in the class, j the death's.
            partners[i][kind] += w[j];
            partners[j][kind] += w[i];
        }
    }
    Brute { counts, partners }
}

struct Sim {
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    weight: Vec<f64>,
    stratum: Vec<u32>,
    x: Vec<f64>,
}

fn simulate(seed: u64, n: usize, delayed: bool) -> Sim {
    let mut rng = Stream::seeded(seed);
    let mut sim = Sim {
        start: vec![],
        stop: vec![],
        status: vec![],
        weight: vec![],
        stratum: vec![],
        x: vec![],
    };
    for _ in 0..n {
        let stop = rng.below(12) as f64 + 1.0;
        sim.stop.push(stop);
        sim.start.push(if delayed {
            (rng.below(stop as usize) as f64).min(stop - 1.0)
        } else {
            0.0
        });
        sim.status.push(i32::from(rng.below(3) != 0));
        sim.weight.push((1 + rng.below(3)) as f64);
        sim.stratum.push(rng.below(3) as u32);
        sim.x.push(rng.below(7) as f64 / 2.0);
    }
    sim
}

fn check(sim: &Sim, delayed: bool, strata: bool) {
    let mut builder = SurvData::builder(&sim.stop, &sim.status).weight(&sim.weight);
    if delayed {
        builder = builder.start(&sim.start);
    }
    if strata {
        builder = builder.stratum(&sim.stratum);
    }
    let data = builder.build().unwrap();
    let opts = ConcordanceOptions {
        influence: true,
        ..ConcordanceOptions::default()
    };
    let got = estimate(&data, &sim.x, &opts).unwrap();
    let death: Vec<bool> = sim.status.iter().map(|&s| s == 1).collect();
    let zero = vec![0_u32; sim.stop.len()];
    let want = brute(
        delayed.then_some(&sim.start[..]),
        &sim.stop,
        &death,
        &sim.weight,
        &sim.x,
        if strata { &sim.stratum } else { &zero },
    );
    assert_eq!(got.counts.concordant, want.counts[0]);
    assert_eq!(got.counts.discordant, want.counts[1]);
    assert_eq!(got.counts.tied_x, want.counts[2]);
    assert_eq!(got.counts.tied_y, want.counts[3]);
    assert_eq!(got.counts.tied_xy, want.counts[4]);
    let pairs = want.counts[0] + want.counts[1] + want.counts[2];
    let somer = (want.counts[0] - want.counts[1]) / pairs;
    let influence = got.influence.as_ref().unwrap();
    // The variance is that of the influence function the result carries.
    assert_eq!(influence.variance(None).unwrap().get(0, 0), got.variance);
    let n = influence.n_units() as f64;
    let dfbeta: Vec<f64> = influence.column(0).iter().map(|v| v / n).collect();
    for (k, p) in want.partners.iter().enumerate() {
        let expected = sim.weight[k] * ((1.0 - somer) * p[0] - (1.0 + somer) * p[1] - somer * p[2])
            / (2.0 * pairs);
        assert!(
            (dfbeta[k] - expected).abs() <= 1e-12 * (1.0 + expected.abs()),
            "row {k}: {} against {expected}",
            dfbeta[k]
        );
    }
}

#[test]
fn counts_and_influences_match_enumeration() {
    for seed in 0..6 {
        for (delayed, strata) in [(false, false), (true, false), (false, true), (true, true)] {
            check(&simulate(seed, 90, delayed), delayed, strata);
        }
    }
}

#[test]
fn reversing_swaps_the_counts_and_flips_the_influence() {
    let sim = simulate(11, 60, true);
    let data = SurvData::builder(&sim.stop, &sim.status)
        .start(&sim.start)
        .build()
        .unwrap();
    let forward = ConcordanceOptions {
        influence: true,
        ..ConcordanceOptions::default()
    };
    let a = estimate(&data, &sim.x, &forward).unwrap();
    let b = estimate(
        &data,
        &sim.x,
        &ConcordanceOptions {
            reverse: true,
            ..forward
        },
    )
    .unwrap();
    assert_eq!(a.counts.concordant, b.counts.discordant);
    assert_eq!(a.counts.discordant, b.counts.concordant);
    assert_eq!(a.concordance.unwrap() + b.concordance.unwrap(), 1.0);
    assert_eq!(a.variance, b.variance);
    assert_eq!(a.cox_variance, b.cox_variance);
    let (da, db) = (a.influence.unwrap(), b.influence.unwrap());
    assert!(
        da.column(0)
            .iter()
            .zip(db.column(0))
            .all(|(p, q)| *p == -*q)
    );
}

#[test]
fn undefined_results_are_none_and_bad_input_is_an_error() {
    let stop = [1.0, 2.0, 3.0];
    let status = [0, 0, 0];
    let data = SurvData::builder(&stop, &status).build().unwrap();
    let r = estimate(&data, &[1.0, 2.0, 3.0], &ConcordanceOptions::default()).unwrap();
    assert!(r.concordance.is_none() && r.variance.is_none() && r.cox_variance.is_none());
    assert!(estimate(&data, &[1.0, 2.0], &ConcordanceOptions::default()).is_err());
    assert!(estimate(&data, &[1.0, f64::NAN, 3.0], &ConcordanceOptions::default()).is_err());
    let start = [0.0; 3];
    let counting = SurvData::builder(&stop, &status)
        .start(&start)
        .build()
        .unwrap();
    let opts = ConcordanceOptions {
        time_weight: TimeWeight::NOverG2,
        ..ConcordanceOptions::default()
    };
    assert!(estimate(&counting, &[1.0, 2.0, 3.0], &opts).is_err());
}
