//! The survival curves, their limits, quantiles and restricted means give
//! identical bits in rayon pools of 1, 2, 3, 4 and 8 threads.
//!
//! The inputs are drawn from a seeded stream on a grid of tenths, so that times
//! tie. The number of rows is odd and above rayon's cut-off for a parallel sort,
//! there are seven strata of unequal size, some rows have weight 0, and subjects
//! have several rows.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the grid values and counts are far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::survfit::{
    ConfLower, ConfType, HazardType, Survfit, SurvfitOptions, SurvivalType, estimate,
};
use epistat_testkit::assert_same_on_pools;

const POOLS: [usize; 5] = [1, 2, 3, 4, 8];

/// A value standing for `None`, so that an absent number differs from any
/// present one.
const ABSENT: f64 = f64::from_bits(0x7ff8_0000_dead_beef);

struct Sample {
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    weight: Vec<f64>,
    stratum: Vec<u32>,
    cluster: Vec<u32>,
    id: Vec<u32>,
}

fn sample(n: usize, seed: u64) -> Sample {
    let mut rng = Stream::seeded(seed);
    let mut s = Sample {
        start: Vec::new(),
        stop: Vec::new(),
        status: Vec::new(),
        weight: Vec::new(),
        stratum: Vec::new(),
        cluster: Vec::new(),
        id: Vec::new(),
    };
    for row in 0..n {
        let start = (rng.uniform() * 30.0).floor() / 10.0;
        s.start.push(start);
        s.stop
            .push(start + 0.1 + (rng.exp1() * 40.0).floor() / 10.0);
        s.status.push(i32::from(rng.below(10) < 4));
        s.weight.push(if rng.below(25) == 0 {
            0.0
        } else {
            0.5 + rng.uniform() * 2.0
        });
        s.stratum.push(5 * rng.below(7) as u32 + 3);
        s.cluster.push(rng.below(600) as u32 * 2);
        // Two rows per subject, so that chains of intervals exist.
        s.id.push((row / 2) as u32 * 3);
    }
    s
}

fn flatten(fit: &Survfit, out: &mut Vec<f64>) {
    let opt = |v: Option<f64>| v.unwrap_or(ABSENT);
    out.extend([fit.n_rows as f64, fit.n_used as f64]);
    for g in &fit.groups {
        out.extend([
            f64::from(g.group),
            g.n as f64,
            g.n_id.map_or(-1.0, |n| n as f64),
        ]);
        for column in [
            &g.time,
            &g.n_risk,
            &g.n_event,
            &g.n_censor,
            &g.surv,
            &g.cumhaz,
            &g.se_cumhaz,
        ] {
            out.extend_from_slice(column);
        }
        for column in [&g.count_risk, &g.count_event, &g.count_censor] {
            out.extend(column.iter().map(|&c| f64::from(c)));
        }
        for column in [&g.se, &g.lower, &g.upper] {
            out.extend(column.iter().map(|&v| opt(v)));
        }
        for q in g.quantiles(&[0.1, 0.25, 0.5, 0.75, 0.9]).unwrap() {
            out.extend([opt(q.time), opt(q.lower), opt(q.upper)]);
        }
        for tau in [0.5, 2.0, 6.0, 40.0] {
            let r = g.rmean(tau).unwrap();
            out.extend([r.mean, r.se]);
        }
    }
}

#[test]
fn estimate_does_not_depend_on_the_number_of_threads() {
    let s = sample(3001, 31);
    for start in [None, Some(s.start.as_slice())] {
        // Model-based variance with and without the id, and the jackknife with
        // clusters, with ids and with each row alone.
        for (cluster, id, robust) in [
            (false, false, Some(false)),
            (false, start.is_some(), None),
            (true, false, None),
            (false, true, Some(true)),
            (false, false, Some(true)),
        ] {
            if robust == Some(true) && start.is_some() && !cluster && !id {
                continue;
            }
            let mut builder = SurvData::builder(&s.stop, &s.status)
                .weight(&s.weight)
                .stratum(&s.stratum);
            if let Some(start) = start {
                builder = builder.start(start);
            }
            if cluster {
                builder = builder.cluster(&s.cluster);
            }
            if id {
                builder = builder.id(&s.id);
            }
            let data = builder.build().unwrap();
            for (survival, hazard, conf_type, conf_lower) in [
                (
                    SurvivalType::KaplanMeier,
                    HazardType::NelsonAalen,
                    ConfType::Log,
                    ConfLower::Usual,
                ),
                (
                    SurvivalType::KaplanMeier,
                    HazardType::FlemingHarrington,
                    ConfType::LogLog,
                    ConfLower::Peto,
                ),
                (
                    SurvivalType::ExpCumulativeHazard,
                    HazardType::NelsonAalen,
                    ConfType::Arcsin,
                    ConfLower::Modified,
                ),
            ] {
                let opts = SurvfitOptions {
                    survival,
                    hazard,
                    conf_type,
                    conf_lower,
                    robust,
                    ..SurvfitOptions::default()
                };
                assert_same_on_pools(&POOLS, || {
                    let fit = estimate(&data, &opts).unwrap();
                    let mut out = Vec::new();
                    flatten(&fit, &mut out);
                    assert!(out.len() > 1000);
                    out
                });
            }
        }
    }
}
