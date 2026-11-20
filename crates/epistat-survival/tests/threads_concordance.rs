//! `concordance::estimate` gives identical bits in rayon pools of 1, 2, 3, 4
//! and 8 threads.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the inputs are a few thousand rows, far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::concordance::{ConcordanceOptions, TimeWeight, estimate};
use epistat_testkit::assert_same_on_pools;

const POOLS: [usize; 5] = [1, 2, 3, 4, 8];

/// A value standing for `None`, so that an absent number differs from any
/// present one.
const ABSENT: u64 = 0x7ff8_0000_dead_beef;

fn opt(v: Option<f64>) -> f64 {
    v.unwrap_or_else(|| f64::from_bits(ABSENT))
}

/// Strata, a cluster, case weights with zeros, fractional and tied times,
/// delayed entry and a score with ties, under each time weight that allows it.
#[test]
fn estimate_and_influence() {
    let n = 4001;
    let mut rng = Stream::seeded(41);
    let mut start = Vec::new();
    let mut stop = Vec::new();
    let mut status = Vec::new();
    let mut weight = Vec::new();
    let mut stratum = Vec::new();
    let mut cluster = Vec::new();
    let mut score = Vec::new();
    for _ in 0..n {
        let x = rng.normal();
        let t = (rng.exp1() * (-0.5 * x).exp() * 6.0).ceil() / 7.0 + 0.1;
        stop.push(t);
        start.push((rng.uniform() * t * 5.0).floor() / 7.0 * 0.8);
        status.push(i32::from(rng.below(3) != 0));
        weight.push(match rng.below(6) {
            0 => 0.0,
            1 => 2.5,
            _ => 1.0 + rng.uniform(),
        });
        stratum.push(rng.below(5) as u32 * 3);
        cluster.push(rng.below(60) as u32);
        score.push((x * 4.0).round() / 4.0);
    }
    for delayed in [false, true] {
        for kind in [
            TimeWeight::N,
            TimeWeight::S,
            TimeWeight::SOverG,
            TimeWeight::NOverG2,
            TimeWeight::I,
        ] {
            if delayed && matches!(kind, TimeWeight::SOverG | TimeWeight::NOverG2) {
                continue;
            }
            let mut builder = SurvData::builder(&stop, &status)
                .weight(&weight)
                .stratum(&stratum)
                .cluster(&cluster);
            if delayed {
                builder = builder.start(&start);
            }
            let data = builder.build().unwrap();
            let opts = ConcordanceOptions {
                time_weight: kind,
                reverse: true,
                ymax: Some(1.2),
                influence: true,
                ..ConcordanceOptions::default()
            };
            assert_same_on_pools(&POOLS, || {
                let r = estimate(&data, &score, &opts).unwrap();
                let c = r.counts;
                let mut out = vec![
                    c.concordant,
                    c.discordant,
                    c.tied_x,
                    c.tied_y,
                    c.tied_xy,
                    opt(r.concordance),
                    opt(r.variance),
                    opt(r.cox_variance),
                ];
                out.extend(r.strata.iter().flat_map(|s| {
                    [
                        f64::from(s.code),
                        s.counts.concordant,
                        s.counts.discordant,
                        s.counts.tied_x,
                    ]
                }));
                out.extend(r.influence.unwrap());
                out
            });
        }
    }
}
