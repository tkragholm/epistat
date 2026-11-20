//! Pseudo-values, their estimates and the regression on them give identical
//! bits in rayon pools of 1, 2, 3, 4 and 8 threads, for every target and both
//! methods.
//!
//! The data have two causes, three strata of unequal size, tied times on a grid
//! of tenths, delayed entry, case weights with some of weight 0, and subjects
//! with several rows. There are more subjects in a stratum than a block of the
//! influence recursion holds, and an odd number of rows.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the grid values and counts are far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::pseudo::{GeeOptions, Link, Method, Pseudo, PseudoOptions, Target, pseudo};
use epistat_testkit::{POOLS, assert_same_on_pools};
use faer::Mat;

struct Sample {
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    weight: Vec<f64>,
    stratum: Vec<u32>,
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
        id: Vec::new(),
    };
    for row in 0..n {
        let start = (rng.uniform() * 30.0).floor() / 10.0;
        s.start.push(start);
        s.stop
            .push(start + 0.1 + (rng.exp1() * 40.0).floor() / 10.0);
        s.status.push(match rng.below(10) {
            0..=2 => 1,
            3..=4 => 2,
            _ => 0,
        });
        s.weight.push(if rng.below(25) == 0 {
            0.0
        } else {
            0.5 + rng.uniform() * 2.0
        });
        s.stratum.push(5 * rng.below(3) as u32 + 3);
        s.id.push((row / 2) as u32 * 3);
    }
    // A subject belongs to one stratum.
    for row in (1..n).step_by(2) {
        s.stratum[row] = s.stratum[row - 1];
    }
    s
}

fn flatten(p: &Pseudo, out: &mut Vec<f64>) {
    out.extend([p.n_rows as f64, p.n_used as f64]);
    out.extend(p.group.iter().map(|&g| f64::from(g)));
    out.extend(p.subject.iter().map(|&g| f64::from(g)));
    out.extend_from_slice(&p.values);
    for g in &p.groups {
        out.push(f64::from(g.group));
        out.extend_from_slice(&g.estimate);
        out.extend(g.beyond.iter().map(|&b| f64::from(u8::from(b))));
    }
}

#[test]
fn pseudo_does_not_depend_on_the_number_of_threads() {
    let s = sample(3001, 53);
    let times = [0.5, 1.5, 3.0, 6.0, 12.0];
    for with_start in [false, true] {
        let mut builder = SurvData::builder(&s.stop, &s.status)
            .n_codes(2)
            .weight(&s.weight)
            .stratum(&s.stratum)
            .id(&s.id);
        if with_start {
            builder = builder.start(&s.start);
        }
        let data = builder.build().unwrap();
        for target in [
            Target::Survival,
            Target::CumulativeIncidence { cause: 1 },
            Target::CumulativeIncidence { cause: 2 },
            Target::RestrictedMean,
            Target::RestrictedTimeLost { cause: 2 },
        ] {
            for method in [Method::InfinitesimalJackknife, Method::Jackknife] {
                let mut opts = PseudoOptions::new(target);
                opts.method = method;
                assert_same_on_pools(&POOLS, || {
                    let p = pseudo(&data, &times, &opts).unwrap();
                    let mut out = Vec::new();
                    flatten(&p, &mut out);
                    assert!(out.len() > 5000);
                    out
                });
            }
        }
    }
}

#[test]
fn regression_does_not_depend_on_the_number_of_threads() {
    let s = sample(3001, 59);
    let times = [1.5, 3.0, 6.0];
    // The pseudo-values of the whole sample, regressed on a covariate.
    let data = SurvData::builder(&s.stop, &s.status)
        .n_codes(2)
        .weight(&s.weight)
        .id(&s.id)
        .build()
        .unwrap();
    for (target, link) in [
        (Target::CumulativeIncidence { cause: 1 }, Link::Log),
        (Target::Survival, Link::Cloglog),
        (Target::RestrictedMean, Link::Identity),
    ] {
        let opts = PseudoOptions::new(target);
        assert_same_on_pools(&POOLS, || {
            let p = pseudo(&data, &times, &opts).unwrap();
            let n = p.n_subjects();
            let covariate = Mat::from_fn(n, 2, |i, j| {
                if j == 0 {
                    f64::from(p.subject[i] % 2 == 0)
                } else {
                    f64::from(p.subject[i] % 7) / 7.0
                }
            });
            let mut gee = GeeOptions::new(link);
            gee.max_iterations = 200;
            let fit = p.regress(Some(covariate.as_ref()), true, &gee).unwrap();
            let mut out = fit.coefficients.clone();
            out.extend_from_slice(&fit.covariance);
            out.extend_from_slice(&fit.fitted);
            out.extend([fit.information_condition, fit.iterations as f64]);
            out
        });
    }
}
