//! `split_lexis`, `aggregate` and `fit_rates` give identical bits in rayon
//! pools of 1, 2, 3, 4 and 8 threads, on random data on a coarse grid so that
//! breaks are met together. The data is large enough for several chunks.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "the grid values and counts are far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::lexis::cells::{CellKey, aggregate};
use epistat_survival::lexis::rates::{RateOptions, fit_rates};
use epistat_survival::lexis::{LexisOptions, TimeScale, split_cells, split_lexis};
use epistat_testkit::{POOLS, assert_same_on_pools};

#[test]
fn lexis_does_not_depend_on_the_number_of_threads() {
    let n = 60_017;
    let mut rng = Stream::seeded(23);
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    let mut stop = Vec::with_capacity(n);
    let mut status = Vec::with_capacity(n);
    for _ in 0..n {
        a.push(rng.below(80) as f64 / 4.0);
        b.push(1980.0 + rng.below(80) as f64 / 4.0);
        stop.push((1 + rng.below(100)) as f64 / 4.0);
        status.push(rng.below(2) as i32);
    }
    let data = SurvData::builder(&stop, &status).build().unwrap();
    let breaks_a: Vec<f64> = (0..=10).map(|k| f64::from(2 * k)).collect();
    let breaks_b: Vec<f64> = (0..=8).map(|k| 1980.0 + 3.0 * f64::from(k)).collect();
    for crop in [false, true] {
        assert_same_on_pools(&POOLS, || {
            let out = split_lexis(
                &data,
                &[
                    TimeScale {
                        entry: &a,
                        breaks: &breaks_a,
                    },
                    TimeScale {
                        entry: &b,
                        breaks: &breaks_b,
                    },
                ],
                &LexisOptions { crop },
            )
            .unwrap();
            let mut v: Vec<f64> = Vec::new();
            v.extend(&out.entry[0]);
            v.extend(&out.entry[1]);
            v.extend(&out.length);
            v.extend(out.interval[0].iter().map(|&x| f64::from(x)));
            v.extend(out.interval[1].iter().map(|&x| f64::from(x)));
            v.extend(out.status.iter().map(|&x| f64::from(x)));
            v.extend(out.row.iter().map(|&x| f64::from(x)));
            let events = out.events(1);
            let cells = aggregate(
                &[
                    CellKey {
                        codes: &out.interval[0],
                        levels: breaks_a.len() + 1,
                    },
                    CellKey {
                        codes: &out.interval[1],
                        levels: breaks_b.len() + 1,
                    },
                ],
                &out.length,
                &events,
            )
            .unwrap();
            v.extend(&cells.length);
            v.extend(&cells.events);
            let ones = vec![1.0; cells.len()];
            let x: Vec<f64> = cells.codes[0].iter().map(|&c| f64::from(c)).collect();
            let fit = fit_rates(
                &cells.length,
                &cells.events,
                &[&ones, &x],
                &RateOptions::default(),
            )
            .unwrap();
            v.extend(fit.fit.coefficients.iter().map(|c| c.unwrap()));
            v.extend(fit.fit.covariance.as_slice().iter().map(|c| c.unwrap()));
            v.push(fit.fit.deviance);
            let direct = split_cells(
                &data,
                &[
                    TimeScale {
                        entry: &a,
                        breaks: &breaks_a,
                    },
                    TimeScale {
                        entry: &b,
                        breaks: &breaks_b,
                    },
                ],
                &[],
                None,
                &LexisOptions { crop },
            )
            .unwrap();
            v.extend(&direct.length);
            v.extend(&direct.events);
            v.extend(direct.pieces.iter().map(|&c| c as f64));
            v
        });
    }
}
