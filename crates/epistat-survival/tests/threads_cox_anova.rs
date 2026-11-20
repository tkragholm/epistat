//! `cox::anova` and `cox::anova_terms` give identical bits in rayon pools of 1,
//! 2, 3, 4 and 8 threads.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the inputs are a few thousand rows, far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::cox::anova::anova_terms;
use epistat_survival::cox::{CoxOptions, Ties};
use epistat_testkit::assert_same_on_pools;
use faer::Mat;

const POOLS: [usize; 5] = [1, 2, 3, 4, 8];
const ABSENT: u64 = 0x7ff8_0000_dead_beef;

#[test]
fn anova_terms_is_the_same_on_every_pool() {
    let n = 1500;
    let mut rng = Stream::seeded(53);
    let mut stop = Vec::new();
    let mut status = Vec::new();
    let mut stratum = Vec::new();
    let mut a = Vec::new();
    let mut b = Vec::new();
    for _ in 0..n {
        let u = rng.normal();
        a.push(u);
        b.push(f64::from(u8::from(rng.below(2) == 0)));
        stop.push((rng.exp1() * (-0.3 * u).exp() * 8.0).ceil() / 10.0 + 0.1);
        status.push(i32::from(rng.below(4) != 0));
        stratum.push(rng.below(3) as u32);
    }
    let x = Mat::from_fn(n, 3, |i, j| match j {
        0 => a[i],
        1 => b[i],
        _ => a[i] * b[i],
    });
    let data = SurvData::builder(&stop, &status)
        .stratum(&stratum)
        .build()
        .unwrap();
    let g0: &[usize] = &[0];
    let g1: &[usize] = &[1, 2];
    for ties in [Ties::Breslow, Ties::Efron] {
        let opts = CoxOptions {
            ties,
            ..CoxOptions::default()
        };
        assert_same_on_pools(&POOLS, || {
            let table = anova_terms(&data, x.as_ref(), &[g0, g1], &opts).unwrap();
            let mut out = Vec::new();
            for row in &table.rows {
                for v in [row.log_likelihood, row.chisq, row.p_value] {
                    out.push(v.unwrap_or_else(|| f64::from_bits(ABSENT)));
                }
                out.push(row.df.map_or(f64::from_bits(ABSENT), |d| d as f64));
            }
            out
        });
    }
}
