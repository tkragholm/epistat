//! `cox::residuals` and `cox::zph` give identical bits in rayon pools of 1, 2,
//! 3, 4 and 8 threads.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the inputs are a few thousand rows, far below the limits of the casts"
)]

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::cox::{
    Collapse, CoxOptions, CoxResidualKind, ResidualOptions, Ties, ZphOptions, ZphTransform, fit,
    residuals, zph,
};
use epistat_testkit::{POOLS, absent, assert_same_on_pools};

#[test]
fn residuals_and_zph() {
    let n = 2203;
    let mut rng = Stream::seeded(53);
    let mut stop = Vec::new();
    let mut start = Vec::new();
    let mut status = Vec::new();
    let mut weight = Vec::new();
    let mut stratum = Vec::new();
    let mut cluster = Vec::new();
    let mut base = Vec::new();
    for _ in 0..n {
        let a = rng.normal();
        let t = (rng.exp1() * (-0.4 * a).exp() * 8.0).ceil() / 10.0 + 0.1;
        stop.push(t);
        start.push((rng.uniform() * t * 8.0).floor() / 10.0 * 0.9);
        status.push(i32::from(rng.below(3) != 0));
        weight.push(if rng.below(7) == 0 { 2.0 } else { 1.0 });
        stratum.push(rng.below(3) as u32);
        cluster.push(rng.below(31) as u32);
        base.push(a);
    }
    let x = Mat::from_fn(n, 3, |i, j| match j {
        2 => base[i] + base[i],
        1 => f64::from(u8::from(i % 3 == 0)),
        _ => base[i],
    });
    let data = SurvData::builder(&stop, &status)
        .start(&start)
        .weight(&weight)
        .stratum(&stratum)
        .cluster(&cluster)
        .build()
        .unwrap();
    let codes: Vec<u32> = (0..n).map(|i| (i % 11) as u32).collect();
    for ties in [Ties::Breslow, Ties::Efron] {
        let cox = CoxOptions {
            ties,
            ..CoxOptions::default()
        };
        assert_same_on_pools(&POOLS, || {
            let model = fit(&data, x.as_ref(), &cox).unwrap();
            let mut out = Vec::new();
            for (kind, collapse) in [
                (CoxResidualKind::Martingale, Collapse::None),
                (CoxResidualKind::Martingale, Collapse::Cluster),
                (CoxResidualKind::Deviance, Collapse::Codes(&codes)),
                (CoxResidualKind::Score, Collapse::None),
                (CoxResidualKind::Schoenfeld, Collapse::None),
                (CoxResidualKind::ScaledSchoenfeld, Collapse::None),
                (CoxResidualKind::Dfbeta, Collapse::Cluster),
                (CoxResidualKind::Dfbetas, Collapse::None),
            ] {
                let opts = ResidualOptions {
                    collapse,
                    ..ResidualOptions::new(kind, cox.clone())
                };
                let got = residuals(&data, x.as_ref(), &model, &opts).unwrap();
                for column in got.columns.iter().flatten() {
                    out.extend_from_slice(column);
                }
                out.extend(got.time.iter().copied());
            }
            for transform in [
                ZphTransform::Km,
                ZphTransform::Rank,
                ZphTransform::Identity,
                ZphTransform::Log,
            ] {
                let groups = [vec![0], vec![1, 2]];
                let opts = ZphOptions {
                    cox: cox.clone(),
                    transform,
                    groups: Some(groups.to_vec()),
                    singledf: false,
                };
                let got = zph(&data, x.as_ref(), &model, &opts).unwrap();
                for t in got.tests.iter().chain(std::iter::once(&got.global)) {
                    let t = t.expect("the test is defined");
                    out.extend([t.statistic, t.df as f64, t.p_value]);
                }
                out.extend(got.x.iter().copied());
                for column in &got.y {
                    out.extend(column.iter().copied().map(absent));
                }
                out.extend(got.variance.as_slice().iter().copied().map(absent));
            }
            out
        });
    }
}
