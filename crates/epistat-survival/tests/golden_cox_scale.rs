//! The bits of `cox::fit` and `cox::score_residuals` on problems of tens of
//! thousands of rows, pinned by a hash of every output field, on one thread
//! and on several. The rows span many blocks of the parallel preparation and
//! risk scores, and the problems take the paths of a fit: tied times, delayed
//! entry, strata, case weights with zeros, an offset, starting values, clusters,
//! a column without variation, more than sixteen columns, and Breslow, Efron
//! and exact ties. The first problem has more rows than a block of the
//! parallel sweep over the risk sets, so its hash is that of the blocked
//! sweep; the others are swept as one running sum, as before the sweep ran in
//! blocks.
//!
//! The bits depend on the platform's `ln` and `exp`, so the test asserts the
//! recorded values only on the target it was recorded on, aarch64 macOS.
//! Elsewhere it computes the hashes and, when `EPISTAT_HASH_OUT` is set,
//! appends them to that file.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the inputs are tens of thousands of rows, far below the limits of the casts"
)]

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvBuf;
use epistat_survival::cox::{CoxModelFit, CoxOptions, Ties, fit, score_residuals};
use epistat_testkit::{Fingerprint, on_threads};

fn hash_options(h: Fingerprint, values: &[Option<f64>]) -> Fingerprint {
    values
        .iter()
        .fold(h.u64(values.len() as u64), |h, &v| h.option_f64(v))
}

fn hash_fit(fit: &CoxModelFit) -> Fingerprint {
    let mut h = hash_options(Fingerprint::new(), &fit.coefficients);
    h = hash_options(h, &fit.se);
    h = hash_options(h, fit.covariance.as_slice());
    h = match &fit.robust_se {
        Some(v) => hash_options(h.u64(1), v),
        None => h.u64(0),
    };
    h = match &fit.robust_covariance {
        Some(v) => hash_options(h.u64(1), v.as_slice()),
        None => h.u64(0),
    };
    for test in [
        fit.tests.wald,
        fit.tests.robust_wald,
        fit.tests.score,
        fit.tests.robust_score,
        fit.tests.likelihood_ratio,
    ] {
        h = match test {
            Some(t) => h.u64(1).f64(t.statistic).u64(t.df as u64).f64(t.p_value),
            None => h.u64(0),
        };
    }
    h.option_f64(fit.log_partial_likelihood)
        .option_f64(fit.null_log_partial_likelihood)
        .f64(fit.n_events)
        .u64(fit.n_rows as u64)
        .u64(fit.n_used as u64)
        .u64(fit.rank as u64)
        .u64(u64::try_from(fit.iterations).unwrap())
        .u64(u64::from(fit.converged))
}

struct Sample {
    x: Mat<f64>,
    status: Vec<i32>,
    start: Vec<f64>,
    stop: Vec<f64>,
    weight: Vec<f64>,
    stratum: Vec<u32>,
    cluster: Vec<u32>,
    offset: Vec<f64>,
}

/// Whole-day times so that events tie, `p` columns of which column 2 is
/// constant, entry times before the exit.
fn sample(n: usize, p: usize, seed: u64) -> Sample {
    let mut rng = Stream::seeded(seed);
    let mut x = Mat::<f64>::zeros(n, p);
    let mut s = Sample {
        x: Mat::zeros(0, 0),
        status: Vec::new(),
        start: Vec::new(),
        stop: Vec::new(),
        weight: Vec::new(),
        stratum: Vec::new(),
        cluster: Vec::new(),
        offset: Vec::new(),
    };
    for i in 0..n {
        let mut eta = 0.0;
        for j in 0..p {
            x[(i, j)] = match j {
                2 => 1.5,
                1 => f64::from(rng.below(3) as u32),
                _ => rng.normal(),
            };
            eta += 0.1 * x[(i, j)] / (1 + j % 4) as f64;
        }
        let stop = (rng.exp1() * (-eta).exp() * 300.0).ceil() + 1.0;
        s.start.push((rng.uniform() * stop).floor().min(stop - 1.0));
        s.stop.push(stop);
        s.status.push(i32::from(rng.uniform() < 0.6));
        s.weight.push(match rng.below(8) {
            0 => 0.0,
            1 => 2.0,
            2 => 0.5,
            _ => 1.0,
        });
        s.stratum.push(rng.below(4) as u32 * 5);
        s.cluster.push((i / 4) as u32);
        s.offset.push(0.1 * rng.normal());
    }
    s.x = x;
    s
}

#[derive(Clone, Copy)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each flag turns one input of the fit on or off"
)]
struct Case {
    entry: bool,
    strata: bool,
    weights: bool,
    clusters: bool,
    offset: bool,
    init: bool,
    ties: Ties,
}

fn run(s: &Sample, case: Case) -> u64 {
    let mut builder = SurvBuf::builder(s.stop.clone(), s.status.clone());
    if case.entry {
        builder = builder.start(s.start.clone());
    }
    if case.strata {
        builder = builder.stratum(s.stratum.clone());
    }
    if case.weights {
        builder = builder.weight(s.weight.clone());
    }
    if case.clusters {
        builder = builder.cluster(s.cluster.clone());
    }
    if case.offset {
        builder = builder.offset(s.offset.clone());
    }
    let buf = builder.build().unwrap();
    let data = buf.as_data();
    let init: Vec<f64> = (0..s.x.ncols()).map(|j| 0.05 * j as f64).collect();
    let opts = CoxOptions {
        ties: case.ties,
        start: case.init.then_some(init),
        ..CoxOptions::default()
    };
    let fitted = fit(&data, s.x.as_ref(), &opts).unwrap();
    assert!(fitted.converged);
    let mut h = hash_fit(&fitted);
    if case.ties != Ties::Exact {
        for column in score_residuals(&data, s.x.as_ref(), &fitted.coefficients, &opts).unwrap() {
            h = match column {
                Some(values) => h.u64(1).f64s(&values),
                None => h.u64(0),
            };
        }
    }
    h.finish()
}

fn fits() -> Vec<u64> {
    let plain = Case {
        entry: false,
        strata: false,
        weights: false,
        clusters: false,
        offset: false,
        init: false,
        ties: Ties::Efron,
    };
    let everything = Case {
        entry: true,
        strata: true,
        weights: true,
        clusters: true,
        offset: true,
        init: false,
        ties: Ties::Efron,
    };
    let narrow = sample(20_011, 5, 20_261_124);
    let wide = sample(9_001, 18, 20_261_125);
    let small = sample(1_500, 3, 20_261_126);
    vec![
        run(&narrow, plain),
        run(&narrow, everything),
        run(
            &narrow,
            Case {
                ties: Ties::Breslow,
                init: true,
                ..everything
            },
        ),
        run(
            &narrow,
            Case {
                entry: false,
                offset: false,
                ..everything
            },
        ),
        run(&wide, plain),
        run(&wide, everything),
        run(
            &small,
            Case {
                ties: Ties::Exact,
                strata: true,
                ..plain
            },
        ),
        run(
            &small,
            Case {
                ties: Ties::Exact,
                entry: true,
                offset: true,
                ..plain
            },
        ),
    ]
}

#[test]
fn the_fits_keep_their_bits_on_every_pool() {
    let expected: [u64; 8] = [
        11_701_549_807_329_737_045,
        9_480_662_531_853_594_351,
        1_634_127_325_466_869_148,
        3_913_290_817_384_065_141,
        16_403_543_435_294_141_440,
        3_861_591_647_582_365_767,
        5_105_298_889_272_943_880,
        6_564_290_637_452_342_372,
    ];
    let results = on_threads(&[1, 3, 8], fits);
    for (i, hash) in results[0].iter().enumerate() {
        epistat_testkit::hash_out(&format!("fits/{i}"), *hash);
    }
    // Off the recording target the pools still have to agree with each other.
    let want: Vec<u64> = if epistat_testkit::RECORDED_PLATFORM {
        expected.to_vec()
    } else {
        results[0].clone()
    };
    for (threads, got) in [1, 3, 8].into_iter().zip(results) {
        assert_eq!(got, want, "{threads} threads");
    }
}
