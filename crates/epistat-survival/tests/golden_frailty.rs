//! The bits of `frailty::fit` on varied inputs, pinned by a hash of every output
//! field, so that a change to any last bit is a failure here. The bits depend on
//! the platform's `ln` and `exp`, so the test runs only on the target it was
//! recorded on, aarch64 macOS. Thread-count independence is checked by
//! `threads_frailty`.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvBuf;
use epistat_survival::cox::Ties;
use epistat_survival::frailty::{FrailtyFit, FrailtyMethod, FrailtyOptions, Theta, fit};
use epistat_testkit::Fingerprint;

fn hash(f: &FrailtyFit) -> u64 {
    let mut h = Fingerprint::new();
    for v in f
        .coefficients
        .iter()
        .chain(&f.se)
        .chain(f.covariance.as_slice())
        .chain(f.covariance_sandwich.as_slice())
        .chain(&f.diagonal_se)
        .chain(f.diagonal_covariance.as_slice())
        .chain(f.diagonal_covariance_sandwich.as_slice())
        .chain(&f.frailties)
        .chain(&f.frailty_variances)
    {
        h = h.option_f64(*v);
    }
    h = h
        .f64(f.theta)
        .u64(f.theta_status as u64)
        .f64(f.kendall_tau)
        .f64(f.log_partial_likelihood)
        .f64(f.penalised_log_likelihood)
        .option_f64(f.marginal_log_likelihood)
        .option_f64(f.cox_log_likelihood);
    if let Some(t) = f.likelihood_ratio {
        h = h.f64(t.statistic).f64(t.p_value);
    }
    for s in &f.history {
        h = h
            .f64(s.theta)
            .f64(s.log_partial_likelihood)
            .f64(s.marginal_log_likelihood)
            .u64(u64::from(s.converged));
    }
    h.u64(u64::try_from(f.iterations).unwrap())
        .u64(u64::from(f.converged))
        .f64(f.n_events)
        .u64(f.n_rows as u64)
        .u64(f.n_used as u64)
        .u64(f.n_clusters as u64)
        .finish()
}

fn run(entry: bool, weights: bool, strata: bool, opts: &FrailtyOptions) -> u64 {
    let n = 600;
    let mut rng = Stream::seeded(91);
    let (mut time, mut start, mut status, mut cluster, mut weight, mut stratum) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let mut x = Mat::<f64>::zeros(n, 2);
    let mut z = 1.0;
    for i in 0..n {
        if i % 3 == 0 {
            z = rng.exp1() * 0.6 + 0.4;
        }
        let a = rng.normal();
        let b = f64::from(u8::from(rng.below(5) < 2));
        let t = rng.exp1() / (0.2 * z * (0.5 * a - 0.4 * b).exp());
        let c = rng.exp1() * 10.0;
        let stop = (t.min(c) * 10.0).ceil() / 10.0;
        time.push(stop);
        start.push(if rng.below(3) == 0 {
            (0.3 * stop * 10.0).floor() / 10.0
        } else {
            0.0
        });
        status.push(i32::from(t <= c));
        cluster.push(u32::try_from(i / 3).unwrap());
        weight.push([0.5, 1.0, 2.0][rng.below(3)]);
        stratum.push(u32::try_from(rng.below(2)).unwrap());
        x[(i, 0)] = a;
        x[(i, 1)] = b;
    }
    let mut b = SurvBuf::builder(time, status).cluster(cluster);
    if entry {
        b = b.start(start);
    }
    if weights {
        b = b.weight(weight);
    }
    if strata {
        b = b.stratum(stratum);
    }
    let buf = b.build().unwrap();
    hash(&fit(&buf.as_data(), x.as_ref(), opts).unwrap())
}

#[test]
fn bits_are_pinned() {
    let fixed = |ties| FrailtyOptions {
        ties,
        theta: Theta::Fixed(0.5),
        ..FrailtyOptions::default()
    };
    let cases: Vec<(&str, u64)> = vec![
        ("fixed efron", run(false, false, false, &fixed(Ties::Efron))),
        (
            "fixed breslow entry weights strata",
            run(true, true, true, &fixed(Ties::Breslow)),
        ),
        (
            "fixed efron entry weights strata",
            run(true, true, true, &fixed(Ties::Efron)),
        ),
        (
            "profile efron entry strata",
            run(true, false, true, &FrailtyOptions::default()),
        ),
        (
            "coxph search breslow",
            run(
                false,
                false,
                false,
                &FrailtyOptions {
                    ties: Ties::Breslow,
                    theta: Theta::coxph(),
                    ..FrailtyOptions::default()
                },
            ),
        ),
        (
            "em breslow",
            run(
                false,
                false,
                false,
                &FrailtyOptions {
                    method: FrailtyMethod::Em {
                        max_iterations: 2000,
                        tolerance: 1e-12,
                    },
                    ..fixed(Ties::Breslow)
                },
            ),
        ),
    ];
    let recorded: Vec<String> = cases
        .iter()
        .map(|(name, h)| format!("(\"{name}\", {h:#018x}),"))
        .collect();
    let want: Vec<(&str, u64)> = EXPECTED.to_vec();
    assert_eq!(cases, want, "recorded:\n{}", recorded.join("\n"));
}

const EXPECTED: &[(&str, u64)] = &[
    ("fixed efron", 0xed38_3ce1_8831_ac73),
    ("fixed breslow entry weights strata", 0x474b_29b3_bc83_59aa),
    ("fixed efron entry weights strata", 0x2746_d7c1_be37_1f67),
    ("profile efron entry strata", 0x1e7e_1ed6_b4c2_a98e),
    ("coxph search breslow", 0x4b6f_e8e0_eb4b_d424),
    ("em breslow", 0x3cc0_b0cf_ac3a_a56f),
];
