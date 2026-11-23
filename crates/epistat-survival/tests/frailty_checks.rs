//! Edge cases of `frailty::fit`: what is refused, aliased columns, clusters
//! without a used row, a model without covariates, theta = 0 as the Cox
//! model, and the end states of the searches.

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::cox::{self, CoxOptions, Ties};
use epistat_survival::frailty::{FrailtyOptions, Method, Theta, ThetaStatus, fit};
use epistat_survival::{SurvBuf, SurvData};

struct Sim {
    time: Vec<f64>,
    status: Vec<i32>,
    cluster: Vec<u32>,
    weight: Vec<f64>,
    x: Mat<f64>,
}

fn simulate(g: usize, seed: u64) -> Sim {
    let mut rng = Stream::seeded(seed);
    let (mut time, mut status, mut cluster, mut weight) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut rows = Vec::new();
    for c in 0..g {
        let z = rng.exp1() * 0.6 + 0.4;
        let size = 2 + rng.below(4);
        let b = f64::from(u8::from(rng.below(2) == 0));
        for _ in 0..size {
            let a = rng.normal();
            let t = rng.exp1() / (0.2 * z * (0.5 * a - 0.4 * b).exp());
            let cens = rng.exp1() * 10.0;
            time.push((t.min(cens) * 10.0).ceil() / 10.0);
            status.push(i32::from(t <= cens));
            cluster.push(u32::try_from(c).unwrap());
            weight.push(1.0);
            rows.push([a, b]);
        }
    }
    let x = Mat::from_fn(rows.len(), 2, |i, j| rows[i][j]);
    Sim {
        time,
        status,
        cluster,
        weight,
        x,
    }
}

fn data(s: &Sim) -> SurvBuf {
    SurvBuf::builder(s.time.clone(), s.status.clone())
        .cluster(s.cluster.clone())
        .build()
        .unwrap()
}

fn fixed(theta: f64, ties: Ties) -> FrailtyOptions<'static> {
    FrailtyOptions {
        ties,
        theta: Theta::Fixed(theta),
        ..FrailtyOptions::default()
    }
}

#[test]
fn refuses_what_is_not_defined() {
    let s = simulate(20, 1);
    let buf = data(&s);
    let d = buf.as_data();
    let no_cluster = SurvData::builder(&s.time, &s.status).build().unwrap();
    assert!(fit(&no_cluster, s.x.as_ref(), &fixed(0.5, Ties::Efron)).is_err());
    assert!(fit(&d, s.x.as_ref(), &fixed(0.5, Ties::Exact)).is_err());
    assert!(fit(&d, s.x.as_ref(), &fixed(-0.1, Ties::Efron)).is_err());
    assert!(fit(&d, s.x.as_ref(), &fixed(f64::NAN, Ties::Efron)).is_err());
    let em_efron = FrailtyOptions {
        method: Method::Em {
            max_iterations: 10,
            tolerance: 1e-10,
        },
        ..fixed(0.5, Ties::Efron)
    };
    assert!(fit(&d, s.x.as_ref(), &em_efron).is_err());
    let mut w = s.weight.clone();
    w[0] = 2.0;
    let weighted = SurvBuf::builder(s.time.clone(), s.status.clone())
        .cluster(s.cluster.clone())
        .weight(w)
        .build()
        .unwrap();
    let profile = FrailtyOptions::default();
    assert!(fit(&weighted.as_data(), s.x.as_ref(), &profile).is_err());
    let at_fixed = fit(&weighted.as_data(), s.x.as_ref(), &fixed(0.5, Ties::Efron)).unwrap();
    assert!(at_fixed.marginal_log_likelihood.is_none());
    assert!(at_fixed.cox_log_likelihood.is_none());
    let short = Mat::<f64>::zeros(3, 2);
    assert!(fit(&d, short.as_ref(), &fixed(0.5, Ties::Efron)).is_err());
}

/// Theta = 0 is the Cox model, which `cox::fit` fits by its own Newton
/// iteration: the two agree within the stopping rules, `1e-9` standard errors.
#[test]
fn theta_zero_is_the_cox_model() {
    let s = simulate(60, 2);
    let buf = data(&s);
    let d = buf.as_data();
    for ties in [Ties::Breslow, Ties::Efron] {
        let f = fit(&d, s.x.as_ref(), &fixed(0.0, ties)).unwrap();
        let c = cox::fit(
            &d,
            s.x.as_ref(),
            &CoxOptions {
                ties,
                eps: 1e-14,
                ..CoxOptions::default()
            },
        )
        .unwrap();
        for k in 0..2 {
            let se = c.standard_errors[k].unwrap();
            assert!((f.coefficients[k].unwrap() - c.coefficients[k].unwrap()).abs() <= 1e-9 * se);
            assert!((f.standard_errors[k].unwrap() - se).abs() <= 1e-9 * se);
            assert_eq!(f.standard_errors[k], f.diagonal_standard_errors[k]);
        }
        let l = c.log_partial_likelihood.unwrap();
        assert!((f.log_partial_likelihood - l).abs() <= 1e-12 * l.abs());
        assert!(f.frailties.iter().all(|w| *w == Some(0.0)));
        assert_eq!(f.kendall_tau.to_bits(), 0.0_f64.to_bits());
    }
}

/// An aliased column is `None` and the others are the fit without it; a
/// cluster whose rows all have weight zero has no frailty.
#[test]
fn aliased_columns_and_empty_clusters() {
    let s = simulate(40, 3);
    let n = s.time.len();
    let x3 = Mat::from_fn(n, 3, |i, j| match j {
        0 => s.x[(i, 0)],
        1 => 2.0 * s.x[(i, 0)],
        _ => s.x[(i, 1)],
    });
    let buf = data(&s);
    let d = buf.as_data();
    let a = fit(&d, x3.as_ref(), &fixed(0.5, Ties::Efron)).unwrap();
    let b = fit(&d, s.x.as_ref(), &fixed(0.5, Ties::Efron)).unwrap();
    assert_eq!(a.coefficients[1], None);
    assert_eq!(a.coefficients[0], b.coefficients[0]);
    assert_eq!(a.coefficients[2], b.coefficients[1]);
    assert!(a.covariance[3..6].iter().all(Option::is_none));

    let mut w = s.weight.clone();
    for (i, wi) in w.iter_mut().enumerate() {
        if s.cluster[i] == 7 {
            *wi = 0.0;
        }
    }
    let wbuf = SurvBuf::builder(s.time.clone(), s.status.clone())
        .cluster(s.cluster.clone())
        .weight(w)
        .build()
        .unwrap();
    let e = fit(&wbuf.as_data(), s.x.as_ref(), &fixed(0.5, Ties::Efron)).unwrap();
    assert_eq!(e.frailties[7], None);
    assert_eq!(e.frailty_variances[7], None);
    assert_eq!(e.n_clusters, 39);
    assert!(
        e.frailties
            .iter()
            .enumerate()
            .all(|(i, w)| i == 7 || w.is_some())
    );
}

/// The model without covariates has only frailties; their exp has mean 1.
#[test]
fn no_covariates() {
    let s = simulate(30, 4);
    let n = s.time.len();
    let buf = data(&s);
    let empty = Mat::<f64>::zeros(n, 0);
    let f = fit(&buf.as_data(), empty.as_ref(), &fixed(0.5, Ties::Efron)).unwrap();
    assert!(f.converged);
    assert!(f.coefficients.is_empty() && f.covariance.is_empty());
    let mean: f64 = f.frailties.iter().map(|w| w.unwrap().exp()).sum::<f64>() / 30.0;
    assert!((mean - 1.0).abs() <= 1e-12, "{mean}");
}

/// coxph's search stops at its fit limit and says so; the profile search and
/// coxph's search keep every theta they fitted, the first being 0.
#[test]
fn search_end_states() {
    let s = simulate(80, 5);
    let buf = data(&s);
    let d = buf.as_data();
    let short = FrailtyOptions {
        theta: Theta::Coxph {
            eps: 1e-5,
            outer_max: 3,
        },
        ..FrailtyOptions::default()
    };
    let f = fit(&d, s.x.as_ref(), &short).unwrap();
    assert_eq!(f.theta_status, ThetaStatus::SearchLimit);
    assert_eq!(f.history.len(), 3);
    assert_eq!(f.history[0].theta.to_bits(), 0.0_f64.to_bits());
    assert_eq!(f.history[1].theta.to_bits(), 1.0_f64.to_bits());
    assert_eq!(f.theta.to_bits(), f.history[2].theta.to_bits());
    let p = fit(&d, s.x.as_ref(), &FrailtyOptions::default()).unwrap();
    assert_eq!(p.theta_status, ThetaStatus::Interior);
    assert_eq!(p.history[0].theta.to_bits(), 0.0_f64.to_bits());
    let best = p
        .history
        .iter()
        .map(|h| h.marginal_log_likelihood)
        .fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(p.marginal_log_likelihood.unwrap().to_bits(), best.to_bits());
    assert!(p.likelihood_ratio.unwrap().p_value < 0.5);
}
