//! Edge cases of `frailty::fit`: what is refused, aliased columns, clusters
//! without a used row, a model without covariates, theta = 0 as the Cox
//! model, the end states of the searches, and the profile score with the
//! estimate, interval and standard error it gives for Breslow ties.

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::cox::{
    self, Collapse, CoxOptions, CoxResidualKind, ResidualOptions, Ties, residuals,
};
use epistat_survival::frailty::{FrailtyMethod, FrailtyOptions, Theta, ThetaStatus, fit};
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

fn fixed(theta: f64, ties: Ties) -> FrailtyOptions {
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
        method: FrailtyMethod::Em {
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
                tolerance: 1e-14,
                ..CoxOptions::default()
            },
        )
        .unwrap();
        for k in 0..2 {
            let se = c.se[k].unwrap();
            assert!((f.coefficients[k].unwrap() - c.coefficients[k].unwrap()).abs() <= 1e-9 * se);
            assert!((f.se[k].unwrap() - se).abs() <= 1e-9 * se);
            assert_eq!(f.se[k], f.diagonal_se[k]);
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
    assert!((0..3).all(|j| a.covariance.get(1, j).is_none()));

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
    assert!(f.coefficients.is_empty() && f.covariance.dim() == 0);
    let mean: f64 = f.frailties.iter().map(|w| w.unwrap().exp()).sum::<f64>() / 30.0;
    assert!((mean - 1.0).abs() <= 1e-12, "{mean}");
}

/// The profile search keeps every theta it fitted, the first being 0.
#[test]
fn search_end_states() {
    let s = simulate(80, 5);
    let buf = data(&s);
    let d = buf.as_data();
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

/// Data with a gamma frailty of variance `theta` (none at 0) in `g` clusters
/// of `size` rows, or of 2 to 5 rows when `size` is 0.
fn simulate_theta(g: usize, size: usize, theta: f64, seed: u64) -> Sim {
    let mut rng = Stream::seeded(seed);
    let (mut time, mut status, mut cluster, mut weight) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut rows = Vec::new();
    for c in 0..g {
        // A gamma variate with mean 1 and variance 0.5, a sum of two
        // exponentials with mean 1/2.
        let z = if theta > 0.0 {
            (rng.exp1() + rng.exp1()) * theta
        } else {
            1.0
        };
        let n = if size > 0 { size } else { 2 + rng.below(4) };
        for _ in 0..n {
            let a = rng.normal();
            let t = rng.exp1() / (0.2 * z * (0.5 * a).exp());
            let cens = rng.exp1() * 10.0;
            time.push(t.min(cens));
            status.push(i32::from(t <= cens));
            cluster.push(u32::try_from(c).unwrap());
            weight.push(1.0);
            rows.push(a);
        }
    }
    let x = Mat::from_fn(rows.len(), 1, |i, _| rows[i]);
    Sim {
        time,
        status,
        cluster,
        weight,
        x,
    }
}

fn breslow_profile() -> FrailtyOptions {
    FrailtyOptions {
        ties: Ties::Breslow,
        ..FrailtyOptions::default()
    }
}

/// The profile score at theta is the derivative of the profile `c`. A
/// central difference with step `h` on theta has a truncation error of about
/// `h^2 c''' / 6`, which the difference between the estimates at `h` and
/// `h / 2` measures (it is 3/4 of the error at `h`), and a rounding error of
/// `2 e / h` with `e` the error of `c`. A fit stops after a Newton step whose
/// decrement was below `1e-10`, so the penalised likelihood, stationary, is
/// within the square of that of its maximum; `e` is then rounding, taken as
/// `1e-15 |c|`.
#[test]
fn the_profile_score_is_the_derivative_of_the_profile() {
    let s = simulate_theta(120, 0, 0.5, 11);
    let buf = data(&s);
    let d = buf.as_data();
    let at = |theta: f64| fit(&d, s.x.as_ref(), &fixed(theta, Ties::Breslow)).unwrap();
    for theta in [0.05, 0.3, 0.8, 2.0] {
        let f = at(theta);
        let score = f.theta_score.unwrap();
        let c = |t: f64| at(t).marginal_log_likelihood.unwrap();
        let diff = |h: f64| (c(theta + h) - c(theta - h)) / (2.0 * h);
        let h = 1e-3 * theta;
        let (d1, d2) = (diff(h), diff(h / 2.0));
        let e = 1e-15 * f.marginal_log_likelihood.unwrap().abs();
        let bound = 2.0 * (4.0 / 3.0) * (d1 - d2).abs() + 4.0 * e / h;
        assert!(
            (d2 - score).abs() <= bound,
            "{theta}: {score} {d2} bound {bound:e}"
        );
    }
}

/// The score at 0 is `(1/2) sum_i [(D_i - Lambda_i)^2 - D_i]`, where
/// `D_i - Lambda_i` is the cluster's sum of the martingale residuals of the
/// Breslow Cox fit. The two Cox fits agree to `1e-9` standard errors
/// (`theta_zero_is_the_cox_model`), which moves each `Lambda_i` by about
/// `1e-9` relative per unit of `|x| se` in the risk scores, here below
/// `1e-8`, and the score by `sum_i |D_i - Lambda_i| 1e-8 Lambda_i`.
#[test]
fn the_score_at_zero_is_the_closed_form() {
    let s = simulate_theta(80, 0, 0.5, 12);
    let buf = data(&s);
    let d = buf.as_data();
    let f = fit(&d, s.x.as_ref(), &fixed(0.0, Ties::Breslow)).unwrap();
    let options = CoxOptions {
        ties: Ties::Breslow,
        tolerance: 1e-14,
        ..CoxOptions::default()
    };
    let c = cox::fit(&d, s.x.as_ref(), &options).unwrap();
    let grouped = residuals(
        &d,
        s.x.as_ref(),
        &c,
        &ResidualOptions {
            collapse: Collapse::Cluster,
            ..ResidualOptions::new(CoxResidualKind::Martingale, options)
        },
    )
    .unwrap();
    let m = grouped.columns[0].as_ref().unwrap();
    let mut events = vec![0.0; 80];
    for (&k, &e) in s.cluster.iter().zip(&s.status) {
        events[k as usize] += f64::from(e);
    }
    let mut want = 0.0;
    let mut bound = 0.0;
    for (&r, &dk) in m.iter().zip(&events) {
        want += 0.5 * (r * r - dk);
        bound += r.abs() * 1e-8 * (dk - r).abs();
    }
    let got = f.score_at_zero.unwrap();
    assert!(
        (got - want).abs() <= bound + 1e-12,
        "{got} {want} {bound:e}"
    );
    assert_eq!(f.theta_score.unwrap().to_bits(), got.to_bits());
}

/// EM and Newton's method on the penalised likelihood reach the same profile
/// maximum with Breslow ties. EM stops when the marginal likelihood changes
/// by less than `1e-14` relative, which leaves it within `1e-9` of the
/// maximum at each theta (`em_equals_penalised`). The two maxima of `c` then
/// agree to `1e-9`, and since `c` is quadratic near its maximum with
/// curvature `1 / se^2` on `ln theta`, the two roots agree to `sqrt(2e-9) se`,
/// plus twice the root finder's tolerance.
#[test]
fn em_and_newton_reach_the_same_profile_maximum() {
    let s = simulate_theta(60, 0, 0.5, 13);
    let buf = data(&s);
    let d = buf.as_data();
    let a = fit(&d, s.x.as_ref(), &breslow_profile()).unwrap();
    let em = FrailtyOptions {
        method: FrailtyMethod::Em {
            max_iterations: 20_000,
            tolerance: 1e-14,
        },
        ..breslow_profile()
    };
    let b = fit(&d, s.x.as_ref(), &em).unwrap();
    assert_eq!(a.theta_status, ThetaStatus::Interior);
    assert_eq!(b.theta_status, ThetaStatus::Interior);
    let (ma, mb) = (
        a.marginal_log_likelihood.unwrap(),
        b.marginal_log_likelihood.unwrap(),
    );
    assert!((ma - mb).abs() <= 1e-9, "{ma} {mb}");
    let se = a.ln_theta_se.unwrap();
    let bound = 2e-9_f64.sqrt() * se + 2e-6;
    let dt = (a.theta.ln() - b.theta.ln()).abs();
    assert!(dt <= bound, "{} {} {dt:e} {bound:e}", a.theta, b.theta);
}

/// The profile at the ends of the interval is `q / 2` below its maximum, to
/// the root finder's tolerance on `ln theta`, `1e-6`, times the slope of `c`
/// in `ln theta` there, `theta c'(theta)`, plus the rounding of `c`. A Wald
/// interval on `ln theta` from the standard error is close to the profile
/// interval when `c` is nearly quadratic; only its order is checked.
#[test]
fn the_interval_ends_are_at_the_cut() {
    let s = simulate_theta(150, 0, 0.5, 14);
    let buf = data(&s);
    let d = buf.as_data();
    let f = fit(&d, s.x.as_ref(), &breslow_profile()).unwrap();
    assert_eq!(f.theta_status, ThetaStatus::Interior);
    let top = f.marginal_log_likelihood.unwrap();
    let interval = f.theta_interval.unwrap();
    let q = epistat_stats::dist::chi_square::quantile(0.95, 1.0);
    let cut = top - q / 2.0;
    let (lower, upper) = (interval.lower.unwrap(), interval.upper.unwrap());
    assert!(
        lower < f.theta && f.theta < upper,
        "{lower} {} {upper}",
        f.theta
    );
    for end in [lower, upper] {
        let e = fit(&d, s.x.as_ref(), &fixed(end, Ties::Breslow)).unwrap();
        let c = e.marginal_log_likelihood.unwrap();
        let bound = 2e-6 * (end * e.theta_score.unwrap()).abs() + 1e-12 * top.abs();
        assert!((c - cut).abs() <= bound, "{end}: {c} {cut} {bound:e}");
    }
    let se = f.ln_theta_se.unwrap();
    let wald = (f.theta.ln() - 1.96 * se, f.theta.ln() + 1.96 * se);
    assert!((wald.0 - lower.ln()).abs() < 1.0 && (wald.1 - upper.ln()).abs() < 1.0);
}

/// Without heterogeneity the score at 0 decides the boundary: a fit whose
/// score at 0 is not positive is the Cox model, with the likelihood ratio
/// p-value of 1 and an interval from 0; one whose score is positive has an
/// interior estimate, whose p-value is half the chi-square tail of Self and
/// Liang.
#[test]
fn the_score_at_zero_decides_the_boundary() {
    let mut boundary = 0;
    let mut interior = 0;
    for seed in 20..30 {
        let s = simulate_theta(100, 3, 0.0, seed);
        let buf = data(&s);
        let d = buf.as_data();
        let f = fit(&d, s.x.as_ref(), &breslow_profile()).unwrap();
        let u0 = f.score_at_zero.unwrap();
        let lr = f.likelihood_ratio.unwrap();
        if u0 <= 0.0 {
            boundary += 1;
            assert_eq!(f.theta_status, ThetaStatus::Boundary, "{seed}");
            assert_eq!(f.theta.to_bits(), 0.0_f64.to_bits());
            assert_eq!(lr.p_value.to_bits(), 1.0_f64.to_bits());
            assert_eq!(f.theta_interval.unwrap().lower, Some(0.0));
            assert!(f.ln_theta_se.is_none());
        } else {
            interior += 1;
            assert_eq!(f.theta_status, ThetaStatus::Interior, "{seed}");
            if lr.statistic > 0.0 {
                let want = 0.5 * epistat_stats::dist::chi_square::sf(lr.statistic, 1.0);
                assert_eq!(lr.p_value.to_bits(), want.to_bits());
            }
        }
    }
    assert!(boundary > 0 && interior > 0, "{boundary} {interior}");
    let s = simulate_theta(100, 3, 0.5, 31);
    let buf = data(&s);
    let f = fit(&buf.as_data(), s.x.as_ref(), &breslow_profile()).unwrap();
    assert_eq!(f.theta_status, ThetaStatus::Interior);
    assert!(f.score_at_zero.unwrap() > 0.0);
    assert!(f.likelihood_ratio.unwrap().p_value < 0.05);
}

/// When every cluster has one row theta is not identified: the search ends on
/// the boundary or is an error. Leaving the interval out gives the same fit.
#[test]
fn singleton_clusters_and_no_interval() {
    for seed in 40..44 {
        let s = simulate_theta(200, 1, 0.5, seed);
        let buf = data(&s);
        let d = buf.as_data();
        for ties in [Ties::Breslow, Ties::Efron] {
            let options = FrailtyOptions {
                ties,
                ..FrailtyOptions::default()
            };
            match fit(&d, s.x.as_ref(), &options) {
                Ok(f) => assert_eq!(f.theta_status, ThetaStatus::Boundary),
                Err(e) => assert!(e.to_string().contains("one row"), "{e}"),
            }
        }
    }
    let s = simulate_theta(60, 0, 0.5, 45);
    let buf = data(&s);
    let d = buf.as_data();
    let with = fit(&d, s.x.as_ref(), &breslow_profile()).unwrap();
    let without = fit(
        &d,
        s.x.as_ref(),
        &FrailtyOptions {
            level: None,
            ..breslow_profile()
        },
    )
    .unwrap();
    assert!(without.theta_interval.is_none());
    assert!(with.theta_interval.is_some());
    assert_eq!(with.theta.to_bits(), without.theta.to_bits());
    assert_eq!(with.coefficients, without.coefficients);
}
