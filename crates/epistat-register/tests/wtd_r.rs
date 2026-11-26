//! The parametric waiting time distribution against the likelihood of
//! Støvring et al. 2016 written out in base R and maximised with `optim`
//! (`tests/data/wtd_reference.R`).
//!
//! Both sides stop at a point where the gradient of the mean negative
//! log-likelihood is small. Near the optimum the objective is quadratic, so
//! two such points are at most `(|g_1| + |g_2|) / lambda_min` apart, with
//! `lambda_min` the smallest eigenvalue of the Hessian that R reports; the
//! test takes twice that. The log of the duration moves by at most twice the
//! largest change of a parameter on the scale of estimation (its derivatives
//! there are 1 and `z sigma < 1` for the log-normal, `-1` and
//! `-log(-log 0.2) / alpha` for the Weibull), so its relative tolerance is
//! four times the parameter bound. The log-likelihood is a sum of `n` logs of
//! order 10, so the two sums differ by rounding of about `n * 10 * n * eps`,
//! below `1e-8` for these `n`.

use epistat_register::wtd::{Direction, Family, fit_wtd, waiting_times};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Window {
    start: i64,
    end: i64,
}

#[derive(Deserialize)]
struct Fit {
    family: String,
    direction: String,
    n: usize,
    x: Vec<f64>,
    loglik: f64,
    duration: f64,
    gradient_norm: f64,
    min_eigen: f64,
}

#[derive(Deserialize)]
struct Reference {
    window: Window,
    person: Vec<u32>,
    day: Vec<i64>,
    fits: Vec<Fit>,
}

#[test]
fn fits_agree_with_the_likelihood_in_r() {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/wtd_reference.json"
    ));
    let persons = *r.person.last().unwrap() as usize + 1;
    let mut offsets = vec![0_usize; persons + 1];
    for &p in &r.person {
        offsets[p as usize + 1] += 1;
    }
    for p in 0..persons {
        offsets[p + 1] += offsets[p];
    }
    #[expect(clippy::cast_precision_loss, reason = "a window of days")]
    let delta = (r.window.end - r.window.start) as f64;
    assert_eq!(r.fits.len(), 4);
    for f in &r.fits {
        let direction = match f.direction.as_str() {
            "forward" => Direction::Forward,
            _ => Direction::Reverse,
        };
        let family = match f.family.as_str() {
            "lognormal" => Family::LogNormal,
            _ => Family::Weibull,
        };
        let times =
            waiting_times(&offsets, &r.day, r.window.start, r.window.end, direction).unwrap();
        assert_eq!(times.len(), f.n, "{} {}", f.direction, f.family);
        let fit = fit_wtd(&times, delta, family).unwrap();
        assert!(fit.converged, "{} {}", f.direction, f.family);
        let g = fit.gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
        let bound = 2.0 * (g + f.gradient_norm) / f.min_eigen;
        assert!(bound < 1e-6, "the bound itself is tight: {bound:e}");
        for j in 0..3 {
            assert!(
                (fit.estimation_scale[j] - f.x[j]).abs() <= bound,
                "{} {} parameter {j}: {} against {} (bound {bound:e})",
                f.direction,
                f.family,
                fit.estimation_scale[j],
                f.x[j]
            );
        }
        let duration = fit.duration(0.8).unwrap();
        assert!(
            (duration / f.duration - 1.0).abs() <= 4.0 * bound,
            "{} {} duration {duration} against {}",
            f.direction,
            f.family,
            f.duration
        );
        assert!((fit.log_likelihood - f.loglik).abs() < 1e-8);
        // The simulated inter-arrival distribution has its 80th percentile at
        // 112.1 days; the estimates are near it.
        assert!((duration - 112.1).abs() < 15.0);
    }
}
