//! Properties of [`epistat_survival::absrisk`] that need no reference: what is
//! refused, where a risk is undefined, the order of the times, and the case
//! weights.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "the inputs are a few hundred rows, far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::absrisk::{
    CauseSpecificCox, CscOptions, NewRows, RiskLimits, RiskOptions, SurvType,
};
use epistat_survival::cox::Ties;
use faer::Mat;

struct Sim {
    stop: Vec<f64>,
    status: Vec<i32>,
    x1: Mat<f64>,
    x2: Mat<f64>,
    stratum: Vec<u32>,
}

fn sim(n: usize, seed: u64) -> Sim {
    let mut rng = Stream::seeded(seed);
    let mut stop = Vec::new();
    let mut status = Vec::new();
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut stratum = Vec::new();
    for _ in 0..n {
        let u = rng.normal();
        let v = f64::from(u8::from(rng.below(2) == 0));
        let t1 = rng.exp1() / (0.10 * (0.5 * u).exp());
        let t2 = rng.exp1() / (0.06 * (-0.6 * v).exp());
        let c = 2.0 + 16.0 * rng.uniform();
        let t = t1.min(t2).min(c);
        stop.push((t * 1000.0).round() / 1000.0 + 0.001);
        status.push(if c < t1.min(t2) {
            0
        } else if t1 < t2 {
            1
        } else {
            2
        });
        a.push(u);
        b.push(v);
        stratum.push(rng.below(2) as u32 + 7);
    }
    Sim {
        stop,
        status,
        x1: Mat::from_fn(n, 1, |i, _| a[i]),
        x2: Mat::from_fn(n, 2, |i, j| if j == 0 { a[i] } else { b[i] }),
        stratum,
    }
}

fn risk_opts(std_err: bool) -> RiskOptions {
    RiskOptions {
        std_err,
        ..RiskOptions::default()
    }
}

#[test]
fn refusals() {
    let s = sim(150, 1);
    let data = SurvData::builder(&s.stop, &s.status)
        .n_codes(2)
        .build()
        .unwrap();
    let designs = [s.x1.as_ref(), s.x2.as_ref()];
    let opts = CscOptions::default();
    let model = CauseSpecificCox::fit(&data, &designs, &opts).unwrap();
    let new_x1 = Mat::from_fn(2, 1, |i, _| i as f64);
    let new_x2 = Mat::from_fn(2, 2, |i, j| (i + j) as f64);
    let new_designs = [new_x1.as_ref(), new_x2.as_ref()];
    let new = NewRows {
        designs: &new_designs,
        stratum: None,
    };
    let ok = model.predict(&new, &[1.0, 5.0], 1, &risk_opts(true));
    assert!(ok.is_ok());
    // A cause outside the codes, and 0 which is censoring.
    assert!(model.predict(&new, &[1.0], 3, &risk_opts(false)).is_err());
    assert!(model.predict(&new, &[1.0], 0, &risk_opts(false)).is_err());
    // A time that is not finite.
    assert!(
        model
            .predict(&new, &[f64::NAN], 1, &risk_opts(false))
            .is_err()
    );
    // The wrong number of designs, and of columns.
    let one = [new_x1.as_ref()];
    assert!(
        model
            .predict(
                &NewRows {
                    designs: &one,
                    stratum: None
                },
                &[1.0],
                1,
                &risk_opts(false)
            )
            .is_err()
    );
    let swapped = [new_x2.as_ref(), new_x1.as_ref()];
    assert!(
        model
            .predict(
                &NewRows {
                    designs: &swapped,
                    stratum: None
                },
                &[1.0],
                1,
                &risk_opts(false)
            )
            .is_err()
    );
    // A level outside (0, 1).
    let bad = RiskOptions {
        conf_level: 1.0,
        ..risk_opts(true)
    };
    assert!(model.predict(&new, &[1.0], 1, &bad).is_err());
    // One code, and a wrong number of designs at the fit.
    let single: Vec<i32> = s.status.iter().map(|&c| i32::from(c != 0)).collect();
    let one_code = SurvData::builder(&s.stop, &single).build().unwrap();
    assert!(CauseSpecificCox::fit(&one_code, &designs, &opts).is_err());
    assert!(CauseSpecificCox::fit(&data, &designs[..1], &opts).is_err());
    // The cause of a survival-type fit is the cause of the fit.
    let survival = CscOptions {
        surv_type: SurvType::Survival { cause: 2 },
        ..opts
    };
    let fitted = CauseSpecificCox::fit(&data, &designs, &survival).unwrap();
    assert!(fitted.predict(&new, &[1.0], 2, &risk_opts(true)).is_ok());
    assert!(fitted.predict(&new, &[1.0], 1, &risk_opts(false)).is_err());
    let out_of_range = CscOptions {
        surv_type: SurvType::Survival { cause: 3 },
        ..opts
    };
    assert!(CauseSpecificCox::fit(&data, &designs, &out_of_range).is_err());
}

#[test]
fn strata_are_required_and_known() {
    let s = sim(150, 2);
    let data = SurvData::builder(&s.stop, &s.status)
        .n_codes(2)
        .stratum(&s.stratum)
        .build()
        .unwrap();
    let designs = [s.x1.as_ref(), s.x2.as_ref()];
    let model = CauseSpecificCox::fit(&data, &designs, &CscOptions::default()).unwrap();
    let new_x1 = Mat::from_fn(2, 1, |i, _| i as f64);
    let new_x2 = Mat::from_fn(2, 2, |i, j| (i + j) as f64);
    let new_designs = [new_x1.as_ref(), new_x2.as_ref()];
    let missing = NewRows {
        designs: &new_designs,
        stratum: None,
    };
    assert!(
        model
            .predict(&missing, &[3.0], 1, &risk_opts(false))
            .is_err()
    );
    // The caller's codes are 7 and 8, not 0 and 1.
    let dense = [0u32, 1];
    let wrong = NewRows {
        designs: &new_designs,
        stratum: Some(&dense),
    };
    assert!(model.predict(&wrong, &[3.0], 1, &risk_opts(false)).is_err());
    let codes = [7u32, 8];
    let right = NewRows {
        designs: &new_designs,
        stratum: Some(&codes),
    };
    let out = model.predict(&right, &[3.0], 1, &risk_opts(true)).unwrap();
    assert!(out.risk.iter().all(Option::is_some));
    // The same rows in the other order give the other stratum's values.
    let swapped = [8u32, 7];
    let swap_designs = [
        Mat::from_fn(2, 1, |i, _| (1 - i) as f64),
        Mat::from_fn(2, 2, |i, j| (1 - i + j) as f64),
    ];
    let refs = [swap_designs[0].as_ref(), swap_designs[1].as_ref()];
    let reversed = model
        .predict(
            &NewRows {
                designs: &refs,
                stratum: Some(&swapped),
            },
            &[3.0],
            1,
            &risk_opts(true),
        )
        .unwrap();
    assert_eq!(out.risk[0], reversed.risk[1]);
    assert_eq!(out.risk[1], reversed.risk[0]);
    assert_eq!(
        out.std_err.as_ref().unwrap()[0],
        reversed.std_err.as_ref().unwrap()[1]
    );
}

#[test]
fn what_the_standard_error_needs() {
    let s = sim(120, 3);
    let weight: Vec<f64> = (0..s.stop.len()).map(|i| 1.0 + (i % 2) as f64).collect();
    let start: Vec<f64> = vec![0.0; s.stop.len()];
    let designs = [s.x1.as_ref(), s.x2.as_ref()];
    let new_x1 = Mat::from_fn(1, 1, |_, _| 0.3);
    let new_x2 = Mat::from_fn(1, 2, |_, j| j as f64);
    let new_designs = [new_x1.as_ref(), new_x2.as_ref()];
    let new = NewRows {
        designs: &new_designs,
        stratum: None,
    };
    // Case weights and entry times are refused for the standard error only.
    let weighted = SurvData::builder(&s.stop, &s.status)
        .n_codes(2)
        .weight(&weight)
        .build()
        .unwrap();
    let model = CauseSpecificCox::fit(&weighted, &designs, &CscOptions::default()).unwrap();
    assert!(model.predict(&new, &[3.0], 1, &risk_opts(true)).is_err());
    assert!(model.predict(&new, &[3.0], 1, &risk_opts(false)).is_ok());
    let entry = SurvData::builder(&s.stop, &s.status)
        .n_codes(2)
        .start(&start)
        .build()
        .unwrap();
    let model = CauseSpecificCox::fit(&entry, &designs, &CscOptions::default()).unwrap();
    assert!(model.predict(&new, &[3.0], 1, &risk_opts(true)).is_err());
    assert!(model.predict(&new, &[3.0], 1, &risk_opts(false)).is_ok());
}

#[test]
fn undefined_where_the_data_end_or_the_survival_is_not_a_probability() {
    let s = sim(200, 4);
    let data = SurvData::builder(&s.stop, &s.status)
        .n_codes(2)
        .build()
        .unwrap();
    let designs = [s.x1.as_ref(), s.x2.as_ref()];
    let model = CauseSpecificCox::fit(&data, &designs, &CscOptions::default()).unwrap();
    let last = s.stop.iter().copied().fold(0.0, f64::max);
    let first_event = s
        .stop
        .iter()
        .zip(&s.status)
        .filter(|&(_, &c)| c != 0)
        .map(|(t, _)| *t)
        .fold(f64::INFINITY, f64::min);
    let new_x1 = Mat::from_fn(2, 1, |i, _| if i == 0 { 0.0 } else { 40.0 });
    let new_x2 = Mat::from_fn(2, 2, |i, _| if i == 0 { 0.0 } else { 40.0 });
    let new_designs = [new_x1.as_ref(), new_x2.as_ref()];
    let new = NewRows {
        designs: &new_designs,
        stratum: None,
    };
    let times = [0.0, first_event * 0.5, first_event, last, last + 1.0];
    let out = model.predict(&new, &times, 1, &risk_opts(true)).unwrap();
    let at = |r: usize, t: usize| out.risk[r * times.len() + t];
    let se = |r: usize, t: usize| out.std_err.as_ref().unwrap()[r * times.len() + t];
    // Before the first event, and at time 0, the risk is 0 with no uncertainty.
    for t in 0..2 {
        assert_eq!(at(0, t), Some(0.0));
        assert_eq!(se(0, t), Some(0.0));
    }
    // After the last follow-up time nothing is defined.
    assert_eq!(at(0, 4), None);
    assert_eq!(se(0, 4), None);
    assert!(out.lower.as_ref().unwrap()[4].is_none());
    // At the last time the risk is a probability.
    let v = at(0, 3).unwrap();
    assert!((0.0..=1.0).contains(&v));
    // A risk score of exp(40 * beta) makes the first hazard increment exceed
    // one, so the survival is not a probability, and the risk is None from
    // then on and Some(0) before the first event.
    assert_eq!(at(1, 1), Some(0.0));
    assert_eq!(at(1, 3), None);
    // The exponential form has no such limit.
    let exp = model
        .predict(
            &new,
            &times,
            1,
            &RiskOptions {
                product_limit: false,
                ..risk_opts(false)
            },
        )
        .unwrap();
    assert!(exp.risk[times.len() + 3].is_some());
}

#[test]
fn the_order_of_the_times_does_not_matter() {
    let s = sim(160, 5);
    let data = SurvData::builder(&s.stop, &s.status)
        .n_codes(2)
        .build()
        .unwrap();
    let designs = [s.x1.as_ref(), s.x2.as_ref()];
    let model = CauseSpecificCox::fit(&data, &designs, &CscOptions::default()).unwrap();
    let new_x1 = Mat::from_fn(3, 1, |i, _| i as f64 - 1.0);
    let new_x2 = Mat::from_fn(3, 2, |i, j| (i * (j + 1)) as f64 * 0.5);
    let new_designs = [new_x1.as_ref(), new_x2.as_ref()];
    let new = NewRows {
        designs: &new_designs,
        stratum: None,
    };
    let sorted = [1.0, 3.0, 3.0, 7.5, 12.0];
    let shuffled = [7.5, 1.0, 12.0, 3.0, 3.0];
    let a = model.predict(&new, &sorted, 2, &risk_opts(true)).unwrap();
    let b = model.predict(&new, &shuffled, 2, &risk_opts(true)).unwrap();
    for r in 0..3 {
        for (i, &t) in shuffled.iter().enumerate() {
            let j = sorted
                .iter()
                .position(|u| u.to_bits() == t.to_bits())
                .unwrap();
            assert_eq!(a.risk[r * 5 + j], b.risk[r * 5 + i]);
            assert_eq!(
                a.std_err.as_ref().unwrap()[r * 5 + j],
                b.std_err.as_ref().unwrap()[r * 5 + i]
            );
            assert_eq!(
                a.lower.as_ref().unwrap()[r * 5 + j],
                b.lower.as_ref().unwrap()[r * 5 + i]
            );
        }
    }
    // No limits without a standard error, and none when asked for none.
    let plain = model.predict(&new, &sorted, 2, &risk_opts(false)).unwrap();
    assert!(plain.lower.is_none() && plain.upper.is_none() && plain.std_err.is_none());
    let none = model
        .predict(
            &new,
            &sorted,
            2,
            &RiskOptions {
                limits: RiskLimits::None,
                ..risk_opts(true)
            },
        )
        .unwrap();
    assert!(none.lower.is_none() && none.std_err.is_some());
    assert_eq!(none.risk, plain.risk);
}

/// A case weight of 2 is two rows with Breslow ties, so the baselines, and the
/// risk, are those of the data with the rows repeated.
#[test]
fn case_weights_are_row_repeats() {
    let s = sim(140, 6);
    let copies: Vec<usize> = (0..s.stop.len())
        .map(|i| if i % 3 == 0 { 2 } else { 1 })
        .collect();
    let weight: Vec<f64> = copies.iter().map(|&c| c as f64).collect();
    let mut stop = Vec::new();
    let mut status = Vec::new();
    let mut rows = Vec::new();
    for (i, &count) in copies.iter().enumerate() {
        for _ in 0..count {
            stop.push(s.stop[i]);
            status.push(s.status[i]);
            rows.push(i);
        }
    }
    let x1 = Mat::from_fn(rows.len(), 1, |i, _| s.x1[(rows[i], 0)]);
    let x2 = Mat::from_fn(rows.len(), 2, |i, j| s.x2[(rows[i], j)]);
    let opts = CscOptions {
        ties: Ties::Breslow,
        ..CscOptions::default()
    };
    let weighted = SurvData::builder(&s.stop, &s.status)
        .n_codes(2)
        .weight(&weight)
        .build()
        .unwrap();
    let repeated = SurvData::builder(&stop, &status)
        .n_codes(2)
        .build()
        .unwrap();
    let designs = [s.x1.as_ref(), s.x2.as_ref()];
    let rep_designs = [x1.as_ref(), x2.as_ref()];
    let b = CauseSpecificCox::fit(&repeated, &rep_designs, &opts).unwrap();
    // The same coefficients in both, from the repeated data.
    let a = CauseSpecificCox::from_fits(&weighted, &designs, b.fits().to_vec(), &opts).unwrap();
    let new_x1 = Mat::from_fn(2, 1, |i, _| i as f64);
    let new_x2 = Mat::from_fn(2, 2, |i, j| (i + j) as f64 * 0.4);
    let new_designs = [new_x1.as_ref(), new_x2.as_ref()];
    let new = NewRows {
        designs: &new_designs,
        stratum: None,
    };
    let times = [2.0, 5.0, 9.0];
    let wa = a.predict(&new, &times, 1, &risk_opts(false)).unwrap();
    let wb = b.predict(&new, &times, 1, &risk_opts(true)).unwrap();
    // The standard error of the repeated data is the one the model with unit
    // weights has.
    assert!(wb.std_err.is_some());
    // The two sum the same positive terms in a different order, and the risk
    // sets differ by the order of the repeated rows: a relative error of at
    // most 4 eps (n + J + 10) in each of the two, with n the rows and J at
    // most n, and the total hazard of these rows is below 1.
    let bound = 8.0 * f64::EPSILON * (2.0 * stop.len() as f64 + 10.0);
    for (g, e) in wa.risk.iter().zip(&wb.risk) {
        let (g, e) = (g.unwrap(), e.unwrap());
        assert!((g - e).abs() <= bound * e, "{g} against {e}");
    }
}
