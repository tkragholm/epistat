//! Properties of `msreg` that need no reference: the model without covariates
//! is the Aalen-Johansen estimate, hazards that are one are one, rows of
//! weight 0 change nothing, an offset is a change of reference, and the input
//! checks.

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::cox::predict::HazardType;
use epistat_survival::msreg::{
    MsBaseline, MsFitOptions, MsOptions, MsTerm, MsTerms, ProbTransOptions, fit,
    observed_transitions, probtrans,
};
use epistat_survival::multistate::{Options, estimate};
use epistat_survival::timefix::TimeFix;

/// Illness-death data for `n` subjects with two covariates: rows with `start`,
/// `stop`, the status, the state the row starts in and the subject.
struct Data {
    id: Vec<u32>,
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    istate: Vec<u32>,
    x: Mat<f64>,
}

fn next(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

fn illness_death(n: usize, seed: u64) -> Data {
    let mut rng = seed;
    let mut d = Data {
        id: Vec::new(),
        start: Vec::new(),
        stop: Vec::new(),
        status: Vec::new(),
        istate: Vec::new(),
        x: Mat::zeros(0, 0),
    };
    let mut covariates = Vec::new();
    for i in 0..n {
        let x1 = f64::from(next(&mut rng) < 0.4);
        let x2 = (next(&mut rng) * 4.0).floor() / 2.0;
        let mut t = 0.0;
        let mut state = 1u32;
        loop {
            // Integer times, so that events and censorings tie.
            let wait = (-(next(&mut rng)).ln() / (0.08 * (0.3 * x1 + 0.1 * x2).exp()))
                .ceil()
                .max(1.0);
            let censor = next(&mut rng) < 0.15;
            let to: u32 = if state == 1 {
                if next(&mut rng) < 0.6 { 2 } else { 3 }
            } else {
                3
            };
            let stop = t + wait;
            d.id.push(u32::try_from(i).unwrap() + 100);
            d.start.push(t);
            d.stop.push(stop);
            d.istate.push(state);
            covariates.push((x1, x2));
            if censor || stop > 40.0 {
                d.status.push(0);
                break;
            }
            d.status.push(i32::try_from(to).unwrap());
            if to == 3 {
                break;
            }
            t = stop;
            state = to;
        }
    }
    d.x = Mat::from_fn(covariates.len(), 2, |i, j| {
        if j == 0 {
            covariates[i].0
        } else {
            covariates[i].1
        }
    });
    d
}

fn data_of(d: &Data) -> SurvData<'_> {
    SurvData::builder(&d.stop, &d.status)
        .start(&d.start)
        .id(&d.id)
        .istate(&d.istate)
        .n_codes(3)
        .build()
        .unwrap()
}

#[test]
fn the_model_without_covariates_is_the_aalen_johansen_estimate() {
    let d = illness_death(300, 7);
    let data = data_of(&d);
    let x = Mat::<f64>::zeros(d.stop.len(), 0);
    let model = fit(&data, x.as_ref(), &MsOptions::default()).unwrap();
    assert!(model.terms.is_empty() && model.coefficients().is_empty());
    assert_eq!(model.fit.coefficients.len() + model.fit.covariance.len(), 0);
    assert_eq!(model.fit.rank, 0);
    let h = model
        .msfit(
            &[],
            &MsFitOptions {
                hazard: Some(HazardType::Breslow),
                se: false,
                ..MsFitOptions::default()
            },
        )
        .unwrap();
    let pt = probtrans(
        &h,
        0.0,
        &ProbTransOptions {
            se: false,
            ..ProbTransOptions::default()
        },
    )
    .unwrap();
    let aj = estimate(
        &data,
        &h.time,
        &Options {
            initial: Some(vec![1.0, 0.0, 0.0]),
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(pt.time.len(), h.time.len() + 1);
    // Both are products of at most `T` factors, each adding a few epsilon.
    let bound = 8.0 * h.time.len() as f64 * f64::EPSILON;
    let mut worst = 0.0f64;
    for (t, _) in h.time.iter().enumerate() {
        for state in 0..3 {
            worst = worst.max(
                (pt.probability_of(t + 1, 0, state).unwrap() - aj.probabilities[t * 3 + state])
                    .abs(),
            );
        }
    }
    eprintln!("probability against Aalen-Johansen: {worst:.2e}, bound {bound:.2e}");
    assert!(worst <= bound);
}

#[test]
fn common_coefficients_on_one_baseline_make_one_hazard() {
    let d = illness_death(300, 11);
    let data = data_of(&d);
    let opts = MsOptions {
        terms: MsTerms::Common,
        baseline: MsBaseline::Shared,
        ..MsOptions::default()
    };
    let model = fit(&data, d.x.as_ref(), &opts).unwrap();
    let h = model.msfit(&[1.0, 1.5], &MsFitOptions::default()).unwrap();
    let k = h.transitions.len();
    assert_eq!(k, 3);
    for a in 0..k {
        assert_eq!(h.cumulative_hazard[a], h.cumulative_hazard[0]);
        for b in 0..k {
            let (va, cab) = (
                h.covariance_of(a, a).unwrap(),
                h.covariance_of(a, b).unwrap(),
            );
            for (x, y) in va.iter().zip(cab) {
                assert!((x - y).abs() <= 8.0 * f64::EPSILON * x.abs(), "{x} {y}");
            }
        }
    }
}

#[test]
fn rows_of_weight_zero_change_nothing() {
    let d = illness_death(200, 3);
    let n = d.stop.len();
    // The first 30 subjects get weight 0 in one run and are removed in the other.
    let weight: Vec<f64> = (0..n)
        .map(|i| if d.id[i] < 130 { 0.0 } else { 1.0 })
        .collect();
    let keep: Vec<usize> = (0..n).filter(|&i| d.id[i] >= 130).collect();
    let data = SurvData::builder(&d.stop, &d.status)
        .start(&d.start)
        .id(&d.id)
        .istate(&d.istate)
        .weight(&weight)
        .n_codes(3)
        .build()
        .unwrap();
    let pick = |v: &[f64]| keep.iter().map(|&i| v[i]).collect::<Vec<f64>>();
    let (stop, start) = (pick(&d.stop), pick(&d.start));
    let status: Vec<i32> = keep.iter().map(|&i| d.status[i]).collect();
    let id: Vec<u32> = keep.iter().map(|&i| d.id[i]).collect();
    let istate: Vec<u32> = keep.iter().map(|&i| d.istate[i]).collect();
    let reduced = SurvData::builder(&stop, &status)
        .start(&start)
        .id(&id)
        .istate(&istate)
        .n_codes(3)
        .build()
        .unwrap();
    let x_reduced = Mat::from_fn(keep.len(), 2, |i, j| d.x[(keep[i], j)]);
    let a = fit(&data, d.x.as_ref(), &MsOptions::default()).unwrap();
    let b = fit(&reduced, x_reduced.as_ref(), &MsOptions::default()).unwrap();
    assert_eq!(a.fit.n_rows, n);
    assert_eq!(b.fit.n_rows, keep.len());
    assert_eq!(a.n_stacked, b.n_stacked);
    for (u, v) in a.fit.coefficients.iter().zip(&b.fit.coefficients) {
        assert_eq!(u.unwrap().to_bits(), v.unwrap().to_bits());
    }
}

#[test]
fn an_offset_in_the_model_is_a_change_of_reference() {
    let d = illness_death(200, 5);
    let data = data_of(&d);
    let offset = vec![0.7; d.stop.len()];
    let plain = fit(&data, d.x.as_ref(), &MsOptions::default()).unwrap();
    let shifted_data = SurvData::builder(&d.stop, &d.status)
        .start(&d.start)
        .id(&d.id)
        .istate(&d.istate)
        .n_codes(3)
        .offset(&offset)
        .build()
        .unwrap();
    let shifted = fit(&shifted_data, d.x.as_ref(), &MsOptions::default()).unwrap();
    let h0 = plain.msfit(&[1.0, 1.0], &MsFitOptions::default()).unwrap();
    let h1 = shifted
        .msfit(
            &[1.0, 1.0],
            &MsFitOptions {
                offset: 0.7,
                ..MsFitOptions::default()
            },
        )
        .unwrap();
    for (a, b) in h0.cumulative_hazard.iter().zip(&h1.cumulative_hazard) {
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() <= 1e-9 * x.abs().max(1e-300), "{x} {y}");
        }
    }
}

#[test]
fn the_transitions_are_those_observed_in_order() {
    let d = illness_death(100, 1);
    let data = data_of(&d);
    assert_eq!(
        observed_transitions(&data, TimeFix::default()).unwrap(),
        vec![(1, 2), (1, 3), (2, 3)]
    );
    // An event that is not on the list censors the transitions out of its state.
    let x = Mat::<f64>::zeros(d.stop.len(), 0);
    let only = [(1, 2), (2, 3)];
    let model = fit(
        &data,
        x.as_ref(),
        &MsOptions {
            transitions: Some(&only),
            ..MsOptions::default()
        },
    )
    .unwrap();
    assert_eq!(model.transitions.len(), 2);
    assert_eq!(model.n_states, 3);
}

fn err<T>(r: epistat_survival::Result<T>) -> String {
    match r {
        Ok(_) => panic!("accepted"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn bad_input_is_refused_with_the_reason() {
    let d = illness_death(60, 2);
    let data = data_of(&d);
    let n = d.stop.len();
    let x = &d.x;
    let msg = err(fit(
        &data,
        Mat::<f64>::zeros(n - 1, 1).as_ref(),
        &MsOptions::default(),
    ));
    assert!(msg.contains("x has length"), "{msg}");
    // Terms.
    for (terms, what) in [
        (vec![MsTerm::covariate(5, &[0])], "column 5 of 2"),
        (vec![MsTerm::covariate(0, &[9])], "transition 9 of 3"),
        (vec![MsTerm::covariate(0, &[])], "no transition"),
        (
            vec![MsTerm::covariate(0, &[0, 1]), MsTerm::covariate(0, &[1])],
            "repeats a coefficient",
        ),
    ] {
        let msg = err(fit(
            &data,
            x.as_ref(),
            &MsOptions {
                terms: MsTerms::Custom(&terms),
                ..MsOptions::default()
            },
        ));
        assert!(msg.contains(what), "{msg}");
    }
    let msg = err(fit(
        &data,
        x.as_ref(),
        &MsOptions {
            baseline: MsBaseline::Groups(&[0, 1]),
            ..MsOptions::default()
        },
    ));
    assert!(msg.contains("baseline"), "{msg}");
    for (list, what) in [
        (vec![(1, 1)], "leaves and enters one state"),
        (vec![(1, 4)], "outside the states"),
        (vec![(1, 2), (1, 2)], "listed twice"),
        (vec![], "empty"),
    ] {
        let msg = err(fit(
            &data,
            x.as_ref(),
            &MsOptions {
                transitions: Some(&list),
                ..MsOptions::default()
            },
        ));
        assert!(msg.contains(what), "{msg}");
    }
    let msg = err(SurvData::builder(&d.stop, &d.status).offset(&[0.0]).build());
    assert!(msg.contains("offset"), "{msg}");
    // Data without an id, with one status code, with an overlap.
    let status1: Vec<i32> = d.status.iter().map(|&s| i32::from(s > 0)).collect();
    let one_code_data = SurvData::builder(&d.stop, &status1)
        .start(&d.start)
        .id(&d.id)
        .build()
        .unwrap();
    let msg = err(fit(&one_code_data, x.as_ref(), &MsOptions::default()));
    assert!(msg.contains("one code"), "{msg}");
    let no_id = SurvData::builder(&d.stop, &d.status)
        .start(&d.start)
        .istate(&d.istate)
        .n_codes(3)
        .build()
        .unwrap();
    let msg = err(fit(&no_id, x.as_ref(), &MsOptions::default()));
    assert!(msg.contains("id"), "{msg}");
    // A row that ends in the state it starts in.
    let mut same = d.status.clone();
    let first = (0..n)
        .find(|&i| d.istate[i] == 2 && d.status[i] == 3)
        .unwrap();
    same[first] = 2;
    let data_same = SurvData::builder(&d.stop, &same)
        .start(&d.start)
        .id(&d.id)
        .istate(&d.istate)
        .n_codes(3)
        .build()
        .unwrap();
    let msg = err(fit(&data_same, x.as_ref(), &MsOptions::default()));
    assert!(msg.contains("the state it starts in"), "{msg}");
    // A row that overlaps the one before it.
    let mut start = d.start.clone();
    let second = (0..n).find(|&i| d.istate[i] == 2).unwrap();
    start[second] -= 1.0;
    let data_overlap = SurvData::builder(&d.stop, &d.status)
        .start(&start)
        .id(&d.id)
        .istate(&d.istate)
        .n_codes(3)
        .build()
        .unwrap();
    let msg = err(fit(&data_overlap, x.as_ref(), &MsOptions::default()));
    assert!(msg.contains("overlap"), "{msg}");
    // A model's predictions.
    let model = fit(&data, x.as_ref(), &MsOptions::default()).unwrap();
    let msg = err(model.msfit(&[1.0], &MsFitOptions::default()));
    assert!(msg.contains("covariate columns"), "{msg}");
    let msg = err(model.msfit(&[1.0, f64::NAN], &MsFitOptions::default()));
    assert!(msg.contains("not finite"), "{msg}");
    let robust = MsFitOptions {
        variance: epistat_survival::cox::predict::Variance::Robust,
        ..MsFitOptions::default()
    };
    assert!(model.msfit(&[1.0, 0.0], &robust).is_ok());
    let model_only = fit(
        &data,
        x.as_ref(),
        &MsOptions {
            robust: false,
            ..MsOptions::default()
        },
    )
    .unwrap();
    assert!(model_only.fit.robust_covariance.is_none());
    let msg = err(model_only.msfit(&[1.0, 0.0], &robust));
    assert!(msg.contains("robust"), "{msg}");
    let h = model.msfit(&[1.0, 0.0], &MsFitOptions::default()).unwrap();
    let msg = err(probtrans(&h, 1e9, &ProbTransOptions::default()));
    assert!(msg.contains("no event time is after"), "{msg}");
    let h_plain = model
        .msfit(
            &[1.0, 0.0],
            &MsFitOptions {
                se: false,
                ..MsFitOptions::default()
            },
        )
        .unwrap();
    let msg = err(probtrans(&h_plain, 0.0, &ProbTransOptions::default()));
    assert!(msg.contains("no covariances"), "{msg}");
    let msg = err(probtrans(&h, f64::NAN, &ProbTransOptions::default()));
    assert!(msg.contains("predt"), "{msg}");
}
