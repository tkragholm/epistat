//! Properties of the later parts of `msreg` that need no reference: a Greenwood
//! variance worked by hand, a fixed horizon that is the product taken from the
//! other end, several profiles that are the profiles one by one, and the input
//! checks.

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::msreg::{
    MsBaseline, MsFitOptions, MsOptions, MsProfile, MsTerm, MsTerms, MsVarType, ProbTransDirection,
    ProbTransMethod, ProbTransOptions, fit, probtrans,
};

/// Five subjects in state 1: a move to state 2 at 1, to 3 at 1, to 3 at 2, a
/// censoring at 3 and a move to 2 at 4.
fn small() -> (Vec<u32>, Vec<f64>, Vec<i32>, Vec<u32>) {
    (
        vec![1, 2, 3, 4, 5],
        vec![1.0, 1.0, 2.0, 3.0, 4.0],
        vec![2, 3, 3, 0, 2],
        vec![1; 5],
    )
}

fn close(a: f64, b: f64) {
    assert!((a - b).abs() <= 1e-14 * b.abs().max(1.0), "{a} is not {b}");
}

#[test]
fn greenwood_variance_by_hand() {
    let (id, stop, status, istate) = small();
    let data = SurvData::builder(&stop, &status)
        .id(&id)
        .istate(&istate)
        .n_codes(3)
        .build()
        .unwrap();
    let none = Mat::<f64>::zeros(5, 0);
    let model = fit(&data, none.as_ref(), &MsOptions::default()).unwrap();
    let h = model
        .msfit(
            &[],
            &MsFitOptions {
                vartype: MsVarType::Greenwood,
                ..MsFitOptions::default()
            },
        )
        .unwrap();
    assert_eq!(h.time, vec![1.0, 2.0, 4.0]);
    // Transition 0 is 1 -> 2 and transition 1 is 1 -> 3, at risk 5, 3, 1.
    close(h.cumulative_hazard[0][2], 1.0 / 5.0 + 1.0);
    let var12 = 4.0 / 125.0;
    let var13 = 4.0 / 125.0 + 2.0 / 27.0;
    let cov = -1.0 / 125.0;
    close(h.covariance_of(0, 0).unwrap()[0], var12);
    close(h.covariance_of(0, 0).unwrap()[2], var12);
    close(h.covariance_of(1, 1).unwrap()[0], 4.0 / 125.0);
    close(h.covariance_of(1, 1).unwrap()[2], var13);
    close(h.covariance_of(0, 1).unwrap()[0], cov);
    close(h.covariance_of(0, 1).unwrap()[2], cov);
    // The Aalen form differs: d / n^2 per event time.
    let aalen = model.msfit(&[], &MsFitOptions::default()).unwrap();
    assert!((aalen.covariance_of(0, 0).unwrap()[2] - var12).abs() > 1e-3);
}

#[test]
fn fixed_horizon_is_the_product_from_the_other_end() {
    let (id, stop, status, istate) = small();
    let data = SurvData::builder(&stop, &status)
        .id(&id)
        .istate(&istate)
        .n_codes(3)
        .build()
        .unwrap();
    let none = Mat::<f64>::zeros(5, 0);
    let model = fit(&data, none.as_ref(), &MsOptions::default()).unwrap();
    let h = model.msfit(&[], &MsFitOptions::default()).unwrap();
    let forward = probtrans(&h, 0.0, &ProbTransOptions::default()).unwrap();
    for predt in [4.0, 3.0] {
        let fixed = probtrans(
            &h,
            predt,
            &ProbTransOptions {
                direction: ProbTransDirection::FixedHorizon,
                ..ProbTransOptions::default()
            },
        )
        .unwrap();
        // The first row is P(0, predt), the forward product up to the last
        // event time that is not after predt.
        let last = h.time.iter().filter(|&&t| t <= predt).count();
        assert_eq!(fixed.time[0].to_bits(), 0.0f64.to_bits());
        for from in 0..3 {
            for to in 0..3 {
                close(
                    fixed.probability_of(0, from, to).unwrap(),
                    forward.probability_of(last, from, to).unwrap(),
                );
            }
        }
        // After the last event time it is the identity: the rows are the last
        // event time and `predt`, or `predt` alone when it is an event time.
        let rows = fixed.time.len();
        assert_eq!(fixed.time[rows - 1].to_bits(), predt.to_bits());
        for from in 0..3 {
            for to in 0..3 {
                assert_eq!(
                    fixed.probability_of(rows - 1, from, to).unwrap().to_bits(),
                    f64::from(u8::from(from == to)).to_bits()
                );
            }
        }
    }
}

#[test]
fn profiles_one_by_one_and_together() {
    let (id, stop, status, istate) = small();
    let x = Mat::from_fn(5, 1, |i, _| i as f64 / 2.0);
    let data = SurvData::builder(&stop, &status)
        .id(&id)
        .istate(&istate)
        .n_codes(3)
        .build()
        .unwrap();
    let model = fit(&data, x.as_ref(), &MsOptions::default()).unwrap();
    let a = vec![0.5];
    let b = vec![2.0];
    let by = [a.clone(), b.clone()];
    let opts = MsFitOptions::default();
    let together = model
        .msfit_many(
            &[
                MsProfile::Same(&a),
                MsProfile::ByTransition(&by),
                MsProfile::Same(&b),
            ],
            &opts,
        )
        .unwrap();
    assert_eq!(together[0], model.msfit(&a, &opts).unwrap());
    assert_eq!(together[2], model.msfit(&b, &opts).unwrap());
    // A profile that differs by transition has each transition's own value.
    assert_eq!(
        together[1].cumulative_hazard[0],
        together[0].cumulative_hazard[0]
    );
    assert_eq!(
        together[1].cumulative_hazard[1],
        together[2].cumulative_hazard[1]
    );
}

fn err<T>(r: Result<T, epistat_survival::Error>) -> String {
    r.err().expect("an error").to_string()
}

#[test]
fn input_checks() {
    let (id, stop, status, istate) = small();
    let x = Mat::from_fn(5, 1, |i, _| i as f64 / 2.0);
    let data = SurvData::builder(&stop, &status)
        .id(&id)
        .istate(&istate)
        .n_codes(3)
        .build()
        .unwrap();
    let greenwood = MsFitOptions {
        vartype: MsVarType::Greenwood,
        ..MsFitOptions::default()
    };
    let with = fit(&data, x.as_ref(), &MsOptions::default()).unwrap();
    assert!(err(with.msfit(&[1.0], &greenwood)).contains("without coefficients"));
    // Greenwood needs a baseline of its own for each transition, and without
    // variances there is nothing to check.
    let none = Mat::<f64>::zeros(5, 0);
    let shared = fit(
        &data,
        none.as_ref(),
        &MsOptions {
            baseline: MsBaseline::Shared,
            ..MsOptions::default()
        },
    )
    .unwrap();
    assert!(err(shared.msfit(&[], &greenwood)).contains("baseline hazard of its own"));
    let no_se = MsFitOptions {
        se: false,
        ..greenwood
    };
    assert!(shared.msfit(&[], &no_se).is_ok());
    // A custom term set with no term is a model without coefficients too.
    let terms: [MsTerm; 0] = [];
    let empty = fit(
        &data,
        x.as_ref(),
        &MsOptions {
            terms: MsTerms::Custom(&terms),
            ..MsOptions::default()
        },
    )
    .unwrap();
    assert!(empty.msfit(&[1.0], &greenwood).is_ok());

    // Profiles by transition.
    let one = [vec![1.0]];
    assert!(
        err(with.msfit_profile(&MsProfile::ByTransition(&one), &MsFitOptions::default()))
            .contains("1 vectors for 2 transitions")
    );
    let bad = [vec![1.0], vec![1.0, 2.0]];
    assert!(
        err(with.msfit_profile(&MsProfile::ByTransition(&bad), &MsFitOptions::default()))
            .contains("2 values for 1 covariate columns")
    );
    let nan = [vec![1.0], vec![f64::NAN]];
    assert!(
        err(with.msfit_profile(&MsProfile::ByTransition(&nan), &MsFitOptions::default()))
            .contains("not finite")
    );

    // Probabilities.
    let h = with.msfit(&[1.0], &MsFitOptions::default()).unwrap();
    let fixed = ProbTransOptions {
        direction: ProbTransDirection::FixedHorizon,
        ..ProbTransOptions::default()
    };
    assert!(err(probtrans(&h, 0.5, &fixed)).contains("no event time is at or before"));
    let plain = with
        .msfit(
            &[1.0],
            &MsFitOptions {
                se: false,
                ..MsFitOptions::default()
            },
        )
        .unwrap();
    let covariance = ProbTransOptions {
        covariance: true,
        method: ProbTransMethod::Greenwood,
        ..ProbTransOptions::default()
    };
    assert!(err(probtrans(&plain, 0.0, &covariance)).contains("no covariances"));
    assert!(err(probtrans(&plain, 0.0, &fixed)).contains("no covariances"));
}
