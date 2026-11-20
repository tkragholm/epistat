//! Cox models without covariates, `coxph(y ~ 1)` and `coxph(y ~ offset(o))`,
//! of survival 3.8-12 against [`epistat_survival::cox::fit`] with a design of
//! no columns, and its `basehaz`, `predict` and `survfit`.
//!
//! # Tolerances
//!
//! With `T` the largest number of event times in a stratum, `n` the number of
//! rows and `xb` the largest absolute offset:
//!
//! - a log partial likelihood is a sum of at most `n` terms of size
//!   `log(n) + xb`, so its error is `8 eps n (log(n) + xb)`, in units of
//!   `max(|value|, 1)`;
//! - a cumulative hazard, a risk and an expected number have the relative
//!   error `8 eps (T + 4 + xb)`, and a linear predictor the same in units of
//!   `max(|value|, 1)`;
//! - a standard error is a square root of a sum over at most `T` times, and a
//!   survival value or a limit has that bound as an absolute one.
//!
//! The bounds are not changed after a run. The worst deviation in units of its
//! bound is printed for each output.

use epistat_survival::SurvData;
use epistat_survival::cox::predict::{
    BaselineOptions, FittedModel, HazardType, NewData, PredictOptions, PredictType, SurvfitOptions,
    basehaz, predict, survfit,
};
use epistat_survival::cox::{CoxOptions, Ties, fit};
use epistat_testkit::{NanRule, load_json, worst_deviation, worst_deviation_defined};
use faer::Mat;
use serde::Deserialize;

#[derive(Deserialize)]
struct Versions {
    r: String,
    survival: String,
    jsonlite: String,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    fits: Vec<RFit>,
}

#[derive(Deserialize)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "one flag per option of the R call"
)]
struct RFit {
    name: String,
    ties: String,
    entry: bool,
    strata: bool,
    weights: bool,
    offset: bool,
    cluster: bool,
    loglik: f64,
    n: usize,
    nevent: f64,
    basehaz: RBase,
    lp: Vec<f64>,
    risk: Vec<f64>,
    expected: Vec<f64>,
    survfit: RCurve,
    survfit_alt: RAlt,
    start: Vec<f64>,
    stop: Vec<f64>,
    event: Vec<i32>,
    stratum: Vec<u32>,
    weight: Vec<f64>,
    offset_values: Vec<f64>,
    id: Vec<u32>,
}

#[derive(Deserialize)]
struct RBase {
    time: Vec<f64>,
    hazard: Vec<f64>,
}

#[derive(Deserialize)]
struct RCurve {
    time: Vec<f64>,
    n_risk: Vec<f64>,
    n_event: Vec<f64>,
    surv: Vec<f64>,
    cumhaz: Vec<f64>,
    std_err: Vec<f64>,
    lower: Vec<Option<f64>>,
    upper: Vec<Option<f64>>,
    strata: Vec<usize>,
}

#[derive(Deserialize)]
struct RAlt {
    ctype: u8,
    surv: Vec<f64>,
    cumhaz: Vec<f64>,
    std_err: Vec<f64>,
    lower: Vec<Option<f64>>,
    upper: Vec<Option<f64>>,
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one pass over the outputs of each fit"
)]
fn null_cox_matches_r() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/cox_null.json"
    ));
    eprintln!(
        "reference: {}, survival {}, jsonlite {}",
        reference.versions.r, reference.versions.survival, reference.versions.jsonlite
    );
    let eps = f64::EPSILON;
    let mut worst = std::collections::BTreeMap::<String, f64>::new();
    for r in &reference.fits {
        let name = &r.name;
        let n = r.stop.len();
        let x = Mat::<f64>::zeros(n, 0);
        let mut builder = SurvData::builder(&r.stop, &r.event);
        if r.entry {
            builder = builder.start(&r.start);
        }
        if r.strata {
            builder = builder.stratum(&r.stratum);
        }
        if r.weights {
            builder = builder.weight(&r.weight);
        }
        if r.cluster {
            builder = builder.cluster(&r.id);
        }
        let data = builder.build().unwrap();
        let opts = CoxOptions {
            ties: if r.ties == "efron" {
                Ties::Efron
            } else {
                Ties::Breslow
            },
            offset: r.offset.then_some(r.offset_values.as_slice()),
            ..CoxOptions::default()
        };
        let model = fit(&data, x.as_ref(), &opts).unwrap();
        let xb = if r.offset {
            r.offset_values.iter().fold(0.0f64, |m, v| m.max(v.abs()))
        } else {
            0.0
        };
        let t_max = r.survfit.strata.iter().copied().max().unwrap() as f64;
        let b_ll = 8.0 * eps * (n as f64) * ((n as f64).ln() + xb);
        let b_sum = 8.0 * eps * (t_max + 4.0 + xb);
        let mut note = |what: &str, value: f64| {
            assert!(value <= 1.0, "{name}: {what} is {value} bounds");
            let e = worst.entry(what.to_string()).or_insert(0.0);
            *e = e.max(value);
        };

        // The fit.
        assert_eq!(model.coefficients.len() + model.standard_errors.len(), 0);
        assert_eq!(model.covariance.len(), 0);
        assert_eq!(model.robust_covariance.is_some(), r.cluster, "{name}");
        assert_eq!(model.rank, 0);
        assert_eq!(model.iterations, 0);
        assert!(model.converged);
        assert!(model.tests.wald.is_none() && model.tests.score.is_none());
        assert!(model.tests.likelihood_ratio.is_none() && model.tests.robust_wald.is_none());
        assert_eq!(model.n_rows, r.n, "{name}: n");
        // R counts the events, the fit sums their case weights.
        let counted = r.event.iter().filter(|&&e| e == 1).count() as f64;
        assert!((r.nevent - counted).abs() < 0.5, "{name}: nevent of R");
        let summed: f64 = (0..n)
            .filter(|&i| r.event[i] == 1)
            .map(|i| if r.weights { r.weight[i] } else { 1.0 })
            .sum();
        assert!(
            (model.n_events - summed).abs() <= 8.0 * eps * summed,
            "{name}: events"
        );
        let ll = model.log_partial_likelihood.unwrap();
        assert_eq!(model.null_log_partial_likelihood, Some(ll));
        note(
            "loglik",
            worst_deviation(&[ll], &[r.loglik], b_ll, 1.0, NanRule::Finite),
        );

        let fitted = FittedModel {
            data: &data,
            x: x.as_ref(),
            fit: &model,
            options: &opts,
        };

        // basehaz at the means, which for a model without covariates is the
        // mean offset.
        let baseline = basehaz(&fitted, &BaselineOptions::default()).unwrap();
        let counts: Vec<usize> = baseline.strata.iter().map(|s| s.time.len()).collect();
        assert_eq!(counts, r.survfit.strata, "{name}: event times per stratum");
        let cat = |f: &dyn Fn(&epistat_survival::cox::predict::StratumBaseline) -> Vec<f64>| {
            baseline.strata.iter().flat_map(f).collect::<Vec<f64>>()
        };
        assert_eq!(cat(&|s| s.time.clone()), r.basehaz.time, "{name}: times");
        note(
            "basehaz",
            worst_deviation(
                &cat(&|s| s.cumulative_hazard.clone()),
                &r.basehaz.hazard,
                b_sum,
                0.0,
                NanRule::Finite,
            ),
        );

        // predict. With an offset the reference of the two differs by a
        // constant, so the differences from the first row are compared.
        let se_opts = PredictOptions {
            se_fit: true,
            ..PredictOptions::default()
        };
        let lp = predict(
            &fitted,
            PredictType::LinearPredictor,
            None,
            &PredictOptions::default(),
        )
        .unwrap();
        let risk = predict(&fitted, PredictType::Risk, None, &PredictOptions::default()).unwrap();
        if r.offset {
            let rel = |v: &[f64]| v.iter().map(|x| x - v[0]).collect::<Vec<f64>>();
            note(
                "lp",
                worst_deviation(
                    &rel(&lp.fit),
                    &rel(&r.lp),
                    b_sum * 2.0,
                    1.0,
                    NanRule::Finite,
                ),
            );
            let ratio = |v: &[f64]| v.iter().map(|x| x / v[0]).collect::<Vec<f64>>();
            note(
                "risk",
                worst_deviation(
                    &ratio(&risk.fit),
                    &ratio(&r.risk),
                    b_sum * 2.0,
                    0.0,
                    NanRule::Finite,
                ),
            );
        } else {
            note(
                "lp",
                worst_deviation(&lp.fit, &r.lp, b_sum, 1.0, NanRule::Finite),
            );
            note(
                "risk",
                worst_deviation(&risk.fit, &r.risk, b_sum, 0.0, NanRule::Finite),
            );
        }
        for kind in [PredictType::LinearPredictor, PredictType::Risk] {
            let with_se = predict(&fitted, kind, None, &se_opts).unwrap();
            assert!(
                with_se.se.as_ref().unwrap().iter().all(|&s| s == 0.0),
                "{name}: no coefficients, no standard error"
            );
        }
        let ex = predict(&fitted, PredictType::Expected, None, &se_opts).unwrap();
        // Rows with an event at a time shared with other events under Efron
        // ties have R's martingale form, which differs by definition.
        let mut tied = std::collections::BTreeMap::<(u32, u64), usize>::new();
        for i in 0..n {
            if r.event[i] == 1 {
                let s = if r.strata { r.stratum[i] } else { 0 };
                *tied.entry((s, r.stop[i].to_bits())).or_default() += 1;
            }
        }
        let keep: Vec<usize> = (0..n)
            .filter(|&i| {
                let s = if r.strata { r.stratum[i] } else { 0 };
                !(r.event[i] == 1 && tied[&(s, r.stop[i].to_bits())] > 1 && r.ties == "efron")
            })
            .collect();
        assert!(keep.len() >= 30, "{name}: too few rows compared");
        let got: Vec<f64> = keep.iter().map(|&i| ex.fit[i]).collect();
        let want: Vec<f64> = keep.iter().map(|&i| r.expected[i]).collect();
        note(
            "expected",
            worst_deviation(&got, &want, b_sum, 0.0, NanRule::Finite),
        );
        let se = ex.se.as_ref().unwrap();
        assert!(se.iter().all(|s| s.is_finite() && *s >= 0.0));
        if !r.entry && !r.strata && !r.offset {
            // R fails on the standard error of an expected number without
            // coefficients. With every risk 1 it is the standard error of the
            // cumulative hazard at the exit time, which survfit gives.
            let rs = &r.survfit;
            let want: Vec<f64> = r
                .stop
                .iter()
                .map(|&t| {
                    let k = rs.time.partition_point(|&u| u <= t);
                    if k == 0 { 0.0 } else { rs.std_err[k - 1] }
                })
                .collect();
            note(
                "expected se",
                worst_deviation(se, &want, b_sum, 0.0, NanRule::Finite),
            );
        }

        // survfit at the reference, one row of new data per stratum.
        let levels: Vec<u32> = if r.strata {
            let mut l = r.stratum.clone();
            l.sort_unstable();
            l.dedup();
            l
        } else {
            vec![0]
        };
        let m_rows = levels.len();
        let weight: Vec<f64> = if r.weights {
            r.weight.clone()
        } else {
            vec![1.0; n]
        };
        let mean_offset = r
            .offset_values
            .iter()
            .zip(&weight)
            .map(|(o, w)| o * w)
            .sum::<f64>()
            / weight.iter().sum::<f64>();
        let offsets = vec![mean_offset; m_rows];
        let new_x = Mat::<f64>::zeros(m_rows, 0);
        let mut new = NewData::new(new_x.as_ref());
        new.stratum = r.strata.then_some(levels.as_slice());
        new.offset = r.offset.then_some(offsets.as_slice());
        let at_events = SurvfitOptions {
            censor_times: false,
            ..SurvfitOptions::default()
        };
        let curves = survfit(&fitted, &new, &at_events).unwrap();
        let alt_kind = if r.survfit_alt.ctype == 1 {
            HazardType::Breslow
        } else {
            HazardType::Efron
        };
        let alt = survfit(
            &fitted,
            &new,
            &SurvfitOptions {
                hazard: Some(alt_kind),
                ..at_events
            },
        )
        .unwrap();
        let mut at = 0;
        for (k, c) in curves.curves.iter().enumerate() {
            let len = r.survfit.strata[k];
            let sl = at..at + len;
            at += len;
            let times = &curves.strata[c.stratum];
            assert_eq!(times.time, r.survfit.time[sl.clone()], "{name}: times");
            let rs = &r.survfit;
            note(
                "n_risk",
                worst_deviation(
                    &times.n_risk,
                    &rs.n_risk[sl.clone()],
                    8.0 * eps * n as f64,
                    0.0,
                    NanRule::Finite,
                ),
            );
            note(
                "n_event",
                worst_deviation(
                    &times.n_event,
                    &rs.n_event[sl.clone()],
                    b_sum,
                    0.0,
                    NanRule::Finite,
                ),
            );
            note(
                "cumhaz",
                worst_deviation(
                    &c.cumulative_hazard,
                    &rs.cumhaz[sl.clone()],
                    b_sum,
                    0.0,
                    NanRule::Finite,
                ),
            );
            note(
                "surv",
                worst_deviation(
                    &c.survival,
                    &rs.surv[sl.clone()],
                    b_sum,
                    1.0,
                    NanRule::Finite,
                ),
            );
            note(
                "std_err",
                worst_deviation(
                    c.std_err.as_ref().unwrap(),
                    &rs.std_err[sl.clone()],
                    b_sum,
                    0.0,
                    NanRule::Finite,
                ),
            );
            note(
                "limits",
                worst_deviation_defined(c.lower.as_ref().unwrap(), &rs.lower[sl.clone()], b_sum)
                    .max(worst_deviation_defined(
                        c.upper.as_ref().unwrap(),
                        &rs.upper[sl.clone()],
                        b_sum,
                    )),
            );
            let (a, ra) = (&alt.curves[k], &r.survfit_alt);
            note(
                "alt cumhaz",
                worst_deviation(
                    &a.cumulative_hazard,
                    &ra.cumhaz[sl.clone()],
                    b_sum,
                    0.0,
                    NanRule::Finite,
                ),
            );
            note(
                "alt surv",
                worst_deviation(
                    &a.survival,
                    &ra.surv[sl.clone()],
                    b_sum,
                    1.0,
                    NanRule::Finite,
                ),
            );
            note(
                "alt std_err",
                worst_deviation(
                    a.std_err.as_ref().unwrap(),
                    &ra.std_err[sl.clone()],
                    b_sum,
                    0.0,
                    NanRule::Finite,
                ),
            );
            note(
                "alt limits",
                worst_deviation_defined(a.lower.as_ref().unwrap(), &ra.lower[sl.clone()], b_sum)
                    .max(worst_deviation_defined(
                        a.upper.as_ref().unwrap(),
                        &ra.upper[sl.clone()],
                        b_sum,
                    )),
            );
        }
    }
    for (what, value) in &worst {
        eprintln!("worst {what}: {value:.3} bounds");
    }
}
