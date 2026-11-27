//! `basehaz`, `predict.coxph` and `survfit.coxph` of survival 3.8-12 against
//! [`epistat_survival::cox::predict`], on the lung data and on simulated
//! counting-process data with strata, weights, an offset and clusters, with
//! Breslow and Efron ties.
//!
//! The functions are given the coefficients and the covariance matrix that
//! `coxph` returned, so what is compared is the prediction and not the fit.
//!
//! # Tolerances
//!
//! With `T` the largest number of event times in a stratum, `p` the number of
//! columns, and `xb` the largest value of `sum_a (|x_a| + |x_a|) |beta_a|` over
//! the rows and the new rows, plus the largest offset:
//!
//! - a risk score `exp(eta)` has a relative error of `(p + 2 + xb) eps`, the
//!   `p` products and sums of `eta` and its magnitude through the
//!   exponential;
//! - a risk-set sum is a compensated sum of positive terms, so it has the
//!   relative error of its terms, and so has a hazard increment. The cumulative
//!   sum over at most `T` times adds `T eps`. The bound for the cumulative
//!   hazard, the baseline, the counts and the expected number is
//!   `8 eps (T + p + 4 + xb)`, relative to the value;
//! - the linear predictor has the absolute error `(p + 2) eps xb`, and the
//!   bound `8 eps (p + 4 + xb)` in units of `max(|value|, 1)`. The risk is
//!   its exponential, with that relative error;
//! - a standard error is the square root of a quadratic form with the
//!   covariance matrix, whose cancellation is at most the condition number
//!   `kappa` of the matrix times `p`, and of a sum over the times, so its
//!   bound is the cumulative one times `p kappa`, relative to the value;
//! - a survival value is in `(0, 1]` and `S` times the cumulative-hazard
//!   error is below the cumulative bound, so that is its absolute bound. A
//!   confidence limit has the error of the standard error times `z se S`,
//!   below the standard-error bound, and the quantile `z` has a few epsilon.
//!
//! The bounds are not changed after a run. The worst deviation in units of
//! its bound is printed for each output.

use epistat_linalg::{Mat, Side};
use epistat_survival::Level;
use epistat_survival::SurvData;
use epistat_survival::cox::predict::{
    BaselineOptions, Centring, ConfType, FittedModel, HazardType, NewData, PredictOptions,
    PredictType, Reference, SurvfitOptions, basehaz, predict, predict_terms, reference_means,
    reference_offset, survfit,
};
use epistat_survival::cox::{CoxOptions, Ties, fit};
use epistat_testkit::{NanRule, load_json, worst_deviation, worst_deviation_defined};
use serde::Deserialize;

#[derive(Deserialize)]
struct Versions {
    r: String,
    survival: String,
    jsonlite: String,
}

#[derive(Deserialize)]
struct Reference_ {
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
    cluster: bool,
    offset: bool,
    robust: bool,
    x: Vec<Vec<f64>>,
    coefficients: Vec<f64>,
    var: Vec<Vec<f64>>,
    naive_var: Vec<Vec<f64>>,
    means: Vec<f64>,
    groups: Vec<Vec<usize>>,
    start: Vec<f64>,
    stop: Vec<f64>,
    event: Vec<i32>,
    stratum: Vec<u32>,
    weight: Vec<f64>,
    offset_values: Vec<f64>,
    new_x: Vec<Vec<f64>>,
    new_stratum: Vec<u32>,
    new_start: Vec<f64>,
    new_stop: Vec<f64>,
    new_offset: Vec<f64>,
    basehaz: RBase,
    predict: std::collections::BTreeMap<String, RPred>,
    expected: RExpected,
    terms: RTerms,
    survfit: Vec<RCurve>,
    survfit_alt: RAlt,
    survfit_means: RMeans,
}

#[derive(Deserialize)]
struct RBase {
    time: Vec<f64>,
    n_event: Vec<f64>,
    n_risk: Vec<f64>,
    strata: Vec<usize>,
    centered: Vec<f64>,
    uncentered: Vec<f64>,
}

#[derive(Deserialize)]
struct RPred {
    fit: Vec<f64>,
    se: Vec<f64>,
    new_fit: Vec<f64>,
    new_se: Vec<f64>,
}

#[derive(Deserialize)]
struct RExpected {
    fit: Vec<f64>,
    se_r: Vec<f64>,
    fit_plain: Vec<f64>,
    se_definition: Vec<f64>,
    fit_definition: Vec<f64>,
    new_fit: Option<Vec<f64>>,
    new_se_r: Option<Vec<f64>>,
    new_se_definition: Option<Vec<f64>>,
    survival: Vec<f64>,
    survival_se: Vec<f64>,
}

#[derive(Deserialize)]
struct RTerms {
    fit: Vec<Vec<f64>>,
    se: Vec<Vec<f64>>,
    constant: f64,
    new_fit: Vec<Vec<f64>>,
    new_se: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct RLimits {
    lower: Vec<Option<f64>>,
    upper: Vec<Option<f64>>,
}

#[derive(Deserialize)]
struct RCurve {
    time: Vec<f64>,
    n_risk: Vec<f64>,
    n_event: Vec<f64>,
    surv: Vec<f64>,
    cumhaz: Vec<f64>,
    std_err: Vec<f64>,
    ci90: RLimits,
    ci: std::collections::BTreeMap<String, RLimits>,
}

#[derive(Deserialize)]
struct RAltCurve {
    time: Vec<f64>,
    surv: Vec<f64>,
    cumhaz: Vec<f64>,
    std_err: Vec<f64>,
    lower: Vec<Option<f64>>,
    upper: Vec<Option<f64>>,
}

#[derive(Deserialize)]
struct RAlt {
    ctype: u8,
    curves: Vec<RAltCurve>,
}

#[derive(Deserialize)]
struct RMeans {
    time: Vec<f64>,
    surv: Vec<f64>,
    cumhaz: Vec<f64>,
    std_err: Vec<f64>,
    lower: Vec<Option<f64>>,
    upper: Vec<Option<f64>>,
    strata: Vec<usize>,
}

fn flatten(m: &[Vec<f64>]) -> Vec<Option<f64>> {
    m.iter().flatten().copied().map(Some).collect()
}

/// The reference curves are R's with `censor = FALSE`, at the event times.
fn at_events() -> SurvfitOptions<'static> {
    SurvfitOptions {
        censor_times: false,
        ..SurvfitOptions::default()
    }
}

fn matrix(rows: &[Vec<f64>]) -> Mat<f64> {
    Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
}

#[derive(Default)]
struct Worst(std::collections::BTreeMap<String, f64>);

impl Worst {
    fn put(&mut self, name: &str, value: f64) {
        let entry = self.0.entry(name.to_string()).or_insert(0.0);
        *entry = entry.max(value);
    }
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one pass over the outputs of each fit"
)]
fn predictions_match_r() {
    let reference: Reference_ = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/cox_predict.json"
    ));
    eprintln!(
        "reference: {}, survival {}, jsonlite {}",
        reference.versions.r, reference.versions.survival, reference.versions.jsonlite
    );
    let eps = f64::EPSILON;
    let mut worst = Worst::default();
    for r in &reference.fits {
        let n = r.stop.len();
        let p = r.coefficients.len();
        let x = matrix(&r.x);
        let new_x = matrix(&r.new_x);
        let status = r.event.clone();
        let mut builder = SurvData::builder(&r.stop, &status);
        if r.entry {
            builder = builder.start(&r.start);
        }
        if r.strata {
            builder = builder.stratum(&r.stratum);
        }
        if r.weights {
            builder = builder.weight(&r.weight);
        }
        if r.offset {
            builder = builder.offset(&r.offset_values);
        }
        let data = builder.build().unwrap();
        let opts = CoxOptions {
            ties: if r.ties == "efron" {
                Ties::Efron
            } else {
                Ties::Breslow
            },
            ..CoxOptions::default()
        };
        let mut model = fit(&data, x.as_ref(), &opts).unwrap();
        // The fit is R's, so what is compared is the prediction.
        model.coefficients = r.coefficients.iter().copied().map(Some).collect();
        model.covariance = flatten(&r.naive_var);
        model.robust_covariance = r.robust.then(|| flatten(&r.var));
        assert_eq!(r.cluster, r.robust);
        let fitted = FittedModel {
            data: &data,
            x: x.as_ref(),
            fit: &model,
            options: &opts,
        };

        // The bounds.
        let t_max = r.basehaz.strata.iter().copied().max().unwrap() as f64;
        let pf = p as f64;
        let mut xb = 0.0f64;
        for a in 0..p {
            let big =
                r.x.iter()
                    .chain(&r.new_x)
                    .map(|row| row[a].abs())
                    .fold(0.0, f64::max);
            xb += 2.0 * big * r.coefficients[a].abs();
        }
        if r.offset {
            xb += r
                .offset_values
                .iter()
                .chain(&r.new_offset)
                .fold(0.0f64, |m, v| m.max(v.abs()));
        }
        let b_sum = 8.0 * eps * (t_max + pf + 4.0 + xb);
        let b_lin = 8.0 * eps * (pf + 4.0 + xb);
        let v = Mat::from_fn(p, p, |i, j| r.var[i][j]);
        let eigen = v.self_adjoint_eigenvalues(Side::Lower).unwrap();
        let kappa = eigen[p - 1] / eigen[0];
        let b_se = b_sum * pf * kappa;
        let name = &r.name;
        let mut note = |what: &str, value: f64| {
            assert!(value <= 1.0, "{name}: {what} is {value} bounds");
            worst.put(what, value);
        };

        // The means.
        let means = reference_means(&fitted, Centring::Coxph).unwrap();
        note(
            "means",
            worst_deviation(
                &means,
                &r.means,
                8.0 * eps * (n as f64),
                1.0,
                NanRule::Finite,
            ),
        );

        // basehaz.
        let baseline = basehaz(&fitted, &BaselineOptions::default()).unwrap();
        let counts: Vec<usize> = baseline.strata.iter().map(|s| s.time.len()).collect();
        assert_eq!(counts, r.basehaz.strata, "{name}: event times per stratum");
        let cat = |f: &dyn Fn(&epistat_survival::cox::predict::StratumBaseline) -> Vec<f64>| {
            baseline.strata.iter().flat_map(f).collect::<Vec<f64>>()
        };
        assert_eq!(cat(&|s| s.time.clone()), r.basehaz.time, "{name}: times");
        note(
            "basehaz n_event",
            worst_deviation(
                &cat(&|s| s.n_events.clone()),
                &r.basehaz.n_event,
                b_sum,
                0.0,
                NanRule::Finite,
            ),
        );
        note(
            "basehaz n_risk",
            worst_deviation(
                &cat(&|s| s.n_risk.clone()),
                &r.basehaz.n_risk,
                8.0 * eps * (n as f64),
                0.0,
                NanRule::Finite,
            ),
        );
        note(
            "basehaz centered",
            worst_deviation(
                &cat(&|s| s.cumulative_hazard.clone()),
                &r.basehaz.centered,
                b_sum,
                0.0,
                NanRule::Finite,
            ),
        );
        let uncentered: Vec<f64> = cat(&|s| s.cumulative_hazard.clone())
            .iter()
            .map(|h| h * baseline.uncentring_factor)
            .collect();
        note(
            "basehaz uncentered",
            worst_deviation(
                &uncentered,
                &r.basehaz.uncentered,
                b_sum,
                0.0,
                NanRule::Finite,
            ),
        );

        // predict, lp and risk.
        let refs = [
            ("strata", Reference::Strata),
            ("sample", Reference::Sample),
            ("zero", Reference::Zero),
        ];
        let mut new = NewData::new(new_x.as_ref());
        new.stratum = r.strata.then_some(r.new_stratum.as_slice());
        new.offset = r.offset.then_some(r.new_offset.as_slice());
        for (label, reference_kind) in refs {
            let opts_p = PredictOptions {
                reference: Some(reference_kind),
                se_fit: true,
                ..PredictOptions::default()
            };
            for (kind, key) in [
                (PredictType::LinearPredictor, "lp"),
                (PredictType::Risk, "risk"),
            ] {
                let rp = &r.predict[&format!("{key}_{label}")];
                let on = predict(&fitted, kind, None, &opts_p).unwrap();
                let off = predict(&fitted, kind, Some(&new), &opts_p).unwrap();
                let (bound, floor) = if key == "lp" {
                    (b_lin, 1.0)
                } else {
                    (b_lin, 0.0)
                };
                note(
                    &format!("{key} {label}"),
                    worst_deviation(&on.fit, &rp.fit, bound, floor, NanRule::Finite).max(
                        worst_deviation(&off.fit, &rp.new_fit, bound, floor, NanRule::Finite),
                    ),
                );
                let lp_se = &r.predict[&format!("lp_{label}")];
                if key == "lp" {
                    note(
                        &format!("lp se {label}"),
                        worst_deviation(
                            on.se.as_ref().unwrap(),
                            &rp.se,
                            b_se,
                            0.0,
                            NanRule::Finite,
                        )
                        .max(worst_deviation(
                            off.se.as_ref().unwrap(),
                            &rp.new_se,
                            b_se,
                            0.0,
                            NanRule::Finite,
                        )),
                    );
                } else {
                    // survival returns se(lp) sqrt(risk); the delta method
                    // is se(lp) risk, which is what is computed.
                    let quirk: Vec<f64> = lp_se
                        .se
                        .iter()
                        .zip(&rp.fit)
                        .map(|(s, risk)| s * risk.sqrt())
                        .collect();
                    note(
                        "risk se, survival's own form",
                        worst_deviation(&quirk, &rp.se, b_lin, 0.0, NanRule::Finite),
                    );
                    let delta: Vec<f64> = lp_se
                        .se
                        .iter()
                        .zip(&rp.fit)
                        .map(|(s, risk)| s * risk)
                        .collect();
                    let delta_new: Vec<f64> = lp_se
                        .new_se
                        .iter()
                        .zip(&rp.new_fit)
                        .map(|(s, risk)| s * risk)
                        .collect();
                    note(
                        &format!("risk se {label}"),
                        worst_deviation(
                            on.se.as_ref().unwrap(),
                            &delta,
                            b_se,
                            0.0,
                            NanRule::Finite,
                        )
                        .max(worst_deviation(
                            off.se.as_ref().unwrap(),
                            &delta_new,
                            b_se,
                            0.0,
                            NanRule::Finite,
                        )),
                    );
                }
            }
        }

        // expected and survival.
        let opts_e = PredictOptions {
            se_fit: true,
            ..PredictOptions::default()
        };
        let ex = predict(&fitted, PredictType::Expected, None, &opts_e).unwrap();
        assert_eq!(
            r.expected.fit_plain, r.expected.fit,
            "{name}: survival's expected number without se.fit"
        );
        // survival's expected number of the fitted data is the martingale
        // form, which differs from the cumulative hazard times the risk for an
        // event at a time shared with other events under Efron ties.
        note(
            "expected",
            worst_deviation(
                &ex.fit,
                &r.expected.fit_definition,
                b_sum,
                0.0,
                NanRule::Finite,
            ),
        );
        let mut tied = std::collections::BTreeMap::<(u32, u64), usize>::new();
        for i in 0..n {
            if r.event[i] == 1 {
                let s = if r.strata { r.stratum[i] } else { 0 };
                *tied.entry((s, r.stop[i].to_bits())).or_default() += 1;
            }
        }
        let unshared: Vec<bool> = (0..n)
            .map(|i| {
                let s = if r.strata { r.stratum[i] } else { 0 };
                !(r.event[i] == 1 && tied[&(s, r.stop[i].to_bits())] > 1 && r.ties == "efron")
            })
            .collect();
        let pick = |values: &[f64]| -> Vec<f64> {
            values
                .iter()
                .zip(&unshared)
                .filter(|(_, u)| **u)
                .map(|(v, _)| *v)
                .collect()
        };
        let compared = unshared.iter().filter(|&&u| u).count();
        let (got, want) = (pick(&ex.fit), pick(&r.expected.fit));
        assert!(
            compared >= 30,
            "{name}: too few rows compared with survival"
        );
        note(
            "expected (survival's fitted form)",
            worst_deviation(&got, &want, b_sum, 0.0, NanRule::Finite),
        );
        let differs = (0..n).any(|i| (ex.fit[i] - r.expected.fit[i]).abs() > 1e-6);
        assert_eq!(
            differs,
            r.ties == "efron" && tied.values().any(|&c| c > 1),
            "{name}: the martingale form of survival should differ exactly with Efron ties"
        );
        // The scale of the standard error of an expected number is the one of
        // its parts: the cumulative variance and `g`.
        note(
            "expected se (definition)",
            worst_deviation(
                ex.se.as_ref().unwrap(),
                &r.expected.se_definition,
                b_se,
                0.0,
                NanRule::Finite,
            ),
        );
        if r.entry {
            let gap = ex
                .se
                .as_ref()
                .unwrap()
                .iter()
                .zip(&r.expected.se_r)
                .map(|(a, b)| (a - b).abs() / a)
                .fold(0.0, f64::max);
            assert!(
                gap > 1e-3,
                "{name}: survival's se of expected with entry times should differ"
            );
        } else {
            note(
                "expected se (survival)",
                worst_deviation(
                    ex.se.as_ref().unwrap(),
                    &r.expected.se_r,
                    b_se,
                    0.0,
                    NanRule::Finite,
                ),
            );
        }
        let sv = predict(&fitted, PredictType::Survival, None, &opts_e).unwrap();
        // survival's survival of the fitted data is exp(-expected) of its
        // martingale form, and its se that of the cumulative-hazard form times
        // that survival; the definition form is compared on every row and
        // survival's own on the rows where the two forms agree.
        let from_definition: Vec<f64> = r
            .expected
            .fit_definition
            .iter()
            .map(|e| (-e).exp())
            .collect();
        let se_from_definition: Vec<f64> = r
            .expected
            .se_definition
            .iter()
            .zip(&from_definition)
            .map(|(s, v)| s * v)
            .collect();
        note(
            "survival",
            worst_deviation(&sv.fit, &from_definition, b_sum, 1.0, NanRule::Finite),
        );
        note(
            "survival se",
            worst_deviation(
                sv.se.as_ref().unwrap(),
                &se_from_definition,
                b_se,
                0.0,
                NanRule::Finite,
            ),
        );
        if !r.entry {
            note(
                "survival (survival's fitted form)",
                worst_deviation(
                    &pick(&sv.fit),
                    &pick(&r.expected.survival),
                    b_sum,
                    1.0,
                    NanRule::Finite,
                ),
            );
            note(
                "survival se (survival's fitted form)",
                worst_deviation(
                    &pick(sv.se.as_ref().unwrap()),
                    &pick(&r.expected.survival_se),
                    b_se,
                    0.0,
                    NanRule::Finite,
                ),
            );
        }
        let mut new_e = new;
        if !r.offset {
            new_e.stop = Some(&r.new_stop);
            new_e.start = r.entry.then_some(r.new_start.as_slice());
            let en = predict(&fitted, PredictType::Expected, Some(&new_e), &opts_e).unwrap();
            note(
                "expected new",
                worst_deviation(
                    &en.fit,
                    r.expected.new_fit.as_ref().unwrap(),
                    b_sum,
                    0.0,
                    NanRule::Finite,
                ),
            );
            note(
                "expected new se (definition)",
                worst_deviation(
                    en.se.as_ref().unwrap(),
                    r.expected.new_se_definition.as_ref().unwrap(),
                    b_se,
                    0.0,
                    NanRule::Finite,
                ),
            );
            if !r.entry {
                note(
                    "expected new se (survival)",
                    worst_deviation(
                        en.se.as_ref().unwrap(),
                        r.expected.new_se_r.as_ref().unwrap(),
                        b_se,
                        0.0,
                        NanRule::Finite,
                    ),
                );
            }
        }

        // terms.
        let groups: Vec<&[usize]> = r.groups.iter().map(Vec::as_slice).collect();
        let tm = predict_terms(&fitted, &groups, None, &opts_e).unwrap();
        let tn = predict_terms(&fitted, &groups, Some(&new), &opts_e).unwrap();
        let flat = |m: &[Vec<f64>]| m.iter().flatten().copied().collect::<Vec<f64>>();
        note(
            "terms",
            worst_deviation(&tm.fit, &flat(&r.terms.fit), b_lin, 1.0, NanRule::Finite).max(
                worst_deviation(
                    &tn.fit,
                    &flat(&r.terms.new_fit),
                    b_lin,
                    1.0,
                    NanRule::Finite,
                ),
            ),
        );
        note(
            "terms se",
            worst_deviation(
                tm.se.as_ref().unwrap(),
                &flat(&r.terms.se),
                b_se,
                0.0,
                NanRule::Finite,
            )
            .max(worst_deviation(
                tn.se.as_ref().unwrap(),
                &flat(&r.terms.new_se),
                b_se,
                0.0,
                NanRule::Finite,
            )),
        );
        note(
            "terms constant",
            worst_deviation(
                &[tm.constant],
                &[r.terms.constant],
                b_lin,
                1.0,
                NanRule::Finite,
            ),
        );

        // survfit.
        let kinds = [
            ("log", ConfType::Log),
            ("log-log", ConfType::LogLog),
            ("plain", ConfType::Plain),
            ("logit", ConfType::Logit),
            ("arcsin", ConfType::Arcsin),
        ];
        let sf_opts = at_events();
        let curves = survfit(&fitted, &new, &sf_opts).unwrap();
        for (i, rc) in r.survfit.iter().enumerate() {
            let c = &curves.curves[i];
            let times = &curves.strata[c.stratum];
            assert_eq!(times.time, rc.time, "{name}: curve {i} times");
            note(
                "curve n_risk",
                worst_deviation(
                    &times.n_risk,
                    &rc.n_risk,
                    8.0 * eps * n as f64,
                    0.0,
                    NanRule::Finite,
                ),
            );
            note(
                "curve n_event",
                worst_deviation(&times.n_event, &rc.n_event, b_sum, 0.0, NanRule::Finite),
            );
            note(
                "curve cumhaz",
                worst_deviation(
                    &c.cumulative_hazard,
                    &rc.cumhaz,
                    b_sum,
                    0.0,
                    NanRule::Finite,
                ),
            );
            note(
                "curve surv",
                worst_deviation(&c.survival, &rc.surv, b_sum, 1.0, NanRule::Finite),
            );
            note(
                "curve std_err",
                worst_deviation(
                    c.se.as_ref().unwrap(),
                    &rc.std_err,
                    b_se,
                    0.0,
                    NanRule::Finite,
                ),
            );
            for (key, kind) in kinds {
                let o = SurvfitOptions {
                    conf_type: kind,
                    ..at_events()
                };
                let cc = survfit(&fitted, &new, &o).unwrap();
                let rl = &rc.ci[key];
                note(
                    &format!("limits {key}"),
                    worst_deviation_defined(cc.curves[i].lower.as_ref().unwrap(), &rl.lower, b_se)
                        .max(worst_deviation_defined(
                            cc.curves[i].upper.as_ref().unwrap(),
                            &rl.upper,
                            b_se,
                        )),
                );
            }
            let o90 = SurvfitOptions {
                level: Level::confidence(0.9).unwrap(),
                ..at_events()
            };
            let c90 = survfit(&fitted, &new, &o90).unwrap();
            note(
                "limits level 0.9",
                worst_deviation_defined(
                    c90.curves[i].lower.as_ref().unwrap(),
                    &rc.ci90.lower,
                    b_se,
                )
                .max(worst_deviation_defined(
                    c90.curves[i].upper.as_ref().unwrap(),
                    &rc.ci90.upper,
                    b_se,
                )),
            );
        }
        // The other hazard type.
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
                ..at_events()
            },
        )
        .unwrap();
        for (i, rc) in r.survfit_alt.curves.iter().enumerate() {
            let c = &alt.curves[i];
            assert_eq!(alt.strata[c.stratum].time, rc.time);
            note(
                "alt cumhaz",
                worst_deviation(
                    &c.cumulative_hazard,
                    &rc.cumhaz,
                    b_sum,
                    0.0,
                    NanRule::Finite,
                ),
            );
            note(
                "alt surv",
                worst_deviation(&c.survival, &rc.surv, b_sum, 1.0, NanRule::Finite),
            );
            note(
                "alt std_err",
                worst_deviation(
                    c.se.as_ref().unwrap(),
                    &rc.std_err,
                    b_se,
                    0.0,
                    NanRule::Finite,
                ),
            );
            note(
                "alt limits",
                worst_deviation_defined(c.lower.as_ref().unwrap(), &rc.lower, b_se).max(
                    worst_deviation_defined(c.upper.as_ref().unwrap(), &rc.upper, b_se),
                ),
            );
        }
        // The curve at the means, in every stratum.
        let m_rows = r.basehaz.strata.len();
        let at_means = Mat::from_fn(m_rows, p, |_, j| r.means[j]);
        let levels: Vec<u32> = if r.strata {
            let mut l = r.stratum.clone();
            l.sort_unstable();
            l.dedup();
            l
        } else {
            vec![0]
        };
        let mut nm = NewData::new(at_means.as_ref());
        nm.stratum = r.strata.then_some(levels.as_slice());
        let case_weight: Vec<f64> = if r.weights {
            r.weight.clone()
        } else {
            vec![1.0; n]
        };
        let mean_offset = [r
            .offset_values
            .iter()
            .zip(&case_weight)
            .map(|(o, w)| o * w)
            .sum::<f64>()
            / case_weight.iter().sum::<f64>()];
        let mean_offsets = vec![mean_offset[0]; m_rows];
        nm.offset = r.offset.then_some(mean_offsets.as_slice());
        if r.offset {
            let from_crate = reference_offset(&fitted).unwrap();
            assert!(
                (from_crate - mean_offset[0]).abs() < 8.0 * eps * mean_offset[0].abs().max(1.0)
            );
        }
        let cm = survfit(&fitted, &nm, &sf_opts).unwrap();
        let mut offset = 0;
        for (k, c) in cm.curves.iter().enumerate() {
            let len = r.survfit_means.strata[k];
            let sl = offset..offset + len;
            offset += len;
            assert_eq!(cm.strata[c.stratum].time, r.survfit_means.time[sl.clone()]);
            note(
                "means curve cumhaz",
                worst_deviation(
                    &c.cumulative_hazard,
                    &r.survfit_means.cumhaz[sl.clone()],
                    b_sum,
                    0.0,
                    NanRule::Finite,
                ),
            );
            note(
                "means curve surv",
                worst_deviation(
                    &c.survival,
                    &r.survfit_means.surv[sl.clone()],
                    b_sum,
                    1.0,
                    NanRule::Finite,
                ),
            );
            note(
                "means curve std_err",
                worst_deviation(
                    c.se.as_ref().unwrap(),
                    &r.survfit_means.std_err[sl.clone()],
                    b_se,
                    0.0,
                    NanRule::Finite,
                ),
            );
            note(
                "means curve limits",
                worst_deviation_defined(
                    c.lower.as_ref().unwrap(),
                    &r.survfit_means.lower[sl.clone()],
                    b_se,
                )
                .max(worst_deviation_defined(
                    c.upper.as_ref().unwrap(),
                    &r.survfit_means.upper[sl.clone()],
                    b_se,
                )),
            );
        }
    }
    for (what, value) in &worst.0 {
        eprintln!("worst deviation, {what}: {value:.3} of its bound");
    }
}
