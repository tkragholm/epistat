//! Every public entry point refuses an empty column, NaN, an infinity, a column
//! one row short and an unsorted column with an `Input` error, through
//! `epistat_testkit::assert_rejects_bad_input`. A kind of input an entry point
//! has no argument for, such as an unsorted column where the order of the rows
//! does not matter, is not tried.

use epistat_core::Bootstrap;
use epistat_core::Classify;
use epistat_core::Level;
use epistat_linalg::Mat;
use epistat_survival::aalen_johansen::{self, AalenJohansenOptions, AalenJohansenVarianceOptions};
use epistat_survival::absrisk::{CauseSpecificCox, CscOptions, NewRows, RiskOptions};
use epistat_survival::aipw::{AipwCensoring, AipwSurvOptions, AipwSurvSpec, aipw_cox};
use epistat_survival::casecohort::{self, CaseCohortMethod, CaseCohortOptions};
use epistat_survival::clogit::{ClogitOptions, clogit};
use epistat_survival::concordance::{self, ConcordanceOptions};
use epistat_survival::condense::{self, CondenseOptions};
use epistat_survival::cox::{self, CoxOptions};
use epistat_survival::crossover::{self, ExposureIntervals, Measure, Observation, Referent};
use epistat_survival::crossover::{Candidates, CaseEvents, ControlRule, sample_controls};
use epistat_survival::fine_gray::{self, FineGrayOptions};
use epistat_survival::flexpar::{self, FlexParOptions, FlexparScale};
use epistat_survival::frailty::{self, FrailtyOptions};
use epistat_survival::informative::{informative_sets, informative_strata};
use epistat_survival::lexis::{
    self, LexisOptions, TimeScale,
    cells::{self, CellKey},
    rates::{self, RateOptions},
};
use epistat_survival::markov::{self, MarkovOptions};
use epistat_survival::metrics::rttright::{self, RttrightOptions};
use epistat_survival::metrics::{
    brier::{self, BrierOptions},
    calibration::{self, CalibrationOptions, Groups},
    recalibration::{self, Bandwidth, RecalibrationOptions, RecalibrationScale, SmoothOptions},
    royston::{self, RoystonOptions},
    score::{self, ScoreOptions},
};
use epistat_survival::msreg::{self, MsOptions};
use epistat_survival::multistate::{self, MultistateOptions, OccupationVarianceOptions};
use epistat_survival::ncc::{self, NccDesign};
use epistat_survival::pseudo::{
    self, PseudoIpcwOptions, PseudoOptions, PseudoRegressionOptions, Target,
};
use epistat_survival::pseudo::{StateOptions, StateTarget};
use epistat_survival::pyears::{self, Dimension, PyearsOptions};
use epistat_survival::ratetable::{LongDim, RateDim, RateTable};
use epistat_survival::relsurv::{
    self, ExcessOptions, ExpectedHazard, ExpectedMethod, NetMethod, NetOptions, NetTransform,
};
use epistat_survival::rmst::{self, RmstOptions, Tau};
use epistat_survival::sccs::{self, PieceSpec, SccsData, SccsOptions};
use epistat_survival::split::{self, SplitOptions};
use epistat_survival::standardise::{
    Counterfactual, ExposureLevel, StandardiseOptions, standardise_cox,
    yates::{self, InverseLink, YatesLevel, YatesOptions, YatesTest},
};
use epistat_survival::surv2::{self, Repeated, StatusKind, Timeline, TimelineOptions};
use epistat_survival::survdiff::{self, SurvdiffOptions};
use epistat_survival::survfit::{self, SurvfitOptions};
use epistat_survival::survreg::{self, SurvregOptions};
use epistat_survival::tmerge::{Observations, Spans, Tmerge, TmergeOptions};
use epistat_survival::weighting::clone::{self, Adherence, CloneInput, Strategy, Trigger};
use epistat_survival::weighting::pooled::{self, PooledEvent, PooledOptions};
use epistat_survival::weighting::trials::{self, PersonPeriod, WeightModel, WeightRows};
use epistat_survival::weighting::{
    Weights, cox as wcox,
    ipcw::{self, CensoringModel, IpcwOptions},
    iptw::{self, IptwOptions},
    km::{self, KmOptions},
};
use epistat_survival::{Error, SurvBuf, SurvData};
use epistat_testkit::{BadInput, assert_rejects_bad_input};
use std::cell::Cell;
use std::fmt::Debug;

const STOP: [f64; 10] = [1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 5.5, 6.0, 7.0, 8.0];
const STATUS: [i32; 10] = [1, 0, 1, 1, 0, 1, 1, 0, 1, 1];
const WEIGHT: [f64; 10] = [1.0, 0.5, 1.5, 1.0, 2.0, 1.0, 0.5, 1.0, 1.5, 1.0];
const START: [f64; 10] = [0.0, 0.5, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 3.0, 0.0];
const GROUP: [u32; 10] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1];

thread_local! {
    /// Set while `rejects` makes the call with nothing spoiled.
    static CLEAN: Cell<bool> = const { Cell::new(false) };
}

fn clean() -> bool {
    CLEAN.with(Cell::get)
}

/// `good` spoiled as `bad` says, or `good` itself for the call with nothing
/// spoiled.
fn col(bad: BadInput, good: &[f64]) -> Vec<f64> {
    if clean() {
        good.to_vec()
    } else {
        bad.column(good)
    }
}

/// `good` with the length spoiled as `bad` says.
fn shortened<T: Clone>(bad: BadInput, good: &[T]) -> Vec<T> {
    match bad {
        _ if clean() => good.to_vec(),
        BadInput::Empty => Vec::new(),
        _ => good[..good.len() - 1].to_vec(),
    }
}

/// Runs `assert_rejects_bad_input`, after checking that the call with nothing
/// spoiled succeeds, so that a refusal is the doing of the bad input and not
/// of the rest of the call.
fn rejects<T: Debug, E: Classify + Debug>(
    label: &str,
    build: impl Fn(BadInput) -> Option<Result<T, E>>,
) {
    CLEAN.with(|c| c.set(true));
    let untouched = build(BadInput::Nan);
    CLEAN.with(|c| c.set(false));
    assert!(
        matches!(untouched, Some(Ok(_))),
        "{label}: the call with nothing spoiled returned {untouched:?}"
    );
    assert_rejects_bad_input(label, build);
}

/// The kinds of bad input that spoil one column whose order does not matter.
fn unordered(bad: BadInput) -> bool {
    clean() || bad != BadInput::Unsorted
}

/// The kinds that spoil a grid of times, which has no length to match.
fn grid(bad: BadInput) -> bool {
    clean() || bad != BadInput::Mismatched
}

/// The kinds that spoil a grid of times that may be none.
fn ascending(bad: BadInput) -> bool {
    clean() || matches!(bad, BadInput::Nan | BadInput::Infinite | BadInput::Unsorted)
}

/// The kinds that spoil a grid of times that may come in any order.
fn free_grid(bad: BadInput) -> bool {
    clean() || matches!(bad, BadInput::Empty | BadInput::Nan | BadInput::Infinite)
}

/// The kinds that put a value that is not finite in a column.
fn non_finite(bad: BadInput) -> bool {
    clean() || matches!(bad, BadInput::Nan | BadInput::Infinite)
}

/// The kinds that can spoil a column of integers: its length.
fn lengths(bad: BadInput) -> bool {
    clean() || matches!(bad, BadInput::Empty | BadInput::Mismatched)
}

/// A matrix of an intercept and `values`, one row for each value.
fn with_intercept(values: &[f64]) -> Mat<f64> {
    Mat::from_fn(values.len(), 2, |i, j| if j == 0 { 1.0 } else { values[i] })
}

/// The entry points that take no column besides the data are called on data
/// built from a spoiled stop column, so the refusal comes from the data the
/// entry point is given.
fn on_data<T>(
    bad: BadInput,
    status: &[i32],
    n_codes: u32,
    call: impl FnOnce(&SurvData<'_>) -> Result<T, Error>,
) -> Option<Result<(), Error>> {
    let stop = col(bad, &STOP);
    unordered(bad).then(|| {
        let data = SurvData::builder(&stop, status).n_codes(n_codes).build()?;
        call(&data).map(|_| ())
    })
}

#[test]
fn survival_data_refuses_bad_columns() {
    rejects("SurvData stop", |bad| {
        let stop = col(bad, &STOP);
        unordered(bad).then(|| SurvData::builder(&stop, &STATUS).build().map(|_| ()))
    });
    rejects("SurvData status", |bad| {
        let status = shortened(bad, &STATUS);
        lengths(bad).then(|| SurvData::builder(&STOP, &status).build().map(|_| ()))
    });
    rejects("SurvData weight", |bad| {
        let weight = col(bad, &WEIGHT);
        unordered(bad).then(|| {
            SurvData::builder(&STOP, &STATUS)
                .weight(&weight)
                .build()
                .map(|_| ())
        })
    });
    rejects("SurvData start", |bad| {
        let start = col(bad, &START);
        unordered(bad).then(|| {
            SurvData::builder(&STOP, &STATUS)
                .start(&start)
                .build()
                .map(|_| ())
        })
    });
    rejects("SurvData offset", |bad| {
        let offset = col(bad, &WEIGHT);
        unordered(bad).then(|| {
            SurvData::builder(&STOP, &STATUS)
                .offset(&offset)
                .build()
                .map(|_| ())
        })
    });
    rejects("SurvData stratum", |bad| {
        let stratum = shortened(bad, &GROUP);
        lengths(bad).then(|| {
            SurvData::builder(&STOP, &STATUS)
                .stratum(&stratum)
                .build()
                .map(|_| ())
        })
    });
    rejects("SurvData cluster", |bad| {
        let cluster = shortened(bad, &GROUP);
        lengths(bad).then(|| {
            SurvData::builder(&STOP, &STATUS)
                .cluster(&cluster)
                .build()
                .map(|_| ())
        })
    });
    rejects("SurvData id", |bad| {
        let id = shortened(bad, &GROUP);
        lengths(bad).then(|| {
            SurvData::builder(&STOP, &STATUS)
                .id(&id)
                .build()
                .map(|_| ())
        })
    });
}

#[test]
fn owned_survival_data_refuses_bad_columns() {
    rejects("SurvBuf stop", |bad| {
        unordered(bad).then(|| SurvBuf::builder(col(bad, &STOP), STATUS.to_vec()).build())
    });
    rejects("SurvBuf status", |bad| {
        lengths(bad).then(|| SurvBuf::builder(STOP.to_vec(), shortened(bad, &STATUS)).build())
    });
    rejects("SurvBuf weight", |bad| {
        unordered(bad).then(|| {
            SurvBuf::builder(STOP.to_vec(), STATUS.to_vec())
                .weight(col(bad, &WEIGHT))
                .build()
        })
    });
    rejects("SurvBuf start", |bad| {
        unordered(bad).then(|| {
            SurvBuf::builder(STOP.to_vec(), STATUS.to_vec())
                .start(col(bad, &START))
                .build()
        })
    });
    rejects("SurvBuf offset", |bad| {
        unordered(bad).then(|| {
            SurvBuf::builder(STOP.to_vec(), STATUS.to_vec())
                .offset(col(bad, &WEIGHT))
                .build()
        })
    });
    rejects("SurvBuf stratum", |bad| {
        lengths(bad).then(|| {
            SurvBuf::builder(STOP.to_vec(), STATUS.to_vec())
                .stratum(shortened(bad, &GROUP))
                .build()
        })
    });
    rejects("SurvBuf cluster", |bad| {
        lengths(bad).then(|| {
            SurvBuf::builder(STOP.to_vec(), STATUS.to_vec())
                .cluster(shortened(bad, &GROUP))
                .build()
        })
    });
    rejects("SurvBuf id", |bad| {
        lengths(bad).then(|| {
            SurvBuf::builder(STOP.to_vec(), STATUS.to_vec())
                .id(shortened(bad, &GROUP))
                .build()
        })
    });
}

const COVARIATE: [f64; 10] = [0.5, -1.0, 0.3, 1.2, -0.4, 0.8, -0.2, 1.5, -0.7, 0.1];
const CAUSE: [i32; 10] = [1, 0, 2, 1, 0, 2, 1, 0, 2, 1];

/// A one-column design matrix.
fn column(values: &[f64]) -> Mat<f64> {
    Mat::from_fn(values.len(), 1, |i, _| values[i])
}

/// Follow-up data with one kind of event.
fn follow_up() -> SurvBuf {
    SurvBuf::builder(STOP.to_vec(), STATUS.to_vec())
        .build()
        .unwrap()
}

/// Follow-up data with two causes.
fn competing() -> SurvBuf {
    SurvBuf::builder(STOP.to_vec(), CAUSE.to_vec())
        .n_codes(2)
        .build()
        .unwrap()
}

#[test]
fn cox_models_refuse_bad_columns() {
    let buf = follow_up();
    let data = buf.as_data();
    let options = CoxOptions::default();
    rejects("cox::fit", |bad| {
        let x = column(&col(bad, &COVARIATE));
        unordered(bad).then(|| cox::fit(&data, x.as_ref(), &options).map(|_| ()))
    });
    let fitted = cox::fit(&data, column(&COVARIATE).as_ref(), &options).unwrap();
    rejects("cox::score_residuals", |bad| {
        let x = column(&col(bad, &COVARIATE));
        unordered(bad).then(|| {
            cox::score_residuals(&data, x.as_ref(), &fitted.coefficients, &options).map(|_| ())
        })
    });
    let both = competing();
    let data = both.as_data();
    rejects("fine_gray::fit", |bad| {
        let x = column(&col(bad, &COVARIATE));
        unordered(bad)
            .then(|| fine_gray::fit(&data, x.as_ref(), &FineGrayOptions::new(1)).map(|_| ()))
    });
}

#[test]
fn curves_refuse_bad_columns() {
    rejects("survfit::estimate", |bad| {
        on_data(bad, &STATUS, 1, |d| {
            survfit::estimate(d, &SurvfitOptions::default())
        })
    });
    rejects("aalen_johansen::estimate", |bad| {
        on_data(bad, &CAUSE, 2, |d| {
            aalen_johansen::estimate(d, &AalenJohansenOptions::default())
        })
    });
    rejects("rmst::rmst2", |bad| {
        let groups = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
        let stop = col(bad, &STOP);
        unordered(bad).then(|| {
            let data = SurvData::builder(&stop, &STATUS).stratum(&groups).build()?;
            rmst::rmst2(&data, &RmstOptions::new(Tau::Fixed(4.0))).map(|_| ())
        })
    });
    let buf = follow_up();
    let data = buf.as_data();
    rejects("concordance::estimate", |bad| {
        unordered(bad).then(|| {
            concordance::estimate(&data, &col(bad, &COVARIATE), &ConcordanceOptions::default())
                .map(|_| ())
        })
    });
    rejects("survdiff::test", |bad| {
        lengths(bad).then(|| {
            survdiff::test(&data, &shortened(bad, &GROUP), &SurvdiffOptions::default()).map(|_| ())
        })
    });
    rejects("survdiff::one_sample", |bad| {
        let expected = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1];
        unordered(bad).then(|| {
            survdiff::one_sample(&data, &col(bad, &expected), &SurvdiffOptions::default())
                .map(|_| ())
        })
    });
}

#[test]
fn multistate_refuses_bad_columns() {
    let p = process();
    let data = multistate_data(&p);
    let times = [1.0, 2.5, 4.0, 5.5, 7.0];
    rejects("multistate::estimate", |bad| {
        non_finite(bad).then(|| {
            multistate::estimate(&data, &col(bad, &times), &MultistateOptions::default())
                .map(|_| ())
        })
    });
    rejects("multistate::variance", |bad| {
        ascending(bad).then(|| {
            multistate::variance(
                &data,
                &col(bad, &times),
                &MultistateOptions::default(),
                &OccupationVarianceOptions::new(0.0),
            )
            .map(|_| ())
        })
    });
}

#[test]
fn regression_models_refuse_bad_columns() {
    let buf = follow_up();
    let data = buf.as_data();
    rejects("survreg::fit", |bad| {
        let x = column(&col(bad, &COVARIATE));
        unordered(bad)
            .then(|| survreg::fit(&data, x.as_ref(), &SurvregOptions::default()).map(|_| ()))
    });
    rejects("flexpar::fit", |bad| {
        let x = column(&col(bad, &COVARIATE));
        unordered(bad).then(|| {
            flexpar::fit(
                &data,
                x.as_ref(),
                &FlexParOptions::new(FlexparScale::Hazard, 2),
            )
            .map(|_| ())
        })
    });
    let grouped = SurvBuf::builder(STOP.to_vec(), STATUS.to_vec())
        .cluster(vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        .build()
        .unwrap();
    let grouped_data = grouped.as_data();
    rejects("frailty::fit", |bad| {
        let x = column(&col(bad, &COVARIATE));
        unordered(bad).then(|| {
            frailty::fit(&grouped_data, x.as_ref(), &FrailtyOptions::default()).map(|_| ())
        })
    });
    let case: Vec<bool> = STATUS.iter().map(|&s| s == 1).collect();
    let set: Vec<u32> = (0..10).map(|i| i / 2).collect();
    rejects("clogit case", |bad| {
        let case = shortened(bad, &case);
        let x = column(&COVARIATE);
        lengths(bad).then(|| clogit(&case, &set, x.as_ref(), &ClogitOptions::default()).map(|_| ()))
    });
    rejects("clogit set", |bad| {
        let set = shortened(bad, &set);
        let x = column(&COVARIATE);
        lengths(bad).then(|| clogit(&case, &set, x.as_ref(), &ClogitOptions::default()).map(|_| ()))
    });
    rejects("clogit x", |bad| {
        let x = column(&col(bad, &COVARIATE));
        unordered(bad)
            .then(|| clogit(&case, &set, x.as_ref(), &ClogitOptions::default()).map(|_| ()))
    });
    rejects("informative_sets", |bad| {
        let x = column(&col(bad, &COVARIATE));
        unordered(bad).then(|| informative_sets(&case, &set, x.as_ref(), None).map(|_| ()))
    });
    rejects("informative_strata", |bad| {
        let x = column(&col(bad, &COVARIATE));
        unordered(bad).then(|| informative_strata(&data, x.as_ref()).map(|_| ()))
    });
}

#[test]
fn sampled_cohorts_refuse_bad_columns() {
    let case = [
        true, false, true, true, false, true, true, false, false, false,
    ];
    let design = |exit: &[f64], case: &[bool]| {
        ncc::inclusion_probabilities(&NccDesign {
            entry: None,
            exit,
            case,
            controls_per_case: 1,
            stratum: None,
        })
    };
    rejects("ncc::inclusion_probabilities exit", |bad| {
        unordered(bad).then(|| design(&col(bad, &STOP), &case).map(|_| ()))
    });
    rejects("ncc::inclusion_probabilities case", |bad| {
        lengths(bad).then(|| design(&STOP, &shortened(bad, &case)).map(|_| ()))
    });
    let probabilities = design(&STOP, &case).unwrap();
    let sampled = [true; 10];
    rejects("ncc::weights sampled", |bad| {
        lengths(bad)
            .then(|| ncc::weights(&probabilities, &shortened(bad, &sampled), &case).map(|_| ()))
    });
    rejects("ncc::weights case", |bad| {
        lengths(bad)
            .then(|| ncc::weights(&probabilities, &sampled, &shortened(bad, &case)).map(|_| ()))
    });
    let weights = ncc::weights(&probabilities, &sampled, &case).unwrap();
    let buf = follow_up();
    let data = buf.as_data();
    let x = column(&COVARIATE);
    let options = CoxOptions::default();
    rejects("ncc::fit_weighted weights", |bad| {
        unordered(bad).then(|| {
            ncc::fit_weighted(&data, x.as_ref(), &col(bad, &weights), &options).map(|_| ())
        })
    });
    rejects("ncc::fit_weighted x", |bad| {
        let x = column(&col(bad, &COVARIATE));
        unordered(bad).then(|| ncc::fit_weighted(&data, x.as_ref(), &weights, &options).map(|_| ()))
    });
    let fitted = ncc::fit_weighted(&data, x.as_ref(), &weights, &options).unwrap();
    rejects("ncc::baseline weights", |bad| {
        unordered(bad).then(|| {
            ncc::baseline(&fitted, &data, x.as_ref(), &col(bad, &weights), &options).map(|_| ())
        })
    });
    let baseline = ncc::baseline(&fitted, &data, x.as_ref(), &weights, &options).unwrap();
    let times = [2.0, 4.0, 6.0];
    // No times and no profiles are a request for an empty result, and the times
    // and the profiles may come in any order.
    rejects("ncc::absolute_risk times", |bad| {
        let new_x = column(&[0.0, 1.0]);
        non_finite(bad).then(|| {
            ncc::absolute_risk(
                &baseline,
                &fitted.coefficients,
                new_x.as_ref(),
                &col(bad, &times),
            )
            .map(|_| ())
        })
    });
    rejects("ncc::absolute_risk new_x", |bad| {
        let new_x = column(&col(bad, &[0.0, 1.0, 2.0, 3.0]));
        non_finite(bad).then(|| {
            ncc::absolute_risk(&baseline, &fitted.coefficients, new_x.as_ref(), &times).map(|_| ())
        })
    });
    rejects("casecohort::fit x", |bad| {
        let x = column(&col(bad, &COVARIATE));
        let subcohort = [false, true, false, true, true, true, true, true, true, true];
        let sizes = [100.0];
        unordered(bad).then(|| {
            casecohort::fit(
                &data,
                x.as_ref(),
                &subcohort,
                None,
                &CaseCohortOptions::new(CaseCohortMethod::Prentice, &sizes),
            )
            .map(|_| ())
        })
    });
    rejects("casecohort::fit subcohort", |bad| {
        let subcohort = [false, true, false, true, true, true, true, true, true, true];
        let subcohort = shortened(bad, &subcohort);
        let sizes = [100.0];
        lengths(bad).then(|| {
            casecohort::fit(
                &data,
                x.as_ref(),
                &subcohort,
                None,
                &CaseCohortOptions::new(CaseCohortMethod::Prentice, &sizes),
            )
            .map(|_| ())
        })
    });
}

#[test]
fn follow_up_tools_refuse_bad_columns() {
    let buf = follow_up();
    let data = buf.as_data();
    // Cut points may come in any order, and none is a split into nothing.
    rejects("split", |bad| {
        let cuts = [2.0, 4.0, 6.0];
        non_finite(bad)
            .then(|| split::split(&data, &col(bad, &cuts), &SplitOptions::default()).map(|_| ()))
    });
    let identified = SurvBuf::builder(STOP.to_vec(), STATUS.to_vec())
        .start(STOP.iter().map(|t| t - 0.5).collect())
        .id((0..10).collect())
        .build()
        .unwrap();
    let identified_data = identified.as_data();
    rejects("condense", |bad| {
        let key = [0_u64; 10];
        lengths(bad).then(|| {
            condense::condense(
                &identified_data,
                &shortened(bad, &key),
                &CondenseOptions {
                    merge_over_events: true,
                },
            )
            .map(|_| ())
        })
    });
    rejects("pyears factor", |bad| {
        let codes = shortened(bad, &GROUP);
        lengths(bad).then(|| {
            pyears::pyears(
                &data,
                &[Dimension::Factor {
                    codes: &codes,
                    levels: 2,
                }],
                &PyearsOptions::default(),
            )
            .map(|_| ())
        })
    });
    rejects("pyears origin", |bad| {
        let origin = col(bad, &[0.0; 10]);
        unordered(bad).then(|| {
            pyears::pyears(
                &data,
                &[Dimension::Time {
                    origin: &origin,
                    cuts: &[0.0, 5.0, 10.0],
                }],
                &PyearsOptions::default(),
            )
            .map(|_| ())
        })
    });
    rejects("pyears cuts", |bad| {
        let cuts = col(bad, &[0.0, 5.0, 10.0]);
        grid(bad).then(|| {
            pyears::pyears(
                &data,
                &[Dimension::Time {
                    origin: &[0.0; 10],
                    cuts: &cuts,
                }],
                &PyearsOptions::default(),
            )
            .map(|_| ())
        })
    });
}

const TIMES: [f64; 3] = [2.0, 4.0, 6.0];
const RISK: [f64; 10] = [0.1, 0.2, 0.3, 0.15, 0.25, 0.35, 0.4, 0.45, 0.5, 0.55];

/// An `n` by `columns` matrix whose entries are `values` repeated.
fn matrix(values: &[f64], columns: usize) -> Mat<f64> {
    Mat::from_fn(values.len(), columns, |i, j| {
        values[i] * (1.0 + 0.1 * j as f64)
    })
}

#[test]
fn scores_refuse_bad_columns() {
    let buf = follow_up();
    let data = buf.as_data();
    let options = ScoreOptions::default();
    rejects("metrics::score::brier predicted", |bad| {
        let predicted = matrix(&col(bad, &RISK), 3);
        unordered(bad)
            .then(|| score::brier(&data, predicted.as_ref(), &TIMES, &options).map(|_| ()))
    });
    rejects("metrics::score::brier times", |bad| {
        let predicted = matrix(&RISK, 3);
        grid(bad).then(|| {
            score::brier(&data, predicted.as_ref(), &col(bad, &TIMES), &options).map(|_| ())
        })
    });
    rejects("metrics::score::null_brier", |bad| {
        grid(bad).then(|| score::null_brier(&data, &col(bad, &TIMES), &options).map(|_| ()))
    });
    rejects("metrics::score::auc marker", |bad| {
        let marker = matrix(&col(bad, &RISK), 3);
        unordered(bad).then(|| score::auc(&data, marker.as_ref(), &TIMES, &options).map(|_| ()))
    });
    rejects("metrics::score::auc times", |bad| {
        let marker = matrix(&RISK, 3);
        grid(bad)
            .then(|| score::auc(&data, marker.as_ref(), &col(bad, &TIMES), &options).map(|_| ()))
    });
    rejects("metrics::brier::estimate predicted", |bad| {
        let predicted = matrix(&col(bad, &RISK), 3);
        unordered(bad).then(|| {
            brier::estimate(&data, predicted.as_ref(), &TIMES, &BrierOptions::default()).map(|_| ())
        })
    });
    rejects("metrics::brier::estimate times", |bad| {
        let predicted = matrix(&RISK, 3);
        free_grid(bad).then(|| {
            brier::estimate(
                &data,
                predicted.as_ref(),
                &col(bad, &TIMES),
                &BrierOptions::default(),
            )
            .map(|_| ())
        })
    });
    rejects("metrics::calibration::estimate", |bad| {
        let groups = CalibrationOptions::new(Groups::Quantiles(2));
        unordered(bad)
            .then(|| calibration::estimate(&data, &col(bad, &RISK), 5.0, &groups).map(|_| ()))
    });
    rejects("metrics::recalibration::slope_intercept", |bad| {
        unordered(bad).then(|| {
            recalibration::slope_intercept(
                &data,
                &col(bad, &RISK),
                5.0,
                &RecalibrationOptions::new(RecalibrationScale::Cloglog),
            )
            .map(|_| ())
        })
    });
    rejects("metrics::recalibration::smoothed_curve", |bad| {
        unordered(bad).then(|| {
            recalibration::smoothed_curve(
                &data,
                &col(bad, &RISK),
                5.0,
                &SmoothOptions::new(Bandwidth::Smooth),
            )
            .map(|_| ())
        })
    });
    rejects("metrics::royston::estimate", |bad| {
        unordered(bad).then(|| {
            royston::estimate(&data, &col(bad, &COVARIATE), &RoystonOptions::default()).map(|_| ())
        })
    });
}

#[test]
fn pseudo_values_refuse_bad_columns() {
    let buf = follow_up();
    let data = buf.as_data();
    rejects("pseudo::pseudo times", |bad| {
        grid(bad).then(|| {
            pseudo::pseudo(
                &data,
                &col(bad, &TIMES),
                &PseudoOptions::new(Target::Survival),
            )
            .map(|_| ())
        })
    });
    rejects("pseudo::pseudo_ipcw times", |bad| {
        let censoring = matrix(&[0.0; 0], 1);
        let _ = censoring;
        let model = matrix(&COVARIATE, 1);
        grid(bad).then(|| {
            pseudo::pseudo_ipcw(
                &data,
                model.as_ref(),
                &col(bad, &TIMES),
                &PseudoIpcwOptions::new(Target::Survival),
            )
            .map(|_| ())
        })
    });
    rejects("pseudo::pseudo_ipcw covariates", |bad| {
        let model = matrix(&col(bad, &COVARIATE), 1);
        unordered(bad).then(|| {
            pseudo::pseudo_ipcw(
                &data,
                model.as_ref(),
                &TIMES,
                &PseudoIpcwOptions::new(Target::Survival),
            )
            .map(|_| ())
        })
    });
    let cluster = [0_u32, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    rejects("pseudo::regression::fit y", |bad| {
        let x = matrix(&COVARIATE, 1);
        let y = col(bad, &RISK);
        unordered(bad).then(|| {
            pseudo::regression::fit(
                x.as_ref(),
                &y,
                &cluster,
                None,
                None,
                None,
                &PseudoRegressionOptions::new(epistat_glm::Link::Identity),
            )
            .map(|_| ())
        })
    });
    rejects("pseudo::regression::fit x", |bad| {
        let x = matrix(&col(bad, &COVARIATE), 1);
        unordered(bad).then(|| {
            pseudo::regression::fit(
                x.as_ref(),
                &RISK,
                &cluster,
                None,
                None,
                None,
                &PseudoRegressionOptions::new(epistat_glm::Link::Identity),
            )
            .map(|_| ())
        })
    });
    rejects("pseudo::regression::fit cluster", |bad| {
        let x = matrix(&COVARIATE, 1);
        let cluster = shortened(bad, &cluster);
        lengths(bad).then(|| {
            pseudo::regression::fit(
                x.as_ref(),
                &RISK,
                &cluster,
                None,
                None,
                None,
                &PseudoRegressionOptions::new(epistat_glm::Link::Identity),
            )
            .map(|_| ())
        })
    });
}

/// The multi-state data of `process`.
fn multistate_data(p: &Process) -> SurvData<'_> {
    SurvData::builder(&p.stop, &p.status)
        .start(&p.start)
        .n_codes(3)
        .id(&p.person)
        .cluster(&p.person)
        .istate(&p.istate)
        .build()
        .unwrap()
}

/// A multi-state process: two rows for each of ten people.
struct Process {
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    istate: Vec<u32>,
    person: Vec<u32>,
}

fn process() -> Process {
    let mut p = Process {
        start: Vec::new(),
        stop: Vec::new(),
        status: Vec::new(),
        istate: Vec::new(),
        person: Vec::new(),
    };
    for id in 0..10_u32 {
        let shift = f64::from(id) * 0.1;
        // State 1 to 2 to 3, or 1 to 2 and back, with the codes of the targets.
        let (second, last) = if id % 3 == 0 { (1, 3) } else { (1, 0) };
        p.start.extend([0.0, 1.0 + shift]);
        p.stop.extend([1.0 + shift, 2.5 + shift]);
        p.status.extend([second + 1, last]);
        p.istate.extend([1, 2]);
        p.person.extend([id, id]);
    }
    p
}

#[test]
fn tests_and_weights_refuse_bad_columns() {
    let p = process();
    let data = SurvData::builder(&p.stop, &p.status)
        .start(&p.start)
        .n_codes(3)
        .id(&p.person)
        .istate(&p.istate)
        .build()
        .unwrap();
    let landmarks = [0.5, 1.2, 2.0];
    let call = |grid: &[f64]| {
        markov::test(
            &data,
            &[1],
            2,
            grid,
            Bootstrap::new(5, 1),
            &MarkovOptions::default(),
        )
        .map(|_| ())
    };
    rejects("markov::test grid", |bad| {
        free_grid(bad).then(|| call(&col(bad, &landmarks)))
    });
    let buf = follow_up();
    let data = buf.as_data();
    let x = matrix(&COVARIATE, 1);
    let treated = [
        true, false, true, false, true, false, true, false, true, false,
    ];
    rejects("iptw::fit treatment", |bad| {
        let treated = shortened(bad, &treated);
        lengths(bad).then(|| iptw::fit(&treated, x.as_ref(), &IptwOptions::default()).map(|_| ()))
    });
    rejects("iptw::fit design", |bad| {
        let design = with_intercept(&col(bad, &COVARIATE));
        unordered(bad)
            .then(|| iptw::fit(&treated, design.as_ref(), &IptwOptions::default()).map(|_| ()))
    });
    // Curves at no times are an empty result, and the times may come in any order.
    rejects("km::estimate times", |bad| {
        non_finite(bad).then(|| {
            km::estimate(
                &data,
                &col(bad, &TIMES),
                &KmOptions::default(),
                &Weights::known(),
            )
            .map(|_| ())
        })
    });
    rejects("ipcw::fit", |bad| {
        let x = matrix(&col(bad, &COVARIATE), 1);
        unordered(bad).then(|| {
            ipcw::fit(
                &data,
                CensoringModel::Cox { x: x.as_ref() },
                &IpcwOptions::default(),
            )
            .map(|_| ())
        })
    });
    let design = with_intercept(&COVARIATE);
    let propensity = iptw::fit(&treated, design.as_ref(), &IptwOptions::default()).unwrap();
    rejects("weighting::cox::fit", |bad| {
        let x = matrix(&col(bad, &COVARIATE), 1);
        unordered(bad).then(|| {
            wcox::fit(
                &data,
                x.as_ref(),
                &CoxOptions::default(),
                &Weights::iptw(&propensity, None),
            )
            .map(|_| ())
        })
    });
}

#[test]
fn lexis_and_rates_refuse_bad_columns() {
    let buf = follow_up();
    let data = buf.as_data();
    let entry = [50.0, 52.0, 55.0, 57.0, 60.0, 62.0, 65.0, 67.0, 70.0, 72.0];
    let breaks = [40.0, 60.0, 80.0, 100.0];
    rejects("lexis::split_lexis entry", |bad| {
        let entry = col(bad, &entry);
        unordered(bad).then(|| {
            lexis::split_lexis(
                &data,
                &[TimeScale {
                    entry: &entry,
                    breaks: &breaks,
                }],
                &LexisOptions::default(),
            )
            .map(|_| ())
        })
    });
    // A scale without breaks never cuts a row, and the breaks are sorted by the call.
    rejects("lexis::split_lexis breaks", |bad| {
        let breaks = col(bad, &breaks);
        non_finite(bad).then(|| {
            lexis::split_lexis(
                &data,
                &[TimeScale {
                    entry: &entry,
                    breaks: &breaks,
                }],
                &LexisOptions::default(),
            )
            .map(|_| ())
        })
    });
    rejects("lexis::split_cells keys", |bad| {
        let codes = shortened(bad, &GROUP);
        lengths(bad).then(|| {
            lexis::split_cells(
                &data,
                &[TimeScale {
                    entry: &entry,
                    breaks: &breaks,
                }],
                &[CellKey {
                    codes: &codes,
                    levels: 2,
                }],
                None,
                &LexisOptions::default(),
            )
            .map(|_| ())
        })
    });
    let length = [1.0, 2.0, 1.5, 3.0, 2.0, 1.0];
    let events = [0.0, 1.0, 0.0, 2.0, 1.0, 0.0];
    let codes = [0_u32, 1, 0, 1, 0, 1];
    rejects("cells::aggregate keys", |bad| {
        let codes = shortened(bad, &codes);
        lengths(bad).then(|| {
            cells::aggregate(
                &[CellKey {
                    codes: &codes,
                    levels: 2,
                }],
                &length,
                &events,
            )
            .map(|_| ())
        })
    });
    rejects("cells::aggregate length", |bad| {
        unordered(bad).then(|| {
            cells::aggregate(
                &[CellKey {
                    codes: &codes,
                    levels: 2,
                }],
                &col(bad, &length),
                &events,
            )
            .map(|_| ())
        })
    });
    rejects("cells::aggregate events", |bad| {
        unordered(bad).then(|| {
            cells::aggregate(
                &[CellKey {
                    codes: &codes,
                    levels: 2,
                }],
                &length,
                &col(bad, &events),
            )
            .map(|_| ())
        })
    });
    let covariate = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    rejects("rates::fit_rates length", |bad| {
        unordered(bad).then(|| {
            rates::fit_rates(
                &col(bad, &length),
                &events,
                &[&covariate],
                &RateOptions::default(),
            )
            .map(|_| ())
        })
    });
    rejects("rates::fit_rates events", |bad| {
        unordered(bad).then(|| {
            rates::fit_rates(
                &length,
                &col(bad, &events),
                &[&covariate],
                &RateOptions::default(),
            )
            .map(|_| ())
        })
    });
    rejects("rates::fit_rates columns", |bad| {
        unordered(bad).then(|| {
            rates::fit_rates(
                &length,
                &events,
                &[&col(bad, &covariate)],
                &RateOptions::default(),
            )
            .map(|_| ())
        })
    });
}

#[test]
fn merging_and_windows_refuse_bad_columns() {
    let base: Vec<u32> = (0..5).collect();
    let id = [0_u32, 1, 2, 3, 4];
    let tstop = [3.0, 4.0, 5.0, 6.0, 7.0];
    rejects("tmerge::Tmerge::new base", |bad| {
        let base = shortened(bad, &base);
        lengths(bad).then(|| {
            Tmerge::new(
                &base,
                &Spans {
                    id: &id,
                    tstart: None,
                    tstop: &tstop,
                },
                TmergeOptions::default(),
            )
            .map(|_| ())
        })
    });
    rejects("tmerge::Tmerge::new spans id", |bad| {
        let ids = shortened(bad, &id);
        lengths(bad).then(|| {
            Tmerge::new(
                &base,
                &Spans {
                    id: &ids,
                    tstart: None,
                    tstop: &tstop,
                },
                TmergeOptions::default(),
            )
            .map(|_| ())
        })
    });
    rejects("tmerge::Tmerge::new tstop", |bad| {
        let stops = col(bad, &tstop);
        unordered(bad).then(|| {
            Tmerge::new(
                &base,
                &Spans {
                    id: &id,
                    tstart: None,
                    tstop: &stops,
                },
                TmergeOptions::default(),
            )
            .map(|_| ())
        })
    });
    rejects("tmerge::Tmerge::new tstart", |bad| {
        let starts = col(bad, &[0.0, 1.0, 1.0, 2.0, 2.0]);
        unordered(bad).then(|| {
            Tmerge::new(
                &base,
                &Spans {
                    id: &id,
                    tstart: Some(&starts),
                    tstop: &tstop,
                },
                TmergeOptions::default(),
            )
            .map(|_| ())
        })
    });
    let mut merged = Tmerge::new(
        &base,
        &Spans {
            id: &id,
            tstart: None,
            tstop: &tstop,
        },
        TmergeOptions::default(),
    )
    .unwrap();
    let obs_id = [0_u32, 1, 2, 3];
    let obs_time = [Some(1.0), Some(2.0), Some(1.5), Some(2.5)];
    rejects("tmerge::Tmerge::tdc id", |bad| {
        let ids = shortened(bad, &obs_id);
        lengths(bad).then(|| {
            merged.clone().tdc(
                "x",
                &Observations {
                    id: &ids,
                    time: &obs_time,
                    value: None,
                },
                None,
            )
        })
    });
    rejects("tmerge::Tmerge::tdc time", |bad| {
        let times: Vec<Option<f64>> = col(bad, &[1.0, 2.0, 1.5, 2.5])
            .into_iter()
            .map(Some)
            .collect();
        unordered(bad).then(|| {
            merged.clone().tdc(
                "x",
                &Observations {
                    id: &obs_id,
                    time: &times,
                    value: None,
                },
                None,
            )
        })
    });
    merged
        .event(
            "y",
            &Observations {
                id: &obs_id,
                time: &obs_time,
                value: None,
            },
        )
        .unwrap();
    rejects("tmerge::Tmerge::from_first_events", |bad| {
        let times: Vec<Option<f64>> = col(bad, &tstop).into_iter().map(Some).collect();
        unordered(bad).then(|| {
            Tmerge::from_first_events(&base, &id, &times, TmergeOptions::default()).map(|_| ())
        })
    });
}

#[test]
fn timelines_and_exposure_windows_refuse_bad_columns() {
    let id = [0_u32, 0, 1, 1, 2];
    let time = [1.0, 2.0, 1.0, 3.0, 2.0];
    let status = [Some(0_u32), Some(1), Some(0), Some(1), Some(0)];
    let options = TimelineOptions {
        status: StatusKind::Binary,
        repeated: Repeated::Collapse,
        lvcf: true,
    };
    rejects("surv2::counting id", |bad| {
        let ids = shortened(bad, &id);
        lengths(bad).then(|| {
            surv2::counting(
                &Timeline {
                    id: &ids,
                    time: &time,
                    status: &status,
                    covariates: &[],
                },
                &options,
            )
            .map(|_| ())
        })
    });
    rejects("surv2::counting time", |bad| {
        let times = col(bad, &time);
        unordered(bad).then(|| {
            surv2::counting(
                &Timeline {
                    id: &id,
                    time: &times,
                    status: &status,
                    covariates: &[],
                },
                &options,
            )
            .map(|_| ())
        })
    });
    rejects("surv2::counting status", |bad| {
        let statuses = shortened(bad, &status);
        lengths(bad).then(|| {
            surv2::counting(
                &Timeline {
                    id: &id,
                    time: &time,
                    status: &statuses,
                    covariates: &[],
                },
                &options,
            )
            .map(|_| ())
        })
    });
    let day = [100_i64, 200, 300, 400];
    let person = [0_u32, 1, 2, 3];
    let referent = Referent::Unidirectional { n: 1, step: 14 };
    rejects("crossover::build_windows event_day", |bad| {
        let days = shortened(bad, &day);
        lengths(bad)
            .then(|| crossover::build_windows(&days, &person, 7, 0, referent, None).map(|_| ()))
    });
    rejects("crossover::build_windows person", |bad| {
        let persons = shortened(bad, &person);
        lengths(bad)
            .then(|| crossover::build_windows(&day, &persons, 7, 0, referent, None).map(|_| ()))
    });
    rejects("crossover::build_windows observation", |bad| {
        let start = shortened(bad, &[0_i64, 0, 0, 0]);
        let end = [1000_i64; 4];
        lengths(bad).then(|| {
            crossover::build_windows(
                &day,
                &person,
                7,
                0,
                referent,
                Some(Observation {
                    start: &start,
                    end: &end,
                }),
            )
            .map(|_| ())
        })
    });
    let windows = crossover::build_windows(&day, &person, 7, 0, referent, None).unwrap();
    rejects("crossover::exposure_in_windows", |bad| {
        let who = [0_u32, 1, 2, 3];
        let start = shortened(bad, &[90_i64, 190, 290, 390]);
        let end = [95_i64, 195, 295, 395];
        lengths(bad).then(|| {
            crossover::exposure_in_windows(
                &windows,
                &ExposureIntervals {
                    person: &who,
                    start: &start,
                    end: &end,
                },
                Measure::Days,
            )
            .map(|_| ())
        })
    });
    let case_group = [true, false, true, false];
    rejects("crossover::case_time_control_design exposure", |bad| {
        let exposure = col(bad, &[1.0, 0.0, 2.0, 0.0]);
        unordered(bad)
            .then(|| crossover::case_time_control_design(&exposure, &case_group).map(|_| ()))
    });
    rejects("crossover::case_time_control_design case_group", |bad| {
        let group = shortened(bad, &case_group);
        lengths(bad)
            .then(|| crossover::case_time_control_design(&[1.0, 0.0, 2.0, 0.0], &group).map(|_| ()))
    });
}

#[test]
fn rate_tables_refuse_bad_columns() {
    let cuts = [0.0, 10.0, 20.0, 100.0];
    let rates = [0.01, 0.02, 0.05];
    rejects("RateTable::new rates", |bad| {
        let rates = col(bad, &rates);
        unordered(bad).then(|| {
            RateTable::new(
                vec![RateDim::Time {
                    cuts: cuts.to_vec(),
                }],
                rates,
                1.0,
            )
            .map(|_| ())
        })
    });
    rejects("RateTable::new cuts", |bad| {
        let cuts = col(bad, &cuts);
        // A cut list of one cut or none has no cell.
        RateTable::new(vec![RateDim::Time { cuts }], rates.to_vec(), 1.0)
            .map(|_| ())
            .into()
    });
    let start = [0.0, 10.0, 20.0];
    let level = [0_u32, 0, 0];
    rejects("RateTable::from_long start", |bad| {
        let start = col(bad, &start);
        unordered(bad).then(|| {
            RateTable::from_long(
                &[LongDim::Time {
                    start: &start,
                    end: 100.0,
                }],
                &rates,
                1.0,
            )
            .map(|_| ())
        })
    });
    rejects("RateTable::from_long level", |bad| {
        let level = shortened(bad, &level);
        lengths(bad).then(|| {
            RateTable::from_long(
                &[
                    LongDim::Time {
                        start: &start,
                        end: 100.0,
                    },
                    LongDim::Factor { level: &level },
                ],
                &rates,
                1.0,
            )
            .map(|_| ())
        })
    });
    rejects("RateTable::from_long rates", |bad| {
        let rates = col(bad, &rates);
        unordered(bad).then(|| {
            RateTable::from_long(
                &[LongDim::Time {
                    start: &start,
                    end: 100.0,
                }],
                &rates,
                1.0,
            )
            .map(|_| ())
        })
    });
    let table = RateTable::new(
        vec![RateDim::Time {
            cuts: cuts.to_vec(),
        }],
        rates.to_vec(),
        1.0,
    )
    .unwrap();
    rejects("RateTable::expected_cells", |bad| {
        let person_time = col(bad, &[1.0, 2.0, 3.0]);
        unordered(bad).then(|| table.expected_cells(&person_time).map(|_| ()))
    });
    rejects("RateTable::expected_keyed length", |bad| {
        let length = col(bad, &[1.0, 2.0, 3.0]);
        unordered(bad).then(|| table.expected_keyed(&[&[1, 2, 3]], &length).map(|_| ()))
    });
    rejects("RateTable::expected_keyed codes", |bad| {
        let codes = shortened(bad, &[1_u32, 2, 3]);
        lengths(bad).then(|| {
            table
                .expected_keyed(&[&codes], &[1.0, 2.0, 3.0])
                .map(|_| ())
        })
    });
    let buf = follow_up();
    let data = buf.as_data();
    let origin = [40.0; 10];
    rejects("RateTable::expected origins", |bad| {
        let origin = col(bad, &origin);
        unordered(bad).then(|| table.expected(&data, &[&origin], &[]).map(|_| ()))
    });
    rejects("ExpectedHazard::from_table origins", |bad| {
        let origin = col(bad, &origin);
        unordered(bad)
            .then(|| ExpectedHazard::from_table(&table, &data, &[&origin], &[]).map(|_| ()))
    });
}

#[test]
fn net_survival_refuses_bad_columns() {
    let cuts = [0.0, 10.0, 20.0, 100.0];
    let table = RateTable::new(
        vec![RateDim::Time {
            cuts: cuts.to_vec(),
        }],
        vec![0.01, 0.02, 0.05],
        1.0,
    )
    .unwrap();
    let buf = follow_up();
    let data = buf.as_data();
    let origin = [40.0; 10];
    let expected = ExpectedHazard::from_table(&table, &data, &[&origin], &[]).unwrap();
    rejects("relsurv::net_survival_test groups", |bad| {
        let groups = shortened(bad, &GROUP);
        lengths(bad)
            .then(|| relsurv::net_survival_test(&data, &groups, &expected, None).map(|_| ()))
    });
    // No times give no curves.
    rejects("relsurv::expected_survival times", |bad| {
        (non_finite(bad) || bad == BadInput::Unsorted).then(|| {
            relsurv::expected_survival(&data, &expected, &col(bad, &TIMES), ExpectedMethod::EdererI)
                .map(|_| ())
        })
    });
    rejects("relsurv::net_survival potential", |bad| {
        let potential = shortened(bad, &[20.0_f64; 10]);
        let options = NetOptions {
            method: NetMethod::Hakulinen,
            potential: Some(&potential),
            ..NetOptions::default()
        };
        lengths(bad).then(|| relsurv::net_survival(&data, &expected, &options).map(|_| ()))
    });
    rejects("relsurv::excess_hazard x", |bad| {
        let x = col(bad, &COVARIATE);
        unordered(bad).then(|| {
            relsurv::excess_hazard(
                &data,
                &[&x],
                &expected,
                &ExcessOptions::new(&[0.0, 4.0, 10.0]),
            )
            .map(|_| ())
        })
    });
    rejects("relsurv::excess_hazard cuts", |bad| {
        let cuts = col(bad, &[0.0, 4.0, 10.0]);
        grid(bad).then(|| {
            relsurv::excess_hazard(&data, &[&COVARIATE], &expected, &ExcessOptions::new(&cuts))
                .map(|_| ())
        })
    });
    let curves = relsurv::net_survival(&data, &expected, &NetOptions::default()).unwrap();
    rejects("relsurv::standardise_net weights", |bad| {
        let weights = shortened(bad, &[1.0]);
        lengths(bad).then(|| {
            relsurv::standardise_net(
                &curves,
                &weights,
                &TIMES,
                Level::default(),
                NetTransform::LogLog,
            )
            .map(|_| ())
        })
    });
}

#[test]
fn standardisation_refuses_bad_columns() {
    let buf = follow_up();
    let data = buf.as_data();
    let x = column(&COVARIATE);
    let options = CoxOptions::default();
    let fitted = cox::fit(&data, x.as_ref(), &options).unwrap();
    let set = [(0_usize, 1.0)];
    let level = ExposureLevel::single(Counterfactual::Columns(&set));
    // Times may come in any order, and none gives no columns.
    rejects("standardise::standardise_cox times", |bad| {
        non_finite(bad).then(|| {
            standardise_cox(
                &data,
                x.as_ref(),
                &fitted,
                &options,
                std::slice::from_ref(&level),
                &col(bad, &TIMES),
                &StandardiseOptions::default(),
            )
            .map(|_| ())
        })
    });
    rejects("standardise::standardise_cox design", |bad| {
        let design = column(&col(bad, &COVARIATE));
        let design_level = ExposureLevel::single(Counterfactual::Design(design.as_ref()));
        unordered(bad).then(|| {
            standardise_cox(
                &data,
                x.as_ref(),
                &fitted,
                &options,
                std::slice::from_ref(&design_level),
                &TIMES,
                &StandardiseOptions::default(),
            )
            .map(|_| ())
        })
    });
    let coefficients = [0.5, -0.2];
    let covariance = [0.04, 0.01, 0.01, 0.09];
    let zero = [(0_usize, 0.0)];
    let base = Mat::from_fn(10, 2, |i, j| if j == 0 { 1.0 } else { COVARIATE[i] });
    let levels = [
        YatesLevel {
            population: Counterfactual::Columns(&zero),
            base: Some(base.as_ref()),
            weights: None,
        },
        YatesLevel {
            population: Counterfactual::Columns(&set),
            base: Some(base.as_ref()),
            weights: None,
        },
    ];
    let yates_options = YatesOptions {
        link: InverseLink::Identity,
        centre: None,
        test: YatesTest::Global,
    };
    rejects("yates::yates coefficients", |bad| {
        let coefficients = col(bad, &coefficients);
        unordered(bad)
            .then(|| yates::yates(&coefficients, &covariance, &levels, &yates_options).map(|_| ()))
    });
    rejects("yates::yates covariance", |bad| {
        let covariance = col(bad, &covariance);
        unordered(bad)
            .then(|| yates::yates(&coefficients, &covariance, &levels, &yates_options).map(|_| ()))
    });
    rejects("yates::wald_tests estimate", |bad| {
        let estimate = col(bad, &[1.0, 2.0]);
        unordered(bad)
            .then(|| yates::wald_tests(&estimate, &covariance, YatesTest::Global).map(|_| ()))
    });
    rejects("yates::wald_tests covariance", |bad| {
        let covariance = col(bad, &covariance);
        unordered(bad)
            .then(|| yates::wald_tests(&[1.0, 2.0], &covariance, YatesTest::Global).map(|_| ()))
    });
}

#[test]
fn absolute_risks_and_aipw_refuse_bad_columns() {
    let buf = competing();
    let data = buf.as_data();
    let design = column(&COVARIATE);
    let designs = [design.as_ref(), design.as_ref()];
    rejects("absrisk::CauseSpecificCox::fit design", |bad| {
        let design = column(&col(bad, &COVARIATE));
        let designs = [design.as_ref(), design.as_ref()];
        unordered(bad)
            .then(|| CauseSpecificCox::fit(&data, &designs, &CscOptions::default()).map(|_| ()))
    });
    let model = CauseSpecificCox::fit(&data, &designs, &CscOptions::default()).unwrap();
    // As for the standardised risks, times may come in any order and may be none.
    rejects("absrisk::CauseSpecificCox::predict times", |bad| {
        let new = [column(&[0.0, 1.0])];
        let new_designs = [new[0].as_ref(), new[0].as_ref()];
        non_finite(bad).then(|| {
            model
                .predict(
                    &NewRows {
                        designs: &new_designs,
                        stratum: None,
                        offset: None,
                    },
                    &col(bad, &TIMES),
                    1,
                    &RiskOptions::default(),
                )
                .map(|_| ())
        })
    });
    rejects("absrisk::CauseSpecificCox::predict designs", |bad| {
        let new = column(&col(bad, &[0.0, 1.0, 2.0, 3.0]));
        let new_designs = [new.as_ref(), new.as_ref()];
        non_finite(bad).then(|| {
            model
                .predict(
                    &NewRows {
                        designs: &new_designs,
                        stratum: None,
                        offset: None,
                    },
                    &TIMES,
                    1,
                    &RiskOptions::default(),
                )
                .map(|_| ())
        })
    });
    let buf = follow_up();
    let data = buf.as_data();
    let outcome = column(&COVARIATE);
    let treated = [
        true, false, true, false, true, false, true, false, true, false,
    ];
    let zero = [(0_usize, 0.0)];
    let one = [(0_usize, 1.0)];
    let propensity = Mat::from_fn(
        10,
        2,
        |i, j| if j == 0 { 1.0 } else { COVARIATE[(i + 3) % 10] },
    );
    rejects("aipw::aipw_cox treatment", |bad| {
        let treatment = shortened(bad, &treated);
        lengths(bad).then(|| {
            aipw_cox(
                &data,
                &AipwSurvSpec {
                    treatment: &treatment,
                    outcome_x: outcome.as_ref(),
                    levels: [
                        Counterfactual::Columns(&zero),
                        Counterfactual::Columns(&one),
                    ],
                    propensity_x: propensity.as_ref(),
                    censoring: AipwCensoring::KaplanMeier { groups: None },
                },
                &TIMES,
                &AipwSurvOptions::default(),
            )
            .map(|_| ())
        })
    });
    rejects("aipw::aipw_cox outcome_x", |bad| {
        let outcome = column(&col(bad, &COVARIATE));
        unordered(bad).then(|| {
            aipw_cox(
                &data,
                &AipwSurvSpec {
                    treatment: &treated,
                    outcome_x: outcome.as_ref(),
                    levels: [
                        Counterfactual::Columns(&zero),
                        Counterfactual::Columns(&one),
                    ],
                    propensity_x: propensity.as_ref(),
                    censoring: AipwCensoring::KaplanMeier { groups: None },
                },
                &TIMES,
                &AipwSurvOptions::default(),
            )
            .map(|_| ())
        })
    });
    rejects("aipw::aipw_cox times", |bad| {
        grid(bad).then(|| {
            aipw_cox(
                &data,
                &AipwSurvSpec {
                    treatment: &treated,
                    outcome_x: outcome.as_ref(),
                    levels: [
                        Counterfactual::Columns(&zero),
                        Counterfactual::Columns(&one),
                    ],
                    propensity_x: propensity.as_ref(),
                    censoring: AipwCensoring::KaplanMeier { groups: None },
                },
                &col(bad, &TIMES),
                &AipwSurvOptions::default(),
            )
            .map(|_| ())
        })
    });
}

#[test]
fn self_controlled_and_multistate_regressions_refuse_bad_columns() {
    let person = [0_u32, 0, 1, 1, 2, 2];
    let length = [10.0, 20.0, 15.0, 15.0, 12.0, 18.0];
    let events = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
    let exposure = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let x = column(&exposure);
    rejects("sccs::fit length", |bad| {
        let length = col(bad, &length);
        unordered(bad).then(|| {
            sccs::fit(
                &SccsData {
                    person: &person,
                    length: &length,
                    events: &events,
                },
                x.as_ref(),
                &SccsOptions::default(),
            )
            .map(|_| ())
        })
    });
    rejects("sccs::fit events", |bad| {
        let events = col(bad, &events);
        unordered(bad).then(|| {
            sccs::fit(
                &SccsData {
                    person: &person,
                    length: &length,
                    events: &events,
                },
                x.as_ref(),
                &SccsOptions::default(),
            )
            .map(|_| ())
        })
    });
    rejects("sccs::fit person", |bad| {
        let person = shortened(bad, &person);
        lengths(bad).then(|| {
            sccs::fit(
                &SccsData {
                    person: &person,
                    length: &length,
                    events: &events,
                },
                x.as_ref(),
                &SccsOptions::default(),
            )
            .map(|_| ())
        })
    });
    rejects("sccs::fit x", |bad| {
        let x = column(&col(bad, &exposure));
        unordered(bad).then(|| {
            sccs::fit(
                &SccsData {
                    person: &person,
                    length: &length,
                    events: &events,
                },
                x.as_ref(),
                &SccsOptions::default(),
            )
            .map(|_| ())
        })
    });
    let start = [0.0, 0.0, 0.0];
    let end = [100.0, 120.0, 90.0];
    let offsets = [0_usize, 1, 2, 3];
    let event_days = [50.0, 60.0, 40.0];
    rejects("sccs::build_pieces start", |bad| {
        let start = col(bad, &start);
        unordered(bad).then(|| {
            sccs::build_pieces(&PieceSpec {
                start: &start,
                end: &end,
                event_offsets: &offsets,
                events: &event_days,
                ..PieceSpec::default()
            })
            .map(|_| ())
        })
    });
    rejects("sccs::build_pieces end", |bad| {
        let end = col(bad, &end);
        unordered(bad).then(|| {
            sccs::build_pieces(&PieceSpec {
                start: &start,
                end: &end,
                event_offsets: &offsets,
                events: &event_days,
                ..PieceSpec::default()
            })
            .map(|_| ())
        })
    });
    rejects("sccs::build_pieces events", |bad| {
        let events = col(bad, &event_days);
        unordered(bad).then(|| {
            sccs::build_pieces(&PieceSpec {
                start: &start,
                end: &end,
                event_offsets: &offsets,
                events: &events,
                ..PieceSpec::default()
            })
            .map(|_| ())
        })
    });
    // No age cuts is one age band.
    rejects("sccs::build_pieces age_cuts", |bad| {
        let cuts = col(bad, &[30.0, 60.0, 90.0]);
        non_finite(bad).then(|| {
            sccs::build_pieces(&PieceSpec {
                start: &start,
                end: &end,
                event_offsets: &offsets,
                events: &event_days,
                age_cuts: &cuts,
                ..PieceSpec::default()
            })
            .map(|_| ())
        })
    });
    let p = process();
    let data = SurvData::builder(&p.stop, &p.status)
        .start(&p.start)
        .n_codes(3)
        .id(&p.person)
        .istate(&p.istate)
        .build()
        .unwrap();
    let covariate: Vec<f64> = (0..20).map(|i| f64::from(i % 5) - 1.5).collect();
    rejects("msreg::fit x", |bad| {
        let x = column(&col(bad, &covariate));
        unordered(bad).then(|| msreg::fit(&data, x.as_ref(), &MsOptions::default()).map(|_| ()))
    });
}

#[test]
fn trial_weights_refuse_bad_columns() {
    let id = [0_u64, 0, 0, 1, 1, 1];
    let period = [0_i64, 1, 2, 0, 1, 2];
    let treatment = [false, false, true, true, true, false];
    let previous = [false, false, false, true, true, true];
    let numerator = Mat::from_fn(6, 1, |_, _| 1.0);
    let denominator = Mat::from_fn(6, 2, |i, j| {
        if j == 0 {
            1.0
        } else {
            [0.1, 0.5, 0.2, 0.7, 0.3, 0.9][i]
        }
    });
    let model = WeightModel {
        numerator: numerator.as_ref(),
        denominator: denominator.as_ref(),
    };
    let data = |id: &'static [u64], period: &'static [i64]| PersonPeriod {
        id,
        period,
        treatment: &treatment,
        previous: &previous,
    };
    let _ = data;
    rejects("trials::switch_weights treatment", |bad| {
        let treatment = shortened(bad, &treatment);
        lengths(bad).then(|| {
            trials::switch_weights(
                &PersonPeriod {
                    id: &id,
                    period: &period,
                    treatment: &treatment,
                    previous: &previous,
                },
                &model,
                &WeightRows::default(),
            )
            .map(|_| ())
        })
    });
    rejects("trials::switch_weights id", |bad| {
        let id = shortened(bad, &id);
        lengths(bad).then(|| {
            trials::switch_weights(
                &PersonPeriod {
                    id: &id,
                    period: &period,
                    treatment: &treatment,
                    previous: &previous,
                },
                &model,
                &WeightRows::default(),
            )
            .map(|_| ())
        })
    });
    rejects("trials::switch_weights period", |bad| {
        let period = shortened(bad, &period);
        lengths(bad).then(|| {
            trials::switch_weights(
                &PersonPeriod {
                    id: &id,
                    period: &period,
                    treatment: &treatment,
                    previous: &previous,
                },
                &model,
                &WeightRows::default(),
            )
            .map(|_| ())
        })
    });
    rejects("trials::switch_weights denominator", |bad| {
        let design = with_intercept(&col(bad, &[0.1, 0.5, 0.2, 0.7, 0.3, 0.9]));
        unordered(bad).then(|| {
            trials::switch_weights(
                &PersonPeriod {
                    id: &id,
                    period: &period,
                    treatment: &treatment,
                    previous: &previous,
                },
                &WeightModel {
                    numerator: numerator.as_ref(),
                    denominator: design.as_ref(),
                },
                &WeightRows::default(),
            )
            .map(|_| ())
        })
    });
    rejects("trials::switch_weights case_weight", |bad| {
        let weights = col(bad, &[1.0; 6]);
        unordered(bad).then(|| {
            trials::switch_weights(
                &PersonPeriod {
                    id: &id,
                    period: &period,
                    treatment: &treatment,
                    previous: &previous,
                },
                &model,
                &WeightRows {
                    fitted: None,
                    case_weight: Some(&weights),
                },
            )
            .map(|_| ())
        })
    });
}

#[test]
fn control_sampling_and_clones_refuse_bad_columns() {
    let day = [100_i64, 200, 300];
    let person = [0_u32, 1, 2];
    let candidate_person = [3_u32, 4, 5, 6, 7, 8];
    let start = [0_i64; 6];
    let end = [1000_i64; 6];
    let first_event: [Option<i64>; 6] = [None; 6];
    rejects("sample_controls day", |bad| {
        let day = shortened(bad, &day);
        lengths(bad).then(|| {
            sample_controls(
                &CaseEvents {
                    day: &day,
                    person: &person,
                    key: None,
                },
                &Candidates {
                    person: &candidate_person,
                    start: &start,
                    end: &end,
                    first_event: &first_event,
                    key: None,
                },
                ControlRule::EventFree,
                1,
                7,
            )
            .map(|_| ())
        })
    });
    rejects("sample_controls candidates", |bad| {
        let ends = shortened(bad, &end);
        lengths(bad).then(|| {
            sample_controls(
                &CaseEvents {
                    day: &day,
                    person: &person,
                    key: None,
                },
                &Candidates {
                    person: &candidate_person,
                    start: &start,
                    end: &ends,
                    first_event: &first_event,
                    key: None,
                },
                ControlRule::EventFree,
                1,
                7,
            )
            .map(|_| ())
        })
    });
    let id = [0_u64, 0, 0, 1, 1, 1];
    let period = [0_i64, 1, 2, 0, 1, 2];
    let treatment = [false, true, true, false, false, true];
    let outcome = [false, false, false, false, false, true];
    let covariate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let strategies = [Strategy::InitiateWhen {
        trigger: Trigger::AtLeast {
            covariate: 0,
            threshold: 0.35,
        },
        grace: 1,
        adherence: Adherence::Continuous,
    }];
    rejects("clone::clone_censor treatment", |bad| {
        let treatment = shortened(bad, &treatment);
        lengths(bad).then(|| {
            clone::clone_censor(
                &CloneInput {
                    id: &id,
                    period: &period,
                    treatment: &treatment,
                    outcome: &outcome,
                    covariates: &[&covariate],
                },
                &strategies,
            )
            .map(|_| ())
        })
    });
    rejects("clone::clone_censor covariates", |bad| {
        let covariate = col(bad, &covariate);
        unordered(bad).then(|| {
            clone::clone_censor(
                &CloneInput {
                    id: &id,
                    period: &period,
                    treatment: &treatment,
                    outcome: &outcome,
                    covariates: &[&covariate],
                },
                &strategies,
            )
            .map(|_| ())
        })
    });
    let buf = follow_up();
    let data = buf.as_data();
    rejects("pooled::fit design", |bad| {
        let design = matrix(&col(bad, &COVARIATE), 1);
        unordered(bad).then(|| {
            pooled::fit(
                &data,
                PooledEvent::Censoring { censored: None },
                design.as_ref(),
                None,
                &PooledOptions::default(),
            )
            .map(|_| ())
        })
    });
    rejects("pseudo::pseudo_states times", |bad| {
        let p = process();
        let times = [1.0, 2.0, 2.5];
        let data = SurvData::builder(&p.stop, &p.status)
            .start(&p.start)
            .n_codes(3)
            .id(&p.person)
            .istate(&p.istate)
            .build()
            .unwrap();
        grid(bad).then(|| {
            pseudo::pseudo_states(
                &data,
                &col(bad, &times),
                &StateOptions::new(StateTarget::Occupation, 0.0),
            )
            .map(|_| ())
        })
    });
    rejects("rttright::estimate times", |bad| {
        free_grid(bad).then(|| {
            rttright::estimate(&data, Some(&col(bad, &TIMES)), &RttrightOptions::default())
                .map(|_| ())
        })
    });
}

#[test]
fn bootstraps_and_variances_refuse_bad_columns() {
    rejects("aalen_johansen::variance", |bad| {
        on_data(bad, &CAUSE, 2, |d| {
            aalen_johansen::variance(
                d,
                &AalenJohansenOptions::default(),
                &AalenJohansenVarianceOptions::default(),
            )
        })
    });
    let p = process();
    let data = multistate_data(&p);
    let times = [1.0, 2.5, 4.0, 5.5, 7.0];
    multistate::bootstrap(
        &data,
        &times,
        &[3.0, 6.0],
        0.0,
        Bootstrap::new(3, 1),
        &MultistateOptions::default(),
    )
    .unwrap();
    rejects("multistate::bootstrap times", |bad| {
        non_finite(bad).then(|| {
            multistate::bootstrap(
                &data,
                &col(bad, &times),
                &[],
                0.0,
                Bootstrap::new(3, 1),
                &MultistateOptions::default(),
            )
            .map(|_| ())
        })
    });
    rejects("multistate::bootstrap horizons", |bad| {
        let horizons = col(bad, &[3.0, 6.0]);
        non_finite(bad).then(|| {
            multistate::bootstrap(
                &data,
                &times,
                &horizons,
                0.0,
                Bootstrap::new(3, 1),
                &MultistateOptions::default(),
            )
            .map(|_| ())
        })
    });
}
