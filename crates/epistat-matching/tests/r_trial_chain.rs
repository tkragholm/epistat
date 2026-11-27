//! The whole chain of a sequential trial emulation against TrialEmulation 0.0.5
//! `initiators`: `expand_trial_columns` of this crate, then
//! `protocol_rows`, `switch_weights`, `censoring_weights`,
//! `trial_weights`, `fit_msm` and `cumulative_incidence_rows` of
//! `epistat_survival::weighting::trials`, on the person-period data of
//! `data/trial_chain.R`, by intention to treat, per protocol with switch
//! weights, and per protocol with switch and censoring weights.
//!
//! # The tolerances
//!
//! The logistic models are fitted by IRLS from the same start with the same
//! stopping rule as `glm`, so a coefficient differs by `8 kappa eps` relative to
//! `max(|c|, 1)`, with `kappa` the condition number of the information matrix,
//! below `10^3` for the designs here (the period and the follow-up time squared
//! run to 7 and 49). A coefficient of a weight model then agrees to `2e-12`. A
//! linear predictor of at most four terms of size at most 10 differs by
//! `eta = 4 * 10 * 2e-12 = 8e-11`, a probability by `eta / 4`, a weight (a
//! ratio of two probabilities or of one minus them) by `2 eta` relative, and
//! the product of up to seven of them by `14 eta`, about `1.1e-9`, relative to
//! weights of at most 7; the weights are compared through their sum and maximum
//! to `1e-8` relative. The weighted outcome model, with weight errors `1.1e-9`
//! and a condition number of `10^3`, has coefficients and a covariance that
//! agree to `1e-6` relative to `max(|v|, 1)` at the most, and the cumulative
//! incidence, whose derivative with respect to a coefficient is below 1, to
//! `1e-6`. Without weights (intention to treat) the outcome model agrees to
//! `8 kappa eps = 2e-12`, which `1e-9` covers.

use chrono::{Days, NaiveDate};
use epistat_linalg::Mat;
use epistat_matching::{
    BaseRecord, PanelColumns, TrialColumns, TrialOptions, TrialPeriod, TrialPerson,
    expand_trial_columns,
};
use epistat_survival::weighting::trials::{
    Fit, Msm, MsmInput, Panel, PersonPeriod, TrialRows, WeightModel, WeightRows, censoring_weights,
    cumulative_incidence_rows, fit_msm, protocol_rows, switch_weights, trial_weights,
};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct PanelJson {
    id: Vec<u64>,
    period: Vec<i64>,
    treatment: Vec<u8>,
    eligible: Vec<u8>,
    outcome: Vec<u8>,
    censored: Vec<u8>,
    x1: Vec<f64>,
    x2: Vec<f64>,
}

#[derive(Deserialize)]
struct Models {
    d0: Vec<f64>,
    n0: Vec<f64>,
    d1: Vec<f64>,
    n1: Vec<f64>,
}

#[derive(Deserialize)]
struct Analysis {
    estimand: String,
    use_censor: bool,
    rows: usize,
    weight_sum: f64,
    weight_max: f64,
    switch_coef: Option<Models>,
    censor_coef: Option<Models>,
    msm_coef: Vec<f64>,
    msm_vcov: Vec<f64>,
    cum_inc_0: Vec<f64>,
    cum_inc_1: Vec<f64>,
}

#[derive(Deserialize)]
struct Reference {
    panel: PanelJson,
    analyses: Vec<Analysis>,
}

fn load() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/trial_chain.json"
    ))
}

fn day(period: i64) -> NaiveDate {
    NaiveDate::from_ymd_opt(2020, 1, 1)
        .and_then(|d| d.checked_add_days(Days::new(30 * u64::try_from(period).ok()?)))
        .expect("a date")
}

/// The panel of the persons, in the order of their rows.
fn panel(p: &PanelJson) -> Vec<TrialPerson<BaseRecord>> {
    let mut panel: Vec<TrialPerson<BaseRecord>> = Vec::new();
    for i in 0..p.id.len() {
        if i == 0 || p.id[i] != p.id[i - 1] {
            panel.push(TrialPerson {
                record: BaseRecord::new(p.id[i].to_string(), day(0)),
                periods: Vec::new(),
            });
        }
        let person = panel.last_mut().expect("a person");
        person.periods.push(TrialPeriod {
            period: p.period[i],
            date: day(p.period[i]),
            eligible: p.eligible[i] == 1,
            treatment: Some(p.treatment[i] == 1),
            event: p.outcome[i] == 1,
        });
    }
    panel
}

fn worst(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / y.abs().max(1.0))
        .fold(0.0, f64::max)
}

#[expect(
    clippy::cast_precision_loss,
    reason = "periods and follow-up times are small integers"
)]
fn design_row(t: &TrialColumns, p: &PanelJson, assigned: bool, f: i64, r: usize, out: &mut [f64]) {
    let s = t.start_row[r];
    let f = f as f64;
    out.copy_from_slice(&[
        1.0,
        f64::from(u8::from(assigned)),
        t.trial_period[r] as f64,
        f,
        f * f,
        p.x1[s],
        p.x2[s],
    ]);
}

/// The weight of every row of the trials.
#[expect(clippy::cast_precision_loss, reason = "periods are small integers")]
fn weights(a: &Analysis, p: &PanelJson, cols: &PanelColumns, t: &TrialColumns) -> Vec<f64> {
    if a.estimand == "ITT" {
        return vec![1.0; t.len()];
    }
    let treatment = cols.known_treatment().expect("known treatment");
    let chosen = protocol_rows(&Panel {
        id: &cols.person,
        period: &cols.period,
        treatment: &treatment,
        eligible: &cols.eligible,
        outcome: &cols.event,
    })
    .expect("rows");
    let data = PersonPeriod {
        id: &cols.person,
        period: &cols.period,
        treatment: &treatment,
        previous: &chosen.previous,
    };
    let n = cols.len();
    let rows = WeightRows {
        fitted: Some(&chosen.fitted),
        case_weight: None,
    };
    let den = Mat::from_fn(n, 4, |i, j| [1.0, p.x1[i], p.x2[i], p.period[i] as f64][j]);
    let num = Mat::from_fn(n, 2, |i, j| [1.0, p.x1[i]][j]);
    let switch = switch_weights(
        &data,
        &WeightModel {
            numerator: num.as_ref(),
            denominator: den.as_ref(),
        },
        &rows,
    )
    .expect("switch weights");
    let models = a.switch_coef.as_ref().expect("switch models");
    let check = |got: &[Vec<f64>], want: [&Vec<f64>; 2], what: &str| {
        for (g, w) in got.iter().zip(want) {
            assert!(worst(g, w) < 2e-12, "{what}: {g:?} against {w:?}");
        }
    };
    check(
        &switch.denominator_coefficients,
        [&models.d0, &models.d1],
        "switch denominator",
    );
    check(
        &switch.numerator_coefficients,
        [&models.n0, &models.n1],
        "switch numerator",
    );
    let mut row_weight = switch.weight;
    if a.use_censor {
        let censored: Vec<bool> = p.censored.iter().map(|&c| c == 1).collect();
        let den = Mat::from_fn(n, 3, |i, j| [1.0, p.x1[i], p.x2[i]][j]);
        let cens = censoring_weights(
            &data,
            &censored,
            &WeightModel {
                numerator: num.as_ref(),
                denominator: den.as_ref(),
            },
            Fit::ByPreviousTreatment,
            &rows,
        )
        .expect("censoring weights");
        let models = a.censor_coef.as_ref().expect("censoring models");
        check(
            &cens.denominator_coefficients,
            [&models.d0, &models.d1],
            "censoring denominator",
        );
        check(
            &cens.numerator_coefficients,
            [&models.n0, &models.n1],
            "censoring numerator",
        );
        for (w, c) in row_weight.iter_mut().zip(&cens.weight) {
            *w *= c;
        }
    }
    trial_weights(
        &data,
        &row_weight,
        &TrialRows {
            id: &t.person,
            trial_period: &t.trial_period,
            follow_up: &t.follow_up_time,
        },
    )
    .expect("trial weights")
}

fn risk(msm: &Msm, p: &PanelJson, t: &TrialColumns, assigned: bool) -> Vec<f64> {
    let starts: Vec<usize> = (0..t.len()).filter(|&r| t.follow_up_time[r] == 0).collect();
    cumulative_incidence_rows(msm, starts.len(), 6, &|i, time, out| {
        let f = i64::try_from(time).expect("small");
        design_row(t, p, assigned, f, starts[i], out);
    })
    .expect("risk")
    .risk
}

#[test]
fn the_chain_agrees_with_trial_emulation() {
    let r = load();
    let p = &r.panel;
    let people = panel(p);
    let cols = PanelColumns::of(&people);
    assert_eq!(cols.len(), p.id.len());
    for a in &r.analyses {
        let options = TrialOptions {
            censor_at_deviation: a.estimand == "PP",
            ..TrialOptions::default()
        };
        let t = expand_trial_columns(&people, &options).expect("valid panel");
        assert_eq!(t.len(), a.rows, "{} rows", a.estimand);
        let weight = weights(a, p, &cols, &t);
        let sum: f64 = weight.iter().sum();
        let max = weight.iter().copied().fold(0.0, f64::max);
        assert!((sum - a.weight_sum).abs() / a.weight_sum < 1e-8, "{sum}");
        assert!((max - a.weight_max).abs() / a.weight_max < 1e-8, "{max}");
        let design = Mat::from_fn(t.len(), 7, |r, j| {
            let mut out = [0.0; 7];
            design_row(&t, p, t.assigned[r], t.follow_up_time[r], r, &mut out);
            out[j]
        });
        let msm = fit_msm(&MsmInput {
            design: design.as_ref(),
            outcome: &t.outcome,
            weight: &weight,
            person: &t.person,
        })
        .expect("outcome model");
        let tol = if a.estimand == "ITT" { 1e-9 } else { 1e-6 };
        let what = format!("{} censoring {}", a.estimand, a.use_censor);
        assert!(worst(&msm.coefficients, &a.msm_coef) < tol, "{what}");
        assert!(worst(&msm.covariance, &a.msm_vcov) < tol, "{what}");
        assert!(
            worst(&risk(&msm, p, &t, false), &a.cum_inc_0) < tol,
            "{what}"
        );
        assert!(
            worst(&risk(&msm, p, &t, true), &a.cum_inc_1) < tol,
            "{what}"
        );
    }
}

/// The spans of the trial columns form the same rows as the columns, and the
/// rows point at the panel rows of their person and period.
#[test]
fn the_columns_point_at_their_panel_rows() {
    let r = load();
    let people = panel(&r.panel);
    let cols = PanelColumns::of(&people);
    let t = expand_trial_columns(&people, &TrialOptions::default()).expect("valid");
    for i in 0..t.len() {
        assert_eq!(cols.person[t.row[i]], t.person[i]);
        assert_eq!(cols.person[t.start_row[i]], t.person[i]);
        assert_eq!(cols.period[t.start_row[i]], t.trial_period[i]);
        assert_eq!(
            cols.period[t.row[i]] - cols.period[t.start_row[i]],
            t.follow_up_time[i]
        );
    }
    let (start, end) = t.spans();
    let rows: Vec<usize> = start.iter().zip(&end).flat_map(|(&a, &b)| a..b).collect();
    assert_eq!(rows, t.row);
    let rows = epistat_matching::expand_trials(&people, &TrialOptions::default()).expect("valid");
    assert_eq!(rows.len(), t.len());
    for (row, i) in rows.iter().zip(0..) {
        assert_eq!(row.trial_period, t.trial_period[i]);
        assert_eq!(row.follow_up_time, t.follow_up_time[i]);
        assert_eq!(row.assigned, t.assigned[i]);
        assert_eq!(row.outcome, t.outcome[i]);
    }
}
