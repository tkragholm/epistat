//! Truncated weights in the outcome model of sequential trials,
//! `WeightLimits::apply` as `epistat_survival::weighting::trials` takes it, against TrialEmulation 0.0.5
//! `initiators(analysis_weights = "p99")` and `"weight_limits"` on the panel of
//! `data/trial_chain.R`, per protocol with switch weights. See
//! `data/trial_limits.R`.
//!
//! # The tolerances
//!
//! As in `r_trial_chain.rs`: the weights before truncation agree to `1.1e-9`
//! relative, and a cut at a type 1 quantile is one of them, chosen by rank,
//! which only a tie within that error could change; there is none here. The
//! outcome model and the risks then agree to `1e-6` relative.

#[path = "common/trial_panel.rs"]
mod trial_panel;
use chrono::{Days, NaiveDate};
use epistat_linalg::Mat;
use epistat_matching::{BaseRecord, TrialColumns, TrialOptions, expand_trial_columns};
use epistat_survival::weighting::trials::{
    Msm, MsmInput, PersonPeriod, Population, ProtocolPanel, TrialRows, TrialWeightFit,
    WeightLimits, WeightModel, WeightRows, censoring_weights, cumulative_incidence, fit_msm,
    protocol_rows, switch_weights, trial_weights,
};
use epistat_testkit::load_json;
use serde::Deserialize;
use trial_panel::{PanelPeriod, PanelPerson, PanelRows, persons};

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
    switch_coef: Option<Models>,
    censor_coef: Option<Models>,
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

#[derive(Deserialize)]
struct Limited {
    analysis_weights: String,
    msm_coef: Vec<f64>,
    msm_vcov: Vec<f64>,
    cum_inc_0: Vec<f64>,
    cum_inc_1: Vec<f64>,
}

#[derive(Deserialize)]
struct Limits {
    analyses: Vec<Limited>,
}

fn day(period: i64) -> NaiveDate {
    NaiveDate::from_ymd_opt(2020, 1, 1)
        .and_then(|d| d.checked_add_days(Days::new(30 * u64::try_from(period).ok()?)))
        .expect("a date")
}

/// The panel of the persons, in the order of their rows.
fn panel(p: &PanelJson) -> Vec<PanelPerson<BaseRecord>> {
    let mut panel: Vec<PanelPerson<BaseRecord>> = Vec::new();
    for i in 0..p.id.len() {
        if i == 0 || p.id[i] != p.id[i - 1] {
            panel.push(PanelPerson {
                record: BaseRecord::new(p.id[i].to_string(), day(0)),
                periods: Vec::new(),
            });
        }
        let person = panel.last_mut().expect("a person");
        person.periods.push(PanelPeriod {
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
fn weights(
    a: &Analysis,
    p: &PanelJson,
    cols: &PanelRows<BaseRecord>,
    t: &TrialColumns,
) -> Vec<f64> {
    if a.estimand == "ITT" {
        return vec![1.0; t.len()];
    }
    let treatment = cols.known_treatment().expect("known treatment");
    let chosen = protocol_rows(&ProtocolPanel {
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
            TrialWeightFit::ByPreviousTreatment,
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
            id: &persons(t),
            trial_period: &t.trial_period,
            follow_up: &t.follow_up_time,
        },
    )
    .expect("trial weights")
}

fn risk(msm: &Msm, p: &PanelJson, t: &TrialColumns, assigned: bool) -> Vec<f64> {
    let starts: Vec<usize> = (0..t.len()).filter(|&r| t.follow_up_time[r] == 0).collect();
    cumulative_incidence(
        msm,
        &Population::Rows {
            rows: starts.len(),
            times: 6,
            design: &|i, time, out| {
                let f = i64::try_from(time).expect("small");
                design_row(t, p, assigned, f, starts[i], out);
            },
        },
    )
    .expect("risk")
    .risk
}

#[test]
fn truncated_weights_agree_with_trial_emulation() {
    let r = load();
    let p = &r.panel;
    let people = panel(p);
    let cols = PanelRows::of(&people);
    let pp = r
        .analyses
        .iter()
        .find(|a| a.estimand == "PP" && !a.use_censor)
        .expect("the per-protocol analysis");
    let options = TrialOptions::default();
    let t = expand_trial_columns(&cols.panel(), &options).expect("valid panel");
    let weight = weights(pp, p, &cols, &t);
    let limits: Limits = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/trial_limits.json"
    ));
    for a in &limits.analyses {
        let rule = match a.analysis_weights.as_str() {
            "p99" => WeightLimits::P99,
            _ => WeightLimits::Values {
                lower: 0.5,
                upper: 3.0,
            },
        };
        let limited = rule.apply(&weight).expect("limits");
        let design = Mat::from_fn(t.len(), 7, |r, j| {
            let mut out = [0.0; 7];
            design_row(&t, p, t.assigned[r], t.follow_up_time[r], r, &mut out);
            out[j]
        });
        let msm = fit_msm(MsmInput {
            design: design.as_ref(),
            outcome: &t.outcome,
            weight: &limited,
            person: &persons(&t),
        })
        .expect("outcome model");
        let what = &a.analysis_weights;
        assert!(worst(&msm.coefficients, &a.msm_coef) < 1e-6, "{what}");
        let covariance: Vec<f64> = msm
            .covariance
            .as_slice()
            .iter()
            .map(|v| v.unwrap())
            .collect();
        assert!(worst(&covariance, &a.msm_vcov) < 1e-6, "{what}");
        assert!(
            worst(&risk(&msm, p, &t, false), &a.cum_inc_0) < 1e-6,
            "{what}"
        );
        assert!(
            worst(&risk(&msm, p, &t, true), &a.cum_inc_1) < 1e-6,
            "{what}"
        );
    }
}
