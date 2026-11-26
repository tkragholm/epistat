//! Every entry point that takes columns refuses an empty column, NaN, an
//! infinity and columns of different lengths with an `Input` error, and
//! every one that takes numbers refuses NaN and an infinity, through
//! `epistat_testkit::assert_rejects_bad_input`. Order does not matter to any
//! of them, so `Unsorted` is not tried.

use epistat_core::Level;
use epistat_epi::decomposition::{Split, kitagawa_poisson};
use epistat_epi::rates::{
    Strata, crude_rate, direct_standardised, rate_difference, rate_ratio, rate_ratio_exact,
    standardised_difference, standardised_ratio,
};
use epistat_epi::rosenbaum::{
    MScore, WilcoxonDistribution, hodges_lehmann_bounds, hodges_lehmann_gamma, m_scores,
    wilcoxon_bounds,
};
use epistat_epi::sir::{sir, sir_homogeneity, sir_trend};
use epistat_epi::{Counts2x2, PersonTime2, bias, evalue, mantel_haenszel, signed_rank};
use epistat_testkit::{BadInput, assert_rejects_bad_input};

const EVENTS: [f64; 4] = [3.0, 5.0, 2.0, 7.0];
const TIME: [f64; 4] = [100.0, 140.0, 90.0, 210.0];
const EVENTS0: [f64; 4] = [4.0, 2.0, 6.0, 3.0];
const TIME0: [f64; 4] = [120.0, 80.0, 150.0, 95.0];
const WEIGHTS: [f64; 4] = [0.2, 0.3, 0.1, 0.4];
const DIFFERENCES: [f64; 5] = [0.4, -0.1, 1.3, 0.7, 2.2];

/// The kinds that spoil one of several columns.
fn columns(bad: BadInput) -> bool {
    bad != BadInput::Unsorted
}

/// The kinds that spoil the one column an entry point takes.
fn one(bad: BadInput) -> bool {
    matches!(bad, BadInput::Empty | BadInput::Nan | BadInput::Infinite)
}

/// The kinds that spoil a single number, and the number spoilt.
fn number(bad: BadInput, x: f64) -> Option<f64> {
    match bad {
        BadInput::Nan | BadInput::Infinite => Some(bad.column(&[x])[0]),
        _ => None,
    }
}

/// The 2 x 2 tables whose exposed cases are `cases`.
fn tables(cases: &[f64]) -> Vec<Counts2x2> {
    cases
        .iter()
        .map(|&a| Counts2x2 {
            exposed_cases: a,
            exposed_noncases: 20.0,
            unexposed_cases: 4.0,
            unexposed_noncases: 30.0,
        })
        .collect()
}

/// The person-time strata whose exposed events are `events`.
fn strata(events: &[f64]) -> Vec<PersonTime2> {
    events
        .iter()
        .map(|&x| PersonTime2 {
            exposed_events: x,
            exposed_time: 100.0,
            unexposed_events: 4.0,
            unexposed_time: 120.0,
        })
        .collect()
}

#[test]
fn rates_refuse_bad_columns() {
    let level = Level::default();
    assert_rejects_bad_input("direct_standardised", |bad| {
        columns(bad).then(|| match bad {
            BadInput::Mismatched => {
                direct_standardised(&EVENTS, &bad.column(&TIME), &WEIGHTS, level)
            }
            _ => direct_standardised(&bad.column(&EVENTS), &TIME, &WEIGHTS, level),
        })
    });
    assert_rejects_bad_input("direct_standardised weights", |bad| {
        columns(bad).then(|| direct_standardised(&EVENTS, &TIME, &bad.column(&WEIGHTS), level))
    });
    let reference = Strata {
        events: &EVENTS0,
        time: &TIME0,
    };
    assert_rejects_bad_input("standardised_ratio", |bad| {
        columns(bad).then(|| {
            let events = bad.column(&EVENTS);
            let group = Strata {
                events: &events,
                time: &TIME,
            };
            standardised_ratio(
                group,
                reference,
                &WEIGHTS,
                level,
                epistat_epi::rates::RatioInterval::F,
            )
        })
    });
    assert_rejects_bad_input("standardised_difference", |bad| {
        columns(bad).then(|| {
            let time = bad.column(&TIME0);
            let reference = Strata {
                events: &EVENTS0,
                time: &time,
            };
            let group = Strata {
                events: &EVENTS,
                time: &TIME,
            };
            standardised_difference(group, reference, &WEIGHTS, level)
        })
    });
}

#[test]
fn rates_refuse_bad_numbers() {
    let level = Level::default();
    assert_rejects_bad_input("crude_rate", |bad| {
        number(bad, 7.0).map(|events| crude_rate(events, 12.5, level, 1000.0))
    });
    assert_rejects_bad_input("crude_rate scale", |bad| {
        number(bad, 1000.0).map(|scale| crude_rate(7.0, 12.5, level, scale))
    });
    assert_rejects_bad_input("rate_ratio", |bad| {
        number(bad, 100.0).map(|t1| rate_ratio(5.0, t1, 9.0, 120.0, level))
    });
    assert_rejects_bad_input("rate_difference", |bad| {
        number(bad, 9.0).map(|e0| rate_difference(5.0, 100.0, e0, 120.0, level))
    });
    assert_rejects_bad_input("rate_ratio_exact", |bad| {
        number(bad, 5.0).map(|e1| rate_ratio_exact(e1, 100.0, 9.0, 120.0, level))
    });
}

#[test]
fn mantel_haenszel_refuses_bad_strata() {
    let level = Level::default();
    let cases = [10.0, 6.0, 3.0, 8.0];
    assert_rejects_bad_input("mantel_haenszel::odds_ratio", |bad| {
        one(bad).then(|| mantel_haenszel::odds_ratio(&tables(&bad.column(&cases)), level))
    });
    assert_rejects_bad_input("mantel_haenszel::risk_ratio", |bad| {
        one(bad).then(|| mantel_haenszel::risk_ratio(&tables(&bad.column(&cases)), level))
    });
    assert_rejects_bad_input("mantel_haenszel::risk_difference", |bad| {
        one(bad).then(|| mantel_haenszel::risk_difference(&tables(&bad.column(&cases)), level))
    });
    assert_rejects_bad_input("mantel_haenszel::cmh_test", |bad| {
        one(bad).then(|| mantel_haenszel::cmh_test(&tables(&bad.column(&cases)), true))
    });
    assert_rejects_bad_input("mantel_haenszel::breslow_day", |bad| {
        one(bad).then(|| mantel_haenszel::breslow_day(&tables(&bad.column(&cases)), true))
    });
    assert_rejects_bad_input("mantel_haenszel::rate_ratio", |bad| {
        one(bad).then(|| mantel_haenszel::rate_ratio(&strata(&bad.column(&EVENTS)), level))
    });
    assert_rejects_bad_input("mantel_haenszel::rate_difference", |bad| {
        one(bad).then(|| mantel_haenszel::rate_difference(&strata(&bad.column(&EVENTS)), level))
    });
    assert_rejects_bad_input("mantel_haenszel::rate_ratio_test", |bad| {
        one(bad).then(|| mantel_haenszel::rate_ratio_test(&strata(&bad.column(&EVENTS)), false))
    });
    assert_rejects_bad_input("mantel_haenszel::rate_ratio_breslow_day", |bad| {
        one(bad)
            .then(|| mantel_haenszel::rate_ratio_breslow_day(&strata(&bad.column(&EVENTS)), true))
    });
    assert_rejects_bad_input("mantel_haenszel::rate_ratio_homogeneity", |bad| {
        one(bad).then(|| mantel_haenszel::rate_ratio_homogeneity(&strata(&bad.column(&EVENTS))))
    });
}

#[test]
fn sir_refuses_bad_input() {
    let level = Level::default();
    let exact = epistat_epi::sir::SirInterval::Exact;
    assert_rejects_bad_input("sir", |bad| {
        number(bad, 12.0).map(|observed| sir(observed, 9.5, level, exact))
    });
    assert_rejects_bad_input("sir_homogeneity", |bad| {
        columns(bad).then(|| match bad {
            BadInput::Mismatched => sir_homogeneity(&EVENTS, &bad.column(&EVENTS0)),
            _ => sir_homogeneity(&bad.column(&EVENTS), &EVENTS0),
        })
    });
    let score = [1.0, 2.0, 3.0, 4.0];
    assert_rejects_bad_input("sir_trend", |bad| {
        columns(bad).then(|| sir_trend(&EVENTS, &EVENTS0, &bad.column(&score)))
    });
}

#[test]
fn kitagawa_poisson_refuses_bad_columns() {
    let level = Level::default();
    let reference = Strata {
        events: &EVENTS0,
        time: &TIME0,
    };
    assert_rejects_bad_input("kitagawa_poisson", |bad| {
        columns(bad).then(|| {
            let events = bad.column(&EVENTS);
            let group = Strata {
                events: &events,
                time: &TIME,
            };
            kitagawa_poisson(group, reference, Split::Reference, level, 1000.0)
        })
    });
    assert_rejects_bad_input("kitagawa_poisson person-time", |bad| {
        columns(bad).then(|| {
            let time = bad.column(&TIME0);
            let reference = Strata {
                events: &EVENTS0,
                time: &time,
            };
            let group = Strata {
                events: &EVENTS,
                time: &TIME,
            };
            kitagawa_poisson(group, reference, Split::Symmetric, level, 1000.0)
        })
    });
}

#[test]
fn rosenbaum_and_signed_rank_refuse_bad_columns() {
    assert_rejects_bad_input("wilcoxon_bounds", |bad| {
        one(bad)
            .then(|| wilcoxon_bounds(&bad.column(&DIFFERENCES), 2.0, WilcoxonDistribution::Exact))
    });
    assert_rejects_bad_input("hodges_lehmann_bounds", |bad| {
        one(bad).then(|| hodges_lehmann_bounds(&bad.column(&DIFFERENCES), &[1.0, 2.0]))
    });
    assert_rejects_bad_input("hodges_lehmann_gamma", |bad| {
        one(bad).then(|| hodges_lehmann_gamma(&bad.column(&DIFFERENCES), 0.0))
    });
    assert_rejects_bad_input("m_scores", |bad| {
        one(bad).then(|| {
            let sets = vec![
                bad.column(&[3.0, 1.0, 2.0]),
                vec![4.0, 2.5],
                vec![1.5, 0.5, 1.0],
            ];
            m_scores(&sets, &MScore::default())
        })
    });
    assert_rejects_bad_input("signed_rank::average_ranks", |bad| {
        matches!(bad, BadInput::Nan | BadInput::Infinite)
            .then(|| signed_rank::average_ranks(&bad.column(&DIFFERENCES)))
    });
}

#[test]
fn evalue_and_bias_refuse_bad_numbers() {
    assert_rejects_bad_input("evalue::evalues_rr", |bad| {
        number(bad, 1.8).map(|rr| evalue::evalues_rr(rr, Some(1.2), Some(2.7), 1.0))
    });
    assert_rejects_bad_input("evalue::evalues_rd", |bad| {
        number(bad, 150.0).map(|a| {
            let t = Counts2x2 {
                exposed_cases: a,
                exposed_noncases: 9_850.0,
                unexposed_cases: 900.0,
                unexposed_noncases: 89_100.0,
            };
            evalue::evalues_rd(&t, 0.0, Level::default())
        })
    });
    let m = bias::Misclass {
        se: [0.9, 0.85],
        sp: [0.95, 0.9],
    };
    assert_rejects_bad_input("bias::correct_exposure", |bad| {
        number(bad, 150.0).map(|a| {
            let t = Counts2x2 {
                exposed_cases: a,
                exposed_noncases: 9_850.0,
                unexposed_cases: 900.0,
                unexposed_noncases: 89_100.0,
            };
            bias::correct_exposure(&t, &m)
        })
    });
}
