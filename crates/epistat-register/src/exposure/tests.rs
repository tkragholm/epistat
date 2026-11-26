#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "test sizes and days are small draws from a generator"
)]

use super::*;
use crate::person_time::days_from_civil;
use epistat_testkit::{SplitMix64, assert_same_on_threads};

/// 100 units at 2 a day last 50 days. Invented for tests.
const N02: Assumption = Assumption {
    dose_default: 2.0,
    dose_min: 1.0,
    dose_max: 4.0,
    max_stock_days: 30.0,
    grace_days: 0.0,
};

fn day(y: i64, m: i64, d: i64) -> i64 {
    days_from_civil(y, m, d)
}

/// One person, one group.
fn periods_of(
    rows: &[(i64, f64)],
    a: Assumption,
    rule: DoseRule,
    stock: Stock,
    pauses: &[(i64, i64)],
) -> Periods {
    let days: Vec<i64> = rows.iter().map(|r| r.0).collect();
    let quantity: Vec<f64> = rows.iter().map(|r| r.1).collect();
    let group = vec![Some(0); rows.len()];
    let d = Dispensings {
        person_offsets: &[0, rows.len()],
        group: &group,
        day: &days,
        quantity: &quantity,
    };
    let (ps, pe): (Vec<i64>, Vec<i64>) = pauses.iter().copied().unzip();
    let offsets = [0, pauses.len()];
    let p = Pauses {
        person_offsets: &offsets,
        start: &ps,
        end: &pe,
    };
    let assumptions = [a];
    let s = Supply {
        assumptions: &assumptions,
        rule,
        stock,
        pauses: (!pauses.is_empty()).then_some(&p),
    };
    exposure_periods(&d, &s).unwrap()
}

fn spans(p: &Periods) -> Vec<(i64, i64)> {
    (0..p.len()).map(|i| (p.start[i], p.end_day(i))).collect()
}

fn simple(rows: &[(i64, f64)]) -> Vec<(i64, i64)> {
    spans(&periods_of(
        rows,
        N02,
        DoseRule::Default,
        Stock::CarryForward,
        &[],
    ))
}

#[test]
fn one_dispensing_lasts_quantity_over_dose() {
    assert_eq!(
        simple(&[(day(2020, 1, 1), 100.0)]),
        [(day(2020, 1, 1), day(2020, 2, 20))]
    );
}

#[test]
fn a_refill_before_the_supply_runs_out_starts_when_it_does() {
    for refill in [day(2020, 1, 21), day(2020, 1, 1)] {
        assert_eq!(
            simple(&[(day(2020, 1, 1), 100.0), (refill, 100.0)]),
            [(day(2020, 1, 1), day(2020, 4, 10))],
            "refill on {refill}"
        );
    }
}

#[test]
fn a_gap_wider_than_the_supply_ends_the_period() {
    assert_eq!(
        simple(&[(day(2020, 1, 1), 100.0), (day(2020, 7, 1), 100.0)]),
        [
            (day(2020, 1, 1), day(2020, 2, 20)),
            (day(2020, 7, 1), day(2020, 8, 20))
        ]
    );
    // A dispensing on the day the supply runs out continues the period.
    assert_eq!(simple(&[(0, 100.0), (50, 100.0)]).len(), 1);
    assert_eq!(simple(&[(0, 100.0), (51, 100.0)]).len(), 2);
}

#[test]
fn stockpiling_beyond_the_cap_is_discarded() {
    let rows: Vec<(i64, f64)> = (1..=6).map(|d| (day(2020, 1, d), 100.0)).collect();
    let capped = simple(&rows);
    assert_eq!(capped.len(), 1);
    assert!(capped[0].1 < day(2020, 5, 1));
    let uncapped = spans(&periods_of(
        &rows,
        Assumption {
            max_stock_days: f64::INFINITY,
            ..N02
        },
        DoseRule::Default,
        Stock::CarryForward,
        &[],
    ));
    assert_eq!(uncapped[0].1, day(2020, 1, 1) + 300);
    let discarded = spans(&periods_of(
        &rows,
        N02,
        DoseRule::Default,
        Stock::Discard,
        &[],
    ));
    assert_eq!(discarded[0].1, day(2020, 1, 6) + 50);
}

#[test]
fn a_grace_period_joins_dispensings_and_adds_no_supply() {
    let with = |grace| {
        spans(&periods_of(
            &[(0, 100.0), (60, 100.0)],
            Assumption {
                grace_days: grace,
                ..N02
            },
            DoseRule::Default,
            Stock::CarryForward,
            &[],
        ))
    };
    // The first supply ends on day 50. A gap of 10 days is within a grace of 10.
    assert_eq!(with(10.0), [(0, 110)]);
    assert_eq!(with(9.0), [(0, 50), (60, 110)]);
}

#[test]
fn pauses_extend_a_supply_by_the_paused_days_inside_it() {
    let p = |pauses: &[(i64, i64)]| {
        spans(&periods_of(
            &[(day(2020, 1, 1), 100.0)],
            N02,
            DoseRule::Default,
            Stock::CarryForward,
            pauses,
        ))
    };
    assert_eq!(
        p(&[(day(2020, 1, 10), day(2020, 1, 20))]),
        [(day(2020, 1, 1), day(2020, 3, 1))]
    );
    assert_eq!(
        p(&[(day(2021, 1, 1), day(2021, 2, 1))]),
        [(day(2020, 1, 1), day(2020, 2, 20))]
    );
    assert_eq!(
        p(&[(day(2019, 1, 1), day(2019, 2, 1))]),
        [(day(2020, 1, 1), day(2020, 2, 20))]
    );
    assert_eq!(
        p(&[
            (day(2020, 1, 10), day(2020, 1, 20)),
            (day(2020, 2, 1), day(2020, 2, 11))
        ]),
        [(day(2020, 1, 1), day(2020, 3, 11))]
    );
}

#[test]
fn an_observed_dose_is_read_off_the_refill_interval_and_clamped() {
    let observed = |rows: &[(i64, f64)]| {
        spans(&periods_of(
            rows,
            N02,
            DoseRule::Observed,
            Stock::CarryForward,
            &[],
        ))
    };
    assert_eq!(
        observed(&[(day(2020, 1, 1), 100.0), (day(2020, 1, 26), 100.0)]),
        [(day(2020, 1, 1), day(2020, 3, 16))]
    );
    // 100 units over 2 days is 50 a day, clamped to 4: a supply of 25 days.
    assert!(
        observed(&[(day(2020, 1, 1), 100.0), (day(2020, 1, 3), 100.0)])[0].1 > day(2020, 1, 26)
    );
    // The last dispensing has no next one and uses the default.
    assert_eq!(
        observed(&[(day(2020, 1, 1), 100.0)]),
        [(day(2020, 1, 1), day(2020, 2, 20))]
    );
}

#[test]
fn a_fixed_duration_counts_each_dispensing() {
    let fixed = |rows: &[(i64, f64)]| {
        periods_of(
            rows,
            N02,
            DoseRule::Fixed { days: 30.0 },
            Stock::CarryForward,
            &[],
        )
    };
    let p = fixed(&[(0, 1.0), (30, 5.0), (100, 2.0), (100, 2.0)]);
    assert_eq!(spans(&p), [(0, 60), (100, 160)]);
    assert_eq!(p.n_dispensing_days, [2, 1]);
    assert_eq!(p.quantity, [6.0, 4.0]);
}

#[test]
fn the_reported_dose_is_the_quantity_over_the_durations() {
    let p = periods_of(
        &[(0, 100.0)],
        N02,
        DoseRule::Default,
        Stock::CarryForward,
        &[],
    );
    assert_eq!(p.daily_dose, [Some(2.0)]);
    assert_eq!(p.quantity, [100.0]);
    assert_eq!(p.n_dispensing_days, [1]);
    // Observed: the quantity-weighted harmonic mean of the doses.
    let p = periods_of(
        &[(0, 100.0), (50, 100.0), (100, 100.0)],
        N02,
        DoseRule::Observed,
        Stock::CarryForward,
        &[],
    );
    // Doses clamp(2, 1, 4) = 2, 2, and the last uses the default 2.
    assert_eq!(p.daily_dose, [Some(2.0)]);
}

#[test]
fn people_and_groups_are_separate_and_unassigned_rows_are_counted() {
    let days = [0, 0, 5, 0, 3];
    let quantity = [100.0; 5];
    let group = [Some(0), Some(1), Some(0), None, Some(0)];
    let d = Dispensings {
        person_offsets: &[0, 2, 5],
        group: &group,
        day: &days,
        quantity: &quantity,
    };
    let assumptions = [
        N02,
        Assumption {
            dose_default: 1.0,
            ..N02
        },
    ];
    let s = Supply {
        assumptions: &assumptions,
        rule: DoseRule::Default,
        stock: Stock::CarryForward,
        pauses: None,
    };
    let p = exposure_periods(&d, &s).unwrap();
    assert_eq!(p.unassigned, 1);
    assert_eq!(p.person, [0, 0, 1]);
    assert_eq!(p.group, [0, 1, 0]);
    // Person 1's rows are unsorted: days 5 and 3 become one period from day 3.
    assert_eq!(p.start, [0, 0, 3]);
    assert_eq!(spans(&p), [(0, 50), (0, 100), (3, 85)]);
}

#[test]
fn refusals() {
    let days = [0];
    let one = |q: f64, g: Option<u32>, a: &[Assumption]| {
        let group = [g];
        let quantity = [q];
        let d = Dispensings {
            person_offsets: &[0, 1],
            group: &group,
            day: &days,
            quantity: &quantity,
        };
        let s = Supply {
            assumptions: a,
            rule: DoseRule::Default,
            stock: Stock::CarryForward,
            pauses: None,
        };
        exposure_periods(&d, &s)
    };
    assert!(one(1.0, Some(0), &[]).is_err(), "no assumptions");
    for q in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let e = one(q, Some(0), &[N02]).unwrap_err().to_string();
        assert!(e.contains("row 0"), "{e}");
    }
    assert!(
        one(1.0, Some(1), &[N02]).is_err(),
        "group without an assumption"
    );
    for bad in [
        Assumption {
            dose_default: 0.0,
            ..N02
        },
        Assumption {
            dose_min: 5.0,
            ..N02
        },
        Assumption {
            max_stock_days: -1.0,
            ..N02
        },
        Assumption {
            max_stock_days: f64::NAN,
            ..N02
        },
        Assumption {
            grace_days: f64::INFINITY,
            ..N02
        },
    ] {
        assert!(one(1.0, Some(0), &[bad]).is_err());
    }
    assert!(one(1.0, Some(0), &[N02]).is_ok());
    assert!(one(1.0, None, &[N02]).unwrap().is_empty());
}

/// The closed form of the module docs: without cap, pauses and grace,
/// `e_i = S_i + max_{j<=i}(t_j - S_{j-1})` with `S` the cumulative duration.
#[test]
fn the_closed_form_holds_without_cap_pauses_and_grace() {
    let mut rng = SplitMix64::new(7);
    let a = Assumption {
        max_stock_days: f64::INFINITY,
        ..N02
    };
    for _ in 0..200 {
        let n = 1 + rng.below(12) as usize;
        let mut t = 0_i64;
        let rows: Vec<(i64, f64)> = (0..n)
            .map(|_| {
                t += 1 + rng.below(60) as i64;
                (t, 2.0 * (1 + rng.below(40)) as f64)
            })
            .collect();
        let p = periods_of(&rows, a, DoseRule::Default, Stock::CarryForward, &[]);
        // Walk the closed form and cut it into periods where the supply ran out.
        let mut cumulative = 0.0;
        let mut best = f64::NEG_INFINITY;
        let mut ends = Vec::new();
        for &(day, quantity) in &rows {
            best = best.max(day as f64 - cumulative);
            cumulative += quantity / 2.0;
            ends.push(cumulative + best);
        }
        // Period ends are the closed-form value at the last row of each period.
        let mut expected = Vec::new();
        for (i, end) in ends.iter().enumerate() {
            let next_gap = rows.get(i + 1).is_none_or(|r| r.0 as f64 > *end);
            if next_gap {
                expected.push(*end);
            }
        }
        assert_eq!(expected.len(), p.len());
        for (got, want) in p.end.iter().zip(&expected) {
            assert!((got - want).abs() <= 4.0 * n as f64 * f64::EPSILON * want.abs().max(1.0));
        }
    }
}

/// A port of the Python loop, which rescans the pauses for every supply.
fn advance_by_rescan(start: f64, days: f64, pauses: &[(i64, i64)]) -> f64 {
    let mut remaining = days;
    let mut position = start;
    for &(a, b) in pauses {
        if (b as f64) <= position {
            continue;
        }
        if (a as f64) >= position + remaining {
            break;
        }
        remaining -= (a as f64 - position).max(0.0);
        position = position.max(b as f64);
    }
    position + remaining
}

#[test]
fn the_pause_pointer_equals_a_rescan_bit_for_bit() {
    let mut rng = SplitMix64::new(11);
    for _ in 0..300 {
        let mut pauses: Vec<(i64, i64)> = (0..rng.below(8))
            .map(|_| {
                let a = rng.below(300) as i64;
                (a, a + rng.below(30) as i64)
            })
            .collect();
        pauses.sort_unstable();
        let n = 1 + rng.below(10) as usize;
        let mut t = 0_i64;
        let rows: Vec<(i64, f64)> = (0..n)
            .map(|_| {
                t += rng.below(40) as i64;
                (t, 1.0 + rng.below(150) as f64)
            })
            .collect();
        let p = periods_of(&rows, N02, DoseRule::Default, Stock::CarryForward, &pauses);
        // The Python recurrence with a rescan, over the same-day sums.
        let mut merged: Vec<(i64, f64)> = Vec::new();
        for &(d, q) in &rows {
            match merged.last_mut() {
                Some(l) if l.0 == d => l.1 += q,
                _ => merged.push((d, q)),
            }
        }
        let mut ends = Vec::new();
        let mut supply_end: Option<f64> = None;
        for &(d, q) in &merged {
            let t = d as f64;
            let start = match supply_end {
                Some(e) if t <= e => t + (e - t).min(30.0),
                _ => {
                    if let Some(e) = supply_end {
                        ends.push(e);
                    }
                    t
                }
            };
            supply_end = Some(advance_by_rescan(start, q / 2.0, &pauses));
        }
        ends.push(supply_end.unwrap());
        assert_eq!(
            p.end.iter().map(|e| e.to_bits()).collect::<Vec<_>>(),
            ends.iter().map(|e| e.to_bits()).collect::<Vec<_>>()
        );
    }
}

#[test]
fn cumulative_dose_at_dispensing_and_prorated() {
    let rows = [(0_i64, 100.0), (10, 100.0), (200, 100.0)];
    let days: Vec<i64> = rows.iter().map(|r| r.0).collect();
    let quantity: Vec<f64> = rows.iter().map(|r| r.1).collect();
    let group = [Some(0); 3];
    let d = Dispensings {
        person_offsets: &[0, 3],
        group: &group,
        day: &days,
        quantity: &quantity,
    };
    let w = Windows {
        person: &[0, 0],
        group: &[0, 0],
        start: &[0, 25],
        end: &[11, 75],
    };
    let at = cumulative_dose(&d, &w, 0, DoseMode::AtDispensing).unwrap();
    assert_eq!(at, [200.0, 0.0]);
    // A lag of 5 days moves the window to [-5, 6) and [20, 70).
    let lagged = cumulative_dose(&d, &w, 5, DoseMode::AtDispensing).unwrap();
    assert_eq!(lagged, [100.0, 0.0]);
    let assumptions = [N02];
    let s = Supply {
        assumptions: &assumptions,
        rule: DoseRule::Default,
        stock: Stock::CarryForward,
        pauses: None,
    };
    // Supplies: [0, 50) with 100 units; [40, 90) with 100 units, started 40
    // days in because the carried stock of 40 days is capped at 30 days from the
    // dispensing on day 10; [200, 250) with 100 units.
    let prorated = cumulative_dose(&d, &w, 0, DoseMode::ProratedOverSupply(&s)).unwrap();
    assert_eq!(prorated, [11.0 * 2.0, 25.0 * 2.0 + 35.0 * 2.0]);
    assert!(cumulative_dose(&d, &w, -1, DoseMode::AtDispensing).is_err());
}

#[test]
fn exposure_on_a_day_with_a_lag() {
    let p = periods_of(
        &[(10, 100.0)],
        N02,
        DoseRule::Default,
        Stock::CarryForward,
        &[],
    );
    let q = PointTable {
        person: &[0, 0, 0, 0, 1],
        group: &[0, 0, 0, 0, 0],
        day: &[9, 10, 59, 60, 30],
    };
    assert_eq!(
        exposed_on(&p, &q, 0).unwrap(),
        [false, true, true, false, false]
    );
    // With a lag of 5 the status on day t is the status on day t - 5.
    assert_eq!(
        exposed_on(&p, &q, 5).unwrap(),
        [false, false, true, true, false]
    );
}

#[test]
fn results_do_not_depend_on_the_thread_count() {
    assert_same_on_threads(|| {
        let mut rng = SplitMix64::new(3);
        let rows: Vec<(i64, f64)> = (0..500)
            .map(|i| (i * 7 + rng.below(5) as i64, 1.0 + rng.below(120) as f64))
            .collect();
        let p = periods_of(
            &rows,
            N02,
            DoseRule::Observed,
            Stock::CarryForward,
            &[(100, 130)],
        );
        p.end
    });
}

#[test]
fn a_fractional_end_rounds_up_to_the_first_day_without_supply() {
    // 49 units at 10 a day last 4.9 days: days 0 to 4 have supply left.
    let a = Assumption {
        dose_default: 10.0,
        ..N02
    };
    let p = periods_of(&[(0, 49.0)], a, DoseRule::Default, Stock::Discard, &[]);
    assert!((p.end[0] - 4.9).abs() < 1e-12);
    assert_eq!((p.end_day(0), p.last_day(0)), (5, 4));
    let q = PointTable {
        person: &[0, 0],
        group: &[0, 0],
        day: &[4, 5],
    };
    assert_eq!(exposed_on(&p, &q, 0).unwrap(), [true, false]);
    let set = p.span_set(0, 1).unwrap();
    assert_eq!(
        (set.start.as_slice(), set.end.as_slice()),
        (&[0][..], &[5][..])
    );
}

/// Every day of a random history is in the set of a group exactly when
/// `exposed_on` reads it as exposed, for fractional and whole supplies.
#[test]
fn span_sets_agree_with_exposed_on_day_by_day() {
    let mut rng = SplitMix64::new(0x5EED_0010);
    for _ in 0..300 {
        let persons = 1 + (rng.next_u64() % 4) as usize;
        let (mut day, mut quantity, mut group, mut offsets) =
            (Vec::new(), Vec::new(), Vec::new(), vec![0]);
        for _ in 0..persons {
            let mut t = (rng.next_u64() % 20) as i64;
            for _ in 0..(rng.next_u64() % 6) {
                day.push(t);
                quantity.push(1.0 + (rng.next_u64() % 60) as f64 / 3.0);
                group.push(Some((rng.next_u64() % 2) as u32));
                t += (rng.next_u64() % 40) as i64;
            }
            offsets.push(day.len());
        }
        let a = Assumption {
            dose_default: 1.5,
            dose_min: 0.5,
            dose_max: 3.0,
            max_stock_days: 10.0,
            grace_days: (rng.next_u64() % 5) as f64 / 2.0,
        };
        let assumptions = [a, a];
        let d = Dispensings {
            person_offsets: &offsets,
            group: &group,
            day: &day,
            quantity: &quantity,
        };
        let rule = if rng.next_u64().is_multiple_of(2) {
            DoseRule::Default
        } else {
            DoseRule::Observed
        };
        let s = Supply {
            assumptions: &assumptions,
            rule,
            stock: Stock::CarryForward,
            pauses: None,
        };
        let p = exposure_periods(&d, &s).unwrap();
        for g in 0..2_u32 {
            let set = p.span_set(g, persons).unwrap();
            for person in 0..persons {
                let rows = set.key_offsets[person]..set.key_offsets[person + 1];
                for t in -2..400_i64 {
                    let in_set = rows.clone().any(|r| set.start[r] <= t && t < set.end[r]);
                    let q = PointTable {
                        person: &[person as u32],
                        group: &[g],
                        day: &[t],
                    };
                    assert_eq!(in_set, exposed_on(&p, &q, 0).unwrap()[0]);
                }
            }
        }
    }
}

#[test]
fn span_set_refuses_a_person_out_of_range() {
    let p = periods_of(&[(0, 10.0)], N02, DoseRule::Default, Stock::Discard, &[]);
    assert!(p.span_set(0, 0).is_err());
    assert!(p.span_set(1, 1).unwrap().is_empty());
}
