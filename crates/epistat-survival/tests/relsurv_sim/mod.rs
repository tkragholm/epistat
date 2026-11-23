//! A simulated cohort with a daily rate table by age, sex and calendar date,
//! for the checks of `relsurv` that need no R reference.

#![allow(dead_code, reason = "each test file uses part of the cohort")]

use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::ratetable::{RateDim, RateTable};
use epistat_survival::relsurv::ExpectedHazard;

/// The first day of the table, 1980-01-01 in days since 1970.
pub const DAY0: f64 = 3652.0;

/// Gompertz in age, a factor for sex and a trend in the calendar year, with
/// whole years of 365 days, the last cell of each time dimension reaching far.
pub fn table() -> RateTable {
    let mut age: Vec<f64> = (0..=110).map(|a| f64::from(a) * 365.0).collect();
    age.push(1e9);
    let mut year: Vec<f64> = (0..=45).map(|y| DAY0 + f64::from(y) * 365.0).collect();
    year.push(1e9);
    let mut rates = Vec::new();
    for y in 0..year.len() - 1 {
        for s in 0..2 {
            for a in 0..age.len() - 1 {
                let log =
                    -10.5 + 0.09 * a as f64 + if s == 0 { 0.35 } else { 0.0 } - 0.012 * y as f64;
                rates.push(log.exp() / 365.0);
            }
        }
    }
    RateTable::new(
        vec![
            RateDim::Time { cuts: age },
            RateDim::Factor { levels: 2 },
            RateDim::Time { cuts: year },
        ],
        rates,
        1.0,
    )
    .unwrap()
}

pub struct Cohort {
    pub age0: Vec<f64>,
    pub date0: Vec<f64>,
    pub sex: Vec<u32>,
    pub x1: Vec<f64>,
    pub x2: Vec<f64>,
    pub group: Vec<u32>,
    pub start: Vec<f64>,
    pub time: Vec<f64>,
    pub status: Vec<i32>,
}

/// `n` persons aged 40 to 80 at diagnosis from 1985 to 2000, with an excess
/// hazard of about 0.2 a year, deaths on whole days, censoring from 150 to
/// 2650 days, and, with `entry`, a third of them entering late.
pub fn cohort(n: usize, seed: u64, entry: bool) -> Cohort {
    let mut rng = Stream::seeded(seed);
    let mut c = Cohort {
        age0: Vec::new(),
        date0: Vec::new(),
        sex: Vec::new(),
        x1: Vec::new(),
        x2: Vec::new(),
        group: Vec::new(),
        start: Vec::new(),
        time: Vec::new(),
        status: Vec::new(),
    };
    for _ in 0..n {
        let x1 = rng.normal();
        let x2 = f64::from(u8::from(rng.uniform() < 0.4));
        let t = 1.0 + (2000.0 * (-(0.4 * x1 + 0.3 * x2)).exp() * rng.exp1()).floor();
        let cens = 150.5 + (rng.uniform() * 2500.0).floor();
        let time = t.min(cens);
        let start = if entry && rng.below(3) == 0 {
            (0.5 * time).floor() + 0.25
        } else {
            0.0
        };
        c.age0
            .push((rng.uniform() * 40.0 * 365.0).floor() + 40.0 * 365.0);
        c.date0
            .push(DAY0 + 5.0 * 365.0 + (rng.uniform() * 15.0 * 365.0).floor());
        c.sex.push(u32::try_from(rng.below(2)).unwrap());
        c.x1.push(x1);
        c.x2.push(x2);
        c.group.push(u32::try_from(rng.below(2)).unwrap());
        c.start.push(start);
        c.time.push(time);
        c.status.push(i32::from(t <= cens));
    }
    c
}

impl Cohort {
    pub fn data(&self, strata: bool) -> SurvData<'_> {
        let mut b = SurvData::builder(&self.time, &self.status).start(&self.start);
        if strata {
            b = b.stratum(&self.group);
        }
        b.build().unwrap()
    }

    pub fn expected(&self, table: &RateTable, data: &SurvData<'_>) -> ExpectedHazard {
        ExpectedHazard::from_table(table, data, &[&self.age0, &self.date0], &[&self.sex]).unwrap()
    }
}
