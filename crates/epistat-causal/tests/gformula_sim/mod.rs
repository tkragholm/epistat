//! Simulated person-period data and models for the g-formula's own tests.
//!
//! Each person has a baseline `V`, and in each period a binary covariate `L`,
//! a normal covariate `Z` and a binary treatment `A`, with a survival outcome
//! `Y` and a competing event `D`; rows end at an event or at censoring. `Yend`
//! is an outcome at the last period, NaN elsewhere.

use epistat_causal::gformula::{
    Action, Competing, Covariate, CovariateKind, Estimate, GFormula, History, Intervention,
    Outcome, PersonPeriods, Rule, Term,
};
use epistat_rng::Stream;

pub struct Sim {
    pub id: Vec<u64>,
    pub period: Vec<usize>,
    pub v: Vec<f64>,
    pub l: Vec<f64>,
    pub z: Vec<f64>,
    pub a: Vec<f64>,
    pub y: Vec<bool>,
    pub d: Vec<bool>,
    pub y_end: Vec<f64>,
}

fn expit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn bernoulli(s: &mut Stream, p: f64) -> f64 {
    f64::from(u8::from(s.uniform() < p))
}

/// `n` persons over `periods` periods.
pub fn simulate(n: usize, periods: usize, seed: u64) -> Sim {
    let mut s = Stream::seeded(seed);
    let mut sim = Sim {
        id: Vec::new(),
        period: Vec::new(),
        v: Vec::new(),
        l: Vec::new(),
        z: Vec::new(),
        a: Vec::new(),
        y: Vec::new(),
        d: Vec::new(),
        y_end: Vec::new(),
    };
    for i in 0..n {
        let v = bernoulli(&mut s, 0.4);
        let (mut lp, mut zp, mut ap) = (0.0, 0.0, 0.0);
        for k in 0..periods {
            let l = if k > 0 && lp > 0.5 {
                1.0
            } else {
                bernoulli(&mut s, expit(-1.2 + 0.5 * zp - 0.6 * ap + 0.3 * v))
            };
            let z = 0.5 * zp + 0.6 * l - 0.3 * ap + 0.2 * v + s.normal();
            let a = bernoulli(&mut s, expit(-0.8 + 1.5 * ap + 0.7 * l + 0.4 * z));
            let d = s.uniform() < expit(-3.5 + 0.4 * l + 0.1 * k as f64);
            let y = !d && s.uniform() < expit(-2.8 + 0.7 * l + 0.3 * z - 0.6 * a + 0.3 * v);
            let y_end = if k + 1 == periods {
                bernoulli(&mut s, expit(-0.5 + 0.6 * l + 0.4 * z - 0.5 * a))
            } else {
                f64::NAN
            };
            sim.id.push(i as u64 * 3 + 7);
            sim.period.push(k);
            sim.v.push(v);
            sim.l.push(l);
            sim.z.push(z);
            sim.a.push(a);
            sim.y.push(y);
            sim.d.push(d);
            sim.y_end.push(y_end);
            if y || d || s.uniform() < 0.03 {
                break;
            }
            (lp, zp, ap) = (l, z, a);
        }
    }
    sim
}

/// Survivors only, for an end-of-follow-up outcome: persons with every
/// period and no event.
pub fn complete(sim: &Sim, periods: usize) -> Sim {
    let mut out = simulate(0, periods, 0);
    let mut r = 0;
    while r < sim.id.len() {
        let mut e = r;
        while e < sim.id.len() && sim.id[e] == sim.id[r] {
            e += 1;
        }
        if e - r == periods {
            for i in r..e {
                out.id.push(sim.id[i]);
                out.period.push(sim.period[i]);
                out.v.push(sim.v[i]);
                out.l.push(sim.l[i]);
                out.z.push(sim.z[i]);
                out.a.push(sim.a[i]);
                out.y.push(false);
                out.d.push(false);
                out.y_end.push(sim.y_end[i]);
            }
        }
        r = e;
    }
    out
}

pub const L: usize = 0;
pub const Z: usize = 1;
pub const A: usize = 2;

pub fn covariates() -> Vec<Covariate> {
    use Term::{Baseline, Current, Intercept, Lag, LaggedCumulativeAverage};
    vec![
        Covariate {
            kind: CovariateKind::Binary,
            terms: vec![Intercept, Lag(Z, 1), Lag(A, 1), Baseline(0)],
            restriction: Some(epistat_causal::gformula::Restriction::Absorbing),
        },
        Covariate {
            kind: CovariateKind::Normal,
            terms: vec![Intercept, Lag(Z, 1), Current(L), Lag(A, 1), Baseline(0)],
            restriction: None,
        },
        Covariate {
            kind: CovariateKind::Binary,
            terms: vec![
                Intercept,
                Lag(A, 1),
                Current(L),
                Current(Z),
                LaggedCumulativeAverage(Z, 1),
            ],
            restriction: None,
        },
    ]
}

pub fn outcome_terms() -> Vec<Term> {
    use Term::{Baseline, CumulativeAverage, Current, Intercept, Period};
    vec![
        Intercept,
        Current(L),
        CumulativeAverage(Z),
        Current(A),
        Baseline(0),
        Term::product(Current(A), Baseline(0)),
        Period,
    ]
}

pub fn competing_terms() -> Vec<Term> {
    vec![Term::Intercept, Term::Current(L), Term::Period]
}

pub fn interventions() -> Vec<Intervention> {
    vec![
        Intervention::set(A, 0.0),
        Intervention::set(A, 1.0),
        Intervention {
            rules: vec![Rule {
                variable: A,
                action: Action::Dynamic(Box::new(
                    |h: &History<'_>, natural: f64| {
                        if h.current(Z) > 0.8 { 1.0 } else { natural }
                    },
                )),
                periods: Some(vec![1, 2, 3]),
            }],
        },
        Intervention {
            rules: vec![Rule {
                variable: Z,
                action: Action::Threshold {
                    lower: f64::NEG_INFINITY,
                    upper: 0.0,
                },
                periods: None,
            }],
        },
    ]
}

impl Sim {
    pub fn baseline(&self) -> [&[f64]; 1] {
        [&self.v]
    }

    pub fn time_varying(&self) -> [&[f64]; 3] {
        [&self.l, &self.z, &self.a]
    }

    pub fn data<'a>(&'a self, baseline: &'a [&'a [f64]], tv: &'a [&'a [f64]]) -> PersonPeriods<'a> {
        PersonPeriods {
            id: &self.id,
            period: &self.period,
            baseline,
            time_varying: tv,
        }
    }

    pub fn survival(&self) -> Outcome<'_> {
        Outcome::Survival {
            event: &self.y,
            terms: outcome_terms(),
            competing: Some(Competing {
                event: &self.d,
                terms: competing_terms(),
            }),
            censoring: None,
        }
    }

    pub fn binary_end(&self) -> Outcome<'_> {
        Outcome::BinaryEnd {
            value: &self.y_end,
            terms: vec![
                Term::Intercept,
                Term::Current(L),
                Term::Current(Z),
                Term::Current(A),
            ],
        }
    }
}

fn push(out: &mut Vec<f64>, e: &Estimate) {
    for x in [e.estimate, e.monte_carlo_se, e.se, e.lower, e.upper] {
        out.push(x.unwrap_or(f64::NAN));
    }
}

/// Every number of a result, in a fixed order.
pub fn flatten(g: &GFormula) -> Vec<f64> {
    let mut out = Vec::new();
    for s in &g.strategies {
        for e in s.value.iter().chain(&s.ratio).chain(&s.difference) {
            push(&mut out, e);
        }
        out.push(s.intervened_persons);
        out.push(s.intervened_person_periods);
    }
    for m in g.models.covariates.iter().flatten() {
        out.extend(&m.coefficients);
        out.push(m.rmse.unwrap_or(f64::NAN));
    }
    out.extend(&g.models.outcome.coefficients);
    if let Some(m) = &g.models.competing {
        out.extend(&m.coefficients);
    }
    out.extend(g.observed.iter().map(|o| o.unwrap_or(f64::NAN)));
    out.push(g.failed_replicates as f64);
    out
}
