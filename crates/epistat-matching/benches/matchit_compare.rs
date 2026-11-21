//! Wall time of the matching methods on a simulated register cohort, to be
//! set against MatchIt on the same data (`benches/matchit/matchit_bench.R`).
//!
//! One process times one method on one cohort size, so that
//! `/usr/bin/time -l` gives the peak memory of that run alone.
//! `benches/matchit/run.sh` is the driver. The environment sets the work:
//!
//! * `SCRATCH`: directory for the cohort file and the results. Nothing is
//!   written inside the repository.
//! * `N`: number of units, for example 100000.
//! * `METHOD`: `gen` writes the cohort file and stops. `load` reads the cohort
//!   and stops, which gives the memory of the data alone. Any other name is
//!   one of the methods below.
//! * `REPS`: timed repetitions, default 3. The median is reported.
//!
//! The cohort is simulated from a fixed seed by `gen` and written to
//! `SCRATCH/cohort_N.csv`. Every later run, in Rust and in R, reads that file,
//! so both sides match identical numbers. Each run writes `SCRATCH/out/rust_METHOD_N.csv`
//! with its pairs or subclasses and appends a line to `SCRATCH/out/rust_times.csv`.
//!
//! Only the matching is timed: the propensity fit, the distances and the
//! selection, as one call of `run_spec`. Reading the file and building the
//! records are not.

use chrono::{Days, NaiveDate};
use epistat_matching::prelude::*;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::io::Write as _;
use std::path::PathBuf;
use std::time::Instant;

const SEED: u64 = 20_261_121;
const COVARIATES: [&str; 5] = ["x1", "x2", "x3", "x4", "x5"];
/// Units per fine stratum `g`, small enough for optimal matching in R.
const FINE_STRATUM: usize = 400;
const CALIPER_STD: f64 = 0.2;
/// Raw caliper on the probability scale for optimal and full matching, which
/// MatchIt does not apply itself.
const CALIPER_RAW_DEFAULT: f64 = 0.2;

fn caliper_raw() -> f64 {
    std::env::var("CALIPER_RAW")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(CALIPER_RAW_DEFAULT)
}

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

fn scratch() -> PathBuf {
    PathBuf::from(
        std::env::var("SCRATCH").expect("SCRATCH names a directory outside the repository"),
    )
}

/// The cohort: a treated share of about 10 percent that depends on five
/// covariates, twenty coarse strata `s`, fine strata `g` of about 400 units,
/// and for risk-set matching a case time for 5 percent of the units and a
/// death time for 20 percent of the rest. Times are days from a common birth
/// date.
fn generate(n: usize, path: &std::path::Path) {
    let mut rng = Rng(SEED ^ n as u64);
    let fine = (n / FINE_STRATUM).max(1) as u64;
    let mut out = String::with_capacity(n * 80);
    out.push_str("id,treat,x1,x2,x3,x4,x5,s,g,t_case,t_death\n");
    for i in 1..=n {
        let x1 = round3(rng.normal());
        let x2 = round3(rng.normal());
        let x3 = u8::from(rng.uniform() < 0.4);
        let x4 = round3((0.5 * rng.normal()).exp());
        let x5 = (rng.uniform() * 5.0).floor();
        let s = rng.next_u64() % 20;
        let g = rng.next_u64() % fine;
        let lp = -2.4 + 0.3 * x1 - 0.2 * x2 + 0.3 * f64::from(x3) + 0.15 * x4 + 0.08 * x5;
        let treat = u8::from(rng.uniform() < 1.0 / (1.0 + (-lp).exp()));
        let case = rng.uniform() < 0.05;
        let t_case = if case {
            (100 + rng.next_u64() % 1900).to_string()
        } else {
            String::new()
        };
        let death = !case && rng.uniform() < 0.2;
        let t_death = if death {
            (100 + rng.next_u64() % 2090).to_string()
        } else {
            String::new()
        };
        writeln!(
            out,
            "{i},{treat},{x1},{x2},{x3},{x4},{x5},s{s},g{g},{t_case},{t_death}"
        )
        .expect("write to string");
    }
    fs::write(path, out).expect("write cohort");
}

struct Row {
    id: String,
    treat: bool,
    x: [f64; 5],
    s: String,
    g: String,
    t_case: Option<u64>,
    t_death: Option<u64>,
}

fn read_cohort(path: &std::path::Path) -> Vec<Row> {
    let text = fs::read_to_string(path).expect("read cohort");
    text.lines()
        .skip(1)
        .map(|line| {
            let f: Vec<&str> = line.split(',').collect();
            let num = |k: usize| f[k].parse::<f64>().expect("number");
            Row {
                id: f[0].to_string(),
                treat: f[1] == "1",
                x: [num(2), num(3), num(4), num(5), num(6)],
                s: f[7].to_string(),
                g: f[8].to_string(),
                t_case: f[9].parse().ok(),
                t_death: f[10].parse().ok(),
            }
        })
        .collect()
}

fn origin() -> NaiveDate {
    NaiveDate::from_ymd_opt(2000, 1, 1).expect("valid date")
}

fn record(row: &Row) -> BalanceRecord {
    let mut r = BalanceRecord::new(row.id.clone(), origin());
    for (name, value) in COVARIATES.iter().zip(row.x) {
        r.covariates
            .insert((*name).to_string(), CovariateValue::Numeric(value));
        r.core.numerics.insert((*name).to_string(), value);
    }
    r.core.strata.insert("s".into(), row.s.clone());
    r.core.strata.insert("g".into(), row.g.clone());
    r.core.strata.insert("x3".into(), format!("{}", row.x[2]));
    r.core.strata.insert("x5".into(), format!("{}", row.x[4]));
    r
}

fn propensity(caliper: Option<f64>) -> DistanceSpec {
    DistanceSpec::Propensity {
        covariates: CovariatesSpec {
            names: COVARIATES.iter().map(|s| (*s).to_string()).collect(),
            missing: MissingSpec::Error,
        },
        link: LinkSpec::Logit,
        estimator: EstimatorSpec::Glm,
        scale: ScaleSpec::Probability,
        sampling_weights: None,
        caliper,
    }
}

fn wide() -> i32 {
    36_500
}

/// The spec of a method name, or `None` for a name that is not a spec method.
fn spec_for(method: &str) -> Option<MatchSpec> {
    let base = MatchSpec {
        method: MatchMethod::Nearest,
        distance: propensity(None),
        birth_window_days: wide(),
        ..MatchSpec::default()
    };
    Some(match method {
        "nn_ps" => base,
        "nn_ps_cal" => MatchSpec {
            distance: propensity(Some(CALIPER_STD)),
            std_caliper: StdCaliperSpec::All(true),
            ..base
        },
        "nn_strata" => MatchSpec {
            exact: vec!["s".into()],
            ..base
        },
        "nn_cal_strata" => MatchSpec {
            distance: propensity(Some(CALIPER_STD)),
            std_caliper: StdCaliperSpec::All(true),
            exact: vec!["s".into()],
            ..base
        },
        "nn_1to3" => MatchSpec { ratio: 3, ..base },
        "nn_mahal" => MatchSpec {
            distance: DistanceSpec::Mahalanobis {
                covariates: CovariatesSpec {
                    names: COVARIATES.iter().map(|s| (*s).to_string()).collect(),
                    missing: MissingSpec::Error,
                },
                covariance: CovarianceSpec::default(),
                transform: TransformSpec::default(),
                caliper: None,
            },
            ..base
        },
        "subclass" => MatchSpec {
            method: MatchMethod::Subclass,
            subclass: SubclassSpec {
                n: 6,
                min_n: 1,
                reference: None,
            },
            ..base
        },
        "exact" => MatchSpec {
            method: MatchMethod::Exact,
            distance: DistanceSpec::default(),
            exact: vec!["s".into(), "x3".into(), "x5".into()],
            ..base
        },
        "cem" => MatchSpec {
            method: MatchMethod::Cem,
            distance: DistanceSpec::default(),
            cem: CemSpec {
                variables: vec![
                    CemVariableSpec::Numeric {
                        name: "x1".into(),
                        binning: BinningSpec::Quantiles { n: 5 },
                    },
                    CemVariableSpec::Numeric {
                        name: "x2".into(),
                        binning: BinningSpec::Quantiles { n: 5 },
                    },
                    CemVariableSpec::Categorical {
                        name: "x3".into(),
                        groups: vec![],
                    },
                    CemVariableSpec::Categorical {
                        name: "x5".into(),
                        groups: vec![],
                    },
                ],
                k2k: None,
            },
            ..base
        },
        "quick" => MatchSpec {
            method: MatchMethod::Quick,
            ..base
        },
        "optimal_cal" => MatchSpec {
            method: MatchMethod::Optimal,
            distance: propensity(Some(caliper_raw())),
            exact: vec!["g".into()],
            ..base
        },
        "full_cal" => MatchSpec {
            method: MatchMethod::Full,
            distance: propensity(Some(caliper_raw())),
            exact: vec!["g".into()],
            ..base
        },
        _ => return None,
    })
}

fn median(times: &mut [f64]) -> f64 {
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

/// Units in no pair or subclass are absent from `assign`.
struct Result {
    /// Wall seconds of each repetition.
    times: Vec<f64>,
    /// Number of units matched or kept.
    matched: usize,
    /// Pairs as (anchor, control) ids, or unit and subclass.
    lines: Vec<(String, String)>,
    header: &'static str,
}

fn write_lines(path: &std::path::Path, header: &str, lines: &[(String, String)]) {
    let mut out = std::io::BufWriter::new(fs::File::create(path).expect("create output"));
    writeln!(out, "{header}").expect("write");
    for (a, b) in lines {
        writeln!(out, "{a},{b}").expect("write");
    }
}

fn output_lines(out: &SpecOutput) -> (usize, Vec<(String, String)>, &'static str) {
    match out {
        SpecOutput::Pairs { outcome, .. } => (
            outcome.matched_cases,
            outcome
                .pairs
                .iter()
                .map(|p| (p.case_id.clone(), p.control_id.clone()))
                .collect(),
            "anchor,control",
        ),
        SpecOutput::Exact(o) => (
            o.retained_units(),
            subclasses(&o.assignments),
            "id,subclass",
        ),
        SpecOutput::Quick(o) => (
            o.retained_units(),
            subclasses(&o.assignments),
            "id,subclass",
        ),
        SpecOutput::Full(o) => (
            o.retained_units(),
            subclasses(&o.assignments),
            "id,subclass",
        ),
        SpecOutput::Subclass { outcome, .. } => (
            outcome.assignments.len(),
            subclasses(&outcome.assignments),
            "id,subclass",
        ),
        SpecOutput::ExposureDensity(_) | SpecOutput::Trials(_) => {
            unreachable!("the benchmark runs no exposure density or trial specification")
        }
    }
}

fn subclasses(map: &std::collections::HashMap<String, usize>) -> Vec<(String, String)> {
    let sorted: BTreeMap<u64, usize> = map
        .iter()
        .map(|(k, v)| (k.parse::<u64>().expect("numeric id"), *v))
        .collect();
    sorted
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn run_standard(
    method: &str,
    spec: &MatchSpec,
    anchors: &[BalanceRecord],
    candidates: &[BalanceRecord],
    reps: usize,
) -> Result {
    let mut times = Vec::new();
    let mut last = None;
    for _ in 0..reps {
        let start = Instant::now();
        let out = run_spec(
            spec,
            SpecInput::Standard {
                anchors,
                candidates,
            },
        )
        .unwrap_or_else(|e| panic!("{method}: {e}"));
        times.push(start.elapsed().as_secs_f64());
        last = Some(out);
    }
    let (matched, mut lines, header) = output_lines(&last.expect("one repetition"));
    if header == "anchor,control" {
        lines.sort_by_key(|(a, b)| (a.parse::<u64>().unwrap_or(0), b.parse::<u64>().unwrap_or(0)));
    }
    Result {
        times,
        matched,
        lines,
        header,
    }
}

fn run_risk_set(rows: &[Row], reps: usize, replace: bool) -> Result {
    let cohort: Vec<RoleTransitionRecord<BalanceRecord>> = rows
        .iter()
        .map(|row| {
            let mut r = record(row);
            r.core.death_date = row.t_death.map(|d| origin() + Days::new(d));
            let when = row.t_case.map(|d| origin() + Days::new(d));
            RoleTransitionRecord::from_record(r, when)
        })
        .collect();
    let spec = MatchSpec {
        mode: MatchMode::RiskSet,
        method: MatchMethod::Random,
        exact: vec!["s".into()],
        birth_window_days: wide(),
        require_alive: true,
        age_limit_years: Some(6),
        replace,
        seed: 7,
        ..MatchSpec::default()
    };
    let mut times = Vec::new();
    let mut last = None;
    for _ in 0..reps {
        let start = Instant::now();
        let out = run_spec(&spec, SpecInput::RiskSet { cohort: &cohort })
            .unwrap_or_else(|e| panic!("risk set: {e}"));
        times.push(start.elapsed().as_secs_f64());
        last = Some(out);
    }
    let (matched, lines, header) = output_lines(&last.expect("one repetition"));
    Result {
        times,
        matched,
        lines,
        header,
    }
}

fn main() {
    let n: usize = std::env::var("N")
        .ok()
        .and_then(|v| v.parse().ok())
        .expect("N");
    let method = std::env::var("METHOD").expect("METHOD");
    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let dir = scratch();
    fs::create_dir_all(dir.join("out")).expect("scratch");
    let cohort_path = dir.join(format!("cohort_{n}.csv"));
    if method == "gen" {
        generate(n, &cohort_path);
        return;
    }
    let rows = read_cohort(&cohort_path);
    if method == "load" {
        let (a, _): (Vec<_>, Vec<_>) = rows.iter().partition(|r| r.treat);
        let records: Vec<BalanceRecord> = rows.iter().map(record).collect();
        println!(
            "loaded {} rows, {} treated, {} records",
            rows.len(),
            a.len(),
            records.len()
        );
        return;
    }
    let (anchors, candidates): (Vec<BalanceRecord>, Vec<BalanceRecord>) = {
        let mut a = Vec::new();
        let mut c = Vec::new();
        for row in &rows {
            if row.treat {
                a.push(record(row));
            } else {
                c.push(record(row));
            }
        }
        (a, c)
    };
    let result = match method.as_str() {
        "riskset" | "riskset_replace" => run_risk_set(&rows, reps, method == "riskset_replace"),
        "balance" => {
            let spec = spec_for("nn_ps").expect("spec");
            let SpecOutput::Pairs { outcome, .. } = run_spec(
                &spec,
                SpecInput::Standard {
                    anchors: &anchors,
                    candidates: &candidates,
                },
            )
            .expect("match") else {
                panic!("pairs")
            };
            let mut times = Vec::new();
            let mut smd = String::new();
            for _ in 0..reps {
                let start = Instant::now();
                let report = balance_report(&anchors, &candidates, &outcome);
                times.push(start.elapsed().as_secs_f64());
                smd.clear();
                for row in &report.numeric {
                    writeln!(
                        smd,
                        "{},{},{},{},{}",
                        row.name,
                        row.smd_pre.unwrap_or(f64::NAN),
                        row.smd_post.unwrap_or(f64::NAN),
                        row.var_ratio_post.unwrap_or(f64::NAN),
                        row.ecdf_mean_diff_post.unwrap_or(f64::NAN)
                    )
                    .expect("write");
                }
            }
            fs::write(
                dir.join(format!("out/rust_balance_{n}.csv")),
                format!("name,smd_pre,smd_post,var_ratio_post,ecdf_mean_post\n{smd}"),
            )
            .expect("write");
            Result {
                times,
                matched: outcome.matched_cases,
                lines: Vec::new(),
                header: "",
            }
        }
        name => {
            let spec = spec_for(name).unwrap_or_else(|| panic!("unknown method {name}"));
            run_standard(name, &spec, &anchors, &candidates, reps)
        }
    };
    if !result.header.is_empty() {
        write_lines(
            &dir.join(format!("out/rust_{method}_{n}.csv")),
            result.header,
            &result.lines,
        );
    }
    let mut times = result.times.clone();
    let med = median(&mut times);
    let min = times.first().copied().unwrap_or(f64::NAN);
    let line = format!(
        "{n},{method},{},{med:.4},{min:.4},{}\n",
        result.times.len(),
        result.matched
    );
    println!(
        "n {n} {method}: median {med:.3} s, min {min:.3} s over {} runs, matched {}",
        result.times.len(),
        result.matched
    );
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("out/rust_times.csv"))
        .expect("times file");
    file.write_all(line.as_bytes()).expect("write times");
}
