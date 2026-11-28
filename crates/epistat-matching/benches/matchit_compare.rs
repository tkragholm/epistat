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
//! * `OUT`: directory for the results, default `SCRATCH/out`.
//! * `RECORDS`: `owned` builds a `BalanceRecord` per unit, with its strata,
//!   numeric fields and covariates in hash maps. Any other value, or none,
//!   builds one columnar `Cohort` and matches its rows. Both give the same
//!   results.
//!
//! The cohort is simulated from a fixed seed by `gen` and written to
//! `SCRATCH/cohort_N.csv`. Every later run, in Rust and in R, reads that file,
//! so both sides match identical numbers. Each run writes `SCRATCH/out/rust_METHOD_N.csv`
//! with its pairs or subclasses and appends a line to `SCRATCH/out/rust_times.csv`.
//!
//! Only the matching is timed: the propensity fit, the distances and the
//! selection, as one call of `run_spec`. Reading the file and building the
//! records are not.

#[path = "../tests/common/trial_panel.rs"]
mod trial_panel;
use chrono::{Days, NaiveDate};
use epistat_core::Bootstrap;
use epistat_matching::CbpsConfig;
use epistat_matching::abadie_imbens::{AbadieImbensData, AbadieImbensSpec, abadie_imbens};
use epistat_matching::effects::{
    BootstrapSpec, Clustering, EffectSpec, MatchedSample, OutcomeModel, UnitData, bootstrap,
    estimate_effects,
};
use epistat_matching::imputation::{
    CompleteDataDf, ImputationApproach, ImputedSet, estimate_pooled, match_imputed,
};
use epistat_matching::multigroup::{VectorMatchSpec, vector_match};
use epistat_matching::output::{MatchDataOptions, Units, get_matches, match_data};
use epistat_matching::prelude::*;
use epistat_matching::sensitivity::{
    MScore, WilcoxonDistribution, hodges_lehmann_bounds, m_statistic_bounds, matched_sets,
    wilcoxon_bounds,
};
use epistat_matching::spec::{
    BinningSpec, CardinalitySpec, CemSpec, CemVariableSpec, CovarianceSpec, CovariatesSpec,
    DistanceSpec, EstimatorSpec, GeneticSpec, LinkSpec, MatchMethod, MatchMode, MissingSpec,
    ScaleSpec, StdCaliperSpec, SubclassSpec, TransformSpec,
};
use epistat_matching::{
    BalanceRecord, CovariateEncodingConfig, CovariateRecord, CovariateValue, ElasticNetCvConfig,
    FineBalance, LogisticRegressionConfig, MatchingCriteria, NearestSelection, PropensityEstimator,
    PropensityLink, PropensityScoreConfig, SpecOutput, balance_report,
    prepare_propensity_distance_config,
};
use epistat_matching::{Cohort, CohortRow};
use std::fmt::Write as _;
use std::fs;
use std::io::{BufRead, Write as _};
use std::path::PathBuf;
use std::time::Instant;
use trial_panel::{PanelPeriod, PanelPerson, PanelRows};

const SEED: u64 = 20_261_121;
const COVARIATES: [&str; 5] = ["x1", "x2", "x3", "x4", "x5"];
/// Units per fine stratum `g`, small enough for optimal matching in R.
const FINE_STRATUM: usize = 400;
const CALIPER_STD: f64 = 0.2;
/// Raw caliper on the probability scale for optimal and full matching, which
/// MatchIt does not apply itself.
const CALIPER_RAW_DEFAULT: f64 = 0.2;
/// Periods of the sequential trial panel and days between their start dates.
const TRIAL_PERIODS: usize = 6;
const TRIAL_SPACING: u64 = 300;
/// Bootstrap replicates of the `effects_boot` method.
const BOOT_REPLICATES: usize = 100;
/// Imputed data sets of the `impute` method.
const IMPUTED_SETS: usize = 3;

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

fn out_dir() -> PathBuf {
    std::env::var("OUT").map_or_else(|_| scratch().join("out"), PathBuf::from)
}

/// The cohort: a treated share of about 10 percent that depends on five
/// covariates, twenty coarse strata `s`, fine strata `g` of about 400 units,
/// and for risk-set matching a case time for 5 percent of the units and a
/// death time for 20 percent of the rest. Times are days from a common birth
/// date.
///
/// Four columns follow, drawn from a second stream so that the first eleven do
/// not depend on them: a family key `fam` shared by about four units, a
/// continuous outcome `y` that depends on the treatment and the covariates, a
/// three-level group `grp` that depends on the covariates, and `ctreat`, a
/// treatment that depends strongly on `x1` and `x2` so that balance binds in
/// cardinality matching.
fn generate(n: usize, path: &std::path::Path) {
    let mut rng = Rng(SEED ^ n as u64);
    let mut extra = Rng(SEED.rotate_left(17) ^ n as u64);
    let fine = (n / FINE_STRATUM).max(1) as u64;
    let families = (n / 4).max(1) as u64;
    let mut out = String::with_capacity(n * 100);
    out.push_str("id,treat,x1,x2,x3,x4,x5,s,g,t_case,t_death,fam,y,grp,ctreat\n");
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
        let fam = extra.next_u64() % families;
        let y = round3(
            1.0 + 0.5 * f64::from(treat) + 0.3 * x1 - 0.2 * x2
                + 0.1 * f64::from(x3)
                + extra.normal(),
        );
        let a = 0.5 * x1 + 0.3 * x2 + 0.5 * extra.normal();
        let grp = if a < -0.4 {
            0
        } else if a < 0.4 {
            1
        } else {
            2
        };
        let clp = -1.5 + 1.2 * x1 + 0.9 * x2;
        let ctreat = u8::from(extra.uniform() < 1.0 / (1.0 + (-clp).exp()));
        writeln!(
            out,
            "{i},{treat},{x1},{x2},{x3},{x4},{x5},s{s},g{g},{t_case},{t_death},f{fam},{y},{grp},{ctreat}"
        )
        .expect("write to string");
    }
    fs::write(path, out).expect("write cohort");
}

struct Row {
    id: u64,
    treat: bool,
    ctreat: bool,
    x: [f64; 5],
    s: u16,
    g: u32,
    fam: u32,
    grp: u8,
    y: f64,
    t_case: Option<u32>,
    t_death: Option<u32>,
}

fn read_cohort(path: &std::path::Path) -> Vec<Row> {
    let file = fs::File::open(path).expect("open cohort");
    let mut lines = std::io::BufReader::with_capacity(1 << 20, file).lines();
    lines.next();
    let mut rows = Vec::new();
    for line in lines {
        let line = line.expect("read line");
        let f: Vec<&str> = line.split(',').collect();
        let num = |k: usize| f[k].parse::<f64>().expect("number");
        rows.push(Row {
            id: f[0].parse().expect("id"),
            treat: f[1] == "1",
            ctreat: f[14] == "1",
            x: [num(2), num(3), num(4), num(5), num(6)],
            s: f[7][1..].parse().expect("s"),
            g: f[8][1..].parse().expect("g"),
            fam: f[11][1..].parse().expect("fam"),
            grp: f[13].parse().expect("grp"),
            y: num(12),
            t_case: f[9].parse().ok(),
            t_death: f[10].parse().ok(),
        });
    }
    rows
}

fn origin() -> NaiveDate {
    NaiveDate::from_ymd_opt(2000, 1, 1).expect("valid date")
}

fn record(row: &Row) -> BalanceRecord {
    let mut r = BalanceRecord::new(row.id.to_string(), origin());
    for (name, value) in COVARIATES.iter().zip(row.x) {
        r.covariates
            .insert((*name).to_string(), CovariateValue::Numeric(value));
        r.core.numerics.insert((*name).to_string(), value);
    }
    r.core.strata.insert("s".into(), format!("s{}", row.s));
    r.core.strata.insert("g".into(), format!("g{}", row.g));
    r.core.strata.insert("x3".into(), format!("{}", row.x[2]));
    r.core.strata.insert("x5".into(), format!("{}", row.x[4]));
    r.core.strata.insert("fam".into(), format!("f{}", row.fam));
    r
}

/// Whether the records are owned `BalanceRecord`s rather than rows of a
/// columnar `Cohort`.
fn owned_records() -> bool {
    std::env::var("RECORDS").is_ok_and(|v| v == "owned")
}

const STRATA: [&str; 5] = ["s", "g", "x3", "x5", "fam"];

fn stratum_value(row: &Row, key: &str) -> String {
    match key {
        "s" => format!("s{}", row.s),
        "g" => format!("g{}", row.g),
        "x3" => format!("{}", row.x[2]),
        "x5" => format!("{}", row.x[4]),
        _ => format!("f{}", row.fam),
    }
}

/// The units of `rows` as one columnar cohort holding what [`record`] puts in
/// each record, with `x` in place of the rows' covariates when given. With
/// `dates`, the death and case dates of [`cohort_of`] too.
fn columns(rows: &[Row], x: Option<&[[f64; 5]]>, dates: bool) -> Cohort {
    let mut builder = Cohort::builder(
        rows.iter().map(|row| row.id.to_string()),
        vec![origin(); rows.len()],
    );
    for (k, name) in COVARIATES.iter().enumerate() {
        let values: Vec<Option<f64>> = match x {
            Some(x) => x.iter().map(|x| Some(x[k])).collect(),
            None => rows.iter().map(|row| Some(row.x[k])).collect(),
        };
        builder = builder.numeric_field_and_covariate(*name, values);
    }
    for key in STRATA {
        builder = builder.stratum(key, rows.iter().map(|row| Some(stratum_value(row, key))));
    }
    if dates {
        let day = |d: Option<u32>| d.map(|d| origin() + Days::new(u64::from(d)));
        builder = builder
            .death_date(rows.iter().map(|row| day(row.t_death)).collect())
            .event_date(rows.iter().map(|row| day(row.t_case)).collect());
    }
    builder.build().expect("valid cohort")
}

/// The two groups of a standard match. `y` is the outcome of the unit with id
/// `i + 1`.
struct Inputs<R> {
    anchors: Vec<R>,
    candidates: Vec<R>,
    y: Vec<f64>,
}

/// The two groups as rows of `cohort`, which holds the units of `rows` in
/// order.
fn standard_rows<'a>(rows: &[Row], hard: bool, cohort: &'a Cohort) -> Inputs<CohortRow<'a>> {
    let mut anchors = Vec::new();
    let mut candidates = Vec::new();
    for (row, unit) in rows.iter().zip(cohort.rows()) {
        if if hard { row.ctreat } else { row.treat } {
            anchors.push(unit);
        } else {
            candidates.push(unit);
        }
    }
    Inputs {
        anchors,
        candidates,
        y: rows.iter().map(|row| row.y).collect(),
    }
}

/// The two groups as owned records, built from the rows, which are consumed.
fn standard_inputs(rows: Vec<Row>, hard: bool) -> Inputs<BalanceRecord> {
    let mut y = Vec::with_capacity(rows.len());
    let mut anchors = Vec::new();
    let mut candidates = Vec::new();
    for row in rows {
        y.push(row.y);
        let treated = if hard { row.ctreat } else { row.treat };
        if treated {
            anchors.push(record(&row));
        } else {
            candidates.push(record(&row));
        }
    }
    Inputs {
        anchors,
        candidates,
        y,
    }
}

fn propensity(caliper: Option<f64>) -> DistanceSpec {
    DistanceSpec::Propensity {
        covariates: covariates(),
        link: LinkSpec::Logit,
        estimator: EstimatorSpec::Glm,
        scale: ScaleSpec::Probability,
        sampling_weights: None,
        caliper,
    }
}

fn covariates() -> CovariatesSpec {
    CovariatesSpec {
        names: COVARIATES.iter().map(|s| (*s).to_string()).collect(),
        missing: MissingSpec::Error,
    }
}

fn mahalanobis() -> DistanceSpec {
    DistanceSpec::Mahalanobis {
        covariates: covariates(),
        covariance: CovarianceSpec::default(),
        transform: TransformSpec::default(),
        caliper: None,
    }
}

fn euclidean() -> DistanceSpec {
    DistanceSpec::Euclidean {
        covariates: covariates(),
        scaled: true,
        caliper: None,
    }
}

/// The window of methods that do not read one.
fn unwindowed() -> i32 {
    MatchSpec {
        seed: Some(0),
        ..MatchSpec::default()
    }
    .birth_window_days
}

fn wide() -> i32 {
    36_500
}

/// The spec of a method name, or `None` for a name that is not a spec method
/// of standard matching.
fn spec_for(method: &str) -> Option<MatchSpec> {
    let base = MatchSpec {
        method: MatchMethod::Nearest,
        distance: propensity(None),
        birth_window_days: wide(),
        seed: Some(0),
        ..MatchSpec::default()
    };
    let strata = vec!["s".to_string()];
    Some(match method {
        "nn_ps" => base,
        "nn_ps_cal" => MatchSpec {
            distance: propensity(Some(CALIPER_STD)),
            std_caliper: StdCaliperSpec::All(true),
            ..base
        },
        "nn_strata" => MatchSpec {
            exact: strata,
            ..base
        },
        "nn_cal_strata" => MatchSpec {
            distance: propensity(Some(CALIPER_STD)),
            std_caliper: StdCaliperSpec::All(true),
            exact: strata,
            ..base
        },
        "nn_1to3" => MatchSpec { ratio: 3, ..base },
        "nn_ps_r4" => MatchSpec { ratio: 4, ..base },
        "nn_strata_r4" => MatchSpec {
            ratio: 4,
            exact: strata,
            ..base
        },
        "nn_replace" => MatchSpec {
            replace: true,
            ..base
        },
        "nn_unique" => MatchSpec {
            unique_by_key: Some("fam".into()),
            ..base
        },
        "nn_mahal" => MatchSpec {
            distance: mahalanobis(),
            ..base
        },
        "nn_mahal_strata" => MatchSpec {
            distance: mahalanobis(),
            exact: strata,
            ..base
        },
        "nn_euclid" => MatchSpec {
            distance: euclidean(),
            ..base
        },
        "nn_euclid_strata" => MatchSpec {
            distance: euclidean(),
            exact: strata,
            ..base
        },
        "subclass" => MatchSpec {
            method: MatchMethod::Subclass,
            subclass: SubclassSpec {
                n: 6,
                min_n: 1,
                reference: None,
            },
            birth_window_days: unwindowed(),
            ..base
        },
        "exact" => MatchSpec {
            method: MatchMethod::Exact,
            distance: DistanceSpec::default(),
            exact: vec!["s".into(), "x3".into(), "x5".into()],
            birth_window_days: unwindowed(),
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
            birth_window_days: unwindowed(),
            ..base
        },
        "quick" => MatchSpec {
            method: MatchMethod::Quick,
            birth_window_days: unwindowed(),
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
        "fine_balance" => MatchSpec {
            method: MatchMethod::Optimal,
            distance: propensity(None),
            exact: vec!["g".into()],
            fine_balance: Some(FineBalance::on_keys(["x5"])),
            ..base
        },
        // Balance binds in the next two: the treatment depends on x1 and x2
        // so strongly that the controls the tolerance asks for are scarcer
        // than the treated units.
        "cardinality" => MatchSpec {
            method: MatchMethod::Cardinality,
            distance: DistanceSpec::default(),
            cardinality: CardinalitySpec {
                covariates: covariates(),
                ratio: Some(1.0),
                tols: vec![0.05],
                std_tols: vec![true],
                ..CardinalitySpec::default()
            },
            birth_window_days: unwindowed(),
            ..base
        },
        "profile" => MatchSpec {
            method: MatchMethod::Cardinality,
            distance: DistanceSpec::default(),
            cardinality: CardinalitySpec {
                covariates: covariates(),
                ratio: None,
                tols: vec![0.05],
                std_tols: vec![true],
                ..CardinalitySpec::default()
            },
            birth_window_days: unwindowed(),
            ..base
        },
        "genetic_eval" => MatchSpec {
            method: MatchMethod::Genetic,
            genetic: GeneticSpec {
                covariates: covariates(),
                pop_size: 0,
                ..GeneticSpec::default()
            },
            seed: Some(7),
            birth_window_days: unwindowed(),
            ..base
        },
        "genetic_short" => MatchSpec {
            method: MatchMethod::Genetic,
            genetic: GeneticSpec {
                covariates: covariates(),
                pop_size: 10,
                max_generations: 2,
                wait_generations: 1,
                hard_generation_limit: true,
                ..GeneticSpec::default()
            },
            seed: Some(7),
            birth_window_days: unwindowed(),
            ..base
        },
        _ => return None,
    })
}

fn median(times: &mut [f64]) -> f64 {
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

/// What a method run leaves for the results file.
struct Timed {
    /// Wall seconds of each repetition.
    times: Vec<f64>,
    /// Number of units matched or kept, or the size of the result.
    matched: usize,
    /// Pairs as (anchor, control) ids, or unit and subclass, for comparison
    /// with R. Empty for methods R is not compared on.
    lines: Vec<(u64, u64)>,
    header: &'static str,
}

fn write_lines(path: &std::path::Path, header: &str, lines: &[(u64, u64)]) {
    let mut out = std::io::BufWriter::new(fs::File::create(path).expect("create output"));
    writeln!(out, "{header}").expect("write");
    for (a, b) in lines {
        writeln!(out, "{a},{b}").expect("write");
    }
}

fn id_number(id: &str) -> u64 {
    id.parse().expect("numeric id")
}

fn subclass_lines<'a>(rows: impl Iterator<Item = (&'a str, usize)>) -> Vec<(u64, u64)> {
    let mut v: Vec<(u64, u64)> = rows.map(|(k, v)| (id_number(k), v as u64)).collect();
    v.sort_unstable();
    v
}

fn output_lines(out: &SpecOutput) -> (usize, Vec<(u64, u64)>, &'static str) {
    match out {
        SpecOutput::Pairs { outcome, .. } | SpecOutput::Genetic { outcome, .. } => {
            let mut lines: Vec<(u64, u64)> = outcome
                .pairs
                .iter()
                .map(|p| {
                    (
                        id_number(p.case_id.as_str()),
                        id_number(p.control_id.as_str()),
                    )
                })
                .collect();
            lines.sort_unstable();
            (outcome.matched_cases, lines, "anchor,control")
        }
        SpecOutput::Exact(o) | SpecOutput::Quick(o) | SpecOutput::Full(o) => (
            o.retained_units(),
            subclass_lines(o.assignments()),
            "id,subclass",
        ),

        SpecOutput::Subclass { outcome, .. } => (
            outcome.retained_units(),
            subclass_lines(outcome.assignments()),
            "id,subclass",
        ),
        SpecOutput::Cardinality(m) => {
            for report in &m.outcome.strata {
                println!(
                    "stratum {}: units {}, treated kept {}, controls kept {}, proven {}, bound {}, nodes {}, pivots {}",
                    report.key,
                    report.units,
                    report.selected_treated,
                    report.selected_control,
                    report.proven_optimal,
                    report.bound,
                    report.nodes,
                    report.pivots
                );
            }
            (
                m.outcome.selected.iter().filter(|s| **s).count(),
                Vec::new(),
                "",
            )
        }
        SpecOutput::ExposureDensity(o) => (o.exposed.len(), Vec::new(), ""),
        SpecOutput::Trials(o) => (
            o.trials.iter().map(|t| t.outcome.pairs.len()).sum(),
            Vec::new(),
            "",
        ),
        _ => (0, Vec::new(), ""),
    }
}

/// Repetitions stop once the total passes this many seconds.
const REPS_BUDGET_SECS: f64 = 20.0;

fn time_reps<T>(reps: usize, mut f: impl FnMut() -> T) -> (Vec<f64>, T) {
    let mut times: Vec<f64> = Vec::new();
    let mut last = None;
    for _ in 0..reps {
        let start = Instant::now();
        let out = f();
        times.push(start.elapsed().as_secs_f64());
        last = Some(out);
        if times.iter().sum::<f64>() > REPS_BUDGET_SECS {
            break;
        }
    }
    (times, last.expect("one repetition"))
}

fn run_standard<R: CovariateRecord>(
    method: &str,
    spec: &MatchSpec,
    input: &Inputs<R>,
    reps: usize,
) -> Timed {
    let (times, out) = time_reps(reps, || {
        run_spec(
            spec,
            SpecInput::Standard {
                cases: &input.anchors,
                controls: &input.candidates,
            },
        )
        .unwrap_or_else(|e| panic!("{method}: {e}"))
    });
    let (matched, lines, header) = output_lines(&out);
    Timed {
        times,
        matched,
        lines,
        header,
    }
}

fn cohort_of(rows: &[Row]) -> Vec<RoleTransitionRecord<BalanceRecord>> {
    rows.iter()
        .map(|row| {
            let mut r = record(row);
            r.core.death_date = row.t_death.map(|d| origin() + Days::new(u64::from(d)));
            let when = row.t_case.map(|d| origin() + Days::new(u64::from(d)));
            RoleTransitionRecord::from_record(r, when)
        })
        .collect()
}

fn risk_set_spec(mode: MatchMode, replace: bool) -> MatchSpec {
    MatchSpec {
        mode,
        method: MatchMethod::Random,
        exact: vec!["s".into()],
        birth_window_days: wide(),
        require_alive: true,
        age_limit_years: Some(6),
        replace,
        seed: Some(7),
        ..MatchSpec::default()
    }
}

fn run_risk_set(rows: Vec<Row>, reps: usize, mode: MatchMode, replace: bool) -> Timed {
    if owned_records() {
        let cohort = cohort_of(&rows);
        drop(rows);
        run_risk_set_on(&cohort, reps, mode, replace)
    } else {
        let columns = columns(&rows, None, true);
        drop(rows);
        run_risk_set_on(&columns.risk_set_rows(), reps, mode, replace)
    }
}

fn run_risk_set_on<R: CovariateRecord>(
    cohort: &[RoleTransitionRecord<R>],
    reps: usize,
    mode: MatchMode,
    replace: bool,
) -> Timed {
    let spec = risk_set_spec(mode, replace);
    let (times, out) = time_reps(reps, || {
        run_spec(&spec, SpecInput::RiskSet { cohort }).unwrap_or_else(|e| panic!("risk set: {e}"))
    });
    let (matched, lines, header) = output_lines(&out);
    Timed {
        times,
        matched,
        lines,
        header,
    }
}

/// People observed in six periods; a person who starts treatment does so in
/// the period `g mod 6`, is eligible until then, and the outcome falls in the
/// period of the case date.
fn trial_panel<R>(
    rows: &[Row],
    mut record_of: impl FnMut(usize, &Row) -> R,
) -> Vec<PanelPerson<R>> {
    rows.iter()
        .enumerate()
        .map(|(i, row)| {
            let r = record_of(i, row);
            let start = (row.g as usize) % TRIAL_PERIODS;
            let event_period = row
                .t_case
                .map(|d| ((u64::from(d) - 1) / TRIAL_SPACING) as usize);
            let periods = (0..TRIAL_PERIODS)
                .map(|k| PanelPeriod {
                    period: i64::try_from(k).expect("small"),
                    date: origin() + Days::new(TRIAL_SPACING * (k as u64 + 1)),
                    eligible: !row.treat || k <= start,
                    treatment: Some(row.treat && k >= start),
                    event: event_period == Some(k),
                })
                .collect();
            PanelPerson { record: r, periods }
        })
        .collect()
}

fn run_trials(rows: Vec<Row>, reps: usize) -> Timed {
    if owned_records() {
        let panel = trial_panel(&rows, |_, row| {
            let mut r = record(row);
            r.core.death_date = row.t_death.map(|d| origin() + Days::new(u64::from(d)));
            r
        });
        drop(rows);
        run_trials_on(&panel, reps)
    } else {
        let columns = columns(&rows, None, true);
        let panel = trial_panel(&rows, |i, _| columns.row(i).expect("row"));
        drop(rows);
        run_trials_on(&panel, reps)
    }
}

/// The cohort's case dates are not read: the panel's periods set the trials.
fn run_trials_on<R: CovariateRecord + Clone>(panel: &[PanelPerson<R>], reps: usize) -> Timed {
    let rows = PanelRows::of(panel);
    let panel = &rows.panel();
    let spec = MatchSpec {
        mode: MatchMode::SequentialTrials,
        method: MatchMethod::Random,
        exact: vec!["s".into()],
        birth_window_days: wide(),
        require_alive: true,
        age_limit_years: Some(6),
        seed: Some(7),
        ..MatchSpec::default()
    };
    let (times, out) = time_reps(reps, || {
        run_spec(&spec, SpecInput::Trials { panel }).unwrap_or_else(|e| panic!("trials: {e}"))
    });
    let (matched, lines, header) = output_lines(&out);
    Timed {
        times,
        matched,
        lines,
        header,
    }
}

fn units_of<R: MatchingRecord>(input: &Inputs<R>) -> Units {
    let ids = input
        .anchors
        .iter()
        .chain(&input.candidates)
        .map(|r| r.id().to_owned())
        .collect();
    let anchor = (0..input.anchors.len() + input.candidates.len())
        .map(|i| i < input.anchors.len())
        .collect();
    Units::new(ids, anchor).expect("distinct ids")
}

fn unit_data_of<R: MatchingRecord>(input: &Inputs<R>, y: &[f64]) -> UnitData {
    let all = || input.anchors.iter().chain(&input.candidates);
    let ids: Vec<String> = all().map(|r| r.id().to_owned()).collect();
    let treated: Vec<bool> = (0..ids.len()).map(|i| i < input.anchors.len()).collect();
    let outcome: Vec<f64> = all()
        .map(|r| y[usize::try_from(id_number(r.id())).expect("id") - 1])
        .collect();
    let mut data = UnitData::new(ids, treated, outcome).expect("unit data");
    for name in ["x1", "x2"] {
        let column = all()
            .map(|r| r.numeric(name).expect("numeric field"))
            .collect();
        data = data.with_covariate(name, column).expect("covariate");
    }
    data
}

fn nearest_pairs<R: CovariateRecord>(input: &Inputs<R>) -> MatchOutcome {
    let spec = spec_for("nn_ps").expect("spec");
    match run_spec(
        &spec,
        SpecInput::Standard {
            cases: &input.anchors,
            controls: &input.candidates,
        },
    )
    .expect("match")
    {
        SpecOutput::Pairs { outcome, .. } => *outcome,
        _ => panic!("pairs"),
    }
}

fn fit_config(link: PropensityLink, estimator: PropensityEstimator) -> PropensityScoreConfig {
    PropensityScoreConfig {
        encoding: CovariateEncodingConfig {
            covariate_keys: COVARIATES.iter().map(|s| (*s).to_string()).collect(),
            include_intercept: true,
            ..CovariateEncodingConfig::default()
        },
        estimator,
        logistic: LogisticRegressionConfig {
            link,
            ..LogisticRegressionConfig::default()
        },
        ..PropensityScoreConfig::default()
    }
}

fn run_fit<R: CovariateRecord>(
    input: &Inputs<R>,
    reps: usize,
    config: &PropensityScoreConfig,
) -> Timed {
    let (times, prepared) = time_reps(reps, || {
        prepare_propensity_distance_config(&input.anchors, &input.candidates, config, None)
            .unwrap_or_else(|e| panic!("fit: {e}"))
    });
    Timed {
        times,
        matched: prepared.estimate.probabilities.len(),
        lines: Vec::new(),
        header: "",
    }
}

fn run_balance<R: CovariateRecord>(input: &Inputs<R>, reps: usize) -> (Timed, String) {
    let outcome = nearest_pairs(input);
    let mut smd = String::new();
    let (times, report) = time_reps(reps, || {
        balance_report(
            &input.anchors,
            &input.candidates,
            &outcome,
            &epistat_matching::BalanceReportOptions::default(),
        )
        .expect("a matched estimand")
    });
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
    (
        Timed {
            times,
            matched: outcome.matched_cases,
            lines: Vec::new(),
            header: "",
        },
        smd,
    )
}

fn run_effects<R: CovariateRecord>(input: &Inputs<R>, reps: usize, boot: bool) -> Timed {
    let outcome = nearest_pairs(input);
    let units = units_of(input);
    let data = unit_data_of(input, &input.y);
    let md = match_data(&outcome, &units, MatchDataOptions::default()).expect("match data");
    let mut spec = EffectSpec::new(OutcomeModel::Linear);
    spec.clustering = Clustering::Pair;
    let (times, rows) = time_reps(reps, || {
        let sample = MatchedSample::from_match_data(&md, &data).expect("sample");
        if boot {
            bootstrap(
                &sample,
                &spec,
                &BootstrapSpec::new(Bootstrap::new(BOOT_REPLICATES, 11)),
            )
            .expect("bootstrap");
        } else {
            estimate_effects(&sample, &spec).expect("effects");
        }
        sample.len()
    });
    Timed {
        times,
        matched: rows,
        lines: Vec::new(),
        header: "",
    }
}

fn run_abadie<R: MatchingRecord>(input: &Inputs<R>, reps: usize) -> Timed {
    let data = unit_data_of(input, &input.y);
    let mut spec = AbadieImbensSpec::new();
    spec.bias_adjust = true;
    spec.var_calc = 2;
    let (times, est) = time_reps(reps, || {
        let units = AbadieImbensData::from_unit_data(&data).expect("data");
        abadie_imbens(&units, &spec).expect("estimate")
    });
    Timed {
        times,
        matched: est.units,
        lines: Vec::new(),
        header: "",
    }
}

fn run_sensitivity<R: CovariateRecord>(input: &Inputs<R>, reps: usize, which: &str) -> Timed {
    let outcome = nearest_pairs(input);
    let units = units_of(input);
    let data = unit_data_of(input, &input.y);
    let long = get_matches(&outcome, &units, false).expect("long matches");
    let gammas = [1.0, 1.5, 2.0, 2.5, 3.0];
    let (times, sets) = time_reps(reps, || matched_sets(&long, &data).expect("sets"));
    if which == "sens_sets" {
        return Timed {
            times,
            matched: sets.len(),
            lines: Vec::new(),
            header: "",
        };
    }
    let (times, ()) = time_reps(reps, || match which {
        "sens_wilcoxon" => {
            wilcoxon_bounds(&sets, &gammas, WilcoxonDistribution::Normal).expect("bounds");
        }
        "sens_hl" => {
            hodges_lehmann_bounds(&sets, &gammas).expect("bounds");
        }
        _ => {
            m_statistic_bounds(&sets, &gammas, &MScore::default()).expect("bounds");
        }
    });
    Timed {
        times,
        matched: sets.len(),
        lines: Vec::new(),
        header: "",
    }
}

/// The covariates of each imputed data set: its own jitter on x1 and x2, as
/// imputed values differ.
fn imputed_covariates(rows: &[Row]) -> Vec<Vec<[f64; 5]>> {
    (0..IMPUTED_SETS)
        .map(|m| {
            let mut jitter = Rng(SEED ^ (m as u64 + 1));
            rows.iter()
                .map(|row| {
                    let mut x = row.x;
                    for value in &mut x[..2] {
                        *value += 0.05 * jitter.normal();
                    }
                    x
                })
                .collect()
        })
        .collect()
}

fn run_impute(rows: Vec<Row>, reps: usize) -> Timed {
    let y: Vec<f64> = rows.iter().map(|r| r.y).collect();
    let xs = imputed_covariates(&rows);
    if owned_records() {
        let sets_data: Vec<Inputs<BalanceRecord>> = xs
            .iter()
            .map(|x| {
                let mut anchors = Vec::new();
                let mut candidates = Vec::new();
                for (row, x) in rows.iter().zip(x) {
                    let mut r = record(row);
                    for (name, v) in ["x1", "x2"].into_iter().zip(x) {
                        r.core.numerics.insert(name.to_string(), *v);
                        r.covariates
                            .insert(name.to_string(), CovariateValue::Numeric(*v));
                    }
                    if row.treat {
                        anchors.push(r);
                    } else {
                        candidates.push(r);
                    }
                }
                Inputs {
                    anchors,
                    candidates,
                    y: Vec::new(),
                }
            })
            .collect();
        drop(rows);
        run_impute_on(&sets_data, &y, reps)
    } else {
        let cohorts: Vec<Cohort> = xs.iter().map(|x| columns(&rows, Some(x), false)).collect();
        let sets_data: Vec<Inputs<CohortRow<'_>>> = cohorts
            .iter()
            .map(|cohort| standard_rows(&rows, false, cohort))
            .collect();
        drop(rows);
        run_impute_on(&sets_data, &y, reps)
    }
}

fn run_impute_on<R: CovariateRecord>(sets_data: &[Inputs<R>], y: &[f64], reps: usize) -> Timed {
    let sets: Vec<ImputedSet<'_, R>> = sets_data
        .iter()
        .map(|s| ImputedSet {
            cases: &s.anchors,
            controls: &s.candidates,
        })
        .collect();
    let data: Vec<UnitData> = sets_data.iter().map(|s| unit_data_of(s, y)).collect();
    let criteria = MatchingCriteria {
        match_ratio: 1,
        ..MatchingCriteria::default()
    };
    let config = PropensityScoreConfig::default();
    let effect = EffectSpec::new(OutcomeModel::Linear);
    let (times, pooled) = time_reps(reps, || {
        let matches = match_imputed(
            &sets,
            ImputationApproach::Within,
            &criteria,
            &NearestSelection::new(0),
            &[],
            &(),
            &config,
        )
        .expect("matches");
        estimate_pooled(
            &sets,
            &matches,
            &data,
            MatchDataOptions::default(),
            &effect,
            CompleteDataDf::Residual,
        )
        .expect("pooled")
    });
    Timed {
        times,
        matched: pooled.per_set.len(),
        lines: Vec::new(),
        header: "",
    }
}

fn run_multigroup(rows: Vec<Row>, reps: usize) -> Timed {
    let groups: Vec<usize> = rows.iter().map(|r| usize::from(r.grp)).collect();
    if owned_records() {
        let records: Vec<BalanceRecord> = rows.iter().map(record).collect();
        drop(rows);
        run_multigroup_on(&records, &groups, reps)
    } else {
        let cohort = columns(&rows, None, false);
        drop(rows);
        let records: Vec<CohortRow<'_>> = cohort.rows().collect();
        run_multigroup_on(&records, &groups, reps)
    }
}

fn run_multigroup_on<R: CovariateRecord>(records: &[R], groups: &[usize], reps: usize) -> Timed {
    let spec = VectorMatchSpec::new(0, 5);
    let (times, out) = time_reps(reps, || {
        vector_match(records, groups, 3, &spec).unwrap_or_else(|e| panic!("vector: {e}"))
    });
    Timed {
        times,
        matched: out.matched_focal.len(),
        lines: Vec::new(),
        header: "",
    }
}

/// Runs a method of standard matching on the two groups, or on the units for
/// the methods that take one sample.
fn run_on_groups<R: CovariateRecord>(
    m: &str,
    input: &Inputs<R>,
    reps: usize,
) -> (Timed, Option<String>) {
    let timed = match m {
        "cardinality" | "profile" => run_standard(m, &spec_for(m).expect("spec"), input, reps),
        "balance" => {
            let (timed, table) = run_balance(input, reps);
            return (timed, Some(table));
        }
        "fit_logit" => run_fit(
            input,
            reps,
            &fit_config(PropensityLink::Logit, PropensityEstimator::GlmLogit),
        ),
        "fit_probit" => run_fit(
            input,
            reps,
            &fit_config(PropensityLink::Probit, PropensityEstimator::GlmLogit),
        ),
        "fit_enet_cv" => run_fit(
            input,
            reps,
            &fit_config(
                PropensityLink::Logit,
                PropensityEstimator::ElasticNetCv(ElasticNetCvConfig {
                    alpha: 1.0,
                    ..ElasticNetCvConfig::new(3)
                }),
            ),
        ),
        "fit_cbps" => run_fit(
            input,
            reps,
            &fit_config(
                PropensityLink::Logit,
                PropensityEstimator::Cbps(CbpsConfig::default()),
            ),
        ),
        "effects" => run_effects(input, reps, false),
        "effects_boot" => run_effects(input, reps, true),
        "abadie" => run_abadie(input, reps),
        "sens_sets" | "sens_wilcoxon" | "sens_hl" | "sens_mscore" => {
            run_sensitivity(input, reps, m)
        }
        name => {
            let spec = spec_for(name).unwrap_or_else(|| panic!("unknown method {name}"));
            run_standard(name, &spec, input, reps)
        }
    };
    (timed, None)
}

/// Builds the two groups of a standard match from the rows, which are
/// consumed, and runs `m` on them.
fn run_standard_method(m: &str, rows: Vec<Row>, reps: usize) -> (Timed, Option<String>) {
    let hard = m == "cardinality" || m == "profile";
    if owned_records() {
        let input = standard_inputs(rows, hard);
        run_on_groups(m, &input, reps)
    } else {
        let cohort = columns(&rows, None, false);
        let input = standard_rows(&rows, hard, &cohort);
        drop(rows);
        run_on_groups(m, &input, reps)
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
    let out = out_dir();
    fs::create_dir_all(&out).expect("output directory");
    let cohort_path = dir.join(format!("cohort_{n}.csv"));
    if method == "gen" {
        generate(n, &cohort_path);
        return;
    }
    let rows = read_cohort(&cohort_path);
    let m = method.as_str();
    let mut balance_table = None;
    let result = match m {
        "load" => {
            let (treated, others) = if owned_records() {
                let input = standard_inputs(rows, false);
                (input.anchors.len(), input.candidates.len())
            } else {
                let cohort = columns(&rows, None, false);
                let input = standard_rows(&rows, false, &cohort);
                drop(rows);
                (input.anchors.len(), input.candidates.len())
            };
            println!("loaded {treated} treated, {others} others");
            return;
        }
        "riskset" | "riskset_replace" => {
            run_risk_set(rows, reps, MatchMode::RiskSet, m == "riskset_replace")
        }
        "expdens" => run_risk_set(rows, reps, MatchMode::ExposureDensity, false),
        "trials" => run_trials(rows, reps),
        "impute" => run_impute(rows, reps),
        "multigroup" => run_multigroup(rows, reps),
        _ => {
            let (timed, table) = run_standard_method(m, rows, reps);
            balance_table = table;
            timed
        }
    };
    if let Some(table) = balance_table {
        fs::write(
            out.join(format!("rust_balance_{n}.csv")),
            format!("name,smd_pre,smd_post,var_ratio_post,ecdf_mean_post\n{table}"),
        )
        .expect("write");
    }
    if !result.header.is_empty() && n <= 1_000_000 {
        write_lines(
            &out.join(format!("rust_{method}_{n}.csv")),
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
        .open(out.join("rust_times.csv"))
        .expect("times file");
    file.write_all(line.as_bytes()).expect("write times");
}
