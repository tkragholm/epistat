//! Wall time of the survival estimators on a simulated register cohort, to be
//! set against R's `survival` and `mstate` on the same data
//! (`benches/survival/survival_bench.R`).
//!
//! One process times one method on one cohort size, so that
//! `/usr/bin/time -l` gives the peak memory of that run alone.
//! `benches/survival/run.sh` is the driver. The environment sets the work:
//!
//! * `SCRATCH`: directory for the cohort file and the results. Nothing is
//!   written inside the repository.
//! * `N`: number of subjects, for example 100000.
//! * `METHOD`: `gen` writes the cohort file and stops, `load` reads it and
//!   stops. Any other name is one of the methods below. A name ending in
//!   `_split` runs the method on the follow-up cut at day 750, and `split`
//!   times the cutting.
//! * `REPS`: timed repetitions, default 3. The median is reported.
//! * `OUT`: directory for the results, default `SCRATCH/out`.
//! * `BOOT`: bootstrap replicates of `aj_boot`, default 50.
//!
//! Methods: `cox_breslow`, `cox_efron` and `cox_exact` (cause-specific Cox on
//! two covariates with delayed entry), `aj` (Aalen-Johansen by arm),
//! `aj_boot` (its clustered bootstrap), `finegray`, `survfit`
//! (Kaplan-Meier by arm), `multistate` (occupation probabilities of an
//! illness-death model) and `split`. Fine-Gray and the bootstrap take no
//! delayed entry and do not run on split data.
//!
//! The cohort is simulated from a fixed seed by `gen` and written to
//! `SCRATCH/cohort_N.csv`. Every later run, in Rust and in R, reads that file.
//! Each run writes `SCRATCH/out/rust_METHOD_N.csv` with the numbers the two
//! sides are compared on, and appends a line to `SCRATCH/out/rust_times.csv`.
//! Only the estimation is timed: reading the file, building the data and, for
//! the `_split` methods, the cutting are not.

#![allow(
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    clippy::similar_names,
    reason = "a benchmark program: the cohort has fewer than 2^32 rows, times are whole days, and the names follow the R script"
)]

use epistat_core::{Bootstrap, Seed};
use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::aalen_johansen::{
    self, AalenJohansenOptions, BootstrapOptions, Draw, RowWindows,
};
use epistat_survival::cox::{self, CoxOptions, Ties};
use epistat_survival::fine_gray::{self, FineGrayOptions};
use epistat_survival::multistate;
use epistat_survival::split::{SplitOptions, split};
use epistat_survival::survfit::{self, SurvfitOptions};
use std::fmt::Write as _;
use std::fs;
use std::io::{BufRead, Write as _};
use std::path::PathBuf;
use std::time::Instant;

const SEED: u64 = 20_261_122;
/// The cut point of the `_split` methods, in days.
const CUT: f64 = 750.0;
/// The times at which the curves are compared.
const GRID: [f64; 14] = [
    100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0,
    1400.0,
];
/// Subjects per cluster of the bootstrap.
const CLUSTER_SIZE: u64 = 4;

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

    /// A whole number of days, at least 1, exponential with the given mean.
    fn days(&mut self, mean: f64) -> f64 {
        (-self.uniform().ln() * mean).ceil().max(1.0)
    }
}

fn scratch() -> PathBuf {
    PathBuf::from(
        std::env::var("SCRATCH").expect("SCRATCH names a directory outside the repository"),
    )
}

fn out_dir() -> PathBuf {
    std::env::var("OUT").map_or_else(|_| scratch().join("out"), PathBuf::from)
}

/// Writes the cohort. Each subject has two covariates, an arm, a delayed
/// entry for 30 percent, a cluster, and an illness-death history observed
/// until a censoring time: the first transition is to illness (status 1) or
/// to death (status 2), and an ill subject goes on to death. `t2` and `s2`
/// are the second leg, empty when there is none. All times are whole days, so
/// events tie.
fn generate(n: usize, path: &std::path::Path) {
    let mut rng = Rng(SEED);
    let mut out = String::with_capacity(n * 70);
    out.push_str("id,x1,x2,arm,entry,time,status,fam,t2,s2\n");
    for i in 0..n {
        let x1 = rng.normal();
        let x2 = rng.normal();
        let arm = u8::from(rng.uniform() < 0.5);
        let entry = if rng.uniform() < 0.3 {
            (rng.uniform() * 300.0).ceil()
        } else {
            0.0
        };
        let tc = (200.0 + rng.uniform() * 1300.0).ceil().max(entry + 1.0);
        let ill = entry + rng.days(1200.0 / (0.5 * x1 + 0.3 * f64::from(arm)).exp());
        let dead = entry + rng.days(4000.0 * (0.3 * x2).exp());
        let (time, status) = if ill <= dead && ill <= tc {
            (ill, 1)
        } else if dead <= tc {
            (dead, 2)
        } else {
            (tc, 0)
        };
        let (t2, s2) = if status == 1 && tc > ill {
            let d2 = ill + rng.days(1000.0);
            if d2 <= tc {
                (d2.to_string(), 3)
            } else {
                (tc.to_string(), 0)
            }
        } else {
            (String::new(), 0)
        };
        let fam = i as u64 / CLUSTER_SIZE;
        writeln!(
            out,
            "{i},{x1:.6},{x2:.6},{arm},{entry},{time},{status},{fam},{t2},{}",
            if t2.is_empty() {
                String::new()
            } else {
                s2.to_string()
            }
        )
        .expect("write to string");
    }
    fs::write(path, out).expect("write cohort");
}

#[derive(Default)]
struct Cohort {
    x1: Vec<f64>,
    x2: Vec<f64>,
    arm: Vec<u32>,
    entry: Vec<f64>,
    time: Vec<f64>,
    status: Vec<i32>,
    fam: Vec<u32>,
    /// Second leg of the illness-death history, NaN and 0 when absent.
    t2: Vec<f64>,
    s2: Vec<i32>,
}

fn read_cohort(path: &std::path::Path) -> Cohort {
    let file = fs::File::open(path).expect("open cohort");
    let mut lines = std::io::BufReader::with_capacity(1 << 20, file).lines();
    lines.next();
    let mut c = Cohort::default();
    for line in lines {
        let line = line.expect("read line");
        let f: Vec<&str> = line.split(',').collect();
        c.x1.push(f[1].parse().expect("x1"));
        c.x2.push(f[2].parse().expect("x2"));
        c.arm.push(f[3].parse().expect("arm"));
        c.entry.push(f[4].parse().expect("entry"));
        c.time.push(f[5].parse().expect("time"));
        c.status.push(f[6].parse().expect("status"));
        c.fam.push(f[7].parse().expect("fam"));
        c.t2.push(f[8].parse().unwrap_or(f64::NAN));
        c.s2.push(f[9].parse().unwrap_or(0));
    }
    c
}

/// Follow-up pieces of a cohort: the columns an estimator reads, possibly cut.
struct Frame {
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    row: Vec<u32>,
}

impl Frame {
    fn whole(c: &Cohort) -> Self {
        Self {
            start: c.entry.clone(),
            stop: c.time.clone(),
            status: c.status.clone(),
            row: (0..c.time.len() as u32).collect(),
        }
    }

    fn cut(c: &Cohort) -> Self {
        let data = SurvData::builder(&c.time, &c.status)
            .start(&c.entry)
            .n_codes(2)
            .build()
            .expect("data");
        let s = split(&data, &[CUT], &SplitOptions::default()).expect("split");
        Self {
            start: s.start,
            stop: s.stop,
            status: s.status,
            row: s.row,
        }
    }

    fn pick<T: Copy>(&self, column: &[T]) -> Vec<T> {
        self.row.iter().map(|&r| column[r as usize]).collect()
    }
}

struct Timed {
    times: Vec<f64>,
    lines: String,
}

fn timed<T>(reps: usize, mut run: impl FnMut() -> T) -> (Vec<f64>, T) {
    let mut times = Vec::new();
    let mut last = None;
    for _ in 0..reps {
        let t0 = Instant::now();
        let r = run();
        times.push(t0.elapsed().as_secs_f64());
        last = Some(r);
    }
    (times, last.expect("at least one repetition"))
}

/// Value of a right-continuous step function at `at`, 0 before the first time.
fn step_at(times: &[f64], values: &[f64], at: f64, before: f64) -> f64 {
    let k = times.partition_point(|&t| t <= at);
    if k == 0 { before } else { values[k - 1] }
}

fn run_cox(ties: Ties, f: &Frame, c: &Cohort, reps: usize) -> Timed {
    let (x1, x2) = (f.pick(&c.x1), f.pick(&c.x2));
    let x = Mat::from_fn(x1.len(), 2, |i, j| if j == 0 { x1[i] } else { x2[i] });
    let data = SurvData::builder(&f.stop, &f.status)
        .start(&f.start)
        .n_codes(2)
        .build()
        .expect("data");
    let opts = CoxOptions {
        ties,
        cause: Some(1),
        ..CoxOptions::default()
    };
    let (times, fit) = timed(reps, || cox::fit(&data, x.as_ref(), &opts).expect("cox"));
    let mut lines = String::new();
    for (k, name) in ["x1", "x2"].iter().enumerate() {
        let _ = writeln!(
            lines,
            "coef_{name},{}\nse_{name},{}",
            fit.coefficients[k].expect("coefficient"),
            fit.se[k].expect("se")
        );
    }
    let _ = writeln!(
        lines,
        "loglik,{}",
        fit.log_partial_likelihood.expect("loglik")
    );
    Timed { times, lines }
}

fn run_aj(f: &Frame, c: &Cohort, reps: usize) -> Timed {
    let arm = f.pick(&c.arm);
    let data = SurvData::builder(&f.stop, &f.status)
        .start(&f.start)
        .stratum(&arm)
        .n_codes(2)
        .build()
        .expect("data");
    let (times, curves) = timed(reps, || {
        aalen_johansen::estimate(&data, &AalenJohansenOptions::default()).expect("aj")
    });
    let mut lines = String::new();
    for g in [0u32, 1] {
        let idx: Vec<usize> = (0..curves.group.len())
            .filter(|&i| curves.group[i] == g)
            .collect();
        let t: Vec<f64> = idx.iter().map(|&i| curves.time[i]).collect();
        for (k, cause) in curves.causes.iter().enumerate() {
            let v: Vec<f64> = idx.iter().map(|&i| curves.cif[k][i]).collect();
            for at in GRID {
                let _ = writeln!(
                    lines,
                    "cif{cause}_arm{g}_t{at},{}",
                    step_at(&t, &v, at, 0.0)
                );
            }
        }
    }
    Timed { times, lines }
}

fn run_aj_boot(c: &Cohort, reps: usize) -> Timed {
    let boot: usize = std::env::var("BOOT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    let n = c.time.len();
    let data = SurvData::builder(&c.time, &c.status)
        .cluster(&c.fam)
        .n_codes(2)
        .build()
        .expect("data");
    let exposed: Vec<bool> = c.arm.iter().map(|&a| a == 1).collect();
    let cells =
        RowWindows::new((0..n as u32).collect(), vec![0, n as u32], n, "cell").expect("cells");
    let opts = BootstrapOptions::new(Draw::PerCell);
    let ages = GRID.to_vec();
    let (times, (yes, no, ok)) = timed(reps, || {
        aalen_johansen::bootstrap(
            &data,
            &exposed,
            &cells,
            Bootstrap::new(boot, Seed(SEED)),
            &ages,
            1,
            &opts,
        )
        .expect("bootstrap")
    });
    assert!(ok[0]);
    let mut lines = String::new();
    for (name, arm) in [("arm1", &yes), ("arm0", &no)] {
        for (a, at) in ages.iter().enumerate() {
            let v = &arm[a * boot..(a + 1) * boot];
            let mean = v.iter().sum::<f64>() / boot as f64;
            let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (boot as f64 - 1.0);
            let _ = writeln!(
                lines,
                "mean_{name}_t{at},{mean}\nsd_{name}_t{at},{}",
                var.sqrt()
            );
        }
    }
    Timed { times, lines }
}

fn run_finegray(c: &Cohort, reps: usize) -> Timed {
    let x = Mat::from_fn(
        c.time.len(),
        2,
        |i, j| if j == 0 { c.x1[i] } else { c.x2[i] },
    );
    let data = SurvData::builder(&c.time, &c.status)
        .n_codes(2)
        .build()
        .expect("data");
    let opts = FineGrayOptions::new(1);
    let (times, fit) = timed(reps, || {
        fine_gray::fit(&data, x.as_ref(), &opts).expect("fg")
    });
    let mut lines = String::new();
    for (k, name) in ["x1", "x2"].iter().enumerate() {
        let _ = writeln!(
            lines,
            "coef_{name},{}\nse_{name},{}",
            fit.coefficients[k].expect("coefficient"),
            fit.se[k].expect("se")
        );
    }
    Timed { times, lines }
}

fn run_survfit(f: &Frame, c: &Cohort, reps: usize) -> Timed {
    let arm = f.pick(&c.arm);
    let event: Vec<i32> = f.status.iter().map(|&s| i32::from(s > 0)).collect();
    let data = SurvData::builder(&f.stop, &event)
        .start(&f.start)
        .stratum(&arm)
        .build()
        .expect("data");
    let (times, fit) = timed(reps, || {
        survfit::estimate(&data, &SurvfitOptions::default()).expect("survfit")
    });
    let mut lines = String::new();
    for g in &fit.groups {
        for at in GRID {
            let s = step_at(&g.time, &g.surv, at, 1.0);
            let k = g.time.partition_point(|&t| t <= at);
            let se = if k == 0 {
                0.0
            } else {
                g.se[k - 1].unwrap_or(f64::NAN)
            };
            let _ = writeln!(
                lines,
                "surv_arm{}_t{at},{s}\nse_arm{}_t{at},{se}",
                g.group, g.group
            );
        }
    }
    Timed { times, lines }
}

fn run_multistate(cut: bool, c: &Cohort, reps: usize) -> Timed {
    // The first leg starts in state 1 and ends in illness (state 2, status 1)
    // or death (state 3, status 2). The second leg starts in state 2 and ends
    // in death. With `cut`, a leg crossing the cut point is two rows.
    let mut id = Vec::new();
    let mut start = Vec::new();
    let mut stop = Vec::new();
    let mut status = Vec::new();
    let mut istate = Vec::new();
    let leg = |id: &mut Vec<u32>,
               st: &mut Vec<f64>,
               sp: &mut Vec<f64>,
               s: &mut Vec<i32>,
               is: &mut Vec<u32>,
               i: u32,
               a: f64,
               b: f64,
               to: i32,
               from: u32| {
        let mut pieces = vec![a];
        if cut && a < CUT && b > CUT {
            pieces.push(CUT);
        }
        pieces.push(b);
        for w in pieces.windows(2) {
            id.push(i);
            st.push(w[0]);
            sp.push(w[1]);
            s.push(if w[1] == b { to } else { 0 });
            is.push(from);
        }
    };
    for i in 0..c.time.len() {
        let to = match c.status[i] {
            1 => 2,
            2 => 3,
            _ => 0,
        };
        leg(
            &mut id,
            &mut start,
            &mut stop,
            &mut status,
            &mut istate,
            i as u32,
            c.entry[i],
            c.time[i],
            to,
            1,
        );
        if c.status[i] == 1 && !c.t2[i].is_nan() {
            let to2 = if c.s2[i] == 3 { 3 } else { 0 };
            leg(
                &mut id,
                &mut start,
                &mut stop,
                &mut status,
                &mut istate,
                i as u32,
                c.time[i],
                c.t2[i],
                to2,
                2,
            );
        }
    }
    let data = SurvData::builder(&stop, &status)
        .start(&start)
        .id(&id)
        .istate(&istate)
        .n_codes(3)
        .build()
        .expect("data");
    // The estimate advances at the times it is given, so these are every
    // distinct stop.
    let mut event_times = stop.clone();
    event_times.sort_by(f64::total_cmp);
    event_times.dedup();
    let (times, est) = timed(reps, || {
        multistate::estimate(
            &data,
            &event_times,
            &multistate::MultistateOptions::default(),
        )
        .expect("multistate")
    });
    let mut lines = String::new();
    for at in GRID {
        let k = event_times.partition_point(|&t| t <= at);
        for s in 0..est.n_states {
            let p = if k == 0 {
                est.initial[s]
            } else {
                est.probabilities[(k - 1) * est.n_states + s]
            };
            let _ = writeln!(lines, "p{}_t{at},{p}", s + 1);
        }
    }
    Timed { times, lines }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
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
    let c = read_cohort(&cohort_path);
    if method == "load" {
        println!("loaded {} subjects", c.time.len());
        return;
    }
    let (base, cut) = method
        .strip_suffix("_split")
        .map_or((method.as_str(), false), |b| (b, true));
    let frame = if cut {
        Frame::cut(&c)
    } else {
        Frame::whole(&c)
    };
    let result = match base {
        "split" => {
            let data = SurvData::builder(&c.time, &c.status)
                .start(&c.entry)
                .n_codes(2)
                .build()
                .expect("data");
            let (times, s) = timed(reps, || {
                split(&data, &[CUT], &SplitOptions::default()).expect("split")
            });
            Timed {
                times,
                lines: format!("pieces,{}\n", s.len()),
            }
        }
        "cox_breslow" => run_cox(Ties::Breslow, &frame, &c, reps),
        "cox_efron" => run_cox(Ties::Efron, &frame, &c, reps),
        "cox_exact" => run_cox(Ties::Exact, &frame, &c, reps),
        "aj" => run_aj(&frame, &c, reps),
        "aj_boot" => run_aj_boot(&c, reps),
        "finegray" => run_finegray(&c, reps),
        "survfit" => run_survfit(&frame, &c, reps),
        "multistate" | "msurvfit" => run_multistate(cut, &c, reps),
        other => panic!("unknown method {other}"),
    };
    fs::write(
        out.join(format!("rust_{method}_{n}.csv")),
        format!("name,value\n{}", result.lines),
    )
    .expect("write");
    let mut times = result.times.clone();
    let med = median(&mut times);
    let min = times.first().copied().unwrap_or(f64::NAN);
    println!(
        "n {n} {method}: median {med:.3} s, min {min:.3} s over {} runs",
        result.times.len()
    );
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out.join("rust_times.csv"))
        .expect("times file");
    writeln!(
        file,
        "{n},{method},{},{med:.4},{min:.4}",
        result.times.len()
    )
    .expect("write times");
}
