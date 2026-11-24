//! Shared by the `tier1_*` benchmark programs of the crates: environment,
//! the data generator's random stream, reading the data files that R reads
//! too, timing, and the result files.
//!
//! One process times one method on one size, so that `/usr/bin/time -v`
//! gives the peak memory of that run alone. `bench/run.sh` is the
//! driver. The environment sets the work:
//!
//! * `SCRATCH`: directory for the data files and the results.
//! * `N`: the size, as the driver defines it for the method.
//! * `METHOD`: `gen_KIND` writes the data file of that kind and stops. Any
//!   other name is a method of the program.
//! * `REPS`: timed repetitions, default 3. The median is reported.
//! * `OUT`: directory for the results, default `SCRATCH/out`.
//!
//! Each run writes `OUT/rust_METHOD_N.csv` with the numbers the two sides are
//! compared on, as `name,value` lines, and appends a line to
//! `OUT/rust_times.csv`. Only the method is timed: reading the file and
//! building the inputs are not.

#![allow(
    dead_code,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a benchmark program: each program uses part of this module, and sizes and codes are small whole numbers"
)]

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::io::{BufRead, Write as _};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub fn scratch() -> PathBuf {
    PathBuf::from(
        std::env::var("SCRATCH").expect("SCRATCH names a directory outside the repository"),
    )
}

pub fn out_dir() -> PathBuf {
    std::env::var("OUT").map_or_else(|_| scratch().join("out"), PathBuf::from)
}

pub fn size() -> usize {
    std::env::var("N")
        .expect("N is the size")
        .parse()
        .expect("N is a whole number")
}

pub fn method() -> String {
    std::env::var("METHOD").expect("METHOD names the method")
}

pub fn reps() -> usize {
    std::env::var("REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3)
}

pub fn data_path(kind: &str, n: usize) -> PathBuf {
    scratch().join(format!("{kind}_{n}.csv"))
}

/// A SplitMix64 stream, the same for every kind of data.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform on (0, 1).
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    pub fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }

    /// A whole number in `0..k`.
    pub fn below(&mut self, k: u64) -> u64 {
        self.next_u64() % k
    }

    pub fn bernoulli(&mut self, p: f64) -> bool {
        self.uniform() < p
    }

    /// Gamma with the given shape and unit scale, by Marsaglia and Tsang, for
    /// shape of at least 1.
    pub fn gamma(&mut self, shape: f64) -> f64 {
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = (1.0 + c * x).powi(3);
            if v > 0.0 && self.uniform().ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }

    /// Poisson, by inversion for a small mean and a normal approximation
    /// above 30.
    pub fn poisson(&mut self, mean: f64) -> f64 {
        if mean > 30.0 {
            return (mean + mean.sqrt() * self.normal()).round().max(0.0);
        }
        let limit = (-mean).exp();
        let (mut k, mut p) = (0.0, 1.0);
        loop {
            p *= self.uniform();
            if p <= limit {
                return k;
            }
            k += 1.0;
        }
    }
}

/// A buffered writer for a data file, with the header written.
pub struct Writer {
    out: std::io::BufWriter<fs::File>,
}

impl Writer {
    pub fn create(path: &Path, header: &str) -> Self {
        let mut out = std::io::BufWriter::with_capacity(
            1 << 20,
            fs::File::create(path).expect("create data file"),
        );
        writeln!(out, "{header}").expect("write header");
        Self { out }
    }

    /// One row of numbers, shortest round-trip form.
    pub fn row(&mut self, values: &[f64]) {
        let mut line = String::with_capacity(values.len() * 12);
        for (i, v) in values.iter().enumerate() {
            if i > 0 {
                line.push(',');
            }
            if v.is_nan() {
                line.push_str("NA");
            } else {
                let _ = write!(line, "{v}");
            }
        }
        writeln!(self.out, "{line}").expect("write row");
    }

    pub fn raw(&mut self, line: &str) {
        writeln!(self.out, "{line}").expect("write row");
    }
}

/// The numeric columns of a data file, by name. `NA` reads as NaN.
pub struct Table {
    names: Vec<String>,
    columns: Vec<Vec<f64>>,
}

impl Table {
    pub fn read(path: &Path) -> Self {
        let file = fs::File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
        let mut lines = std::io::BufReader::with_capacity(1 << 20, file).lines();
        let header = lines.next().expect("header").expect("read header");
        let names: Vec<String> = header.split(',').map(str::to_string).collect();
        let mut columns = vec![Vec::new(); names.len()];
        for line in lines {
            let line = line.expect("read line");
            for (column, field) in columns.iter_mut().zip(line.split(',')) {
                column.push(field.parse::<f64>().unwrap_or(f64::NAN));
            }
        }
        Self { names, columns }
    }

    /// [`Table::read`] with the columns allocated once at the number of rows,
    /// found by a first pass over the file, so that a large file takes the
    /// memory of its values and not that of doubling vectors.
    pub fn read_sized(path: &Path) -> Self {
        use std::io::Read;
        let mut file =
            fs::File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
        let mut buffer = vec![0u8; 1 << 20];
        let mut newlines = 0usize;
        loop {
            let got = file.read(&mut buffer).expect("read");
            if got == 0 {
                break;
            }
            newlines += buffer[..got].iter().filter(|&&b| b == b'\n').count();
        }
        let file = fs::File::open(path).expect("open");
        let mut lines = std::io::BufReader::with_capacity(1 << 20, file).lines();
        let header = lines.next().expect("header").expect("read header");
        let names: Vec<String> = header.split(',').map(str::to_string).collect();
        let mut columns = vec![Vec::with_capacity(newlines); names.len()];
        for line in lines {
            let line = line.expect("read line");
            for (column, field) in columns.iter_mut().zip(line.split(',')) {
                column.push(field.parse::<f64>().unwrap_or(f64::NAN));
            }
        }
        Self { names, columns }
    }

    pub fn col(&self, name: &str) -> &[f64] {
        let k = self
            .names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("no column {name}"));
        &self.columns[k]
    }

    pub fn ints<T: TryFrom<i64>>(&self, name: &str) -> Vec<T> {
        self.col(name)
            .iter()
            .map(|&v| T::try_from(v as i64).unwrap_or_else(|_| panic!("{name}: {v} out of range")))
            .collect()
    }

    pub fn flags(&self, name: &str) -> Vec<bool> {
        self.col(name).iter().map(|&v| v != 0.0).collect()
    }

    pub fn len(&self) -> usize {
        self.columns.first().map_or(0, Vec::len)
    }
}

/// The lines of a file after its header, split on commas.
pub fn read_text_rows(path: &Path) -> Vec<Vec<String>> {
    let file = fs::File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    std::io::BufReader::with_capacity(1 << 20, file)
        .lines()
        .skip(1)
        .map(|l| {
            l.expect("read line")
                .split(',')
                .map(str::to_string)
                .collect()
        })
        .collect()
}

pub fn timed<T>(reps: usize, mut run: impl FnMut() -> T) -> (Vec<f64>, T) {
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

/// The numbers a run is compared on.
#[derive(Default)]
pub struct Lines(String);

impl Lines {
    pub fn push(&mut self, name: &str, value: f64) {
        let _ = writeln!(self.0, "{name},{value}");
    }

    pub fn push_opt(&mut self, name: &str, value: Option<f64>) {
        self.push(name, value.unwrap_or(f64::NAN));
    }

    pub fn push_all(&mut self, prefix: &str, values: &[f64]) {
        for (i, &v) in values.iter().enumerate() {
            self.push(&format!("{prefix}{i}"), v);
        }
    }
}

/// Writes the results and the time of a run.
pub fn finish(method: &str, n: usize, times: &[f64], lines: &Lines) {
    let out = out_dir();
    fs::create_dir_all(&out).expect("create OUT");
    fs::write(
        out.join(format!("rust_{method}_{n}.csv")),
        format!("name,value\n{}", lines.0),
    )
    .expect("write results");
    let mut sorted = times.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    let min = sorted[0];
    let path = out.join("rust_times.csv");
    let fresh = !path.exists();
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("open times");
    if fresh {
        writeln!(file, "n,method,reps,median_s,min_s").expect("write header");
    }
    writeln!(file, "{n},{method},{},{median},{min}", times.len()).expect("write time");
}

/// Counts the entries of a map, for the results of a keyed method.
pub fn tally<K: std::hash::Hash + Eq>(items: impl IntoIterator<Item = K>) -> HashMap<K, usize> {
    let mut map = HashMap::new();
    for k in items {
        *map.entry(k).or_insert(0) += 1;
    }
    map
}
