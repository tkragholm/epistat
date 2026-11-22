//! Measures the resident memory that a [`Cohort`] takes per unit.
//!
//! ```text
//! cargo run --release -p epistat-matching --example cohort_memory -- [units]
//! ```
//!
//! The cohort has a unique id, a birth date, five strata of twenty values each
//! and five numeric covariates with no missing value. The program prints the
//! resident set of the process, read with `ps`, after the cohort is built. The
//! growth between two sizes divided by the difference in units is the memory
//! per unit.

use chrono::NaiveDate;
use epistat_matching::Cohort;

fn resident_kib() -> u64 {
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .expect("ps runs");
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .expect("a number of kibibytes")
}

fn main() {
    let units: usize = std::env::args()
        .nth(1)
        .map_or(1_000_000, |text| text.parse().expect("a number of units"));
    let origin = NaiveDate::from_ymd_opt(2000, 1, 1).unwrap();
    let cohort = {
        let ids: Vec<String> = (0..units).map(|i| format!("p{i:08}")).collect();
        let births: Vec<NaiveDate> = (0..units)
            .map(|i| origin + chrono::Days::new((i % 3650) as u64))
            .collect();
        let mut builder = Cohort::builder(ids, births);
        for k in 0..5_usize {
            builder = builder.stratum(
                format!("s{k}"),
                (0..units).map(|i| Some(format!("v{}", (i * (k + 3)) % 20))),
            );
            builder = builder.numeric_covariate(
                format!("x{k}"),
                (0..units).map(|i| Some(((i * (k + 7)) % 1000) as f64)),
            );
        }
        let cohort = builder.build().expect("a valid cohort");
        println!("{units} units, resident {} KiB", resident_kib());
        cohort
    };
    std::hint::black_box(&cohort);
}
