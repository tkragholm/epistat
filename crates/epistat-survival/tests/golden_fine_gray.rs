//! The Fine-Gray rows and fit, pinned to their bits. Downstream studies depend
//! on these numbers, so this file must pass unchanged through any change to
//! the module.
//!
//! The inputs come from a seeded stream. Each output column is reduced to an
//! FNV-1a hash of its bit patterns ([`Fingerprint`]) and each scalar of a fit
//! is kept as its bits, so a change of the last bit of any value fails. The
//! fit uses `exp` from the platform's maths library, so the values were
//! recorded on aarch64 macOS and another libm could differ in the last bit.
//!
//! To record the table again after an intended change, run with
//! `GOLDEN_RECORD=1 cargo test --test golden_fine_gray -- --nocapture` and
//! paste the printed lines into `EXPECTED`.
//!
//! The bits depend on the platform's `ln` and `exp`, which come from its maths
//! library, so the test asserts the recorded values only on the target it was
//! recorded on, aarch64 macOS. CI's macOS runner is that target. Thread-count
//! independence, which does not depend on the platform, is checked everywhere
//! by the thread tests. Elsewhere it computes the hashes and, when
//! `EPISTAT_HASH_OUT` is set, appends them to that file.

#![expect(
    clippy::unreadable_literal,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    reason = "the table holds recorded bit patterns, and the counts are a few hundred"
)]

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::cox::{CoxModelFit, Ties};
use epistat_survival::fine_gray::{
    FineGrayOptions, FineGrayRows, RowsOptions, fit, risk_rows, rows,
};
use epistat_survival::lowlevel::float_codes;
use epistat_survival::timefix::TimeFix;
use epistat_testkit::Fingerprint;

mod input {
    use super::Stream;

    pub struct Sample {
        pub name: &'static str,
        pub time: Vec<f64>,
        pub event: Vec<i32>,
        pub weight: Vec<f64>,
        pub x: Vec<f64>,
        pub group: Vec<u32>,
    }

    /// `causes` is the number of failure causes, `grid` the number of distinct
    /// time values, `scale` the spacing of the grid, `zeros` the share of
    /// weights set to 0 in percent, `fractional` whether weights vary, and
    /// `censored` the share of censored rows in percent.
    pub fn sample(
        name: &'static str,
        n: usize,
        seed: u64,
        causes: usize,
        grid: usize,
        scale: f64,
        zeros: usize,
        fractional: bool,
        censored: usize,
    ) -> Sample {
        let mut rng = Stream::seeded(seed);
        let mut s = Sample {
            name,
            time: Vec::new(),
            event: Vec::new(),
            weight: Vec::new(),
            x: Vec::new(),
            group: Vec::new(),
        };
        for _ in 0..n {
            s.time.push((1 + rng.below(grid)) as f64 * scale);
            s.event.push(if rng.below(100) < censored {
                0
            } else {
                1 + rng.below(causes) as i32
            });
            s.weight.push(if rng.below(100) < zeros {
                0.0
            } else if fractional {
                0.25 + rng.uniform() * 2.0
            } else {
                1.0
            });
            s.x.push(rng.below(5) as f64 * 0.5 - 0.5);
            s.group.push(rng.below(9) as u32);
        }
        s
    }
}

fn samples() -> Vec<input::Sample> {
    let mut all = vec![
        // Tied integer days, three causes, unit weights.
        input::sample("ties", 60, 101, 3, 15, 1.0, 0, false, 30),
        // Fractional weights with zeros, times on a tenth grid.
        input::sample("weights", 80, 102, 2, 40, 0.1, 15, true, 35),
        // Little censoring, so G stays near 1 and most weights are 1.
        input::sample("sparse_censoring", 50, 103, 2, 30, 0.5, 0, true, 5),
        // No censoring at all.
        input::sample("no_censoring", 30, 104, 2, 12, 1.0, 0, false, 0),
        // Mostly censored, so G falls to zero or near it.
        input::sample("heavy_censoring", 45, 105, 2, 8, 1.0, 10, true, 85),
    ];
    // A continuous covariate with every value distinct, and times without ties.
    let mut rng = Stream::seeded(106);
    let mut distinct = input::sample("distinct", 40, 106, 2, 1, 1.0, 5, true, 30);
    for i in 0..40 {
        distinct.time[i] = 0.5 + i as f64 * 0.37 + rng.uniform() * 0.01;
        distinct.x[i] = rng.uniform() * 3.0 - 1.0;
    }
    all.push(distinct);
    all
}

fn bits(values: &[f64]) -> u64 {
    Fingerprint::new().f64s(values).finish()
}

fn rows_record(rows: &FineGrayRows) -> Vec<u64> {
    let mut out = vec![rows.row.len() as u64];
    out.push(
        rows.row
            .iter()
            .fold(Fingerprint::new(), |hash, &r| hash.u64(u64::from(r)))
            .finish(),
    );
    out.push(bits(&rows.start));
    out.push(bits(&rows.stop));
    out.push(
        rows.status
            .iter()
            .fold(Fingerprint::new(), |hash, &e| hash.u64(e as u64))
            .finish(),
    );
    out.push(bits(&rows.weight));
    out
}

fn opt_bits(value: Option<f64>) -> u64 {
    value.map_or(u64::MAX, f64::to_bits)
}

fn fit_record(fit: Result<CoxModelFit, epistat_survival::Error>) -> Vec<u64> {
    match fit {
        Ok(fit) => vec![
            opt_bits(fit.coefficients[0]),
            opt_bits(fit.se[0]),
            fit.n_events.to_bits(),
            fit.n_rows as u64,
            u64::try_from(fit.iterations).unwrap(),
            u64::from(fit.converged),
            opt_bits(fit.robust_se.and_then(|v| v[0])),
        ],
        Err(error) => {
            let text = error.to_string();
            vec![
                u64::MAX - 1,
                text.bytes()
                    .fold(Fingerprint::new(), |hash, b| hash.u64(u64::from(b)))
                    .finish(),
            ]
        }
    }
}

/// Times are compared as they are.
const ROWS: RowsOptions = RowsOptions {
    timefix: TimeFix::Exact,
};

fn options(cause: u32, ties: Ties) -> FineGrayOptions {
    FineGrayOptions {
        ties,
        timefix: TimeFix::Exact,
        ..FineGrayOptions::new(cause)
    }
}

/// One line per (sample, cause, output), each a vector of recorded bits.
fn record() -> Vec<(String, Vec<u64>)> {
    let mut lines = Vec::new();
    for s in samples() {
        let n = s.time.len();
        let own: Vec<u32> = (0..n as u32).collect();
        let by_x = float_codes(&s.x);
        let x = Mat::from_fn(n, 1, |i, _| s.x[i]);
        let plain = SurvData::builder(&s.time, &s.event)
            .n_codes(3)
            .weight(&s.weight)
            .build()
            .unwrap();
        for cause in 1..=3 {
            let tag = format!("{} cause {cause}", s.name);
            lines.push((
                format!("{tag} expand"),
                rows_record(&rows(&plain, cause, &ROWS).unwrap()),
            ));
            lines.push((
                format!("{tag} risk_rows x"),
                rows_record(&risk_rows(&plain, cause, &by_x, &ROWS).unwrap()),
            ));
            lines.push((
                format!("{tag} risk_rows group"),
                rows_record(&risk_rows(&plain, cause, &s.group, &ROWS).unwrap()),
            ));
            for (ties_name, ties) in [("breslow", Ties::Breslow), ("efron", Ties::Efron)] {
                for (cluster_name, cluster) in [
                    ("none", None),
                    ("subject", Some(own.as_slice())),
                    ("group", Some(s.group.as_slice())),
                ] {
                    let mut builder = SurvData::builder(&s.time, &s.event)
                        .n_codes(3)
                        .weight(&s.weight);
                    if let Some(cluster) = cluster {
                        builder = builder.cluster(cluster);
                    }
                    let data = builder.build().unwrap();
                    let model = fit(&data, x.as_ref(), &options(cause, ties));
                    lines.push((
                        format!("{tag} fit {ties_name} {cluster_name}"),
                        fit_record(model),
                    ));
                }
            }
        }
    }
    lines
}

#[test]
fn refusals_keep_their_messages() {
    let s = &samples()[0];
    let n = s.time.len();
    let x = Mat::from_fn(n, 1, |i, _| s.x[i]);
    let data = SurvData::builder(&s.time, &s.event)
        .n_codes(3)
        .weight(&s.weight)
        .build()
        .unwrap();
    let zero = fit(&data, x.as_ref(), &options(0, Ties::Breslow));
    assert_eq!(
        zero.unwrap_err().to_string(),
        "cause 0 is outside 1..=3; 0 is censoring"
    );
    let short = Mat::from_fn(n - 1, 1, |i, _| s.x[i]);
    let short = fit(&data, short.as_ref(), &options(1, Ties::Breslow));
    assert_eq!(
        short.unwrap_err().to_string(),
        "x has length 59, expected 60"
    );
    assert_eq!(
        risk_rows(&data, 1, &s.group[1..], &ROWS)
            .unwrap_err()
            .to_string(),
        "level has length 59, expected 60"
    );
}

#[test]
fn rows_and_fits_keep_their_bits() {
    let lines = record();
    for (label, values) in &lines {
        epistat_testkit::hash_out_words(label, values);
    }
    if std::env::var_os("GOLDEN_RECORD").is_some() {
        for (label, values) in &lines {
            println!("    (\"{label}\", &{values:?}),");
        }
        return;
    }
    if !epistat_testkit::RECORDED_PLATFORM {
        return;
    }
    assert_eq!(lines.len(), EXPECTED.len());
    for ((label, values), (want_label, want)) in lines.iter().zip(EXPECTED) {
        assert_eq!(label, want_label);
        assert_eq!(values.as_slice(), *want, "{label}");
    }
}

const EXPECTED: &[(&str, &[u64])] = &[
    (
        "ties cause 1 expand",
        &[
            211,
            10856076611835182342,
            13052634834822312299,
            62115402625604898,
            5405767350661835236,
            4262778527861163447,
        ],
    ),
    (
        "ties cause 1 risk_rows x",
        &[
            100,
            236572699758858609,
            9229098289935994007,
            13884464757240149714,
            6786258010436106500,
            8835683452085250323,
        ],
    ),
    (
        "ties cause 1 risk_rows group",
        &[
            132,
            6738029080708519516,
            8935923560406703833,
            943425153074167790,
            10330306353812834564,
            10667542433383554811,
        ],
    ),
    (
        "ties cause 1 fit breslow none",
        &[
            13824713134326763600,
            4599953140261006895,
            4625478292286210048,
            60,
            3,
            1,
            18446744073709551615,
        ],
    ),
    (
        "ties cause 1 fit breslow subject",
        &[
            13824713134326763600,
            4599953140261006895,
            4625478292286210048,
            60,
            3,
            1,
            4599752772874865686,
        ],
    ),
    (
        "ties cause 1 fit breslow group",
        &[
            13824713134326763600,
            4599953140261006895,
            4625478292286210048,
            60,
            3,
            1,
            4599553630932595962,
        ],
    ),
    (
        "ties cause 1 fit efron none",
        &[
            13824916865210317403,
            4599965971821566203,
            4625478292286210048,
            60,
            3,
            1,
            18446744073709551615,
        ],
    ),
    (
        "ties cause 1 fit efron subject",
        &[
            13824916865210317403,
            4599965971821566203,
            4625478292286210048,
            60,
            3,
            1,
            4599896268221748909,
        ],
    ),
    (
        "ties cause 1 fit efron group",
        &[
            13824916865210317403,
            4599965971821566203,
            4625478292286210048,
            60,
            3,
            1,
            4599715551434294345,
        ],
    ),
    (
        "ties cause 2 expand",
        &[
            251,
            2443467642507561508,
            12143683821322375353,
            12367574682993049422,
            16965917916449642148,
            4997845607109364236,
        ],
    ),
    (
        "ties cause 2 risk_rows x",
        &[
            112,
            17595453846443733987,
            500679367268474943,
            15565339960497036642,
            8901546906844172516,
            4996226347465383691,
        ],
    ),
    (
        "ties cause 2 risk_rows group",
        &[
            144,
            12764132777697149420,
            14301852340330520261,
            12149558457898783484,
            7488780664577466596,
            764847748278807463,
        ],
    ),
    (
        "ties cause 2 fit breslow none",
        &[
            13828355714701410971,
            4600001738175051036,
            4625478292286210048,
            60,
            3,
            1,
            18446744073709551615,
        ],
    ),
    (
        "ties cause 2 fit breslow subject",
        &[
            13828355714701410971,
            4600001738175051036,
            4625478292286210048,
            60,
            3,
            1,
            4599233148931055660,
        ],
    ),
    (
        "ties cause 2 fit breslow group",
        &[
            13828355714701410971,
            4600001738175051036,
            4625478292286210048,
            60,
            3,
            1,
            4600946326514264606,
        ],
    ),
    (
        "ties cause 2 fit efron none",
        &[
            13828435015447412243,
            4599981950940338486,
            4625478292286210048,
            60,
            3,
            1,
            18446744073709551615,
        ],
    ),
    (
        "ties cause 2 fit efron subject",
        &[
            13828435015447412243,
            4599981950940338486,
            4625478292286210048,
            60,
            3,
            1,
            4599299621709840322,
        ],
    ),
    (
        "ties cause 2 fit efron group",
        &[
            13828435015447412243,
            4599981950940338486,
            4625478292286210048,
            60,
            3,
            1,
            4600973893517090440,
        ],
    ),
    (
        "ties cause 3 expand",
        &[
            292,
            5712757542556107079,
            11604623004953293015,
            14645855525529024705,
            11093362456315923397,
            7257149509249627550,
        ],
    ),
    (
        "ties cause 3 risk_rows x",
        &[
            112,
            14201051615744436617,
            4938364389814458047,
            3669926632335119549,
            2337462454781012293,
            7860113056242037271,
        ],
    ),
    (
        "ties cause 3 risk_rows group",
        &[
            145,
            12567624057623145546,
            8255429610389643569,
            8727269496423081927,
            18301800416918869989,
            3572614426222386878,
        ],
    ),
    (
        "ties cause 3 fit breslow none",
        &[
            4609296462439179559,
            4603077916696254587,
            4622945017495814144,
            60,
            4,
            1,
            18446744073709551615,
        ],
    ),
    (
        "ties cause 3 fit breslow subject",
        &[
            4609296462439179559,
            4603077916696254587,
            4622945017495814144,
            60,
            4,
            1,
            4601795004584889773,
        ],
    ),
    (
        "ties cause 3 fit breslow group",
        &[
            4609296462439179559,
            4603077916696254587,
            4622945017495814144,
            60,
            4,
            1,
            4603631229564008265,
        ],
    ),
    (
        "ties cause 3 fit efron none",
        &[
            4609347257950544837,
            4603066089085990731,
            4622945017495814144,
            60,
            4,
            1,
            18446744073709551615,
        ],
    ),
    (
        "ties cause 3 fit efron subject",
        &[
            4609347257950544837,
            4603066089085990731,
            4622945017495814144,
            60,
            4,
            1,
            4601845595575003382,
        ],
    ),
    (
        "ties cause 3 fit efron group",
        &[
            4609347257950544837,
            4603066089085990731,
            4622945017495814144,
            60,
            4,
            1,
            4603663053426098730,
        ],
    ),
    (
        "weights cause 1 expand",
        &[
            246,
            16871520064288269269,
            11435149874650908369,
            14059440426762361019,
            4873561969183286277,
            17442036148268203081,
        ],
    ),
    (
        "weights cause 1 risk_rows x",
        &[
            134,
            7378018476476819912,
            14111022479303546678,
            723863529486589483,
            2697395664707952261,
            6651329354937865403,
        ],
    ),
    (
        "weights cause 1 risk_rows group",
        &[
            154,
            3322000605866750506,
            10828022720953986703,
            1519690517846048772,
            8185876718030793477,
            13376841320834613265,
        ],
    ),
    (
        "weights cause 1 fit breslow none",
        &[
            13814062323896115295,
            4598444358967793673,
            4627161928506626658,
            80,
            3,
            1,
            18446744073709551615,
        ],
    ),
    (
        "weights cause 1 fit breslow subject",
        &[
            13814062323896115295,
            4598444358967793673,
            4627161928506626658,
            80,
            3,
            1,
            4599192691152504289,
        ],
    ),
    (
        "weights cause 1 fit breslow group",
        &[
            13814062323896115295,
            4598444358967793673,
            4627161928506626658,
            80,
            3,
            1,
            4599248173466055055,
        ],
    ),
    (
        "weights cause 1 fit efron none",
        &[
            13814013981413712650,
            4598450349758601822,
            4627161928506626658,
            80,
            3,
            1,
            18446744073709551615,
        ],
    ),
    (
        "weights cause 1 fit efron subject",
        &[
            13814013981413712650,
            4598450349758601822,
            4627161928506626658,
            80,
            3,
            1,
            4599256032667280023,
        ],
    ),
    (
        "weights cause 1 fit efron group",
        &[
            13814013981413712650,
            4598450349758601822,
            4627161928506626658,
            80,
            3,
            1,
            4599308071827634306,
        ],
    ),
    (
        "weights cause 2 expand",
        &[
            212,
            11428436244470903030,
            7753462861293012827,
            5294445455955987933,
            6221535312952219588,
            13743311335743834849,
        ],
    ),
    (
        "weights cause 2 risk_rows x",
        &[
            119,
            7535713886389683696,
            15665163584935098496,
            16973881641205677233,
            17154761378921398564,
            6877620174133733545,
        ],
    ),
    (
        "weights cause 2 risk_rows group",
        &[
            154,
            14145592710185389394,
            10906686536659444690,
            8198601775181558953,
            6175217057858660004,
            13013423917665902134,
        ],
    ),
    (
        "weights cause 2 fit breslow none",
        &[
            13802720377640997635,
            4596992799965371572,
            4630119498190085661,
            80,
            2,
            1,
            18446744073709551615,
        ],
    ),
    (
        "weights cause 2 fit breslow subject",
        &[
            13802720377640997635,
            4596992799965371572,
            4630119498190085661,
            80,
            2,
            1,
            4598127984695049302,
        ],
    ),
    (
        "weights cause 2 fit breslow group",
        &[
            13802720377640997635,
            4596992799965371572,
            4630119498190085661,
            80,
            2,
            1,
            4596765342416537613,
        ],
    ),
    (
        "weights cause 2 fit efron none",
        &[
            13800666114049281754,
            4597002799028538712,
            4630119498190085661,
            80,
            2,
            1,
            18446744073709551615,
        ],
    ),
    (
        "weights cause 2 fit efron subject",
        &[
            13800666114049281754,
            4597002799028538712,
            4630119498190085661,
            80,
            2,
            1,
            4598224596143945822,
        ],
    ),
    (
        "weights cause 2 fit efron group",
        &[
            13800666114049281754,
            4597002799028538712,
            4630119498190085661,
            80,
            2,
            1,
            4596913015250267063,
        ],
    ),
    (
        "weights cause 3 expand",
        &[
            386,
            10790460187734859003,
            3062292712778816579,
            3496465401357817100,
            1429458144501584997,
            5313587367904876045,
        ],
    ),
    (
        "weights cause 3 risk_rows x",
        &[
            124,
            349893181861787612,
            6273703596785895102,
            4954709066762944258,
            11984747362187796645,
            4605039417868866631,
        ],
    ),
    (
        "weights cause 3 risk_rows group",
        &[
            183,
            15307932775175031681,
            97872507278031598,
            6858135935604706267,
            12501296745194705797,
            18010321133902356922,
        ],
    ),
    (
        "weights cause 3 fit breslow none",
        &[
            18446744073709551615,
            18446744073709551615,
            9223372036854775808,
            80,
            0,
            0,
            18446744073709551615,
        ],
    ),
    (
        "weights cause 3 fit breslow subject",
        &[
            18446744073709551615,
            18446744073709551615,
            9223372036854775808,
            80,
            0,
            0,
            18446744073709551615,
        ],
    ),
    (
        "weights cause 3 fit breslow group",
        &[
            18446744073709551615,
            18446744073709551615,
            9223372036854775808,
            80,
            0,
            0,
            18446744073709551615,
        ],
    ),
    (
        "weights cause 3 fit efron none",
        &[
            18446744073709551615,
            18446744073709551615,
            9223372036854775808,
            80,
            0,
            0,
            18446744073709551615,
        ],
    ),
    (
        "weights cause 3 fit efron subject",
        &[
            18446744073709551615,
            18446744073709551615,
            9223372036854775808,
            80,
            0,
            0,
            18446744073709551615,
        ],
    ),
    (
        "weights cause 3 fit efron group",
        &[
            18446744073709551615,
            18446744073709551615,
            9223372036854775808,
            80,
            0,
            0,
            18446744073709551615,
        ],
    ),
    (
        "sparse_censoring cause 1 expand",
        &[
            56,
            9598978640730707184,
            3696304918593447205,
            11693942316453486002,
            868853621252584452,
            3538253526420438881,
        ],
    ),
    (
        "sparse_censoring cause 1 risk_rows x",
        &[
            38,
            9077188240731320108,
            1007474159242842597,
            14778134419601264498,
            2413925923570229636,
            13179763808482097750,
        ],
    ),
    (
        "sparse_censoring cause 1 risk_rows group",
        &[
            37,
            12575835416076693325,
            2530698711355456261,
            10338742761297994802,
            9695245388369385732,
            16828159929900191360,
        ],
    ),
    (
        "sparse_censoring cause 1 fit breslow none",
        &[
            13819148473670378323,
            4598496587858753187,
            4628411439846819830,
            50,
            3,
            1,
            18446744073709551615,
        ],
    ),
    (
        "sparse_censoring cause 1 fit breslow subject",
        &[
            13819148473670378323,
            4598496587858753187,
            4628411439846819830,
            50,
            3,
            1,
            4598891886707012207,
        ],
    ),
    (
        "sparse_censoring cause 1 fit breslow group",
        &[
            13819148473670378323,
            4598496587858753187,
            4628411439846819830,
            50,
            3,
            1,
            4598793672878178924,
        ],
    ),
    (
        "sparse_censoring cause 1 fit efron none",
        &[
            13819205274698375977,
            4598486499876981970,
            4628411439846819830,
            50,
            3,
            1,
            18446744073709551615,
        ],
    ),
    (
        "sparse_censoring cause 1 fit efron subject",
        &[
            13819205274698375977,
            4598486499876981970,
            4628411439846819830,
            50,
            3,
            1,
            4598944526485114153,
        ],
    ),
    (
        "sparse_censoring cause 1 fit efron group",
        &[
            13819205274698375977,
            4598486499876981970,
            4628411439846819830,
            50,
            3,
            1,
            4598906049038475797,
        ],
    ),
    (
        "sparse_censoring cause 2 expand",
        &[
            54,
            3576150565356984420,
            8115406626727737317,
            16368226751632206329,
            2371563282759667940,
            8621129474972097529,
        ],
    ),
    (
        "sparse_censoring cause 2 risk_rows x",
        &[
            40,
            14765375603444291733,
            646666125706265637,
            2413579687619247241,
            15479066665475216068,
            17256393403359173950,
        ],
    ),
    (
        "sparse_censoring cause 2 risk_rows group",
        &[
            43,
            12191200459979196414,
            8150291629140578821,
            14200899805961370188,
            4151139389291873348,
            6597677488174816186,
        ],
    ),
    (
        "sparse_censoring cause 2 fit breslow none",
        &[
            4594521213482601961,
            4598771393464409094,
            4629626349262449805,
            50,
            2,
            1,
            18446744073709551615,
        ],
    ),
    (
        "sparse_censoring cause 2 fit breslow subject",
        &[
            4594521213482601961,
            4598771393464409094,
            4629626349262449805,
            50,
            2,
            1,
            4600568104376727643,
        ],
    ),
    (
        "sparse_censoring cause 2 fit breslow group",
        &[
            4594521213482601961,
            4598771393464409094,
            4629626349262449805,
            50,
            2,
            1,
            4601343562509935250,
        ],
    ),
    (
        "sparse_censoring cause 2 fit efron none",
        &[
            4594502148686977813,
            4598785196371404500,
            4629626349262449805,
            50,
            2,
            1,
            18446744073709551615,
        ],
    ),
    (
        "sparse_censoring cause 2 fit efron subject",
        &[
            4594502148686977813,
            4598785196371404500,
            4629626349262449805,
            50,
            2,
            1,
            4600729508483460179,
        ],
    ),
    (
        "sparse_censoring cause 2 fit efron group",
        &[
            4594502148686977813,
            4598785196371404500,
            4629626349262449805,
            50,
            2,
            1,
            4601497804308295902,
        ],
    ),
    (
        "sparse_censoring cause 3 expand",
        &[
            60,
            4797026030145165648,
            10184141430607721253,
            8280427456406823137,
            18229908013310916773,
            17487482820706847964,
        ],
    ),
    (
        "sparse_censoring cause 3 risk_rows x",
        &[
            19,
            636391866897387156,
            16623002304131210309,
            1651145092944684656,
            9512054303180107013,
            17306852038852319721,
        ],
    ),
    (
        "sparse_censoring cause 3 risk_rows group",
        &[
            22,
            7280654419630392952,
            6388534975009323813,
            18119081662664955092,
            12501863912485626085,
            5643936800444166210,
        ],
    ),
    (
        "sparse_censoring cause 3 fit breslow none",
        &[
            18446744073709551615,
            18446744073709551615,
            9223372036854775808,
            50,
            0,
            0,
            18446744073709551615,
        ],
    ),
    (
        "sparse_censoring cause 3 fit breslow subject",
        &[
            18446744073709551615,
            18446744073709551615,
            9223372036854775808,
            50,
            0,
            0,
            18446744073709551615,
        ],
    ),
    (
        "sparse_censoring cause 3 fit breslow group",
        &[
            18446744073709551615,
            18446744073709551615,
            9223372036854775808,
            50,
            0,
            0,
            18446744073709551615,
        ],
    ),
    (
        "sparse_censoring cause 3 fit efron none",
        &[
            18446744073709551615,
            18446744073709551615,
            9223372036854775808,
            50,
            0,
            0,
            18446744073709551615,
        ],
    ),
    (
        "sparse_censoring cause 3 fit efron subject",
        &[
            18446744073709551615,
            18446744073709551615,
            9223372036854775808,
            50,
            0,
            0,
            18446744073709551615,
        ],
    ),
    (
        "sparse_censoring cause 3 fit efron group",
        &[
            18446744073709551615,
            18446744073709551615,
            9223372036854775808,
            50,
            0,
            0,
            18446744073709551615,
        ],
    ),
    (
        "no_censoring cause 1 expand",
        &[
            30,
            12483764871328381252,
            16714763329727612389,
            4716477389514470130,
            12876778340707637476,
            9046458546020157061,
        ],
    ),
    (
        "no_censoring cause 1 risk_rows x",
        &[
            17,
            2876579118146379658,
            15633972250446808005,
            2529224602319950106,
            9062387907908035684,
            7661412197263363284,
        ],
    ),
    (
        "no_censoring cause 1 risk_rows group",
        &[
            21,
            5145228337183312678,
            3880543910414739013,
            13591524034668623642,
            17880497505532810852,
            6619950914403202432,
        ],
    ),
    (
        "no_censoring cause 1 fit breslow none",
        &[
            13828946437294407512,
            4600679539854236072,
            4623507967449235456,
            30,
            4,
            1,
            18446744073709551615,
        ],
    ),
    (
        "no_censoring cause 1 fit breslow subject",
        &[
            13828946437294407512,
            4600679539854236072,
            4623507967449235456,
            30,
            4,
            1,
            4599801053142125013,
        ],
    ),
    (
        "no_censoring cause 1 fit breslow group",
        &[
            13828946437294407512,
            4600679539854236072,
            4623507967449235456,
            30,
            4,
            1,
            4599459102598895553,
        ],
    ),
    (
        "no_censoring cause 1 fit efron none",
        &[
            13828943498554049480,
            4600640262099161262,
            4623507967449235456,
            30,
            4,
            1,
            18446744073709551615,
        ],
    ),
    (
        "no_censoring cause 1 fit efron subject",
        &[
            13828943498554049480,
            4600640262099161262,
            4623507967449235456,
            30,
            4,
            1,
            4599823303493869121,
        ],
    ),
    (
        "no_censoring cause 1 fit efron group",
        &[
            13828943498554049480,
            4600640262099161262,
            4623507967449235456,
            30,
            4,
            1,
            4599471571777968733,
        ],
    ),
    (
        "no_censoring cause 2 expand",
        &[
            30,
            12483764871328381252,
            16714763329727612389,
            9985076638466991664,
            30502340153506372,
            9046458546020157061,
        ],
    ),
    (
        "no_censoring cause 2 risk_rows x",
        &[
            21,
            15410764129436597617,
            3880543910414739013,
            1703860002352090072,
            10850728598535330468,
            16275751446235394396,
        ],
    ),
    (
        "no_censoring cause 2 risk_rows group",
        &[
            24,
            2112632967513324420,
            12325268230655562277,
            11999943486399009264,
            15997441650793037860,
            17917831523468065744,
        ],
    ),
    (
        "no_censoring cause 2 fit breslow none",
        &[
            4607030846361181757,
            4601735947462883738,
            4625478292286210048,
            30,
            4,
            1,
            18446744073709551615,
        ],
    ),
    (
        "no_censoring cause 2 fit breslow subject",
        &[
            4607030846361181757,
            4601735947462883738,
            4625478292286210048,
            30,
            4,
            1,
            4602746818826764218,
        ],
    ),
    (
        "no_censoring cause 2 fit breslow group",
        &[
            4607030846361181757,
            4601735947462883738,
            4625478292286210048,
            30,
            4,
            1,
            4602948632445039099,
        ],
    ),
    (
        "no_censoring cause 2 fit efron none",
        &[
            4607373200896029574,
            4601882895242971594,
            4625478292286210048,
            30,
            4,
            1,
            18446744073709551615,
        ],
    ),
    (
        "no_censoring cause 2 fit efron subject",
        &[
            4607373200896029574,
            4601882895242971594,
            4625478292286210048,
            30,
            4,
            1,
            4603038232047293041,
        ],
    ),
    (
        "no_censoring cause 2 fit efron group",
        &[
            4607373200896029574,
            4601882895242971594,
            4625478292286210048,
            30,
            4,
            1,
            4603357674063383632,
        ],
    ),
    (
        "no_censoring cause 3 expand",
        &[
            30,
            12483764871328381252,
            16714763329727612389,
            15529033449697718181,
            16714763329727612389,
            9046458546020157061,
        ],
    ),
    (
        "no_censoring cause 3 risk_rows x",
        &[
            5,
            15915389831978482146,
            4672095441862941765,
            820272556973227805,
            4672095441862941765,
            2347728363511568357,
        ],
    ),
    (
        "no_censoring cause 3 risk_rows group",
        &[
            9,
            8187244242917186181,
            4524767293552705221,
            8526701099964192285,
            4524767293552705221,
            17063880454256911616,
        ],
    ),
    (
        "no_censoring cause 3 fit breslow none",
        &[
            18446744073709551615,
            18446744073709551615,
            9223372036854775808,
            30,
            0,
            0,
            18446744073709551615,
        ],
    ),
    (
        "no_censoring cause 3 fit breslow subject",
        &[
            18446744073709551615,
            18446744073709551615,
            9223372036854775808,
            30,
            0,
            0,
            18446744073709551615,
        ],
    ),
    (
        "no_censoring cause 3 fit breslow group",
        &[
            18446744073709551615,
            18446744073709551615,
            9223372036854775808,
            30,
            0,
            0,
            18446744073709551615,
        ],
    ),
    (
        "no_censoring cause 3 fit efron none",
        &[
            18446744073709551615,
            18446744073709551615,
            9223372036854775808,
            30,
            0,
            0,
            18446744073709551615,
        ],
    ),
    (
        "no_censoring cause 3 fit efron subject",
        &[
            18446744073709551615,
            18446744073709551615,
            9223372036854775808,
            30,
            0,
            0,
            18446744073709551615,
        ],
    ),
    (
        "no_censoring cause 3 fit efron group",
        &[
            18446744073709551615,
            18446744073709551615,
            9223372036854775808,
            30,
            0,
            0,
            18446744073709551615,
        ],
    ),
    (
        "heavy_censoring cause 1 expand",
        &[
            54,
            5764301390649283428,
            13955911671430510885,
            7616329699969204824,
            5435015249376668325,
            10787731004705580810,
        ],
    ),
    (
        "heavy_censoring cause 1 risk_rows x",
        &[
            54,
            10422496540018956419,
            2474400021684882341,
            13328603483229975640,
            16999004171679470917,
            18328093407411229964,
        ],
    ),
    (
        "heavy_censoring cause 1 risk_rows group",
        &[
            54,
            10422496540018956419,
            2474400021684882341,
            13328603483229975640,
            16999004171679470917,
            18328093407411229964,
        ],
    ),
    (
        "heavy_censoring cause 1 fit breslow none",
        &[
            13817880415521270637,
            4603993107718282416,
            4616993744422866754,
            45,
            3,
            1,
            18446744073709551615,
        ],
    ),
    (
        "heavy_censoring cause 1 fit breslow subject",
        &[
            13817880415521270637,
            4603993107718282416,
            4616993744422866754,
            45,
            3,
            1,
            4600992152773540666,
        ],
    ),
    (
        "heavy_censoring cause 1 fit breslow group",
        &[
            13817880415521270637,
            4603993107718282416,
            4616993744422866754,
            45,
            3,
            1,
            4598848281358629444,
        ],
    ),
    (
        "heavy_censoring cause 1 fit efron none",
        &[
            13817880415521270637,
            4603993107718282416,
            4616993744422866754,
            45,
            3,
            1,
            18446744073709551615,
        ],
    ),
    (
        "heavy_censoring cause 1 fit efron subject",
        &[
            13817880415521270637,
            4603993107718282416,
            4616993744422866754,
            45,
            3,
            1,
            4600992152773540666,
        ],
    ),
    (
        "heavy_censoring cause 1 fit efron group",
        &[
            13817880415521270637,
            4603993107718282416,
            4616993744422866754,
            45,
            3,
            1,
            4598848281358629444,
        ],
    ),
    (
        "heavy_censoring cause 2 expand",
        &[
            57,
            11153188455031427882,
            12180525758500628497,
            4418211135398400096,
            8859237251864465445,
            17411577952684735420,
        ],
    ),
    (
        "heavy_censoring cause 2 risk_rows x",
        &[
            52,
            3467414389448312220,
            9756530872080251881,
            14876346105520297728,
            11197999345027776997,
            8640305102266476143,
        ],
    ),
    (
        "heavy_censoring cause 2 risk_rows group",
        &[
            57,
            7931935033626401898,
            18082612224262472977,
            12648903355739896800,
            5287808455433712389,
            9380657479977293584,
        ],
    ),
    (
        "heavy_censoring cause 2 fit breslow none",
        &[
            4605418086435410663,
            4603438123047070390,
            4620314590584756295,
            45,
            4,
            1,
            18446744073709551615,
        ],
    ),
    (
        "heavy_censoring cause 2 fit breslow subject",
        &[
            4605418086435410663,
            4603438123047070390,
            4620314590584756295,
            45,
            4,
            1,
            4605850095547607248,
        ],
    ),
    (
        "heavy_censoring cause 2 fit breslow group",
        &[
            4605418086435410663,
            4603438123047070390,
            4620314590584756295,
            45,
            4,
            1,
            4606717227919757562,
        ],
    ),
    (
        "heavy_censoring cause 2 fit efron none",
        &[
            4605418086435410663,
            4603438123047070390,
            4620314590584756295,
            45,
            4,
            1,
            18446744073709551615,
        ],
    ),
    (
        "heavy_censoring cause 2 fit efron subject",
        &[
            4605418086435410663,
            4603438123047070390,
            4620314590584756295,
            45,
            4,
            1,
            4605850095547607248,
        ],
    ),
    (
        "heavy_censoring cause 2 fit efron group",
        &[
            4605418086435410663,
            4603438123047070390,
            4620314590584756295,
            45,
            4,
            1,
            4606717227919757562,
        ],
    ),
    (
        "heavy_censoring cause 3 expand",
        &[
            72,
            14379765215969917024,
            6569544370321836977,
            3487498716132553733,
            13389207865668415525,
            6512187990966246998,
        ],
    ),
    (
        "heavy_censoring cause 3 risk_rows x",
        &[
            62,
            15944546965509545552,
            15030594284914396401,
            4259997854432282437,
            11503993979791759845,
            1133668410040019729,
        ],
    ),
    (
        "heavy_censoring cause 3 risk_rows group",
        &[
            62,
            15182886902517434089,
            8215556362534941417,
            1640846178785861997,
            11503993979791759845,
            12350663649496915051,
        ],
    ),
    (
        "heavy_censoring cause 3 fit breslow none",
        &[
            18446744073709551615,
            18446744073709551615,
            9223372036854775808,
            45,
            0,
            0,
            18446744073709551615,
        ],
    ),
    (
        "heavy_censoring cause 3 fit breslow subject",
        &[
            18446744073709551615,
            18446744073709551615,
            9223372036854775808,
            45,
            0,
            0,
            18446744073709551615,
        ],
    ),
    (
        "heavy_censoring cause 3 fit breslow group",
        &[
            18446744073709551615,
            18446744073709551615,
            9223372036854775808,
            45,
            0,
            0,
            18446744073709551615,
        ],
    ),
    (
        "heavy_censoring cause 3 fit efron none",
        &[
            18446744073709551615,
            18446744073709551615,
            9223372036854775808,
            45,
            0,
            0,
            18446744073709551615,
        ],
    ),
    (
        "heavy_censoring cause 3 fit efron subject",
        &[
            18446744073709551615,
            18446744073709551615,
            9223372036854775808,
            45,
            0,
            0,
            18446744073709551615,
        ],
    ),
    (
        "heavy_censoring cause 3 fit efron group",
        &[
            18446744073709551615,
            18446744073709551615,
            9223372036854775808,
            45,
            0,
            0,
            18446744073709551615,
        ],
    ),
    (
        "distinct cause 1 expand",
        &[
            113,
            9848993313983978153,
            13201350292139515720,
            13531563610563995390,
            5092848920263408261,
            4780783243476632427,
        ],
    ),
    (
        "distinct cause 1 risk_rows x",
        &[
            113,
            9825025072764472521,
            15953904967177565416,
            9129019445714759086,
            15915899491211386437,
            1767955011576369213,
        ],
    ),
    (
        "distinct cause 1 risk_rows group",
        &[
            90,
            11275358237932759620,
            2720268670945120719,
            2046846734047827389,
            17225372996954515941,
            16064218759847971562,
        ],
    ),
    (
        "distinct cause 1 fit breslow none",
        &[
            4586914648138128641,
            4599742049366126657,
            4624382940489705421,
            40,
            3,
            1,
            18446744073709551615,
        ],
    ),
    (
        "distinct cause 1 fit breslow subject",
        &[
            4586914648138128641,
            4599742049366126657,
            4624382940489705421,
            40,
            3,
            1,
            4601010917336064657,
        ],
    ),
    (
        "distinct cause 1 fit breslow group",
        &[
            4586914648138128641,
            4599742049366126657,
            4624382940489705421,
            40,
            3,
            1,
            4602391192044122096,
        ],
    ),
    (
        "distinct cause 1 fit efron none",
        &[
            4586914648138128641,
            4599742049366126657,
            4624382940489705421,
            40,
            3,
            1,
            18446744073709551615,
        ],
    ),
    (
        "distinct cause 1 fit efron subject",
        &[
            4586914648138128641,
            4599742049366126657,
            4624382940489705421,
            40,
            3,
            1,
            4601010917336064657,
        ],
    ),
    (
        "distinct cause 1 fit efron group",
        &[
            4586914648138128641,
            4599742049366126657,
            4624382940489705421,
            40,
            3,
            1,
            4602391192044122096,
        ],
    ),
    (
        "distinct cause 2 expand",
        &[
            101,
            11303877155719133150,
            2518427886540491211,
            17750305751200180508,
            12673968376765119812,
            12626620839680648157,
        ],
    ),
    (
        "distinct cause 2 risk_rows x",
        &[
            101,
            14178887622362977790,
            4927776519528465227,
            5060003553587126032,
            5771044568723469924,
            5556165654117616144,
        ],
    ),
    (
        "distinct cause 2 risk_rows group",
        &[
            88,
            17198647898139909168,
            2603941907062765748,
            6136502519524458117,
            7935923766654385124,
            12849654674035288253,
        ],
    ),
    (
        "distinct cause 2 fit breslow none",
        &[
            4598594584325031953,
            4600016638513050267,
            4623687774585965684,
            40,
            3,
            1,
            18446744073709551615,
        ],
    ),
    (
        "distinct cause 2 fit breslow subject",
        &[
            4598594584325031953,
            4600016638513050267,
            4623687774585965684,
            40,
            3,
            1,
            4600927448838030247,
        ],
    ),
    (
        "distinct cause 2 fit breslow group",
        &[
            4598594584325031953,
            4600016638513050267,
            4623687774585965684,
            40,
            3,
            1,
            4599513025026038816,
        ],
    ),
    (
        "distinct cause 2 fit efron none",
        &[
            4598594584325031953,
            4600016638513050267,
            4623687774585965684,
            40,
            3,
            1,
            18446744073709551615,
        ],
    ),
    (
        "distinct cause 2 fit efron subject",
        &[
            4598594584325031953,
            4600016638513050267,
            4623687774585965684,
            40,
            3,
            1,
            4600927448838030247,
        ],
    ),
    (
        "distinct cause 2 fit efron group",
        &[
            4598594584325031953,
            4600016638513050267,
            4623687774585965684,
            40,
            3,
            1,
            4599513025026038816,
        ],
    ),
    (
        "distinct cause 3 expand",
        &[
            175,
            17711925491246047569,
            4120611327746446630,
            16589828299051586098,
            18299157447275024005,
            6106370954659063065,
        ],
    ),
    (
        "distinct cause 3 risk_rows x",
        &[
            175,
            1284169473204680657,
            12999844625514162598,
            16589828299051586098,
            18299157447275024005,
            15935087386676488810,
        ],
    ),
    (
        "distinct cause 3 risk_rows group",
        &[
            124,
            6568390119007049438,
            14915357524732163667,
            4623691329164036532,
            11984747362187796645,
            5718752252420044038,
        ],
    ),
    (
        "distinct cause 3 fit breslow none",
        &[
            18446744073709551615,
            18446744073709551615,
            9223372036854775808,
            40,
            0,
            0,
            18446744073709551615,
        ],
    ),
    (
        "distinct cause 3 fit breslow subject",
        &[
            18446744073709551615,
            18446744073709551615,
            9223372036854775808,
            40,
            0,
            0,
            18446744073709551615,
        ],
    ),
    (
        "distinct cause 3 fit breslow group",
        &[
            18446744073709551615,
            18446744073709551615,
            9223372036854775808,
            40,
            0,
            0,
            18446744073709551615,
        ],
    ),
    (
        "distinct cause 3 fit efron none",
        &[
            18446744073709551615,
            18446744073709551615,
            9223372036854775808,
            40,
            0,
            0,
            18446744073709551615,
        ],
    ),
    (
        "distinct cause 3 fit efron subject",
        &[
            18446744073709551615,
            18446744073709551615,
            9223372036854775808,
            40,
            0,
            0,
            18446744073709551615,
        ],
    ),
    (
        "distinct cause 3 fit efron group",
        &[
            18446744073709551615,
            18446744073709551615,
            9223372036854775808,
            40,
            0,
            0,
            18446744073709551615,
        ],
    ),
];
