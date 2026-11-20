//! The Fine-Gray rows and fit, pinned to the bits they gave when the module
//! took slices. Downstream studies depend on these numbers, so this file must
//! pass unchanged through any change to the module.
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

#![expect(
    deprecated,
    reason = "the test pins the deprecated slice-based functions, which stay as wrappers"
)]
#![expect(
    clippy::unreadable_literal,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    reason = "the table holds recorded bit patterns, and the counts are a few hundred"
)]

use epistat_rng::Stream;
use epistat_survival::cox::{CoxFit, Ties};
use epistat_survival::fine_gray::{FineGrayData, Rows, fine_gray_fit, float_codes};
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

fn rows_record(rows: &Rows) -> Vec<u64> {
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
        rows.event
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

fn fit_record(fit: Result<CoxFit, epistat_survival::Error>) -> Vec<u64> {
    match fit {
        Ok(fit) => vec![
            opt_bits(fit.log_hazard_ratio),
            opt_bits(fit.standard_error),
            fit.n_events.to_bits(),
            fit.n_rows as u64,
            u64::from(fit.iterations),
            u64::from(fit.converged),
            opt_bits(fit.robust_standard_error),
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

/// One line per (sample, cause, output), each a vector of recorded bits.
fn record() -> Vec<(String, Vec<u64>)> {
    let mut lines = Vec::new();
    for s in samples() {
        let n = s.time.len();
        let own: Vec<u32> = (0..n as u32).collect();
        let by_x = float_codes(&s.x);
        for cause in 1..=3 {
            let tag = format!("{} cause {cause}", s.name);
            let data = FineGrayData::new(&s.time, &s.event, &s.weight, cause).unwrap();
            lines.push((format!("{tag} expand"), rows_record(&data.expand())));
            lines.push((
                format!("{tag} risk_rows x"),
                rows_record(&data.risk_rows(&by_x).unwrap()),
            ));
            lines.push((
                format!("{tag} risk_rows group"),
                rows_record(&data.risk_rows(&s.group).unwrap()),
            ));
            for (ties_name, ties) in [("breslow", Ties::Breslow), ("efron", Ties::Efron)] {
                for (cluster_name, cluster) in [
                    ("none", None),
                    ("subject", Some(own.as_slice())),
                    ("group", Some(s.group.as_slice())),
                ] {
                    let fit =
                        fine_gray_fit(&s.time, &s.event, &s.weight, &s.x, cause, ties, cluster);
                    lines.push((
                        format!("{tag} fit {ties_name} {cluster_name}"),
                        fit_record(fit),
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
    let zero = fine_gray_fit(&s.time, &s.event, &s.weight, &s.x, 0, Ties::Breslow, None);
    assert_eq!(
        zero.unwrap_err().to_string(),
        "cause 0 is censoring, not a cause"
    );
    let short = fine_gray_fit(
        &s.time,
        &s.event,
        &s.weight,
        &s.x[1..],
        1,
        Ties::Breslow,
        None,
    );
    assert_eq!(
        short.unwrap_err().to_string(),
        "columns differ in length: x has 59 values against 60 in time"
    );
    let data = FineGrayData::new(&s.time, &s.event, &s.weight, 1).unwrap();
    assert_eq!(
        data.risk_rows(&s.group[1..]).unwrap_err().to_string(),
        "columns differ in length: level has 59 values against 60 in time"
    );
    let cluster = fine_gray_fit(
        &s.time,
        &s.event,
        &s.weight,
        &s.x,
        1,
        Ties::Breslow,
        Some(&s.group[1..]),
    );
    assert_eq!(
        cluster.unwrap_err().to_string(),
        "columns differ in length: cluster has 59 values against 60 in time"
    );
}

#[test]
fn rows_and_fits_keep_their_bits() {
    let lines = record();
    if std::env::var_os("GOLDEN_RECORD").is_some() {
        for (label, values) in &lines {
            println!("    (\"{label}\", &{values:?}),");
        }
        return;
    }
    assert_eq!(lines.len(), EXPECTED.len());
    for ((label, values), (want_label, want)) in lines.iter().zip(EXPECTED) {
        assert_eq!(label, want_label);
        assert_eq!(values.as_slice(), *want, "{label}");
    }
}

/// Recorded from the slice-based functions.
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
            13824713134326763613,
            4599953140261006893,
            4625478292286210048,
            100,
            4,
            1,
            18446744073709551615,
        ],
    ),
    (
        "ties cause 1 fit breslow subject",
        &[
            13824713134326763582,
            4599953140261006895,
            4625478292286210048,
            211,
            4,
            1,
            4599752772874865686,
        ],
    ),
    (
        "ties cause 1 fit breslow group",
        &[
            13824713134326763582,
            4599953140261006895,
            4625478292286210048,
            211,
            4,
            1,
            4599553630932595964,
        ],
    ),
    (
        "ties cause 1 fit efron none",
        &[
            13824916865210317396,
            4599965971821566202,
            4625478292286210048,
            100,
            4,
            1,
            18446744073709551615,
        ],
    ),
    (
        "ties cause 1 fit efron subject",
        &[
            13824916865210317385,
            4599965971821566199,
            4625478292286210048,
            211,
            4,
            1,
            4599896268221748894,
        ],
    ),
    (
        "ties cause 1 fit efron group",
        &[
            13824916865210317385,
            4599965971821566199,
            4625478292286210048,
            211,
            4,
            1,
            4599715551434294329,
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
            13828355714701416423,
            4600001738175051036,
            4625478292286210048,
            112,
            4,
            1,
            18446744073709551615,
        ],
    ),
    (
        "ties cause 2 fit breslow subject",
        &[
            13828355714701416425,
            4600001738175051036,
            4625478292286210048,
            251,
            4,
            1,
            4599233148931057113,
        ],
    ),
    (
        "ties cause 2 fit breslow group",
        &[
            13828355714701416425,
            4600001738175051036,
            4625478292286210048,
            251,
            4,
            1,
            4600946326514266536,
        ],
    ),
    (
        "ties cause 2 fit efron none",
        &[
            13828435015447412245,
            4599981950940338487,
            4625478292286210048,
            112,
            4,
            1,
            18446744073709551615,
        ],
    ),
    (
        "ties cause 2 fit efron subject",
        &[
            13828435015447412244,
            4599981950940338481,
            4625478292286210048,
            251,
            4,
            1,
            4599299621709840315,
        ],
    ),
    (
        "ties cause 2 fit efron group",
        &[
            13828435015447412244,
            4599981950940338481,
            4625478292286210048,
            251,
            4,
            1,
            4600973893517090433,
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
            4609296462456649630,
            4603077916702969678,
            4622945017495814144,
            112,
            6,
            1,
            18446744073709551615,
        ],
    ),
    (
        "ties cause 3 fit breslow subject",
        &[
            4609296462456649621,
            4603077916702969672,
            4622945017495814144,
            292,
            6,
            1,
            4601795004605161543,
        ],
    ),
    (
        "ties cause 3 fit breslow group",
        &[
            4609296462456649621,
            4603077916702969672,
            4622945017495814144,
            292,
            6,
            1,
            4603631229571291250,
        ],
    ),
    (
        "ties cause 3 fit efron none",
        &[
            4609347257965540928,
            4603066089091736557,
            4622945017495814144,
            112,
            6,
            1,
            18446744073709551615,
        ],
    ),
    (
        "ties cause 3 fit efron subject",
        &[
            4609347257965540917,
            4603066089091736540,
            4622945017495814144,
            292,
            6,
            1,
            4601845595592299141,
        ],
    ),
    (
        "ties cause 3 fit efron group",
        &[
            4609347257965540917,
            4603066089091736540,
            4622945017495814144,
            292,
            6,
            1,
            4603663053432362930,
        ],
    ),
    (
        "weights cause 1 expand",
        &[
            259,
            16435329277623428694,
            3257430820204066801,
            5602861988250258702,
            14058164180547989221,
            15898716504008097382,
        ],
    ),
    (
        "weights cause 1 risk_rows x",
        &[
            143,
            9090680091008043851,
            7721562576717539602,
            1564428532097249590,
            18209030831507499973,
            14095908567083613117,
        ],
    ),
    (
        "weights cause 1 risk_rows group",
        &[
            166,
            16311993539693667505,
            3626990188469476935,
            8614782900049262052,
            3883461466012992549,
            10487989894720341406,
        ],
    ),
    (
        "weights cause 1 fit breslow none",
        &[
            13814062323896115220,
            4598444358967793672,
            4627161928506626658,
            138,
            4,
            1,
            18446744073709551615,
        ],
    ),
    (
        "weights cause 1 fit breslow subject",
        &[
            13814062323896115229,
            4598444358967793673,
            4627161928506626658,
            254,
            4,
            1,
            4599192691152504283,
        ],
    ),
    (
        "weights cause 1 fit breslow group",
        &[
            13814062323896115229,
            4598444358967793673,
            4627161928506626658,
            254,
            4,
            1,
            4599248173466055050,
        ],
    ),
    (
        "weights cause 1 fit efron none",
        &[
            13814013981413712601,
            4598450349758601824,
            4627161928506626658,
            138,
            4,
            1,
            18446744073709551615,
        ],
    ),
    (
        "weights cause 1 fit efron subject",
        &[
            13814013981413712649,
            4598450349758601824,
            4627161928506626658,
            254,
            4,
            1,
            4599256032667280028,
        ],
    ),
    (
        "weights cause 1 fit efron group",
        &[
            13814013981413712649,
            4598450349758601824,
            4627161928506626658,
            254,
            4,
            1,
            4599308071827634310,
        ],
    ),
    (
        "weights cause 2 expand",
        &[
            223,
            13505314724725783808,
            16726315640030929738,
            12360119078105271659,
            5038335392323586533,
            2639263828216449190,
        ],
    ),
    (
        "weights cause 2 risk_rows x",
        &[
            128,
            6328537296402822671,
            13496547257531291897,
            9803831757318043223,
            17391681924655314309,
            15343977375622551493,
        ],
    ),
    (
        "weights cause 2 risk_rows group",
        &[
            164,
            10888078724434495046,
            5684829680779166482,
            16382858636899870238,
            1737706455199419397,
            11101304330692271928,
        ],
    ),
    (
        "weights cause 2 fit breslow none",
        &[
            13802720377611465526,
            4596992799965371577,
            4630119498190085660,
            122,
            3,
            1,
            18446744073709551615,
        ],
    ),
    (
        "weights cause 2 fit breslow subject",
        &[
            13802720377611465322,
            4596992799965371573,
            4630119498190085660,
            217,
            3,
            1,
            4598127984695100587,
        ],
    ),
    (
        "weights cause 2 fit breslow group",
        &[
            13802720377611465322,
            4596992799965371573,
            4630119498190085660,
            217,
            3,
            1,
            4596765342416443908,
        ],
    ),
    (
        "weights cause 2 fit efron none",
        &[
            13800666114040080860,
            4597002799028538713,
            4630119498190085660,
            122,
            3,
            1,
            18446744073709551615,
        ],
    ),
    (
        "weights cause 2 fit efron subject",
        &[
            13800666114040080909,
            4597002799028538714,
            4630119498190085660,
            217,
            3,
            1,
            4598224596143953826,
        ],
    ),
    (
        "weights cause 2 fit efron group",
        &[
            13800666114040080909,
            4597002799028538714,
            4630119498190085660,
            217,
            3,
            1,
            4596913015250238871,
        ],
    ),
    (
        "weights cause 3 expand",
        &[
            402,
            1265205352389449747,
            12364429291375277286,
            17581473469500034602,
            646818790841753189,
            980335103324799181,
        ],
    ),
    (
        "weights cause 3 risk_rows x",
        &[
            132,
            7614779903385641545,
            12893239975579535771,
            16552600259842024524,
            8164121070118978981,
            8179885808248214620,
        ],
    ),
    (
        "weights cause 3 risk_rows group",
        &[
            196,
            5257326790617183021,
            7907633178948711327,
            10156789511055491452,
            8464970716091018661,
            5910201450423008723,
        ],
    ),
    (
        "weights cause 3 fit breslow none",
        &[
            18446744073709551615,
            18446744073709551615,
            0,
            129,
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
            0,
            399,
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
            0,
            399,
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
            0,
            129,
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
            0,
            399,
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
            0,
            399,
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
            13819148473670378332,
            4598496587858753187,
            4628411439846819829,
            38,
            4,
            1,
            18446744073709551615,
        ],
    ),
    (
        "sparse_censoring cause 1 fit breslow subject",
        &[
            13819148473670378332,
            4598496587858753187,
            4628411439846819829,
            56,
            4,
            1,
            4598891886707012209,
        ],
    ),
    (
        "sparse_censoring cause 1 fit breslow group",
        &[
            13819148473670378332,
            4598496587858753187,
            4628411439846819829,
            56,
            4,
            1,
            4598793672878178925,
        ],
    ),
    (
        "sparse_censoring cause 1 fit efron none",
        &[
            13819205274698375975,
            4598486499876981969,
            4628411439846819829,
            38,
            4,
            1,
            18446744073709551615,
        ],
    ),
    (
        "sparse_censoring cause 1 fit efron subject",
        &[
            13819205274698375966,
            4598486499876981968,
            4628411439846819829,
            56,
            4,
            1,
            4598944526485114153,
        ],
    ),
    (
        "sparse_censoring cause 1 fit efron group",
        &[
            13819205274698375966,
            4598486499876981968,
            4628411439846819829,
            56,
            4,
            1,
            4598906049038475798,
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
            4594521213498257194,
            4598771393464469351,
            4629626349262449804,
            40,
            4,
            1,
            18446744073709551615,
        ],
    ),
    (
        "sparse_censoring cause 2 fit breslow subject",
        &[
            4594521213498257194,
            4598771393464469352,
            4629626349262449804,
            54,
            4,
            1,
            4600568104377409106,
        ],
    ),
    (
        "sparse_censoring cause 2 fit breslow group",
        &[
            4594521213498257194,
            4598771393464469352,
            4629626349262449804,
            54,
            4,
            1,
            4601343562509974794,
        ],
    ),
    (
        "sparse_censoring cause 2 fit efron none",
        &[
            4594502148698123207,
            4598785196371445664,
            4629626349262449804,
            40,
            4,
            1,
            18446744073709551615,
        ],
    ),
    (
        "sparse_censoring cause 2 fit efron subject",
        &[
            4594502148698123224,
            4598785196371445663,
            4629626349262449804,
            54,
            4,
            1,
            4600729508483937291,
        ],
    ),
    (
        "sparse_censoring cause 2 fit efron group",
        &[
            4594502148698123224,
            4598785196371445663,
            4629626349262449804,
            54,
            4,
            1,
            4601497804308318655,
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
            0,
            19,
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
            0,
            60,
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
            0,
            60,
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
            0,
            19,
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
            0,
            60,
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
            0,
            60,
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
            13828946437294407517,
            4600679539854236073,
            4623507967449235456,
            17,
            5,
            1,
            18446744073709551615,
        ],
    ),
    (
        "no_censoring cause 1 fit breslow subject",
        &[
            13828946437294407518,
            4600679539854236073,
            4623507967449235456,
            30,
            5,
            1,
            4599801053142125011,
        ],
    ),
    (
        "no_censoring cause 1 fit breslow group",
        &[
            13828946437294407518,
            4600679539854236073,
            4623507967449235456,
            30,
            5,
            1,
            4599459102598895551,
        ],
    ),
    (
        "no_censoring cause 1 fit efron none",
        &[
            13828943498554049480,
            4600640262099161261,
            4623507967449235456,
            17,
            5,
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
            5,
            1,
            4599823303493869119,
        ],
    ),
    (
        "no_censoring cause 1 fit efron group",
        &[
            13828943498554049480,
            4600640262099161262,
            4623507967449235456,
            30,
            5,
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
            4607030846361182850,
            4601735947462883739,
            4625478292286210048,
            21,
            5,
            1,
            18446744073709551615,
        ],
    ),
    (
        "no_censoring cause 2 fit breslow subject",
        &[
            4607030846361182850,
            4601735947462883739,
            4625478292286210048,
            30,
            5,
            1,
            4602746818826764522,
        ],
    ),
    (
        "no_censoring cause 2 fit breslow group",
        &[
            4607030846361182850,
            4601735947462883739,
            4625478292286210048,
            30,
            5,
            1,
            4602948632445039387,
        ],
    ),
    (
        "no_censoring cause 2 fit efron none",
        &[
            4607373200896030779,
            4601882895242971600,
            4625478292286210048,
            21,
            5,
            1,
            18446744073709551615,
        ],
    ),
    (
        "no_censoring cause 2 fit efron subject",
        &[
            4607373200896030779,
            4601882895242971600,
            4625478292286210048,
            30,
            5,
            1,
            4603038232047293727,
        ],
    ),
    (
        "no_censoring cause 2 fit efron group",
        &[
            4607373200896030779,
            4601882895242971600,
            4625478292286210048,
            30,
            5,
            1,
            4603357674063384306,
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
            0,
            5,
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
            0,
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
            0,
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
            0,
            5,
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
            0,
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
            0,
            30,
            0,
            0,
            18446744073709551615,
        ],
    ),
    (
        "heavy_censoring cause 1 expand",
        &[
            61,
            10582998941015571521,
            17711088883115803656,
            452193472568091908,
            3704106255675281125,
            12687633463818858454,
        ],
    ),
    (
        "heavy_censoring cause 1 risk_rows x",
        &[
            61,
            14870658743963589382,
            7854656042537321928,
            3488746387808351684,
            7950793257382507909,
            6935791449802860456,
        ],
    ),
    (
        "heavy_censoring cause 1 risk_rows group",
        &[
            61,
            14870658743963589382,
            7854656042537321928,
            3488746387808351684,
            7950793257382507909,
            6935791449802860456,
        ],
    ),
    (
        "heavy_censoring cause 1 fit breslow none",
        &[
            13817880415521270474,
            4603993107718282417,
            4616993744422866754,
            55,
            4,
            1,
            18446744073709551615,
        ],
    ),
    (
        "heavy_censoring cause 1 fit breslow subject",
        &[
            13817880415521270483,
            4603993107718282419,
            4616993744422866754,
            55,
            4,
            1,
            4600992152773540658,
        ],
    ),
    (
        "heavy_censoring cause 1 fit breslow group",
        &[
            13817880415521270483,
            4603993107718282419,
            4616993744422866754,
            55,
            4,
            1,
            4598848281358629435,
        ],
    ),
    (
        "heavy_censoring cause 1 fit efron none",
        &[
            13817880415521270474,
            4603993107718282417,
            4616993744422866754,
            55,
            4,
            1,
            18446744073709551615,
        ],
    ),
    (
        "heavy_censoring cause 1 fit efron subject",
        &[
            13817880415521270483,
            4603993107718282419,
            4616993744422866754,
            55,
            4,
            1,
            4600992152773540658,
        ],
    ),
    (
        "heavy_censoring cause 1 fit efron group",
        &[
            13817880415521270483,
            4603993107718282419,
            4616993744422866754,
            55,
            4,
            1,
            4598848281358629435,
        ],
    ),
    (
        "heavy_censoring cause 2 expand",
        &[
            64,
            13159630784484638540,
            10864007386970512268,
            7302102611607464252,
            8762978574364073541,
            851144651090118100,
        ],
    ),
    (
        "heavy_censoring cause 2 risk_rows x",
        &[
            59,
            8243707393024539706,
            17979487452277158276,
            5405622349998333404,
            13701004502565105669,
            2764028189264948899,
        ],
    ),
    (
        "heavy_censoring cause 2 risk_rows group",
        &[
            64,
            183557547210699468,
            9218523919339621900,
            14979915821932992380,
            6913722502593557797,
            3850653751147594372,
        ],
    ),
    (
        "heavy_censoring cause 2 fit breslow none",
        &[
            4605418086435411266,
            4603438123047070390,
            4620314590584756295,
            53,
            5,
            1,
            18446744073709551615,
        ],
    ),
    (
        "heavy_censoring cause 2 fit breslow subject",
        &[
            4605418086435411273,
            4603438123047070384,
            4620314590584756295,
            58,
            5,
            1,
            4605850095547607553,
        ],
    ),
    (
        "heavy_censoring cause 2 fit breslow group",
        &[
            4605418086435411273,
            4603438123047070384,
            4620314590584756295,
            58,
            5,
            1,
            4606717227919757724,
        ],
    ),
    (
        "heavy_censoring cause 2 fit efron none",
        &[
            4605418086435411266,
            4603438123047070390,
            4620314590584756295,
            53,
            5,
            1,
            18446744073709551615,
        ],
    ),
    (
        "heavy_censoring cause 2 fit efron subject",
        &[
            4605418086435411273,
            4603438123047070384,
            4620314590584756295,
            58,
            5,
            1,
            4605850095547607553,
        ],
    ),
    (
        "heavy_censoring cause 2 fit efron group",
        &[
            4605418086435411273,
            4603438123047070384,
            4620314590584756295,
            58,
            5,
            1,
            4606717227919757724,
        ],
    ),
    (
        "heavy_censoring cause 3 expand",
        &[
            80,
            15358658992329352004,
            12091691843099601569,
            15641888456111033892,
            10832649015740339493,
            4097782033972291354,
        ],
    ),
    (
        "heavy_censoring cause 3 risk_rows x",
        &[
            70,
            13906627805281595412,
            17830229924338671105,
            11450001418250240916,
            6425401157880512229,
            14417627905713592714,
        ],
    ),
    (
        "heavy_censoring cause 3 risk_rows group",
        &[
            70,
            9354139198578731661,
            11750392686037670857,
            5746520141979811676,
            6425401157880512229,
            4888266544836053768,
        ],
    ),
    (
        "heavy_censoring cause 3 fit breslow none",
        &[
            18446744073709551615,
            18446744073709551615,
            0,
            64,
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
            0,
            74,
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
            0,
            74,
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
            0,
            64,
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
            0,
            74,
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
            0,
            74,
            0,
            0,
            18446744073709551615,
        ],
    ),
    (
        "distinct cause 1 expand",
        &[
            115,
            9043946253695687882,
            451829772871641933,
            8798490182951452026,
            7030269069443749285,
            14634681886905131798,
        ],
    ),
    (
        "distinct cause 1 risk_rows x",
        &[
            115,
            14443227874536182762,
            17601015228468579597,
            11476910361869556570,
            4959466874307822949,
            6172038154687064452,
        ],
    ),
    (
        "distinct cause 1 risk_rows group",
        &[
            92,
            5938869006099168071,
            12512124574589589862,
            16189151894316551049,
            7211751599604290821,
            11463684459318006987,
        ],
    ),
    (
        "distinct cause 1 fit breslow none",
        &[
            4586914648138128609,
            4599742049366126659,
            4624382940489705422,
            114,
            4,
            1,
            18446744073709551615,
        ],
    ),
    (
        "distinct cause 1 fit breslow subject",
        &[
            4586914648138128633,
            4599742049366126658,
            4624382940489705422,
            114,
            4,
            1,
            4601010917336064659,
        ],
    ),
    (
        "distinct cause 1 fit breslow group",
        &[
            4586914648138128633,
            4599742049366126658,
            4624382940489705422,
            114,
            4,
            1,
            4602391192044122099,
        ],
    ),
    (
        "distinct cause 1 fit efron none",
        &[
            4586914648138128609,
            4599742049366126659,
            4624382940489705422,
            114,
            4,
            1,
            18446744073709551615,
        ],
    ),
    (
        "distinct cause 1 fit efron subject",
        &[
            4586914648138128633,
            4599742049366126658,
            4624382940489705422,
            114,
            4,
            1,
            4601010917336064659,
        ],
    ),
    (
        "distinct cause 1 fit efron group",
        &[
            4586914648138128633,
            4599742049366126658,
            4624382940489705422,
            114,
            4,
            1,
            4602391192044122099,
        ],
    ),
    (
        "distinct cause 2 expand",
        &[
            103,
            2199200555811044764,
            526114810990463178,
            15036149574746992368,
            14875560421956666948,
            4498106645177716650,
        ],
    ),
    (
        "distinct cause 2 risk_rows x",
        &[
            103,
            12023602301158720252,
            17025150541771463786,
            15138999517881429784,
            12219202932886050660,
            12528820598576896803,
        ],
    ),
    (
        "distinct cause 2 risk_rows group",
        &[
            90,
            17377077898665404786,
            2700922022696897077,
            12163526943070927409,
            16243572017742216420,
            827211139251342194,
        ],
    ),
    (
        "distinct cause 2 fit breslow none",
        &[
            4598594584325031945,
            4600016638513050266,
            4623687774585965685,
            102,
            4,
            1,
            18446744073709551615,
        ],
    ),
    (
        "distinct cause 2 fit breslow subject",
        &[
            4598594584325031953,
            4600016638513050265,
            4623687774585965685,
            102,
            4,
            1,
            4600927448838030255,
        ],
    ),
    (
        "distinct cause 2 fit breslow group",
        &[
            4598594584325031953,
            4600016638513050265,
            4623687774585965685,
            102,
            4,
            1,
            4599513025026038822,
        ],
    ),
    (
        "distinct cause 2 fit efron none",
        &[
            4598594584325031945,
            4600016638513050266,
            4623687774585965685,
            102,
            4,
            1,
            18446744073709551615,
        ],
    ),
    (
        "distinct cause 2 fit efron subject",
        &[
            4598594584325031953,
            4600016638513050265,
            4623687774585965685,
            102,
            4,
            1,
            4600927448838030255,
        ],
    ),
    (
        "distinct cause 2 fit efron group",
        &[
            4598594584325031953,
            4600016638513050265,
            4623687774585965685,
            102,
            4,
            1,
            4599513025026038822,
        ],
    ),
    (
        "distinct cause 3 expand",
        &[
            178,
            11182777530451296627,
            14267287462164708030,
            1658707900212490603,
            9946394348355424869,
            8090036199788255023,
        ],
    ),
    (
        "distinct cause 3 risk_rows x",
        &[
            178,
            12999668620653456723,
            2177835161425484350,
            1658707900212490603,
            9946394348355424869,
            1541851229954623928,
        ],
    ),
    (
        "distinct cause 3 risk_rows group",
        &[
            127,
            12901443112864622652,
            8296185721316498555,
            1013546470059978973,
            2929230124838088837,
            10728941311795180652,
        ],
    ),
    (
        "distinct cause 3 fit breslow none",
        &[
            18446744073709551615,
            18446744073709551615,
            0,
            177,
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
            0,
            177,
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
            0,
            177,
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
            0,
            177,
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
            0,
            177,
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
            0,
            177,
            0,
            0,
            18446744073709551615,
        ],
    ),
];
