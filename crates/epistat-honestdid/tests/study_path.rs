//! The HonestDiD calls of Study I, hashed to the bit.
//!
//! The event study is the one of epistat-did's study path: Study I's doubly
//! robust ATT(g,t) on a simulated panel of diagnosed children's parents and
//! their matched comparator families, with family clusters, aggregated by
//! event time as `did::aggte` does, without the reference period -1 and
//! within event times -5 to 8, its covariance from the family cluster sums of
//! the aggregated influence functions as PHD's `kernels::did` forms it. On it
//! the test makes the calls of `kernels::did::honest`: the relative-magnitude
//! surface for every post period, the average and the average of the first
//! three; the joint path at three values of Mbar with the Gaussian
//! calibration; and the smoothness bound from the pre-trends with the
//! smoothness surface below it; on the event study scaled as the kernels
//! scale it, with Study I's grids and 20,000 draws.
//!
//! A change that moves a hash moves Study I, except as decided (decision 112
//! moves the smoothness bounds at the switch), so a port of these calls to
//! new names must reproduce every hash.
//!
//! The hashes were recorded on aarch64 macOS and are asserted only there.
//! Elsewhere the test computes them and, when `EPISTAT_HASH_OUT` is set,
//! appends them to that file. Print them with
//! `STUDY_PATH_PRINT=1 cargo test -p epistat-honestdid --test study_path -- --nocapture`.

use epistat_core::{Bootstrap, Covariance, EventStudy, Level};
use epistat_did::StaggeredPanel;
use epistat_did::att_gt::{
    self, Aggte, AggteOptions, AggteType, AttGt, AttGtOptions, BasePeriod, ComparisonGroup,
    NuisanceOptions, UnbalancedPanel,
};
use epistat_did::did_cluster_sums;
use epistat_honestdid::honest::{
    self, Calibration, HonestOptions, JointPathOptions, Restriction, RmBound, Target,
};
use epistat_inference::Clusters;
use epistat_linalg::MatRef;
use epistat_testkit::{Fingerprint, SplitMix64};

const CONFIDENCE: f64 = 0.95;
const FIRST_YEAR: i32 = 2000;
const LAST_YEAR: i32 = 2016;
/// Draws behind every least-favorable critical value and the joint path's
/// calibration, Study I's `analysis_honest_least_favorable_draws`.
const DRAWS: usize = 20_000;
const JOINT_SEED: u64 = 20_260_819;
const RM_GRID: [f64; 9] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5];
const JOINT_GRID: [f64; 3] = [0.0, 0.25, 0.5];

/// A fingerprint that hashes NaN and `None` alike, so a value that becomes
/// `Option` without changing leaves the hash alone.
struct Hash(Fingerprint);

impl Hash {
    fn new() -> Self {
        Self(Fingerprint::new())
    }

    fn word(&mut self, value: u64) {
        self.0 = std::mem::take(&mut self.0).u64(value);
    }

    fn int(&mut self, value: i64) {
        self.word(value.cast_unsigned());
    }

    fn text(&mut self, text: &str) {
        self.word(text.len() as u64);
        for byte in text.bytes() {
            self.word(u64::from(byte));
        }
    }

    fn o(&mut self, value: Option<f64>) {
        let value = value.filter(|v| !v.is_nan());
        self.0 = std::mem::take(&mut self.0).option_f64(value);
    }

    fn f(&mut self, value: f64) {
        self.o(Some(value));
    }

    fn finish(&self) -> u64 {
        self.0.finish()
    }
}

fn normal(stream: &mut SplitMix64) -> f64 {
    let u1 = stream.unit().max(f64::MIN_POSITIVE);
    let u2 = stream.unit();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// One person-year of the analysis table, in the kernels' columns.
struct Row {
    time: i32,
    /// The cohort, 0 for never treated, as `kernels::did::NEVER_TREATED`.
    group: i32,
    outcome: f64,
    weight: f64,
    unit: u32,
    cluster: u32,
    comparison_cohort: i32,
    covariates: [f64; 3],
}

fn panel() -> Vec<Row> {
    let mut stream = SplitMix64::new(20_261_126);
    let mut rows = Vec::new();
    let mut next_unit = 0_u32;
    let mut next_family = 0_u32;
    let mut family =
        |stream: &mut SplitMix64, rows: &mut Vec<Row>, cohort: i32, treated: bool, weight: f64| {
            let cluster = next_family * 7 + 3;
            next_family += 1;
            let parents = if stream.unit() < 0.8 { 2 } else { 1 };
            let rare = f64::from(stream.unit() < 0.03);
            for parent in 0..parents {
                // Unit codes that do not sort in family order.
                let unit = next_unit.wrapping_mul(40_503).wrapping_add(17) % (1 << 20);
                next_unit += 1;
                let age = 24.0 + 14.0 * stream.unit();
                let mother = f64::from(parent == 0);
                let level = 20.0 * normal(stream);
                let from = (cohort - 6 + i32::from(stream.unit() < 0.15)).max(FIRST_YEAR);
                let to = (cohort + 8 - i32::from(stream.unit() < 0.2) * 3).min(LAST_YEAR);
                for time in from..=to {
                    let event = time - cohort;
                    let effect = if treated && event >= 0 {
                        -15.0 - 2.0 * f64::from(event)
                    } else {
                        0.0
                    };
                    let outcome = 250.0 + 3.0 * age - 12.0 * mother
                        + level
                        + 4.0 * f64::from(time - FIRST_YEAR)
                        + effect
                        + 15.0 * normal(stream);
                    rows.push(Row {
                        time,
                        group: if treated { cohort } else { 0 },
                        outcome,
                        weight,
                        unit,
                        cluster,
                        comparison_cohort: if treated { 0 } else { cohort },
                        covariates: [age, mother, rare],
                    });
                }
            }
        };
    for case in 0..260 {
        let cohort = if case < 6 {
            FIRST_YEAR
        } else {
            2004 + i32::try_from(stream.below(10)).expect("small")
        };
        family(&mut stream, &mut rows, cohort, true, 1.0);
        let comparators = 2 + stream.below(3);
        for _ in 0..comparators {
            family(
                &mut stream,
                &mut rows,
                cohort,
                false,
                1.0 / comparators as f64,
            );
        }
    }
    rows
}

/// The analysis table as panel columns, sorted by unit and then period, as
/// `kernels::did` hands it to the estimator.
struct Columns {
    unit: Vec<u32>,
    period: Vec<i64>,
    cohort: Vec<Option<i64>>,
    outcome: Vec<f64>,
    weight: Vec<f64>,
    /// Column-major, three columns.
    covariates: Vec<f64>,
    matched: Vec<Option<i64>>,
    cluster: Vec<u32>,
}

fn columns(rows: &[Row]) -> Columns {
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by_key(|&i| (rows[i].unit, rows[i].time));
    let pick = |f: &dyn Fn(&Row) -> f64| order.iter().map(|&i| f(&rows[i])).collect::<Vec<_>>();
    let covariates = (0..3)
        .flat_map(|j| order.iter().map(move |&i| rows[i].covariates[j]))
        .collect();
    Columns {
        unit: order.iter().map(|&i| rows[i].unit).collect(),
        period: order.iter().map(|&i| i64::from(rows[i].time)).collect(),
        cohort: order
            .iter()
            .map(|&i| (rows[i].group != 0).then_some(i64::from(rows[i].group)))
            .collect(),
        outcome: pick(&|row| row.outcome),
        weight: pick(&|row| row.weight),
        covariates,
        matched: order
            .iter()
            .map(|&i| {
                (rows[i].comparison_cohort != 0).then_some(i64::from(rows[i].comparison_cohort))
            })
            .collect(),
        cluster: order.iter().map(|&i| rows[i].cluster).collect(),
    }
}

/// Study I's ATT(g,t): the matched or pooled never-treated comparators, a
/// universal base, the per-cell panel, no overlap refusal, the legacy
/// repeated cross-section computation, and the one-replicate cell bootstrap
/// the kernels set, seeded as the estimator's former default.
fn study_att_gt(
    data: &Columns,
    comparison_group: ComparisonGroup,
    anticipation: i64,
    clustered: bool,
) -> AttGt {
    let covariates = MatRef::from_column_major_slice(&data.covariates, data.unit.len(), 3);
    let builder = StaggeredPanel::builder(&data.unit, &data.period, &data.cohort, &data.outcome)
        .weights(&data.weight)
        .covariates(covariates)
        .comparison_cohort(&data.matched);
    let builder = if clustered {
        builder.clusters(&data.cluster)
    } else {
        builder
    };
    let panel = builder.build().expect("panel");
    let options = AttGtOptions {
        comparison_group,
        base_period: BasePeriod::Universal,
        anticipation,
        unbalanced: UnbalancedPanel::PerCell,
        refuse_poor_overlap: false,
        nuisance: NuisanceOptions {
            legacy_repeated_cross_section: true,
            ..NuisanceOptions::default()
        },
        level: Level::confidence(CONFIDENCE).expect("level"),
        bootstrap: (!clustered).then_some(Bootstrap::new(1, 17_431)),
        ..AttGtOptions::default()
    };
    att_gt::estimate(&panel, &options).expect("ATT(g,t)")
}

/// `kernels::did::honest`'s scale: the root of the largest variance.
fn honest_scale(covariance: &Covariance) -> f64 {
    let largest = (0..covariance.dim())
        .map(|index| covariance.get(index, index).unwrap_or(0.0))
        .fold(0.0_f64, f64::max);
    if largest.is_finite() && largest > 0.0 {
        largest.sqrt()
    } else {
        1.0
    }
}

fn scaled(study: &EventStudy, scale: f64) -> EventStudy {
    let k = study.coefficients().len();
    let covariance: Vec<f64> = (0..k * k)
        .map(|index| {
            let value = study
                .covariance()
                .get(index / k, index % k)
                .expect("defined");
            value / (scale * scale)
        })
        .collect();
    EventStudy::new(
        study
            .coefficients()
            .iter()
            .map(|value| value / scale)
            .collect(),
        Covariance::from_values(k, &covariance).expect("covariance"),
        study.relative_periods().to_vec(),
        study.reference(),
    )
    .expect("event study")
}

/// The event-time curve with its family-clustered covariance, as the kernels
/// form it from the aggregated influence functions, without the reference
/// period -1 and within event times -5 to 8.
fn event_study(dynamic: &Aggte, clusters: &Clusters) -> EventStudy {
    let influence = dynamic.influence();
    let rows: Vec<_> = dynamic
        .points
        .iter()
        .enumerate()
        .filter(|(_, row)| row.key != -1 && (-5..=8).contains(&row.key))
        .collect();
    let sums: Vec<Vec<f64>> = rows
        .iter()
        .map(|(j, _)| did_cluster_sums(influence.column(*j), clusters).expect("sums"))
        .collect();
    let covariance: Vec<f64> = sums
        .iter()
        .flat_map(|left| {
            sums.iter()
                .map(|right| left.iter().zip(right).map(|(a, b)| a * b).sum())
        })
        .collect();
    EventStudy::new(
        rows.iter().map(|(_, row)| row.estimate.estimate).collect(),
        Covariance::from_values(rows.len(), &covariance).expect("covariance"),
        rows.iter().map(|(_, row)| row.key).collect(),
        -1,
    )
    .expect("event study")
}

/// The event-time aggregation of the family-clustered fit and the family
/// clusters, as the kernels form them.
fn dynamic(clustered: &AttGt) -> (Aggte, Clusters) {
    let families = clustered.clusters().expect("clusters").to_vec();
    let clusters = Clusters::from_labels(&families);
    let options = AggteOptions {
        aggregation: AggteType::Dynamic,
        level: Level::confidence(CONFIDENCE).expect("level"),
        ..AggteOptions::default()
    };
    (
        clustered.aggregate(&options).expect("aggregation"),
        clusters,
    )
}

/// `kernels::did::honest`'s three entry points on the event-time curve.
fn honest(out: &mut Vec<(&'static str, u64)>, study: &EventStudy) {
    let level = Level::confidence(CONFIDENCE).expect("level");
    let options = HonestOptions {
        level,
        least_favorable: Bootstrap::new(DRAWS, 0),
        ..HonestOptions::default()
    };
    let scale = honest_scale(study.covariance());
    let input = scaled(study, scale);
    let n_post = input.n_post();
    let mut functionals: Vec<Vec<f64>> = (0..n_post)
        .map(|k| (0..n_post).map(|j| f64::from(j == k)).collect())
        .collect();
    functionals.push(vec![1.0 / n_post as f64; n_post]);
    functionals.push(
        (0..n_post)
            .map(|j| if j < 3 { 1.0 / 3.0 } else { 0.0 })
            .collect(),
    );

    let restriction = Restriction::RelativeMagnitude {
        mbar: RM_GRID.to_vec(),
        bound: RmBound::ParallelTrends,
    };
    let offset = input.n_pre();
    let mut h = Hash::new();
    for post_weights in &functionals {
        let summary = honest::summarize(
            &input,
            &Target::Weights(post_weights.clone()),
            &restriction,
            &options,
        )
        .expect("relative magnitude");
        let estimate: f64 = post_weights
            .iter()
            .enumerate()
            .map(|(index, weight)| weight * input.coefficients()[offset + index])
            .sum();
        let mut variance = 0.0;
        for (i, wi) in post_weights.iter().enumerate() {
            for (j, wj) in post_weights.iter().enumerate() {
                variance += wi
                    * wj
                    * input
                        .covariance()
                        .get(offset + i, offset + j)
                        .expect("defined");
            }
        }
        h.f(estimate * scale);
        h.f(variance.max(0.0).sqrt() * scale);
        for row in &summary.rows {
            h.f(row.value);
            h.o(row.interval.lower.map(|v| v * scale));
            h.o(row.interval.upper.map(|v| v * scale));
            h.o(row.identified.and_then(|set| set.lower).map(|v| v * scale));
            h.o(row.identified.and_then(|set| set.upper).map(|v| v * scale));
            h.text(row.delta.label());
        }
    }
    out.push(("honest/relative_magnitude", h.finish()));

    let mut h = Hash::new();
    for m in JOINT_GRID {
        let region = honest::joint_path(
            &input,
            &Restriction::RelativeMagnitude {
                mbar: vec![m],
                bound: RmBound::ParallelTrends,
            },
            &JointPathOptions {
                level,
                null: 0.0 / scale,
                calibration: Calibration::Gaussian(Bootstrap::new(DRAWS, JOINT_SEED)),
                least_favorable: Bootstrap::new(DRAWS, 0),
                ..JointPathOptions::default()
            },
        )
        .expect("joint path");
        h.f(region.pointwise_level);
        h.o(region.critical_value);
        for point in &region.points {
            h.int(point.period.expect("a period"));
            let original = point.original;
            for value in [
                Some(original.estimate),
                original.se,
                original.lower(),
                original.upper(),
                point.robust.lower,
                point.robust.upper,
                point.identified.lower,
                point.identified.upper,
            ] {
                h.o(value.map(|v| v * scale));
            }
        }
    }
    out.push(("honest/joint_path", h.finish()));

    let m = honest::smoothness_bound_from_pretrends(study, Level::alpha(0.05).expect("level"))
        .expect("smoothness m");
    let mut h = Hash::new();
    h.f(m);
    let grid = vec![0.0, 0.5 * m / scale, m / scale];
    let summary = honest::summarize(
        &input,
        &Target::Weights(functionals[n_post].clone()),
        &Restriction::Smoothness { m: grid },
        &options,
    )
    .expect("smoothness");
    for row in &summary.rows {
        h.f(row.value * scale);
        h.o(row.interval.lower.map(|v| v * scale));
        h.o(row.interval.upper.map(|v| v * scale));
        h.text(row.delta.label());
    }
    out.push(("honest/smoothness", h.finish()));
}

fn cases() -> Vec<(&'static str, u64)> {
    let rows = panel();
    let data = columns(&rows);
    let clustered = study_att_gt(&data, ComparisonGroup::MatchedNeverTreated, 0, true);
    let (dynamic, clusters) = dynamic(&clustered);
    let mut out = Vec::new();
    honest(&mut out, &event_study(&dynamic, &clusters));
    out
}

const GOLDEN: &[(&str, u64)] = &[
    ("honest/relative_magnitude", 0x9e84_13c7_b9f7_560b),
    ("honest/joint_path", 0x40d8_ce11_2e5b_21b2),
    ("honest/smoothness", 0x6b35_a17a_1a79_e667),
];

#[test]
fn study_paths_are_bit_identical_to_the_recorded_hashes() {
    let current = cases();
    for (name, hash) in &current {
        epistat_testkit::hash_out(name, *hash);
    }
    if std::env::var_os("STUDY_PATH_PRINT").is_some() {
        for (name, hash) in &current {
            println!(
                "    (\"{name}\", 0x{:04x}_{:04x}_{:04x}_{:04x}),",
                hash >> 48,
                (hash >> 32) & 0xffff,
                (hash >> 16) & 0xffff,
                hash & 0xffff
            );
        }
    }
    if epistat_testkit::RECORDED_PLATFORM {
        assert_eq!(current.as_slice(), GOLDEN);
    }
}
