//! The ICD-10 coding of the Charlson and Elixhauser conditions by Quan et al.
//! 2005.
//!
//! The 17 Charlson conditions are those of Table 1 and the 31 Elixhauser
//! conditions those of Table 2, ICD-10 column, in the order of the tables,
//! with the tables' names as labels. The specifications are written as the
//! tables print them; the [module](super) states how each printed form is
//! read.
//!
//! The tables give the conditions and their codes only. The hierarchy, in
//! which the severe form of a condition cancels the milder one, is that of the
//! module [`comorbidity`](super), stated with each row's `supersedes`.
//!
//! The weights are those of Charlson et al. 1987, Quan et al. 2011 (Table 4)
//! and van Walraven et al. 2009, one per row of the tables, and the tests
//! check every score against the R package `comorbidity` 1.1.0.

use super::Condition;
use crate::codes::{CodeSet, EraCodeSet};

/// A condition of a Quan table: key, label, ICD-10 specification and the
/// keys it cancels.
struct Row {
    key: &'static str,
    label: &'static str,
    icd10: &'static str,
    supersedes: &'static [&'static str],
}

/// Quan et al. 2005, Table 1, ICD-10 column.
static CHARLSON: [Row; 17] = [
    Row {
        key: "mi",
        label: "Myocardial infarction",
        icd10: "I21.x, I22.x, I25.2",
        supersedes: &[],
    },
    Row {
        key: "chf",
        label: "Congestive heart failure",
        icd10: "I09.9, I11.0, I13.0, I13.2, I25.5, I42.0, I42.5–I42.9, I43.x, I50.x, P29.0",
        supersedes: &[],
    },
    Row {
        key: "pvd",
        label: "Peripheral vascular disease",
        icd10: "I70.x, I71.x, I73.1, I73.8, I73.9, I77.1, I79.0, I79.2, K55.1, K55.8, K55.9, Z95.8, Z95.9",
        supersedes: &[],
    },
    Row {
        key: "cevd",
        label: "Cerebrovascular disease",
        icd10: "G45.x, G46.x, H34.0, I60.x–I69.x",
        supersedes: &[],
    },
    Row {
        key: "dementia",
        label: "Dementia",
        icd10: "F00.x–F03.x, F05.1, G30.x, G31.1",
        supersedes: &[],
    },
    Row {
        key: "cpd",
        label: "Chronic pulmonary disease",
        icd10: "I27.8, I27.9, J40.x–J47.x, J60.x–J67.x, J68.4, J70.1, J70.3",
        supersedes: &[],
    },
    Row {
        key: "rheumd",
        label: "Rheumatic disease",
        icd10: "M05.x, M06.x, M31.5, M32.x–M34.x, M35.1, M35.3, M36.0",
        supersedes: &[],
    },
    Row {
        key: "pud",
        label: "Peptic ulcer disease",
        icd10: "K25.x–K28.x",
        supersedes: &[],
    },
    Row {
        key: "mld",
        label: "Mild liver disease",
        icd10: "B18.x, K70.0–K70.3, K70.9, K71.3–K71.5, K71.7, K73.x, K74.x, K76.0, K76.2–K76.4, K76.8, K76.9, Z94.4",
        supersedes: &[],
    },
    Row {
        key: "diab",
        label: "Diabetes without chronic complication",
        icd10: "E10.0, E10.1, E10.6, E10.8, E10.9, E11.0, E11.1, E11.6, E11.8, E11.9, E12.0, E12.1, E12.6, E12.8, E12.9, E13.0, E13.1, E13.6, E13.8, E13.9, E14.0, E14.1, E14.6, E14.8, E14.9",
        supersedes: &[],
    },
    Row {
        key: "diabwc",
        label: "Diabetes with chronic complication",
        icd10: "E10.2–E10.5, E10.7, E11.2–E11.5, E11.7, E12.2–E12.5, E12.7, E13.2–E13.5, E13.7, E14.2–E14.5, E14.7",
        supersedes: &["diab"],
    },
    Row {
        key: "hp",
        label: "Hemiplegia or paraplegia",
        icd10: "G04.1, G11.4, G80.1, G80.2, G81.x, G82.x, G83.0–G83.4, G83.9",
        supersedes: &[],
    },
    Row {
        key: "rend",
        label: "Renal disease",
        icd10: "I12.0, I13.1, N03.2–N03.7, N05.2–N05.7, N18.x, N19.x, N25.0, Z49.0–Z49.2, Z94.0, Z99.2",
        supersedes: &[],
    },
    Row {
        key: "canc",
        label: "Any malignancy, including lymphoma and leukemia, except malignant neoplasm of skin",
        icd10: "C00.x–C26.x, C30.x–C34.x, C37.x–C41.x, C43.x, C45.x–C58.x, C60.x–C76.x, C81.x–C85.x, C88.x, C90.x–C97.x",
        supersedes: &[],
    },
    Row {
        key: "msld",
        label: "Moderate or severe liver disease",
        icd10: "I85.0, I85.9, I86.4, I98.2, K70.4, K71.1, K72.1, K72.9, K76.5, K76.6, K76.7",
        supersedes: &["mld"],
    },
    Row {
        key: "metacanc",
        label: "Metastatic solid tumor",
        icd10: "C77.x–C80.x",
        supersedes: &["canc"],
    },
    Row {
        key: "aids",
        label: "AIDS/HIV",
        icd10: "B20.x–B22.x, B24.x",
        supersedes: &[],
    },
];

/// Quan et al. 2005, Table 2, ICD-10 column.
static ELIXHAUSER: [Row; 31] = [
    Row {
        key: "chf",
        label: "Congestive heart failure",
        icd10: "I09.9, I11.0, I13.0, I13.2, I25.5, I42.0, I42.5–I42.9, I43.x, I50.x, P29.0",
        supersedes: &[],
    },
    Row {
        key: "carit",
        label: "Cardiac arrhythmias",
        icd10: "I44.1–I44.3, I45.6, I45.9, I47.x–I49.x, R00.0, R00.1, R00.8, T82.1, Z45.0, Z95.0",
        supersedes: &[],
    },
    Row {
        key: "valv",
        label: "Valvular disease",
        icd10: "A52.0, I05.x–I08.x, I09.1, I09.8, I34.x–I39.x, Q23.0–Q23.3, Z95.2–Z95.4",
        supersedes: &[],
    },
    Row {
        key: "pcd",
        label: "Pulmonary circulation disorders",
        icd10: "I26.x, I27.x, I28.0, I28.8, I28.9",
        supersedes: &[],
    },
    Row {
        key: "pvd",
        label: "Peripheral vascular disorders",
        icd10: "I70.x, I71.x, I73.1, I73.8, I73.9, I77.1, I79.0, I79.2, K55.1, K55.8, K55.9, Z95.8, Z95.9",
        supersedes: &[],
    },
    Row {
        key: "hypunc",
        label: "Hypertension, uncomplicated",
        icd10: "I10.x",
        supersedes: &[],
    },
    Row {
        key: "hypc",
        label: "Hypertension, complicated",
        icd10: "I11.x–I13.x, I15.x",
        supersedes: &["hypunc"],
    },
    Row {
        key: "para",
        label: "Paralysis",
        icd10: "G04.1, G11.4, G80.1, G80.2, G81.x, G82.x, G83.0–G83.4, G83.9",
        supersedes: &[],
    },
    Row {
        key: "ond",
        label: "Other neurological disorders",
        icd10: "G10.x–G13.x, G20.x–G22.x, G25.4, G25.5, G31.2, G31.8, G31.9, G32.x, G35.x–G37.x, G40.x, G41.x, G93.1, G93.4, R47.0, R56.x",
        supersedes: &[],
    },
    Row {
        key: "cpd",
        label: "Chronic pulmonary disease",
        icd10: "I27.8, I27.9, J40.x–J47.x, J60.x–J67.x, J68.4, J70.1, J70.3",
        supersedes: &[],
    },
    Row {
        key: "diabunc",
        label: "Diabetes, uncomplicated",
        icd10: "E10.0, E10.1, E10.9, E11.0, E11.1, E11.9, E12.0, E12.1, E12.9, E13.0, E13.1, E13.9, E14.0, E14.1, E14.9",
        supersedes: &[],
    },
    Row {
        key: "diabc",
        label: "Diabetes, complicated",
        icd10: "E10.2–E10.8, E11.2–E11.8, E12.2–E12.8, E13.2–E13.8, E14.2–E14.8",
        supersedes: &["diabunc"],
    },
    Row {
        key: "hypothy",
        label: "Hypothyroidism",
        icd10: "E00.x–E03.x, E89.0",
        supersedes: &[],
    },
    Row {
        key: "rf",
        label: "Renal failure",
        icd10: "I12.0, I13.1, N18.x, N19.x, N25.0, Z49.0–Z49.2, Z94.0, Z99.2",
        supersedes: &[],
    },
    Row {
        key: "ld",
        label: "Liver disease",
        icd10: "B18.x, I85.x, I86.4, I98.2, K70.x, K71.1, K71.3–K71.5, K71.7, K72.x–K74.x, K76.0, K76.2–K76.9, Z94.4",
        supersedes: &[],
    },
    Row {
        key: "pud",
        label: "Peptic ulcer disease excluding bleeding",
        icd10: "K25.7, K25.9, K26.7, K26.9, K27.7, K27.9, K28.7, K28.9",
        supersedes: &[],
    },
    Row {
        key: "aids",
        label: "AIDS/HIV",
        icd10: "B20.x–B22.x, B24.x",
        supersedes: &[],
    },
    Row {
        key: "lymph",
        label: "Lymphoma",
        icd10: "C81.x–C85.x, C88.x, C96.x, C90.0, C90.2",
        supersedes: &[],
    },
    Row {
        key: "metacanc",
        label: "Metastatic cancer",
        icd10: "C77.x–C80.x",
        supersedes: &["solidtum"],
    },
    Row {
        key: "solidtum",
        label: "Solid tumor without metastasis",
        icd10: "C00.x–C26.x, C30.x–C34.x, C37.x–C41.x, C43.x, C45.x–C58.x, C60.x–C76.x, C97.x",
        supersedes: &[],
    },
    Row {
        key: "rheumd",
        label: "Rheumatoid arthritis/collagen vascular diseases",
        icd10: "L94.0, L94.1, L94.3, M05.x, M06.x, M08.x, M12.0, M12.3, M30.x, M31.0–M31.3, M32.x–M35.x, M45.x, M46.1, M46.8, M46.9",
        supersedes: &[],
    },
    Row {
        key: "coag",
        label: "Coagulopathy",
        icd10: "D65–D68.x, D69.1, D69.3–D69.6",
        supersedes: &[],
    },
    Row {
        key: "obes",
        label: "Obesity",
        icd10: "E66.x",
        supersedes: &[],
    },
    Row {
        key: "wloss",
        label: "Weight loss",
        icd10: "E40.x–E46.x, R63.4, R64",
        supersedes: &[],
    },
    Row {
        key: "fed",
        label: "Fluid and electrolyte disorders",
        icd10: "E22.2, E86.x, E87.x",
        supersedes: &[],
    },
    Row {
        key: "blane",
        label: "Blood loss anemia",
        icd10: "D50.0",
        supersedes: &[],
    },
    Row {
        key: "dane",
        label: "Deficiency anemia",
        icd10: "D50.8, D50.9, D51.x–D53.x",
        supersedes: &[],
    },
    Row {
        key: "alcohol",
        label: "Alcohol abuse",
        icd10: "F10, E52, G62.1, I42.6, K29.2, K70.0, K70.3, K70.9, T51.x, Z50.2, Z71.4, Z72.1",
        supersedes: &[],
    },
    Row {
        key: "drug",
        label: "Drug abuse",
        icd10: "F11.x–F16.x, F18.x, F19.x, Z71.5, Z72.2",
        supersedes: &[],
    },
    Row {
        key: "psycho",
        label: "Psychoses",
        icd10: "F20.x, F22.x–F25.x, F28.x, F29.x, F30.2, F31.2, F31.5",
        supersedes: &[],
    },
    Row {
        key: "depre",
        label: "Depression",
        icd10: "F20.4, F31.3–F31.5, F32.x, F33.x, F34.1, F41.2, F43.2",
        supersedes: &[],
    },
];

/// The weights of Charlson et al. 1987 (J Chronic Dis 40:373-83), 1, 2, 3
/// or 6 points per condition, in the order of [`CHARLSON`]. The 1987 index
/// scores leukaemia and lymphoma as conditions of their own, 2 points each;
/// the Quan coding folds them into any malignancy, also 2 points.
pub(super) static CHARLSON_1987: [f64; 17] = [
    1.0, // mi
    1.0, // chf
    1.0, // pvd
    1.0, // cevd
    1.0, // dementia
    1.0, // cpd
    1.0, // rheumd
    1.0, // pud
    1.0, // mld
    1.0, // diab
    2.0, // diabwc
    2.0, // hp
    2.0, // rend
    2.0, // canc
    3.0, // msld
    6.0, // metacanc
    6.0, // aids
];

/// The updated weights of Quan et al. 2011 (Am J Epidemiol 173:676-82),
/// Table 4, in the order of [`CHARLSON`]. Five conditions have the weight 0.
pub(super) static CHARLSON_QUAN_2011: [f64; 17] = [
    0.0, // mi
    2.0, // chf
    0.0, // pvd
    0.0, // cevd
    2.0, // dementia
    1.0, // cpd
    1.0, // rheumd
    0.0, // pud
    2.0, // mld
    0.0, // diab
    1.0, // diabwc
    2.0, // hp
    1.0, // rend
    2.0, // canc
    4.0, // msld
    6.0, // metacanc
    4.0, // aids
];

/// The point weights of van Walraven et al. 2009 (Med Care 47:626-33), in the
/// order of [`ELIXHAUSER`], some of them negative. Van Walraven fitted
/// hypertension as one group, so uncomplicated and complicated hypertension
/// both have its weight, 0.
pub(super) static ELIXHAUSER_VAN_WALRAVEN_2009: [f64; 31] = [
    7.0,  // chf
    5.0,  // carit
    -1.0, // valv
    4.0,  // pcd
    2.0,  // pvd
    0.0,  // hypunc
    0.0,  // hypc
    7.0,  // para
    6.0,  // ond
    3.0,  // cpd
    0.0,  // diabunc
    0.0,  // diabc
    0.0,  // hypothy
    5.0,  // rf
    11.0, // ld
    0.0,  // pud
    0.0,  // aids
    9.0,  // lymph
    12.0, // metacanc
    4.0,  // solidtum
    0.0,  // rheumd
    3.0,  // coag
    -4.0, // obes
    6.0,  // wloss
    5.0,  // fed
    -2.0, // blane
    -2.0, // dane
    0.0,  // alcohol
    -7.0, // drug
    0.0,  // psycho
    -3.0, // depre
];

fn conditions(table: &[Row]) -> Vec<Condition> {
    table
        .iter()
        .map(|row| Condition {
            key: row.key.to_owned(),
            label: row.label.to_owned(),
            codes: EraCodeSet {
                icd8: CodeSet::default(),
                icd10: CodeSet::parse(row.icd10)
                    .unwrap_or_else(|_| unreachable!("the Quan tables parse")),
            },
            supersedes: row.supersedes.iter().map(|&s| s.to_owned()).collect(),
        })
        .collect()
}

/// The 17 Charlson conditions of Table 1.
pub(super) fn charlson() -> Vec<Condition> {
    conditions(&CHARLSON)
}

/// The 31 Elixhauser conditions of Table 2.
pub(super) fn elixhauser() -> Vec<Condition> {
    conditions(&ELIXHAUSER)
}
