//! The Danish coding of the Charlson index, ICD-8 and ICD-10.
//!
//! The 19 conditions, their ICD-10 codes and the Charlson weights are those of
//! Table 1 of Thygesen et al. 2011, who report the positive predictive value
//! of each condition's ICD-10 codes in the Danish National Patient Register.
//! The ICD-8 codes, for diagnoses before 1994, are those of Table A of
//! Christensen et al. 2011, whose ICD-10 codes are the same as Thygesen's and
//! whose weights are Charlson's of 1987. The specifications are written here
//! as the tables print them, ranges included, and [`CodeSet::parse`] reads
//! them, so `E10.2-E10.8` is the range of the tables and not a list of
//! prefixes copied from it.
//!
//! The tables do not state a hierarchy. The map uses the three pairs of the
//! Quan coding and of `heaven::charlsonIndex`: moderate or severe liver
//! disease cancels mild liver disease, diabetes with chronic complications
//! cancels diabetes without, and a metastatic solid tumour cancels any tumour.
//! Leukaemia and lymphoma are conditions of their own, as in Charlson 1987,
//! and are not cancelled.

use super::Condition;
use crate::codes::{CodeSet, EraCodeSet};

/// A condition of the Danish table: key, label, ICD-8 and ICD-10
/// specifications, weight and the keys it cancels.
struct Row {
    key: &'static str,
    label: &'static str,
    icd8: &'static str,
    icd10: &'static str,
    weight: f64,
    supersedes: &'static [&'static str],
}

/// Thygesen et al. 2011, Table 1 (ICD-10), and Christensen et al. 2011,
/// Table A (ICD-8 and weights), in the order of the tables.
static TABLE: [Row; 19] = [
    Row {
        key: "mi",
        label: "Myocardial infarction",
        icd8: "410",
        icd10: "I21, I22, I23",
        weight: 1.0,
        supersedes: &[],
    },
    Row {
        key: "chf",
        label: "Congestive heart failure",
        icd8: "427.09, 427.10, 427.11, 427.19, 428.99, 782.49",
        icd10: "I50, I11.0, I13.0, I13.2",
        weight: 1.0,
        supersedes: &[],
    },
    Row {
        key: "pvd",
        label: "Peripheral vascular disease",
        icd8: "440, 441, 442, 443, 444, 445",
        icd10: "I70, I71, I72, I73, I74, I77",
        weight: 1.0,
        supersedes: &[],
    },
    Row {
        key: "cevd",
        label: "Cerebrovascular disease",
        icd8: "430-438",
        icd10: "I60-I69, G45, G46",
        weight: 1.0,
        supersedes: &[],
    },
    Row {
        key: "dementia",
        label: "Dementia",
        icd8: "290.09-290.19, 293.09",
        icd10: "F00-F03, F05.1, G30",
        weight: 1.0,
        supersedes: &[],
    },
    Row {
        key: "cpd",
        label: "Chronic pulmonary disease",
        icd8: "490-493, 515-518",
        icd10: "J40-J47, J60-J67, J68.4, J70.1, J70.3, J84.1, J92.0, J96.1, J98.2, J98.3",
        weight: 1.0,
        supersedes: &[],
    },
    Row {
        key: "rheumd",
        label: "Connective tissue disease",
        icd8: "712, 716, 734, 446, 135.99",
        icd10: "M05, M06, M08, M09, M30, M31, M32, M33, M34, M35, M36, D86",
        weight: 1.0,
        supersedes: &[],
    },
    Row {
        key: "pud",
        label: "Ulcer disease",
        icd8: "530.91, 530.98, 531-534",
        icd10: "K22.1, K25-K28",
        weight: 1.0,
        supersedes: &[],
    },
    Row {
        key: "mld",
        label: "Mild liver disease",
        icd8: "571, 573.01, 573.04",
        icd10: "B18, K70.0-K70.3, K70.9, K71, K73, K74, K76.0",
        weight: 1.0,
        supersedes: &[],
    },
    Row {
        key: "diab",
        label: "Diabetes",
        icd8: "249.00, 249.06, 249.07, 249.09, 250.00, 250.06, 250.07, 250.09",
        icd10: "E10.0, E10.1, E10.9, E11.0, E11.1, E11.9",
        weight: 1.0,
        supersedes: &[],
    },
    Row {
        key: "hp",
        label: "Hemiplegia",
        icd8: "344",
        icd10: "G81, G82",
        weight: 2.0,
        supersedes: &[],
    },
    Row {
        key: "rend",
        label: "Moderate to severe renal disease",
        icd8: "403, 404, 580-583, 584, 590.09, 593.19, 753.10-753.19, 792",
        icd10: "I12, I13, N00-N05, N07, N11, N14, N17-N19, Q61",
        weight: 2.0,
        supersedes: &[],
    },
    Row {
        key: "diabwc",
        label: "Diabetes with end organ damage",
        icd8: "249.01-249.05, 249.08, 250.01-250.05, 250.08",
        icd10: "E10.2-E10.8, E11.2-E11.8",
        weight: 2.0,
        supersedes: &["diab"],
    },
    Row {
        key: "canc",
        label: "Any tumour",
        icd8: "140-194",
        icd10: "C00-C75",
        weight: 2.0,
        supersedes: &[],
    },
    Row {
        key: "leuk",
        label: "Leukaemia",
        icd8: "204-207",
        icd10: "C91-C95",
        weight: 2.0,
        supersedes: &[],
    },
    Row {
        key: "lymph",
        label: "Lymphoma",
        icd8: "200-203, 275.59",
        icd10: "C81-C85, C88, C90, C96",
        weight: 2.0,
        supersedes: &[],
    },
    Row {
        key: "msld",
        label: "Moderate to severe liver disease",
        icd8: "070.00, 070.02, 070.04, 070.06, 070.08, 573.00, 456.00-456.09",
        icd10: "B15.0, B16.0, B16.2, B19.0, K70.4, K72, K76.6, I85",
        weight: 3.0,
        supersedes: &["mld"],
    },
    Row {
        key: "metacanc",
        label: "Metastatic solid tumour",
        icd8: "195-198, 199",
        icd10: "C76-C80",
        weight: 6.0,
        supersedes: &["canc"],
    },
    Row {
        key: "aids",
        label: "AIDS",
        icd8: "079.83",
        icd10: "B21-B24",
        weight: 6.0,
        supersedes: &[],
    },
];

pub(super) fn conditions() -> Vec<Condition> {
    let parse = |spec: &str| {
        CodeSet::parse(spec).unwrap_or_else(|_| unreachable!("the Danish table parses"))
    };
    TABLE
        .iter()
        .map(|row| Condition {
            key: row.key.to_owned(),
            label: row.label.to_owned(),
            codes: EraCodeSet {
                icd8: parse(row.icd8),
                icd10: parse(row.icd10),
            },
            supersedes: row.supersedes.iter().map(|&s| s.to_owned()).collect(),
        })
        .collect()
}

pub(super) fn weights() -> Vec<f64> {
    TABLE.iter().map(|row| row.weight).collect()
}
