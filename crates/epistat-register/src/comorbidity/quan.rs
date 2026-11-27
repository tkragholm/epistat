//! The ICD-10 coding of the Charlson and Elixhauser conditions by Quan et al.
//! 2005.

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

static CHARLSON: [Row; 17] = [
    Row {
        key: "mi",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "chf",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "pvd",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "cevd",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "dementia",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "cpd",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "rheumd",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "pud",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "mld",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "diab",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "diabwc",
        label: "",
        icd10: "",
        supersedes: &["diab"],
    },
    Row {
        key: "hp",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "rend",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "canc",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "msld",
        label: "",
        icd10: "",
        supersedes: &["mld"],
    },
    Row {
        key: "metacanc",
        label: "",
        icd10: "",
        supersedes: &["canc"],
    },
    Row {
        key: "aids",
        label: "",
        icd10: "",
        supersedes: &[],
    },
];

static ELIXHAUSER: [Row; 31] = [
    Row {
        key: "chf",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "carit",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "valv",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "pcd",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "pvd",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "hypunc",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "hypc",
        label: "",
        icd10: "",
        supersedes: &["hypunc"],
    },
    Row {
        key: "para",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "ond",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "cpd",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "diabunc",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "diabc",
        label: "",
        icd10: "",
        supersedes: &["diabunc"],
    },
    Row {
        key: "hypothy",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "rf",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "ld",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "pud",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "aids",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "lymph",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "metacanc",
        label: "",
        icd10: "",
        supersedes: &["solidtum"],
    },
    Row {
        key: "solidtum",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "rheumd",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "coag",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "obes",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "wloss",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "fed",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "blane",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "dane",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "alcohol",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "drug",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "psycho",
        label: "",
        icd10: "",
        supersedes: &[],
    },
    Row {
        key: "depre",
        label: "",
        icd10: "",
        supersedes: &[],
    },
];

/// The weights of Charlson et al. 1987, in the order of [`CHARLSON`].
pub(super) static CHARLSON_1987: [f64; 17] = [
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 6.0, 6.0,
];

/// The weights of Quan et al. 2011, Table 4, in the order of [`CHARLSON`].
pub(super) static CHARLSON_QUAN_2011: [f64; 17] = [
    0.0, 2.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 4.0, 6.0, 4.0,
];

/// The point weights of van Walraven et al. 2009, in the order of
/// [`ELIXHAUSER`].
pub(super) static ELIXHAUSER_VAN_WALRAVEN_2009: [f64; 31] = [
    7.0, 5.0, -1.0, 4.0, 2.0, 0.0, 0.0, 7.0, 6.0, 3.0, 0.0, 0.0, 0.0, 5.0, 11.0, 0.0, 0.0, 9.0,
    12.0, 4.0, 0.0, 3.0, -4.0, 6.0, 5.0, -2.0, -2.0, 0.0, -7.0, 0.0, -3.0,
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

/// The 17 Charlson conditions.
pub(super) fn charlson() -> Vec<Condition> {
    conditions(&CHARLSON)
}

/// The 31 Elixhauser conditions.
pub(super) fn elixhauser() -> Vec<Condition> {
    conditions(&ELIXHAUSER)
}
