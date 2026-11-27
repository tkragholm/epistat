//! Levels of an ICD-10 code and blocks from a table the caller supplies.
//!
//! The 3-character category and the 4-character subcategory of a normalised
//! ICD-10 code are prefixes, and [`icd10_levels::ancestor`] reads them. The
//! chapter of a code is [`crate::Icd10Chapter`]. The blocks between them, such
//! as `C00-C14`, are a table of the WHO classification, which is under WHO
//! copyright, so [`CodeHierarchy`] takes the table from the caller.

use super::{CodeMap, CodeSet};
use crate::Error;

/// A level of an ICD-10 code that is a prefix of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Icd10Level {
    /// The 3-character category, a letter and two digits.
    Category,
    /// The 4-character subcategory.
    Subcategory,
}

/// Levels of an ICD-10 code.
pub mod icd10_levels {
    use super::Icd10Level;

    /// The prefix of a normalised code at `level`, or [`None`] when the code is
    /// shorter than the level.
    #[must_use]
    pub fn ancestor(normalized: &str, level: Icd10Level) -> Option<&str> {
        let len = match level {
            Icd10Level::Category => 3,
            Icd10Level::Subcategory => 4,
        };
        normalized.get(..len)
    }
}

/// Labelled, non-overlapping ranges of codes, such as the blocks of ICD-10.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodeHierarchy {
    labels: Vec<String>,
    sets: Vec<CodeSet>,
}

impl CodeHierarchy {
    /// A hierarchy from `(label, specification)` pairs, each specification as
    /// [`CodeSet::parse`] reads it.
    ///
    /// # Errors
    ///
    /// [`Error::CodeItem`] when a specification does not parse,
    /// [`Error::DuplicateName`] when a label is used twice, and
    /// [`Error::SharedCode`] when two entries share a code.
    pub fn from_ranges(ranges: &[(&str, &str)]) -> Result<Self, Error> {
        let mut labels: Vec<String> = Vec::with_capacity(ranges.len());
        let mut sets: Vec<CodeSet> = Vec::with_capacity(ranges.len());
        for &(label, spec) in ranges {
            if labels.iter().any(|l| l == label) {
                return Err(Error::DuplicateName {
                    what: "label",
                    name: label.to_owned(),
                });
            }
            let set = CodeSet::parse(spec)?;
            if let Some(k) = sets.iter().position(|s| !s.intersection(&set).is_empty()) {
                return Err(Error::SharedCode {
                    what: "entries",
                    first: labels[k].clone(),
                    second: label.to_owned(),
                });
            }
            labels.push(label.to_owned());
            sets.push(set);
        }
        Ok(Self { labels, sets })
    }

    /// The labels, in the order given.
    #[must_use]
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// The label of the entry that holds a normalised code, if any.
    #[must_use]
    pub fn label_of(&self, normalized: &str) -> Option<&str> {
        self.sets
            .iter()
            .position(|s| s.contains(normalized))
            .map(|k| self.labels[k].as_str())
    }

    /// For each distinct code, the position of its entry in
    /// [`labels`](Self::labels).
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when the hierarchy has more than 64 entries.
    pub fn classify(&self, distinct_codes: &[&str]) -> Result<Vec<Option<u32>>, Error> {
        let map = CodeMap::new(
            self.labels
                .iter()
                .cloned()
                .zip(self.sets.iter().cloned())
                .collect(),
        )?;
        Ok(distinct_codes
            .iter()
            .map(|code| {
                let mask = map.mask(code);
                (mask != 0).then(|| mask.trailing_zeros())
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn category_and_subcategory_are_prefixes() {
        assert_eq!(
            icd10_levels::ancestor("C509", Icd10Level::Category),
            Some("C50")
        );
        assert_eq!(
            icd10_levels::ancestor("C509", Icd10Level::Subcategory),
            Some("C509")
        );
        assert_eq!(icd10_levels::ancestor("C50", Icd10Level::Subcategory), None);
    }

    #[test]
    fn blocks_classify_codes_and_refuse_overlap() {
        let h =
            CodeHierarchy::from_ranges(&[("lip", "C00-C14"), ("digestive", "C15-C26")]).unwrap();
        assert_eq!(h.label_of("C140"), Some("lip"));
        assert_eq!(h.label_of("C15"), Some("digestive"));
        assert_eq!(h.label_of("C27"), None);
        assert_eq!(
            h.classify(&["C01", "C20", "D01"]).unwrap(),
            [Some(0), Some(1), None]
        );
        assert!(CodeHierarchy::from_ranges(&[("a", "C00-C14"), ("b", "C10-C20")]).is_err());
        assert!(CodeHierarchy::from_ranges(&[("a", "C00"), ("a", "C20")]).is_err());
    }
}
