//! ICD-8 beside ICD-10.
//!
//! Danish hospital diagnoses are coded in ICD-8 up to the end of 1993 and in
//! ICD-10 from 1994, and a study that reads the history before 1994 meets both.
//! ICD-8 codes are numeric: three digits for the category, a fourth digit for
//! the subdivision, and in the Danish version a fifth \[1\]. ICD-10 codes begin
//! with a letter. The first character therefore tells the revision, and no
//! code is in both.
//!
//! There is no one-to-one translation between the revisions. The practice in
//! Danish register research is to define each condition twice, once as a list
//! of ICD-8 codes and once as a list of ICD-10 codes, and to take the union of
//! the two flags (see [`EraCodeSet`]). A [`Crosswalk`] is for converting single
//! codes when a study has a table it can cite. It is a structure and a
//! validator and holds no data. Every mapping carries its [`MapKind`], so that
//! an approximate mapping is not used as an exact one.
//!
//! [`normalize_any`] normalises a code of either revision. It leaves
//! [`normalize_diagnosis_code`](crate::normalize_diagnosis_code), which drops
//! a code with no letter, as it is.
//!
//! # References
//!
//! 1. Schmidt M, Schmidt SAJ, Sandegaard JL, Ehrenstein V, Pedersen L,
//!    Sorensen HT. The Danish National Patient Registry: a review of content,
//!    data quality, and research potential. Clin Epidemiol. 2015;7:449-90.

use std::collections::BTreeSet;

use super::CodeSet;
use crate::{Error, normalize_diagnosis_code};

/// The revision of the International Classification of Diseases a code is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Revision {
    /// ICD-8, numeric codes.
    Icd8,
    /// ICD-10, codes that begin with a letter.
    Icd10,
}

/// The revision of a normalised code by its first character, or [`None`] for
/// an empty code or one that begins with another character.
#[must_use]
pub fn revision_of(normalized: &str) -> Option<Revision> {
    match normalized.as_bytes().first()? {
        b'0'..=b'9' => Some(Revision::Icd8),
        b'A'..=b'Z' => Some(Revision::Icd10),
        _ => None,
    }
}

/// Normalise a diagnosis code of either revision.
///
/// A code that begins, after whitespace and punctuation are dropped, with a
/// letter is ICD-10 and is normalised as
/// [`normalize_diagnosis_code`](crate::normalize_diagnosis_code) does, with the
/// same result. A code that begins with a digit is ICD-8 and is kept as its
/// digits, so that `"250.09"` and `"25009"` both give `"25009"`. A code that
/// begins with a digit and holds a letter, an empty code and a code with no
/// letter or digit give [`None`]: nothing is repaired.
#[must_use]
pub fn normalize_any(code: &str) -> Option<(Revision, String)> {
    let clean: String = code
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .map(|c| c.to_ascii_uppercase())
        .collect();
    match clean.as_bytes().first()? {
        b'0'..=b'9' => clean
            .bytes()
            .all(|b| b.is_ascii_digit())
            .then_some((Revision::Icd8, clean)),
        _ => normalize_diagnosis_code(code).map(|n| (Revision::Icd10, n.full_code)),
    }
}

/// A condition as one code set for each revision.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EraCodeSet {
    /// The ICD-8 codes of the condition.
    pub icd8: CodeSet,
    /// The ICD-10 codes of the condition.
    pub icd10: CodeSet,
}

impl EraCodeSet {
    /// Whether a normalised code is in the set of its own revision. A code of
    /// neither revision is in neither.
    #[must_use]
    pub fn contains(&self, normalized: &str) -> bool {
        match revision_of(normalized) {
            Some(Revision::Icd8) => self.icd8.contains(normalized),
            Some(Revision::Icd10) => self.icd10.contains(normalized),
            None => false,
        }
    }
}

/// How an ICD-8 code maps to ICD-10 in a crosswalk the caller supplies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MapKind {
    /// The ICD-8 code has one ICD-10 code and no other ICD-8 code has it.
    OneToOne,
    /// The ICD-8 code has several ICD-10 codes.
    OneToMany,
    /// Several ICD-8 codes have this ICD-10 code.
    ManyToOne,
    /// The mapping is an approximation that the source of the table marks so.
    Approximate,
}

/// One mapping of a [`Crosswalk`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CrosswalkEntry {
    /// The ICD-8 code, normalised.
    pub from: String,
    /// The ICD-10 code, normalised.
    pub to: String,
    /// The kind of the mapping as the table declares it.
    pub kind: MapKind,
}

/// A many-to-many table from ICD-8 to ICD-10 codes, supplied by the caller.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Crosswalk {
    entries: Vec<CrosswalkEntry>,
}

/// How much of a set of ICD-8 codes a [`Crosswalk`] covers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Coverage {
    /// The number of distinct codes asked about.
    pub codes: usize,
    /// How many of them have an entry.
    pub mapped: usize,
    /// The codes with no entry, in byte order.
    pub unmapped: Vec<String>,
    /// How many mapped codes have an entry of kind [`MapKind::OneToOne`].
    /// A code with entries of several kinds counts under each.
    pub one_to_one: usize,
    /// How many mapped codes have an entry of kind [`MapKind::OneToMany`].
    pub one_to_many: usize,
    /// How many mapped codes have an entry of kind [`MapKind::ManyToOne`].
    pub many_to_one: usize,
    /// How many mapped codes have an entry of kind [`MapKind::Approximate`].
    pub approximate: usize,
}

impl Crosswalk {
    /// Build a crosswalk from `(icd8, icd10, kind)` triples.
    ///
    /// The codes are normalised as [`normalize_any`] does, and the sources
    /// must be ICD-8 and the targets ICD-10. The declared kinds are checked
    /// against the table: [`MapKind::OneToOne`] needs a source with one target
    /// that has one source, [`MapKind::OneToMany`] a source with several
    /// targets and [`MapKind::ManyToOne`] a target with several sources.
    /// [`MapKind::Approximate`] is accepted anywhere.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] for a code of the wrong revision or none, a
    /// pair given twice, or a kind that the table contradicts.
    pub fn from_pairs(pairs: &[(&str, &str, MapKind)]) -> Result<Self, Error> {
        let mut entries = Vec::with_capacity(pairs.len());
        for &(from, to, kind) in pairs {
            let Some((Revision::Icd8, from_code)) = normalize_any(from) else {
                return Err(Error::invalid(format!(
                    "crosswalk source {from:?} is not an ICD-8 code"
                )));
            };
            let Some((Revision::Icd10, to_code)) = normalize_any(to) else {
                return Err(Error::invalid(format!(
                    "crosswalk target {to:?} is not an ICD-10 code"
                )));
            };
            entries.push(CrosswalkEntry {
                from: from_code,
                to: to_code,
                kind,
            });
        }
        entries.sort_by(|a, b| (&a.from, &a.to).cmp(&(&b.from, &b.to)));
        if let Some(w) = entries
            .windows(2)
            .find(|w| w[0].from == w[1].from && w[0].to == w[1].to)
        {
            return Err(Error::invalid(format!(
                "crosswalk pair {} to {} is given twice",
                w[0].from, w[0].to
            )));
        }
        let targets_of = |from: &str| entries.iter().filter(|e| e.from == from).count();
        let sources_of = |to: &str| entries.iter().filter(|e| e.to == to).count();
        for e in &entries {
            let (t, s) = (targets_of(&e.from), sources_of(&e.to));
            let consistent = match e.kind {
                MapKind::OneToOne => t == 1 && s == 1,
                MapKind::OneToMany => t > 1,
                MapKind::ManyToOne => s > 1,
                MapKind::Approximate => true,
            };
            if !consistent {
                return Err(Error::invalid(format!(
                    "crosswalk pair {} to {} is declared {:?}, but its source has {t} targets and its target {s} sources",
                    e.from, e.to, e.kind
                )));
            }
        }
        Ok(Self { entries })
    }

    /// The entries of one normalised ICD-8 code, in order of their targets.
    /// Empty when the code has none.
    #[must_use]
    pub fn map_icd8(&self, code: &str) -> &[CrosswalkEntry] {
        let lo = self.entries.partition_point(|e| e.from.as_str() < code);
        let hi = self.entries.partition_point(|e| e.from.as_str() <= code);
        &self.entries[lo..hi]
    }

    /// The coverage of a collection of normalised ICD-8 codes. A code given
    /// twice counts once.
    #[must_use]
    pub fn coverage<'a>(&self, codes: impl Iterator<Item = &'a str>) -> Coverage {
        let distinct: BTreeSet<&str> = codes.collect();
        let mut out = Coverage {
            codes: distinct.len(),
            ..Coverage::default()
        };
        for code in distinct {
            let entries = self.map_icd8(code);
            if entries.is_empty() {
                out.unmapped.push(code.to_owned());
                continue;
            }
            out.mapped += 1;
            let has = |kind| entries.iter().any(|e| e.kind == kind);
            out.one_to_one += usize::from(has(MapKind::OneToOne));
            out.one_to_many += usize::from(has(MapKind::OneToMany));
            out.many_to_one += usize::from(has(MapKind::ManyToOne));
            out.approximate += usize::from(has(MapKind::Approximate));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn revision_is_read_from_the_first_character() {
        assert_eq!(revision_of("24900"), Some(Revision::Icd8));
        assert_eq!(revision_of("E109"), Some(Revision::Icd10));
        assert_eq!(revision_of(""), None);
        assert_eq!(revision_of("-1"), None);
    }

    #[test]
    fn normalize_any_keeps_icd8_and_agrees_with_icd10() {
        assert_eq!(
            normalize_any("250.09"),
            Some((Revision::Icd8, "25009".into()))
        );
        assert_eq!(normalize_any(" 410 "), Some((Revision::Icd8, "410".into())));
        assert_eq!(normalize_any("12J"), None);
        assert_eq!(normalize_any(""), None);
        assert_eq!(normalize_any("..."), None);
        for raw in ["DJ45.0", "dj45", "D55", "DD55", "E10.2", "I428A", "c509"] {
            let icd10 = normalize_diagnosis_code(raw).unwrap().full_code;
            assert_eq!(normalize_any(raw), Some((Revision::Icd10, icd10)), "{raw}");
        }
        assert!(normalize_diagnosis_code("24900").is_none());
    }

    #[test]
    fn era_sets_route_by_revision_and_never_collide() {
        let era = EraCodeSet {
            icd8: CodeSet::parse("290-315, 410").unwrap(),
            icd10: CodeSet::parse("F").unwrap(),
        };
        assert!(era.contains("29009"));
        assert!(era.contains("41009"));
        assert!(!era.contains("316"));
        assert!(era.contains("F200"));
        assert!(!era.contains("G200"));
        assert!(!era.contains(""));
        // A digit-first code is tested against the ICD-8 set only.
        let icd10_only = EraCodeSet {
            icd8: CodeSet::default(),
            icd10: CodeSet::parse("0-9, A-Z").unwrap(),
        };
        assert!(icd10_only.icd10.contains("29009"));
        assert!(!icd10_only.contains("29009"));
    }

    #[test]
    fn crosswalk_checks_kinds_and_reports_coverage() {
        let table = [
            ("250", "E14", MapKind::OneToMany),
            ("250", "E10", MapKind::OneToMany),
            ("410", "I21", MapKind::OneToOne),
            ("411", "I24", MapKind::ManyToOne),
            ("412", "I24", MapKind::ManyToOne),
            ("427", "I49", MapKind::Approximate),
        ];
        let x = Crosswalk::from_pairs(&table).unwrap();
        let hit = x.map_icd8("250");
        assert_eq!(hit.len(), 2);
        assert_eq!(hit[0].to, "E10");
        assert_eq!(x.map_icd8("999").len(), 0);
        let cover = x.coverage(["250", "410", "999", "250", "411", "427"].into_iter());
        assert_eq!(cover.codes, 5);
        assert_eq!(cover.mapped, 4);
        assert_eq!(cover.unmapped, ["999"]);
        assert_eq!(
            (
                cover.one_to_one,
                cover.one_to_many,
                cover.many_to_one,
                cover.approximate
            ),
            (1, 1, 1, 1)
        );

        assert!(Crosswalk::from_pairs(&[("250", "E10", MapKind::OneToMany)]).is_err());
        assert!(
            Crosswalk::from_pairs(&[
                ("410", "I21", MapKind::OneToOne),
                ("411", "I21", MapKind::OneToOne)
            ])
            .is_err()
        );
        assert!(Crosswalk::from_pairs(&[("E10", "250", MapKind::OneToOne)]).is_err());
        assert!(
            Crosswalk::from_pairs(&[
                ("410", "I21", MapKind::OneToOne),
                ("410", "I21", MapKind::OneToOne)
            ])
            .is_err()
        );
    }
}
