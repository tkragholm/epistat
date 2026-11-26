//! Comorbidity indices: the Charlson index and the Elixhauser measure.
//!
//! A comorbidity index summarises the conditions recorded for a person before
//! an index date as a set of flags and a weighted sum of them. The Charlson
//! index \[1\] weights 17 conditions, and its ICD-10 coding is the algorithm of
//! Quan et al. \[2\] with the weights of Charlson \[1\] or the updated weights of
//! Quan et al. \[3\]. The Elixhauser measure \[4\] flags 31 conditions in the ICD-10
//! coding of Quan et al. \[2\] (Tables 1 and 2 for the Charlson and the Elixhauser
//! conditions), and van Walraven et al. \[5\] give it a point
//! score.
//!
//! # What is built in
//!
//! The code lists and weights are those of the R package `comorbidity` 1.1.0
//! \[6\], which takes them from Quan et al. \[2\] (the coding algorithms for the
//! Charlson and the Elixhauser conditions) and from the three weight tables
//! above. They are written to `lists.rs` by `tests/data/comorbidity_lists.R`,
//! and the tests check every code of the alphabet against the package.
//!
//! - [`ComorbidityMap::charlson_quan_icd10`] with [`charlson_original`]
//!   (Charlson 1987) or [`charlson_quan2011`] (Quan 2011, Table 4).
//! - [`ComorbidityMap::elixhauser_quan_icd10`] with
//!   [`elixhauser_van_walraven`] (van Walraven 2009, Table 3).
//!
//! - [`ComorbidityMap::charlson_denmark`] with [`charlson_denmark_weights`]:
//!   the 19 conditions of the Danish coding, with the ICD-10 codes of
//!   Thygesen et al. \[7\] (Table 1), whose positive predictive values in the
//!   Danish National Patient Register they report, and the ICD-8 codes for
//!   the years before 1994 and the Charlson 1987 weights of Christensen et al.
//!   \[8\] (Table A). The tables are written out by hand, with their ranges,
//!   and the tests check the flags and scores against `heaven::charlsonIndex`
//!   given the same list.
//!
//! Other lists are supplied by the caller: [`ComorbidityMap::set_icd10`] and
//! [`ComorbidityMap::set_icd8`] replace the codes of a condition, and
//! [`ComorbidityMap::new`] builds a map from nothing. Which revision a code is
//! in is read from the code, and the ICD-8 codes of a map count only for
//! diagnoses coded in ICD-8, so a study that reads history from 1977 needs a
//! map with both, as the Danish one has.
//!
//! # Method
//!
//! Each condition has one [`EraCodeSet`], a set of ICD-8
//! codes and a set of ICD-10 codes, and a code is tested against the set of its
//! own revision ([`revision_of`]). The distinct codes
//! of a column are classified once by [`ComorbidityMap::classify`] into a mask
//! with bit `k` set when the code is in condition `k`. The flags of an index are
//! the OR of the masks of the diagnoses in its look-back window
//! ([`flags_at_index`]). The hierarchy removes the milder form of a condition
//! when the more severe form is present,
//!
//! ```text
//! flags' = flags & !(OR of cancelled(k) for the set bits k of flags)
//! ```
//!
//! with `cancelled(k)` the mask of the conditions that condition `k` supersedes.
//! In the Charlson index moderate or severe liver disease supersedes mild liver
//! disease, diabetes with chronic complications supersedes diabetes without,
//! and metastatic solid tumour supersedes any malignancy. In the Elixhauser
//! measure complicated hypertension supersedes uncomplicated, complicated
//! diabetes supersedes uncomplicated, and metastatic cancer supersedes solid
//! tumour without metastasis. These are the pairs of `assign0 = TRUE` in the
//! package. The hierarchy is on the flag, not on the code, so a person with
//! codes of both forms has the severe flag only. In the Quan coding leukaemia
//! and lymphoma are part of the Charlson malignancy condition and the
//! metastatic flag cancels them with it, where the 1987 index and the R package
//! `heaven` keep them as conditions of their own.
//!
//! The score is the sum of the weights of the conditions whose flag is set, added
//! in condition order as `f64`. All built-in weights are integers, so the sum is
//! exact.
//!
//! # History too short
//!
//! A score of 0 says that no listed condition was recorded in the window, and
//! says nothing when the window starts before the register began.
//! [`IndexFlags::history_complete`] says whether the window lies inside the
//! recorded history, and [`IndexFlags::score`] returns [`None`] instead of a
//! number for an incomplete window when the strict reading is asked for.
//!
//! # Traps
//!
//! A code shorter than the longest item of a list cannot match every item:
//! `E10` is not in diabetes without complications, which lists `E10.0` and
//! four others. [`Classification::short_codes`] counts them. An ICD-8 code
//! matches no condition of a map with no ICD-8 codes, and
//! [`Classification::icd8_codes`] counts those too. Nothing is repaired.
//!
//! # References
//!
//! 1. Charlson ME, Pompei P, Ales KL, MacKenzie CR. A new method of
//!    classifying prognostic comorbidity in longitudinal studies: development
//!    and validation. J Chronic Dis. 1987;40(5):373-83.
//! 2. Quan H, Sundararajan V, Halfon P, et al. Coding algorithms for defining
//!    comorbidities in ICD-9-CM and ICD-10 administrative data. Med Care.
//!    2005;43(11):1130-9.
//! 3. Quan H, Li B, Couris CM, et al. Updating and validating the Charlson
//!    comorbidity index and score for risk adjustment in hospital discharge
//!    abstracts using data from 6 countries. Am J Epidemiol.
//!    2011;173(6):676-82.
//! 4. Elixhauser A, Steiner C, Harris DR, Coffey RM. Comorbidity measures for
//!    use with administrative data. Med Care. 1998;36(1):8-27.
//! 5. van Walraven C, Austin PC, Jennings A, Quan H, Forster AJ. A
//!    modification of the Elixhauser comorbidity measures into a point system
//!    for hospital death using administrative data. Med Care.
//!    2009;47(6):626-33.
//! 6. Gasparini A. comorbidity: An R package for computing comorbidity scores.
//!    J Open Source Softw. 2018;3(23):648.
//! 7. Thygesen SK, Christiansen CF, Christensen S, Lash TL, Sorensen HT. The
//!    predictive value of ICD-10 diagnostic coding used to assess Charlson
//!    comorbidity index conditions in the population-based Danish National
//!    Registry of Patients. BMC Med Res Methodol. 2011;11:83.
//! 8. Christensen S, Johansen MB, Christiansen CF, Jensen R, Lemeshow S.
//!    Comparison of Charlson comorbidity index with SAPS and APACHE scores for
//!    prediction of mortality following intensive care. Clin Epidemiol.
//!    2011;3:203-11.

mod denmark;
mod lists;

use crate::RegisterError;
use crate::codes::{CodeSet, EraCodeSet, Revision, revision_of};
use crate::events::{EventRows, Lookback};

/// One condition of a comorbidity map.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Condition {
    /// A short identifier, unique in the map.
    pub key: String,
    /// The readable name.
    pub label: String,
    /// The codes of the condition, by revision.
    pub codes: EraCodeSet,
    /// The keys of the milder conditions this one cancels when both are
    /// flagged.
    pub supersedes: Vec<String>,
}

/// A weight for each condition of a map, in the map's order.
#[derive(Clone, Debug, PartialEq)]
pub struct Weights {
    /// The name of the weighting.
    pub name: String,
    /// One weight per condition.
    pub value: Vec<f64>,
}

impl Weights {
    /// Weights with a name.
    #[must_use]
    pub fn new(name: impl Into<String>, value: Vec<f64>) -> Self {
        Self {
            name: name.into(),
            value,
        }
    }
}

/// The weights of Charlson et al. 1987 for the 17 conditions of the Quan coding
/// ([`ComorbidityMap::charlson_quan_icd10`]).
#[must_use]
pub fn charlson_original() -> Weights {
    Weights::new("Charlson 1987", lists::CHARLSON_1987.to_vec())
}

/// The updated weights of Quan et al. 2011 for the 17 conditions of the Quan
/// coding ([`ComorbidityMap::charlson_quan_icd10`]).
#[must_use]
pub fn charlson_quan2011() -> Weights {
    Weights::new("Quan 2011", lists::CHARLSON_QUAN_2011.to_vec())
}

/// The Charlson 1987 weights of the 19 conditions of
/// [`ComorbidityMap::charlson_denmark`], as Christensen et al. 2011 (Table A)
/// list them: 1 for the first ten conditions, 2 for hemiplegia, renal
/// disease, diabetes with end organ damage, any tumour, leukaemia and
/// lymphoma, 3 for moderate to severe liver disease and 6 for metastatic solid
/// tumour and AIDS.
#[must_use]
pub fn charlson_denmark_weights() -> Weights {
    Weights::new("Charlson 1987, Danish coding", denmark::weights())
}

/// The point weights of van Walraven et al. 2009 for the 31 conditions of
/// [`ComorbidityMap::elixhauser_quan_icd10`], some of them negative.
#[must_use]
pub fn elixhauser_van_walraven() -> Weights {
    Weights::new(
        "van Walraven 2009",
        lists::ELIXHAUSER_VAN_WALRAVEN_2009.to_vec(),
    )
}

/// The conditions of an index, at most 64, with the hierarchy between them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComorbidityMap {
    conditions: Vec<Condition>,
    /// For each condition, the mask of the conditions it cancels.
    cancels: Vec<u64>,
}

fn condition_from(spec: &lists::Spec, supersedes: &[&str]) -> Condition {
    Condition {
        key: spec.key.to_owned(),
        label: spec.label.to_owned(),
        codes: EraCodeSet {
            icd8: CodeSet::default(),
            icd10: CodeSet::from_prefixes(spec.prefixes.iter().copied())
                .unwrap_or_else(|_| unreachable!("the generated prefixes are valid")),
        },
        supersedes: supersedes.iter().map(|&s| s.to_owned()).collect(),
    }
}

impl ComorbidityMap {
    /// A map from conditions in the order of their bits.
    ///
    /// # Errors
    ///
    /// [`RegisterError::Input`] for more than 64 conditions, a key used twice, a
    /// `supersedes` key that names no condition or the condition itself.
    pub fn new(conditions: Vec<Condition>) -> Result<Self, RegisterError> {
        if conditions.len() > 64 {
            return Err(RegisterError::input(format!(
                "a comorbidity map holds at most 64 conditions, got {}",
                conditions.len()
            )));
        }
        let mut cancels = Vec::with_capacity(conditions.len());
        for (k, c) in conditions.iter().enumerate() {
            if conditions[..k].iter().any(|o| o.key == c.key) {
                return Err(RegisterError::input(format!(
                    "the condition {:?} is listed twice",
                    c.key
                )));
            }
            let mut mask = 0_u64;
            for key in &c.supersedes {
                let Some(j) = conditions.iter().position(|o| &o.key == key) else {
                    return Err(RegisterError::input(format!(
                        "{:?} supersedes {key:?}, which is not a condition of the map",
                        c.key
                    )));
                };
                if j == k {
                    return Err(RegisterError::input(format!(
                        "{:?} supersedes itself",
                        c.key
                    )));
                }
                mask |= 1 << j;
            }
            cancels.push(mask);
        }
        Ok(Self {
            conditions,
            cancels,
        })
    }

    /// The 17 conditions of the Charlson index in the ICD-10 coding of Quan et
    /// al. 2005 (Table 1), with no ICD-8 codes.
    #[must_use]
    pub fn charlson_quan_icd10() -> Self {
        let supersedes = |key: &str| -> &[&str] {
            match key {
                "msld" => &["mld"],
                "diabwc" => &["diab"],
                "metacanc" => &["canc"],
                _ => &[],
            }
        };
        let conditions = lists::CHARLSON_QUAN_ICD10
            .iter()
            .map(|s| condition_from(s, supersedes(s.key)))
            .collect();
        Self::new(conditions).unwrap_or_else(|_| unreachable!("the built-in map is valid"))
    }

    /// The 19 conditions of the Charlson index in the Danish coding: ICD-10
    /// codes of Thygesen et al. 2011 (Table 1) and ICD-8 codes of Christensen
    /// et al. 2011 (Table A). Moderate or severe liver disease cancels mild
    /// liver disease, diabetes with end organ damage cancels diabetes, and a
    /// metastatic solid tumour cancels any tumour; leukaemia and lymphoma are
    /// conditions of their own. Weights: [`charlson_denmark_weights`].
    #[must_use]
    pub fn charlson_denmark() -> Self {
        Self::new(denmark::conditions())
            .unwrap_or_else(|_| unreachable!("the built-in map is valid"))
    }

    /// The 31 conditions of the Elixhauser measure in the ICD-10 coding of Quan
    /// et al. 2005 (Table 2), with no ICD-8 codes.
    #[must_use]
    pub fn elixhauser_quan_icd10() -> Self {
        let supersedes = |key: &str| -> &[&str] {
            match key {
                "hypc" => &["hypunc"],
                "diabc" => &["diabunc"],
                "metacanc" => &["solidtum"],
                _ => &[],
            }
        };
        let conditions = lists::ELIXHAUSER_QUAN_ICD10
            .iter()
            .map(|s| condition_from(s, supersedes(s.key)))
            .collect();
        Self::new(conditions).unwrap_or_else(|_| unreachable!("the built-in map is valid"))
    }

    /// The conditions in the order of their bits.
    #[must_use]
    pub fn conditions(&self) -> &[Condition] {
        &self.conditions
    }

    fn position(&self, key: &str) -> Result<usize, RegisterError> {
        self.conditions
            .iter()
            .position(|c| c.key == key)
            .ok_or_else(|| RegisterError::input(format!("the map has no condition {key:?}")))
    }

    /// Replace the ICD-10 codes of a condition, for a Danish or study list.
    ///
    /// # Errors
    ///
    /// [`RegisterError::Input`] when no condition has the key.
    pub fn set_icd10(&mut self, key: &str, codes: CodeSet) -> Result<(), RegisterError> {
        let k = self.position(key)?;
        self.conditions[k].codes.icd10 = codes;
        Ok(())
    }

    /// Replace the ICD-8 codes of a condition, for the years before ICD-10.
    ///
    /// # Errors
    ///
    /// [`RegisterError::Input`] when no condition has the key.
    pub fn set_icd8(&mut self, key: &str, codes: CodeSet) -> Result<(), RegisterError> {
        let k = self.position(key)?;
        self.conditions[k].codes.icd8 = codes;
        Ok(())
    }

    /// The mask of one normalised code.
    #[must_use]
    pub fn mask(&self, normalized: &str) -> u64 {
        self.conditions
            .iter()
            .enumerate()
            .filter(|(_, c)| c.codes.contains(normalized))
            .fold(0, |mask, (k, _)| mask | (1 << k))
    }

    /// One mask per distinct normalised code, in the order given, with bit `k`
    /// set when the code is in condition `k`, and counts of what cannot be
    /// classified in full.
    #[must_use]
    pub fn classify(&self, distinct_codes: &[&str]) -> Classification {
        let longest = |era: Revision| {
            self.conditions
                .iter()
                .map(|c| match era {
                    Revision::Icd8 => c.codes.icd8.longest_item(),
                    Revision::Icd10 => c.codes.icd10.longest_item(),
                })
                .max()
                .unwrap_or(0)
        };
        let (longest8, longest10) = (longest(Revision::Icd8), longest(Revision::Icd10));
        let mut out = Classification {
            masks: Vec::with_capacity(distinct_codes.len()),
            short_codes: 0,
            icd8_codes: 0,
            unroutable_codes: 0,
        };
        for code in distinct_codes {
            out.masks.push(self.mask(code));
            match revision_of(code) {
                Some(Revision::Icd8) => {
                    out.icd8_codes += 1;
                    out.short_codes += usize::from(code.len() < longest8);
                }
                Some(Revision::Icd10) => out.short_codes += usize::from(code.len() < longest10),
                None => out.unroutable_codes += 1,
            }
        }
        out
    }

    /// The flags with the milder form of a condition removed where the more
    /// severe form is flagged.
    #[must_use]
    pub fn apply_hierarchy(&self, flags: u64) -> u64 {
        let mut cancelled = 0_u64;
        let mut rest = flags;
        while rest != 0 {
            let k = rest.trailing_zeros() as usize;
            rest &= rest - 1;
            cancelled |= self.cancels.get(k).copied().unwrap_or(0);
        }
        flags & !cancelled
    }

    /// The sum of the weights of the flagged conditions, in condition order.
    ///
    /// # Errors
    ///
    /// [`RegisterError::Input`] when the weights are not one per condition, or
    /// a flag is set beyond the conditions.
    pub fn score(&self, flags: u64, weights: &Weights) -> Result<f64, RegisterError> {
        if weights.value.len() != self.conditions.len() {
            return Err(RegisterError::input(format!(
                "{} weights for {} conditions",
                weights.value.len(),
                self.conditions.len()
            )));
        }
        if self.conditions.len() < 64 && flags >> self.conditions.len() != 0 {
            return Err(RegisterError::input("a flag is set beyond the conditions"));
        }
        let mut sum = 0.0;
        for (k, w) in weights.value.iter().enumerate() {
            if flags >> k & 1 == 1 {
                sum += w;
            }
        }
        Ok(sum)
    }
}

/// The classification of a dictionary of codes by a [`ComorbidityMap`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Classification {
    /// For each code, the mask of its conditions.
    pub masks: Vec<u64>,
    /// Codes shorter than the longest item of the lists of their revision,
    /// which cannot match every item. The caller reports the count.
    pub short_codes: usize,
    /// Codes that are ICD-8. They match only the ICD-8 sets of the map, and
    /// none of a map without ICD-8 codes.
    pub icd8_codes: usize,
    /// Codes that are of neither revision, and match nothing.
    pub unroutable_codes: usize,
}

/// The index dates of persons, in rows sorted by person.
#[derive(Clone, Copy, Debug)]
pub struct IndexTable<'a> {
    /// The person of each index, a position among the persons of
    /// [`EventRows::person_offsets`].
    pub person: &'a [u32],
    /// The index day of each index.
    pub day: &'a [i64],
    /// The first day the register could have recorded the person at each index.
    pub history_start: &'a [i64],
}

/// The flags found for one index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexFlags {
    /// The OR of the masks of the diagnoses in the window, before the hierarchy.
    pub flags: u64,
    /// Whether the window lies inside the recorded history.
    pub history_complete: bool,
}

impl IndexFlags {
    /// The score after the hierarchy. With `strict` set the score of an index
    /// whose window starts before the recorded history is [`None`], because a
    /// score of 0 there would mean that nothing was recorded rather than that
    /// nothing could be.
    ///
    /// # Errors
    ///
    /// As [`ComorbidityMap::score`].
    pub fn score(
        &self,
        map: &ComorbidityMap,
        weights: &Weights,
        strict: bool,
    ) -> Result<Option<f64>, RegisterError> {
        let value = map.score(map.apply_hierarchy(self.flags), weights)?;
        Ok((!strict || self.history_complete).then_some(value))
    }
}

/// The flags of each index: the OR of the masks of the person's diagnoses in
/// the window from the look-back start up to the index day.
///
/// Diagnoses before `history_start` are not used. With `include_index_day` a
/// diagnosis on the index day is in the window.
///
/// # Errors
///
/// [`RegisterError::Input`] when the columns differ in length, a person is out
/// of range, the rows of a person are not sorted by day, the look-back is
/// [`Lookback::None`] or negative, or the rows are not a CSR layout.
pub fn flags_at_index(
    rows: &EventRows<'_>,
    indices: &IndexTable<'_>,
    lookback: Lookback,
    include_index_day: bool,
) -> Result<Vec<IndexFlags>, RegisterError> {
    let n = indices.person.len();
    if indices.day.len() != n || indices.history_start.len() != n {
        return Err(RegisterError::input(
            "person, day and history_start must have one value per index",
        ));
    }
    if lookback == Lookback::None {
        return Err(RegisterError::input(
            "flags need a look-back window, not Lookback::None",
        ));
    }
    let offsets = rows.person_offsets;
    if offsets.is_empty()
        || offsets[0] != 0
        || offsets[offsets.len() - 1] != rows.day.len()
        || rows.mask.len() != rows.day.len()
        || offsets.windows(2).any(|w| w[0] > w[1])
    {
        return Err(RegisterError::input(
            "person_offsets must be a CSR layout over day and mask",
        ));
    }
    let persons = offsets.len() - 1;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let p = indices.person[i] as usize;
        if p >= persons {
            return Err(RegisterError::input(format!(
                "index {i} names person {p} of {persons}"
            )));
        }
        let (a, b) = (offsets[p], offsets[p + 1]);
        let days = &rows.day[a..b];
        if days.windows(2).any(|w| w[0] > w[1]) {
            return Err(RegisterError::input(format!(
                "the rows of person {p} are not sorted by day"
            )));
        }
        let (index, history) = (indices.day[i], indices.history_start[i]);
        let start = lookback.window_start(index, history)?.unwrap_or(history);
        let end = index + i64::from(include_index_day);
        let lo = days.partition_point(|&d| d < start.max(history));
        let hi = days.partition_point(|&d| d < end);
        let flags = rows.mask[a..b][lo..hi.max(lo)]
            .iter()
            .fold(0_u64, |acc, m| acc | m);
        out.push(IndexFlags {
            flags,
            history_complete: start >= history,
        });
    }
    Ok(out)
}
