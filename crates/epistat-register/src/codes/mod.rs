//! Sets of diagnosis and drug codes written as prefixes and ranges.
//!
//! Case definitions, comorbidity maps and drug groups are lists such as
//! `I21, I252, C00-C26, E10.2-E10.5, N02AA`. A code belongs to the set when it
//! starts with a listed prefix or falls in a listed range. [`CodeSet`] holds
//! such a list in a form that answers membership by binary search and supports
//! union, intersection and difference exactly. [`CodeMap`] holds up to 64
//! named sets and classifies the distinct codes of a column once, one bit per
//! set, so that classification costs the number of distinct codes and not the
//! number of rows. [`atc`], [`icd10_levels`] and [`CodeHierarchy`] give the
//! levels of a code, and [`Revision`], [`EraCodeSet`] and [`Crosswalk`] take
//! ICD-8 beside ICD-10.
//!
//! # Matching rule
//!
//! An item of length `L` is a range of `L`-character strings, `lo` to `hi`
//! inclusive in byte order, and a prefix is the range with `lo == hi`. A code
//! `c` matches the item when `len(c) >= L` and `lo <= c[..L] <= hi`. Codes are
//! upper-case ASCII letters and digits, as [`normalize_diagnosis_code`]
//! leaves an ICD-10 code, so byte order puts `C09` before `C10`. A dot in a
//! specification is ignored, so `E10.2` is `E102`. The Danish register prefix
//! `D` is not part of a specification: the caller normalises the codes first.
//!
//! [`normalize_diagnosis_code`]: crate::normalize_diagnosis_code
//!
//! # Representation and algorithm
//!
//! Codes are at most eight characters. A set is stored as one sorted list of
//! disjoint, non-adjacent intervals for each code length `l` from 1 to 8,
//! over the alphabet `0`-`9`, `A`-`Z`, where the list of length `l` holds the strings of exactly `l` characters
//! that the set contains, and the list of length 8 holds the strings of eight
//! or more characters, compared on their first eight. An item of length `L`
//! contributes to every length `l >= L` the interval `[lo, hi]` padded to `l`
//! characters with `0` on the left and `Z` on the right:
//! a string of `l` characters starts with a string in `[lo, hi]` exactly when
//! it lies in the padded interval. Membership is one binary search in the list
//! of the code's length. Union, intersection and difference are sweeps over the
//! eight pairs of lists, and the result is exact for codes of every length
//! because each length is kept apart. A set is empty exactly when all eight
//! lists are.
//!
//! The lists are canonical, so two sets that contain the same codes are
//! equal under `==` however they were built. A code with a character outside
//! the alphabet is in no set.
//!
//! The original items are kept for a set built by [`CodeSet::parse`],
//! [`CodeSet::from_prefixes`] or [`CodeSet::union`] of such sets, because
//! [`CodeMap::longest_match`] needs the length of the prefix that matched.
//! Intersection and difference give sets with no original items.
//!
//! # Traps
//!
//! A three-character code such as `E10` matches no four-character item such as
//! `E10.2`, and a classification of a column of such codes finds fewer cases
//! than the data holds. [`CodeMap::classify`] counts the codes shorter than
//! the longest item of the map and the caller reports the count. Nothing is
//! repaired. Danish codes with a trailing letter (`DI428A`, after
//! normalisation `I428A`) are longer than WHO's four characters, and the
//! prefix rule covers them.

mod hierarchy;
mod icd8;

pub mod atc;

use std::collections::BTreeMap;

use crate::RegisterError;

pub use hierarchy::{CodeHierarchy, Icd10Level, icd10_levels};
pub use icd8::{
    Coverage, Crosswalk, CrosswalkEntry, EraCodeSet, MapKind, Revision, normalize_any, revision_of,
};

/// The longest code a set tells apart. Longer codes are compared on their first
/// eight characters.
const MAX_LEN: usize = 8;

type Key = [u8; MAX_LEN];

/// An interval of keys of one width, `lo` to `hi` inclusive.
type Interval = (Key, Key);

/// One item of a specification: a range of `len`-character strings.
type Item = (u8, Key, Key);

/// The next character of the alphabet `0`-`9`, `A`-`Z`, or [`None`] after `Z`.
fn next_char(byte: u8) -> Option<u8> {
    match byte {
        b'9' => Some(b'A'),
        b'Z' => None,
        _ => Some(byte + 1),
    }
}

/// The previous character of the alphabet, or [`None`] before `0`.
fn previous_char(byte: u8) -> Option<u8> {
    match byte {
        b'A' => Some(b'9'),
        b'0' => None,
        _ => Some(byte - 1),
    }
}

/// The next key of `width` characters in the order of the alphabet, or
/// [`None`] after `ZZ..Z`.
fn successor(mut key: Key, width: usize) -> Option<Key> {
    for byte in key[..width].iter_mut().rev() {
        if let Some(next) = next_char(*byte) {
            *byte = next;
            return Some(key);
        }
        *byte = b'0';
    }
    None
}

/// The previous key of `width` characters, or [`None`] before `00..0`.
fn predecessor(mut key: Key, width: usize) -> Option<Key> {
    for byte in key[..width].iter_mut().rev() {
        if let Some(previous) = previous_char(*byte) {
            *byte = previous;
            return Some(key);
        }
        *byte = b'Z';
    }
    None
}

/// Sort and merge intervals that overlap or touch.
fn normalise(mut list: Vec<Interval>, width: usize) -> Vec<Interval> {
    list.sort_unstable();
    let mut out: Vec<Interval> = Vec::with_capacity(list.len());
    for (lo, hi) in list {
        if let Some(last) = out.last_mut() {
            let touches = last.1 >= lo || successor(last.1, width).is_some_and(|next| next >= lo);
            if touches {
                if hi > last.1 {
                    last.1 = hi;
                }
                continue;
            }
        }
        out.push((lo, hi));
    }
    out
}

fn contains_key(list: &[Interval], key: Key) -> bool {
    let at = list.partition_point(|(lo, _)| *lo <= key);
    at > 0 && key <= list[at - 1].1
}

fn intersect_lists(a: &[Interval], b: &[Interval]) -> Vec<Interval> {
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::new();
    while i < a.len() && j < b.len() {
        let lo = a[i].0.max(b[j].0);
        let hi = a[i].1.min(b[j].1);
        if lo <= hi {
            out.push((lo, hi));
        }
        if a[i].1 < b[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

fn subtract_lists(a: &[Interval], b: &[Interval], width: usize) -> Vec<Interval> {
    let mut out = Vec::new();
    for &(alo, ahi) in a {
        let mut cursor = alo;
        let mut j = b.partition_point(|&(_, bhi)| bhi < alo);
        let mut exhausted = false;
        while j < b.len() && b[j].0 <= ahi {
            let (blo, bhi) = b[j];
            if blo > cursor
                && let Some(end) = predecessor(blo, width)
            {
                out.push((cursor, end));
            }
            if bhi >= ahi {
                exhausted = true;
                break;
            }
            if let Some(next) = successor(bhi, width) {
                cursor = next;
            } else {
                exhausted = true;
                break;
            }
            j += 1;
        }
        if !exhausted {
            out.push((cursor, ahi));
        }
    }
    out
}

/// Validate and upper-case one end of an item, dropping dots.
fn clean_part(part: &str, whole: &str) -> Result<(Key, usize), RegisterError> {
    let mut key = [0_u8; MAX_LEN];
    let mut len = 0;
    for ch in part.chars().filter(|&c| c != '.') {
        if !ch.is_ascii_alphanumeric() {
            return Err(RegisterError::input(format!(
                "code item {whole:?} holds the character {ch:?}, which is neither a letter, a digit nor a dot"
            )));
        }
        if len == MAX_LEN {
            return Err(RegisterError::input(format!(
                "code item {whole:?} is longer than {MAX_LEN} characters"
            )));
        }
        key[len] = ch.to_ascii_uppercase() as u8;
        len += 1;
    }
    if len == 0 {
        return Err(RegisterError::input(format!(
            "code item {whole:?} has an empty code"
        )));
    }
    Ok((key, len))
}

/// A set of codes as the prefixes and ranges of a specification.
///
/// See the [module](self) for the matching rule and the representation.
#[derive(Clone, Debug, Default)]
pub struct CodeSet {
    /// `classes[l - 1]` holds the strings of exactly `l` characters in the
    /// set, the last list the strings of `MAX_LEN` or more.
    classes: [Vec<Interval>; MAX_LEN],
    /// The merged original items by length, when the set was built from items.
    by_len: Option<[Vec<Interval>; MAX_LEN]>,
    /// The length of the longest original item, 0 for an empty set.
    longest_item: usize,
}

impl PartialEq for CodeSet {
    fn eq(&self, other: &Self) -> bool {
        self.classes == other.classes
    }
}

impl Eq for CodeSet {}

impl CodeSet {
    fn from_items(items: Vec<Item>) -> Self {
        let mut by_len: [Vec<Interval>; MAX_LEN] = Default::default();
        let mut longest_item = 0;
        for (len, lo, hi) in items {
            let len = usize::from(len);
            longest_item = longest_item.max(len);
            by_len[len - 1].push((lo, hi));
        }
        for (i, list) in by_len.iter_mut().enumerate() {
            *list = normalise(std::mem::take(list), i + 1);
        }
        Self::from_by_len(by_len, longest_item)
    }

    fn from_by_len(by_len: [Vec<Interval>; MAX_LEN], longest_item: usize) -> Self {
        let mut classes: [Vec<Interval>; MAX_LEN] = Default::default();
        for (i, list) in by_len.iter().enumerate() {
            let len = i + 1;
            for &(lo, mut hi) in list {
                for width in len..=MAX_LEN {
                    // The padded interval of width `width`: where the item says
                    // nothing the left end takes `0`, the least character, and
                    // the right end `Z`, the greatest.
                    for byte in &mut hi[len..width] {
                        *byte = b'Z';
                    }
                    let mut lo = lo;
                    for byte in &mut lo[len..width] {
                        *byte = b'0';
                    }
                    classes[width - 1].push((lo, hi));
                }
            }
        }
        for (i, list) in classes.iter_mut().enumerate() {
            *list = normalise(std::mem::take(list), i + 1);
        }
        Self {
            classes,
            by_len: Some(by_len),
            longest_item,
        }
    }

    /// Parse a specification such as `"I21, I252, C00-C26, E10.2-E10.5"`.
    ///
    /// Whitespace and commas separate items, and dots are ignored. An item is
    /// a prefix, or two prefixes of the same length joined by a hyphen, for the
    /// range between them inclusive. Letters are upper-cased. An empty
    /// specification is the empty set.
    ///
    /// # Errors
    ///
    /// [`RegisterError::Input`] when an item holds a character other than a
    /// letter, a digit or a dot, is longer than 8 characters, has ends of
    /// different lengths or an end before its start, or has more than one
    /// hyphen.
    pub fn parse(spec: &str) -> Result<Self, RegisterError> {
        let mut items = Vec::new();
        for token in spec
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter(|t| !t.is_empty())
        {
            let mut parts = token.split('-');
            let first = parts.next().unwrap_or_default();
            let (lo, len) = clean_part(first, token)?;
            let hi = match parts.next() {
                None => lo,
                Some(second) => {
                    if parts.next().is_some() {
                        return Err(RegisterError::input(format!(
                            "code item {token:?} has more than one hyphen"
                        )));
                    }
                    let (hi, hi_len) = clean_part(second, token)?;
                    if hi_len != len {
                        return Err(RegisterError::input(format!(
                            "range {token:?} has ends of different lengths"
                        )));
                    }
                    if hi < lo {
                        return Err(RegisterError::input(format!(
                            "range {token:?} ends before it starts"
                        )));
                    }
                    hi
                }
            };
            items.push((u8::try_from(len).unwrap_or(u8::MAX), lo, hi));
        }
        Ok(Self::from_items(items))
    }

    /// The set of codes that start with any of the prefixes.
    ///
    /// # Errors
    ///
    /// [`RegisterError::Input`] when a prefix is empty, longer than 8
    /// characters or holds a character other than a letter, a digit or a dot.
    pub fn from_prefixes<'a>(
        prefixes: impl IntoIterator<Item = &'a str>,
    ) -> Result<Self, RegisterError> {
        let mut items = Vec::new();
        for prefix in prefixes {
            let (key, len) = clean_part(prefix, prefix)?;
            items.push((u8::try_from(len).unwrap_or(u8::MAX), key, key));
        }
        Ok(Self::from_items(items))
    }

    /// Whether `normalized` belongs to the set. The code must be upper case and
    /// without dots, as the normalisation functions leave it. An empty code
    /// belongs to no set.
    #[must_use]
    pub fn contains(&self, normalized: &str) -> bool {
        let bytes = normalized.as_bytes();
        if bytes.is_empty() {
            return false;
        }
        let width = bytes.len().min(MAX_LEN);
        if !bytes[..width]
            .iter()
            .all(|b| b.is_ascii_digit() || b.is_ascii_uppercase())
        {
            return false;
        }
        let mut key = [0_u8; MAX_LEN];
        key[..width].copy_from_slice(&bytes[..width]);
        contains_key(&self.classes[width - 1], key)
    }

    /// Whether the set holds no code.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.classes.iter().all(Vec::is_empty)
    }

    /// The length of the longest item of the specification, 0 for an empty
    /// set. A code shorter than this cannot match every item.
    #[must_use]
    pub fn longest_item(&self) -> usize {
        self.longest_item
    }

    /// The codes that are in either set.
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        let longest = self.longest_item.max(other.longest_item);
        if let (Some(a), Some(b)) = (&self.by_len, &other.by_len) {
            let mut merged: [Vec<Interval>; MAX_LEN] = Default::default();
            for (i, list) in merged.iter_mut().enumerate() {
                let mut both = a[i].clone();
                both.extend_from_slice(&b[i]);
                *list = normalise(both, i + 1);
            }
            return Self::from_by_len(merged, longest);
        }
        let mut classes: [Vec<Interval>; MAX_LEN] = Default::default();
        for (i, list) in classes.iter_mut().enumerate() {
            let mut both = self.classes[i].clone();
            both.extend_from_slice(&other.classes[i]);
            *list = normalise(both, i + 1);
        }
        Self {
            classes,
            by_len: None,
            longest_item: longest,
        }
    }

    /// The codes that are in both sets.
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Self {
        let mut classes: [Vec<Interval>; MAX_LEN] = Default::default();
        for (i, list) in classes.iter_mut().enumerate() {
            *list = intersect_lists(&self.classes[i], &other.classes[i]);
        }
        Self {
            classes,
            by_len: None,
            longest_item: self.longest_item.max(other.longest_item),
        }
    }

    /// The codes that are in this set and not in `other`, for exclusions.
    #[must_use]
    pub fn difference(&self, other: &Self) -> Self {
        let mut classes: [Vec<Interval>; MAX_LEN] = Default::default();
        for (i, list) in classes.iter_mut().enumerate() {
            *list = subtract_lists(&self.classes[i], &other.classes[i], i + 1);
        }
        Self {
            classes,
            by_len: None,
            longest_item: self.longest_item.max(other.longest_item),
        }
    }

    /// The length of the longest original item that `normalized` matches, or
    /// [`None`] when none matches or the set has no original items (it came
    /// from an intersection or a difference).
    fn longest_prefix(&self, normalized: &str) -> Option<usize> {
        let by_len = self.by_len.as_ref()?;
        let bytes = normalized.as_bytes();
        (1..=bytes.len().min(MAX_LEN)).rev().find(|&len| {
            let mut key = [0_u8; MAX_LEN];
            key[..len].copy_from_slice(&bytes[..len]);
            contains_key(&by_len[len - 1], key)
        })
    }

    fn has_items(&self) -> bool {
        self.by_len.is_some()
    }
}

/// The result of classifying a dictionary of codes with a [`CodeMap`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Classification {
    /// For each code, a mask with bit `k` set when the code is in set `k`.
    pub masks: Vec<u64>,
    /// How many of the codes are shorter than the longest item of the map and
    /// so cannot match every item of it. The caller reports the count.
    pub short_codes: usize,
}

/// Up to 64 named code sets, classified together with one bit per set.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CodeMap {
    sets: Vec<(String, CodeSet)>,
}

impl CodeMap {
    /// A map of named sets. Set `k` is bit `k` of the masks.
    ///
    /// # Errors
    ///
    /// [`RegisterError::Input`] for more than 64 sets or a name used twice.
    pub fn new(sets: Vec<(String, CodeSet)>) -> Result<Self, RegisterError> {
        if sets.len() > 64 {
            return Err(RegisterError::input(format!(
                "a code map holds at most 64 sets, got {}",
                sets.len()
            )));
        }
        for (k, (name, _)) in sets.iter().enumerate() {
            if sets[..k].iter().any(|(other, _)| other == name) {
                return Err(RegisterError::input(format!(
                    "the code map names the set {name:?} twice"
                )));
            }
        }
        Ok(Self { sets })
    }

    /// The names of the sets, in bit order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.sets.iter().map(|(name, _)| name.as_str())
    }

    /// The number of sets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sets.len()
    }

    /// Whether the map has no sets.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sets.is_empty()
    }

    /// The mask of one code, bit `k` set when the code is in set `k`.
    #[must_use]
    pub fn mask(&self, normalized: &str) -> u64 {
        self.sets
            .iter()
            .enumerate()
            .filter(|(_, (_, set))| set.contains(normalized))
            .fold(0, |mask, (k, _)| mask | (1_u64 << k))
    }

    /// One mask per distinct code, in the order given. The codes must be
    /// normalised. Cost is the number of codes times the number of sets times
    /// a binary search.
    #[must_use]
    pub fn classify(&self, distinct_codes: &[&str]) -> Classification {
        let longest = self
            .sets
            .iter()
            .map(|(_, set)| set.longest_item())
            .max()
            .unwrap_or(0);
        Classification {
            masks: distinct_codes.iter().map(|c| self.mask(c)).collect(),
            short_codes: distinct_codes.iter().filter(|c| c.len() < longest).count(),
        }
    }

    /// For each code, the index of the set whose original item matched the
    /// longest prefix of the code, or [`None`] when no set matches. This is
    /// the assignment of an ATC code to the most specific group that names it.
    ///
    /// # Errors
    ///
    /// [`RegisterError::Input`] when a set has no original items (it came from
    /// an intersection or a difference) or when a code matches two sets with
    /// prefixes of the same length.
    pub fn longest_match(
        &self,
        distinct_codes: &[&str],
    ) -> Result<Vec<Option<u32>>, RegisterError> {
        if let Some((name, _)) = self.sets.iter().find(|(_, set)| !set.has_items()) {
            return Err(RegisterError::input(format!(
                "the set {name:?} has no original items, so a longest prefix cannot be read from it"
            )));
        }
        let mut out = Vec::with_capacity(distinct_codes.len());
        for code in distinct_codes {
            let mut best: Option<(usize, usize)> = None;
            for (k, (name, set)) in self.sets.iter().enumerate() {
                let Some(len) = set.longest_prefix(code) else {
                    continue;
                };
                match best {
                    Some((best_len, _)) if len < best_len => {}
                    Some((best_len, other)) if len == best_len => {
                        return Err(RegisterError::input(format!(
                            "code {code:?} matches the sets {:?} and {name:?} with prefixes of the same length",
                            self.sets[other].0
                        )));
                    }
                    _ => best = Some((len, k)),
                }
            }
            out.push(best.map(|(_, k)| u32::try_from(k).unwrap_or(u32::MAX)));
        }
        Ok(out)
    }
}

/// The distinct values of a column of codes and, for each row, the position of
/// its value among them.
///
/// Classify the distinct values once and [gather](Self::gather) the result back
/// to the rows. The distinct values are in byte order, so the result does not
/// depend on the order of the rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dictionary<'a> {
    distinct: Vec<&'a str>,
    row_code: Vec<u32>,
}

impl<'a> Dictionary<'a> {
    /// The dictionary of a column.
    ///
    /// # Errors
    ///
    /// [`RegisterError::Input`] for more distinct codes than a `u32` holds.
    pub fn new(rows: &[&'a str]) -> Result<Self, RegisterError> {
        let mut seen: BTreeMap<&'a str, u32> = BTreeMap::new();
        for &code in rows {
            seen.entry(code).or_insert(0);
        }
        let distinct: Vec<&'a str> = seen.keys().copied().collect();
        if u32::try_from(distinct.len()).is_err() {
            return Err(RegisterError::input(
                "more distinct codes than a UInt32 holds",
            ));
        }
        for (i, value) in seen.values_mut().enumerate() {
            *value = u32::try_from(i).unwrap_or(u32::MAX);
        }
        let row_code = rows.iter().map(|code| seen[code]).collect();
        Ok(Self { distinct, row_code })
    }

    /// The distinct codes in byte order.
    #[must_use]
    pub fn distinct(&self) -> &[&'a str] {
        &self.distinct
    }

    /// For each row, the position of its code in [`distinct`](Self::distinct).
    #[must_use]
    pub fn row_code(&self) -> &[u32] {
        &self.row_code
    }

    /// One value per row from one value per distinct code.
    ///
    /// # Panics
    ///
    /// When `per_distinct` is shorter than the number of distinct codes.
    #[must_use]
    pub fn gather<T: Copy>(&self, per_distinct: &[T]) -> Vec<T> {
        self.row_code
            .iter()
            .map(|&i| per_distinct[i as usize])
            .collect()
    }
}

#[cfg(test)]
mod tests;
