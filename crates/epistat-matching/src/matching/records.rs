use crate::types::{BalanceRecord, BaseRecord, CovariateRef, CovariateValue, RoleTransitionRecord};
use chrono::NaiveDate;
use std::collections::HashMap;
use std::sync::OnceLock;

/// A unit that can be matched: an identifier, a birth date and the values the constraints and distances read.
pub trait MatchingRecord: Send + Sync {
    /// Unique identifier for the record.
    fn id(&self) -> &str;
    /// The individual's birth date (used for caliper matching).
    fn birth_date(&self) -> NaiveDate;
    /// Map of categorical values used for exact strata matching.
    ///
    /// The matchers read one value at a time through [`stratum`](Self::stratum).
    fn strata(&self) -> &HashMap<String, String>;
    /// The value of the stratum `key`, `None` when the record has none.
    ///
    /// The default reads [`strata`](Self::strata). A record that keeps its
    /// strata elsewhere, as a [`Cohort`](crate::Cohort) row keeps them in
    /// columns, overrides it, and the matchers then never build the map.
    fn stratum(&self, key: &str) -> Option<&str> {
        self.strata().get(key).map(String::as_str)
    }
    /// A named numeric field, for [`crate::constraints::Caliper`] constraints.
    ///
    /// `strata()` carries categorical values for exact matching. This carries
    /// numeric ones, so a job with several numeric constraints, such as a
    /// child's birth date and a parent's birth year, can address each by name.
    ///
    /// `None` means "this record has no such value", and every caliper treats
    /// that as a refusal rather than a pass: an unverifiable constraint has not
    /// been satisfied. The default carries no numeric fields.
    ///
    /// A wrapper type must forward this, as it forwards `strata()`. Without
    /// the forward the code compiles and every pair is refused on a field the
    /// inner record carries, which looks like a caliper that is far too tight.
    /// `RoleTransitionRecord` and `BalanceRecord` forward it.
    fn numeric(&self, _name: &str) -> Option<f64> {
        None
    }
    /// Names the table of columns the record's numeric fields are read from,
    /// for records that are rows of a shared table. Two records that return
    /// the same number have their numeric fields at the same slots.
    ///
    /// A constraint that reads one field of many records finds the slot of its
    /// name once for every table, through [`numeric_slot`](Self::numeric_slot),
    /// and then reads each record with
    /// [`numeric_in_slot`](Self::numeric_in_slot). The default, `None`, has it
    /// read every record by name.
    fn numeric_table(&self) -> Option<usize> {
        None
    }
    /// The slot of the numeric field `name` in the record's table, `None` when
    /// the table has no such field. Only called when
    /// [`numeric_table`](Self::numeric_table) is `Some`.
    fn numeric_slot(&self, _name: &str) -> Option<usize> {
        None
    }
    /// The record's numeric field at `slot`, which
    /// [`numeric_slot`](Self::numeric_slot) returned for a record of the same
    /// table, `None` when the record has no value. Only called when
    /// [`numeric_table`](Self::numeric_table) is `Some`.
    fn numeric_in_slot(&self, _slot: usize) -> Option<f64> {
        None
    }
    /// Optional key for ensuring uniqueness (e.g., family ID).
    fn unique_key(&self) -> Option<&str>;
    /// Optional death date, read by [`MustBeAlive`](crate::constraints::MustBeAlive).
    fn death_date(&self) -> Option<NaiveDate> {
        None
    }
    /// Optional emigration date. If provided, used to derive residency at index
    /// (see [`ResidentAtIndexRecord`]): a record is non-resident from this date
    /// onward.
    fn emigration_date(&self) -> Option<NaiveDate> {
        None
    }
    /// Optional date from which the record is in the population at risk.
    ///
    /// Risk-set matching ([`MatchJob::new_transition`](crate::MatchJob::new_transition))
    /// takes a record into the risk set of a case only when its entry date is
    /// on or before the case's index date, so a record that enters on the
    /// index day is in the set, as in `Epi::ccwc` (`entry <= t`). Leaving is
    /// the opposite: a record that dies, emigrates or has its event on the
    /// index day is already out. With no entry date the record has been at risk
    /// from the start. Standard matching does not read this field.
    fn entry_date(&self) -> Option<NaiveDate> {
        None
    }
}

/// A record with the date of the event that makes a person a case.
pub trait RoleIndexedRecord: MatchingRecord {
    /// The date of the event, if the person has one.
    fn event_date(&self) -> Option<NaiveDate>;
}

/// Record type that can determine residency at a given index date.
pub trait ResidentAtIndexRecord: RoleIndexedRecord {
    /// Whether the person lives in the study area on `index_date`.
    fn is_resident_at(&self, index_date: NaiveDate) -> bool;
}

/// A record that carries covariates, for propensity models and balance reports.
pub trait CovariateRecord: MatchingRecord {
    /// Map of the covariates of the record.
    ///
    /// The propensity models and balance reports read one covariate at a time
    /// through [`covariate`](Self::covariate) and
    /// [`covariate_names`](Self::covariate_names).
    fn covariates(&self) -> &HashMap<String, CovariateValue>;
    /// The covariate `name`, `None` when the record does not carry it.
    ///
    /// The default reads [`covariates`](Self::covariates).
    fn covariate(&self, name: &str) -> Option<CovariateRef<'_>> {
        self.covariates().get(name).map(CovariateRef::from)
    }
    /// The names of the covariates the record carries, in no particular order.
    ///
    /// The default reads [`covariates`](Self::covariates).
    fn covariate_names(&self) -> impl Iterator<Item = &str> {
        self.covariates().keys().map(String::as_str)
    }
    /// Names the table of columns the record's covariates are read from, for
    /// records that are rows of a shared table. Two records that return the
    /// same number have their covariates at the same slots.
    ///
    /// A model that reads one covariate of many records finds the slot of
    /// its name once for every table, through
    /// [`covariate_slot`](Self::covariate_slot), and then reads each record
    /// with [`covariate_in_slot`](Self::covariate_in_slot). The default,
    /// `None`, has it read every record by name.
    fn covariate_table(&self) -> Option<usize> {
        None
    }
    /// The slot of the covariate `name` in the record's table, `None` when the
    /// table has no such covariate. Only called when
    /// [`covariate_table`](Self::covariate_table) is `Some`.
    fn covariate_slot(&self, _name: &str) -> Option<usize> {
        None
    }
    /// The record's covariate at `slot`, which [`covariate_slot`](Self::covariate_slot)
    /// returned for a record of the same table. Only called when
    /// [`covariate_table`](Self::covariate_table) is `Some`.
    fn covariate_in_slot(&self, _slot: usize) -> CovariateRef<'_> {
        CovariateRef::Missing
    }
}

impl MatchingRecord for BaseRecord {
    fn id(&self) -> &str {
        &self.id
    }

    fn birth_date(&self) -> NaiveDate {
        self.birth_date
    }

    fn strata(&self) -> &HashMap<String, String> {
        &self.strata
    }

    fn numeric(&self, name: &str) -> Option<f64> {
        self.numerics.get(name).copied()
    }

    fn unique_key(&self) -> Option<&str> {
        self.unique_key.as_deref()
    }

    fn death_date(&self) -> Option<NaiveDate> {
        self.death_date
    }

    fn emigration_date(&self) -> Option<NaiveDate> {
        self.emigration_date
    }

    fn entry_date(&self) -> Option<NaiveDate> {
        self.entry_date
    }
}

impl MatchingRecord for BalanceRecord {
    fn id(&self) -> &str {
        self.core.id.as_str()
    }

    fn birth_date(&self) -> NaiveDate {
        self.core.birth_date
    }

    fn strata(&self) -> &HashMap<String, String> {
        &self.core.strata
    }

    fn numeric(&self, name: &str) -> Option<f64> {
        self.core.numerics.get(name).copied()
    }

    fn unique_key(&self) -> Option<&str> {
        self.core.unique_key.as_deref()
    }

    fn death_date(&self) -> Option<NaiveDate> {
        self.core.death_date
    }

    fn emigration_date(&self) -> Option<NaiveDate> {
        self.core.emigration_date
    }

    fn entry_date(&self) -> Option<NaiveDate> {
        self.core.entry_date
    }
}

impl CovariateRecord for BalanceRecord {
    fn covariates(&self) -> &HashMap<String, CovariateValue> {
        &self.covariates
    }
}

impl<R: MatchingRecord> MatchingRecord for RoleTransitionRecord<R> {
    crate::delegate_matching_record!(record);
}

impl<R: MatchingRecord> RoleIndexedRecord for RoleTransitionRecord<R> {
    fn event_date(&self) -> Option<NaiveDate> {
        self.transition_date
    }
}

impl<R: MatchingRecord> ResidentAtIndexRecord for RoleTransitionRecord<R> {
    /// Resident at `index_date` unless an emigration date on or before it is known.
    ///
    /// Mirrors [`MatchJob::with_alive_check`](crate::matching::MatchJob::with_alive_check)
    /// semantics: a control with no emigration date is treated as resident, and
    /// one that emigrated strictly after the case index date is still resident.
    fn is_resident_at(&self, index_date: NaiveDate) -> bool {
        self.emigration_date()
            .is_none_or(|emigration| emigration > index_date)
    }
}

impl<R: CovariateRecord> CovariateRecord for RoleTransitionRecord<R> {
    delegate_covariate_record!(record);
}

/// A shared reference is a record of the same kind, so a matcher can take
/// `&R` in place of `R` and work on records it does not own, as sequential
/// trials do with the people of a panel.
impl<R: MatchingRecord + ?Sized> MatchingRecord for &R {
    crate::delegate_matching_record!(*);
}

impl<R: RoleIndexedRecord + ?Sized> RoleIndexedRecord for &R {
    fn event_date(&self) -> Option<NaiveDate> {
        (**self).event_date()
    }
}

impl<R: ResidentAtIndexRecord + ?Sized> ResidentAtIndexRecord for &R {
    fn is_resident_at(&self, index_date: NaiveDate) -> bool {
        (**self).is_resident_at(index_date)
    }
}

impl<R: CovariateRecord + ?Sized> CovariateRecord for &R {
    delegate_covariate_record!(*);
}

/// The slot of one column in each of the first two tables of columns a run of
/// records reads from, found once per table.
///
/// A third table is not cached, and its records are read by name.
#[derive(Default)]
pub(crate) struct SlotCache {
    resolved: [OnceLock<(usize, Option<usize>)>; 2],
}

/// What a [`SlotCache`] knows about a column in a table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CachedSlot {
    /// The column is at this slot.
    At(usize),
    /// The table has no such column.
    Absent,
    /// The cache holds two other tables, so the column is read by name.
    ByName,
}

impl SlotCache {
    /// The slot of the column in `table`, with `find` called to look it up
    /// the first time the table is met.
    pub(crate) fn slot(&self, table: usize, find: impl FnOnce() -> Option<usize>) -> CachedSlot {
        let mut find = Some(find);
        for cell in &self.resolved {
            let &(known, slot) =
                cell.get_or_init(|| (table, find.take().expect("a lookup runs at most once")()));
            if known == table {
                return slot.map_or(CachedSlot::Absent, CachedSlot::At);
            }
        }
        CachedSlot::ByName
    }
}

/// Reads one covariate from a run of records.
///
/// A record that says which table of columns it reads from has the column
/// found once for every table, so a covariate is then read by position and
/// not by name.
pub(crate) struct ColumnReader<'k> {
    key: &'k str,
    slots: SlotCache,
}

impl<'k> ColumnReader<'k> {
    pub(crate) fn new(key: &'k str) -> Self {
        Self {
            key,
            slots: SlotCache::default(),
        }
    }

    pub(crate) const fn key(&self) -> &'k str {
        self.key
    }

    pub(crate) fn read<'r, R: CovariateRecord>(
        &mut self,
        record: &'r R,
    ) -> Option<CovariateRef<'r>> {
        let Some(table) = record.covariate_table() else {
            return record.covariate(self.key);
        };
        match self.slots.slot(table, || record.covariate_slot(self.key)) {
            CachedSlot::At(slot) => Some(record.covariate_in_slot(slot)),
            CachedSlot::Absent => None,
            CachedSlot::ByName => record.covariate(self.key),
        }
    }
}

/// Tells which records of a run need their covariate names read. Records of
/// one table of columns carry the same names, so only the first record of each
/// table does; a record that names no table always does.
#[derive(Default)]
pub(crate) struct NameScan {
    tables: Vec<usize>,
}

impl NameScan {
    pub(crate) fn needs_names<R: CovariateRecord + ?Sized>(&mut self, record: &R) -> bool {
        let Some(table) = record.covariate_table() else {
            return true;
        };
        if self.tables.contains(&table) {
            return false;
        }
        self.tables.push(table);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::caliper_on_field;
    use crate::test_utils::date;
    use crate::{MatchJob, MatchRatio};
    use std::collections::BTreeSet;

    struct Wrapper<R> {
        inner: R,
    }

    impl<R: MatchingRecord> MatchingRecord for Wrapper<R> {
        crate::delegate_matching_record!(inner);
    }

    impl<R: MatchingRecord> RoleIndexedRecord for Wrapper<RoleTransitionRecord<R>> {
        fn event_date(&self) -> Option<NaiveDate> {
            self.inner.transition_date
        }
    }

    impl<R: MatchingRecord> ResidentAtIndexRecord for Wrapper<RoleTransitionRecord<R>> {
        fn is_resident_at(&self, index_date: NaiveDate) -> bool {
            self.emigration_date()
                .is_none_or(|emigration| emigration > index_date)
        }
    }

    /// Names of the `fn` items declared between the first `{` after `start`
    /// and the matching closing brace at column zero or four.
    fn fn_names(source: &str, start: &str, end: &str) -> BTreeSet<String> {
        let from = source.find(start).expect("block start present");
        let body = &source[from..];
        let to = body.find(end).expect("block end present");
        body[..to]
            .lines()
            .filter_map(|line| {
                let line = line.trim_start();
                let rest = line.strip_prefix("fn ")?;
                let name = rest.split(['(', '<']).next()?;
                Some(name.to_owned())
            })
            .collect()
    }

    #[test]
    fn delegate_macro_forwards_every_trait_method() {
        let trait_fns = fn_names(
            include_str!("records.rs"),
            "pub trait MatchingRecord",
            "pub trait RoleIndexedRecord",
        );
        let macro_fns = fn_names(
            include_str!("../lib.rs"),
            "macro_rules! delegate_matching_methods",
            "macro_rules! delegate_covariate_record",
        );
        assert!(!trait_fns.is_empty());
        assert_eq!(trait_fns, macro_fns);
    }

    #[test]
    fn delegate_macro_forwards_every_covariate_method() {
        let trait_fns = fn_names(
            include_str!("records.rs"),
            "pub trait CovariateRecord",
            "impl MatchingRecord for BaseRecord",
        );
        let macro_fns = fn_names(
            include_str!("../lib.rs"),
            "macro_rules! delegate_covariate_record",
            "pub mod abadie_imbens;",
        );
        assert!(!trait_fns.is_empty());
        assert_eq!(trait_fns, macro_fns);
    }

    #[test]
    fn wrapper_returns_what_its_inner_record_returns() {
        let inner = BaseRecord::new("r1", date(2010, 3, 4))
            .with_numeric("mother_birth_year", 1980.0)
            .with_death_date(date(2030, 1, 1))
            .with_emigration_date(date(2025, 6, 1))
            .with_entry_date(date(2011, 2, 3));
        let mut inner = inner;
        inner.strata.insert("sex".to_owned(), "F".to_owned());
        inner.unique_key = Some("family".to_owned());
        let wrapped = Wrapper {
            inner: inner.clone(),
        };

        assert_eq!(wrapped.id(), inner.id());
        assert_eq!(wrapped.birth_date(), inner.birth_date());
        assert_eq!(wrapped.strata(), inner.strata());
        assert_eq!(wrapped.unique_key(), Some("family"));
        assert_eq!(wrapped.death_date(), Some(date(2030, 1, 1)));
        assert_eq!(wrapped.emigration_date(), Some(date(2025, 6, 1)));
        assert_eq!(wrapped.entry_date(), Some(date(2011, 2, 3)));
        assert_eq!(wrapped.numeric("mother_birth_year"), Some(1980.0));
        assert_eq!(wrapped.numeric("absent"), None);
    }

    fn base(id: &str, mother: f64, emigrated: Option<NaiveDate>) -> BaseRecord {
        let record =
            BaseRecord::new(id, date(2010, 1, 1)).with_numeric("mother_birth_year", mother);
        match emigrated {
            Some(day) => record.with_emigration_date(day),
            None => record,
        }
    }

    #[test]
    fn matching_run_is_the_same_for_wrapper_and_inner_record() {
        let rows = [
            ("case", 1980.0, None, Some(date(2012, 1, 1))),
            ("far", 1996.0, None, None),
            ("gone", 1980.2, Some(date(2011, 1, 1)), None),
            ("near", 1980.5, Some(date(2015, 1, 1)), None),
        ];
        let inner: Vec<_> = rows
            .iter()
            .map(|&(id, mother, emigrated, transition)| {
                RoleTransitionRecord::from_record(base(id, mother, emigrated), transition)
            })
            .collect();
        let wrapped: Vec<_> = rows
            .iter()
            .map(|&(id, mother, emigrated, transition)| Wrapper {
                inner: RoleTransitionRecord::from_record(base(id, mother, emigrated), transition),
            })
            .collect();

        let ratio = MatchRatio::new(1).expect("non-zero ratio");
        let direct = MatchJob::new_transition(&inner, 42)
            .with_ratio(ratio)
            .with_constraints(vec![caliper_on_field("mother_birth_year", 1.0)])
            .with_resident_check()
            .run()
            .expect("a valid job");
        let through_macro = MatchJob::new_transition(&wrapped, 42)
            .with_ratio(ratio)
            .with_constraints(vec![caliper_on_field("mother_birth_year", 1.0)])
            .with_resident_check()
            .run()
            .expect("a valid job");

        assert_eq!(direct.pairs.len(), 1);
        assert_eq!(direct.pairs[0].control_id, "near");
        assert_eq!(through_macro.pairs.len(), direct.pairs.len());
        assert_eq!(
            through_macro.pairs[0].case_id.as_str(),
            direct.pairs[0].case_id.as_str()
        );
        assert_eq!(
            through_macro.pairs[0].control_id.as_str(),
            direct.pairs[0].control_id.as_str()
        );
        assert_eq!(through_macro.matched_cases, direct.matched_cases);
        assert_eq!(through_macro.unmatched_cases, direct.unmatched_cases);
    }

    #[test]
    fn column_reader_agrees_with_reading_by_name_over_four_tables() {
        use crate::cohort::Cohort;
        let birth = vec![date(2000, 1, 1); 3];
        let tables: Vec<Cohort> = vec![
            Cohort::builder(["a", "b", "c"], birth.clone())
                .numeric_covariate("x", [Some(1.0), None, Some(3.0)])
                .numeric_covariate("y", [Some(5.0), Some(6.0), Some(7.0)])
                .build()
                .unwrap(),
            Cohort::builder(["d", "e", "f"], birth.clone())
                .numeric_covariate("y", [Some(0.0), Some(1.0), Some(2.0)])
                .numeric_covariate("x", [Some(-1.0), Some(4.0), None])
                .build()
                .unwrap(),
            Cohort::builder(["g", "h", "i"], birth.clone())
                .numeric_covariate("z", [Some(0.0), Some(1.0), Some(2.0)])
                .build()
                .unwrap(),
            Cohort::builder(["j", "k", "l"], birth)
                .numeric_covariate("x", [Some(9.0), Some(8.0), None])
                .build()
                .unwrap(),
        ];
        let rows: Vec<_> = tables.iter().flat_map(Cohort::rows).collect();
        let mut reader = ColumnReader::new("x");
        for _ in 0..2 {
            for row in &rows {
                assert_eq!(reader.read(row), row.covariate("x"), "row {}", row.id());
            }
        }
    }
}
