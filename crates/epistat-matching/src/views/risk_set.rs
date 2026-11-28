//! [`RiskSetCohort`]: one row per person of a register cohort, with the
//! dates risk sets are made of, the input of risk-set matching.

use std::collections::HashMap;

use chrono::NaiveDate;
use epistat_core::Day;

use super::{
    Col, ColumnKind, Columns, ExactMissing, Label, LabelColumn, Pending, length, naive,
    naive_column,
};
use crate::Error;
use crate::cohort::Cohort;
use crate::constraints::ResidenceSpell;
use crate::role_transition::IdOrder;

/// Whether the `to` day of a residence spell is a day of residence.
///
/// There is no default: registers differ, and a wrong reading moves every
/// spell end by a day.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpellEnd {
    /// `to` is the first day abroad, not a day of residence.
    Exclusive,
    /// `to` is the last day of residence, as `epistat_register`'s spells
    /// have it.
    Inclusive,
}

/// Where the people of a cohort live: periods of residence by person, or the
/// date each one emigrated.
///
/// The columns are read when the residence is made, and a problem with them
/// is returned when the cohort is built.
#[derive(Debug)]
pub struct Residence {
    repr: Result<Repr, Error>,
}

#[derive(Debug)]
enum Repr {
    /// Spells by the text of their person's label, with exclusive ends.
    Spells {
        person: Vec<String>,
        from: Vec<Option<Day>>,
        to: Vec<Option<Day>>,
    },
    UntilEmigration(Vec<Option<Day>>),
}

impl Residence {
    /// Periods of residence, one row per spell: the person, by the same
    /// labels as the cohort's ids, and the first and the last day, `to` read
    /// as `end` says. A null `from` is a spell that began before every date
    /// and a null `to` one that has not ended. A person with no spell is
    /// resident on every date; a person with spells, on the dates they
    /// cover.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "a column is passed as the call builds it, Col::unix_days(..), not by reference"
    )]
    pub fn spells<L: Label>(
        person: &[L],
        from: Col<'_, Day>,
        to: Col<'_, Day>,
        end: SpellEnd,
    ) -> Self {
        let read = || -> Result<Repr, Error> {
            let n = person.len();
            let from = super::read_days("residence from", &from, n)?;
            let to = super::read_days("residence to", &to, n)?;
            let to = to
                .into_iter()
                .enumerate()
                .map(|(row, day)| match (end, day) {
                    (SpellEnd::Inclusive, Some(day)) => day
                        .checked_add_days(1)
                        .map(Some)
                        .ok_or_else(|| Error::DateOutOfRange {
                            column: "residence to".to_string(),
                            row,
                        }),
                    (_, day) => Ok(day),
                })
                .collect::<Result<Vec<_>, _>>()?;
            for (row, (a, b)) in from.iter().zip(&to).enumerate() {
                // With the end made exclusive, a spell from after its last
                // day has `to < from`, and one of a single day `to = from + 1`.
                let after = match end {
                    SpellEnd::Exclusive => matches!((a, b), (Some(a), Some(b)) if a > b),
                    SpellEnd::Inclusive => matches!((a, b), (Some(a), Some(b)) if a >= b),
                };
                if after {
                    return Err(Error::SpellOrder {
                        person: person[row].to_string(),
                        row,
                    });
                }
            }
            Ok(Repr::Spells {
                person: person.iter().map(ToString::to_string).collect(),
                from,
                to,
            })
        };
        Self { repr: read() }
    }

    /// Resident until the emigration date, which is not a day of residence;
    /// a null is a person who has not emigrated. One row per person of the
    /// cohort.
    #[must_use]
    #[expect(
        clippy::needless_pass_by_value,
        reason = "a column is passed as the call builds it, Col::unix_days(..), not by reference"
    )]
    pub fn until_emigration(dates: Col<'_, Day>) -> Self {
        let n = dates.len();
        Self {
            repr: super::read_days("emigration date", &dates, n).map(Repr::UntilEmigration),
        }
    }

    /// The periods of each row of a cohort whose ids have the texts `ids`:
    /// `None` for a person resident on every date.
    fn periods(self, ids: &[String]) -> Result<Vec<Option<Vec<ResidenceSpell>>>, Error> {
        match self.repr? {
            Repr::UntilEmigration(dates) => {
                if dates.len() != ids.len() {
                    return Err(length("emigration date", dates.len(), ids.len()));
                }
                dates
                    .iter()
                    .enumerate()
                    .map(|(row, day)| {
                        day.map(|day| {
                            Ok(vec![ResidenceSpell {
                                from: None,
                                to: Some(naive("emigration date", row, day)?),
                            }])
                        })
                        .transpose()
                    })
                    .collect()
            }
            Repr::Spells { person, from, to } => {
                let row_of: HashMap<&str, usize> = ids
                    .iter()
                    .enumerate()
                    .map(|(row, id)| (id.as_str(), row))
                    .collect();
                let mut periods: Vec<Option<Vec<ResidenceSpell>>> = vec![None; ids.len()];
                for (spell, label) in person.iter().enumerate() {
                    let row =
                        *row_of
                            .get(label.as_str())
                            .ok_or_else(|| Error::UnknownSpellPerson {
                                person: label.clone(),
                                row: spell,
                            })?;
                    let date = |column: &str, day: Option<Day>| {
                        day.map(|day| naive(column, spell, day)).transpose()
                    };
                    periods[row]
                        .get_or_insert_with(Vec::new)
                        .push(ResidenceSpell {
                            from: date("residence from", from[spell])?,
                            to: date("residence to", to[spell])?,
                        });
                }
                Ok(periods)
            }
        }
    }
}

/// One row per person of a register cohort, with the dates risk sets are
/// made of: birth, the index date that makes a case, the end of time at risk
/// as a comparator, entry, death and residence.
///
/// A person whose index date is set and who may be a case is a case on that
/// date. Everyone is a comparator from entry up to but not including their
/// own index date, their `comparator_until` date and their death, on the
/// days they are resident. The ids are any [`Label`] and are coded once, in
/// the sorted order of the labels, which is the order in which the matcher
/// takes cases on one date and lists each pool.
///
/// ```
/// use epistat_matching::{Col, Residence, RiskSetCohort, SpellEnd};
///
/// let ids = ["p1", "p2", "p3", "p4"];
/// // Days since 1970-01-01, as a register gives them.
/// let birth = [14_610_i64, 14_620, 14_640, 14_600];
/// let diagnosis = [Some(16_436_i64), None, None, Some(17_000)];
/// let sex = ["F", "F", "M", "F"];
/// let spell_person = ["p2"];
/// let resident_from = [Some(14_620_i64)];
/// let resident_to = [Some(16_000_i64)];
///
/// let cohort = RiskSetCohort::builder(&ids, Col::unix_days(&birth[..]))
///     .index_date(Col::unix_days_nullable(&diagnosis[..]))
///     .residence(Residence::spells(
///         &spell_person,
///         Col::unix_days_nullable(&resident_from[..]),
///         Col::unix_days_nullable(&resident_to[..]),
///         SpellEnd::Exclusive,
///     ))
///     .categorical("sex", &sex[..])
///     .build()?;
/// assert_eq!((cohort.len(), cohort.n_cases()), (4, 2));
/// assert!(cohort.has_residence());
/// # Ok::<(), epistat_matching::Error>(())
/// ```
#[derive(Debug)]
pub struct RiskSetCohort {
    columns: Columns,
    can_be_case: Vec<bool>,
    comparator_until: Option<Vec<Option<NaiveDate>>>,
    residence: Option<Vec<Option<Vec<ResidenceSpell>>>>,
}

impl RiskSetCohort {
    /// Start the cohort from the ids and the birth dates. Birth is required:
    /// the age limit and a birth window read it.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "a column is passed as the call builds it, Col::unix_days(..), not by reference"
    )]
    pub fn builder<'a, L: Label>(
        ids: &'a [L],
        birth_date: Col<'a, Day>,
    ) -> RiskSetCohortBuilder<'a> {
        let mut pending = Pending::new(ids);
        let birth = pending.own_dates("birth date", &birth_date);
        RiskSetCohortBuilder {
            pending,
            birth,
            index: None,
            can_be_case: None,
            comparator_until: None,
            entry: None,
            death: None,
            residence: None,
        }
    }

    /// The number of people.
    #[must_use]
    pub fn len(&self) -> usize {
        self.can_be_case.len()
    }

    /// Whether the cohort has no one, which `build` refuses.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.can_be_case.is_empty()
    }

    /// The number of cases: people with an index date who may be a case.
    #[must_use]
    pub fn n_cases(&self) -> usize {
        self.columns
            .cohort
            .risk_set_rows()
            .iter()
            .zip(&self.can_be_case)
            .filter(|(row, can)| **can && row.transition_date.is_some())
            .count()
    }

    /// Whether the cohort has residence spells or emigration dates.
    #[must_use]
    pub const fn has_residence(&self) -> bool {
        self.residence.is_some()
    }

    /// The named columns and their kinds, in the order they were added.
    pub fn columns(&self) -> impl Iterator<Item = (&str, ColumnKind)> {
        self.columns
            .named
            .iter()
            .map(|column| (column.name.as_str(), column.kind))
    }

    /// The kind of the column `name`.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownColumn`], naming the columns the cohort has, when it
    /// has none called `name`.
    pub fn column_kind(&self, name: &str) -> Result<ColumnKind, Error> {
        self.columns.column(name).map(|column| column.kind)
    }

    /// Checks that the column `name` can be an exact key: it is categorical,
    /// and it has no null unless `missing` makes a null a level of its own.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownColumn`] when there is no such column,
    /// [`Error::WrongColumnKind`] when it is not categorical and
    /// [`Error::NullKey`] at its first null under [`ExactMissing::Refuse`].
    pub fn check_exact_key(&self, name: &str, missing: ExactMissing) -> Result<(), Error> {
        self.columns.check_exact_key(name, missing)
    }

    /// The engine's cohort, with the index date as the event date.
    pub(crate) const fn cohort(&self) -> &Cohort {
        &self.columns.cohort
    }

    /// The order of the people's ids.
    pub(crate) const fn id_order(&self) -> &IdOrder {
        &self.columns.order
    }

    /// Whether each person's index date makes them a case.
    pub(crate) fn can_be_case(&self) -> &[bool] {
        &self.can_be_case
    }

    /// The first day each person is no longer a comparator, when given.
    pub(crate) fn comparator_until(&self) -> Option<&[Option<NaiveDate>]> {
        self.comparator_until.as_deref()
    }

    /// The residence periods of each person, `None` for one resident on
    /// every date, when the cohort has residence.
    pub(crate) fn residence(&self) -> Option<&[Option<Vec<ResidenceSpell>>]> {
        self.residence.as_deref()
    }
}

/// Builds a [`RiskSetCohort`] column by column, from
/// [`RiskSetCohort::builder`].
///
/// Each column is read and checked when it is added. The first problem is
/// returned by [`build`](Self::build).
#[must_use]
pub struct RiskSetCohortBuilder<'a> {
    pending: Pending,
    birth: Vec<Option<Day>>,
    index: Option<Vec<Option<Day>>>,
    can_be_case: Option<&'a [bool]>,
    comparator_until: Option<Vec<Option<Day>>>,
    entry: Option<Vec<Option<Day>>>,
    death: Option<Vec<Option<Day>>>,
    residence: Option<Residence>,
}

impl std::fmt::Debug for RiskSetCohortBuilder<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RiskSetCohortBuilder")
            .field("rows", &self.pending.len())
            .field("error", &self.pending.error)
            .finish_non_exhaustive()
    }
}

impl<'a> RiskSetCohortBuilder<'a> {
    /// The date a person becomes a case. A null: never a case, at risk as a
    /// comparator throughout.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "a column is passed as the call builds it, Col::unix_days(..), not by reference"
    )]
    pub fn index_date(mut self, dates: Col<'a, Day>) -> Self {
        self.index = Some(self.pending.own_dates("index date", &dates));
        self
    }

    /// Whether an index date makes the person a case. Where it is false the
    /// index date only ends the person's time at risk as a comparator.
    /// Without it every index date makes a case.
    pub const fn can_be_case(mut self, flags: &'a [bool]) -> Self {
        self.can_be_case = Some(flags);
        self
    }

    /// The first day a person is no longer a comparator, whatever else
    /// holds. A null: no such day.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "a column is passed as the call builds it, Col::unix_days(..), not by reference"
    )]
    pub fn comparator_until(mut self, dates: Col<'a, Day>) -> Self {
        self.comparator_until = Some(self.pending.own_dates("comparator until", &dates));
        self
    }

    /// The day a person enters the cohort, `entry` of `Epi::ccwc()`. A null:
    /// at risk from birth.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "a column is passed as the call builds it, Col::unix_days(..), not by reference"
    )]
    pub fn entry_date(mut self, dates: Col<'a, Day>) -> Self {
        self.entry = Some(self.pending.own_dates("entry date", &dates));
        self
    }

    /// The date of death. A null: alive throughout.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "a column is passed as the call builds it, Col::unix_days(..), not by reference"
    )]
    pub fn death_date(mut self, dates: Col<'a, Day>) -> Self {
        self.death = Some(self.pending.own_dates("death date", &dates));
        self
    }

    /// Residence spells or emigration dates; see [`Residence`].
    pub fn residence(mut self, residence: Residence) -> Self {
        self.residence = Some(residence);
        self
    }

    /// A categorical column: an exact key of a pass, or a covariate.
    pub fn categorical(mut self, name: &str, values: impl LabelColumn<'a>) -> Self {
        self.pending.categorical(name, &values.into_col());
        self
    }

    /// A numeric column: a caliper field, a score or a covariate. A null is
    /// a missing value; NaN and infinities are refused.
    pub fn numeric(mut self, name: &str, values: impl Into<Col<'a, f64>>) -> Self {
        self.pending.numeric(name, &values.into());
        self
    }

    /// A date column, for a constraint within days.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "a column is passed as the call builds it, Col::unix_days(..), not by reference"
    )]
    pub fn date(mut self, name: &str, values: Col<'a, Day>) -> Self {
        self.pending.date(name, &values);
        self
    }

    /// The cohort.
    ///
    /// # Errors
    ///
    /// The first problem met: no ids ([`Error::Input`]), an id on two rows
    /// ([`Error::DuplicateIdRows`]), two labels that print alike
    /// ([`Error::AmbiguousLabel`]), a column of another length than the ids
    /// ([`Error::ColumnLength`]), two columns of one name
    /// ([`Error::DuplicateColumn`]), a number that is not finite
    /// ([`Error::NonFiniteValue`]), a date out of range
    /// ([`Error::DateOutOfRange`]), a null birth date ([`Error::NullValue`]),
    /// an index date before birth ([`Error::IndexBeforeBirth`]), a spell of a
    /// person not in the cohort ([`Error::UnknownSpellPerson`]) or a spell
    /// that starts after it ends ([`Error::SpellOrder`]).
    pub fn build(mut self) -> Result<RiskSetCohort, Error> {
        let n = self.pending.len();
        if let Some(flags) = self.can_be_case
            && flags.len() != n
        {
            self.pending.fail(length("can be case", flags.len(), n));
        }
        if let Some(error) = self.pending.error.take() {
            return Err(error);
        }
        let birth = self
            .birth
            .iter()
            .enumerate()
            .map(|(row, day)| {
                let day = day.ok_or_else(|| Error::NullValue {
                    column: "birth date".to_string(),
                    row,
                })?;
                naive("birth date", row, day)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(index) = &self.index {
            for (row, (day, born)) in index.iter().zip(&self.birth).enumerate() {
                if let (Some(day), Some(born)) = (day, born)
                    && day < born
                {
                    return Err(Error::IndexBeforeBirth {
                        id: self.pending.ids[row].clone(),
                        row,
                    });
                }
            }
        }
        let dates = |name: &str, days: Option<&Vec<Option<Day>>>| {
            days.map(|days| naive_column(name, days)).transpose()
        };
        let index = dates("index date", self.index.as_ref())?;
        let entry = dates("entry date", self.entry.as_ref())?;
        let death = dates("death date", self.death.as_ref())?;
        let comparator_until = dates("comparator until", self.comparator_until.as_ref())?;
        let residence = self
            .residence
            .map(|residence| residence.periods(&self.pending.ids))
            .transpose()?;
        let can_be_case = self
            .can_be_case
            .map_or_else(|| vec![true; n], <[bool]>::to_vec);
        let columns = self.pending.build(birth, |mut builder| {
            if let Some(index) = index {
                builder = builder.event_date(index);
            }
            if let Some(entry) = entry {
                builder = builder.entry_date(entry);
            }
            if let Some(death) = death {
                builder = builder.death_date(death);
            }
            builder
        })?;
        Ok(RiskSetCohort {
            columns,
            can_be_case,
            comparator_until,
            residence,
        })
    }
}
