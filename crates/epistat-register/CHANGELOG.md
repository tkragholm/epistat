# Changelog

## [0.2.0] - unreleased

The crate grows from person-time, spans and deflation into the main tools for register data: code sets, event definitions, comorbidity indices, drug exposure, Lexis splitting, pedigrees and residence spells. It also has one error type with stable codes.

### Breaking changes

- `RegisterError` is replaced by `epistat_register::Error`, a non-exhaustive enum. `Error::Input` wraps `epistat_core::InputError` where it held a string, and the other failures have named variants with codes starting `register.`. It is no longer `Eq`. `connected_components`, `merge_sorted_spans`, `MergedSpans::bounds` and the `CpiTable` methods return `Error`. Code that matched the message text reads `code()` through `epistat_core::Classify` instead.
- `MergedSpans`, `NormalizedDiagnosis` and `YearOverlap` are `#[non_exhaustive]`. Read their fields and do not build them with a struct literal.
- `person_time::days_from_civil`, `civil_from_days` and `year_from_days`, and `age::LeapBirthday`, are now re-exports of the functions and the type in `epistat_core::day`. Calls and paths that name them keep working, and the results are the same.
- The crate depends on `epistat-core`, `epistat-optim` and `epistat-stats`, and its minimum supported Rust version is 1.89, up from 1.85.
- `YearOverlap.calendar_year` is an `i64`, as the day counts are, where it was an `i32` that a year beyond its range turned into `i32::MAX`. Read it as `i64` or convert with `i32::try_from`.
- `age::birthday_on_age` takes and returns an `epistat_core::Day`, so the epoch of the date is part of its type: write `birthday_on_age(Day::from_unix_days(birth)?, age, rule)` and read `.unix_days()` for the count.

### Added

- `codes`, sets of codes written as prefixes and ranges (`CodeSet`), with classification of the distinct codes of a column, the levels of ATC and ICD-10, ICD-8 beside ICD-10, and crosswalks between them. A trailing `.x` marks a category prefix and an en dash joins a range as a hyphen does.
- `events`, the first event of a definition in a window, whether a person was already a case before it, and whether a second record confirms it.
- `comorbidity`, the Charlson index and the Elixhauser measure from diagnosis codes with look-back windows and hierarchy rules, using the ICD-10 coding of Quan et al. 2005 and the Danish ICD-8 and ICD-10 Charlson coding. The weights are those of Charlson 1987, Quan 2011 and van Walraven 2009.
- `exposure`, dispensings turned into periods of drug exposure and cumulative doses under assumptions the caller declares.
- `intervals`, intersection, difference, complement and clipping of per-key sets of half-open intervals.
- `lexis`, follow-up cut on integer days at calendar years, civil birthdays, periods and times since entry.
- `pedigree`, siblings, cousins, multiple births, family clusters and relatedness from the mother and father of each person.
- `residence`, residence spells and the end of observation from dated emigrations and immigrations.
- `wtd`, the parametric waiting time distribution and the duration of a prescription as a percentile of the time between redemptions, and `pre2dup`, drug use periods from each person's own pattern of purchases by the PRE2DUP method.
- `age::birthday_on_age`, and `person_time::add_months` and `add_years`.
- `MergedSpans::bounds`, the start and end of each merged span.
- `Error`, with `CodeItemProblem`, `CrosswalkProblem` and `PedigreeProblem`, which implements `Classify`.
- `CpiTable::try_new`, which refuses an index that is not finite and positive and a year given twice with `InputError::Domain` at its row, where `CpiTable::new` keeps a zero index and lets the later of two rows win; `CpiTable::years`, the first and last year of the table; and `CpiTable::denmark` with the feature `denmark`.
- `age::age_at`, the age on a `Day` of a person born on a `Day`, beside `age_in_years` on day counts; `person_time::day_from_sas_seconds`, the `Day` and time of day of a SAS datetime; and `Day` re-exported at the crate root. SAS dates enter through `Day::from_sas_days` and `Day::from_sas_column`.
- `wtd::WtdFit` implements `epistat_core::Tidy`: its coefficient table names the parameters on the scale of estimation, without standard errors since the fit computes no information matrix, and its glance reports the log-likelihood, AIC and convergence.

### Deprecated

- `RegisterError`, an alias of `Error`, use `Error`. Deprecated items are removed in the next release that breaks the API.
- `person_time::sas_days_to_unix`, `unix_days_to_sas`, `days_from_civil_sas`, `civil_from_days_sas` and `civil_from_sas_seconds`, the SAS date functions of 0.1 on bare `i64`, whose results a caller could pass where a Unix day is read: use `epistat_core::Day::from_sas_days`, `Day::sas_days`, `Day::from_civil` and `day_from_sas_seconds`. `sas_seconds_to_unix`, `unix_seconds_to_sas` and the two offsets stay.

### Results

No released result changes.

## [0.1.2] - 2026-11-19

- Added the SAS date epoch beside the Unix epoch in `person_time`, with conversions from SAS days and seconds.

## [0.1.1] - 2026-11-19

- Added `icd10`, which normalises ICD-10 diagnosis codes, including the Danish register prefix, and classifies them into the 22 chapters.
- Added `age`, age in whole years from a birth date under a named rule for the leap-day birthday.

## [0.1.0] - 2026-11-19

- First release: person-time split across calendar years, merging of overlapping intervals, connected components of rows that share a key, and deflation to constant prices with the `denmark` consumer price index feature.
