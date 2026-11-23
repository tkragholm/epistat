# epistat-register

Small methods for register data on plain slices.

- `person_time`: day counts to and from civil dates, and the split of an
  interval of days across the calendar years it spans.
- `spans`: merging of overlapping intervals, with a grace, in one sweep.
- `relations`: connected components of rows that share a code in any of several
  key columns, numbered by first row.
- `cpi`: constant-price deflation with an annual price index the caller
  supplies. The Statistics Denmark index, 1900 to 2025, is available with the
  cargo feature `denmark`.
- `icd10`: normalization of ICD-10 diagnosis codes, including the Danish
  register prefix, and their 22 chapters. Serde support with the cargo feature
  `serde`.
- `age`: age in whole years at a date, by civil years or by days over 365.25.
- `codes`: sets of codes as prefixes and ranges (`I21, C00-C26, E10.2-E10.5`),
  classified once per distinct code, with ATC levels, ICD-10 levels, labelled
  blocks from a table the caller supplies, ICD-8 beside ICD-10 and a crosswalk
  structure that holds no data.
- `events`: first event in a window, washout and look-back, prior status with a
  separate state for a history that is too short, and confirmation by a second
  date or contact.
- `comorbidity`: the Charlson index (Quan 2005 coding, Charlson 1987 or Quan
  2011 weights) and the Elixhauser measure (Quan 2005 coding, van Walraven
  weights), checked against the R package `comorbidity`. National code lists
  are supplied by the caller.
- `exposure`: exposure periods from dispensings with a fixed, default or
  observed dose, grace period, stockpiling with a cap, paused days and
  cumulative dose, checked against `AdhereR`. No dose is built in.
- `intervals`: intersection, difference, complement within a window and
  clipping of per-key sets of half-open intervals, checked against day arrays
  in base R.
- `lexis`: cuts follow-up on integer days at calendar years, civil birthdays,
  periods and time since entry, checked against `Epi::splitLexis` and
  `popEpi::splitMulti`.
- `pedigree`: full and half siblings, cousins of any degree, multiple births,
  birth order, family clusters and the coefficient of relationship from the
  mother and father of each person, checked against `kinship2`. Parents of
  unknown paternity are a kind of their own.
- `residence`: residence spells from dated emigrations and immigrations, with
  the reading of a same-day pair named by the caller, residence on a day, the
  first departure after a day and the end of observation.
