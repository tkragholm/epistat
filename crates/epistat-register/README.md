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
  structure that holds no data. Sets, ATC levels, the routing by revision and
  the crosswalk checks are checked against the rules written out in base R.
- `events`: first event in a window, washout and look-back, prior status with a
  separate state for a history that is too short, and confirmation by a second
  date or contact.
- `comorbidity`: the Charlson index (Quan 2005 coding, Charlson 1987 or Quan
  2011 weights) and the Elixhauser measure (Quan 2005 coding, van Walraven
  weights), checked against the R package `comorbidity`, and the Danish
  Charlson coding in ICD-8 and ICD-10 (Thygesen 2011, Christensen 2011),
  checked against `heaven::charlsonIndex`. Other lists are supplied by the
  caller.
- `exposure`: exposure periods from dispensings with a fixed, default or
  observed dose, grace period, stockpiling with a cap, paused days and
  cumulative dose, checked against `AdhereR` and against base R for the dose
  rules, exposure on a day and both cumulative doses. A period ends on the
  first whole day with no supply left. No dose is built in.
- `wtd`: the parametric waiting time distribution of Støvring et al. (2016),
  log-normal or Weibull, and the duration of a prescription as a percentile
  of the time between redemptions, checked against its likelihood written
  out in R.
- `pre2dup`: drug use periods by the PRE2DUP method of Tanskanen et al.
  (2015), from each person's sliding average dose, with hospital stays and
  caller-supplied group and package parameters, checked against the method
  written out in R.
- `intervals`: intersection, difference, complement within a window and
  clipping of per-key sets of half-open intervals, checked against day arrays
  in base R, and the conversions from residence spells, closed intervals with
  open ends, merged spans and exposure periods, and to the `(tstart, tstop]`
  rows and switches of `tmerge`.
- `lexis`: cuts follow-up on integer days at calendar years, civil birthdays,
  periods and time since entry, and labels each piece with its band on every
  source, checked against `Epi::splitLexis`, `Epi::timeBand` and
  `popEpi::splitMulti`.
- `pedigree`: full and half siblings, cousins of any degree, multiple births
  and birth order within a tolerance of days, family clusters of sibships or
  of whole pedigrees, and the coefficient of relationship from the mother and
  father of each person, checked against `kinship2`. Parents of unknown
  paternity are a kind of their own.
- `residence`: residence spells from dated emigrations and immigrations, with
  the reading of a same-day pair named by the caller, residence on a day, the
  first departure after a day and the end of observation.
