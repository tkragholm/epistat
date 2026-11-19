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
