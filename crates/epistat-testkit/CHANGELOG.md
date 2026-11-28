# Changelog

## [0.1.2] - unreleased

The crate gains checks for bad input and for error codes, a small random generator, and a helper that records the hash of a test case. No existing item changes, except that the error checks inherit core's stricter `Classify`.

### Added

- `BadInput`, five kinds of invalid column (such as an empty one or one with a non-finite value), with `BadInput::column` to build the column from a good one and `BadInput::code` for the error code each kind should give.
- `assert_input_error`, `assert_input_error_code` and `assert_rejects_bad_input`, which check that a call returns an input error, optionally with a given code. `assert_rejects_bad_input_with_codes` checks the code of every kind, with the mapping as an argument.
- `assert_codes_unique`, which checks that the codes of a crate's errors are distinct.
- `assert_codes_documented`, which checks that each error's code is written in the doc comment of its variant.
- `assert_close_to`, which checks one value against a `Tolerance` and names it when it fails.
- `SplitMix64`, a small deterministic generator for test data, with `unit`, `uniform` and `below`.
- `hash_out` and `hash_out_words`, which write the hash of a case to the file named by `EPISTAT_HASH_OUT`, and `RECORDED_PLATFORM`, which says whether the recorded hashes apply to the machine running the test.
- The crate depends on `epistat-core`, and reference JSON is parsed with exact float round trip.

### Changed

- The error checks take an error that implements `epistat_core::Classify`, which requires `std::error::Error + Send + Sync + 'static`, so a test's own error type implements `Display` and `Error`.

### Results

No released result changes.

## [0.1.1] - 2026-11-22

- Added pool sizes, absent values and worst deviations to the comparison helpers.
- NaN is checked in every comparison with an R reference, and the rule that left it unchecked is removed.

## [0.1.0] - 2026-11-19

- First release: tolerances, bit fingerprints, thread-count checks and R reference files.
