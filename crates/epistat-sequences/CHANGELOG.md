# Changelog

## [0.3.0] - unreleased

This release gives the crate one typed `Error`, returns `None` where a measure is undefined, adds `fit_typology`, and refuses weights that are negative or not finite.

### Breaking changes

- `Error` replaces `SequenceError`. The string variant `Invalid` is gone and the refusals have their own variants: `Ragged`, `NoRestarts`, `Types`, `NoPositiveWeight` and `UnknownDistance`. A panel without length or alphabet, and a panel of more than 2^32 sequences, are `Error::Input(InputError)` with the shared codes. Match on `Error` variants instead of the message.
- Weights that are negative or not finite are refused with `Error::Input`. `null_sequences`, `transition_costs`, `substitution_costs` and the position-wise and optimal-matching fits checked only the number of weights before.
- An undefined silhouette is `None`. `silhouette` returns `Result<Option<f64>, Error>` where it returned `f64`, which was `0.0` for fewer than two clusters or nothing to score. `Typology::silhouette` is an `Option<f64>`.
- `Agreement::adjusted_rand` and `Agreement::share_agreeing` are `Option<f64>`. Both are `None` for empty input or zero total weight, and `adjusted_rand` is `None` when the index is undefined, for example when both partitions are a single type. `partition_agreement` refuses a negative or NaN weight.
- `refit_agreements` returns `Vec<AgreementPair>`, with `AgreementPair = Option<(Option<f64>, Option<f64>)>`. The outer `None` means the replicate had too few sequences of positive weight to be refitted.
- Cluster labels are `u32` in `silhouette`, `partition_agreement`, `refit_agreements` and `Agreement::correspondence`. Convert `i64` labels with `u32::try_from`.
- `FitSettings` holds `restarts: usize` instead of a closure of the seed, so it has no lifetime and is `Copy`. Each fit draws its restarts with `Restarts::from_seed(seed, restarts)`. Explicit seed lists remain available on the k-medoids functions through `Restarts::from_seeds`. The `RestartSeeds` type is removed.
- `null_silhouettes` takes `bootstrap: Bootstrap` in place of a seed and a replicate count. Write `Bootstrap::new(replicates, seed)`. Replicate `r` draws from `replicate_seed(seed, r)`, as before.
- `transition_costs`, `substitution_costs` and `indel_for` return `Result<_, Error>` and check the number of weights, the symbol range and the table size, where they indexed out of range or truncated.
- The fields of `Panel` are read through `count()`, `length()` and `alphabet()`.
- `Attempt` and `Agreement` are `#[non_exhaustive]`.
- `best_attempt`, `rank_labels` and `weighted_average` are no longer public.
- The minimum supported Rust version is 1.89.

### Added

- `fit_typology(panel, weights, settings, seed)` derives the costs, fits and scores in one call and returns a `Typology` with the `Attempt` and the silhouette.
- `Error` implements `Classify`, with kinds and stable `sequences.*` codes.

### Changed

- K-medoids is fitted once over a distance trait, and eight-sequence blocks of equal length are aligned together, with the same results on any number of threads.

### Deprecated

- `SequenceError` (use `Error`). Deprecated items are removed in the next release that breaks the API.

### Fixed

- `positionwise::kmedoids` did not terminate when a weight was NaN. Weights are now checked first.

### Results

No released result changes. Where a measure was defined, its value is the same as in 0.2.0. The silhouette of fewer than two clusters is now `None` rather than `0.0`.

## [0.2.0] - 2026-11-19

- Restarts are given by the caller as seeds (`Restarts::from_seed`, `Restarts::from_seeds`), and degenerate fits return `None` rather than panicking.
- Multi-start helpers from `epistat-optim` are used for the k-medoids restarts.

## [0.1.0] - 2026-11-19

- First release: optimal-matching and Hamming distances, weighted k-medoids, silhouette width, four null models, partition agreement and replicate loops for null silhouettes and refit agreement.
