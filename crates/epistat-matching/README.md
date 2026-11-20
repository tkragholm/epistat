# epistat-matching

Matched cohorts for register studies. Anchors are matched to candidates
greedily under calipers and constraints, with or without replacement and at a
fixed or falling ratio; or, in a longitudinal cohort, people are matched at the
time they become cases to those still at risk (risk-set matching). Constraints
cover sex, birth date windows, exact strata, same family, alive and resident
at the index date, and any numeric field. Propensity scores come from
`epistat-glm`, plain or penalised, and Mahalanobis distances from a pooled
covariance. The output has the pairs, unit weights for the ATT, ATC or ATE, a
typed reason for every excluded candidate, estimand drift diagnostics, and
balance before and after matching: standardised mean differences, variance
ratios, eCDF and eQQ distances and Cramér's V.

Seeds come from the caller, and results do not depend on the number of
threads. Golden hashes pin the matched sets.

```rust,ignore
use epistat_matching::prelude::*;

let outcome = MatchJob::new_standard(&anchors, &candidates, 42)
    .with_ratio(MatchRatio::new(4).expect("non-zero ratio"))
    .with_birth_window(BirthDateWindowDays::new(30).expect("non-negative window"))
    .with_gender_match()
    .with_exact_matches(["region"])
    .run();

let risk_sets = MatchJob::new_transition(&cohort, 42)
    .with_ratio(MatchRatio::new(1).expect("non-zero ratio"))
    .with_age_limit(AgeLimitYears::new(6).expect("positive age limit"))
    .with_alive_check()
    .with_resident_check()
    .run();
```

The `reporting` feature writes the matching summary and the balance tables
as CSV.

This crate continues `cohort_matching` 0.8.0, whose repository keeps the
earlier history.
