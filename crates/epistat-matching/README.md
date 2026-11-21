# epistat-matching

Matching for cohort studies. The methods follow MatchIt where it has them:
nearest-neighbour, optimal, full, subclass, exact, coarsened exact, quick,
genetic and cardinality matching, with propensity scores, Mahalanobis and
Euclidean distances, calipers, common support and `mahvars`. Anchors can also
be matched greedily under calipers and constraints on sex, birth date, exact
strata, family, alive and resident at the index date, and any numeric field.

Longitudinal cohorts are matched in risk sets at the time a person becomes a
case, by exposure density, or across sequential trials. The output has the
pairs or subclasses, matching weights for the ATT, ATC or ATE, a typed reason
for every excluded candidate, balance before and after matching, effect
estimates with matched-set bootstrap and Abadie-Imbens variances, and
Rosenbaum bounds and E-values.

Seeds come from the caller, and results do not depend on the number of
threads. Golden hashes pin the matched sets.

A typed builder, `MatchJob`, suits Rust code. A serde `MatchSpec` and
`run_spec` describe any method as plain values, for callers that read their
options at run time.

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

The `arrow` feature builds records from Apache Arrow arrays. The `reporting`
feature writes the matching summary and the balance tables as CSV.
