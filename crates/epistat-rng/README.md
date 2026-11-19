# epistat-rng

`SplitMix64` streams and the bootstrap draws built on them: uniform and
bounded integers, shuffles, resample counts, Bayesian bootstrap weights and
Rademacher multiplier sums.

Each draw uses only integer arithmetic and scaling by powers of two, so it is
the same on every platform and can be reproduced exactly in another language.
Replicate seeds come from `replicate_seed(base, r)`, so a replicate draws the
same values whichever thread runs it.
