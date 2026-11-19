# epistat-rng

`SplitMix64` streams and the bootstrap draws built on them: uniform and
bounded integers, shuffles, resample counts, Bayesian bootstrap weights and
Rademacher multiplier sums, and normal, gamma, chi-square, Beta, Wishart and
inverse Wishart draws.

The stream, the uniforms and the bounded integers use only integer arithmetic
and scaling by powers of two, so they are the same on every platform and can be
reproduced exactly in another language. Draws that call `ln` or `cos` agree
only as far as the platform's maths library does.
Replicate seeds come from `replicate_seed(base, r)`, so a replicate draws the
same values whichever thread runs it.
