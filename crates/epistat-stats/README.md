# epistat-stats

Sample quantiles (R's types 1 and 7), the normal distribution function and
its inverse, and floating-point sums in the order Python computes them.

The quantiles and the normal distribution function give the same doubles as
R; the normal quantile is within two units in the last place of R's `qnorm`.
Nothing uses fused multiply-add, so results are the same on every platform.
