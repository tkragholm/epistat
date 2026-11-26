# epistat-epi

Epidemiological measures and sensitivity analyses: rates with exact
intervals, rate ratios (with an exact interval) and differences, directly
standardised rates with Fay-Feuer intervals, the ratio and difference of two
standardised rates, Mantel-Haenszel estimators with the Breslow-Day test,
tests of a common rate ratio over strata, standardised incidence ratios with
exact, Byar, Wald and profile intervals and tests of homogeneity and trend,
the weights of direct standardisation, the Kitagawa decomposition with the
variances of its parts, E-values (checked against `EValue`), quantitative
bias analysis (misclassification, selection, an unmeasured confounder and
misclassified person-time, simple and probabilistic, checked against
`episensr`), Rosenbaum bounds (checked against `rbounds`, `sensitivitymv`,
`sensitivitymw` and `sensitivity2x2xk`) and covariate balance statistics.

The distribution functions, quantiles and sums these rest on are in
`epistat-stats`. Nothing uses fused multiply-add, so results are the same on
every platform.
