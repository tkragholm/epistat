# epistat-survival

Cox regression on one covariate (Breslow and Efron ties, delayed entry, strata,
cluster-robust variance), the Fine-Gray model, Aalen-Johansen cumulative
incidence with a clustered bootstrap, multi-state occupation probabilities and
the Titman-Putter test of the Markov property.

Estimators take slices, bootstrap replicates are seeded from the caller's seeds
through `epistat-rng`, and results do not depend on the number of threads.
