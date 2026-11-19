# epistat-inference

Variances and intervals for estimators: from influence functions (plain and
cluster-robust, as R's `did` computes them), from sandwich estimators with or
without clusters, and from the multiplier and pairs cluster bootstraps, with
percentile, trimmed and bias-corrected intervals, and the joint Wald test.
Clusters are numbered in order of first appearance and every sum runs in a
fixed order, so results do not depend on hashing or on the number of threads.
