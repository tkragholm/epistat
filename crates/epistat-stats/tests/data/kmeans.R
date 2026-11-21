# Reference k-means partitions from stats::kmeans.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/kmeans.R
#
# Twenty simulated data sets of 240 points in 2 to 4 dimensions, drawn from
# 2 to 5 normal components with unit variance and centres about 3 apart, so
# that the clusters overlap a little. Each is clustered from k of its own rows
# as starting centres, chosen with a fixed seed, by algorithm = "Lloyd" and by
# the default "Hartigan-Wong", with iter.max = 1000. A run that R reports as an
# empty cluster or as exceeding its quick-transfer steps is left out.

suppressWarnings(suppressMessages(library(jsonlite)))

set.seed(20261121)
runs <- list()
for (r in 1:20) {
  d <- sample(2:4, 1)
  k <- sample(2:5, 1)
  n <- 240
  centres <- matrix(rnorm(k * d, sd = 2), k, d)
  comp <- sample.int(k, n, replace = TRUE)
  x <- centres[comp, , drop = FALSE] + matrix(rnorm(n * d), n, d)
  start_rows <- sample.int(n, k)
  start <- x[start_rows, , drop = FALSE]
  for (alg in c("Lloyd", "Hartigan-Wong")) {
    fit <- tryCatch(suppressWarnings(kmeans(x, centers = start, algorithm = alg, iter.max = 1000)),
                    error = function(e) NULL)
    if (is.null(fit)) next
    runs[[length(runs) + 1]] <- list(
      algorithm = alg, d = d, k = k,
      points = as.vector(t(x)), start = as.vector(t(start)),
      cluster = fit$cluster - 1L,
      centres = as.vector(t(fit$centers)),
      withinss = fit$withinss, size = fit$size, iter = fit$iter
    )
  }
}
out <- list(versions = list(r = R.version.string), runs = runs)
write(toJSON(out, auto_unbox = TRUE, digits = NA), "tests/data/kmeans.json")
cat(length(runs), "runs\n")
