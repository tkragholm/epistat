# Quantiles of the truncated standard normal from TruncatedNormal::norminvp,
# the function HonestDiD's conditional tests call.
#
# Run from the crate root:
#   Rscript tests/data/generate_truncated_normal_quantile_reference.R
#
# The intervals lie far in a tail, where the normal distribution function
# underflows (the first two are truncation regions of HonestDiD's conditional
# test on its BCdata_EventStudy example under smoothness), and on moderate
# tails where it does not.

library(TruncatedNormal)
library(jsonlite)

# Versions: written with R 4.6.2, TruncatedNormal 2.3 and jsonlite 2.0.0.
stopifnot(packageVersion("TruncatedNormal") == "2.3",
          packageVersion("jsonlite") == "2.0.0")
cat(R.version.string, "\n")

cases <- list(
  c(0.9975885170607642, -65.60229789667807, -41.62083760979225),
  c(0.9547738693467336, -79.00385324319012, -45.06644080412651),
  c(0.3, 41.6, 65.6),
  c(0.5, 38, 39),
  c(0.95, -1000, -40),
  c(0.5, 40, Inf),
  c(0.1, -Inf, -50),
  c(0.999, 37.5, 37.50001),
  c(0.7, -30, -20),
  c(0.2, 20, 30)
)
rows <- lapply(cases, function(case) {
  list(p = case[1], l = case[2], u = case[3],
       quantile = norminvp(case[1], case[2], case[3]))
})
# jsonlite writes an infinite bound as a string, so it is written as the
# largest double, which has the same quantile.
rows <- lapply(rows, function(row) {
  row$l <- max(row$l, -.Machine$double.xmax)
  row$u <- min(row$u, .Machine$double.xmax)
  row
})
write_json(rows, "tests/data/truncated_normal_quantile_ref.json", digits = 17,
           auto_unbox = TRUE)
cat("truncated_normal_quantile_ref.json written.\n")
