# Pooled effects of tests/data/matchthem.json under other complete-data
# degrees of freedom, from mice::pool.scalar().
#
# Reads the per-set estimates and variances that matchthem.R wrote, so the
# matching reference stays as it is. For the "within" approach, the pooled
# result for dfcom = n - k with n = 36, 120 and 1000 and k = 2, and for an
# infinite n.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/matchthem_df.R

suppressMessages({
  library(mice)
  library(jsonlite)
})

ref <- fromJSON("tests/data/matchthem.json", simplifyVector = FALSE)
a <- ref$approaches[[1]]
stopifnot(a$approach == "within")
q <- sapply(a$sets, function(s) s$estimate)
u <- sapply(a$sets, function(s) s$variance)
pool_at <- function(n) {
  p <- pool.scalar(q, u, n = n, k = 2)
  list(dfcom = n - 2, df = p$df, t = p$t, fmi = p$fmi)
}
out <- list(
  versions = list(r = R.version.string, mice = as.character(packageVersion("mice"))),
  cases = lapply(c(36, 120, 1000, Inf), pool_at)
)
write(toJSON(out, auto_unbox = TRUE, digits = 17), "tests/data/matchthem_df.json")
