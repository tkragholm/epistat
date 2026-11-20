# Weighted eQQ distances of MatchIt's summary.matchit(standardize = FALSE), on
# small random samples.
#
# MatchIt computes them in an internal function, qqsum(x, t, w), which this
# script calls directly. Each case is a variable, a group indicator and weights,
# and the script writes the mean and maximum quantile difference. The cases vary
# the sizes of the groups (equal and unequal, down to one unit), the number of
# distinct values (two to eight, so both the few-valued branch and the
# continuous one are used), ties, weights of 1, random positive weights, integer
# weights and zero weights. A variable with only the values 0 and 1 is left out:
# MatchIt reads it from the difference in proportions.
#
# The last element, `defect`, is a hand-made case in which MatchIt's few-valued
# branch reads the pooled maximum for the larger group: group 1 is 1 1 1 2 and
# group 0 is 2 3.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/matchit_eqq.R
suppressMessages({
  library(MatchIt)
  library(jsonlite)
})
set.seed(20261121)

make_case <- function() {
  repeat {
    n1 <- sample(1:12, 1)
    n0 <- if (runif(1) < 0.2) n1 else sample(1:12, 1)
    k <- sample(c(2:8, 40), 1)
    mode <- sample(c("int", "cont", "scaled"), 1)
    draw <- function(n) {
      v <- sample(seq_len(k), n, replace = TRUE)
      switch(mode, int = v, cont = round(v + rnorm(n, 0, 0.3), 2), scaled = v * 2.5 - 1)
    }
    x <- c(draw(n1), draw(n0))
    t <- c(rep(1, n1), rep(0, n0))
    wmode <- sample(c("one", "real", "int", "zero"), 1)
    w <- switch(wmode,
      one = rep(1, n1 + n0),
      real = round(runif(n1 + n0, 0.2, 3), 3),
      int = sample(1:4, n1 + n0, replace = TRUE),
      zero = ifelse(runif(n1 + n0) < 0.3, 0, round(runif(n1 + n0, 0.2, 3), 3)))
    pos1 <- sum(w[t == 1] > 0); pos0 <- sum(w[t == 0] > 0)
    if (pos1 < 1 || pos0 < 1) next
    if (all(x == 0 | x == 1)) next
    perm <- sample(length(x))
    x <- x[perm]; t <- t[perm]; w <- w[perm]
    r <- MatchIt:::qqsum(x, t, w, standardize = FALSE)
    return(list(x = x, group = t, weight = w, mean = unname(r["meandiff"]),
                max = unname(r["maxdiff"])))
  }
}
cases <- lapply(1:400, function(i) make_case())
dx <- c(1, 1, 1, 2, 2, 3); dt <- c(1, 1, 1, 1, 0, 0); dw <- rep(1, 6)
r <- MatchIt:::qqsum(dx, dt, dw, standardize = FALSE)
defect <- list(x = dx, group = dt, weight = dw, mean = unname(r["meandiff"]),
               max = unname(r["maxdiff"]))
out <- list(
  versions = list(r = R.version.string,
                  matchit = as.character(packageVersion("MatchIt")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  cases = cases, defect = defect)
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
write_json(out, file.path(dirname(script), "matchit_eqq.json"), digits = 17,
           na = "string", auto_unbox = TRUE)
