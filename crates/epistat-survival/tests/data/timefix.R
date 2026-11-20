# survival::aeqSurv on right-censored and counting-process times: no ties, ties
# by rounding error near 0, 1, 1e4 and 1e7 (the absolute branch of the rule and
# the relative one), chains longer than the tolerance, intervals whose start
# equals the previous stop up to rounding, a custom tolerance, a collapse to
# zero length that aeqSurv rejects, and infinite times, which aeqSurv mishandles.
# For each case: the input, the tolerance, the output or the error message.
# Seventeen digits are written so that every double reads back exactly.
# Run: Rscript timefix.R > timefix.json
suppressMessages({
  library(jsonlite)
  library(survival)
})
tol0 <- sqrt(.Machine$double.eps)

run <- function(name, stop, start = NULL, tolerance = tol0, status = NULL) {
  n <- length(stop)
  if (is.null(status)) status <- rep(c(1L, 0L), length.out = n)
  x <- if (is.null(start)) Surv(stop, status) else Surv(start, stop, status)
  res <- tryCatch(
    if (identical(tolerance, tol0)) aeqSurv(x) else aeqSurv(x, tolerance = tolerance),
    error = function(e) conditionMessage(e)
  )
  out <- list(name = name, tolerance = tolerance, start = start, stop = stop)
  if (is.character(res)) {
    out$error <- res
  } else {
    m <- unclass(res)
    if (is.null(start)) {
      out$out_stop <- unname(m[, 1])
    } else {
      out$out_start <- unname(m[, 1])
      out$out_stop <- unname(m[, 2])
    }
    out$changed <- !identical(unname(m[, seq_len(ncol(m) - 1)]), unname(unclass(x)[, seq_len(ncol(m) - 1)]))
  }
  out
}

set.seed(20261120)
cases <- list()
add <- function(...) cases[[length(cases) + 1]] <<- run(...)

add("no_ties", c(1, 2.5, 3, 7.25, 10))
add("integers", c(5, 3, 3, 8, 13, 21, 3, 5))
# Absolute branch: values near zero, where the mean is small too.
add("near_zero", c(0, 1e-9, 5e-9, 1e-7, 2e-7, 2e-7 + 1e-9))
# Around 1: 1e-8 apart is tied, 3e-8 apart is not.
add("around_one", c(1, 1 + 1e-8, 1 + 4e-8, 1.5, 1.5 - 2e-9, 2))
# Around 1e4 and 1e7 the absolute difference exceeds the tolerance, so the
# relative branch decides.
add("around_1e4", c(10000, 10000 + 5e-5, 10000 + 3e-4, 10001, 10001 - 1e-5))
add("around_1e7", c(1e7, 1e7 + 0.1, 1e7 + 1, 1e7 + 5, 1e7 + 5 + 0.05))
# Mixed magnitudes: the mean is that of the pooled unique values.
add("mixed_magnitudes", c(0.5, 0.5 + 1e-9, 3000, 3000 + 2e-5, 3000 + 1e-3, 1e5, 1e5 + 1e-4))
# Floating point accumulation, as a date computed by repeated addition.
add("accumulated_tenths", c(cumsum(rep(0.1, 10)), seq(0.1, 1, by = 0.1)))
# A chain of nine links of 0.9 tol: 8 tol from end to end.
step <- 0.9 * tol0
add("chain_absolute", c(0, step * 1:9, 1, 2))
# A chain in the relative branch: links of 0.9 tol * 1e4.
rstep <- 0.9 * tol0 * 1e4
add("chain_relative", c(1e4 + rstep * 0:6, 2e4))
# Counting process: each start is the previous stop up to rounding.
k <- 1:12
add("counting_start_equals_stop", stop = cumsum(rep(0.1, 12)), start = (k - 1) * 0.1)
add("counting_integer_days", stop = c(10, 20, 35, 50), start = c(0, 10, 20, 35))
# Entry at the first stop of another row, with a tie among the stops.
add("counting_shared_values", stop = c(2, 2 + 1e-12, 5, 5), start = c(1, 1 + 1e-12, 2, 2 + 1e-12))
# A custom tolerance.
add("tolerance_1e-3", c(1, 1.0005, 1.01, 1.0102, 5), tolerance = 1e-3)
add("tolerance_zero", c(1, 1 + 1e-12, 2), tolerance = 0)
# A row whose start and stop differ by rounding error only.
add("zero_length_collapse", stop = c(1 + 1e-12, 3), start = c(1, 2))
# The collapse comes through a chain: stop and start are 1.8 tol apart, with a
# value between them from another row.
add("zero_length_chain", stop = c(1 + 1.8 * tol0, 1 + 0.9 * tol0, 9), start = c(1, 0, 3))
# A larger set: noisy copies of 200 dates, as start-stop intervals.
d <- sort(sample(1:5000, 200)) + 0.0
noise <- function(v) v + v * runif(length(v), -1, 1) * 1e-12
bounds <- c(0, d)
start <- noise(bounds[-201])
stop <- noise(d)
add("large_noisy_intervals", stop = stop, start = start)
add("large_noisy_right_censored", stop = noise(rep(d, 3)))

# Infinite times: aeqSurv ignores them when it looks for ties, then maps an
# infinite stop to the largest finite value and loses an infinite start.
# Recorded as text, since JSON has no infinity.
inf_stop <- tryCatch(
  suppressWarnings(unclass(aeqSurv(Surv(c(1, 1 + 1e-12, Inf, 3), c(1, 0, 1, 1))))[, 1]),
  error = function(e) conditionMessage(e))
inf_counting <- tryCatch(
  suppressWarnings(aeqSurv(Surv(c(0, 1, 1 + 1e-12, 3), c(1, 2, 3, Inf), c(1, 0, 1, 1)))),
  error = function(e) conditionMessage(e))
infinite <- list(
  right_censored_input = c("1", "1.000000000001", "Inf", "3"),
  right_censored_output = as.character(inf_stop),
  counting_error = inf_counting
)

cat(toJSON(list(
  r_version = R.version.string,
  survival_version = as.character(packageVersion("survival")),
  jsonlite_version = as.character(packageVersion("jsonlite")),
  long_double_bytes = .Machine$sizeof.longdouble,
  cases = cases, infinite = infinite
), digits = I(17), auto_unbox = TRUE, null = "null", na = "null"))
