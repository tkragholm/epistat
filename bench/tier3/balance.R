# The R side of the tier 3 benchmarks of entropy balancing and CBPS: one process
# runs one method on one size and writes the numbers the Rust side writes, so that
# ../tier1/compare.R can check that both solved the same problem. The data file
# comes from the Rust program (crates/epistat-causal/benches/tier3_balance.rs).
#
# Usage: Rscript balance.R N METHOD SCRATCH
# Environment: OUT (results, default SCRATCH/out) and REPS (timed repetitions,
# default 3).
suppressMessages({library(data.table); library(WeightIt); library(CBPS)})
args <- commandArgs(TRUE)
n <- as.integer(args[1]); method <- args[2]; scratch <- args[3]
out <- Sys.getenv("OUT", file.path(scratch, "out"))
reps <- as.integer(Sys.getenv("REPS", "3"))
dir.create(out, showWarnings = FALSE, recursive = TRUE)
d <- fread(file.path(scratch, sprintf("bal_%d.csv", n)))
for (j in names(d)) if (is.integer(d[[j]])) set(d, j = j, value = as.numeric(d[[j]]))
setDTthreads(0)
timed <- function(f) {
  times <- numeric(reps)
  for (i in seq_len(reps)) {
    t0 <- proc.time()[["elapsed"]]
    value <- f()
    times[i] <- proc.time()[["elapsed"]] - t0
  }
  list(times = times, value = value)
}
wls_diff <- function(w) {
  z <- cbind(1 - d$a, d$a)
  b <- solve(crossprod(z, w * z), crossprod(z, w * d$y))
  b[2] - b[1]
}
summarise <- function(w) c(sum_w = sum(w), sum_w_square = sum(w^2), max_w = max(w), difference = wls_diff(w))
if (method %in% c("ebal_att", "ebal_ate_m2")) {
  est <- if (method == "ebal_att") "ATT" else "ATE"; mom <- if (method == "ebal_att") 1 else 2
  r <- timed(function() weightit(a ~ x1 + x2 + x3 + x4, data = d, method = "ebal",
                                 estimand = est, moments = mom, reltol = 1e-12)$weights)
  w <- r$value
  values <- summarise(w)
} else if (method %in% c("cbps_exact", "cbps_over")) {
  att <- if (method == "cbps_exact") 0 else 1
  ident <- if (method == "cbps_exact") "exact" else "over"
  r <- timed(function() {
    capture.output(f <- CBPS(a ~ x1 + x2 + x3 + x4, data = d, ATT = att, method = ident,
                             iterations = 1000))
    f
  })
  f <- r$value
  values <- c(summarise(f$weights), setNames(coef(f)[, 1], paste0("coef", 0:4)))
} else stop("unknown method ", method)
fwrite(data.table(name = names(values), value = sprintf("%.17g", as.numeric(values))),
       file.path(out, sprintf("r_%s_%d.csv", method, n)), quote = FALSE)
path <- file.path(out, "r_times.csv")
fwrite(data.table(n = n, method = method, reps = length(r$times),
                  median_s = median(r$times), min_s = min(r$times)),
       path, append = file.exists(path))
