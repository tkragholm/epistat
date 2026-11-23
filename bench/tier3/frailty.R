# The R side of the tier 3 benchmark of epistat_survival::frailty, in the layout
# of bench/tier3/survival.R: one process runs one method on one size and writes
# the numbers the Rust side writes
# (crates/epistat-survival/benches/tier3_frailty.rs describes the methods and
# the data file, which the Rust program writes).
#
# Usage: Rscript frailty.R N METHOD SCRATCH
# Environment: OUT (results, default SCRATCH/out) and REPS (timed repetitions,
# default 3).
#
# frailty_fixed, frailty_entry: coxph(... + frailty(cluster, theta = 0.5)), sparse.
# frailty_dense: the same with sparse = FALSE, whose information is dense in the
#   clusters; expect it to run out of time or memory above 10^4 rows.
# frailty_search, frailty_profile: coxph(... + frailty(cluster)), method "em".
# Every fit runs with ties = "efron" and coxph.control(eps = 1e-12, iter.max = 500,
# outer.max = 30), so that both sides solve the score equations to far below the
# standard errors.
suppressMessages({
  library(data.table)
  library(survival)
})
args <- commandArgs(TRUE)
n <- as.integer(args[1]); method <- args[2]; scratch <- args[3]
out <- Sys.getenv("OUT", file.path(scratch, "out"))
reps <- as.integer(Sys.getenv("REPS", "3"))
dir.create(out, showWarnings = FALSE, recursive = TRUE)
timed <- function(f) {
  times <- numeric(reps)
  for (i in seq_len(reps)) {
    t0 <- proc.time()[["elapsed"]]
    value <- f()
    times[i] <- proc.time()[["elapsed"]] - t0
  }
  list(times = times, value = value)
}
finish <- function(label, times, values) {
  d <- data.table(name = names(values), value = sprintf("%.17g", as.numeric(values)))
  fwrite(d, file.path(out, sprintf("r_%s_%d.csv", label, n)), quote = FALSE)
  path <- file.path(out, "r_times.csv")
  row <- data.table(n = n, method = label, reps = length(times),
                    median_s = median(times), min_s = min(times))
  fwrite(row, path, append = file.exists(path))
}

d <- as.data.frame(fread(file.path(scratch, sprintf("frailty_%d.csv", n))))
ctl <- coxph.control(eps = 1e-12, iter.max = 500, outer.max = 30)
lhs <- if (method == "frailty_entry") "Surv(start, time, status)" else "Surv(time, status)"
frail <- switch(method,
  frailty_fixed = , frailty_entry = "frailty(cluster, theta = 0.5)",
  frailty_dense = "frailty(cluster, theta = 0.5, sparse = FALSE)",
  frailty_search = , frailty_profile = "frailty(cluster)")
fm <- as.formula(paste(lhs, "~ x1 + x2 +", frail))
r <- timed(function() suppressWarnings(coxph(fm, data = d, ties = "efron", control = ctl)))
f <- r$value
h <- f$history[[1]]
theta <- unname(if (!is.null(h$history)) h$history[nrow(h$history), 1] else h$theta)
if (method == "frailty_profile") {
  v <- c(c_loglik = h$c.loglik, info_theta = theta)
} else {
  v <- c(beta0 = coef(f)[[1]], beta1 = coef(f)[[2]],
         se0 = sqrt(f$var[1, 1]), se1 = sqrt(f$var[2, 2]),
         loglik = f$loglik[2], c_loglik = h$c.loglik)
  if (method == "frailty_search") v <- c(v, theta = theta)
}
finish(method, r$times, v)
