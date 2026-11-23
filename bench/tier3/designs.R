# The R side of the tier 3 benchmark of epistat_survival::casecohort and
# epistat_survival::ncc, in the layout of bench/tier3/relsurv.R: one process
# runs one method on one size and writes the numbers the Rust side writes
# (crates/epistat-survival/benches/tier3_designs.rs describes the methods and
# the data files, which the Rust program writes).
#
# Usage: Rscript designs.R N METHOD SCRATCH
# Environment: OUT (results, default SCRATCH/out) and REPS (timed repetitions,
# default 3).
#
# cch_prentice, cch_selfprentice, cch_linying: survival::cch with that method
#   and the cohort size N.
# cch_borgan1, cch_borgan2: survival::cch with method I.Borgan or II.Borgan,
#   the stratum and the cohort size of each stratum.
# ncc_probabilities: multipleNCC::KMprob with three controls per case.
# ncc_fit: multipleNCC::wpl with Kaplan-Meier-type weights and the robust
#   variance, on the whole cohort file.
suppressMessages({
  library(data.table); library(survival); library(multipleNCC)
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

if (startsWith(method, "cch_")) {
  d <- as.data.frame(fread(file.path(scratch, sprintf("cc_%d.csv", n))))
  sizes <- as.data.frame(fread(file.path(scratch, sprintf("cc_sizes_%d.csv", n))))
  d$id <- seq_len(nrow(d))
  m <- c(cch_prentice = "Prentice", cch_selfprentice = "SelfPrentice", cch_linying = "LinYing",
         cch_borgan1 = "I.Borgan", cch_borgan2 = "II.Borgan")[[method]]
  stratified <- m %in% c("I.Borgan", "II.Borgan")
  r <- timed(function() {
    if (stratified) {
      cch(Surv(time, status) ~ x1 + x2, data = d, subcoh = ~sub, id = ~id, stratum = ~stratum,
          cohort.size = setNames(sizes$size, sizes$stratum), method = m)
    } else {
      cch(Surv(time, status) ~ x1 + x2, data = d, subcoh = ~sub, id = ~id, cohort.size = n, method = m)
    }
  })
  f <- r$value
  v <- as.matrix(f$var)
  finish(method, r$times, c(setNames(f$coefficients, c("coef0", "coef1")),
                            se0 = sqrt(v[1, 1]), se1 = sqrt(v[2, 2])))
} else {
  d <- as.data.frame(fread(file.path(scratch, sprintf("ncc_%d.csv", n))))
  state <- ifelse(d$case == 1, 2L, ifelse(d$sampled == 1, 1L, 0L))
  if (method == "ncc_probabilities") {
    r <- timed(function() KMprob(d$exit, state, m = 3))
    p <- r$value
    first <- which(d$sampled == 1 & d$case == 0)[1:10]
    finish(method, r$times, c(sum_p = sum(p), setNames(p[first], sprintf("p%d", 0:9))))
  } else if (method == "ncc_fit") {
    r <- timed(function() {
      suppressWarnings(wpl(Surv(exit, case) ~ x1 + x2, data = d, samplestat = state, m = 3,
                           weight.method = "KM", variance = "robust"))
    })
    f <- r$value
    v <- as.matrix(f$var)
    finish(method, r$times, c(setNames(f$coefficients, c("coef0", "coef1")),
                              se0 = sqrt(v[1, 1]), se1 = sqrt(v[2, 2])))
  } else stop("unknown method ", method)
}
