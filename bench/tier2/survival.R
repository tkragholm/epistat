# The R side of the tier 2 benchmarks of epistat-survival, in the layout of
# bench/tier1/bench.R: one process runs one method on one size and writes the
# numbers the Rust side writes (crates/epistat-survival/benches/tier2_survival.rs
# describes the methods and the data files, which the Rust program writes).
#
# Usage: Rscript survival.R N METHOD SCRATCH
# Environment: OUT (results, default SCRATCH/out) and REPS (timed repetitions,
# default 3).
#
# aipw_km, aipw_cox: riskRegression::ate(estimator = "AIPTW") at the same three
#   times, with the same models. The censoring model of aipw_km is a Kaplan-Meier
#   curve by treatment.
# trial_msm: TrialEmulation::initiators(estimand_type = "PP") on the person-period
#   file. It starts a trial in each period in which the person has not yet been
#   treated, where the Rust program starts one in every period, so the sizes of
#   the two analyses differ and the times are not a like-for-like comparison.
suppressMessages({
  library(data.table)
  library(survival)
  library(prodlim)
  library(riskRegression)
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
finish <- function(times, values) {
  d <- data.table(name = names(values), value = sprintf("%.17g", as.numeric(values)))
  fwrite(d, file.path(out, sprintf("r_%s_%d.csv", method, n)), quote = FALSE)
  path <- file.path(out, "r_times.csv")
  row <- data.table(n = n, method = method, reps = length(times),
                    median_s = median(times), min_s = min(times))
  fwrite(row, path, append = file.exists(path))
}

if (method %in% c("aipw_km", "aipw_cox")) {
  d <- as.data.frame(fread(file.path(scratch, sprintf("surv_%d.csv", n))))
  d$a <- factor(d$a, levels = c(0, 1))
  at <- c(2, 5, 8)
  r <- timed(function() {
    fy <- coxph(Surv(time, status) ~ a + x1 + x2, data = d, x = TRUE, y = TRUE, ties = "breslow")
    fa <- glm(a ~ x1 + x2, data = d, family = binomial)
    fc <- if (method == "aipw_km") {
      prodlim(Hist(time, status) ~ a, data = transform(d, status = 1L - status))
    } else {
      coxph(Surv(time, status == 0) ~ a + x1 + x2, data = d, x = TRUE, y = TRUE, ties = "breslow")
    }
    ate(fy, treatment = fa, censor = fc, data = d, times = at, estimator = "AIPTW",
        known.nuisance = TRUE, verbose = FALSE)
  })
  m <- as.data.frame(r$value$meanRisk)
  df <- as.data.frame(r$value$diffRisk)
  v <- c()
  for (k in seq_along(at)) {
    v[sprintf("risk0_%g", at[k])] <- m$estimate[m$time == at[k] & m$treatment == "0"]
    v[sprintf("risk1_%g", at[k])] <- m$estimate[m$time == at[k] & m$treatment == "1"]
    v[sprintf("diff_%g", at[k])] <- df$estimate[df$time == at[k]]
    v[sprintf("diff_se_%g", at[k])] <- df$se[df$time == at[k]]
  }
  finish(r$times, v)
} else if (method == "trial_msm") {
  suppressMessages(library(TrialEmulation))
  d <- as.data.frame(fread(file.path(scratch, sprintf("trial_%d.csv", n))))
  d$treatment <- d$a
  d$eligible <- ave(d$a, d$id, FUN = function(a) as.integer(cumsum(c(0, head(a, -1))) == 0))
  r <- timed(function() {
    dir <- tempfile(); dir.create(dir)
    initiators(d, id = "id", period = "period", treatment = "treatment", outcome = "outcome",
      eligible = "eligible", estimand_type = "PP", model_var = "assigned_treatment",
      outcome_cov = ~x1, switch_d_cov = ~l + x1, switch_n_cov = ~x1,
      include_followup_time = ~followup_time + I(followup_time^2),
      include_trial_period = ~1, data_dir = dir, quiet = TRUE)
  })
  fit <- r$value
  v <- setNames(as.numeric(coef(fit$model)), paste0("coef", seq_along(coef(fit$model)) - 1))
  finish(r$times, v)
} else {
  stop("unknown method ", method)
}
