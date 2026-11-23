# The R side of the g-formula benchmark: gfoRmula::gformula on the data file
# written by crates/epistat-causal/benches/tier3_gformula.rs, one process per
# method and size, writing the numbers the Rust side writes. See
# gformula_run.sh.
#
# Usage: Rscript gformula.R N METHOD SCRATCH
# Environment: OUT (results, default SCRATCH/out) and REPS (default 3).
suppressMessages({library(data.table); library(gfoRmula)})
args <- commandArgs(TRUE)
n <- as.integer(args[1]); method <- args[2]; scratch <- args[3]
out <- Sys.getenv("OUT", file.path(scratch, "out"))
reps <- as.integer(Sys.getenv("REPS", "3"))
dir.create(out, showWarnings = FALSE, recursive = TRUE)
d <- fread(file.path(scratch, sprintf("gf_%d.csv", n)))
K <- 6
dyn_l2 <- function(newdf, pool, intvar, intvals, time_name, t) {
  newdf[, (intvar) := as.numeric(newdf$L2 > intvals[[1]])]
}
dyn_l1 <- function(newdf, pool, intvar, intvals, time_name, t) {
  newdf[, (intvar) := as.numeric(newdf$L1)]
}
ints <- if (method == "gf_dynamic") {
  list(intervention1.A = list(dyn_l2, 0.5), intervention2.A = list(dyn_l1))
} else {
  list(intervention1.A = list(static, rep(0, K)), intervention2.A = list(static, rep(1, K)))
}
run <- function() {
  do.call(gformula, c(list(
    obs_data = copy(d), id = "id", time_points = K, time_name = "t0",
    covnames = c("L1", "L2", "A"), covtypes = c("binary", "normal", "binary"),
    covparams = list(covmodels = c(L1 ~ lag1_L1 + lag_cumavg1_A + V + W,
                                   L2 ~ lag1_L2 + L1 + lag1_A + W,
                                   A ~ lag1_A + L1 + L2 + V)),
    histvars = list(c("L1", "L2", "A"), c("L2"), c("A")),
    histories = c(lagged, cumavg, lagavg), basecovs = c("V", "W"),
    outcome_name = "Y", outcome_type = "survival",
    ymodel = Y ~ L1 + cumavg_L2 + A + V + A:V + t0,
    compevent_name = "D", compevent_model = D ~ L1 + W + t0,
    nsamples = if (method == "gf_boot") 20 else 0, seed = 1, show_progress = FALSE), ints))
}
times <- numeric(reps)
for (i in seq_len(reps)) {
  t0 <- proc.time()[["elapsed"]]
  fit <- run()
  times[i] <- proc.time()[["elapsed"]] - t0
}
values <- c()
for (name in c("L1", "L2", "A", "Y", "D")) {
  b <- unname(fit$coeffs[[name]])
  values <- c(values, setNames(b, sprintf("coef_%s_%d", name, seq_along(b) - 1)))
}
r <- fit$result[k == K - 1]
for (s in seq_len(nrow(r))) {
  values[sprintf("info_risk_%d", s - 1)] <- r[["g-form risk"]][s]
  if (method == "gf_boot") values[sprintf("info_se_%d", s - 1)] <- r[["Risk SE"]][s]
}
v <- data.table(name = names(values), value = sprintf("%.17g", as.numeric(values)))
fwrite(v, file.path(out, sprintf("r_%s_%d.csv", method, n)), quote = FALSE)
path <- file.path(out, "r_times.csv")
fwrite(data.table(n = n, method = method, reps = reps, median_s = median(times), min_s = min(times)),
       path, append = file.exists(path))
