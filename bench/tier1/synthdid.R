# The R side of the synthdid benchmark (crates/epistat-did/benches/tier1_synthdid.rs):
# one process runs one method on one size and writes the numbers the Rust side
# writes. See bench/run.sh for the driver.
#
# Usage: Rscript synthdid.R N METHOD SCRATCH
# Environment: OUT (results, default SCRATCH/out) and REPS (timed repetitions,
# default 3). N is the number of units. synthdid 0.0.9 is not on CRAN; the
# benchmark uses the build from github.com/synth-inference/synthdid at commit
# 70c1ce3eac58e28c30b67435ca377bb48baa9b8a.
suppressMessages({
  library(data.table)
  library(synthdid)
})
args <- commandArgs(TRUE)
n <- as.integer(args[1]); method <- args[2]; scratch <- args[3]
out <- Sys.getenv("OUT", file.path(scratch, "out"))
reps <- as.integer(Sys.getenv("REPS", "3"))
dir.create(out, showWarnings = FALSE, recursive = TRUE)

kind <- switch(method, synthdid_t20 = , synthdid_placebo_t20 = "sdid20",
               synthdid_t100 = , synthdid_jk_t100 = , sc_t100 = "sdid100",
               stop("unknown method ", method))
d <- fread(file.path(scratch, sprintf("%s_%d.csv", kind, n)))
periods <- ncol(d) - 2
Y <- as.matrix(d[, paste0("y", 0:(periods - 1)), with = FALSE])
N0 <- sum(d$treated == 0)
T0 <- (periods * 4) %/% 5
estimator <- if (method == "sc_t100") sc_estimate else synthdid_estimate

times <- numeric(reps)
for (i in seq_len(reps)) {
  t0 <- proc.time()[["elapsed"]]
  est <- estimator(Y, N0, T0)
  se <- switch(method,
               synthdid_jk_t100 = c(synthdid_se(est, method = "jackknife")),
               synthdid_placebo_t20 = c(synthdid_se(est, method = "placebo", replications = 20)),
               NA)
  times[i] <- proc.time()[["elapsed"]] - t0
}
w <- attr(est, "weights")
indexed <- function(prefix, x) setNames(as.numeric(x), paste0(prefix, seq_along(x) - 1))
values <- c(estimate = c(est), indexed("lambda", w$lambda), indexed("omega", w$omega),
            indexed("curve", synthdid_effect_curve(est)),
            info_lambda_steps = if (is.null(w$lambda.vals)) 0 else sum(!is.na(w$lambda.vals)),
            info_omega_steps = sum(!is.na(w$omega.vals)))
if (method == "synthdid_jk_t100") values["se"] <- se
if (method == "synthdid_placebo_t20") values["info_se"] <- se
fwrite(data.table(name = names(values), value = sprintf("%.17g", values)),
       file.path(out, sprintf("r_%s_%d.csv", method, n)), quote = FALSE)
path <- file.path(out, "r_times.csv")
fwrite(data.table(n = n, method = method, reps = reps, median_s = median(times), min_s = min(times)),
       path, append = file.exists(path))
