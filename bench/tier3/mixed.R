# The R side of the mixed model benchmark: lme4::lmer and lme4::glmer at their
# default controls on the data files written by
# crates/epistat-mixed/benches/tier3_mixed.rs, one process per method and size,
# writing the numbers the Rust side writes. See bench/run.sh.
#
# Usage: Rscript mixed.R N METHOD SCRATCH
# Environment: OUT (results, default SCRATCH/out) and REPS (default 3).
suppressMessages({library(data.table); library(lme4)})
args <- commandArgs(TRUE)
n <- as.integer(args[1]); method <- args[2]; scratch <- args[3]
out <- Sys.getenv("OUT", file.path(scratch, "out"))
reps <- as.integer(Sys.getenv("REPS", "3"))
dir.create(out, showWarnings = FALSE, recursive = TRUE)
kind <- if (method == "pairs_agq") "pair" else "mix"
d <- fread(file.path(scratch, sprintf("%s_%d.csv", kind, n)))
d$g <- factor(d$g)
fit_once <- function() {
  switch(method,
    lmm_intercept = lmer(yg ~ x1 + x2 + t + (1 | g), d),
    lmm_slope = lmer(yg ~ x1 + x2 + t + (t | g), d),
    glmm_laplace = glmer(yb ~ x1 + x2 + (1 | g), d, family = binomial),
    glmm_agq = glmer(yb ~ x1 + x2 + (1 | g), d, family = binomial, nAGQ = 7),
    pairs_agq = glmer(yb ~ x1 + x2 + (1 | g), d, family = binomial, nAGQ = 7),
    poisson_laplace = glmer(yc ~ x1 + x2 + (1 | g), d, family = poisson))
}
times <- numeric(reps)
for (i in seq_len(reps)) {
  t0 <- proc.time()[["elapsed"]]
  fit <- fit_once()
  times[i] <- proc.time()[["elapsed"]] - t0
}
theta <- unname(getME(fit, "theta"))
ll <- as.numeric(logLik(fit))
# glmer's logLik leaves out the saturated log likelihood for nAGQ > 1; the
# Rust side always includes it.
if (method %in% c("glmm_agq", "pairs_agq")) ll <- ll + sum(dbinom(d$yb, 1, d$yb, log = TRUE))
# The Rust side reports the standard deviation of a generalised model's random
# intercept, which is lme4's theta, and lme4's theta of a linear model.
values <- c(setNames(theta, paste0("theta", seq_along(theta) - 1)),
            setNames(unname(fixef(fit)), paste0("beta", seq_along(fixef(fit)) - 1)),
            loglik = ll)
v <- data.table(name = names(values), value = sprintf("%.17g", as.numeric(values)))
fwrite(v, file.path(out, sprintf("r_%s_%d.csv", method, n)), quote = FALSE)
path <- file.path(out, "r_times.csv")
fwrite(data.table(n = n, method = method, reps = reps, median_s = median(times), min_s = min(times)),
       path, append = file.exists(path))
