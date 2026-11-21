# Writes a simulated cohort as CSV and times MatchIt's cardinality matching with
# HiGHS on it, for the comparison in `examples/cardinality_timing.rs`.
#
#   Rscript tests/data/cardinality_timing.R N OUTDIR [SECONDS]
#
# The cohort has N units, about a fifth treated, three continuous and two binary
# covariates, and a treated group that differs from the controls by about 0.3
# standard deviations on each covariate. MatchIt gets `ratio = 1`, `tols = 0.1`
# and the time limit SECONDS (default 600).
suppressMessages(library(MatchIt))
args <- commandArgs(trailingOnly = TRUE)
n <- as.integer(args[1])
dir <- args[2]
limit <- if (length(args) >= 3) as.numeric(args[3]) else 600
set.seed(n)
x1 <- rnorm(n); x2 <- rnorm(n, 1, 2); x3 <- rexp(n)
b1 <- rbinom(n, 1, 0.4); b2 <- rbinom(n, 1, 0.25)
lp <- -1.8 + 0.3 * x1 + 0.15 * x2 + 0.3 * x3 + 0.4 * b1 - 0.3 * b2
treat <- rbinom(n, 1, plogis(lp))
d <- data.frame(treat, x1, x2, x3, b1, b2)
write.csv(d, file.path(dir, sprintf("cohort_%d.csv", n)), row.names = FALSE, quote = FALSE)
t0 <- proc.time()[["elapsed"]]
m <- matchit(treat ~ x1 + x2 + x3 + b1 + b2, data = d, method = "cardinality",
             ratio = 1, tols = 0.1, include.obj = TRUE, time = limit)
el <- proc.time()[["elapsed"]] - t0
cat(sprintf("matchit n %d selected %d status %d (%s) seconds %.2f\n", n, sum(m$weights > 0),
            m$obj$status, m$obj$status_message, el))
