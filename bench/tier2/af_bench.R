# The R side of the benchmarks of the attributable fraction of a logistic model
# (crates/epistat-glm/benches/tier2_af.rs): AFglm of the R package AF 0.1.5, with
# a confounder, an interaction and clusters of four rows. AF was archived from
# CRAN on 2026-05-08, so its sources are read with source(): set AF_SRC to the
# directory AF/R of the unpacked tarball
# https://cran.r-project.org/src/contrib/Archive/AF/AF_0.1.5.tar.gz. The file is
# written by the Rust program (METHOD=gen_afd). One process runs one method on
# one size and writes the numbers the Rust side writes.
#
# Usage: Rscript af_bench.R N METHOD SCRATCH
# Environment: AF_SRC, OUT (results, default SCRATCH/out) and REPS (timed
# repetitions, default 3).
# The interval is on the scale of log(1 - AF), as AFglm's own summary gives it.
suppressMessages({ library(data.table); library(survival) })
src <- Sys.getenv("AF_SRC")
if (!nzchar(src)) stop("set AF_SRC to the directory AF/R of AF_0.1.5.tar.gz")
for (f in c("AFfunctions.R", "AFglm.R")) source(file.path(src, f))
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

d <- as.data.frame(fread(file.path(scratch, sprintf("afd_%d.csv", n))))
case_control <- method == "af_case_control"
stopifnot(method %in% c("af_cross", "af_case_control"))
fit <- glm(y ~ e * z, family = binomial, data = d)
r <- timed(function() AFglm(fit, data = d, exposure = "e", clusterid = "cluster",
                            case.control = case_control))
a <- r$value
af <- a$AF.est; se <- sqrt(drop(a$AF.var))
half <- qnorm(0.975) * se / (1 - af)
finish(r$times, c(af = af, se = se, lower = 1 - (1 - af) * exp(half),
                  upper = 1 - (1 - af) * exp(-half)))
