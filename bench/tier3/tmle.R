# The R side of the tier 3 benchmark of targeted maximum likelihood estimation:
# one process runs one method on one size and writes the numbers the Rust side
# writes, so that ../tier1/compare.R can check that both solved the same problem.
# The data file comes from the Rust program
# (crates/epistat-causal/benches/tier3_tmle.rs).
#
# Usage: Rscript tmle.R N METHOD SCRATCH
# Environment: OUT (results, default SCRATCH/out) and REPS (timed repetitions,
# default 3).
suppressMessages({library(data.table); library(tmle)})
args <- commandArgs(TRUE)
n <- as.integer(args[1]); method <- args[2]; scratch <- args[3]
out <- Sys.getenv("OUT", file.path(scratch, "out"))
reps <- as.integer(Sys.getenv("REPS", "3"))
dir.create(out, showWarnings = FALSE, recursive = TRUE)
d <- fread(file.path(scratch, sprintf("tml_%d.csv", n)))
timed <- function(f) {
  times <- numeric(reps)
  for (i in seq_len(reps)) {
    t0 <- proc.time()[["elapsed"]]
    value <- f()
    times[i] <- proc.time()[["elapsed"]] - t0
  }
  list(times = times, value = value)
}
binary <- method != "tmle_ate_gaussian"
all <- method == "tmle_all_binary"
if (!(method %in% c("tmle_ate_binary", "tmle_ate_gaussian", "tmle_all_binary"))) stop("unknown method ", method)
y <- if (binary) d$yb else d$yc
W <- data.frame(x1 = d$x1, x2 = d$x2, x3 = d$x3)
r <- timed(function() suppressWarnings(
  tmle(Y = y, A = d$a, W = W, family = if (binary) "binomial" else "gaussian",
       Q.SL.library = "SL.glm", g.SL.library = "SL.glm", cvQinit = FALSE,
       prescreenW.g = FALSE, evalATT = all)))
e <- r$value$estimates
values <- c(ate = e$ATE$psi, info_se_ate = sqrt(e$ATE$var.psi))
if (all) values <- c(values, att = e$ATT$psi, info_se_att = sqrt(e$ATT$var.psi),
                     atc = e$ATC$psi, info_se_atc = sqrt(e$ATC$var.psi))
fwrite(data.table(name = names(values), value = sprintf("%.17g", as.numeric(values))),
       file.path(out, sprintf("r_%s_%d.csv", method, n)), quote = FALSE)
path <- file.path(out, "r_times.csv")
fwrite(data.table(n = n, method = method, reps = length(r$times),
                  median_s = median(r$times), min_s = min(r$times)),
       path, append = file.exists(path))
