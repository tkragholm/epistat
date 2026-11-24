# The R side of the benchmarks of epistat-glm's generalised estimating
# equations and ordinal regression (crates/epistat-glm/benches/tier2_glm.rs).
# One process runs one method on one size and writes the numbers the Rust side
# writes, so that bench/tier1/compare.R can check that both solved the same
# problem. The data files come from the Rust program (METHOD=gen_gpanel and
# gen_gord).
#
# Usage: Rscript glm_bench.R N METHOD SCRATCH
# Environment: OUT (results, default SCRATCH/out) and REPS (timed repetitions,
# default 3).
#
# brant: brant::brant on a MASS::polr fit. The package differs from Brant's
# formulas in the fifth digit (it copies the upper blocks of the covariance of the
# stacked estimates into the lower ones without transposing, and takes the diagonal
# blocks from glm's vcov at the working weights of the step before the last), so
# the omnibus statistic agrees to about 1e-3 relative; see
# crates/epistat-glm/tests/r_ordinal.rs.
suppressMessages({ library(data.table); library(geepack); library(ordinal) })
args <- commandArgs(TRUE)
n <- as.integer(args[1]); method <- args[2]; scratch <- args[3]
out <- Sys.getenv("OUT", file.path(scratch, "out"))
reps <- as.integer(Sys.getenv("REPS", "3"))
dir.create(out, showWarnings = FALSE, recursive = TRUE)
rd <- function(kind) {
  d <- fread(file.path(scratch, sprintf("%s_%d.csv", kind, n)))
  for (j in names(d)) if (is.integer(d[[j]])) set(d, j = j, value = as.numeric(d[[j]]))
  d
}
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
tight <- geese.control(epsilon = 1e-8, maxit = 100)

if (method %in% c("gee_exch", "gee_ar1", "gee_unstr")) {
  d <- as.data.frame(rd("gpanel"))
  spec <- list(gee_exch = list("b", binomial(), "exchangeable"),
               gee_ar1 = list("y", poisson(), "ar1"),
               gee_unstr = list("y", poisson(), "unstructured"))[[method]]
  # geeglm stops with a segmentation fault on the unstructured structure when the
  # waves of a cluster have gaps, as these data do.
  if (method == "gee_unstr") quit(status = 0)
  fm <- as.formula(paste(spec[[1]], "~ x1 + x2 + x3"))
  r <- timed(function() geeglm(fm, family = spec[[2]], data = d, id = d$id, waves = d$wave,
                               corstr = spec[[3]], control = tight))
  f <- r$value
  v <- c(setNames(coef(f), paste0("coef_", c("int", "x1", "x2", "x3"))),
         setNames(sqrt(diag(f$geese$vbeta)), paste0("se_", c("int", "x1", "x2", "x3"))),
         setNames(f$geese$alpha, paste0("alpha", seq_along(f$geese$alpha) - 1)),
         scale = unname(f$geese$gamma))
  finish(r$times, v)
} else if (method %in% c("ordinal", "ordinal_ppo")) {
  d <- as.data.frame(rd("gord")); d$y <- factor(d$y, ordered = TRUE)
  ppo <- method == "ordinal_ppo"
  fm <- if (ppo) y ~ x1 + x2 + x3 + x4 + x5 else y ~ x1 + x2 + x3 + x4 + x5 + z1 + z2
  r <- timed(function() if (ppo) clm(fm, nominal = ~ z1 + z2, data = d) else clm(fm, data = d))
  f <- r$value; se <- sqrt(diag(vcov(f)))
  nm <- c("x1", "x2", "x3", "x4", "x5", if (!ppo) c("z1", "z2"))
  th <- if (ppo) f$alpha.mat[1, ] else f$alpha
  th_names <- if (ppo) paste0(colnames(f$alpha.mat), ".(Intercept)") else names(f$alpha)
  v <- c(setNames(th, paste0("theta", seq_along(th) - 1)),
         setNames(se[th_names], paste0("se_theta", seq_along(th) - 1)),
         setNames(f$beta, paste0("coef_", nm)), setNames(se[names(f$beta)], paste0("se_", nm)))
  finish(r$times, v)
} else if (method == "brant") {
  suppressMessages({ library(MASS); library(brant) })
  d <- as.data.frame(rd("gord"))
  # brant adds columns named z1, z2, ... to the model frame, so the covariates are renamed
  names(d)[1:7] <- paste0("v", 1:7)
  d$y <- factor(d$y, ordered = TRUE)
  m <- polr(y ~ v1 + v2 + v3 + v4 + v5 + v6 + v7, data = d, Hess = TRUE, method = "logistic")
  r <- timed(function() {
    capture.output(tab <- suppressWarnings(brant(m)))
    tab
  })
  finish(r$times, c(omnibus = r$value["Omnibus", "X2"]))
} else stop("unknown method ", method)
