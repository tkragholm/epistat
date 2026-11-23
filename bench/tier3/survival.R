# The R side of the tier 3 benchmark of epistat_survival::flexpar, in the layout
# of bench/tier2/survival.R: one process runs one method on one size and writes
# the numbers the Rust side writes
# (crates/epistat-survival/benches/tier3_survival.rs describes the methods and
# the data file, which the Rust program writes).
#
# Usage: Rscript survival.R N METHOD SCRATCH
# Environment: OUT (results, default SCRATCH/out) and REPS (timed repetitions,
# default 3).
#
# flexpar_hazard, flexpar_odds, flexpar_probit, flexpar_entry: rstpm2::stpm2 with
#   df = 4, link.type PH, PO or probit, on all rows (Surv(entry, time, status) for
#   flexpar_entry). flexsurv::flexsurvspline is timed too, as flexsurv_METHOD, with
#   the same knots.
# flexpar_tvc: rstpm2::stpm2 with tvc = list(x2 = 2) and delayed entry.
suppressMessages({
  library(data.table)
  library(survival)
  library(rstpm2)
  library(flexsurv)
  library(bbmle)
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

d <- as.data.frame(fread(file.path(scratch, sprintf("flex_%d.csv", n))))
names(d)[names(d) == "start"] <- "entry"
d$event <- d$status
at <- c(0.5, 1.5, 3.0)
entry <- method %in% c("flexpar_entry", "flexpar_tvc")
link <- c(flexpar_hazard = "PH", flexpar_odds = "PO", flexpar_probit = "probit",
          flexpar_entry = "PH", flexpar_tvc = "PH")[[method]]
fscale <- c(PH = "hazard", PO = "odds", probit = "normal")[[link]]
lhs <- if (entry) "Surv(entry, time, event)" else "Surv(time, event)"
fm <- as.formula(paste(lhs, "~ x1 + x2"))
nd <- data.frame(x1 = 0.5, x2 = 1, time = at)

r <- timed(function() {
  if (method == "flexpar_tvc") stpm2(fm, data = d, df = 4, link.type = link, tvc = list(x2 = 2))
  else stpm2(fm, data = d, df = 4, link.type = link)
})
f <- r$value
v <- c(loglik = as.numeric(logLik(f)))
if (method != "flexpar_tvc") v <- c(v, setNames(coef(f)[c("x1", "x2")], c("beta0", "beta1")))
s <- predict(f, newdata = nd, type = "surv")
v <- c(v, setNames(as.numeric(s), sprintf("surv_%g", at)))
if (method == "flexpar_tvc") {
  hr <- predict(f, newdata = data.frame(x1 = 0.5, x2 = 0, time = at), type = "hr", var = "x2",
                exposed = function(data) transform(data, x2 = 1))
  v <- c(v, setNames(as.numeric(hr), sprintf("hr_%g", at)))
}
finish(method, r$times, v)

if (method != "flexpar_tvc") {
  # flexsurvspline with the knots stpm2 places: quantiles of the log event times
  le <- log(d$time[d$event == 1]); kn <- as.numeric(quantile(le, seq(0, 1, length.out = 5)))
  g <- timed(function() {
    flexsurvspline(fm, data = d, scale = fscale, knots = kn[2:4], bknots = kn[c(1, 5)])
  })
  v <- c(loglik = as.numeric(logLik(g$value)))
  v <- c(v, setNames(g$value$res[c("x1", "x2"), "est"], c("beta0", "beta1")))
  s <- summary(g$value, newdata = data.frame(x1 = 0.5, x2 = 1), t = at, type = "survival", ci = FALSE, tidy = TRUE)
  v <- c(v, setNames(s$est, sprintf("surv_%g", at)))
  finish(paste0("flexsurv_", method), g$times, v)
}
