# The R side of the tier 3 benchmark of epistat_survival::relsurv, in the layout
# of bench/tier3/survival.R: one process runs one method on one size and writes
# the numbers the Rust side writes
# (crates/epistat-survival/benches/tier3_relsurv.rs describes the methods and
# the data file, which the Rust program writes).
#
# Usage: Rscript relsurv.R N METHOD SCRATCH
# Environment: OUT (results, default SCRATCH/out) and REPS (timed repetitions,
# default 3).
#
# net_pp, net_e2: relsurv::rs.surv, method pohar-perme or ederer2, type
#   fleming-harrington, precision 1, no delayed entry.
# net_pp_monthly: popEpi::survtab, surv.type surv.rel, relsurv.method pp,
#   surv.method hazard, monthly breaks to 7.5 years, no delayed entry.
# excess_poisson: popEpi::relpois on the data split by popEpi::splitMulti at
#   whole years of follow-up to 5 and at the cuts of the table, with delayed
#   entry, each record with the hazard of its cell.
# excess_flexpar: rstpm2::stpm2 with df = 4 and bhazard, the hazard of the table
#   at the exit, no delayed entry.
suppressMessages({
  library(data.table); library(survival); library(relsurv); library(popEpi)
  library(Epi); library(rstpm2); library(bbmle)
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

# The daily rate table of tests/relsurv_sim, by the same formula.
day0 <- 3652
age_cuts <- (0:110) * 365
year_cuts <- day0 + (0:45) * 365
haz <- array(0, dim = c(length(age_cuts), 2, length(year_cuts)))
for (a in seq_along(age_cuts)) for (s in 1:2) for (y in seq_along(year_cuts)) {
  haz[a, s, y] <- exp(-10.5 + 0.09 * (a - 1) + 0.35 * (s == 1) - 0.012 * (y - 1)) / 365
}
rt <- haz
attributes(rt) <- list(
  dim = dim(haz), dimnames = list(as.character(age_cuts), c("male", "female"), as.character(year_cuts)),
  dimid = c("age", "sex", "year"), type = c(2L, 1L, 3L),
  cutpoints = list(age_cuts, NULL, as.Date(year_cuts, origin = "1970-01-01")), class = "ratetable")

d <- as.data.frame(fread(file.path(scratch, sprintf("rel_%d.csv", n))))
d$sex <- d$sex + 1
d$dt <- as.Date(d$date0, origin = "1970-01-01")
yr <- 365.25
at <- c(1, 3, 5)
surv_at <- function(time, surv) sapply(at, function(y) surv[max(which(time <= y * yr))])

if (method %in% c("net_pp", "net_e2")) {
  m <- if (method == "net_pp") "pohar-perme" else "ederer2"
  r <- timed(function() rs.surv(Surv(time, status) ~ 1, data = d, ratetable = rt,
                                rmap = list(age = age0, sex = sex, year = dt), method = m,
                                type = "fleming-harrington", precision = 1))
  f <- r$value
  finish(method, r$times, setNames(surv_at(f$time, f$surv), sprintf("surv_%g", at)))
} else if (method == "net_pp_monthly") {
  # popEpi takes the table as hazards by the values at which the time scales
  # of the Lexis data enter each cell, here in days.
  pophaz <- CJ(age = age_cuts, sex = 1:2, per = year_cuts)
  pophaz[, haz := haz[cbind(match(age, age_cuts), sex, match(per, year_cuts))]]
  r <- timed(function() {
    x <- Lexis(entry = list(FUT = 0, age = d$age0, per = d$date0),
               exit = list(FUT = d$time), exit.status = d$status, data = d)
    survtab(FUT ~ 1, data = x, surv.type = "surv.rel", relsurv.method = "pp",
            surv.method = "hazard", breaks = list(FUT = seq(0, 7.5 * yr, by = yr / 12)),
            pophaz = pophaz)
  })
  s <- r$value
  finish(method, r$times, setNames(surv_at(s$Tstop, s$r.pp), sprintf("info_surv_%g", at)))
} else if (method == "excess_poisson") {
  cuts <- (0:5) * yr
  r <- timed(function() {
    L <- Lexis(entry = list(fot = d$start, age = d$age0 + d$start, per = d$date0 + d$start),
               exit = list(fot = d$time), exit.status = factor(d$status, 0:1, c("alive", "dead")),
               data = d)
    sl <- splitMulti(L, breaks = list(fot = cuts, age = age_cuts, per = year_cuts))
    invisible(sl[, pop.haz := haz[cbind(findInterval(age, age_cuts), sex, findInterval(per, year_cuts))]])
    relpois(sl, lex.Xst %in% "dead" ~ -1 + FOT + x1 + x2, check = FALSE)
  })
  finish(method, r$times, setNames(coef(r$value), sprintf("coef%d", seq_along(coef(r$value)) - 1)))
} else if (method == "excess_flexpar") {
  r <- timed(function() {
    # the hazard of the cell that ends at the exit, as the Rust side takes it
    bh <- haz[cbind(findInterval(d$age0 + d$time, age_cuts, left.open = TRUE), d$sex,
                    findInterval(d$date0 + d$time, year_cuts, left.open = TRUE))] * yr
    d$time_y <- d$time / yr
    stpm2(Surv(time_y, status) ~ x1 + x2, data = d, df = 4, bhazard = bh)
  })
  f <- r$value
  finish(method, r$times, c(loglik = as.numeric(logLik(f)),
                            setNames(coef(f)[c("x1", "x2")], c("beta0", "beta1"))))
} else stop("unknown method ", method)
