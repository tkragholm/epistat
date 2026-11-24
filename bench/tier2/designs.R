# The R side of the benchmarks of the self-controlled case series, the
# case-crossover windows and the attributable fraction function of a Cox model
# (crates/epistat-survival/benches/tier2_designs.rs). One process runs one method
# on one size and writes the numbers the Rust side writes. The data files come
# from the Rust program (METHOD=gen_scc, gen_cxo, gen_afc).
#
# Usage: Rscript designs.R N METHOD SCRATCH
# Environment: AF_SRC (af_cox only), OUT (results, default SCRATCH/out) and REPS
# (timed repetitions, default 3).
#
# sccs: SCCS::standardsccs on the stacked events, one exposure date per person,
#   the windows [-14, 0), [0, 14) and [14, 60) days around it (the exposure
#   starts 14 days before the date, which puts the cut points at 0, 14 and
#   28 days), and age groups starting at days 200, 400 and 700. The Rust pieces
#   are half open, so the last observed day is end - 1 and the last day of the
#   exposure is the day before the window ends. Coefficients are written in the
#   order of the Rust design: the three windows, then the age groups.
#   standardsccs has no robust variance, so none is compared.
# crossover: the windows of tests/data/crossover.R (time-stratified by month,
#   three days, no lag, observation periods) built for all events at once with
#   data.table, and the days of exposure in each window as the overlap with the
#   union of the intervals of the person. The test script builds the same windows
#   by explicit loops over days and persons, checked against survival::clogit.
# af_cox: AFcoxph of the R package AF 0.1.5 from its source (see af_bench.R). Its
#   standard error leaves out the compensator of the baseline hazard and is too
#   small (the standard errors are written all the same, so the difference
#   between the two sides shows up in the comparison of the numbers).
suppressMessages({ library(data.table); library(survival) })
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

if (method == "sccs") {
  suppressMessages(library(SCCS))
  d <- fread(file.path(scratch, sprintf("scc_%d.csv", n)))
  long <- melt(d, id.vars = c("id", "start", "end", "expo"), measure.vars = c("ev1", "ev2", "ev3"),
               value.name = "aevent", na.rm = TRUE)
  ds <- as.data.frame(long[, .(indiv = id, astart = start, aend = end - 1, aevent,
                               adrug = expo - 14, aedrug = expo + 59)])
  r <- timed(function() standardsccs(event ~ adrug + age, indiv = indiv, astart = astart,
                                     aend = aend, aevent = aevent, adrug = adrug, aedrug = aedrug,
                                     expogrp = c(0, 14, 28),
                                     agegrp = c(200, 400, 700), data = ds))
  cf <- r$value$coefficients[, "coef"]
  v <- setNames(as.numeric(cf), paste0("coef", seq_along(cf) - 1))
  finish(r$times, v)
} else if (method == "crossover") {
  d <- fread(file.path(scratch, sprintf("cxo_%d.csv", n)))
  ev <- d[set == 0][, .(set = .I - 1L, person, event, ostart = start, oend = end)]
  ex <- d[set == 1][, .(person, start = as.integer(start), end = as.integer(end))]
  h <- 3L; lag <- 0L
  r <- timed(function() {
    ev <- copy(ev)
    ev[, `:=`(hs = event - lag - h, he = event - lag - 1L)]
    ev <- ev[hs >= ostart & he <= oend]
    day <- as.Date(ev$event, origin = "1970-01-01")
    month <- format(day, "%Y%m")
    cand <- CJ(i = seq_len(nrow(ev)), s = c(-28L, -21L, -14L, -7L, 7L, 14L, 21L, 28L))
    cand[, same := format(as.Date(ev$event[i] + s, origin = "1970-01-01"), "%Y%m") == month[i]]
    cand <- cand[same == TRUE]
    cand[, `:=`(a = ev$hs[i] + s, b = ev$he[i] + s, t = ev$event[i], os = ev$ostart[i], oe = ev$oend[i])]
    refs <- cand[!(a <= t & t <= b) & a >= os & b <= oe]
    keep <- unique(refs$i)
    cases <- ev[keep, .(set, person, start = hs, end = he, i = keep)]
    refw <- refs[, .(set = ev$set[i], person = ev$person[i], start = a, end = b, i)]
    w <- rbind(cases, refw)
    w[, row := .I]
    # union of the days of each person's intervals, then the overlap with each window
    setorder(ex, person, start)
    ex[, grp := { prev <- shift(cummax(end)); cumsum(is.na(prev) | start > prev) }, by = person]
    un <- ex[, .(start = min(start), end = max(end)), by = .(person, grp)]
    setkey(un, person, start, end)
    wk <- w[, .(person, start = as.integer(start), end = as.integer(end), row)]
    ov <- foverlaps(wk, un, by.x = c("person", "start", "end"), by.y = c("person", "start", "end"),
                    type = "any", nomatch = NULL)
    list(windows = nrow(w), days = ov[, sum(pmin(end, i.end) - pmax(start, i.start) + 1)])
  })
  finish(r$times, c(windows = r$value$windows, exposure_days = r$value$days))
} else if (method == "af_cox") {
  src <- Sys.getenv("AF_SRC")
  if (!nzchar(src)) stop("set AF_SRC to the directory AF/R of AF_0.1.5.tar.gz")
  suppressMessages(library(numDeriv))
  for (f in c("AFfunctions.R", "AFcoxph.R")) source(file.path(src, f))
  d <- as.data.frame(fread(file.path(scratch, sprintf("afc_%d.csv", n))))
  fit <- coxph(Surv(time, status) ~ e + z + e:z, data = d, ties = "breslow", x = TRUE)
  at <- c(1, 2, 4, 8, 16)
  r <- timed(function() AFcoxph(fit, data = d, exposure = "e", times = at))
  a <- r$value
  v <- c(setNames(a$AF.est, paste0("af", seq_along(at) - 1)),
         setNames(sqrt(drop(a$AF.var)), paste0("se", seq_along(at) - 1)))
  finish(r$times, v)
} else stop("unknown method ", method)
