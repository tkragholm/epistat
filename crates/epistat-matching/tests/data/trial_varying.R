# Sequential trials on a time scale of each person's own, with a stratum that
# changes between trials, checked against Epi::ccwc, as the reference for
# `match_trials_with` with per-person dates and period strata.
#
# A panel of 300 children is simulated over 10 periods of 30 days counted
# from each child's own diagnosis date, so period k is a different calendar
# date for every child. Each child has a sex, a birth date, an entry date
# (some enter after diagnosis), and may die or emigrate. In each period a
# child may have a flare, which also makes starting treatment more likely. A
# child is eligible while never treated before.
#
# In the trial of period k, the risk set of an initiator on its own date t
# (its diagnosis date plus 30 k days) is the set of children enrolled in the
# trial who do not initiate in it, of the same sex and the same flare status
# in period k, who have entered on or before t and are alive and resident on
# t. An initiator older than the age limit (12 years, on the birthday) at t is
# not a case. The sets come from Epi::ccwc on the calendar scale, run for each
# case with the non-initiators of its trial (an initiator is not a control in
# its own trial, though on the calendar scale a later initiator is at risk on
# an earlier one's date), the initiation as the failure at t, and the controls
# at risk until the day before death, emigration or the end of data; the
# script stops unless they equal the direct definition above.
#
# Run from the crate directory:
#   Rscript tests/data/trial_varying.R > tests/data/trial_varying.json
suppressMessages({
  library(Epi)
  library(jsonlite)
})
set.seed(20261127)
n <- 300L
steps <- 0:9
step_days <- 30L
day <- function(x) as.integer(as.Date(x))
origin <- day("2016-01-01") + sample(0:700, n, TRUE)
sex <- sample(c("1", "2"), n, TRUE)
birth <- day("2004-06-01") + sample(0:3600, n, TRUE)
entry <- origin - sample(0:200, n, TRUE)
late <- runif(n) < 0.15
entry[late] <- origin[late] + sample(10:150, sum(late), TRUE)
death <- rep(NA_integer_, n)
emig <- rep(NA_integer_, n)
d <- runif(n) < 0.12
death[d] <- origin[d] + sample(20:300, sum(d), TRUE)
e <- !d & runif(n) < 0.12
emig[e] <- origin[e] + sample(20:300, sum(e), TRUE)
end <- origin + step_days * (max(steps) + 1L)
risk_end <- pmin(death, emig, end, na.rm = TRUE)
rows <- list()
for (i in seq_len(n)) {
  pd <- origin[i] + step_days * steps
  obs <- which(pd >= entry[i] & pd < risk_end[i])
  if (length(obs) == 0) next
  flare <- ifelse(runif(length(obs)) < 0.3, "F", "N")
  tr <- integer(length(obs))
  on <- FALSE
  for (k in seq_along(obs)) {
    p <- if (flare[k] == "F") 0.35 else 0.10
    if (!on && runif(1) < p) on <- TRUE else if (on && runif(1) < 0.15) on <- FALSE
    tr[k] <- as.integer(on)
  }
  ev <- as.integer(runif(length(obs)) < 0.02)
  last <- if (any(ev == 1)) which(ev == 1)[1] else length(obs)
  keep <- seq_len(last)
  prior <- c(0L, cumsum(tr)[-length(tr)])
  rows[[i]] <- data.frame(
    idn = i, period = steps[obs][keep], date = pd[obs][keep],
    eligible = as.integer(prior == 0L)[keep], treatment = tr[keep],
    outcome = ev[keep], flare = flare[keep]
  )
}
panel <- do.call(rbind, rows)

limit_years <- 12L
limit_day <- function(b) {
  lt <- as.POSIXlt(as.Date(b, origin = "1970-01-01"))
  feb29 <- lt$mon == 1L & lt$mday == 29L
  lt$mday[feb29] <- 28L
  lt$year <- lt$year + limit_years
  as.integer(as.Date(lt))
}
lim <- limit_day(birth)
final_end <- max(end) + 1L

risk_sets <- list()
for (k in sort(unique(panel$period))) {
  enr <- panel[panel$period == k & panel$eligible == 1L, ]
  idx <- enr$idn
  t <- enr$date
  init <- enr$treatment == 1L
  case <- init & t < lim[idx]
  if (!any(case)) next
  dd <- death[idx]; ee <- emig[idx]
  out <- pmin(dd, ee, final_end, na.rm = TRUE) - 1L
  for (q in which(case)) {
    # One ccwc run per case, with the case and the non-initiators: a later
    # initiator of the trial would be at risk on an earlier initiator's date on
    # the calendar scale, but it is assigned treatment in the trial.
    rows_q <- c(q, which(!init))
    df <- data.frame(t_in = entry[idx][rows_q],
                     t_out = c(t[q], out[!init]),
                     status = c(1L, rep(0L, sum(!init))),
                     sx = sex[idx][rows_q], fl = enr$flare[rows_q])
    keep <- df$t_in <= df$t_out
    map <- rows_q[keep]
    # ccwc stops when the case has no control at all; the set is then empty.
    cc <- tryCatch(suppressWarnings(ccwc(
      entry = t_in, exit = t_out, fail = status, origin = 0, controls = 1000,
      match = list(sx, fl), data = df[keep, ], silent = TRUE
    )), error = function(e) NULL)
    got <- if (is.null(cc)) integer(0) else sort(map[cc$Map[cc$Fail == 0]])
    ok <- !init & sex[idx] == sex[idx[q]] & enr$flare == enr$flare[q] &
      entry[idx] <= t[q] & (is.na(dd) | dd > t[q]) & (is.na(ee) | ee > t[q])
    mine <- sort(which(ok))
    stopifnot(identical(as.integer(got), as.integer(mine)))
    risk_sets[[sprintf("%d:p%03d", k, idx[q])]] <- I(sprintf("p%03d", idx[got]))
  }
}
stopifnot(length(risk_sets) > 50)
out <- list(
  versions = list(R = R.version.string,
    Epi = as.character(packageVersion("Epi")),
    jsonlite = as.character(packageVersion("jsonlite"))),
  age_limit_years = limit_years,
  people = list(id = sprintf("p%03d", seq_len(n)), sex = sex, birth = birth,
    entry = entry, death = death, emigration = emig),
  panel = list(id = sprintf("p%03d", panel$idn), period = panel$period, date = panel$date,
    eligible = panel$eligible, treatment = panel$treatment, outcome = panel$outcome,
    flare = panel$flare),
  risk_sets = risk_sets)
cat(toJSON(out, auto_unbox = TRUE, na = "null"))
