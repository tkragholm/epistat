# Case-crossover windows, exposure per window and the analysis of
# epistat-survival::crossover. The windows are built by explicit loops over
# calendar days with R's Date (weekday by format "%u", month by seq), the
# exposure by the union of the days of each interval, and the analysis by
# survival::clogit, exact, and with a cluster-robust variance by
# method = "approximate" with cluster(person), which coincide in likelihood
# with one case per set. The case-time-control case follows Suissa (1995): the
# sets of control persons carry the same construction at a matched date, and
# the design is exposure and exposure by group.
#
# Cases: time-stratified referents by month and by year with leap days and
# months of 28 to 31 days, unidirectional and bidirectional referents, a lag,
# a hazard window of several days that spans two months, events at the first
# and last day of observation, referents that fall outside observation or on
# the event day, and events whose hazard window starts before observation.
# Run: Rscript crossover.R > crossover.json
suppressMessages({
  library(jsonlite)
  library(survival)
})
set.seed(20261124)

d2 <- function(t) as.Date(t, origin = "1970-01-01")
days <- function(d) as.integer(as.numeric(d))

# Shifts of the referent days, relative to the event day t.
shifts <- function(t, referent) {
  if (referent$type == "uni") return(-referent$step * rev(seq_len(referent$n)))
  if (referent$type == "bi")
    return(c(-referent$step * rev(seq_len(referent$n)), referent$step * seq_len(referent$n)))
  d <- d2(t)
  first <- if (referent$unit == "month") as.Date(format(d, "%Y-%m-01")) else as.Date(format(d, "%Y-01-01"))
  nxt <- if (referent$unit == "month") seq(first, by = "month", length.out = 2)[2] else seq(first, by = "year", length.out = 2)[2]
  all <- seq(first, nxt - 1, by = "day")
  keep <- all[format(all, "%u") == format(d, "%u") & all != d]
  as.integer(as.numeric(keep) - as.numeric(d))
}

build <- function(ev, person, h, lag, referent, obs_start, obs_end) {
  rows <- list(); dropped <- integer(0); outside <- integer(0)
  n_outside <- 0; n_onday <- 0
  for (i in seq_along(ev)) {
    t <- ev[i]
    hs <- t - lag - h; he <- t - lag - 1
    if (hs < obs_start[i] || he > obs_end[i]) { outside <- c(outside, i - 1L); next }
    refs <- list()
    for (s in shifts(t, referent)) {
      a <- hs + s; b <- he + s
      if (a <= t && t <= b) { n_onday <- n_onday + 1; next }
      if (a < obs_start[i] || b > obs_end[i]) { n_outside <- n_outside + 1; next }
      refs[[length(refs) + 1]] <- c(a, b)
    }
    if (length(refs) == 0) { dropped <- c(dropped, i - 1L); next }
    rows[[length(rows) + 1]] <- list(set = i - 1L, person = person[i], start = hs, end = he, case = TRUE)
    for (r in refs) rows[[length(rows) + 1]] <- list(set = i - 1L, person = person[i], start = r[1], end = r[2], case = FALSE)
  }
  w <- data.frame(set = sapply(rows, `[[`, "set"), person = sapply(rows, `[[`, "person"),
                  start = sapply(rows, `[[`, "start"), end = sapply(rows, `[[`, "end"),
                  case = sapply(rows, `[[`, "case"))
  list(w = w, dropped = dropped, outside = outside, n_outside = n_outside, n_onday = n_onday)
}

# Days of exposure in each window: the union of the days of the intervals.
exposure_days <- function(w, ex) {
  sapply(seq_len(nrow(w)), function(r) {
    e <- ex[ex$person == w$person[r], ]
    if (nrow(e) == 0) return(0)
    covered <- unique(unlist(lapply(seq_len(nrow(e)), function(k) e$start[k]:e$end[k])))
    length(intersect(covered, w$start[r]:w$end[r]))
  })
}

cases <- list()
run_case <- function(label, n, h, lag, referent, with_group = FALSE) {
  # events 2018-2021, 70 persons, observation windows that sometimes cut in
  person <- sample(0:69, n, TRUE)
  ev <- sample(days(as.Date("2018-01-01")):days(as.Date("2021-12-31")), n, TRUE)
  # include month ends, February 29 2020 and the first of a month
  special <- days(as.Date(c("2020-02-29", "2019-01-31", "2021-03-01", "2020-12-31", "2018-01-01")))
  ev[seq_along(special)] <- special
  obs_start <- ev - sample(c(0, 2, 15, 40, 80, 200, 3000), n, TRUE, prob = c(.04, .06, .1, .1, .2, .2, .3))
  obs_end <- ev + sample(c(0, 5, 30, 90, 3000), n, TRUE, prob = c(.1, .1, .2, .2, .4))
  b <- build(ev, person, h, lag, referent, obs_start, obs_end)
  # exposure: some intervals per person over the period
  ne <- 600
  ex <- data.frame(person = sample(0:69, ne, TRUE))
  ex$start <- sample(days(as.Date("2017-12-01")):days(as.Date("2021-12-31")), ne, TRUE)
  ex$end <- ex$start + sample(0:60, ne, TRUE)
  w <- b$w
  w$days <- exposure_days(w, ex)
  w$exposed <- as.numeric(w$days > 0)
  group <- NULL
  if (with_group) {
    group <- rbinom(n, 1, 0.5)
    w$group <- group[w$set + 1]
  }
  fit <- NULL
  w$case_i <- as.integer(w$case)
  keep_sets <- unique(w$set)
  if (nrow(w) > 0) {
    ctrl <- coxph.control(eps = 1e-12, toler.chol = 1e-14, iter.max = 100)
    form <- if (with_group) case_i ~ exposed + exposed:group + strata(set) else case_i ~ exposed + strata(set)
    ex_fit <- clogit(form, data = w, method = "exact", control = ctrl)
    form_c <- if (with_group) case_i ~ exposed + exposed:group + strata(set) + cluster(person) else case_i ~ exposed + strata(set) + cluster(person)
    ap_fit <- clogit(form_c, data = w, method = "approximate", control = ctrl)
    fit <- list(coef = I(unname(coef(ex_fit))), vcov = lapply(seq_len(nrow(vcov(ex_fit))), function(i) I(unname(vcov(ex_fit)[i, ]))),
                loglik = I(unname(ex_fit$loglik)),
                approx_coef = I(unname(coef(ap_fit))),
                approx_robust_vcov = lapply(seq_len(nrow(ap_fit$var)), function(i) I(unname(ap_fit$var[i, ]))),
                approx_vcov = lapply(seq_len(nrow(ap_fit$naive.var)), function(i) I(unname(ap_fit$naive.var[i, ]))))
  }
  cases[[length(cases) + 1]] <<- list(label = label, h = h, lag = lag, referent = referent,
    event_day = ev, person = person, obs_start = obs_start, obs_end = obs_end,
    group = group,
    exposure = list(person = ex$person, start = ex$start, end = ex$end),
    windows = list(set = w$set, person = w$person, start = w$start, end = w$end,
                   is_case = w$case, days = w$days),
    sets_without_referent = I(b$dropped), sets_case_outside = I(b$outside),
    n_referents_outside = b$n_outside, n_referents_on_event_day = b$n_onday,
    fit = fit)
}

run_case("time-stratified by month, one day", 400, 1, 0, list(type = "ts", unit = "month"))
run_case("time-stratified by month, three days with a lag", 400, 3, 2, list(type = "ts", unit = "month"))
run_case("time-stratified by year", 300, 2, 1, list(type = "ts", unit = "year"))
run_case("unidirectional", 300, 2, 1, list(type = "uni", n = 3, step = 7))
run_case("bidirectional", 300, 3, 0, list(type = "bi", n = 2, step = 14))
run_case("bidirectional with referent on the event day", 200, 7, 0, list(type = "bi", n = 1, step = 7))
run_case("case-time-control", 600, 1, 0, list(type = "ts", unit = "month"), with_group = TRUE)

out <- list(versions = list(R = R.version.string, survival = as.character(packageVersion("survival"))),
            cases = cases)
cat(toJSON(out, auto_unbox = TRUE, digits = NA, na = "string", null = "null"))
