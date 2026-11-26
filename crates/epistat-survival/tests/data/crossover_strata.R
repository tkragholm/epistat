# Stratified referents of epistat-survival::crossover against CRAN and
# against explicit loops.
#
# 1. rSPARCS::CXover.data, which lists the referent days of single-day events:
#    direction = "month4" with apart = 7 is the same weekday in the same
#    month, Referent::TimeStratified with StratumUnit::Month, and apart = 14
#    keeps the referents 14 and 28 days away, Referent::Stratified with the
#    month and a spacing of 14.
# 2. season::casecross on a daily series of counts with a daily exposure:
#    stratalength = 28 with matchdow = TRUE is the same weekday within fixed
#    28-day strata from the first day of the series, StratumUnit::Fixed with a
#    spacing of 7, the same with 35-day strata, and stratamonth = TRUE with
#    matchdow = TRUE is the month. casecross fits coxph with the count as weight and a stratum
#    per case day, which has the likelihood of clogit with one set per event.
# 3. Explicit loops over calendar days with R's Date for quarters, years with
#    a 35-day spacing and a 30-day hazard window, fixed strata with a lag, and
#    strata given per event, with observation periods that drop referents.
# Run: Rscript crossover_strata.R > crossover_strata.json
suppressMessages({
  library(jsonlite)
  library(rSPARCS)
  library(season)
  library(survival)
})
set.seed(20261126)

d2 <- function(t) as.Date(t, origin = "1970-01-01")
num <- function(d) as.integer(as.numeric(d))

# 1. rSPARCS.
ev <- num(as.Date("2012-01-01")) + sample(0:2900, 400, TRUE)
sparcs <- list()
for (apart in c(7, 14)) {
  out <- CXover.data(data.frame(id = seq_along(ev), date = d2(ev)), date = "date",
                     ID = "id", direction = "month4", apart = apart)
  refs <- lapply(seq_along(ev), function(i) {
    r <- out[out$ID == i & out$status == 0, "Date"]
    I(sort(num(r) - ev[i]))
  })
  sparcs[[length(sparcs) + 1]] <- list(spacing = apart, event_day = ev, shifts = refs)
}

# 2. season::casecross on a daily series.
first <- as.Date("2014-03-01")
dates <- first + 0:(2 * 365 - 1)
x <- round(sin(2 * pi * seq_along(dates) / 365) + rnorm(length(dates), 0, 0.5), 3)
dow <- as.numeric(format(dates, "%u"))
y <- rpois(length(dates), exp(0.4 + 0.25 * x + 0.1 * (dow == 1)))
series <- data.frame(date = dates, y = y, x = x)
cc <- function(...) {
  f <- casecross(y ~ x, data = series, ...)
  list(coef = unname(coef(f$c.model)), var = unname(vcov(f$c.model)[1, 1]),
       ncases = f$ncases)
}
casecross_ref <- list(
  first = num(first), x = x, y = y,
  fixed_dow = cc(exclusion = 2, stratalength = 28, matchdow = TRUE),
  fixed_35 = cc(exclusion = 2, stratalength = 35, matchdow = TRUE),
  month_dow = cc(exclusion = 2, stratamonth = TRUE, matchdow = TRUE))

# 3. Explicit loops.
stratum <- function(t, unit, origin = NA, length = NA) {
  d <- d2(t)
  y <- as.numeric(format(d, "%Y")); m <- as.numeric(format(d, "%m"))
  if (unit == "quarter") {
    q <- (m - 1) %/% 3
    a <- as.Date(sprintf("%04d-%02d-01", y, 3 * q + 1))
    b <- seq(a, by = "3 months", length.out = 2)[2] - 1
  } else if (unit == "year") {
    a <- as.Date(sprintf("%04d-01-01", y)); b <- as.Date(sprintf("%04d-12-31", y))
  } else if (unit == "month") {
    a <- as.Date(sprintf("%04d-%02d-01", y, m)); b <- seq(a, by = "month", length.out = 2)[2] - 1
  } else {
    k <- floor((t - origin) / length)
    return(c(origin + k * length, origin + k * length + length - 1))
  }
  c(num(a), num(b))
}
loop_windows <- function(ev, h, lag, spacing, strata, obs_start, obs_end) {
  rows <- NULL; dropped <- integer(0); outside <- integer(0); n_out <- 0
  for (i in seq_along(ev)) {
    t <- ev[i]; hs <- t - lag - h; he <- t - lag - 1
    if (hs < obs_start[i] || he > obs_end[i]) { outside <- c(outside, i - 1L); next }
    s <- strata[[i]]
    refs <- NULL
    for (u in s[1]:s[2]) {
      if (u == t || (u - t) %% spacing != 0) next
      a <- hs + (u - t); b <- he + (u - t)
      if (a <= t && t <= b) next
      if (a < obs_start[i] || b > obs_end[i]) { n_out <- n_out + 1; next }
      refs <- rbind(refs, c(a, b))
    }
    if (is.null(refs)) { dropped <- c(dropped, i - 1L); next }
    rows <- rbind(rows, data.frame(set = i - 1L, start = c(hs, refs[, 1]),
                                   end = c(he, refs[, 2]), case = c(TRUE, rep(FALSE, nrow(refs)))))
  }
  list(set = rows$set, start = rows$start, end = rows$end, is_case = rows$case,
       sets_without_referent = I(dropped), sets_case_outside_observation = I(outside),
       n_referents_outside_observation = n_out)
}
loops <- list()
add <- function(label, unit, h, lag, spacing, origin = NA, length = NA, custom = FALSE) {
  n <- 300
  ev <- num(as.Date("2011-11-01")) + sample(0:1500, n, TRUE)
  obs_start <- ev - sample(20:500, n, TRUE)
  obs_end <- ev + sample(0:200, n, TRUE)
  strata <- if (custom) {
    lapply(ev, function(t) { a <- t - sample(0:90, 1); c(a, a + sample(90:200, 1)) })
  } else lapply(ev, stratum, unit = unit, origin = origin, length = length)
  strata <- lapply(seq_along(ev), function(i) {
    s <- strata[[i]]; if (custom) s <- c(min(s[1], ev[i]), max(s[2], ev[i])); s
  })
  loops[[length(loops) + 1]] <<- list(label = label, unit = unit, h = h, lag = lag,
    spacing = spacing, origin = origin, length = length, custom = custom,
    event_day = ev, obs_start = obs_start, obs_end = obs_end,
    first = sapply(strata, `[`, 1), last = sapply(strata, `[`, 2),
    windows = loop_windows(ev, h, lag, spacing, strata, obs_start, obs_end))
}
add("quarter, weekly", "quarter", 1, 0, 7)
add("quarter, 5 days with a lag", "quarter", 5, 2, 7)
add("year, 30 days every 35", "year", 30, 0, 35)
add("fixed 28 days, weekly, lag 1", "fixed", 1, 1, 7, origin = num(as.Date("2011-06-06")), length = 28)
add("fixed 90 days, 14 days every 14", "fixed", 14, 0, 14, origin = 15000, length = 90)
add("strata per event, 10 days every 21", "custom", 10, 0, 21, custom = TRUE)

out <- list(versions = list(R = R.version.string,
  rSPARCS = as.character(packageVersion("rSPARCS")),
  season = as.character(packageVersion("season")),
  survival = as.character(packageVersion("survival"))),
  sparcs = sparcs, casecross = casecross_ref, loops = loops)
cat(toJSON(out, auto_unbox = TRUE, digits = NA, null = "null", na = "null"))
