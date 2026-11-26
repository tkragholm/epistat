# The self-controlled case series of epistat-survival::sccs with several
# exposure series, seasons and calendar periods. Case 1: two drugs, age groups
# and two seasons against SCCS::standardsccs. Case 2: calendar periods and
# three seasons on people born on different days, against an explicit R loop
# for the pieces (R's Date for the calendar) and glm Poisson with a dummy for
# each person.
#
# Mapping of the conventions, as in sccs.R. SCCS counts days with both ends
# included, so start = astart, end = aend + 1, and the windows of a drug with
# expogrp = c(0, 10) and aedrug = adrug + 27 are [0, 10) and [10, 28) after
# adrug. Time is age in days and the calendar origin is the date of birth as
# days since 1970-01-01, so age a falls on the day dob + a, which is how
# standardsccs places its seasons. A season that starts on day dd of month mm
# is seasongrp ddmm; dob is ddmmyyyy.
# Run: Rscript sccs_calendar.R > sccs_calendar.json
suppressMessages({
  library(jsonlite)
  library(SCCS)
  library(survival)
})
set.seed(20261126)

rows <- function(M) lapply(seq_len(nrow(M)), function(i) I(unname(M[i, ])))
ddmmyyyy <- function(d) as.numeric(format(d, "%d%m%Y"))

# Case 1: two drugs, one dose each or none, age groups and two seasons.
n <- 500
dob <- as.Date("1990-01-01") + sample(0:3000, n, TRUE)
astart <- sample(200:500, n, TRUE)
aend <- astart + sample(500:900, n, TRUE)
d1 <- ifelse(runif(n) < 0.7, astart + sample(0:400, n, TRUE), NA)
d2 <- ifelse(runif(n) < 0.6, astart + sample(0:400, n, TRUE), NA)
season_rate <- function(age, b) {
  m <- as.numeric(format(b + age, "%m"))
  ifelse(m >= 3 & m < 9, 1, 1.6)
}
ev <- lapply(seq_len(n), function(i) {
  days <- astart[i]:aend[i]
  w <- season_rate(days, dob[i])
  if (!is.na(d1[i])) w <- w * ifelse(days >= d1[i] & days < d1[i] + 10, 3, 1)
  if (!is.na(d2[i])) w <- w * ifelse(days >= d2[i] & days <= d2[i] + 20, 2, 1)
  sort(unique(sample(days, sample(1:2, 1, prob = c(0.8, 0.2)), prob = w)))
})
ne <- sapply(ev, length)
ds <- data.frame(indiv = rep(seq_len(n), ne), astart = rep(astart, ne),
                 aend = rep(aend, ne), aevent = unlist(ev),
                 d1 = rep(d1, ne), e1 = rep(d1 + 27, ne),
                 d2 = rep(d2, ne), e2 = rep(d2 + 20, ne),
                 dob = rep(ddmmyyyy(dob), ne))
age_cuts <- c(500, 800, 1100)
sf <- standardsccs(event ~ d1 + d2 + age + season, indiv = indiv, astart = astart,
                   aend = aend, aevent = aevent, adrug = cbind(d1, d2),
                   aedrug = cbind(e1, e2), expogrp = list(c(0, 10), 0),
                   agegrp = age_cuts, seasongrp = c(0103, 0109), dob = dob,
                   data = ds)
case1 <- list(
  label = "two drugs, age groups and seasons, against SCCS",
  start = astart, end = aend + 1,
  event_offsets = c(0, cumsum(ne)), events = unlist(ev),
  series = list(
    list(offsets = c(0, cumsum(!is.na(d1))), dates = d1[!is.na(d1)],
         windows = list(list(from = 0, to = 10), list(from = 10, to = 28))),
    list(offsets = c(0, cumsum(!is.na(d2))), dates = d2[!is.na(d2)],
         windows = list(list(from = 0, to = 21)))),
  age_cuts = age_cuts,
  origin = as.numeric(dob), calendar_cuts = numeric(0),
  seasons = list(list(month = 3, day = 1), list(month = 9, day = 1)),
  coef = I(unname(sf$coefficients[, "coef"])),
  names = I(rownames(sf$coefficients)),
  vcov = rows(sf$vcov), loglik = I(unname(sf$loglik)))

# Case 2: calendar periods and three seasons, against an explicit loop.
season_of <- function(dates, starts) {
  md <- as.numeric(format(dates, "%m")) * 100 + as.numeric(format(dates, "%d"))
  key <- sapply(starts, function(s) s$month * 100 + s$day)
  k <- sapply(md, function(x) sum(key <= x))
  ifelse(k == 0, length(starts) - 1, k - 1)
}
hand_pieces <- function(start, end, events, b, cal_cuts, starts, age_cuts) {
  out <- NULL
  for (i in seq_along(start)) {
    y0 <- as.numeric(format(b[i] + start[i], "%Y")) - 1
    y1 <- as.numeric(format(b[i] + end[i], "%Y"))
    sb <- NULL
    for (y in y0:y1) for (s in starts) {
      sb <- c(sb, as.numeric(as.Date(sprintf("%04d-%02d-%02d", y, s$month, s$day)) - b[i]))
    }
    cuts <- c(start[i], end[i], age_cuts, cal_cuts - as.numeric(b[i]), sb)
    cuts <- sort(unique(cuts[cuts >= start[i] & cuts <= end[i]]))
    for (m in seq_len(length(cuts) - 1)) {
      a <- cuts[m]; e <- cuts[m + 1]
      r <- data.frame(person = i - 1L, start = a, end = e,
                      y = sum(events[[i]] >= a & events[[i]] < e),
                      band = sum(age_cuts <= a),
                      period = sum(cal_cuts - as.numeric(b[i]) <= a),
                      season = season_of(b[i] + a, starts))
      last <- if (is.null(out)) NULL else out[nrow(out), ]
      if (!is.null(last) && last$person == r$person && last$band == r$band &&
          last$period == r$period && last$season == r$season) {
        out$end[nrow(out)] <- e
        out$y[nrow(out)] <- out$y[nrow(out)] + r$y
      } else out <- rbind(out, r)
    }
  }
  out
}
n <- 300
dob <- as.Date("1985-06-15") + sample(0:4000, n, TRUE)
start <- sample(3000:4000, n, TRUE)
end <- start + sample(700:1500, n, TRUE)
cal_cuts <- as.numeric(as.Date(c("2000-01-01", "2002-07-01")))
starts <- list(list(month = 2, day = 15), list(month = 6, day = 1),
               list(month = 10, day = 20))
ev <- lapply(seq_len(n), function(i) {
  days <- start[i]:(end[i] - 1)
  w <- ifelse(as.numeric(dob[i] + days) >= cal_cuts[1], 1.5, 1) *
    ifelse(season_of(dob[i] + days, starts) == 2, 1.8, 1)
  sort(sample(days, sample(1:2, 1, prob = c(0.85, 0.15)), prob = w))
})
pc <- hand_pieces(start, end, ev, dob, cal_cuts, starts, numeric(0))
X <- cbind(sapply(1:2, function(s) as.numeric(pc$season == s)),
           sapply(1:2, function(c) as.numeric(pc$period == c)))
colnames(X) <- paste0("v", 1:4)
used <- tapply(pc$y, pc$person, sum) > 0
dat <- data.frame(y = pc$y, pid = factor(pc$person), off = log(pc$end - pc$start), X)
dat <- dat[pc$person %in% as.integer(names(used)[used]), ]
g <- glm(y ~ 0 + pid + v1 + v2 + v3 + v4 + offset(off), family = poisson, data = dat,
         control = glm.control(epsilon = 1e-13, maxit = 100))
case2 <- list(
  label = "calendar periods and three seasons, against an explicit loop and glm",
  start = start, end = end,
  event_offsets = c(0, cumsum(sapply(ev, length))), events = unlist(ev),
  series = list(), age_cuts = numeric(0),
  origin = as.numeric(dob), calendar_cuts = cal_cuts, seasons = starts,
  pieces = list(person = pc$person, start = pc$start, end = pc$end, events = pc$y,
                season = pc$season, period = pc$period),
  coef = I(unname(coef(g)[paste0("v", 1:4)])),
  vcov = rows(vcov(g)[paste0("v", 1:4), paste0("v", 1:4)]))

out <- list(versions = list(R = R.version.string, SCCS = as.character(packageVersion("SCCS")),
  survival = as.character(packageVersion("survival"))), cases = list(case1, case2))
cat(toJSON(out, auto_unbox = TRUE, digits = NA, null = "null"))
