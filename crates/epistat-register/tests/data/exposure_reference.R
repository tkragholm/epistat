# Reference results for the dose rules, exposed_on and cumulative_dose of
# epistat-register::exposure.
#
# 1. The observed and the fixed dose rule. The duration of each supply is
#    computed here in base R from the rule as the crate documents it:
#    observed, quantity over the dose read from the time to the next
#    dispensing day, clamp(quantity / gap, dose_min, dose_max), and the
#    default dose for the last supply; fixed, a number of days per
#    dispensing. Dispensings on one day are one supply. The episodes are
#    then AdhereR::compute.treatment.episodes with those durations, carry
#    over and permitted gaps of 0 and 30 days. Quantities are multiples of 4
#    and the doses 1, 2 and 4, so every duration is a whole number of days up
#    to rounding: quantity / (quantity / gap) can miss the gap by an ulp.
#    AdhereR is given the rounded durations, which it takes as whole days.
# 2. exposed_on: whether random days, moved back by a lag, lie in an AdhereR
#    episode of the person, [start, end).
# 3. cumulative_dose: over random windows moved back by a lag, the quantity
#    dispensed in the window, and the quantity spread over the days of
#    supply. The second is computed day by day: each day of supply is drawn
#    from the earliest supply not yet used up, first in, first out, which is
#    what carrying supply forward with no cap means, and the day contributes
#    that supply's quantity over its duration.
#
# Run from the crate directory:
#   Rscript tests/data/exposure_reference.R > tests/data/exposure_reference.json
suppressMessages({library(AdhereR); library(jsonlite)})
set.seed(20261126)
epoch <- as.Date("1970-01-01")
dose_default <- 2; dose_min <- 1; dose_max <- 4; fixed_days <- 30

n <- 120
rows <- list()
for (p in 1:n) {
  k <- sample(c(1, 2, 3, 5, 8), 1)
  t <- 18000 + sample(0:500, 1)
  for (i in 1:k) {
    q <- 4 * sample(5:30, 1)
    rows[[length(rows) + 1]] <- data.frame(id = p - 1L, day = t, quantity = q)
    # Same-day refills, refills inside the supply and gaps around 30 days.
    t <- t + sample(c(0, 5, 10, 20, 25, 30, 40, 60, 90, 120, 200), 1)
  }
}
d <- do.call(rbind, rows)
d <- d[order(d$id, d$day), ]

# One supply per person and day.
supplies <- aggregate(cbind(quantity, rows = 1) ~ id + day, data = d, FUN = sum)
supplies <- supplies[order(supplies$id, supplies$day), ]
observed_duration <- function(s) {
  out <- numeric(nrow(s))
  for (i in seq_len(nrow(s))) {
    last <- i == nrow(s) || s$id[i + 1] != s$id[i]
    dose <- if (last) dose_default else
      min(max(s$quantity[i] / (s$day[i + 1] - s$day[i]), dose_min), dose_max)
    out[i] <- s$quantity[i] / dose
  }
  out
}
supplies$observed <- observed_duration(supplies)
supplies$fixed <- fixed_days * supplies$rows
stopifnot(all(abs(supplies$observed - round(supplies$observed)) < 1e-9))

episodes <- function(duration, gap) {
  x <- data.frame(id = supplies$id, date = as.character(epoch + supplies$day),
                  dur = round(duration), med = "A")
  r <- compute.treatment.episodes(
    x, ID.colname = "id", event.date.colname = "date", event.duration.colname = "dur",
    medication.class.colname = "med", carryover.within.obs.window = TRUE,
    carry.only.for.same.medication = FALSE, consider.dosage.change = FALSE,
    medication.change.means.new.treatment.episode = FALSE,
    dosage.change.means.new.treatment.episode = FALSE,
    maximum.permissible.gap = gap, maximum.permissible.gap.unit = "days",
    maximum.permissible.gap.append.to.episode = FALSE,
    followup.window.start = 0, followup.window.start.unit = "days",
    followup.window.duration = 20000, followup.window.duration.unit = "days",
    date.format = "%Y-%m-%d", suppress.warnings = TRUE)
  r <- r[order(r$id, r$episode.start), ]
  list(id = r$id, start = as.numeric(r$episode.start - epoch),
       end = as.numeric(r$episode.end - epoch))
}
runs <- list()
for (rule in c("observed", "fixed")) for (gap in c(0, 30)) {
  e <- episodes(supplies[[rule]], gap)
  e$rule <- rule; e$gap <- gap
  runs[[length(runs) + 1]] <- e
}

# 2. Points against the observed episodes with a gap of 30.
e30 <- runs[[2]]
m <- 600
point_person <- sample(0:(n - 1), m, replace = TRUE)
point_day <- 18000 + sample(-20:1500, m, replace = TRUE)
point_lag <- sample(c(0, 1, 7, 30, 90), m, replace = TRUE)
exposed <- logical(m)
for (i in seq_len(m)) {
  t <- point_day[i] - point_lag[i]
  j <- e30$id == point_person[i]
  exposed[i] <- any(e30$start[j] <= t & t < e30$end[j])
}

# 3. Day by day, first in first out, for the observed durations.
rate_by_day <- list()
for (p in 0:(n - 1)) {
  s <- supplies[supplies$id == p, ]
  left <- round(s$observed)
  rate <- s$quantity / s$observed
  first <- s$day[1]
  horizon <- max(s$day) + sum(left) + 1
  r <- numeric(horizon - first)
  for (t in first:(horizon - 1)) {
    open <- which(s$day <= t & left > 0)
    if (length(open) > 0) {
      j <- open[1]
      r[t - first + 1] <- rate[j]
      left[j] <- left[j] - 1
    }
  }
  rate_by_day[[p + 1]] <- list(first = first, rate = r)
}
w <- 400
window_person <- sample(0:(n - 1), w, replace = TRUE)
window_start <- 18000 + sample(-50:1500, w, replace = TRUE)
window_end <- window_start + sample(c(0, 1, 30, 90, 365, 730), w, replace = TRUE)
window_lag <- sample(c(0, 30, 365), w, replace = TRUE)
at_dispensing <- numeric(w); prorated <- numeric(w)
for (i in seq_len(w)) {
  lo <- window_start[i] - window_lag[i]; hi <- window_end[i] - window_lag[i]
  j <- d$id == window_person[i] & d$day >= lo & d$day < hi
  at_dispensing[i] <- sum(d$quantity[j])
  rb <- rate_by_day[[window_person[i] + 1]]
  days <- seq_along(rb$rate) + rb$first - 1
  prorated[i] <- sum(rb$rate[days >= lo & days < hi])
}

cat(toJSON(list(
  versions = list(R = as.character(getRversion()),
                  AdhereR = as.character(packageVersion("AdhereR"))),
  dose = list(default = dose_default, min = dose_min, max = dose_max, fixed = fixed_days),
  id = d$id, day = d$day, quantity = d$quantity,
  supplies = list(id = supplies$id, day = supplies$day, observed = supplies$observed),
  episodes = runs,
  points = list(person = point_person, day = point_day, lag = point_lag, exposed = exposed),
  windows = list(person = window_person, start = window_start, end = window_end,
                 lag = window_lag, at_dispensing = at_dispensing, prorated = prorated)
), digits = NA, auto_unbox = TRUE))
