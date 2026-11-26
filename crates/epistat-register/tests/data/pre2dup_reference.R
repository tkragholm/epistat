# Reference drug use periods for epistat-register::pre2dup, written out in
# base R from Tanskanen et al., BMC Med Inform Decis Mak 2015;15:21
# (PRE2DUP), since no CRAN package implements it. The readings of the paper
# are those the crate documents: purchases on one day are one purchase, the
# coefficient of variation uses the sample standard deviation, the lower dose
# limit is checked before the upper one caps the dose, the refill length of
# two purchases in the stockpiling test is capped as one purchase is, and a
# stay longer than the longest stay ends the extension of a period's end.
#
# The script first checks equation 1 against the averages the paper prints
# in Table 2, then simulates purchases in DDD with hospital stays for
# persons of three kinds (regular, irregular with stockpiling, and
# occasional) and writes the periods.
#
# Run from the crate directory:
#   Rscript tests/data/pre2dup_reference.R > tests/data/pre2dup_reference.json
suppressMessages(library(jsonlite))
set.seed(20261126)

glob <- list(max_avg = 300, max_stay = 30, max_ddd = 10, max_single = 150, dvar = 0.5,
             adherence = 0.2)
grp <- list(lower = 0.3, common = 1, upper = 10, min_period = 15, max_refill = 120)

hospital_days <- function(stays, from, to) {
  if (nrow(stays) == 0) return(0)
  sum(pmax(pmin(stays$end, to) - pmax(stays$start, from), 0))
}

# Equation 1 over the runs, with the weights of the first and last purchase.
averages <- function(ddd, t, max_avg) {
  m <- length(ddd); avg <- rep(NA_real_, m)
  s <- 1
  while (s <= m) {
    e <- s
    while (e < m && !is.na(t[e]) && t[e] <= max_avg) e <- e + 1
    if (e - s + 1 >= 3 && all(!is.na(ddd[s:e]))) {
      tt <- t
      if (e < m) tt[e] <- ddd[e] / (ddd[e - 1] / t[e - 1])
      for (i in s:e) {
        if (i == s) {
          num <- 5 * ddd[i] + ddd[i + 1]; den <- 5 * tt[i] + tt[i + 1]
        } else if (i == e) {
          num <- ddd[i - 1] + 5 * ddd[i]; den <- tt[i - 1] + 5 * tt[i]
        } else {
          num <- ddd[i - 1] + 4 * ddd[i] + ddd[i + 1]
          den <- tt[i - 1] + 4 * tt[i] + tt[i + 1]
        }
        avg[i] <- if (!is.na(den) && den > 0) num / den else NA
      }
    }
    s <- e + 1
  }
  v <- avg[!is.na(avg)]
  cv <- if (length(v) >= 2) sd(v) / mean(v) else 0
  list(avg = avg, cv = cv)
}

table2 <- averages(rep(10, 5), c(10, 5, 15, 10, 10), 300)$avg
stopifnot(all(abs(table2 - c(1.09, 1.33, 0.80, 0.92, 1.00)) < 0.005))

periods_of <- function(day, ddd, packages, refill, stays) {
  # One purchase per day.
  u <- sort(unique(day))
  ddd1 <- vapply(u, function(d) sum(ddd[day == d]), 1)
  refill1 <- vapply(u, function(d) sum(refill[day == d] * packages[day == d]), 1)
  single <- ifelse(!is.na(refill1), refill1,
                   ifelse(!is.na(ddd1), ddd1 * (1 + glob$adherence) / grp$common, grp$min_period))
  single <- pmin(single, glob$max_single)
  m <- length(u)
  t <- rep(NA_real_, m)
  if (m >= 2) {
    for (i in 1:(m - 1)) t[i] <- u[i + 1] - u[i] - hospital_days(stays, u[i], u[i + 1])
    t[m] <- ddd1[m] / (ddd1[m - 1] / t[m - 1])
  }
  a <- averages(ddd1, t, glob$max_avg)
  avg <- a$avg; cv <- a$cv
  upper <- min(grp$upper, glob$max_ddd)
  erfl <- function(i, amount) min(amount * (1 + glob$dvar * cv) / min(avg[i], upper), grp$max_refill)
  out <- data.frame()
  first <- 1
  while (first <= m) {
    last <- first; by_single <- FALSE
    repeat {
      if (last == m) break
      if (!is.na(avg[last])) {
        if (avg[last] < grp$lower) { by_single <- TRUE; break }
        if (erfl(last, ddd1[last]) >= t[last]) { last <- last + 1; next }
        if (last > first && !is.na(avg[last - 1]) && avg[last] < avg[last - 1] &&
            !is.na(avg[last + 1]) && avg[last] < avg[last + 1]) {
          if (erfl(last, ddd1[last - 1] + ddd1[last]) >= t[last - 1] + t[last]) {
            last <- last + 1; next
          }
        }
        break
      }
      if (single[last] >= t[last]) { last <- last + 1; next }
      break
    }
    k <- last - first + 1
    base <- if (!is.na(avg[last]) && !by_single)
      ddd1[last] * (1 + 0.5 * cv) / min(avg[last], upper) * (1 + exp(-k)) else single[last]
    end <- u[last] + base
    for (j in seq_len(nrow(stays))) {
      if (stays$end[j] <= u[last]) next
      if (stays$start[j] >= end) break
      if (stays$end[j] - stays$start[j] > glob$max_stay) break
      end <- end + stays$end[j] - max(stays$start[j], u[last])
    }
    end <- max(end, u[first] + grp$min_period)
    hd <- hospital_days(stays, u[first], end)
    total <- sum(ddd1[first:last])
    out <- rbind(out, data.frame(start = u[first], end = end, n = k, ddd = total,
                                 hd = hd, per_day = total / (end - u[first] - hd)))
    first <- last + 1
  }
  out
}

n <- 150
rows <- data.frame(); stays_all <- data.frame()
for (p in 0:(n - 1)) {
  kind <- sample(c("regular", "irregular", "occasional"), 1, prob = c(0.5, 0.3, 0.2))
  t <- sample(0:200, 1); k <- if (kind == "occasional") sample(1:3, 1) else sample(4:14, 1)
  for (i in seq_len(k)) {
    amount <- sample(c(30, 50, 90, 100), 1)
    rows <- rbind(rows, data.frame(person = p, day = t, ddd = amount, packages = 1,
                                   refill = if (runif(1) < 0.2) amount else NA))
    if (runif(1) < 0.05) rows <- rbind(rows, data.frame(person = p, day = t, ddd = 30,
                                                        packages = 1, refill = NA))
    gap <- switch(kind,
      regular = round(amount * runif(1, 0.85, 1.15)),
      irregular = sample(c(3, 7, round(amount * 0.5), round(amount * 1.6), amount, 250), 1),
      occasional = sample(c(100, 200, 400), 1))
    t <- t + gap
  }
  # Hospital stays: a few, some longer than 30 days.
  for (s in seq_len(sample(0:2, 1))) {
    a <- sample(0:1500, 1); len <- sample(c(3, 10, 20, 45), 1)
    stays_all <- rbind(stays_all, data.frame(person = p, start = a, end = a + len))
  }
}
stays_all <- stays_all[order(stays_all$person, stays_all$start), ]
# Stays of a person must not overlap.
keep <- rep(TRUE, nrow(stays_all))
for (i in seq_len(nrow(stays_all))[-1]) {
  if (stays_all$person[i] == stays_all$person[i - 1] && stays_all$start[i] < stays_all$end[i - 1]) keep[i] <- FALSE
}
stays_all <- stays_all[keep, ]

result <- data.frame()
for (p in 0:(n - 1)) {
  r <- rows[rows$person == p, ]
  s <- stays_all[stays_all$person == p, c("start", "end")]
  per <- periods_of(r$day, r$ddd, r$packages, r$refill, s)
  result <- rbind(result, cbind(person = p, per))
}

cat(toJSON(list(
  versions = list(R = as.character(getRversion())),
  group = grp, globals = glob,
  rows = rows, stays = stays_all,
  periods = result
), digits = NA, dataframe = "columns", na = "null"))
