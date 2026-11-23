# Reference results for the event definitions of epistat-register::events, by
# a direct implementation of the rules in base R: first event in a window,
# prior status, confirmation by a second date, and the look-back windows
# (the heaven rule `date >= index - look.back * 365.25 & date <= index` for
# whole days, and civil years by POSIXlt). Run from the crate directory:
#   Rscript tests/data/events_reference.R > tests/data/events_reference.json
suppressMessages(library(jsonlite))
set.seed(20261123)
epoch <- as.Date("1970-01-01")
to_day <- function(d) as.numeric(d - epoch)
from_day <- function(n) epoch + n

# Rules, in the order of the definitions (bit k):
# kind: "days", "years_march1", "years_feb28", "unlimited", "none", "heaven"
rules <- list(
  list(kind = "days", n = 365, require = 1, inclusive = FALSE),
  list(kind = "years_march1", n = 5, require = 1, inclusive = TRUE),
  list(kind = "years_feb28", n = 1, require = 1, inclusive = FALSE),
  list(kind = "unlimited", n = 0, require = 1, inclusive = FALSE),
  list(kind = "none", n = 0, require = 2, within = 30, date = "first", inclusive = FALSE),
  list(kind = "heaven", n = 5, require = 1, inclusive = TRUE),
  list(kind = "none", n = 0, require = 2, within = 10, date = "confirming", inclusive = FALSE),
  list(kind = "days", n = 100, require = 2, within = Inf, date = "first", inclusive = TRUE)
)

years_before <- function(day, n, leap) {
  lt <- as.POSIXlt(from_day(day))
  y <- lt$year + 1900 - n
  target_leap <- (y %% 4 == 0 && y %% 100 != 0) || y %% 400 == 0
  if (lt$mon == 1 && lt$mday == 29 && !target_leap) {
    return(to_day(if (leap == "feb28") as.Date(sprintf("%d-02-28", y)) else as.Date(sprintf("%d-03-01", y))))
  }
  to_day(as.Date(sprintf("%d-%02d-%02d", y, lt$mon + 1, lt$mday)))
}

window_start <- function(r, entry, history) {
  switch(r$kind,
    days = entry - r$n,
    years_march1 = years_before(entry, r$n, "march1"),
    years_feb28 = years_before(entry, r$n, "feb28"),
    unlimited = history,
    none = NA_real_,
    heaven = NA_real_)
}

evaluate <- function(r, k, days, masks, entry, exit, history) {
  has <- bitwAnd(masks, 2^(k - 1)) != 0
  d <- days[has]
  boundary <- entry + as.numeric(r$inclusive)
  out <- list(first = NA_real_, case = NA_real_, prior = "NotAssessed", n = 0L, before = sum(d < history))
  if (r$kind == "heaven") {
    # data.date >= index - look.back * 365.25 & data.date <= index, real valued
    found <- any(d >= entry - r$n * 365.25 & d <= entry & d >= history)
    out$prior <- if (found) "Prevalent" else if (entry - r$n * 365.25 < history) "HistoryTooShort" else "NoneObserved"
  } else if (r$kind != "none") {
    s <- window_start(r, entry, history)
    found <- any(d >= max(s, history) & d < boundary)
    out$prior <- if (found) "Prevalent" else if (s < history) "HistoryTooShort" else "NoneObserved"
  }
  w <- d[d >= max(boundary, history) & d <= exit]
  out$n <- length(w)
  if (length(w) > 0) {
    out$first <- w[1]
    if (r$require == 1) {
      out$case <- w[1]
    } else {
      for (i in seq_along(w)) {
        later <- which(w > w[i] & w - w[i] <= r$within)
        if (length(later) > 0) {
          out$case <- if (r$date == "first") w[i] else w[min(later)]
          break
        }
      }
    }
  }
  out
}

n <- 400
entry <- numeric(n); exit <- numeric(n); history <- numeric(n)
# Entries on 29 February in a quarter of the persons.
leap_days <- to_day(as.Date(sprintf("%d-02-29", seq(2000, 2024, by = 4))))
for (p in 1:n) {
  entry[p] <- if (p %% 4 == 0) sample(leap_days, 1) else to_day(as.Date("2000-01-01")) + sample(0:9000, 1)
  exit[p] <- entry[p] + sample(c(0, 1, 30, 365, 2000), 1)
  history[p] <- entry[p] - sample(c(30, 200, 400, 1000, 2500, 8000), 1)
}
person <- integer(0); day <- numeric(0); mask <- integer(0)
for (p in 1:n) {
  m <- sample(c(0, 1, 2, 3, 5, 12, 40), 1)
  if (m == 0) next
  # Events cluster on the entry, the exit and the window edges.
  anchors <- c(entry[p], exit[p], entry[p] - 365, entry[p] - 366, entry[p] - 1826, entry[p] - 1827,
               entry[p] - 100, entry[p] - 101, entry[p] + 10, entry[p] + 30, history[p], history[p] - 1)
  d <- ifelse(runif(m) < 0.5, sample(anchors, m, replace = TRUE) + sample(c(-1, 0, 0, 1), m, replace = TRUE),
              entry[p] + sample(-3000:2500, m, replace = TRUE))
  d <- sort(d)
  person <- c(person, rep(p - 1L, m)); day <- c(day, d)
  mask <- c(mask, sample(0:255, m, replace = TRUE))
}
# Persons 0-based, rows sorted by person then day.
ref <- lapply(seq_along(rules), function(k) {
  res <- lapply(1:n, function(p) {
    rows <- which(person == p - 1L)
    evaluate(rules[[k]], k, day[rows], mask[rows], entry[p], exit[p], history[p])
  })
  list(first = sapply(res, function(x) x$first), case = sapply(res, function(x) x$case),
       prior = sapply(res, function(x) x$prior), n = sapply(res, function(x) x$n),
       before = sapply(res, function(x) x$before))
})
cat(toJSON(list(
  r_version = R.version.string,
  entry = entry, exit = exit, history = history,
  person = person, day = day, mask = mask,
  rules = ref
), digits = NA, na = "null", auto_unbox = TRUE))
