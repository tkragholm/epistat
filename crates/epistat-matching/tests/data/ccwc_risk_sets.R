# Risk sets and sampling of Epi::ccwc on a small simulated register cohort, as
# the reference for risk-set matching in epistat-matching.
#
# The cohort (400 people, fixed seed) has two matching variables (sex and
# area, four cells), staggered entry, deaths and emigrations, tied failure
# times in one cell, tied failure times across cells, a person over the age
# limit who fails on the day of a case, and people whose death, emigration or
# entry falls exactly on a case's failure day. Times are whole days since
# 1970-01-01.
#
# What ccwc is run on. Our engine puts a person out of the risk set on the day
# of their death, emigration or first event. ccwc keeps a person with
# exit == failure time in the risk set. The data given to ccwc therefore has
# exit = (end of risk) - 1 day for everyone who does not fail as a case, and
# exit = the failure day for a case, which makes "at risk at t" the same
# statement in both. The unshifted version (exit = end of risk) is run as well
# and its differences are recorded under `raw`.
#
# A person fails as a case when the event falls before the sixth birthday
# (the age limit); an event on or after it is a censoring in ccwc and an
# ineligible case in the engine, and the person leaves the risk set on that day
# in both.
#
# For each time scale (calendar days with origin 0, and age in days with
# origin = date of birth):
#  * the risk set of every case from ccwc with controls = 1000 (the whole risk
#    set is returned when it has fewer people than that),
#  * the same set from the register definition, written directly from the data
#    (same cell, entered on or before the failure time, still at risk after it,
#    not the case itself), checked equal to ccwc's here in the script,
#  * the warnings and the number of controls ccwc returns with controls = 4.
# On the calendar scale, 2000 runs of ccwc with controls = 1 (seeds
# 20261200 + r) give, for each case that is alone at its failure time, how
# often each member of its risk set was drawn, and the union over the runs of
# everyone ever drawn (checked equal to the risk set here).
#
# Run from the crate directory:
#   Rscript tests/data/ccwc_risk_sets.R
# Writes tests/data/ccwc_risk_sets.json.
suppressMessages({
  library(Epi)
  library(jsonlite)
})
set.seed(20261120)

n <- 400L
day <- function(x) as.integer(as.Date(x))
sex <- sample(c("1", "2"), n, TRUE)
area <- sample(c("A", "B"), n, TRUE, prob = c(0.55, 0.45))
birth <- day("2000-01-01") + sample(0:3652, n, TRUE)
entry <- pmax(birth, day("2004-01-01")) + sample(0:1800, n, TRUE)
end <- day("2020-12-31")
event <- rep(NA_integer_, n)
is_case <- runif(n) < 0.32
event[is_case] <- entry[is_case] + sample(60:2500, sum(is_case), TRUE)
event[!is.na(event) & event > end] <- NA
death <- rep(NA_integer_, n)
emig <- rep(NA_integer_, n)
d <- is.na(event) & runif(n) < 0.08
death[d] <- entry[d] + sample(30:3000, sum(d), TRUE)
e <- is.na(event) & runif(n) < 0.10
emig[e] <- entry[e] + sample(30:3000, sum(e), TRUE)
death[!is.na(death) & death > end] <- NA
emig[!is.na(emig) & emig > end] <- NA

# A small cell ("2 C", 14 people, all entering 2006-06-01) built so that risk
# sets shrink to a few people: rows 1-4 fail at 100, 300, 500, 700 days after
# entry, rows 5 and 6 fail together at 900, row 7 at 1200 and row 8 at 1500;
# rows 9-11 leave early and rows 12-14 at 1050, 1100 and 1300 days. The
# failure at 900 has 5 people at risk (fewer than 2 x 4, more than 4), the one
# at 1200 has 2 and the one at 1500 has none.
c_rows <- 1:14
sex[c_rows] <- "2"
area[c_rows] <- "C"
birth[c_rows] <- day("2005-01-01") + (0:13) * 3L
entry[c_rows] <- day("2006-06-01")
event[c_rows] <- NA_integer_
death[c_rows] <- NA_integer_
emig[c_rows] <- NA_integer_
e0 <- day("2006-06-01")
event[1:8] <- e0 + c(100L, 300L, 500L, 700L, 900L, 900L, 1200L, 1500L)
emig[9] <- e0 + 200L
death[10] <- e0 + 400L
emig[11] <- e0 + 600L
emig[12] <- e0 + 1050L
death[13] <- e0 + 1100L
emig[14] <- e0 + 1300L

limit_day <- function(b) {
  lt <- as.POSIXlt(as.Date(b, origin = "1970-01-01"))
  feb29 <- lt$mon == 1L & lt$mday == 29L
  lt$mday[feb29] <- 28L
  lt$year <- lt$year + 6L
  as.integer(as.Date(lt))
}
cell <- paste(sex, area)

# Ties: three cases in cell "1 A", two in "2 B", two in different cells, and
# a person over the age limit who fails on the day of a case in "1 B".
# The first candidate keeps its event day. The next candidates born before that
# day take it too, up to k rows in all, so that nobody has an event before
# their own birth.
tie_to <- function(candidates, k) {
  first <- candidates[1]
  later <- candidates[-1]
  later <- later[birth[later] < event[first]]
  rows <- c(first, later)[seq_len(k)]
  stopifnot(!anyNA(rows))
  event[rows[-1]] <<- event[rows[1]]
  entry[rows] <<- pmin(entry[rows], event[rows[1]] - 1L)
}
lim <- limit_day(birth)
in_age <- !is.na(event) & event < lim
over_age <- !is.na(event) & event >= lim
tie_to(which(in_age & cell == "1 A"), 3)
tie_to(which(in_age & cell == "2 B"), 2)
a <- which(in_age & cell == "1 B")
b <- which(in_age & cell == "2 A")
tie_to(c(a[3], b[3:length(b)]), 2)
x <- which(in_age & cell == "1 B")[4]
y <- which(over_age & cell == "1 B")
tie_to(c(x, y), 2)
# After tying, a case's age status can have changed; recompute.
lim <- limit_day(birth)
in_age <- !is.na(event) & event < lim

# Boundary persons for the cases that are alone at their time: death,
# emigration or entry exactly on the failure day.
lone <- which(in_age & cell != "2 C" & !(event %in% event[duplicated(event)]))
bd <- sample(lone, 9)
noncase <- is.na(event)
for (j in seq_along(bd)) {
  i <- bd[j]
  cand <- which(noncase & cell == cell[i] & is.na(death) & is.na(emig))
  p <- sample(cand, 1)
  noncase[p] <- FALSE # one role each
  if (j %% 3 == 1) {
    entry[p] <- min(entry[p], event[i] - 5L)
    death[p] <- event[i]
  } else if (j %% 3 == 2) {
    entry[p] <- min(entry[p], event[i] - 5L)
    emig[p] <- event[i]
  } else {
    entry[p] <- event[i]
  }
}

# Everyone must have entered before leaving the risk set.
risk_end <- pmin(event, death, emig, end + 1L, na.rm = TRUE)
stopifnot(all(entry < risk_end))
fail <- as.integer(!is.na(event) & event < lim)
stopifnot(sum(fail) > 60, any(!is.na(event) & event >= lim))
exit_map <- ifelse(fail == 1L, event, risk_end - 1L)
exit_raw <- ifelse(fail == 1L, event, risk_end)
# A case with an age-limit censoring has fail 0 and exit_raw = event.

df <- data.frame(
  t_in = entry, t_out = exit_map, t_raw = exit_raw, status = fail,
  dob = birth, sx = sex, ar = area
)
case_rows <- which(fail == 1L)
k_small <- 4L

# ccwc evaluates its arguments in `df`, so the exit column is copied to
# df$exit_sel before each call.
run_by_name <- function(col, k, age = FALSE) {
  df$exit_sel <- df[[col]]
  warns <- character()
  res <- withCallingHandlers(
    if (age) {
      ccwc(
        entry = t_in, exit = exit_sel, fail = status, origin = dob,
        controls = k, match = list(sx, ar), data = df, silent = TRUE
      )
    } else {
      ccwc(
        entry = t_in, exit = exit_sel, fail = status, origin = 0,
        controls = k, match = list(sx, ar), data = df, silent = TRUE
      )
    },
    warning = function(w) {
      warns <<- c(warns, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  list(res = res, warnings = warns)
}

sets_from <- function(res) {
  cs <- res[res$Fail == 1, ]
  co <- res[res$Fail == 0, ]
  by_set <- split(co$Map, co$Set)
  lapply(case_rows, function(i) {
    s <- cs$Set[cs$Map == i]
    if (length(s) == 0) integer(0) else sort(unname(by_set[[as.character(s)]]))
  })
}

# Register definition of the risk set of case i at time scale `age`.
register_set <- function(i, age) {
  t_i <- if (age) event[i] - birth[i] else event[i]
  ok <- cell == cell[i] & seq_len(n) != i
  if (age) {
    at <- birth + t_i
  } else {
    at <- rep(t_i, n)
  }
  as.integer(which(ok & entry <= at & at < risk_end))
}

scale_block <- function(age) {
  big <- run_by_name("t_out", 1000, age)
  s_big <- sets_from(big$res)
  s_def <- lapply(case_rows, register_set, age = age)
  stopifnot(identical(lapply(s_big, as.integer), lapply(s_def, as.integer)))
  k4 <- run_by_name("t_out", k_small, age)
  raw <- run_by_name("t_raw", 1000, age)
  s_raw <- sets_from(raw$res)
  # Tie structure on this scale.
  tt <- if (age) event - birth else event
  tie_key <- paste(cell, tt)[case_rows]
  ncase_set <- as.integer(table(tie_key)[tie_key])
  m <- vapply(s_def, length, 1L)
  cs <- k4$res[k4$res$Fail == 1, ]
  co <- k4$res[k4$res$Fail == 0, ]
  list(
    cases = I(case_rows),
    risk_set = lapply(s_def, I),
    ncase_at_time = ncase_set,
    raw_extra = Map(function(r, s) I(setdiff(r, s)), s_raw, s_big),
    raw_missing = Map(function(r, s) I(setdiff(s, r)), s_raw, s_big),
    k4_warnings = I(k4$warnings),
    k4_controls_returned = vapply(case_rows, function(i) {
      s <- cs$Set[cs$Map == i]
      if (length(s) == 0) 0L else sum(co$Set == s)
    }, 1L),
    k4_controls_in_several_sets = sum(table(co$Map) > 1),
    k4_future_cases_as_controls = sum(fail[co$Map] == 1L),
    big_warnings = big$warnings,
    size_check = identical(m, vapply(s_big, length, 1L))
  )
}

cal <- scale_block(FALSE)
agb <- scale_block(TRUE)

# Draw frequencies, calendar scale, controls = 1.
lone_pos <- which(cal$ncase_at_time == 1L & vapply(cal$risk_set, length, 1L) >= 2L)
reps <- 2000L
draws <- matrix(NA_integer_, reps, length(lone_pos))
seen <- vector("list", length(case_rows))
for (q in seq_along(seen)) seen[[q]] <- integer(0)
df$exit_sel <- df$t_out
for (r in seq_len(reps)) {
  set.seed(20261200L + r)
  res <- suppressWarnings(ccwc(
    entry = t_in, exit = exit_sel, fail = status, origin = 0, controls = 1,
    match = list(sx, ar), data = df, silent = TRUE
  ))
  cs <- res[res$Fail == 1, ]
  co <- res[res$Fail == 0, ]
  s <- cs$Set[match(case_rows, cs$Map)]
  first <- co$Map[match(s, co$Set)]
  draws[r, ] <- first[lone_pos]
  by_set <- split(co$Map, co$Set)
  for (q in seq_along(case_rows)) {
    if (!is.na(s[q])) {
      seen[[q]] <- union(seen[[q]], by_set[[as.character(s[q])]])
    }
  }
}
stopifnot(identical(
  lapply(seen, function(v) sort(as.integer(v))),
  lapply(cal$risk_set, as.integer)
))
freq <- lapply(seq_along(lone_pos), function(q) {
  mem <- cal$risk_set[[lone_pos[q]]]
  list(
    case = case_rows[lone_pos[q]],
    members = I(mem),
    counts = I(as.integer(table(factor(draws[, q], levels = mem))))
  )
})
chi <- vapply(freq, function(f) {
  e <- sum(f$counts) / length(f$counts)
  sum((f$counts - e)^2 / e)
}, 0)

out <- list(
  versions = list(
    R = R.version.string,
    Epi = as.character(packageVersion("Epi")),
    jsonlite = as.character(packageVersion("jsonlite"))
  ),
  age_limit_years = 6L,
  k_small = k_small,
  reps = reps,
  people = list(
    sex = sex, area = area, birth = birth, entry = entry,
    event = event, death = death, emigration = emig, end = end,
    in_age = as.integer(fail)
  ),
  calendar = cal,
  age = agb,
  draws = freq,
  r_chi_square = chi
)
write(
  toJSON(out, auto_unbox = TRUE, na = "null", digits = NA),
  file = "tests/data/ccwc_risk_sets.json"
)
cat(
  "cases", length(case_rows), "tied sets",
  sum(cal$ncase_at_time > 1), "lone with draws", length(lone_pos), "\n"
)
cat("raw extra total", sum(lengths(cal$raw_extra)), "\n")
cat("k4 warnings:", cal$k4_warnings, "\n")
cat("max risk set", max(lengths(cal$risk_set)), "min", min(lengths(cal$risk_set)), "\n")
cat("n risk < 4:", sum(lengths(cal$risk_set) < 4), " == 0:", sum(lengths(cal$risk_set) == 0), "\n")
