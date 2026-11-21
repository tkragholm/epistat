# Risk sets and draws of exposure density sampling, from the R package heaven
# (exposureMatch), as the reference for the exposure density mode of
# epistat-matching. Epi::ccwc is run on the same cohort with the exposure as
# the failure, where the two designs coincide.
#
# The cohort (300 people, fixed seed) has two matching variables (sex and area,
# four cells), staggered entry, exposure for about 45 percent of the people, a
# death or an emigration after the exposure for some of the exposed and before
# the end of follow-up for some of the unexposed, exposures on the same day,
# exposures after the age limit, and people whose death, emigration or entry
# falls exactly on the exposure day of another person in their cell. Times are
# whole days since 1970-01-01.
#
# What heaven is run on.
#  * event is 1 for a person exposed before the age limit (a case). The age
#    limit is 12 years, compared on the birthday as in the engine.
#  * case.index is the exposure day, also for people exposed after the age
#    limit, who are controls until then.
#  * end.followup is the first of the exposure, death, emigration and the day
#    after the end of the study. heaven takes a person as a control of a case
#    when the case index is strictly before end.followup, so a person who dies,
#    emigrates or is exposed on the index day is out, as in the engine.
#  * date.terms is entry - 1, so that entry on or before the index day is the
#    same statement as entry - 1 before it. heaven compares the term with the
#    index day on both sides, so a control has entered exactly when the case
#    has, and every case has entered by its exposure day.
#  * n.controls = 100000 returns every member of every risk set.
# heaven draws the controls of each case independently and without repetition
# inside a case, so a person can be a control for several cases and a person
# exposed later is a control for earlier cases.
#
# The same risk sets from Epi::ccwc(controls = 1000), with the exit of every
# person who is not a case one day before the end of risk, which makes "at risk
# at t" the same statement as in heaven. The script stops if the two differ.
# ccwc puts cases with the same exposure day and cell in one set and draws
# their controls together, so only its sets are compared, not its draws.
#
# Draws: 1500 runs of heaven with one control per case (seeds 1 to 1500) give,
# for every case with at least two people at risk, how often each member of the
# risk set was drawn.
#
# Run from the crate directory:
#   Rscript tests/data/exposure_density.R
# Writes tests/data/exposure_density.json.
suppressMessages({
  library(heaven)
  library(Epi)
  library(data.table)
  library(jsonlite)
})
set.seed(20261121)

n <- 300L
day <- function(x) as.integer(as.Date(x))
sex <- sample(c("1", "2"), n, TRUE)
area <- sample(c("A", "B"), n, TRUE, prob = c(0.55, 0.45))
birth <- day("2000-01-01") + sample(0:3652, n, TRUE)
entry <- pmax(birth, day("2004-01-01")) + sample(0:1800, n, TRUE)
end <- day("2020-12-31")
exposed <- runif(n) < 0.45
expo <- rep(NA_integer_, n)
expo[exposed] <- entry[exposed] + sample(60:2500, sum(exposed), TRUE)
expo[!is.na(expo) & expo > end] <- NA
death <- rep(NA_integer_, n)
emig <- rep(NA_integer_, n)
d <- runif(n) < 0.10
death[d] <- ifelse(is.na(expo[d]), entry[d], expo[d]) + sample(30:3000, sum(d), TRUE)
e <- !d & runif(n) < 0.10
emig[e] <- ifelse(is.na(expo[e]), entry[e], expo[e]) + sample(30:3000, sum(e), TRUE)
death[!is.na(death) & death > end] <- NA
emig[!is.na(emig) & emig > end] <- NA

limit_years <- 12L
limit_day <- function(b) {
  lt <- as.POSIXlt(as.Date(b, origin = "1970-01-01"))
  feb29 <- lt$mon == 1L & lt$mday == 29L
  lt$mday[feb29] <- 28L
  lt$year <- lt$year + limit_years
  as.integer(as.Date(lt))
}
cell <- paste(sex, area)

# Ties: three exposures on one day in cell "1 A", two in "2 B", and an
# exposure after the age limit on the day of a case in "1 B".
lim <- limit_day(birth)
in_age <- !is.na(expo) & expo < lim
over_age <- !is.na(expo) & expo >= lim
# The first candidate keeps its exposure day. The next candidates born before
# that day take it too, up to k rows in all, so that nobody is exposed before
# their own birth.
tie_to <- function(candidates, k) {
  first <- candidates[1]
  later <- candidates[-1]
  later <- later[birth[later] < expo[first]]
  rows <- c(first, later)[seq_len(k)]
  stopifnot(!anyNA(rows))
  expo[rows[-1]] <<- expo[rows[1]]
  entry[rows] <<- pmin(entry[rows], expo[rows[1]] - 1L)
}
tie_to(which(in_age & cell == "1 A"), 3)
tie_to(which(in_age & cell == "2 B"), 2)
tie_to(c(which(in_age & cell == "1 B")[1], which(over_age & cell == "1 B")), 2)
# The entry and the exit of a person must stay in order after a tie moved an
# exposure: a person who dies or emigrates stays out of the risk set from the
# exposure day onward, so a death on or before the new exposure day is moved.
death[!is.na(death) & !is.na(expo) & death <= expo] <- NA
emig[!is.na(emig) & !is.na(expo) & emig <= expo] <- NA
lim <- limit_day(birth)
in_age <- !is.na(expo) & expo < lim

# People whose death, emigration or entry falls exactly on the exposure day of
# a case in their cell. Each takes one role: never exposed, no other exit.
noncase <- is.na(expo) & is.na(death) & is.na(emig)
lone <- which(in_age)
bd <- sample(lone, 12)
for (j in seq_along(bd)) {
  i <- bd[j]
  cand <- which(noncase & cell == cell[i])
  p <- sample(cand, 1)
  noncase[p] <- FALSE
  if (j %% 3 == 1) {
    entry[p] <- min(entry[p], expo[i] - 5L)
    death[p] <- expo[i]
  } else if (j %% 3 == 2) {
    entry[p] <- min(entry[p], expo[i] - 5L)
    emig[p] <- expo[i]
  } else {
    entry[p] <- expo[i]
    death[p] <- NA
    emig[p] <- NA
  }
}
# A case has entered by its exposure day.
stopifnot(all(is.na(expo) | entry <= expo))
risk_end <- pmin(expo, death, emig, end + 1L, na.rm = TRUE)
stopifnot(all(entry < risk_end | entry == risk_end & !is.na(expo)))
case <- as.integer(!is.na(expo) & expo < lim)
stopifnot(sum(case) > 50, any(over_age <- !is.na(expo) & expo >= lim))

ptid <- sprintf("p%03d", seq_len(n))
dat <- data.table(
  ptid = ptid, case = case, sex = sex, area = area,
  case.index = as.numeric(expo), end.followup = as.numeric(risk_end),
  entry_m1 = as.numeric(entry - 1L)
)

hv <- function(k, seed) {
  r <- suppressMessages(exposureMatch(
    ptid = "ptid", event = "case", terms = c("sex", "area"), data = dat,
    n.controls = k, case.index = "case.index", end.followup = "end.followup",
    date.terms = "entry_m1", output.count.controls = FALSE,
    seed = seed, progressbar = FALSE, verbose = FALSE
  ))
  cs <- r[r$case == 1, ]
  co <- r[r$case == 0, ]
  case_ptid <- cs$ptid[match(co$case.id, cs$case.id)]
  split(co$ptid, factor(case_ptid, levels = ptid[case == 1]))
}

case_rows <- which(case == 1L)
full <- hv(100000, 0)
sets_heaven <- lapply(full, function(s) sort(as.character(s)))

# The same sets from ccwc, with the exposure as the failure.
df <- data.frame(
  t_in = entry, t_out = ifelse(case == 1L, expo, risk_end - 1L),
  status = case, sx = sex, ar = area
)
cc <- suppressWarnings(ccwc(
  entry = t_in, exit = t_out, fail = status, origin = 0, controls = 1000,
  match = list(sx, ar), data = df, silent = TRUE
))
cs <- cc[cc$Fail == 1, ]
co <- cc[cc$Fail == 0, ]
by_set <- split(co$Map, co$Set)
sets_ccwc <- lapply(case_rows, function(i) {
  s <- cs$Set[cs$Map == i]
  if (length(s) == 0) character(0) else sort(sprintf("p%03d", by_set[[as.character(s)]]))
})
names(sets_ccwc) <- ptid[case_rows]
for (q in seq_along(case_rows)) {
  stopifnot(identical(sets_heaven[[q]], sets_ccwc[[q]]))
}

# Draw frequencies, one control per case.
reps <- 1500L
members <- Filter(function(s) length(s) >= 2, sets_heaven)
counts <- lapply(members, function(s) integer(length(s)))
for (r in seq_len(reps)) {
  g <- hv(1, r)
  for (nm in names(members)) {
    first <- g[[nm]][1]
    pos <- match(first, members[[nm]])
    stopifnot(!is.na(pos))
    counts[[nm]][pos] <- counts[[nm]][pos] + 1L
  }
}
chi <- vapply(counts, function(cnt) {
  e <- sum(cnt) / length(cnt)
  sum((cnt - e)^2 / e)
}, 0)
draws <- lapply(names(members), function(nm) {
  list(case = nm, members = I(members[[nm]]), counts = I(counts[[nm]]))
})

g2 <- hv(3, 11)
multi <- table(unlist(g2))
future <- sum(unlist(g2) %in% ptid[case == 1])

out <- list(
  versions = list(
    R = R.version.string,
    heaven = as.character(packageVersion("heaven")),
    Epi = as.character(packageVersion("Epi")),
    jsonlite = as.character(packageVersion("jsonlite"))
  ),
  age_limit_years = limit_years,
  reps = reps,
  people = list(
    id = ptid, sex = sex, area = area, birth = birth, entry = entry,
    exposure = expo, death = death, emigration = emig, end = end,
    case = case
  ),
  cases = I(ptid[case_rows]),
  risk_set_heaven = lapply(sets_heaven, I),
  risk_set_ccwc = lapply(sets_ccwc, I),
  draws = draws,
  r_chi_square = unname(chi),
  heaven_k3_controls_in_several_sets = sum(multi > 1),
  heaven_k3_future_cases_as_controls = future
)
write(
  toJSON(out, auto_unbox = TRUE, na = "null", digits = NA),
  file = "tests/data/exposure_density.json"
)
cat("cases", length(case_rows), "with draws", length(members), "\n")
cat("set sizes", range(lengths(sets_heaven)), " empty:", sum(lengths(sets_heaven) == 0), "\n")
cat("over-age exposed", sum(!is.na(expo) & expo >= lim), "\n")
cat("k3 several sets", sum(multi > 1), "future cases", future, "\n")
