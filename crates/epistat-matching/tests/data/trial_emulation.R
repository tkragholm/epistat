# Sequential trial emulation checked against the R package TrialEmulation and
# Epi::ccwc, as the reference for `expand_trials` and `match_trials` in
# epistat-matching.
#
# Three panels.
#  * `data_censored` of TrialEmulation, all 725 rows.
#  * `trial_example` of TrialEmulation, the first 12 people by id.
#  * A simulated panel of 316 children observed over 12 periods of 90 days,
#    with sex and area as matching variables, staggered entry, deaths and
#    emigrations that end the observation, treatment that starts, stops and
#    restarts, an outcome that ends the follow-up, and eligibility that is
#    "never treated before" for 70 percent of the people and "observed" for
#    the rest.
#
# Expansion. For every panel the script runs TrialEmulation 0.0.5 as an
# intention-to-treat sequence (set_data on a trial_sequence "ITT", then
# expand_trials) and as a per-protocol sequence ("PP"). For each person and
# trial the reference keeps the number of rows of both, the assigned treatment
# (assigned_treatment of the expanded data) and the follow-up time of the
# outcome. The script stops unless the follow-up times of each person and
# trial run 0, 1, ..., n - 1 without gaps, so the counts describe every row.
# The per-protocol rows of the package are checked here to be the
# intention-to-treat rows cut before the first change of treatment.
#
# Matching. On the simulated panel, in every trial, the risk set of each
# initiator is the set of people enrolled in that trial who do not initiate,
# agree on sex and area, have entered on or before the first day of the trial
# and are alive and resident on it. Initiators and people who initiate after
# the age limit are not controls. The sets are computed from Epi::ccwc (the
# calendar scale, the failure at the first day of the trial, controls = 1000)
# and, independently, directly from the data. The script stops unless the two
# are equal. An initiator older than the age limit (12 years, compared on the
# birthday) is not a case. A person who is not a case leaves on the day before
# the end of risk, which makes "at risk at t" the same statement as in ccwc.
#
# Draws. For the four initiators of the cell "3 C", each alone among the
# initiators of its trial and cell, 600 runs of ccwc with controls = 1 (seeds 1 to 600) give how
# often each member of the set was drawn.
#
# Run from the crate directory:
#   Rscript tests/data/trial_emulation.R
# Writes tests/data/trial_emulation.json.
suppressMessages({
  library(TrialEmulation)
  library(Epi)
  library(data.table)
  library(jsonlite)
})
set.seed(20261122)

expand_te <- function(panel, estimand) {
  ts <- trial_sequence(estimand)
  ts <- suppressWarnings(set_data(ts, copy(as.data.table(panel))))
  ts <- set_outcome_model(ts, adjustment_terms = ~1)
  ts <- set_expansion_options(ts, output = save_to_datatable(), chunk_size = 0)
  ts <- expand_trials(ts)
  x <- as.data.table(ts@expansion@datastore@data)
  setorder(x, id, trial_period, followup_time)
  x
}

# Per person and trial: rows of the PP and of the ITT expansion.
summarise_expansion <- function(panel) {
  panel <- as.data.table(panel)
  itt <- expand_te(panel, "ITT")
  pp <- expand_te(panel, "PP")
  for (x in list(itt, pp)) {
    ok <- x[, all(followup_time == seq_len(.N) - 1L), by = .(id, trial_period)]$V1
    stopifnot(all(ok))
  }
  # The per-protocol rows are the intention-to-treat rows before the first
  # change of treatment.
  itt[, chg := c(0, treatment[-1] != treatment[-.N]), by = .(id, trial_period)]
  cut <- itt[, .SD[cumsum(chg) == 0], by = .(id, trial_period)]
  stopifnot(nrow(cut) == nrow(pp))
  stopifnot(all(cut$id == pp$id), all(cut$trial_period == pp$trial_period),
            all(cut$followup_time == pp$followup_time))
  a <- itt[, .(itt_rows = .N, assigned = assigned_treatment[1],
               outcome_at = if (any(outcome == 1)) followup_time[outcome == 1][1] else NA_integer_),
           by = .(id, trial_period)]
  b <- pp[, .(pp_rows = .N,
              pp_outcome_at = if (any(outcome == 1)) followup_time[outcome == 1][1] else NA_integer_),
          by = .(id, trial_period)]
  m <- merge(a, b, by = c("id", "trial_period"))
  stopifnot(nrow(m) == nrow(a))
  setorder(m, id, trial_period)
  m
}
cols <- function(m) lapply(as.list(m), function(v) I(v))
panel_cols <- function(p) list(
  id = I(p$id), period = I(p$period), eligible = I(p$eligible),
  treatment = I(p$treatment), outcome = I(p$outcome)
)

# 1. data_censored.
dc <- as.data.table(data_censored)[, .(id, period, eligible, treatment, outcome)]
dc_exp <- summarise_expansion(data_censored)

# 2. trial_example, first 12 ids.
ids <- sort(unique(trial_example$id))[1:12]
te <- as.data.table(trial_example)[id %in% ids, .(id, period, eligible, treatment, outcome)]
te_exp <- summarise_expansion(trial_example[trial_example$id %in% ids, ])

# 3. The simulated panel.
n0 <- 300L
n <- 316L
steps <- 0:11
day <- function(x) as.integer(as.Date(x))
day0 <- day("2015-01-01")
step_days <- 90L
pdate <- day0 + step_days * steps
sex <- sample(c("1", "2"), n, TRUE)
area <- sample(c("A", "B"), n, TRUE, prob = c(0.55, 0.45))
birth <- day("2005-01-01") + sample(0:2800, n, TRUE)
entry <- pmax(birth, day("2014-10-01")) + sample(0:700, n, TRUE)
end <- pdate[length(pdate)] + step_days
death <- rep(NA_integer_, n)
emig <- rep(NA_integer_, n)
d <- runif(n) < 0.10
death[d] <- day0 + sample(60:1000, sum(d), TRUE)
e <- !d & runif(n) < 0.12
emig[e] <- day0 + sample(60:1000, sum(e), TRUE)
# A death, an emigration and an entry on the first day of a period.
death[1] <- pdate[5]; emig[1] <- NA; entry[1] <- pdate[1]
emig[2] <- pdate[6]; death[2] <- NA; entry[2] <- pdate[1]
entry[3] <- pdate[7]; death[3] <- NA; emig[3] <- NA
birth[3] <- min(birth[3], pdate[7] - 3000L)
# A cell of its own ("3 C", 16 people) in which one person starts treatment
# in each of the periods 2, 4, 6 and 8 and the rest never do, so each of the
# four initiators is alone among the initiators of its trial and cell. All are
# observed throughout.
extra <- (n0 + 1L):n
sex[extra] <- "3"; area[extra] <- "C"
birth[extra] <- day("2008-01-01") + sample(0:700, length(extra))
entry[extra] <- day0 - 30L; death[extra] <- NA; emig[extra] <- NA
newuser <- runif(n) < 0.7
newuser[extra] <- TRUE
risk_end <- pmin(death, emig, end, na.rm = TRUE)
rows <- list()
for (i in seq_len(n)) {
  obs <- which(pdate >= entry[i] & pdate < risk_end[i])
  if (length(obs) == 0) next
  tr <- integer(length(obs))
  on <- FALSE
  start <- runif(1) < 0.8
  for (k in seq_along(obs)) {
    if (!on && start && runif(1) < 0.25) on <- TRUE
    else if (on && runif(1) < 0.18) on <- FALSE
    else if (!on && start && k > 1 && runif(1) < 0.10) on <- TRUE
    tr[k] <- as.integer(on)
  }
  ev <- as.integer(runif(length(obs)) < 0.02)
  if (i > n0) {
    k <- i - n0
    tr <- if (k <= 4) as.integer(steps[obs] >= 2 * k) else integer(length(obs))
    ev <- integer(length(obs))
  }
  if (any(ev == 1)) ev[seq_along(ev) > which(ev == 1)[1]] <- 0L
  prior <- c(0L, cumsum(tr)[-length(tr)])
  elig <- if (newuser[i]) as.integer(prior == 0L) else rep(1L, length(obs))
  rows[[i]] <- data.frame(
    id = sprintf("p%03d", i), idn = i, period = steps[obs], eligible = elig,
    treatment = tr, outcome = ev
  )
}
sim <- rbindlist(rows)
# TrialEmulation needs numeric ids.
sim_te <- copy(sim)[, id := idn][, idn := NULL]
sim_exp <- summarise_expansion(sim_te)
sim_exp[, idc := sprintf("p%03d", id)]

limit_years <- 12L
limit_day <- function(b) {
  lt <- as.POSIXlt(as.Date(b, origin = "1970-01-01"))
  feb29 <- lt$mon == 1L & lt$mday == 29L
  lt$mday[feb29] <- 28L
  lt$year <- lt$year + limit_years
  as.integer(as.Date(lt))
}
lim <- limit_day(birth)
cell <- paste(sex, area)
final_end <- end + 1L

trials <- sort(unique(sim_exp$trial_period))
risk_sets <- list()
alone <- list()
for (tp in trials) {
  enr <- sim_exp[trial_period == tp]
  idx <- enr$id
  t0 <- pdate[tp + 1]
  init <- enr$assigned == 1
  case <- init & t0 < lim[idx]
  if (!any(case)) next
  dd <- death[idx]; ee <- emig[idx]
  end_i <- pmin(ifelse(init, t0, NA_integer_), dd, ee, final_end, na.rm = TRUE)
  df <- data.frame(
    t_in = entry[idx], t_out = ifelse(case, t0, end_i - 1L),
    status = as.integer(case), sx = sex[idx], ar = area[idx]
  )
  stopifnot(all(df$t_in <= df$t_out | df$status == 1L))
  cc <- suppressWarnings(ccwc(
    entry = t_in, exit = t_out, fail = status, origin = 0, controls = 1000,
    match = list(sx, ar), data = df, silent = TRUE
  ))
  cs <- cc[cc$Fail == 1, ]
  co <- cc[cc$Fail == 0, ]
  by_set <- split(co$Map, co$Set)
  for (q in which(case)) {
    s <- cs$Set[cs$Map == q]
    got <- if (length(s) == 0) integer(0) else sort(by_set[[as.character(s)]])
    # The register definition.
    ok <- !init & cell[idx] == cell[idx[q]] & entry[idx] <= t0 &
      (is.na(dd) | dd > t0) & (is.na(ee) | ee > t0)
    mine <- sort(which(ok))
    stopifnot(identical(as.integer(got), as.integer(mine)))
    risk_sets[[sprintf("%d:%s", tp, enr$idc[q])]] <- I(sprintf("p%03d", idx[got]))
  }
  # Initiators of the cell "3 C", for the draw frequencies.
  tab <- table(cell[idx][case])
  for (q in which(case)) {
    key <- sprintf("%d:%s", tp, enr$idc[q])
    if (cell[idx[q]] == "3 C" && tab[[cell[idx[q]]]] == 1 && length(risk_sets[[key]]) >= 2) {
      alone[[length(alone) + 1]] <- list(trial = tp, q = q, key = key)
    }
  }
}
stopifnot(length(risk_sets) > 100)

chosen <- alone
stopifnot(length(chosen) == 4)
reps <- 600L
draws <- list()
for (a in chosen) {
  tp <- a$trial
  enr <- sim_exp[trial_period == tp]
  idx <- enr$id
  t0 <- pdate[tp + 1]
  init <- enr$assigned == 1
  case <- init & t0 < lim[idx]
  end_i <- pmin(ifelse(init, t0, NA_integer_), death[idx], emig[idx], final_end, na.rm = TRUE)
  df <- data.frame(
    t_in = entry[idx], t_out = ifelse(case, t0, end_i - 1L),
    status = as.integer(case), sx = sex[idx], ar = area[idx]
  )
  members <- risk_sets[[a$key]]
  counts <- integer(length(members))
  for (r in seq_len(reps)) {
    set.seed(r)
    cc <- suppressWarnings(ccwc(
      entry = t_in, exit = t_out, fail = status, origin = 0, controls = 1,
      match = list(sx, ar), data = df, silent = TRUE
    ))
    s <- cc$Set[cc$Fail == 1 & cc$Map == a$q]
    got <- cc$Map[cc$Fail == 0 & cc$Set == s]
    stopifnot(length(got) == 1)
    pos <- match(sprintf("p%03d", idx[got]), members)
    stopifnot(!is.na(pos))
    counts[pos] <- counts[pos] + 1L
  }
  draws[[length(draws) + 1]] <- list(
    key = a$key, members = I(members), counts = I(counts)
  )
}
chi <- vapply(draws, function(d) {
  e <- sum(d$counts) / length(d$counts)
  sum((d$counts - e)^2 / e)
}, 0)

person <- list(
  id = I(sprintf("p%03d", seq_len(n))), sex = I(sex), area = I(area),
  birth = I(birth), entry = I(entry), death = I(death), emigration = I(emig)
)
out <- list(
  versions = list(
    R = R.version.string,
    TrialEmulation = as.character(packageVersion("TrialEmulation")),
    Epi = as.character(packageVersion("Epi")),
    jsonlite = as.character(packageVersion("jsonlite"))
  ),
  age_limit_years = limit_years,
  day0 = day0, step_days = step_days, reps = reps,
  data_censored = list(panel = panel_cols(dc), expansion = cols(dc_exp)),
  trial_example = list(panel = panel_cols(te), expansion = cols(te_exp)),
  simulated = list(
    panel = panel_cols(sim[, .(id = idn, period, eligible, treatment, outcome)]),
    people = person,
    expansion = cols(sim_exp[, .(id, trial_period, itt_rows, assigned, outcome_at,
                                  pp_rows, pp_outcome_at)])
  ),
  risk_sets = risk_sets,
  draws = draws,
  r_chi_square = I(unname(chi))
)
write(toJSON(out, auto_unbox = TRUE, na = "null", digits = NA),
      file = "tests/data/trial_emulation.json")
cat("enrolments", nrow(sim_exp), "initiator sets", length(risk_sets), "draw cases",
    length(draws), "\n")
cat("people with deviation under PP:", sim_exp[pp_rows < itt_rows, .N], "\n")
