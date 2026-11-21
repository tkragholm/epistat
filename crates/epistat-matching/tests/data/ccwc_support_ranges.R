# Range of a score over the people at risk at each case's index date, with the
# risk sets taken from Epi::ccwc, as the reference for common-support trimming
# in risk-set matching in epistat-matching.
#
# A simulated cohort of 220 people (fixed seed) enters at staggered dates and
# 90 of them fail on a day in a short window, so failure days tie. Scores are
# rounded to whole numbers, so scores tie with the ends of the ranges. Times are
# whole days since 1970-01-01.
#
# ccwc is run without matching variables, so each case's risk set is every
# person at risk at its failure time. It keeps a person with exit equal to the
# failure time in the risk set, so everyone who does not fail as a case is
# given exit = end of follow-up - 1 day and a case exit = its failure day, as
# in ccwc_risk_sets.R. The whole risk set is returned (controls = 1000). The
# risk set is checked against the definition written directly from the data
# (entered on or before the failure day, event strictly after it).
#
# For each case the script records the smallest and largest score in its risk
# set (null when the set is empty), and whether the case's own score is
# strictly outside that range. The range of the scores of all cases is
# recorded as well.
#
# Run from the crate directory:
#   Rscript tests/data/ccwc_support_ranges.R
# Writes tests/data/ccwc_support_ranges.json.
suppressMessages({
  library(Epi)
  library(jsonlite)
})
set.seed(20261121)

n <- 220L
day <- function(x) as.integer(as.Date(x))
entry <- day("2005-01-01") + sample(0:1500, n, TRUE)
end <- day("2012-12-31")
is_case <- runif(n) < 0.41
event <- rep(NA_integer_, n)
event[is_case] <- day("2005-02-01") + sample(0:300, sum(is_case), TRUE) * 5L
event[!is.na(event) & event <= entry] <- NA
is_case <- !is.na(event)
score <- round(rnorm(n, sd = 1.3))

df <- data.frame(
  id = seq_len(n), t_in = entry,
  t_out = ifelse(is_case, event, end - 1L),
  status = as.integer(is_case), score = score
)
res <- suppressWarnings(ccwc(
  entry = t_in, exit = t_out, fail = status, origin = 0, controls = 1000,
  data = df, silent = TRUE
))
cs <- res[res$Fail == 1, ]
co <- res[res$Fail == 0, ]
case_rows <- which(is_case)
sets <- lapply(case_rows, function(i) {
  s <- cs$Set[match(i, cs$Map)]
  sort(co$Map[co$Set == s])
})
direct <- lapply(case_rows, function(i) {
  sort(which(entry <= event[i] & (is.na(event) | event > event[i])))
})
stopifnot(identical(lapply(sets, as.integer), lapply(direct, as.integer)))

lo <- vapply(sets, function(s) if (length(s)) min(score[s]) else NA_real_, 0)
hi <- vapply(sets, function(s) if (length(s)) max(score[s]) else NA_real_, 0)
own <- score[case_rows]
outside <- !is.na(lo) & (own < lo | own > hi)
all_cases <- range(own)

out <- list(
  versions = list(R = R.version.string, Epi = as.character(packageVersion("Epi"))),
  entry = entry, event = event, score = score,
  cases = case_rows, set_size = lengths(sets),
  low = lo, high = hi, outside = outside,
  case_range = all_cases,
  at_bounds = sum(!is.na(lo) & (own == lo | own == hi))
)
write_json(out, "tests/data/ccwc_support_ranges.json", digits = NA, auto_unbox = TRUE, na = "null")
cat("cases", length(case_rows), "outside", sum(outside), "at bounds", out$at_bounds, "empty", sum(lengths(sets) == 0), "\n")
