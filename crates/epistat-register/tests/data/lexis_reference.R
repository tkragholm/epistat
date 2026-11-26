# Reference pieces for epistat-register::lexis from Epi::splitLexis and
# popEpi::splitMulti, with calendar time and age both counted in whole days
# (R's scales are numeric, so day counts are exact). Age cuts are at
# ceiling(1461 * a / 4) days, the day a person reaches age a under the
# 365.25-day rule, since a civil birthday is not a fixed number of days. Run
# from the crate directory:
#   Rscript tests/data/lexis_reference.R > tests/data/lexis_reference.json
suppressMessages({library(jsonlite); library(Epi); library(popEpi)})
set.seed(20261123)
epoch <- as.Date("1970-01-01")
day <- function(y, m, d) as.numeric(as.Date(sprintf("%d-%02d-%02d", y, m, d)) - epoch)

n <- 400
birth <- day(1940, 1, 1) + sample(0:22000, n, replace = TRUE)
entry <- birth + sample(0:20000, n, replace = TRUE)
exit <- entry + sample(0:9000, n, replace = TRUE)
# Edge cases: follow-up from a new year to a new year, from a birthday on
# the age scale, and an interval of one day.
entry[1:3] <- day(2001, 1, 1); exit[1:3] <- c(day(2003, 1, 1), day(2002, 1, 1) + 1, day(2001, 1, 2))
ages <- c(5, 18, 40, 65)
age_cut <- ceiling(1461 * ages / 4)
entry[4] <- birth[4] + age_cut[2]; exit[4] <- entry[4] + 2000
entry[5] <- birth[5]; exit[5] <- birth[5] + age_cut[2]
years <- 1940:2100
per_cut <- sapply(years, function(y) day(y, 1, 1))
keep <- exit > entry
id <- seq_len(n)

# Epi prints a note about entry.status on standard output.
invisible(capture.output(lex <- Lexis(entry = list(per = entry[keep], age = (entry - birth)[keep]),
             exit = list(per = exit[keep]), exit.status = 1L, id = id[keep])))
by_epi <- splitLexis(splitLexis(lex, breaks = per_cut, time.scale = "per"),
                     breaks = age_cut, time.scale = "age")
# The label of a piece is the left end of its band on each scale, NA below
# the first break: the year of the calendar band and the age of the age band.
pieces <- function(x) {
  per_left <- timeBand(x, "per", "left")
  age_left <- timeBand(x, "age", "left")
  d <- data.frame(id = x$lex.id, start = x$per, dur = x$lex.dur,
                  year = years[match(per_left, per_cut)],
                  age = ages[match(age_left, age_cut)])
  d[order(d$id, d$start), ]
}
e <- pieces(by_epi)
pm <- splitMulti(lex, per = per_cut, age = age_cut, drop = FALSE)
p <- pieces(pm)
stopifnot(all(e$id == p$id), all(e$start == p$start), all(e$dur == p$dur))
cat(toJSON(list(
  versions = list(R = as.character(getRversion()), Epi = as.character(packageVersion("Epi")),
                  popEpi = as.character(packageVersion("popEpi"))),
  birth = birth, start = entry, end = exit, ages = ages,
  epi = list(id = e$id - 1, start = e$start, dur = e$dur, year = e$year, age = e$age),
  popepi_agrees = TRUE
), digits = NA, auto_unbox = TRUE, na = "null"))
