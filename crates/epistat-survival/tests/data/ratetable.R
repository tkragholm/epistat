# Reference values for epistat_survival::ratetable from survival::pyears with
# the rate table survexp.us (age 0 to 109 years in days, sex, calendar year
# 1940 to 2020, daily hazards), its calendar year retyped as a plain date.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/ratetable.R
#
# The table itself is written too, so the Rust test builds the same RateTable.
# Cohort A: 2000 persons with delayed entry, ages 0 to 100, entry dates 1945 to
# 2015 and follow-up of up to 12 years, so some follow-up passes the last age
# band and the last calendar year, where R keeps the last cell. Cohort B: 300
# persons who enter between 1935 and 1941, so part of their follow-up is
# before the first calendar year and off the table. For the first 150 persons of
# each cohort, pyears is also called person by person, which gives the expected
# count of each row. Values are written with 17 significant digits.
suppressMessages({
  library(jsonlite)
  library(survival)
})
set.seed(20261122)
rt <- survexp.us
# survexp.us is a table of type 4: pyears shifts the calendar year by the
# person's birthday, so the year cell changes at the birthday and the rates
# are not those of the cell of the date. Retyping the third dimension as a
# plain date gives the cell of the date, which is what RateTable looks up.
attr(rt, "type") <- c(2L, 1L, 3L)
att <- attributes(rt)
age_cuts <- att$cutpoints[[1]]
year_cuts <- as.numeric(att$cutpoints[[3]])

cohort <- function(n, date_lo, date_hi, age_hi) {
  date0 <- as.numeric(as.Date(date_lo)) + floor(runif(n) * (as.numeric(as.Date(date_hi)) - as.numeric(as.Date(date_lo))))
  age0 <- floor(runif(n, 0, age_hi * 365.25))
  start <- floor(runif(n, 0, 2 * 365.25))
  stop <- start + 30 + floor(runif(n, 0, 12 * 365.25))
  sex <- sample(1:2, n, TRUE)
  data.frame(start = start, stop = stop, ev = 0, age0 = age0, date0 = date0, sex = sex)
}
run <- function(d) {
  p <- pyears(Surv(start, stop, ev) ~ 1, data = d,
              rmap = list(age = age0, sex = sex, year = date0),
              ratetable = rt, scale = 1, expect = "event")
  list(pyears = p$pyears, offtable = p$offtable, expected = p$expected)
}
make <- function(name, d) {
  total <- run(d)
  per_row <- sapply(1:150, function(i) run(d[i, ])$expected)
  list(name = name, start = d$start, stop = d$stop, age0 = d$age0, date0 = d$date0,
       sex = d$sex - 1, pyears = total$pyears, offtable = total$offtable,
       expected = total$expected, per_row = per_row)
}
a <- make("cohort A", cohort(2000, "1945-01-01", "2015-12-31", 100))
b <- make("cohort B, entry before the table", cohort(300, "1935-01-01", "1941-12-31", 60))

write(toJSON(list(versions = list(r = R.version.string, survival = as.character(packageVersion("survival"))),
                  age_cuts = I(age_cuts), year_cuts = I(year_cuts), rates = I(as.vector(rt)),
                  cohorts = list(a, b)),
             auto_unbox = TRUE, digits = 17, null = "null", na = "null"),
      "tests/data/ratetable.json")
