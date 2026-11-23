# Exposure periods from heaven::medicinMacro (type = "dynamic") for the
# settings in which its rules and those of epistat-register::exposure coincide:
# one drug, one strength, minimum, maximum and default dose all equal, so every
# supply lasts quantity / dose days, with and without admissions that pause the
# supply and with and without a cap on the stock. Run from the crate directory:
#   Rscript tests/data/heaven_reference.R > tests/data/heaven_reference.json
suppressMessages({library(heaven); library(data.table); library(jsonlite)})
set.seed(20261123)
n <- 60
dose <- 2
rows <- list()
adms <- list()
for (p in 0:(n - 1)) {
  k <- sample(1:10, 1)
  t <- 10000 + sample(0:300, 1)
  for (i in 1:k) {
    rows[[length(rows) + 1]] <- data.frame(id = p, day = t,
      q = dose * sample(c(10, 20, 30, 45, 60, 120), 1))
    t <- t + sample(c(0, 5, 10, 15, 20, 30, 45, 60, 90, 150), 1)
  }
  for (j in seq_len(sample(0:2, 1))) {
    a <- 10000 + sample(0:700, 1)
    adms[[length(adms) + 1]] <- data.frame(id = p, s = a, e = a + sample(1:25, 1))
  }
}
d <- do.call(rbind, rows)
a <- do.call(rbind, adms)
# Overlapping or duplicated admissions are not allowed by heaven.
a <- a[order(a$id, a$s), ]
keep <- rep(TRUE, nrow(a))
for (i in seq_len(nrow(a))[-1]) {
  if (a$id[i] == a$id[i - 1] && a$s[i] <= max(a$e[which(keep[seq_len(i - 1)] & a$id[seq_len(i - 1)] == a$id[i])])) keep[i] <- FALSE
}
a <- a[keep, ]
origin <- as.Date("1970-01-01")
dd <- data.table(pnr = d$id, eksd = origin + d$day, strnum = 1, packsize = 1, apk = d$q, atc = "X")
adm_all <- data.table(pnr = a$id, first.indate = origin + a$s, last.outdate = origin + a$e)
adm_none <- adm_all[0]

# heaven pairs the admissions of a data set with its persons by position, and
# when some persons have none the admissions of others are applied to them, so
# each person is run alone, with an admission in 2090 standing in for none.
run_person <- function(p, depot_units, adm) {
  dr <- list(X = list(atc = "X", maxdepot = depot_units,
    period = as.Date(c("1990-01-01", "2090-12-31")),
    doses = list(value = 1, min = dose, max = dose, def = dose)))
  adm_p <- adm[pnr == p]
  if (nrow(adm_p) == 0) adm_p <- data.table(pnr = p, first.indate = as.Date("2090-01-01"), last.outdate = as.Date("2090-01-02"))
  medicinMacro(dr, dd[pnr == p], adm_p)$X
}
run <- function(depot_units, adm) {
  r <- rbindlist(lapply(unique(dd$pnr), run_person, depot_units = depot_units, adm = adm))
  r <- r[dose > 0]
  r <- data.table(id = r$pnr, s = as.numeric(r$firstday - origin), e = as.numeric(r$lastday - origin))
  r <- r[order(id, s)]
  # Adjacent intervals are one period.
  merged <- list()
  for (i in seq_len(nrow(r))) {
    m <- length(merged)
    if (m > 0 && merged[[m]]$id == r$id[i] && merged[[m]]$e == r$s[i]) merged[[m]]$e <- r$e[i]
    else merged[[m + 1]] <- list(id = r$id[i], s = r$s[i], e = r$e[i])
  }
  list(id = I(sapply(merged, `[[`, "id")), start = I(sapply(merged, `[[`, "s")),
       end = I(sapply(merged, `[[`, "e")))
}
# heaven lengthens a supply that is running when an admission starts by more
# than the days of the admission, so the pause cases compare only persons whose
# admissions all start while no supply is running (in the periods without
# pauses), which is where the two rules coincide.
base <- run(1e9, adm_none)
idle <- sapply(0:(n - 1), function(p) {
  b <- as.data.frame(base)[base$id == p, ]
  all(sapply(a$s[a$id == p], function(s0) !any(b$start <= s0 & s0 < b$end)))
})
pause_ok <- (0:(n - 1))[idle]
cases <- list(
  list(name = "no_pauses_no_cap", depot_days = NA, pauses = FALSE, periods = run(1e9, adm_none)),
  list(name = "pauses_no_cap", depot_days = NA, pauses = TRUE, periods = run(1e9, adm_all)),
  list(name = "pauses_cap", depot_days = 40, pauses = TRUE, periods = run(40 * dose, adm_all)),
  list(name = "no_pauses_cap", depot_days = 40, pauses = FALSE, periods = run(40 * dose, adm_none)))
cat(toJSON(list(
  r_version = R.version.string, heaven_version = as.character(packageVersion("heaven")),
  dose = dose, id = d$id, day = d$day, quantity = d$q,
  pause_ok = I(pause_ok), pause_id = I(a$id), pause_start = I(a$s), pause_end = I(a$e), cases = cases
), digits = NA, auto_unbox = TRUE))
