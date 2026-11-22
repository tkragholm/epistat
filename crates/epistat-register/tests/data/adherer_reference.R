# Treatment episodes from AdhereR::compute.treatment.episodes for the
# supply recurrence of epistat-register::exposure with whole-day durations, a
# single medication, no cap on stock, no pauses and a permitted gap of 0, 30
# and 90 days. Run from the crate directory:
#   Rscript tests/data/adherer_reference.R > tests/data/adherer_reference.json
suppressMessages({library(AdhereR); library(jsonlite)})
set.seed(20261123)
epoch <- as.Date("1970-01-01")
n <- 150
persons <- list()
rows <- list()
for (p in 1:n) {
  k <- sample(c(1, 2, 3, 5, 8, 12), 1)
  t <- 18000 + sample(0:2000, 1)
  supply_end <- NA_real_
  for (i in 1:k) {
    dur <- sample(c(7, 14, 28, 30, 60, 90), 1)
    rows[[length(rows) + 1]] <- data.frame(id = p - 1L, day = t, dur = dur)
    # Next dispensing relative to the end of the supply held so far, so that
    # gaps fall on, just inside and just outside the limits.
    start <- if (is.na(supply_end) || t > supply_end) t else supply_end
    supply_end <- start + dur
    gap <- sample(c(-40:-1, 0, 0, 1, 29, 30, 31, 89, 90, 91, 0:120), 1)
    t <- max(t, supply_end + gap)
  }
}
d <- do.call(rbind, rows)
d <- d[order(d$id, d$day), ]
d$date <- as.character(epoch + d$day)
d$med <- "A"
d$quantity <- 2 * d$dur

run <- function(gap) {
  r <- compute.treatment.episodes(
    d, ID.colname = "id", event.date.colname = "date", event.duration.colname = "dur",
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
  list(gap = gap, id = r$id, start = as.numeric(r$episode.start - epoch),
       end = as.numeric(r$episode.end - epoch))
}
cat(toJSON(list(
  r_version = R.version.string,
  adherer_version = as.character(packageVersion("AdhereR")),
  id = d$id, day = d$day, quantity = d$quantity,
  episodes = lapply(c(0, 30, 90), run)
), digits = NA, auto_unbox = TRUE))
