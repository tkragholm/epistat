# Reference results for epistat-register::residence from an independent
# implementation of the stated rules in data.table (no package offers them):
# a day with both directions is dropped, runs in one direction collapse to the
# first, every In opens a spell closed by the next event, a first Out closes a
# spell of unknown start, and a person with no usable event is resident
# throughout. 2000 simulated persons. Run from the crate directory:
#   Rscript tests/data/residence_reference.R > tests/data/residence_reference.json
suppressMessages({library(jsonlite); library(data.table)})
set.seed(20261123)
n <- 2000
k <- sample(0:6, n, replace = TRUE, prob = c(.3, .2, .2, .1, .1, .05, .05))
ev <- data.table(person = rep(0:(n - 1), k))
ev[, `:=`(day = sample(0:120, .N, replace = TRUE), dir = sample(c("Out", "In"), .N, replace = TRUE))]
setorder(ev, person, day)  # direction order within a day is the delivery order
raw <- copy(ev)

d <- ev[, if (uniqueN(dir) == 1) .SD[1], by = .(person, day)]
d[, prev := shift(dir), by = person]
alt <- d[is.na(prev) | dir != prev]
alt[, `:=`(next_day = shift(day, -1), pos = seq_len(.N)), by = person]
spells <- rbind(
  alt[pos == 1 & dir == "Out", .(person, from = NA_real_, to = as.numeric(day))],
  alt[dir == "In", .(person, from = as.numeric(day), to = as.numeric(next_day))]
)
setorder(spells, person, from, na.last = FALSE)

q <- data.table(person = sample(0:(n - 1), 6000, replace = TRUE), day = sample(-10:140, 6000, replace = TRUE))
by_person <- split(spells, factor(spells$person, levels = 0:(n - 1)))
q[, resident := {
  mapply(function(p, dd) {
    s <- by_person[[p + 1]]
    if (nrow(s) == 0) TRUE else any((is.na(s$from) | s$from <= dd) & (is.na(s$to) | dd <= s$to))
  }, person, day)
}]
outs <- alt[dir == "Out", .(person, day)]
q[, departure := {
  mapply(function(p, dd) {
    x <- outs[person == p & day >= dd, day]
    if (length(x)) as.numeric(min(x)) else NA_real_
  }, person, day)
}]
cat(toJSON(list(
  versions = list(R = as.character(getRversion()), data.table = as.character(packageVersion("data.table"))),
  persons = n,
  events = list(person = raw$person, day = raw$day, out = raw$dir == "Out"),
  spells = list(person = spells$person, from = spells$from, to = spells$to),
  queries = list(person = q$person, day = q$day, resident = q$resident, departure = q$departure)
), digits = NA, auto_unbox = TRUE, na = "null"))
