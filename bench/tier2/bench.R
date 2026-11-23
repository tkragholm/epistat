# The R side of the tier 2 benchmarks: one process runs one method on one size
# and writes the numbers the Rust side writes, so that ../tier1/compare.R can
# check that both solved the same problem. See run.sh for the driver. The data
# files come from the Rust programs (crates/epistat-register/benches/
# tier2_register.rs).
#
# Usage: Rscript bench.R N METHOD SCRATCH
# Environment: OUT (results, default SCRATCH/out) and REPS (timed repetitions,
# default 3).
suppressMessages(library(data.table))
args <- commandArgs(TRUE)
n <- as.integer(args[1]); method <- args[2]; scratch <- args[3]
out <- Sys.getenv("OUT", file.path(scratch, "out"))
reps <- as.integer(Sys.getenv("REPS", "3"))
dir.create(out, showWarnings = FALSE, recursive = TRUE)
rd <- function(kind) {
  d <- fread(file.path(scratch, sprintf("%s_%d.csv", kind, n)))
  for (j in names(d)) if (is.integer(d[[j]])) set(d, j = j, value = as.numeric(d[[j]]))
  d
}
setDTthreads(0)
timed <- function(f) {
  times <- numeric(reps)
  for (i in seq_len(reps)) {
    t0 <- proc.time()[["elapsed"]]
    value <- f()
    times[i] <- proc.time()[["elapsed"]] - t0
  }
  list(times = times, value = value)
}
finish <- function(times, values) {
  d <- data.table(name = names(values), value = sprintf("%.17g", as.numeric(values)))
  fwrite(d, file.path(out, sprintf("r_%s_%d.csv", method, n)), quote = FALSE)
  path <- file.path(out, "r_times.csv")
  row <- data.table(n = n, method = method, reps = length(times),
                    median_s = median(times), min_s = min(times))
  fwrite(row, path, append = file.exists(path))
}

if (method == "lexis") {
  suppressMessages({library(Epi); library(popEpi)})
  d <- rd("lx")
  ages <- c(5, 18, 40, 65)
  age_cut <- ceiling(1461 * ages / 4)
  per_cut <- as.numeric(as.Date(sprintf("%d-01-01", 1930:2110)) - as.Date("1970-01-01"))
  r <- timed(function() {
    invisible(capture.output(lex <- Lexis(entry = list(per = d$start, age = d$start - d$birth),
                                          exit = list(per = d$end), exit.status = 1L)))
    splitMulti(lex, per = per_cut, age = age_cut, drop = FALSE)
  })
  finish(r$times, c(pieces = nrow(r$value), days = sum(r$value$lex.dur)))
} else if (method == "intervals") {
  d <- rd("iv")
  r <- timed(function() {
    a <- d[set == 0, .(key, start, end)]; b <- d[set == 1, .(key, start, end)]
    setkey(b, key, start, end)
    ov <- foverlaps(a, b, by.x = c("key", "start", "end"), by.y = c("key", "start", "end"),
                    type = "any", nomatch = NULL)
    ov <- ov[, .(len = pmax(0, pmin(end, i.end) - pmax(start, i.start)))][len > 0]
    clipped <- a[, .(len = pmax(0, pmin(end, 250) - pmax(start, 50)))]
    a_days <- a[, sum(end - start)]
    c(a_days = a_days, intersect_days = ov[, sum(len)], subtract_days = a_days - ov[, sum(len)],
      clip_days = clipped[, sum(len)], intersect_pieces = nrow(ov),
      subtract_pieces = NA_real_)
  })
  finish(r$times, r$value[setdiff(names(r$value), "subtract_pieces")])
} else if (method == "residence") {
  d <- rd("rs")
  r <- timed(function() {
    ev <- copy(d); ev[, dir := ifelse(out == 1, "Out", "In")]
    x <- ev[, if (uniqueN(dir) == 1) .SD[1], by = .(person, day)]
    x[, prev := shift(dir), by = person]
    alt <- x[is.na(prev) | dir != prev]
    alt[, `:=`(next_day = shift(day, -1), pos = seq_len(.N)), by = person]
    sp <- rbind(alt[pos == 1 & dir == "Out", .(person, from = NA_real_, to = as.numeric(day))],
                alt[dir == "In", .(person, from = as.numeric(day), to = as.numeric(next_day))])
    persons <- max(d$person) + 1
    inside <- sp[(is.na(from) | from <= 1500) & (is.na(to) | 1500 <= to), uniqueN(person)]
    with_spell <- sp[, uniqueN(person)]
    dep <- alt[dir == "Out" & day >= 1500, uniqueN(person)]
    c(spells = nrow(sp), from_sum = sp[, sum(from, na.rm = TRUE)], to_sum = sp[, sum(to, na.rm = TRUE)],
      resident = inside + (persons - with_spell), departures = dep)
  })
  finish(r$times, r$value)
} else if (method == "pedigree") {
  d <- rd("pd")
  r <- timed(function() {
    pairs <- function(counts) sum(counts * (counts - 1) / 2)
    same_mother <- d[!is.na(mother), .N, by = mother][, pairs(N)]
    same_father <- d[!is.na(father), .N, by = father][, pairs(N)]
    full <- d[!is.na(mother) & !is.na(father), .N, by = .(mother, father)][, pairs(N)]
    # Family clusters by label propagation over the two parent columns.
    lab <- seq_len(nrow(d))
    repeat {
      old <- lab
      for (col in c("mother", "father")) {
        m <- tapply(lab, d[[col]], min)
        idx <- !is.na(d[[col]])
        lab[idx] <- as.numeric(m[as.character(d[[col]][idx])])
      }
      if (all(old == lab)) break
    }
    c(full = full, same_mother = same_mother, same_father = same_father,
      clusters = length(unique(lab)))
  })
  finish(r$times, r$value)
} else if (method %in% c("bias_exposure", "bias_outcome")) {
  suppressMessages(library(episensr))
  tab <- matrix(c(215, 83, 1021, 1202), 2, byrow = TRUE)
  r <- timed(function() {
    x <- suppressMessages(suppressWarnings(
      if (method == "bias_exposure") {
        probsens(tab, type = "exposure", reps = n, seca = list("trapezoidal", c(.75, .85, .95, 1)),
                 spca = list("triangular", c(.9, 1, .97)))
      } else {
        probsens(tab, type = "outcome", reps = n, seca = list("normal", c(.7, .95, .85, .05)),
                 spca = list("uniform", c(.95, 1)))
      }))
    m <- x$adj_measures
    c(info_rr_median = m[1, 1], info_rr_lower = m[1, 2], info_rr_upper = m[1, 3],
      info_or_median = m[3, 1], info_or_lower = m[3, 2], info_or_upper = m[3, 3],
      info_rr_total_median = m[2, 1], info_rr_total_lower = m[2, 2], info_rr_total_upper = m[2, 3],
      info_or_total_median = m[4, 1], info_or_total_lower = m[4, 2], info_or_total_upper = m[4, 3])
  })
  finish(r$times, r$value)
} else {
  stop("unknown method ", method)
}
