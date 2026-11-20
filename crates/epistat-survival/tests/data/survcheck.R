# survival::survcheck on follow-up with several rows per subject: clean data,
# each flag alone, several flags in one subject, right-censored rows with a
# duplicated id, binary counting data with an overlap, an initial state given
# and absent, ids out of order, ties in stop with different starts, and random
# data from a fixed seed with gaps, overlaps and wrong initial states.
# For each case: the input, then what survcheck reports, with rows counted from
# 1, ids as given, states as codes (0 is "(s0)" as a start and censoring as an
# end), and the transitions as (from, to, count) triples.
# Seventeen digits are written so that every double reads back exactly.
# Run: Rscript survcheck.R > survcheck.json
suppressMessages({
  library(jsonlite)
  library(survival)
})

state_code <- function(label) {
  label <- as.character(label)
  num <- suppressWarnings(as.integer(sub("^s", "", label)))
  num[label %in% c("(s0)", "(censor)")] <- 0L
  num[label == "event"] <- 1L
  num
}
iv <- function(x) I(as.integer(x))

run <- function(name, id, stop, status, start = NULL, istate = NULL, k = 1L) {
  n <- length(stop)
  st <- if (k > 1L) factor(status, 0:k, c("censor", paste0("s", seq_len(k)))) else status
  y <- if (is.null(start)) Surv(stop, st) else Surv(start, stop, st)
  d <- data.frame(id = id, y = y)
  d$y <- y
  if (!is.null(istate)) {
    d$ist <- ifelse(istate == 0L, "(s0)", paste0("s", istate))
    fit <- survcheck(y ~ 1, data = d, id = id, istate = ist)
  } else {
    fit <- survcheck(y ~ 1, data = d, id = id)
  }
  flag <- function(f) {
    rows <- if (is.null(fit[[f]]$row)) integer(0) else fit[[f]]$row
    ids <- if (is.null(fit[[f]]$id)) numeric(0) else fit[[f]]$id
    list(count = unname(fit$flag[[f]]), rows = iv(rows), ids = I(as.numeric(ids)))
  }
  tr <- as.data.frame(as.table(fit$transitions), stringsAsFactors = FALSE)
  tr <- tr[tr$Freq > 0, , drop = FALSE]
  list(
    name = name, k = k, id = I(as.numeric(id)), stop = I(stop),
    start = if (is.null(start)) NULL else I(start), status = iv(status),
    istate = if (is.null(istate)) NULL else iv(istate),
    overlap = flag("overlap"), gap = flag("gap"), jump = flag("jump"),
    teleport = flag("teleport"),
    from = iv(state_code(tr$from)), to = iv(state_code(tr$to)), count = iv(tr$Freq),
    resolved = iv(state_code(fit$istate))
  )
}

set.seed(20261120)
cases <- list()
add <- function(...) cases[[length(cases) + 1]] <<- run(...)

# Clean multi-state data: 1 -> 2 -> 3 and a censored subject.
add("clean_multistate", id = c(1, 1, 1, 2, 2), start = c(0, 4, 9, 0, 6),
    stop = c(4, 9, 12, 6, 10), status = c(2, 3, 0, 1, 0), istate = c(1, 2, 3, 1, 1), k = 3L)
# Each flag alone.
add("overlap_only", id = c(1, 1), start = c(0, 3), stop = c(5, 8), status = c(2, 0),
    istate = c(1, 2), k = 2L)
add("gap_only", id = c(1, 1), start = c(0, 7), stop = c(5, 9), status = c(2, 0),
    istate = c(1, 2), k = 2L)
add("jump_only", id = c(1, 1), start = c(0, 7), stop = c(5, 9), status = c(2, 0),
    istate = c(1, 1), k = 2L)
add("teleport_only", id = c(1, 1), start = c(0, 5), stop = c(5, 9), status = c(2, 0),
    istate = c(1, 1), k = 2L)
# Several flags in one subject, and another subject with one flag.
add("several_flags_one_id", id = c(5, 5, 5, 5, 5, 9, 9),
    start = c(0, 2, 6, 6, 13, 0, 4),
    stop = c(2, 4, 6.5, 8, 15, 3, 9),
    status = c(1, 0, 0, 2, 0, 0, 0), istate = c(1, 1, 2, 3, 1, 2, 1), k = 3L)
# Right-censored data with a repeated id: an implicit overlap from time 0.
add("right_censored_duplicate_id", id = c(1, 2, 2, 3), stop = c(5, 4, 9, 2),
    status = c(1, 2, 0, 0), k = 2L)
add("right_censored_duplicate_id_istate", id = c(1, 2, 2, 3), stop = c(5, 4, 9, 2),
    status = c(1, 2, 0, 0), istate = c(1, 1, 2, 1), k = 2L)
add("right_censored_binary_duplicate", id = c(1, 1, 2), stop = c(5, 7, 3), status = c(1, 0, 0))
# Binary counting data with an overlap, and with a gap.
add("binary_counting_overlap", id = c(1, 1, 2), start = c(0, 4, 0), stop = c(6, 9, 5),
    status = c(0, 1, 0))
add("binary_counting_gap", id = c(1, 1, 2), start = c(0, 8, 0), stop = c(6, 9, 5),
    status = c(0, 1, 0))
# Initial state given and absent on the same follow-up.
add("istate_absent", id = c(1, 1, 1, 2), start = c(0, 3, 8, 0), stop = c(3, 8, 11, 4),
    status = c(2, 0, 1, 0), k = 2L)
add("istate_with_unknown", id = c(1, 1, 2, 2), start = c(0, 3, 0, 5), stop = c(3, 8, 5, 9),
    status = c(2, 0, 0, 1), istate = c(0, 2, 1, 0), k = 2L)
# Ids out of order in the rows, with the rows of a subject out of time order.
add("ids_out_of_order", id = c(30, 10, 20, 10, 30, 20, 10),
    start = c(5, 3, 0, 0, 0, 4, 9), stop = c(9, 6, 4, 3, 5, 8, 12),
    status = c(1, 0, 2, 2, 0, 0, 1), istate = c(2, 1, 1, 1, 1, 2, 1), k = 2L)
# Equal stops with different starts, equal keys otherwise, and a row inside a
# longer one: the order within a subject decides what is flagged.
add("equal_stops", id = c(1, 1, 1, 2, 2), start = c(0, 2, 2, 1, 0), stop = c(5, 5, 5, 3, 3),
    status = c(0, 1, 0, 0, 0), k = 1L)
add("nested_rows", id = c(1, 1, 1), start = c(0, 2, 3), stop = c(10, 5, 4), status = c(0, 0, 1))
# Fractional times that are exactly representable and tie exactly.
add("fractional_exact", id = c(1, 1, 1), start = c(0, 2.5, 5.5), stop = c(2.5, 5.5, 8.25),
    status = c(1, 0, 0), istate = c(1, 2, 1), k = 2L)
# A single row per subject.
add("one_row_per_subject", id = c(4, 3, 2, 1), start = c(0, 0, 0, 0), stop = c(1, 2, 3, 4),
    status = c(0, 1, 2, 0), istate = c(1, 1, 2, 2), k = 2L)

# Random data. Each subject has 1 to 5 rows. A row starts at the previous stop
# (tie), after a gap, or before it (overlap), and the state given for it is the
# state reached, or wrong, or unknown.
random_case <- function(name, n_subj, k, counting, with_istate, p_event = 0.5) {
  id <- start <- stop <- status <- istate <- numeric(0)
  ids <- sample(1000, n_subj)
  for (s in ids) {
    m <- sample(1:5, 1)
    clock <- 0
    reached <- sample(0:k, 1)
    for (j in seq_len(m)) {
      move <- sample(c("tie", "gap", "overlap"), 1, prob = c(0.6, 0.2, 0.2))
      b <- clock + switch(move, tie = 0, gap = sample(1:4, 1),
                          overlap = -sample(0:3, 1))
      if (j == 1) b <- 0
      if (b < 0) b <- 0
      e <- b + sample(1:6, 1)
      stt <- if (runif(1) < p_event) sample(1:k, 1) else 0L
      ist <- if (j == 1) reached else if (runif(1) < 0.7) reached else sample(0:k, 1)
      id <- c(id, s); start <- c(start, b); stop <- c(stop, e)
      status <- c(status, stt); istate <- c(istate, ist)
      clock <- e
      if (stt > 0) reached <- stt
    }
  }
  o <- sample(length(id))
  # Right-censored rows have no start, so a subject's rows share time 0.
  run(name, id = id[o], stop = stop[o], status = status[o],
      start = if (counting) start[o] else NULL,
      istate = if (with_istate) istate[o] else NULL, k = k)
}
cases[[length(cases) + 1]] <- random_case("random_multistate_istate", 300, 3L, TRUE, TRUE)
cases[[length(cases) + 1]] <- random_case("random_multistate_no_istate", 300, 3L, TRUE, FALSE)
cases[[length(cases) + 1]] <- random_case("random_right_censored_istate", 300, 3L, FALSE, TRUE)
cases[[length(cases) + 1]] <- random_case("random_right_censored_no_istate", 300, 3L, FALSE, FALSE)
cases[[length(cases) + 1]] <- random_case("random_binary_counting", 300, 1L, TRUE, FALSE, 0.4)

cat(toJSON(list(
  r_version = R.version.string,
  survival_version = as.character(packageVersion("survival")),
  jsonlite_version = as.character(packageVersion("jsonlite")),
  cases = cases
), digits = I(17), auto_unbox = TRUE, null = "null", na = "null"))
