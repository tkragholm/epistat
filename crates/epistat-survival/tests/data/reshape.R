# survival::survSplit, survival::tmerge and survival::survcondense: the examples
# of their help pages and of the timedep vignette (the veteran, mgus2, aml, cgd0,
# jasa, pbc and pbcseq data), hand-made cases for each rule, and random cases.
# These functions produce times and row indices, not sums, so the Rust
# functions must agree exactly.
# For each case: the input, the call's settings, the output or the error.
# survcondense returns no rows at all when nothing is merged (it drops the rows
# with x[-integer(0), ]); those cases are recorded as they are.
# Run: Rscript reshape.R > reshape.json
suppressMessages({
  library(jsonlite)
  library(survival)
})
I1 <- function(x) if (length(x) == 1) I(x) else x   # keep length-one vectors arrays
num <- function(x) as.numeric(x)

# ---------------------------------------------------------------- survSplit
split_case <- function(name, d, cut, zero = 0, timefix = TRUE, counting = FALSE) {
  d$rowid <- seq_len(nrow(d)) - 1L
  res <- tryCatch(suppressWarnings(
    if (counting) survSplit(Surv(s, e, ev) ~ ., d, cut = cut, zero = zero, episode = "ep",
                            added = "ad", timefix = timefix)
    else survSplit(Surv(e, ev) ~ ., d, cut = cut, zero = zero, episode = "ep",
                   added = "ad", timefix = timefix, start = "s")),
    error = function(err) conditionMessage(err))
  out <- list(name = name, counting = counting, zero = zero, timefix = timefix,
              cut = I1(cut), start = if (counting) I1(num(d$s)), stop = I1(num(d$e)),
              status = I1(as.integer(d$ev)))
  if (is.character(res)) out$error <- res
  else {
    out$out_start <- I1(num(res$s)); out$out_stop <- I1(num(res$e))
    out$out_status <- I1(as.integer(res$ev)); out$out_row <- I1(as.integer(res$rowid))
    out$out_episode <- I1(as.integer(res$ep)); out$out_added <- I1(as.logical(res$ad))
  }
  out
}
set.seed(20261120)
sp <- list()
add_split <- function(...) sp[[length(sp) + 1]] <<- split_case(...)

vet <- veteran
add_split("veteran_60_120", data.frame(e = vet$time, ev = vet$status), c(60, 120))
mg <- mgus2
mg <- mg[!is.na(mg$futime), ]
add_split("mgus2_age_25_100", data.frame(s = mg$age, e = mg$age + mg$futime / 12, ev = mg$death),
          25:100, counting = TRUE)
add_split("aml_10_20_30", data.frame(e = aml$time, ev = aml$status), c(10, 20, 30))
add_split("cut_at_start_and_stop", data.frame(s = c(0, 5, 5, 2), e = c(5, 10, 6, 9), ev = c(1, 0, 1, 1)),
          c(5, 6), counting = TRUE)
add_split("cuts_unsorted_repeated", data.frame(e = c(4, 10, 25), ev = c(1, 1, 0)), c(20, 5, 5, 10, 1))
add_split("cuts_outside", data.frame(e = c(4, 10), ev = c(1, 0)), c(-3, 0, 100))
add_split("cut_at_zero_default", data.frame(e = c(4, 10), ev = c(1, 0)), c(0, 4, 10))
add_split("zero_negative", data.frame(e = c(4, 10), ev = c(1, 0)), c(-2, 3, 8), zero = -5)
add_split("zero_too_high", data.frame(e = c(4, 10), ev = c(1, 0)), c(3), zero = 4)
add_split("zero_nonzero_start", data.frame(e = c(14, 20), ev = c(1, 0)), c(15, 17), zero = 10)
add_split("cut_not_finite", data.frame(e = c(4, 10), ev = c(1, 0)), c(3, Inf))
add_split("one_cut_many_pieces_row", data.frame(s = 0, e = 100, ev = 1), seq(1, 99, by = 2), counting = TRUE)
# near ties between a cut and a stop, with and without the tie rule
nt <- data.frame(e = c(5 + 3e-12, 10, 7.5, 12 + 1e-12), ev = c(1, 0, 1, 1))
add_split("near_tie_timefix", nt, c(5, 7.5 + 1e-13, 11))
add_split("near_tie_exact", nt, c(5, 7.5 + 1e-13, 11), timefix = FALSE)
nt2 <- data.frame(s = c(0, 5 - 2e-12, 8), e = c(5, 9, 12 + 1e-12), ev = c(0, 1, 1))
add_split("near_tie_counting", nt2, c(5, 12), counting = TRUE)
add_split("near_tie_counting_exact", nt2, c(5, 12), counting = TRUE, timefix = FALSE)
for (k in 1:12) {
  n <- sample(5:60, 1)
  grid <- sample(c(1, 2, 5, 10), 1)
  s <- sample(0:20, n, TRUE) / grid
  e <- s + sample(1:25, n, TRUE) / grid
  d <- data.frame(s = s, e = e, ev = sample(0:1, n, TRUE))
  cut <- sample(0:50, sample(0:12, 1), TRUE) / grid
  if (k %% 3 == 0) add_split(paste0("random_right_", k), d[, c("e", "ev")], cut, zero = -1)
  else add_split(paste0("random_counting_", k), d, cut, counting = TRUE)
}

# ------------------------------------------------------------------ tmerge
# data2 tables are lists of columns; NA is null in the JSON.
tm_case <- function(name, base_id, first, ops, options = list()) {
  base <- data.frame(id = base_id, rowid = seq_along(base_id) - 1L)
  opt <- options
  cur <- base
  err <- NULL
  counts <- NULL
  run1 <- function(cl_extra, d2, first_call) {
    cl <- call("tmerge", quote(cur), quote(d2), id = quote(id))
    for (nm in names(cl_extra)) cl[[nm]] <- cl_extra[[nm]]
    if (length(opt)) cl$options <- opt
    suppressWarnings(eval(cl))
  }
  res <- tryCatch({
    if (!is.null(first$span)) {
      d2 <- data.frame(id = first$span$id, tstop = first$span$tstop)
      ex <- list(tstop = quote(tstop))
      if (!is.null(first$span$tstart)) { d2$tstart <- first$span$tstart; ex$tstart <- quote(tstart) }
      cur <- run1(ex, d2, TRUE)
    }
    for (i in seq_along(ops)) {
      op <- ops[[i]]
      d2 <- data.frame(id = op$id, time = op$time)
      args <- list(quote(time))
      if (!is.null(op$value)) {
        d2$value <- op$value
        args <- c(args, list(quote(value)))
      }
      if (!is.null(op$init)) args$init <- op$init
      ex <- list(as.call(c(list(as.name(op$kind)), args)))
      names(ex) <- op$name
      cur <- run1(ex, d2, i == 1 && is.null(first$span))
    }
    cur
  }, error = function(e) conditionMessage(e))
  out <- list(name = name, base_id = I1(as.integer(base_id)),
              delay = if (is.null(options$delay)) 0 else options$delay,
              na_rm = if (is.null(options$na.rm)) TRUE else options$na.rm,
              tdcstart = options$tdcstart,
              ops = lapply(ops, function(o) list(kind = o$kind, name = o$name,
                id = I1(as.integer(o$id)), time = I1(num(o$time)),
                value = if (!is.null(o$value)) I1(num(o$value)), init = o$init)))
  if (!is.null(first$span)) out$span <- list(id = I1(as.integer(first$span$id)),
      tstart = if (!is.null(first$span$tstart)) I1(num(first$span$tstart)),
      tstop = I1(num(first$span$tstop)))
  else out$first_events <- TRUE
  if (is.character(res)) out$error <- res
  else {
    out$out_id <- I1(as.integer(res$id)); out$out_row <- I1(as.integer(res$rowid))
    out$out_tstart <- I1(num(res$tstart)); out$out_tstop <- I1(num(res$tstop))
    vars <- setdiff(names(res), c("id", "rowid", "tstart", "tstop"))
    out$out_vars <- lapply(vars, function(v) list(name = v, values = I1(num(res[[v]]))))
    tc <- attr(res, "tcount")
    out$out_counts <- lapply(seq_len(nrow(tc)), function(i) {
      as.list(tc[i, ])})
  }
  out
}
tms <- list()
add_tm <- function(...) tms[[length(tms) + 1]] <<- tm_case(...)
op <- function(kind, name, id, time, value = NULL, init = NULL)
  list(kind = kind, name = name, id = id, time = time, value = value, init = init)
span <- function(id, tstop, tstart = NULL) list(span = list(id = id, tstop = tstop, tstart = tstart))
fe <- list(span = NULL)

# The numbers of the help page: intervals (0,5], (5,10], (15,20], new data at
# -1, 5, 11, 12 (the C comment in tmerge2).
add_tm("tdc_carry_forward",  1L, span(c(1L, 1L, 1L), c(5, 10, 20), c(0, 5, 15)),
       list(op("tdc", "x", rep(1L, 4), c(-1, 5, 11, 12), c(10, 20, 30, 40))))
add_tm("tdc_no_value", c(1L, 2L), span(c(1L, 2L), c(10, 10)),
       list(op("tdc", "x", c(1L, 1L), c(3, 7))))
add_tm("tdc_init_and_tdcstart", c(1L, 2L), span(c(1L, 2L), c(10, 10)),
       list(op("tdc", "x", 1L, 3, 5, init = 99), op("tdc", "y", 1L, 3, 5)),
       list(tdcstart = -1))
add_tm("tdc_tdcstart_missing_default", c(1L, 2L), span(c(1L, 2L), c(10, 10)),
       list(op("tdc", "x", 1L, 3, 5)))
add_tm("tdc_tie_last_wins", 1L, span(1L, 10),
       list(op("tdc", "x", c(1L, 1L, 1L), c(4, 4, 4), c(1, 2, 3))))
add_tm("tdc_na_rm_true", 1L, span(1L, 10),
       list(op("tdc", "x", c(1L, 1L, 1L), c(2, 4, 6), c(1, NA, 3))))
add_tm("tdc_na_rm_false", 1L, span(1L, 10),
       list(op("tdc", "x", c(1L, 1L, 1L), c(2, 4, 6), c(1, NA, 3))), list(na.rm = FALSE))
add_tm("tdc_update_values", c(1L, 2L), span(c(1L, 2L), c(10, 10)),
       list(op("tdc", "x", c(1L, 2L), c(2, 5), c(1, 2)), op("tdc", "x", 1L, 6, 7)))
add_tm("tdc_update_01", c(1L, 2L), span(c(1L, 2L), c(10, 10)),
       list(op("tdc", "x", 1L, 2), op("tdc", "x", 2L, 3)))
add_tm("tdc_update_01_refused", c(1L, 2L), span(c(1L, 2L), c(10, 10)),
       list(op("tdc", "x", 1L, 2, 5), op("tdc", "x", 2L, 3)))
add_tm("tdc_outside_follow_up", 1L, span(c(1L, 1L), c(5, 20), c(0, 10)),
       list(op("tdc", "x", c(1L, 1L, 1L, 1L), c(-3, 7, 25, 5), c(1, 2, 3, 4))))
add_tm("delay", c(1L, 2L), span(c(1L, 2L), c(20, 20), c(0, 5)),
       list(op("tdc", "x", c(1L, 1L, 2L, 2L), c(0, 4, 5, 9), c(1, 2, 3, 4)),
            op("cumtdc", "c", c(1L, 1L, 2L), c(0, 4, 9))), list(delay = 2))
add_tm("cumtdc_basic", c(1L, 2L), span(c(1L, 2L), c(10, 10)),
       list(op("cumtdc", "c", c(1L, 1L, 2L, 1L), c(2, 5, 3, 5), c(1, 2, 10, 4))))
add_tm("cumtdc_no_value_counts", 1L, span(1L, 10),
       list(op("cumtdc", "c", c(1L, 1L, 1L), c(2, 2, 7))))
add_tm("cumtdc_init", 1L, span(1L, 10),
       list(op("cumtdc", "c", c(1L, 1L), c(2, 7), c(1, 1), init = 100)))
add_tm("cumtdc_value_default_missing", 1L, span(1L, 10),
       list(op("cumtdc", "c", c(1L, 1L), c(4, 7), c(1, 1))))
add_tm("cumtdc_na_rm_false", 1L, span(1L, 10),
       list(op("cumtdc", "c", c(1L, 1L), c(4, 7), c(1, NA))), list(na.rm = FALSE))
add_tm("cumtdc_gap_and_late", 1L, span(c(1L, 1L), c(5, 20), c(0, 10)),
       list(op("cumtdc", "c", c(1L, 1L, 1L, 1L), c(-1, 7, 12, 30), c(1, 2, 3, 4))))
add_tm("cumtdc_update", 1L, span(1L, 10),
       list(op("cumtdc", "c", 1L, 3, 1), op("cumtdc", "c", 1L, 6, 2)))
# events
add_tm("event_edges", 1L, span(c(1L, 1L, 1L), c(5, 10, 20), c(0, 5, 15)),
       list(op("event", "e", rep(1L, 7), c(0, 5, 3, 10, 15, 20, 12), c(1L, 2L, 3L, 4L, 5L, 6L, 7L))))
add_tm("event_late_early_gap", 1L, span(c(1L, 1L), c(5, 20), c(2, 10)),
       list(op("event", "e", rep(1L, 4), c(1, 7, 25, 5), c(1L, 1L, 1L, 1L))))
add_tm("event_no_value", c(1L, 2L, 3L), span(c(1L, 2L, 3L), c(5, 6, 7)),
       list(op("event", "e", c(1L, 2L), c(5, 4))))
add_tm("event_split_and_tie", 1L, span(1L, 10),
       list(op("event", "e", rep(1L, 4), c(3, 3, 6, 6), c(1L, 2L, 3L, 4L))))
add_tm("event_update_after_cover", c(1L, 2L), span(c(1L, 2L), c(10, 10)),
       list(op("event", "e", c(1L, 2L), c(10, 10), c(1L, 1L)),
            op("event", "e", 1L, 4, 2L)))
add_tm("event_two_types", c(1L, 2L), span(c(1L, 2L), c(10, 10)),
       list(op("event", "death", 1L, 10, 1L), op("event", "prog", c(1L, 2L), c(4, 4), c(1L, 1L))))
add_tm("event_na_time_and_value", 1L, span(1L, 10),
       list(op("event", "e", c(1L, 1L, 1L), c(NA, 4, 6), c(1L, NA, 2L))))
add_tm("event_na_rm_false", 1L, span(1L, 10),
       list(op("event", "e", c(1L, 1L, 1L), c(NA, 4, 6), c(1L, NA, 2L))), list(na.rm = FALSE))
add_tm("event_unknown_id", c(1L, 2L), span(c(1L, 2L), c(10, 10)),
       list(op("event", "e", c(1L, 9L, 2L), c(3, 3, 4), c(1L, 1L, 1L))))
add_tm("cumevent_basic", c(1L, 2L), span(c(1L, 2L), c(10, 10)),
       list(op("cumevent", "k", c(1L, 1L, 1L, 2L), c(2, 5, 12, 3), c(1L, 2L, 4L, 5L))))
add_tm("cumevent_no_value", 1L, span(c(1L, 1L), c(5, 20), c(0, 10)),
       list(op("cumevent", "k", rep(1L, 5), c(-1, 3, 5, 8, 15))))
add_tm("cumevent_zero_value", 1L, span(1L, 10),
       list(op("cumevent", "k", rep(1L, 3), c(2, 4, 6), c(1L, 0L, 2L))))
# first call
add_tm("first_events_set_range", c(1L, 2L, 3L), list(span = NULL),
       list(op("event", "e", c(3L, 1L, 2L), c(9, 5, 8), c(1L, 2L, 1L)),
            op("tdc", "x", c(1L, 2L), c(2, 3), c(1, 1))))
add_tm("first_events_not_in_data1", c(1L, 2L), list(span = NULL),
       list(op("event", "e", c(1L, 9L), c(5, 8))))
add_tm("first_call_duplicate_base_id", c(1L, 1L), span(c(1L), c(5)), list())
add_tm("first_call_id_not_in_base", 1L, span(c(1L, 2L), c(5, 6)), list())
add_tm("first_call_base_not_in_spans", c(1L, 2L), span(1L, 5), list())
add_tm("first_call_overlap", 1L, span(c(1L, 1L), c(5, 8), c(0, 4)), list())
add_tm("first_call_touching_ok", 1L, span(c(1L, 1L), c(5, 8), c(0, 5)), list())
add_tm("first_call_start_ge_stop", 1L, span(1L, 5, 5), list())
add_tm("first_call_default_start_negative_stop", 1L, span(1L, -2), list())
add_tm("first_call_order_of_spans", c(5L, 3L, 9L), span(c(9L, 5L, 3L, 5L), c(6, 9, 4, 3), c(1, 4, 0, 0)), list())
add_tm("first_call_order_unique", c(5L, 3L, 9L), span(c(9L, 5L, 3L), c(6, 9, 4)), list())
add_tm("tdc_on_event_name_refused", 1L, span(1L, 10),
       list(op("event", "e", 1L, 5), op("tdc", "e", 1L, 3)))
# a vignette-shaped combination: tdc + events + cumulative counts
add_tm("combined", c(10L, 20L, 30L), span(c(10L, 20L, 30L), c(30, 20, 25)),
       list(op("event", "death", c(10L, 30L), c(30, 25), c(1L, 2L)),
            op("tdc", "therapy", c(10L, 20L), c(5, 12)),
            op("event", "infect", c(10L, 10L, 20L, 20L), c(8, 8, 3, 25), c(1L, 1L, 1L, 1L)),
            op("cumevent", "ninf", c(10L, 10L, 20L), c(8, 15, 3)),
            op("cumtdc", "visits", c(10L, 10L, 30L), c(2, 9, 9))))
# random
rand_tm <- function(seed) {
  set.seed(seed)
  n <- sample(2:10, 1)
  base_id <- sample(100:999, n)
  rows <- list()
  for (id in base_id) {
    k <- sample(1:3, 1, prob = c(.5, .3, .2))
    t <- sample(0:4, 1)
    for (j in seq_len(k)) {
      len <- sample(1:7, 1)
      rows[[length(rows) + 1]] <- c(id, t, t + len)
      t <- t + len + sample(c(0, 0, 1, 3), 1)
    }
  }
  m <- do.call(rbind, rows)
  m <- m[sample(nrow(m)), , drop = FALSE]
  sp <- if (all(m[, 2] == 0) && !any(duplicated(m[, 1])) && runif(1) < .5)
    list(id = m[, 1], tstop = m[, 3]) else list(id = m[, 1], tstop = m[, 3], tstart = m[, 2])
  ops <- list()
  nops <- sample(2:5, 1)
  for (i in seq_len(nops)) {
    kind <- sample(c("tdc", "cumtdc", "event", "cumevent"), 1)
    name <- switch(kind, tdc = sample(c("a", "b"), 1), cumtdc = "c", event = "e", cumevent = paste0("k", i))
    m1 <- sample(0:25, 1) + 1
    ids <- sample(c(base_id, base_id, 1L), m1, TRUE)
    grid <- sample(c(1, 2), 1)
    time <- sample(-2:45, m1, TRUE) / grid
    time[runif(m1) < .05] <- NA
    use_value <- runif(1) < .7
    value <- NULL
    if (use_value) {
      value <- if (kind %in% c("event", "cumevent")) sample(c(0L, 1L, 2L, 3L), m1, TRUE)
               else sample(c(1, 2, 3.5, -1), m1, TRUE)
      if (kind != "cumevent") value[runif(m1) < .1] <- NA
    }
    init <- if (kind %in% c("tdc", "cumtdc") && use_value && runif(1) < .3) 7 else NULL
    ops[[i]] <- op(kind, name, ids, time, value, init)
  }
  options <- list()
  if (runif(1) < .3) options$delay <- sample(c(1, 2), 1)
  if (runif(1) < .3) options$tdcstart <- 0
  add_tm(paste0("random_", seed), base_id, list(span = sp), ops, options)
}
for (s in 1:60) rand_tm(s)
# first-event ranges at random
for (s in 101:110) {
  set.seed(s)
  n <- sample(2:8, 1)
  base_id <- sample(100:999, n)
  ids <- sample(base_id)
  time <- sample(1:30, n, TRUE) / 2
  ops <- list(op("event", "e", ids, time, sample(1:3, n, TRUE)),
              op("tdc", "x", sample(base_id, 8, TRUE), sample(0:30, 8, TRUE) / 2, sample(1:5, 8, TRUE) + 0.0),
              op("event", "e", sample(base_id, 8, TRUE), sample(0:30, 8, TRUE) / 2, sample(1:3, 8, TRUE)))
  ops[[1]]$value <- as.integer(ops[[1]]$value)
  add_tm(paste0("random_first_events_", s), base_id, list(span = NULL), ops)
}
add_tm("first_events_repeated_id", c(1L, 2L), list(span = NULL),
       list(op("event", "e", c(1L, 2L, 1L), c(5, 8, 7), c(1L, 2L, 1L))))

# Documented examples.
# timedep vignette, cgd: nine events per subject in wide form.
cg <- cgd0
cgops <- lapply(1:7, function(k) op("event", "infect", cg$id, cg[[paste0("etime", k)]]))
add_tm("vignette_cgd", cg$id, span(cg$id, cg$futime), cgops)
# timedep vignette, jasa heart transplants.
ja <- jasa
ja$subject <- seq_len(nrow(ja))
ja$futime <- pmax(.5, ja$fu.date - ja$accept.dt)
ja$txtime <- ifelse(ja$tx.date == ja$fu.date, ja$tx.date - ja$accept.dt - .5, ja$tx.date - ja$accept.dt)
add_tm("vignette_jasa", ja$subject, list(span = NULL),
       list(op("event", "death", ja$subject, ja$futime, as.integer(ja$fustat)),
            op("tdc", "trans", ja$subject, as.numeric(ja$txtime))))
# the tmerge help page, pbc and pbcseq.
pb <- subset(pbc, id <= 312)
add_tm("help_pbc", pb$id, list(span = NULL),
       list(op("event", "endpt", pb$id, pb$time, as.integer(pb$status)),
            op("tdc", "ascites", pbcseq$id, pbcseq$day, as.numeric(pbcseq$ascites)),
            op("tdc", "bili", pbcseq$id, pbcseq$day, pbcseq$bili),
            op("tdc", "albumin", pbcseq$id, pbcseq$day, pbcseq$albumin),
            op("tdc", "protime", pbcseq$id, pbcseq$day, pbcseq$protime),
            op("tdc", "alk.phos", pbcseq$id, pbcseq$day, pbcseq$alk.phos)))

# -------------------------------------------------------------- survcondense
cd_case <- function(name, d, merge_over_events = TRUE) {
  ds <- d[order(d$id, d$e), ]
  # Keeping events: a row and its successor get the same block number only when
  # the row has no event, so survcondense cannot merge across an event.
  ds$blk <- ave(ds$ev, ds$id, FUN = function(v) cumsum(v) - v)
  f <- if (merge_over_events) Surv(s, e, ev) ~ x1 + x2 else Surv(s, e, ev) ~ x1 + x2 + blk
  # survcondense gives wrong starts when the rows are not sorted by id and stop
  # and the merges form one block (its two special cases index the rows of the
  # data, not the sorted rows). It is run on the sorted rows, and the kept rows
  # are mapped back to the order of d.
  res <- tryCatch(survcondense(f, ds, id = id, start = "s", end = "e", event = "ev"),
                  error = function(err) conditionMessage(err))
  out <- list(name = name, merge_over_events = merge_over_events, id = I1(as.integer(d$id)),
              start = I1(num(d$s)), stop = I1(num(d$e)), status = I1(as.integer(d$ev)),
              x1 = I1(as.integer(d$x1)), x2 = I1(as.integer(d$x2)))
  if (is.character(res)) out$error <- res
  else {
    # The output has only the columns of the formula, and the kept row is the one
    # with this id and stop, which are unique in these data.
    row <- match(paste(res$id, res$e), paste(d$id, d$e))
    o <- order(row)
    out$out_start <- I1(num(res$s[o])); out$out_stop <- I1(num(res$e[o]))
    out$out_status <- I1(as.integer(res$ev[o])); out$out_row <- I1(as.integer(row[o] - 1L))
  }
  out
}
cds <- list()
add_cd <- function(...) cds[[length(cds) + 1]] <<- cd_case(...)
add_cd("help_pair", data.frame(id = c(1, 1), s = c(0, 10), e = c(10, 25), ev = c(0, 0), x1 = 1, x2 = 3))
add_cd("nothing_to_merge", data.frame(id = c(1, 1, 2), s = c(0, 12, 0), e = c(10, 25, 5), ev = c(0, 0, 1), x1 = 1, x2 = 3))
add_cd("event_in_first_row_r_rule", data.frame(id = c(1, 1), s = c(0, 10), e = c(10, 25), ev = c(1, 0), x1 = 1, x2 = 3))
add_cd("event_in_first_row_keep", data.frame(id = c(1, 1), s = c(0, 10), e = c(10, 25), ev = c(1, 0), x1 = 1, x2 = 3),
       merge_over_events = FALSE)
add_cd("whole_run", data.frame(id = rep(1, 5), s = 0:4, e = 1:5, ev = c(0, 0, 0, 0, 1), x1 = 2, x2 = 2))
add_cd("different_ids_same_times", data.frame(id = c(1, 2), s = c(0, 5), e = c(5, 9), ev = 0, x1 = 1, x2 = 1))
add_cd("shuffled_rows", data.frame(id = c(2, 1, 1, 2, 1), s = c(5, 8, 0, 0, 4), e = c(9, 12, 4, 5, 8), ev = c(0, 1, 0, 0, 0), x1 = 1, x2 = c(1, 1, 1, 1, 7)))
amlsplit <- survSplit(Surv(time, status) ~ ., data = aml, cut = c(10, 20, 30), id = "newid")
add_cd("aml_after_survsplit", data.frame(id = amlsplit$newid, s = amlsplit$tstart, e = amlsplit$time,
       ev = amlsplit$status, x1 = as.integer(amlsplit$x), x2 = 0L))
for (k in 1:25) {
  set.seed(500 + k)
  n <- sample(3:12, 1)
  ids <- sample(1:5, n, TRUE)
  rows <- do.call(rbind, lapply(sort(unique(ids)), function(i) {
    m <- sum(ids == i); t <- sample(0:3, 1); out <- NULL
    for (j in seq_len(m)) { len <- sample(1:4, 1); out <- rbind(out, c(i, t, t + len)); t <- t + len + sample(c(0, 0, 0, 2), 1) }
    out }))
  d <- data.frame(id = rows[, 1], s = rows[, 2], e = rows[, 3], ev = sample(c(0, 0, 0, 1), nrow(rows), TRUE),
                  x1 = sample(1:2, nrow(rows), TRUE), x2 = sample(1:2, nrow(rows), TRUE, prob = c(.8, .2)))
  d <- d[sample(nrow(d)), ]
  add_cd(paste0("random_", k), d, merge_over_events = (k %% 2 == 0))
}

cat(toJSON(list(
  r_version = R.version.string,
  survival_version = as.character(packageVersion("survival")),
  jsonlite_version = as.character(packageVersion("jsonlite")),
  split = sp, tmerge = tms, condense = cds
), digits = I(17), auto_unbox = TRUE, null = "null", na = "null"))
