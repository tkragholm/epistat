# The Titman-Putter test of the Markov property by mstate::MarkovTest (the
# function Titman wrote, shipped in mstate), on a four-state process with a
# reversible middle:
#   1 -> 2 (trans 1), 1 -> 3 (trans 2), 2 -> 1 (trans 3), 2 -> 3 (trans 4),
#   2 -> 4 (trans 5), 4 -> 2 (trans 6); state 3 is absorbing.
# Sojourn times are Weibull and the rate out of state 2 depends on the time
# spent in state 1 before, so the process is not Markov. Times are rounded to
# one decimal, so exits tie, and there are re-entries. Persons are censored
# administratively between 4 and 12.
# For each tested transition the script records the stays (the input of the
# Rust test), the landmark grid, the qualifying states, and from MarkovTest the
# normalised traces (zbar), the variances (est_cov diagonal), orig_stat,
# nobs_grid and the wild-bootstrap p-values (dist = "normal", mean of
# "bootstrap statistic > observed", with B draws). The sums of absolute terms
# of the trace and the largest cumulative intensity are computed here straight
# from the definitions, for the rounding-error bound of the test. The variance
# is a sum of non-negative terms and needs no such sum. The number of stays at risk whose
# person occupied each qualifying state at each landmark is counted here by a
# plain loop, since MarkovTest does not return it.
# Run: Rscript markov.R > markov.json
suppressMessages({
  library(jsonlite)
  library(mstate)
})
set.seed(20261120)
n <- 220
B <- 4000

sojourn <- function(shape, scale) max(0.1, round(rweibull(1, shape, scale), 1))
rows <- list()
for (i in seq_len(n)) {
  cens <- round(runif(1, 4, 12), 1)
  t <- 0; s <- 1; first1 <- NA; from_prev <- 0
  repeat {
    if (s == 1) {
      a <- sojourn(1.6, 4); b <- sojourn(1.2, 9)
      d <- min(a, b); to <- if (a <= b) 2 else 3
    } else if (s == 2) {
      long <- if (is.na(first1)) 0 else first1
      f <- 1 + 0.35 * long                   # memory of the time spent in state 1
      x1 <- sojourn(1.3, 5 * f); x3 <- sojourn(1.0, 7 / f); x4 <- sojourn(1.4, 6)
      d <- min(x1, x3, x4)
      to <- c(1, 3, 4)[which.min(c(x1, x3, x4))]
    } else {
      d <- sojourn(1.2, 3); to <- 2
    }
    if (s == 1 && is.na(first1)) first1 <- d
    stop_t <- round(t + d, 1)
    if (stop_t >= cens) {
      rows[[length(rows) + 1]] <- data.frame(id = i, from = s, start = t, stop = cens, to = 0)
      break
    }
    rows[[length(rows) + 1]] <- data.frame(id = i, from = s, start = t, stop = stop_t, to = to)
    t <- stop_t; s <- to
    if (s == 3) break
  }
}
stays <- do.call(rbind, rows)

trans <- transMat(list(c(2, 3), c(1, 3, 4), c(), c(2)), names = c("1", "2", "3", "4"))
# msdata: one row per possible transition out of the state of each stay
long <- list()
for (r in seq_len(nrow(stays))) {
  st <- stays[r, ]
  targets <- which(!is.na(trans[st$from, ]))
  for (tg in targets) {
    long[[length(long) + 1]] <- data.frame(
      id = st$id, from = st$from, to = tg, trans = trans[st$from, tg],
      Tstart = st$start, Tstop = st$stop, status = as.integer(st$to == tg)
    )
  }
}
msd <- do.call(rbind, long)
msd <- msd[order(msd$id, msd$Tstart, msd$trans), ]
rownames(msd) <- NULL
attr(msd, "trans") <- trans
class(msd) <- c("msdata", "data.frame")

grid <- c(1, 2, 3, 4, 5, 6)
tests <- list()
pkgs <- c("jsonlite", "mstate", "survival")
for (tr in c(1, 2, 4)) {
  tt <- to.trans2(trans)
  tfrom <- tt$from[tt$transno == tr]; tto <- tt$to[tt$transno == tr]
  set.seed(1000 + tr)
  mt <- suppressWarnings(MarkovTest(msd, id = "id", transition = tr, grid = grid,
                                    B = B, dist = "normal"))
  qual <- mt$qualset
  # Rust codes: the origin 0, the other qualifying states 1.., any other state q
  others <- setdiff(qual, tfrom)
  code_of <- function(s) {
    if (s == tfrom) return(0L)
    k <- match(s, others)
    if (is.na(k)) length(qual) else k
  }
  # position of each Rust state in the qualset column order of MarkovTest
  col_of_code <- c(match(tfrom, qual), match(others, qual))
  S <- stays
  S$code <- vapply(S$from, code_of, 1L)
  S$event <- as.integer(S$to == tto)
  at_risk <- S[S$from == tfrom, ]
  # independent counts and absolute sums, by the definitions
  occ <- matrix(0, length(grid), length(qual))
  for (g in seq_along(grid)) {
    for (k in seq_len(nrow(at_risk))) {
      p <- at_risk$id[k]
      cover <- S[S$id == p & S$start < grid[g] & S$stop >= grid[g], ]
      if (nrow(cover) > 0) {
        c0 <- cover$code[nrow(cover)]
        if (c0 < length(qual)) occ[g, c0 + 1] <- occ[g, c0 + 1] + 1
      }
    }
  }
  # Nelson-Aalen of the transition, for the absolute sums
  y <- vapply(at_risk$stop, function(u) sum(at_risk$start < u & at_risk$stop >= u), 1)
  inc <- at_risk$event / y
  H <- function(u) sum(inc[at_risk$stop <= u])
  trace_abs <- matrix(0, length(grid), length(qual))
  for (g in seq_along(grid)) {
    after <- as.numeric(at_risk$stop > grid[g])
    m <- vapply(seq_len(nrow(at_risk)), function(i) {
      at_risk$event[i] * after[i] -
        (H(max(grid[g], at_risk$stop[i])) - H(max(grid[g], at_risk$start[i])))
    }, 1)
    for (j in seq_along(qual)) {
      ind <- vapply(seq_len(nrow(at_risk)), function(i) {
        cover <- S[S$id == at_risk$id[i] & S$start < grid[g] & S$stop >= grid[g], ]
        as.numeric(nrow(cover) > 0 && cover$code[nrow(cover)] == (j - 1))
      }, 1)
      trace_abs[g, j] <- sum(abs(m * ind * after))
    }
  }
  tests[[length(tests) + 1]] <- list(
    transition = tr, from = tfrom, to = tto, qualset = qual,
    col_of_code = col_of_code, grid = grid, B = B,
    stays = list(person = S$id - 1L, start = S$start, stop = S$stop,
                 state = S$code, event = S$event),
    zbar = unname(mt$zbar), variance = unname(apply(mt$est_cov, 1, function(m) diag(m)) |> t()),
    orig_stat = c(mt$orig_stat), nobs_grid = mt$nobs_grid,
    p_stat_wb = c(mt$p_stat_wb), nsub = mt$Nsub, occupied = unname(occ),
    trace_abs = unname(trace_abs), h_max = sum(inc), n_at_risk = nrow(at_risk)
  )
}
versions <- c(list(R = R.version.string),
              setNames(lapply(pkgs, function(p) as.character(packageVersion(p))), pkgs))
cat(toJSON(list(versions = versions, tests = tests), digits = NA, auto_unbox = TRUE, na = "null"))
