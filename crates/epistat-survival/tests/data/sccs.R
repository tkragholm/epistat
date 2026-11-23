# The self-controlled case series of epistat-survival::sccs. Cases: one event
# per person against SCCS::standardsccs, several events per person against
# standardsccs (stacked) and glm Poisson with person dummies, events on window
# and age boundaries, pre-exposure windows with overlapping doses against an
# explicit loop for the pieces, people without events or with one piece, a
# column that is constant within people, and separation.
#
# Mapping of the conventions. SCCS counts days: observation astart..aend, an
# exposure window adrug..aedrug and an age group that starts on day c are all
# inclusive. The Rust pieces are half open on one continuous scale, so
# start = astart, end = aend + 1, window = [0, aedrug - adrug + 1) after the
# exposure date, and a cut point c is c.
#
# The references per case: the coefficients and covariance of the Poisson
# regression with a dummy for each person (equal to the conditional
# estimates), the conditional log likelihood of the formula evaluated by hand
# at them, the person-level sandwich from the numerical gradient of each
# person's log likelihood (numDeriv), and for the first cases the output of
# standardsccs. The script stops if the references disagree with each other
# beyond 1e-8 relative.
# Run: Rscript sccs.R > sccs.json
suppressMessages({
  library(jsonlite)
  library(SCCS)
  library(survival)
  library(numDeriv)
})
set.seed(20261123)

# The pieces by an explicit loop. Windows is a data frame with from and to.
hand_pieces <- function(start, end, events, expo, age_cuts, windows) {
  rows <- list()
  for (i in seq_along(start)) {
    cuts <- c(start[i], end[i], age_cuts[age_cuts > start[i] & age_cuts < end[i]])
    for (d in expo[[i]]) for (k in seq_len(nrow(windows))) {
      for (c in c(d + windows$from[k], d + windows$to[k])) {
        if (is.finite(c) && c > start[i] && c < end[i]) cuts <- c(cuts, c)
      }
    }
    cuts <- sort(unique(cuts))
    prev <- NULL
    for (m in seq_len(length(cuts) - 1)) {
      a <- cuts[m]; b <- cuts[m + 1]
      band <- sum(age_cuts <= a)
      win <- NA_integer_; best <- -Inf
      for (d in expo[[i]]) for (k in seq_len(nrow(windows))) {
        if (d + windows$from[k] <= a && a < d + windows$to[k] && d > best) {
          best <- d; win <- k - 1L
        }
      }
      y <- sum(events[[i]] >= a & events[[i]] < b)
      same <- !is.null(prev) && prev$person == i - 1L && prev$band == band &&
        identical(prev$win, win)
      if (same) {
        rows[[length(rows)]]$end <- b
        rows[[length(rows)]]$y <- rows[[length(rows)]]$y + y
        prev <- rows[[length(rows)]]
      } else {
        prev <- list(person = i - 1L, start = a, end = b, y = y, band = band, win = win)
        rows[[length(rows) + 1]] <- prev
      }
    }
  }
  data.frame(person = sapply(rows, `[[`, "person"), start = sapply(rows, `[[`, "start"),
             end = sapply(rows, `[[`, "end"), y = sapply(rows, `[[`, "y"),
             band = sapply(rows, `[[`, "band"),
             win = sapply(rows, function(r) r$win))
}

design_of <- function(pc, n_win, n_band, extra = NULL) {
  X <- NULL
  for (k in seq_len(n_win)) X <- cbind(X, as.numeric(!is.na(pc$win) & pc$win == k - 1))
  for (b in seq_len(n_band - 1)) X <- cbind(X, as.numeric(pc$band == b))
  if (!is.null(extra)) X <- cbind(X, extra[pc$person + 1])
  X
}

person_ll <- function(beta, X, len, y) {
  eta <- drop(X %*% beta) + log(len)
  m <- max(eta)
  sum(y * eta) - sum(y) * (m + log(sum(exp(eta - m))))
}

rows <- function(M) lapply(seq_len(nrow(M)), function(i) I(unname(M[i, ])))

reference <- function(pc, X) {
  len <- pc$end - pc$start
  keep_p <- tapply(pc$y, pc$person, sum)
  cnt <- tapply(pc$y, pc$person, length)
  used <- as.integer(names(keep_p)[keep_p > 0 & cnt >= 2])
  sel <- pc$person %in% used
  Xs <- X[sel, , drop = FALSE]; ps <- pc[sel, ]
  p <- ncol(X)
  dat <- data.frame(y = ps$y, pid = factor(ps$person), off = log(ps$end - ps$start))
  colnames(Xs) <- paste0("v", seq_len(p))
  dat <- cbind(dat, Xs)
  f <- as.formula(paste("y ~ 0 + pid +", paste(colnames(Xs), collapse = "+"), "+ offset(off)"))
  g <- glm(f, family = poisson, data = dat, control = glm.control(epsilon = 1e-13, maxit = 100))
  nm <- paste0("v", seq_len(p))
  beta <- coef(g)[nm]
  V <- vcov(g)[nm, nm, drop = FALSE]
  ll <- function(b) sum(sapply(used, function(i) {
    r <- ps$person == i
    person_ll(b, Xs[r, , drop = FALSE], ps$end[r] - ps$start[r], ps$y[r])
  }))
  ok <- !is.na(beta)
  S <- NULL
  if (all(ok)) {
    sc <- sapply(used, function(i) {
      r <- ps$person == i
      grad(function(b) person_ll(b, Xs[r, , drop = FALSE], ps$end[r] - ps$start[r], ps$y[r]),
           beta, method.args = list(eps = 1e-5, d = 1e-4, r = 6))
    })
    sc <- matrix(sc, nrow = p)
    meat <- sc %*% t(sc)
    Vm <- V
    rob <- Vm %*% meat %*% Vm
  } else rob <- NULL
  list(beta = I(unname(beta)), cov = if (all(ok)) rows(V) else NULL,
       robust = if (is.null(rob)) NULL else rows(rob),
       loglik = if (all(ok)) ll(beta) else NULL,
       null_loglik = ll(rep(0, p)), n_used = length(used),
       n_events = sum(ps$y))
}

spec_of <- function(start, end, events, expo, age_cuts, windows) {
  list(start = start, end = end,
       event_offsets = c(0, cumsum(sapply(events, length))),
       events = unlist(events),
       exposure_offsets = c(0, cumsum(sapply(expo, length))),
       exposures = unlist(expo), age_cuts = age_cuts,
       windows = lapply(seq_len(nrow(windows)), function(k)
         list(from = windows$from[k], to = windows$to[k])))
}

as_pieces <- function(pc) list(person = pc$person, start = pc$start, end = pc$end,
  events = pc$y, age_band = pc$band, window = ifelse(is.na(pc$win), -1L, pc$win))

cases <- list()
add <- function(label, start, end, events, expo, age_cuts, windows, extra = NULL, sccs = NULL) {
  pc <- hand_pieces(start, end, events, expo, age_cuts, windows)
  X <- design_of(pc, nrow(windows), length(age_cuts) + 1, extra)
  r <- reference(pc, X)
  cases[[length(cases) + 1]] <<- list(label = label,
    spec = spec_of(start, end, events, expo, age_cuts, windows),
    extra = extra, pieces = as_pieces(pc), n_windows = nrow(windows),
    n_age_bands = length(age_cuts) + 1, glm = r, sccs = sccs)
  invisible(r)
}

sccs_out <- function(fit) list(
  coef = I(unname(fit$coefficients[, "coef"])), vcov = rows(fit$vcov),
  loglik = I(unname(fit$loglik)), n = fit$n)

# Case 1 and 2: single and several events, windows [0,10) and [10,28).
make_people <- function(n, multi) {
  astart <- sample(100:300, n, TRUE)
  aend <- astart + sample(400:800, n, TRUE)
  adrug <- astart + sample(20:300, n, TRUE)
  aedrug <- adrug + 27
  ne <- if (multi) sample(1:3, n, TRUE, prob = c(.5, .3, .2)) else rep(1, n)
  ev <- lapply(seq_len(n), function(i) {
    # more events in days 0-9 after the exposure than at other times
    e <- numeric(ne[i])
    for (j in seq_len(ne[i])) {
      e[j] <- if (runif(1) < 0.2) adrug[i] + sample(0:9, 1) else sample(astart[i]:aend[i], 1)
    }
    pmin(e, aend[i])
  })
  # standardsccs drops a repeated event day (unique() on the stacked rows)
  ev <- lapply(ev, function(e) sort(unique(e)))
  ne <- sapply(ev, length)
  list(astart = astart, aend = aend, adrug = adrug, aedrug = aedrug, ev = ev, ne = ne)
}
for (multi in c(FALSE, TRUE)) {
  n <- if (multi) 300 else 400
  P <- make_people(n, multi)
  windows <- data.frame(from = c(0, 10), to = c(10, 28))
  ds <- data.frame(indiv = rep(seq_len(n), P$ne), astart = rep(P$astart, P$ne),
                   aend = rep(P$aend, P$ne), aevent = unlist(P$ev),
                   adrug = rep(P$adrug, P$ne), aedrug = rep(P$aedrug, P$ne))
  sf <- standardsccs(event ~ adrug + age, indiv = indiv, astart = astart, aend = aend,
                     aevent = aevent, adrug = adrug, aedrug = aedrug,
                     expogrp = c(0, 10), agegrp = c(250, 400, 600), data = ds)
  r <- add(if (multi) "several events, against SCCS" else "one event, against SCCS",
           P$astart, P$aend + 1, P$ev, as.list(P$adrug), c(250, 400, 600), windows,
           sccs = sccs_out(sf))
  stopifnot(max(abs(r$beta - sf$coefficients[, "coef"]) / abs(sf$coefficients[, "coef"])) < 1e-6)
  if (!multi) {
    # clogit with offset on the single events, the exact likelihood
    pc <- hand_pieces(P$astart, P$aend + 1, P$ev, as.list(P$adrug), c(250, 400, 600), windows)
    X <- design_of(pc, 2, 4)
    colnames(X) <- paste0("v", 1:5)
    d <- data.frame(pc, X, case = pc$y, len = pc$end - pc$start)
    cl <- clogit(case ~ v1 + v2 + v3 + v4 + v5 + strata(person) + offset(log(len)), data = d,
                 method = "exact",
                 control = coxph.control(eps = 1e-12, toler.chol = 1e-14, iter.max = 100))
    stopifnot(max(abs(coef(cl) - r$beta)) < 1e-8)
    cases[[length(cases)]]$clogit <- list(coef = I(unname(coef(cl))), vcov = rows(vcov(cl)),
                                          loglik = I(unname(cl$loglik)))
  }
}

# Case 3: events on window and age boundaries, observation start and last day.
n <- 120
astart <- sample(100:200, n, TRUE); aend <- astart + sample(300:500, n, TRUE)
adrug <- astart + sample(30:200, n, TRUE); aedrug <- adrug + 27
ev <- lapply(seq_len(n), function(i) {
  pool <- c(adrug[i], adrug[i] + 9, adrug[i] + 10, adrug[i] + 27, adrug[i] + 28,
            astart[i], aend[i], 250, 400)
  pool <- pool[pool >= astart[i] & pool <= aend[i]]
  sort(unique(sample(pool, sample(1:3, 1), replace = TRUE)))
})
ds <- data.frame(indiv = rep(seq_len(n), sapply(ev, length)),
                 astart = rep(astart, sapply(ev, length)), aend = rep(aend, sapply(ev, length)),
                 aevent = unlist(ev), adrug = rep(adrug, sapply(ev, length)),
                 aedrug = rep(aedrug, sapply(ev, length)))
sf <- standardsccs(event ~ adrug + age, indiv = indiv, astart = astart, aend = aend,
                   aevent = aevent, adrug = adrug, aedrug = aedrug,
                   expogrp = c(0, 10), agegrp = c(250, 400), data = ds)
r <- add("events on boundaries, against SCCS", astart, aend + 1, ev, as.list(adrug),
         c(250, 400), data.frame(from = c(0, 10), to = c(10, 28)), sccs = sccs_out(sf))
stopifnot(max(abs(r$beta - sf$coefficients[, "coef"]) / abs(sf$coefficients[, "coef"])) < 1e-6)

# Case 4: a pre-exposure window, several doses with overlapping windows,
# exposure dates before the start of observation, an open-ended last window.
n <- 250
astart <- sample(0:100, n, TRUE); aend <- astart + sample(300:700, n, TRUE)
nd <- sample(0:3, n, TRUE, prob = c(.1, .4, .3, .2))
expo <- lapply(seq_len(n), function(i) {
  if (nd[i] == 0) return(numeric(0))
  sort(astart[i] + sample(-30:400, nd[i]))
})
windows <- data.frame(from = c(-14, 0, 7), to = c(0, 7, 30))
ev <- lapply(seq_len(n), function(i) {
  ne <- sample(1:3, 1)
  e <- sample(astart[i]:(aend[i] - 1), ne, replace = TRUE)
  if (length(expo[[i]]) > 0 && runif(1) < 0.4) e[1] <- expo[[i]][1] + sample(0:6, 1)
  e <- pmin(pmax(e, astart[i]), aend[i] - 1)
  sort(e)
})
add("pre-exposure window and overlapping doses", astart, aend, ev, expo, c(200, 450), windows)

# Case 5: people without events, with one piece, an extra column constant
# within people (aliased) and a column with variation.
n <- 150
astart <- rep(0, n); aend <- sample(200:400, n, TRUE)
expo <- lapply(seq_len(n), function(i) if (i %% 3 == 0) numeric(0) else sample(20:150, 1))
ev <- lapply(seq_len(n), function(i) {
  if (i %% 10 == 0) return(numeric(0))
  sort(sample(0:(aend[i] - 1), sample(1:2, 1), replace = TRUE))
})
extra <- rbinom(n, 1, 0.5)
windows <- data.frame(from = 0, to = 14)
pc <- hand_pieces(astart, aend, ev, expo, numeric(0), windows)
X <- design_of(pc, 1, 1, extra)
r <- reference(pc, X[, 1, drop = FALSE])
cases[[length(cases) + 1]] <- list(label = "dropped people and an aliased column",
  spec = spec_of(astart, aend, ev, expo, numeric(0), windows), extra = extra,
  pieces = as_pieces(pc), n_windows = 1, n_age_bands = 1, glm = r, aliased_extra = TRUE)

# Case 6: separation, every event inside the window.
n <- 60
astart <- rep(0, n); aend <- rep(200, n)
expo <- lapply(seq_len(n), function(i) 50)
ev <- lapply(seq_len(n), function(i) sort(sample(50:59, sample(1:2, 1), replace = TRUE)))
windows <- data.frame(from = 0, to = 10)
pc <- hand_pieces(astart, aend, ev, expo, numeric(0), windows)
cases[[length(cases) + 1]] <- list(label = "separation",
  spec = spec_of(astart, aend, ev, expo, numeric(0), windows), extra = NULL,
  pieces = as_pieces(pc), n_windows = 1, n_age_bands = 1, glm = NULL, separation = TRUE)

out <- list(versions = list(R = R.version.string, SCCS = as.character(packageVersion("SCCS")),
  survival = as.character(packageVersion("survival")),
  numDeriv = as.character(packageVersion("numDeriv"))), cases = cases)
cat(toJSON(out, auto_unbox = TRUE, digits = NA, na = "string", null = "null"))
