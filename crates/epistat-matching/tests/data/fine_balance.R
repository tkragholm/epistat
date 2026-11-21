# Fine, near-fine and refined balance, checked against rcbalance and
# designmatch.
#
# Run from the crate directory:
#   Rscript tests/data/fine_balance.R
# It writes tests/data/fine_balance.json.
#
# rcbalance::rcbalance() solves with rlemon (the LEMON network simplex). It
# scales distances by 1/tol and rounds, so its total is within about
# 0.5 * tol per pair of the optimum, and it weights the levels of refined
# balance with penalties of base 3 (its `penalty` argument), which orders the
# levels only approximately. The balance tables are compared as given.
#
# designmatch::bmatch() with `fine` solves the same pair matching with exact
# fine balance as an integer program (glpk through Rglpk). It is used on the
# scenarios where exact fine balance is feasible.

suppressMessages({
  library(rcbalance)
  library(designmatch)
  library(jsonlite)
})

tol <- 1e-6
set.seed(20261121)

make_data <- function(nt, nc, p_t, p_c, shift = 0.4) {
  lev <- c("p", "q", "r")
  info <- function(n, prob, mu) {
    data.frame(
      a = sample(lev, n, TRUE, prob),
      b = sample(c("u", "v"), n, TRUE),
      s = sample(c("s1", "s2"), n, TRUE),
      x1 = rnorm(n, mu), x2 = rnorm(n, mu / 2),
      stringsAsFactors = FALSE
    )
  }
  list(t = info(nt, p_t, shift), c = info(nc, p_c, 0))
}

dist_matrix <- function(d, caliper = NULL, strata = FALSE) {
  m <- sqrt(outer(d$t$x1, d$c$x1, "-")^2 + outer(d$t$x2, d$c$x2, "-")^2)
  if (!is.null(caliper)) m[m > caliper] <- Inf
  if (strata) m[outer(d$t$s, d$c$s, "!=")] <- Inf
  m
}

balance_of <- function(matches, d, keys, k) {
  mc <- as.vector(matches)
  lab <- function(df) do.call(paste, c(lapply(keys, function(key) df[[key]]), sep = "."))
  tl <- lab(d$t); cl <- lab(d$c)
  cats <- sort(unique(c(tl, cl)))
  list(
    keys = I(keys), categories = I(cats),
    target = I(as.integer(k * table(factor(tl, levels = cats)))),
    matched = I(as.integer(table(factor(cl[mc], levels = cats))))
  )
}

run <- function(name, d, fb, k = 1, caliper = NULL, strata = FALSE, dm = FALSE) {
  m <- dist_matrix(d, caliper, strata)
  out <- list(name = name, k = k, caliper = caliper, strata = strata,
              fb = lapply(fb, I), nt = nrow(d$t), nc = nrow(d$c),
              t = d$t, c = d$c)
  call <- function(mm) {
    tryCatch(
      rcbalance(mm, fb.list = fb, treated.info = d$t, control.info = d$c,
                k = k, tol = tol),
      error = function(e) NULL)
  }
  r <- call(m)
  if (is.null(r)) {
    out$error <- TRUE
    return(out)
  }
  out$error <- FALSE
  mt <- r$matches
  out$matches <- lapply(seq_len(nrow(mt)), function(i) I(as.integer(mt[i, ])))
  out$total <- sum(vapply(seq_len(nrow(mt)), function(i) sum(m[i, mt[i, ]]), 0))
  out$levels <- lapply(fb, function(keys) balance_of(mt, d, keys, k))
  # The optimum is unique when small relative perturbations of the distances
  # give the same pairs.
  same <- TRUE
  for (rep in 1:3) {
    mp <- m * (1 + 1e-4 * runif(length(m)))
    rp <- call(mp)
    if (is.null(rp) || !identical(sort(rp$matches), sort(mt)) ||
        !all(rp$matches == mt)) {
      same <- FALSE
    }
  }
  out$unique <- same
  out$tol <- tol
  if (dm && length(fb) == 1) {
    keys <- fb[[1]]
    cov <- sapply(keys, function(key) as.integer(factor(c(d$t[[key]], d$c[[key]]))))
    fine_var <- if (length(keys) == 1) cov[, 1] else as.integer(factor(do.call(paste, as.data.frame(cov))))
    t_ind <- c(rep(1, nrow(d$t)), rep(0, nrow(d$c)))
    # designmatch wants integer distances; scale as rcbalance does.
    mi <- m; mi[!is.finite(mi)] <- 1e6
    dmat <- round(mi / tol)
    res <- tryCatch(
      bmatch(t_ind = t_ind, dist_mat = dmat,
             fine = list(covs = matrix(fine_var, ncol = 1)),
             total_groups = nrow(d$t),
             solver = list(name = "glpk", t_max = 300, approximate = 0,
                           round_cplex = 0, trace = 0)),
      error = function(e) NULL)
    if (!is.null(res) && !is.null(res$t_id)) {
      out$designmatch <- list(
        total = sum(m[cbind(res$t_id, res$c_id - nrow(d$t))]),
        n = length(res$t_id))
    }
  }
  out
}

runs <- list()
d1 <- make_data(30, 120, c(.4, .35, .25), c(.34, .33, .33))
runs$feasible_k1 <- run("feasible_k1", d1, list("a"), 1, dm = TRUE)
d2 <- make_data(25, 150, c(.4, .35, .25), c(.34, .33, .33))
runs$feasible_k2 <- run("feasible_k2", d2, list("a"), 2)
d3 <- make_data(30, 70, c(.3, .3, .4), c(.5, .45, .05))
runs$near_fine <- run("near_fine", d3, list("a"), 1)
d4 <- make_data(30, 90, c(.3, .3, .4), c(.45, .45, .10))
runs$refined <- run("refined", d4, list("a", c("a", "b")), 1)
runs$interaction <- run("interaction", d1, list(c("a", "b")), 1, dm = TRUE)
runs$caliper <- run("caliper", d1, list("a"), 1, caliper = 1.2)
runs$strata <- run("strata", d3, list("a"), 1, strata = TRUE)
runs$strata_caliper_refined <- run("strata_caliper_refined", d4,
                                   list("a", c("a", "b")), 1,
                                   caliper = 1.9, strata = TRUE)
runs$infeasible_pairs <- run("infeasible_pairs", d1, list("a"), 1, caliper = 0.05)

versions <- list(
  r = R.version.string,
  rcbalance = as.character(packageVersion("rcbalance")),
  designmatch = as.character(packageVersion("designmatch")),
  rlemon = as.character(packageVersion("rlemon"))
)
write_json(list(versions = versions, runs = unname(runs)),
           "tests/data/fine_balance.json", digits = I(17), auto_unbox = TRUE,
           null = "null", na = "null", dataframe = "columns")
for (r in runs) cat(r$name, "error", r$error, "unique", r$unique, "total", r$total,
                    "dm", if (!is.null(r$designmatch)) r$designmatch$total else NA, "\n")
