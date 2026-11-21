# Genetic matching for fixed weights, from MatchIt 4.8.2 (method = "genetic"),
# Matching 4.10-15 (Match and GenMatch) and rgenoud.
#
# MatchIt's genetic method searches for weights with rgenoud, whose random path
# cannot be reproduced. To fix the weights, the script gives genoud a search
# range of width 2e-9 (relative) around the chosen weights, through the
# `Domains` argument that matchit() passes on to GenMatch. Whatever genoud
# returns is then equal to the chosen weights to 1e-9, and the script reads
# the weights it did use from GenMatch's `par`. Three things are recorded
# per run (the header continues below):
#   matchit    the pairs of matchit(method = "genetic") with those weights,
#              as the controls of each treated unit in the order MatchIt gives
#   match      the pairs of Matching::Match(Weight = 3, Weight.matrix = diag(par)),
#              which the script requires to equal `matchit`
#   losses     GenMatch's fitness at those weights for every fit function and
#              loss, from GenMatch itself with the same range, and the
#              p-values behind the default one
# GenMatch rejects fit.func = "qqmax.mean" and "qqmax.max" as invalid although
# its code handles both; the values for them are computed from Match's rows
# with Matching::qqstats (matchit_genetic.R, "losses").
#
# The data are R's lalonde with its continuous covariates jittered, so that no
# two controls tie in distance (Match breaks ties at random from R's stream,
# which the crate cannot reproduce), and a simulated cohort.
#
# The script also records two-sample Kolmogorov-Smirnov p-values from ks.test
# on samples with ties, and the bootstrap p-value of GenMatch repeated with
# different seeds, which the crate checks by distribution.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/matchit_genetic.R
suppressMessages({
  library(MatchIt)
  library(Matching)
  library(rgenoud)
  library(jsonlite)
})
args <- commandArgs(FALSE)
here <- dirname(sub("--file=", "", args[grep("--file=", args)]))
set.seed(20261123)

# ---- data -----------------------------------------------------------------

data(lalonde, package = "MatchIt")
lal <- lalonde
lal$race <- as.character(lal$race)
lal$black <- as.numeric(lal$race == "black")
lal$hispan <- as.numeric(lal$race == "hispan")
rownames(lal) <- NULL
for (v in c("age", "educ", "re74", "re75")) {
  lal[[v]] <- lal[[v]] + rnorm(nrow(lal), 0, 0.05 * sd(lal[[v]]))
}
lal$g <- sample(c("a", "b", "c", "d", "e"), nrow(lal), replace = TRUE)
lal_cov <- c("age", "educ", "re74", "re75", "married", "nodegree", "black", "hispan")
lal$ps <- unname(predict(glm(reformulate(lal_cov, "treat"), data = lal, family = binomial),
                         type = "response"))

n_sim <- 300
sim <- data.frame(treat = rbinom(n_sim, 1, 0.25))
sim$z1 <- rnorm(n_sim) + 0.5 * sim$treat
sim$z2 <- rnorm(n_sim)
sim$z3 <- rexp(n_sim)
sim$b <- rbinom(n_sim, 1, 0.4 + 0.2 * sim$treat)
sim$g <- sample(c("a", "b", "c"), n_sim, replace = TRUE)
sim_cov <- c("z1", "z2", "z3", "b")
sim$ps <- unname(predict(glm(reformulate(sim_cov, "treat"), data = sim, family = binomial),
                         type = "response"))

scenarios <- list(
  lalonde = list(d = lal, cov = lal_cov, exact = "married"),
  sim = list(d = sim, cov = sim_cov, exact = "b")
)

# ---- one run --------------------------------------------------------------

pop_sd <- function(x) sqrt(sum((x - mean(x))^2) / length(x))

# The inputs GenMatch gets from matchit2genetic for m.order = "largest" (the
# default) or "smallest" and "data", rebuilt here.
genetic_inputs <- function(sc, run) {
  d <- sc$d
  treat <- as.integer(d$treat == 1)
  n <- nrow(d)
  ord <- switch(run$order,
    largest = order(d$ps, decreasing = TRUE),
    smallest = order(d$ps),
    data = seq_len(n)
  )
  covs <- as.matrix(d[, sc$cov])
  X <- cbind(covs, ps = d$ps)
  exact_log <- NULL
  if (run$exact) {
    X <- cbind(X, ex = as.integer(factor(d[[sc$exact]])))
    exact_log <- c(rep(FALSE, ncol(X) - 1L), TRUE)
  }
  cal <- NULL
  if (!is.null(run$cal_cov) || !is.null(run$cal_ps)) {
    cal <- setNames(rep(Inf, ncol(X)), colnames(X))
    if (!is.null(run$cal_cov)) {
      cal[names(run$cal_cov)] <- unlist(run$cal_cov) / vapply(names(run$cal_cov), function(v) pop_sd(X[ord, v]), 1)
    }
    if (!is.null(run$cal_ps)) {
      cal["ps"] <- run$cal_ps / pop_sd(d$ps[ord])
    }
  }
  restrict <- NULL
  if (run$anti) {
    tt <- treat[ord]
    gg <- d$g[ord]
    restrict <- do.call("rbind", lapply(which(tt == 1), function(j) {
      rc <- which(tt == 0 & gg == gg[j])
      if (!length(rc)) NULL else cbind(j, rc, -1L)
    }))
  }
  list(Tr = treat[ord], X = X[ord, , drop = FALSE], bal = covs[ord, , drop = FALSE],
       exact = exact_log, caliper = cal, restrict = restrict, ord = ord)
}

pinned <- function(w) cbind(w * (1 - 1e-9), w * (1 + 1e-9))

genmatch_at <- function(inp, run, w, fit = list()) {
  g <- suppressWarnings(do.call(GenMatch, c(list(
    Tr = inp$Tr, X = inp$X, BalanceMatrix = inp$bal, M = run$ratio, exact = inp$exact,
    caliper = inp$caliper, replace = run$replace, estimand = "ATT", ties = FALSE,
    CommonSupport = FALSE, weights = NULL, restrict = inp$restrict,
    pop.size = 6, max.generations = 1, wait.generations = 1, Domains = pinned(w),
    starting.values = w, print.level = 0, distance.tolerance = 0
  ), fit)))
  g
}

run_one <- function(name, sc, run) {
  d <- sc$d
  p <- length(sc$cov) + 1L + as.integer(run$exact)
  w <- round(runif(p, 0.2, 5), 3)
  if (isTRUE(run$equal)) w <- rep(1, p)
  inp <- genetic_inputs(sc, run)
  g <- genmatch_at(inp, run, w)
  par <- g$par
  # MatchIt, with the same range for the weights.
  cal_arg <- NULL
  if (!is.null(run$cal_cov) || !is.null(run$cal_ps)) {
    cal_arg <- c(unlist(run$cal_cov), if (!is.null(run$cal_ps)) c(run$cal_ps))
    names(cal_arg) <- c(names(run$cal_cov), if (!is.null(run$cal_ps)) "")
  }
  mi_args <- list(
    formula = reformulate(sc$cov, "treat"), data = d, method = "genetic", distance = d$ps,
    ratio = run$ratio, replace = run$replace, m.order = run$order, caliper = cal_arg,
    std.caliper = FALSE, exact = if (run$exact) reformulate(sc$exact) else NULL,
    antiexact = if (run$anti) ~g else NULL,
    pop.size = 6, max.generations = 1, wait.generations = 1, Domains = pinned(w),
    starting.values = w
  )
  mi_args <- mi_args[!vapply(mi_args, is.null, TRUE)]
  m <- suppressWarnings(do.call(matchit, mi_args))
  mm <- m$match.matrix
  treated <- as.integer(rownames(mm))
  matchit_rows <- lapply(seq_len(nrow(mm)), function(i) {
    v <- as.integer(mm[i, ])
    v[!is.na(v)]
  })
  # Match with the weights GenMatch reports.
  mo <- suppressWarnings(Match(
    Tr = inp$Tr, X = inp$X, M = run$ratio, exact = inp$exact, caliper = inp$caliper,
    replace = run$replace, estimand = "ATT", ties = FALSE, CommonSupport = FALSE,
    distance.tolerance = 0, Weight = 3, Weight.matrix = diag(par), restrict = inp$restrict,
    version = "fast"
  ))
  by_treated <- split(inp$ord[mo$index.control], inp$ord[mo$index.treated])
  stopifnot(setequal(names(by_treated), as.character(treated[lengths(matchit_rows) > 0])))
  for (i in seq_along(treated)) {
    got <- by_treated[[as.character(treated[i])]]
    stopifnot(setequal(got, matchit_rows[[i]]))
  }
  list(
    scenario = name, run = run[c("order", "ratio", "replace", "exact", "anti")],
    cal_cov = run$cal_cov, cal_ps = run$cal_ps,
    weights = par, order = inp$ord, treated = treated, matchit = matchit_rows,
    match_n = nrow(mo$index.treated), value = g$value, inp = inp
  )
}

runs <- list(
  list(order = "largest", ratio = 1, replace = FALSE, exact = FALSE, anti = FALSE),
  list(order = "largest", ratio = 1, replace = TRUE, exact = FALSE, anti = FALSE),
  list(order = "largest", ratio = 2, replace = FALSE, exact = FALSE, anti = FALSE),
  list(order = "largest", ratio = 2, replace = TRUE, exact = FALSE, anti = FALSE),
  list(order = "smallest", ratio = 3, replace = TRUE, exact = FALSE, anti = FALSE),
  list(order = "data", ratio = 1, replace = FALSE, exact = FALSE, anti = FALSE),
  list(order = "largest", ratio = 1, replace = FALSE, exact = FALSE, anti = FALSE, equal = TRUE),
  list(order = "largest", ratio = 2, replace = FALSE, exact = FALSE, anti = FALSE, cal_ps = 0.1),
  list(order = "largest", ratio = 1, replace = TRUE, exact = FALSE, anti = FALSE, cal_ps = 0.05),
  list(order = "largest", ratio = 2, replace = FALSE, exact = TRUE, anti = FALSE),
  list(order = "largest", ratio = 1, replace = TRUE, exact = FALSE, anti = TRUE),
  list(order = "smallest", ratio = 2, replace = FALSE, exact = TRUE, anti = TRUE, cal_ps = 0.2)
)
cal_runs <- list(
  lalonde = list(age = 4),
  sim = list(z1 = 0.7)
)

out_runs <- list()
for (name in names(scenarios)) {
  sc <- scenarios[[name]]
  for (run in runs) {
    r <- run
    out_runs[[length(out_runs) + 1]] <- run_one(name, sc, r)
  }
  r <- list(order = "largest", ratio = 2, replace = TRUE, exact = FALSE, anti = FALSE,
            cal_cov = cal_runs[[name]])
  out_runs[[length(out_runs) + 1]] <- run_one(name, sc, r)
  r <- list(order = "largest", ratio = 1, replace = FALSE, exact = TRUE, anti = FALSE,
            cal_cov = cal_runs[[name]], cal_ps = 0.1)
  out_runs[[length(out_runs) + 1]] <- run_one(name, sc, r)
}

# ---- losses ---------------------------------------------------------------

fits <- list(
  pvals = list(),
  pvals_minimum = list(loss = 2),
  pvals_no_ks = list(ks = FALSE),
  pvals_unpaired = list(paired = FALSE),
  qqmean.mean = list(fit.func = "qqmean.mean"),
  qqmean.max = list(fit.func = "qqmean.max"),
  qqmax.mean = "qqmax.mean",
  qqmax.max = "qqmax.max",
  qqmedian.median = list(fit.func = "qqmedian.median"),
  qqmedian.max = list(fit.func = "qqmedian.max")
)
for (k in seq_along(out_runs)) {
  r <- out_runs[[k]]
  sc <- scenarios[[r$scenario]]
  run <- r$run
  run$cal_cov <- r$cal_cov
  run$cal_ps <- r$cal_ps
  inp <- r$inp
  use <- if ((k - 1) %% 4 == 0) names(fits) else "pvals"
  losses <- list()
  for (nm in use) {
    if (nm == "pvals") {
      value <- r$value
    } else if (is.character(fits[[nm]])) {
      # GenMatch's check of `fit.func` rejects "qqmax.mean" and "qqmax.max",
      # which its code handles further down (see the header). The values are
      # Matching::qqstats on the rows of Match, summarised as GenMatch's code
      # for them does.
      mo <- suppressWarnings(Match(
        Tr = inp$Tr, X = inp$X, M = run$ratio, exact = inp$exact, caliper = inp$caliper,
        replace = run$replace, estimand = "ATT", ties = FALSE, CommonSupport = FALSE,
        distance.tolerance = 0, Weight = 3, Weight.matrix = diag(r$weights),
        restrict = inp$restrict, version = "fast"
      ))
      mx <- vapply(seq_len(ncol(inp$bal)), function(i) {
        qqstats(inp$bal[mo$index.treated, i], inp$bal[mo$index.control, i], standardize = TRUE)$maxdiff
      }, 1)
      value <- if (nm == "qqmax.mean") mean(mx) else sort(mx, decreasing = TRUE)
    } else {
      g <- genmatch_at(inp, run, r$weights, fits[[nm]])
      value <- g$value
    }
    losses[[length(losses) + 1]] <- list(fit = nm, value = as.numeric(value))
  }
  out_runs[[k]]$losses <- losses
  out_runs[[k]]$inp <- NULL
  out_runs[[k]]$value <- NULL
}

# ---- bootstrap Kolmogorov-Smirnov p-values --------------------------------

# One evaluation of GenMatch (a population of one, no generations) with
# nboots = 500, thirty times, on the first lalonde run: sorted p-values.
r1 <- out_runs[[1]]
sc1 <- scenarios[["lalonde"]]
run1 <- list(order = "largest", ratio = 1, replace = FALSE, exact = FALSE, anti = FALSE)
inp1 <- genetic_inputs(sc1, run1)
boot <- lapply(1:30, function(i) {
  g <- suppressWarnings(GenMatch(
    Tr = inp1$Tr, X = inp1$X, BalanceMatrix = inp1$bal, M = 1, replace = FALSE, ties = FALSE,
    CommonSupport = FALSE, pop.size = 1, max.generations = 0, wait.generations = 0,
    Domains = pinned(r1$weights), starting.values = r1$weights, print.level = 0,
    distance.tolerance = 0, nboots = 500
  ))
  as.numeric(g$value)
})

# ---- Kolmogorov-Smirnov p-values ------------------------------------------

mks <- getFromNamespace("Mks.test", "Matching")
ks_cases <- list()
sizes <- list(c(5, 7), c(8, 8), c(20, 20), c(30, 40), c(60, 60), c(99, 99), c(100, 100),
              c(150, 120), c(185, 185), c(3, 40))
for (sz in sizes) for (kind in c("ties", "continuous", "shifted")) for (rep in 1:2) {
  x <- rnorm(sz[1]); y <- rnorm(sz[2], if (kind == "shifted") 0.4 else 0)
  if (kind == "ties") { x <- round(x * 2) / 2; y <- round(y * 2) / 2 }
  res <- suppressWarnings(mks(x, y))
  ks_cases[[length(ks_cases) + 1]] <- list(x = x, y = y, p = res$p.value,
                                           d = unname(res$statistic), exact = res$exact)
}

# The limiting distribution behind the asymptotic p-value. ks.test passes
# tol = 1e-6 to the series, but below x = 1 it sums only the first term of
# the theta-function form, so its p-value is off by up to 4e-5 (1.6e-5 at
# x = 0.96). `series` is the alternating series with 200 terms, which is
# exact to rounding.
psmirnov <- getFromNamespace("psmirnov", "stats")
kolmogorov_upper <- function(x) 2 * sum((-1)^(0:199) * exp(-2 * (1:200)^2 * x^2))
limit_x <- c(seq(0.3, 1.5, by = 0.05), 0.96, 0.99, 0.999)
limit_cases <- lapply(limit_x, function(x) {
  # Sample sizes 200 and 200 give sqrt(m n / (m + n)) = 10.
  list(x = x, r = psmirnov(x / 10, sizes = c(200, 200), exact = FALSE, lower.tail = FALSE),
       series = kolmogorov_upper(x))
})

# ---- the search -----------------------------------------------------------

# GenMatch's own search, with rgenoud's random path, on two data sets, for
# twelve seeds. Population 60 and 15 generations, a hard limit and no early
# stop, so that every run has the same budget. The crate's search, with the
# same budget, is compared with the distribution of what comes back.
sel <- c(sample(which(lal$treat == 1), 60), sample(which(lal$treat == 0), 180))
lal_small <- lal[sel, ]
rownames(lal_small) <- NULL
search_sc <- list(
  lalonde_small = list(d = lal_small, cov = lal_cov, exact = "married"),
  sim = scenarios[["sim"]]
)
search_ref <- list()
search_budget <- list(pop = 60, generations = 15)
for (nm in names(search_sc)) {
  sc <- search_sc[[nm]]
  run <- list(order = "largest", ratio = 1, replace = FALSE, exact = FALSE, anti = FALSE)
  inp <- genetic_inputs(sc, run)
  start_all_ones <- suppressWarnings(GenMatch(
    Tr = inp$Tr, X = inp$X, BalanceMatrix = inp$bal, M = 1, replace = FALSE, ties = FALSE,
    CommonSupport = FALSE, pop.size = 1, max.generations = 0, wait.generations = 0,
    Domains = pinned(rep(1, ncol(inp$X))), starting.values = rep(1, ncol(inp$X)),
    print.level = 0, distance.tolerance = 0
  ))$value
  values <- lapply(1:12, function(seed) {
    set.seed(1000 + seed)
    g <- suppressWarnings(GenMatch(
      Tr = inp$Tr, X = inp$X, BalanceMatrix = inp$bal, M = 1, replace = FALSE, ties = FALSE,
      CommonSupport = FALSE, pop.size = search_budget$pop,
      max.generations = search_budget$generations, wait.generations = search_budget$generations,
      hard.generation.limit = TRUE, print.level = 0, distance.tolerance = 0
    ))
    as.numeric(g$value)
  })
  search_ref[[nm]] <- list(scenario = nm, ones = as.numeric(start_all_ones), values = values)
}
scenarios <- c(scenarios, search_sc[setdiff(names(search_sc), names(scenarios))])

# ---- output ---------------------------------------------------------------

scen_out <- lapply(names(scenarios), function(nm) {
  sc <- scenarios[[nm]]
  d <- sc$d
  list(name = nm, cov = sc$cov, exact = sc$exact, treat = d$treat, ps = d$ps, g = d$g,
       columns = lapply(setNames(sc$cov, sc$cov), function(v) d[[v]]))
})
for (i in seq_along(out_runs)) {
  out_runs[[i]]$cal_cov <- if (is.null(out_runs[[i]]$cal_cov)) NULL else
    list(name = names(out_runs[[i]]$cal_cov), value = unname(unlist(out_runs[[i]]$cal_cov)))
}
result <- list(
  versions = list(r = R.version.string, matchit = as.character(packageVersion("MatchIt")),
                  matching = as.character(packageVersion("Matching")),
                  rgenoud = as.character(packageVersion("rgenoud")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  scenarios = scen_out, runs = out_runs, search = unname(search_ref),
  budget = search_budget, boot = list(run = r1$weights, sorted = boot),
  ks = ks_cases, limit = limit_cases
)
write_json(result, file.path(here, "matchit_genetic.json"), digits = NA, auto_unbox = TRUE,
           null = "null", na = "null")
cat("runs:", length(out_runs), "ks:", length(ks_cases), "\n")
