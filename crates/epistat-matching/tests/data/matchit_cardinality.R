# Reference cardinality and profile matching from MatchIt's
# matchit(method = "cardinality") with the HiGHS solver.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/matchit_cardinality.R
#
# Each case holds the treatment, the model matrix that MatchIt balances
# (factors as indicator columns), the tolerances as one value per column, the
# options, and what MatchIt returns: the weights, HiGHS's status and objective
# for each exact stratum, and the number of units selected. A case has an
# "optimal" flag that is true only when HiGHS reports the optimum in every
# stratum, and only those cases are held to the optimum. For the cases with
# mahvars the file also holds the coordinates that MatchIt pairs on (its
# transformed mahvars, the Euclidean distance of which is the Mahalanobis
# distance), the pairs it makes, and the total distance of its pairs.
#
# Data: lalonde, and a simulated cohort of 1000 units with six covariates.
suppressMessages({
  library(MatchIt)
  library(jsonlite)
})

set.seed(20261121)
data(lalonde)
lalonde$treat <- as.integer(lalonde$treat)

n <- 1000
sim <- data.frame(
  x1 = rnorm(n), x2 = rnorm(n, 1, 2), x3 = rexp(n),
  b1 = rbinom(n, 1, 0.4), b2 = rbinom(n, 1, 0.25),
  g = factor(sample(c("a", "b", "c", "d"), n, TRUE, c(.4, .3, .2, .1)))
)
lp <- with(sim, -1.2 + 0.5 * x1 + 0.2 * x2 + 0.3 * x3 + 0.6 * b1 - 0.4 * b2 +
             0.5 * (g == "a") - 0.3 * (g == "d"))
sim$treat <- rbinom(n, 1, plogis(lp))
sim$sw <- runif(n, 0.5, 2)
sim$stratum <- sample(1:3, n, TRUE)

cases <- list()

datasets <- list()

add_case <- function(name, data, formula, ..., mahvars = NULL, tols = .05,
                     std.tols = TRUE, s.weights = NULL, exact = NULL,
                     estimand = "ATT", ratio = 1) {
  args <- list(formula, data = data, method = "cardinality",
               estimand = estimand, ratio = ratio, tols = tols,
               std.tols = std.tols, include.obj = TRUE, time = 120, ...)
  if (!is.null(mahvars)) args$mahvars <- mahvars
  if (!is.null(s.weights)) args$s.weights <- s.weights
  if (!is.null(exact)) args$exact <- exact
  m <- do.call(matchit, args)
  X <- MatchIt:::get_covs_matrix(formula, data)
  dname <- if (identical(data, lalonde)) "lalonde" else "sim"
  if (is.null(datasets[[dname]])) {
    datasets[[dname]] <<- list(
      treat = as.integer(data$treat),
      x = lapply(seq_len(ncol(X)), function(j) as.numeric(X[, j])),
      colnames = colnames(X))
  }
  assign <- MatchIt:::get_assign(X)
  expand <- function(v) if (length(v) == 1) rep(v, ncol(X)) else if (length(v) == max(assign)) v[assign] else v
  objs <- if (!is.null(exact)) m$obj else list(m$obj)
  statuses <- vapply(objs, function(o) as.integer(o$status), 1L)
  keys <- if (!is.null(exact)) {
    ex <- MatchIt:::exactify(model.frame(exact, data = data), nam = rownames(data), sep = ", ", include_vars = TRUE)
    match(names(m$obj), levels(ex))
  } else 0L
  case <- list(
    name = name,
    estimand = estimand,
    ratio = if (is.na(ratio)) NULL else ratio,
    dataset = dname,
    tols = expand(tols),
    std_tols = expand(std.tols),
    s_weights = if (is.null(s.weights)) NULL else as.numeric(s.weights),
    exact = if (is.null(exact)) NULL else as.integer(ex),
    weights = as.numeric(m$weights),
    n_selected = sum(m$weights > 0),
    status = statuses,
    optimal = all(statuses == 7L),
    objective = vapply(objs, function(o) as.numeric(o$objective_value), 1)
  )
  if (!is.null(mahvars)) {
    Z <- MatchIt:::transform_covariates(mahvars, data = data, method = "mahalanobis",
                                        s.weights = s.weights, treat = data$treat,
                                        discarded = rep(FALSE, nrow(data)))
    pair <- as.integer(m$subclass)
    # Total distance over the treated and control units of each pair.
    pair_total <- 0
    for (g in unique(pair[!is.na(pair)])) {
      idx <- which(pair == g)
      tr <- idx[data$treat[idx] == 1]
      co <- idx[data$treat[idx] == 0]
      pair_total <- pair_total + sum(sqrt(rowSums((Z[rep(tr, each = length(co)), , drop = FALSE] -
                                                     Z[rep(co, times = length(tr)), , drop = FALSE])^2)))
    }
    W <- MatchIt:::get_covs_matrix_for_dist(mahvars, data)
    case$pair_raw <- lapply(seq_len(ncol(W)), function(j) as.numeric(W[, j]))
    case$pair_z <- lapply(seq_len(ncol(Z)), function(j) as.numeric(Z[, j]))
    case$pair <- pair
    case$pair_total <- pair_total
  }
  cases[[length(cases) + 1]] <<- case
  invisible(NULL)
}

lf <- treat ~ age + educ + race + married + nodegree + re74 + re75
# One value per model term, in the order of the formula.
mixed_tols <- c(.1, .1, .1, .1, .1, 1000, 1000)
mixed_std <- c(TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE)

add_case("lalonde_att_1", lalonde, lf, tols = .1)
add_case("lalonde_att_1_default", lalonde, lf)
add_case("lalonde_att_2", lalonde, lf, tols = .15, ratio = 2)
add_case("lalonde_att_3", lalonde, lf, tols = .2, ratio = 3)
add_case("lalonde_atc_1", lalonde, lf, tols = .1, estimand = "ATC")
add_case("lalonde_att_profile", lalonde, lf, tols = .1, ratio = NA)
add_case("lalonde_att_profile_tight", lalonde, lf, tols = .05, ratio = NA)
add_case("lalonde_atc_profile", lalonde, lf, tols = .1, estimand = "ATC", ratio = NA)
add_case("lalonde_ate_profile", lalonde, lf, tols = .1, estimand = "ATE", ratio = NA)
add_case("lalonde_ate_1", lalonde, lf, tols = .15, estimand = "ATE", ratio = 1)
add_case("lalonde_ate_2", lalonde, lf, tols = .2, estimand = "ATE", ratio = 2)
add_case("lalonde_att_raw_tols", lalonde, lf, tols = mixed_tols, std.tols = mixed_std)
add_case("lalonde_att_exact", lalonde, lf, tols = .15, exact = ~married)
add_case("lalonde_ate_profile_exact", lalonde, lf, tols = .2, estimand = "ATE", ratio = NA,
         exact = ~nodegree)
add_case("lalonde_att_pairs", lalonde, lf, tols = .15, mahvars = ~age + educ + re74)
add_case("lalonde_ate_pairs", lalonde, lf, tols = .2, estimand = "ATE", ratio = 1,
         mahvars = ~age + educ + re74)
sw <- runif(nrow(lalonde), 0.5, 2)
add_case("lalonde_att_profile_sw", lalonde, lf, tols = .15, ratio = NA, s.weights = sw)
add_case("lalonde_ate_profile_sw", lalonde, lf, tols = .2, estimand = "ATE", ratio = NA,
         s.weights = sw)

sf <- treat ~ x1 + x2 + x3 + b1 + b2 + g
add_case("sim_att_1", sim, sf, tols = .05)
add_case("sim_att_2", sim, sf, tols = .1, ratio = 2)
add_case("sim_att_profile", sim, sf, tols = .05, ratio = NA)
add_case("sim_atc_1", sim, sf, tols = .05, estimand = "ATC")
add_case("sim_ate_profile", sim, sf, tols = .05, estimand = "ATE", ratio = NA)
add_case("sim_ate_1", sim, sf, tols = .1, estimand = "ATE", ratio = 1)
add_case("sim_att_exact", sim, sf, tols = .1, exact = ~stratum)
add_case("sim_att_pairs", sim, sf, tols = .1, mahvars = ~x1 + x2 + x3)
add_case("sim_ate_profile_sw", sim, sf, tols = .1, estimand = "ATE", ratio = NA,
         s.weights = sim$sw)

out <- list(
  versions = list(r = R.version.string,
                  matchit = as.character(packageVersion("MatchIt")),
                  highs = as.character(packageVersion("highs")),
                  optmatch = as.character(packageVersion("optmatch"))),
  datasets = datasets,
  cases = cases
)
write(toJSON(out, auto_unbox = TRUE, digits = NA, null = "null", na = "null"),
      "tests/data/matchit_cardinality.json")
cat("cases:", length(cases), " optimal:", sum(vapply(cases, function(x) x$optimal, TRUE)), "\n")
