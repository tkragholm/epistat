# Reference effect estimates after matching, from marginaleffects.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/effects.R
#
# The script matches `lalonde` with MatchIt in six ways, as the MatchIt
# vignette "Estimating effects after matching" does, and fits the outcome
# model on the matched rows with the matching weights. For each fit it records
# the coefficients, the cluster-robust covariance from sandwich::vcovCL and the
# average effect among the units of the estimand from
# marginaleffects::avg_comparisons(), with the weights as `wts`:
#   difference   comparison = "difference"
#   ratio        comparison = "ratio", the ratio of the weighted mean
#                predictions (binary outcome only)
#   lnratio      comparison = "lnratio"
#   lnor         comparison = "lnor", the log of the odds ratio of the
#                weighted mean predictions
# The logistic fits run to a relative deviance change of 1e-14 instead of R's
# default 1e-8, so that the comparison does not measure R's stopping rule.
# The Rust test reads the matched rows, so it does not depend on the matching
# code, only on the effect code.
#
# Matches (propensity score from a logistic model of age, educ, race,
# married, nodegree, re74 and re75):
#   nearest            1:1 without replacement, ATT, rows of match_data()
#   full               full matching, ATT and ATE, rows of match_data()
#   subclass           6 subclasses, ATT and ATC, rows of match_data()
#   replace_ratio2     2:1 with replacement, ATT, rows of get_matches()
# Outcomes: `re78` (linear model) and `re78 > 0` (logistic model).
# Models: treatment only; treatment with age and educ; and with the
# interactions of treatment with age and educ.
# Variance: the clusters of the vignette, `~subclass` without replacement and
# `~subclass + id` with it. With replacement `~id` is recorded as well.

suppressMessages({
  library(MatchIt)
  library(marginaleffects)
  library(sandwich)
  library(jsonlite)
})

data(lalonde)
ps_formula <- treat ~ age + educ + race + married + nodegree + re74 + re75
unit_names <- rownames(lalonde)

cases <- list()

add_case <- function(name, estimand, rows, clustering) {
  cases[[length(cases) + 1]] <<- list(name = name, estimand = estimand,
                                      rows = rows, clustering = clustering)
}

from_match_data <- function(m) {
  md <- match_data(m, drop.unmatched = TRUE)
  list(unit = match(rownames(md), unit_names), weight = md$weights,
       subclass = as.integer(as.character(md$subclass)), data = md)
}

from_get_matches <- function(m) {
  gm <- get_matches(m)
  list(unit = match(gm$id, unit_names), weight = gm$weights,
       subclass = as.integer(as.character(gm$subclass)), data = gm)
}

m_nearest <- matchit(ps_formula, data = lalonde, method = "nearest")
m_full_att <- matchit(ps_formula, data = lalonde, method = "full")
m_full_ate <- matchit(ps_formula, data = lalonde, method = "full",
                      estimand = "ATE")
m_sub_att <- matchit(ps_formula, data = lalonde, method = "subclass",
                     subclass = 6)
m_sub_atc <- matchit(ps_formula, data = lalonde, method = "subclass",
                     subclass = 6, estimand = "ATC")
m_replace <- matchit(ps_formula, data = lalonde, method = "nearest",
                     replace = TRUE, ratio = 2)

specs <- list(
  list("nearest", "ATT", from_match_data(m_nearest), list("pair")),
  list("full_att", "ATT", from_match_data(m_full_att), list("pair")),
  list("full_ate", "ATE", from_match_data(m_full_ate), list("pair")),
  list("subclass_att", "ATT", from_match_data(m_sub_att), list("pair")),
  list("subclass_atc", "ATC", from_match_data(m_sub_atc), list("pair")),
  list("replace_ratio2", "ATT", from_get_matches(m_replace),
       list("pair_unit", "unit")))

models <- list(
  list(covariates = character(0), interactions = FALSE),
  list(covariates = c("age", "educ"), interactions = FALSE),
  list(covariates = c("age", "educ"), interactions = TRUE))

cluster_of <- function(data, clustering) {
  switch(clustering,
         pair = data$subclass,
         unit = data$id,
         pair_unit = data.frame(subclass = data$subclass, id = data$id))
}

newdata_of <- function(data, estimand) {
  switch(estimand,
         ATT = subset(data, treat == 1),
         ATC = subset(data, treat == 0),
         ATE = data)
}

effect <- function(fit, data, estimand, clustering, comparison) {
  cl <- cluster_of(data, clustering)
  vc <- if (is.data.frame(cl)) ~subclass + id else if (clustering == "unit") ~id else ~subclass
  a <- avg_comparisons(fit, variables = "treat", vcov = vc,
                       newdata = newdata_of(data, estimand), wts = "weights",
                       comparison = comparison)
  list(estimate = a$estimate, se = a$std.error)
}

results <- list()
for (spec in specs) {
  name <- spec[[1]]; estimand <- spec[[2]]; mm <- spec[[3]]
  data <- mm$data
  # Both match tables carry the unit under a different name.
  data$id <- if (is.null(data$id)) rownames(data) else data$id
  data$subclass <- as.integer(as.character(data$subclass))
  data$y <- lalonde$re78[mm$unit]
  data$b <- as.numeric(lalonde$re78[mm$unit] > 0)
  data$age <- lalonde$age[mm$unit]
  data$educ <- lalonde$educ[mm$unit]
  for (clustering in spec[[4]]) {
    for (outcome in c("continuous", "binary")) {
      for (model in models) {
        rhs <- if (length(model$covariates) == 0) "treat" else if (model$interactions)
          paste0("treat * (", paste(model$covariates, collapse = " + "), ")") else
          paste(c("treat", model$covariates), collapse = " + ")
        suppressWarnings(fit <- if (outcome == "continuous")
          lm(as.formula(paste("y ~", rhs)), data = data, weights = weights)
        else glm(as.formula(paste("b ~", rhs)), data = data, weights = weights,
                 family = binomial,
                 control = glm.control(epsilon = 1e-14, maxit = 100)))
        cl <- cluster_of(data, clustering)
        v <- vcovCL(fit, cluster = cl)
        res <- list(
          case = name, clustering = clustering, outcome = outcome,
          covariates = model$covariates, interactions = model$interactions,
          coefficients = unname(coef(fit)),
          vcov = unname(split(v, row(v))),
          difference = effect(fit, data, estimand, clustering, "difference"))
        if (outcome == "binary") {
          res$ratio <- effect(fit, data, estimand, clustering, "ratio")
          res$lnratio <- effect(fit, data, estimand, clustering, "lnratio")
          res$lnor <- effect(fit, data, estimand, clustering, "lnor")
        }
        results[[length(results) + 1]] <- res
      }
    }
  }
  add_case(name, estimand, mm$unit, spec[[4]])
  cases[[length(cases)]]$weight <- mm$weight
  cases[[length(cases)]]$subclass <- mm$subclass
}

out <- list(
  versions = list(r = R.version.string,
                  matchit = as.character(packageVersion("MatchIt")),
                  marginaleffects = as.character(packageVersion("marginaleffects")),
                  sandwich = as.character(packageVersion("sandwich")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  lalonde = list(treat = lalonde$treat, re78 = lalonde$re78,
                 age = lalonde$age, educ = lalonde$educ),
  cases = cases, results = results)
write_json(out, "tests/data/effects.json", digits = I(17), auto_unbox = TRUE,
           na = "null")
