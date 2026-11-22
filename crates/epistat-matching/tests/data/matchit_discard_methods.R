# Common-support trimming with the methods subclass, full, optimal, quick and
# genetic, from MatchIt 4.8.2: matchit(method = ..., distance = "glm",
# discard = policy, reestimate = TRUE or FALSE) on R's lalonde.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/matchit_discard_methods.R
#
# The continuous covariates of lalonde are jittered by a small amount, so that
# no two units share a score and no two distances tie. For every run the file
# holds the units MatchIt discarded (row numbers), the distance MatchIt matched
# on and the result of the method:
#   subclass   the subclass of every unit, the weights and the cutpoints q.cut.
#              Without reestimate these are matchit()'s own, whose cutpoints
#              are quantiles over all units of the reference group, discarded
#              ones included; the crate does the same. With reestimate the
#              discarded units keep their first-fit scores while the kept units
#              have second-fit scores, and matchit() takes its quantiles over
#              that mixture of two fits. That is a defect of matchit(): the
#              mixture is the distribution of neither fit and it moves the
#              cutpoints and the subclasses with the number of units discarded.
#              With reestimate the reference is matchit() run on the kept units
#              alone with their second-fit distance (`subclass`, `weights`,
#              `q_cut`), which is what the crate computes, and
#              `q_cut_with_discard` holds the cutpoints of the call with the
#              discard, so that the test asserts both.
#   full       the subclass of every unit and the weights; `unique` is TRUE when
#              four perturbations of the distance by 1e-5 standard deviations
#              give the same partition, so that the optimum is a single one
#   optimal    the controls of each treated unit and `unique` as for full
#   quick      the subclass of every unit and the weights
#   genetic    the controls of each treated unit, with the weights of the
#              search pinned to 1 by a range of relative width 2e-9, so that
#              R's random search does not enter
# `discarded` and `distance` come from matchit() itself.
suppressMessages({
  library(MatchIt)
  library(jsonlite)
})
args <- commandArgs(FALSE)
here <- dirname(sub("--file=", "", args[grep("--file=", args)]))

data(lalonde)
set.seed(20261125)
lal <- data.frame(
  treat = lalonde$treat, age = lalonde$age, educ = lalonde$educ,
  black = as.numeric(lalonde$race == "black"),
  hispan = as.numeric(lalonde$race == "hispan"),
  married = lalonde$married, nodegree = lalonde$nodegree,
  re74 = lalonde$re74, re75 = lalonde$re75)
for (v in c("age", "educ", "re74", "re75")) {
  lal[[v]] <- lal[[v]] + rnorm(nrow(lal), 0, 0.05 * sd(lal[[v]]))
}
covs <- setdiff(names(lal), "treat")
f <- reformulate(covs, "treat")

partition_of <- function(subclass) {
  keep <- which(!is.na(subclass))
  groups <- split(keep, subclass[keep])
  sort(vapply(groups, function(g) paste(sort(g), collapse = ","), ""))
}
pairs_of <- function(m) {
  mm <- m$match.matrix
  rows <- lapply(seq_len(nrow(mm)), function(i) {
    v <- as.integer(mm[i, ])
    v[!is.na(v)]
  })
  list(treated = as.integer(rownames(mm)), controls = rows)
}
signature_of <- function(method, m) {
  if (method == "optimal") {
    p <- pairs_of(m)
    paste(vapply(seq_along(p$treated), function(i)
      paste(p$treated[i], paste(sort(p$controls[[i]]), collapse = "-")), ""),
      collapse = ";")
  } else {
    paste(partition_of(m$subclass), collapse = ";")
  }
}

# Whether the optimum is a single one: the distance of the run, perturbed, with
# the units MatchIt discarded given as a logical vector, gives the same match.
is_unique <- function(method, m, extra) {
  sig <- signature_of(method, m)
  d <- unname(m$distance)
  sc <- 1e-5 * sd(d)
  all(vapply(1:4, function(k) {
    pm <- tryCatch(suppressWarnings(do.call(matchit, c(list(
      formula = f, data = lal, method = method,
      distance = d + rnorm(length(d), 0, sc), discard = m$discarded),
      extra))), error = function(e) NULL)
    !is.null(pm) && identical(signature_of(method, pm), sig)
  }, logical(1)))
}

run_one <- function(method, policy, reestimate, extra = list(), link = "logit",
                    estimand = "ATT") {
  m <- suppressWarnings(do.call(matchit, c(list(
    formula = f, data = lal, method = method, distance = "glm", link = link,
    discard = policy, reestimate = reestimate, estimand = estimand), extra)))
  run <- list(method = method, policy = policy, reestimate = reestimate,
              estimand = estimand, link = link,
              extra = extra, discarded = which(m$discarded),
              distance = unname(m$distance))
  if (method %in% c("full", "quick")) {
    run$subclass <- ifelse(is.na(m$subclass), NA, as.integer(m$subclass))
    run$weights <- unname(m$weights)
  }
  if (method == "subclass") {
    run$q_cut_with_discard <- unname(m$q.cut)
    run$subclass <- ifelse(is.na(m$subclass), NA, as.integer(m$subclass))
    run$weights <- unname(m$weights)
    run$q_cut <- unname(m$q.cut)
  }
  if (method == "subclass" && reestimate) {
    kept <- which(!m$discarded)
    mk <- suppressWarnings(do.call(matchit, c(list(
      formula = f, data = lal[kept, ], method = "subclass",
      distance = unname(m$distance)[kept], estimand = estimand), extra)))
    sub <- rep(NA_integer_, nrow(lal))
    sub[kept] <- as.integer(mk$subclass)
    w <- rep(0, nrow(lal))
    w[kept] <- unname(mk$weights)
    run$subclass <- ifelse(is.na(sub), NA, sub)
    run$weights <- w
    run$q_cut <- unname(mk$q.cut)
  }
  if (method %in% c("optimal", "genetic")) run$pairs <- pairs_of(m)
  if (method %in% c("full", "optimal")) {
    run$unique <- is_unique(method, m, extra)
  }
  run
}

# optmatch stops its solver at a relative tolerance of 0.001 by default, which
# leaves the total distance up to about 0.5 per cent above the optimum on this
# data. The runs set it to 1e-9, as the other full and optimal references do.
TOL <- 1e-9

runs <- list()
add <- function(...) runs[[length(runs) + 1]] <<- run_one(...)
for (policy in c("treated", "control", "both")) {
  for (reest in c(FALSE, TRUE)) {
    add("subclass", policy, reest, list(subclass = 6))
    add("subclass", policy, reest, list(subclass = 4), estimand = "ATE")
    add("full", policy, reest, list(tol = TOL))
    add("optimal", policy, reest, list(ratio = 1, tol = TOL))
    add("optimal", policy, reest, list(ratio = 2, tol = TOL))
    add("quick", policy, reest, link = "linear.logit")
    add("genetic", policy, reest,
        list(pop.size = 6, max.generations = 1, wait.generations = 1,
             Domains = cbind(rep(1 - 1e-9, length(covs) + 1),
                             rep(1 + 1e-9, length(covs) + 1)),
             starting.values = rep(1, length(covs) + 1)))
  }
}
out <- list(
  versions = list(r = R.version.string,
                  matchit = as.character(packageVersion("MatchIt")),
                  optmatch = as.character(packageVersion("optmatch")),
                  quickmatch = as.character(packageVersion("quickmatch")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  data = list(treat = lal$treat, names = covs,
              covariates = lapply(covs, function(v) lal[[v]])),
  runs = runs)
write(toJSON(out, auto_unbox = TRUE, digits = I(17), null = "null", na = "null"),
      file.path(here, "matchit_discard_methods.json"))
cat(length(runs), "runs\n")
