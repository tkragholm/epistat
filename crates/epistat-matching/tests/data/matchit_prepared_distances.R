# Pairs of MatchIt 4.8.2's nearest neighbour matching on its covariate
# distances, for the crate to match with distances it prepares itself from the
# covariates: distance = "mahalanobis", "robust_mahalanobis", "euclidean" and
# "scaled_euclidean", with the covariates given to matchit() and no
# transformed covariate matrix passed between R and Rust.
#
# For every run the file holds, per treated unit in row order, the units
# matched to it in order of assignment, for a range of m.order, ratio,
# replacement, a caliper on a covariate and exact matching. No treated unit
# has two controls at the same distance (the script checks every distance
# matrix), so matchit() has no tie to break within a row and the crate must
# give its pairs. A few pairs of equal distances in different rows are not a
# tie unless both are the nearest pair of their rows.
#
# Data: a sample of lalonde and a simulated set, each with a factor and binary
# covariates. The continuous covariates are made unique by perturbations far
# below any difference that matters, so that no two distances are equal.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/matchit_prepared_distances.R
suppressMessages({
  library(MatchIt)
  library(jsonlite)
})
args <- commandArgs(FALSE)
here <- dirname(sub("--file=", "", args[grep("--file=", args)]))

data(lalonde)
set.seed(20261125)
tr <- sample(which(lalonde$treat == 1), 50)
co <- sample(which(lalonde$treat == 0), 150)
lal <- lalonde[sample(c(tr, co)), c("treat", "age", "educ", "race", "married", "nodegree", "re74", "re75")]
rownames(lal) <- NULL
lal$race <- as.character(lal$race)
lal$re74 <- lal$re74 + 1e-3 * seq_len(nrow(lal))
lal$re75 <- lal$re75 + 1e-3 * rev(seq_len(nrow(lal))) * 1.37
lal$age <- lal$age + 1e-4 * seq_len(nrow(lal)) * 0.71
lal$s <- paste0(lal$race, lal$married)

n <- 180
sim <- data.frame(treat = rbinom(n, 1, 0.3))
sim$z1 <- rnorm(n) + 0.5 * sim$treat
sim$z2 <- rnorm(n)
sim$b <- rbinom(n, 1, 0.4)
sim$grp <- sample(c("a", "b", "c"), n, replace = TRUE)
sim$age <- round(runif(n, 20, 40)) + 1e-4 * seq_len(n)
sim$s <- sample(c("a", "b"), n, replace = TRUE)

datasets <- list(
  lalonde = list(d = lal, f = treat ~ age + educ + race + married + nodegree + re74 + re75),
  sim = list(d = sim, f = treat ~ z1 + z2 + b + grp + age)
)

distances <- c("mahalanobis", "robust_mahalanobis", "euclidean", "scaled_euclidean")
dist_fun <- list(
  mahalanobis = mahalanobis_dist, robust_mahalanobis = robust_mahalanobis_dist,
  euclidean = euclidean_dist, scaled_euclidean = scaled_euclidean_dist
)

specs <- list()
add <- function(order, ratio, replace = FALSE, covcal = NULL, exact = FALSE, estimand = "ATT",
                covcal_std = NULL)
  specs[[length(specs) + 1]] <<- list(order = order, ratio = ratio, replace = replace,
                                      covcal = covcal, exact = exact, estimand = estimand,
                                      covcal_std = covcal_std)
for (o in c("data", "closest", "farthest")) for (r in 1:3) add(o, r)
add("data", 2, replace = TRUE)
add("data", 1, replace = TRUE)
add("data", 2, covcal = 4)
add("closest", 2, covcal = 4)
add("data", 2, covcal_std = 0.5)
add("closest", 2, covcal_std = 0.25)
add("data", 2, exact = TRUE)
add("farthest", 2, exact = TRUE)
add("data", 2, estimand = "ATC")
add("closest", 3, estimand = "ATC")

runs <- list()
data_out <- list()
for (dname in names(datasets)) {
  d <- datasets[[dname]]$d
  f <- datasets[[dname]]$f
  covs <- setdiff(all.vars(f), "treat")
  data_out[[dname]] <- list(
    treat = d$treat,
    numerics = lapply(Filter(is.numeric, d[covs]), function(x) x),
    factors = lapply(Filter(is.character, d[covs]), function(x) x),
    s = d$s, age = d$age, age_sd = sd(d$age))
  for (dist in distances) {
    D <- unname(dist_fun[[dist]](f, d))
    if (any(apply(D, 1, anyDuplicated) > 0)) stop(sprintf("equal distances in a row: %s %s", dname, dist))
    # Equal distances in different rows can matter only to closest and farthest,
    # and only where both are the nearest pair of their rows.
    equal_across <- sum(duplicated(as.vector(D)))
    for (sp in specs) {
      focal <- if (sp$estimand == "ATC") 0L else 1L
      fi <- which(d$treat == focal)
      cal <- if (!is.null(sp$covcal)) c(age = sp$covcal) else if (!is.null(sp$covcal_std)) c(age = sp$covcal_std) else NULL
      f_m <- matchit(f, data = d, method = "nearest", distance = dist,
                     m.order = sp$order, ratio = sp$ratio, replace = sp$replace,
                     estimand = sp$estimand,
                     exact = if (sp$exact) ~ s else NULL,
                     caliper = cal, std.caliper = !is.null(sp$covcal_std))
      mm <- f_m$match.matrix
      rows <- lapply(seq_along(fi), function(i) { v <- mm[i, ]; as.integer(v[!is.na(v)]) })
      runs[[length(runs) + 1]] <- list(data = dname, distance = dist, spec = sp,
                                       focal_rows = fi, matchit = rows, equal_across = equal_across)
    }
  }
}

# What matchit() does with a caliper on a covariate distance: every caliper
# must be named, so a caliper on the distance itself, in standard deviation
# units or not, is an error.
refusal <- function(...) tryCatch({ matchit(..., data = lal, method = "nearest"); "accepted" },
                                  error = function(e) conditionMessage(e))
refusals <- list(
  mahalanobis_std = refusal(treat ~ age + educ + re74, distance = "mahalanobis", caliper = 0.5, std.caliper = TRUE),
  mahalanobis_raw = refusal(treat ~ age + educ + re74, distance = "mahalanobis", caliper = 0.5, std.caliper = FALSE),
  euclidean_std = refusal(treat ~ age + educ + re74, distance = "euclidean", caliper = 0.5, std.caliper = TRUE),
  scaled_euclidean_raw = refusal(treat ~ age + educ + re74, distance = "scaled_euclidean", caliper = 0.5, std.caliper = FALSE)
)

out <- list(
  refusals = refusals,
  versions = list(r = R.version.string, matchit = as.character(packageVersion("MatchIt")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  data = data_out, runs = runs
)
write(toJSON(out, auto_unbox = TRUE, digits = I(17), null = "null", na = "null"),
      file.path(here, "matchit_prepared_distances.json"))
cat(length(runs), "runs\n")
