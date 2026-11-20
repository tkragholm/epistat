# survival::concordance on a numeric predictor: the lung data with a Cox linear
# predictor (reverse = TRUE) and with a score that ties (ph.ecog), tied event
# times, case weights, strata, a cluster, delayed entry (counting-process
# Surv(start, stop, event)), every timewt the function offers (S/G and n/G2 need
# right-censored data), ymin and ymax, timefix on and off, a stratum with a single
# event, twelve strata (survival then makes one stratum over disjoint time
# ranges), and a random data set of 3000 rows with fractional times.
# For each case: the data set it ran on, the call, the counts, the concordance,
# the infinitesimal jackknife variance, the variance of the Cox score statistic
# (cvar) and the dfbeta of each row. Seventeen digits are written so that every
# double reads back exactly. The versions of R, survival and jsonlite are in it.
# Run: Rscript concordance.R > concordance.json
suppressMessages({
  library(jsonlite)
  library(survival)
})
set.seed(20261120)

lung2 <- lung[complete.cases(lung[, c("time", "status", "age", "sex", "ph.ecog", "wt.loss")]), ]
n1 <- nrow(lung2)
lp <- predict(coxph(Surv(time, status) ~ age + sex + ph.ecog + wt.loss, lung2), type = "lp")

datasets <- list()
datasets$lung <- list(
  stop = lung2$time, status = lung2$status - 1L,
  lp = unname(lp), ecog = lung2$ph.ecog, age = lung2$age,
  weight = round(runif(n1, 0.5, 3), 2),
  stratum = as.integer(lung2$sex), cluster = lung2$inst,
  start = pmin(floor(lung2$time * runif(n1, 0, 0.8)), lung2$time - 1)
)
datasets$lung$cluster[is.na(datasets$lung$cluster)] <- 99L
# Twelve strata, a few events in each, and one stratum with one event.
st12 <- rep(1:12, length.out = n1)
datasets$lung$stratum12 <- st12
one <- which(st12 == 12 & datasets$lung$status == 1L)
datasets$lung$status12 <- datasets$lung$status
datasets$lung$status12[one[-1]] <- 0L
# Random data, fractional times, integer-free weights, a score with ties.
n2 <- 3000
x2 <- round(rnorm(n2), 1)
t2 <- round(rexp(n2, 0.1 * exp(0.3 * x2)), 3) + 0.001
c2 <- round(runif(n2, 0, 30), 3) + 0.001
datasets$random <- list(
  stop = pmin(t2, c2), status = as.integer(t2 <= c2), x = x2,
  weight = round(runif(n2, 0.2, 4), 3),
  stratum = sample(1:4, n2, replace = TRUE),
  cluster = sample(1:300, n2, replace = TRUE),
  start = round(runif(n2, 0, 0.5) * pmin(t2, c2), 3)
)
datasets$random$start <- pmin(datasets$random$start, datasets$random$stop - 0.001)

cases <- list()
run <- function(name, data, x, timewt = "n", reverse = FALSE, weights = FALSE,
                strata = NULL, cluster = FALSE, delayed = FALSE, ymin = NULL,
                ymax = NULL, timefix = TRUE, status = "status", keep = TRUE) {
  d <- datasets[[data]]
  df <- data.frame(stop = d$stop, status = d[[status]], x = d[[x]])
  if (delayed) df$start <- d$start
  if (weights) df$w <- d$weight
  if (!is.null(strata)) df$s <- d[[strata]]
  if (cluster) df$cl <- d$cluster
  lhs <- if (delayed) "Surv(start, stop, status)" else "Surv(stop, status)"
  rhs <- if (is.null(strata)) "x" else "x + strata(s)"
  args <- list(as.formula(paste(lhs, "~", rhs)), data = df, timewt = timewt,
               reverse = reverse, influence = 1, timefix = timefix)
  if (weights) args$weights <- df$w
  if (cluster) args$cluster <- df$cl
  if (!is.null(ymin)) args$ymin <- ymin
  if (!is.null(ymax)) args$ymax <- ymax
  if (!is.null(strata)) args$keepstrata <- keep
  fit <- do.call(concordance, args)
  count <- if (is.matrix(fit$count)) colSums(fit$count) else fit$count
  list(name = name, data = data, x = x, timewt = timewt, reverse = reverse,
       weights = weights, strata = strata, cluster = cluster, delayed = delayed,
       ymin = ymin, ymax = ymax, timefix = timefix, status = status,
       count = unname(count[c("concordant", "discordant", "tied.x", "tied.y", "tied.xy")]),
       concordance = unname(fit$concordance), var = unname(fit$var),
       cvar = unname(fit$cvar), dfbeta = unname(fit$dfbeta), n = fit$n)
}

cases[[length(cases) + 1]] <- run("lung cox lp, reverse", "lung", "lp", reverse = TRUE)
cases[[length(cases) + 1]] <- run("lung ecog, ties in x", "lung", "ecog")
cases[[length(cases) + 1]] <- run("lung age, forward", "lung", "age")
cases[[length(cases) + 1]] <- run("lung weights", "lung", "ecog", weights = TRUE)
cases[[length(cases) + 1]] <- run("lung strata", "lung", "lp", reverse = TRUE, strata = "stratum")
cases[[length(cases) + 1]] <- run("lung strata and weights", "lung", "age", weights = TRUE, strata = "stratum")
cases[[length(cases) + 1]] <- run("lung cluster", "lung", "lp", reverse = TRUE, cluster = TRUE)
cases[[length(cases) + 1]] <- run("lung delayed entry", "lung", "ecog", delayed = TRUE)
cases[[length(cases) + 1]] <- run("lung delayed entry, weights, strata", "lung", "age", delayed = TRUE, weights = TRUE, strata = "stratum")
for (tw in c("S", "S/G", "n/G2", "I")) {
  cases[[length(cases) + 1]] <- run(paste("lung timewt", tw), "lung", "lp", timewt = tw, reverse = TRUE)
  cases[[length(cases) + 1]] <- run(paste("lung weights timewt", tw), "lung", "ecog", timewt = tw, weights = TRUE)
  cases[[length(cases) + 1]] <- run(paste("lung strata timewt", tw), "lung", "age", timewt = tw, strata = "stratum")
}
for (tw in c("S", "I")) {
  cases[[length(cases) + 1]] <- run(paste("lung delayed entry timewt", tw), "lung", "ecog", timewt = tw, delayed = TRUE)
  cases[[length(cases) + 1]] <- run(paste("lung delayed entry strata timewt", tw), "lung", "age", timewt = tw, delayed = TRUE, strata = "stratum")
}
cases[[length(cases) + 1]] <- run("lung ymax", "lung", "lp", reverse = TRUE, ymax = 400)
cases[[length(cases) + 1]] <- run("lung ymax timewt S", "lung", "lp", timewt = "S", reverse = TRUE, ymax = 400)
cases[[length(cases) + 1]] <- run("lung ymin", "lung", "lp", reverse = TRUE, ymin = 150)
cases[[length(cases) + 1]] <- run("lung ymin delayed entry", "lung", "ecog", delayed = TRUE, ymin = 150)
cases[[length(cases) + 1]] <- run("lung timefix off", "lung", "ecog", timefix = FALSE)
cases[[length(cases) + 1]] <- run("twelve strata", "lung", "age", strata = "stratum12")
cases[[length(cases) + 1]] <- run("twelve strata timewt I", "lung", "age", timewt = "I", strata = "stratum12")
# survival's keepstrata default drops the per-stratum counts above ten strata
# and runs the stratified data as one over disjoint time ranges.
cases[[length(cases) + 1]] <- run("twelve strata, one stratum over disjoint times", "lung", "age", strata = "stratum12", keep = FALSE)
cases[[length(cases) + 1]] <- run("twelve strata, disjoint times, weights, timewt I", "lung", "ecog", timewt = "I", weights = TRUE, strata = "stratum12", keep = FALSE)
cases[[length(cases) + 1]] <- run("twelve strata, one with a single event, timewt S", "lung", "age", timewt = "S", strata = "stratum12", status = "status12")
cases[[length(cases) + 1]] <- run("random", "random", "x")
cases[[length(cases) + 1]] <- run("random reverse weights strata cluster", "random", "x", reverse = TRUE, weights = TRUE, strata = "stratum", cluster = TRUE)
for (tw in c("S", "S/G", "n/G2", "I")) {
  cases[[length(cases) + 1]] <- run(paste("random weights timewt", tw), "random", "x", timewt = tw, weights = TRUE)
}
cases[[length(cases) + 1]] <- run("random delayed entry weights strata", "random", "x", delayed = TRUE, weights = TRUE, strata = "stratum")
cases[[length(cases) + 1]] <- run("random delayed entry timewt S weights", "random", "x", delayed = TRUE, weights = TRUE, timewt = "S")

cat(toJSON(list(
  versions = list(r = R.version.string, survival = as.character(packageVersion("survival")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  datasets = datasets, cases = cases
), digits = I(17), auto_unbox = TRUE, null = "null", na = "null"))
