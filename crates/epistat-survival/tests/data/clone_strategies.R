# Cloning and censoring under the strategies of
# epistat_survival::weighting::clone beyond "start within a grace period":
# a window of starting that opens after time zero, a minimum duration of
# treatment, and dynamic strategies set off by a binary and by a continuous
# covariate; written out with base R from their definitions (no package
# implements them), with the weights of a logistic model of the deviation of
# each strategy on its rows at risk.
#
# Person-period data are simulated with a flare indicator f and a continuous
# covariate l that both drive starting treatment and the outcome, and that
# depend on the earlier treatment. The strategies, in order, are
#   1. start in one of the periods 2 to 4, not before, and stay on;
#   2. start in one of the periods 0 to 3 and stay on for at least 3 periods;
#   3. start within 1 period of the first flare, not before, and stay on;
#   4. start within 2 periods of l first falling below -0.3, not before, with
#      no constraint after the start;
#   5. never start.
# The file holds the input, every clone row (its input row, strategy,
# deviation, whether it is at risk, outcome), the fitted probability of
# deviating and the weight of each clone row.
# Numbers are written with 17 digits.
#
# Run from the crate directory:
#   Rscript tests/data/clone_strategies.R > tests/data/clone_strategies.json
suppressMessages(library(jsonlite))
set.seed(20261126)
n <- 800
tmax <- 9
rows <- vector("list", n)
for (i in seq_len(n)) {
  l <- rnorm(1)
  a_prev <- 0
  out <- list()
  for (p in 0:tmax) {
    l <- 0.5 * l + 0.3 * a_prev + rnorm(1, 0, 0.7)
    f <- rbinom(1, 1, plogis(-1.6 + 0.6 * l - 0.5 * a_prev))
    pa <- if (a_prev == 1) plogis(2.2 - 0.5 * l) else plogis(-2.4 + 1.2 * f - 0.6 * l + 0.1 * p)
    a <- rbinom(1, 1, pa)
    y <- rbinom(1, 1, plogis(-4 - 0.6 * a + 0.4 * l + 0.5 * f + 0.05 * p))
    out[[length(out) + 1]] <- data.frame(id = i - 1, period = p, treatment = a,
      outcome = y, f = f, l = round(l, 4))
    a_prev <- a
    if (y == 1) break
  }
  rows[[i]] <- do.call(rbind, out)
}
d <- do.call(rbind, rows)
d$row <- seq_len(nrow(d)) - 1

# type: "window" (first, last), "when" (trigger, grace), or "never"; dur is
# the number of periods treatment must go on from the start (Inf: always, 1:
# no constraint).
strategies <- list(
  list(type = "window", first = 2, last = 4, dur = Inf),
  list(type = "window", first = 0, last = 3, dur = 3),
  list(type = "when", trigger = function(r) r$f >= 1, grace = 1, dur = Inf),
  list(type = "when", trigger = function(r) r$l < -0.3, grace = 2, dur = 1),
  list(type = "never"))

clones <- list()
for (id in unique(d$id)) {
  di <- d[d$id == id, ]
  for (arm in seq_along(strategies)) {
    s <- strategies[[arm]]
    started <- NA
    trig <- NA
    for (k in seq_len(nrow(di))) {
      r <- di[k, ]
      j <- r$period
      a <- r$treatment
      if (s$type == "never") {
        risk <- TRUE; dev <- a == 1
      } else {
        if (s$type == "when" && is.na(trig) && s$trigger(r)) trig <- j
        first <- if (s$type == "when") (if (is.na(trig)) Inf else trig) else s$first
        last <- if (s$type == "when") (if (is.na(trig)) Inf else trig + s$grace) else s$last
        if (!is.na(started)) {
          risk <- (j - started) < s$dur; dev <- risk && a == 0
        } else if (j < first) {
          risk <- TRUE; dev <- a == 1
        } else if (j == last) {
          risk <- TRUE; dev <- a == 0
        } else {
          risk <- FALSE; dev <- FALSE
        }
      }
      ev <- !dev && r$outcome == 1
      clones[[length(clones) + 1]] <- data.frame(row = r$row, id = id, arm = arm - 1,
        period = j, dev = as.integer(dev), risk = as.integer(risk),
        outcome = as.integer(ev), f = r$f, l = r$l)
      if (dev) break
      if (a == 1 && is.na(started)) started <- j
      if (ev) break
    }
  }
}
cl <- do.call(rbind, clones)
cl$pd <- NA_real_
for (arm in seq_along(strategies) - 1) {
  sel <- cl$arm == arm
  at_risk <- sel & cl$risk == 1
  md <- glm(dev ~ l + f + period, data = cl[at_risk, ], family = binomial)
  cl$pd[sel] <- predict(md, newdata = cl[sel, ], type = "response")
}
cl$factor <- ifelse(cl$risk == 1, 1 / (1 - cl$pd), 1)
cl$weight <- ave(cl$factor, interaction(cl$id, cl$arm), FUN = cumprod)
out <- list(
  versions = list(R = as.character(getRversion()),
    jsonlite = as.character(packageVersion("jsonlite"))),
  input = list(id = d$id, period = d$period, treatment = d$treatment,
    outcome = d$outcome, f = d$f, l = d$l),
  clones = list(row = cl$row, arm = cl$arm, dev = cl$dev, risk = cl$risk,
    outcome = cl$outcome, pd = cl$pd, weight = cl$weight))
cat(toJSON(out, digits = I(17), auto_unbox = TRUE, na = "null"))
