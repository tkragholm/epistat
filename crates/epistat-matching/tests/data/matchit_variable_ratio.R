# Pairs and weights of MatchIt 4.8.2's variable ratio nearest neighbour
# matching (ratio, min.controls, max.controls), which is Ming and
# Rosenbaum's (2000) extremal matching.
#
# For every run the file holds, per focal unit in row order, the units matched
# to it in order of assignment:
#   matchit   what matchit() returned (match.matrix), and its weights
#   model     the pairs of a short loop written here from the documented
#             behaviour: the numbers of controls from the paper's rule, the
#             anchors ranked by score, candidates ranked by distance with ties
#             broken by row order, and rounds in which an anchor takes a
#             control until it has its number.
#   counts    the numbers of controls per focal unit the loop used
#   ties      how many times the loop chose between candidates at the same
#             distance. Where this is 0, the loop and matchit() must agree
#             unless the numbers of controls differ; the script stops if not.
#   matchit_counts_ok  whether matchit()'s own numbers of controls (the sizes
#             of its rows, with replacement and no caliper) equal the loop's.
# MatchIt's numbers of controls differ from the paper's where the unit that
# gets neither the minimum nor the maximum would get fewer than min.controls
# (kmax is taken from n1 - 1 where the paper takes n1). The runs with
# defect = true are those: where matchit() runs with replacement and without
# caliper or exact, the script asserts that it gives a unit fewer than
# min.controls. The loop keeps to min.controls, and so does the crate.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/matchit_variable_ratio.R
suppressMessages({
  library(MatchIt)
  library(jsonlite)
})
args <- commandArgs(FALSE)
here <- dirname(sub("--file=", "", args[grep("--file=", args)]))

data(lalonde, package = "MatchIt")
set.seed(20261123)
tr <- sample(which(lalonde$treat == 1), 60)
co <- sample(which(lalonde$treat == 0), 180)
lal <- lalonde[sample(c(tr, co)), ]
lal$race <- as.character(lal$race)
rownames(lal) <- NULL
lal$black <- as.numeric(lal$race == "black")
lal$hispan <- as.numeric(lal$race == "hispan")
lal$s <- paste0(lal$race, lal$married)
fit <- glm(treat ~ age + educ + black + hispan + married + nodegree + re74 + re75,
           data = lal, family = binomial)
lal$ps <- unname(predict(fit, type = "response"))
lal$lin <- unname(predict(fit, type = "link"))
rowno <- seq_len(nrow(lal))
lal$ps_u <- lal$ps + 1e-10 * rowno

n_sim <- 200
sim <- data.frame(treat = rbinom(n_sim, 1, 0.3))
sim$z1 <- rnorm(n_sim) + 0.5 * sim$treat
sim$z2 <- rnorm(n_sim)
sim$age <- round(runif(n_sim, 20, 40))
sim$s <- sample(c("a", "b", "c"), n_sim, replace = TRUE)
sim$ps <- unname(predict(glm(treat ~ z1 + z2, data = sim, family = binomial), type = "response"))
sim$tied <- round(sim$ps * 20) / 20
# A set in which the treated units have the lower scores, so the ranking runs
# the other way.
low <- sim
low$treat <- 1L - low$treat
low$ps <- 1 - low$ps
low$tied <- round(low$ps * 20) / 20

scenarios <- list(
  lalonde_prob = list(d = lal, dist = "ps", kind = "vec"),
  lalonde_linear = list(d = lal, dist = "lin", kind = "vec"),
  lalonde_prob_unique = list(d = lal, dist = "ps_u", kind = "vec"),
  sim_tied = list(d = sim, dist = "tied", kind = "vec"),
  sim_continuous = list(d = sim, dist = "ps", kind = "vec"),
  low_continuous = list(d = low, dist = "ps", kind = "vec"),
  sim_mahvars = list(d = sim, dist = "ps", kind = "mahvars", mah = ~ z1 + z2)
)

# The numbers of controls of n1 focal units, fewest first, from the paper's
# rule: m controls in all, k_max units with max.controls, one unit with the
# number that makes the total, and the rest with min.controls.
paper_counts <- function(n1, ratio, a, b) {
  m <- round(ratio * n1)
  kmax <- floor((m - a * n1) / (b - a))
  kmin <- n1 - kmax - 1
  c(rep(a, kmin), m - a * kmin - b * kmax, rep(b, kmax))
}

# D: distance, focal units (rows) by pool units (columns); elig: pairs the
# calipers and exact allow; rv: controls per focal unit.
model_match <- function(D, elig, key, order, rv, replace) {
  nf <- nrow(D)
  np <- ncol(D)
  avail <- rep(TRUE, np)
  chosen <- replicate(nf, integer(0), simplify = FALSE)
  ties <- 0L
  find <- function(i, k) {
    cand <- which(elig[i, ] & avail & !(seq_len(np) %in% chosen[[i]]))
    if (!length(cand)) return(integer(0))
    o <- cand[order(D[i, cand], cand)]
    top <- o[seq_len(min(k + 1L, length(o)))]
    if (anyDuplicated(D[i, top]) > 0) ties <<- ties + 1L
    o[seq_len(min(k, length(o)))]
  }
  take <- function(i, j) {
    chosen[[i]] <<- c(chosen[[i]], j)
    if (!replace) avail[j] <<- FALSE
  }
  ord <- switch(order,
    largest = order(key, decreasing = TRUE),
    smallest = order(key),
    seq_len(nf)
  )
  if (replace) {
    for (i in ord) for (j in find(i, rv[i])) take(i, j)
  } else if (!(order %in% c("closest", "farthest"))) {
    active <- ord
    for (r in seq_len(max(rv))) {
      nxt <- integer(0)
      for (i in active) {
        j <- find(i, 1L)
        if (length(j)) {
          take(i, j)
          if (length(chosen[[i]]) < rv[i]) nxt <- c(nxt, i)
        }
      }
      active <- nxt
      if (!any(avail)) break
    }
  } else {
    closest <- order == "closest"
    active <- seq_len(nf)
    for (r in seq_len(max(rv))) {
      if (!any(avail)) break
      ent_i <- integer(0); ent_j <- integer(0); ent_d <- numeric(0)
      for (i in active) {
        j <- find(i, 1L)
        if (length(j)) { ent_i <- c(ent_i, i); ent_j <- c(ent_j, j); ent_d <- c(ent_d, D[i, j]) }
      }
      live <- rep(TRUE, length(ent_i))
      matched <- integer(0)
      while (any(live) && any(avail)) {
        idx <- which(live)
        dd <- ent_d[idx]
        best <- if (closest) idx[order(dd, idx)[1]] else idx[order(-dd, idx)[1]]
        if (sum(ent_d[idx] == ent_d[best]) > 1) ties <- ties + 1L
        i <- ent_i[best]
        if (avail[ent_j[best]]) {
          take(i, ent_j[best]); matched <- c(matched, i); live[best] <- FALSE
        } else {
          j <- find(i, 1L)
          if (length(j)) { ent_j[best] <- j; ent_d[best] <- D[i, j] } else live[best] <- FALSE
        }
      }
      active <- sort(matched[vapply(matched, function(i) length(chosen[[i]]) < rv[i], logical(1))])
    }
  }
  list(rows = chosen, ties = ties)
}

spec <- function(order = "largest", ratio = 2, min = 1, max = 4, replace = FALSE,
                 estimand = "ATT", cal = NULL, exact = FALSE) {
  list(order = order, ratio = ratio, min = min, max = max, replace = replace,
       estimand = estimand, cal = cal, exact = exact)
}

# The numbers of controls as MatchIt 4.8.2 computes them, fewest first (the
# documented rule of the paper with n1 - 1 where it has n1, and a zero shifted
# to the next unit), to tell the runs in which it differs from the paper's.
matchit_counts <- function(n1, ratio, a, b) {
  m <- round(ratio * n1)
  kmax <- floor((m - a * (n1 - 1)) / (b - a))
  kmin <- n1 - kmax - 1
  v <- c(rep(a, kmin), m - (a * kmin + b * kmax), rep(b, kmax))
  while (any(v == 0)) {
    i <- which(v == 0)[1]
    v[i] <- 1
    if (i == length(v)) break
    v[i + 1] <- v[i + 1] - 1
  }
  v
}

specs_for <- function(kind) {
  out <- list()
  for (o in c("largest", "smallest", "data", "closest", "farthest")) {
    out[[length(out) + 1]] <- spec(o, 2, 1, 4)
    out[[length(out) + 1]] <- spec(o, 2.5, 1, 5)
    out[[length(out) + 1]] <- spec(o, 3, 2, 6)
  }
  out[[length(out) + 1]] <- spec("largest", 2, 1, 12)
  out[[length(out) + 1]] <- spec("largest", 1.5, 1, 2)
  out[[length(out) + 1]] <- spec("largest", 3, 1, 5, replace = TRUE)
  out[[length(out) + 1]] <- spec("smallest", 2.5, 2, 5, replace = TRUE)
  out[[length(out) + 1]] <- spec("data", 3, 2, 6, replace = TRUE)
  out[[length(out) + 1]] <- spec(NULL, 2, 1, 4)
  out[[length(out) + 1]] <- spec(NULL, 2, 1, 4, estimand = "ATC")
  out[[length(out) + 1]] <- spec("smallest", 1.5, 1, 3, estimand = "ATC")
  out[[length(out) + 1]] <- spec("largest", 2, 1, 4, cal = 0.25)
  out[[length(out) + 1]] <- spec("closest", 2, 1, 4, cal = 0.25)
  out[[length(out) + 1]] <- spec("largest", 2, 1, 4, cal = 0.5, replace = TRUE)
  out[[length(out) + 1]] <- spec("largest", 2, 1, 4, exact = TRUE)
  out[[length(out) + 1]] <- spec("smallest", 2, 1, 4, exact = TRUE)
  out
}

runs <- list()
data_out <- list()
mismatch <- 0L
for (sname in names(scenarios)) {
  sc <- scenarios[[sname]]
  d <- sc$d
  n <- nrow(d)
  rownames(d) <- as.character(seq_len(n))
  dist_vec <- d[[sc$dist]]
  X <- NULL
  if (!is.null(sc$mah)) {
    X <- MatchIt:::transform_covariates(sc$mah, data = d, method = "mahalanobis",
                                        treat = as.integer(d$treat))
    attr(X, "treat") <- NULL
    X <- unclass(X)
    dimnames(X) <- NULL
  }
  data_out[[sname]] <- list(
    treat = d$treat, s = d$s, age = d$age, dist = dist_vec,
    maha = if (!is.null(X)) lapply(seq_len(ncol(X)), function(j) X[, j]) else NULL,
    kind = sc$kind
  )
  sp_list <- specs_for(sc$kind)
  for (sp in sp_list) {
    focal <- if (sp$estimand == "ATC") 0L else 1L
    fi <- which(d$treat == focal)
    pi <- which(d$treat != focal)
    key <- dist_vec[fi]
    sd_all <- sd(dist_vec)
    D <- abs(outer(dist_vec[fi], dist_vec[pi], "-"))
    elig <- matrix(TRUE, length(fi), length(pi))
    if (!is.null(sp$cal)) elig <- elig & (D <= sp$cal * sd_all)
    if (sp$exact) elig <- elig & outer(d$s[fi], d$s[pi], "==")
    # In the mahvars scenario the pairs are ranked by the Mahalanobis distance
    # of the transformed covariates, with the score as the caliper only.
    if (sc$kind == "mahvars") D <- as.matrix(dist(X))[fi, pi, drop = FALSE]
    default_order <- if (sp$estimand == "ATC") "smallest" else "largest"
    eff_order <- if (is.null(sp$order)) default_order else sp$order
    m_eff <- if (sp$replace && eff_order %in% c("closest", "farthest")) "data" else eff_order

    n1 <- length(fi)
    counts <- paper_counts(n1, sp$ratio, sp$min, sp$max)
    stopifnot(sum(counts) == round(sp$ratio * n1), all(counts >= sp$min), all(counts <= sp$max))
    above <- mean(dist_vec[fi]) > mean(dist_vec[pi])
    rv <- rep(NA_integer_, n1)
    rv[order(dist_vec[fi], decreasing = above)] <- counts
    mod <- model_match(D, elig, key, m_eff, rv, sp$replace)
    defect <- !identical(as.numeric(matchit_counts(n1, sp$ratio, sp$min, sp$max)), as.numeric(counts))

    warns <- character(0)
    f <- withCallingHandlers(
      tryCatch(
        matchit(
          treat ~ age, data = d, method = "nearest", distance = dist_vec,
          mahvars = if (sc$kind == "mahvars") sc$mah else NULL,
          estimand = sp$estimand, m.order = sp$order, ratio = sp$ratio,
          min.controls = sp$min, max.controls = sp$max, replace = sp$replace,
          exact = if (sp$exact) ~ s else NULL,
          caliper = if (!is.null(sp$cal)) c(sp$cal) else NULL, std.caliper = TRUE
        ),
        error = function(e) e
      ),
      warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") }
    )
    if (inherits(f, "error")) stop(sprintf("%s: %s", sname, conditionMessage(f)))
    mm <- f$match.matrix
    rows <- lapply(seq_along(fi), function(i) { v <- mm[i, ]; as.integer(v[!is.na(v)]) })
    model_rows <- lapply(mod$rows, function(v) as.integer(pi[v]))
    agrees <- identical(rows, model_rows)
    # With replacement and no caliper or exact stratum every unit gets the
    # number it was given, so the sizes of matchit()'s rows are its numbers.
    plain <- sp$replace && is.null(sp$cal) && !sp$exact
    their_counts <- if (plain) lengths(rows) else NULL
    counts_ok <- if (plain) identical(as.integer(their_counts), as.integer(rv)) else NA
    if (plain) {
      stopifnot(identical(isTRUE(counts_ok), !defect))
      if (defect) stopifnot(min(their_counts) < sp$min)
    }
    if (!defect && !agrees && mod$ties == 0L) {
      mismatch <- mismatch + 1L
      cat("MODEL DISAGREES WITHOUT TIES:", sname, jsonlite::toJSON(sp, auto_unbox = TRUE), "\n")
    }
    runs[[length(runs) + 1]] <- list(
      scenario = sname, spec = sp, defect = defect, matchit = rows, model = model_rows,
      counts = as.integer(rv), matchit_counts_ok = counts_ok,
      matchit_weights = unname(f$weights),
      ties = mod$ties, agrees = agrees, warnings = unique(warns)
    )
  }
}
cat(length(runs), "runs;", sum(vapply(runs, function(r) r$agrees, logical(1))), "agree with the model;",
    sum(vapply(runs, function(r) r$ties == 0L, logical(1))), "have no ties;",
    mismatch, "disagree without ties\n")
if (mismatch > 0) quit(status = 1)

out <- list(
  versions = list(r = R.version.string, matchit = as.character(packageVersion("MatchIt")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  scenarios = data_out, runs = runs
)
write(toJSON(out, auto_unbox = TRUE, digits = I(17), null = "null", na = "null"),
      file.path(here, "matchit_variable_ratio.json"))
