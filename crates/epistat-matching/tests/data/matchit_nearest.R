# Pairs of MatchIt 4.8.2's method = "nearest" with every deterministic
# m.order, for a distance vector on two scales, Mahalanobis distance, mahvars,
# calipers (raw and in standard deviation units), exact and antiexact.
#
# For every run the file holds, per focal unit in row order, the units matched
# to it in order of assignment:
#   matchit   what matchit() returned (match.matrix)
#   model     the pairs of a short loop written here from the documented
#             behaviour: candidates ranked by distance, ties broken by row
#             order, the focal units taken in the documented order, and, without
#             replacement, in rounds. For m.order = "closest" and "farthest" the
#             pairs are taken in order of their distance, and a unit whose
#             control has gone takes the next nearest in its place by the new
#             distance.
#   ties      how many times the model chose between candidates at the same
#             distance (for closest and farthest also entries of the same
#             distance in the queue). Where this is 0, the model has no tie to
#             break and matchit() must agree with it; the script stops if not.
# The crate must reproduce the model in every run, and matchit() in every run
# without ties.
#
# Distances: for a numeric distance the matching distance is the absolute
# difference. For Mahalanobis and mahvars the script takes MatchIt's own
# transformed covariate matrix (MatchIt:::transform_covariates) and stores its
# columns, so the crate's distance is the Euclidean distance between those
# rows, which is MatchIt's.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/matchit_nearest.R
suppressMessages({
  library(MatchIt)
  library(jsonlite)
})
args <- commandArgs(FALSE)
here <- dirname(sub("--file=", "", args[grep("--file=", args)]))

# ---- data -----------------------------------------------------------------

data(lalonde, package = "MatchIt")
set.seed(20261122)
tr <- sample(which(lalonde$treat == 1), 60)
co <- sample(which(lalonde$treat == 0), 180)
lal <- lalonde[sample(c(tr, co)), ]
lal$race <- as.character(lal$race)
rownames(lal) <- NULL
lal$black <- as.numeric(lal$race == "black")
lal$hispan <- as.numeric(lal$race == "hispan")
lal$s <- paste0(lal$race, lal$married)
lal$fam <- sample(sprintf("f%02d", 1:40), nrow(lal), replace = TRUE)
fit <- glm(treat ~ age + educ + black + hispan + married + nodegree + re74 + re75,
           data = lal, family = binomial)
lal$ps <- unname(predict(fit, type = "response"))
lal$lin <- unname(predict(fit, type = "link"))
# The same scores and covariates made unique by a perturbation far below any
# difference that matters, so that no two units share a distance. lalonde has
# many identical covariate rows, which give identical scores.
rowno <- seq_len(nrow(lal))
lal$ps_u <- lal$ps + 1e-10 * rowno
lal$lin_u <- lal$lin + 1e-9 * rowno
lal$educ_u <- lal$educ + 1e-7 * rowno
lal$re74_u <- lal$re74 + 1e-6 * rowno

n_sim <- 160
sim <- data.frame(treat = rbinom(n_sim, 1, 0.3))
sim$z1 <- rnorm(n_sim) + 0.5 * sim$treat
sim$z2 <- rnorm(n_sim)
sim$age <- round(runif(n_sim, 20, 40))
sim$s <- sample(c("a", "b", "c"), n_sim, replace = TRUE)
sim$fam <- sample(sprintf("f%02d", 1:30), n_sim, replace = TRUE)
sim$ps <- unname(predict(glm(treat ~ z1 + z2, data = sim, family = binomial), type = "response"))
# Scores on a grid of 0.05 (division, so each is the double nearest to its decimal), so many candidates share a distance.
sim$tied <- round(sim$ps * 20) / 20

scenarios <- list(
  lalonde_prob = list(d = lal, dist = "ps", kind = "vec", cov = c("age", "educ", "re74")),
  lalonde_linear = list(d = lal, dist = "lin", kind = "vec", cov = c("age", "educ", "re74")),
  lalonde_prob_unique = list(d = lal, dist = "ps_u", kind = "vec", cov = c("age", "educ", "re74")),
  lalonde_linear_unique = list(d = lal, dist = "lin_u", kind = "vec", cov = c("age", "educ", "re74")),
  sim_tied = list(d = sim, dist = "tied", kind = "vec", cov = "age"),
  sim_continuous = list(d = sim, dist = "ps", kind = "vec", cov = "age"),
  lalonde_mahalanobis = list(d = lal, kind = "maha", mah = ~ age + educ + black + hispan + married + nodegree + re74 + re75, cov = "age"),
  sim_mahalanobis = list(d = sim, kind = "maha", mah = ~ z1 + z2 + age, cov = "age"),
  lalonde_mahalanobis_unique = list(d = lal, kind = "maha", mah = ~ age + educ_u + black + hispan + married + nodegree + re74_u + re75, cov = "age"),
  sim_mahvars = list(d = sim, dist = "ps", kind = "mahvars", mah = ~ z1 + z2, cov = "age"),
  lalonde_mahvars = list(d = lal, dist = "ps", kind = "mahvars", mah = ~ age + educ + re74 + re75, cov = "age"),
  lalonde_mahvars_unique = list(d = lal, dist = "ps_u", kind = "mahvars", mah = ~ age + educ_u + re74_u + re75, cov = "age")
)

# ---- the model ------------------------------------------------------------

# D: distance, focal units (rows) by pool units (columns); elig: which pairs
# the calipers, exact and antiexact allow; key: the focal units' distance
# scores for "largest" and "smallest".
model_match <- function(D, elig, key, order, ratio, replace) {
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
    for (i in ord) for (j in find(i, ratio)) take(i, j)
  } else if (!(order %in% c("closest", "farthest"))) {
    active <- ord
    for (r in seq_len(ratio)) {
      nxt <- integer(0)
      for (i in active) {
        j <- find(i, 1L)
        if (length(j)) {
          take(i, j)
          if (length(chosen[[i]]) < ratio) nxt <- c(nxt, i)
        }
      }
      active <- nxt
      if (!any(avail)) break
    }
  } else {
    closest <- order == "closest"
    active <- seq_len(nf)
    for (r in seq_len(ratio)) {
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
      active <- sort(matched[vapply(matched, function(i) length(chosen[[i]]) < ratio, logical(1))])
    }
  }
  list(rows = chosen, ties = ties)
}

# ---- runs -----------------------------------------------------------------

spec <- function(order = "largest", ratio = 1, replace = FALSE, estimand = "ATT",
                 cal = NULL, std = TRUE, covcal = NULL, exact = FALSE, anti = FALSE) {
  list(order = order, ratio = ratio, replace = replace, estimand = estimand,
       cal = cal, std = std, covcal = covcal, exact = exact, anti = anti)
}

specs_for <- function(kind) {
  all_orders <- if (kind == "maha") c("data", "closest", "farthest") else c("largest", "smallest", "data", "closest", "farthest")
  out <- list()
  for (o in all_orders) for (r in c(1, 2, 3)) out[[length(out) + 1]] <- spec(o, r)
  o1 <- all_orders[1]; o2 <- if (kind == "maha") "closest" else "smallest"
  out[[length(out) + 1]] <- spec(o1, 1, TRUE)
  out[[length(out) + 1]] <- spec(o1, 3, TRUE)
  out[[length(out) + 1]] <- spec(o2, 2, TRUE)
  if (kind != "maha") {
    out[[length(out) + 1]] <- spec(NULL, 2)
    out[[length(out) + 1]] <- spec(NULL, 2, estimand = "ATC")
    out[[length(out) + 1]] <- spec("largest", 1, estimand = "ATC")
  }
  for (o in unique(c(o1, "closest"))) {
    if (kind != "maha") {
      out[[length(out) + 1]] <- spec(o, 2, cal = 0.25)
      out[[length(out) + 1]] <- spec(o, 2, cal = 0.25, std = FALSE)
      out[[length(out) + 1]] <- spec(o, 2, cal = 0.5, replace = TRUE)
    }
    out[[length(out) + 1]] <- spec(o, 2, covcal = 4)
    if (kind != "maha") out[[length(out) + 1]] <- spec(o, 2, cal = 0.25, covcal = 4)
  }
  for (o in unique(c(o1, o2, "closest"))) out[[length(out) + 1]] <- spec(o, 2, exact = TRUE)
  for (o in unique(c(o1, "farthest"))) out[[length(out) + 1]] <- spec(o, 2, anti = TRUE)
  out[[length(out) + 1]] <- spec(o1, 2, cal = if (kind != "maha") 0.5 else NULL, covcal = 5, exact = TRUE, anti = TRUE)
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
  dist_vec <- if (!is.null(sc$dist)) d[[sc$dist]] else NULL
  X <- NULL
  if (!is.null(sc$mah)) {
    X <- MatchIt:::transform_covariates(sc$mah, data = d, method = "mahalanobis",
                                        treat = as.integer(d$treat))
    attr(X, "treat") <- NULL
    X <- unclass(X)
    dimnames(X) <- NULL
  }
  data_out[[sname]] <- list(
    treat = d$treat, s = d$s, fam = d$fam, age = d$age,
    dist = dist_vec, maha = if (!is.null(X)) lapply(seq_len(ncol(X)), function(j) X[, j]) else NULL,
    kind = sc$kind
  )
  for (sp in specs_for(sc$kind)) {
    focal <- if (sp$estimand == "ATC") 0L else 1L
    fi <- which(d$treat == focal)
    pi <- which(d$treat != focal)
    key <- if (!is.null(dist_vec)) dist_vec[fi] else NULL
    sd_all <- if (!is.null(dist_vec)) sd(dist_vec) else NA
    D <- if (sc$kind == "vec") {
      abs(outer(dist_vec[fi], dist_vec[pi], "-"))
    } else {
      as.matrix(dist(X))[fi, pi, drop = FALSE]
    }
    gate <- if (sc$kind == "maha" || is.null(sp$cal)) NULL else {
      w <- if (sp$std) sp$cal * sd_all else sp$cal
      abs(outer(dist_vec[fi], dist_vec[pi], "-")) <= w
    }
    elig <- matrix(TRUE, length(fi), length(pi))
    if (!is.null(gate)) elig <- elig & gate
    if (!is.null(sp$covcal)) elig <- elig & (abs(outer(d$age[fi], d$age[pi], "-")) <= sp$covcal)
    if (sp$exact) elig <- elig & outer(d$s[fi], d$s[pi], "==")
    if (sp$anti) elig <- elig & outer(d$fam[fi], d$fam[pi], "!=")
    default_order <- if (sc$kind == "maha") "data" else if (sp$estimand == "ATC") "smallest" else "largest"
    eff_order <- if (is.null(sp$order)) default_order else sp$order
    eff_replace <- sp$replace
    m_eff <- if (eff_replace && eff_order %in% c("closest", "farthest")) "data" else eff_order
    mod <- model_match(D, elig, key, m_eff, sp$ratio, eff_replace)

    cal <- c(); stdv <- c()
    if (!is.null(sp$cal) && sc$kind != "maha") { cal <- c(cal, sp$cal); stdv <- c(stdv, sp$std); names(cal) <- ""}
    if (!is.null(sp$covcal)) { nm <- c(names(cal), "age"); cal <- c(cal, sp$covcal); names(cal) <- nm; stdv <- c(stdv, FALSE) }
    warns <- character(0)
    f <- withCallingHandlers(
      tryCatch(
        matchit(
          if (sc$kind == "maha") update(sc$mah, treat ~ .) else treat ~ age,
          data = d, method = "nearest",
          distance = if (sc$kind == "maha") "mahalanobis" else dist_vec,
          mahvars = if (sc$kind == "mahvars") sc$mah else NULL,
          estimand = sp$estimand, m.order = sp$order, ratio = sp$ratio, replace = sp$replace,
          exact = if (sp$exact) ~ s else NULL, antiexact = if (sp$anti) ~ fam else NULL,
          caliper = if (length(cal)) cal else NULL,
          std.caliper = if (length(stdv)) stdv else TRUE
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
    if (!agrees && mod$ties == 0L) {
      mismatch <- mismatch + 1L
      cat("MODEL DISAGREES WITHOUT TIES:", sname, jsonlite::toJSON(sp, auto_unbox = TRUE), "\n")
    }
    runs[[length(runs) + 1]] <- list(
      scenario = sname, spec = sp, matchit = rows, model = model_rows,
      ties = mod$ties, agrees = agrees, warnings = unique(warns)
    )
  }
}
cat(length(runs), "runs;", sum(vapply(runs, function(r) r$agrees, logical(1))), "agree with the model;",
    sum(vapply(runs, function(r) r$ties == 0L, logical(1))), "have no ties;",
    mismatch, "disagree without ties\n")
if (mismatch > 0) quit(status = 1)

# Where matchit() breaks a tie between controls at the same distance from a
# treated unit, as its documentation does not say: the control with the
# highest row when they lie below the treated unit's score, the lowest row
# when they lie above it, whatever the documentation's "order in which the data
# appear". Three controls share a score on one side of one treated unit.
tie_case <- function(treat, dist) {
  d <- data.frame(treat = treat, x = seq_along(treat))
  as.integer(matchit(treat ~ x, data = d, distance = dist, method = "nearest",
                     replace = TRUE)$match.matrix[1, 1])
}
tie_cases <- list(
  below = list(treat = c(0, 0, 0, 1, 0), dist = c(0.4, 0.4, 0.4, 0.5, 0.9),
               matchit = tie_case(c(0, 0, 0, 1, 0), c(0.4, 0.4, 0.4, 0.5, 0.9))),
  above = list(treat = c(1, 0, 0, 0, 0), dist = c(0.5, 0.6, 0.6, 0.6, 0.9),
               matchit = tie_case(c(1, 0, 0, 0, 0), c(0.5, 0.6, 0.6, 0.6, 0.9)))
)

out <- list(
  tie_cases = tie_cases,
  versions = list(r = R.version.string, matchit = as.character(packageVersion("MatchIt")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  scenarios = data_out, runs = runs
)
write(toJSON(out, auto_unbox = TRUE, digits = I(17), null = "null", na = "null"),
      file.path(here, "matchit_nearest.json"))
