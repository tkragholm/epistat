# Timeline data turned into counting-process intervals by survival::fromtimeline
# (surv2counting, the conversion that coxph, survfit and survcheck run on a Surv2
# response), survival 3.8-12.
#
# Cases: the examples of the Surv2 help page, a handful of hand-made edge cases
# (missing status, single observations, unsorted rows, subjects that differ in
# having an initial state), the pbcseq timelines of the timeline vignette, and
# randomized timelines with every combination of status kind, repeated and lvcf.
# A case where R stops with an error is written with error = true.
# Run: Rscript surv2.R > surv2.json
suppressMessages({
  library(jsonlite)
  library(survival)
})
set.seed(20261122)

# d: data.frame(id, time, status [integer, NA allowed], covariates)
# kind: "binary" or m = number of states
run <- function(name, d, m, repeated, lvcf, covs) {
  rownames(d) <- NULL
  d$ev <- if (m == 0) d$status else factor(d$status, 0:m, c("cen", paste0("s", seq_len(m))))
  f <- as.formula(paste("Surv2(time, ev) ~", if (length(covs)) paste(covs, collapse = " + ") else "1"))
  rep <- list(events = TRUE, collapse = FALSE, first = "first")[[repeated]]
  res <- tryCatch(
    suppressWarnings(fromtimeline(f, data = d, id = id, repeated = rep, lvcf = lvcf,
                                  yname = c("tstart", "tstop", "status"))),
    error = function(e) conditionMessage(e))
  inp <- list(id = d$id, time = d$time, status = d$status,
              covs = lapply(covs, function(v) I(d[[v]])))
  out <- list(name = name, m = m, repeated = repeated, lvcf = lvcf, input = inp)
  if (is.character(res)) {
    out$error <- res
    return(out)
  }
  status <- res$status
  if (is.factor(status)) status <- as.integer(status) - 1L
  out$result <- list(
    row = as.integer(rownames(res)) - 1L,
    start = if ("tstop" %in% names(res)) I(res$tstart) else NULL,
    stop = I(if ("tstop" %in% names(res)) res$tstop else res$tstart),
    status = I(status),
    istate = if ("istate" %in% names(res)) I(as.integer(res$istate)) else NULL,
    covs = lapply(covs, function(v) I(res[[v]])))
  out
}

cases <- list()
add <- function(...) cases[[length(cases) + 1]] <<- run(...)
all_modes <- function(name, d, m, covs = character(0)) {
  for (rp in c("events", "collapse", "first")) for (lv in c(TRUE, FALSE))
    add(paste(name, rp, lv), d, m, rp, lv, covs)
}

# Surv2 help page: CU, CU, MCI, CU, MCI, MCI, dementia  (states 1 = CU, 2 = MCI, 3 = dementia)
help1 <- data.frame(id = 1L, time = 1:7, status = c(1, 1, 2, 1, 2, 2, 3))
all_modes("help_example", help1, 3)
# the same with a second subject that has a different path, and a covariate
help2 <- data.frame(id = rep(1:2, c(7, 4)), time = c(1:7, c(2, 5, 6, 9)),
                    status = c(1, 1, 2, 1, 2, 2, 3, 1, 3, 3, 2),
                    x = c(1, NA, 3, NA, NA, 6, 7, 9, NA, 8, 8))
all_modes("help_two_subjects", help2, 3, "x")

# binary status, repeated events, missing status and covariates
bin <- data.frame(id = c(4, 4, 4, 4, 9, 9, 2, 2, 2, 7),
                  time = c(3, 1, 2, 4, 5, 1, 10, 20, 15, 3),
                  status = c(0, 0, 1, 1, 0, NA, 0, 1, 0, 0),
                  x = c(NA, 1.5, 2.5, 3.5, NA, NA, 4, NA, 6, 7))
all_modes("binary_edge", bin, 0, "x")

# one row per subject after conversion (no start column)
two <- data.frame(id = rep(1:5, each = 2), time = rep(c(0, 4), 5) + rep(0:4, each = 2),
                  status = c(0, 1, 0, 0, 0, 1, 0, 1, 0, 0), x = c(1:10))
all_modes("two_rows_each", two, 0, "x")
# competing risks, no initial state, missing status that is taken as censor
cmp <- data.frame(id = rep(1:4, c(3, 2, 4, 3)), time = c(0, 2, 5, 0, 3, 0, 1, 2, 6, 0, 4, 8),
                  status = c(0, 0, 2, 0, 1, 0, NA, 0, 3, 0, 0, 1))
all_modes("competing", cmp, 3)
# some subjects have an initial state and some do not: R stops
mixed <- data.frame(id = rep(1:2, each = 3), time = rep(1:3, 2), status = c(1, 1, 2, 0, 0, 2))
add("mixed_initial", mixed, 3, "collapse", TRUE, character(0))
# an initial event in binary data
binit <- data.frame(id = rep(1:2, each = 3), time = rep(1:3, 2), status = c(1, 0, 1, 1, 1, 0))
add("binary_initial", binit, 0, "collapse", TRUE, character(0))
# a duplicated time
dup <- data.frame(id = c(1, 1, 1), time = c(1, 2, 2), status = c(0, 0, 1))
add("duplicated_time", dup, 0, "collapse", TRUE, character(0))

# pbcseq, as in the timeline vignette
temp <- with(subset(pbcseq, !duplicated(id)),
             data.frame(id = id, day = futime, death = 1 * (status == 2)))
pdata3 <- cbind(pbcseq[, -(2:3)], death = 0)
pdata3 <- merge(pdata3, temp, all = TRUE)
pdata3$bili3 <- cut(pdata3$bili, c(0, 1, 4, 50), c("normal", "1-4", "4+"))
tmp <- with(pdata3, ifelse(!duplicated(id, fromLast = TRUE), 4 * death, as.numeric(bili3)))
pb <- data.frame(id = pdata3$id, time = pdata3$day, bili = pdata3$bili, chol = pdata3$chol,
                 albumin = pdata3$albumin)
pb_bin <- pb; pb_bin$status <- as.integer(pdata3$death)
pb_ms <- pb; pb_ms$status <- ifelse(is.na(tmp), NA, as.integer(tmp))
add("pbcseq binary", pb_bin, 0, "collapse", TRUE, c("bili", "chol", "albumin"))
add("pbcseq binary nolvcf", pb_bin, 0, "collapse", FALSE, c("bili", "chol", "albumin"))
add("pbcseq states", pb_ms, 4, "collapse", TRUE, c("bili", "chol"))
add("pbcseq states events", pb_ms, 4, "events", TRUE, c("bili", "chol"))
add("pbcseq states first", pb_ms, 4, "first", TRUE, c("bili", "chol"))

# randomized timelines
random_case <- function(i) {
  n_id <- sample(5:40, 1)
  m <- sample(c(0, 2, 3), 1)
  initial <- m > 0 && runif(1) < 0.5
  rows <- list()
  allow_bad <- runif(1) < 0.12
  ids <- sample(1000L, n_id) * 3L + 5L
  for (j in seq_len(n_id)) {
    k <- sample(1:7, 1, prob = c(2, 3, 3, 3, 2, 2, 1))
    t <- sort(sample(0:60, k)) + sample(c(0, 0.5), 1)
    if (m == 0) {
      s <- c(rbinom(k - 1, 1, 0.12), rbinom(1, 1, 0.4))
      s[1] <- 0
      s <- if (k > 1) s else 0
    } else if (initial) {
      s <- integer(k)
      s[1] <- sample(1:m, 1)
      for (u in seq_len(k)[-1]) s[u] <- sample(c(0, s[u - 1], s[u - 1], 1:m), 1)
    } else {
      s <- ifelse(runif(k) < 0.25, sample(1:m, k, replace = TRUE), 0)
      s[1] <- 0
    }
    miss <- runif(k) < 0.08
    if (initial && !allow_bad) miss[1] <- FALSE
    s[miss] <- NA
    rows[[j]] <- data.frame(id = ids[j], time = t, status = s,
                            x1 = ifelse(runif(k) < 0.3, NA, round(rnorm(k), 3)),
                            x2 = ifelse(runif(k) < 0.1, NA, sample(1:5, k, replace = TRUE)))
  }
  d <- do.call(rbind, rows)
  # an initial state needs everyone to have one: a missing first status would
  # make it missing, which is taken as 0 and so breaks the rule. Keep some of
  # those cases, since R stops there and Rust must too.
  d[sample(nrow(d)), ]
}
for (i in 1:60) {
  d <- random_case(i)
  m <- if (is.factor(d$status)) 0 else max(c(0, d$status), na.rm = TRUE)
  m <- if (all(d$status %in% c(0, 1, NA)) && i %% 3 != 0) 0 else max(m, 2)
  rp <- sample(c("events", "collapse", "first"), 1)
  lv <- runif(1) < 0.7
  cs <- sample(list(c("x1"), c("x1", "x2"), c("x2"), character(0)), 1)[[1]]
  add(paste("random", i), d, m, rp, lv, cs)
}

versions <- list(R = R.version.string, survival = as.character(packageVersion("survival")),
                 jsonlite = as.character(packageVersion("jsonlite")))
cat(toJSON(list(versions = versions, cases = cases), digits = I(17), auto_unbox = TRUE,
           na = "null", null = "null"))
