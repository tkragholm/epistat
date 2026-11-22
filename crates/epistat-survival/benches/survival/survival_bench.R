# R side of benches/survival_compare.rs: the same problems on the same cohort
# file, with survival and mstate.
#
# Usage: Rscript survival_bench.R N METHOD SCRATCH
# Environment: REPS (default 3), OUT (default SCRATCH/out), BOOT (default 50).
# Writes OUT/r_METHOD_N.csv, the numbers compare.R sets against the Rust ones,
# and appends n,method,reps,median_s,min_s to OUT/r_times.csv. Only the
# estimation is timed, as on the Rust side: reading the file, building the
# data and, for the _split methods, the cutting are not.
suppressMessages({library(survival); library(mstate); library(data.table)})
args <- commandArgs(TRUE)
n <- args[1]; method <- args[2]; scratch <- args[3]
reps <- as.integer(Sys.getenv("REPS", "3"))
out <- Sys.getenv("OUT", file.path(scratch, "out"))
boot <- as.integer(Sys.getenv("BOOT", "50"))
dir.create(out, showWarnings = FALSE, recursive = TRUE)
CUT <- 750
GRID <- seq(100, 1400, by = 100)
SEED <- 20261122

d <- fread(file.path(scratch, sprintf("cohort_%s.csv", n)))
cut <- grepl("_split$", method)
base <- sub("_split$", "", method)

timed <- function(f) {
  times <- numeric(reps); res <- NULL
  for (r in seq_len(reps)) { times[r] <- system.time(res <- f())[["elapsed"]] }
  list(times = times, res = res)
}
split_data <- function(df) {
  df <- as.data.frame(df)
  df$ev <- as.integer(df$status > 0)
  s <- survSplit(Surv(entry, time, ev) ~ ., data = df, cut = CUT,
                 start = "entry", end = "time", event = "ev", episode = "ep")
  s$status <- s$status * s$ev
  s
}
frame <- function() if (cut) split_data(d) else as.data.frame(d)

step_at <- function(times, values, at, before) {
  k <- findInterval(at, times)
  ifelse(k == 0, before, values[pmax(k, 1)])
}
lines <- NULL
emit <- function(name, value) lines <<- c(lines, sprintf("%s,%.15g", name, value))

if (base == "split") {
  r <- timed(function() split_data(d))
  emit("pieces", nrow(r$res))
} else if (grepl("^cox_", base)) {
  df <- frame()
  ties <- sub("^cox_", "", base)
  r <- timed(function() coxph(Surv(entry, time, status == 1) ~ x1 + x2, data = df, ties = ties))
  # The fit's own components, since summary() does not accept the exact
  # method with start-stop data.
  for (k in 1:2) {
    emit(paste0("coef_x", k), unname(r$res$coefficients)[k])
    emit(paste0("se_x", k), sqrt(r$res$var[k, k]))
  }
  emit("loglik", r$res$loglik[2])
} else if (base == "aj") {
  df <- frame()
  r <- timed(function() survfit(Surv(entry, time, factor(status, 0:2)) ~ arm, data = df, id = id))
  sm <- summary(r$res, times = GRID, extend = TRUE)
  g <- as.integer(sub("arm=", "", as.character(sm$strata)))
  for (k in c("1", "2")) for (i in seq_along(GRID))
    emit(sprintf("cif%s_arm%d_t%g", k, g[i], sm$time[i]), sm$pstate[i, which(colnames(sm$pstate) == k)])
} else if (base == "aj_boot") {
  df <- as.data.frame(d)
  cl <- match(df$fam, sort(unique(df$fam)))
  ncl <- max(cl)
  set.seed(SEED)
  r <- timed(function() {
    res <- array(0, c(boot, length(GRID), 2))
    for (b in seq_len(boot)) {
      w <- rexp(ncl); w <- w / sum(w) * ncl
      df$w <- w[cl]
      fit <- survfit(Surv(time, factor(status, 0:2)) ~ arm, data = df, weights = w)
      sm <- summary(fit, times = GRID, extend = TRUE)
      g <- as.integer(sub("arm=", "", as.character(sm$strata)))
      res[b, , 1] <- sm$pstate[g == 1, "1"]
      res[b, , 2] <- sm$pstate[g == 0, "1"]
    }
    res
  })
  for (a in 1:2) for (i in seq_along(GRID)) {
    nm <- c("arm1", "arm0")[a]
    emit(sprintf("mean_%s_t%g", nm, GRID[i]), mean(r$res[, i, a]))
    emit(sprintf("sd_%s_t%g", nm, GRID[i]), sd(r$res[, i, a]))
  }
} else if (base == "finegray") {
  df <- as.data.frame(d)
  r <- timed(function() {
    fg <- finegray(Surv(time, factor(status, 0:2)) ~ x1 + x2, data = df, etype = "1")
    coxph(Surv(fgstart, fgstop, fgstatus) ~ x1 + x2, data = fg, weights = fgwt)
  })
  s <- summary(r$res)$coefficients
  for (v in c("x1", "x2")) { emit(paste0("coef_", v), s[v, "coef"]); emit(paste0("se_", v), s[v, "se(coef)"]) }
} else if (base == "survfit") {
  df <- frame()
  df$ev <- as.integer(df$status > 0)
  r <- timed(function() survfit(Surv(entry, time, ev) ~ arm, data = df))
  sm <- summary(r$res, times = GRID, extend = TRUE)
  g <- as.integer(sub("arm=", "", as.character(sm$strata)))
  for (i in seq_along(GRID)) {
    emit(sprintf("surv_arm%d_t%g", g[i], sm$time[i]), sm$surv[i])
    emit(sprintf("se_arm%d_t%g", g[i], sm$time[i]), sm$std.err[i])
  }
} else if (base %in% c("multistate", "msurvfit")) {
  # Legs: from state 1 to illness (2) or death (3), then from 2 to death.
  leg1 <- data.table(id = d$id, start = d$entry, stop = d$time,
                     status = c(0L, 2L, 3L)[d$status + 1L], istate = 1L)
  has2 <- d$status == 1 & !is.na(d$t2)
  leg2 <- data.table(id = d$id[has2], start = d$time[has2], stop = d$t2[has2],
                     status = ifelse(d$s2[has2] == 3, 3L, 0L), istate = 2L)
  legs <- rbind(leg1, leg2)
  if (cut) {
    cross <- legs$start < CUT & legs$stop > CUT
    a <- legs[cross]; b <- legs[cross]
    a$stop <- CUT; a$status <- 0L; b$start <- CUT
    legs <- rbind(legs[!cross], a, b)
  }
  setorder(legs, id, start)
  if (base == "msurvfit") {
    legs[, st := factor(status, 0:3)]
    legs[, is := factor(istate, 1:3)]
    r <- timed(function() survfit(Surv(start, stop, st) ~ 1, data = legs, id = id, istate = is))
    sm <- summary(r$res, times = GRID, extend = TRUE)
    for (i in seq_along(GRID)) for (s in 1:3)
      emit(sprintf("p%d_t%g", s, sm$time[i]), sm$pstate[i, which(colnames(sm$pstate) == as.character(s))])
  } else {
    tmat <- transMat(x = list(c(2, 3), 3, c()), names = c("healthy", "ill", "dead"))
    f1 <- legs[istate == 1]; f2 <- legs[istate == 2]
    long <- rbind(
      data.table(id = f1$id, Tstart = f1$start, Tstop = f1$stop, status = as.integer(f1$status == 2L), trans = 1L),
      data.table(id = f1$id, Tstart = f1$start, Tstop = f1$stop, status = as.integer(f1$status == 3L), trans = 2L),
      data.table(id = f2$id, Tstart = f2$start, Tstop = f2$stop, status = as.integer(f2$status == 3L), trans = 3L))
    long <- as.data.frame(long)
    r <- timed(function() {
      cx <- coxph(Surv(Tstart, Tstop, status) ~ strata(trans), data = long, method = "breslow")
      msf <- msfit(cx, newdata = data.frame(trans = 1:3), trans = tmat)
      probtrans(msf, predt = 0, variance = FALSE)[[1]]
    })
    pt <- r$res
    for (i in seq_along(GRID)) for (s in 1:3)
      emit(sprintf("p%d_t%g", s, GRID[i]), pt[[paste0("pstate", s)]][findInterval(GRID[i], pt$time)])
  }
} else stop("unknown method ", method)

writeLines(c("name,value", lines), file.path(out, sprintf("r_%s_%s.csv", method, n)))
med <- median(r$times)
cat(sprintf("n %s %s: median %.3f s over %d runs\n", n, method, med, reps))
cat(sprintf("%s,%s,%d,%.4f,%.4f\n", n, method, reps, med, min(r$times)),
    file = file.path(out, "r_times.csv"), append = TRUE)
