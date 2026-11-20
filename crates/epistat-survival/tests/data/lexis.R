# Lexis splitting on several time scales against Epi::splitLexis (applied scale
# by scale) and popEpi::splitMulti, and Poisson rate models against glm() with
# Epi::ci.lin and Epi::ci.exp.
#
# Cases: the three-row cohort of the splitLexis help page, the nickel cohort of
# Epi, the first rows of popEpi's sire, random data on a grid of quarters
# (exact in binary, so breaks met together are met exactly and every output
# must agree bit for bit), random data in tenths (where breaks are met
# together only up to rounding), and one row whose last piece is 1e-9 long.
# For each case: the input, the pieces of repeated splitLexis, those of
# splitMulti with drop = FALSE, and those of splitMulti with drop = TRUE.
# The pieces are ordered by row and by the first time scale. Both functions
# drop a piece shorter than sqrt(.Machine$double.eps); the last case records it.
# Seventeen digits are written so that every double reads back exactly.
# Run: Rscript lexis.R > lexis.json
suppressMessages({
  library(jsonlite)
  library(Epi)
  library(popEpi)
  library(data.table)
})
set.seed(20261120)
# Epi and popEpi print notes with cat; keep them out of the JSON.
sink(stderr())

pieces <- function(x, scales) {
  x <- as.data.frame(x)
  x <- x[order(x$row, x[[scales[1]]]), ]
  list(
    row = I(as.integer(x$row)),
    entry = lapply(scales, function(s) I(as.numeric(x[[s]]))),
    dur = I(as.numeric(x$lex.dur)),
    status = I(as.integer(as.character(x$lex.Xst)))
  )
}

run <- function(name, entry, dur, status, breaks, crop_breaks = NULL) {
  scales <- names(entry)
  L <- Lexis(entry = entry, duration = dur, entry.status = 0L,
             exit.status = as.integer(status), id = seq_along(dur))
  L$row <- L$lex.id  # Lexis drops rows of duration 0, so number them first
  x <- L
  for (s in scales) if (!is.null(breaks[[s]])) {
    x <- splitLexis(x, breaks = breaks[[s]], time.scale = s)
  }
  m <- splitMulti(L, breaks = breaks[!vapply(breaks, is.null, TRUE)], drop = FALSE)
  out <- list(
    name = name, scales = scales,
    dur = I(as.numeric(dur)), status = I(as.integer(status)),
    entry = lapply(entry, function(v) I(as.numeric(v))),
    breaks = lapply(scales, function(s) I(as.numeric(breaks[[s]]))),
    epi = pieces(x, scales), multi = pieces(m, scales)
  )
  if (!is.null(crop_breaks)) {
    mc <- splitMulti(L, breaks = crop_breaks, drop = TRUE)
    out$crop_breaks <- lapply(scales, function(s) I(as.numeric(crop_breaks[[s]])))
    out$crop <- pieces(mc, scales)
  }
  out
}

cases <- list()

# the bogus cohort of the help page
xcoh <- data.frame(
  birth = c("14/07/1952", "01/04/1954", "10/06/1987"),
  entry = c("04/08/1965", "08/09/1972", "23/12/1991"),
  exit = c("27/06/1997", "23/05/1995", "24/07/1998"), fail = c(1, 0, 1))
bt <- cal.yr(xcoh$birth, format = "%d/%m/%Y")
en <- cal.yr(xcoh$entry, format = "%d/%m/%Y")
ex <- cal.yr(xcoh$exit, format = "%d/%m/%Y")
cases[[length(cases) + 1]] <- run("xcoh", list(per = en, age = en - bt, tfe = 0 * en),
  ex - en, xcoh$fail,
  list(per = seq(1960, 2000, 5), age = seq(10, 50, 10), tfe = c(0, 5, 10, 20)),
  list(per = c(1970, 1995), age = c(15, 45), tfe = c(0, 20)))

# nickel
data(nickel)
cases[[length(cases) + 1]] <- run("nickel",
  list(per = nickel$agein + nickel$dob, age = nickel$agein, tfe = nickel$agein - nickel$age1st),
  nickel$ageout - nickel$agein, as.integer(nickel$icd %in% c(162, 163)),
  list(per = seq(1930, 1990, 10), age = seq(40, 95, 5), tfe = c(0, 1, 5, 10, 20, 30, 40)),
  list(per = c(1940, 1980), age = c(50, 85), tfe = c(0, 40)))

# popEpi's sire, the first rows
data(sire)
s <- sire[1:1500]
per <- get.yrs(s$dg_date)
cases[[length(cases) + 1]] <- run("sire",
  list(fot = 0 * per, per = per, age = s$dg_age),
  get.yrs(s$ex_date) - per, as.integer(s$status == 1),
  list(fot = seq(0, 20, 1), per = seq(1990, 2015, 5), age = seq(0, 100, 5)),
  list(fot = c(0, 10), per = c(1995, 2010), age = c(20, 90)))

# quarters: exact in binary
n <- 400
ea <- sample(0:80, n, TRUE) / 4
eb <- 30 + sample(0:80, n, TRUE) / 4
d <- (1 + sample(0:59, n, TRUE)) / 4
st <- sample(0:2, n, TRUE)
cases[[length(cases) + 1]] <- run("quarters",
  list(a = ea, b = eb, c = 0 * ea), d, st,
  list(a = seq(0, 40, 2), b = seq(30, 70, 3), c = seq(0, 20, 5)),
  list(a = c(2, 30), b = c(33, 60), c = c(0, 15)))

# tenths: breaks met together up to rounding
ea <- sample(0:200, n, TRUE) / 10
eb <- 30 + sample(0:200, n, TRUE) / 10
d <- (1 + sample(0:149, n, TRUE)) / 10
st <- sample(0:1, n, TRUE)
cases[[length(cases) + 1]] <- run("tenths",
  list(a = ea, b = eb), d, st,
  list(a = seq(0, 40, 0.5), b = seq(30, 70, 1.5)),
  list(a = c(1, 35), b = c(33, 60)))

# a last piece of 1e-9
cases[[length(cases) + 1]] <- run("short_piece",
  list(a = c(0, 0), b = c(10, 10)), c(5 + 1e-9, 7), c(1, 1),
  list(a = c(0, 5), b = NULL))

# Poisson models for rates on the nickel pieces split on age and time since
# first exposure: glm with an offset, ci.lin and ci.exp
nl <- Lexis(entry = list(per = nickel$agein + nickel$dob, age = nickel$agein,
                         tfe = nickel$agein - nickel$age1st),
            duration = nickel$ageout - nickel$agein, entry.status = 0L,
            exit.status = as.integer(nickel$icd %in% c(162, 163)))
nl$exposure <- nickel$exposure
nl$row <- seq_len(nrow(nl))
nl <- splitLexis(nl, breaks = c(50, 60, 70), time.scale = "age")
nl <- splitLexis(nl, breaks = c(0, 20, 30, 40), time.scale = "tfe")
nl <- as.data.frame(nl)
nl$ab <- cut(nl$age, c(0, 50, 60, 70, 200), right = FALSE)
nl$tb <- cut(nl$tfe, c(-100, 20, 30, 40, 200), right = FALSE)
nl$ev <- as.integer(as.character(nl$lex.Xst))
fit <- glm(ev ~ ab + tb + exposure, family = poisson, offset = log(lex.dur), data = nl)
X <- model.matrix(fit)
p <- ncol(X)
unit <- function(...) { v <- rep(0, p); v[c(...)] <- 1; v }
# rates: the reference cell at exposure 0 and 5, a cell in the third age and
# second time band, and a ratio of two cells
ctr <- rbind(unit(1), { v <- unit(1); v[p] <- 5; v }, unit(1, 3, 6), unit(3) - unit(2))
colnames(ctr) <- colnames(X)
cl <- ci.lin(fit, ctr.mat = ctr)
ce <- ci.exp(fit, ctr.mat = ctr)
sm <- summary(fit)
rates <- list(
  length = I(nl$lex.dur), events = I(nl$ev),
  columns = lapply(seq_len(ncol(X)), function(j) I(as.numeric(X[, j]))),
  names = colnames(X),
  coefficients = I(as.numeric(coef(fit))), se = I(as.numeric(sqrt(diag(vcov(fit))))),
  vcov = I(as.numeric(vcov(fit))),
  deviance = fit$deviance, iterations = fit$iter,
  ctr = lapply(seq_len(nrow(ctr)), function(i) I(as.numeric(ctr[i, ]))),
  ci_lin = lapply(seq_len(nrow(cl)), function(i) I(as.numeric(cl[i, c("Estimate", "StdErr", "2.5%", "97.5%", "P")]))),
  ci_exp = lapply(seq_len(nrow(ce)), function(i) I(as.numeric(ce[i, ])))
)

out_json <- toJSON(list(
  r_version = R.version.string,
  epi_version = as.character(packageVersion("Epi")),
  popepi_version = as.character(packageVersion("popEpi")),
  cases = cases, rates = rates
), digits = I(17), auto_unbox = TRUE, na = "null")
sink()
cat(out_json)
