# The event-dependent exposure case series of epistat-survival::sccs against
# SCCS::eventdepenexp (SCCS 1.8) on the package's rotavirus vaccine and
# intussusception data, rotdat: two doses, observation from day 42 to 183 of
# age, one or two risk windows after a gap of one day, ten age groups of two
# weeks, and the doses sharing or not sharing their coefficients.
#
# Mapping of the conventions, as in sccs.R: start = sta, end = end + 1, and
# with aedrug = adrug + 21 the windows of expogrp = 1 are [1, 22) and those
# of expogrp = c(1, 8) are [1, 8) and [8, 22) after the dose; the day of the
# dose itself is the gap.
# Run: Rscript sccs_eventdep.R > sccs_eventdep.json
suppressMessages({
  library(jsonlite)
  library(SCCS)
})
data(rotdat)
rows <- function(M) lapply(seq_len(nrow(M)), function(i) I(unname(M[i, ])))
d <- rotdat[order(rotdat$case), ]
doses <- lapply(seq_len(nrow(d)), function(i) {
  x <- c(d$rv[i], d$rvd2[i]); x[!is.na(x)]
})
fits <- list()
for (expogrp in list(1, c(1, 8))) for (same in c(TRUE, FALSE)) {
  f <- eventdepenexp(indiv = case, astart = sta, aend = end, aevent = intus,
                     adrug = cbind(rv, rvd2), aedrug = cbind(rv + 21, rvd2 + 21),
                     expogrp = expogrp, sameexpopar = same, agegrp = seq(56, 168, 14),
                     dataformat = "multi", data = d)
  cuts <- c(expogrp, 22)
  fits[[length(fits) + 1]] <- list(
    same = same,
    windows = lapply(seq_along(expogrp), function(k) list(from = cuts[k], to = cuts[k + 1])),
    coef = I(unname(f$coefficients[, "coef"])), names = I(rownames(f$coefficients)),
    vcov = rows(f$VarCov))
}
out <- list(versions = list(R = R.version.string, SCCS = as.character(packageVersion("SCCS")),
  gnm = as.character(packageVersion("gnm"))),
  start = d$sta, end = d$end + 1, event = d$intus,
  dose_offsets = c(0, cumsum(sapply(doses, length))), doses = unlist(doses),
  age_cuts = seq(56, 168, 14), fits = fits)
cat(toJSON(out, auto_unbox = TRUE, digits = NA, null = "null"))
