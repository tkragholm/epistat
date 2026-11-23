# The two-period DRDID references: the locally efficient doubly robust
# estimator on a panel and on repeated cross-sections.
#
# Run from the crate root:  Rscript tests/data/generate_drdid_reference.R
#
# drdid_panel_ref.json: DRDID::drdid_panel on 500 units drawn from the LaLonde
# sample with the CPS comparison group, as in the example of ?drdid_panel but
# with fewer units and without re74, keeping the ATT, its standard error and the
# influence function.
#
# drdid_rc_efficient_ref.json: DRDID::drdid_rc, the estimator att_gt calls with
# est_method = "dr" and panel = FALSE, on every group-time cell of did's mpdta
# read as repeated cross-sections. A cell holds the cohort first treated in
# `group` and the never-treated counties in the years g - 1 and `time`, with
# post = (year == time) and the design cbind(1, lpop).

library(DRDID)
library(did)
library(jsonlite)

# Versions: written with R 4.6.2, DRDID 1.3.0, did 2.5.1 and jsonlite 2.0.0.
stopifnot(packageVersion("DRDID") == "1.3.0",
          packageVersion("did") == "2.5.1",
          packageVersion("jsonlite") == "2.0.0")
cat(R.version.string, "\n")

base_path <- "tests/data/"

# --- Panel ---
lalonde <- subset(nsw, nsw$treated == 0 | nsw$sample == 2)
set.seed(123)
lalonde <- lalonde[sort(sample(seq_len(nrow(lalonde)), 500)), ]
covariates <- as.matrix(cbind(1, lalonde$age, lalonde$educ, lalonde$black,
                              lalonde$married, lalonde$nodegree, lalonde$hisp))
panel <- drdid_panel(y1 = lalonde$re78, y0 = lalonde$re75, D = lalonde$experimental,
                     covariates = covariates, inffunc = TRUE)
panel_ref <- list(
  att = panel$ATT,
  se = panel$se,
  y1 = lalonde$re78,
  y0 = lalonde$re75,
  treat = lalonde$experimental,
  covariates = unname(covariates),
  influence = as.vector(panel$att.inf.func)
)
write_json(panel_ref, paste0(base_path, "drdid_panel_ref.json"), digits = 17, auto_unbox = TRUE)
cat("drdid_panel_ref.json written.\n")

# --- Repeated cross-sections ---
data(mpdta)
years <- sort(unique(mpdta$year))
cells <- list()
for (g in sort(setdiff(unique(mpdta$first.treat), 0))) {
  base <- g - 1
  for (t in setdiff(years, base)) {
    cell <- mpdta[mpdta$year %in% c(base, t) & mpdta$first.treat %in% c(g, 0), ]
    post <- as.numeric(cell$year == t)
    d <- as.numeric(cell$first.treat == g)
    fit <- drdid_rc(y = cell$lemp, post = post, D = d,
                    covariates = cbind(1, cell$lpop), inffunc = TRUE)
    cells[[length(cells) + 1]] <- list(
      group = g, baseline_time = base, time = t,
      att = fit$ATT, se = fit$se, inffunc = as.vector(fit$att.inf.func),
      y = cell$lemp, post = post, d = d, lpop = cell$lpop
    )
  }
}
write_json(cells, paste0(base_path, "drdid_rc_efficient_ref.json"), digits = 17, auto_unbox = TRUE)
cat("drdid_rc_efficient_ref.json written.\n")
