# The relative-magnitudes reference for the first post-period effect.
#
# Run from the crate root:  Rscript tests/data/generate_rm_first_post_reference.R
#
# Writes honest_did_rm_ref.json: the original confidence set of the first post
# period's effect in HonestDiD's BCdata_EventStudy, and at Mbar 0.5 and 1 the
# DeltaRM identified set and the C-LF confidence set. The grid of the
# confidence set is explicit, centred on the estimate and spanning +/- 40 of its
# standard errors in 10,000 points, as in generate_rm_avg_reference.R, so the
# returned bounds are interior grid points rather than the edge of HonestDiD's
# default grid. HonestDiD seeds its least-favorable simulation itself (seed 0,
# 1,000 draws), so the file is reproducible. The identified set comes from
# HonestDiD's internal .compute_IDset_DeltaRM, the function
# createSensitivityResults_relativeMagnitudes uses.

library(HonestDiD)
library(jsonlite)

# Versions: written with R 4.6.2, HonestDiD 0.2.8 and jsonlite 2.0.0.
stopifnot(packageVersion("HonestDiD") == "0.2.8",
          packageVersion("jsonlite") == "2.0.0")
cat(R.version.string, "\n")

out_path <- "tests/data/honest_did_rm_ref.json"

data(BCdata_EventStudy)
betahat <- BCdata_EventStudy$betahat
sigma <- BCdata_EventStudy$sigma
num_pre <- length(BCdata_EventStudy$prePeriodIndices)
num_post <- length(BCdata_EventStudy$postPeriodIndices)
l_vec <- basisVector(1, num_post)

post <- BCdata_EventStudy$postPeriodIndices
centre <- as.numeric(t(l_vec) %*% betahat[post])
spread <- sqrt(as.numeric(t(l_vec) %*% sigma[post, post] %*% l_vec))
grid_lb <- centre - 40 * spread
grid_ub <- centre + 40 * spread

original <- constructOriginalCS(betahat = betahat, sigma = sigma,
                                numPrePeriods = num_pre, numPostPeriods = num_post,
                                l_vec = l_vec, alpha = 0.05)

rows <- lapply(c(0.5, 1.0), function(mbar) {
  result <- createSensitivityResults_relativeMagnitudes(
    betahat = betahat, sigma = sigma,
    numPrePeriods = num_pre, numPostPeriods = num_post,
    method = "C-LF", Mbarvec = c(mbar), l_vec = l_vec, alpha = 0.05,
    gridPoints = 10000, grid.lb = grid_lb, grid.ub = grid_ub
  )
  id <- HonestDiD:::.compute_IDset_DeltaRM(Mbar = mbar, trueBeta = betahat, l_vec = l_vec,
                                           numPrePeriods = num_pre, numPostPeriods = num_post)
  lb <- result$lb[[1]]
  ub <- result$ub[[1]]
  # The confidence set contains the identified set and lies inside the grid.
  stopifnot(lb <= id$id.lb, ub >= id$id.ub, lb > grid_lb + 1e-8, ub < grid_ub - 1e-8)
  cat(sprintf("Mbar %.1f: C-LF [%.9f, %.9f], identified set [%.9f, %.9f]\n",
              mbar, lb, ub, id$id.lb, id$id.ub))
  list(lb = lb, ub = ub, method = "C-LF", Delta = "DeltaRM", Mbar = mbar,
       id_lb = id$id.lb, id_ub = id$id.ub)
})

rm_ref <- list(
  original = list(lb = as.numeric(original$lb), ub = as.numeric(original$ub),
                  method = "Original"),
  relative_magnitude = rows
)
write_json(rm_ref, out_path, digits = 15, auto_unbox = TRUE)
cat("honest_did_rm_ref.json written.\n")
