# epistat

Rust crates for statistics in epidemiology and register research.

| Crate | Contents |
|---|---|
| [`epistat-causal`](crates/epistat-causal) | Augmented inverse probability weighting for binary and continuous outcomes, with efficient and stacked variances |
| [`epistat-core`](crates/epistat-core) | The shared types: confidence levels, intervals, estimates, covariances, codes for labels, the person-period key, civil dates, input checks and error kinds |
| [`epistat-did`](crates/epistat-did) | Difference-in-differences: staggered ATT(g,t) and its aggregations, doubly robust two-period estimators, fixed-effects, imputation, switching and synthetic estimators |
| [`epistat-honestdid`](crates/epistat-honestdid) | HonestDiD sensitivity analysis of event studies and the power of the pre-trends test |
| [`epistat-glm`](crates/epistat-glm) | Generalised linear models fitted by IRLS as R's `glm.fit` does |
| [`epistat-impute`](crates/epistat-impute) | Multiple imputation by chained equations: normal, predictive mean matching, logistic and multinomial methods, passive columns and survival predictors |
| [`epistat-inference`](crates/epistat-inference) | Influence-function and sandwich variances, cluster-robust and bootstrap intervals |
| [`epistat-linalg`](crates/epistat-linalg) | Dense solves with fallbacks and row-blocked regression products on faer, independent of the thread count |
| [`epistat-matching`](crates/epistat-matching) | Nearest-neighbour and risk-set matching under calipers and register constraints, with weights, exclusion reasons and balance statistics |
| [`epistat-mixed`](crates/epistat-mixed) | Linear and generalised linear mixed models with one grouping factor, by REML, ML, the Laplace approximation and adaptive Gauss-Hermite quadrature, as `lme4` fits them |
| [`epistat-optim`](crates/epistat-optim) | BFGS with a Moré-Thuente line search, Nelder-Mead and a dense simplex method |
| [`epistat-register`](crates/epistat-register) | Calendar-year person-time, interval merging, connected components over shared keys and constant-price deflation |
| [`epistat-rng`](crates/epistat-rng) | `SplitMix64` streams and bootstrap draws that reproduce across threads, platforms and languages |
| [`epistat-semicontinuous`](crates/epistat-semicontinuous) | Two-part, Tweedie, log-normal and marginalised two-part models for semicontinuous outcomes such as costs |
| [`epistat-sequences`](crates/epistat-sequences) | Optimal-matching and Hamming distances, k-medoids, silhouettes and typology validation for state sequences |
| [`epistat-stats`](crates/epistat-stats) | Distribution functions, sample quantiles, sums, weighted moments and k-means, checked against R and Python |
| [`epistat-epi`](crates/epistat-epi) | Rates, Mantel-Haenszel and standardised estimates, SIRs, E-values, bias analysis, Rosenbaum bounds and balance statistics, checked against R |
| [`epistat-survival`](crates/epistat-survival) | Cox, Fine-Gray and multi-state regression, Aalen-Johansen and multi-state occupation with cluster bootstraps, Kaplan-Meier, pseudo-observations and prediction metrics |
| [`epistat-testkit`](crates/epistat-testkit) | Tolerances, bit fingerprints, thread-count checks and R reference files, for tests |

## Reference implementations

The tests compare results with the R and Python packages below, through
reference values that scripts in each crate's `tests/data` (`tests/r` in
`epistat-semicontinuous`) write with them. Nothing in epistat is copied from
these packages. The reference data and the scripts that write them are left
out of the published packages.

- `AdhereR` (register): treatment episodes of drug supply, `compute.treatment.episodes`.
- `AF` (glm, survival): attributable fractions, `AFglm` and `AFcoxph`, run from the archived package's source.
- `bacondecomp` (did): the Goodman-Bacon decomposition of two-way fixed effects, `bacon`.
- `boot` (matching): bootstrap intervals, `boot.ci`.
- `brant` (glm): the Brant test of proportional odds.
- `CBPS` (causal, matching): covariate balancing propensity score weights and coefficients.
- `cmprsk` (survival): Fine-Gray regression and cumulative incidence, `crr` and `cuminc`.
- `comorbidity` (register): Charlson and Elixhauser flags and scores.
- `contdid` (did): the average causal response to a continuous dose, `cont_did`.
- `DasGuptR` (epi): Das Gupta's rate decomposition.
- `designmatch` (matching): pair matching with fine balance, `bmatch`.
- `did` (did): group-time ATTs, their aggregations and the multiplier bootstrap, `att_gt`, `aggte` and `mboot`.
- `did2s` (did): two-stage difference-in-differences.
- `didimputation` (did): the imputation estimator of Borusyak, Jaravel and Spiess.
- `DIDmultiplegtDYN` (did): dynamic effects of de Chaisemartin and D'Haultfoeuille, `did_multiplegt_dyn`.
- `DRDID` (did): doubly robust, outcome regression and IPW difference-in-differences for panels and repeated cross-sections.
- `ebal` (causal): entropy balancing weights for the ATT.
- `Epi` (inference, matching, register, survival): risk-set sampling, Lexis splitting and linear contrasts, `ccwc`, `splitLexis`, `timeBand` and `ci.lin`.
- `episensr` (epi): bias analysis for misclassification, selection and confounding, simple and probabilistic.
- `etm` (survival): Aalen-Johansen transition probabilities.
- `etwfe` (did): extended two-way fixed effects, linear and GLM, with `emfx`.
- `EValue` (epi, matching): E-values for ratios, differences and hazards.
- `eventglm` (survival): regression on pseudo-observations, `cumincglm` and `rmeanglm`.
- `fixest` (did, glm): fixed-effects OLS, `sunab` and event-study regressions, `feols` and `fixef`.
- `flexsurv` (survival): Royston-Parmar spline models, `flexsurvspline`.
- `gee` (glm): generalised estimating equations.
- `geepack` (glm, survival): GEE fits and regression on pseudo-values, `geeglm` and `geese.fit`.
- `gfoRmula` (causal): the parametric g-formula, with censoring.
- `glmnet` (glm, matching, semicontinuous): elastic-net paths and cross-validation.
- `heaven` (matching, register): exposure density sampling, drug exposure periods and the Danish Charlson index.
- `HonestDiD` (did, honestdid): the sensitivity analysis of Rambachan and Roth, its confidence sets and FLCI.
- `Hmisc` (linalg): restricted cubic spline bases and knots, `rcspline.eval`.
- `ipw` (survival): point and time-varying treatment and censoring weights, `ipwpoint` and `ipwtm`.
- `kinship2` (register): kinship coefficients of pedigrees.
- `lme4` (mixed): linear and generalised linear mixed models, `lmer` and `glmer`, and Gauss-Hermite rules.
- `marginaleffects` (matching): average effects after matching, `avg_comparisons`.
- `MASS` (glm): negative binomial and proportional odds models, `glm.nb` and `polr`.
- `Matching` (matching): Abadie-Imbens matching and genetic matching, `Match` and `GenMatch`.
- `MatchIt` (epi, matching): matching methods, matching weights and balance summaries.
- `MatchThem` (matching): matching within multiply imputed data.
- `metafor` (epi): Mantel-Haenszel pooled measures with the Breslow-Day and Tarone tests, `rma.mh`.
- `mice` (impute, inference, matching): the imputation steps and Rubin's rules, `mice`, `pool`, `pool.scalar` and `D3`.
- `MissingDiD` (did, Python): the authors' replication code of Testa, Kennedy and Reimherr for a missing pre-treatment outcome, run with numpy, pandas and statsmodels.
- `mitml` (inference): the D1 and D2 pooled tests, `testModels`.
- `mpmath` (stats, Python): regularised incomplete beta and gamma functions at high precision.
- `mstate` (survival): multi-state regression, transition probabilities and the Markov test, `msfit`, `probtrans` and `MarkovTest`.
- `multipleNCC` (survival): inclusion probabilities and weighted Cox fits for nested case-control data.
- `mvtnorm` (honestdid): multivariate normal probabilities for the power of pre-trend tests.
- `nnet` (causal, glm, impute, matching): multinomial logit fits, `multinom`.
- `optmatch` (matching): optimal full and pair matching.
- `ordinal` (glm): cumulative link models, `clm`.
- `popEpi` (epi, register, survival): rates, SIRs, Lexis splitting and Poisson excess hazard models.
- `pretrends` (honestdid): the power of pre-trend tests of Roth.
- `prodlim` (survival): Kaplan-Meier and cumulative incidence curves and jackknife pseudo-values.
- `pseudo` (survival): pseudo-observations for survival, restricted means and cumulative incidence.
- `PSweight` (survival): overlap and matching weights, `SumStat`.
- `quickmatch` (matching): generalised full matching.
- `R` (epi, glm, inference, linalg, optim, stats, survival): `glm`, `lm`, `optimize`, the `p`, `q` and special functions of `stats`, `kmeans`, `p.adjust`, `poisson.test`, `mantelhaen.test` and `splines::ns`.
- `rbounds` (matching): Rosenbaum bounds for the Wilcoxon and Hodges-Lehmann statistics.
- `rcbalance` (matching): refined covariate balance matching.
- `relsurv` (survival): net survival by Pohar Perme, Ederer and Hakulinen, its tests and crude probabilities.
- `riskRegression` (causal, survival): AIPW and g-formula risks, absolute risks, Brier scores, AUC and calibration, `ate`, `CSC` and `Score`.
- `rms` (glm, inference, survival): ordinal regression, spline contrasts and `cph`, `orm` and `Predict`.
- `rSPARCS` (survival): case-crossover referent days.
- `rstpm2` (survival): flexible parametric and excess hazard models with predictions, `stpm2`.
- `sandwich` (glm, matching, survival): robust and cluster-robust covariances, `vcovHC`, `vcovCL` and `estfun`.
- `SCCS` (survival): self-controlled case series, `standardsccs`, `semisccs`, `eventdepenexp` and `samplesize`.
- `season` (survival): the time-stratified case-crossover design, `casecross`.
- `sensitivity2x2xk` (matching): Rosenbaum sensitivity analysis for binary outcomes.
- `sensitivitymv` and `sensitivitymw` (matching): Rosenbaum sensitivity analysis and intervals for matched sets.
- `statmod` (glm, semicontinuous): the Tweedie family of `glm`.
- `stdReg` (survival): standardised survival from Cox models, `stdCoxph`.
- `survival` (stats, survival): `coxph`, `survfit`, `survreg`, `cox.zph`, `clogit`, `cch`, `finegray`, `pyears`, `survdiff`, `concordance`, `royston`, `yates`, `brier`, `survexp`, `pseudo`, `tmerge`, `survSplit` and `cipoisson`.
- `survRM2` (survival): restricted mean survival times, with covariates, `rmst2`.
- `synthdid` (did): synthetic difference-in-differences, its standard errors and effect curves.
- `timeROC` (survival): the time-dependent AUC.
- `tmle` (causal): targeted maximum likelihood with `glm` learners.
- `TrialEmulation` (matching, survival): the sequential trial expansion, its weights and the per-protocol marginal structural model.
- `triplediff` (did): triple differences, `ddd`.
- `TruncatedNormal` (honestdid): truncated normal quantiles.
- `VGAM` (glm): partial proportional odds models, `vglm`.
- `WeightIt` (causal, survival): entropy balancing, overlap and matching weights, and the M-estimation variance of weighted Cox fits.

## Contributing

The golden, fingerprint and study-path tests assert their recorded hashes only
on aarch64 macOS. On other targets they compute the hashes and assert nothing:
on x86 the last bits depend on the vector extensions faer selects at run time,
so two runs are comparable only on the same machine.
To check that a change keeps every hash on any target, run
`scripts/hashes.sh before.txt` on the base commit and `scripts/hashes.sh
after.txt` on the head, then `diff before.txt after.txt`. Each line is
`<crate>/<test-file>/<case> <hash>`, and a line that differs is a hash that moved.
The script builds only the tests that write hashes. Naming packages after the
output file limits it to those packages, which is enough for a branch that
can only affect them and the packages that depend on them.

Licensed under either of MIT or Apache-2.0, at your option.
