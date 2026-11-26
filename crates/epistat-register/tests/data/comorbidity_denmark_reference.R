# Reference flags and scores for ComorbidityMap::charlson_denmark from
# heaven::charlsonIndex and heaven::findCondition, given the Danish list.
#
# The list is written out here from the two tables it comes from, in the
# prefix form heaven matches: ICD-10 codes with the register's leading D,
# ICD-8 codes as their digits.
#   Thygesen SK et al. BMC Med Res Methodol 2011;11:83, Table 1 (ICD-10).
#   Christensen S et al. Clin Epidemiol 2011;3:203-11, Table A (ICD-8 and
#   weights; its ICD-10 codes are Thygesen's).
# A range such as I60-I69 or 290.09-290.19 becomes every prefix in it.
#
# 1. masks: every ICD-10 code of three and four characters with the Danish
#    prefix, and every ICD-8 code of three and five digits, against the list
#    by findCondition(match = "start"). Only the codes that match a
#    condition are written, with the conditions they match.
# 2. persons: simulated diagnoses with dates and an index date per person,
#    scored by charlsonIndex with look.back = 5 (5 * 365.25 days, the index
#    day included) and its hierarchy and weights.
#
# Run from the crate directory:
#   Rscript tests/data/comorbidity_denmark_reference.R > tests/data/comorbidity_denmark_reference.json
suppressMessages({library(jsonlite); library(heaven); library(data.table)})
set.seed(20261126)

# Every prefix of the same width from lo to hi, for codes of a fixed letter.
span <- function(letter, lo, hi, width) {
  paste0(letter, formatC(lo:hi, width = width, flag = "0"))
}
icd10 <- function(...) paste0("D", c(...))
list_dk <- list(
  mi = c("410", icd10("I21", "I22", "I23")),
  chf = c("42709", "42710", "42711", "42719", "42899", "78249",
          icd10("I50", "I110", "I130", "I132")),
  pvd = c(as.character(440:445), icd10("I70", "I71", "I72", "I73", "I74", "I77")),
  cevd = c(as.character(430:438), icd10(span("I", 60, 69, 2), "G45", "G46")),
  dementia = c(as.character(29009:29019), "29309",
               icd10(span("F", 0, 3, 2), "F051", "G30")),
  cpd = c(as.character(c(490:493, 515:518)),
          icd10(span("J", 40, 47, 2), span("J", 60, 67, 2), "J684", "J701", "J703",
                "J841", "J920", "J961", "J982", "J983")),
  rheumd = c("712", "716", "734", "446", "13599",
             icd10("M05", "M06", "M08", "M09", span("M", 30, 36, 2), "D86")),
  pud = c("53091", "53098", as.character(531:534), icd10("K221", span("K", 25, 28, 2))),
  mld = c("571", "57301", "57304",
          icd10("B18", span("K", 700, 703, 3), "K709", "K71", "K73", "K74", "K760")),
  diab = c("24900", "24906", "24907", "24909", "25000", "25006", "25007", "25009",
           icd10("E100", "E101", "E109", "E110", "E111", "E119")),
  hp = c("344", icd10("G81", "G82")),
  rend = c("403", "404", as.character(580:583), "584", "59009", "59319",
           as.character(75310:75319), "792",
           icd10("I12", "I13", span("N", 0, 5, 2), "N07", "N11", "N14",
                 span("N", 17, 19, 2), "Q61")),
  diabwc = c(as.character(c(24901:24905, 24908, 25001:25005, 25008)),
             icd10(span("E", 102, 108, 3), span("E", 112, 118, 3))),
  canc = c(as.character(140:194), icd10(span("C", 0, 75, 2))),
  leuk = c(as.character(204:207), icd10(span("C", 91, 95, 2))),
  lymph = c(as.character(200:203), "27559",
            icd10(span("C", 81, 85, 2), "C88", "C90", "C96")),
  msld = c("07000", "07002", "07004", "07006", "07008", "57300",
           as.character(45600:45609),
           icd10("B150", "B160", "B162", "B190", "K704", "K72", "K766", "I85")),
  metacanc = c(as.character(195:199), icd10(span("C", 76, 80, 2))),
  aids = c("07983", icd10(span("B", 21, 24, 2)))
)
# ICD-8 codes below 100 keep their leading zero as digits.
stopifnot(all(nchar(list_dk$msld[1:5]) == 5))
keys <- names(list_dk)
# charlsonIndex takes its weights and its hierarchy from its own condition
# names, so the list is passed under those names. The weights and the
# hierarchy are then heaven's, which checks the crate's against them.
heaven_name <- c(
  mi = "myocardial.infarction", chf = "heart.failure",
  pvd = "peripheral.vascular.disease", cevd = "cerebrovascular.disease",
  dementia = "dementia", cpd = "chronic.pulmonary.disease",
  rheumd = "rheumatic.disease", pud = "peptic.ulcer.disease",
  mld = "mild.liver.disease", diab = "diabetes.without.complications",
  hp = "hemiplegia.paraplegia", rend = "renal.disease",
  diabwc = "diabetes.with.complications", canc = "any.malignancy",
  leuk = "leukemia", lymph = "lymphoma", msld = "severe.liver.disease",
  metacanc = "metastatic.solid.tumor", aids = "AIDS.HIV"
)
stopifnot(setequal(heaven_name, names(heaven::charlson.codes)))
list_heaven <- setNames(list_dk, heaven_name[keys])

# 1. Every code against the list.
letters10 <- LETTERS
codes10 <- c(as.vector(outer(letters10, formatC(0:99, width = 2, flag = "0"), paste0)),
             as.vector(outer(letters10, formatC(0:999, width = 3, flag = "0"), paste0)))
codes8 <- c(formatC(0:999, width = 3, flag = "0"), formatC(0:99999, width = 5, flag = "0"))
universe <- data.table(raw = c(paste0("D", codes10), codes8))
universe[, id := .I]
hits <- findCondition(universe, vars = "raw", keep = c("id", "raw"),
                      conditions = list_dk, match = "start")
hits[, bit := match(X, keys) - 1]
masks <- hits[, list(mask = sum(unique(2^bit))), by = list(id, raw)][order(id)]

# 2. Persons with dated diagnoses, drawn from the codes that match and from
# codes next to them, so that the windows and the hierarchy are exercised.
pool <- c(masks$raw, sample(universe$raw, 400))
n <- 300
k <- sample(1:8, n, replace = TRUE)
epoch <- as.Date("1970-01-01")
index <- as.Date("2015-01-01") + sample(0:2000, n, replace = TRUE)
dx <- data.table(
  pnr = rep(seq_len(n), k),
  code = sample(pool, sum(k), replace = TRUE),
  date = rep(index, k) - sample(-400:3000, sum(k), replace = TRUE)
)
# Edge days: a diagnosis on the index day and one on each side of the edge
# of the look-back.
dx[1, date := index[dx$pnr[1]]]
dx[2, date := index[dx$pnr[2]] - floor(5 * 365.25)]
dx[3, date := index[dx$pnr[3]] - floor(5 * 365.25) - 1]
dx[, charlson.date := index[pnr]]
ci <- charlsonIndex(dx, ptid = "pnr", vars = "code", data.date = "date",
                    charlson.date = "charlson.date", look.back = 5,
                    icd.codes = list_heaven)
score <- rep(0, n)
score[ci[[1]]$pnr] <- ci[[1]]$charlson.index
elements <- ci[[2]]
# A condition that no person has is not a column of the result.
for (name in setdiff(heaven_name, names(elements))) elements[[name]] <- 0
flags <- rep(0, n)
el <- as.matrix(elements[, heaven_name[keys], with = FALSE])
flags[elements$pnr] <- as.vector(el %*% (2^(seq_along(keys) - 1)))

cat(toJSON(list(
  versions = list(R = as.character(getRversion()),
                  heaven = as.character(packageVersion("heaven")),
                  data.table = as.character(packageVersion("data.table"))),
  keys = keys,
  universe = list(icd10 = length(codes10), icd8 = length(codes8)),
  masks = list(raw = masks$raw, mask = masks$mask),
  persons = list(
    person = dx$pnr - 1, code = dx$code,
    day = as.numeric(dx$date - epoch),
    index = as.numeric(index - epoch),
    flags = flags, score = score
  )
), digits = NA, auto_unbox = TRUE))
