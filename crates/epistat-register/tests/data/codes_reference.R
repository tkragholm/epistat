# Reference results for epistat-register::codes, written from the rules the
# crate documents, in base R with byte-order comparison (no R package holds
# these rules).
#
# 1. Code sets: random specifications of prefixes and ranges over a small
#    alphabet, and every code of one to four characters over it. A code
#    matches an item of length L from lo to hi when it has at least L
#    characters and lo <= its first L characters <= hi. Union, intersection
#    and difference are the logical operations on the membership vectors.
#    The longest match of a code over two sets is the set whose matching item
#    is longest, and a tie when both match with items of one length.
# 2. ATC levels: the level of a code from its length and pattern, a letter,
#    two digits, a letter, a letter and two digits, and its ancestors as the
#    prefixes of the shorter levels.
# 3. ICD-8 routing: normalisation of codes of either revision. A code that
#    begins with a digit after punctuation and spaces are dropped is ICD-8
#    when it holds only digits; otherwise the ICD-10 rule applies: trim and
#    upper-case, drop a leading D that a letter follows (the Danish register
#    prefix), drop characters that are not letters or digits and any digits
#    before the first letter. Membership of a two-revision condition is the
#    set of the code's own revision.
# 4. Crosswalks: random tables of ICD-8 to ICD-10 pairs with declared kinds,
#    some declared wrongly. A table is refused when a pair is given twice or
#    a kind is contradicted: one to one needs a source with one target that
#    has one source, one to many a source with several targets, many to one a
#    target with several sources. For accepted tables, the coverage of a set
#    of query codes.
#
# Run from the crate directory:
#   Rscript tests/data/codes_reference.R > tests/data/codes_reference.json
suppressMessages(library(jsonlite))
invisible(Sys.setlocale("LC_COLLATE", "C"))
set.seed(20261126)

# 1. Code sets.
alphabet <- c("0", "1", "9", "A", "B", "C")
universe <- character(0)
for (len in 1:4) {
  grid <- do.call(expand.grid, c(rep(list(alphabet), len), stringsAsFactors = FALSE))
  universe <- c(universe, do.call(paste0, grid))
}
random_item <- function() {
  len <- sample(1:3, 1)
  a <- paste(sample(alphabet, len, replace = TRUE), collapse = "")
  if (runif(1) < 0.4) {
    b <- paste(sample(alphabet, len, replace = TRUE), collapse = "")
    ends <- sort(c(a, b))
    list(len = len, lo = ends[1], hi = ends[2])
  } else {
    list(len = len, lo = a, hi = a)
  }
}
spec_of <- function(items) {
  # Dots inside an item are ignored, so one is put after the first character
  # of some items.
  text <- vapply(items, function(it) {
    dot <- function(s) if (nchar(s) > 1 && runif(1) < 0.3)
      paste0(substr(s, 1, 1), ".", substr(s, 2, nchar(s))) else s
    if (it$lo == it$hi) dot(it$lo) else paste0(dot(it$lo), "-", dot(it$hi))
  }, "")
  paste(text, collapse = ", ")
}
matches <- function(items) {
  m <- rep(FALSE, length(universe)); best <- rep(0L, length(universe))
  for (it in items) {
    head <- substr(universe, 1, it$len)
    hit <- nchar(universe) >= it$len & head >= it$lo & head <= it$hi
    m <- m | hit
    best[hit] <- pmax(best[hit], it$len)
  }
  list(member = m, longest = best)
}
sets <- list()
for (k in 1:25) {
  a <- replicate(sample(1:5, 1), random_item(), simplify = FALSE)
  b <- replicate(sample(1:5, 1), random_item(), simplify = FALSE)
  ma <- matches(a); mb <- matches(b)
  longest <- ifelse(ma$longest == 0 & mb$longest == 0, -1L,
              ifelse(ma$longest == mb$longest, -2L,
               ifelse(ma$longest > mb$longest, 0L, 1L)))
  # Membership as one character per code of the universe.
  bits <- function(x) paste(ifelse(x, "1", "0"), collapse = "")
  sets[[k]] <- list(
    a = spec_of(a), b = spec_of(b),
    a_in = bits(ma$member), b_in = bits(mb$member),
    union = bits(ma$member | mb$member),
    intersection = bits(ma$member & mb$member),
    difference = bits(ma$member & !mb$member),
    # n for no match, t for a tie, a or b for the set.
    longest = paste(c("n", "t", "a", "b")[longest + 3], collapse = ""),
    longest_item = max(vapply(c(a, b), function(it) it$len, 1L))
  )
}

# 2. ATC levels.
atc_chars <- c(LETTERS[1:6], "N", as.character(0:9))
atc <- replicate(1500, {
  len <- sample(c(0:8, 1, 3, 4, 5, 7, 7), 1)
  # Most codes follow the pattern and some break it at one place.
  pattern <- c("L", "D", "D", "L", "L", "D", "D", "D")
  chars <- vapply(seq_len(len), function(i) {
    if (runif(1) < 0.9) {
      if (pattern[i] == "L") sample(LETTERS[1:6], 1) else sample(as.character(0:9), 1)
    } else sample(atc_chars, 1)
  }, "")
  paste(chars, collapse = "")
})
level_regex <- c("^[A-Z]$", "^[A-Z][0-9]{2}$", "^[A-Z][0-9]{2}[A-Z]$",
                 "^[A-Z][0-9]{2}[A-Z]{2}$", "^[A-Z][0-9]{2}[A-Z]{2}[0-9]{2}$")
level_len <- c(1, 3, 4, 5, 7)
level <- vapply(atc, function(x) {
  hit <- which(vapply(level_regex, function(r) grepl(r, x), TRUE))
  if (length(hit)) hit - 1L else -1L
}, 1L, USE.NAMES = FALSE)
ancestors <- lapply(seq_along(atc), function(i) {
  vapply(1:5, function(l) if (level[i] >= l - 1) substr(atc[i], 1, level_len[l]) else "", "")
})

# 3. ICD-8 routing.
raw <- c("410.99", "DI219", "411.09", "d55", "D55", "DD55", "dj45.0", " 250 09 ",
         "4A1", "", ".", "-", "DE10", "E10.9", "D", "DA", "7", "12-3", "D.55",
         "q99", "D-J45", "000.00")
pool8 <- c("410", "41099", "250", "25009", "4109", "411", "41109", "070", "07000")
pool10 <- c("I21", "I219", "E10", "E109", "D55", "J450", "Q99", "I2")
for (i in 1:300) {
  x <- if (runif(1) < 0.5) sample(pool8, 1) else paste0(if (runif(1) < 0.7) "D" else "", sample(pool10, 1))
  if (runif(1) < 0.3 && nchar(x) > 3) x <- paste0(substr(x, 1, 3), ".", substr(x, 4, nchar(x)))
  if (runif(1) < 0.2) x <- tolower(x)
  if (runif(1) < 0.1) x <- paste0(" ", x, " ")
  raw <- c(raw, x)
}
normalize_icd10 <- function(code) {
  t <- toupper(trimws(code))
  if (t == "") return(NA_character_)
  t <- gsub("[[:space:]]", "", t)
  if (substr(t, 1, 1) == "D" && grepl("^[A-Z]", substr(t, 2, 2))) t <- substr(t, 2, nchar(t))
  t <- gsub("[^A-Z0-9]", "", t)
  t <- sub("^[0-9]*", "", t)
  if (t == "") NA_character_ else t
}
route <- function(code) {
  clean <- toupper(gsub("[^A-Za-z0-9]", "", code))
  if (clean == "") return(c(NA_character_, NA_character_))
  if (grepl("^[0-9]", clean)) {
    if (grepl("^[0-9]+$", clean)) c("Icd8", clean) else c(NA_character_, NA_character_)
  } else {
    n <- normalize_icd10(code)
    if (is.na(n)) c(NA_character_, NA_character_) else c("Icd10", n)
  }
}
routed <- t(vapply(raw, route, c("", ""), USE.NAMES = FALSE))
# A condition: infarction as 410 in ICD-8 and I21 in ICD-10.
era8 <- list(list(len = 3, lo = "410", hi = "410"))
era10 <- list(list(len = 3, lo = "I21", hi = "I21"))
in_items <- function(code, items) any(vapply(items, function(it)
  nchar(code) >= it$len && substr(code, 1, it$len) >= it$lo && substr(code, 1, it$len) <= it$hi, TRUE))
era_member <- vapply(seq_along(raw), function(i) {
  if (is.na(routed[i, 1])) FALSE
  else if (routed[i, 1] == "Icd8") in_items(routed[i, 2], era8)
  else in_items(routed[i, 2], era10)
}, TRUE)

# 4. Crosswalks.
kinds <- c("OneToOne", "OneToMany", "ManyToOne", "Approximate")
codes8 <- c("410", "41099", "250", "25009", "070", "07000", "344")
codes10 <- c("I21", "I219", "E10", "E109", "B15", "G81", "G82")
tables <- list()
for (k in 1:60) {
  m <- sample(1:7, 1)
  from <- sample(codes8, m, replace = TRUE)
  to <- sample(codes10, m, replace = TRUE)
  if (runif(1) < 0.15 && m > 1) { from[m] <- from[1]; to[m] <- to[1] }
  targets <- vapply(from, function(f) sum(from == f), 1L)
  sources <- vapply(to, function(t) sum(to == t), 1L)
  kind <- ifelse(targets == 1 & sources == 1, "OneToOne",
           ifelse(targets > 1, "OneToMany", ifelse(sources > 1, "ManyToOne", "OneToOne")))
  # Some pairs are declared approximate, and some wrongly.
  kind[runif(m) < 0.2] <- "Approximate"
  wrong <- runif(m) < 0.15
  kind[wrong] <- sample(kinds, sum(wrong), replace = TRUE)
  dup <- any(duplicated(paste(from, to)))
  consistent <- all(ifelse(kind == "OneToOne", targets == 1 & sources == 1,
                    ifelse(kind == "OneToMany", targets > 1,
                    ifelse(kind == "ManyToOne", sources > 1, TRUE))))
  ok <- !dup && consistent
  query <- unique(sample(c(codes8, "999"), sample(1:6, 1)))
  cov <- NULL
  if (ok) {
    mapped <- query[query %in% from]
    has <- function(kd) sum(vapply(mapped, function(q) any(kind[from == q] == kd), TRUE))
    cov <- list(codes = length(query), mapped = length(mapped),
                unmapped = sort(setdiff(query, from)),
                one_to_one = has("OneToOne"), one_to_many = has("OneToMany"),
                many_to_one = has("ManyToOne"), approximate = has("Approximate"))
  }
  tables[[k]] <- list(from = from, to = to, kind = kind, ok = ok, query = query,
                      coverage = cov)
}

cat(toJSON(list(
  versions = list(R = as.character(getRversion())),
  alphabet = alphabet, universe = universe,
  sets = sets,
  atc = list(code = atc, level = level, ancestors = ancestors),
  routing = list(raw = raw, revision = routed[, 1], code = routed[, 2], member = era_member),
  crosswalks = tables
), digits = NA, auto_unbox = TRUE, na = "null"))
