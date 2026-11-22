#!/usr/bin/env bash
# Runs the Rust matching program and the MatchIt script on the same simulated
# cohorts, one process per method and size, and records wall time, peak memory
# and whether R finished within the time limit.
#
#   SCRATCH=/some/dir/outside/the/repo benches/matchit/run.sh
#
# Settings, all optional:
#   SIZES     default "100000 1000000 10000000"
#   METHODS   default every method
#   LIMIT     seconds allowed to one process, default 1800
#   IMPLS     "rust r" (default), or just one of them
#   FORCE_REPS  timed repetitions for every run, instead of the defaults
#   OUT       results directory, default SCRATCH/out
#   THREADS   sets RAYON_NUM_THREADS for the Rust program
#   NOSKIP    set to run every size whatever the previous time was
#   NOCOMPARE set to skip compare.R
#
# Results, in OUT:
#   runs.csv        impl,n,method,status,process_wall_s,peak_rss_bytes
#   rust_times.csv  n,method,reps,median_s,min_s,matched   (matching only)
#   r_times.csv     the same for R
#   compare_N.csv   what compare.R prints for each size
#
# LIMIT applies to both programs. A method is skipped at a size when its time
# at the previous size, scaled up by the ratio of the sizes to the power of the
# exponent seen between the last two sizes (at least 1), already exceeds LIMIT,
# or when it did not finish at the previous size. Peak memory is
# `maximum resident set size` of /usr/bin/time -l, which includes reading the
# cohort; the method "load" is the same process with no matching.
set -u
: "${SCRATCH:?set SCRATCH to a directory outside the repository}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CRATE="$(cd "$HERE/../.." && pwd)"
SIZES="${SIZES:-100000 1000000 10000000}"
OUT="${OUT:-$SCRATCH/out}"
METHODS="${METHODS:-load nn_ps nn_ps_cal nn_strata nn_cal_strata nn_1to3 nn_ps_r4 nn_strata_r4 nn_replace nn_unique nn_mahal nn_mahal_strata nn_euclid nn_euclid_strata subclass exact cem quick optimal_cal full_cal fine_balance cardinality profile genetic_eval genetic_short riskset riskset_replace expdens trials balance fit_logit fit_probit fit_enet_cv fit_cbps effects effects_boot abadie sens_sets sens_wilcoxon sens_hl sens_mscore impute multigroup}"
# Methods with no R counterpart in matchit_bench.R.
NO_R=" nn_unique fine_balance expdens trials effects_boot sens_sets impute multigroup "
LIMIT="${LIMIT:-1800}"
IMPLS="${IMPLS:-rust r}"
export PATH="/opt/homebrew/opt/r/bin:$PATH"
mkdir -p "$OUT"
[ -f "$OUT/runs.csv" ] || echo "impl,n,method,status,process_wall_s,peak_rss_bytes" > "$OUT/runs.csv"

BIN="$(cd "$CRATE" && cargo bench --bench matchit_compare --no-run --message-format=json 2>/dev/null |
  python3 -c "
import json, sys
last = None
for line in sys.stdin:
    try:
        j = json.loads(line)
    except ValueError:
        continue
    if j.get('reason') == 'compiler-artifact' and j.get('executable') and j['target']['name'] == 'matchit_compare':
        last = j['executable']
print(last)")"

reps_for() { # impl n method
  if [ -n "${FORCE_REPS:-}" ]; then echo "$FORCE_REPS"
  elif [ "$2" -ge 1000000 ]; then echo 1
  else echo 3; fi
}

previous_time() { # impl method n_prev
  awk -F, -v n="$3" -v m="$2" '$1==n && $2==m {t=$4} END {print t}' "$OUT/$1_times.csv" 2>/dev/null
}

record() { # impl n method status wallfile
  local rss wall
  rss=$(awk '/maximum resident set size/ {print $1}' "$5")
  wall=$(awk '/ real / {print $1}' "$5")
  echo "$1,$2,$3,$4,${wall:-NA},${rss:-NA}" >> "$OUT/runs.csv"
}

# Runs one method of one implementation under the time limit.
run_one() { # impl n method
  local impl=$1 n=$2 m=$3 skip="" pt pp st status tf
  [ "$impl" = r ] && [[ "$NO_R" == *" $m "* ]] && return
  if [ -n "$prev" ] && [ -z "${NOSKIP:-}" ]; then
    pt=$(previous_time "$impl" "$m" "$prev")
    pp=""
    [ -n "$prev2" ] && pp=$(previous_time "$impl" "$m" "$prev2")
    if [ -n "$pt" ] && awk -v t="$pt" -v tp="${pp:-0}" -v a="$n" -v b="$prev" -v c="${prev2:-0}" -v l="$LIMIT" \
         'BEGIN{e=1; if (tp>0 && c>0 && t>1) {e=log(t/tp)/log(b/c); if (e<1) e=1}; exit !(t*exp(e*log(a/b)) > l)}'; then skip=1; fi
    # A method that did not finish at the previous size will not finish now.
    if grep -q "^$impl,$prev,$m,\(timeout\|skipped\|error[0-9]*\)" "$OUT/runs.csv"; then skip=1; fi
  fi
  if [ -n "$skip" ]; then
    echo "$impl,$n,$m,skipped,NA,NA" >> "$OUT/runs.csv"
    return
  fi
  tf="$OUT/time_${impl}_${m}_${n}.txt"
  if [ "$impl" = rust ]; then
    N=$n METHOD=$m REPS="$(reps_for rust "$n" "$m")" SCRATCH="$SCRATCH" OUT="$OUT" \
      /usr/bin/time -l -o "$tf" gtimeout -k 30 "$LIMIT" "$BIN" > "$OUT/log_rust_${m}_${n}.txt" 2>&1
  else
    REPS="$(reps_for r "$n" "$m")" OUT="$OUT" /usr/bin/time -l -o "$tf" \
      gtimeout -k 30 "$LIMIT" Rscript "$HERE/matchit_bench.R" "$n" "$m" "$SCRATCH" \
      > "$OUT/log_r_${m}_${n}.txt" 2>&1
  fi
  st=$?
  status=ok; [ $st -eq 124 ] && status=timeout; [ $st -ne 0 ] && [ $st -ne 124 ] && status="error$st"
  record "$impl" "$n" "$m" "$status" "$tf"
}

[ -n "${THREADS:-}" ] && export RAYON_NUM_THREADS="$THREADS"
prev=""
prev2=""
for n in $SIZES; do
  [ -f "$SCRATCH/cohort_$n.csv" ] || N=$n METHOD=gen SCRATCH="$SCRATCH" "$BIN"
  for m in $METHODS; do
    for impl in $IMPLS; do run_one "$impl" "$n" "$m"; done
  done
  [ -n "${NOCOMPARE:-}" ] || OUT="$OUT" Rscript "$HERE/compare.R" "$n" "$SCRATCH" > "$OUT/compare_$n.csv" 2>&1
  prev2=$prev
  prev=$n
done
