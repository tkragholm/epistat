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
#
# Results, in SCRATCH/out:
#   runs.csv        impl,n,method,status,process_wall_s,peak_rss_bytes
#   rust_times.csv  n,method,reps,median_s,min_s,matched   (matching only)
#   r_times.csv     the same for R
#   compare_N.csv   what compare.R prints for each size
#
# LIMIT applies to both programs. A method is skipped at a size when its time
# at the previous size, scaled up linearly by the ratio of the sizes, already
# exceeds LIMIT, or when it did not finish at the previous size. Peak memory is
# `maximum resident set size` of /usr/bin/time -l, which includes reading the
# cohort; the method "load" is the same process with no matching.
set -u
: "${SCRATCH:?set SCRATCH to a directory outside the repository}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CRATE="$(cd "$HERE/../.." && pwd)"
SIZES="${SIZES:-100000 1000000 10000000}"
METHODS="${METHODS:-load nn_ps nn_ps_cal nn_strata nn_cal_strata nn_1to3 nn_mahal subclass exact cem quick optimal_cal full_cal riskset balance}"
LIMIT="${LIMIT:-1800}"
IMPLS="${IMPLS:-rust r}"
export PATH="/opt/homebrew/opt/r/bin:$PATH"
mkdir -p "$SCRATCH/out"
[ -f "$SCRATCH/out/runs.csv" ] || echo "impl,n,method,status,process_wall_s,peak_rss_bytes" > "$SCRATCH/out/runs.csv"

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
  elif [ "$2" -ge 10000000 ]; then echo 1
  elif [ "$1" = r ] && [ "$2" -ge 1000000 ]; then echo 1
  elif [ "$1" = r ] && [ "$2" -ge 100000 ] && [[ " nn_mahal optimal_cal full_cal riskset " == *" $3 "* ]]; then echo 1
  else echo 3; fi
}

previous_time() { # impl method n_prev
  awk -F, -v n="$3" -v m="$2" '$1==n && $2==m {t=$4} END {print t}' "$SCRATCH/out/$1_times.csv" 2>/dev/null
}

record() { # impl n method status wallfile
  local rss wall
  rss=$(awk '/maximum resident set size/ {print $1}' "$5")
  wall=$(awk '/ real / {print $1}' "$5")
  echo "$1,$2,$3,$4,${wall:-NA},${rss:-NA}" >> "$SCRATCH/out/runs.csv"
}

# Runs one method of one implementation under the time limit.
run_one() { # impl n method
  local impl=$1 n=$2 m=$3 skip="" pt st status tf
  if [ -n "$prev" ]; then
    pt=$(previous_time "$impl" "$m" "$prev")
    if [ -n "$pt" ] && awk -v t="$pt" -v a="$n" -v b="$prev" -v l="$LIMIT" 'BEGIN{exit !(t*a/b > l)}'; then skip=1; fi
    # A method that did not finish at the previous size will not finish now.
    if grep -q "^$impl,$prev,$m,\(timeout\|skipped\)" "$SCRATCH/out/runs.csv"; then skip=1; fi
  fi
  if [ -n "$skip" ]; then
    echo "$impl,$n,$m,skipped,NA,NA" >> "$SCRATCH/out/runs.csv"
    return
  fi
  tf="$SCRATCH/out/time_${impl}_${m}_${n}.txt"
  if [ "$impl" = rust ]; then
    N=$n METHOD=$m REPS="$(reps_for rust "$n" "$m")" SCRATCH="$SCRATCH" \
      /usr/bin/time -l -o "$tf" gtimeout -k 30 "$LIMIT" "$BIN" > "$SCRATCH/out/log_rust_${m}_${n}.txt" 2>&1
  else
    REPS="$(reps_for r "$n" "$m")" /usr/bin/time -l -o "$tf" \
      gtimeout -k 30 "$LIMIT" Rscript "$HERE/matchit_bench.R" "$n" "$m" "$SCRATCH" \
      > "$SCRATCH/out/log_r_${m}_${n}.txt" 2>&1
  fi
  st=$?
  status=ok; [ $st -eq 124 ] && status=timeout; [ $st -ne 0 ] && [ $st -ne 124 ] && status="error$st"
  record "$impl" "$n" "$m" "$status" "$tf"
}

prev=""
for n in $SIZES; do
  [ -f "$SCRATCH/cohort_$n.csv" ] || N=$n METHOD=gen SCRATCH="$SCRATCH" "$BIN"
  for m in $METHODS; do
    for impl in $IMPLS; do run_one "$impl" "$n" "$m"; done
  done
  Rscript "$HERE/compare.R" "$n" "$SCRATCH" > "$SCRATCH/out/compare_$n.csv" 2>&1
  prev=$n
done
