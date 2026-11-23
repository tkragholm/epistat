#!/usr/bin/env bash
# Times chained-equation imputation in Rust and in mice on the same simulated
# data, one process per method and size, with the memory cap and time limit of
# bench/tier2/run.sh.
#
#   SCRATCH=/some/dir/outside/the/repo bench/tier3/impute_run.sh
#
# Settings, all optional: SIZES (default "10000 100000 1000000"), METHODS
# (default "pmm norm logreg polyreg"), LIMIT (seconds per process, default
# 600), REPS (default 1 at 10^6 rows, otherwise 3), OUT, BIN (a built
# tier3_impute program instead of cargo bench).
#
# Results in OUT: runs.csv (impl,n,method,status,process_wall_s,peak_rss_kb),
# rust_times.csv and r_times.csv (n,method,reps,median_s,min_s) and the numbers
# each side wrote for the comparison. The imputed values agree in distribution
# only, so the comparison is the mean, variance and level shares of one
# imputation and not a value-by-value match.
set -u
: "${SCRATCH:?set SCRATCH to a directory outside the repository}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SIZES="${SIZES:-10000 100000 1000000}"
METHODS="${METHODS:-pmm norm logreg polyreg}"
LIMIT="${LIMIT:-600}"
OUT="${OUT:-$SCRATCH/out}"
mkdir -p "$OUT"
[ -f "$OUT/runs.csv" ] || echo "impl,n,method,status,process_wall_s,peak_rss_kb" > "$OUT/runs.csv"
BIN="${BIN:-$(cd "$ROOT" && cargo bench -p epistat-impute --bench tier3_impute --no-run --message-format=json 2>/dev/null |
  python3 -c "
import json, sys
for line in sys.stdin:
    try:
        j = json.loads(line)
    except ValueError:
        continue
    if j.get('reason') == 'compiler-artifact' and j.get('executable') and j['target']['name'] == 'tier3_impute':
        print(j['executable'])")}"
for n in $SIZES; do
  reps=3; [ "$n" -ge 1000000 ] && reps=1
  [ -f "$SCRATCH/imp_$n.csv" ] || N=$n METHOD=gen_imp SCRATCH="$SCRATCH" OUT="$OUT" "$BIN"
  for m in $METHODS; do
    for impl in rust r; do
      tf="$OUT/time_${impl}_${m}_${n}.txt"
      if [ "$impl" = rust ]; then
        N=$n METHOD=$m REPS=$reps SCRATCH="$SCRATCH" OUT="$OUT" \
          python3 "$HERE/../tier1/measure.py" "$tf" "$LIMIT" "$BIN" > "$OUT/log_rust_${m}_${n}.txt" 2>&1
      else
        REPS=$reps OUT="$OUT" \
          python3 "$HERE/../tier1/measure.py" "$tf" "$LIMIT" Rscript "$HERE/impute.R" "$n" "$m" "$SCRATCH" \
          > "$OUT/log_r_${m}_${n}.txt" 2>&1
      fi
      st=$?; status=ok; [ $st -eq 124 ] && status=timeout; [ $st -ne 0 ] && [ $st -ne 124 ] && status="error$st"
      rss=$(awk -F': ' '/Maximum resident set size/ {print $2}' "$tf")
      wall=$(awk -F': ' '/Elapsed/ {print $2}' "$tf")
      echo "$impl,$n,$m,$status,${wall:-NA},${rss:-NA}" >> "$OUT/runs.csv"
    done
  done
  [ -n "${KEEP_DATA:-}" ] || rm -f "$SCRATCH/imp_$n.csv"
done
