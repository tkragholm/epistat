#!/bin/sh
# Records the hash of every pinned result in the workspace, so that two commits
# can be compared on a machine where the recorded constants are not asserted
# (the constants were recorded on aarch64 macOS; on other targets the golden,
# fingerprint and study-path tests only compute their hashes).
#
# Usage, from the workspace root:
#   scripts/hashes.sh before.txt     on the base commit
#   scripts/hashes.sh after.txt      on the head
#   diff before.txt after.txt        every line that differs is a moved hash
#
# The output has one line per hash, `<crate>/<test-file>/<case> <hex hash>`,
# sorted. Each test appends its lines to the file named by EPISTAT_HASH_OUT
# (see `epistat_testkit::hash_out`), one write per line, so tests running on
# several threads share the file without a lock. The script runs the whole
# workspace's tests with all features, in the `ci` profile when the workspace
# defines it, and stops with an error if a test fails, because a failed test
# may not have written its lines. Ignored tests are not run.
#
# CARGO_CMD names the cargo to run (default `cargo`).
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: scripts/hashes.sh OUTPUT-FILE" >&2
    exit 2
fi
out=$1
cd "$(dirname "$0")/.."

raw=$(mktemp)
trap 'rm -f "$raw"' EXIT
: > "$raw"

if grep -q '^\[profile\.ci\]' Cargo.toml; then
    profile="--profile ci"
else
    profile=""
fi

# shellcheck disable=SC2086 # $profile is empty or two words
EPISTAT_HASH_OUT=$raw ${CARGO_CMD:-cargo} test $profile --workspace --all-features --no-fail-fast

# The same case twice with different hashes would hide a difference between
# two runs, so it is an error.
LC_ALL=C sort "$raw" > "$out"
LC_ALL=C awk '$1 == key && $2 != hash { print "conflicting hashes for " $1 > "/dev/stderr"; bad = 1 }
              { key = $1; hash = $2 } END { exit bad }' "$out"
echo "$(wc -l < "$out" | tr -d ' ') hashes written to $out"
