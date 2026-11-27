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
# A branch's comparison may be limited to the packages it can affect, which
# are its changed packages and every package that depends on them (the
# reverse dependencies in `cargo metadata`), by naming them after the output
# file:
#   scripts/hashes.sh after.txt epistat-did epistat-survival
# Run the same package list on the base commit and on the head. The full run,
# with no package named, is the default and the final check.
#
# The output has one line per hash, `<crate>/<test-file>/<case> <hex hash>`,
# sorted. Each test appends its lines to the file named by EPISTAT_HASH_OUT
# (see `epistat_testkit::hash_out`), one write per line, so tests running on
# several threads share the file without a lock.
#
# Only the test targets that can write a hash line are built and run, with all
# features, in the `ci` profile when the workspace defines it. They are found
# from the source on every run: an integration test file under a crate's
# tests/ directory that mentions hash_out, and the library unit tests of a
# crate whose src/ mentions it (doctests write no hashes). The names come from the [[test]]
# entries of the crate's Cargo.toml, or the file name where there is none. If
# a mention cannot be tied to such a target, for instance because it sits in a
# helper module under tests/ or in an example, the script selects every test
# target of the crate (tests/ helpers) or stops with an error (anything else),
# so a hash test cannot go unrecorded. The script stops with an error if a test
# fails, because a failed test may not have written its lines. Ignored tests
# are not run.
#
# CARGO_CMD names the cargo to run (default `cargo`).
set -eu

if [ "$#" -lt 1 ]; then
    echo "usage: scripts/hashes.sh OUTPUT-FILE [PACKAGE ...]" >&2
    exit 2
fi
out=$1
shift
cd "$(dirname "$0")/.."

raw=$(mktemp)
tmp=$(mktemp -d)
trap 'rm -rf "$raw" "$tmp"' EXIT
: > "$raw"

if grep -q '^\[profile\.ci\]' Cargo.toml; then
    profile="--profile ci"
else
    profile=""
fi

# The package name of a crate directory, from its manifest.
package_name() {
    awk '/^\[/ { in_package = ($0 == "[package]"); next }
         in_package && /^name[ \t]*=/ { gsub(/^[^"]*"|".*$/, ""); print; exit }' "$1/Cargo.toml"
}

# `<name> <path>` for each [[test]] entry of a crate's manifest.
test_entries() {
    awk 'function flush() { if (name != "") print name, (path == "" ? "tests/" name ".rs" : path); name = ""; path = "" }
         /^\[/ { flush(); in_test = ($0 == "[[test]]"); next }
         in_test && /^name[ \t]*=/ { n = $0; gsub(/^[^"]*"|".*$/, "", n); name = n }
         in_test && /^path[ \t]*=/ { p = $0; gsub(/^[^"]*"|".*$/, "", p); path = p }
         END { flush() }' "$1/Cargo.toml"
}

# Every package of the workspace, `<name> <directory>`.
: > "$tmp/all"
for dir in crates/*; do
    [ -f "$dir/Cargo.toml" ] && echo "$(package_name "$dir") $dir" >> "$tmp/all"
done

: > "$tmp/chosen"
if [ "$#" -eq 0 ]; then
    cp "$tmp/all" "$tmp/chosen"
else
    for pkg in "$@"; do
        if ! grep "^$pkg " "$tmp/all" >> "$tmp/chosen"; then
            echo "hashes.sh: no workspace package named $pkg" >&2
            exit 2
        fi
    done
fi

# The defining crate mentions hash_out in its own source without writing lines.
: > "$tmp/pkgs"      # packages with a test target to run
: > "$tmp/tests"     # test target names
: > "$tmp/libs"      # packages whose library unit tests run
bad=0
while read -r pkg dir; do
    [ "$pkg" = epistat-testkit ] && skip_src=1 || skip_src=0
    # A source mention outside the testkit's definition: the unit tests.
    if [ "$skip_src" -eq 0 ] && [ -d "$dir/src" ] && grep -rqE 'hash_out' "$dir/src" --include='*.rs'; then
        echo "$pkg" >> "$tmp/libs"
    fi
    # Anything else that mentions it and is not a test file or the library.
    for sub in examples benches build.rs; do
        if [ -e "$dir/$sub" ] && grep -rqE 'hash_out' "$dir/$sub" --include='*.rs'; then
            echo "hashes.sh: $dir/$sub mentions hash_out but is not a test target" >&2
            bad=1
        fi
    done
    [ -d "$dir/tests" ] || continue
    test_entries "$dir" > "$tmp/entries"
    all_targets=0
    : > "$tmp/mine"
    grep -rlE 'hash_out' "$dir/tests" --include='*.rs' | LC_ALL=C sort > "$tmp/files" || true
    while read -r file; do
        [ -n "$file" ] || continue
        rel=${file#"$dir"/}
        # The target this file is the root of: a [[test]] entry, else a file
        # directly under tests/, else tests/NAME/main.rs.
        name=$(awk -v p="$rel" '$2 == p { print $1; exit }' "$tmp/entries")
        if [ -z "$name" ] && ! grep -q '^autotests *= *false' "$dir/Cargo.toml"; then
            case $rel in
                tests/*/main.rs) name=${rel#tests/}; name=${name%/main.rs} ;;
                tests/*/*) ;;
                tests/*.rs) name=${rel#tests/}; name=${name%.rs} ;;
            esac
        fi
        if [ -n "$name" ]; then
            echo "$name" >> "$tmp/mine"
        else
            all_targets=1
        fi
    done < "$tmp/files"
    if [ "$all_targets" -eq 1 ]; then
        # A helper module under tests/ writes the lines for whichever targets
        # include it, so every target of the crate runs.
        echo "$pkg" >> "$tmp/pkgs"
        echo "$pkg" >> "$tmp/alltests"
    elif [ -s "$tmp/mine" ]; then
        echo "$pkg" >> "$tmp/pkgs"
        cat "$tmp/mine" >> "$tmp/tests"
    fi
done < "$tmp/chosen"
if [ "$bad" -ne 0 ]; then
    exit 1
fi

run() {
    # shellcheck disable=SC2086 # $profile is empty or two words
    EPISTAT_HASH_OUT=$raw ${CARGO_CMD:-cargo} test $profile --all-features --no-fail-fast "$@"
}

# Integration tests, one cargo run for all of them. A `--test NAME` selects
# the target of that name in every selected package that has one, so a name
# shared by two crates (study_path) runs in both.
if [ -s "$tmp/tests" ] || [ -s "${tmp}/alltests" ]; then
    set --
    if [ -s "$tmp/alltests" ]; then
        # Whole crates: every integration test of the crate.
        while read -r pkg; do set -- "$@" -p "$pkg"; done < "$tmp/alltests"
        run "$@" --tests
    fi
    if [ -s "$tmp/tests" ]; then
        set --
        while read -r pkg; do
            grep -qx "$pkg" "${tmp}/alltests" 2>/dev/null || set -- "$@" -p "$pkg"
        done < "$tmp/pkgs"
        for name in $(LC_ALL=C sort -u "$tmp/tests"); do set -- "$@" --test "$name"; done
        run "$@"
    fi
fi
if [ -s "$tmp/libs" ]; then
    set --
    while read -r pkg; do set -- "$@" -p "$pkg"; done < "$tmp/libs"
    run "$@" --lib
fi

# The same case twice with different hashes would hide a difference between
# two runs, so it is an error.
LC_ALL=C sort "$raw" > "$out"
LC_ALL=C awk '$1 == key && $2 != hash { print "conflicting hashes for " $1 > "/dev/stderr"; bad = 1 }
              { key = $1; hash = $2 } END { exit bad }' "$out"
echo "$(wc -l < "$out" | tr -d ' ') hashes written to $out"
