use std::fs::OpenOptions;
use std::io::Write;

use crate::Fingerprint;

/// Whether the recorded hashes were produced on this target, aarch64 macOS.
/// Tests assert against their recorded constants only where this is true.
pub const RECORDED_PLATFORM: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));

/// Appends one line, `<crate>/<test-file>/<case> <hex hash>`, to the file named
/// by the environment variable `EPISTAT_HASH_OUT`, and does nothing when the
/// variable is not set.
///
/// Whitespace in `case` becomes `_`, so that a line is a name and a hash.
///
/// The crate is read from `CARGO_PKG_NAME` and the test file from the name of
/// the running test binary, so the same call in the same test gives the same
/// line on every platform. Each line is written with one append, so tests that
/// run on several threads, or several binaries in turn, can share the file.
/// `scripts/hashes.sh` sorts the lines and writes them to a file, so that the
/// hashes of two commits can be compared with `diff` on a target where the
/// recorded constants are not asserted.
///
/// # Panics
///
/// Panics if the variable is set and the file cannot be opened or written.
pub fn hash_out(case: &str, hash: u64) {
    let Some(path) = std::env::var_os("EPISTAT_HASH_OUT") else {
        return;
    };
    let package = std::env::var("CARGO_PKG_NAME").unwrap_or_else(|_| "unknown".into());
    let case = case.replace(char::is_whitespace, "_");
    let line = format!("{package}/{}/{case} {hash:016x}\n", test_file());
    OpenOptions::new()
        .append(true)
        .create(true)
        .open(&path)
        .and_then(|mut file| file.write_all(line.as_bytes()))
        .unwrap_or_else(|e| panic!("cannot append to EPISTAT_HASH_OUT {}: {e}", path.display()));
}

/// [`hash_out`] of the digest of `words` and their count, for a case recorded
/// as a list of integers.
pub fn hash_out_words(case: &str, words: &[u64]) {
    let hash = words
        .iter()
        .fold(Fingerprint::new().u64(words.len() as u64), |h, &w| h.u64(w));
    hash_out(case, hash.finish());
}

/// The name of the running test binary without the hash cargo appends.
fn test_file() -> String {
    let exe = std::env::current_exe().ok();
    let stem = exe
        .as_ref()
        .and_then(|path| path.file_stem())
        .map_or_else(|| "unknown".into(), |s| s.to_string_lossy().into_owned());
    match stem.rsplit_once('-') {
        Some((name, suffix))
            if suffix.len() == 16 && suffix.bytes().all(|b| b.is_ascii_hexdigit()) =>
        {
            name.to_string()
        }
        _ => stem,
    }
}
