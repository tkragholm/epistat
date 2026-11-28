//! The promises of the API that are about what does not compile.

#[test]
fn options_literals_end_in_the_defaults() {
    let t = trybuild::TestCases::new();
    t.pass("tests/ui/options_literal_with_defaults.rs");
    t.compile_fail("tests/ui/exhaustive_options_literal.rs");
}
