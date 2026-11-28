// An options literal that names every documented field and leaves out the
// hidden one does not compile.
use epistat_core::Extensible;

#[derive(Default)]
pub struct FitOptions {
    pub max_iterations: usize,
    pub tolerance: f64,
    #[doc(hidden)]
    pub _extensible: Extensible,
}

fn main() {
    let _ = FitOptions {
        max_iterations: 50,
        tolerance: 1e-8,
    };
}
