// An options literal that ends in the defaults compiles.
use epistat_core::Extensible;

#[derive(Default)]
pub struct FitOptions {
    pub max_iterations: usize,
    pub tolerance: f64,
    #[doc(hidden)]
    pub _extensible: Extensible,
}

fn main() {
    let options = FitOptions {
        max_iterations: 50,
        ..Default::default()
    };
    assert_eq!(options.max_iterations, 50);
}
