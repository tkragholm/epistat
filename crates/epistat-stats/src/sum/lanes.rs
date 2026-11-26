//! Sums and dot products over eight fixed lanes.
//!
//! A sequential sum adds each term to one running total, so every addition
//! waits for the one before it. These functions keep eight running totals, one
//! per position modulo 8, so eight additions are independent and the loop
//! compiles to vector additions. The width is fixed in the source and does not
//! follow the hardware, so the result has the same bits on every target and
//! for every thread count.
//!
//! # Order of operations
//!
//! For `values` of length `n`, the first `n - n mod 8` terms are added into the
//! lanes, lane `j` taking the terms at positions `j`, `j + 8`, `j + 16` and so
//! on in order. The lanes are folded onto each other, lane `j` gaining lane
//! `j + 4`, then `j + 2`, then `j + 1`, so the total is
//! `((l0 + l4) + (l2 + l6)) + ((l1 + l5) + (l3 + l7))`. The last `n mod 8` terms
//! are added to a separate total from left to right, and the result is the lane
//! total plus that tail. With fewer than 8 terms the lanes hold zero and the
//! result is the sequential sum. The terms of a product or other map are
//! rounded one by one before they are added, as in a sequential loop, and no
//! product is fused with an addition.
//!
//! The error of the lane sum is bounded as for any recursive sum by
//! `(n / 8 + 10) u` times the sum of the absolute terms, `u = 2^-53`, a
//! factor of about 8 below the sequential bound for long inputs. It is not the
//! correctly rounded sum. Where a result is defined by an order of operations,
//! as in [`crate::sum::blocked`], [`crate::sum::compensated`] and
//! [`crate::sum::exact`], those functions are used instead.

const LANES: usize = 8;

#[inline]
fn fold(lanes: [f64; LANES]) -> f64 {
    ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6]))
        + ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]))
}

/// The sum of `f(v)` over `values`, in the order described in the module
/// documentation. An empty input gives 0.
#[inline]
pub fn sum_map(values: &[f64], f: impl Fn(f64) -> f64) -> f64 {
    let (chunks, rest) = values.as_chunks::<LANES>();
    let mut lanes = [0.0_f64; LANES];
    for chunk in chunks {
        for (lane, &v) in lanes.iter_mut().zip(chunk) {
            *lane += f(v);
        }
    }
    let mut tail = 0.0;
    for &v in rest {
        tail += f(v);
    }
    fold(lanes) + tail
}

/// The sum of `f(a, b)` over the pairs of two slices of the same length, in the
/// order described in the module documentation.
///
/// # Panics
///
/// Panics when the lengths differ.
#[inline]
pub fn sum_map2<A: Copy, B: Copy>(a: &[A], b: &[B], f: impl Fn(A, B) -> f64) -> f64 {
    assert_eq!(a.len(), b.len(), "slices of different lengths");
    let (ca, ra) = a.as_chunks::<LANES>();
    let (cb, rb) = b.as_chunks::<LANES>();
    let mut lanes = [0.0_f64; LANES];
    for (x, y) in ca.iter().zip(cb) {
        for ((lane, &u), &v) in lanes.iter_mut().zip(x).zip(y) {
            *lane += f(u, v);
        }
    }
    let mut tail = 0.0;
    for (&u, &v) in ra.iter().zip(rb) {
        tail += f(u, v);
    }
    fold(lanes) + tail
}

/// The sum of `values`.
#[must_use]
#[inline]
pub fn sum(values: &[f64]) -> f64 {
    sum_map(values, |v| v)
}

/// The sum of the squares of `values`.
#[must_use]
#[inline]
pub fn sum_squares(values: &[f64]) -> f64 {
    sum_map(values, |v| v * v)
}

/// The sum of the squared deviations of `values` from `centre`.
#[must_use]
#[inline]
pub fn centred_squares(values: &[f64], centre: f64) -> f64 {
    sum_map(values, |v| (v - centre) * (v - centre))
}

/// The dot product of `a` and `b`.
///
/// # Panics
///
/// Panics when the lengths differ.
#[must_use]
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    sum_map2(a, b, |x, y| x * y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sequential(values: &[f64], f: impl Fn(f64) -> f64) -> f64 {
        let mut s = 0.0;
        for &v in values {
            s += f(v);
        }
        s
    }

    fn data(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| ((i * 2_654_435_761) % 1_000_003) as f64 / 997.0 - 500.0)
            .collect()
    }

    #[test]
    fn fewer_than_eight_terms_are_the_sequential_sum() {
        for n in 0..8 {
            let x = data(n);
            assert_eq!(sum(&x).to_bits(), sequential(&x, |v| v).to_bits());
            assert_eq!(
                sum_squares(&x).to_bits(),
                sequential(&x, |v| v * v).to_bits()
            );
        }
        assert_eq!(sum(&[-0.0]).to_bits(), sequential(&[-0.0], |v| v).to_bits());
    }

    #[test]
    fn the_order_is_eight_lanes_folded_in_a_tree_then_the_tail() {
        for n in [8, 9, 15, 16, 17, 63, 64, 65, 1_000, 1_003] {
            let x = data(n);
            let body = n - n % 8;
            let mut l = [0.0_f64; 8];
            for (i, &v) in x[..body].iter().enumerate() {
                l[i % 8] += v;
            }
            let mut tail = 0.0;
            for &v in &x[body..] {
                tail += v;
            }
            let expected =
                (((l[0] + l[4]) + (l[2] + l[6])) + ((l[1] + l[5]) + (l[3] + l[7]))) + tail;
            assert_eq!(sum(&x).to_bits(), expected.to_bits(), "n = {n}");
        }
    }

    #[test]
    fn the_sum_is_close_to_the_exact_sum() {
        for n in [8, 100, 10_001, 100_000] {
            let x = data(n);
            let abs: f64 = x.iter().map(|v| v.abs()).sum();
            let exact = crate::sum::exact(x.iter().copied());
            let bound = (n as f64 / 8.0 + 10.0) * f64::EPSILON / 2.0 * abs;
            assert!((sum(&x) - exact).abs() <= bound, "n = {n}");
        }
    }

    #[test]
    fn dot_and_centred_squares_use_the_same_order() {
        for n in [0, 1, 7, 8, 9, 100, 1_001] {
            let (a, b) = (data(n), data(n + 3));
            let b = &b[3..];
            let products: Vec<f64> = a.iter().zip(b).map(|(x, y)| x * y).collect();
            assert_eq!(dot(&a, b).to_bits(), sum(&products).to_bits());
            let dev: Vec<f64> = a.iter().map(|v| (v - 1.5) * (v - 1.5)).collect();
            assert_eq!(centred_squares(&a, 1.5).to_bits(), sum(&dev).to_bits());
        }
    }

    #[test]
    fn non_finite_terms_propagate_as_in_a_sequential_sum() {
        let mut x = data(20);
        x[11] = f64::NAN;
        assert!(sum(&x).is_nan());
        x[11] = f64::INFINITY;
        assert!(sum(&x).is_infinite() && sum(&x) > 0.0);
        x[12] = f64::NEG_INFINITY;
        assert!(sum(&x).is_nan());
    }

    #[test]
    #[should_panic(expected = "different lengths")]
    fn dot_of_unequal_lengths_panics() {
        let _ = dot(&[1.0], &[1.0, 2.0]);
    }
}
