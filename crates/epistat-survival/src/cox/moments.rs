//! The moment sums of the Cox risk-set sweep and the information update from
//! them. Both are written for a covariate count fixed at compile time, so the
//! inner loops are full rows of known length that the compiler vectorises.
//! The cells above the diagonal of the second moment are computed and never
//! read.

/// Adds `c x` to `m1` and `c x x'` to the lower triangle of `m2` (a full
/// `p * p` array, or empty when the second moment is not wanted). Covariate
/// counts up to 16 have their own compiled loops.
pub(super) fn add(c: f64, x: &[f64], m1: &mut [f64], m2: &mut [f64]) {
    if m2.is_empty() {
        for (cell, &xa) in m1.iter_mut().zip(x) {
            *cell += c * xa;
        }
        return;
    }
    macro_rules! dispatch {
        ($($p:literal)*) => {
            match x.len() {
                $($p => add_fixed::<$p>(c, x, m1, m2),)*
                _ => add_rows(c, x, m1, m2),
            }
        };
    }
    dispatch!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16);
}

fn add_fixed<const P: usize>(c: f64, x: &[f64], m1: &mut [f64], m2: &mut [f64]) {
    let x: &[f64; P] = x.try_into().expect("a row of P covariates");
    let m1: &mut [f64; P] = m1.try_into().expect("P first moments");
    for a in 0..P {
        let cx = c * x[a];
        m1[a] += cx;
        let row: &mut [f64; P] = (&mut m2[a * P..(a + 1) * P])
            .try_into()
            .expect("a row of P second moments");
        for b in 0..P {
            row[b] += cx * x[b];
        }
    }
}

fn add_rows(c: f64, x: &[f64], m1: &mut [f64], m2: &mut [f64]) {
    let p = x.len();
    for a in 0..p {
        let cx = c * x[a];
        m1[a] += cx;
        for (cell, &xb) in m2[a * p..a * p + a + 1].iter_mut().zip(x) {
            *cell += cx * xb;
        }
    }
}

/// The quotients `d1[a] / d0`, formed once per column.
pub(super) fn quotients(d1: &[f64], d0: f64, e: &mut [f64]) {
    for (q, &d) in e.iter_mut().zip(d1) {
        *q = d / d0;
    }
}

/// Adds `u (d2 / d0 - e[a] e[b])` to the lower triangle of `information`, with
/// `d2 = risk - fraction * dying` cell by cell and `e` from [`quotients`].
pub(super) fn add_information(
    p: usize,
    u: f64,
    d0: f64,
    fraction: f64,
    risk: &[f64],
    dying: &[f64],
    e: &[f64],
    information: &mut [f64],
) {
    for a in 0..p {
        let ea = e[a];
        let range = a * p..a * p + a + 1;
        for (((cell, &r), &d), &eb) in information[range.clone()]
            .iter_mut()
            .zip(&risk[range.clone()])
            .zip(&dying[range])
            .zip(&e[..=a])
        {
            let d2 = r - fraction * d;
            *cell += u * (d2 / d0 - ea * eb);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_add(c: f64, x: &[f64], m1: &mut [f64], m2: &mut [f64]) {
        let p = x.len();
        for a in 0..p {
            let cx = c * x[a];
            m1[a] += cx;
            if !m2.is_empty() {
                for (b, xb) in x.iter().enumerate().take(a + 1) {
                    m2[a * p + b] += cx * xb;
                }
            }
        }
    }

    fn reference_information(
        p: usize,
        u: f64,
        d0: f64,
        fraction: f64,
        risk: &[f64],
        dying: &[f64],
        d1: &[f64],
        information: &mut [f64],
    ) {
        for a in 0..p {
            for b in 0..=a {
                let d2 = risk[a * p + b] - fraction * dying[a * p + b];
                information[a * p + b] += u * (d2 / d0 - (d1[a] / d0) * (d1[b] / d0));
            }
        }
    }

    fn values(state: &mut u64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|_| {
                *state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((*state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
            })
            .collect()
    }

    #[test]
    fn sums_and_information_equal_the_reference_bit_for_bit() {
        let mut state = 7u64;
        for p in [1usize, 2, 3, 5, 7, 8, 9, 15, 16, 17, 20, 33] {
            for second in [true, false] {
                let size = if second { p * p } else { 0 };
                let (mut m1, mut m2) = (vec![0.0; p], vec![0.0; size]);
                let (mut r1, mut r2) = (vec![0.0; p], vec![0.0; size]);
                let (mut d1m, mut d2m) = (vec![0.0; p], vec![0.0; size]);
                for row in 0..50 {
                    let x = values(&mut state, p);
                    let c = if row % 7 == 3 { -0.75 } else { 1.25 } * (1.0 + row as f64 / 9.0);
                    add(c, &x, &mut m1, &mut m2);
                    reference_add(c, &x, &mut r1, &mut r2);
                    if row % 5 == 0 {
                        add(c, &x, &mut d1m, &mut d2m);
                    }
                }
                assert_eq!(bits(&m1), bits(&r1), "m1, p = {p}");
                if !second {
                    continue;
                }
                for a in 0..p {
                    for b in 0..=a {
                        assert_eq!(m2[a * p + b].to_bits(), r2[a * p + b].to_bits());
                    }
                }
                let d0 = 41.5;
                let fraction = 0.5;
                let d1: Vec<f64> = (0..p).map(|a| m1[a] - fraction * d1m[a]).collect();
                let mut e = vec![0.0; p];
                quotients(&d1, d0, &mut e);
                let (mut info, mut expected) = (vec![0.25; p * p], vec![0.25; p * p]);
                add_information(p, 0.8, d0, fraction, &m2, &d2m, &e, &mut info);
                reference_information(p, 0.8, d0, fraction, &r2, &d2m, &d1, &mut expected);
                for a in 0..p {
                    for b in 0..=a {
                        assert_eq!(
                            info[a * p + b].to_bits(),
                            expected[a * p + b].to_bits(),
                            "information, p = {p}"
                        );
                    }
                }
            }
        }
    }

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }
}
