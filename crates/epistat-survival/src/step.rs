//! A right-continuous step function read at a time from the left or the right.

/// A step function: `first` before the first time, then `value[i]` from
/// `time[i]` on. `time` is ascending.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Steps {
    pub time: Vec<f64>,
    pub first: f64,
    pub value: Vec<f64>,
}

impl Steps {
    /// The value at `t`, including a step at `t`.
    pub fn at(&self, t: f64) -> f64 {
        match self.time.partition_point(|&s| s <= t) {
            0 => self.first,
            k => self.value[k - 1],
        }
    }

    /// The value just before `t`, so without a step at `t`.
    pub fn before(&self, t: f64) -> f64 {
        match self.time.partition_point(|&s| s < t) {
            0 => self.first,
            k => self.value[k - 1],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Steps;

    #[test]
    fn a_step_at_t_is_read_by_at_and_not_by_before() {
        let steps = Steps {
            time: vec![1.0, 2.0],
            first: 5.0,
            value: vec![0.5, 0.25],
        };
        assert_eq!(steps.at(0.5).to_bits(), 5.0_f64.to_bits());
        assert_eq!(steps.before(1.0).to_bits(), 5.0_f64.to_bits());
        assert_eq!(steps.at(1.0).to_bits(), 0.5_f64.to_bits());
        assert_eq!(steps.before(2.0).to_bits(), 0.5_f64.to_bits());
        assert_eq!(steps.at(9.0).to_bits(), 0.25_f64.to_bits());
        assert_eq!(steps.before(9.0).to_bits(), 0.25_f64.to_bits());
    }
}
