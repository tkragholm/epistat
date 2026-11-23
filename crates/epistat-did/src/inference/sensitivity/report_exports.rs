use std::fmt::Write;

use serde::Serialize;

use super::{
    HonestDirectionalRegion, HonestJointPathMethod, HonestJointPathRegion, HonestMultiFlciResult,
};

const HONEST_REPORT_SCHEMA_V1: &str = "honestdid_joint_report_v1";

const fn method_label(method: HonestJointPathMethod) -> &'static str {
    match method {
        HonestJointPathMethod::Bonferroni => "bonferroni",
        HonestJointPathMethod::GaussianSimulated => "gaussian_simulated",
    }
}

/// One row of a joint-path report, for one post period.
#[derive(Debug, Clone, Serialize)]
pub struct HonestJointPathReportRow {
    /// Version label of the report schema, `honestdid_joint_report_v1`.
    pub schema_version: &'static str,
    /// Calibration method of the joint region: `bonferroni` or
    /// `gaussian_simulated`.
    pub method: &'static str,
    /// Requested joint confidence level, a probability such as 0.95.
    pub confidence_level: f64,
    /// Confidence level of each period's scalar interval, a probability at least
    /// `confidence_level`.
    pub pointwise_confidence_level: f64,
    /// Two-sided max-`|t|` critical value that corresponds to the pointwise
    /// level.
    pub calibrated_max_t_critical_value: f64,
    /// Label of the post-treatment period the row is for.
    pub post_period: i32,
    /// Point estimate of the post-period coefficient.
    pub estimate: f64,
    /// Standard error of `estimate`.
    pub se: f64,
    /// Value that `robustly_significant` is tested against.
    pub null_value: f64,
    /// Value of the sensitivity parameter, `M` or `Mbar`.
    pub sensitivity_value: f64,
    /// Lower end of the Wald interval under exact parallel trends.
    pub original_ci_low: f64,
    /// Upper end of the Wald interval under exact parallel trends.
    pub original_ci_high: f64,
    /// Lower end of the identified set.
    pub identified_lb: f64,
    /// Upper end of the identified set.
    pub identified_ub: f64,
    /// Lower end of the robust confidence interval, run at the pointwise level.
    pub robust_ci_low: f64,
    /// Upper end of the robust confidence interval, run at the pointwise level.
    pub robust_ci_high: f64,
    /// Whether the robust interval excludes `null_value`.
    pub robustly_significant: bool,
}

/// One row of a directional-region report, for one functional.
#[derive(Debug, Clone, Serialize)]
pub struct HonestDirectionalReportRow {
    /// Version label of the report schema, `honestdid_joint_report_v1`.
    pub schema_version: &'static str,
    /// Calibration method of the joint region: `bonferroni` or
    /// `gaussian_simulated`.
    pub method: &'static str,
    /// Requested joint confidence level, a probability such as 0.95.
    pub confidence_level: f64,
    /// Confidence level of each functional's scalar interval, a probability at
    /// least `confidence_level`.
    pub pointwise_confidence_level: f64,
    /// Two-sided max-`|t|` critical value that corresponds to the pointwise
    /// level.
    pub calibrated_max_t_critical_value: f64,
    /// Number of distinct directions in the region.
    pub direction_count: usize,
    /// Number of those directions drawn at random.
    pub random_direction_count: usize,
    /// Number of direction sets built: 1 for a fixed set, the enrichment rounds
    /// for an adaptive one.
    pub iterations_run: usize,
    /// Whether the calibrated pointwise level stabilised before the limits were
    /// reached. `true` for a fixed direction set.
    pub converged: bool,
    /// Zero-based position of the functional in the region's points.
    pub functional_id: usize,
    /// Functional weights in post-period order.
    pub post_weights: Vec<f64>,
    /// Point estimate of the functional.
    pub estimate: f64,
    /// Standard error of `estimate`.
    pub se: f64,
    /// Value that `robustly_significant` is tested against.
    pub null_value: f64,
    /// Value of the sensitivity parameter, `M` or `Mbar`.
    pub sensitivity_value: f64,
    /// Lower end of the Wald interval under exact parallel trends.
    pub original_ci_low: f64,
    /// Upper end of the Wald interval under exact parallel trends.
    pub original_ci_high: f64,
    /// Lower end of the identified set.
    pub identified_lb: f64,
    /// Upper end of the identified set.
    pub identified_ub: f64,
    /// Lower end of the robust confidence interval, run at the pointwise level.
    pub robust_ci_low: f64,
    /// Upper end of the robust confidence interval, run at the pointwise level.
    pub robust_ci_high: f64,
    /// Whether the robust interval excludes `null_value`.
    pub robustly_significant: bool,
}

/// One row of a simultaneous interval report, for one functional.
#[derive(Debug, Clone, Serialize)]
pub struct HonestMultiFlciReportRow {
    /// Version label of the report schema, `honestdid_joint_report_v1`.
    pub schema_version: &'static str,
    /// Calibration method of the joint region: `bonferroni` or
    /// `gaussian_simulated`.
    pub method: &'static str,
    /// Requested joint confidence level, a probability such as 0.95.
    pub confidence_level: f64,
    /// Confidence level of each functional's scalar interval, a probability at
    /// least `confidence_level`.
    pub pointwise_confidence_level: f64,
    /// Two-sided max-`|t|` critical value that corresponds to the pointwise
    /// level.
    pub calibrated_max_t_critical_value: f64,
    /// Zero-based position of the functional in the result's points.
    pub functional_id: usize,
    /// Functional weights in post-period order.
    pub post_weights: Vec<f64>,
    /// Lower end of the least-favorable interval, run at the pointwise level.
    pub interval_low: f64,
    /// Upper end of the least-favorable interval, run at the pointwise level.
    pub interval_high: f64,
    /// Lower end of the Wald interval under exact parallel trends.
    pub original_ci_low: f64,
    /// Upper end of the Wald interval under exact parallel trends.
    pub original_ci_high: f64,
    /// Lower end of the identified set.
    pub identified_lb: f64,
    /// Upper end of the identified set.
    pub identified_ub: f64,
    /// Value that `robustly_significant` is tested against.
    pub null_value: f64,
    /// Whether the interval excludes `null_value`.
    pub robustly_significant: bool,
}

/// Joint-path report with its schema version.
#[derive(Debug, Clone, Serialize)]
pub struct HonestJointPathReport {
    /// Version label of the report schema, `honestdid_joint_report_v1`.
    pub schema_version: &'static str,
    /// Report rows.
    pub rows: Vec<HonestJointPathReportRow>,
}

/// Directional-region report with its schema version.
#[derive(Debug, Clone, Serialize)]
pub struct HonestDirectionalReport {
    /// Version label of the report schema, `honestdid_joint_report_v1`.
    pub schema_version: &'static str,
    /// Report rows.
    pub rows: Vec<HonestDirectionalReportRow>,
}

/// Simultaneous interval report with its schema version.
#[derive(Debug, Clone, Serialize)]
pub struct HonestMultiFlciReport {
    /// Version label of the report schema, `honestdid_joint_report_v1`.
    pub schema_version: &'static str,
    /// Report rows.
    pub rows: Vec<HonestMultiFlciReportRow>,
}

/// One report row per post period of a joint-path region, in the order of its
/// points.
#[must_use]
pub fn build_honest_joint_path_report_rows(
    region: &HonestJointPathRegion,
) -> Vec<HonestJointPathReportRow> {
    region
        .points
        .iter()
        .map(|point| HonestJointPathReportRow {
            schema_version: HONEST_REPORT_SCHEMA_V1,
            method: method_label(region.method),
            confidence_level: region.confidence_level,
            pointwise_confidence_level: region.pointwise_confidence_level,
            calibrated_max_t_critical_value: region.calibrated_max_t_critical_value,
            post_period: point.post_period,
            estimate: point.assessment.estimate,
            se: point.assessment.se,
            null_value: point.assessment.null_value,
            sensitivity_value: point.assessment.sensitivity_value,
            original_ci_low: point.assessment.original_ci.0,
            original_ci_high: point.assessment.original_ci.1,
            identified_lb: point.assessment.identified_set.0,
            identified_ub: point.assessment.identified_set.1,
            robust_ci_low: point.assessment.robust_ci.0,
            robust_ci_high: point.assessment.robust_ci.1,
            robustly_significant: point.assessment.robustly_significant,
        })
        .collect()
}

/// One report row per functional of a directional region, in the order of its
/// points.
#[must_use]
pub fn build_honest_directional_report_rows(
    region: &HonestDirectionalRegion,
) -> Vec<HonestDirectionalReportRow> {
    region
        .points
        .iter()
        .enumerate()
        .map(|(functional_id, point)| HonestDirectionalReportRow {
            schema_version: HONEST_REPORT_SCHEMA_V1,
            method: method_label(region.method),
            confidence_level: region.confidence_level,
            pointwise_confidence_level: region.pointwise_confidence_level,
            calibrated_max_t_critical_value: region.calibrated_max_t_critical_value,
            direction_count: region.diagnostics.direction_count,
            random_direction_count: region.diagnostics.random_direction_count,
            iterations_run: region.diagnostics.iterations_run,
            converged: region.diagnostics.converged,
            functional_id,
            post_weights: point.post_weights.clone(),
            estimate: point.assessment.estimate,
            se: point.assessment.se,
            null_value: point.assessment.null_value,
            sensitivity_value: point.assessment.sensitivity_value,
            original_ci_low: point.assessment.original_ci.0,
            original_ci_high: point.assessment.original_ci.1,
            identified_lb: point.assessment.identified_set.0,
            identified_ub: point.assessment.identified_set.1,
            robust_ci_low: point.assessment.robust_ci.0,
            robust_ci_high: point.assessment.robust_ci.1,
            robustly_significant: point.assessment.robustly_significant,
        })
        .collect()
}

/// One report row per functional of a simultaneous interval result, in the
/// order of its points.
#[must_use]
pub fn build_honest_multi_flci_report_rows(
    result: &HonestMultiFlciResult,
) -> Vec<HonestMultiFlciReportRow> {
    result
        .points
        .iter()
        .enumerate()
        .map(|(functional_id, point)| HonestMultiFlciReportRow {
            schema_version: HONEST_REPORT_SCHEMA_V1,
            method: method_label(result.method),
            confidence_level: result.confidence_level,
            pointwise_confidence_level: result.pointwise_confidence_level,
            calibrated_max_t_critical_value: result.calibrated_max_t_critical_value,
            functional_id,
            post_weights: point.post_weights.clone(),
            interval_low: point.flci.0,
            interval_high: point.flci.1,
            original_ci_low: point.original_ci.0,
            original_ci_high: point.original_ci.1,
            identified_lb: point.identified_set.0,
            identified_ub: point.identified_set.1,
            null_value: point.null_value,
            robustly_significant: point.robustly_significant,
        })
        .collect()
}

/// Renders the joint-path report rows as CSV text with a header line. Floating
/// point values have six decimals.
#[must_use]
pub fn render_honest_joint_path_report_csv(region: &HonestJointPathRegion) -> String {
    let rows = build_honest_joint_path_report_rows(region);
    let mut out = String::from(
        "schema_version,method,confidence_level,pointwise_confidence_level,calibrated_max_t_critical_value,post_period,estimate,se,null_value,sensitivity_value,original_ci_low,original_ci_high,identified_lb,identified_ub,robust_ci_low,robust_ci_high,robustly_significant\n",
    );
    for row in rows {
        let _ = writeln!(
            out,
            "{},{},{:.6},{:.6},{:.6},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
            row.schema_version,
            row.method,
            row.confidence_level,
            row.pointwise_confidence_level,
            row.calibrated_max_t_critical_value,
            row.post_period,
            row.estimate,
            row.se,
            row.null_value,
            row.sensitivity_value,
            row.original_ci_low,
            row.original_ci_high,
            row.identified_lb,
            row.identified_ub,
            row.robust_ci_low,
            row.robust_ci_high,
            row.robustly_significant
        );
    }
    out
}

/// Renders the directional-region report rows as CSV text with a header line.
/// Floating point values have six decimals. The `post_weights` column of the
/// rows is not included.
#[must_use]
pub fn render_honest_directional_report_csv(region: &HonestDirectionalRegion) -> String {
    let rows = build_honest_directional_report_rows(region);
    let mut out = String::from(
        "schema_version,method,confidence_level,pointwise_confidence_level,calibrated_max_t_critical_value,direction_count,random_direction_count,iterations_run,converged,functional_id,estimate,se,null_value,sensitivity_value,original_ci_low,original_ci_high,identified_lb,identified_ub,robust_ci_low,robust_ci_high,robustly_significant\n",
    );
    for row in rows {
        let _ = writeln!(
            out,
            "{},{},{:.6},{:.6},{:.6},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
            row.schema_version,
            row.method,
            row.confidence_level,
            row.pointwise_confidence_level,
            row.calibrated_max_t_critical_value,
            row.direction_count,
            row.random_direction_count,
            row.iterations_run,
            row.converged,
            row.functional_id,
            row.estimate,
            row.se,
            row.null_value,
            row.sensitivity_value,
            row.original_ci_low,
            row.original_ci_high,
            row.identified_lb,
            row.identified_ub,
            row.robust_ci_low,
            row.robust_ci_high,
            row.robustly_significant
        );
    }
    out
}

/// Renders the simultaneous interval report rows as CSV text with a header line.
/// Floating point values have six decimals. The `post_weights` column of the
/// rows is not included.
#[must_use]
pub fn render_honest_multi_flci_report_csv(result: &HonestMultiFlciResult) -> String {
    let rows = build_honest_multi_flci_report_rows(result);
    let mut out = String::from(
        "schema_version,method,confidence_level,pointwise_confidence_level,calibrated_max_t_critical_value,functional_id,interval_low,interval_high,original_ci_low,original_ci_high,identified_lb,identified_ub,null_value,robustly_significant\n",
    );
    for row in rows {
        let _ = writeln!(
            out,
            "{},{},{:.6},{:.6},{:.6},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
            row.schema_version,
            row.method,
            row.confidence_level,
            row.pointwise_confidence_level,
            row.calibrated_max_t_critical_value,
            row.functional_id,
            row.interval_low,
            row.interval_high,
            row.original_ci_low,
            row.original_ci_high,
            row.identified_lb,
            row.identified_ub,
            row.null_value,
            row.robustly_significant
        );
    }
    out
}

/// Render a JSON joint-path report payload.
///
/// # Errors
/// Returns an error if JSON serialization fails.
pub fn render_honest_joint_path_report_json(
    region: &HonestJointPathRegion,
) -> Result<String, String> {
    serde_json::to_string_pretty(&HonestJointPathReport {
        schema_version: HONEST_REPORT_SCHEMA_V1,
        rows: build_honest_joint_path_report_rows(region),
    })
    .map_err(|err| format!("failed to serialize joint path report json: {err}"))
}

/// Render a JSON directional report payload.
///
/// # Errors
/// Returns an error if JSON serialization fails.
pub fn render_honest_directional_report_json(
    region: &HonestDirectionalRegion,
) -> Result<String, String> {
    serde_json::to_string_pretty(&HonestDirectionalReport {
        schema_version: HONEST_REPORT_SCHEMA_V1,
        rows: build_honest_directional_report_rows(region),
    })
    .map_err(|err| format!("failed to serialize directional report json: {err}"))
}

/// Render a JSON multi-functional least-favorable interval report payload.
///
/// # Errors
/// Returns an error if JSON serialization fails.
pub fn render_honest_multi_flci_report_json(
    result: &HonestMultiFlciResult,
) -> Result<String, String> {
    serde_json::to_string_pretty(&HonestMultiFlciReport {
        schema_version: HONEST_REPORT_SCHEMA_V1,
        rows: build_honest_multi_flci_report_rows(result),
    })
    .map_err(|err| format!("failed to serialize honest multi-flci report json: {err}"))
}
