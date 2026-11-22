use chrono::NaiveDate;
use std::collections::{HashMap, HashSet};

/// An episode of one person on one index date.
/// ///
/// /// An exposed episode is a case and an unexposed one a control.
#[derive(Debug, Clone)]
pub struct ReconstructedEpisode {
    /// The person.
    pub person_id: String,
    /// The index date of the episode.
    pub index_date: NaiveDate,
    /// Whether the person is exposed on the index date.
    pub exposed: bool,
    /// The sex code, if known.
    pub sex: Option<i32>,
    /// The municipality, if known.
    pub municipality: Option<String>,
    /// The region, if known.
    pub region: Option<String>,
    /// The family, if known. A case and a control of one family are not matched.
    pub family_id: Option<String>,
    /// The birth date, if known.
    pub birth_date: Option<NaiveDate>,
}

/// The geographical level at which a pair was matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconstructedMatchingTier {
    /// The case and the control live in the same municipality.
    Municipality,
    /// The case and the control live in the same region but not in the same municipality.
    RegionFallback,
}

/// One control matched to one case.
#[derive(Debug, Clone)]
pub struct ReconstructedMatchedPair {
    /// The case.
    pub case_id: String,
    /// The control.
    pub control_id: String,
    /// The index date shared by the two.
    pub index_date: NaiveDate,
    /// The level at which the pair was matched.
    pub tier: ReconstructedMatchingTier,
    /// The difference in birth dates in days, when both are known.
    pub birth_date_distance_days: Option<i64>,
}

/// Options of [`reconstruct_case_control_pairs`].
#[derive(Debug, Clone, Copy)]
pub struct ReconstructedMatchingOptions {
    /// Most controls one case can take. At least 1.
    pub max_ratio: usize,
    /// Largest birth-date difference in days between a case and a control, or
    /// `None` for no limit. Not negative. A pair with an unknown birth date
    /// has an unknown difference and does not meet a limit.
    pub match_caliper_days: Option<i64>,
}

/// Invalid options of [`reconstruct_case_control_pairs`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconstructedMatchingError {
    /// `max_ratio` is 0.
    ZeroMaxRatio,
    /// `match_caliper_days` is negative.
    NegativeCaliper(i64),
}

impl std::fmt::Display for ReconstructedMatchingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroMaxRatio => f.write_str("max_ratio must be at least 1"),
            Self::NegativeCaliper(days) => {
                write!(f, "match_caliper_days is {days}, which is negative")
            }
        }
    }
}

impl std::error::Error for ReconstructedMatchingError {}

/// The result of [`reconstruct_case_control_pairs`].
#[derive(Debug, Default)]
pub struct ReconstructedMatchingOutput {
    /// The cases that could be given controls.
    pub eligible_case_ids: HashSet<String>,
    /// The cases with at least one control.
    pub matched_case_ids: HashSet<String>,
    /// The controls matched to a case.
    pub matched_control_ids: HashSet<String>,
    /// The cases that have a control matched by municipality.
    pub municipality_case_ids: HashSet<String>,
    /// The controls matched by municipality.
    pub municipality_control_ids: HashSet<String>,
    /// The cases that have a control matched by region.
    pub region_case_ids: HashSet<String>,
    /// The controls matched by region.
    pub region_control_ids: HashSet<String>,
    /// Every pair, in the order the controls were read.
    pub matched_pairs: Vec<ReconstructedMatchedPair>,
}

#[derive(Clone, Copy)]
struct CaseState<'a> {
    row: &'a ReconstructedEpisode,
    assigned: usize,
}

type CaseGroupIndex = HashMap<(NaiveDate, i32), HashMap<String, Vec<usize>>>;

/// Match each unexposed episode to an exposed one on the same index date and
/// sex, by municipality and then by region.
///
/// # Errors
///
/// [`ReconstructedMatchingError`] for a `max_ratio` of 0 or a negative
/// caliper.
pub fn reconstruct_case_control_pairs(
    rows: &[ReconstructedEpisode],
    options: ReconstructedMatchingOptions,
) -> Result<ReconstructedMatchingOutput, ReconstructedMatchingError> {
    if options.max_ratio == 0 {
        return Err(ReconstructedMatchingError::ZeroMaxRatio);
    }
    if let Some(days) = options.match_caliper_days.filter(|days| *days < 0) {
        return Err(ReconstructedMatchingError::NegativeCaliper(days));
    }
    let max_ratio = options.max_ratio;
    let caliper_days = options.match_caliper_days;

    let mut case_states = dedup_case_states(rows);
    if case_states.is_empty() {
        return Ok(ReconstructedMatchingOutput::default());
    }

    let (municipality_case_index, region_case_index) = build_case_indices(&case_states);

    let mut output = ReconstructedMatchingOutput {
        eligible_case_ids: case_states
            .iter()
            .map(|case| case.row.person_id.clone())
            .collect(),
        ..ReconstructedMatchingOutput::default()
    };

    for control_row in rows.iter().filter(|row| !row.exposed) {
        let selected = select_case_for_control(
            control_row,
            &case_states,
            &municipality_case_index,
            &region_case_index,
            caliper_days,
            max_ratio,
        );

        if let Some((case_idx, tier)) = selected {
            case_states[case_idx].assigned = case_states[case_idx].assigned.saturating_add(1);
            let case_row = case_states[case_idx].row;
            record_reconstructed_match(&mut output, case_row, control_row, tier);
        }
    }

    Ok(output)
}

fn dedup_case_states(rows: &[ReconstructedEpisode]) -> Vec<CaseState<'_>> {
    let mut case_states = Vec::<CaseState<'_>>::new();
    let mut seen_case_episodes = HashSet::<(&str, NaiveDate)>::new();
    for row in rows.iter().filter(|row| row.exposed) {
        let key = (row.person_id.as_str(), row.index_date);
        if seen_case_episodes.insert(key) {
            case_states.push(CaseState { row, assigned: 0 });
        }
    }
    case_states
}

fn build_case_indices(case_states: &[CaseState<'_>]) -> (CaseGroupIndex, CaseGroupIndex) {
    let mut municipality_case_index = CaseGroupIndex::new();
    let mut region_case_index = CaseGroupIndex::new();
    for (case_idx, case) in case_states.iter().enumerate() {
        let sex_key = case.row.sex.unwrap_or(-1);
        if let Some(municipality) = normalized_group_value(case.row.municipality.as_deref()) {
            municipality_case_index
                .entry((case.row.index_date, sex_key))
                .or_default()
                .entry(municipality.to_string())
                .or_default()
                .push(case_idx);
        }
        if let Some(region) = normalized_group_value(case.row.region.as_deref()) {
            region_case_index
                .entry((case.row.index_date, sex_key))
                .or_default()
                .entry(region.to_string())
                .or_default()
                .push(case_idx);
        }
    }
    (municipality_case_index, region_case_index)
}

fn select_case_for_control(
    control_row: &ReconstructedEpisode,
    case_states: &[CaseState<'_>],
    municipality_case_index: &CaseGroupIndex,
    region_case_index: &CaseGroupIndex,
    caliper_days: Option<i64>,
    max_ratio: usize,
) -> Option<(usize, ReconstructedMatchingTier)> {
    let sex_key = control_row.sex.unwrap_or(-1);
    normalized_group_value(control_row.municipality.as_deref())
        .and_then(|municipality| {
            municipality_case_index
                .get(&(control_row.index_date, sex_key))
                .and_then(|index| index.get(municipality))
                .and_then(|candidate_indices| {
                    select_best_case(
                        case_states,
                        candidate_indices,
                        control_row,
                        caliper_days,
                        max_ratio,
                    )
                })
                .map(|case_idx| (case_idx, ReconstructedMatchingTier::Municipality))
        })
        .or_else(|| {
            normalized_group_value(control_row.region.as_deref()).and_then(|region| {
                region_case_index
                    .get(&(control_row.index_date, sex_key))
                    .and_then(|index| index.get(region))
                    .and_then(|candidate_indices| {
                        select_best_case(
                            case_states,
                            candidate_indices,
                            control_row,
                            caliper_days,
                            max_ratio,
                        )
                    })
                    .map(|case_idx| (case_idx, ReconstructedMatchingTier::RegionFallback))
            })
        })
}

fn record_reconstructed_match(
    output: &mut ReconstructedMatchingOutput,
    case_row: &ReconstructedEpisode,
    control_row: &ReconstructedEpisode,
    tier: ReconstructedMatchingTier,
) {
    output.matched_case_ids.insert(case_row.person_id.clone());
    output
        .matched_control_ids
        .insert(control_row.person_id.clone());

    match tier {
        ReconstructedMatchingTier::Municipality => {
            output
                .municipality_case_ids
                .insert(case_row.person_id.clone());
            output
                .municipality_control_ids
                .insert(control_row.person_id.clone());
        }
        ReconstructedMatchingTier::RegionFallback => {
            output.region_case_ids.insert(case_row.person_id.clone());
            output
                .region_control_ids
                .insert(control_row.person_id.clone());
        }
    }

    output.matched_pairs.push(ReconstructedMatchedPair {
        case_id: case_row.person_id.clone(),
        control_id: control_row.person_id.clone(),
        index_date: case_row.index_date,
        tier,
        birth_date_distance_days: birth_date_distance_days(case_row, control_row),
    });
}

fn select_best_case(
    case_states: &[CaseState<'_>],
    candidate_indices: &[usize],
    control_row: &ReconstructedEpisode,
    caliper_days: Option<i64>,
    max_ratio: usize,
) -> Option<usize> {
    let mut selected: Option<(usize, usize, i64, &str)> = None;
    for case_idx in candidate_indices {
        let Some(case_state) = case_states.get(*case_idx) else {
            continue;
        };
        if case_state.assigned >= max_ratio {
            continue;
        }
        let case_row = case_state.row;
        if !case_control_eligible(case_row, control_row, caliper_days) {
            continue;
        }
        let distance_days = birth_date_distance_days(case_row, control_row).unwrap_or(i64::MAX);
        let candidate = (
            *case_idx,
            case_state.assigned,
            distance_days,
            case_row.person_id.as_str(),
        );
        match selected {
            None => selected = Some(candidate),
            Some(current) => {
                if candidate.1 < current.1
                    || (candidate.1 == current.1 && candidate.2 < current.2)
                    || (candidate.1 == current.1
                        && candidate.2 == current.2
                        && candidate.3 < current.3)
                {
                    selected = Some(candidate);
                }
            }
        }
    }
    selected.map(|choice| choice.0)
}

fn case_control_eligible(
    case_row: &ReconstructedEpisode,
    control_row: &ReconstructedEpisode,
    caliper_days: Option<i64>,
) -> bool {
    if case_row.person_id == control_row.person_id {
        return false;
    }
    let case_family = normalized_group_value(case_row.family_id.as_deref());
    let control_family = normalized_group_value(control_row.family_id.as_deref());
    if case_family.is_some() && case_family == control_family {
        return false;
    }
    caliper_days.is_none_or(|caliper| {
        birth_date_distance_days(case_row, control_row).is_some_and(|days| days <= caliper)
    })
}

fn birth_date_distance_days(
    case_row: &ReconstructedEpisode,
    control_row: &ReconstructedEpisode,
) -> Option<i64> {
    Some(
        case_row
            .birth_date?
            .signed_duration_since(control_row.birth_date?)
            .num_days()
            .abs(),
    )
}

fn normalized_group_value(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::{
        ReconstructedEpisode, ReconstructedMatchingError, ReconstructedMatchingOptions,
        ReconstructedMatchingTier, reconstruct_case_control_pairs,
    };
    use chrono::NaiveDate;

    fn d(year: i32, month: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(year, month, day).expect("hardcoded valid date")
    }

    #[test]
    fn invalid_options_are_errors() {
        let options = |max_ratio, match_caliper_days| ReconstructedMatchingOptions {
            max_ratio,
            match_caliper_days,
        };
        assert_eq!(
            reconstruct_case_control_pairs(&[], options(0, None)).unwrap_err(),
            ReconstructedMatchingError::ZeroMaxRatio
        );
        assert_eq!(
            reconstruct_case_control_pairs(&[], options(1, Some(-1))).unwrap_err(),
            ReconstructedMatchingError::NegativeCaliper(-1)
        );
    }

    /// A caliper of 0 allows the same birth date only, and a pair with an
    /// unknown birth date does not meet any caliper.
    #[test]
    fn a_caliper_of_zero_is_not_no_caliper() {
        let row = |id: &str, exposed, birth: Option<NaiveDate>| ReconstructedEpisode {
            person_id: id.to_string(),
            index_date: d(2021, 1, 1),
            exposed,
            sex: Some(1),
            municipality: Some("0101".to_string()),
            region: Some("H".to_string()),
            family_id: None,
            birth_date: birth,
        };
        let rows = vec![
            row("case", true, Some(d(2010, 1, 1))),
            row("same", false, Some(d(2010, 1, 1))),
            row("other", false, Some(d(2010, 1, 2))),
            row("unknown", false, None),
        ];
        let run = |caliper| {
            reconstruct_case_control_pairs(
                &rows,
                ReconstructedMatchingOptions {
                    max_ratio: 3,
                    match_caliper_days: caliper,
                },
            )
            .unwrap()
            .matched_control_ids
            .len()
        };
        assert_eq!(run(Some(0)), 1);
        assert_eq!(run(None), 3);
    }

    #[test]
    fn reconstructs_pairs_with_municipality_then_region_fallback() {
        let rows = vec![
            ReconstructedEpisode {
                person_id: "case-a".to_string(),
                index_date: d(2021, 1, 1),
                exposed: true,
                sex: Some(1),
                municipality: Some("0101".to_string()),
                region: Some("H".to_string()),
                family_id: Some("fam-a".to_string()),
                birth_date: Some(d(2010, 1, 1)),
            },
            ReconstructedEpisode {
                person_id: "ctrl-a".to_string(),
                index_date: d(2021, 1, 1),
                exposed: false,
                sex: Some(1),
                municipality: Some("0101".to_string()),
                region: Some("H".to_string()),
                family_id: Some("fam-z".to_string()),
                birth_date: Some(d(2010, 1, 2)),
            },
            ReconstructedEpisode {
                person_id: "ctrl-b".to_string(),
                index_date: d(2021, 1, 1),
                exposed: false,
                sex: Some(1),
                municipality: Some("9999".to_string()),
                region: Some("H".to_string()),
                family_id: Some("fam-y".to_string()),
                birth_date: Some(d(2010, 1, 3)),
            },
        ];
        let output = reconstruct_case_control_pairs(
            &rows,
            ReconstructedMatchingOptions {
                max_ratio: 2,
                match_caliper_days: Some(31),
            },
        )
        .unwrap();
        assert_eq!(output.matched_pairs.len(), 2);
        assert!(
            output
                .matched_pairs
                .iter()
                .any(|pair| pair.tier == ReconstructedMatchingTier::Municipality)
        );
        assert!(
            output
                .matched_pairs
                .iter()
                .any(|pair| pair.tier == ReconstructedMatchingTier::RegionFallback)
        );
    }
}
