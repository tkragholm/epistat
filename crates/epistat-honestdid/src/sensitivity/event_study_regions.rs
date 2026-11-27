#[path = "event_study_directional_region.rs"]
mod directional_region;
#[path = "event_study_joint_path.rs"]
mod joint_path;
#[path = "event_study_optimization_surface.rs"]
mod optimization_surface;

pub use directional_region::assess_honest_event_study_directional_region_with_config;
#[cfg(test)]
pub(in crate::sensitivity::event_study_assessment) use directional_region::assess_honest_event_study_directional_region_with_optional_prepared_relative_magnitude;
#[cfg(test)]
pub use joint_path::assess_honest_event_study_joint_path_region;
pub use joint_path::assess_honest_event_study_joint_path_region_with_config;
#[cfg(test)]
pub(in crate::sensitivity::event_study_assessment) use joint_path::assess_honest_event_study_joint_path_region_with_optional_prepared_relative_magnitude;
pub use optimization_surface::{
    assess_honest_event_study_optimization_surface_region_adaptive_with_config,
    assess_honest_event_study_optimization_surface_region_with_config,
};

#[cfg(test)]
pub(in crate::sensitivity) use optimization_surface::assess_honest_event_study_optimization_surface_region_adaptive_with_config_counted;
