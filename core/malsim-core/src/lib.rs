//! Rust core of `malsim`, the MAL simulator. Depends on `maltoolbox-attackgraph`
//! (from the `mal-toolbox` `rust-rewrite` branch) for the attack graph types
//! this simulator steps through.

pub mod attack_surface;
pub mod attacker_step;
pub mod defender_step;
pub mod defense_surface;
pub mod event_logger;
pub mod false_alerts;
pub mod graph_state;
pub mod graph_utils;
pub mod necessity;
pub mod observability;
pub mod ttc;

#[cfg(test)]
mod test_fixtures;
