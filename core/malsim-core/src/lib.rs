//! Rust core of `malsim`, the MAL simulator. Depends on `maltoolbox-attackgraph`
//! (from the `mal-toolbox` `rust-rewrite` branch) for the attack graph types
//! this simulator steps through.

pub mod attack_surface;
pub mod defense_surface;
pub mod graph_state;
pub mod graph_utils;
pub mod necessity;
pub mod ttc;

#[cfg(test)]
mod test_fixtures;
