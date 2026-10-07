//! Rust core of `malsim`, the MAL simulator. Depends on `maltoolbox-attackgraph`
//! (from the `mal-toolbox` `rust-rewrite` branch) for the attack graph types
//! this simulator steps through.

pub mod graph_state;
pub mod necessity;
pub mod ttc;
