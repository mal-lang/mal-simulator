//! Rust port of `python/malsim/mal_simulator/graph_state.py` and the
//! graph-node-dependent half of `ttc_utils.py` (`default_ttc_dist`,
//! `TTCDist.from_node`, `attack_step_ttc_value(s)`,
//! `get_pre_enabled_defenses`, `get_impossible_attack_steps`) - see
//! `PORTING_NOTES.md` §5 Phase A3.
//!
//! Each graph-dependent function here is split into a thin
//! `AttackGraphNode`-reading wrapper plus a graph-independent helper that
//! does the actual logic over plain data (`&TtcDist`, `Option<&Value>`,
//! `&str`) - e.g. `attack_step_ttc_value` delegates to
//! `ttc_value_for_dist`, `resolve_ttc_dist` delegates to
//! `resolve_ttc_dist_from_parts`. This isn't a stylistic preference: it's
//! what makes the actual logic unit-testable without a real
//! `AttackGraphNode`, which (per `PORTING_NOTES.md` §10) needs a real
//! `maltoolbox_language::graph::LanguageGraph` to mint - a dev-dependency
//! this port deliberately didn't add yet. The thin wrappers themselves
//! are untested one-liners for the same reason.

use std::collections::{HashMap, HashSet};
use std::fmt;

use maltoolbox_attackgraph::{AttackGraph, AttackGraphNode, AttackGraphNodeId};
use rand::Rng;
use serde_json::Value;

use crate::necessity::{calculate_necessity, NecessityError};
use crate::ttc::{named_ttc_dist, TtcDist, TtcDistError};

/// Port of `config/sim_settings.py`'s `TTCMode`. Pulled forward from its
/// usual home (full `MalSimulatorSettings` porting is A9's job) because
/// `attack_step_ttc_value`'s mode dispatch needs it now.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TtcMode {
    EffortBasedPerStepSample,
    PerStepSample,
    PreSample,
    ExpectedValue,
    Disabled,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GraphStateError {
    TtcDist(TtcDistError),
    Necessity(NecessityError),
    /// Mirrors Python's `default_ttc_dist`'s `ValueError` for a
    /// `step_type` that isn't `defense`/`or`/`and`.
    NoDefaultTtcDist(String),
}

impl From<TtcDistError> for GraphStateError {
    fn from(e: TtcDistError) -> Self {
        GraphStateError::TtcDist(e)
    }
}

impl From<NecessityError> for GraphStateError {
    fn from(e: NecessityError) -> Self {
        GraphStateError::Necessity(e)
    }
}

impl fmt::Display for GraphStateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphStateError::TtcDist(e) => write!(f, "{e}"),
            GraphStateError::Necessity(e) => write!(f, "{e}"),
            GraphStateError::NoDefaultTtcDist(step_type) => write!(
                f,
                "can only get default TTC of defense and attack steps, not of \"{step_type}\" steps"
            ),
        }
    }
}

impl std::error::Error for GraphStateError {}

/// Port of `default_ttc_dist`. Takes `step_type.as_str()` rather than a
/// whole `AttackGraphNode` (Python's version only ever reads `.type` off
/// the node) - pure string dispatch, graph-independent.
pub fn default_ttc_dist_for_step_type(step_type: &str) -> Result<TtcDist, GraphStateError> {
    match step_type {
        "defense" => Ok(named_ttc_dist("Disabled").expect("'Disabled' is a known named dist")),
        "or" | "and" => Ok(named_ttc_dist("Instant").expect("'Instant' is a known named dist")),
        other => Err(GraphStateError::NoDefaultTtcDist(other.to_string())),
    }
}

/// Port of `TTCDist.from_node`'s logic, decomposed onto plain data
/// (`ttc_json`/`step_type`) instead of a whole `AttackGraphNode` - see
/// module docs for why.
pub fn resolve_ttc_dist_from_parts(
    ttc_json: Option<&Value>,
    step_type: &str,
    ttc_dist_override: Option<&TtcDist>,
) -> Result<TtcDist, GraphStateError> {
    if let Some(dist) = ttc_dist_override {
        return Ok(dist.clone());
    }
    match ttc_json {
        Some(value) => Ok(TtcDist::from_dict(value)?),
        None => default_ttc_dist_for_step_type(step_type),
    }
}

/// Thin `AttackGraphNode`-reading wrapper around
/// `resolve_ttc_dist_from_parts`.
pub fn resolve_ttc_dist(
    node: &AttackGraphNode,
    ttc_dist_override: Option<&TtcDist>,
) -> Result<TtcDist, GraphStateError> {
    resolve_ttc_dist_from_parts(
        node.ttc.as_ref(),
        node.step_type.as_str(),
        ttc_dist_override,
    )
}

/// Graph-independent half of `attack_step_ttc_value`: what to do with an
/// already-resolved `TtcDist` for a given `TtcMode`.
pub fn ttc_value_for_dist(
    ttc_dist: &TtcDist,
    ttc_mode: TtcMode,
    rng: &mut impl Rng,
) -> Option<f64> {
    match ttc_mode {
        TtcMode::ExpectedValue => Some(ttc_dist.expected_value()),
        TtcMode::PreSample => Some(ttc_dist.sample_value(rng)),
        TtcMode::EffortBasedPerStepSample | TtcMode::PerStepSample | TtcMode::Disabled => None,
    }
}

/// Port of `attack_step_ttc_value`.
pub fn attack_step_ttc_value(
    node: &AttackGraphNode,
    ttc_dist_override: Option<&TtcDist>,
    ttc_mode: TtcMode,
    rng: &mut impl Rng,
) -> Result<Option<f64>, GraphStateError> {
    let ttc_dist = resolve_ttc_dist(node, ttc_dist_override)?;
    Ok(ttc_value_for_dist(&ttc_dist, ttc_mode, rng))
}

/// Port of `attack_step_ttc_values`. Iterates `graph.attack_steps`
/// (Rust's precomputed equivalent of Python's `graph.attack_steps`
/// property).
pub fn attack_step_ttc_values(
    graph: &AttackGraph,
    rng: &mut impl Rng,
    ttc_mode: TtcMode,
    ttc_dist_overrides: Option<&HashMap<AttackGraphNodeId, TtcDist>>,
) -> Result<HashMap<AttackGraphNodeId, f64>, GraphStateError> {
    let mut values = HashMap::new();
    for &node_id in &graph.attack_steps {
        let node = &graph.nodes[node_id];
        let override_dist = ttc_dist_overrides.and_then(|m| m.get(&node_id));
        if let Some(value) = attack_step_ttc_value(node, override_dist, ttc_mode, rng)? {
            values.insert(node_id, value);
        }
    }
    Ok(values)
}

/// Graph-independent half of `get_pre_enabled_defenses`'s per-node body.
pub fn is_pre_enabled_for_dist(ttc_dist: &TtcDist, sample: bool, rng: &mut impl Rng) -> bool {
    let p0 = ttc_dist.success_probability(0);
    if p0 == 0.0 {
        // never succeeds -> pre enabled
        // TODO: is this correct?
        return true;
    }
    if p0 == 1.0 {
        // always succeeds -> not pre enabled
        return false;
    }
    sample && ttc_dist.attempt_bernoulli(rng)
}

/// Port of `get_pre_enabled_defenses`.
pub fn get_pre_enabled_defenses(
    graph: &AttackGraph,
    sample: bool,
    rng: &mut impl Rng,
) -> Result<HashSet<AttackGraphNodeId>, GraphStateError> {
    let mut pre_enabled = HashSet::new();
    for &node_id in &graph.defense_steps {
        let node = &graph.nodes[node_id];
        if node.step_type.as_str() != "defense" {
            continue;
        }
        let ttc_dist = resolve_ttc_dist(node, None)?;
        if is_pre_enabled_for_dist(&ttc_dist, sample, rng) {
            pre_enabled.insert(node_id);
        }
    }
    Ok(pre_enabled)
}

/// Graph-independent half of `is_impossible_attack_step`.
pub fn is_impossible_for_dist(ttc_dist: &TtcDist, rng: &mut impl Rng) -> bool {
    !ttc_dist.attempt_bernoulli(rng)
}

/// Port of `is_impossible_attack_step`.
pub fn is_impossible_attack_step(
    node: &AttackGraphNode,
    ttc_dist_override: Option<&TtcDist>,
    rng: &mut impl Rng,
) -> Result<bool, GraphStateError> {
    let ttc_dist = resolve_ttc_dist(node, ttc_dist_override)?;
    Ok(is_impossible_for_dist(&ttc_dist, rng))
}

/// Port of `get_impossible_attack_steps`.
pub fn get_impossible_attack_steps(
    graph: &AttackGraph,
    rng: &mut impl Rng,
    ttc_dist_overrides: Option<&HashMap<AttackGraphNodeId, TtcDist>>,
) -> Result<HashSet<AttackGraphNodeId>, GraphStateError> {
    let mut impossible = HashSet::new();
    for &node_id in &graph.attack_steps {
        let node = &graph.nodes[node_id];
        let override_dist = ttc_dist_overrides.and_then(|m| m.get(&node_id));
        if is_impossible_attack_step(node, override_dist, rng)? {
            impossible.insert(node_id);
        }
    }
    Ok(impossible)
}

/// Port of `GraphState`.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphState {
    pub ttc_values: HashMap<AttackGraphNodeId, f64>,
    pub pre_enabled_defenses: HashSet<AttackGraphNodeId>,
    pub impossible_attack_steps: HashSet<AttackGraphNodeId>,
    pub necessity_per_node: HashMap<AttackGraphNodeId, bool>,
}

/// Port of `compute_initial_graph_state`. Takes the three relevant
/// `MalSimulatorSettings` fields directly rather than the whole settings
/// struct - full `MalSimulatorSettings` porting is A9's job (settings
/// flattening across the FFI boundary, §2.4), and this function doesn't
/// need any of its other fields.
pub fn compute_initial_graph_state(
    graph: &AttackGraph,
    ttc_mode: TtcMode,
    run_defense_step_bernoullis: bool,
    run_attack_step_bernoullis: bool,
    rng: &mut impl Rng,
) -> Result<GraphState, GraphStateError> {
    let ttc_values = attack_step_ttc_values(graph, rng, ttc_mode, None)?;
    let pre_enabled_defenses = get_pre_enabled_defenses(graph, run_defense_step_bernoullis, rng)?;
    let impossible_attack_steps = if run_attack_step_bernoullis {
        get_impossible_attack_steps(graph, rng, None)?
    } else {
        HashSet::new()
    };
    let necessity_per_node = calculate_necessity(graph, &pre_enabled_defenses)?;

    Ok(GraphState {
        ttc_values,
        pre_enabled_defenses,
        impossible_attack_steps,
        necessity_per_node,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttc::{DistFunction, Operation};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn dist(function: DistFunction, args: &[f64]) -> TtcDist {
        TtcDist::new(function, args.to_vec()).unwrap()
    }

    // --- default_ttc_dist_for_step_type ---

    #[test]
    fn default_ttc_dist_defense_is_disabled() {
        assert_eq!(
            default_ttc_dist_for_step_type("defense").unwrap(),
            named_ttc_dist("Disabled").unwrap()
        );
    }

    #[test]
    fn default_ttc_dist_or_and_and_is_instant() {
        assert_eq!(
            default_ttc_dist_for_step_type("or").unwrap(),
            named_ttc_dist("Instant").unwrap()
        );
        assert_eq!(
            default_ttc_dist_for_step_type("and").unwrap(),
            named_ttc_dist("Instant").unwrap()
        );
    }

    #[test]
    fn default_ttc_dist_unsupported_step_type_errors() {
        for step_type in ["exist", "notExist", "something_else"] {
            let err = default_ttc_dist_for_step_type(step_type).unwrap_err();
            assert!(matches!(err, GraphStateError::NoDefaultTtcDist(_)));
        }
    }

    // --- resolve_ttc_dist_from_parts ---

    #[test]
    fn resolve_ttc_dist_override_wins_regardless_of_ttc_json_or_step_type() {
        let override_dist = dist(DistFunction::Exponential, &[0.5]);
        let resolved = resolve_ttc_dist_from_parts(None, "defense", Some(&override_dist)).unwrap();
        assert_eq!(resolved, override_dist);

        let ttc_json = dist(DistFunction::Uniform, &[1.0, 2.0]).to_dict();
        let resolved =
            resolve_ttc_dist_from_parts(Some(&ttc_json), "or", Some(&override_dist)).unwrap();
        assert_eq!(resolved, override_dist);
    }

    #[test]
    fn resolve_ttc_dist_parses_explicit_ttc_json() {
        let explicit = dist(DistFunction::Binomial, &[10.0, 0.1])
            .with_combine(dist(DistFunction::Bernoulli, &[0.5]), Operation::Addition);
        let ttc_json = explicit.to_dict();
        let resolved = resolve_ttc_dist_from_parts(Some(&ttc_json), "or", None).unwrap();
        assert_eq!(resolved, explicit);
    }

    #[test]
    fn resolve_ttc_dist_falls_back_to_default_when_no_ttc_json() {
        let resolved = resolve_ttc_dist_from_parts(None, "defense", None).unwrap();
        assert_eq!(resolved, named_ttc_dist("Disabled").unwrap());

        let err = resolve_ttc_dist_from_parts(None, "exist", None).unwrap_err();
        assert!(matches!(err, GraphStateError::NoDefaultTtcDist(_)));
    }

    // --- ttc_value_for_dist ---

    #[test]
    fn ttc_value_expected_value_mode_returns_expected_value() {
        let d = dist(DistFunction::Exponential, &[0.1]);
        let mut rng = StdRng::seed_from_u64(0);
        assert_eq!(
            ttc_value_for_dist(&d, TtcMode::ExpectedValue, &mut rng),
            Some(d.expected_value())
        );
    }

    #[test]
    fn ttc_value_pre_sample_mode_matches_direct_sample_value_call() {
        let d = dist(DistFunction::Exponential, &[0.1]);
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(42);
        assert_eq!(
            ttc_value_for_dist(&d, TtcMode::PreSample, &mut rng_a),
            Some(d.sample_value(&mut rng_b))
        );
    }

    #[test]
    fn ttc_value_other_modes_return_none() {
        let d = dist(DistFunction::Exponential, &[0.1]);
        let mut rng = StdRng::seed_from_u64(0);
        for mode in [
            TtcMode::Disabled,
            TtcMode::PerStepSample,
            TtcMode::EffortBasedPerStepSample,
        ] {
            assert_eq!(ttc_value_for_dist(&d, mode, &mut rng), None);
        }
    }

    // --- is_pre_enabled_for_dist ---

    // Note the apparent inversion here: `success_probability(0)` is
    // `dist.cdf(0)`, which for `Bernoulli(p)` is `1 - p` - *not* `p`
    // itself. So the *named* "Disabled" dist (`Bernoulli(0.0)`) has
    // `cdf(0) == 1.0` (the "always succeeds" branch -> not pre-enabled),
    // and "Enabled" (`Bernoulli(1.0)`) has `cdf(0) == 0.0` (the "never
    // succeeds" branch -> pre-enabled). This matches Python's
    // `get_pre_enabled_defenses` exactly, including its own "TODO: is
    // this correct?" comment on this branch - ported as-is, not "fixed".

    #[test]
    fn pre_enabled_degenerate_disabled_dist_is_never_pre_enabled() {
        let d = named_ttc_dist("Disabled").unwrap(); // Bernoulli(0.0), cdf(0) == 1.0
        let mut rng = StdRng::seed_from_u64(0);
        assert!(!is_pre_enabled_for_dist(&d, false, &mut rng));
        assert!(!is_pre_enabled_for_dist(&d, true, &mut rng));
    }

    #[test]
    fn pre_enabled_degenerate_enabled_dist_is_always_pre_enabled() {
        let d = named_ttc_dist("Enabled").unwrap(); // Bernoulli(1.0), cdf(0) == 0.0
        let mut rng = StdRng::seed_from_u64(0);
        assert!(is_pre_enabled_for_dist(&d, false, &mut rng));
        assert!(is_pre_enabled_for_dist(&d, true, &mut rng));
    }

    #[test]
    fn pre_enabled_non_degenerate_without_sampling_is_never_pre_enabled() {
        let d = dist(DistFunction::Bernoulli, &[0.5]);
        let mut rng = StdRng::seed_from_u64(0);
        assert!(!is_pre_enabled_for_dist(&d, false, &mut rng));
    }

    #[test]
    fn pre_enabled_non_degenerate_with_sampling_matches_attempt_bernoulli() {
        let d = dist(DistFunction::Bernoulli, &[0.5]);
        let mut rng_a = StdRng::seed_from_u64(7);
        let mut rng_b = StdRng::seed_from_u64(7);
        assert_eq!(
            is_pre_enabled_for_dist(&d, true, &mut rng_a),
            d.attempt_bernoulli(&mut rng_b)
        );
    }

    // --- is_impossible_for_dist ---

    #[test]
    fn impossible_is_negated_attempt_bernoulli() {
        let d = dist(DistFunction::Bernoulli, &[0.5]);
        let mut rng_a = StdRng::seed_from_u64(3);
        let mut rng_b = StdRng::seed_from_u64(3);
        assert_eq!(
            is_impossible_for_dist(&d, &mut rng_a),
            !d.attempt_bernoulli(&mut rng_b)
        );
    }

    #[test]
    fn impossible_without_bernoulli_is_never_impossible() {
        // attempt_bernoulli unconditionally succeeds when there's no
        // Bernoulli anywhere in the dist (see ttc.rs) - so a plain
        // Exponential can never be "impossible" regardless of rng draws.
        let d = dist(DistFunction::Exponential, &[0.1]);
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..10 {
            assert!(!is_impossible_for_dist(&d, &mut rng));
        }
    }
}
