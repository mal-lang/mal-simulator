//! Rust port of `python/malsim/mal_simulator/attack_surface.py` - see
//! `PORTING_NOTES.md` §5 Phase A5.
//!
//! `node_is_actionable` (`graph_utils.py`) stays Python per §2.4 - it
//! operates on a `NodePropertyRule` directly. Per A5's own description,
//! this module takes the *already-flattened* actionability id-set
//! instead: `actionable_steps: None` means "no rule configured" (every
//! node actionable, mirroring `node_is_actionable`'s `if agent_actionability`
//! being falsy), `Some(set)` means exactly the ids in `set` are actionable
//! (mirroring a flattened `NodePropertyRule.per_node()` result, which
//! already baked in its own per-node default). This module never
//! constructs or reads a `NodePropertyRule`.

use std::collections::{HashSet, VecDeque};

use maltoolbox_attackgraph::{AttackGraph, AttackGraphNodeId};

use crate::graph_utils::{node_is_live, node_is_necessary, node_is_traversable, GraphUtilsError};

/// Port of `node_is_actionable`, operating on the already-flattened id-set
/// instead of a `NodePropertyRule` - see module docs.
fn node_is_actionable_flat(
    actionable_steps: Option<&HashSet<AttackGraphNodeId>>,
    node_id: AttackGraphNodeId,
) -> bool {
    match actionable_steps {
        Some(steps) => steps.contains(&node_id),
        None => true,
    }
}

fn causal_mode_is_effect(graph: &AttackGraph, node_id: AttackGraphNodeId) -> bool {
    graph.nodes[node_id]
        .causal_mode
        .is_some_and(|m| m.as_str() == "effect")
}

/// Port of `get_effects_of_attack_step`.
///
/// Python recomputes `has_visited = performed | set(effects)` fresh every
/// loop iteration; here `visited` is instead grown incrementally (starting
/// from `performed_nodes` + `attack_step_id`, gaining each newly-found
/// effect immediately) - `effects` only ever grows and nothing is ever
/// removed from `performed_nodes`, so at every check point `visited` holds
/// exactly the same members the Python union would have recomputed. Purely
/// an idiomatic/efficiency difference (no repeated set-union allocation per
/// iteration), not a behavior change - see `PORTING_NOTES.md` §2.7.
pub fn get_effects_of_attack_step(
    graph: &AttackGraph,
    attack_step_id: AttackGraphNodeId,
    performed_nodes: &HashSet<AttackGraphNodeId>,
    impossible_attack_steps: &HashSet<AttackGraphNodeId>,
    enabled_defenses: &HashSet<AttackGraphNodeId>,
    necessity_per_node: &std::collections::HashMap<AttackGraphNodeId, bool>,
) -> Result<HashSet<AttackGraphNodeId>, GraphUtilsError> {
    let mut effects: HashSet<AttackGraphNodeId> = HashSet::new();
    if !node_is_live(graph, attack_step_id) {
        // Only consider nodes that still exist in the attack graph.
        return Ok(effects);
    }

    let mut visited: HashSet<AttackGraphNodeId> = performed_nodes.clone();
    visited.insert(attack_step_id);

    let mut potential_effects: VecDeque<AttackGraphNodeId> = graph.nodes[attack_step_id]
        .children
        .iter()
        .copied()
        .filter(|&child_id| causal_mode_is_effect(graph, child_id))
        .collect();

    while let Some(effect_id) = potential_effects.pop_front() {
        if visited.contains(&effect_id) {
            continue;
        }
        if node_is_traversable(
            graph,
            effect_id,
            &visited,
            impossible_attack_steps,
            enabled_defenses,
            necessity_per_node,
        )? {
            visited.insert(effect_id);
            effects.insert(effect_id);
            potential_effects.extend(
                graph.nodes[effect_id]
                    .children
                    .iter()
                    .copied()
                    .filter(|&child_id| causal_mode_is_effect(graph, child_id)),
            );
        }
    }

    Ok(effects)
}

/// Port of `get_attack_surface`. Takes `skip_compromised`/`skip_unnecessary`
/// directly (the two fields of `AttackSurfaceSettings` this function reads)
/// rather than a ported settings struct - full settings porting is A9's
/// job, same pattern as `graph_state::compute_initial_graph_state`. The
/// resulting flat argument list (mirroring the independent pieces of
/// `MalSimulatorState`/`AttackSurfaceSettings` the Python function reads
/// off `sim_state`/`settings`) trips clippy's default arg-count lint;
/// bundling three of them into a one-off struct for this function alone
/// would be an abstraction with no other caller, so the lint is silenced
/// here instead.
#[allow(clippy::too_many_arguments)]
pub fn get_attack_surface(
    graph: &AttackGraph,
    skip_compromised: bool,
    skip_unnecessary: bool,
    actionable_steps: Option<&HashSet<AttackGraphNodeId>>,
    performed_nodes: &HashSet<AttackGraphNodeId>,
    from_nodes: Option<&HashSet<AttackGraphNodeId>>,
    impossible_attack_steps: &HashSet<AttackGraphNodeId>,
    enabled_defenses: &HashSet<AttackGraphNodeId>,
    necessity_per_node: &std::collections::HashMap<AttackGraphNodeId, bool>,
) -> Result<HashSet<AttackGraphNodeId>, GraphUtilsError> {
    let from_nodes = from_nodes.unwrap_or(performed_nodes);

    // A performed node that deleted its own backing asset (a dyna-MAL
    // model effect targeting `self`) is no longer live: the core always
    // keeps a *live* node's children/parents correctly unlinked from
    // anything removed, but a dead node's own `.children` is only the
    // frozen pre-removal snapshot mal-toolbox keeps so `performed_nodes`
    // can still hold a readable reference to it - never a source of new
    // live actions.
    let mut from_node_children: HashSet<AttackGraphNodeId> = HashSet::new();
    for &parent_id in from_nodes {
        if !node_is_live(graph, parent_id) {
            continue;
        }
        from_node_children.extend(graph.nodes[parent_id].children.iter().copied());
    }

    let mut surface = HashSet::new();
    for node_id in from_node_children {
        // Nodes marked as effects are not actions/attacks.
        if causal_mode_is_effect(graph, node_id) {
            continue;
        }
        if skip_compromised && performed_nodes.contains(&node_id) {
            continue;
        }
        if skip_unnecessary && !node_is_necessary(necessity_per_node, node_id)? {
            continue;
        }
        if !node_is_actionable_flat(actionable_steps, node_id) {
            continue;
        }
        if !node_is_traversable(
            graph,
            node_id,
            performed_nodes,
            impossible_attack_steps,
            enabled_defenses,
            necessity_per_node,
        )? {
            continue;
        }
        surface.insert(node_id);
    }

    Ok(surface)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::{add_dummy_node, dummy_graph};
    use maltoolbox_language::graph::attack_step::CausalMode;
    use std::collections::HashMap;

    fn mark_effect(graph: &mut AttackGraph, node_id: AttackGraphNodeId) {
        graph.nodes[node_id].causal_mode = Some(CausalMode::Effect);
    }

    // --- get_effects_of_attack_step ---

    #[test]
    fn effects_dead_attack_step_returns_empty() {
        let mut graph = dummy_graph();
        let step = add_dummy_node(&mut graph, "DummyOrAttackStep");
        graph.remove_node(step).unwrap();

        let empty = HashSet::new();
        let empty_map = HashMap::new();
        let effects =
            get_effects_of_attack_step(&graph, step, &empty, &empty, &empty, &empty_map).unwrap();
        assert!(effects.is_empty());
    }

    #[test]
    fn effects_follow_chain_of_effect_children_only() {
        let mut graph = dummy_graph();
        let step = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let effect1 = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let effect2 = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let non_effect_child = add_dummy_node(&mut graph, "DummyOrAttackStep");

        mark_effect(&mut graph, effect1);
        mark_effect(&mut graph, effect2);
        // non_effect_child keeps the default (action) causal mode.

        graph.nodes[step].children.insert(effect1);
        graph.nodes[step].children.insert(non_effect_child);
        graph.nodes[effect1].parents.insert(step);
        graph.nodes[non_effect_child].parents.insert(step);

        graph.nodes[effect1].children.insert(effect2);
        graph.nodes[effect2].parents.insert(effect1);

        let empty = HashSet::new();
        let empty_map = HashMap::new();
        let effects =
            get_effects_of_attack_step(&graph, step, &empty, &empty, &empty, &empty_map).unwrap();

        // Only the effect-tagged descendants are collected, transitively -
        // the action-mode child is never followed even though it's a
        // child of `step`.
        assert_eq!(effects, [effect1, effect2].into_iter().collect());
    }

    #[test]
    fn effects_stop_at_already_visited_node() {
        let mut graph = dummy_graph();
        let step = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let effect = add_dummy_node(&mut graph, "DummyOrAttackStep");
        mark_effect(&mut graph, effect);

        graph.nodes[step].children.insert(effect);
        graph.nodes[effect].parents.insert(step);

        let performed: HashSet<_> = [effect].into_iter().collect();
        let empty_map = HashMap::new();
        let empty = HashSet::new();

        // `effect` is already performed, so it must not be re-added even
        // though it's reachable as an effect child of `step`.
        let effects =
            get_effects_of_attack_step(&graph, step, &performed, &empty, &empty, &empty_map)
                .unwrap();
        assert!(effects.is_empty());
    }

    #[test]
    fn effects_do_not_cross_blocked_and_step() {
        let mut graph = dummy_graph();
        let step = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let other_parent = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let effect = add_dummy_node(&mut graph, "DummyAndAttackStep");
        mark_effect(&mut graph, effect);

        graph.nodes[step].children.insert(effect);
        graph.nodes[effect].parents.insert(step);
        // `effect` is an `and` step with a second, unperformed necessary
        // parent - not traversable yet, so it must not be collected.
        graph.nodes[effect].parents.insert(other_parent);

        let necessity: HashMap<_, _> = [(step, true), (other_parent, true)].into_iter().collect();
        let empty = HashSet::new();

        let effects =
            get_effects_of_attack_step(&graph, step, &empty, &empty, &empty, &necessity).unwrap();
        assert!(effects.is_empty());
    }

    // --- get_attack_surface ---

    fn surface(
        graph: &AttackGraph,
        skip_compromised: bool,
        skip_unnecessary: bool,
        actionable_steps: Option<&HashSet<AttackGraphNodeId>>,
        performed_nodes: &HashSet<AttackGraphNodeId>,
        necessity: &HashMap<AttackGraphNodeId, bool>,
    ) -> HashSet<AttackGraphNodeId> {
        let empty = HashSet::new();
        get_attack_surface(
            graph,
            skip_compromised,
            skip_unnecessary,
            actionable_steps,
            performed_nodes,
            None,
            &empty,
            &empty,
            necessity,
        )
        .unwrap()
    }

    #[test]
    fn attack_surface_includes_traversable_action_children_of_performed_nodes() {
        let mut graph = dummy_graph();
        let performed_node = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let child = add_dummy_node(&mut graph, "DummyOrAttackStep");
        graph.nodes[performed_node].children.insert(child);
        graph.nodes[child].parents.insert(performed_node);

        let performed: HashSet<_> = [performed_node].into_iter().collect();
        let necessity: HashMap<_, _> = [(child, true)].into_iter().collect();

        let result = surface(&graph, true, false, None, &performed, &necessity);
        assert_eq!(result, [child].into_iter().collect());
    }

    #[test]
    fn attack_surface_excludes_effect_children() {
        let mut graph = dummy_graph();
        let performed_node = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let effect_child = add_dummy_node(&mut graph, "DummyOrAttackStep");
        mark_effect(&mut graph, effect_child);
        graph.nodes[performed_node].children.insert(effect_child);
        graph.nodes[effect_child].parents.insert(performed_node);

        let performed: HashSet<_> = [performed_node].into_iter().collect();
        let necessity: HashMap<_, _> = [(effect_child, true)].into_iter().collect();

        let result = surface(&graph, true, false, None, &performed, &necessity);
        assert!(result.is_empty());
    }

    #[test]
    fn attack_surface_skip_compromised_excludes_already_performed_nodes() {
        let mut graph = dummy_graph();
        let parent = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let child = add_dummy_node(&mut graph, "DummyOrAttackStep");
        graph.nodes[parent].children.insert(child);
        graph.nodes[child].parents.insert(parent);

        // Both nodes already performed - with skip_compromised the already-
        // performed child is excluded even though it's otherwise traversable.
        let performed: HashSet<_> = [parent, child].into_iter().collect();
        let necessity: HashMap<_, _> = [(child, true)].into_iter().collect();

        let with_skip = surface(&graph, true, false, None, &performed, &necessity);
        assert!(with_skip.is_empty());

        let without_skip = surface(&graph, false, false, None, &performed, &necessity);
        assert_eq!(without_skip, [child].into_iter().collect());
    }

    #[test]
    fn attack_surface_skip_unnecessary_excludes_unnecessary_nodes() {
        let mut graph = dummy_graph();
        let parent = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let child = add_dummy_node(&mut graph, "DummyOrAttackStep");
        graph.nodes[parent].children.insert(child);
        graph.nodes[child].parents.insert(parent);

        let performed: HashSet<_> = [parent].into_iter().collect();
        let necessity: HashMap<_, _> = [(child, false)].into_iter().collect();

        let with_skip = surface(&graph, true, true, None, &performed, &necessity);
        assert!(with_skip.is_empty());

        let without_skip = surface(&graph, true, false, None, &performed, &necessity);
        assert_eq!(without_skip, [child].into_iter().collect());
    }

    #[test]
    fn attack_surface_respects_flattened_actionable_steps() {
        let mut graph = dummy_graph();
        let parent = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let actionable_child = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let non_actionable_child = add_dummy_node(&mut graph, "DummyOrAttackStep");
        graph.nodes[parent].children.insert(actionable_child);
        graph.nodes[parent].children.insert(non_actionable_child);
        graph.nodes[actionable_child].parents.insert(parent);
        graph.nodes[non_actionable_child].parents.insert(parent);

        let performed: HashSet<_> = [parent].into_iter().collect();
        let necessity: HashMap<_, _> = [(actionable_child, true), (non_actionable_child, true)]
            .into_iter()
            .collect();

        // No rule at all (`None`) -> every node actionable.
        let all_actionable = surface(&graph, true, false, None, &performed, &necessity);
        assert_eq!(
            all_actionable,
            [actionable_child, non_actionable_child]
                .into_iter()
                .collect()
        );

        // A rule present -> only the ids explicitly in the set are actionable.
        let actionable_set: HashSet<_> = [actionable_child].into_iter().collect();
        let restricted = surface(
            &graph,
            true,
            false,
            Some(&actionable_set),
            &performed,
            &necessity,
        );
        assert_eq!(restricted, [actionable_child].into_iter().collect());
    }

    #[test]
    fn attack_surface_from_nodes_overrides_performed_nodes_as_starting_point() {
        let mut graph = dummy_graph();
        let old_performed = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let old_child = add_dummy_node(&mut graph, "DummyOrAttackStep");
        graph.nodes[old_performed].children.insert(old_child);
        graph.nodes[old_child].parents.insert(old_performed);

        let new_node = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let new_child = add_dummy_node(&mut graph, "DummyOrAttackStep");
        graph.nodes[new_node].children.insert(new_child);
        graph.nodes[new_child].parents.insert(new_node);

        let performed: HashSet<_> = [old_performed, new_node].into_iter().collect();
        let from_nodes: HashSet<_> = [new_node].into_iter().collect();
        let necessity: HashMap<_, _> = [(old_child, true), (new_child, true)].into_iter().collect();
        let empty = HashSet::new();

        let result = get_attack_surface(
            &graph,
            true,
            false,
            None,
            &performed,
            Some(&from_nodes),
            &empty,
            &empty,
            &necessity,
        )
        .unwrap();

        // Only `new_child` (child of `from_nodes`), not `old_child`.
        assert_eq!(result, [new_child].into_iter().collect());
    }

    #[test]
    fn attack_surface_excludes_children_of_non_live_from_nodes() {
        let mut graph = dummy_graph();
        let dead_parent = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let child = add_dummy_node(&mut graph, "DummyOrAttackStep");
        graph.nodes[dead_parent].children.insert(child);
        graph.nodes[child].parents.insert(dead_parent);

        let performed: HashSet<_> = [dead_parent].into_iter().collect();
        let necessity: HashMap<_, _> = [(child, true)].into_iter().collect();

        graph.remove_node(dead_parent).unwrap();

        // `dead_parent` no longer exists in the graph, so its (frozen,
        // pre-removal) `.children` snapshot must not be a source of new
        // attack-surface nodes.
        let result = surface(&graph, true, false, None, &performed, &necessity);
        assert!(result.is_empty());
    }

    /// Port of the property `tests/test_attacker.py::test_attack_surface_traininglang`
    /// asserted, on a hand-built reduction of the trainingLang scenario
    /// (`traininglang_scenario.yml`: entry points `User:3:phishing` and
    /// `Host:0:connect`) instead of the scenario fixture itself (needs
    /// Phase C's not-yet-ported scenario-YAML loader to build in Rust).
    /// Enabling defenses must shrink the attack surface, with necessity
    /// recomputed per phase, mirroring the simulator's recompute after a
    /// defender step. Settings mirror `AttackSurfaceSettings` defaults
    /// (`skip_compromised=True`, `skip_unnecessary=False`).
    #[test]
    fn attack_surface_shrinks_as_defenses_are_enabled() {
        let mut graph = dummy_graph();
        // User:3:phishing / Host:0:connect - the scenario's entry points.
        let phishing = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let connect = add_dummy_node(&mut graph, "DummyAndAttackStep");
        // Host:0:authenticate - an `or` step the attacker hasn't reached.
        let auth = add_dummy_node(&mut graph, "DummyOrAttackStep");
        // User:3:notPresent / Host:0:notPresent.
        let user_np = add_dummy_node(&mut graph, "DummyDefenseAttackStep");
        let host_np = add_dummy_node(&mut graph, "DummyDefenseAttackStep");
        // User:3:compromise (`and`: phishing + user notPresent).
        let compromise = add_dummy_node(&mut graph, "DummyAndAttackStep");
        // Host:0:access (`and`: connect + host notPresent + authenticate).
        let access = add_dummy_node(&mut graph, "DummyAndAttackStep");

        let link = |graph: &mut AttackGraph, parent, child| {
            graph.nodes[parent].children.insert(child);
            graph.nodes[child].parents.insert(parent);
        };
        link(&mut graph, phishing, compromise);
        link(&mut graph, user_np, compromise);
        link(&mut graph, connect, access);
        link(&mut graph, host_np, access);
        link(&mut graph, auth, access);

        let performed: HashSet<_> = [phishing, connect].into_iter().collect();
        let no_impossible = HashSet::new();

        let surface_with = |impossible: &HashSet<AttackGraphNodeId>,
                            enabled: &HashSet<AttackGraphNodeId>| {
            let necessity = crate::necessity::calculate_necessity(&graph, enabled).unwrap();
            get_attack_surface(
                &graph, true, false, None, &performed, None, impossible, enabled, &necessity,
            )
            .unwrap()
        };

        // No defenses enabled: `compromise` is reachable (its disabled
        // defense parent is unnecessary, so not required), `access` is not
        // (its necessary `auth` parent is unperformed).
        let none_enabled = HashSet::new();
        assert_eq!(
            surface_with(&no_impossible, &none_enabled),
            [compromise].into_iter().collect()
        );

        // "This wont help, already compromised" - enabling Host:0:notPresent
        // only blocks `access`, which wasn't on the surface anyway.
        let host_enabled: HashSet<_> = [host_np].into_iter().collect();
        assert_eq!(
            surface_with(&no_impossible, &host_enabled),
            [compromise].into_iter().collect()
        );

        // "This should block the attack from further propagating" -
        // enabling User:3:notPresent too blocks `compromise`, leaving an
        // empty surface and a terminated attacker.
        let both_enabled: HashSet<_> = [host_np, user_np].into_iter().collect();
        let blocked_surface = surface_with(&no_impossible, &both_enabled);
        assert!(blocked_surface.is_empty());
        assert!(crate::attacker_step::attacker_is_terminated(
            &blocked_surface,
            &HashSet::new(),
            &performed
        ));

        // Rust-only extra: an impossible attack step is excluded the same
        // way an enabled-defense-blocked one is.
        let impossible: HashSet<_> = [compromise].into_iter().collect();
        assert!(surface_with(&impossible, &none_enabled).is_empty());
    }
}
