//! The pure-Rust reset/step orchestration of `malsim`'s simulator over a
//! *dynamic* instance model/attack graph, where model effects attached to
//! attack steps/defenses add/remove assets and associations (and so attack
//! graph nodes) mid-episode: a port of
//! `python/malsim/dyna_mal_simulator/simulator.py::dyna_reset`/`dyna_step`.
//!
//! Built by composition over [`Simulator`] - the Rust analogue of
//! `DynaMalSimulator(MalSimulator)` (`PORTING_NOTES.md` §11): reset is
//! "restore the model to its pristine snapshot, then the `Simulator`
//! reset", and step swaps `defender_step`/`attacker_step` for the
//! model-effect-aware `dyna_defender_step`/`dyna_attacker_step` but shares
//! the `Simulator`'s per-agent bookkeeping (`update_attacker_runtimes`/
//! `update_defender_runtimes`).

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;

use maltoolbox_attackgraph::{AttackGraph, AttackGraphNodeId};
use maltoolbox_model::Model;

use crate::dyna_attacker_step::dyna_attacker_step;
use crate::dyna_defender_step::dyna_defender_step;
use crate::model_effects::ModEffectOp;
use crate::model_state::{capture_model_snapshot, reset_model_effects, ModelSnapshot};
use crate::settings::{FlatAgentSettings, MalSimulatorSettings};
use crate::simulator::{
    agent_actions, check_action_agents, update_attacker_runtimes, update_defender_runtimes,
    AttackerResults, SimState, Simulator, SimulatorError, StepOutcome,
};

/// Everything a single `DynaSimulator::step` produced: the same per-agent
/// deltas as a static step, plus the model modifications its model effects
/// made - mirrors `DynaMalSimulatorState` extending `MalSimulatorState`
/// with a `modification_record`.
#[derive(Debug, Clone)]
pub struct DynaStepOutcome {
    pub step: StepOutcome,
    /// This step's model-effect modification record, in the order the
    /// modifications were applied. Non-empty means the graph (and
    /// `graph_state`, which grows to cover newly created nodes) may have
    /// changed shape this step.
    pub step_modification_record: Vec<ModEffectOp>,
}

/// Port of `DynaMalSimulator`'s reset/step orchestration, over a dynamic
/// instance model/attack graph. Both the `Model` and the `AttackGraph`
/// built from it are shared handles (§2.2, §6 B3), mutated in place by
/// model effects.
pub struct DynaSimulator {
    sim: Simulator,
    model: Rc<RefCell<Model>>,
    /// The model as it was at `DynaSimulator::new` - every `reset` restores
    /// it (mirrors `DynaMalSimulator.__init__` capturing
    /// `attack_graph.model.to_dict()` once, before any mutation).
    snapshot: ModelSnapshot,
}

impl DynaSimulator {
    /// A dyna simulator over `graph`/`model` (`graph` must have been built
    /// from `model`): captures the pristine model snapshot every `reset`
    /// restores to.
    pub fn new(graph: Rc<RefCell<AttackGraph>>, model: Rc<RefCell<Model>>) -> Self {
        let snapshot = capture_model_snapshot(&model.borrow());
        DynaSimulator {
            sim: Simulator::new(graph),
            model,
            snapshot,
        }
    }

    pub fn graph(&self) -> &Rc<RefCell<AttackGraph>> {
        self.sim.graph()
    }

    pub fn model(&self) -> &Rc<RefCell<Model>> {
        &self.model
    }

    /// The state built by the last successful `reset` (and advanced by every
    /// `step` since) - `None` before the first successful `reset`.
    pub fn state(&self) -> Option<&SimState> {
        self.sim.state()
    }

    /// Restores the shared `Model`/`AttackGraph` to the snapshot captured at
    /// `new` (`reset_model_effects`). Restoring an already-pristine model
    /// is a no-op.
    ///
    /// `reset` does this itself, but a caller that resolves per-node agent
    /// settings against the graph (`FlatAgentSettings`) must call this
    /// first: nodes that the previous episode's model effects removed are
    /// only regenerated (with new ids) by the restore, so flattening
    /// against the mutated graph would miss them.
    pub fn restore_model(&mut self) -> Result<(), SimulatorError> {
        let mut graph = self.sim.graph().borrow_mut();
        let mut model = self.model.borrow_mut();
        reset_model_effects(&mut graph, &mut model, &self.snapshot)?;
        Ok(())
    }

    /// Resets the simulator: first restores the shared `Model`/`AttackGraph`
    /// (`restore_model`, mirroring `dyna_reset`'s `reset_model_effects`
    /// call), then resets exactly like `Simulator::reset`. `agents` must
    /// already be resolved against the restored graph, so call
    /// `restore_model` before flattening them.
    pub fn reset(
        &mut self,
        settings: &MalSimulatorSettings,
        agents: Vec<(String, FlatAgentSettings)>,
        seed: u64,
    ) -> Result<&SimState, SimulatorError> {
        self.restore_model()?;
        self.sim.reset(settings, agents, seed)
    }

    /// Same two-phase shape as `Simulator::step`, except defenders/attackers
    /// act via `dyna_defender_step`/`dyna_attacker_step` (model-effect-aware,
    /// mutating the shared `AttackGraph`/`Model` in place and folding any
    /// newly-created nodes into `graph_state`/`enabled_defenses`
    /// internally). `actions` works as in `Simulator::step`.
    pub fn step(
        &mut self,
        actions: &HashMap<String, Vec<AttackGraphNodeId>>,
    ) -> Result<DynaStepOutcome, SimulatorError> {
        let graph_rc = self.sim.graph().clone();
        let model_rc = self.model.clone();
        let state = self.sim.state_mut()?;
        check_action_agents(state, actions)?;

        let attacker_names: Vec<String> = state.attackers.keys().cloned().collect();
        let defender_names: Vec<String> = state.defenders.keys().cloned().collect();
        let mut modification_record: Vec<ModEffectOp> = Vec::new();

        // --- Defenders act first ---
        let mut step_enabled_defenses: HashSet<AttackGraphNodeId> = HashSet::new();
        {
            let mut graph = graph_rc.borrow_mut();
            let mut model = model_rc.borrow_mut();
            for name in &defender_names {
                let runtime = &state.defenders[name];
                let (enabled, ops) = dyna_defender_step(
                    &mut graph,
                    &mut model,
                    &mut state.rng,
                    state.settings.ttc_mode,
                    state.settings.run_defense_step_bernoullis,
                    state.settings.run_attack_step_bernoullis,
                    agent_actions(actions, name),
                    &runtime.action_surface,
                    &mut state.graph_state,
                    &mut state.enabled_defenses,
                )?;
                step_enabled_defenses.extend(enabled);
                modification_record.extend(ops);
            }
        }
        // `dyna_defender_step` only folds *newly model-effect-created*
        // defense nodes into `enabled_defenses` internally (via
        // `fold_new_nodes_into_graph_state`) - the defenses actually
        // enabled by this step's requested actions still need merging
        // here, same as `Simulator::step` does for `defender_step`.
        state
            .enabled_defenses
            .extend(step_enabled_defenses.iter().copied());

        // --- Attackers act afterwards ---
        let mut step_compromised_nodes: HashSet<AttackGraphNodeId> = HashSet::new();
        let mut attacker_results: AttackerResults = BTreeMap::new();
        {
            let mut graph = graph_rc.borrow_mut();
            let mut model = model_rc.borrow_mut();
            for name in &attacker_names {
                let runtime = &state.attackers[name];
                let (compromised, attempted, ops) = dyna_attacker_step(
                    &mut graph,
                    &mut model,
                    &mut state.rng,
                    state.settings.ttc_mode,
                    state.settings.run_defense_step_bernoullis,
                    state.settings.run_attack_step_bernoullis,
                    agent_actions(actions, name),
                    &runtime.entry_points,
                    &runtime.action_surface,
                    &runtime.performed_nodes,
                    &runtime.num_attempts,
                    runtime.ttc_dist_overrides.as_ref(),
                    Some(&runtime.ttc_values),
                    &mut state.graph_state,
                    &mut state.enabled_defenses,
                )?;
                step_compromised_nodes.extend(compromised.iter().copied());
                modification_record.extend(ops);
                attacker_results.insert(name.clone(), (compromised, attempted));
            }
        }

        let graph = graph_rc.borrow();
        let attackers = update_attacker_runtimes(&graph, state, &attacker_names, attacker_results)?;
        let defenders = update_defender_runtimes(
            &graph,
            state,
            &defender_names,
            &step_enabled_defenses,
            &step_compromised_nodes,
        )?;

        Ok(DynaStepOutcome {
            step: StepOutcome {
                attackers,
                defenders,
                step_enabled_defenses,
            },
            step_modification_record: modification_record,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::FlatAttackerSettings;
    use crate::test_fixtures::wiper_attack_graph;

    /// Settings with every bernoulli off, so tests are not at the mercy of
    /// sampled impossible steps/pre-enabled defenses.
    fn settings() -> MalSimulatorSettings {
        MalSimulatorSettings {
            run_defense_step_bernoullis: false,
            run_attack_step_bernoullis: false,
            ..MalSimulatorSettings::default()
        }
    }

    fn attacker(entry_points: &[AttackGraphNodeId]) -> FlatAgentSettings {
        FlatAgentSettings::Attacker(FlatAttackerSettings {
            entry_points: entry_points.iter().copied().collect(),
            ..FlatAttackerSettings::default()
        })
    }

    fn actions(
        entries: &[(&str, Vec<AttackGraphNodeId>)],
    ) -> HashMap<String, Vec<AttackGraphNodeId>> {
        entries
            .iter()
            .map(|(name, nodes)| (name.to_string(), nodes.clone()))
            .collect()
    }

    #[test]
    fn step_before_reset_fails() {
        let (graph, model) = wiper_attack_graph();
        let mut sim =
            DynaSimulator::new(Rc::new(RefCell::new(graph)), Rc::new(RefCell::new(model)));
        assert!(sim.state().is_none());
        assert!(matches!(
            sim.step(&HashMap::new()),
            Err(SimulatorError::NotReset)
        ));
    }

    #[test]
    fn dyna_reset_and_step_run_model_effects() {
        let (graph, model) = wiper_attack_graph();
        let infect = graph.full_name_to_node["InfectedDevice:infect"];
        let initial_node_count = graph.nodes.len();
        let graph_rc = Rc::new(RefCell::new(graph));
        let model_rc = Rc::new(RefCell::new(model));
        let mut sim = DynaSimulator::new(graph_rc.clone(), model_rc.clone());

        let s = MalSimulatorSettings {
            compromise_entrypoints_at_start: false,
            ..settings()
        };
        let agents = || vec![("WiperController".to_string(), attacker(&[infect]))];
        sim.reset(&s, agents(), 42).unwrap();

        let outcome = sim
            .step(&actions(&[("WiperController", vec![infect])]))
            .unwrap();
        assert!(outcome.step.attackers["WiperController"]
            .step_performed_nodes
            .contains(&infect));
        assert!(!outcome.step_modification_record.is_empty());
        assert!(graph_rc.borrow().nodes.len() > initial_node_count);
        assert!(model_rc
            .borrow()
            .assets
            .values()
            .any(|a| a.name == "Wiper-7"));

        // A second reset restores the pristine model/graph.
        sim.reset(&s, agents(), 42).unwrap();
        assert_eq!(graph_rc.borrow().nodes.len(), initial_node_count);
    }

    #[test]
    fn restore_model_undoes_model_effects_and_is_idempotent() {
        let (graph, model) = wiper_attack_graph();
        let infect = graph.full_name_to_node["InfectedDevice:infect"];
        let graph_rc = Rc::new(RefCell::new(graph));
        let mut sim = DynaSimulator::new(graph_rc.clone(), Rc::new(RefCell::new(model)));

        // Restoring a pristine model changes nothing (not even node ids),
        // so `reset`'s own restore after an explicit one is harmless.
        let pristine: Vec<_> = graph_rc.borrow().nodes.keys().collect();
        sim.restore_model().unwrap();
        assert_eq!(graph_rc.borrow().nodes.keys().collect::<Vec<_>>(), pristine);

        let s = MalSimulatorSettings {
            compromise_entrypoints_at_start: false,
            ..settings()
        };
        let agents = || vec![("WiperController".to_string(), attacker(&[infect]))];
        sim.reset(&s, agents(), 1).unwrap();
        sim.step(&actions(&[("WiperController", vec![infect])]))
            .unwrap();
        assert!(graph_rc.borrow().nodes.len() > pristine.len());

        // The explicit restore brings the graph back before `reset` runs,
        // and `reset`'s own restore is then a no-op.
        sim.restore_model().unwrap();
        let restored: Vec<_> = graph_rc.borrow().nodes.keys().collect();
        assert_eq!(restored.len(), pristine.len());
        sim.reset(&s, agents(), 1).unwrap();
        assert_eq!(graph_rc.borrow().nodes.keys().collect::<Vec<_>>(), restored);
    }
}
