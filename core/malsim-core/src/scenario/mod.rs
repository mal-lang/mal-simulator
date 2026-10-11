//! Rust-only scenario loading - the independent Rust implementation of
//! `python/malsim/scenario/` + `python/malsim/config/` used by an embedding
//! Rust program with no Python runtime (`PORTING_NOTES.md` §2.6, §7). The
//! PyO3 path never calls into this module: Python keeps its own `Scenario`
//! and flattens settings itself (§2.4).

pub mod agent_settings;
pub mod flatten;
pub mod loading;
pub mod node_property_rule;

pub use agent_settings::{
    agent_settings_from_dict, AgentSettings, AgentSettingsError, AgentType, AttackerSettings,
    DefenderSettings, EntryPoints,
};
pub use flatten::{flatten_attacker_settings, flatten_defender_settings, get_entry_points};
pub use loading::{
    load_scenario_dict, recursive_update, validate_scenario_dict, ScenarioFileError,
};
pub use node_property_rule::{NodePropertyRule, NodePropertyRuleError, RuleValue, StepValues};

use std::cell::RefCell;
use std::fmt;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use maltoolbox_attackgraph::{AttackGraph, AttackGraphNodeId};
use maltoolbox_language::LanguageGraph;
use maltoolbox_model::Model;
use rand::Rng;
use serde_json::{Map, Value};

use crate::settings::{FlatAgentSettings, MalSimulatorSettings, SettingsError};

#[derive(Debug)]
pub enum ScenarioError {
    File(ScenarioFileError),
    Settings(SettingsError),
    AgentSettings(AgentSettingsError),
    /// The scenario's `agents` field is missing or not a mapping. (Python
    /// accepts `agent_settings` in validation but then reads `agents`,
    /// raising `KeyError` - see `PORTING_NOTES.md` §12.)
    InvalidAgents(String),
    /// `lang_file` could not be loaded into a `LanguageGraph`.
    Language {
        lang_file: String,
        message: String,
    },
    /// `model`/`model_file` could not be loaded into a `Model`.
    Model(String),
    /// The attack graph could not be built from the model.
    AttackGraph(String),
}

impl fmt::Display for ScenarioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScenarioError::File(e) => write!(f, "{e}"),
            ScenarioError::Settings(e) => write!(f, "{e}"),
            ScenarioError::AgentSettings(e) => write!(f, "{e}"),
            ScenarioError::InvalidAgents(msg) => write!(f, "Invalid scenario agents: {msg}"),
            ScenarioError::Language { lang_file, message } => {
                write!(f, "Could not load language '{lang_file}': {message}")
            }
            ScenarioError::Model(msg) => write!(f, "Could not load model: {msg}"),
            ScenarioError::AttackGraph(msg) => write!(f, "Could not build attack graph: {msg}"),
        }
    }
}

impl std::error::Error for ScenarioError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ScenarioError::File(e) => Some(e),
            ScenarioError::Settings(e) => Some(e),
            ScenarioError::AgentSettings(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ScenarioFileError> for ScenarioError {
    fn from(e: ScenarioFileError) -> Self {
        ScenarioError::File(e)
    }
}

impl From<SettingsError> for ScenarioError {
    fn from(e: SettingsError) -> Self {
        ScenarioError::Settings(e)
    }
}

impl From<AgentSettingsError> for ScenarioError {
    fn from(e: AgentSettingsError) -> Self {
        ScenarioError::AgentSettings(e)
    }
}

/// Port of `Scenario`: a language, a model, the attack graph built from
/// them, agent settings (attackers with entry points/goals already
/// resolved to node ids) and simulator settings.
///
/// The graph and model are held as `Rc<RefCell<..>>`, the handle types
/// `Simulator` takes (§2.2), so a loaded scenario plugs straight into a
/// simulator.
pub struct Scenario {
    /// `lang_file` as given (made absolute by `load_from_file`).
    pub lang_file: String,
    /// The model file, if the model was given as `model_file` rather than
    /// inline (Python's `_model_file`).
    pub model_file: Option<PathBuf>,
    pub lang_graph: Rc<LanguageGraph>,
    pub model: Rc<RefCell<Model>>,
    pub attack_graph: Rc<RefCell<AttackGraph>>,
    /// Attackers first, then defenders, each in scenario-file order (same
    /// as Python's `Scenario.agent_settings`).
    pub agent_settings: Vec<AgentSettings<AttackGraphNodeId>>,
    pub sim_settings: MalSimulatorSettings,
}

/// Port of `LanguageGraph.load_from_file` (`.mal`, `.mar`, `.yml`/`.yaml`,
/// `.json`). Python's `.git` URL support is not ported (§12).
fn load_language_graph(lang_file: &str) -> Result<LanguageGraph, ScenarioError> {
    let language_error = |message: String| ScenarioError::Language {
        lang_file: lang_file.to_string(),
        message,
    };
    if lang_file.ends_with(".git") {
        return Err(language_error(
            "loading a language from a git URL is not supported by the Rust scenario loader".into(),
        ));
    }
    maltoolbox_language::load_language_graph_from_file(lang_file)
        .map_err(|e| language_error(e.to_string()))
}

impl Scenario {
    /// Port of `Scenario.__init__`: loads the language and model, builds the
    /// attack graph and resolves attacker entry points/goals against it.
    pub fn new(
        lang_file: &str,
        model: ModelSource,
        agents: Vec<AgentSettings<String>>,
        sim_settings: MalSimulatorSettings,
    ) -> Result<Scenario, ScenarioError> {
        let lang_graph = Rc::new(load_language_graph(lang_file)?);

        let (model, model_file) = match model {
            ModelSource::File(path) => {
                let model = maltoolbox_model::load_from_file(&path, lang_graph.clone())
                    .map_err(|e| ScenarioError::Model(e.to_string()))?;
                (model, Some(path))
            }
            ModelSource::Dict(dict) => {
                let model = maltoolbox_model::from_dict(&dict, lang_graph.clone())
                    .map_err(|e| ScenarioError::Model(e.to_string()))?;
                (model, None)
            }
        };

        let attack_graph = AttackGraph::from_model(&model)
            .map_err(|e| ScenarioError::AttackGraph(e.to_string()))?;

        let mut attackers = Vec::new();
        let mut defenders = Vec::new();
        for agent in agents {
            match agent {
                AgentSettings::Attacker(a) => attackers.push(AgentSettings::Attacker(
                    a.convert_to_attack_graph_nodes(&attack_graph)?,
                )),
                AgentSettings::Defender(d) => defenders.push(AgentSettings::Defender(d)),
            }
        }
        attackers.extend(defenders);

        Ok(Scenario {
            lang_file: lang_file.to_string(),
            model_file,
            lang_graph,
            model: Rc::new(RefCell::new(model)),
            attack_graph: Rc::new(RefCell::new(attack_graph)),
            agent_settings: attackers,
            sim_settings,
        })
    }

    /// Port of `Scenario.from_dict`: validates the dict, parses agents
    /// (skipping `null` ones) and `sim_settings`, then builds the scenario.
    pub fn from_dict(scenario: &Map<String, Value>) -> Result<Scenario, ScenarioError> {
        validate_scenario_dict(scenario)?;

        let agents = match scenario.get("agents") {
            Some(Value::Object(agents)) => agents,
            Some(other) => {
                return Err(ScenarioError::InvalidAgents(format!(
                    "expected a mapping, got {other}"
                )))
            }
            None => return Err(ScenarioError::InvalidAgents("'agents' is required".into())),
        };
        let agent_settings = agents
            .iter()
            .filter(|(_, d)| !d.is_null())
            .map(|(name, d)| agent_settings_from_dict(name, d))
            .collect::<Result<Vec<_>, _>>()?;

        // `scenario_dict.get('model') or scenario_dict['model_file']`
        let model = match scenario.get("model") {
            Some(model) if is_truthy(model) => ModelSource::Dict(model.clone()),
            _ => match scenario.get("model_file") {
                Some(Value::String(path)) => ModelSource::File(PathBuf::from(path)),
                _ => return Err(ScenarioFileError::MissingOneOf("model_file", "model").into()),
            },
        };

        let sim_settings = match scenario.get("sim_settings") {
            None | Some(Value::Null) => MalSimulatorSettings::default(),
            Some(v) => MalSimulatorSettings::from_value(v)?,
        };

        let lang_file = scenario
            .get("lang_file")
            .and_then(Value::as_str)
            .ok_or(ScenarioFileError::PathNotAString("lang_file"))?;

        Scenario::new(lang_file, model, agent_settings, sim_settings)
    }

    /// Port of `Scenario.load_from_file` (without `**override_keys`; apply
    /// overrides to `load_scenario_dict`'s result and call `from_dict`).
    pub fn load_from_file(scenario_file: impl AsRef<Path>) -> Result<Scenario, ScenarioError> {
        Scenario::from_dict(&load_scenario_dict(scenario_file)?)
    }

    /// Port of the `attacker_settings` property, in scenario order.
    pub fn attacker_settings(&self) -> impl Iterator<Item = &AttackerSettings<AttackGraphNodeId>> {
        self.agent_settings.iter().filter_map(|a| match a {
            AgentSettings::Attacker(a) => Some(a),
            AgentSettings::Defender(_) => None,
        })
    }

    /// Port of the `defender_settings` property, in scenario order.
    pub fn defender_settings(&self) -> impl Iterator<Item = &DefenderSettings> {
        self.agent_settings.iter().filter_map(|a| match a {
            AgentSettings::Defender(d) => Some(d),
            AgentSettings::Attacker(_) => None,
        })
    }

    /// The flat per-agent inputs for `Simulator::reset`, mirroring what
    /// Python's `MalSimulator.reset` builds before calling native: one
    /// entry-point set sampled per attacker (attackers first, in order),
    /// then every agent's settings flattened against the current graph.
    ///
    /// For a dyna simulator, call `Simulator::restore_model` before this
    /// on every reset after the first, so the rules see nodes that the
    /// previous episode's model effects removed and the restore brought
    /// back.
    pub fn flatten_agents(&self, rng: &mut impl Rng) -> Vec<(String, FlatAgentSettings)> {
        let graph = self.attack_graph.borrow();
        let model = self.model.borrow();
        let mut flat = Vec::with_capacity(self.agent_settings.len());
        for attacker in self.attacker_settings() {
            let entry_points = get_entry_points(attacker, rng);
            flat.push((
                attacker.name.clone(),
                FlatAgentSettings::Attacker(flatten_attacker_settings(
                    attacker,
                    &graph,
                    &model,
                    &entry_points,
                )),
            ));
        }
        for defender in self.defender_settings() {
            flat.push((
                defender.name.clone(),
                FlatAgentSettings::Defender(flatten_defender_settings(defender, &graph, &model)),
            ));
        }
        flat
    }
}

/// Where `Scenario::new` gets its model from (Python's `model: Model |
/// dict | str`; an already-built `Model` is not accepted since the model
/// must share the scenario's freshly loaded `LanguageGraph`).
#[derive(Debug, Clone)]
pub enum ModelSource {
    File(PathBuf),
    Dict(Value),
}

/// Python truthiness of a JSON value, for the `get(..) or ..` idioms.
fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

#[cfg(test)]
mod tests;
