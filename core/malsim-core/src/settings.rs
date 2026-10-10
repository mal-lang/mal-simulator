//! Settings shared by the Rust-only scenario loader (`crate::scenario`) and
//! the `Simulator`: a port of `python/malsim/config/sim_settings.py`
//! (`MalSimulatorSettings`, `AttackSurfaceSettings`, `RewardMode`) plus the
//! flat, id-resolved per-agent inputs the `Simulator` consumes - the Rust
//! counterpart of what `native_settings.py`'s `flatten_*_settings` produce
//! on the Python side. See `PORTING_NOTES.md` §11.

use std::collections::{HashMap, HashSet};
use std::fmt;

use maltoolbox_attackgraph::AttackGraphNodeId;
use serde_json::{Map, Value};

pub use crate::graph_state::TtcMode;
use crate::ttc::TtcDist;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SettingsError {
    /// A settings value that must be a mapping isn't one.
    NotAMapping(&'static str),
    /// A key the settings dataclass doesn't have (Python raises a
    /// `TypeError` from the dataclass constructor).
    UnknownField {
        settings: &'static str,
        field: String,
    },
    /// A field with a value of the wrong type or an unknown enum name.
    InvalidValue { field: String, reason: String },
}

impl fmt::Display for SettingsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SettingsError::NotAMapping(what) => write!(f, "{what} must be a mapping"),
            SettingsError::UnknownField { settings, field } => {
                write!(f, "{settings} got an unexpected keyword argument '{field}'")
            }
            SettingsError::InvalidValue { field, reason } => {
                write!(f, "Invalid value for '{field}': {reason}")
            }
        }
    }
}

impl std::error::Error for SettingsError {}

/// Port of `RewardMode`. Carried through settings for inspection only:
/// rewards are not computed in Rust (§2.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RewardMode {
    Cumulative,
    OneOff,
    ExpectedTtc,
    SampleTtc,
}

impl RewardMode {
    /// Parses the Python enum member name (`RewardMode[name]`).
    pub fn from_name(name: &str) -> Option<RewardMode> {
        match name {
            "CUMULATIVE" => Some(RewardMode::Cumulative),
            "ONE_OFF" => Some(RewardMode::OneOff),
            "EXPECTED_TTC" => Some(RewardMode::ExpectedTtc),
            "SAMPLE_TTC" => Some(RewardMode::SampleTtc),
            _ => None,
        }
    }

    /// The Python enum member name (`RewardMode.name`).
    pub fn name(self) -> &'static str {
        match self {
            RewardMode::Cumulative => "CUMULATIVE",
            RewardMode::OneOff => "ONE_OFF",
            RewardMode::ExpectedTtc => "EXPECTED_TTC",
            RewardMode::SampleTtc => "SAMPLE_TTC",
        }
    }
}

/// Port of `AttackSurfaceSettings`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttackSurfaceSettings {
    /// If true, do not add already compromised nodes to the attack surface.
    pub skip_compromised: bool,
    /// If true, do not add unnecessary nodes to the attack surface.
    pub skip_unnecessary: bool,
}

impl Default for AttackSurfaceSettings {
    fn default() -> Self {
        AttackSurfaceSettings {
            skip_compromised: true,
            skip_unnecessary: false,
        }
    }
}

impl AttackSurfaceSettings {
    /// Port of `AttackSurfaceSettings(**d)`.
    pub fn from_value(value: &Value) -> Result<Self, SettingsError> {
        let obj = value
            .as_object()
            .ok_or(SettingsError::NotAMapping("attack_surface"))?;
        let mut settings = AttackSurfaceSettings::default();
        for (key, v) in obj {
            match key.as_str() {
                "skip_compromised" => settings.skip_compromised = bool_field(key, v)?,
                "skip_unnecessary" => settings.skip_unnecessary = bool_field(key, v)?,
                _ => {
                    return Err(SettingsError::UnknownField {
                        settings: "AttackSurfaceSettings",
                        field: key.clone(),
                    })
                }
            }
        }
        Ok(settings)
    }
}

/// Port of `MalSimulatorSettings`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MalSimulatorSettings {
    /// Defined but unread, same as on the Python side (§2.5).
    pub uncompromise_untraversable_steps: bool,
    pub ttc_mode: TtcMode,
    pub seed: Option<u64>,
    pub attack_surface: AttackSurfaceSettings,
    /// If true, each attacker compromises its entry points at the start of
    /// the simulation.
    pub compromise_entrypoints_at_start: bool,
    /// If true, sample defense step bernoullis to decide their initial states.
    pub run_defense_step_bernoullis: bool,
    /// If true, sample attack step bernoullis to decide if they are
    /// impossible/exist.
    pub run_attack_step_bernoullis: bool,
}

impl Default for MalSimulatorSettings {
    fn default() -> Self {
        MalSimulatorSettings {
            uncompromise_untraversable_steps: false,
            ttc_mode: TtcMode::Disabled,
            seed: None,
            attack_surface: AttackSurfaceSettings::default(),
            compromise_entrypoints_at_start: true,
            run_defense_step_bernoullis: true,
            run_attack_step_bernoullis: true,
        }
    }
}

impl MalSimulatorSettings {
    /// Port of `MalSimulatorSettings(**d)` including `__post_init__`'s
    /// string-to-enum (`ttc_mode`) and dict-to-dataclass (`attack_surface`)
    /// conversions. `null` for a field leaves its default only for `seed`,
    /// whose Python default is `None`.
    pub fn from_value(value: &Value) -> Result<Self, SettingsError> {
        let obj: &Map<String, Value> = value
            .as_object()
            .ok_or(SettingsError::NotAMapping("sim_settings"))?;
        let mut settings = MalSimulatorSettings::default();
        for (key, v) in obj {
            match key.as_str() {
                "uncompromise_untraversable_steps" => {
                    settings.uncompromise_untraversable_steps = bool_field(key, v)?
                }
                "ttc_mode" => {
                    let name = v
                        .as_str()
                        .ok_or_else(|| invalid(key, "expected a TTCMode name"))?;
                    settings.ttc_mode = TtcMode::from_name(name)
                        .ok_or_else(|| invalid(key, &format!("unknown TTCMode '{name}'")))?;
                }
                "seed" => {
                    settings.seed = match v {
                        Value::Null => None,
                        _ => Some(
                            v.as_u64()
                                .ok_or_else(|| invalid(key, "expected a non-negative integer"))?,
                        ),
                    }
                }
                "attack_surface" => settings.attack_surface = AttackSurfaceSettings::from_value(v)?,
                "compromise_entrypoints_at_start" => {
                    settings.compromise_entrypoints_at_start = bool_field(key, v)?
                }
                "run_defense_step_bernoullis" => {
                    settings.run_defense_step_bernoullis = bool_field(key, v)?
                }
                "run_attack_step_bernoullis" => {
                    settings.run_attack_step_bernoullis = bool_field(key, v)?
                }
                _ => {
                    return Err(SettingsError::UnknownField {
                        settings: "MalSimulatorSettings",
                        field: key.clone(),
                    })
                }
            }
        }
        Ok(settings)
    }
}

fn invalid(field: &str, reason: &str) -> SettingsError {
    SettingsError::InvalidValue {
        field: field.to_string(),
        reason: reason.to_string(),
    }
}

fn bool_field(field: &str, value: &Value) -> Result<bool, SettingsError> {
    value
        .as_bool()
        .ok_or_else(|| invalid(field, "expected a boolean"))
}

/// One attacker's id-resolved settings, as `Simulator::reset` consumes
/// them - the Rust counterpart of `native_settings.py::
/// flatten_attacker_settings`'s output. `entry_points` is a single,
/// already-sampled set (multiple entry-point sets are sampled by the
/// caller before reset).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FlatAttackerSettings {
    pub entry_points: HashSet<AttackGraphNodeId>,
    pub goals: HashSet<AttackGraphNodeId>,
    /// `None`: no actionability rule (every node actionable).
    pub actionable_steps: Option<HashSet<AttackGraphNodeId>>,
    /// `None`: no TTC overrides.
    pub ttc_dists: Option<HashMap<AttackGraphNodeId, TtcDist>>,
}

/// One defender's id-resolved settings - the Rust counterpart of
/// `native_settings.py::flatten_defender_settings`'s output.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FlatDefenderSettings {
    /// `None`: no actionability rule (every node actionable).
    pub actionable_steps: Option<HashSet<AttackGraphNodeId>>,
    /// `None`: no observability rule (every node observable).
    pub observable_steps: Option<HashSet<AttackGraphNodeId>>,
    pub false_positive_rates: Option<HashMap<AttackGraphNodeId, f64>>,
    pub false_negative_rates: Option<HashMap<AttackGraphNodeId, f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FlatAgentSettings {
    Attacker(FlatAttackerSettings),
    Defender(FlatDefenderSettings),
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn defaults_match_python_dataclass() {
        let s = MalSimulatorSettings::default();
        assert!(!s.uncompromise_untraversable_steps);
        assert_eq!(s.ttc_mode, TtcMode::Disabled);
        assert_eq!(s.seed, None);
        assert!(s.attack_surface.skip_compromised);
        assert!(!s.attack_surface.skip_unnecessary);
        assert!(s.compromise_entrypoints_at_start);
        assert!(s.run_defense_step_bernoullis);
        assert!(s.run_attack_step_bernoullis);
        assert_eq!(MalSimulatorSettings::from_value(&json!({})), Ok(s));
    }

    #[test]
    fn from_value_converts_enum_names_and_nested_attack_surface() {
        let s = MalSimulatorSettings::from_value(&json!({
            "ttc_mode": "PRE_SAMPLE",
            "seed": 42,
            "attack_surface": {"skip_unnecessary": true},
            "run_attack_step_bernoullis": false,
        }))
        .unwrap();
        assert_eq!(s.ttc_mode, TtcMode::PreSample);
        assert_eq!(s.seed, Some(42));
        assert!(s.attack_surface.skip_compromised);
        assert!(s.attack_surface.skip_unnecessary);
        assert!(!s.run_attack_step_bernoullis);
    }

    #[test]
    fn from_value_rejects_unknown_fields_and_values() {
        assert!(matches!(
            MalSimulatorSettings::from_value(&json!({"bogus": 1})),
            Err(SettingsError::UnknownField { .. })
        ));
        assert!(matches!(
            MalSimulatorSettings::from_value(&json!({"ttc_mode": "FAST"})),
            Err(SettingsError::InvalidValue { .. })
        ));
        assert!(matches!(
            MalSimulatorSettings::from_value(&json!({"attack_surface": {"bogus": true}})),
            Err(SettingsError::UnknownField { .. })
        ));
    }

    #[test]
    fn enum_names_round_trip() {
        for mode in [
            TtcMode::EffortBasedPerStepSample,
            TtcMode::PerStepSample,
            TtcMode::PreSample,
            TtcMode::ExpectedValue,
            TtcMode::Disabled,
        ] {
            assert_eq!(TtcMode::from_name(mode.name()), Some(mode));
        }
        for mode in [
            RewardMode::Cumulative,
            RewardMode::OneOff,
            RewardMode::ExpectedTtc,
            RewardMode::SampleTtc,
        ] {
            assert_eq!(RewardMode::from_name(mode.name()), Some(mode));
        }
    }
}
