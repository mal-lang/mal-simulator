//! Rust port of the scenario-file half of `python/malsim/scenario/scenario.py`
//! (`load_scenario_dict`, `_extend_scenario`, `recursive_update`,
//! `path_relative_to_file_dir` and `_validate_scenario_dict`), see
//! `PORTING_NOTES.md` §7 C2.
//!
//! Scenario dicts are `serde_json::Value`s (§11): YAML is parsed with
//! `serde_yaml` straight into that type, which is also what mal-toolbox's
//! `maltoolbox_model::from_dict` takes for an inline `model:`.

use std::fmt;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

/// Port of `deprecated_fields`.
pub const DEPRECATED_FIELDS: [&str; 8] = [
    "attacker_agent_class",
    "defender_agent_class",
    "attacker_entry_points",
    "rewards",
    "observable_steps",
    "actionable_steps",
    "false_positive_rates",
    "false_negative_rates",
];

/// One entry of `required_fields`: a single required field, or a pair of
/// which exactly one must be present.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequiredField {
    One(&'static str),
    OneOf(&'static str, &'static str),
}

/// Port of `required_fields`.
pub const REQUIRED_FIELDS: [RequiredField; 3] = [
    RequiredField::OneOf("agents", "agent_settings"),
    RequiredField::One("lang_file"),
    RequiredField::OneOf("model_file", "model"),
];

/// Port of `allowed_fields` (already flattened, as
/// `_validate_scenario_dict` uses it).
pub const ALLOWED_FIELDS: [&str; 6] = [
    "agents",
    "agent_settings",
    "lang_file",
    "model_file",
    "model",
    "sim_settings",
];

#[derive(Debug)]
pub enum ScenarioFileError {
    Io {
        path: PathBuf,
        source: io::Error,
    },
    Yaml {
        path: PathBuf,
        source: serde_yaml::Error,
    },
    /// The scenario file's top level is not a mapping.
    NotAMapping(PathBuf),
    /// A path-valued field (`extends`, `lang_file`, `model_file`) is not a
    /// string.
    PathNotAString(&'static str),
    /// Python's `SyntaxError("Scenario setting '...' is deprecated ...")`.
    DeprecatedField(String),
    /// Python's `SyntaxError("Scenario setting '...' is not supported")`.
    UnsupportedField(String),
    /// Python's `RuntimeError("Setting '...' required in scenario file")`.
    MissingField(&'static str),
    /// Python's `RuntimeError("One of '(...)' is required in scenario file")`.
    MissingOneOf(&'static str, &'static str),
    /// Python's `RuntimeError("Only one of '(...)' is allowed in scenario file")`.
    BothOf(&'static str, &'static str),
}

impl fmt::Display for ScenarioFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScenarioFileError::Io { path, source } => {
                write!(
                    f,
                    "Could not read scenario file {}: {source}",
                    path.display()
                )
            }
            ScenarioFileError::Yaml { path, source } => {
                write!(
                    f,
                    "Could not parse scenario file {}: {source}",
                    path.display()
                )
            }
            ScenarioFileError::NotAMapping(path) => {
                write!(f, "Scenario file {} must contain a mapping", path.display())
            }
            ScenarioFileError::PathNotAString(key) => {
                write!(f, "Scenario setting '{key}' must be a string path")
            }
            ScenarioFileError::DeprecatedField(key) => write!(
                f,
                "Scenario setting '{key}' is deprecated, see README or ./tests/testdata/scenarios"
            ),
            ScenarioFileError::UnsupportedField(key) => {
                write!(f, "Scenario setting '{key}' is not supported")
            }
            ScenarioFileError::MissingField(key) => {
                write!(f, "Setting '{key}' required in scenario file")
            }
            ScenarioFileError::MissingOneOf(a, b) => {
                write!(f, "One of '('{a}', '{b}')' is required in scenario file")
            }
            ScenarioFileError::BothOf(a, b) => {
                write!(
                    f,
                    "Only one of '('{a}', '{b}')' is allowed in scenario file"
                )
            }
        }
    }
}

impl std::error::Error for ScenarioFileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ScenarioFileError::Io { source, .. } => Some(source),
            ScenarioFileError::Yaml { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Port of `_validate_scenario_dict`: rejects deprecated/unsupported keys,
/// then checks required keys (and that exactly one of each pair is given).
pub fn validate_scenario_dict(scenario: &Map<String, Value>) -> Result<(), ScenarioFileError> {
    for key in scenario.keys() {
        if DEPRECATED_FIELDS.contains(&key.as_str()) {
            return Err(ScenarioFileError::DeprecatedField(key.clone()));
        }
        if !ALLOWED_FIELDS.contains(&key.as_str()) {
            return Err(ScenarioFileError::UnsupportedField(key.clone()));
        }
    }

    for required in REQUIRED_FIELDS {
        match required {
            RequiredField::OneOf(a, b) => {
                let (has_a, has_b) = (scenario.contains_key(a), scenario.contains_key(b));
                if !has_a && !has_b {
                    return Err(ScenarioFileError::MissingOneOf(a, b));
                }
                if has_a && has_b {
                    return Err(ScenarioFileError::BothOf(a, b));
                }
            }
            RequiredField::One(key) => {
                if !scenario.contains_key(key) {
                    return Err(ScenarioFileError::MissingField(key));
                }
            }
        }
    }
    Ok(())
}

/// Port of `recursive_update`: deep-merges `new` over `old`. Where both
/// sides hold a mapping the merge recurses; otherwise `new`'s value wins
/// if present. An explicit `null` in `new` removes the key altogether.
pub fn recursive_update(old: &Map<String, Value>, new: &Map<String, Value>) -> Map<String, Value> {
    let mut combined = Map::new();
    for key in old
        .keys()
        .chain(new.keys().filter(|k| !old.contains_key(*k)))
    {
        // Explicit `null` allows overriding values to None (i.e. removing them).
        if matches!(new.get(key), Some(Value::Null)) {
            continue;
        }
        let merged = match (old.get(key), new.get(key)) {
            (Some(Value::Object(old_sub)), Some(Value::Object(new_sub))) => {
                Value::Object(recursive_update(old_sub, new_sub))
            }
            (_, Some(new_value)) => new_value.clone(),
            (Some(old_value), None) => old_value.clone(),
            (None, None) => unreachable!("key comes from one of the two maps"),
        };
        combined.insert(key.clone(), merged);
    }
    combined
}

/// Port of `path_relative_to_file_dir`: `rel_path` joined onto the
/// directory of the (symlink-resolved) `file`. Like `os.path.join`, an
/// absolute `rel_path` is returned unchanged.
pub fn path_relative_to_file_dir(
    rel_path: &str,
    file: &Path,
) -> Result<PathBuf, ScenarioFileError> {
    let real_file = file
        .canonicalize()
        .map_err(|source| ScenarioFileError::Io {
            path: file.to_path_buf(),
            source,
        })?;
    let file_dir = real_file.parent().unwrap_or_else(|| Path::new("/"));
    Ok(file_dir.join(rel_path))
}

/// Port of `_extend_scenario`: loads the scenario at
/// `original_scenario_path`, merges `overriding_scenario` over it and
/// drops the `extends` key.
fn extend_scenario(
    original_scenario_path: &Path,
    overriding_scenario: &Map<String, Value>,
) -> Result<Map<String, Value>, ScenarioFileError> {
    let original_scenario = load_scenario_dict(original_scenario_path)?;
    let mut resulting_scenario = recursive_update(&original_scenario, overriding_scenario);
    resulting_scenario.remove("extends");
    Ok(resulting_scenario)
}

fn read_yaml_mapping(scenario_file: &Path) -> Result<Map<String, Value>, ScenarioFileError> {
    let file = File::open(scenario_file).map_err(|source| ScenarioFileError::Io {
        path: scenario_file.to_path_buf(),
        source,
    })?;
    let value: Value = serde_yaml::from_reader(file).map_err(|source| ScenarioFileError::Yaml {
        path: scenario_file.to_path_buf(),
        source,
    })?;
    match value {
        Value::Object(map) => Ok(map),
        _ => Err(ScenarioFileError::NotAMapping(scenario_file.to_path_buf())),
    }
}

fn path_field<'a>(
    scenario: &'a Map<String, Value>,
    key: &'static str,
) -> Result<Option<&'a str>, ScenarioFileError> {
    match scenario.get(key) {
        None => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(_) => Err(ScenarioFileError::PathNotAString(key)),
    }
}

/// Port of `load_scenario_dict`: reads a scenario YAML file, resolves
/// `extends` (recursively, relative to this file), and makes `lang_file`
/// (unless it's a `git@` URL) and `model_file` paths relative to this
/// file's directory.
///
/// Does not validate the keys - like Python, that happens when the dict
/// is turned into a scenario (`validate_scenario_dict`).
pub fn load_scenario_dict(
    scenario_file: impl AsRef<Path>,
) -> Result<Map<String, Value>, ScenarioFileError> {
    let scenario_file = scenario_file.as_ref();
    let mut scenario = read_yaml_mapping(scenario_file)?;

    if let Some(extends) = path_field(&scenario, "extends")? {
        let original_scenario_path = path_relative_to_file_dir(extends, scenario_file)?;
        scenario = extend_scenario(&original_scenario_path, &scenario)?;
    }

    // Convert path relative to scenario file if lang_file is a local file.
    // (Python indexes `scenario['lang_file']` unconditionally and raises a
    // `KeyError` when it's missing; here a missing `lang_file` is left for
    // `validate_scenario_dict` to report.)
    if let Some(lang_file) = path_field(&scenario, "lang_file")? {
        if !lang_file.starts_with("git@") {
            let resolved = path_relative_to_file_dir(lang_file, scenario_file)?;
            scenario.insert("lang_file".into(), path_to_value(&resolved));
        }
    }

    if let Some(model_file) = path_field(&scenario, "model_file")? {
        let resolved = path_relative_to_file_dir(model_file, scenario_file)?;
        scenario.insert("model_file".into(), path_to_value(&resolved));
    }

    Ok(scenario)
}

fn path_to_value(path: &Path) -> Value {
    Value::String(path.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn scenarios_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/testdata/scenarios")
    }

    fn obj(value: Value) -> Map<String, Value> {
        match value {
            Value::Object(map) => map,
            _ => panic!("expected an object"),
        }
    }

    #[test]
    fn recursive_update_merges_nested_mappings() {
        let old = obj(json!({
            "a": 1,
            "nested": {"keep": 1, "override": 1, "deeper": {"x": 1}},
        }));
        let new = obj(json!({
            "b": 2,
            "nested": {"override": 2, "deeper": {"y": 2}},
        }));
        assert_eq!(
            Value::Object(recursive_update(&old, &new)),
            json!({
                "a": 1,
                "b": 2,
                "nested": {"keep": 1, "override": 2, "deeper": {"x": 1, "y": 2}},
            })
        );
    }

    #[test]
    fn recursive_update_explicit_null_removes_key() {
        let old =
            obj(json!({"agents": {"Attacker1": {"type": "attacker"}, "Defender1": {}}, "x": 1}));
        let new = obj(json!({"agents": {"Attacker1": null}, "x": null, "y": null}));
        assert_eq!(
            Value::Object(recursive_update(&old, &new)),
            json!({"agents": {"Defender1": {}}})
        );
    }

    #[test]
    fn recursive_update_non_mapping_replaces_mapping_and_vice_versa() {
        let old = obj(json!({"a": {"x": 1}, "b": 1, "c": [1, 2]}));
        let new = obj(json!({"a": 5, "b": {"y": 2}, "c": [3]}));
        assert_eq!(
            Value::Object(recursive_update(&old, &new)),
            json!({"a": 5, "b": {"y": 2}, "c": [3]})
        );
    }

    #[test]
    fn validate_rejects_deprecated_and_unsupported_fields() {
        let base = json!({"agents": {}, "lang_file": "x", "model_file": "y"});
        assert!(validate_scenario_dict(&obj(base.clone())).is_ok());

        let mut deprecated = obj(base.clone());
        deprecated.insert("rewards".into(), json!({}));
        assert!(matches!(
            validate_scenario_dict(&deprecated),
            Err(ScenarioFileError::DeprecatedField(k)) if k == "rewards"
        ));

        let mut unsupported = obj(base);
        unsupported.insert("bogus".into(), json!(1));
        assert!(matches!(
            validate_scenario_dict(&unsupported),
            Err(ScenarioFileError::UnsupportedField(k)) if k == "bogus"
        ));
    }

    #[test]
    fn validate_checks_required_fields() {
        assert!(matches!(
            validate_scenario_dict(&obj(json!({"lang_file": "x", "model_file": "y"}))),
            Err(ScenarioFileError::MissingOneOf("agents", "agent_settings"))
        ));
        assert!(matches!(
            validate_scenario_dict(&obj(json!({"agents": {}, "model_file": "y"}))),
            Err(ScenarioFileError::MissingField("lang_file"))
        ));
        assert!(matches!(
            validate_scenario_dict(&obj(json!({
                "agents": {}, "lang_file": "x", "model_file": "y", "model": {}
            }))),
            Err(ScenarioFileError::BothOf("model_file", "model"))
        ));
    }

    #[test]
    fn load_resolves_paths_relative_to_scenario_file() {
        let scenario =
            load_scenario_dict(scenarios_dir().join("traininglang_scenario.yml")).unwrap();
        let lang_file = PathBuf::from(scenario["lang_file"].as_str().unwrap());
        let model_file = PathBuf::from(scenario["model_file"].as_str().unwrap());
        assert!(lang_file.is_absolute() && lang_file.exists());
        assert!(model_file.is_absolute() && model_file.exists());
        assert!(lang_file.ends_with("langs/org.mal-lang.trainingLang-1.0.0.mar"));
    }

    #[test]
    fn load_keeps_git_lang_file_untouched() {
        let scenario =
            load_scenario_dict(scenarios_dir().join("simple_scenario_git_url.yml")).unwrap();
        assert!(scenario["lang_file"].as_str().unwrap().starts_with("git@"));
    }

    #[test]
    fn extend_scenario_overrides_rewards() {
        // Port of `test_extend_scenario`'s dict-level half.
        let scenario =
            load_scenario_dict(scenarios_dir().join("traininglang_scenario_extended.yml")).unwrap();
        assert!(!scenario.contains_key("extends"));
        let agents = scenario["agents"].as_object().unwrap();
        assert_eq!(agents.len(), 2);
        let defender_rewards = &agents["Defender1"]["rewards"]["by_asset_name"];
        assert_eq!(defender_rewards["Host:0"]["notPresent"], json!(1));
        assert_eq!(defender_rewards["Data:2"]["modify"], json!(1));
        assert!(PathBuf::from(scenario["lang_file"].as_str().unwrap()).exists());
    }

    #[test]
    fn extend_scenario_deeper_removes_null_agent() {
        // Port of `test_extend_scenario_deeper`'s dict-level half.
        let scenario = load_scenario_dict(
            scenarios_dir().join("sub/traininglang_scenario_extended_again.yml"),
        )
        .unwrap();
        let agents = scenario["agents"].as_object().unwrap();
        assert_eq!(agents.keys().collect::<Vec<_>>(), vec!["Defender1"]);
        assert!(PathBuf::from(scenario["lang_file"].as_str().unwrap()).exists());
        assert!(PathBuf::from(scenario["model_file"].as_str().unwrap()).exists());
    }

    #[test]
    fn extend_scenario_override_lang_model_resolves_against_overriding_file() {
        let scenario = load_scenario_dict(
            scenarios_dir().join("sub/traininglang_scenario_override_lang_model.yml"),
        )
        .unwrap();
        let lang_file = PathBuf::from(scenario["lang_file"].as_str().unwrap());
        let model_file = PathBuf::from(scenario["model_file"].as_str().unwrap());
        assert!(lang_file.exists() && model_file.exists());
        // Relative to `sub/`, not to the extended scenario's directory.
        assert!(lang_file.starts_with(scenarios_dir().canonicalize().unwrap().join("sub")));
        assert_eq!(scenario["agents"].as_object().unwrap().len(), 2);
    }

    /// Every `.yml` under `tests/testdata/scenarios`, recursively.
    fn fixture_paths(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                fixture_paths(&path, out);
            } else if path.extension().and_then(|e| e.to_str()) == Some("yml") {
                out.push(path);
            }
        }
    }

    #[test]
    fn every_fixture_loads_and_validates_like_python() {
        // These three still use the deprecated top-level `rewards` key;
        // Python's `_validate_scenario_dict` rejects them the same way.
        const DEPRECATED_FORMAT: [&str; 3] = [
            "no_entry_points_simple_scenario.yml",
            "no_existing_attacker_in_model_scenario.yml",
            "run_demo_scenario.yml",
        ];
        let mut paths = Vec::new();
        fixture_paths(&scenarios_dir(), &mut paths);
        assert!(paths.len() > 30);
        for path in paths {
            let scenario =
                load_scenario_dict(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let file_name = path.file_name().unwrap().to_str().unwrap();
            let result = validate_scenario_dict(&scenario);
            if DEPRECATED_FORMAT.contains(&file_name) {
                assert!(
                    matches!(result, Err(ScenarioFileError::DeprecatedField(ref k)) if k == "rewards"),
                    "{file_name}: {result:?}"
                );
            } else {
                result.unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            }
        }
    }
}
