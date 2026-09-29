//! Native acceleration for the hot path of `get_attack_surface`.
//!
//! Mirrors the semantics of `node_is_blocked` / `node_is_traversable` in
//! `malsim/mal_simulator/graph_utils.py`. The graph topology (parents,
//! children, node kind, existence status, necessity, impossibility) is
//! static for the lifetime of one simulation episode, so it is uploaded
//! once into an `AttackGraphIndex` and then queried many times per step
//! with only the dynamic `performed_nodes` / `enabled_defenses` sets
//! crossing the Python/Rust boundary.

use std::collections::{HashMap, HashSet};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    And,
    Or,
    Exist,
    NotExist,
    Defense,
    Other,
}

fn parse_kind(code: u8) -> PyResult<Kind> {
    match code {
        0 => Ok(Kind::And),
        1 => Ok(Kind::Or),
        2 => Ok(Kind::Exist),
        3 => Ok(Kind::NotExist),
        4 => Ok(Kind::Defense),
        5 => Ok(Kind::Other),
        other => Err(PyValueError::new_err(format!(
            "unknown node kind code {other}"
        ))),
    }
}

#[pyclass(module = "malsim_native")]
struct AttackGraphIndex {
    /// Dense index -> node id, kept around so the index can be pickled
    /// (e.g. when a `MalSimulator` is sent across a process boundary by
    /// vectorized Gymnasium/PettingZoo envs).
    ids: Vec<i64>,
    kind: Vec<Kind>,
    /// Only meaningful for Kind::Exist / Kind::NotExist.
    existence_status: Vec<bool>,
    necessary: Vec<bool>,
    impossible: Vec<bool>,
    parents: Vec<Vec<u32>>,
    id_to_index: HashMap<i64, u32>,
}

impl AttackGraphIndex {
    fn to_indices(&self, ids: &[i64]) -> PyResult<Vec<u32>> {
        ids.iter()
            .map(|id| {
                self.id_to_index
                    .get(id)
                    .copied()
                    .ok_or_else(|| PyValueError::new_err(format!("unknown node id {id}")))
            })
            .collect()
    }

    fn to_index_set(&self, ids: &[i64]) -> PyResult<HashSet<u32>> {
        Ok(self.to_indices(ids)?.into_iter().collect())
    }

    /// Whether `parent_idx`, acting as a parent, blocks traversal of a child
    /// (mirrors `_node_blocks_children` in graph_utils.py).
    fn parent_blocks(&self, parent_idx: u32, enabled_defenses: &HashSet<u32>) -> bool {
        match self.kind[parent_idx as usize] {
            Kind::Exist => !self.existence_status[parent_idx as usize],
            Kind::NotExist => self.existence_status[parent_idx as usize],
            Kind::Defense => enabled_defenses.contains(&parent_idx),
            Kind::And | Kind::Or | Kind::Other => false,
        }
    }

    fn is_blocked(&self, idx: u32, enabled_defenses: &HashSet<u32>) -> bool {
        let i = idx as usize;
        if self.impossible[i] {
            return true;
        }
        match self.kind[i] {
            Kind::And => self.parents[i]
                .iter()
                .any(|&p| self.parent_blocks(p, enabled_defenses)),
            Kind::Or => self.parents[i]
                .iter()
                .all(|&p| self.parent_blocks(p, enabled_defenses)),
            _ => false,
        }
    }

    fn is_traversable(&self, idx: u32, performed: &HashSet<u32>, enabled_defenses: &HashSet<u32>) -> bool {
        let i = idx as usize;
        if !matches!(self.kind[i], Kind::And | Kind::Or) {
            return false;
        }
        if self.is_blocked(idx, enabled_defenses) {
            return false;
        }
        let parents_reached = self.parents[i].iter().any(|p| performed.contains(p));
        if !parents_reached {
            return false;
        }
        match self.kind[i] {
            Kind::Or => true,
            Kind::And => self.parents[i]
                .iter()
                .filter(|&&p| self.necessary[p as usize])
                .all(|p| performed.contains(p)),
            _ => unreachable!(),
        }
    }
}

#[pymethods]
impl AttackGraphIndex {
    #[new]
    fn new(
        ids: Vec<i64>,
        kinds: Vec<u8>,
        existence_status: Vec<bool>,
        necessary: Vec<bool>,
        impossible: Vec<bool>,
        parents: Vec<Vec<i64>>,
    ) -> PyResult<Self> {
        let n = ids.len();
        if kinds.len() != n
            || existence_status.len() != n
            || necessary.len() != n
            || impossible.len() != n
            || parents.len() != n
        {
            return Err(PyValueError::new_err(
                "all parallel arrays passed to AttackGraphIndex must have the same length",
            ));
        }

        let mut id_to_index = HashMap::with_capacity(n);
        for (idx, &id) in ids.iter().enumerate() {
            id_to_index.insert(id, idx as u32);
        }

        let kind = kinds
            .iter()
            .copied()
            .map(parse_kind)
            .collect::<PyResult<Vec<_>>>()?;

        let mut index = AttackGraphIndex {
            ids,
            kind,
            existence_status,
            necessary,
            impossible,
            parents: Vec::with_capacity(n),
            id_to_index,
        };

        for parent_ids in &parents {
            let mut resolved = Vec::with_capacity(parent_ids.len());
            for id in parent_ids {
                let idx = *index.id_to_index.get(id).ok_or_else(|| {
                    PyValueError::new_err(format!("unknown parent node id {id}"))
                })?;
                resolved.push(idx);
            }
            index.parents.push(resolved);
        }

        Ok(index)
    }

    /// Filter `candidate_ids` down to those that are traversable given the
    /// current `performed_ids` and `enabled_defense_ids`. Order of the
    /// returned ids matches the input order.
    fn filter_traversable(
        &self,
        candidate_ids: Vec<i64>,
        performed_ids: Vec<i64>,
        enabled_defense_ids: Vec<i64>,
    ) -> PyResult<Vec<i64>> {
        let performed = self.to_index_set(&performed_ids)?;
        let enabled_defenses = self.to_index_set(&enabled_defense_ids)?;

        let mut result = Vec::with_capacity(candidate_ids.len());
        for id in candidate_ids {
            let idx = *self
                .id_to_index
                .get(&id)
                .ok_or_else(|| PyValueError::new_err(format!("unknown node id {id}")))?;
            if self.is_traversable(idx, &performed, &enabled_defenses) {
                result.push(id);
            }
        }
        Ok(result)
    }

    /// Single-node traversability check, used by effect-propagation BFS.
    fn is_traversable_single(
        &self,
        node_id: i64,
        performed_ids: Vec<i64>,
        enabled_defense_ids: Vec<i64>,
    ) -> PyResult<bool> {
        let performed = self.to_index_set(&performed_ids)?;
        let enabled_defenses = self.to_index_set(&enabled_defense_ids)?;
        let idx = *self
            .id_to_index
            .get(&node_id)
            .ok_or_else(|| PyValueError::new_err(format!("unknown node id {node_id}")))?;
        Ok(self.is_traversable(idx, &performed, &enabled_defenses))
    }

    /// Support `pickle` (and therefore `copy.deepcopy` and sending a
    /// `MalSimulator`/`GraphState` across a multiprocessing boundary, as
    /// vectorized Gymnasium/PettingZoo envs do) by reconstructing the index
    /// from the same parallel arrays the constructor takes.
    fn __reduce__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let kinds: Vec<u8> = self
            .kind
            .iter()
            .map(|k| match k {
                Kind::And => 0u8,
                Kind::Or => 1,
                Kind::Exist => 2,
                Kind::NotExist => 3,
                Kind::Defense => 4,
                Kind::Other => 5,
            })
            .collect();
        let parents: Vec<Vec<i64>> = self
            .parents
            .iter()
            .map(|ps| ps.iter().map(|&idx| self.ids[idx as usize]).collect())
            .collect();

        let cls = py.get_type_bound::<Self>();
        let args = (
            self.ids.clone(),
            kinds,
            self.existence_status.clone(),
            self.necessary.clone(),
            self.impossible.clone(),
            parents,
        );
        Ok((cls, args).into_py(py))
    }
}

#[pymodule]
fn malsim_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AttackGraphIndex>()?;
    Ok(())
}
