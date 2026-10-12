// Native acceleration for the hot path of `get_attack_surface`.
//
// Mirrors the semantics of `node_is_blocked` / `node_is_traversable` in
// `malsim/mal_simulator/graph_utils.py`, and is a line-for-line port of the
// PyO3/Rust implementation in rust/src/lib.rs (see that file for the
// original design notes). The graph topology (parents, children, node
// kind, existence status, necessity, impossibility) is static for the
// lifetime of one simulation episode, so it is uploaded once into an
// AttackGraphIndex and then queried many times per step with only the
// dynamic `performed_nodes` / `enabled_defenses` sets crossing the
// Python/C++ boundary.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

enum class Kind : uint8_t {
    And = 0,
    Or = 1,
    Exist = 2,
    NotExist = 3,
    Defense = 4,
    Other = 5,
};

static Kind parse_kind(uint8_t code) {
    switch (code) {
        case 0: return Kind::And;
        case 1: return Kind::Or;
        case 2: return Kind::Exist;
        case 3: return Kind::NotExist;
        case 4: return Kind::Defense;
        case 5: return Kind::Other;
        default:
            throw std::invalid_argument(
                "unknown node kind code " + std::to_string(code));
    }
}

class AttackGraphIndex {
public:
    AttackGraphIndex(
        std::vector<int64_t> ids,
        const std::vector<uint8_t>& kinds,
        std::vector<bool> existence_status,
        std::vector<bool> necessary,
        std::vector<bool> impossible,
        const std::vector<std::vector<int64_t>>& parents
    )
        : ids_(std::move(ids)),
          existence_status_(std::move(existence_status)),
          necessary_(std::move(necessary)),
          impossible_(std::move(impossible))
    {
        const size_t n = ids_.size();
        if (kinds.size() != n || existence_status_.size() != n ||
            necessary_.size() != n || impossible_.size() != n ||
            parents.size() != n) {
            throw std::invalid_argument(
                "all parallel arrays passed to AttackGraphIndex must have "
                "the same length");
        }

        id_to_index_.reserve(n);
        for (size_t idx = 0; idx < n; ++idx) {
            id_to_index_[ids_[idx]] = static_cast<uint32_t>(idx);
        }

        kind_.reserve(n);
        for (uint8_t code : kinds) {
            kind_.push_back(parse_kind(code));
        }

        parents_.reserve(n);
        for (const auto& parent_ids : parents) {
            std::vector<uint32_t> resolved;
            resolved.reserve(parent_ids.size());
            for (int64_t id : parent_ids) {
                auto it = id_to_index_.find(id);
                if (it == id_to_index_.end()) {
                    throw std::invalid_argument(
                        "unknown parent node id " + std::to_string(id));
                }
                resolved.push_back(it->second);
            }
            parents_.push_back(std::move(resolved));
        }
    }

    // Filter `candidate_ids` down to those that are traversable given the
    // current `performed_ids` and `enabled_defense_ids`. Order of the
    // returned ids matches the input order.
    std::vector<int64_t> filter_traversable(
        const std::vector<int64_t>& candidate_ids,
        const std::vector<int64_t>& performed_ids,
        const std::vector<int64_t>& enabled_defense_ids
    ) const {
        auto performed = to_index_set(performed_ids);
        auto enabled_defenses = to_index_set(enabled_defense_ids);

        std::vector<int64_t> result;
        result.reserve(candidate_ids.size());
        for (int64_t id : candidate_ids) {
            uint32_t idx = index_of(id);
            if (is_traversable(idx, performed, enabled_defenses)) {
                result.push_back(id);
            }
        }
        return result;
    }

    // Single-node traversability check, used by effect-propagation BFS.
    bool is_traversable_single(
        int64_t node_id,
        const std::vector<int64_t>& performed_ids,
        const std::vector<int64_t>& enabled_defense_ids
    ) const {
        auto performed = to_index_set(performed_ids);
        auto enabled_defenses = to_index_set(enabled_defense_ids);
        return is_traversable(index_of(node_id), performed, enabled_defenses);
    }

    // Support `pickle` (and therefore `copy.deepcopy` and sending a
    // `MalSimulator`/`GraphState` across a multiprocessing boundary, as
    // vectorized Gymnasium/PettingZoo envs do) by reconstructing the index
    // from the same parallel arrays the constructor takes.
    py::tuple getstate() const {
        std::vector<uint8_t> kinds;
        kinds.reserve(kind_.size());
        for (Kind k : kind_) {
            kinds.push_back(static_cast<uint8_t>(k));
        }

        std::vector<std::vector<int64_t>> parents;
        parents.reserve(parents_.size());
        for (const auto& ps : parents_) {
            std::vector<int64_t> ids;
            ids.reserve(ps.size());
            for (uint32_t idx : ps) {
                ids.push_back(ids_[idx]);
            }
            parents.push_back(std::move(ids));
        }

        return py::make_tuple(
            ids_, kinds, existence_status_, necessary_, impossible_, parents);
    }

private:
    // Whether `parent_idx`, acting as a parent, blocks traversal of a
    // child (mirrors `_node_blocks_children` in graph_utils.py).
    bool parent_blocks(
        uint32_t parent_idx,
        const std::unordered_set<uint32_t>& enabled_defenses
    ) const {
        switch (kind_[parent_idx]) {
            case Kind::Exist:
                return !existence_status_[parent_idx];
            case Kind::NotExist:
                return existence_status_[parent_idx];
            case Kind::Defense:
                return enabled_defenses.count(parent_idx) != 0;
            default:
                return false;
        }
    }

    bool is_blocked(
        uint32_t idx,
        const std::unordered_set<uint32_t>& enabled_defenses
    ) const {
        if (impossible_[idx]) {
            return true;
        }
        const auto& parents = parents_[idx];
        switch (kind_[idx]) {
            case Kind::And:
                for (uint32_t p : parents) {
                    if (parent_blocks(p, enabled_defenses)) return true;
                }
                return false;
            case Kind::Or:
                for (uint32_t p : parents) {
                    if (!parent_blocks(p, enabled_defenses)) return false;
                }
                return true;
            default:
                return false;
        }
    }

    bool is_traversable(
        uint32_t idx,
        const std::unordered_set<uint32_t>& performed,
        const std::unordered_set<uint32_t>& enabled_defenses
    ) const {
        Kind k = kind_[idx];
        if (k != Kind::And && k != Kind::Or) {
            return false;
        }
        if (is_blocked(idx, enabled_defenses)) {
            return false;
        }
        const auto& parents = parents_[idx];
        bool parents_reached = false;
        for (uint32_t p : parents) {
            if (performed.count(p)) {
                parents_reached = true;
                break;
            }
        }
        if (!parents_reached) {
            return false;
        }
        if (k == Kind::Or) {
            return true;
        }
        // Kind::And
        for (uint32_t p : parents) {
            if (necessary_[p] && !performed.count(p)) {
                return false;
            }
        }
        return true;
    }

    uint32_t index_of(int64_t id) const {
        auto it = id_to_index_.find(id);
        if (it == id_to_index_.end()) {
            throw std::invalid_argument("unknown node id " + std::to_string(id));
        }
        return it->second;
    }

    std::unordered_set<uint32_t> to_index_set(
        const std::vector<int64_t>& ids
    ) const {
        std::unordered_set<uint32_t> result;
        result.reserve(ids.size());
        for (int64_t id : ids) {
            result.insert(index_of(id));
        }
        return result;
    }

    // Dense index -> node id, kept around so the index can be pickled.
    std::vector<int64_t> ids_;
    std::vector<Kind> kind_;
    // Only meaningful for Kind::Exist / Kind::NotExist.
    std::vector<bool> existence_status_;
    std::vector<bool> necessary_;
    std::vector<bool> impossible_;
    std::vector<std::vector<uint32_t>> parents_;
    std::unordered_map<int64_t, uint32_t> id_to_index_;
};

PYBIND11_MODULE(malsim_native, m) {
    py::class_<AttackGraphIndex>(m, "AttackGraphIndex")
        .def(py::init<
                 std::vector<int64_t>,
                 const std::vector<uint8_t>&,
                 std::vector<bool>,
                 std::vector<bool>,
                 std::vector<bool>,
                 const std::vector<std::vector<int64_t>>&>(),
             py::arg("ids"),
             py::arg("kinds"),
             py::arg("existence_status"),
             py::arg("necessary"),
             py::arg("impossible"),
             py::arg("parents"))
        .def("filter_traversable", &AttackGraphIndex::filter_traversable,
             py::arg("candidate_ids"),
             py::arg("performed_ids"),
             py::arg("enabled_defense_ids"))
        .def("is_traversable_single", &AttackGraphIndex::is_traversable_single,
             py::arg("node_id"),
             py::arg("performed_ids"),
             py::arg("enabled_defense_ids"))
        .def(py::pickle(
            [](const AttackGraphIndex& self) { return self.getstate(); },
            [](py::tuple t) {
                if (t.size() != 6) {
                    throw std::runtime_error("invalid AttackGraphIndex pickle state");
                }
                return AttackGraphIndex(
                    t[0].cast<std::vector<int64_t>>(),
                    t[1].cast<std::vector<uint8_t>>(),
                    t[2].cast<std::vector<bool>>(),
                    t[3].cast<std::vector<bool>>(),
                    t[4].cast<std::vector<bool>>(),
                    t[5].cast<std::vector<std::vector<int64_t>>>());
            }));
}
