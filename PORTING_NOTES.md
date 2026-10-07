# Porting notes: moving malsim's simulator core to Rust

This document is the working plan for porting `MalSimulator`'s (and later
`DynaMalSimulator`'s) hot-path logic from Python into the `core/malsim-core`
/ `py-bindings/malsim-pyo3` Rust crates scaffolded in `Cargo.toml` /
`py-bindings/Cargo.toml`. It is meant to be resumable across sessions:
section 0 tracks what's actually done: keep it current as work lands.

## 0. Status

Legend: `[ ]` not started, `[~]` in progress, `[x]` done.

- [x] Phase 0 - Rust workspace scaffolding (`core/`, `py-bindings/`,
      `malsim-core` depending on `maltoolbox-attackgraph`, `malsim-pyo3`
      depending on `malsim-core` + pyo3). Builds; no logic yet.
- [ ] Phase 0.5 - Architectural decisions confirmed (see §2). Done as of
      2026-10-07; revisit if reality disagrees once code is written.
- [ ] Phase A - `MalSimulator` port (§5)
  - [x] A1 - Shared-graph handle extraction proven end to end. Not via a
        direct `PyAttackGraph` pyclass downcast as §2.2 originally
        specified (that doesn't work across independently-built `cdylib`s
        - see §10) - via a `PyCapsule` mal-toolbox's
        `PyAttackGraph::__inner_capsule__()` hands out instead (upstream
        change landed at mal-toolbox commit `c854d1d6`). `malsim-pyo3` no
        longer depends on `maltoolbox-attackgraph-py` at all (only the
        pure `maltoolbox-attackgraph` crate, for the `AttackGraph` type
        the capsule's pointer is cast to); `pyproject.toml` builds via
        maturin; `tests/test_native.py::test_native_node_count_matches_python`
        is the committed smoke test. See §10 for the full writeup,
        including a real double-free bug the first capsule-consumer
        attempt had and how it was fixed.
  - [x] A2 - Port `TtcDist` + RNG plumbing (`ttc_utils.py`). Landed in
        `core/malsim-core/src/ttc.rs`: `DistFunction`, `Operation`,
        `TtcDist` (`expected_value`, `sample_value`, `success_probability`,
        `attempt_ttc_with_effort`, `attempt_bernoulli`, `from_dict`/
        `to_dict` via `serde_json::Value`), `named_ttc_dist`. Deliberately
        excludes `default_ttc_dist`/`TTCDist.from_node`/`from_name` (graph-
        node-dependent, scoped to A3 instead - see §4/§5's A3 description)
        so this module stays independent of `maltoolbox_attackgraph` and
        unit-testable alone. Backed by the `statrs` crate (CDF/mean/
        sampling for Bernoulli/Exp/Binomial/Gamma/LogNormal/Uniform) +
        `rand` for RNG plumbing - see §10 for why and the parameter-order
        gotchas found doing this (Binomial, Gamma). 26 Rust-native tests
        in `ttc.rs`'s `#[cfg(test)]` module: one per `DistFunction` variant
        for `expected_value` (exact, since it's closed-form/non-random),
        the full `combine_with`/`combine_op` composition suite ported 1:1
        from `test_ttc_utils.py::test_all_ttc_distributions` (also exact,
        same reason), `from_dict`/`to_dict` round-trips, error cases, and
        structural/statistical (not exact-value) checks for the RNG-
        touching methods. `cargo test`/`cargo clippy`/`cargo fmt --check`
        all clean. Per §2.1, checked `grep -rn seed= tests/` first - see
        §9 for the specific pre-existing Python tests this flags for
        follow-up at A9 (none of them break *now*, since Python's
        `ttc_utils.py` is untouched until A9 wires native in).
  - [ ] A3 - Port static graph_state computation (`graph_state.py`,
        `graph_processing.py` necessity propagation)
  - [ ] A4 - Port graph traversal predicates (`graph_utils.py` minus
        actionability/reward)
  - [ ] A5 - Port attack surface / defense surface / effects computation
  - [ ] A6 - Port false-alert + detector log generation
  - [ ] A7 - Port attacker_step / defender_step orchestration
  - [ ] A8 - `malsim-pyo3` native `Simulator` pyclass: `reset_native`/
        `step_native` returning plain Python primitives
  - [ ] A9 - Rewrite Python `MalSimulator.reset()`/`.step()` to delegate to
        native, rebuild `AttackerState`/`DefenderState` from native output
  - [ ] A10 - Rewrite remaining `MalSimulator` query methods to read
        native-backed state where needed
  - [ ] A11 - Full existing test suite green with native backend; delete
        now-dead pure-Python hot-path modules (or demote to `#[cfg(test)]`
        oracle comparisons - TBD per-module, see §5)
- [ ] Phase B - `DynaMalSimulator` port (§6, detailed sub-plan TBD after A)
- [ ] Phase C - Rust-only library API (§7)
  - [ ] C1 - Port `NodePropertyRule`'s dict-shape + `.value()`/`.per_node()`
        matching as an independent Rust utility (not shared with the
        Python-path flattening in §2.4 - see §2.6)
  - [ ] C2 - Port scenario YAML loading: field validation, `extends`
        merge (`recursive_update`), path resolution
  - [ ] C3 - Port agent-settings-from-dict construction (entry points,
        goals, resolved rule maps, reward_mode, ttc overrides); `policy`
        field parsed but not instantiated (§2.6)
  - [ ] C4 - Wire scenario loading to mal-toolbox's pure-Rust
        language/model/attack-graph construction (`maltoolbox-language`/
        `maltoolbox-model`/`maltoolbox-attackgraph`, no PyO3)
  - [ ] C5 - `Simulator::reset`/`::step` library API usable standalone
        (original Phase C scope)
  - [ ] C6 - Schema-parity test: run the same scenario YAML fixtures
        (`tests/testdata/scenarios/*.yml`) through both the Python
        `Scenario` and the new Rust loader, compare resulting settings

## 1. Goals and non-goals

**Goals** (from the porting request):
- `MalSimulator` and `DynaMalSimulator`'s `reset()`/`step()` (and the
  other public query methods on them) run on Rust code under the hood.
- `AttackerState`, `DefenderState`, `MalSimulatorState`, `Scenario`,
  `AttackerSettings`/`DefenderSettings`/`MalSimulatorSettings`/
  `NodePropertyRule` stay real, hand-written Python classes - "mirrored
  in Python" - not replaced by native pyclasses. Python users keep
  today's types, `isinstance` checks, pickling, equality, etc.
- Non-breaking: the existing test suite (`tests/test_mal_simulator.py`,
  `tests/test_dyna_mal_simulator.py`, `tests/test_scenario.py`,
  `tests/test_ttc_utils.py`, ~3,850 lines, 155 tests passing today) is the
  compatibility contract and must stay green throughout, not just at the
  end.
- A Rust-only path exists: an embedding Rust program can build a
  simulator and call `reset()`/`step()` without any Python involved.
- That Rust-only path also implements the same functionality as the
  Python `Scenario` class: loading a scenario YAML file (language +
  model + agent settings + sim settings, including `extends` merging)
  into a ready-to-run graph/settings pair, not just stepping an
  already-built graph. See §2.6 and §7.
- Every ported module gets Rust-native tests equivalent to the relevant
  Python tests, where relevant (not just reliance on the Python suite
  staying green) - see §8.
- `python/malsim/envs/`, `python/malsim/policies/`, and
  `python/malsim/visualization/` are explicitly out of scope for porting
  - left exactly as-is - and double as integration tests: if they still
    work unmodified against the new native-backed `MalSimulator`, that's
    strong evidence the public surface didn't shift under them.

**Non-goals** (scoped out, confirmed via §2):
- Bit-identical reproducibility of seeded runs across the Python-RNG →
  Rust-RNG transition (see §2.1).
- Touching the Python `Scenario` class itself (`scenario/scenario.py`)
  - it stays exactly as-is for the Python-bound path, unchanged, per §4.
  (Scope note, not a non-goal: a *separate*, additive Rust implementation
  of the same scenario-loading functionality is in scope for Phase C -
  see §2.6/§7 - since a Rust-only caller has no Python `Scenario` to call
  into. This is new Rust code, not a port/replacement of
  `scenario.py`.)
- Porting reward computation (`rewards.py`) to Rust - see §2.4. It stays
  pure Python, unchanged.
- Decision-agent / policy logic (`policies/`) - explicitly user-pluggable
  Python, can't move to Rust in general. (A *subset* of it might get a
  Rust port in Phase C for the standalone-Rust story, but that's additive,
  not a requirement - see §7.)

## 2. Architectural decisions

### 2.1 RNG / TTC reproducibility: statistically-equivalent only

TTC sampling (`ttc_utils.py`), false-positive/negative generation
(`false_alerts.py`), detector true/false-positive rolls
(`event_logger.py`), and pre-enabled-defense Bernoulli draws all pull from
numpy's `default_rng` (PCG64), repeatedly per step. Reproducing numpy/
scipy's *exact* output for a given seed in Rust is impractical in general:
Gamma/LogNormal sampling in scipy uses rejection algorithms that consume a
variable, data-dependent number of underlying draws per sample, not a
simple inverse-CDF - matching that bit-for-bit would mean reimplementing
scipy's internal C sampling algorithms exactly.

**Decision:** after the port, the same seed may produce a
*different-but-statistically-equivalent* sequence of outcomes. This is a
one-time, documented compatibility break for anyone who recorded "golden"
seeded trajectories before the port - not an ongoing concern. It unblocks
porting all RNG-consuming sampling into Rust (via the `rand`/`rand_distr`
crates) for real speedups.

**Consequence for tests:** any existing test that asserts an *exact*
sampled value or exact sequence of actions for a fixed seed will need to
be identified and either relaxed (assert statistical/structural
properties instead) or re-pinned against the new Rust RNG's output once
ported. Check for these explicitly in A2/A6 rather than assuming none
exist - `grep -n seed= tests/` first.

### 2.2 Shared live graph with mal-toolbox's PyO3 layer

> **Updated after A1 - see §10 for the full story.** The mechanism
> originally specified below (direct `PyAttackGraph.inner` extraction via
> a Rust-level pyclass downcast) does **not** work: PyO3 pyclasses from a
> shared dependency crate get a separate, unrelated type object in every
> independently-built `cdylib` that statically links it, so
> `malsim-pyo3`'s compiled copy of `PyAttackGraph` is never recognized as
> the same type as `maltoolbox-pyo3`'s. What's actually implemented (and
> confirmed working end to end, including under a double-free/leak stress
> test) is a `PyCapsule` handoff: mal-toolbox's `PyAttackGraph` exposes
> `__inner_capsule__()`, and `malsim-pyo3` calls it and reconstructs the
> `Rc<RefCell<AttackGraph>>` from the capsule's pointer. The *goal*
> described below (same live, shared, zero-copy graph on both sides) is
> unchanged and achieved - only the *mechanism* differs from what's
> written here historically. `malsim-pyo3` no longer depends on
> `maltoolbox-attackgraph-py` at all as a result (see §10) - only on the
> pure `maltoolbox-attackgraph` crate.

mal-toolbox's `rust-rewrite` branch already ships `py-bindings/
maltoolbox-attackgraph-py`, whose `PyAttackGraph` wraps
`pub inner: Rc<RefCell<maltoolbox_attackgraph::AttackGraph>>` (plus
`lang_graph_py`/`model_py` handles). Its own doc comment anticipates this
exact need: *"mal-simulator depends on `attack_graph.model` directly"*
and *"malsim's hot loop reads `.parents` per traversability check"*.

**Original decision (superseded, kept for history - see the note
above):** `malsim-pyo3` depends directly on `maltoolbox-attackgraph-py`
(git dependency, `rust-rewrite` branch) and extracts `PyAttackGraph.inner`
to operate on the *same* `Rc<RefCell<AttackGraph>>` Python's
`maltoolbox.AttackGraph` object holds. Zero-copy, always in sync (critical
for `DynaMalSimulator`, where the graph mutates at runtime via model
effects - both the Rust and Python views must see the same mutations
immediately).

**Consequence:** `malsim-core` (pure Rust, no PyO3) depends only on the
plain `maltoolbox-attackgraph` crate and is written against owned/shared
`AttackGraph` values with no PyO3 awareness. The `Rc<RefCell<...>>`
sharing is a `malsim-pyo3`-only concern (see §3) - `malsim-core`'s
`Simulator` always takes `Rc<RefCell<AttackGraph>>` as its graph handle
type (even for the pure-Rust §7 path, which just constructs that handle
itself, owning the only reference) so there is exactly one code path, not
two.

**Risk that materialized (see §10):** `malsim-pyo3` would have been
coupled to an *unpublished, same-org, internal* PyO3 crate's struct
layout (`pub inner`, field names) had the direct-downcast approach
worked. It didn't get the chance to bite as a layout-drift risk, because
the approach itself turned out not to work at all for a more fundamental
reason (type identity across `cdylib`s) - superseded by the `PyCapsule`
approach, whose only cross-module contract is a stable name string.

### 2.3 Phasing: MalSimulator fully stabilized before DynaMalSimulator

`MalSimulator` steps a *fixed-shape* graph (TTC/attack-surface/defense
logic only). `DynaMalSimulator` additionally mutates the graph/model at
runtime (`model_effects.py`, `process_assoc_traversal.py`: asset/
association add/remove as a consequence of attack steps) - a materially
bigger and riskier port.

**Decision:** land and verify the complete `MalSimulator` port (Phase A,
including the Python-mirroring boundary pattern and full non-breaking
test parity) before starting `DynaMalSimulator` (Phase B). Phase B reuses
Phase A's proven FFI/mirroring pattern rather than inventing it under
more complexity at once. Phase B's detailed step breakdown is deliberately
*not* fully fleshed out yet (§6) - write it once Phase A's patterns (how
state crosses the boundary, how settings get flattened, how query methods
split between Rust-backed and pure-Python) are proven to actually work.

### 2.4 NodePropertyRule DSL stays resolved in Python; rewards stay pure Python

`NodePropertyRule` (`config/node_property_rule.py`) is a small per-node
DSL (match by asset name/type) used for `rewards`, `actionable_steps`,
`observable_steps`, `false_positive_rates`, `false_negative_rates`, and
`ttc_dists` overrides on `AttackerSettings`/`DefenderSettings`. It already
has `.per_node(attack_graph) -> dict[full_name, value]`, which flattens
any rule against a concrete graph.

**Decision:**
- `NodePropertyRule` itself is **not** ported to Rust. Its DSL evaluation
  stays exactly as today, in Python, unchanged.
- Everywhere a resolved rule is needed inside the Rust hot loop
  (actionability + observability + FP/FN rates + TTC overrides, which
  *are* read per-node, per-step inside attack-surface/defense-surface/
  false-alert computation), Python resolves it to a flat `dict[node_id,
  value]` (or `set[node_id]` for booleans) **once**, at `reset()` time,
  and hands that plain data across the FFI boundary. Rust never parses or
  evaluates the DSL.
- **Rewards are not ported to Rust at all.** `rewards.py`'s reward
  closures are a separate, pull-based API (`sim.agent_reward(state)`),
  not part of `step()`'s return value - confirmed by reading
  `simulator.py`: `step()` returns only `self._agent_states`, and reward
  closures are built once in `__init__` and called on demand. They
  operate on cheap set arithmetic (`step_performed_nodes`, etc.) over the
  already-mirrored Python dataclasses. There's no performance reason to
  move this, and leaving it alone removes an entire category of
  porting/non-breaking risk for zero cost. `node_reward()`/
  `NodePropertyRule[float]` for rewards stay untouched, pure Python.
- `TTCDist` (`ttc_utils.py`) is the one exception that's genuinely
  two-sided: it must stay a real, user-constructible Python class (users
  author `ttc_dists` overrides as `TTCDist(...)` in scenario config), but
  its *sampling* needs a Rust-side equivalent for the hot loop. Bridge via
  the existing `TTCDist.to_dict()`/`from_dict()` methods: Python resolves
  `ttc_dists: NodePropertyRule[TTCDist]` to `dict[node_id, ttc_dict]`
  (via `.per_node()` + `.to_dict()` per value) and hands that plain dict
  across; Rust parses the same shape into its own `TtcDist` enum. The
  default (non-overridden) per-node TTC distribution is derived from
  `node.ttc`/`node.type` already available on the (shared) graph, so most
  nodes need no override entry at all.

This resolves what would otherwise have been a 5th architectural question
without needing to ask it - the design is well-constrained by what's
actually in the hot loop vs. not.

### 2.5 Internal freedoms that don't affect the public contract

Not decisions requiring sign-off, but worth recording so nobody
"fixes" them later thinking they're bugs:

- `AttackerState.previous_state`/`DefenderState.previous_state` currently
  form an unbounded linked list back through every prior step (each new
  state point to the literal previous state object). Confirmed via
  `grep -rn previous_state` that nothing outside the `*_state_factories.py`
  one-step-back `step_*` properties ever reads more than one level deep.
  The Rust-backed implementation is free to *not* reconstruct this chain
  beyond one level (i.e. only ever set `previous_state` to the immediately
  prior Python state object, never deeper) - this is an internal memory
  characteristic, not observable behavior, and the existing behavior
  already doesn't rely on deep traversal.
- `MalSimulatorSettings.uncompromise_untraversable_steps` is defined but
  not read anywhere in the current implementation (verified by grep) -
  carry it through unchanged (mirror the field), don't implement new
  behavior for it as part of this port.

### 2.6 Rust-only path needs its own Scenario + NodePropertyRule equivalent

§2.4 decided that `NodePropertyRule` stays Python-only, with Python
flattening it to plain data before crossing into `malsim-pyo3`. That
decision assumed Python is always present to do the flattening. The
Rust-only path (Phase C, §7) breaks that assumption: an embedding Rust
program with no Python runtime has no `Scenario`/`NodePropertyRule` to
call into, so if it's going to load a scenario YAML file at all (now a
goal - see §1), it needs its own way to parse the same YAML shapes and
evaluate the same by-asset-type/by-asset-name matching.

**Decision:** port `NodePropertyRule`'s matching semantics
(`by_asset_type`/`by_asset_name` dict lookup, precedence: asset_name >
asset_type > default) to Rust as a **second, independent**
implementation used only by the Rust-only scenario loader - it does
*not* replace or get shared with the Python-side `NodePropertyRule` used
by the PyO3 path (§2.4 still stands for that path: Python flattens,
Rust never parses the DSL there). Two implementations of a small,
well-specified, RNG-free dict-lookup rule is a reasonable, low-risk
trade for keeping §2.4's simplicity on the PyO3 path while still letting
the Rust-only path be genuinely self-sufficient. `NodePropertyRule.to_dict
()`/`from_dict()`'s existing JSON-ish shape is the de facto schema both
implementations target, which keeps them honest against each other.

**Scope boundary - `policy` is not instantiable in Rust.** Scenario YAML's
`agents.<name>.policy` names a Python decision-agent class
(`agent_settings_factories.py`'s `policy_name_to_class`, e.g.
`BreadthFirstAttacker`). Per §7's "library API only" decision, no
built-in policies are ported to Rust. The Rust scenario loader parses and
carries the `policy` field (so round-tripping/inspection works) but never
instantiates or drives anything from it - same as today's Python
`AgentRuntimeMixin.agent`, which is simply not exercised by the Rust-only
path. The embedding Rust caller is responsible for its own action
selection regardless of what a scenario file's `policy` field says.

**What's reused, not reimplemented:** the language/model/attack-graph
construction itself (`create_attack_graph`-equivalent) is *not*
duplicated - it's already available as pure Rust via mal-toolbox's
`maltoolbox-language`/`maltoolbox-model`/`maltoolbox-attackgraph` crates
(the same ones `malsim-core` already depends on / will depend on
transitively). Only malsim's *own* scenario-file schema (the YAML shape
in `tests/testdata/scenarios/*.yml`: `agents`, `sim_settings`, `extends`,
field validation/deprecation) and the `NodePropertyRule` matching above
are new Rust code.

### 2.7 Rust code style, structure, and terminology

The sections above (and §4's table) describe *what* logic moves to Rust
and *why* - they are deliberately silent on *where* it lives inside
`core/malsim-core`/`py-bindings/malsim-pyo3`, because that's a separate
decision with a separate answer:

- **Crate/module layout does not have to mirror `python/malsim`'s file
  layout.** There is no requirement that `ttc_utils.py` becomes `ttc.rs`,
  that `attack_surface.py` becomes `attack_surface.rs`, or that the
  Python package tree (`mal_simulator/`, `config/`, `scenario/`) shows up
  as a matching Rust module tree. Organize the Rust code the way Rust
  code should be organized - by ownership, by what borrows what, by
  trait boundaries - and let that structure diverge from Python's module
  boundaries wherever Rust's idioms want it to. §4's table is a map of
  *content* (which Python behavior ends up native vs. stays Python), not
  a map of *location*.
- **Stay close to a line-for-line port of the logic - written in
  idiomatic Rust, not a syntactic transliteration.** These are not in
  tension and neither one wins at the other's expense. Follow the
  Python implementation's steps, order of operations, and decision
  points closely enough that someone who knows `attack_surface.py` can
  read the Rust function and follow along almost line by line - that
  faithfulness is a real goal of this port, not just a description of
  expected difficulty. At the same time, express each step the way
  idiomatic Rust would: `Result`/`Option` instead of Python's exceptions/
  `None`-checks, iterators instead of manually-indexed loops, real enums
  instead of Python's loosely-typed dict/string dispatch, ownership/
  borrowing used properly. What to avoid is "horrible" Rust that fights
  the language just to look more like the Python source syntactically
  (e.g. a hand-rolled index-based `while` loop where Python used `for`,
  just because that's what the Python line looked like) - that's a
  transliteration, not a port, and it's worse for the original author
  too, since it obscures the logic behind unidiomatic noise instead of
  making it easy to follow.
- **But keep the terminology recognizable.** The whole point of this
  plan is that the original Python author can read the Rust code and
  know what they're looking at. Reuse malsim's own vocabulary for the
  same concepts - `attack_surface`, `defense_surface`, `necessity`,
  `ttc_values`, `action_surface`, `performed_nodes`, `enabled_defenses`,
  `impossible_steps`, and so on - adapted only for Rust naming
  conventions (`snake_case` functions/fields, `PascalCase` types, e.g.
  `TtcDist` not `TTCDist`). Don't rename a concept to different
  vocabulary just because it landed in a different module than its
  Python namesake; a reader should be able to grep malsim's Python
  source for a name and find its Rust counterpart close by in spirit,
  even if not in the same relative path.
- **Divergences get written down here, not as code comments.** When
  porting a module turns up a place where the Rust implementation's
  structure, behavior, or API shape genuinely differs from the Python
  original in a way the original author needs to know - not just "this
  landed in a different file" - record it in this document (§10), the
  same way mal-toolbox's own `PORTING_NOTES.md` keeps a running
  "Architecture: deliberate design differences" / "Confirmed upstream
  bugs fixed, not reproduced" log. A Rust doc comment is for someone
  reading that one file; this document is what the Python author
  actually reads to understand the port, so that's where the callout
  belongs - in addition to, not instead of, normal doc comments that
  explain the Rust code on its own terms.

## 3. Target architecture

```
                      ┌────────────────────────────────┐
 Rust-only caller ──▶ │ core/malsim-core (pure Rust)     │──▶ depends on
                      │  scenario::load_from_file(yaml)  │    maltoolbox-
                      │   - own NodePropertyRule (§2.6)  │    language /
                      │   - field validation / extends   │    -model /
                      │  Simulator<Rc<RefCell<AG>>>       │    -attackgraph
                      └───────────────┬──────────────────┘    (pure Rust)
                                      │ path dependency
                      ┌───────────────┴──────────────┐
 Python caller ──▶    │ py-bindings/malsim-pyo3       │
  (via malsim._native)│  unwraps PyAttackGraph.inner  │──▶ depends on
                      │  (maltoolbox-attackgraph-py)  │    maltoolbox's
                      │  Simulator pyclass:           │    py-bindings
                      │   reset_native()/step_native()│    workspace
                      │   → plain dict/set/float      │    (git dep)
                      └───────────────┬──────────────┘
                                      │ plain Python primitives only
                      ┌───────────────┴──────────────┐
                      │ python/malsim/mal_simulator/  │
                      │  MalSimulator.reset()/.step() │
                      │  rebuilds AttackerState/       │
                      │  DefenderState dataclasses     │
                      │  from native output            │
                      └────────────────────────────────┘
```

Note: the Rust-only `scenario::load_from_file` path and the Python-bound
`py-bindings/malsim-pyo3` path are independent consumers of the same
`Simulator` core - the Python path never calls the Rust scenario loader
(Python keeps using its own `Scenario` class, unchanged, per §2.6), and
the Rust-only path never touches `malsim-pyo3`/PyO3 at all.

(The module paths in the diagram - `scenario::`, `Simulator<...>` - are
illustrative of the dependency/data-flow shape, not a mandated namespace.
Per §2.7, the actual module layout inside each crate is free to be
whatever's idiomatic.)

Key rule for the FFI boundary (both directions): **only plain data
crosses it** - ids (`i64`, matching `maltoolbox_attackgraph::ids::
AttackGraphNodeId`), strings, floats, bools, and flat dicts/sets/lists
thereof. No Python object (beyond the one shared `AttackGraph` handle)
and no Rust struct is ever handed across directly. This is what makes the
Python dataclasses genuinely "mirrored" rather than wrapped: `malsim-pyo3`
hands the Python shim raw ids and primitives; the shim resolves ids to
real `AttackGraphNode` objects via `attack_graph.nodes[id]` (already O(1)
in mal-toolbox) and constructs the unchanged dataclasses itself.

`malsim-core`'s `Simulator` owns:
- `graph: Rc<RefCell<maltoolbox_attackgraph::AttackGraph>>`
- `settings: SimSettings` (port of `MalSimulatorSettings` +
  `AttackSurfaceSettings`; `RewardMode` is **not** included - see §2.4)
- per-episode `GraphState` equivalent: ttc values by node id, pre-enabled
  defenses, impossible attack steps, necessity-by-id (all computed once
  at `reset()`)
- runtime `enabled_defenses: HashSet<NodeId>` (grows during `step()`)
- per-agent runtime structs (`AttackerRuntime`/`DefenderRuntime`): the
  Rust-side equivalent of the mutable parts of `AttackerState`/
  `DefenderState` (performed/attempted/observed node id sets, action
  surface, num_attempts, iteration, performed_nodes_order, logs) plus the
  flattened per-agent settings from §2.4 (actionable ids, observable ids,
  FP/FN rate maps, ttc override map) - **not** rewards.

## 4. File/module inventory & disposition

This table maps *content* - which Python file's behavior ends up native
vs. stays Python - not *location*. None of the parenthetical mentions of
`malsim-core`/`malsim-pyo3` below mandate a specific Rust module path or
filename; per §2.7, where each piece lands inside the Rust crates is an
implementation-time decision driven by Rust idioms, not a 1:1 mirror of
this table's left column.

| Python file | Disposition |
|---|---|
| `mal_simulator/ttc_utils.py` | Logic ported to `malsim-core`. `TTCDist` class stays in Python too, as the user-facing config type (§2.4) - becomes a thin class whose sampling delegates to native where convenient, but whose construction/`to_dict`/`from_dict` stay as today. |
| `mal_simulator/graph_processing.py` | Ported to `malsim-core` (necessity propagation; pure graph algorithm, no RNG, no DSL). |
| `mal_simulator/graph_utils.py` | `node_is_blocked`/`node_is_traversable`/`node_is_live` ported. `node_is_actionable`/`node_reward` stay Python (operate on `NodePropertyRule` directly, not in the flattened hot path - see §2.4) - called rarely, outside `step()`. |
| `mal_simulator/attack_surface.py` | Ported (`get_attack_surface`, `get_effects_of_attack_step`) - hot path, iterates graph each step. |
| `mal_simulator/defense_surface.py` | Ported (`get_defense_surface`) - same reason. |
| `mal_simulator/false_alerts.py` | Ported (RNG-per-node, hot path). |
| `mal_simulator/event_logger.py` | Ported (detector TP/FP rolls are RNG-per-detector, hot path). `LogEntry` the *Python dataclass* stays; Rust returns plain data, Python re-wraps into `LogEntry` (needs `detector`/`trigger` resolved from ids back to real objects, same pattern as nodes). |
| `mal_simulator/observability.py` | Ported (`observed_nodes`); `node_is_observable` for the rarely-called public query can stay Python-side against the raw rule, same as actionability. |
| `mal_simulator/attacker_step.py` | Ported (`attacker_step`, `attempt_attacker_step`, termination check). |
| `mal_simulator/defender_step.py` | Ported (`defender_step`, termination check). |
| `mal_simulator/graph_state.py` | Ported (`compute_initial_graph_state`) - becomes part of native `reset()`. |
| `mal_simulator/simulator_state.py` | Python `MalSimulatorState` dataclass stays (mirrored); its *construction* reads native output instead of calling the above directly. |
| `mal_simulator/attacker_state_factories.py` / `defender_state_factories.py` | Stay as Python factory functions (same names/signatures where feasible), rewritten internally to build dataclasses from native `reset`/`step` output instead of recomputing via the now-ported modules. |
| `mal_simulator/attacker_state.py` / `defender_state.py` / `agent_state.py` | **Unchanged.** This is the literal "mirrored in Python" contract - do not touch these dataclass definitions. |
| `mal_simulator/simulator.py` | `MalSimulator` class: `reset()`/`step()` internals rewritten to call native; all other public methods audited one by one (§5, A10) to either delegate to native state or stay against the Python-side mirror - signatures unchanged either way. |
| `mal_simulator/node_getters.py`, `state_query.py`, `agent_states.py`, `reset_agent.py`, `simulator_static_data.py` | Stay Python, unchanged or near-unchanged - thin glue over already-mirrored Python state, not hot-path, no RNG. |
| `mal_simulator/rewards.py` | **Unchanged** (§2.4). |
| `mal_simulator/run_simulation.py` | **Unchanged** - drives `.reset()`/`.step()` through the public API only. |
| `config/node_property_rule.py`, `config/agent_settings.py`, `config/agent_settings_factories.py`, `config/sim_settings.py` | **Unchanged** for the Python-bound path. New: (a) a small translation module (e.g. `mal_simulator/native_settings.py`) that flattens `AttackerSettings`/`DefenderSettings`/`MalSimulatorSettings` into the plain-data shape native `reset()` expects (§2.4); (b) for the Rust-only path only, an independent Rust port of `NodePropertyRule`'s matching + the agent-settings-from-dict construction these files do (§2.6, §7 C1/C3) - additive new Rust code, not a port/replacement of these Python files. |
| `scenario/scenario.py` | **Unchanged** for the Python-bound path - setup-time-only, not a hot loop, already backed by mal-toolbox's Rust core transitively. For the Rust-only path, an independent Rust scenario loader is added (§2.6, §7 C2/C4) covering the same YAML schema (field validation, `extends` merge, path resolution) - new code, not a port of this file. |
| `envs/`, `policies/`, `visualization/` | **Untouched by design** (per the request) - serve as integration tests. |

## 5. Phase A: `MalSimulator` port - detailed steps

Each step should land as its own PR/commit with the full Python test
suite green *and* a Rust-native test ported from the equivalent Python
test(s) for whatever that step just moved (§8) before moving to the next
- this is where "non-breaking" is hardest to hold, so steps are kept
small on purpose.

**A1. Wire up the shared graph handle, prove nothing else yet.**
Add `maltoolbox-attackgraph-py` as a git dependency (`rust-rewrite`
branch) to `py-bindings/malsim-pyo3/Cargo.toml`. Write one `#[pyfunction]`
that takes a Python `maltoolbox.AttackGraph`, extracts
`PyAttackGraph.inner`, and returns e.g. the node count read through the
shared `Rc<RefCell<...>>`. Write a Python-side smoke test that builds a
scenario's attack graph, calls this function, and confirms the count
matches `len(attack_graph.nodes)`. This validates §2.2 end-to-end before
anything else depends on it.

**A2. Port `TtcDist` + RNG.**
Add `rand`/`rand_distr` (or equivalent) to `malsim-core`. Port
`DistFunction`/`Operation`/`TTCDist` (`expected_value`, `sample_value`,
`success_probability`, `attempt_ttc_with_effort`, `attempt_bernoulli`,
`from_dict`/`to_dict`-compatible parsing) as a Rust enum/struct
(`TtcDist`). Unit-test against hand-computed expected values (not against
Python's exact samples - see §2.1) for each `DistFunction` variant and
for the `combine_with`/`combine_op` composition case. Check
`grep -rn seed= tests/` first per §2.1 and flag any test asserting exact
sampled values for follow-up.

**A3. Port static graph-state computation.**
Port `compute_initial_graph_state`'s pieces: `attack_step_ttc_values`,
`get_pre_enabled_defenses`, `get_impossible_attack_steps` (depends on A2),
and `graph_processing.py`'s necessity propagation (independent of A2,
could be done in parallel). Unit-test each against a small hand-built
graph fixture, comparing against today's Python output for the
*non-random* parts (necessity, which nodes get a TTC value entry) and
against distributional properties for the random parts.

**A4. Port graph traversal predicates.**
`node_is_blocked`, `node_is_traversable`, `node_is_live` - pure graph
logic over `AttackGraphNode.{type,parents,children,existence_status}`
plus the runtime `enabled_defenses`/`impossible_attack_steps`/
`necessity_per_node` from A3. No RNG, no DSL - should be close to a
line-for-line port. (`node_is_actionable`/`node_reward` are *not* ported
here - see §2.4/§4.)

**A5. Port attack surface / defense surface / effects.**
`get_attack_surface`, `get_effects_of_attack_step`, `get_defense_surface`.
Takes the *already-flattened* actionability id-set (§2.4) as a plain
argument - this step should not need to know `NodePropertyRule` exists.

**A6. Port false-alert + detector log generation.**
`generate_false_negatives`/`generate_false_positives`
(`false_alerts.py`), `observed_nodes` (`observability.py`),
`collect_logs`/`collect_false_positives` (`event_logger.py`). All
RNG-per-node/per-detector - lean on A2's RNG plumbing. `LogEntry`'s
`detector`/`trigger` fields need id-based equivalents on the Rust side
(detector id + node id), resolved back to real objects only when crossing
into Python.

**A7. Port step orchestration.**
`attacker_step`/`attempt_attacker_step`/`attacker_is_terminated`
(`attacker_step.py`), `defender_step`/`defender_is_terminated`
(`defender_step.py`). This is where A2-A6 compose into the actual
per-agent step logic. Port the assertion in both
(`node == sim_state.attack_graph.nodes[node.id]`) as a Rust-side
id-membership check with an equivalent error, not a silent skip.

**A8. `malsim-pyo3` native `Simulator` pyclass.**
Expose a `_native.Simulator` (or similar, not part of malsim's public
API - an implementation detail named so it's obviously internal) with
`reset_native(settings_dict, agents_dict, seed) -> dict` and
`step_native(actions: dict[str, list[int]]) -> dict` returning plain
nested dicts/lists/floats (no custom pyclasses for the return shape -
keep it boring and inspectable). This is the first point where A1-A7 are
exercised together through Python.

**A9. Rewrite `MalSimulator.reset()`/`.step()`.**
Write the settings-flattening translation module (§4's
`native_settings.py`). Rewrite `MalSimulator.__init__`/`.reset()`/
`.step()` (module-level `reset()`/`step()` functions in
`simulator.py`) to call the native layer and rebuild
`AttackerState`/`DefenderState` via (rewritten) `attacker_state_factories.
py`/`defender_state_factories.py` from the native output, resolving ids
back to `AttackGraphNode` objects. Keep the *outer* `MalSimulator.reset`/
`.step` signatures byte-for-byte identical. At this point the full
existing test suite is the acceptance gate - get it green before A10.

**A10. Audit every other public method on `MalSimulator`.**
Go through each one (`node_ttc_value`, `node_is_actionable`,
`node_reward`, `node_is_observable`, `node_false_positive_rate`,
`node_false_negative_rate`, `node_is_blocked`, `node_is_necessary`,
`node_is_enabled_defense`, `node_is_compromised`, `compromised_nodes`,
`node_is_traversable`, `get_node`, `agent_reward_by_name`, `agent_reward`,
`agent_is_terminated`, `done`, `alive_agents`, `agent_states`) and decide,
case by case, whether it now reads native-backed runtime state (e.g.
`node_ttc_value` without an agent name reads `sim_state.graph_state.
ttc_values`, which is native-computed) or stays against the pure-Python
mirror (e.g. `node_is_actionable`, which reads a `NodePropertyRule`
directly per §2.4). Signature and return type must not change either way.

**A11. Full parity pass + cleanup.**
Full test suite green. Decide per now-Rust-shadowed Python module
(`attack_surface.py`, `ttc_utils.py`'s sampling internals, etc.) whether
to delete it outright or keep it temporarily behind a feature flag / as a
cross-check oracle during a stabilization period - lean towards deleting
once A9/A10 are solid, since keeping two implementations around is itself
a drift risk. Run `envs/`/`policies/`/`visualization/` as the integration
check (per Goals) - they should need zero changes.

## 6. Phase B: `DynaMalSimulator` port (coarse - detail after Phase A)

`DynaMalSimulator` layers on top of the same `AttackerState`/
`DefenderState`/reward machinery (confirmed: it imports and reuses
`mal_simulator.attacker_state_factories.create_attacker_state`,
`rewards.py`'s reward-fn builders, etc.) but replaces the step functions
with `dyna_mal_simulator/attacker_step.py` / `defender_step.py`, which can
mutate the underlying `Model`/`AttackGraph` at runtime
(`model_effects.py`, 346 lines; `process_assoc_traversal.py`, 318 lines;
`model_state.py`).

The good news: the hardest part - live graph mutation with correct
id/reference bookkeeping - is *already solved* in mal-toolbox's Rust core
(`AssetSnapshot`, `partially_regenerate_graph`, documented in
mal-toolbox's own `PORTING_NOTES.md` §2). Phase B is mostly about porting
malsim's *model-effect application rules* (what a dyna-MAL effect
declaration does to the model) on top of primitives mal-toolbox's Rust
side already exposes, re-using §2.2's shared-graph/shared-model handle.

Deferred until Phase A lands and its patterns are validated:
- Exact module breakdown mirroring §5's granularity, including the
  per-step Rust-test-porting requirement from §8 (equivalent tests from
  `tests/test_dyna_mal_simulator.py` per module, same "where relevant"
  carve-outs).
- Whether model-mutation effects need their own RNG-consuming paths
  (check `model_effects.py` for `rng` usage once this phase starts).
- Whether `DynaMalSimulatorState` needs new native-side fields beyond
  what `MalSimulatorState`'s port already provides.

## 7. Phase C: Rust-only library API + scenario loading

Per the confirmed decisions: **library API only** (not a CLI, not ported
built-in policies), but - per the amendment in §1/§2.6 - also able to
load a scenario file end-to-end, not just step an already-built graph.
`malsim-core`'s public `Simulator::reset`/`::step` (operating on
Rust-native ids/types) must be directly usable by an embedding Rust
program with no PyO3 and no Python runtime involved - this should fall
out of Phase A's design for free, since `malsim-core` is pure Rust by
construction (§2.2/§3) and the `py-bindings/malsim-pyo3` crate is the
*only* consumer that adds PyO3. The embedding caller supplies its own
action-selection code (no built-in agent policies ship in Rust).

**C1. Port `NodePropertyRule`'s matching, independently (§2.6).**
A small Rust type (e.g. `malsim_core::scenario::NodeRule<T>`) implementing
the same `by_asset_type`/`by_asset_name`/`default` precedence as
`node_property_rule.py`'s `.value()`, plus a `.per_node(&AttackGraph,
&Model) -> HashMap<NodeId, T>` equivalent of `.per_node()`. Unit-test
against the same precedence cases `tests/` already covers for
`NodePropertyRule` (by-name beats by-type beats default).

**C2. Port scenario YAML schema handling.**
Field validation (required/allowed/deprecated fields), `extends` merge
(`recursive_update`'s deep-merge-with-explicit-None-override semantics),
and path resolution relative to the scenario file - port of
`scenario.py`'s `_validate_scenario_dict`/`recursive_update`/
`load_scenario_dict`. Use `serde_yaml` (already a `mal-toolbox` workspace
dependency, so no new dependency family). Unit-test the merge semantics
directly (nested-dict override, explicit-`null`-removes-key) since those
are the easiest part to get subtly wrong.

**C3. Port agent-settings-from-dict construction.**
Per-agent dict → entry points/goals (as node ids, resolved against the
loaded graph), `reward_mode`, resolved rule maps (via C1) for rewards/
actionable_steps/observable_steps/false_positive_rates/
false_negative_rates/ttc overrides. Port of `agent_settings_factories.py`
minus `policy_name_to_class` - parse and carry the `policy` field as an
opaque string (§2.6), never instantiate it.

**C4. Wire to mal-toolbox's pure-Rust graph construction.**
Call `maltoolbox-language`/`maltoolbox-model`/`maltoolbox-attackgraph`
(already a `malsim-core` dependency path per Phase 0) to go from
`lang_file`/`model`/`model_file` to a built `AttackGraph`, mirroring what
`create_attack_graph(lang_graph, model)` does on the Python side. This is
the step that proves C1-C3 compose into something that actually produces
a `Simulator`-ready graph + settings pair.

**C5. `Simulator` library-API smoke test.**
An integration test or example *within `core/malsim-core`* (a `#[test]`
or `examples/` entry, not a separate binary crate) that loads a real
scenario fixture via C2-C4, constructs a `Simulator`, and runs a few
manual `step()` calls with hand-picked actions - proving the library
compiles and runs with zero Python/PyO3 in the dependency graph, and that
scenario loading and stepping actually compose end to end.

**C6. Schema-parity test against the Python oracle.**
Run the *same* scenario YAML fixtures used by `tests/test_scenario.py`
(`tests/testdata/scenarios/*.yml`) through both Python's `Scenario.
load_from_file` and the new Rust loader, and assert they agree on the
resolved, non-random shape: same entry points/goals (by node full name),
same resolved actionable/observable/rate maps, same sim settings. This is
the two-independent-implementations risk from §2.6/§9 made concrete and
checked by CI, not just asserted in prose.

## 8. Testing & non-breaking verification strategy

- **Each ported module gets its own Rust-native tests, equivalent to the
  relevant Python tests, not just a pass-through reliance on the Python
  suite staying green.** For every lettered step in §5/§6/§7 that ports a
  Python module, identify the Python test(s) exercising that module's
  behavior (e.g. A2 ↔ `tests/test_ttc_utils.py`; A3-A7 ↔ the relevant
  tests in `tests/test_mal_simulator.py`; C1 ↔ `NodePropertyRule`
  precedence cases; C2 ↔ `test_scenario.py`'s `extends`/validation cases)
  and write a Rust `#[test]` covering the same case/input/expected-shape,
  not just the same code path incidentally. "Where relevant" carves out
  genuinely Python-only concerns with no Rust equivalent - pickling
  (`__getstate__`/`__setstate__`), `copy.deepcopy` semantics, anything
  asserting on Python object identity/`isinstance` - mirroring
  mal-toolbox's own `PORTING_NOTES.md` §1's precedent for what gets
  excluded and why (list it explicitly per phase when skipped, don't
  just drop it silently). Per §2.1, a test asserting an *exact* sampled
  value for a seed is not portable as-is; port its *structural*
  assertion (e.g. "an impossible step never gets compromised") instead.
- The existing Python suite (`tests/test_mal_simulator.py`,
  `test_dyna_mal_simulator.py`, `test_scenario.py`, `test_ttc_utils.py`)
  remains the primary compatibility contract for the Python-bound path.
  It must stay green after *every* step in §5, not just at phase
  boundaries - run it before moving to the next lettered step.
- `uv run mypy python/malsim tests`, `uv run ruff check`, `uv run ruff
  format --check` stay part of the gate throughout (per
  `.claude/CLAUDE.md`'s existing CI contract) - the Python-side rewrites
  in A9/A10 are still strictly-typed Python.
- For each ported module, prefer a short-lived *shadow comparison* during
  development (call both the old Python function and the new native path
  on the same input, assert they agree on the non-random fields) over
  trusting a read-through - this is how mal-toolbox's own Rust port
  caught real divergences (see its `PORTING_NOTES.md` §3-4 for the kind
  of bug this catches, e.g. a Rust draft applying a filter to only one of
  three structurally-similar cases).
- `envs/`, `policies/`, `visualization/` run unmodified as integration
  tests per the Goals section - if they break, something in the "mirrored
  in Python" contract moved when it shouldn't have.
- Per §2.1, explicitly grep for seed-pinned exact-value assertions before
  porting any RNG-touching module and resolve them deliberately (relax or
  re-pin), not by accident.

## 9. Open risks to keep watching

- **Seed-pinned exact-sampled-value tests found during A2's `grep -rn
  seed= tests/` check (per §2.1) - will need relaxing or re-pinning once
  Python's `ttc_utils.py` actually starts delegating to native RNG (A9),
  not before.** `tests/test_ttc_utils.py::test_ttcs_effort_based` seeds
  `np.random.default_rng(10)` once and asserts exact `True`/`False`
  outcomes of `attempt_ttc_with_effort` at specific effort levels (e.g.
  "1 effort will not succeed in this seed" / "500 effort will succeed in
  this seed") - this is exactly the kind of test §2.1 says is not
  portable bit-for-bit across the numpy → Rust RNG transition.
  `test_bernoulli` (same file) is a softer case: it seeds `default_rng(10)`
  and asserts both `True` and `False` appear across 10 `attempt_bernoulli`
  draws - a structural property, but still pinned to one seed's specific
  draw sequence producing both within only 10 tries, so it could in
  principle flake under a different RNG even though it's not an exact-
  value assertion. `test_probs_utils` (same file) also seeds but doesn't
  assert on the sampled value at all, so it's unaffected. None of these
  break today - Python's `ttc_utils.py` is unchanged by A2, so they're
  still exercising the pure-Python/numpy/scipy path. Flagging now (as
  A2's step description requires) so A9 - the step that actually wires
  `MalSimulator` to call into native RNG-consuming code - doesn't
  rediscover this from scratch; revisit `test_ttcs_effort_based`
  specifically (relax to a monotonicity/structural assertion, mirroring
  what `ttc.rs`'s own `attempt_ttc_with_effort_success_rate_matches_
  probability` test does) at that point.
- §2.2's coupling to `maltoolbox-attackgraph-py`'s internal struct layout
  - re-verify `PyAttackGraph.inner`'s visibility/shape on every
    `mal-toolbox` git dependency bump.
- Performance: resolving native-returned id sets back into Python
  `AttackGraphNode` objects (to rebuild `performed_nodes`/
  `action_surface` etc. each step) is inherent to the "mirrored, not
  wrapped" design (§2) and isn't free. If it turns out to dominate once
  A9 lands, the optimization path is incremental resolution (cache
  resolved nodes, only resolve newly-added ids each step and extend the
  existing Python set) rather than resolving the whole accumulated set
  from scratch - not needed for correctness, only revisit if profiling
  says so.
- mal-toolbox's `rust-rewrite` branch is itself a moving target (own
  `PORTING_NOTES.md` still lists some deferred work) - re-check its
  `PORTING_NOTES.md` §6 ("what's left to do") periodically in case
  something malsim depends on shifts under it.
- §2.6's two independent `NodePropertyRule`/scenario-schema
  implementations (Python's, kept as-is, and the new Rust one for the
  standalone path) can drift silently if either side's schema changes
  without the other noticing - there's no shared source of truth between
  them by construction. §7's C6 schema-parity test is the mitigation;
  treat any future change to `scenario.py`'s YAML schema or
  `node_property_rule.py`'s precedence rules as requiring a matching
  Rust-side change plus a re-run of C6, not just a Python-side test
  update.

## 10. Differences log

Per §2.7: this is where a *specific* divergence between the Rust
implementation and the Python original gets recorded, as it's found or
decided during implementation - not as a Rust doc comment, since this
file is what the original Python author reads to understand the port.
"Divergence" here means something beyond "this landed in a different
Rust module" (that's expected and covered by §2.7 already) - it means a
case where the Rust code's structure, behavior, or API shape actually
differs from what reading `python/malsim` would lead you to expect.

Each entry: which Python code it's about, what the Rust side does
differently, and why. Empty for now - no porting code has been written
yet (§0 - still Phase 0). Add entries here as Phase A/B/C steps land;
don't let this section stay empty once code exists, and don't let a
divergence's only record be a comment buried in the `.rs` file that
introduced it.

**§2.2's cross-extension-module `PyAttackGraph` downcast does not work as
written (found during A1).** §2.2 assumed that `malsim-pyo3` could depend
on `maltoolbox-attackgraph-py` as a Cargo dependency, receive a Python
`maltoolbox.AttackGraph` object, and downcast/extract it back to that
crate's `PyAttackGraph` pyclass to reach `.inner: Rc<RefCell<AttackGraph>>`.
Implemented exactly as specified (pinned both to mal-toolbox commit
`493f738f3de915adf1fb348d38a3cb471a1936fd`, added the dependency, wrote a
`#[pyfunction] node_count(graph: &Bound<'_, PyAttackGraph>)`) and it fails
at runtime with `TypeError: 'AttackGraph' object is not an instance of
'AttackGraph'`, even though the object genuinely is a
`maltoolbox._native.AttackGraph`.

Root cause: PyO3 pyclasses from a shared dependency crate get a *separate,
independently-initialized* Python type object in every final `cdylib` that
statically links the crate. `maltoolbox-attackgraph-py` is an `rlib`, not a
`dylib`, so it is compiled into both `maltoolbox._native.so` (via
`maltoolbox-pyo3`) and `malsim._native.so` (via `malsim-pyo3`) as two
independent copies, each with its own `LazyTypeObject` static. The
resulting Python-visible types have the same name/module string but are
not the same object and have no subclass relationship, so
`obj.downcast::<PyAttackGraph>()` / typed-pyclass-parameter extraction
always fails across this boundary - this isn't a bug in the pinned commit
or a one-off mistake, it's a structural limit of static-linking the same
PyO3 pyclass crate into two separately built extension modules. Confirmed
empirically, not just in theory - see this repo's git history around the
date this entry was added for the exact repro.

Options considered: (a) a `PyCapsule`-based raw-handle export added
upstream in mal-toolbox, (b) making the shared crate an actual `dylib`
both extensions link against at runtime, (c) not sharing the live mutable
graph at all for Phase A (reading the graph's static structure via
ordinary Python-level attribute access once per `reset()`, deferring the
cross-module live-handle problem to Phase B). (b) was assessed and
rejected: it would require both `maltoolbox` and `mal-simulator` wheels to
agree on a runtime library search path despite being independently
pip-installed packages, and standard wheel-repair tooling (`auditwheel`/
`delocate`) actively works against this by vendoring external shared-lib
dependencies into each wheel independently - likely to silently
reintroduce two copies (the exact bug this would exist to fix) with no
obvious CI signal. (a) was chosen.

**Resolution: `PyAttackGraph::__inner_capsule__()` added upstream
(mal-toolbox commit `c854d1d6567ecb8851cf52a340a3ec5b673467f4`), consumed
from `malsim-pyo3` via a `PyCapsule`.** `PyAttackGraph` gained a
`__inner_capsule__(&self, py) -> PyResult<Bound<'py, PyCapsule>>` method:
it clones `self.inner` (bumping the `Rc`'s strong count), leaks that clone
via `Rc::into_raw`, and wraps the resulting pointer in a `PyCapsule` named
`"maltoolbox._native.AttackGraph.inner"` with a destructor that reclaims
and drops exactly that one clone when the capsule is GC'd.

On the `malsim-pyo3` side, `extract_shared_graph()`
(`py-bindings/malsim-pyo3/src/lib.rs`) calls `graph.call_method0
("__inner_capsule__")`, verifies the capsule's name matches the same
constant, and reconstructs an `Rc<RefCell<AttackGraph>>` from its pointer.
Consequence: `malsim-pyo3` no longer needs `maltoolbox-attackgraph-py` as
a Cargo dependency at all - it only depends on the pure
`maltoolbox-attackgraph` crate (for the `AttackGraph` type the raw pointer
is cast to), since there's no pyclass to downcast to anymore. This also
drops the "internal struct layout" coupling risk §2.2 originally flagged
for `PyAttackGraph.inner`'s visibility - the capsule's string name is now
the only cross-module contract, and it's a stable one by construction.

**A genuine double-free bug surfaced and was fixed while implementing the
consumer side - worth recording since it's easy to reintroduce.** The
first implementation called `Rc::from_raw(ptr)` directly on the capsule's
pointer. This is wrong: `__inner_capsule__` parks exactly *one* strong
reference via `Rc::into_raw`, which its own capsule destructor reclaims
and drops when the capsule is GC'd - calling `Rc::from_raw` on the same
pointer a second time (from the consumer side) reclaims that *same*
reference again, so it gets dropped twice (once when the consumer's local
`Rc` goes out of scope, once later when the capsule's destructor runs).
This didn't fail the first, simple smoke test (single call, no cleanup
before process exit) - it only surfaced under a stress test that created
many graphs, called `node_count`, deleted the graph, and ran `gc.collect()`
in a loop, crashing with `malloc(): unaligned tcache chunk detected`
within the first couple of iterations. Fix: call
`Rc::increment_strong_count(ptr)` *before* `Rc::from_raw(ptr)`, so the
consumer mints its own independent strong reference instead of stealing
the capsule's. Verified after the fix with a 2000-iteration stress test
(create graph, call `node_count` x5, delete, periodic `gc.collect()`)
showing no crash and flat (non-growing) RSS after an initial warm-up -
i.e. checked for *both* a double-free and a leak, not just the crash.
Anyone writing a second consumer of this same capsule (or a similar one
in Phase B) needs the `increment_strong_count` step too - it's not
specific to `node_count`, it's inherent to consuming this kind of
capsule at all.

**A2: chose the `statrs` crate for distribution CDF/mean/sampling,
instead of hand-rolling special functions or only using `rand_distr`
for sampling.** `rand_distr` only provides sampling, not CDF or mean -
`success_probability` (`dist.cdf(effort)` in Python) and
`expected_value` (`dist.expect()`/a closed-form mean in Python) both need
real CDF/mean implementations, which for Gamma/Binomial requires the
regularized incomplete gamma/beta functions. `statrs` (0.19.1, default
features trimmed to just `std`+`rand` - the `nalgebra` default feature
pulls in `nalgebra`/`glam` for multivariate distributions this module
never uses) provides `ContinuousCDF`/`DiscreteCDF`/`statrs::statistics::
Distribution` (mean) *and* implements `rand`'s `Distribution<f64>` for
sampling, for exactly the six distributions needed (Bernoulli, Exp,
Binomial, Gamma, LogNormal, Uniform) - one dependency covers all three
needs instead of reimplementing special functions by hand. Resolves to
`rand` 0.10.3 + `statrs` 0.19.1 compatibly; no version-mismatch issues
found.

**A2: two parameter-order/parameterization mismatches between scipy's
and statrs's constructors - got the translation wrong once before fixing
it, worth flagging for whoever next touches `ttc.rs`.** Both are handled
correctly in the landed code (`TtcDist::binomial`/`TtcDist::gamma` in
`ttc.rs`), but the mismatch is easy to reintroduce:
- **Binomial:** Python's `args` are `[n, p]` (`n, p = args; binom(n=n,
  p=p)`), but `statrs::distribution::Binomial::new` takes `(p, n)` - the
  *opposite* argument order. `ttc.rs`'s `binomial()` helper does
  `StatrsBinomial::new(self.args[1], self.args[0] as u64)` - swapped
  deliberately, not a typo.
- **Gamma:** Python's `args` are `[shape, scale]` (`gamma(a=shape,
  scale=scale)`), but `statrs::distribution::Gamma::new` takes `(shape,
  rate)` where `rate = 1 / scale`, not `(shape, scale)`. `ttc.rs`'s
  `gamma()` helper does `Gamma::new(self.args[0], 1.0 / self.args[1])`.
  Exponential has the same scipy `scale`-vs-statrs `rate` split
  (`expon(scale=1/rate)` in Python vs. `Exp::new(rate)` in statrs, both
  parameterized by rate already, so no inversion needed there - only
  Gamma's *scale* parameter needs inverting to a *rate* for statrs).

LogNormal and Uniform parameter order match directly (`(mean, std)` →
`LogNormal::new(location, scale)`; `(low, high)` → `Uniform::new(min,
max)`) - no translation needed, confirmed against both scipy's and
statrs's own doc-comment formulas for `expected_value`/mean before
relying on it, not just by matching test numbers.

**A2: `success_probability` deliberately does not consult
`combine_with`, matching Python's existing behavior exactly rather than
"fixing" what looks like it could be an oversight.** Python's
`TTCDist.success_probability` is `self.dist.cdf(effort)` - it never
looks at `self.combine_with`, even for distributions like
`HardAndUncertain` that are a combination. `ttc.rs`'s
`success_probability` mirrors this precisely (see its
`success_probability_ignores_combine_with` test, which asserts a plain
`Exponential(0.1)` and the same distribution combined with
`Bernoulli(0.5)` give identical `success_probability` results) - this is
called out here per §2.7's rule, in case a future reader assumes it's a
bug to be fixed rather than intentionally-preserved behavior.
