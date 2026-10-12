# Porting notes: moving malsim's simulator core to Rust

Reference for the port of `MalSimulator`'s and `DynaMalSimulator`'s
hot-path logic from Python into the `core/malsim-core` /
`py-bindings/malsim-pyo3` Rust crates. The port is complete; this file
records the design, the standing conventions, and the places where the
Rust code deliberately differs from (or deliberately copies a quirk of)
the Python original. Section numbers are referenced from code comments
(`PORTING_NOTES.md §10` etc.), so keep them stable.

## 0. Status

All phases are done (last change 2026-10-11). One line per step, with
where it landed, is in §5 (Phase A, `MalSimulator`), §6 (Phase B,
`DynaMalSimulator`) and §7 (Phase C, Rust-only library API + scenario
loading).

- [x] Phase 0 - Rust workspace scaffolding (`core/`, `py-bindings/`; two
      separate Cargo workspaces).
- [x] Phase A - `MalSimulator` runs on native (A1-A11).
- [x] Phase B - `DynaMalSimulator` runs on native (B1-B7). The two B5
      bug classes code comments refer to as "§0 B5" (stale node ids
      after a model effect, and the missing-TTC-value gap) are described
      in §10 "Dyna path".
- [x] Phase C - `malsim-core` usable without Python, including scenario
      loading (C1-C6).
- [x] Static and dynamic simulators split into `Simulator` /
      `DynaSimulator` (§11).

Gates for any further change (all must stay clean):

```bash
cargo test && cargo clippy -- -D warnings && cargo fmt --check   # in the repo root
cargo test && cargo clippy -- -D warnings && cargo fmt --check   # in py-bindings/
uv run pytest tests examples/*        # includes `integration`
uv run mypy python/malsim tests
uv run ruff check && uv run ruff format --check
```

If `tests/testdata/scenarios/**` or the scenario schema changes,
regenerate the parity golden file with `python -m tests.scenario_parity`
(§8).

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
  `tests/test_ttc_utils.py`) is the compatibility contract and must stay
  green.
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
  Python, can't move to Rust in general. No built-in policies are ported
  (§7).

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
porting all RNG-consuming sampling into Rust (via the `rand`/`statrs`
crates) for real speedups.

**Consequence for tests:** a test that asserts an *exact* sampled value
or exact sequence of actions for a fixed seed is either relaxed (assert
statistical/structural properties instead) or pinned against the Rust
RNG's output (§8).

### 2.2 Shared live graph and model with mal-toolbox's PyO3 layer

Python's `maltoolbox.AttackGraph`/`Model` objects and `malsim-core`
operate on the *same* `Rc<RefCell<AttackGraph>>`/`Rc<RefCell<Model>>`:
zero-copy and always in sync, which `DynaMalSimulator` needs because the
graph and model mutate at runtime.

**Mechanism: a `PyCapsule` handoff, not a pyclass downcast.** A PyO3
pyclass from a shared `rlib` crate gets a separate, unrelated type object
in every `cdylib` that statically links it, so `malsim._native` can never
downcast an object created by `maltoolbox._native` (`TypeError:
'AttackGraph' object is not an instance of 'AttackGraph'`). Instead
mal-toolbox's `PyAttackGraph` and `PyModel` each expose
`__inner_capsule__()`, which returns a named `PyCapsule`
(`"maltoolbox._native.AttackGraph.inner"` /
`"maltoolbox._native.Model.inner"`) holding one `Rc::into_raw` clone that
the capsule's destructor drops. `malsim-pyo3`'s `extract_shared_graph`/
`extract_shared_model` (`lib.rs`) check the name and rebuild the `Rc`.
The capsule name is the only cross-module contract, so `malsim-pyo3`
depends only on the pure `maltoolbox-attackgraph`/`maltoolbox-model`
crates, not on mal-toolbox's `-py` crates.

**Any capsule consumer must call `Rc::increment_strong_count(ptr)` before
`Rc::from_raw(ptr)`.** The capsule owns exactly one strong reference;
`from_raw` alone steals it and it is dropped twice (a double free that a
single-call smoke test does not show, only a create/use/delete/
`gc.collect()` loop does).

**Consequence:** `malsim-core` (pure Rust, no PyO3) depends only on the
plain mal-toolbox crates. Its simulators always take
`Rc<RefCell<AttackGraph>>` (and `Rc<RefCell<Model>>`) as handle types,
also on the Rust-only path, which just owns the only reference - so there
is one code path, not two.

### 2.3 Phasing

`MalSimulator` (Phase A) was ported and stabilized before
`DynaMalSimulator` (Phase B), which reuses Phase A's FFI/mirroring
pattern. Both are done; code comments saying `DynaMalSimulator` "still
uses the pure-Python path" predate Phase B.

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
  (via malsim._native)│  shared graph/model handles   │──▶ pure
                      │  via PyCapsule (§2.2)         │    maltoolbox
                      │  Simulator / DynaSimulator:   │    crates only
                      │   reset_native()/step_native()│    (git dep)
                      │   → plain dict/set/float      │
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
crosses it** - ids (`AttackGraphNode.id: i64`, see §10), strings,
floats, bools, and flat dicts/sets/lists thereof. No Python object
(beyond the shared `AttackGraph`/`Model` handles) and no Rust struct is
ever handed across directly. This is what makes the
Python dataclasses genuinely "mirrored" rather than wrapped: `malsim-pyo3`
hands the Python shim raw ids and primitives; the shim resolves ids to
real `AttackGraphNode` objects via `attack_graph.nodes[id]` (already O(1)
in mal-toolbox) and constructs the unchanged dataclasses itself.

`malsim-core`'s `Simulator` owns:
- `graph: Rc<RefCell<maltoolbox_attackgraph::AttackGraph>>`
- the `MalSimulatorSettings` passed at `reset` (§11); rewards are not
  computed in Rust (§2.4)
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

`DynaSimulator` wraps a `Simulator` and adds the shared `Model` handle
and its pristine snapshot (§11).

## 4. File/module inventory

Where each piece of behavior lives now. Rust module layout is not required
to mirror Python's (§2.7); this is a map, not a rule.

| Python (`python/malsim/`) | What it holds now | Rust counterpart (`core/malsim-core/src/`) |
|---|---|---|
| `mal_simulator/simulator.py` | `MalSimulator`; module-level `reset()`/`step()` delegating to `_native.Simulator`; `_graph_state_from_native`, `_ordered_new_nodes`, `MALSimulatorStaticData`. Public query methods unchanged. | `simulator.rs` |
| `mal_simulator/native_settings.py` | `flatten_sim_settings`/`flatten_attacker_settings`/`flatten_defender_settings` (§2.4). | `scenario/flatten.rs` (Rust-only path) |
| `mal_simulator/attacker_state_factories.py`, `defender_state_factories.py` | Only `create_*_state_from_native`, `get_entry_points`, `_log_entry_from_native`. | - |
| `mal_simulator/attacker_state.py`, `defender_state.py`, `agent_state.py`, `agent_states.py`, `simulator_state.py` | The mirrored dataclasses. **Do not change their shape.** | `AttackerRuntime`/`DefenderRuntime`/`SimState` in `simulator.rs` |
| `mal_simulator/graph_utils.py`, `state_query.py`, `node_getters.py` | Live: back `MalSimulator`'s public query methods (`node_is_blocked`, `node_is_traversable`, `node_ttc_value`, ...). Not dead code (§9). | `graph_utils.rs` (used inside `step`) |
| `mal_simulator/ttc_utils.py` | `TTCDist` and helpers, the user-facing config type. | `ttc.rs` |
| `mal_simulator/graph_state.py` | Only `GraphState`. | `graph_state.rs`, `necessity.rs`, `viability.rs` |
| `mal_simulator/event_logger.py` | Only `LogEntry`. | `event_logger.rs` |
| `mal_simulator/attacker_step.py`, `defender_step.py` | Only `attacker_is_terminated`/`defender_is_terminated`. | `attacker_step.rs`, `defender_step.rs` |
| `mal_simulator/false_alerts.py`, `observability.py` | Only the per-node rule getters. | `false_alerts.rs`, `observability.rs` |
| (deleted) `attack_surface.py`, `defense_surface.py`, `graph_processing.py`, `reset_agent.py`, `simulator_static_data.py` | - | `attack_surface.rs`, `defense_surface.rs`, `necessity.rs`, `viability.rs` |
| `mal_simulator/rewards.py`, `run_simulation.py` | Unchanged, pure Python (§2.4). | - |
| `dyna_mal_simulator/simulator.py`, `simulator_state.py` | `DynaMalSimulator`, `dyna_reset`/`dyna_step` delegating to `_native.DynaSimulator`; `DynaMalSimulatorState`, `AssetOp`/`AssocOp`, `DetachedAsset`, `modification_record_from_native`. | `dyna_simulator.rs`, `dyna_attacker_step.rs`, `dyna_defender_step.rs`, `dyna_graph_state.rs`, `model_effects.rs`, `assoc_traversal.rs`, `model_state.rs` |
| `config/*`, `scenario/scenario.py` | Unchanged; used by the Python path only. | Independent Rust loader (§2.6): `settings.rs`, `scenario/{mod,loading,node_property_rule,agent_settings,flatten}.rs` |
| `envs/`, `policies/`, `visualization/` | Untouched by design; they double as integration tests. | - |

`py-bindings/malsim-pyo3/src/`: `lib.rs` (capsule extraction, plus the
test-support functions `node_count`, `set_detector_rates`,
`model_asset_count`, `model_add_asset_native` - none are public API),
`simulator.rs` (`Simulator` pyclass and the dict conversions both
pyclasses share), `dyna_simulator.rs` (`DynaSimulator` pyclass). It only
parses dicts, maps errors and builds output (§11). Stubs:
`python/malsim/_native-stubs/__init__.pyi`.

Rust test fixtures: `core/malsim-core/src/test_fixtures.rs` compiles
`tests/testdata/langs/dummy_lang.mal` (the file `tests/conftest.py::
dummy_lang_graph` uses) and wiperLang via `maltoolbox_language::
from_mal_spec`. A real `AttackGraphNode` with a specific step type cannot
be hand-built (`AttackStepId` is a slotmap key only a `LanguageGraph`
mints), which is why graph-reading functions in `graph_state.rs` are
split into a thin node-reading wrapper plus a `*_from_parts`/`*_for_dist`
helper over plain data.

## 5. Phase A: `MalSimulator` port (done)

- **A1** - shared graph handle via `PyCapsule` (§2.2).
- **A2** - `ttc.rs`: `DistFunction`, `Operation`, `TtcDist`,
  `named_ttc_dist`, on `statrs` + `rand`.
- **A3** - `graph_state.rs` (`TtcMode`, TTC values, pre-enabled defenses,
  impossible steps, `compute_initial_graph_state`), `necessity.rs`.
- **A4** - `graph_utils.rs`: `node_is_live`, `node_is_necessary`,
  `node_is_blocked`, `node_is_traversable`.
- **A5** - `attack_surface.rs` (`get_attack_surface`,
  `get_effects_of_attack_step`), `defense_surface.rs`.
- **A6** - `false_alerts.rs`, `observability.rs`, `event_logger.rs`.
- **A7** - `attacker_step.rs`, `defender_step.rs`.
- **A8** - `_native.Simulator` pyclass with `reset_native`/`step_native`
  returning plain dicts/lists.
- **A9** - `MalSimulator.reset()`/`.step()` delegate to native;
  `native_settings.py`; `create_*_state_from_native`. Followed by the
  delta-only `step_native` wire format (§10).
- **A10** - audit of the other public `MalSimulator` methods: no code
  change. They read the mirrored Python state; `node_is_blocked`/
  `node_is_traversable` still evaluate in Python (§9).
- **A11** - dead Python hot-path code deleted (§4, §11).

## 6. Phase B: `DynaMalSimulator` port (done)

- **B1** - `assoc_traversal.rs`, `model_effects.rs`.
- **B2** - `model_state.rs` (snapshot/reconcile/reset),
  `dyna_graph_state.rs`, `dyna_attacker_step.rs`,
  `dyna_defender_step.rs`.
- **B3** - shared `Model` handle via `PyModel::__inner_capsule__`
  (upstream mal-toolbox change, §2.2).
- **B4** - native dyna reset/step entry points, now the
  `_native.DynaSimulator` pyclass (§11); `step_modification_record`
  output.
- **B5** - `DynaMalSimulator.reset()`/`.step()` delegate to native.
  Fixed two bug classes on the way: stale node ids after a model effect,
  and Python's `GraphState` missing nodes created mid-episode (the
  "TTC-gap fix"). Both in §10 "Dyna path".
- **B6** - audit of inherited query methods against a graph mutated
  mid-episode: no code change; locked in by
  `test_inherited_query_methods_follow_graph_mutated_by_model_effects`.
- **B7** - dead dyna Python deleted; viability/pruning ported to
  `viability.rs`.

## 7. Phase C: Rust-only library API + scenario loading (done)

Library API only: no CLI and no built-in policies in Rust. The embedding
program selects actions itself.

- **C1** - `scenario/node_property_rule.rs`: `NodePropertyRule<T:
  RuleValue>`.
- **C2** - `scenario/loading.rs`: YAML loading, `extends` merge
  (`recursive_update`), path resolution, validation. Uses `serde_yaml`.
- **C3** - `settings.rs` (`MalSimulatorSettings`, `Flat*Settings`),
  `scenario/agent_settings.rs`, `scenario/flatten.rs`.
- **C4** - `scenario/mod.rs`: `Scenario` (`load_from_file`, `from_dict`,
  `flatten_agents(rng)`), `ModelSource`.
- **C5** - orchestration moved from `malsim-pyo3` into public
  `malsim_core::simulator`; smoke test
  `core/malsim-core/tests/scenario_simulator.rs` uses only the public
  API (no pyo3 in `cargo tree -p malsim-core`).
- **C6** - Rust/Python scenario parity via committed golden JSON (§8).

## 8. Testing rules

- **Ported code gets Rust-native tests covering the same cases as the
  Python tests, not just a green Python suite.** Python-only concerns
  (pickling, `deepcopy`, object identity) are excluded; say so explicitly
  when skipping one.
- **Never assert an exact sampled value or exact RNG-dependent sequence
  against Python's output** (§2.1). Assert the structural property, or
  pin to this port's own deterministic-per-seed output. Before touching
  RNG-consuming code, `grep -rn seed= tests/`: several tests are pinned
  to the current native RNG consumption order and will shift if it
  changes.
- **Do not assert on `HashMap`/`HashSet`/`frozenset` iteration order.**
  Rust's default hasher is seeded per process, so an order-dependent
  test passes or fails per process invocation, not per run within one
  process. Run a suspect test in several separate `cargo test`/`pytest`
  processes.
- The Python suite is the compatibility contract for the Python path,
  and `envs/`, `policies/`, `visualization/` must keep working
  unmodified.
- **Scenario parity (C6):** `python -m tests.scenario_parity` writes
  `tests/testdata/scenario_parity.json` (each fixture's resolved,
  non-random shape; a fixture Python refuses records only its error
  type). `tests/test_scenario_parity.py` asserts the file is current;
  `core/malsim-core/tests/scenario_parity.rs` asserts the Rust loader
  matches it. Known divergences are an explicit list in the Rust test
  (currently only `wrong_agent_classes_scenario.yml`, §12 C3).

## 9. Open risks

- **Two independent scenario/`NodePropertyRule` implementations** (§2.6)
  can drift. Any change to `scenario.py`'s YAML schema or
  `node_property_rule.py`'s precedence needs the matching Rust change
  and a regenerated parity file (§8).
- **Two implementations of the blocked/traversable predicates:**
  `graph_utils.py` (behind `MalSimulator.node_is_blocked`/
  `.node_is_traversable`) and `malsim-core::graph_utils` (inside
  `step`). No dedicated cross-check exists; change both together.
- **mal-toolbox's `rust-rewrite` branch is a moving target.** Both
  `Cargo.toml`s and `pyproject.toml` track the branch. The capsule names
  and `__inner_capsule__` (§2.2) are the contract to re-check after an
  upstream change.
- **`uv`/`maturin` can install a stale `mal-toolbox` build** (seen
  repeatedly on this machine, root cause unknown): the installed
  `maltoolbox/_native*.so` lacks a symbol that exists in the resolved
  commit, even after `uv cache clean mal-toolbox`. Before blaming Rust
  code for a Python-level failure after an upstream change, check
  `strings <installed .so> | grep <symbol>`. Workaround: fresh `git
  clone` of mal-toolbox, `maturin build --release --manifest-path
  py-bindings/maltoolbox-pyo3/Cargo.toml -o <dir>`, `uv pip install
  --reinstall-package mal-toolbox <wheel>`, then use `uv run --no-sync`
  for the rest of the session (plain `uv run` re-syncs and reinstalls
  the stale build).
- **Native memory grows across repeated `DynaMalSimulator` runs that
  apply model effects** (~110-125 KB/run on `wiper_scenario.yml`).
  Upstream and pre-existing: valgrind attributes it to
  `maltoolbox/_native.so`, none to malsim's crates, and the old
  pure-Python dyna path grew faster. The gc-based leak tests cannot see
  it. Repro: build `DynaMalSimulator.from_scenario`, `run_simulation`
  with `RandomAgent`, repeat 300x, read `VmRSS` after `gc.collect()` +
  `malloc_trim(0)`. Not yet reported upstream.
- **Entry points or goals whose asset is removed mid-episode** are not
  handled (a removed goal can never be performed). Pre-existing;
  reset-time staleness is handled (§11).

## 10. Differences log

Per §2.7: where the Rust implementation's structure, behavior or API
shape differs from what reading `python/malsim` would lead you to expect,
and Python quirks that were ported on purpose. Add an entry whenever a
change introduces a new divergence; a comment in the `.rs` file is not
enough.

### FFI boundary and Python-side reconstruction

- **Node ids crossing the boundary are `AttackGraphNode.id: i64`**, the
  stable user-facing id, not `AttackGraphNodeId` (an opaque slotmap key).
  `to_node_id`/`stable_ids` in `malsim-pyo3/src/simulator.rs` translate.
- **`stable_ids` sorts its output**, and agents are processed in name
  order (§12 C5). Otherwise `HashSet`/`HashMap` order, random per
  process, would make a fixed seed irreproducible across processes.
- **`reset_native` returns full state; `step_native` returns deltas.**
  Monotonic fields come back as `step_*` keys (`step_enabled_defenses`,
  `step_performed_nodes`, `step_attempted_nodes`,
  `step_compromised_nodes`, `step_observed_nodes`, `step_logs`) and
  `create_*_state_from_native` merges them into `previous_state`
  (`previous_state is None` means "this is reset output, full shape").
  `num_attempts` is not sent; Python increments it from
  `step_attempted_nodes`. `action_surface`, `iteration`, `terminated`
  are always full values. Resending accumulated state made the port
  slower than pure Python and log handling O(episode²); regression
  test: `test_logger_incremental_log_merge_matches_full_rerun`.
- **Graph-state fields (`ttc_values`, `necessity_per_node`,
  `impossible_attack_steps`, `pre_enabled_defenses`) are sent at reset
  only**, except on a dyna step that ran a model effect - see "Dyna
  path".
- **Two RNG streams.** `reset()` draws one `native_seed` from the Python
  `rng` and native seeds its own `StdRng` from it for all TTC/Bernoulli/
  detector sampling. The Python `rng` is used only for `get_entry_points`
  and the `SAMPLE_TTC` reward mode. A fixed seed is reproducible, but
  the seed-to-outcome mapping differs from pre-port (§2.1).
- **Multiple entry-point sets are sampled in Python**
  (`get_entry_points(attack_graph, ...)`), permanently; only the chosen
  set crosses the boundary.
- **`num_attempts` is sparse in Rust, dense in Python.**
  `create_attacker_state_from_native` rebuilds
  `dict.fromkeys(attack_graph.attack_steps, 0)` plus the counts.
- **`iteration` is 0-indexed in native, 1-indexed on
  `AttackerState`/`DefenderState`.** The factories add 1 and use native's
  un-incremented value as the `performed_nodes_order` key. Getting this
  wrong corrupts those keys silently.
- **Per-attacker `ttc_dists` are override-only.** With overrides
  configured, `AttackerState.ttc_values`/`.impossible_steps` hold only
  the overridden nodes, not the graph-wide map. The resolved rule values
  are treated as predefined-distribution *names* (`TTCDist.from_name`),
  despite the field's declared `NodePropertyRule[TTCDist]` type.
- **`recording` order is reconstructed** by `_ordered_new_nodes`:
  requested actions first in request order, then effect-chain-only nodes
  in implementation-defined order. With several successful actions in one
  step the old per-action interleaving is not reproduced.
- **An unknown agent name in `actions` raises `ValueError`** from native
  (Python's `_pre_step_check` raises `KeyError` first on the normal
  path).
- **Pickling:** `_native_sim` / `_dyna_native_sim` are dropped by
  `__getstate__` and there is no `__setstate__`, so `reset()`/`step()`
  on an unpickled simulator raise `AttributeError` (same as the reward
  closures always did). Restoring would need Rust-side RNG state
  serialization.
- **`reset()` reads the settings captured in `__init__`**
  (`MALSimulatorStaticData`) for both simulators; reassigning
  `sim.sim_settings` afterwards has no effect.

### mal-toolbox binding behavior to know about

- **`AttackGraphNode.children`/`.parents` return a cached Python set.**
  `node.children.add(x)` never reaches the real graph, which native
  reads; only assignment does. Tests use
  `tests/conftest.py::connect_nodes`.
- **`AttackGraphNode.detectors` is likewise a Python-side cache, with no
  setter.** To change detector rates on the real graph use the
  test-support `_native.set_detector_rates(graph, node_id, label,
  tprate, fprate)`. Native reads each node's own `detectors` map and
  never the graph-level `attack_graph.detectors` list.
- **`AttackGraphNode.__hash__` includes the owning graph's pointer**, so
  a `frozenset` of nodes iterates in a different order in every process.
  Anything tie-breaking on that order (`ShortestPathAttacker`, searchers
  with `ActionOrdering.NOTHING`) is not reproducible across processes;
  tests must not pin it. `TTCSoftMinAttacker` needs an explicit `seed`
  in its config to be deterministic.
- **Natively removed assets leave no tombstone.** mal-toolbox records
  tombstones only in `PyModel.remove_asset`; native calls the core
  `Model::remove_asset`, so `model.get_asset_by_id` returns `None`
  afterwards. Hence `AssetOp.asset`/`AssocOp.assoc` are `ModelAsset |
  DetachedAsset`: the live handle when it still resolves, otherwise a
  `DetachedAsset(id, type, name)` built from the native record.

### Python quirks ported as-is (do not "fix" without asking)

- `TtcDist::success_probability` ignores `combine_with`.
- Pre-enabled defenses test `success_probability(0)`, i.e. `cdf(0) = 1 -
  p` for a Bernoulli: the named `"Enabled"` dist takes the "never
  succeeds" branch (pre-enabled), `"Disabled"` the "always succeeds" one.
  Python has a `# TODO: is this correct?` there.
- `attempt_attacker_step` resolves the TTC dist before the `Disabled`
  check, so a malformed `ttc` dict errors even with TTCs disabled.
- The `EXPECTED_VALUE`/`PRE_SAMPLE` comparison is effectively
  `num_attempts_before + 2 >= ttc_value`.
- Entry points bypass both the action-surface and the traversability
  check in `attacker_step` (Python: `# TODO: should this actually be the
  case?`).
- `node_is_blocked`: an `and` node is blocked if *any* parent blocks it,
  an `or` node only if *all* do. So a parentless `or` node is "blocked"
  and a parentless `and` node is not.
- Detector rates use Python truthiness: `None` and `0.0` are falsy, a
  negative rate is truthy but never fires. `collect_logs` calls
  `get_context` before the true-positive roll (so its errors surface
  even for a false negative); `collect_false_positives` calls
  `get_random_context` only for a detector that fires. The RNG is drawn
  only when the rate is truthy.
- `sample_size`'s `quantity is None` branch does not clip to `max_size`
  (unreachable with a finite `max_size` today).
- More in §12 (C3 `_validate_agent_dict`, C4 `agents` vs
  `agent_settings`).

### Deliberate deviations

- **`statrs` parameter order differs from scipy:** `Binomial::new(p, n)`
  (Python args are `[n, p]`) and `Gamma::new(shape, rate)` with `rate =
  1 / scale`. `ttc.rs` swaps/inverts on purpose.
- **`NodePropertyRule` arguments become flattened sets/maps** (§2.4):
  `Option<&HashSet<AttackGraphNodeId>>` for actionability/observability
  (`None` = no rule configured), `Option<&HashMap<_, f64>>` for rates.
  `attack_surface.rs`, `defense_surface.rs` and `observability.rs` each
  keep a private `node_is_*_flat` helper rather than sharing one in
  `graph_utils.rs`, where it would look like the ported
  `node_is_actionable`.
- **`get_attack_surface` has `#[allow(clippy::too_many_arguments)]`**
  instead of a bundling struct with a single caller.
- **`get_effects_of_attack_step` grows one `visited` set** instead of
  recomputing `performed | set(effects)` per iteration; same result.
- **`LogEntry` is id-based in Rust:** `trigger: AttackGraphNodeId` and
  `DetectorId = (AttackGraphNodeId, String)` (node plus its label key),
  since mal-toolbox's `Detector` has no id. Python re-wraps into the
  real `LogEntry`.
- **`collect_logs` requires `model_asset` only for nodes that have
  detectors**, matching where Python's assert sits.
- **Python's `assert node == attack_graph.nodes[node.id]` becomes a
  liveness check** (`node_is_live`) returning `NodeNotInGraph`.
- **`state_query.py::node_ttc_value`'s precedence lives in
  `attacker_step.rs::resolve_ttc_value`** (override, else graph-level
  value, else `AttackerStepError::MissingTtcValue`).
- **`viability.rs` vs `graph_processing.py`:**
  `prune_unviable_and_unnecessary_nodes` validates every necessity entry
  before removing anything (Python fails half-pruned) and removes in
  deterministic graph order; `make_node_unviable` returns only the set
  of nodes made unviable; a missing child entry in
  `propagate_viability_from_node` is `MissingViability`, whereas
  `necessity.rs` treats it as "changed" (pre-existing inconsistency);
  `ViabilityError` is `Debug` only because it wraps upstream
  `GraphError`; Python's logging calls are dropped (no logging
  dependency in the crate).

### Dyna path

- **Node ids from accumulated state can go stale mid-episode.** A model
  effect can remove assets, including the backing asset of the step that
  triggered it, and nothing recreates those nodes. Any id taken from
  historical or cross-step state (`performed_nodes`, `enabled_defenses`,
  `GraphState` maps, per-agent TTC overrides, a sibling in the same
  effect chain, log context) must be looked up with `graph.nodes.get(id)`
  / `node_is_live` and skipped when gone; never index `graph.nodes[id]`
  on it (that panics the whole Python process with `invalid SlotMap key
  used`). Ids read from the live graph in the same call are safe.
  `stable_ids`/`id_value_map` drop dead ids centrally. When debugging
  this class, print at the mutation site in `execute_model_effects`
  (`new_assets`/`removed_assets`/`new_nodes`) before theorizing: two
  backtrace-based diagnoses of it were wrong.
- **Graph-state fields are resent in full on a dyna step whose
  `step_modification_record` is non-empty** (`insert_graph_state_fields`
  in `build_step_output`). Python's `dyna_step` detects this by
  `'ttc_values' in native_out['sim_state']` and replaces its
  `GraphState` wholesale. A full resend rather than a delta because
  `necessity_per_node` is recomputed for the whole graph on every fold.
  Test: `test_native_dyna_simulator_step_resends_graph_state_only_on_model_effect`.
- **`execute_model_effects` does not fold new nodes into `GraphState`**
  (Python's does). It returns `(modification_record, new_node_ids)`;
  `dyna_attacker_step`/`dyna_defender_step` call
  `dyna_graph_state.rs::fold_new_nodes_into_graph_state` right after.
- **There is no `dyna_attempt_attacker_step`.** Python's differs from
  the static one only by `num_attempts.get(node, 0)`, which the Rust
  `attempt_attacker_step` already does.
- **No `DynaMalSimulatorState` struct in Rust;** dyna step functions take
  `&mut GraphState` and `&mut enabled_defenses` separately.
- **`ModelSnapshot` is a typed struct, and `reconcile_model_to_snapshot`
  computes a real minimal diff.** Python's version compares int ids
  against string dict keys, so it always falls back to removing and
  re-adding every association. Same end state; this is the one place
  correctness was chosen over bug-for-bug fidelity.
- **The pristine model snapshot is captured natively** in
  `DynaSimulator`'s constructor; Python no longer calls
  `model.to_dict()` or stores a snapshot.
- **Modification records carry self-contained `AssetRef { id,
  asset_type, name }`**, captured when each op is recorded, because a
  later op in the same record can invalidate an earlier id. Wire keys:
  `kind` (`asset`/`assoc`), `type` (`ADDITIVE`/`SUBTRACTIVE`),
  `asset_{id,type,name}` or `left_asset_*`/`right_asset_*` +
  `field_name`. Only `DynaSimulator.step_native` sends
  `step_modification_record` (§12).
- **Bidirectional associations:** `Model::add_associated_assets` updates
  both sides, so which side's tuple lands in a reconcile/modification
  record depends on `HashMap` order. Tests assert on the resulting model
  state, or accept either side.

## 11. Conventions

Standing decisions made by the user that later work follows without
asking again. Each entry names the phase it was decided in.

- **Adding a crate dependency is always a question for the user**, also
  when it only promotes a transitive dependency to a direct one. The
  same goes for design choices with several defensible options: present
  them, don't pick silently.

- **Python code shadowed by native is deleted, not kept as an oracle**
  (decided at A11/B7). Once a Python function no longer has a live caller
  (one reachable from `MalSimulator`/`DynaMalSimulator`'s public API,
  `native_settings.py`, `rewards.py`, `run_simulation.py`, `config/`,
  `scenario/`, `envs/`, `policies/`, `visualization/` or `examples/`), it
  is deleted outright: no feature flag, no test-only reference copy, no
  deprecation period. Python tests of the deleted code are deleted with
  it only if Rust tests cover the same behavior; where they don't, the
  missing Rust tests are written first.
- **Live code left behind in an otherwise-dead module stays in that
  module** (decided at A11/B7). The file shrinks to just the live items
  rather than having them moved somewhere tidier, so import paths don't
  change. Example: `event_logger.py` now holds only `LogEntry`, and
  `attacker_step.py` only `attacker_is_terminated`.
- **Dead Python functionality with no Rust port gets ported, not dropped,
  when deleting it would otherwise lose test coverage** (decided at B7
  for `graph_processing.py`'s viability/pruning half). Applies even when
  nothing calls the code: coverage is kept by porting the code and its
  tests to `malsim-core`, not by keeping the Python copy alive.

- **Orchestration lives in `malsim-core`** (decided at C, 2026-10-10;
  the "one `Simulator`" part was superseded on 2026-10-11, see the next
  bullet). All reset/step orchestration (per-agent runtime state,
  `SimState`, static and dyna reset/step) lives in public `malsim-core`
  types that take plain-Rust settings structs and return a core error
  type. `malsim-pyo3` is only a thin
  wrapper: it parses Python dicts into those core structs and builds
  Python output from core results. New orchestration logic never goes
  into `malsim-pyo3`.
- **Static and dynamic simulators are separate types** (decided
  2026-10-11, replacing C's single `Simulator` with `new_dyna`). The user
  asked for "a simulator object for running the simulator with static
  instance model/attack graph" and a "DynaSimulator" for the dynamic one.
  - `malsim_core::simulator::Simulator` runs a static graph.
    `malsim_core::dyna_simulator::DynaSimulator` *contains* a `Simulator`,
    plus the shared `Model` handle and its snapshot. It reuses the
    `Simulator`'s reset and per-agent bookkeeping and swaps in only the
    dyna step functions. This is composition, the Rust analogue of
    `DynaMalSimulator(MalSimulator)`: no step-mode trait or generic, and
    no `Deref` from `DynaSimulator` to `Simulator`.
  - Python sees two pyclasses with the same method names:
    `_native.Simulator(graph)` and `_native.DynaSimulator(graph, model)`,
    both with `reset_native(settings, agents, seed)`/`step_native(actions)`.
    The `Model` is captured and snapshotted in `DynaSimulator`'s
    constructor (mirroring `DynaMalSimulator.__init__`), not passed per
    call. No `dyna_`-prefixed method names.
  - Both types return the one `SimulatorError` enum (the error convention
    below).
- **The Rust scenario types keep their Python names** (decided at C).
  They live in `malsim_core::scenario` (with submodules) and are named
  `NodePropertyRule<T>`, `AttackerSettings`, `DefenderSettings`,
  `AgentSettings` (an enum over the two), `Scenario` and
  `Scenario::load_from_file`. The exception is `MalSimulatorSettings`,
  `AttackSurfaceSettings` and `RewardMode`, which live in a top-level
  `malsim_core::settings` module (see the next bullets). Like Python, settings keep their
  *unresolved* `NodePropertyRule`s. A separate resolve step turns them
  into the id-keyed maps/sets the `Simulator` consumes. This applies
  §2.7's terminology rule to type names as well.
- **Settings shared by the scenario loader and the `Simulator` live in
  a top-level `malsim_core::settings` module** (decided at C). This
  covers `MalSimulatorSettings`, `AttackSurfaceSettings`, `RewardMode`
  and the flat inputs below; `TtcMode` stays in `graph_state.rs` and is
  re-exported there. `scenario` and `simulator` both depend on
  `settings`; `simulator` never depends on `scenario`.
- **The flat, id-resolved per-agent `Simulator` inputs are
  `FlatAttackerSettings`, `FlatDefenderSettings` and the
  `FlatAgentSettings` enum** (decided at C). They are named after
  `native_settings.py`'s `flatten_*_settings`, which produce the same
  shape on the Python side.
- **`Simulator` API: settings are passed at reset, mirroring
  `reset_native`** (decided at C). `Simulator::new(graph)`,
  `DynaSimulator::new(graph, model)`, `reset(&MalSimulatorSettings,
  agents, seed) -> Result<&SimState, _>`, `step(&actions) ->
  Result<StepOutcome, _>` (per-step deltas, like `step_native`;
  `DynaSimulator::step` returns `DynaStepOutcome`, which adds the
  modification record) and
  `state() -> Option<&SimState>` for the full accumulated state. Multiple
  entry-point sets are sampled before reset by the caller: Python's
  `get_entry_points`, or a Rust helper on `Scenario`.
- **`DynaSimulator::restore_model()` restores a dyna model before agents
  are flattened** (decided after C, 2026-10-11; it moved from `Simulator`
  to `DynaSimulator` in the split above). It is a no-op for an
  already-pristine model, and `reset` still restores too (a no-op diff
  by then). Every dyna reset caller calls it *before* resolving per-node
  settings into `FlatAgentSettings`: Python's `dyna_reset` through pyo3's
  `DynaSimulator.restore_model_native()`, and Rust callers before
  `Scenario::flatten_agents`.
- **Attacker entry points/goals are re-resolved by full name on every
  reset** (decided 2026-10-11). The Rust `Scenario` keeps a private
  full-name copy of its attacker settings, and `Scenario::flatten_agents`
  resolves it against the current graph on each call. It now returns
  `Result<_, AgentSettingsError>`; `UnknownNode` only happens if the
  caller skipped `restore_model`. The public `agent_settings` keeps the
  load-time ids. On the Python side, `DynaMalSimulator` captures the same
  full-name copy at construction and `dyna_reset` re-resolves it after
  the restore.
- **`NodePropertyRule<T: RuleValue>` is generic** (decided at C). The `RuleValue` trait
  covers three things: parsing from a JSON value, what a list-form entry
  means (`true` for bool, `1.0` for f64, an error for `TtcDist`), and
  Python truthiness. Truthiness matters because precedence is
  `by_asset_name or by_asset_type or default`, so a falsy value falls
  through, exactly as in Python.
- **Rust/Python parity checks use committed golden JSON** (decided at C). A Python
  generator dumps the resolved, non-random shape of each fixture to
  committed JSON. A pytest asserts that the JSON is current, and a Rust
  `#[test]` asserts that the Rust implementation matches it. No test-only
  FFI hooks, and `cargo test` never shells out to Python.
- **The dynamic scenario-dict representation is `serde_json::Value`** (decided at C).
  YAML is parsed with `serde_yaml` (as §7 C2 specified; it is already
  in the lockfile via mal-toolbox) straight into `serde_json::Value`.
  That is the type mal-toolbox's `maltoolbox_model::from_dict` and node
  `ttc` already use. Typed structs are built from it by hand-written
  conversion code, not by `serde` derive.
- **Errors (existing convention, restated at C):** each module gets one
  hand-written error enum with `Display` + `std::error::Error` and
  `From` impls for wrapped errors. No `thiserror`.

## 12. Decisions

Ambiguities resolved during implementation without asking (per the
standing instructions: the most reasonable choice, recorded here). Each
entry names its phase.

- **B7: the deleted-module reachability was decided against `examples/`
  and everything in `python/malsim` outside the hot-path modules as live
  roots.** Downstream code importing the deleted internals (e.g.
  `malsim.mal_simulator.attack_surface`) breaks. That is accepted by the
  §11 "delete outright" convention; none of those modules was exported
  from a package `__init__`.
- **mal-toolbox `main` is the original pure-Python toolbox.** This repo
  builds against mal-toolbox's `rust-rewrite` branch; `main` is only a
  behavioral reference.
- **C1: `NodePropertyRule.default` (the dataclass field) is not ported.**
  `value()` never reads it (it uses its `default` *argument*), and
  `from_dict()` never sets it, so it's dead state. Rust's `value()`
  returns `Option<T>` instead; `None` is where Python returns its
  `default` argument, so `rule.value(..).unwrap_or(d)` is the exact call.
- **C1: `per_node()` is keyed by node id, not full name** (§7 C1's
  `HashMap<NodeId, T>`). Full names are one `full_name_of` away and ids are
  what the `Simulator` consumes.
- **C1: TTC override values are parsed (and validated) when the rule is
  loaded.** Python keeps the raw YAML value and only calls
  `TTCDist.from_name` at reset time (`_flatten_ttc_dists`). The Rust rule
  holds parsed `TtcDist`s, so an unknown name fails at scenario load
  instead of at the first reset. Only the name form is accepted, as on
  the Python path. A dict-form TTC is rejected there too.
- **C1: value parsing is typed but Python-lenient where Python visibly
  is.** `f64` rules accept booleans (`True == 1` in Python). `bool` rules
  accept only booleans; Python would accept any truthy value, but none of
  the fixtures rely on that. An explicit `null` step value is treated as
  absent, which is what Python's `or` chain does with it.
- **C2: a scenario file without `lang_file` is reported by validation,
  not by path resolution.** Python's `load_scenario_dict` indexes
  `scenario['lang_file']` unconditionally and raises a bare `KeyError`.
  Rust skips path resolution for a missing key and lets
  `validate_scenario_dict` report `MissingField("lang_file")`. Both reject
  the file; only the error differs.
- **C2: YAML 1.2 vs PyYAML's YAML 1.1.** `serde_yaml` follows YAML 1.2,
  so `yes`/`no`/`on`/`off` are strings, not booleans as in
  `yaml.safe_load`. No fixture uses those spellings (checked by grep).
  `serde_yaml` is marked deprecated upstream; it's used anyway because §7
  C2 specified it and mal-toolbox already depends on the same version.
  Swapping it out later only touches `read_yaml_mapping`.
- **C2: `recursive_update` key order is deterministic** (old keys first,
  then new-only keys). Python iterates a `set` of keys, so its order is
  arbitrary, and nothing depends on it.
- **C3: unknown `policy` names are accepted** (carried opaquely, per
  §2.6). Python's `agent_settings_from_dict` raises `LookupError` for a
  name outside `policy_name_to_class`. The Rust loader has no policy
  registry, so `wrong_agent_classes_scenario.yml` loads in Rust and fails
  in Python. C6's parity test records this case as an expected
  divergence.
- **C3: `_validate_agent_dict`'s missing-key check is ported as
  written, i.e. it doesn't check.** Python tests `illegal_keys` twice
  instead of `missing_keys`, so a missing `policy` is accepted. Rust
  accepts it too. A missing `type` is still an error in both (Python
  raises `KeyError` when reading it; Rust returns `MissingType`).
- **C3: agent fields are type-checked where Python is duck-typed.**
  `goals` must be a list of strings (Python's `frozenset(d['goals'])`
  would turn a bare string into a set of characters), and `config` must
  be a mapping or `null` (`null` is treated as `{}`). The `sim_settings`
  booleans must be booleans, and `seed` a non-negative integer. No
  fixture is affected.
- **C3: `get_entry_points` with an empty list of alternatives returns no
  entry points.** Python's `rng.choice([])` raises instead. The case
  can't come from YAML (`entry_points: []` parses as an empty *single*
  set, as in Python), only from programmatic construction.
- **C3: `MalSimulatorSettings` is `Copy` and carries `seed:
  Option<u64>`.** Python allows any int. A negative seed would be
  rejected by numpy's `default_rng` anyway.
- **C4: `lang_file` git URLs are not supported by the Rust loader.**
  Python's `LanguageGraph.load_from_file` clones `.git` URLs (the
  `integration`-marked `simple_scenario_git_url.yml` test). Supporting
  that in Rust would need a git dependency or shelling out to `git`, for
  a path that's already network-only on the Python side. Rust returns
  `ScenarioError::Language` with an explanatory message instead.
- **C4: `Scenario` has no `to_dict`/`save_to_file`.** §7 asks only for
  loading. A writer can be added later on top of
  `NodePropertyRule::to_value`.
- **C4: `agents` must be present even though validation accepts
  `agent_settings`.** `_validate_scenario_dict` treats `('agents',
  'agent_settings')` as a one-of pair, but `from_dict` then reads
  `scenario_dict['agents']` and raises `KeyError`. Rust mirrors that by
  returning `InvalidAgents` rather than silently aliasing
  `agent_settings`.
- **C4: `sim_settings: null` means default settings.** Python would call
  `MalSimulatorSettings(**None)` and raise a `TypeError`. Treating it like
  an absent key is the natural reading, and no fixture has it.
- **C4: `Scenario::new` takes a `ModelSource` (file path or inline
  dict), not an already-built `Model`.** The model must share the
  scenario's freshly loaded `Rc<LanguageGraph>`, which a caller-built
  `Model` can't guarantee. Python's `Model`-instance branch has no safe
  Rust equivalent.
- **C5: agents are processed in name order (`BTreeMap`), not insertion
  order.** The pre-C5 pyo3 code used `HashMap`s, whose per-process random
  iteration order made RNG consumption with more than one attacker or
  defender nondeterministic across processes for a fixed seed. Sorted
  order is deterministic. It differs from Python's dict order, but seeded
  runs were never bit-compatible with Python anyway (§2.1). Native output
  dicts are also keyed in name order now.
- **Dyna reset order: restore the model, then resolve per-node
  settings.** `dyna_reset` calls `restore_model_native()` before
  flattening, and re-resolves attacker entry points/goals by full name
  (§11). Flattening against the previous episode's graph either raised
  `ValueError: node id N is not part of this simulator's attack graph`
  or left restored nodes out of the new episode's maps; a stale Python
  `AttackGraphNode` cannot even be read (`.full_name` raises
  `LookupError`). Ids of nodes that no longer exist are accepted by
  native and stay inert (slotmap keys are versioned, so they never alias
  a new node). Regression tests:
  `test_reset_restored_nodes_keep_rule_settings` and
  `test_reset_re_resolves_entry_points_and_goals_on_restored_nodes` in
  `test_dyna_mal_simulator.py`, with Rust twins of the same names
  (prefixed `dyna_reset_`) in `tests/scenario_simulator.rs`.
- **C5: `SimulatorError::ModelState` boxes its payload**, following the
  existing `DynaAttackerStepError::ModelEffects(Box<..>)` precedent
  (clippy `result_large_err`).
- **C5: model asset ids keep counting across dyna resets** (`Wiper-7`,
  then `Wiper-8`, ... for the asset wiperLang's effect adds). This was
  checked against Python's `DynaMalSimulator`, which does the same: the
  model's id counter isn't part of the restored snapshot. The Rust smoke
  test looks new nodes up by pattern instead of pinning a name. Left as
  is (matches Python).
- **C6: YAML `!!set` is read from `serde_yaml`'s untagged form.**
  PyYAML loads `entry_points: !!set {A: null}` as a Python `set`, which
  `_load_entry_points` accepts. `serde_yaml` drops the tag and yields the
  mapping `{A: null}`. The Rust loader therefore treats a mapping whose
  values are all `null` as a set, for `entry_points` and `goals`. This is
  slightly more permissive than Python: an *untagged* all-`null` mapping
  is a `ValueError` there. Found by the parity test on
  `detector_lang_scenario.yml`.
- **C6: the golden file records rate maps as `{len, nonzero}`.** Full
  per-node rate maps would make the file several times larger for no
  extra coverage (every non-listed node is `0.0`, which `len` already
  pins).
- **Dyna reset: `DynaMalSimulator.agent_settings` gets fresh
  `AttackerSettings` objects on every reset.** `dyna_reset` now returns the
  re-resolved settings as a fourth tuple element, and both callers
  (`__init__`, `reset`) store them. Code that held on to an old
  `sim.agent_settings[...]` object across a reset keeps the previous
  episode's node objects. Rules and every other field are the same objects
  (shallow copy), so the reward closures built in `__init__`, which only
  read rules, are unaffected. `dyna_reset` isn't exported from
  `malsim.dyna_mal_simulator`, so changing its signature breaks no public
  API. Entry points or goals whose asset is removed *mid-episode* are
  still not handled (e.g. a removed goal can never be performed); that
  case predates the port and is a different question from the
  reset-time staleness fixed here.
- **Simulator split (2026-10-11): the static `Simulator.step_native`
  output no longer has `step_modification_record`.** The static path
  never runs model effects, so it was always `[]`, and the core
  `StepOutcome` no longer carries the field. Only
  `DynaSimulator.step_native` sends it.
  `test_native_simulator_step_output_is_delta_only(_attacker_only)` now
  assert the exact key set `{'step_enabled_defenses'}`, so they are as
  strict as before. No Python code read the key on the static path.
- **Simulator split: `DynaMalSimulator` keeps its native handle in
  `_dyna_native_sim`, not the inherited `_native_sim`.** `MalSimulator`
  types `_native_sim` as `_native.Simulator`, and `DynaSimulator` is
  deliberately not a subtype of it (§11). `DynaMalSimulator` overrides
  every method that reads the handle (`reset`/`step`), so a dyna instance
  never needs the base attribute. `DynaMalSimulator.__getstate__` drops
  `_dyna_native_sim` from pickles, the same way the base drops
  `_native_sim`.
- **Simulator split: the pyo3 `NotReset` message names the pyclass**
  (`"Simulator.step_native() called before reset_native()"` /
  `"DynaSimulator.step_native() ..."`). Both are still `ValueError`s. The
  old dyna-only "no Model is attached" error is gone, because a
  `DynaSimulator` always has its model.
