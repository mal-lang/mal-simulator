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
- [x] Phase A - `MalSimulator` port (§5)
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
  - [x] A3 - Port static graph_state computation (`graph_state.py`,
        `graph_processing.py` necessity propagation). Landed in
        `core/malsim-core/src/graph_state.rs` (`TtcMode`,
        `default_ttc_dist_for_step_type`, `resolve_ttc_dist(_from_parts)`,
        `ttc_value_for_dist`, `attack_step_ttc_value(s)`,
        `is_pre_enabled_for_dist`, `get_pre_enabled_defenses`,
        `is_impossible_for_dist`/`is_impossible_attack_step`,
        `get_impossible_attack_steps`, `GraphState`,
        `compute_initial_graph_state`) and
        `core/malsim-core/src/necessity.rs` (`evaluate_necessity`,
        `propagate_necessity_from_node`, `calculate_necessity`) - viability
        (`calculate_viability`/`evaluate_viability`/
        `prune_unviable_and_unnecessary_nodes`) deliberately not ported,
        confirmed dead/deprecated code (see §10). `compute_initial_graph_state`
        takes `ttc_mode`/`run_defense_step_bernoullis`/
        `run_attack_step_bernoullis` directly rather than a ported
        `MalSimulatorSettings` struct - full settings porting is A9's job.
        Each graph-dependent function is split into a thin
        `AttackGraphNode`-reading wrapper plus a graph-independent helper
        operating on plain data, specifically so the logic is unit-testable
        without a real graph (see below and §10). 32 new Rust-native tests
        across both modules; `cargo test`/`cargo clippy`/`cargo fmt --check`
        all clean.
        **Deferred, explicitly, per §8's "list it explicitly when skipped"
        rule:** no Rust-native tests for `necessity.rs` at all, and none
        for the thin `AttackGraphNode`-reading wrappers in `graph_state.rs`
        (`resolve_ttc_dist`, `attack_step_ttc_value(s)`,
        `get_pre_enabled_defenses`, `is_impossible_attack_step`,
        `get_impossible_attack_steps`, `compute_initial_graph_state`) -
        every one of these needs a real `AttackGraphNode` with a specific
        `step_type`, and that type's id (`AttackStepId`) is a slotmap key
        only a real `maltoolbox_language::graph::LanguageGraph` can mint
        (confirmed: it can't be faked via `Default`/a literal, unlike
        `AttackGraphNodeId` elsewhere in these modules, which these
        functions never need to mint themselves). Building one requires
        `maltoolbox-language` as a new dev-dependency; asked the user,
        who chose to defer rather than add it in this step (see §10).
        This means `tests/test_graph_processing.py`'s necessity cases
        (`test_necessity_necessary`, `test_necessity_unnecessary`,
        `test_analyzers_apriori_propagate_necessity`) have no Rust-native
        counterpart yet. Revisit once a step actually needs the
        dependency (A4+ will hit the exact same wall for traversal
        predicates) - add it then and backfill these at the same time.
  - [x] A4 - Port graph traversal predicates (`graph_utils.py` minus
        actionability/reward). Landed in `core/malsim-core/src/graph_utils.rs`:
        `is_attack_step_type`/`is_attack_step`, `node_is_live`,
        `node_is_necessary`, `node_blocks_children_from_parts` (+ thin
        `node_blocks_children` wrapper), `node_is_blocked`,
        `and_traversable`, `node_is_traversable`, plus a `GraphUtilsError`
        enum mirroring the module's assertion/`KeyError`/`TypeError`
        failure modes. `node_is_actionable`/`node_reward` intentionally
        excluded per §2.4/§4 - unchanged, stay Python. Like `necessity.rs`,
        `node_is_blocked`/`node_is_traversable` aren't split into a
        graph-independent helper (the logic *is* graph traversal); unlike
        `necessity.rs`, `node_blocks_children`/`is_attack_step` still get
        the graph-state.rs-style split since their bodies are plain-data
        already. Matches on `node.step_type.as_str()`, not the
        `AttackStepType` enum directly, since that enum isn't re-exported
        by `maltoolbox-attackgraph` and `maltoolbox-language` is kept a
        test-only dependency of this crate (see below and §10).
        **Added `maltoolbox-language` as a `[dev-dependencies]` crate
        dependency of `malsim-core`** (workspace-level entry pinned to the
        same `mal-toolbox` git rev as `maltoolbox-attackgraph`, so no new
        external dependency tree) - asked the user per A3's deferred
        question; they chose to add it now *and* backfill A3's deferred
        tests in the same change, rather than defer again. This unblocks
        real `AttackGraphNode` fixtures (`core/malsim-core/src/
        test_fixtures.rs`, compiling `tests/testdata/langs/dummy_lang.mal`
        via `maltoolbox_language::from_mal_spec`, mirroring
        `tests/conftest.py::dummy_lang_graph`) for both this phase's tests
        and `necessity.rs`'s. 15 new Rust-native tests in
        `graph_utils.rs`, including a direct port of
        `tests/test_graph_processing.py::test_node_is_blocked`'s fixture
        and assertions. `necessity.rs` backfilled with 3 tests ported from
        `test_necessity_necessary`/`test_necessity_unnecessary`/
        `test_analyzers_apriori_propagate_necessity` (§0's A3 entry
        updated isn't needed - the deferral note there still accurately
        describes what was deferred *at the time*; this entry and §10
        record the backfill). `cargo test`/`cargo clippy`/`cargo fmt
        --check` all clean (59 tests total in `malsim-core` now); full
        Python suite (`uv run pytest tests -m "not integration"`, 156
        tests) still green, untouched by this phase.
  - [x] A5 - Port attack surface / defense surface / effects computation.
        Landed in `core/malsim-core/src/attack_surface.rs`
        (`get_effects_of_attack_step`, `get_attack_surface`) and
        `core/malsim-core/src/defense_surface.rs` (`get_defense_surface`),
        both mirroring their Python namesakes 1:1 as separate modules (per
        §2.7, not mandated, but kept consistent with A3/A4's module split).
        Per A5's own scope note, neither function knows `NodePropertyRule`
        exists: `node_is_actionable` is replaced by a private
        `node_is_actionable_flat(actionable_steps: Option<&HashSet<
        AttackGraphNodeId>>, node_id)` in each module (`None` = no rule
        configured = every node actionable, mirroring `if agent_actionability`
        being falsy; `Some(set)` = exactly the already-flattened actionable
        ids) - see each module's doc comment. `get_attack_surface` takes
        `skip_compromised`/`skip_unnecessary` as plain `bool`s rather than a
        ported `AttackSurfaceSettings`, same pattern A3 used for
        `MalSimulatorSettings` (full settings porting is still A9's job).
        14 new Rust-native tests (11 in `attack_surface.rs`, 4 in
        `defense_surface.rs`); `cargo test`/`cargo clippy`/`cargo fmt
        --check` all clean (88 tests total in `malsim-core` now); full
        Python suite (156 tests) still green, untouched by this phase - no
        Python file changed.
        **No new crate dependency needed** - reuses A4's `maltoolbox-language`
        dev-dependency and `test_fixtures.rs` fixtures, so no "ask the user"
        trigger this phase.
        **Testing note, different from A3/A4's precedent:** unlike
        `node_is_blocked`, these three functions have no existing *isolated*
        Python unit test to port 1:1 - `tests/test_attacker.py`/
        `test_defender.py`/`test_mal_simulator.py::test_actions_effects`
        only exercise them indirectly through a fully-built scenario and
        running simulator (checked via `grep -rln` across `tests/`). Per
        §8's general requirement ("identify the Python test(s) exercising
        that module's behavior... not just the same code path
        incidentally"), the Rust tests here are authored directly against
        the Python source's documented/implemented semantics using
        `test_fixtures.rs`'s hand-built dummy graphs, rather than being a
        line-for-line port of a specific existing pytest - closest in
        spirit to A4's `node_is_blocked` test but without a single source
        pytest to mirror. See §10 for the two implementation details (the
        actionability flattening shape, the `get_effects_of_attack_step`
        set-growth idiom, and the `get_attack_surface` arg-count lint
        allowance) worth a future reader's attention.
  - [x] A6 - Port false-alert + detector log generation. Landed in
        `core/malsim-core/src/false_alerts.rs` (`node_false_negative_rate`,
        `generate_false_negatives`, `node_false_positive_rate`,
        `generate_false_positives`), `core/malsim-core/src/observability.rs`
        (`observed_nodes`, plus a private `node_is_observable_flat` - third
        independent copy of A5's already-flattened-id-set idiom, same
        reasoning as `attack_surface.rs`/`defense_surface.rs`'s two copies),
        and `core/malsim-core/src/event_logger.rs` (`collect_logs`,
        `collect_false_positives`, `get_context`, `get_random_context`,
        plus an `EventLoggerError` enum mirroring `assert`/`StopIteration`
        failure modes, same pattern as A4's `GraphUtilsError`/A3's
        `NecessityError`). `NodePropertyRule[float]`/`[bool]` rate/
        observability rules are taken as already-flattened
        `Option<&HashMap<AttackGraphNodeId, f64>>`/
        `Option<&HashSet<AttackGraphNodeId>>` per §2.4, same pattern as A5.
        Per its own scope note, `LogEntry`'s Python `detector`/`trigger`
        fields became id-based on the Rust side: `trigger:
        AttackGraphNodeId`, and a new `DetectorId = (AttackGraphNodeId,
        String)` type alias (node id + its label key in that node's
        `detectors` map) stands in for a detector, since mal-toolbox's
        `Detector` type has no id of its own - see §10 for the full
        reasoning. 29 new Rust-native tests (13 in `false_alerts.rs`, 6 in
        `observability.rs`, 10 in `event_logger.rs`); `cargo test`/`cargo
        clippy`/`cargo fmt --check` all clean (103 tests total in
        `malsim-core` now); full Python suite (156 tests) still green,
        untouched by this phase - no Python file changed.
        **No new crate dependency needed** - reuses `rand`/`RngExt`
        (already used by `ttc.rs`) and `maltoolbox-language`'s test-only
        dev-dependency/`test_fixtures.rs` from A4, so no "ask the user"
        trigger this phase. Detector fixtures for tests are hand-built
        directly (`Detector { .. }` literals inserted into a dummy node's
        `.detectors` map) rather than via MAL language syntax - mal-
        toolbox's `Detector` needs no `LanguageGraph`-minted id (unlike
        `AttackStepId`), so `tests/testdata/langs/dummy_lang.mal` (which
        declares no detectors) didn't need extending.
        **Testing note, same situation as A5:** no existing *isolated*
        Python unit test to port 1:1 for `false_alerts.py`/
        `observability.py` (checked via `grep -rln` - only exercised
        indirectly through `tests/test_mal_simulator.py`'s
        `test_simulator_false_positives*`/`test_simulator_false_negatives`/
        the observability steps around line 309, all seed-pinned but only
        on structural/count assertions, not exact values, so none needed
        flagging per §2.1). `tests/test_event_logger.py` *does* exercise
        `event_logger.py`'s behavior somewhat more directly (forcing
        detector tp/fp rates and asserting on `DefenderState.logs`), but
        still through a fully-built scenario + running simulator, not an
        isolated unit test - its specific edge cases (tprate=1.0
        deterministic true positive, fprate=1.0 deterministic false
        positive, a *negative* tprate being "truthy but never fires") were
        ported as direct `collect_logs`/`collect_false_positives` unit
        tests instead (`collect_logs_tprate_one_always_true_positive`,
        `collect_false_positives_fprate_one_always_fires`,
        `collect_logs_tprate_negative_is_truthy_but_never_fires`), rather
        than a line-for-line port of the scenario-level test.
  - [x] A7 - Port attacker_step / defender_step orchestration. Landed in
        `core/malsim-core/src/attacker_step.rs` (`attacker_is_terminated`,
        `attempt_attacker_step`, `attacker_step_effects`, `attacker_step`,
        plus an `AttackerStepError` enum composing `GraphUtilsError`/
        `GraphStateError` with two bespoke variants - `MissingTtcValue`,
        `NodeNotInGraph`) and `core/malsim-core/src/defender_step.rs`
        (`defender_step`, `defender_is_terminated`, plus a
        `DefenderStepError` enum with just `NodeNotInGraph`). No
        `AttackerState`/`DefenderState`/`AgentStates` exist on the Rust
        side yet (A9's job - §3); per-agent inputs these functions need
        (`action_surface`, `goals`, `performed_nodes`, `num_attempts`,
        per-agent ttc overrides) are taken as plain already-flattened
        `HashSet`/`HashMap` arguments, continuing A3/A5/A6's pattern - see
        §10 for `state_query.py::node_ttc_value`'s pull-forward into this
        module and two preserved-as-is behavioral oddities worth a future
        reader's attention. 22 new Rust-native tests (16 in
        `attacker_step.rs`, 6 in `defender_step.rs`), authored directly
        against the Python source's documented/implemented semantics using
        `test_fixtures.rs`'s dummy graphs (same situation as A5/A6: no
        existing *isolated* Python unit test for `attacker_step`/
        `defender_step` - `tests/test_mal_simulator.py::test_attacker_step`/
        `test_defender_step` exist but only exercise these functions
        through a fully-built corelang scenario, checked via `grep -rln`);
        `cargo test`/`cargo clippy`/`cargo fmt --check` all clean (125
        tests total in `malsim-core` now); full Python suite (156 tests)
        still green, untouched by this phase - no Python file changed.
        **No new crate dependency needed** - reuses A2's `rand`, A4's
        `maltoolbox-language` dev-dependency/`test_fixtures.rs`, and A3/
        A5's own modules (`graph_state::resolve_ttc_dist`/`TtcMode`,
        `graph_utils::node_is_live`/`node_is_traversable`,
        `attack_surface::get_effects_of_attack_step`), so no "ask the
        user" trigger this phase.
  - [x] A8 - `malsim-pyo3` native `Simulator` pyclass: `reset_native`/
        `step_native` returning plain Python primitives. Landed in
        `py-bindings/malsim-pyo3/src/simulator.rs`: a `#[pyclass(name =
        "Simulator", module = "malsim._native", unsendable)]` holding the
        shared `Rc<RefCell<AttackGraph>>` (via A1's `extract_shared_graph`,
        now `pub(crate)` so this module can reuse it) plus an
        `Option<SimState>` (`None` until `reset_native` runs).
        `reset_native(settings: dict, agents: dict, seed: int) -> dict`
        parses a flattened settings dict (§2.4 shape: `ttc_mode` as its
        enum variant's name string, both `AttackSurfaceSettings` fields,
        both bernoulli toggles, `compromise_entrypoints_at_start` - all
        with the same defaults as `MalSimulatorSettings`/
        `AttackSurfaceSettings`) and a per-agent dict (`"type"`:
        `"attacker"`/`"defender"` plus already-flattened id
        sets/maps for entry points/goals/actionable/observable steps/FP
        and FN rates - no `NodePropertyRule` parsing here, per §2.4),
        composes A2-A7's ported functions into a full reset (mirroring
        `reset_agent.py`'s `initial_attacker_state`/
        `initial_defender_state` - entry-point compromise-at-start,
        initial action surfaces, initial `observed_nodes`/logs for
        defenders from the pre-compromised set) and `step_native(actions:
        dict[str, list[int]]) -> dict` composes them into a full step
        (defenders act first via `defender_step`, folding newly-enabled
        defenses into `enabled_defenses` before any attacker acts via
        `attacker_step`, then every agent's action surface/observed
        nodes/logs are recomputed - same ordering and two-phase
        "compute-then-update" shape as `simulator.py::step`, verified
        line-by-line against it, not just read-through). All node
        references crossing the FFI boundary (both directions, both
        functions) are the stable `AttackGraphNode.id: i64` -
        `AttackGraph::id_to_node`/`graph.nodes[id].id` are the two
        translation directions (`to_node_id`/`stable_ids` helpers) to/from
        the internal `AttackGraphNodeId` slotmap key malsim-core's
        hot-path functions use - see §10 for why the i64 is the right
        choice here, not the slotmap key. Return shape is deliberately
        boring nested `dict`/`list`/`bool`/`float` (no custom pyclasses),
        per A8's own plan text. Explicit scope cuts (left for A9, which
        has the real settings-flattening to extend this properly): no
        per-agent TTC distribution overrides, no "multiple entry point
        sets, sampled at reset" support (`AttackerSettings.entry_points`
        as `tuple[Set, ...]`) - only a single flat entry-point set per
        attacker; no rewards (unchanged, pure Python, per §2.4 regardless
        of phase). `tests/test_native.py` gained 5 tests exercising
        `_native.Simulator` end to end through a real scenario's attack
        graph (entry-point compromise-at-start on/off, stepping an
        attacker action, calling `step_native` before `reset_native`, an
        unknown agent name in `actions`) - no existing isolated Python
        unit test to port 1:1 for this module (it doesn't correspond to
        any single Python file), same situation A5-A7 were in; written
        directly against `simulator.py`/`reset_agent.py`/
        `attacker_state_factories.py`/`defender_state_factories.py`'s
        documented/implemented semantics instead, checked line-by-line
        (see §10 for the specific call sites compared). `cargo test`/
        `cargo clippy`/`cargo fmt --check` clean in both the root
        (`malsim-core`, 125 tests, unchanged by this phase) and
        `py-bindings` (separate) Cargo workspaces; full Python suite
        (`uv run pytest tests -m "not integration"`, 161 tests including
        the 5 new ones) green; `uv run mypy python/malsim tests`/`uv run
        ruff check`/`uv run ruff format --check` clean (the 3 remaining
        mypy errors predate this phase - confirmed via `git stash`, none
        in files this phase touches).
        **New direct crate dependency, asked the user per standing
        policy - approved.** `rand = "0.10.3"` added as a *direct*
        dependency of `py-bindings/malsim-pyo3/Cargo.toml` (previously
        only present transitively via `malsim-core`, and only in the
        *other*, separate root `Cargo.toml` workspace) - the native
        `Simulator` pyclass owns a `StdRng` seeded from `reset_native`'s
        `seed` argument across the whole `SimState`'s lifetime (reset
        through every subsequent `step_native` call), so `malsim-pyo3`
        itself needs to construct one, not just consume one passed in.
        Same pinned version already used by `malsim-core`, so no new
        version/feature-set to reconcile.
        **Process note, not an architecture decision - recorded for
        completeness.** A first draft of this phase's implementation was
        produced by a subagent that had been scoped to research-only
        (mapping the Python orchestration semantics below) but instead
        wrote and started debugging the actual Rust implementation
        unprompted, including the `rand` dependency edit above before it
        had been asked about. The draft was stopped mid-build-fix,
        reviewed in full against `malsim-core`'s real function signatures
        and the Python source line-by-line (not trusted at face value),
        and fixed up (a lifetime bug in the agent-config parsing loop, two
        clippy lints, this section's missing `PORTING_NOTES.md`/`.pyi`
        stub updates, and the `_pre_step_check`-equivalent unknown-agent
        error this entry mentions above, which the draft hadn't included)
        before being treated as this phase's real output.
  - [x] A9 - Rewrite Python `MalSimulator.reset()`/`.step()` to delegate to
        native, rebuild `AttackerState`/`DefenderState` from native output.
        Landed across both crates and the Python package:
        - `py-bindings/malsim-pyo3/src/simulator.rs` extended (not just
          consumed as-is): per-agent `ttc_dists` overrides are now parsed
          in `reset_native` (`parse_ttc_dist_from_py`/
          `extract_optional_ttc_dist_overrides`/`attacker_ttc_overrides`,
          built directly against `ttc::TtcDist::new`/`with_combine`/
          `named_ttc_dist` rather than `serde_json::Value`, so no new
          direct `serde_json` dependency was needed on this crate - see
          §10), wired into `attacker_step`'s existing (previously
          hardcoded `None, None`) override parameters, and exposed per
          attacker as `ttc_values`/`impossible_steps` in `build_output`.
          `build_output`'s `sim_state` dict also gained `ttc_values`/
          `impossible_attack_steps`/`necessity_per_node`/
          `pre_enabled_defenses` (previously only `enabled_defenses`) so
          Python can reconstruct a full `GraphState`. "Multiple entry
          point sets, sampled at reset" (A8's other scope cut) is
          deliberately *not* added to native - resolved in Python instead
          (see below) - see §10 for why.
        - New `python/malsim/mal_simulator/native_settings.py`:
          `flatten_sim_settings`/`flatten_attacker_settings`/
          `flatten_defender_settings`, resolving each `NodePropertyRule`
          against the graph *once* per §2.4, reusing the existing
          `node_is_actionable`/`node_is_observable`/
          `node_false_positive_rate`/`node_false_negative_rate` helpers
          per node (not reimplementing `NodePropertyRule.value()`'s
          precedence) for exact semantic parity with the pure-Python path.
        - New `create_attacker_state_from_native`/
          `create_defender_state_from_native` functions added
          *alongside* (not replacing) `create_attacker_state`/
          `create_defender_state` in `attacker_state_factories.py`/
          `defender_state_factories.py` - see §10 for why the old
          functions had to stay untouched (`DynaMalSimulator`). Resolve
          ids back to real `AttackGraphNode`/`LogEntry`/`Detector` objects
          from native's output; unlike the old factories, most fields are
          read directly from native's already-full-episode-accumulated
          output rather than merged against `previous_state` - only
          `performed_nodes_order` (pure Python bookkeeping, no native
          equivalent) is still built incrementally via a diff. `num_attempts`
          is deliberately densified back to
          `dict.fromkeys(attack_graph.attack_steps, 0)` plus native's
          sparse overlay - see §10.
        - `attacker_state_factories.py::get_entry_points`'s signature
          changed from `(sim_state, ...)` to `(attack_graph, ...)` (its
          body only ever read `sim_state.attack_graph`) so `MalSimulator`'s
          new `reset()` can resolve "multiple entry point sets" sampling
          *before* a `MalSimulatorState` exists yet to pass into
          `reset_native`; its one caller (`initial_attacker_state`)
          updated to pass `sim_state.attack_graph` - pure signature
          narrowing, no behavior change, `DynaMalSimulator` doesn't call
          this function at all (confirmed via grep).
        - `MalSimulator.__init__`/`.reset()`/`.step()` (outer signatures
          unchanged) now hold one `malsim._native.Simulator` instance
          (`self._native_sim`) for the simulator's lifetime, calling
          `reset_native`/`step_native` once per call and rebuilding
          `AttackerState`/`DefenderState`/`MalSimulatorState` from the
          output via the new native-driven factories above. `_native_sim`
          is excluded from `__getstate__` the same way
          `_defender_reward_fns`/`_attacker_reward_fns` already are - see
          §10 for why this is an existing-pattern extension, not a new
          limitation. Module-level `reset()`/`step()` lost their `rng`-only
          consumption inside `step()` (native owns stepping's RNG
          entirely now) - `step()`'s signature dropped the now-unused
          `rng` parameter. A new `_ordered_new_nodes` helper reconstructs
          `recording`'s per-step node lists (explicitly-requested actions
          first in request order, then any effect-chain-only nodes) since
          native returns these as unordered sets, not Python's old
          order-preserving sequential loop - see §10.
        - New test-support-only `_native.set_detector_rates(graph,
          node_id, label, tprate, fprate)` pyfunction (same tier as A1's
          `node_count` - not a real public API) added to
          `py-bindings/malsim-pyo3/src/lib.rs`, plus a new
          `collect_logs`-model_asset-check fix in
          `core/malsim-core/src/event_logger.rs` and a `tests/conftest.py`
          `connect_nodes` helper - all three are fixes for genuine bugs/
          gaps this phase's end-to-end wiring exposed for the first time;
          see §10 for each.
        - Full test suite green: `uv run pytest tests` (162, including
          `integration`) and `uv run pytest examples/*` (6) all pass;
          `uv run mypy python/malsim tests` has the same 3 pre-existing
          errors A8 already found (confirmed via `git diff --stat` on the
          3 affected files - none touched by A9); `uv run ruff check`/
          `ruff format --check` clean. `cargo test`/`clippy`/`fmt --check`
          clean in both the root (`malsim-core`, 126 tests, +1 from this
          phase's `collect_logs` fix) and `py-bindings` (separate)
          workspaces.
        - **No new crate dependency this phase** - the per-agent
          `ttc_dists` override parsing deliberately avoided needing
          `serde_json` as a direct `malsim-pyo3` dependency (see above and
          §10), so the standing "ask before adding a new crate dependency"
          policy wasn't triggered.
        - Several pre-existing seed-pinned exact-value/exact-order tests
          broke as a direct, expected consequence of §2.1 (RNG) and a
          newly-identified sibling category (collection iteration order -
          see §10) - all identified, relaxed to structural assertions or
          re-pinned to this port's own (still fully deterministic per
          seed) output, never silently left broken. Full list in §9.
  - [x] A10 - Audited every other public method on `MalSimulator`
        (`node_ttc_value`, `node_is_actionable`, `node_reward`,
        `node_is_observable`, `node_false_positive_rate`,
        `node_false_negative_rate`, `node_is_blocked`, `node_is_necessary`,
        `node_is_enabled_defense`, `node_is_compromised`,
        `compromised_nodes`, `node_is_traversable`, `get_node`,
        `agent_reward_by_name`, `agent_reward`, `agent_is_terminated`,
        `done`, `alive_agents`, `agent_states`) against §5's own two
        example criteria. **Result: zero code changes** - confirmed via
        `git diff c23cf57 67c6438 -- python/malsim/mal_simulator/
        simulator.py` (the A8→A9 diff) that every one of these method
        bodies, and all of `graph_utils.py`/`state_query.py`/
        `node_getters.py` (`git diff c23cf57 HEAD` on those three files is
        empty), is byte-for-byte unchanged since before A9 - they already
        satisfied A10's intent without being touched, because A9 kept
        `GraphState`/`MalSimulatorState`/`AttackerState`/`DefenderState`'s
        *shape* identical to the pre-port shape (§3's "mirrored, not
        wrapped") rather than these methods being rewritten to reach past
        that mirror. Sorted into three categories, not just §5's two:
        - **Reads native-computed data directly** (§5's first example
          category): `node_ttc_value` (no-`agent_name` branch:
          `sim_state.graph_state.ttc_values[node]`, agent branch:
          `state_query.node_ttc_value`, both backed by A9's
          `_graph_state_from_native`/per-agent `ttc_values` override map),
          `node_is_necessary` (`sim_state.graph_state.necessity_per_node`,
          native-computed once at reset), `compromised_nodes`/
          `node_is_compromised`/`node_is_enabled_defense` (iterate
          `AttackerState.performed_nodes`/`DefenderState.performed_nodes`
          in `self._agent_states`, native-mirrored every step since A9).
        - **Stays against the pure-Python mirror, never touches native**
          (§5's second example category, per §2.4): `node_is_actionable`,
          `node_reward`, `node_is_observable`, `node_false_positive_rate`,
          `node_false_negative_rate` (all read a `NodePropertyRule`
          straight off `self.agent_settings[agent_name]`), `agent_reward`/
          `agent_reward_by_name` (pure-Python reward closures, §2.4,
          unaffected by any phase of this port).
        - **A third, hybrid category §5's text didn't anticipate**:
          `node_is_blocked`/`node_is_traversable` read native-computed,
          per-episode-mirrored inputs (`sim_state.graph_state.
          impossible_attack_steps`/`necessity_per_node`,
          `sim_state.enabled_defenses`) but still run the final AND/OR
          blocked/traversable predicate as plain Python over those inputs,
          rather than calling back into `malsim-core::graph_utils`'s A4
          port for a single-node query. Deliberately left this way: these
          are O(parents) pure-data checks with no RNG, called ad hoc
          outside the hot `step()` loop (which already goes fully through
          native) - routing a one-node query back across the FFI boundary
          would add call overhead for no measurable benefit. Flagged as a
          two-implementations-can-drift risk in §9, same shape as §9's
          existing `NodePropertyRule` entry.
        **Confirmed no `MalSimulator` method reaches for `self._native_sim`
        outside `reset()`/`step()`** - relevant because `DynaMalSimulator`
        (`dyna_mal_simulator/simulator.py`) subclasses `MalSimulator`,
        overrides `__init__`/`reset()`/`step()` with its own pure-Python
        versions, and never sets `self._native_sim` at all; all 19 audited
        methods are inherited unmodified and operate generically on
        `self.sim_state`/`self._agent_states`/`self.agent_settings`, so
        they work correctly against either subclass's state without
        caring which one built it.
        **Consequence for A11:** `graph_utils.py`'s `node_is_blocked`/
        `node_is_necessary`/`node_is_traversable` and all of
        `state_query.py`/`node_getters.py` are *not* dead code shadowed by
        the Rust port, even once A11 lands - `MalSimulator`'s public query
        API calls them directly, by this phase's explicit decision. A11's
        "delete now-dead pure-Python hot-path modules" sweep should treat
        these as permanently kept, not as deferred-deletion candidates.
        **No new crate dependency, no Rust code touched this phase** - A10
        is a pure audit of already-correct Python, so the standing
        "ask before adding a dependency" policy and §8's "write a
        Rust-native test for whatever this step ported" both have nothing
        to trigger: nothing was ported, only inspected. Full gate re-run
        to confirm nothing regressed while auditing: `uv run pytest tests
        -m "not integration"` (161 passed), `uv run mypy python/malsim
        tests` (no issues), `uv run ruff check` / `ruff format --check`
        (clean) - all green, unchanged from A9's exit state.
  - [x] A11 - Full existing test suite green with native backend; delete
        now-dead pure-Python hot-path modules. **Landed together with B7
        in one cleanup, since both phases delete code from the same
        reachability analysis and several deleted modules (e.g.
        `attack_surface.py`, `ttc_utils.py`'s graph helpers) were last
        reachable only from the pre-B4 dyna path.** See B7 below for the
        full writeup; the A11-specific part is the `mal_simulator/` half of
        the deletions. User decisions taken here (recorded in §11):
        delete outright rather than keep oracles, and keep live survivors
        in their original modules.
- [x] Phase B - `DynaMalSimulator` port (§6)
  - [x] B1 - Port association-traversal evaluation + model-effect
        application (`process_assoc_traversal.py`, `model_effects.py`) to
        `malsim-core`. Landed in `core/malsim-core/src/assoc_traversal.rs`
        (`sample_size`, `apply_quantity_filter`, `traverse_association_chain`
        + its private `assoc_traversal`/`glob_assoc_traversal`/
        `assoc_set_traversal`/`resolve_terminal_traversal` helpers,
        `parse_addition`, `parse_removal`, plus an `AssocTraversalError`
        enum) and `core/malsim-core/src/model_effects.rs` (`apply_model_effect`,
        `execute_model_effects`, the four `target_op` closures as private
        `add_asset_op`/`remove_asset_op`/`add_assoc_op`/`remove_assoc_op`
        functions, `AssetOp`/`AssocOp`/`ModEffectOp` - this port's
        `simulator_state.py` equivalent, since no `MalSimulatorState`-style
        struct exists in `malsim-core` to extend, see below - and a
        `ModelEffectsError` enum). Operates entirely on `i64` asset ids
        (never `ModelAsset` object references, unlike Python) and `AssetId`
        language-graph type ids for not-yet-created assets - both already
        how `maltoolbox-model`/`maltoolbox-attackgraph`'s Rust types work
        (`AttackGraphNode.model_asset: Option<i64>`, `Model::associated_assets:
        HashMap<String, HashSet<i64>>`), so no new id/reference scheme was
        invented. 14 new Rust-native tests (12 in `assoc_traversal.rs`, 2 in
        `model_effects.rs`); `cargo test`/`cargo clippy -D warnings`/`cargo
        fmt --check` all clean (140 tests total in `malsim-core` now, from
        126 pre-B1); full Python suite (164 tests)
        still green, untouched by this phase - no Python file changed.
        **New direct crate dependencies, asked the user per standing policy -
        approved.** `maltoolbox-model` (previously transitive via
        `maltoolbox-attackgraph`, same "promote transitive to direct" shape
        as A8's `rand`) *and* `maltoolbox-language` (previously a
        `[dev-dependencies]`-only entry since A4 - promoted to a normal
        dependency because this phase's *production* code, not just tests,
        needs to name `LanguageGraphModelEffect`/`AssocTraversalChain`/
        `DynTarget`/`QuantityFilter`/etc. directly, and `maltoolbox-attackgraph`
        doesn't re-export them) - both already exist in the workspace at the
        pinned `mal-toolbox` rev, no new external dependency tree. See §10 for
        the deliberate deviations this phase introduced from the Python
        source (the B1/B2 module split for `execute_model_effects`, and two
        ported-as-unreachable-in-Rust Python quirks).
        **Testing approach:** `tests/test_dyna_mal_simulator.py::test_assoc_traversal`/
        `test_apply_model_effect` were ported close to line-for-line (same
        `wiperLang.mal`/`wiper_model.yml` fixtures, now also compiled/loaded
        directly in Rust via `test_fixtures.rs`'s new `wiper_lang_graph`/
        `wiper_attack_graph` helpers - mirrors A4's `dummy_lang_graph`
        precedent); `test_apply_model_effect_modification_record_partially_regenerates_graph`
        has no Rust equivalent of its exact fixture (the
        `dynamic_remove_many_assoc` scenario needs Phase C's not-yet-ported
        scenario-YAML loader to build outside Python) - its *property*
        (removal's modification record, replayed through
        `partially_regenerate_graph`, matches a graph rebuilt fresh from the
        same mutated model) is instead exercised via wiperLang's
        `Wiper:trigger` subtractive effect in a hand-built
        `execute_model_effects_removal_matches_fresh_graph_rebuild` test.
  - [x] B2 - Port model-snapshot reconciliation + dyna step orchestration
        (`model_state.py`, dyna `graph_state.py`/`attacker_step.py`/
        `defender_step.py`, `simulator_state.py`'s `DynaMalSimulatorState`)
        to `malsim-core`, composing B1 with Phase A's existing functions.
        Landed in `core/malsim-core/src/dyna_graph_state.rs`
        (`add_new_nodes_to_graph_state` + a `pub(crate)` `fold_new_nodes_into_graph_state`
        helper shared by both dyna step modules below), `core/malsim-core/
        src/dyna_attacker_step.rs` (`dyna_attacker_step` + a
        `DynaAttackerStepError` enum - no separate `dyna_attempt_attacker_step`,
        see below), `core/malsim-core/src/dyna_defender_step.rs`
        (`dyna_defender_step` + `DynaDefenderStepError`), and
        `core/malsim-core/src/model_state.rs` (`capture_model_snapshot`,
        `reconcile_model_to_snapshot`, `reset_model_effects`, a
        `ModelSnapshot`/`ModelSnapshotAsset` pair standing in for Python's
        `dict[str, Any]` snapshot shape - see §10 for why this isn't a
        literal port of that shape - plus a `ModelStateError` enum).
        **No `DynaMalSimulatorState` struct added** - confirmed (per A3/A5/
        A8's precedent and A9's note that full settings/state-struct porting
        is deferred to the native-`Simulator`-equivalent integration phase)
        that `malsim-core` has no `MalSimulatorState`-shaped struct at all to
        extend; B2's functions take `graph_state: &mut GraphState`/
        `enabled_defenses: &mut HashSet<AttackGraphNodeId>` as separate
        `&mut` parameters instead, mirroring A7's existing flattened-args
        pattern. That composition (building an actual dyna state struct) is
        B4/B5's job once a real native dyna `Simulator` needs one, same as
        A8/A9 did for Phase A. 9 new Rust-native tests (1 in
        `dyna_graph_state.rs`, 3 in `dyna_attacker_step.rs`, 3 in
        `dyna_defender_step.rs`, 2 in `model_state.rs`); `cargo test`/`cargo
        clippy -D warnings`/`cargo fmt --check` all clean (149 tests total in
        `malsim-core` now, from B1's exit count of 140); full Python suite
        (164 tests) still green, untouched - no Python file changed.
        **No new crate dependency this phase** - reuses B1's `maltoolbox-model`/
        `maltoolbox-language` promotions and Phase A's existing `attacker_step`/
        `defender_step`/`graph_state`/`graph_utils`/`necessity` modules, so the
        standing "ask before adding a dependency" policy wasn't triggered.
        **`dyna_attempt_attacker_step` doesn't exist as a separate Rust
        function.** Python's version is identical to A7's `attempt_attacker_step`
        except it resolves `agent.num_attempts.get(node, 0)` instead of
        `agent.num_attempts[node]` (a node created mid-simulation may not be
        seeded in the dict yet) - but A7's Rust `attacker_step` *already*
        resolves its `num_attempts_before` argument the same defensive way
        (`num_attempts.get(&node_id).copied().unwrap_or(0)`, for the identical
        reason), so `dyna_attacker_step` reuses `attacker_step::attempt_attacker_step`
        directly with no behavioral gap - confirmed by diffing the two Python
        functions line-by-line, not assumed.
        **Testing approach:** no isolated Python unit test covers the full
        wrapped dyna step (`test_attacker_step`/`test_remove_before_add`/the
        scenario-level `test_int_dynamic_test_lang*` cases all go through a
        fully-built scenario + running simulator, same situation A5-A7 were
        in) - Rust tests are hand-built directly against the Python source's
        documented/implemented semantics instead, using wiperLang (entry-point
        compromise of `InfectedDevice:infect` triggering its model effect, then
        confirming the newly-regenerated `Wiper-7:*` nodes are folded into the
        returned `graph_state`) for `dyna_attacker_step`, and a `dummy_lang.mal`
        defense step (no model effects declared, so `dummy_lang.mal` didn't
        need extending) for `dyna_defender_step`'s wiring-only checks.
        **One flaky test found and fixed before landing, not worked around
        later.** An early version of `reconcile_model_to_snapshot`'s test
        asserted a *specific* association tuple (`(infected_device, "sendTo",
        c2_server)`) appeared in the returned `new_associations` set; this
        failed in roughly 4 of 5 separate `cargo test` invocations (never
        within one invocation's repeated runs) because `Model::add_associated_assets`
        updates *both* sides of a bidirectional association (e.g. `sendTo`/
        `receiveFrom`) symmetrically, so depending on `HashMap` iteration
        order (randomized per-process) of `reconcile_model_to_snapshot`'s
        asset loop, *either* asset's field-diff pass ends up the one that
        explicitly records the fix - the other side gets the correct end
        state for free, with nothing recorded under its own tuple. Root-caused
        by hand-tracing `add_associated_assets`'s opposite-fieldname update,
        then confirmed by a 25-iteration repeat-run stress test before and
        after the fix (deterministic pass after relaxing the assertion to
        accept either side's tuple, or just the resulting model state).
  - [x] B3 - Prove a shared `Model` handle end to end (A1-equivalent).
        **Upstream mal-toolbox change (this phase's one piece reaching
        outside this repo, per §6):** added `PyModel::__inner_capsule__`
        in `py-bindings/maltoolbox-model-py/src/model.rs`, mirroring
        `PyAttackGraph::__inner_capsule__` (`c854d1d6`) exactly - same
        `PyCapsule`-with-destructor shape, same strong-count-bump-then-
        `Rc::from_raw` extraction discipline, capsule name
        `"maltoolbox._native.Model.inner"`. Landed as mal-toolbox commit
        `b96258bad474282b975245d9848f7c25d195d508` on `rust-rewrite`
        (pushed to `origin` - this repo's git deps fetch over HTTPS from
        the public remote, not a local checkout, so the commit had to be
        on `origin` before `malsim-pyo3` could pin it). Both repos'
        `cargo test`/`clippy -D warnings`/`fmt --check` clean before and
        after.
        **Landed in this repo:** `Cargo.toml`/`py-bindings/Cargo.toml`'s
        `maltoolbox-*` git deps re-pinned from `c854d1d6` to `b96258bad`;
        `maltoolbox-model` added as a *direct* dependency of
        `py-bindings/malsim-pyo3/Cargo.toml` (previously only transitive
        via `malsim-core` → `maltoolbox-attackgraph`, same "promote
        transitive to direct" shape as A8's `rand` and B1's own promotions
        of this exact crate - asked the user per standing policy,
        approved). `py-bindings/malsim-pyo3/src/lib.rs` gained
        `extract_shared_model` (Model counterpart of A1's
        `extract_shared_graph`, `pub(crate)` for B4 to reuse), and two
        test-support-only `#[pyfunction]`s at the same tier as A1's
        `node_count`/A9's `set_detector_rates` - not real public API:
        `model_asset_count` (read-only smoke test, mirrors `node_count`)
        and `model_add_asset_native` (mutates via the shared handle,
        calling `maltoolbox_model::Model::add_asset` directly) - the
        latter exists specifically to prove B3's *double-visibility*
        requirement in both directions, not just one (see below).
        `python/malsim/_native-stubs/__init__.pyi` gained both functions'
        signatures.
        **Test:** `tests/test_native.py::
        test_native_model_asset_count_matches_python_and_sees_both_sides_mutations`
        - loads `simple_scenario.yml`, checks `_native.model_asset_count`
        against `len(scenario.model.assets)`, adds an asset via Python's
        `model.add_asset(...)` and re-checks (proves native sees a
        Python-side mutation), then adds one via the new
        `model_add_asset_native` and checks the new id shows up in
        `model.assets` (proves Python sees a native-side mutation) - this
        second direction is the one thing A1's own smoke test never needed
        to prove, since Phase A never mutates the shared graph from both
        sides at once (§6). Full gate green: `uv run pytest tests`
        (165 passed) and `examples/*` (6 passed); `mypy`/`ruff check`/
        `ruff format --check` clean; `cargo test`/`clippy -D warnings`/
        `fmt --check` clean in both the root (`malsim-core`, 149 tests,
        unchanged by this phase) and `py-bindings` workspaces.
        **Local build-environment gotcha found and worked around, not a
        code defect - see §10 for the full writeup.** `uv run maturin
        develop --uv`'s own dependency-resolution step for the
        `mal-toolbox` git dependency intermittently installed a wheel
        whose compiled `.so` was missing `PyModel.__inner_capsule__`
        entirely, even immediately after `uv cache clean mal-toolbox` and
        even though `direct_url.json` correctly reported the new commit
        sha. Root-caused to the `uv`/`maturin` build pipeline specifically
        in this sandbox, not the Rust source: a plain `cargo build`
        against the exact same checkout (with `PYO3_PYTHON` set to match)
        and, more convincingly, a from-scratch `git clone` +
        `maturin build --release` (the same pipeline a real install uses)
        both produced a correct wheel exposing the method - confirmed via
        `strings` on the `.so` and an actual Python import. Workaround
        used to unblock this phase's test run: build the wheel via the
        from-scratch clone + `maturin build --release` path directly and
        `uv pip install --reinstall-package mal-toolbox <that wheel>`
        instead of relying on `uv run maturin develop`'s own resolution
        for the `mal-toolbox` dependency specifically. Flagged in §9 as a
        risk for whoever picks up B4 next in this same sandbox - don't
        assume `uv run maturin develop` alone reflects a fresh mal-toolbox
        commit; verify with `strings <the installed .so> | grep
        __inner_capsule__` (or equivalent) before trusting a failing
        Python-level test actually indicates a Rust-side bug.
  - [x] B4 - Native dyna reset/step entry points in `malsim-pyo3`
        (A8-equivalent). Landed in `py-bindings/malsim-pyo3/src/simulator.rs`:
        extended the existing `Simulator` pyclass (per an explicit "extend vs.
        new sibling pyclass" decision asked of and answered by the user -
        extend, to reuse `AttackerRuntime`/`DefenderRuntime`/`SimState`/the
        output-building helpers rather than duplicate ~650 lines of near-
        identical bookkeeping) with a `dyna: Option<DynaHandle>` field
        (`DynaHandle { model: Rc<RefCell<Model>>, snapshot: ModelSnapshot }`,
        `None` until first used - plain `MalSimulator` never populates it)
        and two new pymethods: `dyna_reset_native(settings, agents, model,
        seed)` and `dyna_step_native(actions)`. `reset_native`'s body was
        split into a private `do_reset` helper both `reset_native` and
        `dyna_reset_native` call (the latter first restoring the shared
        `Model`/`AttackGraph` to the pristine snapshot via B2's
        `reset_model_effects`); `step_native`'s "update attacker/defender
        runtimes" blocks were extracted into `update_attacker_runtimes`/
        `update_defender_runtimes` methods shared with `dyna_step_native`
        (which calls B2's `dyna_attacker_step`/`dyna_defender_step` instead
        of A7's plain `attacker_step`/`defender_step` for the actual
        stepping, then reuses the same update/output-building helpers).
        **Second design decision asked of and answered by the user:** the
        pristine model snapshot (needed to restore model-effect mutations on
        every `dyna_reset_native` call) is captured *natively*, via B2's
        `capture_model_snapshot` on the live shared `Model`, the first time
        `dyna_reset_native`'s `model` argument is seen - not passed in from
        Python as a dict, despite this phase's own §6 planning text having
        sketched a `model_snapshot` parameter. Python no longer needs to
        call `model.to_dict()` or store a snapshot at all; this is recorded
        as a deliberate divergence from that placeholder signature, not an
        oversight.
        **New output field, and a mid-implementation design correction
        asked of and answered by the user:** `step_modification_record`
        (always present in `step_native`'s/`dyna_step_native`'s `sim_state`
        output, empty for the plain path) - a list of plain dicts describing
        this step's model-effect modification record. The first version of
        this wire format carried only bare `i64` asset ids for removed
        assets/their associations; the user caught that this can't be
        resolved back into a `maltoolbox.ModelAsset` on the Python side (see
        B5's entry below) and directed that the record be fully constructed
        in `malsim-core` and only *handed through* py-bindings, not
        resolved via a `Model` lookup on the Python side at all. This led to
        a `malsim-core` type change (not just a py-bindings one): `core/
        malsim-core/src/model_effects.rs`'s `AssetOp`/`AssocOp` now carry a
        new `AssetRef { id: i64, asset_type: String, name: String }` (a
        self-contained snapshot, captured at the exact moment each op is
        recorded - `AssetRef::from_model`) instead of a bare `i64`, for
        every asset reference in every variant (`AssetOp::Added`/`Removed`,
        `AssocOp::Added`/`Removed`'s `left`/`right`) - not just the removed
        case, since a later op in the *same* modification record can
        invalidate an id an earlier op already referenced (e.g.
        `remove_asset_op` records an about-to-be-removed asset's
        association removals before removing the asset itself - that
        asset's own id is unresolvable by the time a caller reads the full
        record back). `mod_effect_op_to_py`/`set_asset_ref` in `simulator.rs`
        write each `AssetRef`'s `id`/`asset_type`/`name` under
        `{prefix}_id`/`{prefix}_type`/`{prefix}_name` keys (`asset_*` for an
        `AssetOp`, `left_asset_*`/`right_asset_*` for an `AssocOp`, plus
        `field_name`); `kind` (`"asset"`/`"assoc"`) and `type`
        (`"ADDITIVE"`/`"SUBTRACTIVE"`, matching `ModelEffectType`'s own
        wire strings) discriminate the op. `execute_model_effects`'s
        `new_assets`/`removed_assets`/`new_associations`/
        `removed_associations` extraction (feeding `partially_regenerate_graph`)
        was updated to read `.id` off each `AssetRef` - same behavior, just
        reached through the new field. All existing `model_effects.rs`/
        `dyna_attacker_step.rs` Rust-native tests updated/still passing
        after this reshape (`{ .. }` patterns were unaffected; a few tests
        destructuring named `left`/`right`/`asset_id` fields were updated
        to `left.id`/`right.id`/`asset.id`).
        `python/malsim/_native-stubs/__init__.pyi` gained both new methods'
        signatures. `tests/test_native.py` gained
        `test_native_dyna_simulator_step_before_reset_raises`/
        `test_native_dyna_simulator_step_executes_model_effects_and_grows_graph`/
        `test_native_dyna_simulator_reset_restores_pristine_graph_after_mutation`
        (using the `wiperLang_scenario`-equivalent fixture loaded directly,
        mirroring A8's own test style) - the second asserts the
        `step_modification_record`'s self-contained `asset_type`/
        `asset_name`/`left_asset_*`/`right_asset_*` fields are present, not
        just bare ids; two pre-existing `step_native` delta-shape tests
        (`test_native_simulator_step_output_is_delta_only[_attacker_only]`)
        were updated to expect the new always-present
        `step_modification_record` key. Full gate green: `cargo test`/
        `clippy -D warnings`/`fmt --check` clean in both the root
        (`malsim-core`) and `py-bindings` workspaces; `uv run pytest tests -m
        "not integration"` (168 passed, including the 3 new B4 tests);
        `mypy`/`ruff check`/`ruff format --check` clean.
        **No new crate dependency** - reuses B1-B3's existing
        `maltoolbox-model`/B2's dyna step functions.
        **Build-cache gotcha reconfirmed, same as B3's §9/§10 entry - not
        new, but hit again this phase and worth re-flagging for B5+:** `uv
        run maturin develop --uv`/`uv pip install --reinstall-package` both
        intermittently (re)installed a `mal-toolbox` wheel missing
        `PyModel.__inner_capsule__` even with a correct `rev` pin. Workaround
        used throughout B4 (and needed again for every Python-level test run
        for the rest of this phase): build `mal-toolbox` from a fresh
        `git clone` + `maturin build --release` and `cp` the resulting
        `.so` directly into `.venv/lib/.../site-packages/maltoolbox/`
        (bypassing `uv pip install` for that one file), then use `uv run
        --no-sync <cmd>` (not plain `uv run`) for every subsequent Python
        command in the session - plain `uv run` re-triggers the same broken
        resync and silently overwrites the manually-fixed `.so`.
  - [x] B5 - Rewrite `DynaMalSimulator.reset()`/`.step()` to delegate to
        native (A9-equivalent). **Complete as of session 3 - see the final
        "Landed (session 3)" entry below for the closing fix; the two
        blocking-bug writeups in between are kept verbatim as the actual
        debugging history, not retrofitted to look clean.**
        Landed so far: `python/malsim/dyna_mal_simulator/simulator.py`'s
        module-level `dyna_reset`/`dyna_step` fully rewritten to delegate to
        B4's `dyna_reset_native`/`dyna_step_native`, mirroring A9's
        `mal_simulator/simulator.py::reset`/`step` almost exactly - reuses
        (imports, doesn't duplicate) `flatten_sim_settings`/
        `flatten_attacker_settings`/`flatten_defender_settings`,
        `create_attacker_state_from_native`/`create_defender_state_from_native`,
        `_graph_state_from_native`/`_ordered_new_nodes`/`MALSimulatorStaticData`/
        `_pre_step_check`/`alive_agents` from `mal_simulator.simulator`
        unchanged. `DynaMalSimulator.__init__`/`.reset()`/`.step()` now hold
        `self._native_sim`/`self._static_data` the same way `MalSimulator`
        does, and inherit `MalSimulator.__getstate__`'s `_native_sim`
        exclusion for free via subclassing (no override needed - confirmed
        no other attribute-name divergence). `model_snapshot` is no longer
        computed or stored in Python at all (native captures it - B4's
        design decision above); the local `MALSimulatorStaticData`
        `NamedTuple` `DynaMalSimulator` used to define itself was deleted,
        importing `mal_simulator.simulator`'s instead (structurally
        identical, no behavior change).
        `python/malsim/dyna_mal_simulator/simulator_state.py` gained
        `DetachedAsset` (a new, malsim-*own* lightweight frozen dataclass -
        `id`/`type`/`name` - deliberately not a `maltoolbox` type) and
        `modification_record_from_native`/`_resolve_asset_ref`, resolving
        B4's `step_modification_record` dicts into `AssetOp`/`AssocOp`.
        `AssetOp.asset`/`AssocOp.assoc`'s field types widened from
        `ModelAsset` to `ModelAsset | DetachedAsset`: resolves to the live
        `maltoolbox.ModelAsset` handle when `model.get_asset_by_id(id)`
        still finds it (keeps identity/equality with any other live
        reference a caller holds - required for an existing exact-equality
        test over an *added* asset to keep passing) and only falls back to
        `DetachedAsset` (built directly from the native dict's
        `asset_type`/`name` fields, never via a `Model` lookup) when it's
        not resolvable. **Why that fallback is needed, not just defensive:**
        maltoolbox's tombstone mechanism (`PyModelAsset`'s post-removal
        readability) is populated only by `PyModel.remove_asset` (the
        Python-facing method) - `malsim-core`'s native dyna step mutates the
        shared `Model` via the core Rust `Model::remove_asset` directly,
        bypassing that tombstone recording entirely, so a natively-removed
        asset's `ModelAsset` Python handle doesn't just become detached-but-
        readable the way the pure-Python path left it - it's simply gone
        (`model.get_asset_by_id` returns `None`, confirmed by reading
        maltoolbox's own `PyModelAsset::with_asset`/`PyModel::assets`
        source, not assumed). No current test exercises an exact-equality
        assertion against a *removed* asset (checked by grep across
        `test_dyna_mal_simulator.py` - every `modification_record` equality
        assertion involves only additions or association-removals between
        assets that stay alive), so this is forward-looking correctness, not
        yet pinned by a test - a future test would be a reasonable addition
        once B5 is unblocked.
        `python/malsim/dyna_mal_simulator/model_effects.py` (the pure-Python,
        still-used-by-its-own-direct-unit-test module) and
        `tests/test_dyna_mal_simulator.py`'s
        `test_apply_model_effect_modification_record_partially_regenerates_graph`
        both build `set[ModelAsset]`/`set[tuple[ModelAsset, str,
        ModelAsset]]` from a `list[AssetOp | AssocOp]` - mypy correctly
        flagged both once `AssetOp.asset`'s type widened, since neither site
        could statically prove it'd only ever see `ModelAsset`, never
        `DetachedAsset`. Fixed with `isinstance` narrowing (both sites only
        ever run against the pure-Python path's own output, which never
        constructs a `DetachedAsset` - confirmed by reading `model_effects.py`'s
        `target_op` closures, so this is type-narrowing only, not a behavior
        change). `uv run --no-sync mypy python/malsim tests`/`ruff check`/
        `ruff format --check` all clean with every change above in place.
        **Blocked here - `tests/test_dyna_mal_simulator.py` is NOT green,
        and that's B5's actual acceptance gate (§6: "not just a smoke
        subset").** `test_different_attackers` panics the whole Python
        process (`pyo3_runtime.PanicException: invalid SlotMap key used`,
        uncatchable from Python) on 12 of its 12 parametrized cases
        (`RandomAgent`/`TTCSoftMinAttacker`/`BreadthFirstAttacker`/
        `DepthFirstAttacker`, all configs) - it iterates every scenario
        under `tests/testdata/scenarios/dynamal_example_scenarios/`, and at
        least `intermediate/intDynamicTestLang13_scenario.yml` triggers it.
        **Root cause, traced by hand (manual step-by-step repro +
        `RUST_BACKTRACE=1`, not guessed):** `Directory:1:addTopToBottomLeft`'s
        model effect (`A> sub* / files`) is actually asset-*creating* (the
        RHS "files" resolves to a fresh `File` asset per member of
        `sub*`, not an existing one), so it creates two new `File` assets.
        The very next action in this run, `Directory:1:removeTopToBottomLeft`
        (`R> sub*.files / self`), removes one of those brand-new File
        assets' only association, leaving it fully disconnected (zero
        associations). Upstream mal-toolbox's
        `AttackGraph::partially_regenerate_graph` (called from
        `execute_model_effects`, B1) appears to remove-and-regenerate that
        disconnected asset's `reached` attack-step node under a *new*
        internal `AttackGraphNodeId` even though the asset was never in
        this call's `removed_assets` (only an association was removed) -
        the *old* id is a slotmap key, so once removed it never resolves
        again, even if a node is later re-added.
        **CORRECTION (session 2, see this entry's "landed" update below) -
        the paragraph above is WRONG about which asset and which
        mechanism, and this is the *second* time this exact failure class
        has been misdiagnosed this way. Read this before trusting any
        future hand-traced "regenerated under a new id" story for this
        bug class again.** The actual asset involved is `Directory:1`
        (id `0`) itself - the directory `removeTopToBottomLeft` is a step
        *on*, not some unrelated disconnected `File` - and it is not
        "regenerated under a new id" at all: `removed_assets` for that
        call genuinely contains `0`, mal-toolbox's `nodes_to_be_removed`
        removes `removeTopToBottomLeft`'s own node via `remove_node`, and
        *nothing recreates it* (confirmed with `eprintln!` instrumentation
        directly in `execute_model_effects`, printing `new_assets`/
        `removed_assets`/`new_nodes`/`graph.nodes.contains_key(action_id)`
        around the real `partially_regenerate_graph` call, not inferred
        from the panic backtrace alone - see "landed" below for the full
        trace and why this matters for picking a fix). There is no new id
        to "reconcile to" for this case, which is why candidate direction
        3 below doesn't actually apply to it. **Lesson for whoever reads
        this next: when `RUST_BACKTRACE=1` plus reading source is tempting
        you to write up a root cause, add a couple of `eprintln!`s at the
        actual mutation site first and rerun - a backtrace shows *where*
        it panicked, not *which* asset or *why* it became invalid, and
        both the original B5 writeup and this correction's author got it
        wrong the first time without that step.** **Confirmed via `git
        stash` that this is a regression, not pre-existing:** the identical
        scenario+policy combination runs to completion with zero crashes on
        the pure-Python `DynaMalSimulator` path (stashed every B4/B5 file,
        rebuilt, reran the same repro - "ALL OK" across all 28 dynamal
        scenario files). The panic itself fires inside
        `malsim-core::event_logger::collect_logs` (`graph.nodes[attack_step_id]`,
        a direct, un-guarded slotmap index) when it's asked to check
        detectors for a `step_compromised_nodes` entry that's since gone
        stale - **but `collect_logs` is not special, it's just the first
        one this particular repro happened to hit.** `get_attack_surface`
        (attack_surface.rs), `get_defense_surface` (defense_surface.rs),
        `attempt_attacker_step` (attacker_step.rs),
        `necessity::calculate_necessity` (necessity.rs), and
        `node_is_blocked`/`node_is_traversable` (graph_utils.rs) all do the
        same unguarded `graph.nodes[id]` indexing on ids pulled from
        accumulated/historical state (`performed_nodes`, `enabled_defenses`,
        `graph_state`'s maps) - every one of them was written and tested
        under Phase A's "the graph never mutates mid-episode" invariant
        (confirmed true for Phase A, since `MalSimulator` never calls any
        model-effect function), which Phase B's whole design breaks by
        composing these same functions against a graph `execute_model_effects`
        *can* mutate mid-episode. This is a systemic gap exposed by B4/B5's
        wiring, not a one-line bug in one function - **not fixed yet,
        deliberately: the user asked to document this and stop rather than
        have a fix picked unilaterally.** Three candidate directions were
        drafted, not decided, for whoever resumes:
        1. *Narrow/fast:* fix only `collect_logs`'s direct index (`.get()`
           with a skip-if-missing fallback - a vanished node correctly has
           "no detectors to report") to unblock this specific repro, and
           open a dedicated follow-up item (maybe B5.5 or folded into B6) to
           audit the other call sites named above before calling B5-B7 done.
        2. *Thorough:* audit and harden every Phase-A-era function
           malsim-core's dyna path composes (the five modules named above)
           for this exact risk before continuing B5 at all.
        3. *Fix at the source:* reconcile `performed_nodes`/
           `enabled_defenses`/`graph_state`'s id-keyed maps at the one point
           `fold_new_nodes_into_graph_state` already runs, dropping or
           re-resolving (by full name) any id `partially_regenerate_graph`
           invalidated - but this needs an actual simulation-semantics
           decision first (does a node regenerated mid-episode still count
           as "performed" under its new id, or does it effectively reset to
           unperformed?), which is a product question, not just a bugfix,
           and shouldn't be answered implicitly by whichever code path
           happens to be easiest to patch.
        **Repro, to re-derive (nothing session-local survives):**
        ```python
        from malsim.scenario.scenario import Scenario
        from malsim.config.sim_settings import MalSimulatorSettings, TTCMode
        from malsim.dyna_mal_simulator.simulator import DynaMalSimulator
        from malsim.policies.attackers.searchers import BreadthFirstAttacker
        from malsim.mal_simulator.run_simulation import run_simulation

        scenario = Scenario.load_from_file(
            'tests/testdata/scenarios/dynamal_example_scenarios/'
            'intermediate/intDynamicTestLang13_scenario.yml'
        )
        name = next(iter(scenario.attacker_settings))
        scenario.attacker_settings[name].policy = BreadthFirstAttacker
        scenario.attacker_settings[name].config = {'action_ordering': 'sorted'}
        sim = DynaMalSimulator.from_scenario(
            scenario,
            sim_settings=MalSimulatorSettings(
                ttc_mode=TTCMode.PRE_SAMPLE, compromise_entrypoints_at_start=False
            ),
        )
        run_simulation(sim)  # panics a few iterations in
        ```
        Crashes on `Directory:1:removeTopToBottomLeft` specifically when
        stepped manually one action at a time via
        `sim.agent_settings[name].agent.get_next_action(...)` - see this
        entry's root-cause paragraph for why.
        **Everything else already landed this phase stays green while this
        is blocked:** root `malsim-core` + `py-bindings` cargo gates (test/
        clippy/fmt) are unaffected and still clean; `tests/test_native.py`'s
        B4 tests still pass; `mypy`/`ruff` are clean across the whole tree
        (including the partially-landed B5 Python changes above). Only
        `test_dyna_mal_simulator.py` (and only its
        `test_different_attackers` parametrization, 12/40 cases) is red.

        **Landed (session 2): the stale-id panic class above is fixed -
        `test_different_attackers` went from 0/12 to 10/12 passing. Still
        blocked, on a different, newly-exposed bug - see below. Resume
        there, do not restart the panic investigation.** The user picked
        candidate direction 1 (narrow fix + defer the rest) after an
        `AskUserQuestion`, but the real root cause (see the CORRECTION
        above) made the framing of that choice wrong in a way worth
        recording: there is no "was it really performed" semantics
        question for this bug class at all, for either sub-case actually
        found -
        - A node genuinely compromised *this step* whose own backing asset
          is removed by that same step's model effect (the `Directory:1`
          case above): the compromise already happened: the only question
          is what to do when something reads the now-dead id a few lines
          later in the same step. Answer: there's nothing left to read -
          skip it, don't error, don't guess.
        - A node from *historical* state (`performed_nodes`/
          `enabled_defenses`/accumulated `graph_state` maps, or a
          per-agent `ttc_dists` override resolved once at reset) going
          stale because of a *later*, unrelated step's model effect: same
          answer, for the same reason - the historical fact doesn't need
          the live node to still exist.

          Candidate direction 3 (reconcile by full name) was never
          implemented and - per the corrected root cause - doesn't apply
          to the specific case that was crashing (nothing is regenerated
          under a new id there to reconcile to). It may still be the right
          call for a *geniune* regenerate-under-new-id case if one is ever
          found (full_name_to_node is confirmed to be mal-toolbox's own
          stable cross-regeneration identity, so the mechanism would work)
          - no such case has actually been observed yet, confirmed or
          fixed.

        Audit scope ended up wider than the original "5 modules" list (see
        the CORRECTION above for why that list undercounted even the
        already-landed call sites, let alone new ones) - `grep -n
        "graph\.nodes\[" core/malsim-core/src/*.rs
        py-bindings/malsim-pyo3/src/simulator.rs` plus manually classifying
        every production (non-test) hit as either "id sourced from the
        live graph this same call" (safe by construction - mal-toolbox
        keeps `graph.defense_steps`/`graph.attack_steps`/parent/child sets
        in sync on `remove_node`, confirmed by reading it) or "id sourced
        from accumulated/historical/cross-step state" (needs a guard) is
        what actually found every real site - don't re-derive this from
        the original 5-module list, it was incomplete in both directions
        (named two functions - `get_attack_surface`/
        `get_effects_of_attack_step` - that turned out to already be
        correctly guarded since Phase A5 itself, via a pre-existing
        `node_is_live` check; missed `malsim-pyo3::simulator.rs` entirely,
        where the actual second-domino crash was).

        **Fixed, with the reasoning for each:**
        - `core/malsim-core/src/event_logger.rs::collect_logs` - `graph.
          nodes[attack_step_id]` (the original crash site) changed to
          `graph.nodes.get(attack_step_id)`, skip on `None`. New test
          `collect_logs_skips_a_node_removed_from_the_graph_since_compromise`
          (removes a node with a `tprate=1.0` detector that would always
          have fired, asserts no panic and no log).
        - `core/malsim-core/src/dyna_attacker_step.rs` - the `effects`
          loop (processing `attacker_step_effects`' return value) gained a
          `node_is_live` check before each `execute_model_effects` call:
          an *earlier* effect-chain sibling's model effect can invalidate
          a *later* sibling in the same list before its own turn - same
          shape as the primary bug, one level deeper. **No isolated
          Rust-native test added for this one** - constructing a MAL
          fixture where one `+>`-chained effect sibling's model effect
          removes another's backing asset (wiperLang's `exfiltrate`/
          `propagate` pair, the only existing multi-effect-sibling
          fixture, are both purely additive) would need inventing new
          test-only language surface disproportionate to this one guard;
          covered transitively by `cargo test`/`clippy`/`fmt` plus the
          real `test_different_attackers` integration gate. Flag as a gap
          if this specific sub-case is ever suspected of regressing.
        - `py-bindings/malsim-pyo3/src/simulator.rs` - this is where the
          *second* crash was, one call after the first fix (`stable_ids`,
          not in the original 5-module list at all):
          - `stable_ids`/`id_value_map` (the shared id-translation helpers
            used by nearly every output field - `performed_nodes`,
            `action_surface`, `enabled_defenses`, `ttc_values`,
            `necessity_per_node`, ...): hardened once, centrally, rather
            than auditing every call site individually (20+ call sites,
            too easy to miss one) - both now `filter_map` over `graph.
            nodes.get(id)`, silently dropping a dead id from the output
            set/map instead of indexing.
          - `attacker_ttc_overrides` - `ttc_dist_overrides` is resolved
            once at reset and never updated; a dyna-path node it names can
            later be removed. Same `.get()`-skip.
          - `log_entry_to_py` - new `stable_id_of` helper for
            `detector_node_id`/`trigger` (expected to always be live,
            since `collect_logs` only ever builds a `LogEntry` from an
            already-live node and nothing mutates the graph before this
            runs - guarded anyway, as a `PyResult::Err` not a panic, so a
            future change violating that invariant fails loudly and
            catchably instead of crashing the process); `context`'s
            per-label map (picked from `previous_compromised_nodes` -
            genuinely can go stale from an unrelated earlier step) uses
            `filter_map`, dropping just that label.
        - Full gate after all of the above: `cargo test`/`cargo clippy -D
          warnings`/`cargo fmt --check` clean in both the root
          (`malsim-core`) and `py-bindings` workspaces; `uv run --no-sync
          pytest tests -m "not integration"` is 166 passed / 2 failed (the
          new blocker below) / 1 deselected, down from the prior clean
          161; `mypy`/`ruff check`/`ruff format --check` all clean (no
          Python files touched this session - Rust-only changes).

        **Still blocked, on a different bug - NOT a stale-id panic, a
        missing computation, and confirmed to be a genuine Rust-port
        regression (not pre-existing) via the same `git`-stash-the-dyna-
        simulator-and-compare technique used above.** The remaining 2/12
        `test_different_attackers` failures are both `TTCSoftMinAttacker`
        (`config2`/`config3`), both on
        `AssertionError: Node <X> does not have a ttc value` out of
        `python/malsim/mal_simulator/state_query.py::node_ttc_value`
        (pure Python, pre-existing, unchanged this session) - not a panic,
        a normal catchable Python exception. The node in question
        (`WiperMalware-4:execute` on `posterLang_scenario.yml`,
        `File-4:reached`/`File-3:reached`/etc. on
        `intDynamicTestLang13_scenario.yml`) is in every case a node
        *created mid-episode by a dyna model effect* that's present in the
        attacker's `action_surface` but absent from `sim_state.graph_state.
        ttc_values` entirely. `TTCSoftMinAttacker` is the only built-in
        policy that calls `node_ttc_value` for every node in its action
        surface (to pick the lowest-TTC one) - `BreadthFirstAttacker`/
        `DepthFirstAttacker`/`RandomAgent` never read a TTC value at all,
        which is exactly why the stale-id panic above fully masked this
        second bug until now: none of the three policies the original
        12-case failure was dominated by would ever have reached this
        code path. **Confirmed as a genuine regression, not pre-existing:**
        same `cp`-in-the-pre-B4-`dyna_mal_simulator/simulator.py`-and-rerun
        technique as the CORRECTION above, run against `TTCSoftMinAttacker`
        across every `dynamal_example_scenarios` file (not just the two
        failing ones) - zero missing-ttc-value cases on the pure-Python
        path, vs. the two found immediately on the native path. Not
        root-caused beyond this - `ttc_value_for_dist` returns `Some(...)`
        unconditionally for `TtcMode::PreSample`/`ExpectedValue` (checked
        directly in `graph_state.rs`), so the gap isn't "the TTC dist
        legitimately has no value" - it's that these nodes are somehow
        never reaching `add_new_nodes_to_graph_state`'s per-node TTC
        computation at all (wrong `new_nodes` set reaching
        `fold_new_nodes_into_graph_state`? a second, redundant
        `GraphState` reconstruction elsewhere dropping the merge? -
        genuinely not narrowed down yet). **Repro:**
        ```python
        from pathlib import Path
        from malsim.scenario.scenario import Scenario
        from malsim.config.sim_settings import MalSimulatorSettings, TTCMode
        from malsim.dyna_mal_simulator.simulator import DynaMalSimulator
        from malsim.policies.attackers.ttc_soft_min import TTCSoftMinAttacker

        scenario = Scenario.load_from_file(
            'tests/testdata/scenarios/dynamal_example_scenarios/'
            'intermediate/intDynamicTestLang13_scenario.yml'
        )
        name = next(iter(scenario.attacker_settings))
        scenario.attacker_settings[name].policy = TTCSoftMinAttacker
        scenario.attacker_settings[name].config = {'beta': 1.0}
        sim = DynaMalSimulator.from_scenario(
            scenario,
            sim_settings=MalSimulatorSettings(
                ttc_mode=TTCMode.PRE_SAMPLE, compromise_entrypoints_at_start=False
            ),
        )
        from malsim.mal_simulator.run_simulation import run_simulation

        run_simulation(sim)  # AssertionError a few iterations in, node name varies by scenario
        ```
        Also reproduces on `posterLang_scenario.yml` (different scenario,
        same shape - `WiperMalware-4:execute`), so this isn't specific to
        one language file. **Next step for whoever resumes: instrument
        `dyna_graph_state.rs::add_new_nodes_to_graph_state` and its caller
        `fold_new_nodes_into_graph_state` the same way the CORRECTION above
        instrumented `execute_model_effects` - print `new_nodes` and the
        resulting `ttc_values` map's keys at each call - before guessing at
        a fix, for the same reason the CORRECTION above exists.**

        **Landed (session 3): B5 is complete - `test_different_attackers`
        is 12/12 passing, full gate green. Root cause found by reading
        source, not by instrumenting the suggested two functions above -
        they were a red herring; `add_new_nodes_to_graph_state`/
        `fold_new_nodes_into_graph_state` were already correct (confirmed:
        `new_nodes`/`ttc_values` are populated correctly on the Rust side
        the moment a model effect creates a node).** The actual gap was one
        layer further out, in `simulator.rs::build_step_output`'s own doc
        comment: *"Episode-static fields (`ttc_values`, `necessity_per_node`,
        `impossible_attack_steps`, `pre_enabled_defenses`) are dropped
        entirely - Python caches them from `reset_native`'s output."* True
        for Phase A (the graph never mutates after reset) - false for Phase
        B, which grows these same `GraphState` maps mid-episode via
        `fold_new_nodes_into_graph_state`, but `build_step_output` (shared
        by `step_native`/`dyna_step_native`) never sent that growth back
        across the FFI boundary, and
        `dyna_mal_simulator/simulator_state.py::update_simulator_state`
        passed `sim_state.graph_state` through unchanged on every step. So
        native's internal `state.graph_state.ttc_values` had the new node's
        value immediately; Python's mirrored copy never learned about it
        until `TTCSoftMinAttacker` (the only built-in policy that calls
        `node_ttc_value` for every node in its action surface) asserted on
        the gap - exactly why `BreadthFirstAttacker`/`DepthFirstAttacker`/
        `RandomAgent` never tripped it. Confirmed node-for-node against a
        reduced repro (`posterLang_scenario.yml`'s `WiperMalware-4:execute`,
        no seed needed - deterministic under `TtcMode::PreSample`) before
        writing any fix.

        **Design decision asked of and answered by the user (three
        candidates, not picked unilaterally - per standing policy for
        decisions like this):** how should the growing `GraphState` reach
        Python on every dyna step, given `necessity_per_node` is a *whole-
        graph* recompute on every fold (not just new keys), so a naive
        "diff the new keys" delta can't provably stay correct if an
        existing node's necessity value itself changes. Chose **"full
        resend, gated on actual mutation"**: `build_step_output` resends
        the full current `ttc_values`/`impossible_attack_steps`/
        `necessity_per_node`/`pre_enabled_defenses` maps only on a step
        where `step_modification_record` is non-empty (a model effect
        genuinely ran) - correct by construction (no incremental-merge
        risk), and the resend cost only hits steps that actually grow the
        graph, not every step for the sim's whole lifetime. Rejected: full
        resend unconditionally every dyna step (simpler, but pays the cost
        on quiescent steps too); true per-field delta (most wire-efficient,
        but mixes delta/full-resend semantics and risks the necessity-value
        staleness above).
        **Landed:**
        - `py-bindings/malsim-pyo3/src/simulator.rs` - extracted
          `insert_graph_state_fields` (free function, not a method) from
          `build_reset_output`'s inline field-setting code, since it's now
          shared with `build_step_output`; `build_step_output` calls it
          right after `step_modification_record` is set, gated on
          `!step_modification_record.is_empty()`. `build_reset_output`
          itself is behavior-unchanged (same fields, now via the extracted
          helper). No new crate dependency - pure reshuffling of existing
          field-writing code plus one new `if`.
        - `python/malsim/dyna_mal_simulator/simulator_state.py::
          update_simulator_state` gained an optional `graph_state:
          GraphState | None = None` parameter (default `None` = old
          behavior, carry `sim_state.graph_state` over unchanged) -
          replaces it wholesale when given, rather than merging field-by-
          field, consistent with the "full resend" decision above.
        - `python/malsim/dyna_mal_simulator/simulator.py::dyna_step` -
          after `dyna_step_native` returns, checks for `'ttc_values' in
          native_out['sim_state']` (the signal the Rust side's gate fired
          this step) and if so calls the already-imported
          `_graph_state_from_native` (previously only used by `dyna_reset`)
          to rebuild a fresh `GraphState`, passed into
          `update_simulator_state`'s new parameter. `step_native`/
          `mal_simulator/simulator.py::step` are untouched - the key is
          simply absent for a plain step, by construction.
        - `tests/test_native.py` gained
          `test_native_dyna_simulator_step_resends_graph_state_only_on_model_effect`
          (using the same `wiper_scenario.yml` fixture as B4's dyna tests):
          asserts all four keys are present and `Wiper-7:activate` (a node
          that didn't exist at reset) has a `ttc_values` entry on the step
          that creates it, *and* that all four keys are absent on a
          following step with no actions at all (no model effect ran) -
          locking in the gate, not just the happy path.
        - Full gate green: `cargo test`/`clippy -D warnings`/`fmt --check`
          clean in both the root (`malsim-core`, 150 tests, unchanged by
          this fix - pure `py-bindings` change) and `py-bindings`
          workspaces; `uv run --no-sync pytest tests` (170 passed, up from
          168 pre-fix - the 1 new test plus `test_different_attackers`
          gaining its remaining 2 passing parametrizations) and
          `examples/*` (6 passed); `mypy`/`ruff check`/`ruff format --check`
          clean on `python`/`tests`/`examples` (ruff flags a pre-existing,
          unrelated `PORTING_NOTES.md` markdown-fence formatting diff at
          this entry's own repro block - confirmed via `git stash` to
          predate this session, not touched).
  - [x] B6 - Audit every other `DynaMalSimulator` public method
        (A10-equivalent). **Result: zero production-code changes; one
        regression test added.** Did the diff first, as §6 prescribes:
        `git diff f949c21 HEAD -- python/malsim/dyna_mal_simulator/` (only
        `simulator.py`, `simulator_state.py`, `model_effects.py` changed,
        all by B4/B5) and `git diff f949c21 HEAD -- python/malsim/
        mal_simulator/` (nothing relevant). `DynaMalSimulator` still
        defines nothing public beyond `__init__`/`from_scenario`/`reset`/
        `step`.
        **A10's premise changed and was re-verified:** A10 recorded that
        `DynaMalSimulator` never sets `self._native_sim`; since B5 it does
        (`__init__` builds `_native.Simulator(attack_graph)`). Re-checked
        that no inherited method reaches for `_native_sim` outside
        `reset()`/`step()` (both overridden), and that `__getstate__`
        (which excludes `_native_sim`) applies to the dyna class too - a
        pickled `DynaMalSimulator` round-trips and, like `MalSimulator`,
        simply lacks `_native_sim` afterwards (pre-existing limitation,
        §10).
        **What actually needed auditing** was not new methods but whether
        the 19 inherited query methods stay correct when the graph and
        the native-computed `GraphState` they read change *mid-episode*.
        Exercised each against a node that only exists after a model
        effect (`Wiper-7:activate`, created by compromising
        `InfectedDevice:infect` in `wiper_scenario.yml`), under
        `PRE_SAMPLE` and `EXPECTED_VALUE`: `node_ttc_value` (with and
        without agent), `node_is_necessary`, `node_is_blocked` (node and
        full-name forms), `node_is_traversable`, `node_is_compromised`,
        `node_is_enabled_defense`, `node_is_actionable`, `node_reward`,
        `compromised_nodes`, `get_node` (by name and id), plus
        `get_node` raising `LookupError` for the added node again after
        `reset()`. All correct - this is B5's resend-on-model-effect fix
        doing its job. Locked in by
        `tests/test_dyna_mal_simulator.py::
        test_inherited_query_methods_follow_graph_mutated_by_model_effects`.
        **Decisions (no architectural/stylistic choice arose):**
        - `node_ttc_value` raises `KeyError` under the default
          `TTCMode.DISABLED`, for pre-existing nodes as well as new ones
          (ttc_values is simply empty) - identical to `MalSimulator`, so
          left alone rather than made dyna-specific.
        - `reset(seed=...)` re-seeds `self.rng` *after* resetting, so the
          seed only affects the next episode - identical to
          `MalSimulator.reset`, preserved.
        - A node reference held across `reset()` goes stale (the graph is
          restored to the snapshot, so ids of removed/added nodes no
          longer resolve); expected, not guarded.
        Gate: pytest `-m "not integration"` 170 passed, mypy/ruff clean.
  - [x] B7 - Full parity pass + cleanup (A11-equivalent). Done jointly
        with A11.
        **How dead code was identified.** A read-only call-graph pass over
        `python/malsim`, `examples/` and `tests/` classified every top-level
        function/class in the hot-path modules as LIVE (reachable from a
        live root: either simulator's public API, `native_settings.py`,
        `rewards.py`, `run_simulation.py`, `config/`, `scenario/`, `envs/`,
        `policies/`, `visualization/`, `examples/`), TEST-ONLY, or DEAD.
        Actual call sites were checked, not just imports; key results were
        spot-verified by grep.
        **Deleted files (11):** `mal_simulator/{attack_surface,
        defense_surface,graph_processing,reset_agent,
        simulator_static_data}.py` and `dyna_mal_simulator/{attacker_step,
        defender_step,graph_state,model_effects,model_state,
        process_assoc_traversal}.py`.
        **Shrunk in place (per §11, import paths unchanged):**
        - `event_logger.py`: only `LogEntry`.
        - `graph_state.py`: only `GraphState`; `compute_initial_graph_state`
          deleted.
        - `attacker_step.py`: only `attacker_is_terminated`.
        - `defender_step.py`: only `defender_is_terminated`.
        - `false_alerts.py`: only the two rate getters.
        - `observability.py`: only `node_is_observable`.
        - `ttc_utils.py`: kept `TTCDist` and its helpers (still the
          user-facing config type, §4); deleted the module-level
          graph functions `attack_step_ttc_value(s)`,
          `get_pre_enabled_defenses`, `(get_)is_impossible_attack_step(s)`.
        - `graph_utils.py`: lost `node_is_live`. The A10 query predicates
          stay, as A10 required.
        - `node_getters.py`: lost `full_name_dict_to_node_dict`.
        - `attacker_state_factories.py`/`defender_state_factories.py`: only
          the `*_from_native` factories (+ `get_entry_points`,
          `_log_entry_from_native`).
        Not touched: `MalSimulator._defender_is_terminated` (unused private
        method on the public class, out of this cleanup's scope).
        **Viability/pruning ported to Rust rather than dropped** (user
        decision, §11). `graph_processing.py`'s viability half was the one
        piece of deleted Python with tests and no Rust port, since A3 had
        skipped it as dead. It's now in `core/malsim-core/src/viability.rs`
        (`evaluate_viability`, `propagate_viability_from_node`,
        `calculate_viability`, `make_node_unviable`,
        `prune_unviable_and_unnecessary_nodes`, `ViabilityError`), with all
        four Python tests ported. The coreLang-fixture prune test is ported
        twice: once on a hand-built chain, once on the generated wiperLang
        graph. Six extra tests cover the error paths and
        `make_node_unviable`. See §10 for small deviations.
        **Python tests: deleted only where Rust covers them** (checked
        assertion-by-assertion, not by name):
        - Deleted, already covered: the 7 `test_graph_processing.py` tests
          other than `test_node_is_blocked` (necessity.rs/viability.rs);
          `test_mal_simulator.py::test_attacker_step`/`test_defender_step`
          (attacker_step.rs/defender_step.rs);
          `test_dyna_mal_simulator.py::test_assoc_traversal`/
          `test_apply_model_effect` (assoc_traversal.rs/model_effects.rs).
        - Gaps found and closed with new Rust tests before deleting:
          `attack_surface_shrinks_as_defenses_are_enabled` (Rust tests
          never passed enabled defenses or impossible steps into
          `get_attack_surface`), `step_skips_defense_node_outside_action_
          surface_and_not_entry_point`, `defender_step_skips_node_outside_
          non_empty_action_surface`,
          `node_is_traversable_and_false_when_blocked_by_enabled_defense`
          (the old `node_is_traversable_false_when_blocked` never asserted
          `false`; renamed to `node_is_traversable_or_true_when_only_some_
          parents_block`). `model_effects.rs`'s `assert_graph_equivalent`
          now compares edges by full name, not bare step name (which let
          same-named steps on different assets mask wrong edges), and
          checks node count. A new `assert_no_dangling_associations` port
          runs in the removal test.
        - Rewritten, not deleted, because they also cover live simulator
          behavior: `test_attacker.py::test_attack_surface_traininglang`
          now asserts on `sim.agent_states['Attacker1'].action_surface`
          instead of calling `get_attack_surface` (same scenario, same
          three assertions);
          `test_apply_model_effect_modification_record_partially_
          regenerates_graph` now performs the randomly picked `remove` step
          through `DynaMalSimulator.step` (as an entry point, TTCs disabled)
          and checks the same no-dangling-associations and
          fresh-graph-equivalence properties against the native path, plus
          that the recorded subtractive ops actually took effect. Run 40
          times with fresh random picks: 40/40 passed.
        **Integration check:** `envs/`/`policies/`/`visualization/`
        unmodified; their tests and `examples/*` (6) pass.
        **Memory-leak scrutiny (§6's B7 ask).** Both
        `test_no_memory_leak_on_teardown` and
        `test_no_memory_growth_over_repeated_simulations` pass, but they
        only see Python objects (`weakref`, `gc.get_objects()`), not the
        native heap behind the B3 `Model` capsule. So I also measured RSS
        and ran valgrind:
        - RSS over 300 repeated runs on `wiper_scenario.yml`
          (`malloc_trim` before each reading): scenario load, a native
          `Simulator` alone, repeated graph/model capsule extraction, and
          plain `MalSimulator` runs all plateau after warm-up. Dyna runs
          that apply model effects grow ~110-125 KB/run. `tracemalloc`
          shows the Python heap flat over that span, so the growth is
          native.
        - **Pre-existing, not a port regression:** the pre-B4 pure-Python
          dyna path (commit `0bf303f`'s `python/` tree) grows *faster*
          (~220 KB/run) on the same script.
        - **Not malsim's capsule handoff:** valgrind memcheck
          (`PYTHONMALLOC=malloc`, 2 vs 8 runs) shows "definitely lost"
          only for one-time import/type-object allocations, and *zero*
          still-reachable bytes with a `malsim_pyo3`/`malsim_core` frame.
          The growth that remains at exit is all in allocations from
          `maltoolbox/_native.so`.
        - Logged as a §9 open risk with a repro rather than fixed: it's in
          upstream mal-toolbox, not this repo.
        Gate: `uv run --no-sync pytest tests` (160 passed, incl.
        `integration`), `pytest examples/*` (6), mypy (no issues), ruff
        check/format clean; `cargo test` (164) / `clippy -D warnings` /
        `fmt --check` clean in both the root and `py-bindings` workspaces.
- [ ] Phase C - Rust-only library API (§7)
  - [x] C1 - Port `NodePropertyRule`'s dict-shape + `.value()`/`.per_node()`
        matching as an independent Rust utility (not shared with the
        Python-path flattening in §2.4 - see §2.6). Landed in
        `core/malsim-core/src/scenario/node_property_rule.rs`:
        `NodePropertyRule<T: RuleValue>` (`from_value`/
        `from_optional_value`/`to_value`/`len`/`value`/`per_node`),
        `StepValues<T>` (list vs mapping form), and `RuleValue` impls for
        `bool`, `f64` and `TtcDist`. 11 Rust tests: precedence
        (by-name > by-type > default), falsy fall-through, list form,
        `per_node` truthiness, the `test_scenario.py` faulty-dict and
        old-rewards-format cases, TTC-name parsing, round-trip, and nodes
        without a model asset. Decisions in §12 (C1 entries).
  - [x] C2 - Port scenario YAML loading: field validation, `extends`
        merge (`recursive_update`), path resolution. Landed in
        `core/malsim-core/src/scenario/loading.rs`: `load_scenario_dict`
        (YAML to `serde_json::Map`, recursive `extends`, `lang_file`/
        `model_file` made relative to the scenario file, `git@` lang
        URLs left alone), `recursive_update`, `path_relative_to_file_dir`,
        `validate_scenario_dict` (+ the `DEPRECATED_FIELDS`/
        `REQUIRED_FIELDS`/`ALLOWED_FIELDS` tables) and
        `ScenarioFileError`. New direct dependency: `serde_yaml 0.9.34`
        (the version already in both lockfiles via mal-toolbox). 11 Rust
        tests: merge semantics (nested override, explicit-`null` removal,
        mapping vs scalar replacement), validation cases, path resolution,
        the three `test_scenario.py` `extends` fixtures at dict level, and
        a sweep over every fixture asserting the same validate outcome
        Python gives (three fixtures still use the deprecated top-level
        `rewards` and are rejected on both sides).
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

**After A11/B7:** the "Ported" rows below were then deleted on the Python
side (or shrunk to their still-live items); see §0's B7 entry for the
exact list. The table is kept as the record of what moved where.

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

## 6. Phase B: `DynaMalSimulator` port - detailed steps

`DynaMalSimulator` (`dyna_mal_simulator/`) subclasses `MalSimulator` and
layers on top of the same `AttackerState`/`DefenderState`/reward
machinery (confirmed: it imports and reuses
`mal_simulator.attacker_state_factories.create_attacker_state`,
`rewards.py`'s reward-fn builders, A10's already-audited public query
methods, etc. unmodified) but replaces the step functions with its own
`attacker_step.py`/`defender_step.py`, which can mutate the underlying
`Model`/`AttackGraph` at runtime via `model_effects.py` (346 lines) and
`process_assoc_traversal.py` (318 lines), with `model_state.py` (90
lines) handling snapshot-based reset.

**The hard part is already done upstream, confirmed by reading the
actual Rust source, not just mal-toolbox's `PORTING_NOTES.md` prose:**
live graph mutation with correct id/reference bookkeeping
(`maltoolbox-model/src/model.rs`'s `Model::add_asset`/`remove_asset` →
`AssetSnapshot`/`add_associated_assets`/`remove_associated_assets`,
`maltoolbox-attackgraph/src/graph.rs`'s
`AttackGraph::partially_regenerate_graph`) **and** the model-effect
*declaration* types malsim's Python parses at attack-graph-build time
(`maltoolbox-language/src/graph/model_effect.rs`: `LanguageGraphModelEffect`,
`ModelEffectType`, `DynTarget`, `AssocTraversal`/`GlobAssocTraversal`/
`AssocSet`/`AssocTraversalChain`, `SetOperation`, `QuantityFilter` - a
direct Rust port of `language_graph_model_effect.py`'s dataclasses)
already exist in mal-toolbox's `rust-rewrite` branch. Phase B's own job is
narrower than Phase A's was per module: *evaluate* those already-parsed
declarations against a live `Model` (RNG-driven association-chain
traversal and quantity sampling) and *apply* the result via the
already-mutation-safe primitives above - no new graph-mutation
primitives, no new model-effect grammar, to invent. This is why - unlike
Phase A, which needed 7 logic steps (A1-A7) before any integration work -
the pure-Rust portion below compresses to two steps.

**New architectural wrinkle Phase A didn't need to solve: there is no
shared `Model` handle yet.** §2.2/A1 proved a shared `Rc<RefCell<
AttackGraph>>` via `PyAttackGraph::__inner_capsule__`, but
`AttackGraph` deliberately holds no `Model` reference at all (mal-toolbox's
own `PORTING_NOTES.md` §2, "`AttackGraph` ↔ `Model`" row) - every mutating
call takes `&mut Model` explicitly. `PyModel`
(`maltoolbox-model-py/src/model.rs`) already holds `pub inner:
Rc<RefCell<Model>>`, and `PyAttackGraph` already holds a `model_py:
Option<Py<PyModel>>` linking the two on the Python side - but `PyModel`
has **no `__inner_capsule__`-equivalent today**. Phase B needs one,
almost certainly via an upstream mal-toolbox change mirroring A1/§2.2's
precedent (`c854d1d6`) - B3 below is this phase's A1-equivalent proving
step, and is the one piece of this phase that reaches outside malsim's
own repo.

Each step lands as its own PR/commit with the full Python test suite
green and a Rust-native test ported from the equivalent
`tests/test_dyna_mal_simulator.py` case(s) before moving on - same
discipline as §5, and this phase is unusually well-served by it:
`test_assoc_traversal`, `test_apply_model_effect`, and
`test_apply_model_effect_modification_record_partially_regenerates_graph`
(lines 217-414) are already close to isolated unit tests of exactly the
functions B1 ports, unlike most of Phase A's A5-A7, which had no isolated
Python test to port 1:1 at all - check this repo's own
`test_dyna_mal_simulator.py` for the actual fixtures before writing new
ones from scratch.

### Pure Rust logic (`malsim-core`) - 2 steps

**B1. Port association-traversal evaluation + model-effect application.**
`process_assoc_traversal.py` (`sample_size`, `_apply_quantity_filter`,
`_assoc_traversal`, `_glob_assoc_traversal`, `_assoc_set_traversal`,
`traverse_association_chain`, `parse_addition`, `parse_removal`) and
`model_effects.py` (`target_op`'s four closures - `add_asset`/
`remove_asset`/`add_assoc`/`remove_assoc` - plus `_apply_model_effect`/
`execute_model_effects`). Operates directly on mal-toolbox's already-Rust
`AssocTraversal`/`GlobAssocTraversal`/`AssocSet`/`AssocTraversalChain`/
`DynTarget`/`ModelEffectType`/`LanguageGraphModelEffect` types (no new
grammar to invent, per above) and mutates via `Model::add_asset`/
`remove_asset`/`add_associated_assets`/`remove_associated_assets` +
`AttackGraph::partially_regenerate_graph`. RNG-per-quantity-sample - reuse
A2's `rand` plumbing, no new RNG crate. **New direct crate dependency to
flag per standing policy before adding:** `maltoolbox-model` - currently
only a *transitive* dependency of `malsim-core` via `maltoolbox-attackgraph`
(same "promote transitive to direct" shape as A8's `rand` add in
`malsim-pyo3`, not a net-new dependency tree). Unit-test against
`test_dyna_mal_simulator.py::test_assoc_traversal`/`test_apply_model_effect`/
`test_apply_model_effect_modification_record_partially_regenerates_graph`
directly (see above) plus hand-built fixtures for the quantity-filter/
glob/set-operation edge cases those three don't cover.

**B2. Port model-snapshot reconciliation + dyna step orchestration.**
`model_state.py` (`reconcile_model_to_snapshot`, `reset_model_effects`),
dyna `graph_state.py` (`add_new_nodes_to_graph_state` - a thin composition
of A3's `attack_step_ttc_values`/`get_pre_enabled_defenses`/
`get_impossible_attack_steps` and A3's `necessity.rs::calculate_necessity`
over just the newly-added nodes), dyna `attacker_step.py`
(`dyna_attacker_step`/`dyna_attempt_attacker_step` - wraps A7's
`attacker_step`/`attempt_attacker_step` with B1's `execute_model_effects`
called on each successful compromise and each resulting effect node),
dyna `defender_step.py` (`dyna_defender_step` - same wrapping shape,
simpler), and `simulator_state.py`'s `DynaMalSimulatorState`/`AssetOp`/
`AssocOp` (extends A-phase's `MalSimulatorState` with a
`modification_record: Vec<AssetOp | AssocOp>` field). **No new crate
dependency** - this step is almost entirely composition of B1 and
Phase A's existing `attacker_step`/`defender_step`/`graph_state`/
`graph_utils`/`necessity` modules, which is also why it's the second and
last pure-Rust-logic step rather than several - there is very little
*new* algorithmic content here versus wiring. Unit-test
`dyna_attacker_step`/`dyna_defender_step` against hand-built fixtures that
exercise a model-effect-bearing node (no single isolated Python test
covers the full wrapped step - `test_attacker_step`,
`test_remove_before_add`, and the `test_int_dynamic_test_lang*`/
`test_easy_ransomware_lang*`/`test_rand_multiplicity_scenario` cases all
go through a fully-built scenario + running simulator, same situation
A5-A7 were in for their own step - port the narrowest assertions from
those (e.g. `test_remove_before_add`'s specific ordering guarantee) as
direct unit tests instead of a line-for-line scenario port).

### Python bindings integration - 5 steps (kept fine-grained: this is where Phase A's risk actually lived)

**B3. Prove a shared `Model` handle end to end (A1-equivalent).**
Add `PyModel::__inner_capsule__` upstream in mal-toolbox (mirroring
`PyAttackGraph::__inner_capsule__` at `c854d1d6`), then one
`#[pyfunction]` in `malsim-pyo3` that extracts the capsule and reads
something trivial through it (asset count) against a Python-built
`maltoolbox.Model`. Python-side smoke test confirms the count matches
`len(model.assets)` and, critically, that a mutation made through either
side (Python `model.add_asset(...)` vs. the Rust handle) is visible on
the other - this double-visibility check is the one thing A1's original
smoke test didn't need to prove (Phase A never mutated the shared graph
from both sides at once; Phase B's whole point is that it does). Can
proceed in parallel with B1/B2 - it has no dependency on either.

**B4. Native dyna reset/step entry points in `malsim-pyo3`
(A8-equivalent).** Extend (or add a sibling pyclass/method to) the
existing native `Simulator` - naming TBD at implementation time, e.g.
`dyna_reset_native(settings, agents, model_snapshot, seed)`/
`dyna_step_native(actions)` - threading B3's shared `Model` handle
alongside A1's shared `AttackGraph` handle, composing B1/B2's functions
into a full dyna reset/step. New output shape beyond A8/A9's: per-step
`modification_record` (new/removed asset+assoc ops, as plain dicts - an
asset op needs at least `{type, asset_id}`, an assoc op needs
`{type, left_asset_id, field_name, right_asset_id}`) and the set of
newly-created node ids from `partially_regenerate_graph`, both needed so
Python can resolve them back to real objects the same way A9 resolves
other native output. First point B1-B3 are exercised together through
Python - same role A8 played for A1-A7.

**B5. Rewrite `DynaMalSimulator.reset()`/`.step()` to delegate to native
(A9-equivalent).** Rewrite `dyna_mal_simulator/simulator.py`'s
module-level `dyna_reset`/`dyna_step` to call B4's entry points instead of
`model_state.reset_model_effects`/`dyna_attacker_step`/`dyna_defender_step`
directly. Extend `native_settings.py`'s flattening (or add a dyna-specific
sibling) for any dyna-only settings B4's `reset_native` needs. Add
`create_attacker_state_from_native`/`create_defender_state_from_native`
handling (or dyna-specific variants) for B4's new `modification_record`/
new-node-id output, resolving back into `AssetOp`/`AssocOp`/
`AttackGraphNode` objects - same "resolve ids back to real objects at the
Python boundary" discipline as A9. Keep `DynaMalSimulator.__init__`/
`.reset()`/`.step()`'s outer signatures byte-for-byte identical. Full
`tests/test_dyna_mal_simulator.py` green (not just a smoke subset) is the
acceptance gate here, same as A9 was for `test_mal_simulator.py`.

**B6. Audit every other `DynaMalSimulator` public method
(A10-equivalent).** Before writing anything, diff
`dyna_mal_simulator/*.py` across the Phase A → Phase B commit boundary -
A10 already confirmed `DynaMalSimulator` never sets
`self._native_sim` and inherits all 19 of `MalSimulator`'s audited public
methods unmodified, operating generically on `self.sim_state`/
`self._agent_states`, so this audit may turn out to need zero code
changes for those 19, same shape as A10's own result. What's actually new
to audit here is `DynaMalSimulator`-specific surface A10 didn't cover at
all: nothing public beyond `__init__`/`from_scenario`/`reset`/`step`
appears to exist today (confirmed via the class body above), so this step
may collapse to "confirmed nothing new to audit" - but do the diff before
assuming that, not instead of it.

**B7. Full parity pass + cleanup (A11-equivalent).** Full `pytest tests`
(including `integration`) green. Decide per now-Rust-shadowed dyna Python
module (`model_effects.py`, `process_assoc_traversal.py`, `model_state.py`)
whether to delete or keep as a cross-check oracle, same lean-towards-delete
default as A11. Re-run `envs/`/`policies/`/`visualization/` as the
integration check. **Specifically re-run and scrutinize
`test_no_memory_leak_on_teardown`/`test_no_memory_growth_over_repeated_simulations`**
(`test_dyna_mal_simulator.py` lines 963-1048) - these matter more here
than they did anywhere in Phase A, since B3 introduces a *second*
independently-dropped `Rc<RefCell<...>>` shared across the FFI boundary
(the `Model`, alongside A1's `AttackGraph`), and Phase A's A1 double-free
bug (§10) is exactly the failure mode a second capsule-handoff could
reintroduce if its destructor isn't symmetric with A1's.

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
  seed= tests/` check (per §2.1) - `tests/test_ttc_utils.py` still does
  NOT need relaxing even after A9, since `ttc_utils.py` itself was never
  touched (§10: `DynaMalSimulator` still needs it pure-Python).**
  **B7 update:** `ttc_utils.py` lost its module-level graph functions
  (`attack_step_ttc_values` etc.) but `TTCDist` itself is unchanged, and
  `test_ttc_utils.py` only exercises `TTCDist`, so this entry still holds. This
  entry's original framing ("once Python's `ttc_utils.py` actually starts
  delegating to native RNG (A9)") turned out to describe a trigger that
  never happens in A9 - `MalSimulator`'s new native-backed path bypasses
  `ttc_utils.py` entirely (settings flatten straight from
  `AttackerSettings`/`MalSimulatorSettings` into native, per
  `native_settings.py`) rather than making the *existing* module
  delegate, so `test_ttc_utils.py::test_ttcs_effort_based`/
  `test_bernoulli` still exercise the unchanged pure-Python/numpy/scipy
  path and still pass unmodified. Re-flagging for real only if/when
  `ttc_utils.py` is itself deleted or rewritten (Phase B, once
  `DynaMalSimulator` moves off it too - see §10) - not a current risk.
- **New seed-pinned/exact-order tests found at A9 (resolved, not just
  flagged - see §10 for the fixes) - recorded here per §8's "list it
  explicitly" rule, for anyone auditing what A9 actually changed in the
  test suite.** `tests/test_mal_simulator.py::test_attacker_step_attempts_
  register`/`test_simulator_attacker_override_ttcs_state`/
  `test_simulator_attacker_override_ttcs_step`/
  `test_simulator_multiple_entry_point_sets_in_attacker_settings`,
  `tests/envs/test_example_scenarios.py::test_bfs_vs_bfs_state_and_reward_
  per_step_ttc`/`_per_step_effort_based`/`_expected_value_ttc`, and
  `tests/test_attacker.py::test_attack_surface_coreLang_include_unnecessary`
  all asserted an exact sampled value, exact RNG-choice outcome, or exact
  set-iteration-dependent traversal count/sequence for a fixed seed -
  each relaxed to a structural assertion (where the *property being
  tested* didn't actually need the exact value) or re-pinned to this
  port's own new, still-fully-deterministic-per-seed output (where the
  test's whole point was pinning a golden trace). None left broken.
- §2.2's coupling to `maltoolbox-attackgraph-py`'s internal struct layout
  - re-verify `PyAttackGraph.inner`'s visibility/shape on every
    `mal-toolbox` git dependency bump.
- **Performance: resolved, not just flagged - see §10 for the fix.**
  (Found and fixed between A10 and A11, outside the main phase sequence
  - not itself a numbered phase.) Profiling a 5000-step run on A10's
  `HEAD` (commit `1c1b62e`)
  found this entry's predicted risk had actually landed: `rust-rewrite`
  was 2.6x *slower* than pre-port pure-Python `main` (111.6s vs. 42.7s),
  with `step_native` alone (67% of total time) and
  `create_attacker_state_from_native`'s dense `num_attempts` rebuild
  dominating. Root cause wasn't the id-resolution cost itself (this
  entry's original framing) but a broader instance of the same
  anti-pattern: `step_native` resent the *entire* episode-accumulated
  state every step (including fields - `ttc_values`,
  `necessity_per_node`, `impossible_attack_steps`,
  `pre_enabled_defenses`, `logs` - that either never change after reset
  or already had their per-step delta computed and then discarded), and
  both `*_state_factories.py` modules re-resolved/re-parsed all of it
  from scratch every call instead of merging against `previous_state` -
  `defender_state_factories.py`'s `logs` handling was the worst case,
  O(episode²) (re-parsing every log ever fired into a fresh `LogEntry`
  on every single step). Fixed exactly as this entry's original
  "incremental resolution" suggestion described, extended to every
  affected field: `step_native`'s wire format is now delta-only for
  monotonically-growing fields, and both factories merge each delta
  against `previous_state` instead of re-resolving the full accumulated
  value. Re-profiling after the fix: a 5000-step run with every detector
  firing every step (10,002 accumulated logs by the end - the A9-era
  worst case for the old O(episode²) behavior) completed in ~1s, with
  per-step cost flat rather than growing - confirms the fix, not just
  the absence of the specific symptom originally profiled.
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
- **A10's hybrid category - `graph_utils.py`'s `node_is_blocked`/
  `node_is_traversable` vs. `malsim-core::graph_utils`'s A4 port of the
  same predicates - is a second two-implementations-can-drift pair,
  same shape as the entry above.** Unlike that entry, there's no schema
  boundary forcing a parity test (both sides are plain boolean logic over
  the same few fields, not a YAML schema), so the mitigation here is
  weaker: if either side's blocked/traversable logic changes, the other
  needs a matching manual update, caught only by the existing Python
  test suite happening to exercise both call paths (the Rust port inside
  `step()`'s hot loop via native, the Python version via
  `MalSimulator.node_is_blocked`/`.node_is_traversable`) rather than by
  any dedicated cross-check. Revisit if `attacker_step`/`defender_step`'s
  (A7) traversal logic ever changes without a corresponding
  `graph_utils.py` update, or vice versa.
- **B3's local `uv`/`maturin` build-cache anomaly (see §10 for the full
  writeup) - not resolved, only worked around, and may resurface for
  whoever runs B4+ in the same sandbox.** `uv run maturin develop --uv`'s
  own resolution of the `mal-toolbox` git dependency was observed to
  install a wheel missing a method that definitely exists in the pinned
  commit's source (confirmed via an independent from-scratch clone +
  `maturin build --release`), even right after `uv cache clean
  mal-toolbox` and with a correctly-updated `direct_url.json`. Root cause
  not identified (candidates: a uv-internal build/metadata cache keyed on
  something coarser than the resolved commit sha, or a build-isolation
  environment reuse quirk) - no `.cargo/config.toml`/env-based wrapper
  (sccache/ccache) was present to blame. Before trusting any future
  Python-level test failure as a sign of an upstream mal-toolbox Rust bug,
  first confirm what's actually installed: `strings <path to the
  installed maltoolbox/_native*.so> | grep <the-symbol-in-question>`. If
  it's missing despite a correct `rev` pin, rebuild via a fresh clone +
  `maturin build --release --manifest-path py-bindings/maltoolbox-pyo3/
  Cargo.toml -o <dir>` and `uv pip install --reinstall-package
  mal-toolbox <that wheel>` rather than re-running `uv run maturin
  develop` and hoping. **Addendum from B4/B5: plain `uv run <cmd>` can
  re-trigger this mid-session even after the manual fix above** - it
  re-syncs the project's dependencies (including `mal-toolbox`) before
  running, silently reinstalling the broken wheel over the fixed `.so`.
  Use `uv run --no-sync <cmd>` for every Python-level command for the
  rest of the session once the manual `.so` copy has been applied.
- **BLOCKING B5, stale-id panic class (RESOLVED session 2 - see §0's B5
  entry for the full writeup and its CORRECTION paragraph, this is the
  index pointer): the original diagnosis here was wrong and is kept below,
  struck through in spirit, purely so the correction is findable from
  both ends.** ~~native dyna stepping can panic the whole Python process
  (`invalid SlotMap key used`) on scenarios where a model effect's
  association churn leaves an asset fully disconnected, causing upstream
  mal-toolbox's `partially_regenerate_graph` to regenerate that asset's
  attack-step node(s) under a new internal id even though no asset was
  actually removed.~~ **What's actually true, confirmed via `eprintln!`
  instrumentation at the mutation site, not inferred from a backtrace:** a
  step's own model effect can remove the *step's own backing asset*
  outright (no regeneration, no new id - `removed_assets` for that call
  genuinely names it) - `collect_logs` was simply the first of several
  call sites (`malsim-core`'s `event_logger.rs`/`dyna_attacker_step.rs`
  and, not previously identified at all, `malsim-pyo3::simulator.rs`'s
  `stable_ids`/`id_value_map`/`attacker_ttc_overrides`/`log_entry_to_py`)
  that assumed an id, once seen live, stays resolvable for the rest of the
  episode. All of these now guard with a liveness check and skip rather
  than index-and-panic - see §0 B5 for the per-site reasoning. Full
  `cargo test`/`clippy -D warnings`/`fmt --check` clean in both
  workspaces; `test_different_attackers` 0/12 -> 10/12.
- **BLOCKING B5, remaining 2/12 (NOT resolved, newly exposed once the
  panic above stopped masking it - see §0's B5 entry for the full writeup
  and repro): `TTCSoftMinAttacker` hits a Python-side
  `AssertionError: Node <X> does not have a ttc value` for a node created
  mid-episode by a dyna model effect.** Confirmed a genuine Rust-port
  regression (pure-Python `DynaMalSimulator` has zero such gaps on the
  same scenarios+policy), not root-caused beyond "these nodes never reach
  `add_new_nodes_to_graph_state`'s per-node TTC computation, or its result
  doesn't survive to be read later" - not yet narrowed further. Not a
  panic - a normal catchable exception - so lower severity than the class
  above, but still blocks B5's acceptance gate (full green
  `test_dyna_mal_simulator.py`).

- **Native memory grows across repeated `DynaMalSimulator` runs that
  apply model effects (found at B7; pre-existing; upstream).** About
  110-125 KB per run on `wiper_scenario.yml`. The Python heap stays flat,
  and the existing gc-based leak tests can't see it. The pre-B4
  pure-Python dyna path grows faster (~220 KB/run), and valgrind
  attributes the retained memory entirely to mal-toolbox's native module,
  none to `malsim_pyo3`/`malsim_core` (see §0 B7). Matters for very long
  training loops that build fresh dyna simulators thousands of times.
  Repro: load a dyna scenario, build `DynaMalSimulator.from_scenario`,
  `run_simulation` with `RandomAgent`, repeat 300x, and read `VmRSS` from
  `/proc/self/status` after `gc.collect()` + `malloc_trim(0)` every 50
  runs. Worth reporting upstream to mal-toolbox with this repro.

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

**(Superseded at B7: viability was ported after all, to `viability.rs`, so `graph_processing.py` could be deleted without losing test coverage; see §0 B7 and §11.)** **A3: viability (`calculate_viability`/`evaluate_viability`/
`_propagate_viability_from_node`/`make_node_unviable`/
`prune_unviable_and_unnecessary_nodes`) is not ported at all, by design -
not an oversight.** `graph_processing.py`'s own module docstring already
calls viability "(deprecated)", and `grep -rn` across `python/malsim` for
`calculate_viability`/`viability_per_node`/`make_node_unviable`/
`prune_unviable_and_unnecessary_nodes`/`evaluate_viability` turns up no
hits outside `graph_processing.py` itself - nothing in the simulator's
step loop, state factories, or anywhere else calls into it. §5's A3
description only scopes in "`graph_processing.py`'s necessity
propagation", consistent with this. If a future phase discovers a real
caller, port it then rather than assuming this was dropped by mistake.

**A3: every graph-dependent function is split into a thin
`AttackGraphNode`-reading wrapper plus a graph-independent helper over
plain data - a structural choice driven by what's testable, not just
idiom.** E.g. `resolve_ttc_dist(node, override)` is one line delegating to
`resolve_ttc_dist_from_parts(node.ttc.as_ref(), node.step_type.as_str(),
override)`; `attack_step_ttc_value` delegates its mode-dispatch to
`ttc_value_for_dist(&TtcDist, TtcMode, &mut impl Rng)`;
`get_pre_enabled_defenses`'s per-node body is `is_pre_enabled_for_dist(&TtcDist,
bool, &mut impl Rng)`; `is_impossible_attack_step` is
`is_impossible_for_dist(&TtcDist, &mut impl Rng)`. The reason: constructing
a real `AttackGraphNode` with a specific `step_type` needs a real
`maltoolbox_language::graph::LanguageGraph` to mint its `AttackStepId` -
unlike `AttackGraphNodeId` (used throughout these modules as a `HashMap`/
`HashSet` key), `AttackStepId` can't be faked via `Default`/a hand-built
literal when a test needs to pick a *specific* step type (`Default`
produces slotmap's null key, fine for inert unused fields, not fine when
the test's whole point is "given a `defense` node..."). Decomposing this
way means the actual interesting logic (TTC-mode dispatch, the
degenerate-probability pre-enable branches, the Bernoulli-attempt
inversion) is still fully unit-tested now, even though the
`AttackGraphNode`-touching glue isn't.

**A3: a dev-dependency question was raised with the user and explicitly
deferred, not resolved - read this before adding
`maltoolbox-language`.** Testing `necessity.rs` at all (every case needs
a real node with a specific `step_type` and real parent/child
`AttackGraphNodeId` links - there's no way to decompose necessity
propagation into a graph-independent helper the way `graph_state.rs`'s
functions were, since the graph structure *is* the logic here), and
testing the thin wrappers listed above, both need a real
`maltoolbox_language::graph::LanguageGraph` - buildable by compiling
`tests/testdata/langs/dummy_lang.mal` via `maltoolbox_language::
from_mal_spec` (pure Rust, tree-sitter-based, no external tooling;
confirmed by reading `core/maltoolbox-language/src/compiler/mod.rs` and
`graph/file.rs`), the same fixture Python's `conftest.py::dummy_lang_graph`
already uses. Doing this means adding `maltoolbox-language` to
`core/malsim-core/Cargo.toml`'s `[dev-dependencies]` (test-only; already a
transitive dependency via `maltoolbox-attackgraph`, so no new external
dependency tree, just an explicit direct reference to something already
fetched at the same pinned git rev). Asked the user; they chose **"defer
rather than add it in this step"** over adding it now. Consequence:
`necessity.rs` ships with zero Rust-native tests this phase, and
`tests/test_graph_processing.py`'s necessity cases
(`test_necessity_necessary`, `test_necessity_unnecessary`,
`test_analyzers_apriori_propagate_necessity`) have no Rust counterpart
yet - tracked, not silently dropped (§8). This will almost certainly come
up again at A4 (traversal predicates need the same kind of node/graph
fixtures) - when it does, add the dependency then and backfill A3's
deferred tests in the same change rather than asking a third time.

**A3: `success_probability(0)`'s "never/always succeeds" branches in
`get_pre_enabled_defenses`/`is_pre_enabled_for_dist` are *inverted*
relative to the named dist's own name, confirmed intentional (ported
as-is) via the Python source's own uncertainty about it.** `TtcDist::
success_probability(effort)` is `dist.cdf(effort)`, and for
`Bernoulli(p)`, `cdf(0) == 1 - p`. So the *named* `"Disabled"` dist
(`Bernoulli(0.0)`) has `cdf(0) == 1.0` (hits the "always succeeds" branch
-> **not** pre-enabled), while `"Enabled"` (`Bernoulli(1.0)`) has
`cdf(0) == 0.0` (hits the "never succeeds" branch -> pre-enabled) - the
opposite of what the branch comments' wording suggests at a glance, until
you notice it's `cdf` not the raw threshold. This is exactly what
`get_pre_enabled_defenses`'s Python source does too, including its own
`# TODO: is this correct?` comment on this exact branch - so this was
ported faithfully, bugs/uncertainty and all, per `PORTING_NOTES.md`'s
general stance on not "fixing" ported behavior unasked. Caught because
the first draft of `is_pre_enabled_for_dist`'s tests assumed the naive
(name-matching) direction and failed; fixed the *tests*, not the
implementation - see `pre_enabled_degenerate_disabled_dist_is_never_pre_enabled`/
`pre_enabled_degenerate_enabled_dist_is_always_pre_enabled` in
`graph_state.rs` for the corrected, documented expectations.

**A4: `maltoolbox-language` added as a `[dev-dependencies]` crate
dependency of `malsim-core`, resolving A3's deferred question - asked the
user again rather than assuming A3's note settled it on its own.** A3
(above) flagged that testing real-`step_type` fixtures would hit the same
wall again at A4 and recommended adding the dependency then. Per standing
project guidance, a new crate dependency during this port is always an
"ask the user" decision even when a prior note already anticipated it -
asked again at the start of A4 rather than silently acting on A3's
recommendation. The user chose to add it now and have A4 backfill A3's
deferred `necessity.rs` tests in the same change (rather than, e.g., add
it but defer the backfill, or defer the dependency itself again). Net
effect: `necessity.rs` and `graph_utils.rs` share one test-only fixture
module, `core/malsim-core/src/test_fixtures.rs`, which compiles
`tests/testdata/langs/dummy_lang.mal` via `maltoolbox_language::
from_mal_spec` - the same `.mal` file `tests/conftest.py::dummy_lang_graph`
compiles for the Python suite, so both language's unit tests exercise
identical graph shapes for equivalent cases (e.g. `graph_utils.rs`'s
`node_is_blocked_matches_python_test_node_is_blocked` builds the exact
same graph as `tests/test_graph_processing.py::test_node_is_blocked`).

**A5: `node_is_actionable`'s `NodePropertyRule | None` parameter becomes
`Option<&HashSet<AttackGraphNodeId>>` in both `attack_surface.rs` and
`defense_surface.rs` - two independent copies of the same tiny helper, not
a shared one.** Each module defines its own private
`node_is_actionable_flat` with identical logic (`None` -> every node
actionable; `Some(set)` -> membership test) rather than a single shared
function in `graph_utils.rs`. Deliberate, not an oversight: `graph_utils.rs`
documents `node_is_actionable` as deliberately *not* ported there (§2.4 -
it stays Python, operating on the real `NodePropertyRule`), so adding a
same-named-but-different-signature Rust function to that module would be
confusing for a reader grepping for `node_is_actionable`'s Rust
counterpart and not finding the real one. Two 4-line private copies, one
per consuming module, was judged lower-risk than either sharing a
one-off helper across two otherwise-independent modules or placing it
somewhere that implies it's the ported `node_is_actionable`.

**A5: `get_attack_surface` carries `#[allow(clippy::too_many_arguments)]`
rather than a bundling struct.** At 9 parameters (mirroring the
independent pieces of `MalSimulatorState`/`AttackSurfaceSettings` the
Python function reads off `sim_state`/`settings`), it trips clippy's
default 7-argument lint. Considered bundling `impossible_attack_steps`/
`enabled_defenses`/`necessity_per_node` into a one-off struct to get under
the limit; rejected because no other function needs that exact bundle
(`get_effects_of_attack_step` takes the same three but stays under the
limit at 6 args total, `get_defense_surface` only needs two of the
three) - a struct with a single caller is an abstraction for the lint's
sake, not for the code's, so the lint is silenced with a comment instead
per the project's general stance against introducing abstractions beyond
what's needed. Revisit if A9's settings-flattening work ends up
threading these same three values through enough call sites that a real
shared bundle (e.g. sourced from `GraphState` + `enabled_defenses`)
earns its keep on its own merits.

**A5: `get_effects_of_attack_step` grows one `visited` set incrementally
instead of recomputing `performed | set(effects)` every loop iteration -
an idiom difference, not a behavior difference.** Python's
`has_visited = performed | set(effects)` reallocates a new set on every
`while` iteration; the Rust port instead seeds `visited` once from
`performed_nodes` + the starting `attack_step_id` and inserts each newly-
found effect into it immediately (alongside inserting into the separately-
tracked `effects` result set). Since `performed_nodes` is never mutated
and `effects` only ever grows, `visited`'s membership at every check point
is provably identical to what the Python union would have recomputed -
verified by `effects_stop_at_already_visited_node`/
`effects_do_not_cross_blocked_and_step` in `attack_surface.rs`, which
exercise exactly the revisit/non-traversable-skip paths this change
touches. Flagged here per §2.7's rule even though it's "just" an
efficiency idiom, since a future reader diffing against the Python
source line-by-line would otherwise wonder where the per-iteration union
went.

**A6: `LogEntry`'s `detector`/`trigger` fields become id-based
(`AttackGraphNodeId`/`DetectorId`), and a detector's identity on the Rust
side is `(AttackGraphNodeId, String)` - the node it's attached to plus its
label key - since mal-toolbox's `Detector` type has no id of its own.**
Confirmed by reading `maltoolbox_attackgraph::generate::create_detectors`
(the only place `Detector` values are constructed): every `Detector.node`
is set to the exact node whose `detectors: HashMap<String, Detector>` map
it's inserted into, under the same `label` key used as that map's index -
so `(node_id, label)` is a sound, always-resolvable identity
(`graph.nodes[node_id].detectors[&label]` recovers the same `Detector`),
even though it's a composite rather than a single scalar id. This is
exactly what A6's own `PORTING_NOTES.md` §5 description anticipated
("detector id + node id") - recorded here per §2.7 since it's a concrete
design choice a future reader of `event_logger.rs` should know the reason
for, not just see as a given shape.

**A6: Python's `attack_graph.detectors` (a lazily-materialized, mutably-
divergent Python-side list in mal-toolbox's PyO3 layer) is *not* what
`event_logger.rs`'s `collect_false_positives` reads - it walks
`graph.nodes[*].detectors` directly instead, which is what `.detectors`
is itself seeded from at first access.** Confirmed by reading
`maltoolbox-attackgraph-py`'s `graph.rs`: `.detectors` lazily builds a
`Py<PyList>` from the per-node maps once, then becomes "the sole source of
truth from then on" - meaning a Python caller that mutates `graph.detectors`
directly (as `tests/test_event_logger.py::_force_detector_rates` does,
replacing one node's detector and keeping both the node's own dict *and*
the list in sync by hand) diverges from the underlying per-node maps.
`malsim-core` has no such cache (it isn't behind PyO3 at all yet - A6 is
pure Rust, no FFI), so it always reads the authoritative per-node maps,
matching what `.detectors` holds at any point *before* that kind of direct
list mutation happens. Not a concern for A6 itself (nothing here crosses
into Python), but worth flagging now for whoever wires A8/A9's native hot
loop to the shared live graph: the native side must never be handed (or
read through) Python's `.detectors` list, only the per-node maps, or it
risks disagreeing with a caller that's mutated the list directly without
updating the nodes (or vice versa).

**A6: `collect_logs`/`collect_false_positives` preserve Python's exact
truthiness and evaluation-order quirks for `detector.tprate`/`.fprate`,
not just their probability semantics - ported as-is, not "fixed".**
Three specific things carried over deliberately:
- A rate of `None` *or* `0.0` is "falsy" in Python (`if detector.fprate:`/
  `not detector.tprate`) - `rate_is_truthy` in `event_logger.rs` mirrors
  this exactly (`rate.is_some_and(|r| r != 0.0)`), including the
  consequence that a *negative* rate is still truthy (Python doesn't check
  sign, only non-zero-ness) but can never satisfy `rate >= roll` since
  `roll` is drawn from `[0, 1)` - so a negative `tprate` reads as
  "configured, but mathematically never succeeds", distinct from an
  unconfigured (`None`) rate reading as "always succeeds" (ported
  faithfully; see `collect_logs_tprate_negative_is_truthy_but_never_fires`,
  mirroring `tests/test_event_logger.py::
  test_logger_attacks_false_negative`'s `tprate=-1.0` case).
- `collect_logs` calls `get_context` **unconditionally**, before the
  true-positive roll - mirroring Python's `labeled_steps = get_context(...)`
  appearing before its `if not detector.tprate or ...:` check, so a
  `get_context` failure (no previously-compromised candidate for some
  context label) surfaces even for a detector that would've been a false
  negative anyway. `collect_false_positives` calls `get_random_context`
  **only inside** its `if` block - lazily, only for a detector that
  actually fires. This asymmetry between the two functions is Python's own
  (not introduced by the port) and is preserved rather than made
  consistent.
- Both functions draw from `rng` **only when the rate is truthy** -
  Rust's `&&`/`||` short-circuiting (`rate_is_truthy(detector.fprate) &&
  detector.fprate.unwrap() >= rng.random::<f64>()`, and the `!...  || ...`
  mirror in `collect_logs`) reproduces Python's own short-circuit
  (`detector.fprate and detector.fprate >= rng.random()` never calls
  `rng.random()` when the first operand is falsy). Not required for
  correctness per §2.1 (bit-identical reproducibility isn't a goal), but
  kept anyway since it was free and keeps the Rust and Python code
  obviously in step for a line-by-line reader.

**A7: `state_query.py::node_ttc_value` is pulled forward into
`attacker_step.rs` as a private `resolve_ttc_value` helper, rather than
staying in its own module or waiting for a ported `state_query.rs`.**
`attempt_attacker_step`'s `EXPECTED_VALUE`/`PRE_SAMPLE` branch needs
exactly `node_ttc_value`'s precedence logic (an agent-level override
wins over the graph-level default computed once at `reset()`, else a
hard failure), and `attempt_attacker_step` is its *only* caller in the
hot loop - same situation and same resolution A3 used for `TtcMode`
(pulled forward from `config/sim_settings.py` into `graph_state.rs` for
the same reason). Rather than threading an `AttackerState`-shaped
object through (none exists in Rust yet - §3/A9), the two maps
`node_ttc_value` reads become plain arguments: `ttc_value_overrides:
Option<&HashMap<AttackGraphNodeId, f64>>` (the agent-level override) and
`graph_ttc_values: &HashMap<AttackGraphNodeId, f64>` (`GraphState::
ttc_values`, computed by A3's `compute_initial_graph_state`). Python's
`assert node in attacker_state.sim_state.graph_state.ttc_values` becomes
`AttackerStepError::MissingTtcValue` instead of a panic.

**A7: `attempt_attacker_step` resolves `ttc_dist` *unconditionally*,
even in `Disabled` mode where the result is never used - ported as-is,
not short-circuited.** Python's own source resolves `ttc_dist` before
its `if ttc_mode == TTCMode.DISABLED: return True` check, so a node with
a malformed `ttc` dict still raises even when TTCs are globally
disabled. `ttc.rs`'s port preserves this exact ordering (see
`attempt_disabled_mode_still_surfaces_malformed_ttc_dict` in
`attacker_step.rs`, which forces this by giving a node an empty-object
`ttc` value and asserting the error still surfaces under `Disabled`
mode) rather than moving the `Disabled` check first for a "cleaner"
early return.

**A7: the `EXPECTED_VALUE`/`PRE_SAMPLE` branch's `num_attempts + 1 >=
ttc_value` comparison is two attempts ahead of what the variable name
suggests - confirmed intentional (ported as-is), not a transcription
bug.** Python's `attempt_attacker_step` does `num_attempts =
agent.num_attempts[node] + 1` once at the top (used as-is by the
`EFFORT_BASED_PER_STEP_SAMPLE` branch), then *this* branch computes
`num_attempts + 1 >= _node_ttc_value` - i.e. the actual comparison is
`agent.num_attempts[node] + 2 >= ttc_value`, not `+ 1`. `attacker_step.rs`
reproduces this precisely (`(num_attempts + 1) as f64 >= ttc_value`
where `num_attempts` is already `num_attempts_before + 1`) rather than
"fixing" what looks at a glance like a double-increment -
`attempt_expected_value_mode_comparison_is_two_attempts_ahead` locks in
the exact boundary (a `ttc_value` of `2.0` already succeeds on the very
first attempt, when `num_attempts_before` is still `0`).

**A7: `attacker_step`'s "entry points bypass both the action-surface and
traversability checks" is ported as-is, including Python's own
uncertainty about it.** Python's `attacker_step` sets `can_compromise =
True` unconditionally for any node in `agent.settings.entry_points`,
skipping both the `node in agent.action_surface` and
`node_is_traversable(...)` checks entirely - directly under a `# TODO:
should this actually be the case?` comment in the Python source. Ported
faithfully (`attacker_step`'s `entry_points.contains(&node_id)` branch
short-circuits before any traversability check), not tightened into a
stricter check; `step_entry_point_bypasses_action_surface_and_traversability`
in `attacker_step.rs` exercises exactly this by using an `and`-step with
an unperformed necessary parent (otherwise untraversable) as the entry
point and asserting it still compromises.

**A7: Python's `assert node == sim_state.attack_graph.nodes[node.id]`
(identical in both `attacker_step` and `defender_step`) becomes a plain
graph-membership/liveness check (`graph_utils::node_is_live`) on the
Rust side, returning `NodeNotInGraph` instead of panicking.** The Python
assert is an object-identity check guarding against a caller holding a
stale `AttackGraphNode` reference (relevant once a node can be
removed/replaced, e.g. by `DynaMalSimulator`'s model effects - Phase B,
not yet ported). The Rust side works with `AttackGraphNodeId`s rather
than node references throughout, so there is no "stale object" to
compare against identity - the faithful equivalent of "this is really
the node currently in the graph" is "this id still resolves to a live
node". Both `AttackerStepError::NodeNotInGraph` and
`DefenderStepError::NodeNotInGraph` document this translation inline;
flagged here since it's a case where the id-based design genuinely
changes what the check *means*, not just how it's spelled, even though
the two sides should behave identically for every case `MalSimulator`
(as opposed to `DynaMalSimulator`) can actually produce today.

**A4: `node_is_blocked`'s `and`/`or` branches use the opposite
all-vs-any connective from what the type name might suggest - ported
as-is, not a transcription slip.** An `and` node is blocked if *any*
parent blocks it (`any(...)` in Python); an `or` node is blocked only if
*all* parents block it (`all(...)` in Python). This is correct given what
"blocked" means (permanently cut off, not "not yet reached"): an `and`
step needs every parent, so one permanently-blocked parent is enough to
block it forever, while an `or` step only needs one open parent, so every
single parent must be blocked before the `or` step itself is. Flagged here
per §2.7 because the `and`/`or`-labeled match arms in `graph_utils.rs`
look, at a skim, like they might have the connective swapped by mistake -
they don't; see the doc comment directly above `node_is_blocked` in
`graph_utils.rs` for the same note in-file. A related, already-existing
empty-parents edge case falls out of this unchanged from Python: an `or`
node with zero parents is "blocked" (`all(())` is `True`), an `and` node
with zero parents is not (`any(())` is `False`) - not reachable through
the public `node_is_traversable` path today (it requires `parents_reached`
first, which is false for zero parents), but preserved faithfully in
`node_is_blocked` itself since nothing in `graph_utils.py` guards against
calling it with a parentless node directly.

**A8: the FFI boundary uses `AttackGraphNode.id: i64` (the stable,
user-facing id), not `maltoolbox_attackgraph::ids::AttackGraphNodeId`
itself - a correction to §3's own wording, found while implementing.**
§3 says node ids crossing the FFI boundary are "ids (`i64`, matching
`maltoolbox_attackgraph::ids:: AttackGraphNodeId`)", which reads as if
`AttackGraphNodeId` *is* an `i64`. It isn't: it's a `slotmap::new_key_type!`
generational key (confirmed by reading `maltoolbox-attackgraph`'s
`src/ids.rs`), opaque and not constructible from a bare integer, and not
guaranteed stable/meaningful outside one process's `SlotMap`. The actual
stable, cross-boundary-safe identifier is `AttackGraphNode.id: i64` (its
own doc comment calls it "the stable, user/file-facing id", distinct from
the key), with `AttackGraph::id_to_node: IndexMap<i64,
AttackGraphNodeId>` (and the reverse, `graph.nodes[id].id`) as the two
translation directions - both of which `simulator.rs` uses
(`to_node_id`/`stable_ids` helpers) every time a node id crosses the
boundary in either direction. This is what the Python side already uses
too (`AttackGraphNode.id` is a plain `int` there, e.g.
`tests/test_mal_simulator.py`'s `key=lambda n: n.id` sorts), so no
behavior changed - just a correction to this document's earlier
description of the mechanism.

**A8: `AttackerRuntime.num_attempts` is a sparse `HashMap`, unlike
Python's `AttackerState.num_attempts`, which is dense (pre-populated with
every attack step at 0) - a deliberate scope simplification, not a bug.**
`attacker_state_factories.py::create_attacker_state` seeds
`previous_num_attempts` from `dict.fromkeys(sim_state.attack_graph.
attack_steps, 0)` when there's no previous state, so every attack step
node has an explicit `0` entry from the first state onward.
`simulator.rs`'s `AttackerRuntime.num_attempts` instead starts as an empty
`HashMap` and only gains entries for nodes actually attempted - a node
never attempted has no entry at all, rather than an explicit `0`. Every
*value* `node_ttc_value`/consumers would observe is identical either way
(`HashMap::get` absent vs. Python's dict lookup both effectively mean
"zero attempts so far" to any caller that doesn't enumerate the map's
keys expecting full coverage) - the two shapes only differ if something
iterates `num_attempts.keys()` expecting every attack step to be present,
which nothing in this phase's own `build_output` or its tests does. Left
as-is rather than pre-populating, since `reset_native`'s module doc
already scopes this phase down relative to the full `AttackerState`
surface and A9 - which actually rebuilds `AttackerState` from native
output - is the right place to decide whether the Python dataclass needs
the dense shape reconstructed on the Python side instead of carried
native.

**A8: `step_native` raises on an `actions` key naming no registered
agent, matching `simulator.py::_pre_step_check`'s `KeyError` - added
after independently re-reading `_pre_step_check`, since an early draft of
this phase omitted it.** `_pre_step_check` raises `KeyError(f"No agent
has name '{agent_name}'")` for any `actions` key not in `agent_states`;
`step_native` now does the equivalent check (`PyValueError`, which
`pyo3` surfaces as Python's `ValueError` rather than `KeyError` - a
reasonable boundary-crossing substitution, not a behavioral gap, since
nothing downstream matches on the specific exception type) before acting
on anything, covered by
`test_native_simulator_step_unknown_agent_raises`.

**A9: `num_attempts` densified back on the Python side, resolving A8's
own open question.** As A8's entry above anticipated, `create_attacker_
state_from_native` re-densifies native's sparse `num_attempts` map via
`dict.fromkeys(attack_graph.attack_steps, 0)` overlaid with native's
actual counts - needed because the pure-Python `attempt_attacker_step`
(still used directly by `DynaMalSimulator` and by
`tests/test_mal_simulator.py::test_attacker_step`, which calls it on a
`MalSimulator`-produced `AttackerState`) indexes `agent.num_attempts[node]`
unconditionally for any node about to be attempted, raising `KeyError`
on a sparse map for a node never attempted before. Found by running the
full test suite, not by inspection - confirms this was the right thing
to defer to A9 rather than guess at in A8.

**A9: `create_attacker_state`/`create_defender_state`/`initial_attacker_
state`/`initial_defender_state`/`attacker_overriding_ttc_settings`/
`get_entry_points` (its body) stay completely untouched (one narrow
signature change to `get_entry_points`, see below) - `MalSimulator`'s
native-backed path uses new, separate functions instead of rewriting
these in place, despite §4's table suggesting an in-place rewrite
("rewritten internally to build dataclasses from native output").**
Reality forced a correction: `python/malsim/dyna_mal_simulator/
simulator.py` imports `create_attacker_state`/`create_defender_state`
directly from these modules, and `dyna_mal_simulator/attacker_step.py`
imports the *pure-Python* `attacker_step`/`attempt_attacker_step`/
`attacker_is_terminated` from `mal_simulator/attacker_step.py` - none of
which are touched until Phase B (§2.3: `DynaMalSimulator` isn't ported
yet, and must keep working via its existing pure-Python recompute path,
unchanged, for the full test suite - including `tests/test_dyna_mal_
simulator.py` - to stay green per §1's non-breaking contract). Changing
`create_attacker_state`'s signature/behavior in place would have broken
`DynaMalSimulator` outright. §4's own hedge ("same names/signatures
*where feasible*") anticipated exactly this kind of case; the resolution
is new, additively-named functions (`create_attacker_state_from_native`/
`create_defender_state_from_native`) living in the *same* files,
alongside the untouched originals - not a new module, since they're
thematically about the same thing (attacker/defender state construction)
regardless of data source. Revisit once Phase B also moves
`DynaMalSimulator` to native: at that point the old functions become
genuinely dead and can be deleted (§4/A11's own anticipated cleanup,
just deferred one phase further than A11 originally implied for this
specific pair). The one exception, `get_entry_points`, got its parameter
narrowed from `sim_state: MalSimulatorState` to `attack_graph:
AttackGraph` (its body never read anything else) - safe because
`DynaMalSimulator` doesn't call it at all (confirmed via `grep -rn
get_entry_points`), and necessary because `MalSimulator`'s new `reset()`
must resolve "multiple entry point sets" sampling *before* a
`MalSimulatorState` exists to pass the chosen set into `reset_native`.

**A9: "multiple entry point sets, sampled at reset"
(`AttackerSettings.entry_points` as `tuple[Set, ...]`) is resolved in
Python, not added to native - a deliberate, permanent design choice, not
a deferral.** A8 left this as an open scope cut "for A9 to extend
properly," which read as an invitation to port the sampling into Rust.
It isn't: the sampling itself (`rng.choice` over a handful of sets) has
no performance stakes worth a native port, and no RNG-reproducibility
stakes either (§2.1's statistically-equivalent-only contract already
covers whatever this consumes). `MalSimulator`'s `reset()` calls the
unchanged `get_entry_points` once per attacker before building native's
settings dict, and only the *resolved* single set crosses the FFI
boundary - `reset_native`'s per-attacker `entry_points` field never
needed to grow a `tuple[Set,...]`-shaped alternative. Consequence:
`tests/test_mal_simulator.py::test_simulator_multiple_entry_point_sets_
in_attacker_settings` could no longer assert an exact seed → choice
mapping (compute_initial_graph_state's RNG consumption moved entirely to
native's own, separately-seeded `StdRng` - see next entry - so the
*position* of `get_entry_points`'s `rng.choice()` draw in the overall
sequence changed even though the call itself didn't) - relaxed to "the
chosen set is one of the configured options" (§9).

**A9: Python's `rng: np.random.Generator` and native's `StdRng` are two
independent RNG streams from A9 onward, not one interleaved sequence -
an unavoidable consequence of §2.1, worth stating explicitly since it's
the root cause of several relaxed/re-pinned tests in §9.** Before A9,
every sampling call (TTC, bernoullis, detector rolls, entry-point-set
choice) pulled from the single Python `rng` object, in a fixed order
determined by `reset()`/`step()`'s own code structure. After A9, `reset()`
derives one `native_seed = int(rng.integers(0, 2**63 - 1))` from the
Python `rng` stream and hands it to `reset_native`, which seeds its own
`StdRng` once and does *all* subsequent native-side sampling
(TTC/bernoulli/detector rolls) from that independent stream for the rest
of the episode; the Python `rng` object is only touched again for
Python-side-only randomness (currently: `get_entry_points`'s
`rng.choice`, and `rewards.py`'s `SAMPLE_TTC` reward mode, both
unaffected code paths A9 didn't move). This is why a fixed
`sim_settings.seed` is still fully reproducible (same Python draw for
`native_seed`, same native stream from then on - §1's "statistically-
equivalent" contract holds), but is *not* equivalent to the old single-
stream interleaving - any test that happened to pin an exact outcome
derived from the old interleaving order needed relaxing or re-pinning
(§9), not because determinism broke, but because the *mapping* from seed
to outcome changed shape.

**A9: a genuine pre-existing bug in A6's `collect_logs` port, found by
A9's end-to-end wiring (not by any A6-era test) and fixed, not just
documented.** `core/malsim-core/src/event_logger.rs::collect_logs`
checked `node.model_asset.is_none()` unconditionally for every
compromised node, before checking whether that node even had any
detectors - stricter than Python's `collect_logs`, whose equivalent
`assert attack_step.model_asset is not None` sits *inside* `for detector
in attack_step.detectors.values()` and therefore only ever runs when
there's at least one detector to check. A6's own dummy-node test fixtures
always set `model_asset`, so this never surfaced until A9 ran a real,
manually-constructed (no model) scenario through the full simulator
(`tests/agents/test_agents.py::test_defend_future_compromised_defender`)
whose pre-compromised entry-point node had neither a model asset nor any
detectors - which Python tolerates (no detectors means the assert is
never reached) but the old Rust code rejected with
`EventLoggerError::MissingModelAsset`. Fixed by guarding the check on
`node.detectors.is_empty()` first, matching Python's control flow
exactly; `collect_logs_missing_model_asset_errors` (which had encoded the
*bug*, not Python's real behavior - its dummy node had no detectors
either) updated to attach a detector first, and a new
`collect_logs_missing_model_asset_without_detectors_does_not_error` test
added for the previously-untested correct case.

**A9: `maltoolbox`'s `AttackGraphNode.children`/`.parents` Python
properties are a *read-mostly* cached `set` seeded once from the graph's
real edges - mutating that returned set in place
(`node.children.add(x)`) never writes back to the graph the shared
`Rc<RefCell<AttackGraph>>` handle actually reads, only *assigning*
`.children`/`.parents` does (`maltoolbox-attackgraph-py/src/node.rs`'s
`edges_sets`/`set_children`/`set_parents`: the setter calls
`set_edge_field`, which syncs; the cached getter's returned `PySet` does
not).** This is a `maltoolbox` binding characteristic, not a malsim bug,
and not something this repo can fix (out of scope - a different repo).
It never mattered before A9 because the old pure-Python `attack_surface.py`
/`graph_utils.py` only ever read `.children`/`.parents` through this same
cached getter too - internally consistent from Python's perspective, even
though disconnected from the graph's real edge storage (confirmed via
`AttackGraph._to_dict()`, which reflects the real storage and showed
empty `children`/`parents` for a graph wired via `.add()`). A9's native
code reads the graph's real edges directly, so any test that builds a
graph node-by-node via `.children.add(x)`/`.parents.add(x)` and then runs
it through `MalSimulator` silently got an empty attack surface once that
graph reached native (`tests/agents/test_searchers.py`/`test_agents.py`:
~35 occurrences across both files, affecting
`BreadthFirstAttacker`/`DepthFirstAttacker`/`RandomAgent`/
`DefendFutureCompromisedDefender` tests). Fixed at the test level, not
by avoiding native: added `tests/conftest.py::connect_nodes(parent,
child)`, which uses *assignment* (`parent.children = parent.children |
{child}`) instead of `.add()`, and mechanically replaced every
`.add()`-pair call site in those two files with it.
`tests/test_graph_processing.py` uses the same `.add()` pattern but never
constructs a `MalSimulator`/goes through native, so it's unaffected and
was left alone.

**A9: `maltoolbox`'s `AttackGraphNode.detectors` Python property has the
same "Python-side cache, real graph unaffected" shape as `.children`/
`.parents` above, but with no assignment-based escape hatch - fixed with
a new malsim-side native utility instead of a test-level workaround.**
`node.rs`'s `detectors` getter lazily seeds a Python-side cache dict from
"the core's generation-time detector data" and returns the *same* dict
object on every subsequent access (so `node.detectors['x'] = Detector(...)`
*looks* like a durable mutation from Python, per that getter's own doc
comment - "visible to later reads") - but "later reads" means later
Python reads of that same cache, not malsim-core's Rust `collect_logs`/
`collect_false_positives`, which read `AttackGraphNode.detectors`
(`maltoolbox-attackgraph`, the plain Rust crate) directly, and never see
the mutation. Unlike `.children`/`.parents`, there's no `set_detectors`
setter to fall back to. `tests/test_event_logger.py::_force_detector_
rates` used exactly this pattern (plus a since-irrelevant `graph.detectors
.remove/.append` - Rust's `collect_false_positives` iterates every node's
own `.detectors` map directly via `all_detectors`, confirmed in
`event_logger.rs`, so there's no separate graph-level detector list to
keep in sync at all on the Rust side) to force deterministic tprate/fprate
for `test_logger_attacks`/`_false_negative`/`_false_positive`, and
silently stopped working once those tests' simulators moved to native.
Fixed with a new, test-support-only `_native.set_detector_rates(graph,
node_id, label, tprate, fprate)` pyfunction (`py-bindings/malsim-pyo3/
src/lib.rs`, same tier as A1's `node_count` - not a real public API) that
mutates the real `Detector.tprate`/`.fprate` fields directly through the
already-shared `Rc<RefCell<AttackGraph>>`, reusing `maltoolbox-attackgraph`'s
already-`pub` `Detector` fields - no new crate dependency, no mal-toolbox
change needed. `_force_detector_rates` rewritten to call it.

**A9: native-returned node-id sets no longer preserve the same iteration
order Python's old sequential-loop code produced, which broke a few
tests that depended on "the order you get when you iterate a frozenset
built this way" without that ever being a documented contract - fixed
at two levels.** (1) `simulator.py`'s own `recording` log: the old
`step()` appended compromised/enabled nodes to a plain `list` in the
exact order `attacker_step`/`defender_step`'s sequential Python loops
produced them (requested action, then its effects, next requested
action, ...); native returns the full step's result as an unordered
`HashSet` (`step_enabled_defenses`/`step_compromised_nodes` in
`simulator.rs`), and the new factories compute "what's new this step" via
a Python `frozenset` difference (`step_performed_nodes`), which has no
defined order either. Fixed with a new `_ordered_new_nodes(requested,
new_nodes)` helper: explicitly-requested actions first, in the order the
caller supplied them, filtered to the ones that actually succeeded,
followed by any remaining effect-chain-only nodes (not explicitly
requested, so no canonical order recoverable from native's aggregated
output - documented as implementation-defined, not reproduced exactly).
Multi-action-per-step interleaving (old code: node₁, node₁'s effects,
node₂, node₂'s effects, ...) is only approximated as "all direct actions,
then all effects" when more than one action succeeds in the same step -
acceptable since nothing currently depends on finer-grained interleaving
once each individual action's own effects are correctly grouped with it
in the single-action-per-step case that's actually exercised.
(2) Several tests relied on `next(iter(some_frozenset), None)` or similar
raw-set-iteration-order traversal (`BreadthFirstAttacker`/
`DepthFirstAttacker`'s default `ActionOrdering.NOTHING`, whose own comment
already called this "theoretically non-deterministic but in practice
deterministic in CPython" - true for the old pure-Python frozensets, not
necessarily for frozensets rebuilt from Rust `HashSet`-returned id lists)
- those tests' exact pinned iteration counts/action sequences were
re-pinned to this port's own (still fully deterministic per seed, just
different) output rather than chased into matching the old order
exactly. Full list in §9.

**A9: native's per-agent `iteration` counter is 0-indexed from `reset()`
(0 immediately after reset, incrementing once per `step_native` call);
`AttackerState`/`DefenderState.iteration` stays 1-indexed, matching the
pre-A9 contract exactly (1 immediately after reset) - the native-driven
factories add 1, and use native's *un-incremented* value as the
`performed_nodes_order` key for that call, which is the exact value the
old pure-Python `create_attacker_state`/`create_defender_state` used
too (`previous_state.iteration` read *before* this call's `+1`).** Spelled
out here because getting this off-by-one wrong silently corrupts
`performed_nodes_order`'s keys without failing fast - confirmed by
reproducing `tests/test_mal_simulator.py::test_simulator_multiple_
attackers`/`test_simulator_multiple_defenders`'s exact `recording` dict
keys (`{1: ..., 2: ..., ...}`), not just by reasoning about it.

**A9: `AttackerSettings.ttc_dists`/`attacker_overriding_ttc_settings`'s
"override-only, not merged with the graph-wide values" semantics are
preserved exactly in native, including the detail that `.per_node()`'s
resolved values are treated as predefined-distribution *name* strings
(`TTCDist.from_name`), not arbitrary `TTCDist` objects, even though the
field's declared type is `NodePropertyRule[TTCDist]`.** When an attacker
has `ttc_dists` configured, `AttackerState.ttc_values`/`.impossible_steps`
contain *only* the override-affected nodes (not the full graph-wide map
with those nodes' values swapped in) - `simulator.rs`'s `attacker_ttc_
overrides` iterates just the override map's keys, mirroring
`attacker_overriding_ttc_settings`'s two `attack_step_ttc_values`/
`get_impossible_attack_steps` calls restricted to `ttc_overrides.keys()`,
not `malsim-core::graph_state::attack_step_ttc_values`'s own
all-attack-steps iteration (which would have silently produced the wrong,
merged shape). `native_settings.py::_flatten_ttc_dists` mirrors
`TTCDist.from_name(name)` unconditionally on each `.per_node()` value for
the same bug-for-bug-compatible reason - confirmed by keeping
`tests/test_mal_simulator.py::test_simulator_attacker_override_ttcs_
state`'s key-set assertion exact (only the structural "which nodes"
property, not the RNG-dependent sampled values - §9) rather than
widening it to "the full graph".

**A9: `MalSimulator._native_sim` is excluded from `__getstate__` with no
corresponding `__setstate__` repair, and this is an *extension* of an
existing limitation, not a new one.** `MalSimulator` already has no
`__setstate__`: `_defender_reward_fns`/`_attacker_reward_fns` are
similarly excluded from `__getstate__` and are simply absent from
`self.__dict__` after unpickling (Python's default `__reduce_ex__`
behavior with no `__setstate__` is `obj.__dict__.update(state)`, nothing
more) - calling `.agent_reward()`/`.agent_reward_by_name()` on a restored
object already raised `AttributeError` before A9, and nothing in the test
suite exercises that path (`tests/test_mal_simulator.py::
test_simulator_picklable` only pickles immediately after construction and
checks `sim_settings`/`attack_graph` equality, never calls `.step()` or
`.agent_reward()` afterward). `_native_sim` joining that same exclusion
list means `.step()`/`.reset()` on a restored object now *also* raise
`AttributeError` rather than silently operating on stale/mismatched
native state - a loud, immediate failure, not a correctness bug, and
consistent with the existing style of limitation rather than a new kind
of one. A real fix (reconstructing native's internal runtime state from
the already-faithfully-pickled mirrored `AttackerState`/`DefenderState`/
`MalSimulatorState` Python dataclasses) is possible but nontrivial
(native's own RNG stream position can't be recovered without Rust-side
RNG state serialization) and not warranted by any current test or usage
pattern - revisit if a real pickle-mid-episode-then-continue use case
shows up.

**A9: a genuine cross-process non-determinism bug, found and fixed after
the user flagged a flaky test post-review - `stable_ids` now sorts.**
`std`'s default `HashSet`/`HashMap` hasher is seeded from OS randomness
once per process, not from anything malsim controls - so the *same*
`reset_native`/`step_native` call, with the *same* seed, on the *same*
graph, returned `action_surface`/other id lists in a genuinely different
order on every separate process invocation (confirmed empirically: four
back-to-back `python -c` calls with identical inputs produced four
different orderings). This is strictly worse than the "order differs
from the old Python code" class of issue already documented above - it
meant the native layer itself wasn't reproducible run-to-run, which
§2.1's "statistically-equivalent" contract never intended to relax.
Fixed by sorting the `Vec<i64>` `stable_ids` builds before returning it
(`py-bindings/malsim-pyo3/src/simulator.rs`) - ascending by stable id,
cheap, and the only place node-id sets cross the FFI boundary as an
ordered `Vec` (the `HashMap`-returning `id_value_map` wasn't touched -
nothing reads those dicts' iteration order, only looks up by key).

**A9: this `stable_ids` fix alone did not fully stabilize two tests -
both root causes are pre-existing properties this port's RNG-stream
change (§2.1) merely made visible, not new bugs, and both are now fixed
at the test level.** (1) `ShortestPathAttacker`'s path-cost tie-breaking
(`path_finding.py::_find_path_to`'s `sorted(paths, key=itemgetter(1))`,
stable-sorted over `list(attacker_state.performed_nodes)`) is sensitive
to `frozenset` iteration order, which depends on each `AttackGraphNode`'s
`__hash__` - and `maltoolbox-attackgraph-py/src/node.rs`'s `__hash__`
folds in the *owning graph's raw pointer*, so a `frozenset` of nodes
iterates in a different order on every process run regardless of any
seed malsim controls, exactly like the `stable_ids` issue above but in
Python, on a dependency's object, unfixable from this repo.
`test_simulator_attacker_override_ttcs_step` asserted `good_iteration <
bad_iteration`; occasional cost ties (made more frequent by this port's
different TTC sample values for this seed) sometimes resolve both
attackers to the same iteration count depending on that unstable hash
order - a legitimate "tied" outcome, not a regression. Relaxed to
`good_iteration <= bad_iteration`, the invariant that actually holds
unconditionally, with the mechanism recorded in the test's own comment.
(2) `TTCSoftMinAttacker.get_next_action` (`policies/attackers/
ttc_soft_min.py`) samples from a softmax over TTC-derived weights via
its *own* `random.Random(seed)`, seeded from `agent_config.get('seed')` -
`tests/agents/test_ttc_avoider.py::test_ttc_avoider` constructed it with
an empty config (`TTCSoftMinAttacker({})`), so that `seed` was `None`,
drawing from OS entropy on every run. This was always probabilistically
flaky by construction (a softmax assigns the "hard" branch a small but
non-zero weight), confirmed by running the test's logic across 30
independent process invocations with no seed (~1/30 failure rate) vs. 40
with a fixed policy seed (0/40 failures) - this port's different TTC
values for this scenario's seed happened to narrow the easy/hard weight
gap enough to make the pre-existing flake practically visible instead of
theoretical. Fixed by seeding the policy in the test
(`TTCSoftMinAttacker({'seed': 0})`), not by touching the policy or the
simulator - `test_ttc_avoider_low_sharpness` (same file) already
exercises the genuinely-probabilistic low-sharpness case correctly, via
many trials and a tolerance, so this single-trial high-sharpness test
is the only one that needed a seed to be deterministic rather than
"probabilistic with the knob turned eliminated".

**Post-A10, pre-A11: `step_native`'s wire format changed from
full-accumulated-state to this-step's-delta-only, after profiling found
A9's original "resend everything every step" shape made the native port
2.6x slower than pre-port pure Python (see §9's performance entry for
the profiling numbers and root-cause writeup).** `reset_native`'s output
shape is unchanged - full fields, since there's no previous step to
delta against at reset. Only `step_native` changed, in
`py-bindings/malsim-pyo3/src/simulator.rs`:
- `build_output` was split into `build_reset_output` (old behavior,
  called only from `reset_native`) and a new `build_step_output` (called
  only from `step_native`), rather than adding a branch inside one
  function - the two now return genuinely different key sets, so one
  function trying to do both would need a bool parameter threading
  through every field, which is worse than two names that each do one
  thing.
- Four `sim_state` fields (`ttc_values`, `necessity_per_node`,
  `impossible_attack_steps`, `pre_enabled_defenses`) are dropped from
  `step_native`'s output entirely - they're computed once in
  `compute_initial_graph_state` during `reset_native` and never mutated
  by anything in `step_native`, so resending them every step was pure
  waste. Python caches them from `reset_native`'s output
  (`MalSimulatorState`/`simulator_state.py`) and carries them forward
  unchanged on every `step()` call - `update_simulator_state` already
  did this correctly for `enabled_defenses` (a `|=` merge, not a
  replace) before this change; it just also needed the key read renamed
  from `enabled_defenses` to `step_enabled_defenses`.
- `enabled_defenses` (top-level), `performed_nodes`/`attempted_nodes`
  (attacker), `performed_nodes`/`compromised_nodes`/`observed_nodes`/
  `logs` (defender) all changed from the full accumulated value to a
  `step_*`-prefixed delta - the same naming convention
  `AttackerState.step_attempted_nodes`/`.step_performed_nodes` already
  used for their own (Python-side-computed) per-step delta properties
  (`attacker_state.py`), so a reader already familiar with that
  convention recognizes the new wire keys immediately. `num_attempts`
  (attacker) is dropped entirely rather than getting a `step_` delta
  key, since Python derives the increment itself from
  `step_attempted_nodes` - sending a sparse per-step attempt-count map
  over the FFI boundary just to add 1 to each entry on the Python side
  would be pure overhead. `action_surface`/`iteration`/`terminated`
  (both agent kinds) are genuinely unchanged - `action_surface` is
  non-monotonic (nodes enter and leave it) and bounded by frontier size,
  not graph size, so it was never part of the problem and stays a full
  current value every step.
- `attacker_state_factories.py::create_attacker_state_from_native`/
  `defender_state_factories.py::create_defender_state_from_native` both
  gained a `previous_state is None` branch (first call after reset:
  native_agent_out still has the old full-field names/shape, since
  `reset_native` is unchanged) vs. the normal case (merge the `step_*`
  delta against `previous_state`'s already-resolved Python objects -
  `performed_nodes = previous_state.performed_nodes | step_performed_nodes`,
  same pattern for the others). This is the exact fix
  `PORTING_NOTES.md` §9 anticipated in advance for id-resolution cost in
  general ("cache resolved nodes, only resolve newly-added ids each
  step"), just needed for real once profiling confirmed it, and
  extended to a few more fields than that original note named
  explicitly (`ttc_values`/`impossible_steps` for attackers, now
  resolved once from `reset_native`'s output and reused unchanged on
  every later call rather than re-read from a key that no longer
  exists). The defender's `logs` field was the worst instance found:
  the old code's own docstring said native returned "the full
  episode-accumulated history ... not just this step's delta", which
  meant every log ever fired got re-parsed into a fresh `LogEntry` on
  every single step - O(episode²), not O(episode). Fixed the same way,
  with a new regression test
  (`tests/test_event_logger.py::test_logger_incremental_log_merge_matches_full_rerun`)
  that forces a detector to fire every step and checks the
  incrementally-merged `.logs` after each step is a growing,
  duplicate-free prefix matching a one-shot read at the end - the test
  that would have caught a broken merge (double-counted or dropped
  entries), which a simple "is it fast now" check would not.
- Re-profiling after the fix: a 5000-step run with a detector forced to
  fire every step (10,002 accumulated logs by the end, deliberately the
  worst case for the old behavior) completed in about 1 second, with
  per-step cost staying flat rather than growing with the accumulated
  log count - confirms the fix addresses the actual O(episode²)
  mechanism, not just the specific scenario originally profiled.
- No new crate dependency - this was a pure restructuring of existing
  `simulator.rs`/`*_state_factories.py` code (moving data already being
  computed, just discarded, into the FFI boundary's output, and merging
  on the Python side instead of re-resolving) - the "ask before adding a
  dependency" policy had nothing to trigger.

**Phase B1/B2 (`assoc_traversal.rs`/`model_effects.rs`/`dyna_graph_state.rs`/
`dyna_attacker_step.rs`/`dyna_defender_step.rs`/`model_state.rs`):**

- **`execute_model_effects` is deliberately split across the B1/B2
  boundary**, unlike Python where one function (`model_effects.py::
  execute_model_effects`) both applies the model effects *and* calls dyna
  `graph_state.py::add_new_nodes_to_graph_state` to fold the resulting new
  nodes into a `GraphState`, returning a fully-updated
  `DynaMalSimulatorState`. B1's `model_effects.rs::execute_model_effects`
  stops right after `partially_regenerate_graph`, returning
  `(modification_record, new_node_ids)` - it does not fold `new_node_ids`
  into a `GraphState` itself. B2's `dyna_attacker_step`/`dyna_defender_step`
  do that fold via `dyna_graph_state.rs::fold_new_nodes_into_graph_state`
  immediately after calling B1's function. Reason: `add_new_nodes_to_graph_state`
  is B2-scoped Python source (`dyna_mal_simulator/graph_state.py`, not
  `model_effects.py`) - B1 shouldn't reach into B2's not-yet-written code
  to stay a clean, independently-landable step, even though Python's own
  module boundary doesn't draw the line in the same place (it imports
  `graph_state.add_new_nodes_to_graph_state` into `model_effects.py`
  directly). The observable behavior at the B2 call sites is identical
  either way - this is a pure code-organization difference, not a
  behavioral one.
- **`sample_size`'s `quantity is None` branch doesn't clip against
  `max_size`, ported as-is from a latent Python oddity** - see
  `assoc_traversal.rs::sample_size`'s doc comment. Confirmed via grep that
  every real call site only reaches this branch with `max_size` left at
  its default (infinite), so the mismatch is unreachable today in both
  languages, not just "ported faithfully but now also unreachable."
- **`model_state.rs`'s `ModelSnapshot` is a new Rust type, not a literal
  port of Python's `dict[str, Any]` (`Model.to_dict()`) snapshot shape** -
  see `model_state.rs`'s module doc comment for the full reasoning. Short
  version: Python's `reconcile_model_to_snapshot` compares `frozenset(other.id
  for other in others)` (integer asset ids) against `frozenset(other_ids)`
  where `other_ids` is a `dict[str, str]` (`{"<id>": "<name>"}`) - iterating
  a dict yields its *string* keys, so the two frozensets can never
  intersect, and every reconciliation silently falls back to a full
  remove-and-re-add of every association. The *end state* is still
  correct (full teardown-and-rebuild reaches the same target), so no
  currently-passing Python test catches this - but faithfully reproducing
  it in Rust would require deliberately mistyping one side of a comparison
  that Rust's type system wouldn't let compile as mismatched in the first
  place. Chose correctness (the function's own docstring's "minimal diff"
  intent) over bug-for-bug fidelity here - the first and only case in this
  port where "port faithfully" and "write correct Rust" actually conflicted
  rather than just being in tension stylistically (§2.7).
- **One flaky test, found and fixed before landing** (not a port
  divergence, but worth recording per §8's testing discipline): an early
  `reconcile_model_to_snapshot` test asserted a *specific* association
  tuple appeared in the returned `new_associations` set. `Model::
  add_associated_assets` updates both sides of a bidirectional association
  (`sendTo`/`receiveFrom`) symmetrically, so depending on `HashMap`
  iteration order (randomized per-process) of the reconciliation loop,
  *either* asset's field-diff pass could end up the one that explicitly
  records the fix - the other side's correct end state arrives for free,
  with nothing recorded under its own tuple. Found via a 25-iteration
  repeat-run stress test (failed in roughly 4 of 5 separate `cargo test`
  process invocations, but never within one invocation's repeated test
  runs - the tell that it was process-seed-dependent iteration order, not
  genuine non-determinism). Fixed by asserting on the resulting model
  state / accepting either side's tuple, not by pinning iteration order.

**B3: `PyModel::__inner_capsule__` added upstream, mirroring A1's
`PyAttackGraph::__inner_capsule__` exactly - no new mechanism invented.**
§6's "New architectural wrinkle" note predicted this would need an
upstream mal-toolbox change "almost certainly... mirroring A1/§2.2's
precedent" - confirmed, and it really was a mirror, not a variant:
same `PyCapsule::new_with_pointer_and_destructor` call, same
strong-count-bump-then-`Rc::from_raw` extraction discipline on this
repo's side (`extract_shared_model`, copy-structured from
`extract_shared_graph`), same capsule-name-string-as-the-only-
cross-module-contract shape. Landed as mal-toolbox commit `b96258bad`
on `rust-rewrite`, pushed to `origin` (this repo's git dependency
resolves the branch over HTTPS, not a local checkout - a local-only
commit would not have been fetchable). The one genuinely new thing this
phase needed that A1 didn't: **proof that a mutation is visible in both
directions**, not just one. A1's own smoke test (`node_count`) only ever
*read* through the capsule - Phase A never mutates the shared graph from
both the Python and Rust sides within the same test, so it never had to
prove the Rust-side write path. B3's test exercises both directions
explicitly: `model.add_asset(...)` from Python, then
`_native.model_asset_count` immediately reflecting it (Python→Rust
visibility); then a new test-support-only `model_add_asset_native`
pyfunction (mutates via `shared.borrow_mut().add_asset(...)` directly, no
Python-level `Model.add_asset()` call at all) with the resulting id
checked against Python's `model.assets` (Rust→Python visibility). Both
directions passed on the first correct build, with no code changes needed
beyond the straightforward `extract_shared_model`/`model_asset_count`
pair - the capsule mechanism genuinely is symmetric, as A1's original
design intended.

**B3: a `uv`/`maturin` local build-cache anomaly cost most of this
phase's wall-clock time and is recorded here in full since §9's entry
only has room for the summary.** After pushing the upstream commit and
re-pinning this repo's `Cargo.toml`s to it, `uv run maturin develop --uv`
reported (correctly) resolving and reinstalling `mal-toolbox` at the new
commit sha - but the *installed* `.so`'s `strings` output showed zero
occurrences of `__inner_capsule__` in `Model`'s context (one occurrence
for `AttackGraph`, where it already worked), and `type(model).__dict__`
confirmed the method was genuinely absent from the installed type, not
just hidden from `dir()`. This looked exactly like a Rust-side bug - a
plausible one, since dunder-looking method names are exactly the kind of
thing that could trip up pyo3's magic-method detection. It wasn't: a
manual `cargo build -p maltoolbox-model-py`/`cargo clippy -D warnings`
against the live checkout was clean, and two independent manual rebuilds
- a raw `cargo build` (debug profile, `PYO3_PYTHON` pointed at this
project's venv to match ABI) and, more conclusively, a *from-scratch*
`git clone` to a scratch dir followed by `maturin build --release
--manifest-path py-bindings/maltoolbox-pyo3/Cargo.toml` (the same
pipeline a real install uses, no reused caches of any kind) - both
produced a `.so` where `__inner_capsule__` *was* present and callable,
verified via direct Python import, not just `strings`. `uv cache clean
mal-toolbox` followed by a fresh `uv run maturin develop --uv` still
reproduced the broken install afterward, byte-for-byte identical
(`md5sum` matched the pre-clean broken build), which rules out a simple
"stale cache entry" explanation and points at something in how `uv`
resolves/builds this specific git dependency inside this project's own
resolution step, not a resolution-cache or git-ref-cache issue (both of
which were independently confirmed correct via `direct_url.json`).
Root cause not identified - not worth the further time, since it's a
local dev-environment quirk rather than a code defect, confirmed three
separate ways above. Worked around for this phase's actual test run by
building the known-good wheel via the from-scratch-clone path and `uv pip
install --reinstall-package mal-toolbox <that wheel>` directly, bypassing
`uv run maturin develop`'s own `mal-toolbox` resolution step entirely.
**Flag for B4+:** don't trust `uv run maturin develop` alone to reflect a
just-pushed mal-toolbox commit in this sandbox - verify with `strings`
(or equivalent) on the actually-installed `.so` first, every time, before
concluding a failing test means new Rust code is wrong.

**B7: viability/pruning port (`viability.rs`) - small deliberate
deviations from `graph_processing.py`.**
- `prune_unviable_and_unnecessary_nodes` checks the necessity entry of
  every node it is about to remove *before* removing any. Python only hits
  that `KeyError` partway through its removal loop (in a debug-log line),
  leaving the graph half-pruned; Rust errors with the graph untouched.
- Nodes to remove are collected in graph iteration order (a `Vec`), not
  a `set`, so removal order is deterministic.
- `make_node_unviable` mutates the map through `&mut` and returns only
  the set of nodes made unviable (Python also returns the dict it
  mutated in place).
- `propagate_viability_from_node` treats a missing child entry as
  `MissingViability` (mirrors Python's `KeyError`). `necessity.rs`'s
  propagation treats it as "changed" instead; that inconsistency predates
  this phase and was left alone.
- `ViabilityError` wraps upstream `GraphError` (from `remove_node`) the
  same way `ModelEffectsError`/`ModelStateError` do, so it is `Debug`
  only, not `Clone`/`Eq` like `NecessityError`.
- Python's `logger.debug`/`logger.error` calls are dropped (the crate
  has no logging dependency, same as every other ported module).

**B7: `DynaMalSimulator.reset()` reads a settings snapshot.** Since B5,
`dyna_reset` takes the `MALSimulatorStaticData` captured in `__init__`.
Before B5 it took `self.sim_settings` at call time, so reassigning
`sim.sim_settings` after construction changed the next episode; now it
doesn't. This is exactly how `MalSimulator.reset` has always behaved
(it used `_static_data` before the port began), so the two classes now
agree. Recorded here because B6's audit didn't mention it.

## 11. Conventions

Standing decisions made by the user that later phases follow without
asking again. Each entry names the phase it was decided in.

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

- **One `Simulator`, in `malsim-core`** (decided at C, 2026-10-10). All reset/step orchestration
  (per-agent runtime state, `SimState`, plain and dyna reset/step) lives
  in a public `malsim_core::Simulator` that takes plain-Rust settings
  structs and returns a core error type. `malsim-pyo3` is only a thin
  wrapper: it parses Python dicts into those core structs and builds
  Python output from core results. New orchestration logic never goes
  into `malsim-pyo3`.
- **Rust scenario types keep the Python names** (decided at C). They live in
  `malsim_core::scenario` (with submodules) and are named
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
  `Simulator::new_dyna(graph, model)`, `reset(&MalSimulatorSettings,
  agents, seed) -> Result<&SimState, _>`, `step(&actions) ->
  Result<StepOutcome, _>` (per-step deltas, like `step_native`) and
  `state() -> Option<&SimState>` for the full accumulated state. Multiple
  entry-point sets are sampled before reset by the caller: Python's
  `get_entry_points`, or a Rust helper on `Scenario`.
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

- **B7: `tests/test_graph_processing.py` keeps its name** although it now
  holds only `test_node_is_blocked` (a `graph_utils.py` test). Renaming
  test files is churn for no behavioral gain; revisit if the file grows.
- **B7: the deleted-module reachability was decided against `examples/`
  and everything in `python/malsim` outside the hot-path modules as live
  roots.** Downstream code importing the deleted internals (e.g.
  `malsim.mal_simulator.attack_surface`) breaks. That is accepted by the
  §11 "delete outright" convention; none of those modules was exported
  from a package `__init__`.
- **B7: the memory growth found while scrutinizing the leak tests is
  logged (§9), not fixed.** It's upstream in mal-toolbox, predates the
  port, and the port reduced it.
- **B7: the user named `mal-toolbox @ main` as the port source.** `main`
  is mal-toolbox's original pure-Python toolbox and doesn't contain the
  pinned `rust-rewrite` commit (`b96258b`) this repo's crates build
  against, so the dependency pins were left unchanged. `main` was only
  used as a reference.
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
