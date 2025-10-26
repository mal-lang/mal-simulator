from re import S
from gymnasium.spaces import Box, MultiBinary, Space, Discrete, MultiDiscrete
import numpy as np
from typing import NamedTuple, Sequence
from numpy.typing import NDArray
from maltoolbox.attackgraph import AttackGraph
from maltoolbox.language import LanguageGraphAssociation
from malsim.envs.serialization import LangSerializer
from malsim.mal_simulator import MalSimAgentState, MalSimAttackerState


class Assets(NamedTuple):
    type: NDArray[np.int64]
    id: NDArray[np.int64]


class AttackStep(NamedTuple):
    type: NDArray[np.int64]
    id: NDArray[np.int64]
    logic_class: NDArray[np.int64]
    tags: NDArray[np.int64]
    compromised: NDArray[np.bool]
    attempts: NDArray[np.int64]
    traversable: NDArray[np.bool]


class Association(NamedTuple):
    type: NDArray[np.int64]


class LogicGate(NamedTuple):
    type: NDArray[np.int64]


class MALObsInstance(NamedTuple):
    """A MAL observation instance.

    Assets, attack steps, associations and logic gates are disjunct sets of nodes.
    """

    time: np.int64

    assets: Assets
    attack_steps: AttackStep
    associations: Association | None
    logic_gates: LogicGate | None

    step2asset: NDArray[np.int64]
    assoc2asset: NDArray[np.int64] | None
    step2step: NDArray[np.int64]
    logic2step: NDArray[np.int64] | None
    step2logic: NDArray[np.int64] | None


class MALObs(Space[MALObsInstance]):

    def __init__(
        self,
        lang_serializer: LangSerializer,
        use_logic_gates: bool,
        seed: int | np.random.Generator | None = None,
    ):
        self.use_logic_gates = use_logic_gates
        # MAL Features
        self.attack_step_tags = Discrete(
            len(set(lang_serializer.attack_step_tag.values()))
        )
        if use_logic_gates:
            self.logic_gate_type = Discrete(2)  # 0 for AND, 1 for OR
        else:
            self.logic_gate_type = None

        # Language Features
        self.asset_type = Discrete(len(set(lang_serializer.asset_type.values())))
        if isinstance(
            lang_serializer.attack_step_type[
                next(iter(lang_serializer.attack_step_type))
            ],
            dict,
        ):
            attack_step_type_ids = {
                lang_serializer.attack_step_type[asset_type][attack_step_name]
                for asset_type in lang_serializer.attack_step_type
                for attack_step_name in lang_serializer.attack_step_type[asset_type]
            }
            self.attack_step_type = Discrete(len(attack_step_type_ids))
        else:
            self.attack_step_type = Discrete(
                len(set(lang_serializer.attack_step_type.values()))
            )
        self.attack_step_class = Discrete(
            len(set(lang_serializer.attack_step_class.values()))
        )
        self.association_type = Discrete(
            len(set(lang_serializer.association_type.values()))
        )

        # Simulator Features
        self.time = Box(0, np.inf, shape=[], dtype=np.int64)
        self.attack_step_compromised = Box(0, 1, shape=[], dtype=np.bool)
        self.attack_step_attempts = Box(0, np.inf, shape=[], dtype=np.int64)
        self.attack_step_traversable = Box(0, 1, shape=[], dtype=np.bool)
        self.attack_step_id = Box(0, np.inf, shape=[], dtype=np.int64)
        self.asset_id = Box(0, np.inf, shape=[], dtype=np.int64)

        super().__init__(None, None, seed)

    @property
    def is_np_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return False

    def _generate_sample_space(
        self, base_space: None | Box | Discrete, num: int
    ) -> Box | MultiDiscrete | None:
        raise NotImplementedError(
            "Sample space generation is not implemented for MALObs"
        )

    def seed(self, seed: int | None = None):

        if seed is None:
            return {
                "self": super().seed(None),
                "attack_step_tags": self.attack_step_tags.seed(None),
                "logic_gate_type": self.logic_gate_type.seed(None),
                "asset_type": self.asset_type.seed(None),
                "attack_step_type": self.attack_step_type.seed(None),
                "attack_step_class": self.attack_step_class.seed(None),
                "association_type": self.association_type.seed(None),
                "time": self.time.seed(None),
                "attack_step_compromised": self.attack_step_compromised.seed(None),
                "attack_step_attempts": self.attack_step_attempts.seed(None),
                "attack_step_traversable": self.attack_step_traversable.seed(None),
            }

        elif isinstance(seed, int):
            super_seed = super().seed(seed)
            node_seed = int(self.np_random.integers(np.iinfo(np.int32).max))
            # this is necessary such that after int or list/tuple seeding, the Graph PRNG are equivalent
            # REFERENCE: https://gymnasium.farama.org/_modules/gymnasium/spaces/graph/#Graph
            super().seed(seed)
            return {
                "self": super_seed,
                "attack_step_tags": self.attack_step_tags.seed(node_seed),
                "logic_gate_type": self.logic_gate_type.seed(node_seed),
                "asset_type": self.asset_type.seed(node_seed),
                "attack_step_type": self.attack_step_type.seed(node_seed),
                "attack_step_class": self.attack_step_class.seed(node_seed),
                "association_type": self.association_type.seed(node_seed),
                "time": self.time.seed(node_seed),
                "attack_step_compromised": self.attack_step_compromised.seed(node_seed),
                "attack_step_attempts": self.attack_step_attempts.seed(node_seed),
                "attack_step_traversable": self.attack_step_traversable.seed(node_seed),
            }

        else:
            raise TypeError(f"Expects `None` or int, actual type: {type(seed)}")

    def contains(self, x: MALObsInstance) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if isinstance(x, MALObsInstance):
            attack_step_tags_valid = self.attack_step_tags.contains(
                x.attack_steps.tags[0]
            )
            if self.use_logic_gates:
                logic_gate_type_valid = self.logic_gate_type.contains(x.logic_gates.type[0])
            else:
                logic_gate_type_valid = x.logic_gates is None and x.logic2step is None and x.step2logic is None

            asset_type_valid = self.asset_type.contains(x.assets.type[0])
            attack_step_type_valid = self.attack_step_type.contains(
                x.attack_steps.type[0]
            )
            attack_step_class_valid = self.attack_step_class.contains(
                x.attack_steps.logic_class[0]
            )
            if x.associations is not None:
                association_type_valid = self.association_type.contains(
                    x.associations.type[0]
                )
            else:
                association_type_valid = True

            time_valid = self.time.contains(x.time)
            attack_step_compromised_valid = self.attack_step_compromised.contains(
                x.attack_steps.compromised[0]
            )
            attack_step_attempts_valid = self.attack_step_attempts.contains(
                x.attack_steps.attempts[0]
            )
            attack_step_traversable_valid = self.attack_step_traversable.contains(
                x.attack_steps.traversable[0]
            )

            feature_valid = (
                time_valid
                and attack_step_compromised_valid
                and attack_step_attempts_valid
                and attack_step_traversable_valid
                and asset_type_valid
                and attack_step_type_valid
                and attack_step_class_valid
                and association_type_valid
                and logic_gate_type_valid
                and attack_step_tags_valid
            )

            # Edges
            if (isinstance(x.step2asset, np.ndarray)
                and isinstance(x.assoc2asset, np.ndarray | None)
                and isinstance(x.step2step, np.ndarray)
                and isinstance(x.logic2step, np.ndarray | None)
                and isinstance(x.step2logic, np.ndarray | None)
            ):
                if (
                    np.issubdtype(x.step2asset.dtype, np.integer)
                    and (x.assoc2asset is None or np.issubdtype(x.assoc2asset.dtype, np.integer))
                    and np.issubdtype(x.step2step.dtype, np.integer)
                    and (not self.use_logic_gates or np.issubdtype(x.logic2step.dtype, np.integer))
                    and (not self.use_logic_gates or np.issubdtype(x.step2logic.dtype, np.integer))
                ):
                    step2asset_valid = (
                        ((x.step2asset[0] >= 0) & (x.step2asset[0] < x.attack_steps.type.shape[0])).all()
                        and ((x.step2asset[1] >= 0) & (x.step2asset[1] < x.assets.type.shape[0])).all()
                    )
                    if x.assoc2asset is not None:
                        assoc2asset_valid = (
                            ((x.assoc2asset[0] >= 0) & (x.assoc2asset[0] < x.associations.type.shape[0])).all()
                            and ((x.assoc2asset[1] >= 0) & (x.assoc2asset[1] < x.assets.type.shape[0])).all()
                        )
                    else:
                        assoc2asset_valid = True
                    step2step_valid = (
                        ((x.step2step[0] >= 0) & (x.step2step[0] < x.attack_steps.type.shape[0])).all()
                        and ((x.step2step[1] >= 0) & (x.step2step[1] < x.attack_steps.type.shape[0])).all()
                    )
                    if self.use_logic_gates:
                        logic2step_valid = (
                            ((x.logic2step[0] >= 0) & (x.logic2step[0] < x.logic_gates.type.shape[0])).all()
                            and ((x.logic2step[1] >= 0) & (x.logic2step[1] < x.attack_steps.type.shape[0])).all()
                        )
                        step2logic_valid = (
                            ((x.step2logic[0] >= 0) & (x.step2logic[0] < x.attack_steps.type.shape[0])).all()
                            and ((x.step2logic[1] >= 0) & (x.step2logic[1] < x.logic_gates.type.shape[0])).all()
                        )
                    else:
                        logic2step_valid = True
                        step2logic_valid = True

                    edge_valid = (
                        step2asset_valid
                        and assoc2asset_valid
                        and step2step_valid
                        and logic2step_valid
                        and step2logic_valid
                    )

                    return feature_valid and edge_valid

        return False

    def __repr__(self) -> str:
        repr_str = (
            f"MALObs(time={self.time.shape}, "
            f"attack_step_compromised={self.attack_step_compromised.shape}, "
            f"attack_step_attempts={self.attack_step_attempts.shape}, "
            f"attack_step_traversable={self.attack_step_traversable.shape}, "
            f"asset_type={self.asset_type.shape}, "
            f"attack_step_type={self.attack_step_type.shape}, "
            f"attack_step_class={self.attack_step_class.shape}, "
            f"association_type={self.association_type.shape})"
        )
        if self.use_logic_gates:
            repr_str = repr_str[:-1] + f", logic_gate_type={self.logic_gate_type.shape})"
        return repr_str


    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MALObs):
            return False

        return (
            self.time == other.time
            and self.attack_step_compromised == other.attack_step_compromised
            and self.attack_step_attempts == other.attack_step_attempts
            and self.attack_step_traversable == other.attack_step_traversable
            and self.asset_type == other.asset_type
            and self.attack_step_type == other.attack_step_type
            and self.attack_step_class == other.attack_step_class
            and self.association_type == other.association_type
            and self.logic_gate_type == other.logic_gate_type
        )

    def to_jsonable(self, sample_n: Sequence[MALObsInstance]) -> list[dict[str, list[int | float]]]:
        """Convert a batch of samples from this space to a JSONable data type."""
        ret_n = []
        for sample in sample_n:
            ret = {
                "time": sample.time,
                "attack_step_compromised": sample.attack_steps.compromised.tolist(),
                "attack_step_attempts": sample.attack_steps.attempts.tolist(),
                "attack_step_traversable": sample.attack_steps.traversable.tolist(),
                "asset_type": sample.assets.type.tolist(),
                "attack_step_type": sample.attack_steps.type.tolist(),
                "attack_step_class": sample.attack_steps.logic_class.tolist(),
                "association_type": sample.associations.type.tolist(),
                "logic_gate_type": sample.logic_gates.type.tolist(),
                "step2asset": sample.step2asset.tolist(),
                "assoc2asset": sample.assoc2asset.tolist(),
                "step2step": sample.step2step.tolist(),
                "logic2step": sample.logic2step.tolist(),
                "step2logic": sample.step2logic.tolist(),
            }
            ret_n.append(ret)
        return ret_n

    def from_jsonable(self, sample_n: Sequence[dict[str, list[list[int] | list[float]]]]) -> list[MALObsInstance]:
        """Convert a JSONable data type to a batch of samples from this space."""
        ret_n: list[MALObsInstance] = []
        for sample in sample_n:
            attack_step = AttackStep(
                type=np.array(sample["attack_step_type"], dtype=self.attack_step_type.dtype),
                logic_class=np.array(sample["attack_step_class"], dtype=self.attack_step_class.dtype),
                tags=np.array(sample["attack_step_tags"], dtype=self.attack_step_tags.dtype),
                compromised=np.array(sample["attack_step_compromised"], dtype=self.attack_step_compromised.dtype),
                attempts=np.array(sample["attack_step_attempts"], dtype=self.attack_step_attempts.dtype),
                traversable=np.array(sample["attack_step_traversable"], dtype=self.attack_step_traversable.dtype),
            )
            assets = Assets(type=np.array(sample["asset_type"], dtype=self.asset_type.dtype))
            associations = Association(
                type=np.array(sample["association_type"], dtype=self.association_type.dtype),
            )
            logic_gates = LogicGate(
                type=np.array(sample["logic_gate_type"], dtype=self.logic_gate_type.dtype),
            )
            ret = MALObsInstance(
                time=int(sample["time"]),
                attack_steps=attack_step,
                assets=assets,
                associations=associations,
                logic_gates=logic_gates,
                step2asset=np.array(sample["step2asset"], dtype=np.int64),
                assoc2asset=np.array(sample["assoc2asset"], dtype=np.int64),
                step2step=np.array(sample["step2step"], dtype=np.int64),
                logic2step=np.array(sample["logic2step"], dtype=np.int64),
                step2logic=np.array(sample["step2logic"], dtype=np.int64),
            )
            ret_n.append(ret)
        return ret_n


    def sample(self, num_assets: int) -> MALObsInstance:
        """Sample a random MAL observation."""
        raise NotImplementedError("Sampling is not implemented for MALObs")


def attacker_state2graph(state: MalSimAttackerState, lang_serializer: LangSerializer, use_logic_gates: bool) -> MALObsInstance:
    visible_assets = (
        {node.model_asset for node in state.performed_nodes} 
        | {node.model_asset for node in state.action_surface}
    )
    assets = Assets(
        type=np.array([lang_serializer.asset_type[asset.type] for asset in visible_assets]),
        id=np.array([asset.id for asset in visible_assets]),
    )

    visible_attack_steps = {
        node for node in state.sim.attack_graph.nodes.values() if node.model_asset in visible_assets
    }
    if lang_serializer.split_attack_step_types:
        attack_step_type = np.array([lang_serializer.attack_step_type[node.model_asset.type][node.name] for node in visible_attack_steps])
    else:
        attack_step_type = np.array([lang_serializer.attack_step_type[node.name] for node in visible_attack_steps])
    attack_steps = AttackStep(
        type=attack_step_type,
        id=np.array([node.id for node in visible_attack_steps]),
        logic_class=np.array([lang_serializer.attack_step_class[node.type] for node in visible_attack_steps]),
        tags=np.array([lang_serializer.attack_step_tag[node.tags[0] if len(node.tags) > 0 else None] for node in visible_attack_steps]),
        compromised=np.array([state.sim.node_is_compromised(node) for node in visible_attack_steps]),
        attempts=np.array([state.num_attempts.get(node, 0) for node in visible_attack_steps]),
        traversable=np.array([state.sim.node_is_traversable(state.performed_nodes, node) for node in visible_attack_steps]),
    )
    step2asset = {(node.id, node.model_asset.id) for node in visible_attack_steps}
    step2step = {(node.id, child.id) for node in visible_attack_steps for child in node.children if child in visible_attack_steps}

    if len(visible_assets) > 1:
        associations: list[tuple[LanguageGraphAssociation, int, int]] = []
        assoc2asset = set()
        for asset in visible_assets:
            for fieldname, other_assets in asset.associated_assets.items():
                assoc = asset.lg_asset.associations[fieldname]
                for other_asset in filter(lambda asset: asset in visible_assets, other_assets):
                    asset1_id, asset2_id = sorted([asset.id, other_asset.id])
                    if (assoc, asset1_id, asset2_id) not in associations:
                        associations.append((assoc, asset1_id, asset2_id))
                        assoc_idx = associations.index((assoc, asset1_id, asset2_id))
                        assoc2asset.add((assoc_idx, asset1_id))
                        assoc2asset.add((assoc_idx, asset2_id))
        associations = Association(
            type=np.array([lang_serializer.association_type[assoc.name] for (assoc, _, _) in associations]),
        )
        assoc2asset = np.array(list(zip(*assoc2asset)))
    else:
        associations = None
        assoc2asset = None

    if use_logic_gates:
        visible_and_or_steps = [node for node in visible_attack_steps if node.type in ('and', 'or')]
        logic_gates = LogicGate(type=np.array([(0 if node.type == 'and' else 1) for node in visible_and_or_steps]))
        step2logic = set()
        logic2step = set()
        for logic_gate_id, node in enumerate(visible_and_or_steps):
            logic2step.add((logic_gate_id, node.id))
            for child in node.children:
                step2logic.add((child.id, logic_gate_id))
        logic2step = np.array(list(zip(*logic2step)))
        step2logic = np.array(list(zip(*step2logic)))
    else:
        logic_gates = None
        logic2step = None
        step2logic = None

    return MALObsInstance(
        time=state.sim.cur_iter,
        assets=assets,
        attack_steps=attack_steps,
        associations=associations,
        logic_gates=logic_gates,
        step2asset=np.array(list(zip(*step2asset))),
        step2step=np.array(list(zip(*step2step))),
        assoc2asset=assoc2asset,
        logic2step=logic2step,
        step2logic=step2logic,
    )

class AttackGraphNodeSpace(Discrete):
    def __init__(self, attack_graph: AttackGraph, seed: int | np.random.Generator | None = None):
        super().__init__(n=max(attack_graph.nodes.keys()), seed=seed)