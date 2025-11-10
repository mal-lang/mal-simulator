from gymnasium.spaces import Box, Space, Discrete, MultiDiscrete
from gymnasium import spaces

from maltoolbox.model import Model
import numpy as np
from typing import Any, NamedTuple, Sequence
from numpy.typing import NDArray
from .serialization import LangSerializer
from malsim.mal_simulator import MalSimulator


class Asset(NamedTuple):
    """An asset in the attack graph.

    Attributes:
        type: The type of the asset (AssetType).
        id: The ID of the asset in the model.
    """

    type: NDArray[np.int64]
    id: NDArray[np.int64]


class Step(NamedTuple):
    """A node in the attack graph.

    Attributes:
        type: The type of the step (AssetType, AttackStepName).
        id: The ID of the step in the attack graph.
        logic_class: One of {'and', 'or', 'defense', 'exist', 'notExist'}.
        tags: The first @-tag of the step.
        compromised: Whether the step is observed as compromised.
        observable: Whether the step can be observed to be compromised.
        attempts: The number of times the step has been performed.
            Only relevant for the attacker.
        action_mask: Whether the step can be performed.
    """

    type: NDArray[np.int64]
    id: NDArray[np.int64]
    logic_class: NDArray[np.int64]
    tags: NDArray[np.int64]
    compromised: NDArray[np.bool_]
    observable: NDArray[np.bool_]
    attempts: NDArray[np.int64] | None
    action_mask: NDArray[np.bool_]


class Association(NamedTuple):
    """An association between two assets.

    Attributes:
        type: The type of the association (AssociationType).
    """

    type: NDArray[np.int64]


class LogicGate(NamedTuple):
    """A representation of the logic class for a node in the attack graph.

    Should only be used for nodes with logic class 'and' or 'or'.

    Attributes:
        id: The ID of the node which the logic gate belongs to.
        type: The type of the logic gate (LogicGateType).
    """

    id: NDArray[np.int64]
    type: NDArray[np.int64]


class MALObsInstance(NamedTuple):
    """A MAL observation instance.

    Assets, attack/defense steps, associations and logic gates are different objects that exist in the simulation.
    step2asset, assoc2asset, step2step, logic2step and step2logic are the links between these objects.
    """

    time: np.int64

    assets: Asset
    steps: Step
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
        seed: int | np.random.Generator | None = None,
    ):
        # MAL Features
        self.attack_step_tags = Discrete(
            max(set(lang_serializer.attack_step_tag.values())) + 1
        )

        # TODO: Move this into the lang serializer
        self.logic_gate_type = Discrete(2)  # 0 for AND, 1 for OR

        # Language Features
        self.asset_type = Discrete(max(set(lang_serializer.asset_type.values())) + 1)
        if isinstance(
            lang_serializer.attack_step_type[
                next(iter(lang_serializer.attack_step_type))
            ],
            dict,
        ):
            attack_step_type_ids = lang_serializer.attack_step_type.values()
            self.attack_step_type = Discrete(len(attack_step_type_ids) + 1)
        else:
            self.attack_step_type = Discrete(
                max(set(lang_serializer.attack_step_type.values())) + 1
            )
        self.attack_step_class = Discrete(
            max(set(lang_serializer.attack_step_class.values())) + 1
        )
        self.association_type = Discrete(
            max(set(lang_serializer.association_type.values())) + 1
        )

        # Simulator Features
        self.time = Box(0, np.inf, shape=[], dtype=np.int64)
        self.attack_step_observable = Box(0, 1, shape=[], dtype=np.int8)
        self.attack_step_compromised = Box(0, 1, shape=[], dtype=np.int8)
        self.attack_step_attempts = Box(0, np.inf, shape=[], dtype=np.int64)
        self.attack_step_action_mask = Box(0, 1, shape=[], dtype=np.int8)
        self.attack_step_id = Box(0, np.inf, shape=[], dtype=np.int64)
        self.asset_id = Box(0, np.inf, shape=[], dtype=np.int64)

        super().__init__(None, None, seed)

    @property
    def is_np_flattenable(self) -> bool:
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return False

    def _generate_sample_space(
        self, base_space: None | Box | Discrete, num: int
    ) -> Box | MultiDiscrete | None:
        raise NotImplementedError(
            "Sample space generation is not implemented for MALObs"
        )

    def seed(self, seed: int | None = None) -> int | list[int] | dict[str, int]:
        if seed is None:
            seeds = {
                "self": super().seed(None),
                "attack_step_tags": self.attack_step_tags.seed(None),
                "asset_type": self.asset_type.seed(None),
                "attack_step_type": self.attack_step_type.seed(None),
                "attack_step_class": self.attack_step_class.seed(None),
                "association_type": self.association_type.seed(None),
                "time": self.time.seed(None),
                "attack_step_observable": self.attack_step_observable.seed(None),
                "attack_step_compromised": self.attack_step_compromised.seed(None),
                "attack_step_attempts": self.attack_step_attempts.seed(None),
                "attack_step_action_mask": self.attack_step_action_mask.seed(None),
                "logic_gate_type": self.logic_gate_type.seed(None),
            }
            return seeds  # type: ignore

        elif isinstance(seed, int):
            super_seed = super().seed(seed)
            node_seed = int(self.np_random.integers(np.iinfo(np.int32).max))
            # this is necessary such that after int, the Graph PRNG are equivalent
            # REFERENCE: https://gymnasium.farama.org/_modules/gymnasium/spaces/graph/#Graph
            super().seed(seed)
            seeds = {
                "self": super_seed,
                "attack_step_tags": self.attack_step_tags.seed(node_seed),
                "asset_type": self.asset_type.seed(node_seed),
                "attack_step_type": self.attack_step_type.seed(node_seed),
                "attack_step_class": self.attack_step_class.seed(node_seed),
                "association_type": self.association_type.seed(node_seed),
                "time": self.time.seed(node_seed),
                "attack_step_observable": self.attack_step_observable.seed(node_seed),
                "attack_step_compromised": self.attack_step_compromised.seed(node_seed),
                "attack_step_attempts": self.attack_step_attempts.seed(node_seed),
                "attack_step_action_mask": self.attack_step_action_mask.seed(node_seed),
                "logic_gate_type": self.logic_gate_type.seed(node_seed),
            }
            return seeds  # type: ignore

        else:
            raise TypeError(f"Expects `None` or int, actual type: {type(seed)}")

    def contains(self, x: MALObsInstance) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if isinstance(x, MALObsInstance):
            attack_step_tags_valid = self.attack_step_tags.contains(x.steps.tags[0])
            if (
                isinstance(x.logic_gates, LogicGate)
                and x.step2logic is not None
                and x.logic2step is not None
            ):
                logic_gate_type_valid = all(
                    x.logic_gates.type[i] in self.logic_gate_type
                    for i in range(len(x.logic_gates.type))
                )
            else:
                logic_gate_type_valid = False

            asset_type_valid = all(
                x.assets.type[i] in self.asset_type for i in range(len(x.assets.type))
            )
            attack_step_type_valid = all(
                x.steps.type[i] in self.attack_step_type
                for i in range(len(x.steps.type))
            )
            attack_step_class_valid = all(
                x.steps.logic_class[i] in self.attack_step_class
                for i in range(len(x.steps.logic_class))
            )
            if x.associations is not None:
                association_type_valid = all(
                    x.associations.type[i] in self.association_type
                    for i in range(len(x.associations.type))
                )
            else:
                association_type_valid = True

            time_valid = self.time.contains(x.time)
            attack_step_compromised_valid = all(
                x.steps.compromised[i] in self.attack_step_compromised
                for i in range(len(x.steps.compromised))
            )
            attack_step_observable_valid = all(
                x.steps.observable[i] in self.attack_step_observable
                for i in range(len(x.steps.observable))
            )
            if x.steps.attempts is not None:
                attack_step_attempts_valid = all(
                    x.steps.attempts[i] in self.attack_step_attempts
                    for i in range(len(x.steps.attempts))
                )
            else:
                attack_step_attempts_valid = True
            if x.steps.action_mask is not None:
                attack_step_action_mask_valid = all(
                    x.steps.action_mask[i] in self.attack_step_action_mask
                    for i in range(len(x.steps.action_mask))
                )
            else:
                attack_step_action_mask_valid = True

            feature_valid = (
                time_valid
                and attack_step_compromised_valid
                and attack_step_observable_valid
                and attack_step_attempts_valid
                and attack_step_action_mask_valid
                and asset_type_valid
                and attack_step_type_valid
                and attack_step_class_valid
                and association_type_valid
                and logic_gate_type_valid
                and attack_step_tags_valid
            )

            # Edges
            if (
                isinstance(x.step2asset, np.ndarray)
                and isinstance(x.assoc2asset, np.ndarray | None)
                and isinstance(x.step2step, np.ndarray)
                and isinstance(x.logic2step, np.ndarray | None)
                and isinstance(x.step2logic, np.ndarray | None)
            ):
                if (
                    np.issubdtype(x.step2asset.dtype, np.integer)
                    and (
                        x.assoc2asset is None
                        or np.issubdtype(x.assoc2asset.dtype, np.integer)
                    )
                    and np.issubdtype(x.step2step.dtype, np.integer)
                    and (
                        x.logic2step is None
                        or np.issubdtype(x.logic2step.dtype, np.integer)
                    )
                    and (
                        x.step2logic is None
                        or np.issubdtype(x.step2logic.dtype, np.integer)
                    )
                ):
                    step2asset_valid = (
                        (x.step2asset[0] >= 0)
                        & (x.step2asset[0] < x.steps.type.shape[0])
                    ).all() and (
                        (x.step2asset[1] >= 0)
                        & (x.step2asset[1] < x.assets.type.shape[0])
                    ).all()
                    if x.assoc2asset is not None and isinstance(
                        x.associations, Association
                    ):
                        assoc2asset_valid = (
                            (x.assoc2asset[0] >= 0)
                            & (x.assoc2asset[0] < x.associations.type.shape[0])
                        ).all() and (
                            (x.assoc2asset[1] >= 0)
                            & (x.assoc2asset[1] < x.assets.type.shape[0])
                        ).all()
                    else:
                        assoc2asset_valid = True
                    step2step_valid = (
                        (x.step2step[0] >= 0) & (x.step2step[0] < x.steps.type.shape[0])
                    ).all() and (
                        (x.step2step[1] >= 0) & (x.step2step[1] < x.steps.type.shape[0])
                    ).all()
                    if (
                        isinstance(x.logic_gates, LogicGate)
                        and x.step2logic is not None
                        and x.logic2step is not None
                    ):
                        logic2step_valid = (
                            (x.logic2step[0] >= 0)
                            & (x.logic2step[0] < x.logic_gates.type.shape[0])
                        ).all() and (
                            (x.logic2step[1] >= 0)
                            & (x.logic2step[1] < x.steps.type.shape[0])
                        ).all()
                        step2logic_valid = (
                            (x.step2logic[0] >= 0)
                            & (x.step2logic[0] < x.steps.type.shape[0])
                        ).all() and (
                            (x.step2logic[1] >= 0)
                            & (x.step2logic[1] < x.logic_gates.type.shape[0])
                        ).all()
                    else:
                        logic2step_valid = False
                        step2logic_valid = False

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
            f"attack_step_observable={self.attack_step_observable.shape}, "
            f"attack_step_attempts={self.attack_step_attempts.shape}, "
            f"attack_step_action_mask={self.attack_step_action_mask.shape}, "
            f"asset_type={self.asset_type.shape}, "
            f"attack_step_type={self.attack_step_type.shape}, "
            f"attack_step_class={self.attack_step_class.shape}, "
            f"association_type={self.association_type.shape})"
        )
        repr_str = (
            repr_str[:-1] + f", logic_gate_type={self.logic_gate_type.shape})"
        )
        return repr_str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MALObs):
            return False

        return (
            self.time == other.time
            and self.attack_step_compromised == other.attack_step_compromised
            and self.attack_step_observable == other.attack_step_observable
            and self.attack_step_attempts == other.attack_step_attempts
            and self.attack_step_action_mask == other.attack_step_action_mask
            and self.asset_type == other.asset_type
            and self.attack_step_type == other.attack_step_type
            and self.attack_step_class == other.attack_step_class
            and self.association_type == other.association_type
            and self.logic_gate_type == other.logic_gate_type
        )

    def to_jsonable(self, sample_n: Sequence[MALObsInstance]) -> list[Any]:
        """Convert a batch of samples from this space to a JSONable data type."""
        ret_n = []
        for sample in sample_n:
            ret = {
                "time": sample.time,
                "step_compromised": sample.steps.compromised.tolist(),
                "step_attempts": sample.steps.attempts.tolist() if sample.steps.attempts is not None else None,
                "step_action_mask": sample.steps.action_mask.tolist() if sample.steps.action_mask is not None else None,
                "asset_type": sample.assets.type.tolist(),
                "asset_id": sample.assets.id.tolist(),
                "step_type": sample.steps.type.tolist(),
                "step_tags": sample.steps.tags.tolist(),
                "step_id": sample.steps.id.tolist(),
                "step_class": sample.steps.logic_class.tolist(),
                "step_observable": sample.steps.observable.tolist(),
                "association_type": (
                    sample.associations.type.tolist() if sample.associations else None
                ),
                "logic_gate_id": (
                    sample.logic_gates.id.tolist() if sample.logic_gates else None
                ),
                "logic_gate_type": (
                    sample.logic_gates.type.tolist() if sample.logic_gates else None
                ),
                "step2asset": sample.step2asset.tolist(),
                "assoc2asset": (
                    sample.assoc2asset.tolist() if sample.assoc2asset is not None else None
                ),
                "step2step": sample.step2step.tolist(),
                "logic2step": (
                    sample.logic2step.tolist() if sample.logic2step is not None else None
                ),
                "step2logic": (
                    sample.step2logic.tolist() if sample.step2logic is not None else None
                ),
            }
            ret_n.append(ret)
        return ret_n

    def from_jsonable(self, sample_n: Sequence[Any]) -> list[MALObsInstance]:
        """Convert a JSONable data type to a batch of samples from this space."""
        ret_n: list[MALObsInstance] = []
        for sample in sample_n:
            step = Step(
                type=np.array(
                    sample["step_type"], dtype=self.attack_step_type.dtype
                ),
                id=np.array(sample["step_id"], dtype=self.attack_step_id.dtype),
                logic_class=np.array(
                    sample["step_class"], dtype=self.attack_step_class.dtype
                ),
                tags=np.array(
                    sample["step_tags"], dtype=self.attack_step_tags.dtype
                ),
                compromised=np.array(
                    sample["step_compromised"],
                    dtype=self.attack_step_compromised.dtype,
                ),
                observable=np.array(
                    sample["step_observable"],
                    dtype=self.attack_step_observable.dtype,
                ),
                attempts=(
                    np.array(sample["step_attempts"], dtype=self.attack_step_attempts.dtype)
                    if sample["step_attempts"] is not None else None
                ),
                action_mask=(
                    np.array(sample["step_action_mask"], dtype=self.attack_step_action_mask.dtype)
                ),
            )
            assets = Asset(
                type=np.array(sample["asset_type"], dtype=self.asset_type.dtype),
                id=np.array(sample["asset_id"], dtype=self.asset_id.dtype),
            )
            associations = (
                Association(
                    type=np.array(
                        sample["association_type"], dtype=self.association_type.dtype
                    ),
                )
                if sample["association_type"]
                else None
            )
            logic_gates = (
                LogicGate(
                    id=np.array(sample["logic_gate_id"], dtype=self.attack_step_id.dtype),
                    type=np.array(
                        sample["logic_gate_type"], dtype=self.logic_gate_type.dtype
                    ),
                )
                if sample["logic_gate_type"] and self.logic_gate_type
                else None
            )
            ret = MALObsInstance(
                time=np.int64(sample["time"]),
                steps=step,
                assets=assets,
                associations=associations,
                logic_gates=logic_gates,
                step2asset=np.array(sample["step2asset"], dtype=np.int64),
                assoc2asset=(
                    np.array(sample["assoc2asset"], dtype=np.int64)
                    if sample["assoc2asset"]
                    else None
                ),
                step2step=np.array(sample["step2step"], dtype=np.int64),
                logic2step=(
                    np.array(sample["logic2step"], dtype=np.int64)
                    if sample["logic2step"]
                    else None
                ),
                step2logic=(
                    np.array(sample["step2logic"], dtype=np.int64)
                    if sample["step2logic"]
                    else None
                ),
            )
            ret_n.append(ret)
        return ret_n

    def sample(
        self, mask: Any | None = ..., probability: Any | None = ...
    ) -> MALObsInstance:
        """Sample a random MAL observation."""
        raise NotImplementedError("Sampling is not implemented for MALObs")

class MALAttackerObs(MALObs):

    def contains(self, x: MALObsInstance) -> bool:
        if not super().contains(x):
            return False
        # Attacker observations must include attempts
        return x.steps.attempts is not None

class MALDefenderObs(MALObs):

    def contains(self, x: MALObsInstance) -> bool:
        # Defender observations must NOT include attempts
        if x.steps.attempts is not None:
            return False
        return super().contains(x)

class MALObsAttackStepSpace(Discrete):
    """A space over the actionable attack steps for the attacker.
    
    The space is dependent on a specific indexing of the attack steps in the MALObsInstance.
    """
    def __init__(
        self, sim: MalSimulator, seed: int | np.random.Generator | None = None
    ):
        # NOTE: The corresponding MALObsInstance should have the same sorting
        # of the actionable steps as the action space
        actionable_attack_steps = list(sorted({
            node for node in sim.attack_graph.nodes.values()
            if node.type in ("and", "or")
        }, key=lambda step: step.id))
        super().__init__(n=len(actionable_attack_steps), seed=seed)

        self.actionability = np.array([
            sim.node_is_actionable(step)
            for step in actionable_attack_steps
        ])

    def sample(
        self,
        mask: Any | None = None,
        probability: Any | None = None,
    ) -> np.int64:
        if mask is not None and isinstance(mask, np.ndarray):
            mask = mask.astype(np.int8)
        if mask is not None and isinstance(mask, np.ndarray) and mask.shape[0] < self.n:
            mask = np.concatenate([mask, np.zeros(self.n - mask.shape[0], dtype=np.int8)])
        # TODO: Decide if actionability should be used for attacker as well
        return super().sample(mask=mask, probability=probability)

class MALObsDefenseStepSpace(Discrete):
    """A space over the actionable defense steps for the defender.
    
    The space is dependent on a specific indexing of the defense steps in the MALObsInstance.
    """
    def __init__(
        self, sim: MalSimulator, seed: int | np.random.Generator | None = None
    ):
        # NOTE: The corresponding MALObsInstance should have the same sorting
        # of the actionable steps as the action space
        actionable_defense_steps = list(sorted({
            node for node in sim.attack_graph.nodes.values()
            if node.type == "defense"
        }, key=lambda step: step.id))
        super().__init__(n=len(actionable_defense_steps), seed=seed)

        self.actionability = np.array([
            sim.node_is_actionable(step)
            for step in actionable_defense_steps
        ])

    def sample(
        self,
        mask: Any | None = None,
        probability: Any | None = None,
    ) -> np.int64:
        base_mask = self.actionability.astype(np.int8)
        if mask is not None:
            if isinstance(mask, np.ndarray) and mask.shape[0] < self.n:
                mask = np.concatenate([mask, np.zeros(self.n - mask.shape[0], dtype=np.int8)])
            elif isinstance(mask, np.ndarray) and mask.shape[0] > self.n:
                mask = mask[:self.n]
            base_mask = base_mask & mask
        return super().sample(mask=base_mask, probability=probability)

    def __repr__(self) -> str:
        return f"MALObsDefenderActionSpace(n={self.n})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MALObsDefenseStepSpace):
            return False
        return self.n == other.n and bool(np.all(self.actionability == other.actionability))

    def contains(self, x: Any) -> bool:
        return bool(super().contains(x)) and bool(self.actionability[int(x)])

class AssetThenAction(spaces.Tuple):
    """A space over the (asset, lang action for asset).
    
    The space is dependent on a specific indexing of the assets in the MALObsInstance.
    """
    def __init__(
        self, model: Model, lang_serializer: LangSerializer, seed: int | np.random.Generator | None = None
    ):
        if not lang_serializer.split_attack_step_types:
            raise ValueError("lang_serializer must be split_attack_step_types=True to use MALObsAttackerAssetAction")

        self.asset = Discrete(len(model.assets))
        # NOTE: Serializer indicies should be contiguous
        self.action = Discrete(max(lang_serializer.attack_step_type.values()) + 1) # +1 beacuse they are indicies

        super().__init__((self.asset, self.action), seed=seed)

    def _action_mask_for_asset(self, asset_idx: int, obs: MALObsInstance) -> np.ndarray:
        asset_step_indicies = obs.step2asset[0, np.where(obs.step2asset[1] == asset_idx)[0]]
        asset_step_types = obs.steps.type[asset_step_indicies]
        action_mask = np.zeros(self.action.n, dtype=np.int8)
        action_mask[asset_step_types] = True
        return action_mask

    def mask(self, obs: MALObsInstance) -> tuple[np.ndarray, np.ndarray]:
        asset_mask = np.zeros(self.asset.n, dtype=np.int8)
        asset_mask[:obs.assets.type.shape[0]] = True
        action_mask = np.zeros((self.asset.n, self.action.n), dtype=np.int8)
        for asset_idx in range(obs.assets.type.shape[0]):
            action_mask[asset_idx] = self._action_mask_for_asset(asset_idx, obs)
        return (asset_mask, action_mask)

    def sample(self, mask: tuple[Any | None, ...] | None = None, probability: tuple[Any | None, ...] | None = None) -> tuple[Any, ...]:
        if mask is not None:
            if not isinstance(mask, tuple) or not len(mask) == 2 or not isinstance(mask[0], np.ndarray) or not isinstance(mask[1], np.ndarray):
                raise ValueError("mask must be a tuple of length 2 with numpy arrays or None")

            asset = self.asset.sample(mask=mask[0])
            action = self.action.sample(mask=mask[1][asset])
            return (asset, action)

        return super().sample()

class ActionThenAsset(spaces.Tuple):
    """A space over the (lang action, asset to perform action on).
    
    The space is dependent on a specific indexing of the assets in the MALObsInstance.
    """
    def __init__(
        self, model: Model, lang_serializer: LangSerializer, seed: int | np.random.Generator | None = None
    ):
        if not lang_serializer.split_attack_step_types:
            raise ValueError("lang_serializer must be split_attack_step_types=True to use MALObsAttackerAssetAction")

        self.asset = Discrete(len(model.assets))
        # NOTE: Serializer indicies should be contiguous
        self.action = Discrete(max(lang_serializer.attack_step_type.values()))

        super().__init__((self.asset, self.action), seed=seed)

    def _asset_mask_for_action(self, action_type: int, obs: MALObsInstance) -> np.ndarray:
        asset_mask = np.zeros(self.asset.n, dtype=np.int8)
        actionable_step_indicies = np.where((obs.steps.type == action_type) & obs.steps.action_mask)[0]
        asset_indicies = obs.step2asset[1, np.where(np.isin(obs.step2asset[0], actionable_step_indicies))[0]]
        asset_mask[asset_indicies] = True
        return asset_mask

    def mask(self, obs: MALObsInstance) -> tuple[np.ndarray, np.ndarray]:
        action_mask = np.zeros(self.action.n, dtype=np.int8)
        actionable_step_types = np.unique(obs.steps.type[obs.steps.action_mask])
        action_mask[actionable_step_types] = True
        asset_mask = np.zeros((self.action.n, self.asset.n), dtype=np.int8)
        for action_type in actionable_step_types:
            asset_mask[action_type] = self._asset_mask_for_action(action_type, obs)
        return (action_mask, asset_mask)

    def sample(self, mask: tuple[Any | None, ...] | None = None, probability: tuple[Any | None, ...] | None = None) -> tuple[Any, ...]:
        if mask is not None:
            if not isinstance(mask, tuple) or not len(mask) == 2 or not isinstance(mask[0], np.ndarray) or not isinstance(mask[1], np.ndarray):
                raise ValueError("mask must be a tuple of length 2 with numpy arrays or None")

            action = self.action.sample(mask=mask[0])
            asset = self.asset.sample(mask=mask[1][action])
            return (action, asset)

        return super().sample()