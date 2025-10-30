from gymnasium.spaces import Box, Space, Discrete, MultiDiscrete

import numpy as np
from typing import Any, NamedTuple, Sequence
from numpy.typing import NDArray
from .serialization import LangSerializer

class Asset(NamedTuple):
    type: NDArray[np.int64]
    id: NDArray[np.int64]


class AttackStep(NamedTuple):
    type: NDArray[np.int64]
    id: NDArray[np.int64]
    logic_class: NDArray[np.int64]
    tags: NDArray[np.int64]
    compromised: NDArray[np.bool]
    attempts: NDArray[np.int64] | None
    traversable: NDArray[np.bool] | None


class Association(NamedTuple):
    type: NDArray[np.int64]


class LogicGate(NamedTuple):
    type: NDArray[np.int64]


class MALObsInstance(NamedTuple):
    """A MAL observation instance.

    Assets, attack steps, associations and logic gates are disjunct sets of nodes.
    """

    time: np.int64

    assets: Asset
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
            max(set(lang_serializer.attack_step_tag.values())) + 1
        )

        self.logic_gate_type: Discrete | None = None
        if use_logic_gates:
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
        self.attack_step_compromised = Box(0, 1, shape=[], dtype=np.int8)
        self.attack_step_attempts = Box(0, np.inf, shape=[], dtype=np.int64)
        self.attack_step_traversable = Box(0, 1, shape=[], dtype=np.int8)
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
                "attack_step_compromised": self.attack_step_compromised.seed(None),
                "attack_step_attempts": self.attack_step_attempts.seed(None),
                "attack_step_traversable": self.attack_step_traversable.seed(None),
            }
            if self.logic_gate_type:
                seeds["logic_gate_type"] = self.logic_gate_type.seed(None)
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
                "attack_step_compromised": self.attack_step_compromised.seed(node_seed),
                "attack_step_attempts": self.attack_step_attempts.seed(node_seed),
                "attack_step_traversable": self.attack_step_traversable.seed(node_seed),
            }
            if self.logic_gate_type:
                seeds["logic_gate_type"] = self.logic_gate_type.seed(node_seed)
            return seeds  # type: ignore

        else:
            raise TypeError(f"Expects `None` or int, actual type: {type(seed)}")

    def contains(self, x: MALObsInstance) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if isinstance(x, MALObsInstance):
            attack_step_tags_valid = self.attack_step_tags.contains(
                x.attack_steps.tags[0]
            )
            if self.use_logic_gates and isinstance(x.logic_gates, LogicGate) and self.logic_gate_type and x.step2logic is not None and x.logic2step is not None:
                logic_gate_type_valid = all(
                    x.logic_gates.type[i] in self.logic_gate_type
                    for i in range(len(x.logic_gates.type))
                )
            else:
                logic_gate_type_valid = (
                    x.logic_gates is None and x.logic2step is None and x.step2logic is None
                )

            asset_type_valid = all(
                x.assets.type[i] in self.asset_type for i in range(len(x.assets.type))
            )
            attack_step_type_valid = all(
                x.attack_steps.type[i] in self.attack_step_type
                for i in range(len(x.attack_steps.type))
            )
            attack_step_class_valid = all(
                x.attack_steps.logic_class[i] in self.attack_step_class
                for i in range(len(x.attack_steps.logic_class))
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
                x.attack_steps.compromised[i] in self.attack_step_compromised
                for i in range(len(x.attack_steps.compromised))
            )
            if x.attack_steps.attempts is not None:
                attack_step_attempts_valid = all(
                    x.attack_steps.attempts[i] in self.attack_step_attempts
                    for i in range(len(x.attack_steps.attempts))
                )
            else:
                attack_step_attempts_valid = True
            if x.attack_steps.traversable is not None:
                attack_step_traversable_valid = all(
                    x.attack_steps.traversable[i] in self.attack_step_traversable
                    for i in range(len(x.attack_steps.traversable))
                )
            else:
                attack_step_traversable_valid = True

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
                    and (x.logic2step is None or np.issubdtype(x.logic2step.dtype, np.integer))
                    and (x.step2logic is None or np.issubdtype(x.step2logic.dtype, np.integer))
                ):
                    step2asset_valid = (
                        ((x.step2asset[0] >= 0) & (x.step2asset[0] < x.attack_steps.type.shape[0])).all()
                        and ((x.step2asset[1] >= 0) & (x.step2asset[1] < x.assets.type.shape[0])).all()
                    )
                    if x.assoc2asset is not None and isinstance(x.associations, Association):
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
                    if self.use_logic_gates and isinstance(x.logic_gates, LogicGate) and self.logic_gate_type and x.step2logic is not None and x.logic2step is not None:
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
        if self.use_logic_gates and self.logic_gate_type:
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

    def to_jsonable(self, sample_n: Sequence[MALObsInstance]) -> list[dict[str, Any]]:
        """Convert a batch of samples from this space to a JSONable data type."""
        ret_n = []
        for sample in sample_n:
            ret = {
                "time": sample.time,
                "attack_step_compromised": sample.attack_steps.compromised.tolist(),
                "attack_step_attempts": sample.attack_steps.attempts.tolist(),
                "attack_step_traversable": sample.attack_steps.traversable.tolist(),
                "asset_type": sample.assets.type.tolist(),
                "asset_id": sample.assets.id.tolist(),
                "attack_step_type": sample.attack_steps.type.tolist(),
                "attack_step_tags": sample.attack_steps.tags.tolist(),
                "attack_step_id": sample.attack_steps.id.tolist(),
                "attack_step_class": sample.attack_steps.logic_class.tolist(),
                "association_type": (sample.associations.type.tolist() if sample.associations else None),
                "logic_gate_type": (sample.logic_gates.type.tolist() if sample.logic_gates else None),
                "step2asset": sample.step2asset.tolist(),
                "assoc2asset": (sample.assoc2asset.tolist() if sample.assoc2asset else None),
                "step2step": sample.step2step.tolist(),
                "logic2step": (sample.logic2step.tolist() if sample.logic2step else None),
                "step2logic": (sample.step2logic.tolist() if sample.step2logic else None),
            }
            ret_n.append(ret)
        return ret_n

    def from_jsonable(self, sample_n: Sequence[dict[str, Any]]) -> list[MALObsInstance]:
        """Convert a JSONable data type to a batch of samples from this space."""
        ret_n: list[MALObsInstance] = []
        for sample in sample_n:
            attack_step = AttackStep(
                type=np.array(
                    sample["attack_step_type"], dtype=self.attack_step_type.dtype
                ),
                id=np.array(
                    sample["attack_step_id"], dtype=self.attack_step_id.dtype
                ),
                logic_class=np.array(
                    sample["attack_step_class"], dtype=self.attack_step_class.dtype
                ),
                tags=np.array(
                    sample["attack_step_tags"], dtype=self.attack_step_tags.dtype
                ),
                compromised=np.array(
                    sample["attack_step_compromised"], dtype=self.attack_step_compromised.dtype
                ),
                attempts=np.array(
                    sample["attack_step_attempts"], dtype=self.attack_step_attempts.dtype
                ),
                traversable=np.array(
                    sample["attack_step_traversable"], dtype=self.attack_step_traversable.dtype
                ),
            )
            assets = Asset(
                type=np.array(sample["asset_type"], dtype=self.asset_type.dtype),
                id=np.array(sample["asset_id"], dtype=self.asset_id.dtype),
            )
            associations = Association(
                type=np.array(sample["association_type"], dtype=self.association_type.dtype),
            ) if sample["association_type"] else None
            logic_gates = LogicGate(
                type=np.array(sample["logic_gate_type"], dtype=self.logic_gate_type.dtype),
            ) if sample["logic_gate_type"] and self.logic_gate_type else None
            ret = MALObsInstance(
                time=np.int64(sample["time"]),
                attack_steps=attack_step,
                assets=assets,
                associations=associations,
                logic_gates=logic_gates,
                step2asset=np.array(sample["step2asset"], dtype=np.int64),
                assoc2asset=np.array(sample["assoc2asset"], dtype=np.int64) if sample["assoc2asset"] else None,
                step2step=np.array(sample["step2step"], dtype=np.int64),
                logic2step=np.array(sample["logic2step"], dtype=np.int64) if sample["logic2step"] else None,
                step2logic=np.array(sample["step2logic"], dtype=np.int64) if sample["step2logic"] else None,
            )
            ret_n.append(ret)
        return ret_n


    def sample(self, mask: Any | None = ..., probability: Any | None = ...) -> MALObsInstance:
        """Sample a random MAL observation."""
        raise NotImplementedError("Sampling is not implemented for MALObs")





class MALAttackerObs(MALObs):

    def contains(self, x: MALObsInstance) -> bool:
        if not super().contains(x):
            return False
        # Attacker observations must include attempts and traversable arrays
        return (
            x.attack_steps.attempts is not None
            and x.attack_steps.traversable is not None
        )


class MALDefenderObs(MALObs):

    def contains(self, x: MALObsInstance) -> bool:
        # Defender observations must NOT include attempts and traversable
        if x.attack_steps.attempts is not None or x.attack_steps.traversable is not None:
            return False
        return super().contains(x)

    def to_jsonable(self, sample_n: Sequence[MALObsInstance]) -> list[dict[str, Any]]:
        ret_n = []
        for sample in sample_n:
            # Force attempts and traversable to None for defender
            ret = {
                "time": sample.time,
                "attack_step_compromised": sample.attack_steps.compromised.tolist(),
                "attack_step_attempts": None,
                "attack_step_traversable": None,
                "asset_type": sample.assets.type.tolist(),
                "asset_id": sample.assets.id.tolist(),
                "attack_step_type": sample.attack_steps.type.tolist(),
                "attack_step_tags": sample.attack_steps.tags.tolist(),
                "attack_step_id": sample.attack_steps.id.tolist(),
                "attack_step_class": sample.attack_steps.logic_class.tolist(),
                "association_type": (sample.associations.type.tolist() if sample.associations else None),
                "logic_gate_type": (sample.logic_gates.type.tolist() if sample.logic_gates else None),
                "step2asset": sample.step2asset.tolist(),
                "assoc2asset": (sample.assoc2asset.tolist() if sample.assoc2asset else None),
                "step2step": sample.step2step.tolist(),
                "logic2step": (sample.logic2step.tolist() if sample.logic2step else None),
                "step2logic": (sample.step2logic.tolist() if sample.step2logic else None),
            }
            ret_n.append(ret)
        return ret_n

    def from_jsonable(self, sample_n: Sequence[dict[str, Any]]) -> list[MALObsInstance]:
        ret_n: list[MALObsInstance] = []
        for sample in sample_n:
            attack_step = AttackStep(
                type=np.array(
                    sample["attack_step_type"], dtype=self.attack_step_type.dtype
                ),
                id=np.array(
                    sample["attack_step_id"], dtype=self.attack_step_id.dtype
                ),
                logic_class=np.array(
                    sample["attack_step_class"], dtype=self.attack_step_class.dtype
                ),
                tags=np.array(
                    sample["attack_step_tags"], dtype=self.attack_step_tags.dtype
                ),
                compromised=np.array(
                    sample["attack_step_compromised"], dtype=self.attack_step_compromised.dtype
                ),
                attempts=None,
                traversable=None,
            )
            assets = Asset(
                type=np.array(sample["asset_type"], dtype=self.asset_type.dtype),
                id=np.array(sample["asset_id"], dtype=self.asset_id.dtype),
            )
            associations = Association(
                type=np.array(sample["association_type"], dtype=self.association_type.dtype),
            ) if sample["association_type"] else None
            logic_gates = LogicGate(
                type=np.array(sample["logic_gate_type"], dtype=self.logic_gate_type.dtype),
            ) if sample.get("logic_gate_type") and self.logic_gate_type else None
            ret = MALObsInstance(
                time=np.int64(sample["time"]),
                attack_steps=attack_step,
                assets=assets,
                associations=associations,
                logic_gates=logic_gates,
                step2asset=np.array(sample["step2asset"], dtype=np.int64),
                assoc2asset=np.array(sample["assoc2asset"], dtype=np.int64) if sample["assoc2asset"] else None,
                step2step=np.array(sample["step2step"], dtype=np.int64),
                logic2step=np.array(sample["logic2step"], dtype=np.int64) if sample["logic2step"] else None,
                step2logic=np.array(sample["step2logic"], dtype=np.int64) if sample["step2logic"] else None,
            )
            ret_n.append(ret)
        return ret_n