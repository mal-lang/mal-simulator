import json
import logging

from collections import deque
from typing import Any, Deque, Dict, List, Set, Union, Optional

import numpy as np

logger = logging.getLogger(__name__)


def get_new_targets(
    observation: dict, discovered_targets: Set[int], mask: tuple
) -> List[int]:
    attack_surface = mask[1]
    surface_indexes = list(np.flatnonzero(attack_surface))
    new_targets = [idx for idx in surface_indexes if idx not in discovered_targets]
    return new_targets, surface_indexes


class PassiveAttacker:
    def __init__(self, agent_config: dict) -> None:
        pass

    def compute_action_from_dict(self, observation, mask):
        return (0, None)


class BreadthFirstAttacker:
    def __init__(self, agent_config: dict) -> None:
        self.targets: Deque[int] = deque([])
        self.current_target: int = None
        self.attack_graph = agent_config.get('attack_graph')
        self.logs = []

        seed = agent_config.get('seed', np.random.SeedSequence().entropy)

        self.rng = (
            np.random.default_rng(seed)
            if agent_config.get('randomize', False)
            else None
        )

    def compute_action_from_dict(self, observation: Dict[str, Any], mask: tuple):
        new_targets, surface_indexes = get_new_targets(observation, self.targets, mask)

        # Add new targets to the back of the queue
        # if desired, shuffle the new targets to make the attacker more unpredictable
        if self.rng:
            self.rng.shuffle(new_targets)
        for c in new_targets:
            self.targets.appendleft(c)

        self.current_target, done = self.select_next_target(
            self.current_target, self.targets, surface_indexes
        )

        if done:
            logger.debug(
                'Attacker Breadth First agent does not have '
                'any valid targets it will terminate'
            )
            with open(f'{self.__class__.__name__}-logs.json', 'w') as f:
                json.dump(self.logs, f)
                self.logs = []

        # TODO: this does not have Context objects etc
        else:
            astep = self.attack_graph.nodes[self.current_target]
            for _, detector in self.attack_graph.nodes[
                self.current_target
            ].detectors.items():
                log = {}
                log['timestamp'] = observation['timestamp']
                log['agent'] = self.__class__.__name__
                log['_detector'] = detector['name']

                assets = []
                for label, log_asset in detector['context'].items():
                    try:
                        log[label] = str(self.asset.name)
                    except AttributeError:
                        lg_asset = self.attack_graph.lang_graph.get_asset_by_name(
                            log_asset
                        )
                        asset_types = [lg_asset.name] + [
                            a.name for a in lg_asset.sub_assets
                        ]
                        asset = [
                            asset
                            for asset in self.attack_graph.model.assets
                            if asset.type in asset_types
                        ].pop()

                        self.assets = {str(asset.name): asset}
                        log[label] = str(asset.name)

                if log:
                    self.logs.append(log)
                    print(log)

        return (int(not done), self.current_target)

    @staticmethod
    def select_next_target(
        current_target: int,
        targets: Union[List[int], Deque[int]],
        attack_surface: Set[int],
    ) -> tuple[Optional[int], bool]:
        # If the current target was not compromised, put it
        # back, but on the bottom of the stack.
        if current_target in attack_surface:
            targets.appendleft(current_target)
            current_target = targets.pop()

        try:
            return targets.pop(), False
        except IndexError:
            return None, True


class DepthFirstAttacker:
    def __init__(self, agent_config: dict) -> None:
        self.current_target = -1
        self.targets: List[int] = []
        self.attack_graph = agent_config.get('attack_graph')
        self.logs = []

        seed = (
            agent_config['seed']
            if agent_config.get('seed', None)
            else np.random.SeedSequence().entropy
        )
        self.rng = (
            np.random.default_rng(seed)
            if agent_config.get('randomize', False)
            else None
        )

    def compute_action_from_dict(self, observation: Dict[str, Any], mask: tuple):
        new_targets, surface_indexes = get_new_targets(observation, self.targets, mask)

        # Add new targets to the top of the stack
        if self.rng:
            self.rng.shuffle(new_targets)
        for c in new_targets:
            self.targets.append(c)

        self.current_target, done = self.select_next_target(
            self.current_target, self.targets, surface_indexes
        )

        if done:
            logger.debug(
                'Attacker Depth First agent does not have '
                'any valid targets it will terminate'
            )
            with open(f'{self.__class__.__name__}-logs.json', 'w') as f:
                json.dump(self.logs, f)
                self.logs = []

        # TODO: this does not have Context objects etc
        else:
            for _, detector in self.attack_graph.nodes[
                self.current_target
            ].detectors.items():
                log = {}
                log['timestamp'] = observation['timestamp']
                log['agent'] = self.__class__.__name__
                log['_detector'] = detector['name']

                for label, log_asset in detector['context'].items():
                    try:
                        log[label] = str(self.asset.name)
                    except AttributeError:
                        lg_asset = self.attack_graph.lang_graph.get_asset_by_name(
                            log_asset
                        )
                        asset_types = [lg_asset.name] + [
                            a.name for a in lg_asset.sub_assets
                        ]
                        asset = [
                            asset
                            for asset in self.attack_graph.model.assets
                            if asset.type in asset_types
                        ].pop()

                        self.assets = {str(asset.name): asset}
                        log[label] = str(asset.name)
                if log:
                    self.logs.append(log)
                    print(log)

        return (int(not done), self.current_target)
        # self.current_target = None if done else self.current_target
        # action = 0 if done else 1
        # return (action, self.current_target)

    @staticmethod
    def select_next_target(
        current_target: int,
        targets: Union[List[int], Deque[int]],
        attack_surface: Set[int],
    ) -> int:
        if current_target in attack_surface:
            return current_target, False

        while current_target not in attack_surface:
            if len(targets) == 0:
                return None, True

            current_target = targets.pop()

        return current_target, False


AGENTS = {
    BreadthFirstAttacker.__name__: BreadthFirstAttacker,
    DepthFirstAttacker.__name__: DepthFirstAttacker,
}
