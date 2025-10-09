"""TYR Monitor REST API client"""

from __future__ import annotations
import logging
from typing import Any, Optional

import requests
from maltoolbox.attackgraph import AttackGraph, AttackGraphNode
from maltoolbox.model import Model

logger = logging.getLogger(__name__)
DEFAULT_TIMEOUT = 30

class MalSimGUIClient():
    """A client that can talk to the TYR Monitor REST API"""

    def __init__(
            self,
            host: str = "localhost",
            port: int = 8888,
        ):
        self.protocol = "http"
        self.host = host
        self.port = port

        self._test_connection()

    def _create_url(self, endpoint: str) -> str:
        return f"{self.protocol}://{self.host}:{self.port}/{endpoint}"

    def _send_request(
            self,
            method: str,
            endpoint: str,
            json_content: Optional[Any] = None
        ) -> requests.Response:
        """Send a request to the REST API"""
        url = self._create_url(endpoint)
        if method == 'GET':
            res = requests.get(url, timeout=DEFAULT_TIMEOUT)
        elif method == 'POST':
            res = requests.post(
                url, json=json_content, timeout=DEFAULT_TIMEOUT
            )
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        res.raise_for_status()
        return res

    def _test_connection(self) -> None:
        """Tries to run a request to see if api is up"""
        try:
            self._send_request('GET', '')
        except requests.ConnectionError as e:
            raise Exception(
                "Could not connect to malsim gui API. "
                "Is the malsim gui API actually running? "
                "Please refer to README."
            ) from e

    def upload_model(self, model: Model) -> None:
        """Uploads a serialized model to the POST endpoint of the API"""
        self._send_request(
            'POST',
            'model',
            json_content=model._to_dict()
        )

    def upload_attack_graph(self, attack_graph: AttackGraph) -> None:
        """Uploads a serialized graph to the POST endpoint of the API"""
        self._send_request(
            'POST',
            'attack_graph',
            json_content=attack_graph._to_dict()
        )

    def upload_performed_nodes(
            self, new_performed_nodes: list[AttackGraphNode], iteration: int
        ) -> None:
        """Uploads newly performed nodes to API"""
        self._send_request(
            'POST',
            'performed_nodes',
            json_content=[
                {'node_id': n.id, 'iteration': iteration}
                for n in new_performed_nodes
            ]
        )

    def upload_latest_attack_steps(
            self, latest_steps: dict[int, list[dict[str, Any]]]) -> None:
        """Upload dict of latest attack step ids mapped to alert logs"""
        self._send_request(
            'POST',
            'latest_attack_steps',
            json_content=latest_steps
        )

    def upload_defender_suggestions(
            self, suggestions: dict[str, dict[int, dict[Any, Any]]]) -> None:
        """
        Uploads dict of suggestions that maps agent name
        to a dict mapping node id to suggestion info.
        """
        self._send_request(
            'POST',
            'defender_suggestions',
            json_content=suggestions
        )

    def get_defender_action(self) -> Any:
        """Get selected defender action from API"""
        return self._send_request(
            'GET',
            'defender_action'
        ).json()

    def clear_defender_action(self) -> None:
        """Post None to selected action from API"""
        self._send_request(
            'POST',
            'defender_action',
            json_content={'iteration': -1, 'node_id': None}
        )

    def get_reward_value(self) -> Any:
        """Get reward value for the current iteration from API"""
        return self._send_request(
            'GET',
            'reward_value'
        ).json()

    def set_reward_value(self, iteration: int, reward: float) -> None:
        """Post reward value for iteration to API"""
        self._send_request(
            'POST',
            'reward_value',
            json_content={'iteration': iteration, 'reward': reward}
        )

    def reset(self) -> None:
        """Reset the API contents"""
        self._send_request('POST', 'reset')

    def upload_initial_state(
            self, attack_graph: AttackGraph
        ) -> None:
        """Reset the REST API and upload model and graph"""
        print("Resetting and uploading initial state to API")
        self.reset()
        assert attack_graph.model, "Graph needs model connections for GUI"
        self.upload_model(attack_graph.model)
        self.upload_attack_graph(attack_graph)
