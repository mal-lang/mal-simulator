"""Utility functions for handling probabilities"""

from __future__ import annotations
import logging
import math
from enum import Enum
from dataclasses import dataclass

from typing import Any, Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.stats import expon, gamma, binom, lognorm, uniform, bernoulli

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraphNode
    from scipy.stats._distn_infrastructure import (
        rv_continuous_frozen,
        rv_discrete_frozen
    )

logger = logging.getLogger(__name__)

class DistFunction(Enum):
    BERNOULLI = "Bernoulli"
    EXPONENTIAL = "Exponential"
    BINOMIAL = "Binomial"
    GAMMA = "Gamma"
    LOG_NORMAL = "LogNormal"
    UNIFORM = "Uniform"

class Operation(Enum):
    # TODO: check that these match MAL
    SUM = "sum"
    SUBTRACT = "sub"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    EXPONENTIATION = "exponentiation"

def default_ttc_dist(node: AttackGraphNode) -> TTCDist:
    """ttc distribution if no ttc is set in lang or model"""
    if node.type == 'defense':
        return named_ttc_dists['Disabled']
    if node.type in ('or', 'and'):
        return named_ttc_dists['Instant']

    # Other steps have no ttc
    raise ValueError(
        f'Can only get default TTC of defense and attack steps, not of "{node.type}" steps'
    )


class TTCDist:

    def __init__(
        self,
        function: DistFunction,
        args: list[float],
        combine_with: Optional[TTCDist] = None,
        combine_op: Optional[Operation] = None
    ):
        self.function = function
        self.args = args

        if function == DistFunction.BERNOULLI:
            self.dist = bernoulli(self.args[0])
        elif function == DistFunction.BINOMIAL:
            self.dist = binom(n=args[0], p=args[1])
        elif function == DistFunction.EXPONENTIAL:
            self.dist = expon(scale=1 / args[0])
        elif function == DistFunction.GAMMA:
            self.dist = gamma(args[0])
        elif function == DistFunction.LOG_NORMAL:
            self.dist = lognorm(args[0])
        elif function == DistFunction.UNIFORM:
            self.dist = uniform()
        else:
            raise ValueError(f"Unknown distribution {function}")

        self.combine_with = combine_with
        self.combine_op = combine_op

    @property
    def expected_value(self):
        """Return the expected value of a TTCDist"""

        value = self.dist.expect()

        if self.combine_with:
            # Combine with other distribution if TTC is combined
            if self.combine_op == Operation.SUM:
                value += self.combine_with.expected_value
            elif self.combine_op == Operation.SUBTRACT:
                value -= self.combine_with.expected_value
            elif self.combine_op == Operation.MULTIPLY:
                value *= self.combine_with.expected_value
            elif self.combine_op == Operation.DIVIDE:
                value /= self.combine_with.expected_value
            elif self.combine_op == Operation.EXPONENTIATION:
                value = float(pow(value, self.combine_with.expected_value))
            else:
                raise ValueError(f"Unknown operation {self.combine_op}")

        return value

    def sample_value(self, rng: Optional[np.random.Generator] = None):
        """Sample a value from the TTC Distribution"""
        value = self.dist.rvs(random_state=rng)

        if self.combine_with:
            # Combine with other distribution if TTC is combined
            if self.combine_op == Operation.SUM:
                value += self.combine_with.sample_value(rng)
            elif self.combine_op == Operation.SUBTRACT:
                value -= self.combine_with.sample_value(rng)
            elif self.combine_op == Operation.MULTIPLY:
                value *= self.combine_with.sample_value(rng)
            elif self.combine_op == Operation.DIVIDE:
                value /= self.combine_with.sample_value(rng)
            elif self.combine_op == Operation.EXPONENTIATION:
                value = float(pow(value, self.combine_with.sample_value(rng)))
            else:
                raise ValueError(f"Unknown operation {self.combine_op}")

        return value

    def success_probability(self, effort: int) -> float:
        """The probability to succeed with given effort (previous attempts)"""
        return self.dist.cdf(effort)

    def attempt_ttc_with_effort(
        self, effort: int, rng: Optional[np.random.Generator] = None
    ) -> bool:
        """
        Attempt to compromise a step by sampling a success probability
        proportional to the TTC distribution, given previous attempts.
        Note: Ignores Bernoullis.
        """
        if not rng:
            rng = np.random.default_rng()

        success_prob = self.success_probability(effort)
        return success_prob > rng.random()

    def attempt_bernoulli(self, rng: Optional[np.random.Generator] = None):
        """Attempt bernoulli from a TTC Distribution"""
        rng = rng or np.random.default_rng()
        if self.function == DistFunction.BERNOULLI:
            threshold = float(self.args[0])
            bernoulli_success = rng.random() <= threshold
            return bernoulli_success
        elif self.combine_with:
            return self.combine_with.attempt_bernoulli(rng)
        else:
            return True

    @classmethod
    def from_dict(cls, ttc_dict: dict) -> TTCDist:
        """TTCs are stored as dict in the MAL-toolbox, convert to TTCDist
        - If the TTC provided is a predefined name, return that.
        - Otherwise create the TTCDist object (recursively)
        """

        if 'name' in ttc_dict:
            if ttc_dict['name'] in named_ttc_dists:
                # Predefined step ttc set in language, fetch from dict
                return named_ttc_dists[ttc_dict['name']]

            else:
                return TTCDist(
                    DistFunction[ttc_dict['name'].upper()],
                    args=ttc_dict['arguments']
                )

        # If lhs, rhs is set, we combine the distributions
        return cls(
            DistFunction[ttc_dict['lhs']['name']],
            ttc_dict['lhs']['arguments'],
            combine_with = TTCDist.from_dict(ttc_dict['rhs']),
            combine_op = Operation[ttc_dict['operation'].upper()]
        )

    @classmethod
    def from_node(cls, node: AttackGraphNode,) -> TTCDist:
        """Create a TTCDist based on an AttackGraphNode"""

        if node.ttc is None:
            return default_ttc_dist(node)

        # Not predefined, must parse the dict to create a TTCDist
        return TTCDist.from_dict(node.ttc)


# These are MAL supported mappings from name to distribution 
named_ttc_dists: dict[str, TTCDist] = {
    'EasyAndUncertain': TTCDist(
        DistFunction.BERNOULLI, [0.5]
    ),
    'HardAndUncertain': TTCDist(
        DistFunction.EXPONENTIAL, [0.1],
        combine_with=TTCDist(
            DistFunction.BERNOULLI, [0.5]
        ),
        combine_op=Operation.MULTIPLY
    ),
    'VeryHardAndUncertain': TTCDist(
        DistFunction.EXPONENTIAL, [0.01],
        combine_with=TTCDist(
            DistFunction.BERNOULLI, [0.5]
        ),
        combine_op=Operation.MULTIPLY
    ),
    'EasyAndCertain': TTCDist(
        DistFunction.EXPONENTIAL, [1.0]
    ),
    'HardAndCertain': TTCDist(
        DistFunction.EXPONENTIAL, [0.1]
    ),
    'VeryHardAndCertain': TTCDist(
        DistFunction.EXPONENTIAL, [0.01]
    ),
    'Enabled': TTCDist(
        DistFunction.BERNOULLI, [1.0]
    ),
    'Instant': TTCDist(
        DistFunction.BERNOULLI, [1.0]
    ),
    'Disabled': TTCDist(
        DistFunction.BERNOULLI, [0.0]
    )
}
