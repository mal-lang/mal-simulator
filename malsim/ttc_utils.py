"""Utility functions for handling probabilities"""

from __future__ import annotations
import logging
from enum import Enum

from typing import Any, Optional, TYPE_CHECKING

import numpy as np
from scipy.stats import expon, gamma, binom, lognorm, uniform, bernoulli

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraphNode

logger = logging.getLogger(__name__)


class DistFunction(Enum):
    BERNOULLI = 'Bernoulli'
    EXPONENTIAL = 'Exponential'
    BINOMIAL = 'Binomial'
    GAMMA = 'Gamma'
    LOGNORMAL = 'LogNormal'
    UNIFORM = 'Uniform'


class Operation(Enum):
    # TODO: check that these match MAL
    ADDITION = 'addition'
    SUBTRACTION = 'subtraction'
    MULTIPLICATION = 'multiplication'
    DIVISION = 'division'
    EXPONENTIATION = 'exponentiation'


def perform_operation(op: Operation, a: float, b: float) -> float:
    if op == Operation.ADDITION:
        return a + b
    elif op == Operation.SUBTRACTION:
        return a - b
    elif op == Operation.MULTIPLICATION:
        return a * b
    elif op == Operation.DIVISION:
        return a / b
    elif op == Operation.EXPONENTIATION:
        return float(a**b)
    else:
        raise ValueError(f'Unknown operation {op}')


def default_ttc_dist(node: AttackGraphNode) -> TTCDist:
    """ttc distribution if no ttc is set in lang or model"""
    if node.type == 'defense':
        return named_ttc_dists['Disabled']
    if node.type in ('or', 'and'):
        return named_ttc_dists['Instant']

    # Other steps have no ttc
    raise ValueError(
        'Can only get default TTC of defense and attack steps,'
        f' not of "{node.type}" steps'
    )


class TTCDist:
    def __init__(
        self,
        function: DistFunction,
        args: list[float],
        combine_with: Optional[TTCDist] = None,
        combine_op: Optional[Operation] = None,
    ):
        self.function = function
        self.args = args

        if function == DistFunction.BERNOULLI:
            self.dist = bernoulli(self.args[0])
        elif function == DistFunction.BINOMIAL:
            n, p = args
            self.dist = binom(n=n, p=p)
        elif function == DistFunction.EXPONENTIAL:
            (rate,) = args
            self.dist = expon(scale=1 / rate)
        elif function == DistFunction.GAMMA:
            shape, scale = args
            self.dist = gamma(a=shape, scale=scale)
        elif function == DistFunction.LOGNORMAL:
            mean, std = args
            self.dist = lognorm(s=std, scale=np.exp(mean))
        elif function == DistFunction.UNIFORM:
            self.dist = uniform()
        else:
            raise ValueError(f'Unknown distribution {function}')

        self.combine_with = combine_with
        self.combine_op = combine_op

    @property
    def expected_value(self) -> float:
        """Return the expected value of a TTCDist"""

        if self.function == DistFunction.BERNOULLI:
            # Expected value of Bernoulli should affect existence (for attack steps)
            # and initial state (defenses). When fetching expected value we assume
            # the Bernoulli trial was successful.
            value = 1.0
        else:
            value = float(self.dist.expect())

        if self.combine_with and self.combine_op:
            value = perform_operation(
                self.combine_op, value, self.combine_with.expected_value
            )
        return value

    def sample_value(self, rng: Optional[np.random.Generator] = None) -> float:
        """Sample a value from the TTC Distribution"""
        rng = rng or np.random.default_rng()

        if self.function == DistFunction.BERNOULLI:
            # Sampled value of Bernoulli should affect existence (for attack steps)
            # and initial state (defenses). When fetching expected value we assume
            # the Bernoulli trial was successful.
            value = 1.0
        else:
            value = float(self.dist.rvs(random_state=rng))
        if self.combine_with and self.combine_op:
            value = perform_operation(
                self.combine_op, value, self.combine_with.sample_value(rng)
            )
        return value

    def success_probability(self, effort: int) -> float:
        """The probability to succeed with given effort (previous attempts)"""
        return float(self.dist.cdf(effort))

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

    def attempt_bernoulli(self, rng: Optional[np.random.Generator] = None) -> bool:
        """Attempt bernoulli from a TTC Distribution"""
        rng = rng or np.random.default_rng()
        if self.function == DistFunction.BERNOULLI:
            # If current TTCDist is a Bernoulli, sample from it to attempt
            threshold = float(self.args[0])
            bernoulli_success = rng.random() <= threshold
            return bernoulli_success
        elif self.combine_with:
            # Dig deeper in the distribution after Bernoullis
            return self.combine_with.attempt_bernoulli(rng)
        else:
            # If no Bernoulli set, attempt is successful
            return True

    @classmethod
    def from_dict(cls, ttc_dict: dict[str, Any]) -> TTCDist:
        """TTCs are stored as dict in the MAL-toolbox, convert to TTCDist
        - If the TTC provided is a predefined name, return that.
        - Otherwise create the TTCDist object (recursively)
        """

        all_dist_fun_names = [d.value for d in DistFunction]

        if 'name' in ttc_dict:
            if ttc_dict['name'] in named_ttc_dists:
                # Predefined step ttc set in language, fetch from dict
                return named_ttc_dists[ttc_dict['name']]

            else:
                if ttc_dict['name'] not in all_dist_fun_names:
                    raise ValueError(
                        f"Unknown distribution function name '{ttc_dict['name']}'. \n"
                        f'Must be one of: {", ".join(all_dist_fun_names)}'
                    )
                return TTCDist(
                    DistFunction[ttc_dict['name'].upper()], args=ttc_dict['arguments']
                )

        # If lhs, rhs is set, we combine the distributions
        lhs = TTCDist.from_dict(ttc_dict['lhs'])
        lhs.combine_op = Operation[ttc_dict['type'].upper()]
        lhs.combine_with = TTCDist.from_dict(ttc_dict['rhs'])
        return lhs

    @classmethod
    def from_node(
        cls,
        node: AttackGraphNode,
    ) -> TTCDist:
        """Create a TTCDist based on an AttackGraphNode"""

        if node.ttc is None:
            return default_ttc_dist(node)

        # Not predefined, must parse the dict to create a TTCDist
        return TTCDist.from_dict(node.ttc)


# These are MAL supported mappings from name to distribution
named_ttc_dists: dict[str, TTCDist] = {
    'EasyAndUncertain': TTCDist(DistFunction.BERNOULLI, [0.5]),
    'HardAndUncertain': TTCDist(
        DistFunction.EXPONENTIAL,
        [0.1],
        combine_with=TTCDist(DistFunction.BERNOULLI, [0.5]),
        combine_op=Operation.MULTIPLICATION,
    ),
    'VeryHardAndUncertain': TTCDist(
        DistFunction.EXPONENTIAL,
        [0.01],
        combine_with=TTCDist(DistFunction.BERNOULLI, [0.5]),
        combine_op=Operation.MULTIPLICATION,
    ),
    'EasyAndCertain': TTCDist(DistFunction.EXPONENTIAL, [1.0]),
    'HardAndCertain': TTCDist(DistFunction.EXPONENTIAL, [0.1]),
    'VeryHardAndCertain': TTCDist(DistFunction.EXPONENTIAL, [0.01]),
    'Enabled': TTCDist(DistFunction.BERNOULLI, [1.0]),
    'Instant': TTCDist(DistFunction.BERNOULLI, [1.0]),
    'Disabled': TTCDist(DistFunction.BERNOULLI, [0.0]),
}
