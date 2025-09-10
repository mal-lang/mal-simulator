"""Utility functions for handling probabilities"""

from __future__ import annotations
import logging
import math
import random
from enum import Enum

from typing import Any, Optional, TYPE_CHECKING
from numbers import Number

import numpy as np
from scipy.stats import expon

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraphNode

logger = logging.getLogger(__name__)

class ProbCalculationMethod(Enum):
    SAMPLE = 1
    EXPECTED = 2


def sample_prob(
        node: AttackGraphNode,
        probs_dict: dict[str, Any],
        calculated_bernoullis: dict[AttackGraphNode, float]
    ) -> float:
    """Calculate the sampled value from a probability distribution function
    Arguments:
    probs_dict      - a dictionary containing the probability distribution
                      function

    Return:
    The float value obtained from calculating the sampled value corresponding
    to the function provided.
    """

    if probs_dict is None:
        raise ValueError('Probabilities dictionary was missing.')

    if probs_dict['type'] != 'function':
        raise ValueError('Sample probability method requires a function '
            f'probability distribution, but got "{probs_dict["type"]}"')

    match(probs_dict['name']):
        case 'Bernoulli':
            if node in calculated_bernoullis:
                return calculated_bernoullis[node]
            value = random.random()
            threshold = float(probs_dict['arguments'][0])
            res = math.inf if value > threshold else 1.0
            calculated_bernoullis[node] = res
            return res

        case 'Exponential':
            lambd = float(probs_dict['arguments'][0])
            return random.expovariate(lambd)

        case 'Binomial':
            n = int(probs_dict['arguments'][0])
            p = float(probs_dict['arguments'][1])
            # TODO: Someone with basic probabilities competences should
            # actually check if this is correct.
            return random.binomialvariate(n, p)

        case 'Gamma':
            alpha = float(probs_dict['arguments'][0])
            beta = float(probs_dict['arguments'][1])
            return random.gammavariate(alpha, beta)

        case 'LogNormal':
            mu = float(probs_dict['arguments'][0])
            sigma = float(probs_dict['arguments'][1])
            return random.lognormvariate(mu, sigma)

        case 'Uniform':
            a = float(probs_dict['arguments'][0])
            b = float(probs_dict['arguments'][1])
            return random.uniform(a, b)

        case 'Pareto' | 'Truncated Normal':
            raise NotImplementedError(f'"{probs_dict["name"]}" '
                'probability distribution is not currently '
                'supported!')

        case _:
            raise ValueError('Unknown probability distribution '
                f'function encountered "{probs_dict["name"]}"!')


def expected_prob(node: AttackGraphNode, probs_dict: dict[str, Any]) -> float:
    """Calculate the expected value from a probability distribution function
    Arguments:
    probs_dict      - a dictionary containing the probability distribution
                      function

    Return:
    The float value obtained from calculating the expected value corresponding
    to the function provided.
    """

    if probs_dict is None:
        raise ValueError('Probabilities dictionary was missing.')

    if probs_dict['type'] != 'function':
        raise ValueError('Expected value probability method requires a '
            'function probability distribution, but got '
            f'"{probs_dict["type"]}"')

    match(probs_dict['name']):
        case 'Bernoulli':
            # This expected value estimation was added so that we can use
            # non-zero and non-unit Bernoulli values. Failing a Bernoulli
            # yields an infinite value and multiplying any non-zero value by
            # infinity does nothing. We decided to simply use the
            # multiplicative inverse(e.g. 0.1 -> 10, 0.25 -> 4) since most
            # often the Bernoulli is used a multiplication factor.
            threshold = (
                1 / float(probs_dict['arguments'][0])
                if probs_dict['arguments'][0] != 0
                else math.inf
            )
            return threshold

        case 'Exponential':
            lambd = float(probs_dict['arguments'][0])
            return 1/lambd

        case 'Binomial':
            n = int(probs_dict['arguments'][0])
            p = float(probs_dict['arguments'][1])
            # TODO: Someone with basic probabilities competences should
            # actually check if this is correct.
            return n * p

        case 'Gamma':
            alpha = float(probs_dict['arguments'][0])
            beta = float(probs_dict['arguments'][1])
            return alpha * beta

        case 'LogNormal':
            mu = float(probs_dict['arguments'][0])
            sigma = float(probs_dict['arguments'][1])
            return float(pow(math.e, (mu + (pow(sigma, 2)/2))))

        case 'Uniform':
            a = float(probs_dict['arguments'][0])
            b = float(probs_dict['arguments'][1])
            return (a + b)/2

        case 'Pareto' | 'Truncated Normal':
            raise NotImplementedError(f'"{probs_dict["name"]}" '
                'probability distribution is not currently '
                'supported!')

        case _:
            raise ValueError('Unknown probability distribution '
                f'function encountered "{probs_dict["name"]}"!')


def calculate_prob(
    node: AttackGraphNode,
    probs_dict: Optional[dict[str, Any]],
    method: ProbCalculationMethod,
    calculated_bernoullis: dict[AttackGraphNode, float]
) -> float:
    """Calculate the value from a probability distribution
    Arguments:
    probs_dict      - a dictionary containing the probability distribution
                      function
    method          - the method to use in calculating the probability
                      values(currently supporting sampled or expected values)

    Return:
    The float value obtained from calculating the probability distribution.

    TTC Distributions in MAL:
    https://mal-lang.org/mal-langspec/apidocs/org.mal_lang.langspec/org/mal_lang/langspec/ttc/TtcDistribution.html
    """
    if not probs_dict:
        return math.nan

    match(probs_dict['type']):
        case 'addition' | 'subtraction' | 'multiplication' | \
                'division' | 'exponentiation':
            lv = calculate_prob(
                node, probs_dict['lhs'], method, calculated_bernoullis
            )
            rv = calculate_prob(
                node, probs_dict['rhs'], method, calculated_bernoullis
            )
            match(probs_dict['type']):
                case 'addition':
                    return lv + rv
                case 'subtraction':
                    return lv - rv
                case 'multiplication':
                    return lv * rv
                case 'division':
                    return lv / rv
                case 'exponentiation':
                    return float(pow(lv, rv))

        case 'function':
            match(method):
                case ProbCalculationMethod.SAMPLE:
                    return sample_prob(node, probs_dict, calculated_bernoullis)
                case ProbCalculationMethod.EXPECTED:
                    return expected_prob(node, probs_dict)
                case _:
                    raise ValueError('Unknown Probability Calculation method '
                    f'encountered "{method}"!')

        case _:
            raise ValueError('Unknown probability distribution type '
            f'encountered "{probs_dict["type"]}"!')

### SANDOR TTC IMPLEMENTATION

def attempt_step_ttc(
    node: AttackGraphNode,
    step_working_time: int, rng: np.random.Generator
) -> bool:
    """Attempt to compromise a step by sampling a Bernoulli"""
    assert node.ttc, f"Node {node} has no ttc"
    success_prob = (
        get_time_distribution(node.ttc)
        .success_probability(step_working_time)
    )
    return success_prob > rng.random()


class TTCDist:
    def __init__(self, obj):
        self.obj = obj
        self.is_scalar = isinstance(obj, Number)

    def rvs(self, size=None, **kwargs):
        if self.is_scalar:
            return self.obj if size is None else np.full(size, self.obj)
        return self.obj.rvs(size=size, **kwargs)

    def success_probability(self, t: int) -> float:
        if self.is_scalar:
            return 1.0 if t >= self.obj else 0.0
        return self.obj.cdf(t)


def get_time_distribution(ttc: Optional[dict]) -> TTCDist:
    """Get TTC Distribution from predefined names in MAL"""
    if not ttc:
        return TTCDist(0.0)

    name = ttc.get("name")
    arguments = ttc.get("arguments", [])

    if name == "VeryHardAndUncertain":
        # Ber(0.5) * Exp(0.01)
        return TTCDist(expon(scale=1 / 0.01))

    elif name == "HardAndUncertain":
        # Ber(0.5) * Exp(0.1)
        return TTCDist(expon(scale=1 / 0.1))

    elif name == "EasyAndCertain":
        # Exp(1.0)
        return TTCDist(expon(scale=1 / 1.0))

    elif name == "EasyAndUncertain":
        # Ber(0.5)
        return TTCDist(0.0)

    elif name == "HardAndCertain":
        # Exp(0.1)
        return TTCDist(expon(scale=1 / 0.1))

    elif name == "VeryHardAndCertain":
        # Exp(0.01)
        return TTCDist(expon(scale=1 / 0.01))

    elif name == "Infinity":
        # No sampling, effectively infinity
        return TTCDist(float("inf"))

    elif name == "Zero":
        # No sampling, effectively zero
        return TTCDist(0.0)

    elif name == "Enabled":
        # Always 1
        return TTCDist(1.0)

    elif name == "Disabled":
        # Always 0
        return TTCDist(0.0)

    elif name == "Exponential":
        # Exponential distribution with lambda
        if len(arguments) != 1:
            raise ValueError(
                "Exponential requires exactly one argument (lambda)."
            )
        lam = arguments[0]
        return TTCDist(expon(scale=1 / lam))

    else:
        raise ValueError(f"Unsupported distribution name: {name}")
