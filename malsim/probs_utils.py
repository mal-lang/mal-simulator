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


predef_ttcs: dict[str, dict] = {
    'EasyAndUncertain':
    {
        'arguments': [0.5],
        'name': 'Bernoulli',
        'type': 'function'
    },
    'HardAndUncertain':
    {
        'lhs':
        {
            'arguments': [0.1],
            'name': 'Exponential',
            'type': 'function'
        },
        'rhs':
        {
            'arguments': [0.5],
            'name': 'Bernoulli',
            'type': 'function'
        },
        'type': 'multiplication'
    },
    'VeryHardAndUncertain':
    {
        'lhs':
        {
            'arguments': [0.01],
            'name': 'Exponential',
            'type': 'function'
        },
        'rhs':
        {
            'arguments': [0.5],
            'name': 'Bernoulli',
            'type': 'function'
        },
        'type': 'multiplication'
    },
    'EasyAndCertain':
    {
        'arguments': [1.0],
        'name': 'Exponential',
        'type': 'function'
    },
    'HardAndCertain':
    {
        'arguments': [0.1],
        'name': 'Exponential',
        'type': 'function'
    },
    'VeryHardAndCertain':
    {
        'arguments': [0.01],
        'name': 'Exponential',
        'type': 'function'
    },
    'Enabled':
    {
        'arguments': [1.0],
        'name': 'Bernoulli',
        'type': 'function'
    },
    'Instant':
    {
        'arguments': [1.0],
        'name': 'Bernoulli',
        'type': 'function'
    },
    'Disabled':
    {
        'arguments': [0.0],
        'name': 'Bernoulli',
        'type': 'function'
    },
}


def default_ttc(node: AttackGraphNode) -> Optional[dict[str, Any]]:
    """ttc distribution if no ttc is set in lang or model"""
    if node.type == 'defense':
        return predef_ttcs['Disabled']
    if node.type in ('or', 'and'):
        return predef_ttcs['Instant']

    # Other steps have no ttc
    return None


def get_ttc_dict(
        node: AttackGraphNode,
    ) -> Optional[dict[str, Any]]:
    """Convert step TTC to a TTC distribution dict if needed

    - If the TTC provided is a predefined name replace it with the
      ttc distribution dict it corresponds to.
    - Otherwise return the TTC distribution as is.

    Arguments:
    node - Attack graph node to get ttc dict for

    Returns:
    A dict with the steps TTC distribution or None if no ttc applies
    """

    step_ttc = node.ttc or default_ttc(node)
    if not step_ttc:
        return None

    if 'name' in step_ttc and step_ttc['name'] in predef_ttcs:
        # Predefined step ttc set in language, fetch from dict
        step_ttc = predef_ttcs[step_ttc['name']].copy()

    return step_ttc


class ProbCalculationMethod(Enum):
    SAMPLE = 1
    EXPECTED = 2


def _sample_value(
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


def _expected_value(probs_dict: dict[str, Any]) -> float:
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


def ttc_value_from_ttc_dict(
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
            lv = ttc_value_from_ttc_dict(
                node, probs_dict['lhs'], method, calculated_bernoullis
            )
            rv = ttc_value_from_ttc_dict(
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
                    return _sample_value(
                        node, probs_dict, calculated_bernoullis
                    )
                case ProbCalculationMethod.EXPECTED:
                    return _expected_value(probs_dict)
                case _:
                    raise ValueError('Unknown Probability Calculation method '
                    f'encountered "{method}"!')

        case _:
            raise ValueError(
                'Unknown probability distribution type '
                f'encountered "{probs_dict["type"]}"!'
            )

def ttc_value_from_node(
    node: AttackGraphNode,
    method: ProbCalculationMethod,
    calculated_bernoullis: dict[AttackGraphNode, float]
):
    ttc_dict = get_ttc_dict(node)
    return ttc_value_from_ttc_dict(
        node, ttc_dict, method, calculated_bernoullis
    )

### SANDOR TTC IMPLEMENTATION

def attempt_step_ttc(
    node: AttackGraphNode,
    step_working_time: int, rng: np.random.Generator
) -> bool:
    """Attempt to compromise a step by sampling a Bernoulli"""

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
