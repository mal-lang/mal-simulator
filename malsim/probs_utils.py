"""Utility functions for handling probabilities"""

from __future__ import annotations
import logging
import math
import random
from enum import Enum

from typing import Any, Optional, TYPE_CHECKING

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
        # TODO: Do we want a configurable default? There have been projects
        # that have asked for small, but non-zero values for all attack steps.
        return 0

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
    if probs_dict['type'] != 'function':
        raise ValueError('Expected value probability method requires a '
            'function probability distribution, but got '
            f'"{probs_dict["type"]}"')

    match(probs_dict['name']):
        case 'Bernoulli':
            # TODO: What is the expected value of non-unit non-zero Bernoulli?
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
