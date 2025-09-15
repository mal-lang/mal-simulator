"""Unit tests for the probabilities utilities module"""

import numpy as np

from maltoolbox.model import Model
from maltoolbox.attackgraph.attackgraph import AttackGraph
from maltoolbox.language.languagegraph import LanguageGraph

from malsim.ttc_utils import (
    ttc_value_from_node,
    ProbCalculationMethod,
    predef_ttcs,
    get_ttc_dict,
)
from .conftest import path_testdata

def test_probs_utils(model: Model) -> None:
    """Test TTC calculation for nodes"""

    app = model.add_asset('Application')
    creds = model.add_asset('Credentials')
    user = model.add_asset('User')
    identity = model.add_asset('Identity')
    vuln = model.add_asset('SoftwareVulnerability')

    identity.add_associated_assets('credentials', {creds})
    user.add_associated_assets('userIds', {identity})
    app.add_associated_assets('highPrivAppIAMs', {identity})
    app.add_associated_assets('vulnerabilities', {vuln})

    attack_graph = AttackGraph(model.lang_graph, model)

    for node in attack_graph.nodes.values():
        #TODO: Actually check some of the results
        ttc_value_from_node(
            node,
            ProbCalculationMethod.SAMPLE,
            {},
            np.random.default_rng(10)
        )

    for node in attack_graph.nodes.values():
        #TODO: Actually check some of the results
        ttc_value_from_node(
            node,
            ProbCalculationMethod.EXPECTED,
            {},
            np.random.default_rng(10)
        )


def test_get_ttc_dict_defenses(corelang_lang_graph: LanguageGraph) -> None:
    """Make sure TTCs are set correctly for defenses"""

    model = Model.load_from_file(
        path_testdata("models/simple_example_model.yml"),
        corelang_lang_graph
    )
    attack_graph = AttackGraph(
        lang_graph=corelang_lang_graph, model=model
    )

    expected_bernoulli_0_defenses = {
        # Disabled in lang
        'Credentials:10:notPhishable',
        'Data:5:notPresent',
        'Group:13:notPresent',
        'IDPS 1:notPresent',
        'OS App:notPresent',
        'Program 2:notPresent',
        'Credentials:9:notPhishable',
        'Identity:11:notPresent',
        'Credentials:7:notPhishable',
        'Identity:8:notPresent',
        'Credentials:6:notPhishable',
        'SoftwareVulnerability:4:notPresent',

        # Disabled in lang, suppressed (does not matter here)
        'SoftwareVulnerability:4:networkAccessRequired',
        'SoftwareVulnerability:4:localAccessRequired',
        'SoftwareVulnerability:4:physicalAccessRequired',
        'SoftwareVulnerability:4:highPrivilegesRequired',
        'SoftwareVulnerability:4:userInteractionRequired',
        'SoftwareVulnerability:4:confidentialityImpactLimitations',
        'SoftwareVulnerability:4:integrityImpactLimitations',
        'SoftwareVulnerability:4:highComplexityExploitRequired',

        # TTC not set in lang, disabled by default
        'OS App:supplyChainAuditing',
        'Program 2:supplyChainAuditing',
        'User:12:securityAwareness',
        'IDPS 1:supplyChainAuditing',
        'Program 1:supplyChainAuditing',

        # Enabled in lang, disabled in model
        'Credentials:9:unique',
        'User:12:noPasswordReuse',
        'Credentials:7:unique',
        'Credentials:6:unique',

    }

    for node in attack_graph.nodes.values():
        if node.type == 'defense' and get_ttc_dict(node) != predef_ttcs['Disabled']:
            assert node.full_name not in expected_bernoulli_0_defenses

        if node.type == 'defense' and get_ttc_dict(node) == predef_ttcs['Disabled']:
            assert node.full_name in expected_bernoulli_0_defenses


def test_get_ttc_dict_attacksteps(corelang_lang_graph: LanguageGraph) -> None:
    """Make sure TTCs are set correctly for attacks"""

    model = Model.load_from_file(
        path_testdata("models/simple_example_model.yml"),
        corelang_lang_graph
    )
    attack_graph = AttackGraph(
        lang_graph=corelang_lang_graph, model=model
    )

    instant_steps = set(
        n for n in attack_graph.nodes.values()
        if n.type in ('or', 'and')
        and get_ttc_dict(n) == predef_ttcs['Instant']
    )
    assert len(instant_steps) == 454

    # Check some nodes that have diferent TTC
    bypass_sa = attack_graph.get_node_by_full_name('User:12:bypassSecurityAwareness')
    assert bypass_sa
    assert get_ttc_dict(bypass_sa) == predef_ttcs['VeryHardAndUncertain']
