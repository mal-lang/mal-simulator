"""Unit tests for the probabilities utilities module"""
import numpy as np

from maltoolbox.model import Model
from maltoolbox.attackgraph.attackgraph import AttackGraph
from maltoolbox.language.languagegraph import LanguageGraph

from malsim.ttc_utils import TTCDist, named_ttc_dists

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
        if node.type not in ('exist', 'notExist'):
            TTCDist.from_node(node).sample_value(np.random.default_rng(10))

    for node in attack_graph.nodes.values():
        #TODO: Actually check some of the results
        if node.type not in ('exist', 'notExist'):
            TTCDist.from_node(node).expected_value



def test_bernoulli(model: Model) -> None:
    """Test bernoulli calculation for nodes"""

    attack_graph = AttackGraph(model.lang_graph, model)
    uncertain_step = attack_graph.get_node_by_full_name('IDPS 1:bypassContainerization')
    assert uncertain_step
    rng = np.random.default_rng(10)

    bernoulli_samples = (
        TTCDist.from_node(uncertain_step).attempt_bernoulli(rng) for _ in range(10)
    )
    # A step will give True
    assert True in bernoulli_samples
    # But it should also give False since it is uncertain
    assert False in bernoulli_samples


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
        if node.type in ('exist', 'notExist'):
            continue

        ttc_dist = TTCDist.from_node(node)
        if node.type == 'defense' and not (
            ttc_dist.function.name == "BERNOULLI" and ttc_dist.args == [0.0]
        ):
            assert node.full_name not in expected_bernoulli_0_defenses

        if node.type == 'defense' and (
            ttc_dist.function.name == "BERNOULLI" and ttc_dist.args == [0.0]
        ):
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
        and TTCDist.from_node(n) == named_ttc_dists['Instant']
    )
    assert len(instant_steps) == 454

    # Check some nodes that have diferent TTC
    bypass_sa = attack_graph.get_node_by_full_name('User:12:bypassSecurityAwareness')
    assert bypass_sa
    assert TTCDist.from_node(bypass_sa) == named_ttc_dists['VeryHardAndUncertain']


def test_ttcs_effort_based(model: Model) -> None:
    """Test effort based TTC calculation for nodes"""

    app = model.add_asset('Application', name='App')
    creds = model.add_asset('Credentials')
    user = model.add_asset('User', name='User1')
    identity = model.add_asset('Identity', name='User1Identity')
    vuln = model.add_asset('SoftwareVulnerability')

    identity.add_associated_assets('credentials', {creds})
    user.add_associated_assets('userIds', {identity})
    app.add_associated_assets('highPrivAppIAMs', {identity})
    app.add_associated_assets('vulnerabilities', {vuln})

    attack_graph = AttackGraph(model.lang_graph, model)

    rng = np.random.default_rng(10)
    very_hard_uncertain_node = attack_graph.get_node_by_full_name(
        'App:bypassSupplyChainAuditing'
    )
    assert very_hard_uncertain_node
    ttc_dist = TTCDist.from_node(very_hard_uncertain_node)
    assert round(ttc_dist.success_probability(1), 2) == 0.01 # success prob after 1 attempt is 1%
    assert not ttc_dist.attempt_ttc_with_effort(1, rng)      # 1 effort will not succeed in this seed
    assert ttc_dist.attempt_ttc_with_effort(500, rng)        # 500 effort will succeed in this seed

    hard_uncertain_node = attack_graph.get_node_by_full_name(
        'User1:credentialTheft'
    )
    assert hard_uncertain_node
    ttc_dist = TTCDist.from_node(hard_uncertain_node)
    assert round(ttc_dist.success_probability(1), 2) == 0.1 # success prob after 1 attempt is 10%
    assert not ttc_dist.attempt_ttc_with_effort(1, rng)     # 1 effort will not succeed in this seed
    assert ttc_dist.attempt_ttc_with_effort(50, rng)        # 50 effort will succeed in this seed

    instant_node = attack_graph.get_node_by_full_name('App:attemptRead')
    assert instant_node
    ttc_dist = TTCDist.from_node(instant_node)
    assert round(ttc_dist.success_probability(1), 2) == 1.0 # success prob after 1 attempt is 100%
    assert ttc_dist.attempt_ttc_with_effort(1, rng)         # 1 effort will succeed always (instant)

    exp_001_node = attack_graph.get_node_by_full_name('User1:deliverMaliciousRemovableMedia')
    assert exp_001_node
    ttc_dist = TTCDist.from_node(exp_001_node)
    assert round(ttc_dist.success_probability(1), 2) == 0.01 # success prob after 1 attempt is 1%
    assert not ttc_dist.attempt_ttc_with_effort(1, rng)      # 1 effort will not succeed in this seed
    assert ttc_dist.attempt_ttc_with_effort(500, rng)        # 500 effort will succeed in this seed
