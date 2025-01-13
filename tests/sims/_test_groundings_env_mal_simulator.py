"""Test MalSimulator class"""

from maltoolbox.attackgraph import AttackGraph, Attacker
from malsim.sims import MalSimulator, MalSimGroundingsEnv, MalSimAttacker
from malsim.scenario import load_scenario


def test_malsimulator_observe_and_reward_attacker_defender():
    """Run attacker and defender actions and make sure
    rewards and observation states are updated correctly"""

    def verify_defender_obs_state(groundings):
        """Make sure obs state looks as expected"""
        for grounding, state in groundings.items():
            node = env._grounding_name_to_object[grounding]
            if state is True:
                assert node.is_compromised() or node.is_enabled_defense()
            elif state is False:
                assert (
                    not node.is_compromised()
                    and not node.is_enabled_defense()
                ), f"{node.full_name} not correct state {state}"

    attack_graph, _ = load_scenario(
        'tests/testdata/scenarios/traininglang_scenario.yml')
    # Create the simulator
    env = MalSimGroundingsEnv(MalSimulator(attack_graph))

    attacker = env.sim.attack_graph.attackers[0]
    attacker_agent_name = "Attacker1"
    env.register_attacker(attacker_agent_name, attacker.id)

    defender_agent_name = "Defender1"
    env.register_defender(defender_agent_name)

    # Prepare nodes that will be stepped through in order
    user_3_compromise = env.sim.attack_graph\
        .get_node_by_full_name("User:3:compromise")
    host_0_authenticate = env.sim.attack_graph\
        .get_node_by_full_name("Host:0:authenticate")
    host_0_access = env.sim.attack_graph\
        .get_node_by_full_name("Host:0:access")
    host_0_notPresent = env.sim.attack_graph\
        .get_node_by_full_name("Host:0:notPresent")
    data_2_read = env.sim.attack_graph\
        .get_node_by_full_name("Data:2:read")

    # Step with attacker action
    user_3_compromise_grounding = \
        env.obs_model.attack_step_node_to_grounding_name(user_3_compromise)

    groundings = env.step({
        defender_agent_name: {},
        attacker_agent_name: {
            user_3_compromise_grounding: True
        }
    })

    # Verify obs state
    verify_defender_obs_state(groundings)

    # Step with attacker again
    host_0_authenticate_grounding = \
        env.obs_model.attack_step_node_to_grounding_name(host_0_authenticate)

    groundings = env.step({
            defender_agent_name: {},
            attacker_agent_name: {host_0_authenticate_grounding: True}
        }
    )

    # Verify obs state
    verify_defender_obs_state(groundings)

    # Step attacker again
    host_0_access_grounding = \
        env.obs_model.attack_step_node_to_grounding_name(host_0_access)

    groundings = env.step({
            defender_agent_name: {},
            attacker_agent_name: {host_0_access_grounding: True}
        }
    )

    # Verify obs state
    verify_defender_obs_state(groundings)

    # Step defender and attacker
    # Attacker wont be able to traverse Data:2:read since
    # Host:0:notPresent is activated before
    host_0_notPresent_grounding = \
        env.obs_model.attack_step_node_to_grounding_name(host_0_notPresent)
    data_2_read_grounding = \
        env.obs_model.attack_step_node_to_grounding_name(data_2_read)

    groundings = env.step({
            defender_agent_name: {host_0_notPresent_grounding: True},
            attacker_agent_name: {data_2_read_grounding: True}
        }
    )

    # Attacker obs state should look the same as before
    verify_defender_obs_state(groundings)
