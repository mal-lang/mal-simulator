from dataclasses import dataclass

@dataclass
class MalSimulatorSettings():
    """Contains settings used in MalSimulator"""

    # uncompromise_untraversable_steps
    # - Uncompromise (evict attacker) from nodes/steps that are no longer
    #   traversable (often because a defense kicked in) if set to True
    # otherwise:
    # - Leave the node/step compromised even after it becomes untraversable
    uncompromise_untraversable_steps: bool = False

    # cumulative_defender_obs
    # - Defender sees the status of the whole attack graph if set to True
    # otherwise:
    # - Defender only sees the status of nodes changed in the current step
    cumulative_defender_obs: bool = True
