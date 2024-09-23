from dataclasses import dataclass

@dataclass
class MalSimulatorSettings():
    """Contains settings used in MalSimulator"""
    # Attacker observation settings
    evict_attacker_from_defended_step: bool = False
    # Defender observation settings
    remember_previous_steps_in_defender_obs: bool = True
