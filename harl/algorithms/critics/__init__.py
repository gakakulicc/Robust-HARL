"""Critic registry."""
from harl.algorithms.critics.v_critic import VCritic
from harl.algorithms.critics.continuous_q_critic import ContinuousQCritic
from harl.algorithms.critics.twin_continuous_q_critic import TwinContinuousQCritic
from harl.algorithms.critics.soft_twin_continuous_q_critic import (
    SoftTwinContinuousQCritic,
)
from harl.algorithms.critics.discrete_q_critic import DiscreteQCritic

CRITIC_REGISTRY = {
    "happo": VCritic,
    "happo_advt": VCritic,
    "happo_traitor": VCritic,
    "hatrpo": VCritic,
    "haa2c": VCritic,
    "mappo": VCritic,
    "mappo_advt": VCritic,
    "mappo_traitor": VCritic,
    "haddpg": ContinuousQCritic,
    "haddpg_advt": ContinuousQCritic,
    "haddpg_traitor": ContinuousQCritic,
    "hatd3": TwinContinuousQCritic,
    "hatd3_advt": TwinContinuousQCritic,
    "hatd3_traitor": TwinContinuousQCritic,
    "hasac": SoftTwinContinuousQCritic,
    "masac": SoftTwinContinuousQCritic,
    "hasac_advt": SoftTwinContinuousQCritic,
    "hasac_traitor": SoftTwinContinuousQCritic,
    "had3qn": DiscreteQCritic,
    "maddpg": ContinuousQCritic,
    "matd3": TwinContinuousQCritic,
}
