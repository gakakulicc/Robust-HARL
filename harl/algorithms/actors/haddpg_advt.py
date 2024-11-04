from copy import deepcopy
import torch
from harl.utils.envs_tools import check
from harl.algorithms.actors.hasac_advt import HASACAdvt
from harl.algorithms.actors.haddpg import HADDPG


class HADDPGAdvt(HASACAdvt):
    def __init__(self, args, obs_space, act_space, num_agents, device=torch.device("cpu")):
        super(HADDPGAdvt, self).__init__(args, obs_space, act_space, num_agents, device)
        self.haddpg = HADDPG(args, obs_space, act_space, device)

    def get_actions(self, obs, add_noise):
        """Get actions for observations.
        Args:
            obs: (np.ndarray) observations of actor, shape is (n_threads, dim) or (batch_size, dim)
            add_noise: (bool) whether to add noise
        Returns:
            actions: (torch.Tensor) actions taken by this actor, shape is (n_threads, dim) or (batch_size, dim)
        """
        obs = check(obs).to(**self.tpdv)
        actions = self.haddpg.get_actions(obs, add_noise)
        return actions

    def get_target_actions(self, obs):
        """Get target actor actions for observations.
        Args:
            obs: (np.ndarray) observations of target actor, shape is (batch_size, dim)
        Returns:
            actions: (torch.Tensor) actions taken by target actor, shape is (batch_size, dim)
        """
        obs = check(obs).to(**self.tpdv)
        return self.haddpg.get_target_actions(obs)