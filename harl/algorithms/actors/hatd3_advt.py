import torch
from harl.utils.envs_tools import check
from harl.algorithms.actors.hasac_advt import HASACAdvt
from harl.algorithms.actors.hatd3 import HATD3


class HATD3Advt(HASACAdvt):
    def __init__(self, args, obs_space, act_space, num_agents, device=torch.device("cpu")):
        self.hatd3 = HATD3(args, obs_space, act_space, device)
        super(HATD3Advt, self).__init__(args, obs_space, act_space, num_agents, device)


    def get_actions(self, obs, add_noise):
        """Get actions for observations.
        Args:
            obs: (np.ndarray) observations of actor, shape is (n_threads, dim) or (batch_size, dim)
            add_noise: (bool) whether to add noise
        Returns:
            actions: (torch.Tensor) actions taken by this actor, shape is (n_threads, dim) or (batch_size, dim)
        """
        obs = check(obs).to(**self.tpdv)
        actions = self.hatd3.get_actions(obs, add_noise)
        return actions

    def get_target_actions(self, obs):
        """Get target actor actions for observations.
        Args:
            obs: (np.ndarray) observations of target actor, shape is (batch_size, dim)
        Returns:
            actions: (torch.Tensor) actions taken by target actor, shape is (batch_size, dim)
        """
        obs = check(obs).to(**self.tpdv)
        return self.hatd3.get_target_actions(obs)

    def soft_update(self):
        """Soft update target actor."""
        self.hatd3.soft_update()

    def turn_on_grad(self):
        """Turn on grad for actor parameters."""
        self.hatd3.turn_on_grad()
        for p in self.adv_actor.parameters():
            p.requires_grad = True

    def turn_off_grad(self):
        """Turn off grad for actor parameters."""
        self.hatd3.turn_off_grad()
        for p in self.adv_actor.parameters():
            p.requires_grad = False