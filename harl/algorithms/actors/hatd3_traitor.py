import torch
from harl.utils.envs_tools import check
from harl.algorithms.actors.hasac_traitor import HASACTraitor
from harl.algorithms.actors.hatd3 import HATD3


class HATD3Traitor(HASACTraitor):
    def __init__(self, args, obs_space, act_space, num_agents, device=torch.device("cpu")):
        super(HATD3Traitor, self).__init__(args, obs_space, act_space, num_agents, device)
        self.hatd3 = HATD3(args, obs_space, act_space, device)

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

    def get_logits(self, obs, available_actions=None,
                    deterministic=False, agent_id=0):
        """Compute actions and value function predictions for the given inputs.
        Args:
            obs (np.ndarray): local agent inputs to the actor.
            rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        obs = check(obs).to(**self.tpdv)
        actions = self.hatd3.actor(obs)

        if self.actor.action_type == "Discrete":
            if available_actions is not None:
                available_actions = check(available_actions).to(**self.tpdv)
                actions = actions.clone()
                actions[available_actions == 0] = -1e10

        return actions

    def act(self, obs, available_actions=None, deterministic=False, agent_id=0):
        """Compute actions using the given inputs.
        Args:
            obs (np.ndarray): local agent inputs to the actor.
            rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray) denotes which actions are available to agent
                                    (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        obs = check(obs).to(**self.tpdv)
        actions = self.hatd3.actor(obs)
        return actions