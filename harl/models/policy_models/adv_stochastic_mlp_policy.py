import copy
import torch
import torch.nn as nn
from harl.utils.envs_tools import check
from harl.models.base.cnn import CNNBase
from harl.models.base.mlp import MLPBase
from harl.models.base.rnn import RNNLayer
from harl.models.base.act import ACTLayer
from harl.models.policy_models.stochastic_mlp_policy import StochasticMlpPolicy
from harl.utils.envs_tools import get_shape_from_obs_space


class AdvStochasticMlpPolicy(StochasticMlpPolicy):
    def __init__(self, args, obs_space, action_space, num_agents, device=torch.device("cpu")):
        args = copy.deepcopy(args)
        if "adv_hidden_sizes" in args:
            args['hidden_sizes'] = args["adv_hidden_sizes"]
        super(AdvStochasticMlpPolicy, self).__init__(args, obs_space, action_space, device)
        self.super_adversary = args["super_adversary"]  # whether the adversary has defenders' policies
        self.adv_hidden_sizes = args["hidden_sizes"]
        # self.adv_hidden_sizes = args["adv_hidden_sizes"]
        # print(self.adv_hidden_sizes)
        # exit(0)
        self.obs_offset = args.get("obs_offset", 0)
        obs_shape = copy.deepcopy(get_shape_from_obs_space(obs_space))

        if self.super_adversary:
            obs_shape[0] = obs_shape[0] + (num_agents - 1) * get_shape_from_act_space(action_space)
        obs_shape[0] = obs_shape[0] + self.obs_offset
        self.obs_len = obs_shape[0]

        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        self.act = ACTLayer(action_space, self.adv_hidden_sizes[-1],
                            self.initialization_method, self.gain, args)

        self.to(device)

    def forward(self, obs, available_actions=None, stochastic=True):
        """Compute actions from the given inputs.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            stochastic: (bool) whether to sample from action distribution or return the mode.
        Returns:
            actions: (torch.Tensor) actions to take.
        """
        obs = check(obs).to(**self.tpdv)
        deterministic = not stochastic
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic
        )
        return actions, action_log_probs