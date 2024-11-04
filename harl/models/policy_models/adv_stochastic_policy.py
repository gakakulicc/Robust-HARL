import copy
import torch
import torch.nn as nn
from harl.utils.envs_tools import check
from harl.models.base.cnn import CNNBase
from harl.models.base.mlp import MLPBase
from harl.models.base.rnn import RNNLayer
from harl.models.base.act import ACTLayer
from harl.models.policy_models.stochastic_policy import StochasticPolicy
from harl.utils.envs_tools import get_shape_from_obs_space


class AdvStochasticPolicy(StochasticPolicy):
    def __init__(self, args, obs_space, action_space, num_agents, device=torch.device("cpu")):
        args = copy.deepcopy(args)
        if "adv_hidden_sizes" in args:
            args['hidden_sizes'] = args["adv_hidden_sizes"]
        super(AdvStochasticPolicy, self).__init__(args, obs_space, action_space, device)
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

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            self.rnn = RNNLayer(self.adv_hidden_sizes[-1], self.adv_hidden_sizes[-1],
                                self.recurrent_N, self.initialization_method)

        self.act = ACTLayer(action_space, self.adv_hidden_sizes[-1],
                            self.initialization_method, self.gain, args)

        self.to(device)