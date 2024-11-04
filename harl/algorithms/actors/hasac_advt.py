"""HASAC algorithm."""
import torch
from harl.models.policy_models.squashed_gaussian_policy import SquashedGaussianPolicy
from harl.models.policy_models.stochastic_mlp_policy import StochasticMlpPolicy
from harl.utils.discrete_util import gumbel_softmax
from harl.utils.envs_tools import check
from harl.utils.models_tools import update_linear_schedule
from harl.algorithms.actors.off_policy_base import OffPolicyBase
from harl.models.policy_models.adv_stochastic_mlp_policy import AdvStochasticMlpPolicy


class HASACAdvt(OffPolicyBase):
    def __init__(self, args, obs_space, act_space, num_agents, device=torch.device("cpu")):
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.polyak = args["polyak"]
        self.lr = args["lr"]
        self.device = device
        self.action_type = act_space.__class__.__name__

        self.opti_eps = args["opti_eps"]
        self.weight_decay = args["weight_decay"]
        self.adv_lr = args["adv_lr"]
        self.super_adversary = args["super_adversary"]  # whether the adversary has defenders' policies
        self.belief = args["belief"]
        self.num_agents = num_agents

        if act_space.__class__.__name__ == "Box":
            self.actor = SquashedGaussianPolicy(args, obs_space, act_space, device)
            self.adv_actor = AdvStochasticMlpPolicy(args, obs_space, act_space, num_agents, device)
        else:
            self.actor = StochasticMlpPolicy(args, obs_space, act_space, device)
            self.adv_actor = AdvStochasticMlpPolicy(args, obs_space, act_space, num_agents, device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.adv_actor_optimizer = torch.optim.Adam(self.adv_actor.parameters(),
                                                    lr=self.adv_lr, eps=self.opti_eps,
                                                    weight_decay=self.weight_decay)
        self.turn_off_grad()

    def lr_decay(self, step, steps):
        """Decay the actor and critic learning rates.
        Args:
            step: (int) current training step.
            steps: (int) total number of training steps.
        """
        super().lr_decay(step, steps)
        update_linear_schedule(self.adv_actor_optimizer, step, steps, self.adv_lr)

    def get_adv_actions(self, obs, available_actions=None, stochastic=True, agent_id=0):
        """Get actions for observations.
        Args:
            obs: (np.ndarray) observations of actor, shape is (n_threads, dim) or (batch_size, dim)
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            stochastic: (bool) stochastic actions or deterministic actions
        Returns:
            actions: (torch.Tensor) actions taken by this actor, shape is (n_threads, dim) or (batch_size, dim)
        """
        obs = check(obs).to(**self.tpdv)
        actions, _ = self.adv_actor(obs, available_actions, stochastic)
        return actions

    def get_actions(self, obs, available_actions=None, stochastic=True, agent_id=0):
        """Get actions for observations.
        Args:
            obs: (np.ndarray) observations of actor, shape is (n_threads, dim) or (batch_size, dim)
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            stochastic: (bool) stochastic actions or deterministic actions
        Returns:
            actions: (torch.Tensor) actions taken by this actor, shape is (n_threads, dim) or (batch_size, dim)
        """
        obs = check(obs).to(**self.tpdv)
        if self.action_type == "Box":
            actions, _ = self.actor(obs, stochastic=stochastic, with_logprob=False)
        else:
            actions = self.actor(obs, available_actions, stochastic)
        return actions

    def get_adv_actions_with_logprobs(self, obs, available_actions=None, stochastic=True):
        """Get actions and logprobs of actions for observations.
        Args:
            obs: (np.ndarray) observations of actor, shape is (batch_size, dim)
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            stochastic: (bool) stochastic actions or deterministic actions
        Returns:
            actions: (torch.Tensor) actions taken by this actor, shape is (batch_size, dim)
            logp_actions: (torch.Tensor) log probabilities of actions taken by this actor, shape is (batch_size, 1)
        """
        obs = check(obs).to(**self.tpdv)
        if self.action_type == "MultiDiscrete":
            logits = self.adv_actor.get_logits(obs, available_actions)
            actions = []
            logp_actions = []
            for logit in logits:
                action = gumbel_softmax(
                    logit, hard=True, device=self.device
                )  # onehot actions
                logp_action = torch.sum(action * logit, dim=-1, keepdim=True)
                actions.append(action)
                logp_actions.append(logp_action)
            actions = torch.cat(actions, dim=-1)
            logp_actions = torch.cat(logp_actions, dim=-1)
        elif self.action_type == "Discrete":
            logits = self.adv_actor.get_logits(obs, available_actions)
            actions = gumbel_softmax(
                logits, hard=True, device=self.device
            )  # onehot actions
            logp_actions = torch.sum(actions * logits, dim=-1, keepdim=True)
        elif self.action_type == "Box":
            actions, logp_actions = self.adv_actor(
                obs, available_actions, stochastic=stochastic
            )

        return actions, logp_actions

    def get_actions_with_logprobs(self, obs, available_actions=None, stochastic=True):
        """Get actions and logprobs of actions for observations.
        Args:
            obs: (np.ndarray) observations of actor, shape is (batch_size, dim)
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            stochastic: (bool) stochastic actions or deterministic actions
        Returns:
            actions: (torch.Tensor) actions taken by this actor, shape is (batch_size, dim)
            logp_actions: (torch.Tensor) log probabilities of actions taken by this actor, shape is (batch_size, 1)
        """
        obs = check(obs).to(**self.tpdv)
        if self.action_type == "Box":
            actions, logp_actions = self.actor(
                obs, stochastic=stochastic, with_logprob=True
            )
        elif self.action_type == "Discrete":
            logits = self.actor.get_logits(obs, available_actions)
            actions = gumbel_softmax(
                logits, hard=True, device=self.device
            )  # onehot actions
            logp_actions = torch.sum(actions * logits, dim=-1, keepdim=True)
        elif self.action_type == "MultiDiscrete":
            logits = self.actor.get_logits(obs, available_actions)
            actions = []
            logp_actions = []
            for logit in logits:
                action = gumbel_softmax(
                    logit, hard=True, device=self.device
                )  # onehot actions
                logp_action = torch.sum(action * logit, dim=-1, keepdim=True)
                actions.append(action)
                logp_actions.append(logp_action)
            actions = torch.cat(actions, dim=-1)
            logp_actions = torch.cat(logp_actions, dim=-1)
        return actions, logp_actions

    def save(self, save_dir, id):
        """Save the actor."""
        torch.save(
            self.actor.state_dict(), str(save_dir) + "/actor_agent" + str(id) + ".pt"
        )
        torch.save(
            self.adv_actor.state_dict(), str(save_dir) + "/adv_actor_agent" + str(id) + ".pt"
        )

    def restore(self, model_dir, id):
        """Restore the actor."""
        actor_state_dict = torch.load(str(model_dir) + "/actor_agent" + str(id) + ".pt")
        self.actor.load_state_dict(actor_state_dict)

    def adv_restore(self, adv_model_dir, id):
        """Restore the actor."""
        adv_actor_state_dict = torch.load(str(adv_model_dir) + "/adv_actor_agent" + str(id) + ".pt")
        self.adv_actor.load_state_dict(adv_actor_state_dict)


    def turn_on_grad(self):
        """Turn on grad for actor parameters."""
        for p in self.actor.parameters():
            p.requires_grad = True
        for p in self.adv_actor.parameters():
            p.requires_grad = True

    def turn_off_grad(self):
        """Turn off grad for actor parameters."""
        for p in self.actor.parameters():
            p.requires_grad = False
        for p in self.adv_actor.parameters():
            p.requires_grad = False