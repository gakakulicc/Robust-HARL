import numpy as np
import torch
import torch.nn as nn
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm, update_linear_schedule
from harl.utils.trans_tools import softmax
from harl.models.policy_models.stochastic_policy import StochasticPolicy


class MAPPOState:
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize Base class.
        Args:
            args: (dict) arguments.
            obs_space: (gym.spaces or list) observation space.
            act_space: (gym.spaces) action space.
            device: (torch.device) device to use for tensor operations.
        """
        # save arguments
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.data_chunk_length = args["data_chunk_length"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.action_aggregation = args["action_aggregation"]

        self.lr = args["lr"]
        self.opti_eps = args["opti_eps"]
        self.weight_decay = args["weight_decay"]
        # save observation and action spaces
        self.obs_space = obs_space
        self.act_space = act_space
        print(self.obs_space)
        # create actor network
        self.actor = StochasticPolicy(args, self.obs_space, self.act_space, self.device)
        self.noise_actor = StochasticPolicy(args, self.obs_space, self.obs_space, self.device)
        # create actor optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        self.noise_actor_optimizer = torch.optim.Adam(
            self.noise_actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]
        self.obs_epsilon = args["obs_epsilon"]

    def lr_decay(self, episode, episodes):
        """Decay the learning rates.
        Args:
            episode: (int) current training episode.
            episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.noise_actor_optimizer, episode, episodes, self.lr)

    def get_actions(
        self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False, agent_id=0
    ):
        """Compute actions for the given inputs.
        Args:
            obs: (np.ndarray) local agent inputs to the actor.
            rnn_states_actor: (np.ndarray) if actor has RNN layer, RNN states for actor.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        obs_noise, obs_noise_log_probs, _ = self.noise_actor(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )
        obs_noise_numpy = obs_noise.detach().cpu().numpy()
        actions, action_log_probs, rnn_states_actor = self.actor(
            obs + self.obs_epsilon * obs_noise_numpy, rnn_states_actor, masks, available_actions, deterministic
        )
        return obs_noise, obs_noise_log_probs, actions, action_log_probs, rnn_states_actor

    def evaluate_actions(
            self,
            obs,
            rnn_states_actor,
            action,
            masks,
            available_actions=None,
            active_masks=None,
    ):
        """Get action logprobs, entropy, and distributions for actor update.
        Args:
            obs: (np.ndarray / torch.Tensor) local agent inputs to the actor.
            rnn_states_actor: (np.ndarray / torch.Tensor) if actor has RNN layer, RNN states for actor.
            action: (np.ndarray / torch.Tensor) actions whose log probabilities and entropy to compute.
            masks: (np.ndarray / torch.Tensor) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                    (if None, all actions available)
            active_masks: (np.ndarray / torch.Tensor) denotes whether an agent is active or dead.
        """

        (
            action_log_probs,
            dist_entropy,
            action_distribution,
        ) = self.actor.evaluate_actions(
            obs, rnn_states_actor, action, masks, available_actions, active_masks
        )
        return action_log_probs, dist_entropy, action_distribution

    def act(
        self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False, agent_id=0
    ):
        """Compute actions using the given inputs.
        Args:
            obs: (np.ndarray) local agent inputs to the actor.
            rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray) denotes which actions are available to agent
                                    (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )
        return actions, rnn_states_actor

    def update(self, sample):
        """Update actor network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
        Returns:
            policy_loss: (torch.Tensor) actor(policy) loss value.
            dist_entropy: (torch.Tensor) action entropies.
            actor_grad_norm: (torch.Tensor) gradient norm from actor update.
            imp_weights: (torch.Tensor) importance sampling weights.
        """
        (
            obs_batch,
            obs_noise_batch,
            old_obs_noise_log_probs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
        ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        old_obs_noise_log_probs_batch = check(old_obs_noise_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)

        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # reshape to do in a single forward pass for all steps
        action_log_probs, dist_entropy, _ = self.evaluate_actions(
            obs_batch + self.obs_epsilon * obs_noise_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )
        # update actor
        imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1,
            keepdim=True,
        )

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self.use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.actor_optimizer.zero_grad()

        (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())

        self.actor_optimizer.step()

        obs_noise_log_probs_batch, noise_dist_entropy, _ = self.noise_actor.evaluate_actions(
            obs_batch,
            rnn_states_batch,
            obs_noise_batch,
            masks_batch,
            None,
            active_masks_batch,
        )

        # update actor
        noise_imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(obs_noise_log_probs_batch - old_obs_noise_log_probs_batch),
            dim=-1,
            keepdim=True,
        )

        noise_surr1 = - noise_imp_weights * adv_targ
        noise_surr2 = - (
                torch.clamp(noise_imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
                * adv_targ
        )

        if self.use_policy_active_masks:
            noise_policy_action_loss = (
                                               -torch.sum(torch.min(noise_surr1, noise_surr2), dim=-1, keepdim=True)
                                               * active_masks_batch
                                       ).sum() / active_masks_batch.sum()
        else:
            noise_policy_action_loss = -torch.sum(
                torch.min(noise_surr1, noise_surr2), dim=-1, keepdim=True
            ).mean()

        self.noise_actor_optimizer.zero_grad()

        (noise_policy_action_loss - noise_dist_entropy * self.entropy_coef).backward()

        if self.use_max_grad_norm:
            noise_actor_grad_norm = nn.utils.clip_grad_norm_(
                self.noise_actor.parameters(), self.max_grad_norm
            )
        else:
            noise_actor_grad_norm = get_grad_norm(self.noise_actor.parameters())

        self.noise_actor_optimizer.step()

        return policy_loss, dist_entropy, actor_grad_norm, imp_weights, \
            noise_policy_action_loss, noise_dist_entropy, noise_actor_grad_norm, noise_imp_weights

    def train(self, actor_buffer, advantages, state_type):
        """Perform a training update for non-parameter-sharing MAPPO using minibatch GD.
        Args:
            actor_buffer: (OnPolicyActorBuffer) buffer containing training data related to actor.
            advantages: (np.ndarray) advantages.
            state_type: (str) type of state.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0

        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return train_info

        if state_type == "EP":
            advantages_copy = advantages.copy()
            advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = actor_buffer.recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch, self.data_chunk_length
                )
            elif self.use_naive_recurrent_policy:
                data_generator = actor_buffer.naive_recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch
                )
            else:
                data_generator = actor_buffer.feed_forward_generator_actor(
                    advantages, self.actor_num_mini_batch
                )

            for sample in data_generator:
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(sample)

                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def share_param_train(self, actor_buffer, advantages, num_agents, state_type):
        """Perform a training update for parameter-sharing MAPPO using minibatch GD.
        Args:
            actor_buffer: (list[OnPolicyActorBuffer]) buffer containing training data related to actor.
            advantages: (np.ndarray) advantages.
            num_agents: (int) number of agents.
            state_type: (str) type of state.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['ratio'] = 0

        train_info['noise_policy_loss'] = 0
        train_info['noise_dist_entropy'] = 0
        train_info['noise_actor_grad_norm'] = 0
        train_info['noise_ratio'] = 0

        if state_type == "EP":
            advantages_ori_list = []
            advantages_copy_list = []
            for agent_id in range(num_agents):
                advantages_ori = advantages.copy()
                advantages_ori_list.append(advantages_ori)
                advantages_copy = advantages.copy()
                advantages_copy[actor_buffer[agent_id].active_masks[:-1] == 0.0] = np.nan
                advantages_copy_list.append(advantages_copy)
            advantages_ori_tensor = np.array(advantages_ori_list)
            advantages_copy_tensor = np.array(advantages_copy_list)
            mean_advantages = np.nanmean(advantages_copy_tensor)
            std_advantages = np.nanstd(advantages_copy_tensor)
            normalized_advantages = (advantages_ori_tensor - mean_advantages) / (
                std_advantages + 1e-5
            )
            advantages_list = []
            for agent_id in range(num_agents):
                advantages_list.append(normalized_advantages[agent_id])
        elif state_type == "FP":
            advantages_list = []
            for agent_id in range(num_agents):
                advantages_list.append(advantages[:, :, agent_id])

        for _ in range(self.ppo_epoch):
            data_generators = []
            for agent_id in range(num_agents):
                if self.use_recurrent_policy:
                    data_generator = actor_buffer[agent_id].recurrent_generator_actor(
                        advantages_list[agent_id],
                        self.actor_num_mini_batch,
                        self.data_chunk_length,
                    )
                elif self.use_naive_recurrent_policy:
                    data_generator = actor_buffer[agent_id].naive_recurrent_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch
                    )
                else:
                    data_generator = actor_buffer[agent_id].feed_forward_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch
                    )
                data_generators.append(data_generator)

            for _ in range(self.actor_num_mini_batch):
                batches = [[] for _ in range(10)]
                for generator in data_generators:
                    sample = next(generator)
                    for i in range(10):
                        batches[i].append(sample[i])
                for i in range(9):
                    batches[i] = np.concatenate(batches[i], axis=0)
                if batches[9][0] is None:
                    batches[9] = None
                else:
                    batches[9] = np.concatenate(batches[9], axis=0)
                policy_loss, dist_entropy, actor_grad_norm, imp_weights, \
                    noise_policy_loss, noise_dist_entropy, noise_actor_grad_norm, noise_imp_weights = self.update(
                    tuple(batches))

                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['ratio'] += imp_weights.mean()

                train_info['noise_policy_loss'] += noise_policy_loss.item()
                train_info['noise_dist_entropy'] += noise_dist_entropy.item()
                train_info['noise_actor_grad_norm'] += noise_actor_grad_norm
                train_info['noise_ratio'] += noise_imp_weights.mean()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        """Prepare for training."""
        self.actor.train()

    def prep_rollout(self):
        """Prepare for rollout."""
        self.actor.eval()