"""On-policy buffer for actor."""

import torch
import numpy as np
from harl.utils.trans_tools import _flatten, _sa_cast
from harl.utils.envs_tools import get_shape_from_obs_space, get_shape_from_act_space


class OnPolicyActorBufferAdvt:
    """On-policy buffer for actor data storage."""

    def __init__(self, args, obs_space, act_space):
        """Initialize on-policy actor buffer.
        Args:
            args: (dict) arguments
            obs_space: (gym.Space or list) observation space
            act_space: (gym.Space) action space
        """
        self.episode_length = args["episode_length"]
        self.n_rollout_threads = args["n_rollout_threads"]
        self.hidden_sizes = args["hidden_sizes"]
        self.rnn_hidden_size = self.hidden_sizes[-1]
        self.recurrent_n = args["recurrent_n"]

        obs_shape = get_shape_from_obs_space(obs_space)

        if isinstance(obs_shape[-1], list):
            obs_shape = obs_shape[:1]

        # Buffer for observations of this actor.
        self.obs = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, *obs_shape),
            dtype=np.float32,
        )

        # Buffer for rnn states of this actor.
        self.rnn_states = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )
        self.adv_rnn_states = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        # Buffer for available actions of this actor.
        if act_space.__class__.__name__ == "Discrete":
            self.available_actions = np.ones(
                (self.episode_length + 1, self.n_rollout_threads, act_space.n),
                dtype=np.float32,
            )
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        # Buffer for actions of this actor.
        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32
        )
        self.adv_actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32
        )

        # Buffer for action log probs of this actor.
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32
        )
        self.adv_action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32
        )

        self.rewards = np.ones((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        # Buffer for masks of this actor. Masks denotes at which point should the rnn states be reset.
        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)

        # Buffer for active masks of this actor. Active masks denotes whether the agent is alive.
        self.active_masks = np.ones_like(self.masks)
        self.adv_active_masks = np.ones_like(self.masks)

        self.factor = None

        self.step = 0

    def update_factor(self, factor):
        """Save factor for this actor."""
        self.factor = factor.copy()

    def insert(
        self,
        obs,
        rnn_states,
        adv_rnn_states,
        actions,
        adv_actions,
        action_log_probs,
        adv_action_log_probs,
        rewards,
        masks,
        active_masks=None,
        adv_active_masks=None,
        available_actions=None,
    ):
        """Insert data into actor buffer."""
        self.obs[self.step + 1] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.adv_rnn_states[self.step + 1] = adv_rnn_states.copy()
        self.actions[self.step] = actions.copy()
        self.adv_actions[self.step] = adv_actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.adv_action_log_probs[self.step] = adv_action_log_probs.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if adv_active_masks is not None:
            self.adv_active_masks[self.step + 1] = adv_active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """After an update, copy the data at the last step to the first position of the buffer."""
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.adv_rnn_states[0] = self.adv_rnn_states[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        self.adv_active_masks[0] = self.adv_active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def feed_forward_generator_actor(
        self, advantages, actor_num_mini_batch=None, mini_batch_size=None
    ):
        """Training data generator for actor that uses MLP network."""

        # get episode_length, n_rollout_threads, mini_batch_size
        episode_length, n_rollout_threads = self.actions.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        if mini_batch_size is None:
            assert batch_size >= actor_num_mini_batch, (
                f"The number of processes ({n_rollout_threads}) "
                f"* the number of steps ({episode_length}) = {n_rollout_threads * episode_length}"
                f" is required to be greater than or equal to the number of actor mini batches ({actor_num_mini_batch})."
            )
            mini_batch_size = batch_size // actor_num_mini_batch

        # shuffle indices
        rand = torch.randperm(batch_size).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(actor_num_mini_batch)
        ]

        # Combine the first two dimensions (episode_length and n_rollout_threads) to form batch.
        # Take obs shape as an example:
        # (episode_length + 1, n_rollout_threads, *obs_shape) --> (episode_length, n_rollout_threads, *obs_shape)
        # --> (episode_length * n_rollout_threads, *obs_shape)
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        obs_next = self.obs[1:].reshape(-1, *self.obs.shape[2:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])  # actually not used, just for consistency
        adv_rnn_states = self.adv_rnn_states[:-1].reshape(-1, *self.adv_rnn_states.shape[2:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        adv_actions = self.adv_actions.reshape(-1, self.adv_actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(
                -1, self.available_actions.shape[-1]
            )
        rewards = self.rewards.reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        adv_active_masks = self.adv_active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        adv_action_log_probs = self.adv_action_log_probs.reshape(-1, self.adv_action_log_probs.shape[-1])
        if self.factor is not None:
            factor = self.factor.reshape(-1, self.factor.shape[-1])
        advantages = advantages.reshape(-1, 1)


        for indices in sampler:
            # obs shape:
            # (episode_length * n_rollout_threads, *obs_shape) --> (mini_batch_size, *obs_shape)
            obs_batch = obs[indices]
            obs_next_batch = obs_next[indices]
            rnn_states_batch = rnn_states[indices]
            adv_rnn_states_batch = adv_rnn_states[indices]
            actions_batch = actions[indices]
            adv_actions_batch = adv_actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            rewards_batch = rewards[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            adv_active_masks_batch = adv_active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            old_adv_action_log_probs_batch = adv_action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            if self.factor is None:
                yield obs_batch, obs_next_batch, rnn_states_batch, adv_rnn_states_batch, actions_batch, adv_actions_batch, rewards_batch, masks_batch, active_masks_batch, adv_active_masks_batch, old_action_log_probs_batch, old_adv_action_log_probs_batch, adv_targ, available_actions_batch
            else:
                factor_batch = factor[indices]
                yield obs_batch, obs_next_batch, rnn_states_batch, adv_rnn_states_batch, actions_batch, adv_actions_batch, rewards_batch, masks_batch, active_masks_batch, adv_active_masks_batch, old_action_log_probs_batch, old_adv_action_log_probs_batch, adv_targ, available_actions_batch, factor_batch

    def naive_recurrent_generator_actor(self, advantages, actor_num_mini_batch):
        """Training data generator for actor that uses RNN network.
        This generator does not split the trajectories into chunks,
        and therefore maybe less efficient than the recurrent_generator_actor in training.
        """
        raise NotImplementedError
        # get n_rollout_threads and num_envs_per_batch
        n_rollout_threads = self.actions.shape[1]
        assert n_rollout_threads >= actor_num_mini_batch, (
            f"The number of processes ({n_rollout_threads}) "
            f"has to be greater than or equal to the number of "
            f"mini batches ({actor_num_mini_batch})."
        )
        num_envs_per_batch = n_rollout_threads // actor_num_mini_batch

        # shuffle indices
        perm = torch.randperm(n_rollout_threads).numpy()

        T, N = self.episode_length, num_envs_per_batch

        # prepare data for each mini batch
        for batch_id in range(actor_num_mini_batch):
            start_id = batch_id * num_envs_per_batch
            ids = perm[start_id : start_id + num_envs_per_batch]
            obs_batch = _flatten(T, N, self.obs[:-1, ids])
            actions_batch = _flatten(T, N, self.actions[:, ids])
            masks_batch = _flatten(T, N, self.masks[:-1, ids])
            active_masks_batch = _flatten(T, N, self.active_masks[:-1, ids])
            old_action_log_probs_batch = _flatten(T, N, self.action_log_probs[:, ids])
            adv_targ = _flatten(T, N, advantages[:, ids])
            if self.available_actions is not None:
                available_actions_batch = _flatten(T, N, self.available_actions[:-1, ids])
            else:
                available_actions_batch = None
            if self.factor is not None:
                factor_batch = _flatten(T, N, self.factor[:, ids])
            rnn_states_batch = self.rnn_states[0, ids]

            if self.factor is not None:
                yield obs_batch, rnn_states_batch, actions_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch, factor_batch
            else:
                yield obs_batch, rnn_states_batch, actions_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch

    def recurrent_generator_actor(self, advantages, actor_num_mini_batch, data_chunk_length):
        """Training data generator for actor that uses RNN network.
        This generator splits the trajectories into chunks of length data_chunk_length,
        and therefore maybe more efficient than the naive_recurrent_generator_actor in training.
        """

        # get episode_length, n_rollout_threads, and mini_batch_size
        episode_length, n_rollout_threads = self.actions.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length
        mini_batch_size = data_chunks // actor_num_mini_batch

        assert episode_length % data_chunk_length == 0, (
            f"episode length ({episode_length}) must be a multiple of data chunk length ({data_chunk_length})."
        )
        assert data_chunks >= 2, "need larger batch size"

        # shuffle indices
        rand = torch.randperm(data_chunks).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(actor_num_mini_batch)
        ]

        # The following data operations first transpose the first two dimensions of the data (episode_length, n_rollout_threads)
        # to (n_rollout_threads, episode_length), then reshape the data to (n_rollout_threads * episode_length, *dim).
        # Take obs shape as an example:
        # (episode_length + 1, n_rollout_threads, *obs_shape) --> (episode_length, n_rollout_threads, *obs_shape)
        # --> (n_rollout_threads, episode_length, *obs_shape) --> (n_rollout_threads * episode_length, *obs_shape)
        if len(self.obs.shape) > 3:
            obs = self.obs[:-1].transpose(1, 0, 2, 3, 4).reshape(-1, *self.obs.shape[2:])
            obs_next = self.obs[1:].transpose(1, 0, 2, 3, 4).reshape(-1, *self.obs.shape[2:])
        else:
            obs = _sa_cast(self.obs[:-1])
            obs_next = _sa_cast(self.obs[1:])
        actions = _sa_cast(self.actions)
        adv_actions = _sa_cast(self.adv_actions)
        action_log_probs = _sa_cast(self.action_log_probs)
        adv_action_log_probs = _sa_cast(self.adv_action_log_probs)
        advantages = _sa_cast(advantages)
        rewards = _sa_cast(self.rewards)
        masks = _sa_cast(self.masks[:-1])
        active_masks = _sa_cast(self.active_masks[:-1])
        adv_active_masks = _sa_cast(self.adv_active_masks[:-1])
        if self.factor is not None:
            factor = _sa_cast(self.factor)
        rnn_states = (
            self.rnn_states[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.rnn_states.shape[2:])
        )
        adv_rnn_states = self.adv_rnn_states[:-1].transpose(
            1, 0, 2, 3).reshape(-1, *self.adv_rnn_states.shape[2:])
        if self.available_actions is not None:
            available_actions = _sa_cast(self.available_actions[:-1])

        # generate mini-batches
        for indices in sampler:
            obs_batch = []
            obs_next_batch = []
            rnn_states_batch = []
            adv_rnn_states_batch = []
            actions_batch = []
            adv_actions_batch = []
            available_actions_batch = []
            rewards_batch = []
            masks_batch = []
            active_masks_batch = []
            adv_active_masks_batch = []
            old_action_log_probs_batch = []
            old_adv_action_log_probs_batch = []
            adv_targ = []
            factor_batch = []
            for index in indices:
                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N Dim]-->[N T Dim]-->[T*N,Dim]-->[L,Dim]
                obs_batch.append(obs[ind:ind + data_chunk_length])
                obs_next_batch.append(obs_next[ind:ind + data_chunk_length])
                actions_batch.append(actions[ind:ind + data_chunk_length])
                adv_actions_batch.append(adv_actions[ind:ind + data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(
                        available_actions[ind:ind + data_chunk_length])
                rewards_batch.append(rewards[ind:ind + data_chunk_length])
                masks_batch.append(masks[ind:ind + data_chunk_length])
                active_masks_batch.append(
                    active_masks[ind:ind + data_chunk_length])
                adv_active_masks_batch.append(
                    adv_active_masks[ind:ind + data_chunk_length])
                old_action_log_probs_batch.append(
                    action_log_probs[ind:ind + data_chunk_length])
                old_adv_action_log_probs_batch.append(
                    adv_action_log_probs[ind:ind + data_chunk_length])
                adv_targ.append(advantages[ind:ind + data_chunk_length])
                # size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[1,Dim]
                rnn_states_batch.append(rnn_states[ind])
                adv_rnn_states_batch.append(adv_rnn_states[ind])
                if self.factor is not None:
                    factor_batch.append(factor[ind:ind + data_chunk_length])
            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (N, L, Dim)
            obs_batch = np.stack(obs_batch, axis=1)
            obs_next_batch = np.stack(obs_next_batch, axis=1)

            actions_batch = np.stack(actions_batch, axis=1)
            adv_actions_batch = np.stack(adv_actions_batch, axis=1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, axis=1)
            if self.factor is not None:
                factor_batch = np.stack(factor_batch, axis=1)
            rewards_batch = np.stack(rewards_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            adv_active_masks_batch = np.stack(adv_active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            old_adv_action_log_probs_batch = np.stack(old_adv_action_log_probs_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(
                N, *self.rnn_states.shape[2:])
            adv_rnn_states_batch = np.stack(adv_rnn_states_batch).reshape(
                N, *self.adv_rnn_states.shape[2:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            obs_batch = _flatten(L, N, obs_batch)
            obs_next_batch = _flatten(L, N, obs_next_batch)
            actions_batch = _flatten(L, N, actions_batch)
            adv_actions_batch = _flatten(L, N, adv_actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(
                    L, N, available_actions_batch)
            else:
                available_actions_batch = None
            if self.factor is not None:
                factor_batch = _flatten(L, N, factor_batch)
            rewards_batch = _flatten(L, N, rewards_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            adv_active_masks_batch = _flatten(L, N, adv_active_masks_batch)
            old_action_log_probs_batch = _flatten(
                L, N, old_action_log_probs_batch)
            old_adv_action_log_probs_batch = _flatten(
                L, N, old_adv_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)
            if self.factor is not None:
                yield obs_batch, obs_next_batch, rnn_states_batch, adv_rnn_states_batch, actions_batch, \
                    adv_actions_batch, rewards_batch, masks_batch, active_masks_batch, adv_active_masks_batch, \
                    old_action_log_probs_batch, old_adv_action_log_probs_batch, adv_targ, available_actions_batch, factor_batch
            else:
                yield obs_batch, obs_next_batch, rnn_states_batch, adv_rnn_states_batch, actions_batch, \
                    adv_actions_batch, rewards_batch, masks_batch, active_masks_batch, adv_active_masks_batch, \
                    old_action_log_probs_batch, old_adv_action_log_probs_batch, adv_targ, available_actions_batch