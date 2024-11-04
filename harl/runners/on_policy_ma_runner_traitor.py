import time
import numpy as np
import torch
import setproctitle
from harl.common.valuenorm import ValueNorm
from harl.common.fgsm import FGSM
from harl.common.buffers.on_policy_actor_buffer_advt import OnPolicyActorBufferAdvt
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.common.buffers.on_policy_critic_buffer_fp import OnPolicyCriticBufferFP
from harl.algorithms.actors import ALGO_REGISTRY
from harl.algorithms.critics.v_critic import VCritic
from harl.utils.trans_tools import _t2n, softmax
from harl.utils.envs_tools import (
    make_eval_env,
    make_train_env,
    make_render_env,
    set_seed,
    get_num_agents,
)
from harl.utils.models_tools import init_device
from harl.utils.configs_tools import init_dir, save_config
from harl.envs import LOGGER_REGISTRY


class OnPolicyMARunnerTraitor:
    """Runner for on-policy algorithms (adv training)."""

    def __init__(self, args, algo_args, env_args):
        """Initialize the OnPolicyMARunnerAdvt class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        self.save_id = 0

        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args

        self.hidden_sizes = algo_args["model"]["hidden_sizes"]
        self.rnn_hidden_size = self.hidden_sizes[-1]
        self.recurrent_n = algo_args["model"]["recurrent_n"]
        self.action_aggregation = algo_args["algo"]["action_aggregation"]
        self.state_type = env_args.get("state_type", "EP")
        self.share_param = algo_args["algo"]["share_param"]
        self.fixed_order = algo_args["algo"]["fixed_order"]

        # adv training
        self.adv_prob = algo_args["algo"].get("adv_prob", 0.5)  # probability of having adversary
        self.obs_adversary = env_args.get("obs_agent_adversary", False)  # adding adversary on observation
        self.agent_adversary = algo_args["algo"].get("agent_adversary", 0)  # who is the adversary
        self.victim_interval = algo_args["algo"].get("victim_interval",
                                                     1)  # use it if update adversary for multiple times
        self.random_adversary = (
                    self.agent_adversary < 0)  # if self.agent_adversary<0, then we randomly assign adversary
        self.episode_adversary = False  # which episode contains the adversary
        self.load_critic = algo_args["algo"].get("load_critic", False)
        self.load_adv_actor = algo_args["algo"].get("load_adv_actor", False)
        self.load_maddpg = algo_args["algo"].get("load_maddpg", False)
        self.super_adversary = algo_args["algo"].get("super_adversary",
                                                     False)  # whether the adversary has defenders' policies
        self.adapt_adversary = algo_args["algo"].get("adapt_adversary", False)
        self.state_adversary = algo_args["algo"].get("state_adversary", False)
        self.render_mode = algo_args["render"].get("render_mode", None)

        self.env_name = args["env"]

        set_seed(algo_args["seed"])
        self.device = init_device(algo_args["device"])
        if not self.algo_args["render"]["use_render"]:  # train, not render
            self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
                args["env"],
                env_args,
                args["algo"],
                args["exp_name"],
                algo_args["seed"]["seed"],
                logger_path=algo_args["logger"]["log_dir"],
            )
            save_config(args, algo_args, env_args, self.run_dir)
        # set the title of the process
        setproctitle.setproctitle(
            str(args["algo"]) + "-" + str(args["env"]) + "-" + str(args["exp_name"])
        )

        # set the config of env
        if self.algo_args["render"]["use_render"]:  # make envs for rendering
            (
                self.envs,
                self.manual_render,
                self.manual_expand_dims,
                self.manual_delay,
                self.env_num,
            ) = make_render_env(args["env"], algo_args["seed"]["seed"], env_args)
        else:  # make envs for training and evaluation
            self.envs = make_train_env(
                args["env"],
                algo_args["seed"]["seed"],
                algo_args["train"]["n_rollout_threads"],
                env_args,
            )
            self.eval_envs = (
                make_eval_env(
                    args["env"],
                    algo_args["seed"]["seed"],
                    algo_args["eval"]["n_eval_rollout_threads"],
                    env_args,
                )
                if algo_args["eval"]["use_eval"]
                else None
            )
        self.num_agents = get_num_agents(args["env"], env_args, self.envs)

        self.adapt_adv_probs = np.zeros(self.num_agents)

        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space)

        # actor
        if self.share_param:
            self.actor = []
            agent = ALGO_REGISTRY[args["algo"]](
                {**algo_args["model"], **algo_args["algo"]},
                self.envs.observation_space[0],
                self.envs.action_space[0],
                num_agents=self.num_agents,
                device=self.device,
            )
            self.actor.append(agent)
            for agent_id in range(1, self.num_agents):
                assert (
                    self.envs.observation_space[agent_id]
                    == self.envs.observation_space[0]
                ), "Agents have heterogeneous observation spaces, parameter sharing is not valid."
                assert (
                    self.envs.action_space[agent_id] == self.envs.action_space[0]
                ), "Agents have heterogeneous action spaces, parameter sharing is not valid."
                self.actor.append(self.actor[0])
        else:
            self.actor = []
            for agent_id in range(self.num_agents):
                agent = ALGO_REGISTRY[args["algo"]](
                    {**algo_args["model"], **algo_args["algo"]},
                    self.envs.observation_space[agent_id],
                    self.envs.action_space[agent_id],
                    num_agents=self.num_agents,
                    device=self.device,
                )
                self.actor.append(agent)

        if self.algo_args["render"]["use_render"] is False:  # train, not render
            self.actor_buffer = []
            for agent_id in range(self.num_agents):
                ac_bu = OnPolicyActorBufferAdvt(
                    {**algo_args["train"], **algo_args["model"]},
                    self.envs.observation_space[agent_id],
                    self.envs.action_space[agent_id],
                )
                self.actor_buffer.append(ac_bu)

            share_observation_space = self.envs.share_observation_space[0]
            self.critic = VCritic(
                {**algo_args["model"], **algo_args["algo"]},
                share_observation_space,
                device=self.device,
            )
            if self.state_type == "EP":
                # EP stands for Environment Provided, as phrased by MAPPO paper.
                # In EP, the global states for all agents are the same.
                self.critic_buffer = OnPolicyCriticBufferEP(
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                    share_observation_space,
                )
            elif self.state_type == "FP":
                # FP stands for Feature Pruned, as phrased by MAPPO paper.
                # In FP, the global states for all agents are different, and thus needs the dimension of the number of agents.
                self.critic_buffer = OnPolicyCriticBufferFP(
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                    share_observation_space,
                    self.num_agents,
                )
            else:
                raise NotImplementedError

            if self.algo_args["train"]["use_valuenorm"] is True:
                self.value_normalizer = ValueNorm(1, device=self.device)
            else:
                self.value_normalizer = None

            self.logger = LOGGER_REGISTRY[args["env"]](
                args, algo_args, env_args, self.num_agents, self.writter, self.run_dir
            )

        if self.state_adversary:
            self.fgsm = FGSM(algo_args["algo"], self.obs_adversary, self.actor, device=self.device)
        self.state_adversary_all = algo_args["algo"].get('state_adversary_all', False)

        if self.algo_args["train"]["model_dir"] is not None:  # restore model
            self.restore()

    def run(self):
        """Run the training (or rendering) pipeline."""
        if self.algo_args["render"]["use_render"] is True:
            if self.render_mode == "traitor":
                self.render_adv()
            else:
                self.render()
                self.render_adv()
            return
        print("start running")
        self.warmup()

        episodes = (
            int(self.algo_args["train"]["num_env_steps"])
            // self.algo_args["train"]["episode_length"]
            // self.algo_args["train"]["n_rollout_threads"]
        )

        self.logger.init(episodes)  # logger callback at the beginning of training
        self.logger.episode_init(0)

        if self.algo_args["eval"]["use_eval"]:
            if self.state_adversary:
                self.eval_adv_state()
                return

        for episode in range(1, episodes + 1):
            if self.algo_args["train"][
                "use_linear_lr_decay"
            ]:  # linear decay of learning rate
                if self.share_param:
                    self.actor[0].lr_decay(episode, episodes)
                else:
                    for agent_id in range(self.num_agents):
                        self.actor[agent_id].lr_decay(episode, episodes)
                self.critic.lr_decay(episode, episodes)

            self.logger.episode_init(
                episode
            )  # logger callback at the beginning of each episode

            self.prep_rollout()  # change to eval mode

            if self.random_adversary:
                if self.adapt_adversary:
                    self.agent_adversary = np.random.choice(range(self.num_agents), p=softmax(-1 * self.adapt_adv_probs / 1))
                else:
                    self.agent_adversary = np.random.choice(range(self.num_agents))

            if episode % self.victim_interval == 0:  # which means some episodes are not adversary
                self.episode_adversary = (np.random.rand(self.algo_args["train"]["n_rollout_threads"]) < self.adv_prob)
            else:
                self.episode_adversary = (np.random.rand(self.algo_args["train"]["n_rollout_threads"]) < 2) # all True

            for step in range(self.algo_args["train"]["episode_length"]):
                # Sample actions from actors and values from critics
                (
                    values,
                    actions,
                    adv_actions,
                    action_log_probs,
                    adv_action_log_probs,
                    rnn_states,
                    adv_rnn_states,
                    rnn_states_critic,
                ) = self.collect_adv(step)
                input_actions = actions.copy()
                input_actions[self.episode_adversary, self.agent_adversary] = adv_actions[
                    self.episode_adversary, self.agent_adversary]
                # actions: (n_threads, n_agents, action_dim)
                (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                ) = self.envs.step(actions)
                # obs: (n_threads, n_agents, obs_dim)
                # share_obs: (n_threads, n_agents, share_obs_dim)
                # rewards: (n_threads, n_agents, 1)
                # dones: (n_threads, n_agents)
                # infos: (n_threads)
                # available_actions: (n_threads, ) of None or (n_threads, n_agents, action_number)
                data = (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                    values,
                    actions,
                    adv_actions,
                    action_log_probs,
                    adv_action_log_probs,
                    rnn_states,
                    adv_rnn_states,
                    rnn_states_critic,
                )

                self.logger.per_step(data)  # logger callback at each step

                self.insert(data)  # insert data into buffer

            # compute return and update network
            self.compute()
            self.prep_training()  # change to train mode

            if episode % self.victim_interval == 0:
                actor_train_infos, critic_train_info = self.train()
            else:
                actor_train_infos, critic_train_info = self.train_adv()

            # log information
            if episode % self.algo_args["train"]["log_interval"] == 0:
                self.logger.episode_log(
                    actor_train_infos,
                    critic_train_info,
                    self.actor_buffer,
                    self.critic_buffer,
                )

            # eval
            if episode % self.algo_args["train"]["eval_interval"] == 0:
                if self.algo_args["eval"]["use_eval"]:
                    self.prep_rollout()
                    self.eval()
                    self.eval_adv()
                self.save()

            self.after_update()

    def warmup(self):
        """Warm up the replay buffer."""
        # reset env
        obs, share_obs, available_actions = self.envs.reset()
        # replay buffer
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].obs[0] = obs[:, agent_id].copy()
            if self.actor_buffer[agent_id].available_actions is not None:
                self.actor_buffer[agent_id].available_actions[0] = available_actions[
                    :, agent_id
                ].copy()
        if self.state_type == "EP":
            self.critic_buffer.share_obs[0] = share_obs[:, 0].copy()
        elif self.state_type == "FP":
            self.critic_buffer.share_obs[0] = share_obs.copy()

    def get_adv_actions(self, obs, action_probs):
        # (n_threads, n_agents, n_actions)
        action_max = np.expand_dims(action_probs.argmax(axis=-1), axis=-1)
        action_s2 = np.flip(action_max, axis=1)
        action_s1 = 1 - action_s2
        current_state = obs[:, :, 0:1]
        actions = np.zeros_like(action_max)
        actions[current_state==0] = action_s1[current_state==0]
        actions[current_state==1] = action_s2[current_state==1]

        return actions

    @torch.no_grad()
    def collect_adv(self, step):
        action_collector = []
        adv_action_collector = []
        action_log_prob_collector = []
        adv_action_log_prob_collector = []
        rnn_state_collector = []
        adv_rnn_state_collector = []
        for agent_id in range(self.num_agents):
            if self.obs_adversary:
                adv_len = self.actor_buffer[agent_id].obs[step].shape[1] - self.num_agents + self.agent_adversary
                self.actor_buffer[agent_id].obs[step][self.episode_adversary, adv_len] = 1
            action, action_log_prob, rnn_state = self.actor[agent_id].get_actions(self.actor_buffer[agent_id].obs[step],
                                                                                  self.actor_buffer[
                                                                                      agent_id].rnn_states[step],
                                                                                  self.actor_buffer[agent_id].masks[
                                                                                      step],
                                                                                  self.actor_buffer[
                                                                                      agent_id].available_actions[
                                                                                      step] if self.actor_buffer[
                                                                                                   agent_id].available_actions is not None else None,
                                                                                  agent_id=agent_id)
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
        for agent_id in range(self.num_agents):
            if self.super_adversary:
                def_act = np.concatenate([*action_collector[-self.num_agents:][:agent_id],
                                          *action_collector[-self.num_agents:][agent_id + 1:]], axis=-1)
                adv_obs = np.concatenate([self.actor_buffer[agent_id].obs[step], softmax(def_act)], axis=-1)
            else:
                adv_obs = self.actor_buffer[agent_id].obs[step]
            adv_action, adv_action_log_prob, adv_rnn_state = self.actor[agent_id].get_adv_actions(adv_obs,
                                                                                                  self.actor_buffer[
                                                                                                      agent_id].adv_rnn_states[
                                                                                                      step],
                                                                                                  self.actor_buffer[
                                                                                                      agent_id].masks[
                                                                                                      step],
                                                                                                  self.actor_buffer[
                                                                                                      agent_id].available_actions[
                                                                                                      step] if
                                                                                                  self.actor_buffer[
                                                                                                      agent_id].available_actions is not None else None)
            adv_action_collector.append(_t2n(adv_action))
            adv_action_log_prob_collector.append(_t2n(adv_action_log_prob))
            adv_rnn_state_collector.append(_t2n(adv_rnn_state))
        # [self.envs, agents, dim]
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        adv_actions = np.array(adv_action_collector).transpose(1, 0, 2)
        adv_action_log_probs = np.array(adv_action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)
        adv_rnn_states = np.array(adv_rnn_state_collector).transpose(1, 0, 2, 3)

        if self.state_type == "EP":
            if self.obs_adversary:
                adv_len = self.critic_buffer.share_obs[step].shape[1] - self.num_agents + self.agent_adversary
                self.critic_buffer.share_obs[step][self.episode_adversary, adv_len] = 1
            value, rnn_state_critic = self.critic.get_values(self.critic_buffer.share_obs[step],
                                                             self.critic_buffer.rnn_states_critic[step],
                                                             self.critic_buffer.masks[step])
            values = _t2n(value)
            rnn_states_critic = _t2n(rnn_state_critic)
        elif self.state_type == "FP":
            if self.obs_adversary:
                adv_len = self.critic_buffer.share_obs[step].shape[1] - self.num_agents + self.agent_adversary
                self.critic_buffer.share_obs[step][self.episode_adversary, :, adv_len] = 1
            value, rnn_state_critic = self.critic.get_values(np.concatenate(self.critic_buffer.share_obs[step]),
                                                             np.concatenate(self.critic_buffer.rnn_states_critic[step]),
                                                             np.concatenate(self.critic_buffer.masks[step]))
            values = np.array(np.split(_t2n(value), self.algo_args["train"]["n_rollout_threads"]))
            rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.algo_args["train"]["n_rollout_threads"]))

        return values, actions, adv_actions, action_log_probs, adv_action_log_probs, rnn_states, adv_rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
            values, actions, adv_actions, action_log_probs, adv_action_log_probs, \
            rnn_states, adv_rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(
        ), self.num_agents, self.recurrent_n, self.rnn_hidden_size), dtype=np.float32)
        adv_rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(
        ), self.num_agents, self.recurrent_n, self.rnn_hidden_size), dtype=np.float32)

        if self.state_type == "EP":
            rnn_states_critic[dones_env == True] = np.zeros(
                ((dones_env == True).sum(), self.recurrent_n, self.rnn_hidden_size), dtype=np.float32)
        elif self.state_type == "FP":
            rnn_states_critic[dones_env == True] = np.zeros(
                ((dones_env == True).sum(), self.num_agents, self.recurrent_n, self.rnn_hidden_size), dtype=np.float32)

        # masks use 0 to mask out threads that just finish
        masks = np.ones(
            (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # active_masks use 0 to mask out agents that have died
        active_masks = np.ones(
            (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(
            ((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        adv_active_masks = active_masks.copy()
        active_masks[self.episode_adversary, self.agent_adversary] = 0
        adv_active_masks[~self.episode_adversary] = 0
        adv_active_masks[:, np.arange(self.num_agents) != self.agent_adversary] = 0

        # bad_masks use 0 to denote truncation and 1 to denote termination
        if self.state_type == "EP":
            bad_masks = np.array(
                [[0.0] if "bad_transition" in info[0].keys() and info[0]["bad_transition"] == True else [1.0] for info
                 in infos])
        elif self.state_type == "FP":
            bad_masks = np.array([[[0.0] if "bad_transition" in info[agent_id].keys() and info[agent_id][
                'bad_transition'] == True else [1.0] for agent_id in range(self.num_agents)] for info in infos])

        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].insert(obs[:, agent_id], rnn_states[:, agent_id], adv_rnn_states[:, agent_id],
                                               actions[:, agent_id], adv_actions[:, agent_id],
                                               action_log_probs[:, agent_id], adv_action_log_probs[:, agent_id],
                                               rewards[:, agent_id], masks[:, agent_id], active_masks[:, agent_id],
                                               adv_active_masks[:, agent_id],
                                               available_actions[:, agent_id] if available_actions[
                                                                                     0] is not None else None)

        if self.state_type == "EP":
            self.critic_buffer.insert(
                share_obs[:, 0], rnn_states_critic, values, rewards[:, 0], masks[:, 0], bad_masks)
        elif self.state_type == "FP":
            self.critic_buffer.insert(
                share_obs, rnn_states_critic, values, rewards, masks, bad_masks)

    @torch.no_grad()
    def compute(self):
        """Compute returns and advantages.
        Compute critic evaluation of the last state,
        and then let buffer compute returns, which will be used during training.
        """
        if self.state_type == "EP":
            next_value, _ = self.critic.get_values(
                self.critic_buffer.share_obs[-1],
                self.critic_buffer.rnn_states_critic[-1],
                self.critic_buffer.masks[-1],
            )
            next_value = _t2n(next_value)
        elif self.state_type == "FP":
            next_value, _ = self.critic.get_values(
                np.concatenate(self.critic_buffer.share_obs[-1]),
                np.concatenate(self.critic_buffer.rnn_states_critic[-1]),
                np.concatenate(self.critic_buffer.masks[-1]),
            )
            next_value = np.array(
                np.split(_t2n(next_value), self.algo_args["train"]["n_rollout_threads"])
            )
        self.critic_buffer.compute_returns(next_value, self.value_normalizer)

    def train(self):
        """Training procedure for MAPPO."""
        actor_train_infos = []

        # compute advantages
        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[
                :-1
            ] - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            advantages = (
                self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
            )

        # normalize advantages for FP
        if self.state_type == "FP":
            active_masks_collector = [
                self.actor_buffer[i].active_masks for i in range(self.num_agents)
            ]
            active_masks_array = np.stack(active_masks_collector, axis=2)
            advantages_copy = advantages.copy()
            advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        # update actors
        if self.share_param:
            actor_train_info = self.actor[0].share_param_train(
                self.actor_buffer, advantages.copy(), self.num_agents, self.state_type
            )
            for _ in torch.randperm(self.num_agents):
                actor_train_infos.append(actor_train_info)
        else:
            for agent_id in range(self.num_agents):
                if self.state_type == "EP":
                    actor_train_info = self.actor[agent_id].train(
                        self.actor_buffer[agent_id], advantages.copy(), "EP"
                    )
                elif self.state_type == "FP":
                    actor_train_info = self.actor[agent_id].train(
                        self.actor_buffer[agent_id],
                        advantages[:, :, agent_id].copy(),
                        "FP",
                    )
                actor_train_infos.append(actor_train_info)

        # update critic
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        return actor_train_infos, critic_train_info

    def train_adv(self):
        actor_train_infos = []

        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[:-1] - \
                self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            advantages = self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]

        if self.state_type == "FP":
            active_masks_collector = [self.actor_buffer[i].active_masks for i in range(self.num_agents)]
            active_masks_array = np.stack(active_masks_collector, axis = 2)
            advantages_copy = advantages.copy()
            advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

            # update actors
            if self.share_param:
                actor_train_info = self.actor[0].share_param_train_adv(
                    self.actor_buffer, advantages.copy(), self.num_agents, self.state_type
                )
                for _ in torch.randperm(self.num_agents):
                    actor_train_infos.append(actor_train_info)
            else:
                for agent_id in range(self.num_agents):
                    if self.state_type == "EP":
                        actor_train_info = self.actor[agent_id].train_adv(
                            self.actor_buffer[agent_id], advantages.copy(), "EP"
                        )
                    elif self.state_type == "FP":
                        actor_train_info = self.actor[agent_id].train_adv(
                            self.actor_buffer[agent_id],
                            advantages[:, :, agent_id].copy(),
                            "FP",
                        )
                    actor_train_infos.append(actor_train_info)

        return actor_train_infos, {}

    def after_update(self):
        """Do the necessary data operations after an update.
        After an update, copy the data at the last step to the first position of the buffer.
        This will be used for then generating new actions.
        """
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].after_update()
        self.critic_buffer.after_update()

    @torch.no_grad()
    def eval(self):
        """Evaluate the model."""
        self.logger.eval_init()  # logger callback at the beginning of evaluation
        eval_episode = 0

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (
                self.algo_args["eval"]["n_eval_rollout_threads"],
                self.num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (self.algo_args["eval"]["n_eval_rollout_threads"], self.num_agents, 1),
            dtype=np.float32,
        )

        while True:
            eval_actions_collector = []
            for agent_id in range(self.num_agents):
                eval_actions, temp_rnn_state = self.actor[agent_id].act(
                    eval_obs[:, agent_id],
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    eval_available_actions[:, agent_id]
                    if eval_available_actions[0] is not None
                    else None,
                    deterministic=True,
                )
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)
            eval_data = (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            )
            self.logger.eval_per_step(
                eval_data
            )  # logger callback at each step of evaluation

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[
                eval_dones_env == True
            ] = np.zeros(  # if env is done, then reset rnn_state to all zero
                (
                    (eval_dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_n,
                    self.rnn_hidden_size,
                ),
                dtype=np.float32,
            )

            eval_masks = np.ones(
                (self.algo_args["eval"]["n_eval_rollout_threads"], self.num_agents, 1),
                dtype=np.float32,
            )
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
            )

            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    self.logger.eval_thread_done(
                        eval_i
                    )  # logger callback when an episode is done

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                self.logger.eval_log(
                    eval_episode
                )  # logger callback at the end of evaluation
                break

    @torch.no_grad()
    def eval_adv(self):
        if self.random_adversary:
            for i in range(self.num_agents):
                self._eval_adv(i)
        else:
            self._eval_adv(self.agent_adversary)

    @torch.no_grad()
    def _eval_adv(self, adv_id):
        self.logger.eval_init()
        eval_episode = 0

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        if self.obs_adversary:
            adv_len = eval_obs.shape[2] - self.num_agents + adv_id
            eval_obs[:, :, adv_len] = 1
        eval_rnn_states = np.zeros((eval_obs.shape[0], self.num_agents,
                                    self.recurrent_n, self.rnn_hidden_size), dtype=np.float32)
        eval_adv_rnn_states = np.zeros((eval_obs.shape[0], self.num_agents,
                                        self.recurrent_n, self.rnn_hidden_size), dtype=np.float32)
        eval_masks = np.ones((eval_obs.shape[0], self.num_agents, 1), dtype=np.float32)

        obs_offset = self.algo_args['algo'].get('obs_offset', 0)
        eval_transfer_obs = np.zeros((eval_obs.shape[0], self.num_agents, eval_obs.shape[2] + obs_offset))
        if obs_offset >= 0:
            eval_transfer_obs[:, :, :eval_obs.shape[2]] = eval_obs
        else:
            eval_transfer_obs = eval_obs[:, :, :eval_transfer_obs.shape[2]]
        if obs_offset > 0:
            transfer_adv_len = eval_obs.shape[2] - self.num_agents + adv_id
            eval_transfer_obs[:, :, transfer_adv_len] = 1

        while True:
            eval_actions_collector = []
            eval_adv_actions_collector = []
            for agent_id in range(self.num_agents):
                eval_actions, temp_rnn_state = \
                    self.actor[agent_id].act(eval_obs[:, agent_id],
                                             eval_rnn_states[:, agent_id],
                                             eval_masks[:, agent_id],
                                             eval_available_actions[:, agent_id] if eval_available_actions[
                                                                                        0] is not None else None,
                                             deterministic=False,
                                             agent_id=agent_id)
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            for agent_id in range(self.num_agents):
                if self.super_adversary:
                    def_act = np.concatenate([*eval_actions_collector[-self.num_agents:][:agent_id],
                                              *eval_actions_collector[-self.num_agents:][agent_id + 1:]], axis=-1)
                    # adv_obs = np.concatenate([eval_obs[:, agent_id], softmax(def_act)], axis=-1)
                    adv_obs = np.concatenate([eval_transfer_obs[:, agent_id], softmax(def_act)], axis=-1)
                else:
                    # adv_obs = eval_obs[:, agent_id]
                    adv_obs = eval_transfer_obs[:, agent_id]
                eval_adv_actions, temp_adv_rnn_state = \
                    self.actor[agent_id].act_adv(adv_obs,
                                                 eval_adv_rnn_states[:, agent_id],
                                                 eval_masks[:, agent_id],
                                                 eval_available_actions[:, agent_id] if eval_available_actions[
                                                                                            0] is not None else None,
                                                 deterministic=False)
                eval_adv_rnn_states[:, agent_id] = _t2n(temp_adv_rnn_state)
                eval_adv_actions_collector.append(_t2n(eval_adv_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
            eval_adv_actions = np.array(eval_adv_actions_collector).transpose(1, 0, 2)
            eval_actions[:, adv_id] = eval_adv_actions[:, adv_id]

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(
                eval_actions)
            if self.obs_adversary:
                adv_len = eval_obs.shape[2] - self.num_agents + adv_id
                eval_obs[:, :, adv_len] = 1
            if obs_offset >= 0:
                eval_transfer_obs[:, :, :eval_obs.shape[2]] = eval_obs
            else:
                eval_transfer_obs = eval_obs[:, :, :eval_transfer_obs.shape[2]]
            if obs_offset > 0:
                transfer_adv_len = eval_obs.shape[2] - self.num_agents + adv_id
                eval_transfer_obs[:, :, transfer_adv_len] = 1

            eval_data = eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions
            self.logger.eval_per_step(eval_data)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(
            ), self.num_agents, self.recurrent_n, self.rnn_hidden_size), dtype=np.float32)

            eval_masks = np.ones(
                (eval_obs.shape[0], self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    self.logger.eval_thread_done(eval_i)

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                # eval_log returns whether the current model should be saved
                ret_mean = self.logger.eval_log_adv(eval_episode, adv_id)
                break

        self.adapt_adv_probs[adv_id] = np.mean(self.logger.eval_episode_rewards)

        with open("transfer_{}.csv".format(self.env_name), "a") as fp:
            # fp.write("{},{},random,{},{:.4f}\n".format(self.env_name, self.args.exp_name, adv_id, ret_mean))
            fp.write("{},{},{},{:.4f}\n".format(self.env_name, self.args["exp_name"], adv_id, ret_mean))

    @torch.no_grad()
    def eval_adv_state(self):
        if self.state_adversary_all:
            self._eval_adv_state_all()
        elif self.random_adversary:
            for i in range(self.num_agents):
                self._eval_adv_state(i)
        else:
            return self._eval_adv_state(self.agent_adversary)

    @torch.no_grad()
    def _eval_adv_state(self, adv_id):
        self.logger.eval_init()
        eval_episode = 0
        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        if self.obs_adversary:
            adv_len = eval_obs.shape[2] - self.num_agents + adv_id
            eval_obs[:, :, adv_len] = 1
        eval_rnn_states = np.zeros((eval_obs.shape[0], self.num_agents,
                                    self.recurrent_n, self.rnn_hidden_size), dtype=np.float32)
        eval_adv_rnn_states = np.zeros((eval_obs.shape[0], self.num_agents,
                                        self.recurrent_n, self.rnn_hidden_size), dtype=np.float32)
        eval_masks = np.ones((eval_obs.shape[0], self.num_agents, 1), dtype=np.float32)

        while True:
            eval_actions_collector = []
            eval_adv_actions_collector = []
            for agent_id in range(self.num_agents):
                if agent_id == adv_id:
                    obs = self.fgsm(eval_obs[:, agent_id],
                                    eval_rnn_states[:, agent_id],
                                    eval_adv_rnn_states[:, agent_id],
                                    eval_masks[:, agent_id],
                                    eval_available_actions[:, agent_id] if eval_available_actions[
                                                                               0] is not None else None,
                                    agent_id=agent_id)
                else:
                    obs = eval_obs[:, agent_id]

                eval_actions, temp_rnn_state = \
                    self.actor[agent_id].act(obs,
                                            eval_rnn_states[:, agent_id],
                                            eval_masks[:, agent_id],
                                            eval_available_actions[:, agent_id] if eval_available_actions[
                                                                                       0] is not None else None,
                                            deterministic=False,
                                            agent_id=agent_id)

                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            for agent_id in range(self.num_agents):
                eval_adv_actions, temp_adv_rnn_state = \
                    self.actor[agent_id].act_adv(eval_obs[:, agent_id],  # [:, :-self.num_agents],
                                                 eval_adv_rnn_states[:, agent_id],
                                                 eval_masks[:, agent_id],
                                                 eval_available_actions[:, agent_id] if eval_available_actions[
                                                                                            0] is not None else None,
                                                 deterministic=False)
                eval_adv_rnn_states[:, agent_id] = _t2n(temp_adv_rnn_state)
                eval_adv_actions_collector.append(_t2n(eval_adv_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
            eval_adv_actions = np.array(eval_adv_actions_collector).transpose(1, 0, 2)
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(
                eval_actions)
            if self.obs_adversary:
                adv_len = eval_obs.shape[2] - self.num_agents + adv_id
                eval_obs[:, :, adv_len] = 1
            eval_data = eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions
            self.logger.eval_per_step(eval_data)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(
            ), self.num_agents, self.recurrent_n, self.rnn_hidden_size), dtype=np.float32)

            eval_masks = np.ones(
                (eval_obs.shape[0], self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    self.logger.eval_thread_done(eval_i)

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                # eval_log returns whether the current model should be saved
                ret_mean = self.logger.eval_log_adv(eval_episode, adv_id)
                break

        self.adapt_adv_probs[adv_id] = np.mean(self.logger.eval_episode_rewards)

        with open("transfer_{}.csv".format(self.env_name), "a") as fp:
            fp.write(
                "{},{},eps{:.1f},{},{:.4f}\n".format(self.env_name, self.args["exp_name"], self.algo_args["algo"]["eps"],
                                                     adv_id, ret_mean))
            
    @torch.no_grad()
    def _eval_adv_state_all(self):
        self.logger.eval_init()
        eval_episode = 0
        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        if self.obs_adversary:
            # adv_len = eval_obs.shape[2] - self.num_agents + adv_id
            # eval_obs[:, :, adv_len] = 1
            eval_obs[:, :, -self.num_agents:] = 1
        eval_rnn_states = np.zeros((eval_obs.shape[0], self.num_agents,
                                    self.recurrent_n, self.rnn_hidden_size), dtype=np.float32)
        eval_adv_rnn_states = np.zeros((eval_obs.shape[0], self.num_agents,
                                        self.recurrent_n, self.rnn_hidden_size), dtype=np.float32)
        eval_masks = np.ones((eval_obs.shape[0], self.num_agents, 1), dtype=np.float32)

        while True:
            eval_actions_collector = []
            eval_adv_actions_collector = []
            for agent_id in range(self.num_agents):
                obs = self.fgsm(eval_obs[:, agent_id],
                                eval_rnn_states[:, agent_id],
                                eval_adv_rnn_states[:, agent_id],
                                eval_masks[:, agent_id],
                                eval_available_actions[:, agent_id] if eval_available_actions[
                                                                            0] is not None else None,
                                agent_id=agent_id)

                eval_actions, temp_rnn_state = \
                    self.actor[agent_id].act(obs,
                                            eval_rnn_states[:, agent_id],
                                            eval_masks[:, agent_id],
                                            eval_available_actions[:, agent_id] if eval_available_actions[
                                                                                       0] is not None else None,
                                            deterministic=False,
                                            agent_id=agent_id)

                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            for agent_id in range(self.num_agents):
                eval_adv_actions, temp_adv_rnn_state = \
                    self.actor[agent_id].act_adv(eval_obs[:, agent_id],  # [:, :-self.num_agents],
                                                 eval_adv_rnn_states[:, agent_id],
                                                 eval_masks[:, agent_id],
                                                 eval_available_actions[:, agent_id] if eval_available_actions[
                                                                                            0] is not None else None,
                                                 deterministic=False)
                eval_adv_rnn_states[:, agent_id] = _t2n(temp_adv_rnn_state)
                eval_adv_actions_collector.append(_t2n(eval_adv_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
            eval_adv_actions = np.array(eval_adv_actions_collector).transpose(1, 0, 2)
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(
                eval_actions)
            if self.obs_adversary:
                # adv_len = eval_obs.shape[2] - self.num_agents + adv_id
                # eval_obs[:, :, adv_len] = 1
                eval_obs[:, :, -self.num_agents:] = 1
            eval_data = eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions
            self.logger.eval_per_step(eval_data)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(
            ), self.num_agents, self.recurrent_n, self.rnn_hidden_size), dtype=np.float32)

            eval_masks = np.ones(
                (eval_obs.shape[0], self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    self.logger.eval_thread_done(eval_i)

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                # eval_log returns whether the current model should be saved
                ret_mean = self.logger.eval_log_adv(eval_episode, '_all')
                break

        # self.adapt_adv_probs[adv_id] = np.mean(self.logger.eval_episode_rewards)

        with open("transfer_{}.csv".format(self.env_name), "a") as fp:
            fp.write(
                "{},{},eps{:.1f},{},{:.4f}\n".format(self.env_name, self.args["exp_name"], self.algo_args["algo"]["eps"],
                                                     'all_agents', ret_mean))

    def prep_rollout(self):
        """Prepare for rollout."""
        for agent_id in range(self.num_agents):
            self.actor[agent_id].prep_rollout()
        self.critic.prep_rollout()

    def prep_training(self):
        """Prepare for training."""
        for agent_id in range(self.num_agents):
            self.actor[agent_id].prep_training()
        self.critic.prep_training()

    def save(self):
        """Save model parameters."""
        for agent_id in range(self.num_agents):
            policy_actor = self.actor[agent_id].actor
            torch.save(
                policy_actor.state_dict(),
                str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt",
            )
            adv_policy_actor = self.actor[agent_id].adv_actor
            torch.save(
                adv_policy_actor.state_dict(),
                str(self.save_dir) + "/adv_actor_agent" + str(agent_id) + ".pt",
            )
        policy_critic = self.critic.critic
        torch.save(
            policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + ".pt"
        )
        if self.value_normalizer is not None:
            torch.save(
                self.value_normalizer.state_dict(),
                str(self.save_dir) + "/value_normalizer" + ".pt",
            )
        self.save_id = self.save_id + 1

    def restore(self):
        """Restore model parameters."""
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(
                str(self.algo_args["train"]["model_dir"])
                + "/actor_agent"
                + str(agent_id)
                + ".pt"
            )
            self.actor[agent_id].actor.load_state_dict(policy_actor_state_dict)
            if self.load_adv_actor:
                adv_policy_actor_state_dict = torch.load(
                    str(self.algo_args["train"]["adv_model_dir"]) + '/adv_actor_agent' + str(agent_id) + '.pt', map_location=torch.device('cpu'))
                self.actor[agent_id].adv_actor.load_state_dict(
                    adv_policy_actor_state_dict)
        if not self.algo_args["render"]["use_render"]:
            policy_critic_state_dict = torch.load(
                str(self.algo_args["train"]["model_dir"]) + "/critic_agent" + ".pt"
            )
            self.critic.critic.load_state_dict(policy_critic_state_dict)
            if self.value_normalizer is not None:
                value_normalizer_state_dict = torch.load(
                    str(self.algo_args["train"]["model_dir"])
                    + "/value_normalizer"
                    + ".pt"
                )
                self.value_normalizer.load_state_dict(value_normalizer_state_dict)

    def close(self):
        """Close environment, writter, and logger."""
        if self.algo_args["render"]["use_render"]:
            self.envs.close()
        else:
            self.envs.close()
            if self.algo_args["eval"]["use_eval"] and self.eval_envs is not self.envs:
                self.eval_envs.close()
            self.writter.export_scalars_to_json(str(self.log_dir + "/summary.json"))
            self.writter.close()
            self.logger.close()