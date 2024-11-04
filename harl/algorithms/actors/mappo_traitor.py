from harl.algorithms.actors.mappo_advt import MAPPOAdvt

class MAPPOTraitor(MAPPOAdvt):
    def train(self, actor_buffer, advantages, state_type):
        return self.train_adv(actor_buffer, advantages, state_type)

    def share_param_train(self, actor_buffer, advantages, num_agents, state_type):
        return self.share_param_train_adv(actor_buffer, advantages, num_agents, state_type)

    def get_logits(self, obs, rnn_states_actor, masks, available_actions=None,
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
        action_logits = self.actor.get_logits(obs,
                                            rnn_states_actor,
                                            masks,
                                            available_actions,
                                            deterministic)
        return action_logits

    def act_with_probs(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False, agent_id=0):
        """Compute actions using the given inputs.
        Args:
            obs (np.ndarray): local agent inputs to the actor.
            rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray) denotes which actions are available to agent
                                    (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, action_probs, rnn_states_actor = self.actor.forward_with_probs(
            obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, action_probs, rnn_states_actor

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False, agent_id=0):
        """Compute actions using the given inputs.
        Args:
            obs (np.ndarray): local agent inputs to the actor.
            rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray) denotes which actions are available to agent
                                    (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor