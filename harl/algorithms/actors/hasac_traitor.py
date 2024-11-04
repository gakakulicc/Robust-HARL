from harl.algorithms.actors.hasac_advt import HASACAdvt
from harl.utils.envs_tools import check

class HASACTraitor(HASACAdvt):
    def train(self):
        return

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
        action_logits = self.actor.get_transit(obs, available_actions, deterministic)
        return action_logits

    def get_adv_logits(self, obs,  available_actions=None,
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
        action_logits = self.adv_actor.get_transit(obs,
                                                available_actions,
                                                deterministic)
        return action_logits

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
        stochastic = not deterministic
        actions, _, = self.actor(obs, available_actions, stochastic)
        return actions

    def act_adv(self, obs, available_actions=None, deterministic=False, agent_id=0):
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
        stochastic = not deterministic
        actions, _, = self.adv_actor(obs, available_actions, stochastic)
        return actions