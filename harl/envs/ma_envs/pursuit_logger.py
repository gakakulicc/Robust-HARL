from harl.common.base_logger import BaseLogger

class PursuitLogger(BaseLogger):
    def get_task_name(self):
        obs_mode = self.env_args["obs_mode"].replace("_", "")
        dynamics = self.env_args["dynamics"].replace("_", "")
        evader_policy = self.env_args["evader_policy"]
        return f"{obs_mode}-{dynamics}-{evader_policy}"
        # return f"{self.env_args['scenario']}"

    def eval_per_step(self, eval_data):
        """Log evaluation information per step."""
        (
            eval_obs,
            eval_share_obs,
            eval_rewards,
            eval_dones,
            eval_infos,
            eval_available_actions,
        ) = eval_data
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            self.one_episode_rewards[eval_i].append(eval_infos[eval_i][0]['true_rewards'][0][0])
        self.eval_infos = eval_infos