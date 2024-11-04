<h1 align="center"> HARL robustness test </h1>

This repository is a robustness testing code for HARL. In order to conduct extensive robustness testing on HASAC and benchmarks, we made modifications to the HARL code by integrating it into the Pursuit Evade environment and introducing different noise and attack methods, including Environment Uncertainty，State Uncertainty, Reward Uncertainty, Action Uncertainty。
- Environment Uncertainty: Differences between simulation and reality, such as friction or weight changes, cause dynamics uncertainties. Adds Gaussian noise to agent positions. Tested on varying texture surfaces in real-world scenarios. We add noise in the environment to test the robustness of the trained model under environmental uncertainty. You can find the relevant code under `harl/envs/ma_envs/base`.
- State Uncertainty: Sensing inaccuracies can lead to state uncertainties. Introduces worst-case noise using PGD to agent states. Evaluates with varying $\ell_\infty$-bounded perturbation $\epsilon$. We add noise in agents' sending to test the robustness of the trained model in State Uncertainty. You can find relevant codes with the suffix `_traitor` under the `harl/runner`.
- Reward Uncertainty: Inaccurate or poorly defined rewards lead to reward uncertainties. Introduces truncated Gaussian noise to rewards during training, evaluates without noise. We add noise during reward to test the robustness of model training to Reward Uncertainty. You can find the relevant code under `harl/envs/ma_envs/envs/point_envs/pursuit_evasion`.
- Action Uncertainty: Software/hardware errors or adversarial actions can disrupt agent actions. Trains agents with worst-case policies to minimize team rewards, tests in both simulations and real-world. We replace an agent with an adaptive to test the robustness of the action uncertainty of the model under an agent is comparable. You can find the relevant code with the suffix `_advt` under `harl/runner`and `harl/algorithms`.

## Installation
Please note that this code is based on HARL, so the necessary environment dependencies for HARL should be downloaded before using this code. The installed instructions are similar to HARL:
```shell
conda create -n harl python=3.8
conda activate harl
# Install pytorch>=1.9.0 (CUDA>=11.0) manually
git clone https://github.com/gakakulicc/Robust-HARL.git
cd Robust-HARL
pip install -e .
```
In addition, you may need to install several environment support, including [Gym](https://www.gymlibrary.dev/), [SMAC](https://github.com/oxwhirl/smac), [SMACv2](https://github.com/oxwhirl/smacv2), [MAMuJoCo](https://github.com/schroederdewitt/multiagent_mujoco), [MPE](https://pettingzoo.farama.org/environments/mpe/), [Google Research Football](https://github.com/google-research/football), [Bi-DexterousHands](https://github.com/PKU-MARL/DexterousHands), [Light Aircraft Game](https://github.com/liuqh16/CloseAirCombat). If you have not installed these environments, please install them according to the official instructions or the [HARL](https://github.com/PKU-MARL/HARL) tutorial


## Usage
If you want to use this code, you can obtain the relevant code in the corresponding uncertainty experiment folder. You can modify the environment related parameters in `harl/configs/envs_cfgs`, and modify the algorithm related parameters or import the trained model in `harl/configs/algos_cfgs`. The default parameters in these two directories are the ones we have adjusted, and you can use these parameters to reproduce our results. 

In all experiments, you can use the following commands to conduct the experiment: 
```shell
python train.py --exp<EXPERIMENT> --algo<ALGO> --env<ENV> --noise<NOISE> --exp_name<EXPOMENT NAME> 
```
`exp` represents robustness experiments under various noises, the options are `"env"`, `"state"`, `"reward"`, and `"action"`. `algo` is the selected algorithm, we give four options: `"mappo"`, `"happo"`, `"hatd3"` and `"hasac"`. `env` is the experimental environment, in this experiment, only `"pursuit"` is used; `noise` is the noise introduced in the experiment, and the input type is `float`. The recommended input range here is `0-2`. In the action uncertainty experiment, the noise parameter does not work. `exp_name` is the name of the folder where you want to save the experiment results, and the input type is `str`.
It should be noted that in the experiment of reward uncertainty, environmental uncertainty and state uncertainty, the trained model should be imported before the experiment. 

These methods can also refer to the shell examples provided. You can modify the required parameters and then run the corresponding shell script.
