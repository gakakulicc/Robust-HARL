<h1 align="center"> HARL robustness test </h1>

This repository is a robustness testing code for HARL. In order to conduct extensive robustness testing on HASAC and benchmarks, we made modifications to the HARL code by integrating it into the Pursuit Evade environment and introducing different noise and attack methods, including Environment Uncertainty，State Uncertainty, Reward Uncertainty, Action Uncertainty。
- Environment Uncertainty: add noise in the environment to test the robustness of the trained model under environmental uncertainty. You can find the relevant code under `env/harl/envs/ma_envs/base`.
- State Uncertainty: add noise in agents' sending to test the robustness of the trained model in State Uncertainty. You can find the relevant code under `state/harl/runner`.
- Reward Uncertainty: add noise during reward to test the robustness of model training to Reward Uncertainty. You can find the relevant code under `reward/harl/envs/ma_envs/envs/point_envs/pursuit_evasion`.
- Action Uncertainty: replace an agent with an adaptive to test the robustness of the action uncertainty of the model under an agent is comparable. You can find the relevant code under `action/harl/runner`and `action/harl/algorithms`.

## Installation
Please note that this code is based on HARL, so the necessary environment dependencies for HARL should be downloaded before using this code. 

## Usage
If you want to use this code, you can obtain the relevant code in the corresponding uncertainty experiment folder. You can modify the environment related parameters in `harl/configs/envs_cfgs`, and modify the algorithm related parameters or import the trained model in `harl/configs/algos_cfgs`. The default parameters in these two directories are the ones we have adjusted, and you can use these parameters to reproduce our results. 

In the Reward Uncertainty and Action Uncertainty experiments, you can use the following commands to conduct the experiment: 
```shell
python train.py --Algo<ALGO> --env<ENV> --exp_name<EXPOMENT NAME> 
```

In the Environment Uncertainty, State Uncertainty experiment, you can use the following commands to conduct the experiment: 
```shell
python train.py --Algo<ALGO> --env<ENV> --noise<NOISE> --exp_name<EXPOMENT NAME> 
```
These usage methods can also refer to the `train.sh` example provided, modify the parameters you need, and run train.sh.
