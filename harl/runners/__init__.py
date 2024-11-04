"""Runner registry."""
from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.runners.on_policy_ha_runner_advt import OnPolicyHARunnerAdvt
from harl.runners.on_policy_ha_runner_traitor import OnPolicyHARunnerTraitor
from harl.runners.on_policy_ma_runner import OnPolicyMARunner
from harl.runners.on_policy_ma_runner_advt import OnPolicyMARunnerAdvt
from harl.runners.on_policy_ma_runner_traitor import OnPolicyMARunnerTraitor
from harl.runners.on_policy_ma_runner_state import OnPolicyMARunnerState
from harl.runners.off_policy_ha_runner import OffPolicyHARunner
from harl.runners.off_policy_ha_runner_advt import OffPolicyHARunnerAdvt
from harl.runners.off_policy_ha_runner_traitor import OffPolicyHARunnerTraitor
from harl.runners.off_policy_ma_runner import OffPolicyMARunner

RUNNER_REGISTRY = {
    "happo": OnPolicyHARunner,
    "happo_advt": OnPolicyHARunnerAdvt,
    "happo_traitor": OnPolicyHARunnerTraitor,
    "hatrpo": OnPolicyHARunner,
    "haa2c": OnPolicyHARunner,
    "haddpg": OffPolicyHARunner,
    "haddpg_advt": OffPolicyHARunnerAdvt,
    "haddpg_traitor": OffPolicyHARunnerTraitor,
    "hatd3": OffPolicyHARunner,
    "hatd3_advt": OffPolicyHARunnerAdvt,
    "hatd3_traitor": OffPolicyHARunnerTraitor,
    "hasac": OffPolicyHARunner,
    "hasac_advt": OffPolicyHARunnerAdvt,
    "hasac_traitor": OffPolicyHARunnerTraitor,
    "had3qn": OffPolicyHARunner,
    "maddpg": OffPolicyMARunner,
    "masac": OffPolicyMARunner,
    "matd3": OffPolicyMARunner,
    "mappo": OnPolicyMARunner,
    "mappo_advt": OnPolicyMARunnerAdvt,
    "mappo_traitor": OnPolicyMARunnerTraitor,
    "mappo_state": OnPolicyMARunnerState,
}
