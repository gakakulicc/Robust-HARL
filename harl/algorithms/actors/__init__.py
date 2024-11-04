"""Algorithm registry."""
from harl.algorithms.actors.happo import HAPPO
from harl.algorithms.actors.happo_advt import HAPPOAdvt
from harl.algorithms.actors.happo_traitor import HAPPOTraitor
from harl.algorithms.actors.hatrpo import HATRPO
from harl.algorithms.actors.haa2c import HAA2C
from harl.algorithms.actors.haddpg import HADDPG
from harl.algorithms.actors.haddpg_advt import HADDPGAdvt
from harl.algorithms.actors.haddpg_traitor import HADDPGTraitor
from harl.algorithms.actors.hatd3 import HATD3
from harl.algorithms.actors.hatd3_advt import HATD3Advt
from harl.algorithms.actors.hatd3_traitor import HATD3Traitor
from harl.algorithms.actors.hasac import HASAC
from harl.algorithms.actors.masac import MASAC
from harl.algorithms.actors.hasac_advt import HASACAdvt
from harl.algorithms.actors.hasac_traitor import HASACTraitor
from harl.algorithms.actors.had3qn import HAD3QN
from harl.algorithms.actors.maddpg import MADDPG
from harl.algorithms.actors.matd3 import MATD3
from harl.algorithms.actors.mappo import MAPPO
from harl.algorithms.actors.mappo_advt import MAPPOAdvt
from harl.algorithms.actors.mappo_traitor import MAPPOTraitor
from harl.algorithms.actors.mappo_state import MAPPOState

ALGO_REGISTRY = {
    "happo": HAPPO,
    "happo_advt": HAPPOAdvt,
    "mhappo_traitor": HAPPOTraitor,
    "hatrpo": HATRPO,
    "haa2c": HAA2C,
    "haddpg": HADDPG,
    "haddpg_advt": HADDPGAdvt,
    "haddpg_traitor": HADDPGTraitor,
    "hatd3": HATD3,
    "hatd3_advt": HATD3Advt,
    "hatd3_traitor": HATD3Traitor,
    "hasac": HASAC,
    "masac": MASAC,
    "hasac_advt": HASACAdvt,
    "hasac_traitor": HASACTraitor,
    "had3qn": HAD3QN,
    "maddpg": MADDPG,
    "matd3": MATD3,
    "mappo": MAPPO,
    "mappo_advt": MAPPOAdvt,
    "mappo_traitor": MAPPOTraitor,
    "mappo_state": MAPPOState,
}
