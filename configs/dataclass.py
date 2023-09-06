from dataclasses import dataclass, field
from typing import Dict, Union

PARAM = Dict[str, Union[int,float]]
WEIGHT = str
NETWORK = Dict[str, Union[PARAM, WEIGHT]]

@dataclass
class TaskConfig:
    localize            : Dict[str, NETWORK]
    check_liveness      : Dict[str, NETWORK]
    check_mask          : Dict[str, NETWORK]
    estimate_headpose   : Dict[str, NETWORK]
    extract_vector      : Dict[str, NETWORK]
    extract_emotion     : Dict[str, NETWORK]
    extract_agegender   : Dict[str, NETWORK]
