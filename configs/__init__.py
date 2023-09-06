from .dataclass import *
from pathlib import Path
import yaml

def load_config() -> TaskConfig:
    f = (Path(__file__).parent/"config.yaml").open("r")
    yaml_data = yaml.safe_load(f)
    
    config = TaskConfig(**yaml_data)
    return config 

TASK_CONFIG = load_config()
