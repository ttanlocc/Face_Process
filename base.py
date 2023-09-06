import onnxruntime
import random
import numpy as np
from pathlib import Path
from numpy.typing import NDArray
from typing import Any, List
from .utils import count_gpus, get_memory_free_MiB
from abc import ABC, abstractclassmethod

__dir__ = Path(__file__).parent

class ONNXBaseTask(ABC):
    num_gpus: int = count_gpus()

    def __init__(self, weight: str) -> None:
        self.session = self.initialize_session(weight)
        self.input_metadata = self.session.get_inputs()[0]
        self.prepare_input = self.setup_prepare_input()
        
        # warmup model
        input_height, input_width = self.input_metadata.shape[-2:]
        temp = np.zeros((1, 3, int(input_height) if int(input_height) > 0 else 320, int(input_width) if int(input_width) > 0 else 320), dtype=np.float32)
        self.run_session(temp)

    @abstractclassmethod
    def process_output(self, raw_outputs: List[NDArray], **kwargs) -> Any:
        pass

    @abstractclassmethod
    def setup_prepare_input(self):
        pass

    def call(self, image) -> Any:
        input_height, input_width = self.input_metadata.shape[-2:]

        # predict
        input_value = self.prepare_input(image, height=input_height, width=input_width)
        raw_outputs = self.run_session(input_value)

        return self.process_output(raw_outputs)
    
    def run_session(self, input_value: NDArray) -> List[NDArray]:
        input_dict = {self.input_metadata.name : input_value}

        return self.session.run(None, input_dict)
    
    def initialize_session(self, weight: str):
        # get avaiable runtime
        providers=[]
        if self.num_gpus == 0:
            providers += [("CPUExecutionProvider", {})]
        else:
            providers += [(
                "CUDAExecutionProvider", 
                {
                    "device_id": random.choice([i for i in range(self.num_gpus) if get_memory_free_MiB(i) >= 1000])
                }
            )]

        # init session
        return onnxruntime.InferenceSession(
            str(__dir__.parent.parent.parent/weight),
            providers=providers
        )
        








