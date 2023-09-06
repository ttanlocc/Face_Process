import numpy as np
import random
from dataclasses import dataclass
from typing import List
from .base import ONNXBaseTask
from .utils import (
    prepare_input_wraper, get_new_box
)

@dataclass
class LivenessValue:
    is_real: bool
    score: float

class CheckLivenessTask(ONNXBaseTask):
    classes: List[bool]=[False, True, False] #["Fake", "Real", "Fake"]

    def __init__(self, weight: str, real_threshold: float=0.915):
        self.scale = float(weight[:3])
        self.real_threshold = real_threshold
        
        super().__init__(weight)


    def __call__(self, image, bounding_box: List=[]) -> LivenessValue:
        if len(bounding_box) > 0:
            image_height, image_width = image.shape[:-1]

            x1, y1, x2, y2 = get_new_box(
                bbox=bounding_box,
                src_h=image_height,
                src_w=image_width,
                scale=self.scale
            )
            image = image[y1:y2+1, x1:x2+1]
            
        return self.call(image)
    
    def process_output(self, raw_outputs) -> LivenessValue:
        # make softmax to migrate probabilities belong to [fake, real, fake]
        probs  = np.exp(raw_outputs[0][0]) / sum(np.exp(raw_outputs[0][0]))

        # get class id and class score
        if probs[1] >= self.real_threshold:
            class_id = 1
            class_score = probs[1]

        else:
            class_id = random.choice([0,2])
            class_score = 1-probs[1]

        return LivenessValue(
            is_real=self.classes[class_id],
            score=class_score.tolist()
        )
    
    def setup_prepare_input(self):
        return prepare_input_wraper(
            inter=1, 
            color_space="BGR",
            mean=None, 
            std =None,
            is_scale=False
        )

if __name__ == "__main__":
    import cv2
    from pathlib import Path
    from .utils import parse_args
    __dir__ = Path(__file__).parent

    args = parse_args()
    task = CheckLivenessTask(str(__dir__.parent/args.weight_path))
    res  = task(
        cv2.imread(str(__dir__.parent/args.image_path))
    )
    print(res)