import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from numpy.typing import NDArray
from .base import ONNXBaseTask
from .utils import (
    prepare_input_wraper
)

@dataclass
class MaskValue:
    is_wear: bool
    score: float

class CheckMaskTask(ONNXBaseTask):
    classes = [True, False] #["Mask", "WithoutMask"]

    def __call__(self, image, bounding_box: List=[]) -> MaskValue:
        clone = image.copy()

        if len(bounding_box) > 0:
            x1, y1, x2, y2 = bounding_box
            clone = clone[y1:y2+1, x1:x2+1]

        return self.call(clone)
    
    def process_output(self, raw_outputs) -> MaskValue:
        class_id    = np.argmax(raw_outputs[0][0])
        class_score = raw_outputs[0][0][class_id]

        return MaskValue(
            is_wear=self.classes[class_id],
            score=class_score.tolist()
        )
    
    def setup_prepare_input(self):
        return prepare_input_wraper(
            inter=1, 
            color_space="RGB",
            mean=[0.485, 0.456, 0.406], 
            std =[0.229, 0.224, 0.225],
            is_scale=True
        )

if __name__ == "__main__":
    import cv2
    from pathlib import Path
    from .utils import parse_args
    __dir__ = Path(__file__).parent

    args = parse_args()
    task = CheckMaskTask(str(__dir__.parent/args.weight_path))
    res  = task(
        cv2.imread(str(__dir__.parent/args.image_path))
    )
    print(res)