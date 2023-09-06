import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from numpy.typing import NDArray
from .base import ONNXBaseTask
from .utils import (
    prepare_input_wraper, align_face
)

@dataclass
class Emotion:
    emotion: bool
    score: float

class ExtractEmotionTask(ONNXBaseTask):
    classes = ['Anger', 'Disgust', 'Fear', 'Sadness', 'Happiness', 'Surprise', 'Neutral', 'Contempt']

    def __call__(self, image, bounding_box: List[int]=[], landmark: List[int]=[]) -> Emotion:
        clone = image.copy()

        if len(bounding_box) > 0 and len(landmark) > 0:
            try:
                clone = align_face(clone, bounding_box, landmark, True)
            except:
                clone = align_face(clone, bounding_box, landmark, False)
        
        return self.call(clone)
    
    def process_output(self, raw_outputs) -> Emotion:
        class_id    = np.argmax(raw_outputs[0][0])
        class_score = raw_outputs[0][0][class_id]

        return Emotion(
            emotion=self.classes[class_id],
            score=class_score.tolist()
        )
    
    def setup_prepare_input(self):
        return prepare_input_wraper(
            inter=1, 
            color_space="RGB",
            mean=[123.675, 116.28, 103.53], 
            std =[58.395, 57.12, 57.375],
            is_scale=False
        )

if __name__ == "__main__":
    import cv2
    from pathlib import Path
    from .utils import parse_args
    __dir__ = Path(__file__).parent

    args = parse_args()
    task = ExtractEmotionTask(str(__dir__.parent/args.weight_path))
    res  = task(
        cv2.imread(str(__dir__.parent/args.image_path))
    )
    print(res)