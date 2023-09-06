import numpy as np
import random
from dataclasses import dataclass
from typing import List
from .base import ONNXBaseTask
from .utils import prepare_input_wrapper

@dataclass
class NSFWPrediction:
    label: str
    score: float

class CheckNSFWTask(ONNXBaseTask):
    classes: List[str] = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']

    def __init__(self, weight: str, threshold: float = 0.5):
        self.threshold = threshold
        super().__init__(weight)

    def __call__(self, image) -> NSFWPrediction:
        return self.call(image)

    def process_output(self, raw_outputs) -> NSFWPrediction:
        probabilities = np.exp(raw_outputs[0][0]) / sum(np.exp(raw_outputs[0][0]))
        max_prob_index = np.argmax(probabilities)
        max_prob_score = probabilities[max_prob_index]

        if max_prob_score >= self.threshold:
            predicted_class = self.classes[max_prob_index]
        else:
            predicted_class = 'unknown'

        return NSFWPrediction(label=predicted_class, score=max_prob_score.tolist())

    def setup_prepare_input(self):
        return prepare_input_wrapper(
            inter=1,
            color_space="RGB",
            mean=None,
            std=None,
            is_scale=False
        )

if __name__ == "__main__":
    import cv2
    from pathlib import Path
    from .utils import parse_args
    __dir__ = Path(__file__).parent

    args = parse_args()
    task = CheckNSFWTask(str(__dir__.parent / args.weight_path))
    res = task(
        cv2.imread(str(__dir__.parent / args.image_path))
    )
    print(res)
