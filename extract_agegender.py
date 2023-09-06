import numpy as np
import random
from dataclasses import dataclass
from typing import List
from .base import ONNXBaseTask
from .utils import (
    prepare_input_wraper, class_letterbox
)

@dataclass
class AgeGender:
    gender: str
    age: str

    gender_score: float
    age_score: float

class ExtractAgeGenderTaskV1(ONNXBaseTask):
    gender_classes: List[str]=['male', 'female']
    age_classes: List[str]=['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']

    def __call__(self, image, bounding_box: List[int]=[]) -> AgeGender:
        clone = image.copy()

        if len(bounding_box) > 0:
            x1, y1, x2, y2 = bounding_box
            clone = clone[y1:y2+1, x1:x2+1]

        return self.call(clone)

    def process_output(self, raw_outputs) -> AgeGender:
        # get logic
        gender_logic = raw_outputs[0][0][7:9]
        age_logic = raw_outputs[0][0][9:18]

        # use softmax to convert logic to probability
        gender_probs  = np.exp(gender_logic) / sum(np.exp(gender_logic))
        age_probs  = np.exp(age_logic) / sum(np.exp(age_logic))

        # get class and score
        gender_id, age_id =  np.argmax(gender_probs), np.argmax(age_probs)
        gender_score, age_score = gender_probs[gender_id], age_probs[age_id]

        return AgeGender(
            gender = self.gender_classes[gender_id],
            age = self.age_classes[age_id],
            gender_score=gender_score.tolist(),
            age_score=age_score.tolist()
        )
    
    def setup_prepare_input(self):
        return prepare_input_wraper(
            inter=1, 
            color_space="RGB",
            mean=[0.485, 0.456, 0.406], 
            std =[0.229, 0.224, 0.225],
            is_scale=True
        )

class ExtractAgeGenderTaskV2(ONNXBaseTask):
    gender_classes: List[str]=['male', 'female']
    age_classes: List[str]=['0-5', '6-10', '11-15', '16-18', '19-24', '25-34', '35-44', '45-54', '55-64', '65+']
    min_age: int=1
    max_age: int=95
    avg_age: int=48
    
    def __call__(self, image, bounding_box: List[int]=[]) -> AgeGender:
        if len(bounding_box) > 0:
            x1, y1, x2, y2 = bounding_box
            image = image[y1:y2+1, x1:x2+1]

        input_height, input_width = self.input_metadata.shape[-2:]
        img = class_letterbox(image, new_shape=(input_height, input_width))

        return self.call(img)

    def process_output(self, raw_outputs) -> AgeGender:
        # get gender id and score
        gender_logic = raw_outputs[0][0][:-1]
        gender_probs = np.exp(gender_logic) / sum(np.exp(gender_logic))

        gender_id, gender_score = np.argmax(gender_probs), np.max(gender_probs)

        # get age regression
        age_reg = raw_outputs[0][0][-1]
        age = age_reg * (self.max_age - self.min_age) + self.avg_age
        age = round(age, 2)

        for cat in self.age_classes:
            s, e = cat.split("-")
            if int(s) <= age <= int(e):
                age_cat = cat
                break

        return AgeGender(
            gender = self.gender_classes[gender_id],
            age = age_cat,
            gender_score=gender_score.tolist(),
            age_score=age.tolist()
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
    task = ExtractAgeGenderTaskV2(str(__dir__.parent/args.weight_path))
    res  = task(
        cv2.imread(str(__dir__.parent/args.image_path))
    )
    print(res)