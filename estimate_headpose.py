from dataclasses import dataclass
from typing import List
from .base import ONNXBaseTask
from .utils import (
    prepare_input_wraper, align_face
)

@dataclass
class HeadPose:
    roll: float
    yaw: float
    pitch: float
    
class EstimateHeadPoseTask(ONNXBaseTask):
    def __call__(self, image, bounding_box: List[int]=[], landmark: List[int]=[]) -> HeadPose:
        clone = image.copy()

        if len(bounding_box) > 0 and len(landmark) > 0:
            try:
                clone = align_face(clone, bounding_box, landmark, True)
            except:
                clone = align_face(clone, bounding_box, landmark, False)
        
        return self.call(clone)

    def process_output(self, raw_outputs) -> HeadPose:
        roll, yaw, pitch = raw_outputs

        return HeadPose(
            roll=roll[0].tolist(),
            yaw=yaw[0].tolist(),
            pitch=pitch[0].tolist()
        )
    
    def setup_prepare_input(self):
        return prepare_input_wraper(
            inter=1, 
            color_space="RGB",
            mean=.5, 
            std =.5,
            is_scale=True
        )

if __name__ == "__main__":
    import cv2
    from pathlib import Path
    from .utils import parse_args
    __dir__ = Path(__file__).parent

    args = parse_args()
    task = EstimateHeadPoseTask(str(__dir__.parent/args.weight_path))
    res  = task(
        cv2.imread(str(__dir__.parent/args.image_path))
    )
    print(res)