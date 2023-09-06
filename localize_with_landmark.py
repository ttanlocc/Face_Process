import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from numpy.typing import NDArray
from .base import ONNXBaseTask
from .utils import (
    prepare_input_wraper, 
    prior_box, decode_boxes, decode_landmarks, 
    non_max_suppression, get_input_size, get_largest_bbox
)

@dataclass
class LocationWithLandmark:
    # score: float
    bounding_box: List[int]
    landmark: List[int] # left_eye, right_eye, nose, left_mouth, right_mouth

class LocalizeTask(ONNXBaseTask):
    steps           = [8, 16, 32]
    min_sizes       = [[16, 32], [64, 128], [256, 512]]
    variances       = [0.1, 0.2]
    nms_threshold   = 0.3
    keep_top_k      = 150

    def __init__(self, weight: str, score_threshold=0.6, limit_side_len=320):
        self.score_threshold = score_threshold
        self.limit_side_len = limit_side_len
        super().__init__(weight)

    
    def __call__(self, image, multi_outputs: bool=False) -> List[LocationWithLandmark]:
        return self.call(image, multi_outputs=multi_outputs)
    
    def call(self, image, multi_outputs: bool=False) -> List[LocationWithLandmark]:
        # variables for prepare_input and process_output
        image_height, image_width= image.shape[:-1]
        input_height, input_width = get_input_size(
            image_height=image_height,
            image_width=image_width,
            limit_side_len=self.limit_side_len
        )

        # predict
        input_value = self.prepare_input(image, height=input_height, width=input_width)
        raw_outputs = self.run_session(input_value)

        # process output
        outputs = self.process_output(
            raw_outputs, 
            target_size=[input_height, input_width],
            image_size=[image_height, image_width])
        
        list_output = []
        if len(outputs[0]) > 0:
            # deal with case that multi_outputs not be used
            if not multi_outputs:
                list_bbox, list_score, list_landmark = outputs
                largest_bbox_index = get_largest_bbox(list_bbox)

                outputs = (
                    list_bbox[[largest_bbox_index]], 
                    list_score[[largest_bbox_index]], 
                    list_landmark[[largest_bbox_index]]
                )
            
            # wrap output
            for (bbox, score, landmark) in zip(*outputs):
                list_output += [
                    LocationWithLandmark(
                        # score=score.tolist(),
                        bounding_box=bbox.tolist(),
                        landmark=landmark.tolist()
                    )
                ]
                
        return list_output
            
    def process_output(self, raw_outputs, target_size: List[int], image_size: List[int]) -> Tuple[NDArray]:
        '''
            target_size: [InputHeight, InputWidth]
            image_size : [ImageHeight, ImageWidth]
        '''
        # extract outputs
        list_bbox       = raw_outputs[0][0, :]
        list_score      = raw_outputs[1][0,:,1]
        list_landmark   = raw_outputs[2][0,:]

        # remove bbox with low confidence
        if max(list_score) <= self.score_threshold:
            return [], [], []

        # decode bbox and landmark
        priors = prior_box(
            width=target_size[1], 
            height=target_size[0], 
            steps=self.steps, 
            min_sizes=self.min_sizes
        )
        
        list_bbox = decode_boxes(
           list_bbox, priors, self.variances, image_size[::-1]
        )

        list_landmark = decode_landmarks(
            list_landmark, priors, self.variances, image_size[::-1]
        )
        # update bbox, landmark, score
        index = np.where(list_score >= self.score_threshold)[0]
        list_bbox, list_score, list_landmark = (
            list_bbox[index], 
            list_score[index], 
            list_landmark[index]
        )
            

        # do nms
        nms_index  = non_max_suppression(
            bboxes=list_bbox, 
            scores=list_score, 
            thresh=self.nms_threshold,
            keep_top_k=self.keep_top_k, 
            mode="Union"
        )
        outputs = (
                list_bbox[nms_index].astype(np.int64), 
                list_score[nms_index], 
                list_landmark[nms_index].astype(np.int64)
        )
        
        return outputs
    
    def setup_prepare_input(self):
        return prepare_input_wraper(
            inter=1, 
            color_space="RGB",
            mean=[104., 117., 123.], 
            std=[1.  , 1.  , 1. ], 
            is_scale=False,
        )

if __name__ == "__main__":
    import cv2
    from pathlib import Path
    from .utils import parse_args
    __dir__ = Path(__file__).parent

    args = parse_args()
    task = LocalizeTask(str(__dir__.parent/args.weight_path))
    result  = task(
        cv2.imread(str(__dir__.parent/args.image_path))
    )
    print(result)

    # visualize
    colors = [(255, 0, 0), (255, 255, 0), (255, 255, 255), (0, 255, 255), (0, 0, 255)]

    image  = cv2.imread(str(__dir__.parent/args.image_path))
    for loc in result:
        if len(loc.bounding_box) == 0:
            continue

        # draw rectangle around object
        start_point, end_point = loc.bounding_box[:2], loc.bounding_box[2:]
        cv2.rectangle(image, tuple(start_point), tuple(end_point), (255, 0, 0), 2)
        
        # draw 5 keypoints's face
        landmark = np.reshape(loc.landmark, (-1, 2))
        for (x, y), color in zip(landmark, colors):
            cv2.circle(image, (x, y), 1, color, 4)

    cv2.imshow("localize", image)
    cv2.waitKey(0)