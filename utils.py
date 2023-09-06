import numpy as np
import cv2
import subprocess
import math
from itertools import product as product
from numpy.typing import NDArray
from typing import List
import argparse
import pynvml
from dataclasses import dataclass
from skimage import transform

def parse_args():
    @dataclass
    class Argument:
        image_path: str
        weight_path: str

    # parse argument
    parser = argparse.ArgumentParser(
        prog="Run AI Tasks",
        description="call builded task belong to Face",
    )
    parser.add_argument(
        "--image", type=str, default="samples/An_2000.jpg", help="path to tested image"
    )
    parser.add_argument(
        "--weight", type=str, default="weights/retinaface_mobilev3.onnx", help="path to weight"
    )

    args = parser.parse_args()
    return Argument(
        image_path=args.image,
        weight_path=args.weight
    )
    
def get_memory_free_MiB(gpu_index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.free // 1024 ** 2

def count_gpus():
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=count', '--format=csv,noheader'], encoding='utf-8')
        num_gpus = int(output.strip().split('\n')[0])
    except subprocess.CalledProcessError:
        num_gpus = 0
    
    return num_gpus

def prepare_input_wraper(inter=1, mean=None, std=None, channel_first=True, color_space="BGR", is_scale=False):
    '''
        THIS PROCESS WAY WILL OPTIMIZE RUNTIME (scaling will bit slower than no scaling)
        ==========================================================================
        inter: resize type (0: Nearest, 1: Linear, 2: Cubic)

        is_scale: whether we scale image in range(0,1) to normalize or not
            NOTE: image normalize with scale DIFFERENT normalize no scale
        mean: expected value of distribution
        std: standard deviation of distribution

        channel_first: True is (c,h,w), False is (h,w,c)
        color_space: BGR (default of cv2), RGB
        ==========================================================================

    '''
    if mean is not None and std is not None:
        mean = mean if isinstance(mean, list) or isinstance(mean, tuple) else [mean]*3
        std  = std if isinstance(std, list) or isinstance(std, tuple) else [std]*3

    def call(img: NDArray, width: int, height: int):
        '''
            weight: input width of input model
            height: input height of input model
        '''

        if img.shape[0] != height or img.shape[1] != width:
            image = cv2.resize(img.copy(),  (width, height), interpolation=inter)
        else:
            image = img.copy()
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if color_space == "RGB" else image
        image = image.transpose((2,0,1)) if channel_first else image
        image = image.astype(np.float32)

        # scale image in range(0,1)
        if is_scale:
            image /= 255

        if mean is not None and std is not None:
            if channel_first:
                image[0, :, :] -= mean[0]; image[1, :, :] -= mean[1]; image[2, :, :] -= mean[2]
                image[0, :, :] /= std[0] ; image[1, :, :] /= std[1] ; image[2, :, :] /= std[2]
            else:
                image[:, :, 0] -= mean[0]; image[:, :, 1] -= mean[1]; image[:, :, 2] -= mean[2]
                image[:, :, 0] /= std[0] ; image[:, :, 1] /= std[1] ; image[:, :, 2] /= std[2]

        return image[np.newaxis, :]

    return call

# =============================External Process image
def class_letterbox(im, new_shape=(640, 640), color=(0, 0, 0), scaleup=True):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    if im.shape[0] == new_shape[0] and im.shape[1] == new_shape[1]:
        return im

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    # ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im

def get_new_box(src_w: int, src_h: int, bbox: List[int], scale: float):
    x, y, xmax, ymax = bbox
    box_w = (xmax - x)
    box_h = (ymax - y)

    # Re-calculate scale ratio
    scale = min((src_h-1)/box_h, min((src_w-1)/box_w, scale))

    # get new width and height with scale ratio
    new_width = box_w * scale
    new_height = box_h * scale
    center_x, center_y = box_w/2+x, box_h/2+y

    # calculate bbox with new width and height
    left_top_x = center_x-new_width/2
    left_top_y = center_y-new_height/2
    right_bottom_x = center_x+new_width/2
    right_bottom_y = center_y+new_height/2

    # bbox must be in image
    if left_top_x < 0:
        right_bottom_x -= left_top_x
        left_top_x = 0

    if left_top_y < 0:
        right_bottom_y -= left_top_y
        left_top_y = 0

    if right_bottom_x > src_w-1:
        left_top_x -= right_bottom_x-src_w+1
        right_bottom_x = src_w-1

    if right_bottom_y > src_h-1:
        left_top_y -= right_bottom_y-src_h+1
        right_bottom_y = src_h-1

    return int(left_top_x), int(left_top_y),\
            int(right_bottom_x), int(right_bottom_y)

def align_face(image: NDArray, bounding_box: List[int], landmark: List[int], use_bbox: int=True):
    src = np.array(landmark).reshape(-1, 2)
    if use_bbox:
        # crop face
        x1, y1, x2, y2 = bounding_box
        image = image[y1:y2+1, x1:x2+1]

        # align
        src -= np.array([x1, y1])

    des = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [38.2946, 92.3655],
            [70.7299, 92.2041],
        ]
    )

    trans = transform.SimilarityTransform()
    trans.estimate(src, des)

    return cv2.warpAffine(image, trans.params[:2, :], dsize=(112, 112))

# =============================DETECT
def get_largest_bbox(bboxes: NDArray) -> NDArray:
    # compute bbox area
    hbbox, wbbox = (
        bboxes[:, 3] - bboxes[:, 1],
        bboxes[:, 2] - bboxes[:, 0],
    )
    area = hbbox*wbbox

    return np.argmax(area)

def get_input_size(image_height: int, image_width: int, limit_side_len: int) -> List[int]:
    '''
        image_size: [ImageHeight, ImageWidth]
    '''
    if max(image_height, image_width) >= limit_side_len:
        ratio = (
            float(limit_side_len) / image_height
            if image_height < image_width
            else float(limit_side_len) / image_width
        )
    else:
        ratio = 1.

    input_height = int((ratio*image_height // 32) * 32)
    input_width  = int((ratio*image_width // 32) * 32)

    return input_height, input_width

def prior_box(width: int, height: int, steps: List[int], min_sizes: List[List[int]]) -> NDArray:
    anchors = []

    feature_maps = [
        [math.ceil(height / step), math.ceil(width / step)] for step in steps
    ]
    for k, f in enumerate(feature_maps):
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes[k]:
                s_kx = min_size / width
                s_ky = min_size / height
                dense_cx = [x * steps[k] / width for x in [j + 0.5]]
                dense_cy = [y * steps[k] / height for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky]

    return np.reshape(anchors, (-1, 4))

def decode_boxes(bboxes: NDArray, priors: NDArray, variances: List[float], scale_factor: List[float]) -> NDArray:
    bboxes = np.concatenate(
        (
            priors[:, :2] + bboxes[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(bboxes[:, 2:] * variances[1]),
        ),
        axis=1,
    )

    bboxes[:, :2] -= bboxes[:, 2:] / 2
    bboxes[:, 2:] += bboxes[:, :2]

    return bboxes * np.array(scale_factor * 2)

def decode_landmarks(landmarks: NDArray, priors: NDArray, variances: List[float], scale_factor: List[float]) -> NDArray:
    landmarks = np.concatenate(
        (
            priors[:, :2] + landmarks[:, :2] * variances[0] * priors[:, 2:],
            priors[:, :2] + landmarks[:, 2:4] * variances[0] * priors[:, 2:],
            priors[:, :2] + landmarks[:, 4:6] * variances[0] * priors[:, 2:],
            priors[:, :2] + landmarks[:, 6:8] * variances[0] * priors[:, 2:],
            priors[:, :2] + landmarks[:, 8:10] * variances[0] * priors[:, 2:],
        ),
        axis=1,
    )

    return landmarks * np.array(scale_factor * 5)

def intersection_over_union(bbox: NDArray, bboxes: NDArray, mode="Union") -> NDArray:
    """
    Caculate IoU between detect and ground truth boxes
    :param crop_box:numpy array (4, )
    :param bboxes:numpy array (n, 4):x1, y1, x2, y2
    :return:
    numpy array, shape (n, ) Iou
    """
    bbox_area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
    areas = (bboxes[:, 2] - bboxes[:, 0] + 1) * (bboxes[:, 3] - bboxes[:, 1] + 1)

    xx1 = np.maximum(bbox[0], bboxes[:, 0])
    yy1 = np.maximum(bbox[1], bboxes[:, 1])
    xx2 = np.minimum(bbox[2], bboxes[:, 2])
    yy2 = np.minimum(bbox[3], bboxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h 
    if mode == "Union":
        over = inter / (bbox_area + areas - inter)

    elif mode == "Minimum":
        over = inter / np.minimum(bbox_area, areas)

    return over

def non_max_suppression(bboxes: NDArray, scores: NDArray, thresh: float, keep_top_k:int=100, mode:str="Union") -> List[int]:
    """
    Bước 1: Tính diện tích của từng bbox
    Bước 2: Sort score của từng bbox theo thứ tự giảm dần và lấy vị trí index của chúng
    Bước 3: Theo thứ tự giảm dần của score, ta lấy bbox này giao với các bbox còn lại,
    sau đó loại bỏ bớt các vị trí mà phần giao của 2 bbox lớn hơn THRESHOLD
    """
    # Sắp xếp độ tư tin giảm giần (lấy index)
    order = scores.argsort()[::-1][:keep_top_k]

    # Duyệt qua từng bbox với độ tự tin giảm dần để loại bỏ những bbox trùng nhau
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        iou = intersection_over_union(bboxes[i], bboxes[order[1:]], mode=mode)

        # keep (cập nhật lại order bằng những gì còn lại sau khi loại bỏ)
        inds = np.where(iou <= thresh)[0]  # [1,2,3,6,45,....]
        order = order[inds + 1]

    return keep