# Image preprocessing

from typing import Tuple

import cv2
import numpy as np


def letterbox(
        image: np.ndarray,
        new_shape: Tuple[int] = (640, 640),
        color: Tuple[int] = (114, 114, 114),
        auto: bool = True,
        stretch: bool = False,
        scaleup: bool = True,
        stride: int = 32,
        bboxes: np.ndarray = None
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = image.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = new_shape + (new_shape + stride) % stride
        new_shape = (new_shape, new_shape)
    else:
        new_shape = [x + (x + stride) % stride for x in new_shape]

    if image.shape[:2] == new_shape:
        return image, 1., (0, 0)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif stretch:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    if bboxes is None:
        return image, ratio, (dw, dh), None
    else:
        bboxes[:, 0::2] *= ratio[0]
        bboxes[:, 1::2] *= ratio[1]
        bboxes[:, 0::2] += dw
        bboxes[:, 1::2] += dh
        return image, ratio, (dw, dh), bboxes


def img_scaling(image: np.ndarray):
    if image.dtype == "uint8":
        image = image.astype(np.float32)
    return image / 255.0


def img_normalize(
        image: np.ndarray,
        mean: np.ndarray = np.array([0.485, 0.456, 0.406]),
        std: np.ndarray = np.array([0.299, 0.224, 0.225])
):
    if image.dtype == "uint8":
        image = image.astype(np.float32)
    image -= mean
    image /= std
    return image


def img_bgr2rgb(image: np.ndarray):
    return image[..., ::-1]


def dim_swap(image: np.ndarray, swap: Tuple[int] = (2, 0, 1)):
    # (height, width, channels) -> (channels, height, width)
    return image.transpose(swap)


def as_contiguous(image: np.ndarray):
    return np.ascontiguousarray(image, dtype=np.float32)


class Preprocessing:
    def __init__(
            self,
            img_size: Tuple[int] = 1280,
            auto: bool=True,
            scaling: bool = True,
            normalize: bool = True,
            bgr2rgb: bool = True,
            swap: bool = True,
            contiguous: bool = True
    ):
        self.img_size = img_size
        self.auto = auto
        self.scaling = scaling
        self.normalize = normalize
        self.bgr2rgb = bgr2rgb
        self.swap = swap
        self.contiguous = contiguous

    def __call__(self, image, labels=None):
        image, _, _, labels = letterbox(image, self.img_size, auto=self.auto, bboxes=labels)
        if self.scaling or self.normalize:
            image = img_scaling(image)
        if self.normalize:
            image = img_normalize(image)
        if self.bgr2rgb:
            image = img_bgr2rgb(image)
        if self.swap:
            image = dim_swap(image)
        if self.contiguous:
            image = as_contiguous(image)
        if labels is None:
            return image
        else:
            return image, labels

