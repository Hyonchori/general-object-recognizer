import numpy as np
import torch

from .for_train.image_augmentations import letterbox


def segment2box(segment):
    # Convert 1 segment label to 1 box label, (xy1, xy2, ...) -> (xyxy)
    x, y = segment.T
    return np.array([x.min(), y.min(), x.max(), y.max()])


def filtering_labels(labels, img_w, img_h, iou_thr=0.3):
    # Filtering a labels smaller than the threshold in the ratio
    # between the size of viewable and the original
    for label in labels:
        if label == "bbox":
            bbox = labels[label]
            valid_indices = filtering_bboxes_indices(bbox, img_w, img_h, iou_thr)
            labels[label] = labels[label][valid_indices]
        elif label == "segmentation":
            segment = labels[label][:, :-1]
            xs = segment[:, 0::2]
            ys = segment[:, 1::2]
            bbox = np.stack([
                np.min(xs, axis=1), np.min(ys, axis=1), np.max(xs, axis=1), np.max(ys, axis=1)
            ]).T
            valid_indices = filtering_bboxes_indices(bbox, img_w, img_h, iou_thr)
            labels[label] = labels[label][valid_indices]
    return labels


def filtering_bboxes_indices(bbox, img_w, img_h, iou_thr=0.3):
    # Filtering a bbox(n, 5) smaller than the threshold in the ratio
    # between the size of viewable and the original
    origin_area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
    vbox = np.zeros((len(bbox), 4))
    vbox[:, 0] = np.max((bbox[:, 0], np.zeros(len(bbox))), axis=0)
    vbox[:, 1] = np.max((bbox[:, 1], np.zeros(len(bbox))), axis=0)
    vbox[:, 2] = np.min((bbox[:, 2], np.ones(len(bbox)) * img_w), axis=0)
    vbox[:, 3] = np.min((bbox[:, 3], np.ones(len(bbox)) * img_h), axis=0)
    viewable_area = (vbox[:, 2] - vbox[:, 0]) * (vbox[:, 3] - vbox[:, 1])
    iou = viewable_area / origin_area
    valid_indices = iou > iou_thr
    return valid_indices


class Preprocessing:
    def __init__(
            self,
            img_size=(720, 1280),
            scaling: bool = True,
            normalize: bool = True,
            mean=np.array([0.485, 0.456, 0.406]),
            std=np.array([0.299, 0.224, 0.225]),
            bgr2rgb: bool = True,
            swap: bool = True,
            swap_channels=(2, 0, 1),
            contiguous: bool = True,
            to_tensor: bool = True
    ):
        self.img_size = img_size
        self.scaling = scaling
        self.normalize = normalize
        self.bgr2rgb = bgr2rgb
        self.swap = swap
        self.contiguous = contiguous
        self.to_tensor = to_tensor

        self.mean = mean
        self.std = std
        self.swap_channels = swap_channels

    def __call__(self, img: np.ndarray, labels=None, img_size=None):
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if img_size is not None or self.img_size is not None:
            img_size = img_size if img_size is not None else self.img_size
            img, labels, _, _ = letterbox(img, labels, img_size, auto=False, dnn_pad=True)
        if self.scaling or self.normalize:
            img /= 255.0
        if self.normalize:
            img -= self.mean
            img /= self.std
        if self.bgr2rgb:
            img = img[..., ::-1]
        if self.swap:
            img = img.transpose(self.swap_channels)
        if self.contiguous:
            img = np.ascontiguousarray(img)
        if self.to_tensor:
            img = torch.from_numpy(img)
            if labels is not None:
                for label in labels:
                    labels[label] = torch.from_numpy(labels[label])
        return img, labels

