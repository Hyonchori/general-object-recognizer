# Image augmentation techniques
import copy
import math
import random

import cv2
import numpy as np

from ..data_utils import segcls2seg


def resample_segment(
        segment: list,
        n: int = 500
):
    # Up-sample an (s + 1) segment
    segment += segment[0: 2]
    xs = segment[0::2]
    ys = segment[1::2]
    xy = np.concatenate([xs, ys]).reshape(2, -1).T

    x = np.linspace(0, len(xy) - 1, n)
    xp = np.arange(len(xy))
    xy = np.concatenate([np.interp(x, xp, xy[:, i]) for i in range(2)]).reshape(2, -1).T
    return xy.reshape(-1).tolist()


def get_mosaic_coordinate(mosaic_index, xc, yc, w, h, input_h, input_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


def get_mosaic_img(img_list, label_list, shape_list, input_h, input_w):
    mosaic_labels = {k: [] for k in label_list[0]}

    # mosaic center x, y
    yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
    xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

    for i, (img, _labels, (h0, w0)) in enumerate(zip(img_list, label_list, shape_list)):
        scale = min(1. * input_h / h0, 1. * input_w / w0)
        img = cv2.resize(
            img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
        )
        h, w, _ = img.shape
        if i == 0:
            mosaic_img = np.full((input_h * 2, input_w * 2, 3), 114, dtype=np.uint8)

        # large image box and small image box
        (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
            i, xc, yc, w, h, input_h, input_w
        )
        mosaic_img[l_y1: l_y2, l_x1: l_x2] = img[s_y1: s_y2, s_x1: s_x2]
        padw, padh = l_x1 - s_x1, l_y1 - s_y1

        labels = copy.deepcopy(_labels)
        labels["cls"][0] = 3
        for label in labels:
            if len(labels[label]) > 0:
                if label in ["bbox", "segmentation"]:
                    labels[label] = scale_and_shift_labels(labels[label], scale, padw, padh)
                mosaic_labels[label].append(labels[label])
    print("mosaic")
    for label in mosaic_labels:
        mosaic_labels[label] = np.concatenate(mosaic_labels[label], 0)
        print(f"\t{label}: {mosaic_labels[label].shape} ->")
        if label == "bbox":
            valid_indices = get_valid_bbox_indices(
                mosaic_labels[label], 2 * input_w, 2 * input_h
            )
            mosaic_labels[label] = mosaic_labels[label][valid_indices]
        elif label == "segmentation":
            valid_indices = get_valid_segment_indices(
                mosaic_labels[label], 2 * input_w, input_h
            )
            mosaic_labels[label] = mosaic_labels[label][valid_indices]
        print(f"\t{label}: {mosaic_labels[label].shape}")

    return mosaic_img, mosaic_labels


def scale_and_shift_labels(labels, scale, w_shift, h_shift):
    # scale and shift segments(n, (4 or s) + 1)
    if isinstance(scale, tuple):
        x_scale = scale[0]
        y_scale = scale[1]
    else:
        x_scale = y_scale = scale
    labels[:, 0::2][:, :-1] = labels[:, 0::2][:, :-1] * x_scale + w_shift
    labels[:, 1::2] = labels[:, 1::2] * y_scale + h_shift
    return labels


def get_valid_bbox_indices(bboxes, width, height):
    # filtering bboxes(n, 5) that exist in out of image
    valid_indices = []
    for bbox in bboxes:
        valid = bbox[0] < width and bbox[1] < height and bbox[2] > 0 and bbox[3] > 0
        valid_indices.append(valid)
    return valid_indices


def get_valid_segment_indices(segments, width, height):
    # filtering segments(n, s + 1) that exist in out of image
    valid_indices = []
    for segment in segments:
        seg = segcls2seg(segment)
        x, y = seg.T
        bbox = [x.min(), y.min(), x.max(), y.max()]
        valid = bbox[0] < width and bbox[1] < height and bbox[2] > 0 and bbox[3] > 0
        valid_indices.append(valid)
    return valid_indices


def random_perspective(
        img: np.ndarray,
        labels: dict,
        degrees: int = 10,
        translate: float = 0.1,
        scale: float = 0.1,
        shear: float = 10.0,
        perspective: float = 0.0,
        border=(0, 0)
):
    height = img.shape[0] + border[0] * 2
    width = img.shape[1] + border[1] * 2

    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation(pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation(pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective(about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective(about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(scale[0], scale[1])
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear(deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear(deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations is IMPORTANT(right to left)
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    print("\nperspective")
    for label in labels:
        print(f"\t{label}: {labels[label].shape} ->")
        if label == "bbox":
            bboxes = labels[label]
            n = len(bboxes)
            xy = np.ones((n * 4, 3))
            xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
            xy = xy @ M.T
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            bboxes[:, :4] = new

        elif label == "segmentation":
            segments = labels[label]
            for i, segment in enumerate(segments):
                seg = segcls2seg(segment)
                xy = np.ones((len(seg), 3))
                xy[:, :2] = seg
                xy = xy @ M.T
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]
                xy = xy.reshape(-1)
                labels[label][i][:len(xy)] = xy
        print(f"\t{label}: {labels[label].shape}")
    return img, labels


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
    # box1(4,n), box2(4,n)
    # Compute candidate boxes which include following 5 things:
    # box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (
        (w2 > wh_thr)
        & (h2 > wh_thr)
        & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
        & (ar < ar_thr)
    )  # candidates

