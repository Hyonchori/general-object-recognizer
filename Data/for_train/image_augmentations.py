# Image augmentation techniques

import math
import random

import cv2
import numpy as np


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


def resample_segments(segments, n=1000):
    # Up-sample an (n, 2) segments
    for i, (seg, cls) in enumerate(segments):
        seg = np.append(seg, seg[0:1], axis=0)  # Interpolate empty between first and last points
        x = np.linspace(0, len(seg) - 1, n)
        xp = np.arange(len(seg))
        segments[i][0] = np.concatenate([np.interp(x, xp, seg[:, i]) for i in range(2)]).reshape(2, -1).T
    return segments


def segment2box(segment):
    # Convert 1 segment label to 1 box label, (xy1, xy2, ...) -> (xyxy)
    x, y = segment.T
    return np.array([x.min(), y.min(), x.max(), y.max()])


def get_valid_segment_indices(segments, width, height):
    # filtering segments that exist in out of image
    valid_indices = []
    for segment in segments:
        x, y = segment.T
        valid = x.min() < width and y.min() < height and x.max() > 0 and y.max() > 0
        valid_indices.append(valid)
    return valid_indices


def random_perspective(
        img: np.ndarray,
        labels: dict,
        degrees: int = 10,
        translate: float = 0.1,
        scale: float = 0.1,
        shear: int = 10,
        perspective: float = 0.0,
        border: int = (0, 0)
):
    height = img.shape[0] + border[0] * 2
    width = img.shape[1] + border[1] * 2

    # Center
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
    s = random.uniform(1 - scale, 1 + scale)
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

    for label in labels:
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

            i = box_candidates(box1=bboxes[:, :4].T * s, box2=new.T, area_thr=0.1)
            bboxes = bboxes[i]
            bboxes[:, :4] = new[i]
            labels[label] = bboxes

        elif label == "segmentation":
            segments = labels[label]
            seg_r = resample_segments(segments)
            for i, (seg, cls) in enumerate(seg_r):
                xy = np.ones((len(seg), 3))
                xy[:, :2] = seg
                xy = xy @ M.T
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]
                labels[label][i][0] = xy

    return img, labels
