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


def get_valid_segment_indices(segments, width, height):
    # filtering segments(n, 2) that exist in out of image
    valid_indices = []
    for segment in segments:
        x, y = segment.T
        bbox = [x.min(), y.min(), x.max(), y.max()]
        valid = bbox[0] < width and bbox[1] < height and bbox[2] > 0 and bbox[3] > 0
        valid_indices.append(valid)
    return valid_indices


def get_valid_bbox_indices(bboxes, width, height):
    # filtering bboxes(n, 5) that exist in out of image
    valid_indices = []
    for bbox in bboxes:
        valid = bbox[0] < width and bbox[1] < height and bbox[2] > 0 and bbox[3] > 0
        valid_indices.append(valid)
    return valid_indices


def scale_and_shift_bboxes(bboxes, scale, w_shift, h_shift):
    # scale and shift bboxes(n, 5)
    if isinstance(scale, tuple):
        x_scale = scale[0]
        y_scale = scale[1]
    else:
        x_scale, y_scale = scale, scale
    bboxes[:, 0::2][:, :-1] = bboxes[:, 0::2][:, :-1] * x_scale + w_shift
    bboxes[:, 1::2] = bboxes[:, 1::2] * y_scale + h_shift
    return bboxes


def scale_and_shift_segments(segments, scale, w_shift, h_shift):
    # scale and shift segments(n, 2)
    if isinstance(scale, tuple):
        x_scale = scale[0]
        y_scale = scale[1]
    else:
        x_scale, y_scale = scale, scale
    for seg_i in range(segments.shape[0]):
        segments[seg_i][0][:, 0:1] = segments[seg_i][0][:, 0:1] * x_scale + w_shift
        segments[seg_i][0][:, 1:2] = segments[seg_i][0][:, 1:2] * y_scale + h_shift
    return segments


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


def get_mosaic_img(img_infos, mosaic_labels, input_w, input_h):
    # mosaic center x, y
    yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
    xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

    for i, (img, _labels, (h0, w0)) in enumerate(img_infos):
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

        labels = _labels.copy()
        for label in labels:
            if len(labels[label]) > 0:
                if label == "bbox":
                    labels[label] = scale_and_shift_bboxes(labels[label], scale, padw, padh)
                elif label == "segmentation":
                    labels[label] = scale_and_shift_segments(labels[label], scale, padw, padh)
                mosaic_labels[label].append(labels[label])

    for label in mosaic_labels:
        mosaic_labels[label] = np.concatenate(mosaic_labels[label], 0)
        if label == "bbox":
            valid_bbox_indices = get_valid_bbox_indices(
                mosaic_labels[label], 2 * input_w, 2 * input_h
            )
            mosaic_labels[label] = mosaic_labels[label][valid_bbox_indices]
        elif label == "segmentation":
            valid_segment_indices = get_valid_segment_indices(
                mosaic_labels[label][:, 0], 2 * input_w, 2 * input_h
            )
            mosaic_labels[label] = mosaic_labels[label][valid_segment_indices]

    return mosaic_img, mosaic_labels


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


def get_mixup_img(img, labels, img2, labels2):
    # applies mixup augmentation
    r = np.random.beta(32., 32.)  # mixup ratio, alpha=beta=32.0
    img = (img * r + img2 * (1 - r)).astype(np.uint8)
    for label in labels:
        labels[label] = np.concatenate((labels[label], labels2[label]), 0)
    return img, labels


def letterbox(
        img: np.ndarray,
        labels=None,
        new_shape=(640, 640),
        color: int = (114, 114, 114),
        auto: bool = True,
        stretch: bool = False,
        stride: int = 32,
        fit: bool = False
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[: 2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # if fit == True: all components of new_shape should be divided by stride for model input
    if fit:
        new_shape = [x + (x + stride) % stride for x in new_shape]

    if img.shape[:2] == new_shape:
        return img, labels, 1., (0, 0)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

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
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    if labels is not None:
        for label in labels:
            if len(labels[label]) > 0:
                if label == "bbox":
                    labels[label] = scale_and_shift_bboxes(labels[label], ratio, dw, dh)
                elif label == "segmentation":
                    labels[label] = scale_and_shift_segments(labels[label], ratio, dw, dh)

    return img, labels, ratio, (dw, dh)


def color_aug(img):
    def _convert(img, alpha=1, beta=0):
        tmp = img.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        img[:] = tmp

    img = img.copy()

    if random.randrange(2):
        _convert(img, beta=random.uniform(-32, 32))

    if random.randrange(2):
        _convert(img, alpha=random.uniform(0.5, 1.5))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if random.randrange(2):
        tmp = img[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        img[:, :, 0] = tmp

    if random.randrange(2):
        _convert(img[:, :, 1], alpha=random.uniform(0.5, 1.5))

    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img


def flip_lr(img, labels):
    _, width, _ = img.shape
    if random.randrange(2):
        img = img[:, ::-1].astype(np.uint8)
        for label in labels:
            if label == "bbox":
                labels[label][:, 0::2][:, :-1] = width - labels[label][:, 2::-2]
            elif label == "segmentation":
                segments = labels[label][:, 0]
                for seg in segments:
                    seg[:, 0] = width - seg[:, 0]
    return img, labels


class BlurAug:
    def __init__(self):
        import imgaug.augmenters as iaa
        self.aug = iaa.OneOf([
            iaa.GaussianBlur(),
            iaa.MotionBlur()
        ])

    def __call__(self, image):
        image = self.aug(image=image)
        return image


class NoiseAug:
    def __init__(self):
        import imgaug.augmenters as iaa
        self.aug = iaa.OneOf([
            iaa.AdditiveGaussianNoise(),
            iaa.Dropout()
        ])

    def __call__(self, image):
        image = self.aug(image=image)
        return image


class WeatherAug:
    def __init__(self):
        import imgaug.augmenters as iaa
        self.aug = iaa.OneOf([
            iaa.Fog(),
            iaa.Snowflakes(),
            iaa.Rain()
        ])

    def __call__(self, image):
        image = self.aug(image=image)
        return image


if __name__ == "__main__":
    import imgaug.augmenters as iaa
    aug = iaa.Fog()
    img_path = "/home/daton/Downloads/coco/val2017/000000000776.jpg"
    img = cv2.imread(img_path)
    cv2.imshow("img", img)

    img = aug(image=img)
    cv2.imshow("img1", img)
    cv2.waitKey(0)
