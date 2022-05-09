import cv2
import numpy as np

from .for_train.image_augmentations import letterbox


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()


def plot_labels(img, labels):
    ref_img = np.zeros_like(img)
    for label in labels:
        if label == "bbox":
            # bbox: (n, 5) = ((x1, y1, x2, y2, cls), (x1, y1, x2, y2, cls), ...)
            for bbox in labels["bbox"]:
                color = colors(bbox[-1], True)
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                              color, 2)
        elif label == "segmentation":
            # seg: (n, 2) = ((x1, y1), (x2, y2), ...)
            for seg, cls in labels["segmentation"]:
                color = colors(cls, True)
                cv2.fillPoly(ref_img, [seg.astype(np.int64)], color)
    img = cv2.addWeighted(img, 1, ref_img, 0.5, 0)
    cv2.imshow("img", img)
    cv2.waitKey(0)


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
            origin_area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
            vbbox = np.zeros((len(bbox), 4))
            vbbox[:, 0] = np.max((bbox[:, 0], vbbox[:, 0]), axis=0)
            vbbox[:, 1] = np.max((bbox[:, 1], vbbox[:, 1]), axis=0)
            vbbox[:, 2] = np.min((bbox[:, 2], np.ones(len(bbox)) * img_w), axis=0)
            vbbox[:, 3] = np.min((bbox[:, 3], np.ones(len(bbox)) * img_h), axis=0)
            viewable_area = (vbbox[:, 2] - vbbox[:, 0]) * (vbbox[:, 3] - vbbox[:, 1])
            iou = viewable_area / origin_area
            valid_indices = iou > iou_thr
            labels[label] = labels[label][valid_indices]

        elif label == "segmentation":
            valid_indices = []
            for seg, _ in labels[label]:
                bbox = segment2box(seg)
                origin_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                vbbox = [max(0, bbox[0]), max(0, bbox[1]), min(img_w, bbox[2]), min(img_h, bbox[3])]
                viewable_area = (vbbox[2] - vbbox[0]) * (vbbox[3] - vbbox[1])
                iou = viewable_area / origin_area
                valid_indices.append(iou >= iou_thr)
            labels[label] = labels[label][valid_indices]

    return labels


class Preprocessing:
    def __init__(
            self,
            img_size=(720, 1280),
            scaling: bool = True,
            normalize: bool = True,
            bgr2rgb: bool = True,
            swap: bool = True
    ):
        self.img_size = img_size
        self.scaling = scaling
        self.normalize = normalize
        self.bgr2rgb = bgr2rgb
        self.swap = swap

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.299, 0.224, 0.225])
        self.swap_channels = (2, 0, 1)

    def __call__(self, img, labels=None):
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if self.img_size is not None:
            img, labels, _, _ = letterbox(img, labels, self.img_size, auto=False, fit=True)
        if self.scaling or self.normalize:
            img /= 255.0
        if self.normalize:
            img -= self.mean
            img /= self.std
        if self.bgr2rgb:
            img = img[..., ::-1]
        if self.swap:
            img = img.transpose(self.swap_channels)
        return img, labels

