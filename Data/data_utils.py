import numpy as np


def segcls2seg(seg_cls):
    seg = seg_cls[:-1]
    xs = seg[0::2]
    ys = seg[1::2]
    return np.concatenate([xs, ys]).reshape(2, -1).T


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
            print(labels[label].shape)
            valid_indices = filtering_bboxes_indices(bbox, img_w, img_h, iou_thr)
            labels[label] = labels[label][valid_indices]
            print(labels[label].shape)

        elif label == "segmentation":
            print(labels[label].shape)
            segment = labels[label][:, :-1]
            xs = segment[:, 0::2]
            ys = segment[:, 1::2]
            bbox = np.stack([
                np.min(xs, axis=1), np.min(ys, axis=1), np.max(xs, axis=1), np.max(ys, axis=1)
            ]).T
            valid_indices = filtering_bboxes_indices(bbox, img_w, img_h, iou_thr)
            labels[label] = labels[label][valid_indices]
            print(labels[label].shape)

    return labels


def filtering_bboxes_indices(bbox, img_w, img_h, iou_thr=0.3):
    # Filtering a bbox(n, 5) smaller than the threshold in the ratio
    # between the size of viewable and the original
    origin_area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
    vbox = np.zeros((len(bbox), 4))
    vbox[:, 0] = np.max((bbox[:, 0], vbox[:, 0]), axis=0)
    vbox[:, 1] = np.max((bbox[:, 1], vbox[:, 1]), axis=0)
    vbox[:, 2] = np.max((bbox[:, 2], np.ones(len(bbox)) * img_w), axis=0)
    vbox[:, 3] = np.max((bbox[:, 3], np.ones(len(bbox)) * img_h), axis=0)
    viewable_area = (vbox[:, 2] - vbox[:, 0]) * (vbox[:, 3] - vbox[:, 1])
    iou = viewable_area / origin_area
    valid_indices = iou > iou_thr
    return valid_indices

