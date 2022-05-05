# Dataset that augmentation techniques are added

import random

import cv2
import numpy as np
from torch.utils.data import Dataset

from general_object_recognizer.Data.for_train.image_augmentations import \
    random_perspective, get_valid_segment_indices


def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
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


class AugmentedDataset(Dataset):
    def __init__(
            self,
            dataset: Dataset,
            img_size: int = (720, 1280),
            enable_mosaic: bool = True,
            enable_mixup: bool = True,
            basic_aug: bool = True,
            blur_aug: bool = True,
            noise_aug: bool = True,
            weather_aug: bool = True,
            preproc=None
    ):
        super().__init__()
        self.dataset = dataset
        self.img_size = img_size
        self.enable_mosaic = enable_mosaic
        self.enable_mixup = enable_mixup
        self.basic_aug = basic_aug
        self.blur_aug = blur_aug
        self.noise_aug = noise_aug
        self.weather_aug = weather_aug
        self.preproc = preproc

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.enable_mosaic:
            mosaic_labels = {t: [] for t in self.dataset.targets}
            input_dim = self.dataset.img_size
            input_h, input_w = input_dim[0], input_dim[1]

            # mosaic center x, y
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # 3 additional image indices
            indices = [idx] + [random.randint(0, len(self.dataset) - 1) for _ in range(3)]
            for i, index in enumerate(indices):
                img, _labels, (h0, w0), _ = self.dataset.pull_item(index)
                scale = min(1. * input_h / h0, 1. * input_w / w0)
                img = cv2.resize(
                    img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
                )
                h, w, _ = img.shape
                print(img.shape)
                if i == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, 3), 114, dtype=np.uint8)

                # large image box and small image box
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                    mosaic_img, i, xc, yc, w, h, input_h, input_w
                )
                mosaic_img[l_y1: l_y2, l_x1: l_x2] = img[s_y1: s_y2, s_x1: s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                labels = _labels.copy()
                for label in labels:
                    if len(labels[label]) > 0:
                        if label == "bbox":
                            labels[label][:, 0::2][:, :-1] = labels[label][:, 0::2][:, :-1] * scale + padw
                            labels[label][:, 1::2] = labels[label][:, 1::2] * scale + padh
                        elif label == "segmentation":
                            for seg_i in range(labels[label].shape[0]):
                                labels[label][seg_i][0][:, 0:1] = labels[label][seg_i][0][:, 0:1] * scale + padw
                                labels[label][seg_i][0][:, 1:2] = labels[label][seg_i][0][:, 1:2] * scale + padh
                        mosaic_labels[label].append(labels[label])

            for label in mosaic_labels:
                mosaic_labels[label] = np.concatenate(mosaic_labels[label], 0)
                if label == "bbox":
                    mosaic_labels[label] = mosaic_labels[label][mosaic_labels[label][:, 0] < 2 * input_w]
                    mosaic_labels[label] = mosaic_labels[label][mosaic_labels[label][:, 1] < 2 * input_h]
                    mosaic_labels[label] = mosaic_labels[label][mosaic_labels[label][:, 2] > 0]
                    mosaic_labels[label] = mosaic_labels[label][mosaic_labels[label][:, 3] > 0]
                elif label == "segmentation":
                    print(mosaic_labels[label][:, 0])
                    print(len(mosaic_labels[label]))
                    valid_segment_indices = get_valid_segment_indices(
                        mosaic_labels[label][:, 0], 2 * input_w, 2 * input_h
                    )
                    mosaic_labels[label] = mosaic_labels[label][valid_segment_indices]
                    print(len(mosaic_labels[label]))

            for label in mosaic_labels:
                if label == "bbox":
                    for bbox in mosaic_labels["bbox"]:
                        color = colors(bbox[-1], True)
                        cv2.rectangle(mosaic_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                      color, 2)
                elif label == "segmentation":
                    for seg, cls in mosaic_labels["segmentation"]:
                        color = colors(cls, True)
                        for x, y in seg:
                            cv2.circle(mosaic_img, (int(x), int(y)), 1, color, -1)
            cv2.imshow("img", mosaic_img)
            cv2.waitKey(0)


if __name__ == "__main__":
    img_dir = "/media/jhc/4AD250EDD250DEAF/dataset/coco/val2017"
    annot_path = "/media/jhc/4AD250EDD250DEAF/dataset/coco/annotations_trainval2017/annotations/instances_val2017.json"

    from general_object_recognizer.Data.for_train.coco_dataset import COCODataset
    dataset = COCODataset(
        img_dir=img_dir,
        annot_path=annot_path
    )
    dataset = AugmentedDataset(dataset)
    for items in dataset:
        print(items)
        break
