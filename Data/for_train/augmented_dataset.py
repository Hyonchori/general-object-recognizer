# Dataset that augmentation techniques are added

import random

import cv2
import numpy as np
from torch.utils.data import Dataset

from general_object_recognizer.Data.for_train.image_augmentations import \
    get_mosaic_img, random_perspective, get_mixup_img, \
    scale_and_shift_bboxes , scale_and_shift_segments, \
    get_valid_segment_indices, get_valid_bbox_indices


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

            # basic augmentation (random perspective, mixup)
            basic_aug: bool = True,
            degrees: int = 10,
            translate: float = 0.1,
            scale: float = (0.5, 1.5),
            shear: float = 2.0,
            perspective: float = 0.0,
            mixup_prob: float = 1.0,

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
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.mixup_prob = mixup_prob

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

            # 3 additional image indices
            indices = [idx] + [random.randint(0, len(self.dataset) - 1) for _ in range(3)]
            img_infos = [self.dataset.pull_item(index)[:-1] for index in indices]

            # get mosaic image
            mosaic_img, mosaic_labels = get_mosaic_img(
                img_infos, mosaic_labels, input_w, input_h
            )

            # apply random perspective
            mosaic_img, mosaic_labels = random_perspective(
                mosaic_img,
                mosaic_labels,
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
                perspective=self.perspective,
                border=(-input_h // 2, -input_w // 2),
            )

            if self.enable_mixup and random.random() < self.mixup_prob:
                mixup_labels = {t: [] for t in self.dataset.targets}
                while len(mixup_labels["bbox"]) == 0:
                    mixup_idx = random.randint(0, len(self.dataset) - 1)
                    mixup_labels = self.dataset.load_anno(mixup_idx)
                mixup_img, mixup_labels, (mixup_h, mixup_w), _ = self.dataset.pull_item(mixup_idx)
                ratio = min(input_h / mixup_h, input_w / mixup_w)
                mixup_img = cv2.resize(
                    mixup_img,
                    (int(mixup_h * ratio), int(mixup_w * ratio)), interpolation=cv2.INTER_LINEAR
                )
                for label in mixup_labels:
                    if len(mixup_labels[label]) > 0:
                        if label == "bbox":
                            mixup_labels[label] = scale_and_shift_bboxes(mixup_labels[label], ratio, 0, 0)
                        elif label == "segmentation":
                            mixup_labels[label] = scale_and_shift_segments(mixup_labels[label], ratio, 0, 0)
                cv2.imshow('mix', mixup_img)

            for label in mosaic_labels:
                if label == "bbox":
                    for bbox in mosaic_labels["bbox"]:
                        color = colors(bbox[-1], True)
                        cv2.rectangle(mosaic_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                      color, 2)
                elif label == "segmentation":
                    ref_img = np.zeros_like(mosaic_img)
                    for seg, cls in mosaic_labels["segmentation"]:
                        color = colors(cls, True)
                        cv2.fillPoly(ref_img, [seg.astype(np.int64)], color)
            mosaic_img = cv2.addWeighted(mosaic_img, 1, ref_img, 0.5, 0)
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
