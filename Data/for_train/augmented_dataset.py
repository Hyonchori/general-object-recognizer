# Dataset that augmentation techniques are added

import random

import cv2
import numpy as np
from torch.utils.data import Dataset

from general_object_recognizer.Data.for_train.image_augmentations import \
    get_mosaic_img, random_perspective, letterbox, get_mixup_img, color_aug, flip_lr, \
    scale_and_shift_bboxes , scale_and_shift_segments, \
    get_valid_segment_indices, get_valid_bbox_indices
from general_object_recognizer.Data.data_utils import plot_labels
# from .image_augmentations import (get_mosaic_img, random_perspective, letterbox, get_mixup_img)
# from ..data_utils import plot_labels


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
            color_prob: float = 0.5,
            flip_prob: float = 0.5,

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
        self.color_prob = color_prob
        self.flip_prob = flip_prob

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
                mixup_img, mixup_labels, _, _ = self.dataset.pull_item(mixup_idx)
                mixup_img, mixup_labels, _, _ = letterbox(
                    mixup_img,
                    mixup_labels,
                    (input_h, input_w),
                    auto=False
                )
                mosaic_img, mosaic_labels = get_mixup_img(
                    mosaic_img,
                    mosaic_labels,
                    mixup_img,
                    mixup_labels
                )

            # basic aug (_distort, flip)
            cv2.imshow("img0", mosaic_img)
            print(random.random())
            if random.random() < self.color_prob:
                mosaic_img = color_aug(mosaic_img)
            if random.random() < self.flip_prob:

                mosaic_img, mosaic_labels = flip_lr(mosaic_img, mosaic_labels)
            cv2.imshow("img_a", mosaic_img)
            plot_labels(mosaic_img, mosaic_labels)


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
