# Dataset that augmentation techniques are added

import random
from copy import deepcopy

import numpy as np
from torch.utils.data import Dataset

from .image_augmentations import (get_mosaic_img, random_perspective, letterbox, get_mixup_img,
                                  color_aug, flip_lr, BlurAug, NoiseAug, WeatherAug)
from ..data_utils import filtering_labels

from general_object_recognizer.Infer.infer_utils import plot_labels


class AugmentedDataset(Dataset):
    def __init__(
            self,
            dataset: Dataset,
            img_size=(720, 1280),
            no_aug: bool = False,

            # mosaic augmentation
            enable_mosaic: bool = True,
            mosaic_prob: float = 1.0,

            # basic augmentation (random perspective, mixup, color, flip)
            degrees: int = 10,
            translate: float = 0.1,
            scale: float = (0.5, 1.5),
            shear: float = 2.0,
            perspective: float = 0.0,
            mixup_prob: float = 1.0,
            color_prob: float = 0.5,
            flip_prob: float = 0.5,

            # additional augmentation (blur, noise, weather)
            enable_blur: bool = True,
            enable_noise: bool = True,
            enable_weather: bool = True,
            additional_prob: float = 0.5,

            preproc=None
    ):
        super().__init__()
        self.dataset = dataset
        self.img_size = img_size
        self.no_aug = no_aug

        self.enable_mosaic = enable_mosaic
        self.mosaic_prob = mosaic_prob

        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.mixup_prob = mixup_prob
        self.color_prob = color_prob
        self.flip_prob = flip_prob

        self.additional_aug = []
        if enable_blur:
            self.additional_aug.append(BlurAug())
        if enable_noise:
            self.additional_aug.append(NoiseAug())
        if enable_weather:
            self.additional_aug.append(WeatherAug())
        self.enable_add = enable_blur | enable_noise | enable_weather
        self.additional_prob = additional_prob

        self.preproc = preproc

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if not self.no_aug:
            input_dim = self.dataset.img_size
            input_h, input_w = input_dim[0], input_dim[1]

            if self.enable_mosaic and random.random() < self.mosaic_prob:

                # 3 additional image indices
                indices = [idx] + [random.randint(0, len(self.dataset) - 1) for _ in range(3)]
                candi = [self.dataset.pull_item(index)[:-1] for index in indices]
                img_list = [x[0] for x in candi]
                label_list = [x[1] for x in candi]
                shape_list = [x[2] for x in candi]

                # get mosaic-augmented image using 4 images
                img, labels = get_mosaic_img(
                    img_list, label_list, shape_list, input_h, input_w
                )
            else:
                img, labels, _, _ = self.dataset.pull_item(idx)

            # apply random perspective
            img, labels = random_perspective(
                img,
                labels,
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
                perspective=self.perspective,
                border=(-input_h // 2, -input_w // 2)
            )

            # apply mixup augmentation
            if random.random() < self.mixup_prob:
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
                img, labels = get_mixup_img(
                    img,
                    labels,
                    mixup_img,
                    mixup_labels
                )

            # basic augmentations (distort color, flip left-right)
            if random.random() < self.color_prob:
                img = color_aug(img)
            if random.random() < self.flip_prob:
                img, labels = flip_lr(img, labels)

            # apply additional augmentations
            if self.enable_add and random.random() < self.additional_prob:
                add_idx = random.randint(0, len(self.additional_aug) - 1)
                add_aug = self.additional_aug[add_idx]
                img = add_aug(image=img)

            # filtering invalid labels (out of image, too small, etc ...)
            labels = filtering_labels(labels, input_w, input_h)

            # preprocessing if self.preproc is not None
            img_p = img.copy()
            labels_p = deepcopy(labels)
            if self.preproc is not None:
                img_p, labels_p = self.preproc(img_p, labels_p)
            return img, img_p, labels, labels_p, img.shape[:2], np.array([idx])
        else:
            img, labels, img_shape, img_id = self.dataset.pull_item(idx)
            img_p = img.copy()
            labels_p = deepcopy(labels)
            if self.preproc is not None:
                img_p, labels_p = self.preproc(img_p, labels_p)
            return img, img_p, labels, labels_p, img_shape, img_id

    def collate_fn(self, batch):
        return self.dataset.collate_fn(batch)
