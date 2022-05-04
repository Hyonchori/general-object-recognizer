# Dataset that augmentation techniques are added

import random

import cv2
import numpy as np
from torch.utils.data import Dataset

from general_object_recognizer.Data.for_train.image_augmentations import \
    random_perspective


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
            indices = [idx] + [random.randint(0, len(self.dataset) - 1) for _ in range(3)]
            for i_mosaic, index in enumerate(indices):
                img, labels, _, _ = self.dataset.pull_item(index)
                for label in labels:
                    if label == "bbox":
                        for bbox in labels["bbox"]:
                            color = colors(bbox[-1], True)
                            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                          color)
                    elif label == "segmentation":
                        for seg, cls in labels["segmentation"]:
                            color = colors(cls, True)
                            for x, y in seg:
                                cv2.circle(img, (int(x), int(y)), 1, color, -1)

                img_p, labels = random_perspective(img, labels)
                for label in labels:
                    if label == "bbox":
                        for bbox in labels["bbox"]:
                            color = colors(bbox[-1], True)
                            cv2.rectangle(img_p, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                          color)
                    elif label == "segmentation":
                        for seg, cls in labels["segmentation"]:
                            color = colors(cls, True)
                            for x, y in seg:
                                cv2.circle(img_p, (int(x), int(y)), 1, color, -1)

                cv2.imshow("img0", img)
                cv2.imshow("img_p", img_p)
                cv2.waitKey(0)
        pass


if __name__ == "__main__":
    img_dir = "/home/daton/Downloads/coco/val2017"
    annot_path = "/home/daton/Downloads/coco/annotations_trainval2017/annotations/instances_val2017.json"

    from general_object_recognizer.Data.for_train.coco_dataset import COCODataset
    dataset = COCODataset(
        img_dir=img_dir,
        annot_path=annot_path
    )
    dataset = AugmentedDataset(dataset)
    for items in dataset:
        print(items)
        break
