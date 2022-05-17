import numpy as np

from Data.for_train.coco_dataset import COCODataset
from Data.for_train.augmented_dataset import AugmentedDataset
from Data.data_utils import Preprocessing


if __name__ == "__main__":
    val_dir = "/home/daton/Downloads/coco/val2017"
    ann_path = "/home/daton/Downloads/coco/annotations_trainval2017/annotations/instances_val2017.json"

    val_dir = "/media/jhc/4AD250EDD250DEAF/dataset/coco/val2017"
    ann_path = "/media/jhc/4AD250EDD250DEAF/dataset/coco/annotations_trainval2017/annotations/instances_val2017.json"

    dataset = COCODataset(val_dir, ann_path)
    preproc = Preprocessing()
    dataset = AugmentedDataset(dataset, preproc=preproc, flip_prob=1.0)

    for img, img_p, labels, labels_p, img_shape, img_id in dataset:
        print(img_p.shape)
        for label in labels:
            print(labels[label].shape)
        break
