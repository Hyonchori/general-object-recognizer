import time
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from Data.for_train.coco_dataset import COCODataset
from Data.for_train.augmented_dataset import AugmentedDataset
from Data.data_utils import Preprocessing
from Data.for_train.dataloader import InfiniteDataLoader
from Data.for_train.prefetcher import DataPrefertcher

from Model.yolox import YOLOX

from Train.losses.detection_loss import DetectionLosses

if __name__ == "__main__":
    val_dir = "/home/daton/Downloads/coco/val2017"
    ann_path = "/home/daton/Downloads/coco/annotations_trainval2017/annotations/instances_val2017.json"

    #val_dir = "/media/jhc/4AD250EDD250DEAF/dataset/coco/val2017"
    #ann_path = "/media/jhc/4AD250EDD250DEAF/dataset/coco/annotations_trainval2017/annotations/instances_val2017.json"

    dataset = COCODataset(val_dir, ann_path, targets="bbox")
    preproc = Preprocessing()
    dataset = AugmentedDataset(dataset, preproc=preproc, flip_prob=1.0)

    dataloader = InfiniteDataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = YOLOX().to(device)
    print(len(dataloader))
    get_loss = DetectionLosses()

    cnt = 0
    t0 = time.time()
    for img, img_p, labels, labels_p, img_shape, img_id in dataloader:
        print(f"\n{cnt} {img_p.shape}")
        cnt += 1
        img_p = img_p.to(device)
        labels_p["bbox"] = labels_p["bbox"].to(device)

        pt0 = time.time()
        pred, x_shifts, y_shifts, expanded_strides, origin_preds, dtype = model(img_p)
        pt1 = time.time()
        print(f"\tpred time: {pt1 - pt0:.3f}")

        get_loss(
            img_p.shape, pred, x_shifts, y_shifts, expanded_strides, labels_p["bbox"], origin_preds, dtype
        )

        t1 = time.time()
        print(f"elapsed time: {t1 - t0:.3f}")
        t0 = time.time()

    prefetcher = DataPrefertcher(dataloader)

    cnt = 0
    t0 = time.time()
    for _ in range(len(dataloader)):
        imgs, targets = prefetcher.next()
        print(f"\n{cnt} {imgs.shape} {imgs.shape[-2:]}")
        cnt += 1
        img_p = imgs.to("cuda")

        pt0 = time.time()
        pred, x_shifts, y_shifts, expanded_strides, origin_preds, dtype = model(img_p)
        pt1 = time.time()
        print(f"\tpred time: {pt1 - pt0:.3f}")

        get_loss(
            imgs.shape, pred, x_shifts, y_shifts, expanded_strides, targets["bbox"], origin_preds, dtype
        )

        t1 = time.time()
        print(f"elapsed time: {t1 - t0:.3f}")
        t0 = time.time()
