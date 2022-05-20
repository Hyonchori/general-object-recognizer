import time
import sys

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data.dataloader import DataLoader

from Data.for_train.coco_dataset import COCODataset
from Data.for_train.augmented_dataset import AugmentedDataset
from Data.data_utils import Preprocessing
from Data.for_train.dataloader import InfiniteDataLoader
from Data.for_train.prefetcher import DataPrefertcher

from Model.yolox import YOLOX

from Train.losses.detection_loss import DetectionLosses
from Train.train_utils import get_total_and_free_memory_in_Mb, occupy_memory
from Train.optimizer import get_optimizer

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
    #device = "cpu"
    model = YOLOX().to(device)
    print(len(dataloader))
    get_loss = DetectionLosses()

    opt = get_optimizer(1, 10, model)
    scaler = torch.cuda.amp.GradScaler()

    cnt = 0
    t0 = time.time()
    occupy_memory(device)
    for e in range(2):
        ts = time.time()
        for img, img_p, labels, labels_p, img_shape, img_id in tqdm(dataloader):
            #print(f"\n{cnt} {img_p.shape}")
            cnt += 1
            img_p = img_p.to(device)
            labels_p["bbox"] = labels_p["bbox"].to(device)

            pt0 = time.time()
            pred, x_shifts, y_shifts, expanded_strides, origin_preds, dtype = model(img_p)
            pt1 = time.time()
            #print(f"\tpred time: {pt1 - pt0:.3f}")

            total_loss = get_loss(
                img_p.shape, pred, x_shifts, y_shifts, expanded_strides, labels_p["bbox"], origin_preds, dtype
            )[0]
            #print(total_loss)
            opt.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(opt)
            scaler.update()

            t1 = time.time()
            #print(f"elapsed time: {t1 - t0:.3f}")
            t0 = time.time()
        te = time.time()
        print(f"train_time: {(te - ts)/60:.2f}m")

    save_path = f"test_{e}.pth"
    torch.save(model.state_dict(), save_path)

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
        total, used = get_total_and_free_memory_in_Mb(device)
        print(total, used)

        t1 = time.time()
        print(f"elapsed time: {t1 - t0:.3f}")
        t0 = time.time()
