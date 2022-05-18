import time
import numpy as np
from torch.utils.data.dataloader import DataLoader

from Data.for_train.coco_dataset import COCODataset
from Data.for_train.augmented_dataset import AugmentedDataset
from Data.data_utils import Preprocessing
from Data.for_train.dataloader import InfiniteDataLoader
from Data.for_train.prefetcher import DataPrefertcher

from Model.yolox import YOLOX

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

    model = YOLOX().cuda()
    print(len(dataloader))

    '''cnt = 0
    t0 = time.time()
    for img, img_p, labels, labels_p, img_shape, img_id in dataloader:
        print(f"{cnt} {img_p.shape}")
        cnt += 1
        img_p = img_p.to("cuda")
        pred = model(img_p)
        t1 = time.time()
        print(f"elapsed time: {t1 - t0:.3f}")
        t0 = time.time()'''

    prefetcher = DataPrefertcher(dataloader)
    cnt = 0
    t0 = time.time()
    for _ in range(len(dataloader)):
        imgs, targets = prefetcher.next()
        print(f"{cnt} {imgs.shape}")
        cnt += 1
        img_p = imgs.to("cuda")
        pred = model(img_p)
        t1 = time.time()
        print(f"elapsed time: {t1 - t0:.3f}")
        t0 = time.time()
