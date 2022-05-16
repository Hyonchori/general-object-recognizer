# COCO format dataset
import os
import warnings
from copy import deepcopy

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO

from .image_augmentations import resample_segment

warnings.filterwarnings("ignore")


class COCODataset(Dataset):
    def __init__(
            self,
            img_dir: str = None,
            annot_path: str = None,
            targets=("cls", "bbox", "segmentation"),
            max_labels: int = 500,
            img_size=(720, 1280),
            resample_segment: bool = True,
            preproc=None
    ):
        super().__init__()
        self.img_dir = img_dir
        self.annot_path = annot_path
        self.targets = targets
        self.max_labels = max_labels
        self.resample_segment = resample_segment

        self.coco = COCO(self.annot_path)
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.cats = tuple([c["name"] for c in cats])
        self.annotations = self._load_coco_annotations()
        self.img_size = img_size
        self.preproc = preproc

    def __len__(self):
        return len(self.ids)

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        labels = {t: [] for t in self.targets}
        for obj in annotations:
            cls = obj["category_id"]
            for t in self.targets:
                if t == "cls":
                    labels[t].append([cls])
                elif t == "bbox":
                    x1 = obj[t][0]
                    y1 = obj[t][1]
                    x2 = x1 + obj[t][2]
                    y2 = y1 + obj[t][3]
                    if obj["area"] > 0 and x2 > x1 and y2 > y1:  # filtering invalid bbox
                        clean_xyxy = [x1, y1, x2, y2, cls]
                        labels[t].append(clean_xyxy)
                elif t == "segmentation":
                    if obj["area"] > 0:
                        seg = obj[t][0]
                        if self.resample_segment:
                            seg = resample_segment(seg)
                        clean_seg = seg + [cls]
                        labels[t].append(clean_seg)
                else:
                    raise f"Invalid target mode is given: {t}"

        for target in labels:
            labels[target] = np.array(labels[target])

        img_shape = (height, width)
        file_name = im_ann["file_name"] if "file_name" in im_ann else f"{id_:012}.jpg"

        del im_ann, annotations
        return labels, img_shape, file_name

    def load_anno(self, index):
        return self.annotations[index][0]

    def pull_item(self, index):
        id_ = self.ids[index]
        labels, img_shape, file_name = self.annotations[index]
        img_path = os.path.join(self.img_dir, file_name)
        img = cv2.imread(img_path)
        assert img is not None, f"Given image path {img_path} is wrong!"
        return img, labels, img_shape, np.array([id_])

    def __getitem__(self, index):
        img0, labels0, img_shape, img_id = self.pull_item(index)
        img = img0.copy()
        labels = deepcopy(labels0)
        if self.preproc is not None:
            img, labels = self.preproc(img, labels0)
        return img0, img, labels0, labels, img_shape, img_id

    def collate_fn(self, batch):
        img0, img, labels0, labels, img_shape, img_id = zip(*batch)

        return img0, img

