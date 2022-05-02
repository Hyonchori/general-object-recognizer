# torch Dataset for train detection model
import os
import random
from typing import Tuple

import cv2
import numpy as np
from torch.utils.data.dataset import Dataset
from pycocotools.coco import COCO

from .detection_augmentations import random_perspective, mixup, TrainTransform


class COCODataset(Dataset):
    """
    COCO dataset class.
    Images should be located in 'data_dir'
    """
    def __init__(
            self,
            data_dir: str = None,
            json_path: str = None,
            img_size: Tuple[int] = (720, 1280),
            preproc: TrainTransform = None
    ):
        super().__init__()
        self.data_dir = data_dir
        self.json_path = json_path

        self.coco = COCO(self.json_path)
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._class = tuple([c["name"] for c in cats])
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
        objs = []
        for obj in annotations:
            x1 = obj["bbox"][0]
            y1 = obj["bbox"][1]
            x2 = x1 + obj["bbox"][2]
            y2 = y1 + obj["bbox"][3]
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)
        res = np.zeros((num_objs, 6))

        for i, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[i, 0: 4] = obj["clean_bbox"]
            res[i, 4] = cls

        file_name = im_ann["file_name"] if "file_name" in im_ann else f"{id_:012}.jpg"
        img_shape = (height, width)

        del im_ann, annotations
        return res, img_shape, file_name

    def load_anno(self, index):
        return self.annotations[index][0]

    def pull_item(self, index):
        id_ = self.ids[index]
        res, img_info, file_name = self.annotations[index]
        img_path = os.path.join(self.data_dir, file_name)
        img = cv2.imread(img_path)
        assert img is not None
        return img, res.copy(), img_info, np.array([id_])

    def __getitem__(self, index):
        img0, target, img_info, img_id = self.pull_item(index)
        img = img0.copy()
        if self.preproc is not None:
            img, target = self.preproc(img0.copy(), target)
        return img0, img, target, img_info, img_id


class AugmentedCOCODataset(COCODataset):
    def __init__(
            self,
            data_dir: str = None,
            json_path: str = None,
            img_size: Tuple[int] = (720, 1280),
            preproc: TrainTransform = None,
            enable_mosaic: bool = True,
            enable_mixup: bool = True,
            degrees: float = 10.0,
            translate: float = 0.1,
            scale: Tuple[float] = (0.5, 1.5),
            shear: float = 2.0,
            perspective: float = 0.0,
            mixup_prob: float = 0.5,
            weather_prob: float = 0.5
    ):
        super().__init__(data_dir, json_path, img_size, preproc)
        self.enable_mosaic = enable_mosaic
        self.enable_mixup = enable_mixup
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.mixup_prob = mixup_prob
        self.weather_prob = weather_prob

        if weather_prob != 0:
            import imgaug.augmenters as iaa
            #self.weather_aug = iaa.

    def __getitem__(self, idx):
        if self.enable_mosaic:
            mosaic_labels = []
            input_size = self.img_size
            input_h, input_w = input_size[:]

            # mosaic center
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # 3 additional image indices
            indices = [idx] + [random.randint(0, len(self)) for _ in range(3)]

            for i, index in enumerate(indices):
                img, _labels, (h0, w0), _ = self.pull_item(index)
                scale = min(1. * input_h / h0, 1 * input_w / w0)
                img = cv2.resize(
                    img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
                )
                # generate mosaic image
                (h, w, c) = img.shape[:3]
                if i == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)

                # suffix l: large image, s: small image in mosaic image
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = self.get_mosaic_coordinate(
                    mosaic_img, i, xc, yc, w, h, input_h, input_w
                )
                mosaic_img[l_y1: l_y2, l_x1: l_x2] = img[s_y1: s_y2, s_x1: s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

    @staticmethod
    def get_mosaic_coordinate(self, image, mosaic_index, xc, yc, w, h, input_h, input_w):
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



if __name__ == "__main__":
    val_dir = "/home/daton/Downloads/coco/val2017"
    val_path = "/home/daton/Downloads/coco/annotations_trainval2017/annotations/instances_val2017.json"
    from general_object_recognizer.Data.for_train.detection_augmentations import TrainTransform
    preproc = TrainTransform(img_size=(720, 1280), swap=False)
    #dataset = COCODataset(val_dir, val_path, preproc=preproc)
    dataset = AugmentedCOCODataset(val_dir, val_path, preproc=preproc)

    for img0, img, targets, img_info, img_id in dataset:
        print(img.shape)
        print(img.dtype == "uint8")
        '''for t in targets:
            print(t, "??")
            cv2.rectangle(img, list(map(int, t[:2])), list(map(int, t[2:4])), [255, 0, 255], 2)
        '''
        cv2.imshow("img0", img0)
        cv2.imshow("img", img)
        cv2.waitKey(0)
