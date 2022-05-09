# COCO format dataset
import os
import warnings
from copy import deepcopy

import cv2
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO

warnings.filterwarnings("ignore")


class COCODataset(Dataset):
    def __init__(
            self,
            img_dir: str = None,
            annot_path: str = None,
            targets: str = ("cls", "bbox", "segmentation"),
            img_size=(720, 1280),
            preproc=None
    ):
        super().__init__()
        self.img_dir = img_dir
        self.annot_path = annot_path
        self.targets = targets

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
                    labels[t].append(cls)
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
                        xs = obj[t][0][0::2]
                        ys = obj[t][0][1::2]
                        seg = np.concatenate([xs, ys]).reshape(2, -1).T
                        clean_seg = [seg, cls]
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
        return img, labels.copy(), img_shape, np.array([id_])

    def __getitem__(self, index):
        img0, labels0, img_shape, img_id = self.pull_item(index)
        img = img0.copy()
        labels = deepcopy(labels0)
        if self.preproc is not None:
            img, labels = self.preproc(img, labels0)
        return img0, img, labels0, labels, img_shape, img_id


if __name__ == "__main__":
    img_dir = "/home/daton/Downloads/coco/val2017"
    annot_path = "/home/daton/Downloads/coco/annotations_trainval2017/annotations/instances_val2017.json"

    import json
    with open(annot_path) as f:
        data = json.load(f)

    # print(data)
    dataset = COCODataset(img_dir=img_dir,
                          annot_path=annot_path,)


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

    for img0, img, labels0, labels, img_shape, img_id in dataset:
        print(img0.shape)
        for label in labels:
            if label == "bbox":
                for bbox in labels["bbox"]:
                    color = colors(bbox[-1], True)
                    cv2.rectangle(img0, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                  color)
            elif label == "segmentation":
                for seg_cls in labels["segmentation"]:
                    print(seg_cls)
                    seg, cls = seg_cls[0], seg_cls[-1]
                    color = colors(cls, True)
                    for x, y in seg:
                        cv2.circle(img0, (int(x), int(y)), 1, color, -1)

        cv2.imshow("img0", img0)
        cv2.waitKey(0)
