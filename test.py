import cv2

from Data.for_train.coco_dataset import COCODataset
from Data.for_train.augmented_dataset import AugmentedDataset
from Data.data_utils import Preprocessing, plot_labels


if __name__ == "__main__":
    val_dir = "/home/daton/Downloads/coco/val2017"
    val_path = "/home/daton/Downloads/coco/annotations_trainval2017/annotations/instances_val2017.json"

    img_size = (720, 1280)
    preproc = Preprocessing(img_size=img_size)
    dataset = COCODataset(val_dir, val_path, img_size=img_size, preproc=preproc)
    dataset = AugmentedDataset(dataset, img_size=img_size, preproc=preproc)

    for img0, img, labels0, labels, img_info, img_id in dataset:
        plot_labels(img0, labels0)